"""MuJoCo backend for RobotInterface: position-actuated arm with quintic
joint interpolation, optional passive viewer, IK-based TCP moves, and
optional gripper control.
"""

import time
from pathlib import Path

import mujoco
import numpy as np

from .ik import solve_ik, solve_ik_trajectory
from .kinematics import manipulability, pose_jacobian, solve_joint_velocity
from .robot_interface import RobotInterface
from .trajectory import JointTrajectory, plan_cartesian_line, plan_ptp_quintic
from .viewer import PassiveViewer, configure_low_power_rendering

_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "xarm6" / "scene.xml"

# Names recognised as gripper actuators (order = priority)
_GRIPPER_ACTUATOR_NAMES = ("gripper", "gripper_motor", "finger_actuator")


def _quintic(t: np.ndarray, q0: np.ndarray, q1: np.ndarray, T: float) -> np.ndarray:
    """Zero vel/acc quintic interpolation, t in [0, T]."""
    s = np.clip(t / T, 0.0, 1.0)
    s5 = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    return q0 + (q1 - q0) * s5[:, None]


class SimRobot(RobotInterface):
    def __init__(self, model_path: str | Path = _MODEL_PATH, timestep: float | None = None,
                 viewer: bool = False, *, viewer_hz: float = 30.0,
                 low_power_rendering: bool = False):
        if not np.isfinite(viewer_hz) or viewer_hz <= 0:
            raise ValueError("viewer_hz must be finite and positive")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        if low_power_rendering:
            configure_low_power_rendering(self.model)
        if timestep is not None:
            self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        # separate MjData for IK/FK evaluation: never touched by the viewer thread
        self.eval_data = mujoco.MjData(self.model)
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        if site_id < 0:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "link_tcp")
        self._site = site_id

        # --- Detect gripper actuator and compute arm/gripper boundaries ---
        self._gripper_actuator_id: int | None = None
        self._gripper_ctrl_range: np.ndarray | None = None
        self._n_arm_actuators: int = self.model.nu  # default: all actuators are arm

        for i in range(self.model.nu):
            name = self.model.actuator(i).name
            if name in _GRIPPER_ACTUATOR_NAMES:
                self._gripper_actuator_id = i
                self._gripper_ctrl_range = self.model.actuator(i).ctrlrange.copy()
                self._n_arm_actuators = i  # arm actuators are indices [0, i)
                break

        # Compute arm qpos/dof boundary from arm actuator joint targets.
        # Arm actuators always map to the first N joints in the kinematic tree.
        self._n_arm_qpos: int = self.model.nq
        self._n_arm_dof: int = self.model.nv
        self._gripper_open_ctrl: float = 0.0
        self._gripper_close_ctrl: float = 0.0
        if self._gripper_actuator_id is not None:
            max_qpos = 0
            max_dof = 0
            for i in range(self._n_arm_actuators):
                trntype = self.model.actuator_trntype[i]
                if trntype == mujoco.mjtTrn.mjTRN_JOINT:
                    jnt_id = self.model.actuator_trnid[i, 0]
                    max_qpos = max(max_qpos, self.model.jnt_qposadr[jnt_id] + 1)
                    max_dof = max(max_dof, self.model.jnt_dofadr[jnt_id] + 1)
            self._n_arm_qpos = max_qpos
            self._n_arm_dof = max_dof

            lo, hi = self._gripper_ctrl_range
            open_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "open")
            close_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "close")
            if open_key >= 0 and close_key >= 0:
                self._gripper_open_ctrl = float(self.model.key(open_key).ctrl[self._gripper_actuator_id])
                self._gripper_close_ctrl = float(self.model.key(close_key).ctrl[self._gripper_actuator_id])
            else:
                self._gripper_open_ctrl = 0.0
                self._gripper_close_ctrl = float(hi if abs(hi) > abs(lo) else lo)

        # --- Initialise state ---
        self._viewer = None
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # home
        # MuJoCo < 3.15 pads incomplete keyframe qpos vectors with zeros while
        # >= 3.15 pads with qpos0, which would drop free-joint scene props
        # (e.g. the pick-place cube) to the world origin. Keyframes define the
        # robot pose; floating props always keep their authored qpos0 pose.
        for j in range(self.model.njnt):
            if int(self.model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                adr = self.model.jnt_qposadr[j]
                self.data.qpos[adr:adr + 7] = self.model.qpos0[adr:adr + 7]
        self.data.ctrl[:self._n_arm_actuators] = self.data.qpos[:self._n_arm_qpos]
        mujoco.mj_forward(self.model, self.data)
        if viewer:
            self._viewer = PassiveViewer(self.model, self.data)
        # wall-clock origin for real-time sync (only used when viewer is open)
        self._wall_start = time.monotonic() - self.data.time
        self._viewer_interval = 1.0 / viewer_hz
        self._last_viewer_sync = time.monotonic()

    # -- arm state ---------------------------------------------------------
    def get_joint_positions(self) -> np.ndarray:
        """Arm joint positions only (excludes gripper joints)."""
        return self.data.qpos[:self._n_arm_qpos].copy()

    def get_tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.site_xpos[self._site].copy()
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self._site])
        return pos, quat

    # -- stepping ----------------------------------------------------------
    def _sync(self, *, force: bool = False) -> None:
        if self._viewer is not None:
            now = time.monotonic()
            if force or now - self._last_viewer_sync >= self._viewer_interval:
                self._viewer.sync()
                self._last_viewer_sync = time.monotonic()

    def _step(self) -> None:
        """One physics step with gravity-compensation feedforward (like real
        xArm position mode, which has model-based compensation built in)."""
        self.data.qfrc_applied[:self._n_arm_dof] = self.data.qfrc_bias[:self._n_arm_dof]
        mujoco.mj_step(self.model, self.data)
        self.data.qfrc_applied[:] = 0.0
        # real-time sync: sleep until wall-clock catches up with sim time
        if self._viewer is not None:
            wall_elapsed = time.monotonic() - self._wall_start
            sim_ahead = self.data.time - wall_elapsed
            if sim_ahead > 0:
                time.sleep(sim_ahead)
        self._sync()

    def _step_until(self, condition, timeout: float) -> None:
        t0 = self.data.time
        while self.data.time - t0 < timeout:
            if self._viewer is not None and not self._viewer.is_running():
                return
            self._step()
            if condition():
                break
        self._sync(force=True)

    def settle(self, duration: float = 1.0) -> None:
        """Step physics for a while (let the arm reach current ctrl)."""
        self._step_until(lambda: False, duration)

    # -- arm commands ------------------------------------------------------
    def set_joint_positions(self, q: np.ndarray) -> None:
        """Set arm joint targets only (does not affect gripper)."""
        self.data.ctrl[:self._n_arm_actuators] = q

    def move_joints(self, q_goal: np.ndarray, duration: float, wait: bool = True) -> None:
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive")
        # Continue the previous position command. Restarting from measured
        # qpos would introduce a target jump whenever the servo still lags.
        q0 = self.data.ctrl[:self._n_arm_actuators].copy()
        q_goal = np.asarray(q_goal, dtype=float)
        n_steps = int(np.ceil(duration / self.model.opt.timestep))
        times = np.minimum(np.arange(1, n_steps + 1) * self.model.opt.timestep, duration)
        traj = _quintic(times, q0, q_goal, duration)
        for q in traj:
            self.data.ctrl[:self._n_arm_actuators] = q
            self._step()
            if self._viewer is not None and not self._viewer.is_running():
                return
        if wait:
            self.settle(0.3)

    def move_tcp(self, pos: np.ndarray, quat: np.ndarray, duration: float,
                 wait: bool = True, *, steps: int = 50) -> None:
        """Move along a straight Cartesian path with shortest-path rotation."""
        pos0, quat0 = self.get_tcp_pose()
        s = np.linspace(0.0, 1.0, steps + 1)[1:]
        waypoints = pos0 + s[:, None] * (np.asarray(pos) - pos0)
        quat_traj = None
        if quat is not None:
            quat_goal = np.asarray(quat, dtype=float)
            q0_conj = np.array([quat0[0], -quat0[1], -quat0[2], -quat0[3]])
            # The integrator right-multiplies a tool-frame rotation increment.
            q_delta_local = np.empty(4)
            mujoco.mju_mulQuat(q_delta_local, q0_conj, quat_goal)  # conj(start) * goal
            rotvec_local = np.empty(3)
            mujoco.mju_quat2Vel(rotvec_local, q_delta_local, 1.0)
            quat_traj = np.empty((steps, 4))
            for i in range(steps):
                quat_traj[i] = quat0
                mujoco.mju_quatIntegrate(quat_traj[i], rotvec_local, s[i])
        # solve IK per waypoint on the isolated eval_data (viewer thread never
        # touches it), then track the joint path with quintic segments.
        # Use full qpos as q_start so IK preserves gripper state.
        q_start = self.data.qpos.copy()
        q_traj, ok = solve_ik_trajectory(self.model, self.eval_data, "attachment_site",
                                         waypoints, quat_traj, q_start)
        if not ok:
            raise RuntimeError(f"IK did not converge for target pos={pos} quat={quat}")
        seg = max(duration / steps, 4 * self.model.opt.timestep)
        for q in q_traj:
            # IK returns full qpos; extract arm portion for move_joints
            self.move_joints(q[:self._n_arm_qpos], seg, wait=False)
        if wait:
            self.settle(0.3)

    def execute_trajectory(self, traj: JointTrajectory, wait: bool = True) -> None:
        """Execute a time-parameterized JointTrajectory.

        Interpolates control targets at simulation timestep resolution model.opt.timestep.
        """
        if traj.n_steps == 0:
            return

        dt_sim = self.model.opt.timestep
        t_start = float(traj.times[0])
        t_end = float(traj.times[-1])
        cur_t = t_start

        while cur_t <= t_end + 0.5 * dt_sim:
            q_target, _, _ = traj.sample(cur_t)
            self.data.ctrl[:self._n_arm_actuators] = q_target[:self._n_arm_actuators]
            self._step()
            if self._viewer is not None and not self._viewer.is_running():
                return
            cur_t += dt_sim

        # Ensure exact final configuration
        self.data.ctrl[:self._n_arm_actuators] = traj.q[-1, :self._n_arm_actuators]
        if wait:
            self.settle(0.3)

    def plan_joint_trajectory(self, q_goal: np.ndarray, max_vel: float | np.ndarray = 1.57,
                              max_acc: float | np.ndarray = 4.0,
                              duration: float | None = None) -> JointTrajectory:
        """Plan a time-parameterized quintic joint trajectory from current state to q_goal."""
        q0 = self.get_joint_positions()
        return plan_ptp_quintic(q0, q_goal, max_vel=max_vel, max_acc=max_acc,
                                duration=duration, dt=self.model.opt.timestep)

    def plan_cartesian_trajectory(self, pos: np.ndarray, quat: np.ndarray | None = None,
                                  max_lin_vel: float = 0.20, max_lin_acc: float = 1.0,
                                  site_name: str | None = None) -> tuple[JointTrajectory, bool]:
        """Plan a Cartesian straight-line trajectory from current TCP pose to (pos, quat)."""
        site = site_name if site_name is not None else self.model.site(self._site).name
        p0, quat0 = self.get_tcp_pose()
        return plan_cartesian_line(p0, quat0, pos, quat, self.model, site_name=site,
                                   max_lin_vel=max_lin_vel, max_lin_acc=max_lin_acc,
                                   q_init=self.data.qpos.copy(), dt=self.model.opt.timestep,
                                   eval_data=self.eval_data)

    def ik_trajectory(self, waypoints_pos: np.ndarray, quat: np.ndarray | None,
                      pos_tol: float = 5e-5) -> tuple[np.ndarray, bool]:
        """IK along Cartesian waypoints using the isolated eval_data."""
        # Use full qpos so IK operates on the complete kinematic chain
        return solve_ik_trajectory(self.model, self.eval_data, "attachment_site",
                                   np.asarray(waypoints_pos, dtype=float), quat,
                                   self.data.qpos.copy(), pos_tol=pos_tol)

    def fk_positions(self, q_traj: np.ndarray) -> np.ndarray:
        """FK of a joint trajectory -> TCP positions (N,3) on eval_data.

        q_traj rows may be arm-only (n_arm,) or full (nq,).
        """
        out = np.empty((len(q_traj), 3))
        for i, q in enumerate(q_traj):
            if len(q) < self.model.nq:
                # Pad arm-only joint array with current gripper state
                full = self.data.qpos.copy()
                full[:len(q)] = q
                self.eval_data.qpos[:] = full
            else:
                self.eval_data.qpos[:] = q
            mujoco.mj_forward(self.model, self.eval_data)
            out[i] = self.eval_data.site_xpos[self._site]
        return out

    def tcp_jacobian(self, q: np.ndarray | None = None) -> np.ndarray:
        """(6, nv) world-frame TCP Jacobian on eval_data; site twist = J @ qd."""
        return pose_jacobian(self.model, self.eval_data,
                             self.model.site(self._site).name, q)

    def solve_tcp_velocity(self, twist: np.ndarray, q: np.ndarray | None = None, *,
                           damping: float = 1e-3) -> np.ndarray:
        """Velocity-level IK: joint velocity for desired TCP twist [v; omega] (world)."""
        return solve_joint_velocity(self.model, self.eval_data,
                                    self.model.site(self._site).name, twist, q,
                                    damping=damping)

    def tcp_manipulability(self, q: np.ndarray | None = None, *,
                           include_rotation: bool = True) -> float:
        """Yoshikawa manipulability of the TCP; near 0 at singularities."""
        return manipulability(self.model, self.eval_data,
                              self.model.site(self._site).name, q,
                              include_rotation=include_rotation)

    # -- gripper control ---------------------------------------------------
    @property
    def has_gripper(self) -> bool:
        return self._gripper_actuator_id is not None

    def get_gripper_position(self) -> float:
        """Current gripper target (0.0 = fully open, 1.0 = fully closed)."""
        if not self.has_gripper:
            raise NotImplementedError("This robot has no gripper")
        ctrl = self.data.ctrl[self._gripper_actuator_id]
        span = self._gripper_close_ctrl - self._gripper_open_ctrl
        if abs(span) < 1e-12:
            return 0.0
        return float(np.clip((ctrl - self._gripper_open_ctrl) / span, 0.0, 1.0))

    def set_gripper(self, pos: float) -> None:
        """Set gripper target (0.0 = fully open, 1.0 = fully closed)."""
        if not self.has_gripper:
            raise NotImplementedError("This robot has no gripper")
        pos = float(np.clip(pos, 0.0, 1.0))
        target_ctrl = self._gripper_open_ctrl + pos * (self._gripper_close_ctrl - self._gripper_open_ctrl)
        self.data.ctrl[self._gripper_actuator_id] = target_ctrl

    def open_gripper(self, wait: bool = True) -> None:
        """Fully open the gripper."""
        self.set_gripper(0.0)
        if wait:
            self.settle(0.5)

    def close_gripper(self, wait: bool = True) -> None:
        """Fully close the gripper."""
        self.set_gripper(1.0)
        if wait:
            self.settle(0.5)

    # -- misc --------------------------------------------------------------
    def solve_tcp(self, pos, quat=None, q_init=None):
        """One-shot IK helper: q reaching (pos, quat) from current state."""
        return solve_ik(self.model, self.eval_data, "attachment_site",
                        np.asarray(pos, dtype=float),
                        None if quat is None else np.asarray(quat, dtype=float),
                        q_init=q_init)

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
