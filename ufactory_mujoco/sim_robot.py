"""MuJoCo backend for RobotInterface: position-actuated arm with quintic
joint interpolation, optional passive viewer, IK-based TCP moves, and
optional gripper control.
"""

from collections.abc import Callable
from contextlib import contextmanager
import math
import time
from pathlib import Path

import mujoco
import numpy as np

from .bio_g2 import BioG2ControlOutput, BioGripperG2Controller, MuJoCoBioG2Adapter
from .bio_g2 import (
    MAX_GAP_MM as BIO_G2_MAX_GAP_MM,
    MIN_GAP_MM as BIO_G2_MIN_GAP_MM,
    STROKE_PER_FINGER_M as BIO_G2_STROKE_M,
    q_to_gap as bio_g2_q_to_gap,
)
from .ft_sensor import AI1500Reader, SensorReading, has_ft_sensor
from .g2 import ControlOutput, G2Controller
from .g2 import (
    GRIPPER_G2_SIM_CLOSE_DRIVE,
    GRIPPER_G2_SIM_OPEN_DRIVE,
    G2Model,
    gripper_g2_gap_m_to_sim_drive,
    gripper_g2_sim_drive_to_gap_m,
)
from .grippers import (
    GRIPPER_ACTUATOR_NAMES,
    GripperSpec,
    bilateral_finger_contact,
    bodies_in_contact,
    drive_to_gap,
    gap_to_position,
    identify_gripper,
    measured_bio_g2_gap_m,
    measured_g2_pad_gap_mm,
    position_to_gap,
    require_bilateral_contact,
)
from .kinematics import manipulability, solve_ik_trajectory, solve_joint_velocity
from .compose import load_model
from .robots import MODELS_ROOT
from .robot_interface import RobotInterface
from .trajectory import JointTrajectory
from .viewer import PassiveViewer, configure_low_power_rendering

_MODEL_PATH = MODELS_ROOT / "xarm6" / "scene.xml"


def _quintic(t: np.ndarray, q0: np.ndarray, q1: np.ndarray, T: float) -> np.ndarray:
    """Zero vel/acc quintic interpolation, t in [0, T]."""
    s = np.clip(t / T, 0.0, 1.0)
    s5 = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    return q0 + (q1 - q0) * s5[:, None]


class SimRobot(RobotInterface):
    """MuJoCo implementation of ``RobotInterface``.

    Arm position servos, an optional gripper (calibrated G2 servo when present), IK helpers
    and an optional passive viewer. ``model_path`` is a model name, MJCF path, ``MjSpec`` or
    ``MjModel`` (used as is) and defaults to the bundled xArm6 scene.
    """
    def __init__(self, model_path: str | Path | mujoco.MjSpec | mujoco.MjModel = _MODEL_PATH,
                 timestep: float | None = None,
                 viewer: bool = False, *, viewer_hz: float = 30.0,
                 low_power_rendering: bool = False):
        if not np.isfinite(viewer_hz) or viewer_hz <= 0:
            raise ValueError("viewer_hz must be finite and positive")
        self.model = load_model(model_path)
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
            if name in GRIPPER_ACTUATOR_NAMES:
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

        self._gripper_spec: GripperSpec | None = identify_gripper(self.model)
        if (
            self._gripper_actuator_id is not None
            and self._gripper_spec is not None
            and self._gripper_spec.kind == "jaw"
            and self._gripper_spec.drive_range_rad[0] != self._gripper_spec.drive_range_rad[1]
        ):
            self._gripper_open_ctrl = float(self._gripper_spec.drive_range_rad[0])
            self._gripper_close_ctrl = float(self._gripper_spec.drive_range_rad[1])

        self._g2_model: G2Model | None = None
        self._g2_controller: G2Controller | None = None
        self._g2_driver_qpos_adr: int | None = None
        self._g2_driver_dof_adr: int | None = None
        self._g2_last_output: ControlOutput | None = None
        self._bio_g2_adapter: MuJoCoBioG2Adapter | None = None
        self._bio_g2_controller: BioGripperG2Controller | None = None
        self._bio_g2_last_output: BioG2ControlOutput | None = None
        self._last_active_gripper_speed_percent: float = 100.0
        self._step_hooks: list[Callable[[], None]] = []

        self._arm_q_lo = np.full(self._n_arm_qpos, -np.inf, dtype=np.float64)
        self._arm_q_hi = np.full(self._n_arm_qpos, np.inf, dtype=np.float64)
        for j in range(self.model.njnt):
            jtype = int(self.model.jnt_type[j])
            if jtype in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)) and int(
                self.model.jnt_limited[j]
            ):
                adr = int(self.model.jnt_qposadr[j])
                if 0 <= adr < self._n_arm_qpos:
                    self._arm_q_lo[adr] = float(self.model.jnt_range[j, 0])
                    self._arm_q_hi[adr] = float(self.model.jnt_range[j, 1])

        # --- Initialise state ---
        self._viewer = None
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # home
        else:
            mujoco.mj_resetData(self.model, self.data)
        # Keyframes define only the robot joints; restore floating scene props
        # (e.g. the pick-place cube) to their authored qpos0 pose.
        for j in range(self.model.njnt):
            if int(self.model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                adr = self.model.jnt_qposadr[j]
                self.data.qpos[adr:adr + 7] = self.model.qpos0[adr:adr + 7]
        self.data.ctrl[:self._n_arm_actuators] = self.data.qpos[:self._n_arm_qpos]

        if (
            self._gripper_spec is not None
            and self._gripper_spec.name == "gripper_g2"
            and self._gripper_actuator_id is not None
        ):
            self._gripper_open_ctrl = GRIPPER_G2_SIM_OPEN_DRIVE
            self._gripper_close_ctrl = GRIPPER_G2_SIM_CLOSE_DRIVE
            drv_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint")
            if drv_jnt >= 0:
                self._g2_driver_qpos_adr = int(self.model.jnt_qposadr[drv_jnt])
                self._g2_driver_dof_adr = int(self.model.jnt_dofadr[drv_jnt])
                # Allow G2Controller's integral preload + velocity feedforward bias beyond nominal stroke range
                self.model.actuator_ctrllimited[self._gripper_actuator_id] = 0
                init_q = float(self.data.qpos[self._g2_driver_qpos_adr])
                init_gap_mm = gripper_g2_sim_drive_to_gap_m(init_q) * 1000.0
                self._g2_model = G2Model.load()
                self._g2_controller = G2Controller(self._g2_model, initial_position_mm=init_gap_mm)
                self._g2_controller.command(init_gap_mm, speed_percent=100.0, force_percent=100.0)
                limit = float(self._g2_controller.torque_limit_nm[0])
                self.model.actuator_forcerange[self._gripper_actuator_id] = (-limit, limit)
                self.data.ctrl[self._gripper_actuator_id] = gripper_g2_gap_m_to_sim_drive(init_gap_mm / 1000.0)
        elif (
            self._gripper_spec is not None
            and self._gripper_spec.name == "bio_gripper_g2"
            and self._gripper_actuator_id is not None
        ):
            self._gripper_open_ctrl = BIO_G2_STROKE_M
            self._gripper_close_ctrl = 0.0
            self._bio_g2_adapter = MuJoCoBioG2Adapter(
                self.model, self.data, self._gripper_actuator_id
            )
            qr, _, ql, _ = self._bio_g2_adapter.state()
            init_gap_mm = float(np.clip(bio_g2_q_to_gap(qr, ql), BIO_G2_MIN_GAP_MM, BIO_G2_MAX_GAP_MM))
            self._bio_g2_controller = BioGripperG2Controller(self._bio_g2_adapter)
            self._bio_g2_controller.reset(init_gap_mm)
            self._bio_g2_controller.command(init_gap_mm, speed_percent=100.0, force_percent=100.0)

        mujoco.mj_forward(self.model, self.data)
        self._ft_sensor: AI1500Reader | None = None
        self._ft_sensor_last_reading: SensorReading | None = None
        if has_ft_sensor(self.model):
            if not np.isclose(self.model.opt.timestep, 0.001, rtol=0.0, atol=1e-12):
                raise ValueError("FT sensor models require timestep=0.001 s")
            self._ft_sensor = AI1500Reader(self.model, self.data)
            self._update_ft_sensor()
        if viewer:
            self._viewer = PassiveViewer(self.model, self.data)
        # wall-clock origin for real-time sync (only used when viewer is open)
        self._wall_start = time.monotonic() - self.data.time
        self._viewer_interval = 1.0 / viewer_hz
        self._last_viewer_sync = time.monotonic()

    # -- arm state ---------------------------------------------------------
    @property
    def n_arm(self) -> int:
        """Arm joint count, excluding gripper joints."""
        return self._n_arm_qpos

    @property
    def n_arm_actuators(self) -> int:
        """Number of arm actuators (excluding gripper actuator)."""
        return self._n_arm_actuators

    @property
    def has_viewer(self) -> bool:
        """Whether an interactive viewer window was opened for this session."""
        return self._viewer is not None

    @property
    def viewer_is_running(self) -> bool:
        """Whether an interactive viewer is currently open and running."""
        return self._viewer is not None and self._viewer.is_running()

    def sync_viewer(self) -> None:
        """Synchronize the passive viewer immediately if one is open."""
        if self._viewer is not None:
            self._viewer.sync()


    def _require_arm_q(self, q: np.ndarray, name: str) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape != (self.n_arm,):
            raise ValueError(f"{name} must have shape ({self.n_arm},), got {q.shape}")
        if not np.all(np.isfinite(q)):
            raise ValueError(f"{name} must contain only finite values")
        if self.n_arm > 0 and (
            np.any(q < self._arm_q_lo - 1e-4) or np.any(q > self._arm_q_hi + 1e-4)
        ):
            raise ValueError(
                f"{name} exceeds arm joint limits [{self._arm_q_lo.tolist()}, {self._arm_q_hi.tolist()}], "
                f"got {q.tolist()}"
            )
        return q

    def get_joint_positions(self) -> np.ndarray:
        """Arm joint positions only (excludes gripper joints)."""
        return self.data.qpos[:self._n_arm_qpos].copy()

    def get_tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """TCP pose: (position (3,), orientation quaternion wxyz (4,))."""
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
        if (
            self._g2_controller is not None
            and self._gripper_actuator_id is not None
            and self._g2_driver_qpos_adr is not None
            and self._g2_driver_dof_adr is not None
        ):
            q_drv = float(self.data.qpos[self._g2_driver_qpos_adr])
            v_drv = float(self.data.qvel[self._g2_driver_dof_adr])
            meas_gap_mm = measured_g2_pad_gap_mm(self.model, self.data)
            dt = float(self.model.opt.timestep)
            output = self._g2_controller.update(
                q_drv, v_drv, dt, measured_gap_mm=meas_gap_mm
            )
            self._g2_last_output = output
            limit = float(output.torque_limit_nm[0])
            self.model.actuator_forcerange[self._gripper_actuator_id] = (-limit, limit)
            vel_bias = (self._g2_controller.kv / self._g2_controller.kp) * float(output.velocity_rad_s[0])
            self.data.ctrl[self._gripper_actuator_id] = float(output.position_rad[0]) + vel_bias
        elif self._bio_g2_controller is not None:
            dt = float(self.model.opt.timestep)
            self._bio_g2_last_output = self._bio_g2_controller.apply(dt)
        mujoco.mj_step(self.model, self.data)
        self.data.qfrc_applied[:] = 0.0
        for hook in tuple(self._step_hooks):
            hook()
        if self._ft_sensor is not None:
            last_time = self._ft_sensor._last_time
            needs_reset = (
                last_time is not None
                and abs(self.data.time - last_time - self.model.opt.timestep) > 1e-9
                and abs(self.data.time - last_time) > 1e-9
            )
            self._update_ft_sensor(reset=needs_reset)
        # real-time sync: sleep until wall-clock catches up with sim time
        if self._viewer is not None:
            wall_elapsed = time.monotonic() - self._wall_start
            sim_ahead = self.data.time - wall_elapsed
            if sim_ahead > 0:
                time.sleep(sim_ahead)
        self._sync()

    def step(self) -> None:
        """Advance simulation by one physics step with arm gravity compensation."""
        self._step()

    @contextmanager
    def step_hook(self, callback: Callable[[], None]):
        """Register ``callback`` to run immediately after each ``mj_step`` inside the ``with`` block."""
        self._step_hooks.append(callback)
        try:
            yield
        finally:
            if callback in self._step_hooks:
                self._step_hooks.remove(callback)

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

    def settle_until(self, condition, timeout: float) -> bool:
        """Step physics until ``condition()`` holds; return False on timeout."""
        t0 = self.data.time
        while self.data.time - t0 < timeout:
            if self._viewer is not None and not self._viewer.is_running():
                return False
            self._step()
            if condition():
                self._sync(force=True)
                return True
        self._sync(force=True)
        return False

    # -- arm commands ------------------------------------------------------
    def set_joint_positions(self, q: np.ndarray) -> None:
        """Set arm joint targets only (does not affect gripper)."""
        q = self._require_arm_q(q, "q")
        self.data.ctrl[:self.n_arm] = q

    def move_joints(self, q_goal: np.ndarray, duration: float, wait: bool = True) -> None:
        """Move the arm to ``q_goal`` with quintic interpolation over ``duration`` s, from the last command."""
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive")
        q_goal = self._require_arm_q(q_goal, "q_goal")
        # Continue the previous position command. Restarting from measured
        # qpos would introduce a target jump whenever the servo still lags.
        q0 = self.data.ctrl[:self.n_arm].copy()
        n_steps = int(np.ceil(duration / self.model.opt.timestep))
        times = np.minimum(np.arange(1, n_steps + 1) * self.model.opt.timestep, duration)
        traj = _quintic(times, q0, q_goal, duration)
        for q in traj:
            self.data.ctrl[:self.n_arm] = q
            self._step()
            if self._viewer is not None and not self._viewer.is_running():
                return
        if wait:
            self.settle(0.3)

    def move_tcp(self, pos: np.ndarray, quat: np.ndarray, duration: float,
                 wait: bool = True, *, steps: int = 50) -> None:
        """Move along a straight Cartesian path with shortest-path rotation."""
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive")
        pos = np.asarray(pos, dtype=float)
        if pos.shape != (3,) or not np.all(np.isfinite(pos)):
            raise ValueError(f"pos must be a finite 3-element vector, got {pos.shape}")
        pos0, quat0 = self.get_tcp_pose()
        s = np.linspace(0.0, 1.0, steps + 1)[1:]
        waypoints = pos0 + s[:, None] * (pos - pos0)
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
        q_traj, ok = solve_ik_trajectory(self.model, self.eval_data, self._tcp_site_name(),
                                         waypoints, quat_traj, q_start)
        if not ok:
            raise RuntimeError(f"IK did not converge for target pos={pos} quat={quat}")
        seg = max(duration / steps, 4 * self.model.opt.timestep)
        for q in q_traj:
            # IK returns full qpos; extract arm portion for move_joints
            self.move_joints(q[:self.n_arm], seg, wait=False)
        if wait:
            self.settle(0.3)

    def execute_trajectory(self, traj: JointTrajectory, wait: bool = True) -> None:
        """Execute a time-parameterized JointTrajectory.

        Interpolates control targets at simulation timestep resolution model.opt.timestep.
        """
        if traj.n_joints != self.n_arm:
            raise ValueError(
                f"trajectory n_joints must be {self.n_arm}, got {traj.n_joints}"
            )
        if traj.n_steps == 0:
            return
        if self.n_arm > 0 and (
            np.any(traj.q < self._arm_q_lo[None, :] - 1e-4)
            or np.any(traj.q > self._arm_q_hi[None, :] + 1e-4)
        ):
            raise ValueError("trajectory contains joint positions outside arm joint limits")

        dt_sim = self.model.opt.timestep
        t_start = float(traj.times[0])
        t_end = float(traj.times[-1])
        cur_t = t_start

        while cur_t <= t_end + 0.5 * dt_sim:
            q_target, _, _ = traj.sample(cur_t)
            self.data.ctrl[:self.n_arm] = q_target
            self._step()
            if self._viewer is not None and not self._viewer.is_running():
                return
            cur_t += dt_sim

        # Ensure exact final configuration
        self.data.ctrl[:self.n_arm] = traj.q[-1]
        if wait:
            self.settle(0.3)


    def ik_trajectory(self, waypoints_pos: np.ndarray, quat: np.ndarray | None,
                      pos_tol: float = 5e-5) -> tuple[np.ndarray, bool]:
        """IK along Cartesian waypoints using the isolated eval_data."""
        # Use full qpos so IK operates on the complete kinematic chain
        return solve_ik_trajectory(self.model, self.eval_data, self._tcp_site_name(),
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

    def solve_tcp_velocity(self, twist: np.ndarray, q: np.ndarray | None = None, *,
                           damping: float = 1e-3) -> np.ndarray:
        """Velocity-level IK: arm joint velocity ``(n_arm,)`` for desired TCP twist [v; omega] (world)."""
        q_full = self._eval_qpos(q, "q")
        qd_full = solve_joint_velocity(self.model, self.eval_data,
                                       self.model.site(self._site).name, twist, q_full,
                                       damping=damping)
        return qd_full[:self._n_arm_dof]

    def tcp_manipulability(self, q: np.ndarray | None = None, *,
                           include_rotation: bool = True) -> float:
        """Yoshikawa manipulability of the TCP; near 0 at singularities."""
        q_full = self._eval_qpos(q, "q")
        return manipulability(self.model, self.eval_data,
                              self.model.site(self._site).name, q_full,
                              include_rotation=include_rotation)

    # -- gripper control ---------------------------------------------------
    @property
    def has_gripper(self) -> bool:
        """Whether a gripper actuator is present."""
        return self._gripper_actuator_id is not None

    @property
    def gripper_kind(self) -> str:
        """``none``, ``jaw``, or ``vacuum``."""
        if self._gripper_spec is None:
            return "none"
        return self._gripper_spec.kind

    @property
    def g2_controller(self) -> G2Controller | None:
        """Calibrated Gripper G2 controller instance when Gripper G2 is mounted."""
        return self._g2_controller

    @property
    def bio_g2_controller(self) -> BioGripperG2Controller | None:
        """Calibrated Bio Gripper G2 controller instance when Bio Gripper G2 is mounted."""
        return self._bio_g2_controller

    # -- force/torque sensor -----------------------------------------------
    @property
    def has_ft_sensor(self) -> bool:
        """Whether an AI1500 6-axis force/torque sensor is present in the model."""
        return self._ft_sensor is not None

    @property
    def ft_sensor(self) -> AI1500Reader | None:
        """Attached AI1500Reader instance, or None when no sensor is present."""
        return self._ft_sensor

    @property
    def ft_sensor_reading(self) -> SensorReading | None:
        """Most recent 6-axis sensor reading, or None when no sensor is present."""
        return self._ft_sensor_last_reading

    def _update_ft_sensor(self, *, reset: bool = False, zero: bool = False) -> SensorReading:
        if self._ft_sensor is None:
            raise NotImplementedError("This robot has no force/torque sensor")
        if self._n_arm_dof > 0:
            # Match _step()'s gravity-compensated arm state so mj_forward does not
            # evaluate F/T readings as if the arm were free-falling without compensation.
            mujoco.mj_forward(self.model, self.data)
            self.data.qfrc_applied[: self._n_arm_dof] = self.data.qfrc_bias[: self._n_arm_dof]
        try:
            if zero:
                self._ft_sensor.zero()
            if reset:
                self._ft_sensor.reset()
            reading = self._ft_sensor.update()
        finally:
            if self._n_arm_dof > 0:
                self.data.qfrc_applied[: self._n_arm_dof] = 0.0
        self._ft_sensor_last_reading = reading
        return reading

    def get_ft_sensor_reading(self) -> SensorReading:
        """Refresh and return the current 6-axis force/torque sensor reading."""
        return self._update_ft_sensor()

    def reset_ft_sensor(self) -> SensorReading:
        """Reset the 200 Hz sample schedule and take an immediate reading."""
        return self._update_ft_sensor(reset=True)

    def zero_ft_sensor(self) -> SensorReading:
        """Tare the F/T sensor at the current state so ``ext`` reads zero."""
        return self._update_ft_sensor(reset=True, zero=True)

    @property
    def gripper_control_output(self) -> ControlOutput | BioG2ControlOutput | None:
        """Most recent control output from the calibrated Gripper G2 or Bio Gripper G2 controller."""
        if self._g2_last_output is not None:
            return self._g2_last_output
        return self._bio_g2_last_output

    @property
    def gripper_actuator_id(self) -> int | None:
        """Actuator index of the attached gripper, or ``None`` if none is present."""
        return self._gripper_actuator_id

    @property
    def gripper_ctrl_range(self) -> tuple[float, float] | None:
        """Actuator control range ``(lo, hi)`` of the attached gripper, or ``None``."""
        if self._gripper_ctrl_range is None:
            return None
        return (float(self._gripper_ctrl_range[0]), float(self._gripper_ctrl_range[1]))

    @property
    def gripper_qpos(self) -> np.ndarray:
        """Non-arm joint positions (e.g. gripper mechanism joints), copied as 1-D array."""
        return self.data.qpos[self._n_arm_qpos:].copy()

    def gripper_wait_duration(self) -> float:
        """Recommended settle duration in seconds for the current gripper command."""
        return self._g2_wait_duration()

    def get_gripper_position(self) -> float:
        """Current gripper target (0.0 = fully open, 1.0 = fully closed)."""
        if not self.has_gripper:
            raise NotImplementedError("This robot has no gripper")
        if (self._g2_controller is not None or self._bio_g2_controller is not None) and self._gripper_spec is not None:
            return gap_to_position(self._gripper_spec, self.get_gripper_gap())
        ctrl = self.data.ctrl[self._gripper_actuator_id]
        span = self._gripper_close_ctrl - self._gripper_open_ctrl
        if abs(span) < 1e-12:
            return 0.0
        return float(np.clip((ctrl - self._gripper_open_ctrl) / span, 0.0, 1.0))

    def _resolve_gripper_speed_force(
        self,
        cur_speed: float,
        cur_force: float,
        speed_percent: float | None,
        force_percent: float | None,
    ) -> tuple[float, float]:
        if speed_percent is None:
            spd = self._last_active_gripper_speed_percent if cur_speed <= 0.0 else cur_speed
        else:
            spd = float(speed_percent)
            if not math.isfinite(spd) or not (0.0 <= spd <= 100.0):
                raise ValueError(f"speed_percent must be finite in [0, 100], got {speed_percent!r}")
            if spd > 0.0:
                self._last_active_gripper_speed_percent = spd
        if force_percent is None:
            frc = cur_force
        else:
            frc = float(force_percent)
            if not math.isfinite(frc) or not (1.0 <= frc <= 100.0):
                raise ValueError(f"force_percent must be finite in [1, 100], got {force_percent!r}")
        return spd, frc

    def set_gripper(
        self,
        pos: float,
        *,
        speed_percent: float | None = None,
        force_percent: float | None = None,
    ) -> None:
        """Set gripper target (0.0 = fully open, 1.0 = fully closed).

        For Gripper G2 / Bio Gripper G2, ``speed_percent=0`` pauses jaw motion while
        holding contact preload; subsequent calls with ``speed_percent=None`` automatically
        resume at the last positive speed percent.
        """
        if not self.has_gripper:
            raise NotImplementedError("This robot has no gripper")
        pos_f = float(pos)
        if not math.isfinite(pos_f):
            raise ValueError(f"gripper position must be finite, got {pos!r}")
        pos_clamped = float(np.clip(pos_f, 0.0, 1.0))
        if (self._g2_controller is not None or self._bio_g2_controller is not None) and self._gripper_spec is not None:
            gap_m = position_to_gap(self._gripper_spec, pos_clamped)
            self.set_gripper_gap(gap_m, speed_percent=speed_percent, force_percent=force_percent)
            return
        if speed_percent is not None and not math.isfinite(float(speed_percent)):
            raise ValueError(f"speed_percent must be finite, got {speed_percent!r}")
        if force_percent is not None and not math.isfinite(float(force_percent)):
            raise ValueError(f"force_percent must be finite, got {force_percent!r}")
        target_ctrl = self._gripper_open_ctrl + pos_clamped * (self._gripper_close_ctrl - self._gripper_open_ctrl)
        self.data.ctrl[self._gripper_actuator_id] = target_ctrl

    def get_gripper_gap(self, *, measured: bool = False) -> float:
        """Finger gap in meters: the commanded target, or the physical pad gap when ``measured``."""
        spec = self._require_jaw()
        if measured:
            return self._measured_gripper_gap(spec)
        if self._g2_controller is not None:
            return float(self._g2_controller.position_mm[0] / 1000.0)
        if self._bio_g2_controller is not None:
            return float(self._bio_g2_controller.position_mm / 1000.0)
        return position_to_gap(spec, self.get_gripper_position())

    def _measured_gripper_gap(self, spec: GripperSpec) -> float:
        if self._g2_controller is not None:
            return measured_g2_pad_gap_mm(self.model, self.data) / 1000.0
        if self._bio_g2_controller is not None or spec.name == "bio_gripper_g2":
            return measured_bio_g2_gap_m(self.model, self.data)
        drive_qpos = float(self.data.qpos[self._n_arm_qpos])
        if (
            self._gripper_actuator_id is not None
            and int(self.model.actuator_trntype[self._gripper_actuator_id]) == int(mujoco.mjtTrn.mjTRN_JOINT)
        ):
            jnt_id = int(self.model.actuator_trnid[self._gripper_actuator_id, 0])
            drive_qpos = float(self.data.qpos[int(self.model.jnt_qposadr[jnt_id])])
        return drive_to_gap(
            spec,
            drive_qpos,
            open_ctrl=self._gripper_open_ctrl,
            close_ctrl=self._gripper_close_ctrl,
        )

    def set_gripper_gap(
        self,
        gap_m: float,
        *,
        speed_percent: float | None = None,
        force_percent: float | None = None,
    ) -> None:
        """Command a finger gap in meters, with optional speed/force percent for Gripper G2 / Bio Gripper G2.

        ``speed_percent=0`` pauses jaw motion while holding contact preload; subsequent
        commands omitting ``speed_percent`` resume at the last positive speed percent.
        """
        if self.gripper_kind == "vacuum":
            raise NotImplementedError("vacuum gripper has no jaw gap")
        gap_f = float(gap_m)
        if not math.isfinite(gap_f):
            raise ValueError(f"gap_m must be finite, got {gap_m!r}")
        spec = self._require_jaw()
        norm_pos = gap_to_position(spec, gap_f)
        if self._g2_controller is not None and self._gripper_actuator_id is not None:
            spd, frc = self._resolve_gripper_speed_force(
                float(self._g2_controller.speed_percent[0]),
                float(self._g2_controller.force_percent[0]),
                speed_percent,
                force_percent,
            )
            self._g2_controller.command(gap_f * 1000.0, speed_percent=spd, force_percent=frc)
            limit = float(self._g2_controller.torque_limit_nm[0])
            self.model.actuator_forcerange[self._gripper_actuator_id] = (-limit, limit)
            self.data.ctrl[self._gripper_actuator_id] = gripper_g2_gap_m_to_sim_drive(gap_f)
            return
        if self._bio_g2_controller is not None:
            spd, frc = self._resolve_gripper_speed_force(
                float(self._bio_g2_controller.speed_percent),
                float(self._bio_g2_controller.force_percent),
                speed_percent,
                force_percent,
            )
            self._bio_g2_controller.command(gap_f * 1000.0, speed_percent=spd, force_percent=frc)
            return
        self.set_gripper(norm_pos, speed_percent=speed_percent, force_percent=force_percent)

    def bilateral_finger_contact(self, object_body: str) -> bool:
        """Whether both finger pads touch ``object_body``."""
        spec = self._require_jaw()
        return bilateral_finger_contact(self.model, self.data, spec, object_body)

    def require_bilateral_contact(self, object_body: str) -> None:
        """Raise GraspError unless both finger pads touch ``object_body``."""
        spec = self._require_jaw()
        require_bilateral_contact(self.model, self.data, spec, object_body)

    def cup_contact(self, object_body: str) -> bool:
        """Whether the vacuum cup touches ``object_body``."""
        if self.gripper_kind != "vacuum" or self._gripper_spec is None:
            raise NotImplementedError("This robot has no vacuum cup")
        return bodies_in_contact(
            self.model, self.data, self._gripper_spec.cup_body, object_body,
        )


    def _g2_wait_duration(self) -> float:
        if self._g2_controller is not None:
            spd_mm_s = float(self._g2_controller.requested_speed_mm_s[0])
            if spd_mm_s <= 0.0:
                return 0.5
            travel_mm = abs(float(self._g2_controller.position_mm[0] - self._g2_controller.reference_gap_mm[0]))
            return max(0.5, travel_mm / spd_mm_s + 0.12)
        if self._bio_g2_controller is not None:
            if self._bio_g2_controller.paused or self._bio_g2_controller.speed_mm_s <= 0.0:
                return 0.5
            gap_spd_mm_s = 2.0 * float(self._bio_g2_controller.speed_mm_s)
            travel_mm = abs(
                float(self._bio_g2_controller.position_mm - bio_g2_q_to_gap(self._bio_g2_controller.reference_q_m))
            )
            return max(0.5, travel_mm / gap_spd_mm_s + 0.12)
        return 0.5

    def open_gripper(
        self,
        wait: bool = True,
        *,
        speed_percent: float | None = None,
        force_percent: float | None = None,
    ) -> None:
        """Fully open the gripper."""
        if speed_percent is None and force_percent is None:
            self.set_gripper(0.0)
        else:
            self.set_gripper(0.0, speed_percent=speed_percent, force_percent=force_percent)
        if wait:
            self.settle(self._g2_wait_duration())

    def close_gripper(
        self,
        wait: bool = True,
        *,
        speed_percent: float | None = None,
        force_percent: float | None = None,
    ) -> None:
        """Fully close the gripper."""
        if speed_percent is None and force_percent is None:
            self.set_gripper(1.0)
        else:
            self.set_gripper(1.0, speed_percent=speed_percent, force_percent=force_percent)
        if wait:
            self.settle(self._g2_wait_duration())

    # -- misc --------------------------------------------------------------

    def close(self) -> None:
        """Close the viewer if one is open."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
