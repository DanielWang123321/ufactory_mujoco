"""MuJoCo backend for RobotInterface: position-actuated arm with quintic
joint interpolation, optional passive viewer, and IK-based TCP moves.
"""

import time
from pathlib import Path

import mujoco
import numpy as np

from .ik import solve_ik, solve_ik_trajectory
from .robot_interface import RobotInterface
from .viewer import PassiveViewer

_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "xarm6" / "scene.xml"


def _quintic(t: np.ndarray, q0: np.ndarray, q1: np.ndarray, T: float) -> np.ndarray:
    """Zero vel/acc quintic interpolation, t in [0, T]."""
    s = np.clip(t / T, 0.0, 1.0)
    s5 = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    return q0 + (q1 - q0) * s5[:, None]


class SimRobot(RobotInterface):
    def __init__(self, model_path: str | Path = _MODEL_PATH, timestep: float | None = None,
                 viewer: bool = False):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        if timestep is not None:
            self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        # separate MjData for IK/FK evaluation: never touched by the viewer thread
        self.eval_data = mujoco.MjData(self.model)
        self._site = self.model.site("attachment_site").id
        self._viewer = None
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # home
        self.data.ctrl[:] = self.data.qpos[:]
        mujoco.mj_forward(self.model, self.data)
        if viewer:
            self._viewer = PassiveViewer(self.model, self.data)
        # wall-clock origin for real-time sync (only used when viewer is open)
        self._wall_start = time.monotonic() - self.data.time

    # -- state -------------------------------------------------------------
    def get_joint_positions(self) -> np.ndarray:
        return self.data.qpos.copy()

    def get_tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.site_xpos[self._site].copy()
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self._site])
        return pos, quat

    # -- stepping ----------------------------------------------------------
    def _sync(self) -> None:
        if self._viewer is not None:
            self._viewer.sync()

    def _step(self) -> None:
        """One physics step with gravity-compensation feedforward (like real
        xArm position mode, which has model-based compensation built in)."""
        self.data.qfrc_applied[:] = self.data.qfrc_bias
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
            self._step()
            if condition():
                return

    def settle(self, duration: float = 1.0) -> None:
        """Step physics for a while (let the arm reach current ctrl)."""
        self._step_until(lambda: False, duration)

    # -- commands ----------------------------------------------------------
    def set_joint_positions(self, q: np.ndarray) -> None:
        self.data.ctrl[:] = q

    def move_joints(self, q_goal: np.ndarray, duration: float, wait: bool = True) -> None:
        q0 = self.data.qpos.copy()
        q_goal = np.asarray(q_goal, dtype=float)
        n_steps = int(duration / self.model.opt.timestep)
        traj = _quintic(np.arange(n_steps + 1) * self.model.opt.timestep, q0, q_goal, duration)
        for q in traj:
            self.data.ctrl[:] = q
            self._step()
            if self._viewer is not None and not self._viewer.is_running():
                return
        if wait:
            self.settle(0.3)

    def move_tcp(self, pos: np.ndarray, quat: np.ndarray, duration: float,
                 wait: bool = True, *, steps: int = 50) -> None:
        pos0, quat0 = self.get_tcp_pose()
        s = np.linspace(0.0, 1.0, steps + 1)[1:]
        waypoints = pos0 + s[:, None] * (np.asarray(pos) - pos0)
        quat_traj = None
        if quat is not None:
            quat_goal = np.asarray(quat, dtype=float)
            q0_conj = np.array([quat0[0], -quat0[1], -quat0[2], -quat0[3]])
            q_err = np.empty(4)
            mujoco.mju_mulQuat(q_err, quat_goal, q0_conj)  # goal * conj(start)
            vel_full = np.empty(3)
            mujoco.mju_quat2Vel(vel_full, q_err, 1.0)
            quat_traj = np.empty((steps, 4))
            for i in range(steps):
                quat_traj[i] = quat0
                mujoco.mju_quatIntegrate(quat_traj[i], vel_full, s[i])
        # solve IK per waypoint on the isolated eval_data (viewer thread never
        # touches it), then track the joint path with quintic segments
        q_start = self.data.qpos.copy()
        q_traj, ok = solve_ik_trajectory(self.model, self.eval_data, "attachment_site",
                                         waypoints, quat_traj, q_start)
        if not ok:
            raise RuntimeError(f"IK did not converge for target pos={pos} quat={quat}")
        seg = max(duration / steps, 4 * self.model.opt.timestep)
        for q in q_traj:
            self.move_joints(q, seg, wait=False)
        if wait:
            self.settle(0.3)

    def ik_trajectory(self, waypoints_pos: np.ndarray, quat: np.ndarray | None,
                      pos_tol: float = 5e-5) -> tuple[np.ndarray, bool]:
        """IK along Cartesian waypoints using the isolated eval_data."""
        return solve_ik_trajectory(self.model, self.eval_data, "attachment_site",
                                   np.asarray(waypoints_pos, dtype=float), quat,
                                   self.get_joint_positions(), pos_tol=pos_tol)

    def fk_positions(self, q_traj: np.ndarray) -> np.ndarray:
        """FK of a joint trajectory (N,nv) -> TCP positions (N,3) on eval_data."""
        out = np.empty((len(q_traj), 3))
        for i, q in enumerate(q_traj):
            self.eval_data.qpos[:] = q
            mujoco.mj_forward(self.model, self.eval_data)
            out[i] = self.eval_data.site_xpos[self._site]
        return out

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
