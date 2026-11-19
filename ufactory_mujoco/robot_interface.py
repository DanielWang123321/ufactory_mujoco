"""Abstract robot control interface shared by ``SimRobot`` (MuJoCo) and ``XArmRobot`` (hardware).

Task code programs against ``RobotInterface`` so the same sequence runs in simulation and on a real arm.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from .kinematics import fk_pose, solve_ik
from .trajectory import DEFAULT_MAX_ROT_VEL, JointTrajectory, plan_cartesian_line, plan_ptp_quintic

if TYPE_CHECKING:
    from .ft_sensor import SensorReading
    from .grippers import GripperSpec


class RobotInterface(ABC):
    """Arm interface with optional gripper (positions in meters/radians).

    Joint commands use this arm's ``n_arm`` length (5, 6, or 7). A vector of
    any other length is rejected.
    """

    @property
    @abstractmethod
    def n_arm(self) -> int:
        """Arm joint count, excluding gripper joints."""

    # -- arm state ----------------------------------------------------------
    @abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        """Current arm joint angles (excludes gripper), shape (n_arm,), rad."""

    @abstractmethod
    def set_joint_positions(self, q: np.ndarray) -> None:
        """Command arm joint target directly (no interpolation; sim: ctrl, real: sdk).

        ``q`` must have shape ``(n_arm,)``.
        """

    @abstractmethod
    def get_tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """TCP pose: (position (3,), orientation quaternion wxyz (4,))."""

    @abstractmethod
    def move_joints(self, q_goal: np.ndarray, duration: float, wait: bool = True) -> None:
        """Move arm to ``q_goal`` (shape ``(n_arm,)``) over ``duration`` seconds.

        Backend notes:
        - ``SimRobot`` steps a quintic trajectory for ``duration`` seconds (``wait=False``
          skips only the trailing ``settle``).
        - ``XArmRobot`` commands controller joint PTP capped by ``max_joint_speed_rad_s``
          (actual duration may exceed ``duration`` when speed-limited; ``wait=False``
          returns immediately after dispatching the SDK command).
        """

    @abstractmethod
    def move_tcp(self, pos: np.ndarray, quat: np.ndarray, duration: float,
                 wait: bool = True) -> None:
        """Move TCP to pose ``(pos (3,), quat wxyz (4,))``.

        Backend notes:
        - ``SimRobot`` interpolates a straight Cartesian path with shortest-path SLERP.
        - ``XArmRobot`` solves endpoint IK and moves in joint space via ``move_joints``.
        To execute identical straight-line Cartesian paths on both backends, use
        ``plan_cartesian_trajectory`` followed by ``execute_trajectory``.
        """

    @abstractmethod
    def execute_trajectory(self, traj: JointTrajectory, wait: bool = True) -> None:
        """Execute a pre-planned joint trajectory (JointTrajectory).

        ``traj.n_joints`` must equal ``n_arm``.
        """

    # -- gripper control ----------------------------------------------------
    @property
    def has_gripper(self) -> bool:
        """Whether a position-controlled jaw is attached."""
        return False

    @property
    def gripper_kind(self) -> str:
        """``none``, ``jaw``, or ``vacuum``."""
        return "none"

    def get_gripper_position(self) -> float:
        """Current gripper position (0.0 = fully open, 1.0 = fully closed)."""
        raise NotImplementedError("This robot has no gripper")

    def set_gripper(
        self,
        pos: float,
        *,
        speed_percent: float | None = None,
        force_percent: float | None = None,
    ) -> None:
        """Set gripper target position (0.0 = fully open, 1.0 = fully closed)."""
        raise NotImplementedError("This robot has no gripper")

    def get_gripper_gap(self, *, measured: bool = False) -> float:
        """Finger gap in meters: commanded by default, physical when ``measured``. Vacuum grippers have no gap."""
        raise NotImplementedError("This robot has no jaw gripper")

    def set_gripper_gap(
        self,
        gap_m: float,
        *,
        speed_percent: float | None = None,
        force_percent: float | None = None,
    ) -> None:
        """Command a finger gap in meters, with optional speed/force percent."""
        raise NotImplementedError("This robot has no jaw gripper")

    def bilateral_finger_contact(self, object_body: str) -> bool:
        """Whether both finger pads touch ``object_body``."""
        raise NotImplementedError("This robot has no jaw gripper")

    def require_bilateral_contact(self, object_body: str) -> None:
        """Raise GraspError unless both finger pads touch ``object_body``."""
        if not self.bilateral_finger_contact(object_body):
            from .grippers import GraspError
            raise GraspError(f"no bilateral finger-pad contact with {object_body}; refusing to lift")

    def cup_contact(self, object_body: str) -> bool:
        """Whether the vacuum cup touches ``object_body``."""
        raise NotImplementedError("This robot has no vacuum cup")

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
            self.settle(0.5)

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
            self.settle(0.5)

    def settle(self, duration: float = 1.0) -> None:
        """Step physics for a while (default no-op; overridden by sim)."""

    # -- force/torque sensor -----------------------------------------------
    @property
    def has_ft_sensor(self) -> bool:
        """Whether a 6-axis force/torque sensor is available on this robot."""
        return False

    def get_ft_sensor_reading(self) -> SensorReading:
        """Read the current 6-axis force/torque sensor wrench."""
        raise NotImplementedError("This robot has no force/torque sensor")

    # -- shared helpers on the backend's MuJoCo model (model, data, eval_data) -------
    def _sync_qpos(self) -> None:
        """Refresh ``data.qpos`` from the robot (hardware backends read the controller)."""

    def _tcp_site_name(self) -> str:
        return self.model.site(self._site).name

    def _require_jaw(self) -> GripperSpec:
        if self.gripper_kind == "vacuum":
            raise NotImplementedError("vacuum gripper has no jaw gap")
        if self._gripper_spec is None or self._gripper_spec.kind != "jaw":
            raise NotImplementedError("This robot has no jaw gripper")
        return self._gripper_spec

    def _eval_qpos(self, q: np.ndarray | None, name: str = "q") -> np.ndarray:
        """Resolve ``q`` (``None``, arm-only ``(n_arm,)``, or full ``(nq,)``) against current ``data.qpos``."""
        if q is None:
            self._sync_qpos()
            return self.data.qpos.copy()
        arr = np.asarray(q, dtype=np.float64)
        if arr.ndim != 1 or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be a finite 1-D array, got shape {arr.shape}")
        if len(arr) == self.model.nq:
            return arr.copy()
        if len(arr) == self.n_arm:
            full = self.data.qpos.copy()
            full[: self.n_arm] = arr
            return full
        raise ValueError(
            f"{name} must have length {self.n_arm} (n_arm) or {self.model.nq} (nq), got {len(arr)}"
        )

    def solve_tcp(self, pos, quat=None, q_init=None):
        """IK for a TCP pose seeded from the current state (or ``q_init``); returns ``(q, converged)``."""
        seed = self._eval_qpos(q_init, "q_init")
        return solve_ik(self.model, self.eval_data, self._tcp_site_name(),
                        np.asarray(pos, dtype=float),
                        None if quat is None else np.asarray(quat, dtype=float),
                        q_init=seed)

    def plan_joint_trajectory(self, q_goal: np.ndarray, max_vel: float | np.ndarray = 1.57,
                              max_acc: float | np.ndarray = 4.0,
                              duration: float | None = None) -> JointTrajectory:
        """Plan a time-parameterized quintic joint trajectory from the current joints to ``q_goal``."""
        q0 = self.get_joint_positions()
        return plan_ptp_quintic(q0, q_goal, max_vel=max_vel, max_acc=max_acc,
                                duration=duration, dt=self.model.opt.timestep)

    def plan_cartesian_trajectory(self, pos: np.ndarray, quat: np.ndarray | None = None,
                                  max_lin_vel: float = 0.20, max_lin_acc: float = 1.0,
                                  site_name: str | None = None, *,
                                  max_rot_vel: float = DEFAULT_MAX_ROT_VEL) -> tuple[JointTrajectory, bool]:
        """Plan an arm-joint straight Cartesian line from the current site pose; returns ``(trajectory, ok)``."""
        site = site_name if site_name is not None else self._tcp_site_name()
        if site_name is None:
            p0, quat0 = self.get_tcp_pose()
        else:
            self._sync_qpos()
            p0, quat0 = fk_pose(self.model, self.eval_data, site, self.data.qpos.copy())
        traj_full, ok = plan_cartesian_line(
            p0, quat0, pos, quat, self.model, site_name=site,
            max_lin_vel=max_lin_vel, max_lin_acc=max_lin_acc, max_rot_vel=max_rot_vel,
            q_init=self.data.qpos.copy(), dt=self.model.opt.timestep,
            eval_data=self.eval_data,
        )
        n = self.n_arm
        arm_traj = JointTrajectory(
            times=traj_full.times,
            q=traj_full.q[:, :n],
            qd=traj_full.qd[:, :n],
            qdd=traj_full.qdd[:, :n],
        )
        return arm_traj, ok

    # -- lifecycle ----------------------------------------------------------
    def close(self) -> None:
        """Release backend resources (optional)."""
