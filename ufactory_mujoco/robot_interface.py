"""Abstract robot control interface shared by ``SimRobot`` (MuJoCo) and ``XArmRobot`` (hardware).

Task code programs against ``RobotInterface`` so the same sequence runs in simulation and on a real arm.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .ft_sensor import SensorReading
    from .trajectory import JointTrajectory


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

    # -- lifecycle ----------------------------------------------------------
    def close(self) -> None:
        """Release backend resources (optional)."""
