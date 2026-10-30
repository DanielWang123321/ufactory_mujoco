"""Abstract robot control interface.

Decouples examples from the concrete backend so a real xArm (via
xarm-python-sdk) can replace the MuJoCo simulation later:

    class XArmRobot(RobotInterface):
        def __init__(self, ip): from xarm.wrapper import XArmAPI; ...
        # implement the same methods with sdk calls

Examples program exclusively against RobotInterface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
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
        """Move arm to q_goal with quintic interpolation over `duration` seconds.

        ``q_goal`` must have shape ``(n_arm,)``.
        """

    @abstractmethod
    def move_tcp(self, pos: np.ndarray, quat: np.ndarray, duration: float,
                 wait: bool = True) -> None:
        """Move TCP to pose (pos (3,), quat wxyz (4,)) using IK, interpolated."""

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

    def get_gripper_gap(self) -> float:
        """Commanded finger gap in meters. Vacuum grippers have no gap."""
        raise NotImplementedError("This robot has no jaw gripper")

    def get_measured_gripper_gap(self) -> float:
        """Measured physical finger gap in meters."""
        return self.get_gripper_gap()

    def set_gripper_gap(
        self,
        gap_m: float,
        *,
        speed_percent: float | None = None,
        force_percent: float | None = None,
    ) -> None:
        """Command a finger gap in meters, with optional speed/force percent."""
        raise NotImplementedError("This robot has no jaw gripper")

    def set_gripper_command(
        self,
        position_mm: float,
        *,
        speed_percent: float = 50.0,
        force_percent: float = 50.0,
    ) -> None:
        """Command Gripper G2 target opening in mm with speed (%) and force (%)."""
        self.set_gripper_gap(
            float(position_mm) / 1000.0,
            speed_percent=speed_percent,
            force_percent=force_percent,
        )

    def bilateral_finger_contact(self, object_body: str) -> bool:
        """Whether both finger pads touch ``object_body``."""
        raise NotImplementedError("This robot has no jaw gripper")

    def require_bilateral_contact(self, object_body: str) -> None:
        """Raise GraspError unless both finger pads touch ``object_body``."""
        if not self.bilateral_finger_contact(object_body):
            from .grippers import GraspError
            raise GraspError(f"无双侧指垫接触，拒绝抬起: {object_body}")

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

    # -- lifecycle ----------------------------------------------------------
    def close(self) -> None:
        """Release backend resources (optional)."""
