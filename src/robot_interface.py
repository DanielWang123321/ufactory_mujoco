"""Abstract robot control interface.

Decouples examples from the concrete backend so a real xArm (via
xarm-python-sdk) can replace the MuJoCo simulation later:

    class XArmRobot(RobotInterface):
        def __init__(self, ip): from xarm.wrapper import XArmAPI; ...
        # implement the same methods with sdk calls

Examples program exclusively against RobotInterface.
"""

from abc import ABC, abstractmethod

import numpy as np


class RobotInterface(ABC):
    """Minimal 6-DoF arm interface with optional gripper (positions in meters/radians)."""

    # -- arm state ----------------------------------------------------------
    @abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        """Current arm joint angles (excludes gripper), shape (n_arm,), rad."""

    @abstractmethod
    def set_joint_positions(self, q: np.ndarray) -> None:
        """Command arm joint target directly (no interpolation; sim: ctrl, real: sdk)."""

    @abstractmethod
    def get_tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """TCP pose: (position (3,), orientation quaternion wxyz (4,))."""

    @abstractmethod
    def move_joints(self, q_goal: np.ndarray, duration: float, wait: bool = True) -> None:
        """Move arm to q_goal with quintic interpolation over `duration` seconds."""

    @abstractmethod
    def move_tcp(self, pos: np.ndarray, quat: np.ndarray, duration: float,
                 wait: bool = True) -> None:
        """Move TCP to pose (pos (3,), quat wxyz (4,)) using IK, interpolated."""

    # -- gripper control ----------------------------------------------------
    @property
    def has_gripper(self) -> bool:
        """Whether a gripper is attached."""
        return False

    def get_gripper_position(self) -> float:
        """Current gripper position (0.0 = fully open, 1.0 = fully closed)."""
        raise NotImplementedError("This robot has no gripper")

    def set_gripper(self, pos: float) -> None:
        """Set gripper target position (0.0 = fully open, 1.0 = fully closed)."""
        raise NotImplementedError("This robot has no gripper")

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

    def settle(self, duration: float = 1.0) -> None:
        """Step physics for a while (default no-op; overridden by sim)."""

    # -- lifecycle ----------------------------------------------------------
    def close(self) -> None:
        """Release backend resources (optional)."""
