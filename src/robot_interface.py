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
    """Minimal 6-DoF arm interface (positions in meters/radians)."""

    @abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        """Current joint angles, shape (6,), rad."""

    @abstractmethod
    def set_joint_positions(self, q: np.ndarray) -> None:
        """Command joint target directly (no interpolation; sim: ctrl, real: sdk)."""

    @abstractmethod
    def get_tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """TCP pose: (position (3,), orientation quaternion wxyz (4,))."""

    @abstractmethod
    def move_joints(self, q_goal: np.ndarray, duration: float, wait: bool = True) -> None:
        """Move to q_goal with quintic interpolation over `duration` seconds."""

    @abstractmethod
    def move_tcp(self, pos: np.ndarray, quat: np.ndarray, duration: float,
                 wait: bool = True) -> None:
        """Move TCP to pose (pos (3,), quat wxyz (4,)) using IK, interpolated."""

    def close(self) -> None:
        """Release backend resources (optional)."""
