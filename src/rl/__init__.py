"""Reinforcement-learning task environments (roadmap 0.0.5, 0.0.7).

Requires only mujoco + numpy; gymnasium is optional (fallback base class) and
stable-baselines3 is required for training only (requirements-rl.txt).
"""

from .reach_env import (
    ROBOT_PROFILES,
    RobotProfile,
    ReachEnv,
    TARGET_HIGH,
    TARGET_LOW,
    XArm6ReachEnv,
)

__all__ = [
    "ReachEnv",
    "XArm6ReachEnv",
    "RobotProfile",
    "ROBOT_PROFILES",
    "TARGET_LOW",
    "TARGET_HIGH",
]
