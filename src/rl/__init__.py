"""Reinforcement-learning task environments (roadmap 0.0.5).

Requires only mujoco + numpy; gymnasium is optional (fallback base class) and
stable-baselines3 is required for training only (requirements-rl.txt).
"""

from .reach_env import (
    TARGET_HIGH,
    TARGET_LOW,
    XArm6ReachEnv,
)

__all__ = ["XArm6ReachEnv", "TARGET_LOW", "TARGET_HIGH"]
