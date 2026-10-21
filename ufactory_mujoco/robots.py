"""Typed registry for the five UFACTORY arms.

Identity, gripper compatibility, and the frozen Reach workspace live here.
Task YAML selects a robot and a scene; it does not override these values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .model_paths import MODELS_ROOT

SCENES = ("scene", "reach", "obstacles", "pick_place", "box_pack")

# Combined MJCF directories that exist today. A compatible gripper missing
# from this table is declared but not built yet.
_COMBO_DIRS: dict[tuple[str, str], str] = {
    ("lite6", "lite6_gripper"): "lite6_with_gripper",
    ("lite6", "lite6_vacuum_gripper"): "lite6_with_vacuum_gripper",
    ("xarm5", "gripper_g2"): "xarm5_with_gripper",
    ("xarm6", "gripper_g2"): "xarm6_with_gripper",
    ("xarm6", "bio_gripper_g2"): "xarm6_with_bio_gripper",
    ("xarm7", "gripper_g2"): "xarm7_with_gripper",
    ("uf850", "gripper_g2"): "uf850_with_gripper",
}

_PICK_PLACE_DIRS: dict[tuple[str, str], str] = {
    ("xarm5", "gripper_g2"): "xarm5_with_gripper",
    ("xarm6", "gripper_g2"): "xarm6_with_gripper",
    ("xarm7", "gripper_g2"): "xarm7_with_gripper",
    ("uf850", "gripper_g2"): "uf850_with_gripper",
    ("lite6", "lite6_gripper"): "lite6_with_gripper",
}


@dataclass(frozen=True)
class ReachWorkspace:
    """Frozen reach-workspace calibration (0.0.7). YAML cannot override this."""

    target_low: tuple[float, float, float]
    target_high: tuple[float, float, float]
    shoulder: tuple[float, float, float]
    reach_min: float
    reach_max: float


@dataclass(frozen=True)
class RobotProfile:
    """Reach-field view of an arm. ``n_joints`` equals ``ArmSpec.dof``."""

    robot: str
    n_joints: int
    target_low: tuple[float, float, float]
    target_high: tuple[float, float, float]
    shoulder: tuple[float, float, float]
    reach_min: float
    reach_max: float


@dataclass(frozen=True)
class ArmSpec:
    """Arm identity. ``grippers`` is the compatible set, not the built assets."""

    key: str
    dof: int
    tcp_site: str
    grippers: tuple[str, ...]
    reach: ReachWorkspace


def _arm(key: str, dof: int, grippers: tuple[str, ...], reach: ReachWorkspace) -> ArmSpec:
    return ArmSpec(key, dof, "attachment_site", grippers, reach)


# Values produced by scripts/measure_reach_workspace.py (FK cloud + IK
# reachability >= 90%). The xArm 6 entry keeps the production 0.0.5 box;
# its reach_max was raised 0.55 -> 0.56 so the radial band covers every box
# corner (the far bottom corner is 0.5586 m from the shoulder).
ARMS: dict[str, ArmSpec] = {
    "lite6": _arm(
        "lite6", 6, ("lite6_gripper", "lite6_vacuum_gripper"),
        ReachWorkspace((0.102, -0.2389, 0.0986), (0.2553, 0.2389, 0.4031),
                       (0.0, 0.0, 0.2435), 0.0918, 0.4228),
    ),
    "xarm5": _arm(
        "xarm5", 5, ("gripper_g2", "bio_gripper_g2"),
        ReachWorkspace((0.1386, -0.3631, 0.155), (0.3914, 0.3631, 0.5834),
                       (0.0, 0.0, 0.267), 0.1248, 0.6827),
    ),
    "xarm6": _arm(
        "xarm6", 6, ("gripper_g2", "bio_gripper_g2"),
        ReachWorkspace((0.25, -0.25, 0.05), (0.45, 0.25, 0.35),
                       (0.0, 0.0, 0.267), 0.15, 0.56),
    ),
    "xarm7": _arm(
        "xarm7", 7, ("gripper_g2", "bio_gripper_g2"),
        ReachWorkspace((0.199, -0.3954, 0.2102), (0.4231, 0.3954, 0.5341),
                       (0.0, 0.0, 0.267), 0.1791, 0.7015),
    ),
    "uf850": _arm(
        "uf850", 6, ("gripper_g2", "bio_gripper_g2"),
        ReachWorkspace((0.1956, -0.4869, 0.2071), (0.5182, 0.4869, 0.735),
                       (0.0, 0.0, 0.364), 0.1761, 0.8823),
    ),
}


def robot_profile(key: str) -> RobotProfile:
    """Old field names: ``n_joints`` equals ``ArmSpec.dof``."""
    arm = ARMS[key]
    workspace = arm.reach
    return RobotProfile(
        arm.key, arm.dof,
        workspace.target_low, workspace.target_high, workspace.shoulder,
        workspace.reach_min, workspace.reach_max,
    )


ROBOT_PROFILES: dict[str, RobotProfile] = {key: robot_profile(key) for key in ARMS}


def _config_error(message: str) -> Exception:
    from .config import ConfigError
    return ConfigError(message)


def resolve_arm_scene(robot: str, gripper: str | None, scene: str) -> Path:
    """Return an existing MJCF. Incompatible and missing assets raise ConfigError."""
    if robot not in ARMS:
        raise _config_error(f"未知机械臂 {robot!r}")
    if scene not in SCENES:
        raise _config_error(f"未知场景 {scene!r}")
    arm = ARMS[robot]
    if gripper is not None and gripper not in arm.grippers:
        raise _config_error(f"{robot} 与夹爪 {gripper} 不兼容")

    if scene == "reach":
        if gripper is not None:
            raise _config_error(f"场景 {scene!r} 不能同时指定夹爪")
        path = MODELS_ROOT / robot / "scene_reach.xml"
    elif scene == "obstacles":
        if robot != "xarm6" or gripper is not None:
            raise _config_error(f"场景 {scene!r} 仅适用于 xarm6 裸臂")
        path = MODELS_ROOT / "xarm6" / "scene_obstacles.xml"
    elif scene in ("pick_place", "box_pack"):
        combo = _PICK_PLACE_DIRS.get((robot, gripper or ""))
        if combo is None:
            raise _config_error(f"场景 {scene!r} 没有 {robot} + {gripper} 的任务场景")
        path = MODELS_ROOT / combo / f"scene_{scene}.xml"
    elif gripper is None:
        path = MODELS_ROOT / robot / "scene.xml"
    else:
        combo = _COMBO_DIRS.get((robot, gripper))
        if combo is None:
            raise _config_error(f"{robot} 与夹爪 {gripper} 兼容但 MJCF 尚未构建")
        path = MODELS_ROOT / combo / "scene.xml"

    if not path.is_file():
        if gripper is not None and (robot, gripper) not in _COMBO_DIRS:
            raise _config_error(f"{robot} 与夹爪 {gripper} 兼容但 MJCF 尚未构建")
        raise _config_error(f"场景文件不存在: {path}")
    return path
