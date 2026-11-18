"""Typed registry for the five UFACTORY arms and their compatible grippers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .model_paths import MODELS_ROOT

SCENES = ("scene", "obstacles", "pick_place", "box_pack")

# Combined MJCF directories that exist today. A compatible gripper missing
# from this table is declared but not built yet.
_COMBO_DIRS: dict[tuple[str, str], str] = {
    ("lite6", "lite6_gripper"): "lite6_with_gripper",
    ("lite6", "lite6_vacuum_gripper"): "lite6_with_vacuum_gripper",
    ("xarm5", "gripper_g2"): "xarm5_with_gripper",
    ("xarm5", "bio_gripper_g2"): "xarm5_with_bio_gripper",
    ("xarm5", "vacuum_gripper"): "xarm5_with_vacuum_gripper",
    ("xarm5", "ft_sensor"): "xarm5_with_ft_sensor",
    ("xarm5", "ft_sensor_and_gripper"): "xarm5_with_ft_sensor_and_gripper",
    ("xarm5", "ft_sensor_and_vacuum_gripper"): "xarm5_with_ft_sensor_and_vacuum_gripper",
    ("xarm6", "gripper_g2"): "xarm6_with_gripper",
    ("xarm6", "bio_gripper_g2"): "xarm6_with_bio_gripper",
    ("xarm6", "vacuum_gripper"): "xarm6_with_vacuum_gripper",
    ("xarm6", "ft_sensor"): "xarm6_with_ft_sensor",
    ("xarm6", "ft_sensor_and_gripper"): "xarm6_with_ft_sensor_and_gripper",
    ("xarm6", "ft_sensor_and_vacuum_gripper"): "xarm6_with_ft_sensor_and_vacuum_gripper",
    ("xarm7", "gripper_g2"): "xarm7_with_gripper",
    ("xarm7", "bio_gripper_g2"): "xarm7_with_bio_gripper",
    ("xarm7", "vacuum_gripper"): "xarm7_with_vacuum_gripper",
    ("xarm7", "ft_sensor"): "xarm7_with_ft_sensor",
    ("xarm7", "ft_sensor_and_gripper"): "xarm7_with_ft_sensor_and_gripper",
    ("xarm7", "ft_sensor_and_vacuum_gripper"): "xarm7_with_ft_sensor_and_vacuum_gripper",
    ("uf850", "gripper_g2"): "uf850_with_gripper",
    ("uf850", "bio_gripper_g2"): "uf850_with_bio_gripper",
    ("uf850", "vacuum_gripper"): "uf850_with_vacuum_gripper",
    ("uf850", "ft_sensor"): "uf850_with_ft_sensor",
    ("uf850", "ft_sensor_and_gripper"): "uf850_with_ft_sensor_and_gripper",
    ("uf850", "ft_sensor_and_vacuum_gripper"): "uf850_with_ft_sensor_and_vacuum_gripper",
}

_PICK_PLACE_DIRS: dict[tuple[str, str], str] = {
    ("xarm5", "gripper_g2"): "xarm5_with_gripper",
    ("xarm6", "gripper_g2"): "xarm6_with_gripper",
    ("xarm7", "gripper_g2"): "xarm7_with_gripper",
    ("uf850", "gripper_g2"): "uf850_with_gripper",
    ("lite6", "lite6_gripper"): "lite6_with_gripper",
}


@dataclass(frozen=True)
class ArmSpec:
    """Arm identity. ``grippers`` is the compatible set, not the built assets."""

    key: str
    dof: int
    tcp_site: str
    grippers: tuple[str, ...]


def _arm(key: str, dof: int, grippers: tuple[str, ...]) -> ArmSpec:
    return ArmSpec(key, dof, "attachment_site", grippers)


ARMS: dict[str, ArmSpec] = {
    "lite6": _arm("lite6", 6, ("lite6_gripper", "lite6_vacuum_gripper")),
    "xarm5": _arm(
        "xarm5",
        5,
        (
            "gripper_g2",
            "bio_gripper_g2",
            "vacuum_gripper",
            "ft_sensor",
            "ft_sensor_and_gripper",
            "ft_sensor_and_vacuum_gripper",
        ),
    ),
    "xarm6": _arm(
        "xarm6",
        6,
        (
            "gripper_g2",
            "bio_gripper_g2",
            "vacuum_gripper",
            "ft_sensor",
            "ft_sensor_and_gripper",
            "ft_sensor_and_vacuum_gripper",
        ),
    ),
    "xarm7": _arm(
        "xarm7",
        7,
        (
            "gripper_g2",
            "bio_gripper_g2",
            "vacuum_gripper",
            "ft_sensor",
            "ft_sensor_and_gripper",
            "ft_sensor_and_vacuum_gripper",
        ),
    ),
    "uf850": _arm(
        "uf850",
        6,
        (
            "gripper_g2",
            "bio_gripper_g2",
            "vacuum_gripper",
            "ft_sensor",
            "ft_sensor_and_gripper",
            "ft_sensor_and_vacuum_gripper",
        ),
    ),
}


def _config_error(message: str) -> Exception:
    from .config import ConfigError
    return ConfigError(message)


def resolve_arm_scene(robot: str, gripper: str | None, scene: str) -> Path:
    """Return an existing MJCF. Incompatible and missing assets raise ConfigError."""
    if robot not in ARMS:
        raise _config_error(f"unknown robot {robot!r}")
    if scene not in SCENES:
        raise _config_error(f"unknown scene {scene!r}")
    arm = ARMS[robot]
    if gripper is not None and gripper not in arm.grippers:
        raise _config_error(f"{robot} does not support gripper {gripper}")

    if scene == "obstacles":
        if robot != "xarm6" or gripper is not None:
            raise _config_error(f"scene {scene!r} is only available for the bare xarm6")
        path = MODELS_ROOT / "xarm6" / "scene_obstacles.xml"
    elif scene in ("pick_place", "box_pack"):
        combo = _PICK_PLACE_DIRS.get((robot, gripper or ""))
        if combo is None:
            raise _config_error(f"scene {scene!r} has no task scene for {robot} + {gripper}")
        path = MODELS_ROOT / combo / f"scene_{scene}.xml"
    elif gripper is None:
        path = MODELS_ROOT / robot / "scene.xml"
    else:
        combo = _COMBO_DIRS.get((robot, gripper))
        if combo is None:
            raise _config_error(f"{robot} + {gripper} MJCF is not built yet")
        path = MODELS_ROOT / combo / "scene.xml"

    if not path.is_file():
        if gripper is not None and (robot, gripper) not in _COMBO_DIRS:
            raise _config_error(f"{robot} + {gripper} MJCF is not built yet")
        from .config import ModelResourceError
        raise ModelResourceError(f"scene file does not exist: {path}")
    return path
