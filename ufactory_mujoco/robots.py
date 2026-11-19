"""Resource roots, typed arm registry, and static/composed MJCF model resolution."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parent
# Wheels/sdists carry data inside the package. Editable checkouts keep the
# authored XML and configuration paths, including their original hash inputs.
RESOURCE_ROOT = _PACKAGE / "data" if (_PACKAGE / "data").is_dir() else _PACKAGE.parent
CONFIGS_ROOT = RESOURCE_ROOT / "configs"
MODELS_ROOT = Path(os.environ.get("UFACTORY_MUJOCO_MODELS", RESOURCE_ROOT / "models")).expanduser()

SCENES = ("scene", "obstacles", "pick_place", "box_pack")

_XARM_SERIES = ("xarm5", "xarm6", "xarm7", "uf850")
_XARM_GRIPPER_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("gripper_g2", "with_gripper"),
    ("bio_gripper_g2", "with_bio_gripper"),
    ("vacuum_gripper", "with_vacuum_gripper"),
    ("ft_sensor", "with_ft_sensor"),
    ("ft_sensor_and_gripper", "with_ft_sensor_and_gripper"),
    ("ft_sensor_and_vacuum_gripper", "with_ft_sensor_and_vacuum_gripper"),
)
_XARM_GRIPPERS: tuple[str, ...] = tuple(key for key, _ in _XARM_GRIPPER_SUFFIXES)

# Combined MJCF names that exist today (static or dynamically composed via MjSpec).
# A compatible gripper missing from this table is declared but not built yet.
_COMBO_DIRS: dict[tuple[str, str], str] = {
    ("lite6", "lite6_gripper"): "lite6_with_gripper",
    ("lite6", "lite6_vacuum_gripper"): "lite6_with_vacuum_gripper",
    **{
        (arm, gripper): f"{arm}_{suffix}"
        for arm in _XARM_SERIES
        for gripper, suffix in _XARM_GRIPPER_SUFFIXES
    },
}

_PICK_PLACE_KEYS: tuple[tuple[str, str], ...] = (
    ("lite6", "lite6_gripper"),
    *((arm, "gripper_g2") for arm in _XARM_SERIES),
)
_PICK_PLACE_DIRS: dict[tuple[str, str], str] = {key: _COMBO_DIRS[key] for key in _PICK_PLACE_KEYS}


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
    "xarm5": _arm("xarm5", 5, _XARM_GRIPPERS),
    "xarm6": _arm("xarm6", 6, _XARM_GRIPPERS),
    "xarm7": _arm("xarm7", 7, _XARM_GRIPPERS),
    "uf850": _arm("uf850", 6, _XARM_GRIPPERS),
}


def available_models(models_root: Path | None = None) -> list[str]:
    """Return sorted names of all static and dynamically composed models."""
    from .compose import COMPOSED_MODELS

    mroot = Path(models_root) if models_root is not None else MODELS_ROOT
    static = (
        {d.name for d in mroot.iterdir() if d.is_dir() and (d / "scene.xml").is_file()}
        if mroot.is_dir()
        else set()
    )
    composed = set(COMPOSED_MODELS) if static else set()
    return sorted(static | composed)


def resolve_model_path(model_arg: str, *, scene: bool = True) -> Path:
    """Resolve a model name ('xarm6', 'xarm6_with_ft_sensor', ...) or XML path to a loadable file."""
    from .compose import is_composed_model, resolve_composed_path

    p = Path(model_arg)
    if p.is_file():
        return p
    candidates = (
        [MODELS_ROOT / model_arg / "scene.xml", MODELS_ROOT / model_arg / f"{model_arg}.xml"]
        if scene
        else [MODELS_ROOT / model_arg / f"{model_arg}.xml", MODELS_ROOT / model_arg / "scene.xml"]
    )
    for c in candidates:
        if c.is_file():
            return c
    if is_composed_model(model_arg):
        return resolve_composed_path(
            model_arg,
            scene=scene,
            models_root=MODELS_ROOT,
            resource_root=RESOURCE_ROOT,
        )
    raise FileNotFoundError(
        f"Model '{model_arg}' not found. Searched:\n"
        f"  - {p}\n"
        f"  - {candidates[0]}\n"
        f"  - {candidates[1]}\n"
        f"Available models: {available_models()}"
    )


def _config_error(message: str) -> Exception:
    from .config import ConfigError
    return ConfigError(message)


def resolve_arm_scene(robot: str, gripper: str | None, scene: str) -> Path:
    """Return an existing or composed MJCF. Incompatible and missing assets raise ConfigError."""
    from .compose import is_composed_model, resolve_composed_path

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
        static_path = MODELS_ROOT / combo / "scene.xml"
        if static_path.is_file():
            return static_path
        if is_composed_model(combo):
            try:
                return resolve_composed_path(
                    combo,
                    scene=True,
                    models_root=MODELS_ROOT,
                    resource_root=RESOURCE_ROOT,
                )
            except FileNotFoundError as exc:
                from .config import ModelResourceError
                raise ModelResourceError(f"scene file does not exist: {static_path}") from exc
        path = static_path

    if not path.is_file():
        if gripper is not None and (robot, gripper) not in _COMBO_DIRS:
            raise _config_error(f"{robot} + {gripper} MJCF is not built yet")
        from .config import ModelResourceError
        raise ModelResourceError(f"scene file does not exist: {path}")
    return path
