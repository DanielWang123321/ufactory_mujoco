"""Model registry: resource roots, arm specs, and the data used to assemble composite models.

This module holds data only and imports nothing else from the package. ``compose`` builds
models from it with ``MjSpec.attach``; ``config`` resolves task scenes.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parent
CONFIGS_ROOT = _PACKAGE / "configs"
MODELS_ROOT = Path(os.environ.get("UFACTORY_MUJOCO_MODELS", _PACKAGE / "models")).expanduser()

SCENES = ("scene", "obstacles", "pick_place", "box_pack")
TASK_SCENES = ("pick_place", "box_pack")


class ConfigError(ValueError):
    """A task configuration failed strict validation."""


class ModelResourceError(ConfigError):
    """A valid task selection refers to a missing model resource."""


@dataclass(frozen=True)
class ArmSpec:
    """Arm identity. ``grippers`` is the compatible set, not the built assets."""

    key: str
    dof: int
    tcp_site: str
    grippers: tuple[str, ...]


_XARM_SERIES = ("xarm5", "xarm6", "xarm7", "uf850")
# Gripper keys accepted by the xArm series and UF850, and the model-name suffix of each.
_XARM_GRIPPER_SUFFIXES = {
    "gripper_g2": "with_gripper",
    "bio_gripper_g2": "with_bio_gripper",
    "vacuum_gripper": "with_vacuum_gripper",
    "ft_sensor": "with_ft_sensor",
    "ft_sensor_and_gripper": "with_ft_sensor_and_gripper",
    "ft_sensor_and_vacuum_gripper": "with_ft_sensor_and_vacuum_gripper",
}
_LITE6_GRIPPER_SUFFIXES = {"lite6_gripper": "with_gripper", "lite6_vacuum_gripper": "with_vacuum_gripper"}

ARMS: dict[str, ArmSpec] = {
    "lite6": ArmSpec("lite6", 6, "attachment_site", tuple(_LITE6_GRIPPER_SUFFIXES)),
    **{arm: ArmSpec(arm, dof, "attachment_site", tuple(_XARM_GRIPPER_SUFFIXES))
       for arm, dof in (("xarm5", 5), ("xarm6", 6), ("xarm7", 7), ("uf850", 6))},
}

# Firmware / URDF variant keys of the same arms.
ROBOT_KEY_ALIASES = {"xarm5_1305": "xarm5", "xarm6_1305": "xarm6", "xarm7_1305": "xarm7"}

# Gripper used by the pick-place and box-pack task scenes of each arm.
TASK_GRIPPERS = {"lite6": "lite6_gripper", **{arm: "gripper_g2" for arm in _XARM_SERIES}}


def combo_name(robot: str, gripper: str) -> str:
    """Model name of ``robot`` with gripper key ``gripper`` (e.g. ``xarm6_with_gripper``)."""
    suffixes = _LITE6_GRIPPER_SUFFIXES if robot == "lite6" else _XARM_GRIPPER_SUFFIXES
    return f"{robot}_{suffixes[gripper]}"


# ----------------------------------------------------------------------------
# Composite recipes
# ----------------------------------------------------------------------------

CAMERAS = ("d405", "d435", "gemini_305", "zed_x_nano")


@dataclass(frozen=True)
class Recipe:
    """Parts of a composite model. A camera is always carried by the camera stand."""

    arm: str | None
    tool: str | None = None
    ft_sensor: bool = False
    camera: str | None = None
    base_plate: bool = False


def _recipes() -> dict[str, Recipe]:
    recipes = {f"camera_stand_with_{cam}": Recipe(None, camera=cam) for cam in CAMERAS}
    recipes.update({f"{arm}_with_base_plate": Recipe(arm, base_plate=True) for arm in ARMS})
    recipes["lite6_with_gripper"] = Recipe("lite6", "lite6_gripper")
    recipes["lite6_with_vacuum_gripper"] = Recipe("lite6", "lite6_vacuum_gripper")
    for arm in _XARM_SERIES:
        recipes[f"{arm}_with_gripper"] = Recipe(arm, "gripper_g2")
        recipes[f"{arm}_with_bio_gripper"] = Recipe(arm, "bio_gripper_g2")
        recipes[f"{arm}_with_vacuum_gripper"] = Recipe(arm, "vacuum_gripper")
        recipes[f"{arm}_with_d435"] = Recipe(arm, camera="d435")
        recipes[f"{arm}_with_gripper_and_d435"] = Recipe(arm, "gripper_g2", camera="d435")
        recipes[f"{arm}_with_ft_sensor"] = Recipe(arm, ft_sensor=True)
        recipes[f"{arm}_with_ft_sensor_and_gripper"] = Recipe(arm, "gripper_g2", ft_sensor=True)
        recipes[f"{arm}_with_ft_sensor_and_vacuum_gripper"] = Recipe(arm, "vacuum_gripper", ft_sensor=True)
    for cam in CAMERAS:
        recipes.setdefault(f"xarm6_with_gripper_and_{cam}", Recipe("xarm6", "gripper_g2", camera=cam))
    return dict(sorted(recipes.items()))


RECIPES: dict[str, Recipe] = _recipes()


# ----------------------------------------------------------------------------
# Mounting data (values of the validated 0.1.7 assemblies)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Tool:
    """How an end effector part is mounted on the arm flange.

    ``tcp`` is the part's site that becomes ``attachment_site``; ``None`` adds one at
    ``suction_site``. Exclude pairs may use ``{flange}``, ``{prev}`` (the flange's parent
    link) and ``{root}``.
    """

    root: str
    tcp: str | None = "link_tcp"
    actuator: str | None = None
    mesh_prefix: str = ""
    drop_root_excludes: bool = False
    excludes: tuple[tuple[str, str], ...] = (("{flange}", "{root}"),)


_LITE6_TOOL_EXCLUDES = (("link4", "link6"), ("{flange}", "{root}"), ("{prev}", "{root}"), ("link4", "{root}"))

TOOLS: dict[str, Tool] = {
    "gripper_g2": Tool("xarm_gripper_base_link", actuator="gripper", mesh_prefix="g2_", drop_root_excludes=True),
    "bio_gripper_g2": Tool("bio_gripper_g2_base_link"),
    "vacuum_gripper": Tool("xarm_vacuum_gripper_link", tcp=None,
                           excludes=(("{flange}", "{root}"), ("{prev}", "{root}"))),
    "lite6_gripper": Tool("uflite_gripper_link", excludes=_LITE6_TOOL_EXCLUDES),
    "lite6_vacuum_gripper": Tool("uflite_vacuum_gripper_link", tcp=None, excludes=_LITE6_TOOL_EXCLUDES),
}

# `ready` arm pose of anything mounted on these flanges: the zero-roll IK pose that puts the
# Gripper G2 TCP at (300, 0, 300) mm, pointing down. `home` stays all-zero joints.
COMBO_READY: dict[str, tuple[float, ...]] = {
    "xarm5": (0, -0.63088336, -0.837867545, 1.468750905, 0),
    "xarm6": (0, -0.63088336, -0.837867545, 0, 1.468750905, 0),
    "xarm7": (0, -0.612697406, 0, 0.831413847, 0, 1.444111253, 0),
    "uf850": (0, 0.305871495, -0.534714507, 0, -0.840586002, 0),
}
# At all-zero joints a tool on these flanges reaches below the floor, so their combinations
# also get an `init` keyframe (the `ready` pose) to start from.
COMBO_INIT_ARMS = frozenset({"xarm5", "xarm6", "xarm7"})

# Scene statistic (center, extent) of arms carrying a tool or camera.
COMBO_STATISTIC = {"lite6": ((0.25, 0, 0.25), 0.7), **{arm: ((0.3, 0, 0.3), 0.9) for arm in _XARM_SERIES}}

# Bio Gripper G2 flange mount (pos, quat) per arm.
BIO_GRIPPER_MOUNT: dict[str, tuple[tuple[float, ...], tuple[float, ...] | None]] = {
    "xarm5": ((0.026044, -0.000867, -0.000697), (0.999993824591, 0, 0, -0.00351436776576)),
    "xarm6": ((0.027018, -0.000715, -0.000692), (0.999993767297, 0, 0, -0.00353063266485)),
    "xarm7": ((0.026931, -0.000882, -0.00076), (0.99999230379, 0, 0, -0.0039233099351)),
    "uf850": ((0.024657, 0, 0), None),
}

# Base plate part, least-squares fit of link_base on the plate (pos, yaw) per arm.
BASE_PLATE_MOUNT: dict[str, tuple[str, tuple[float, float, float], float]] = {
    "xarm5": ("xarm_base_plate", (-0.00012433, -0.00038722, -0.00001016), 0.000002356),
    "xarm6": ("xarm_base_plate", (0.00007243, 0.00002863, -0.00004715), -0.00726035),
    "xarm7": ("xarm_base_plate", (-0.00057017, 0.00006319, 0.00001328), -0.00725905),
    "uf850": ("uf850_base_plate", (-0.00001021, 0.00000350, 0.00000420), 0.0),
    "lite6": ("lite6_base_plate", (0.00298920, 0.00001614, 0.00171732), 0.000000298),
}

# Camera stand on a flange, and cameras on the stand (camera_stand coordinates).
STAND_FLANGE_QUAT = (0.707106781, 0.707106781, 0, 0)
STAND_FLANGE_LIFT = 0.003  # the stand's flange plate raises the gripper or TCP by 3 mm
STAND_MOUNT_TARGET = (0.0675, 0.003, 0.0)
_R_STANDARD = ((0, 1, 0), (0, 0, 1), (1, 0, 0))
_R_GEMINI = ((0, 0, 1), (0, -1, 0), (1, 0, 0))
# camera -> (mount site offset in the camera body, clearance along stand y, rotation)
CAMERA_MOUNT = {
    "d405": ((0.0, 0.0, -0.023), 0.0, _R_STANDARD),
    "d435": ((0.0, 0.0, -0.025), 0.000055, _R_STANDARD),
    "gemini_305": ((0.018348755, 0.023076703, -0.017177002), 0.0, _R_GEMINI),
    "zed_x_nano": ((0.0, 0.0, 0.0), 0.0, _R_STANDARD),
}

# Task scene geometry that is not part of the hashed task configuration.
TASK_STATISTIC_CENTER = {
    ("lite6", "pick_place"): (0.20, 0.05, 0.15), ("lite6", "box_pack"): (0.20, 0.02, 0.15),
    **{(arm, task): (x, 0, 0.15) for arm, x in (("xarm5", 0.28), ("xarm6", 0.3), ("xarm7", 0.32), ("uf850", 0.35))
       for task in TASK_SCENES},
}
TASK_STATISTIC_EXTENT = 0.9
# Cube stands (box half sizes) raise the Lite 6 task cubes by 20 mm.
TASK_STANDS = {"lite6": {"pick_place": (0.022, 0.022, 0.01), "box_pack": (0.016, 0.016, 0.01)}}
