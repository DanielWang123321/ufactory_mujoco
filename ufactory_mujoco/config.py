"""Strict task configuration: defaults, overlay, and a stable content hash."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .robots import resolve_arm_scene

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
EXECUTORS = ("servo_cartesian", "servo_j")
_HASH_FIELDS = ("schema_version", "robot", "gripper", "scene", "executor",
                "pick_place", "box_pack")


class ConfigError(ValueError):
    """A task configuration failed strict validation."""


@dataclass(frozen=True)
class PickPlaceConfig:
    """Cube task geometry. Heights are in the flange frame before the TCP offset."""

    cube_xy: tuple[float, float]
    target_xy: tuple[float, float]
    cube_half_m: float
    cube_mass_kg: float
    grasp_gap_m: float
    approach_m: float
    lift_z_m: float
    flange_grasp_z_m: float
    cart_v_m_s: float
    cart_a_m_s2: float


@dataclass(frozen=True)
class BoxPackConfig:
    """Box-pack task geometry. Heights are in the flange frame before the TCP offset."""

    feed_xy: tuple[tuple[float, float], ...]
    drop_xy: tuple[tuple[float, float], ...]
    cube_half_m: float
    cube_mass_kg: float
    grasp_gap_m: float
    box_center_xy: tuple[float, float]
    box_inner_half_xy: tuple[float, float]
    box_floor_m: float
    box_wall_m: float
    box_rim_z_m: float
    release_above_rim_m: float
    approach_m: float
    lift_z_m: float
    flange_grasp_z_m: float
    cart_v_m_s: float
    cart_a_m_s2: float
    speed_scale: float
    settle_timeout_s: float


@dataclass(frozen=True)
class TaskConfig:
    """Resolved task selection. Reach calibration is not part of this object."""

    schema_version: int
    robot: str
    gripper: str | None
    scene: str
    executor: str
    pick_place: PickPlaceConfig
    box_pack: BoxPackConfig
    scene_path: Path
    sha256: str
    sources: tuple[str, ...]


def config_sha256(payload: Mapping[str, Any]) -> str:
    """SHA-256 of canonical JSON. Callers pass only the hashed fields."""
    encoded = json.dumps(
        {key: payload[key] for key in _HASH_FIELDS},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"无法读取配置 {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"配置根节点必须是 mapping: {path}")
    return raw


def _merge_strict(base: dict[str, Any], overlay: Mapping[str, Any], *, path: str = "") -> None:
    for key, value in overlay.items():
        dotted = f"{path}.{key}" if path else str(key)
        if key not in base:
            raise ConfigError(f"未知配置字段: {dotted}")
        original = base[key]
        if isinstance(original, dict):
            if not isinstance(value, Mapping):
                raise ConfigError(f"{dotted} 必须是 mapping")
            _merge_strict(original, value, path=dotted)
        else:
            base[key] = value


def _vec2(value: Any, field: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ConfigError(f"{field} 必须是长度为 2 的数组")
    xy = tuple(float(item) for item in value)
    if not all(np_isfinite(item) for item in xy):
        raise ConfigError(f"{field} 必须是有限数值")
    return xy


def np_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _positive(value: Any, field: str, *, allow_zero: bool = False) -> float:
    number = float(value)
    if not np_isfinite(number) or (number < 0.0 if allow_zero else number <= 0.0):
        raise ConfigError(f"{field} 必须是{'非负' if allow_zero else '正'}数")
    return number


def _require_pick_place(data: Mapping[str, Any]) -> PickPlaceConfig:
    if not isinstance(data, Mapping):
        raise ConfigError("pick_place 必须是 mapping")
    return PickPlaceConfig(
        cube_xy=_vec2(data["cube_xy"], "pick_place.cube_xy"),
        target_xy=_vec2(data["target_xy"], "pick_place.target_xy"),
        cube_half_m=_positive(data["cube_half_m"], "pick_place.cube_half_m"),
        cube_mass_kg=_positive(data["cube_mass_kg"], "pick_place.cube_mass_kg"),
        grasp_gap_m=_positive(data["grasp_gap_m"], "pick_place.grasp_gap_m", allow_zero=True),
        approach_m=_positive(data["approach_m"], "pick_place.approach_m"),
        lift_z_m=_positive(data["lift_z_m"], "pick_place.lift_z_m"),
        flange_grasp_z_m=_positive(data["flange_grasp_z_m"], "pick_place.flange_grasp_z_m"),
        cart_v_m_s=_positive(data["cart_v_m_s"], "pick_place.cart_v_m_s"),
        cart_a_m_s2=_positive(data["cart_a_m_s2"], "pick_place.cart_a_m_s2"),
    )


def _vec2_list(value: Any, field: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list | tuple) or len(value) == 0:
        raise ConfigError(f"{field} 必须是非空数组")
    return tuple(_vec2(item, f"{field}[{i}]") for i, item in enumerate(value))


def _require_box_pack(data: Mapping[str, Any]) -> BoxPackConfig:
    if not isinstance(data, Mapping):
        raise ConfigError("box_pack 必须是 mapping")
    feed_xy = _vec2_list(data["feed_xy"], "box_pack.feed_xy")
    drop_xy = _vec2_list(data["drop_xy"], "box_pack.drop_xy")
    if len(feed_xy) != len(drop_xy):
        raise ConfigError("box_pack.feed_xy 与 box_pack.drop_xy 长度必须一致")
    return BoxPackConfig(
        feed_xy=feed_xy,
        drop_xy=drop_xy,
        cube_half_m=_positive(data["cube_half_m"], "box_pack.cube_half_m"),
        cube_mass_kg=_positive(data["cube_mass_kg"], "box_pack.cube_mass_kg"),
        grasp_gap_m=_positive(data["grasp_gap_m"], "box_pack.grasp_gap_m", allow_zero=True),
        box_center_xy=_vec2(data["box_center_xy"], "box_pack.box_center_xy"),
        box_inner_half_xy=_vec2(data["box_inner_half_xy"], "box_pack.box_inner_half_xy"),
        box_floor_m=_positive(data["box_floor_m"], "box_pack.box_floor_m"),
        box_wall_m=_positive(data["box_wall_m"], "box_pack.box_wall_m"),
        box_rim_z_m=_positive(data["box_rim_z_m"], "box_pack.box_rim_z_m"),
        release_above_rim_m=_positive(data["release_above_rim_m"], "box_pack.release_above_rim_m"),
        approach_m=_positive(data["approach_m"], "box_pack.approach_m"),
        lift_z_m=_positive(data["lift_z_m"], "box_pack.lift_z_m"),
        flange_grasp_z_m=_positive(data["flange_grasp_z_m"], "box_pack.flange_grasp_z_m"),
        cart_v_m_s=_positive(data["cart_v_m_s"], "box_pack.cart_v_m_s"),
        cart_a_m_s2=_positive(data["cart_a_m_s2"], "box_pack.cart_a_m_s2"),
        speed_scale=_positive(data["speed_scale"], "box_pack.speed_scale"),
        settle_timeout_s=_positive(data["settle_timeout_s"], "box_pack.settle_timeout_s"),
    )


def _pick_payload(task: PickPlaceConfig) -> dict[str, Any]:
    return {
        "cube_xy": [task.cube_xy[0], task.cube_xy[1]],
        "target_xy": [task.target_xy[0], task.target_xy[1]],
        "cube_half_m": task.cube_half_m,
        "cube_mass_kg": task.cube_mass_kg,
        "grasp_gap_m": task.grasp_gap_m,
        "approach_m": task.approach_m,
        "lift_z_m": task.lift_z_m,
        "flange_grasp_z_m": task.flange_grasp_z_m,
        "cart_v_m_s": task.cart_v_m_s,
        "cart_a_m_s2": task.cart_a_m_s2,
    }


def _box_pack_payload(task: BoxPackConfig) -> dict[str, Any]:
    return {
        "feed_xy": [[xy[0], xy[1]] for xy in task.feed_xy],
        "drop_xy": [[xy[0], xy[1]] for xy in task.drop_xy],
        "cube_half_m": task.cube_half_m,
        "cube_mass_kg": task.cube_mass_kg,
        "grasp_gap_m": task.grasp_gap_m,
        "box_center_xy": [task.box_center_xy[0], task.box_center_xy[1]],
        "box_inner_half_xy": [task.box_inner_half_xy[0], task.box_inner_half_xy[1]],
        "box_floor_m": task.box_floor_m,
        "box_wall_m": task.box_wall_m,
        "box_rim_z_m": task.box_rim_z_m,
        "release_above_rim_m": task.release_above_rim_m,
        "approach_m": task.approach_m,
        "lift_z_m": task.lift_z_m,
        "flange_grasp_z_m": task.flange_grasp_z_m,
        "cart_v_m_s": task.cart_v_m_s,
        "cart_a_m_s2": task.cart_a_m_s2,
        "speed_scale": task.speed_scale,
        "settle_timeout_s": task.settle_timeout_s,
    }


def load_task_config(overlay: str | Path | None = None,
                     overrides: Mapping[str, Any] | None = None) -> TaskConfig:
    """Load configs/default.yaml and apply a strict partial overlay."""
    base = _read_yaml(DEFAULT_CONFIG_PATH)
    sources = [str(DEFAULT_CONFIG_PATH)]
    if overlay is not None:
        overlay_path = Path(overlay)
        _merge_strict(base, _read_yaml(overlay_path))
        sources.append(str(overlay_path))
    if overrides:
        _merge_strict(base, overrides)
    schema_version = base["schema_version"]
    robot = base["robot"]
    gripper = base["gripper"]
    scene = base["scene"]
    executor = base["executor"]
    if type(schema_version) is not int or schema_version != 1:
        raise ConfigError("schema_version 必须为 1")
    if not isinstance(robot, str) or not robot:
        raise ConfigError("robot 必须是非空字符串")
    if gripper is not None and (not isinstance(gripper, str) or not gripper):
        raise ConfigError("gripper 必须是非空字符串或 null")
    if not isinstance(scene, str) or not scene:
        raise ConfigError("scene 必须是非空字符串")
    if executor not in EXECUTORS:
        raise ConfigError(f"未知执行器 {executor!r}")
    pick_place = _require_pick_place(base["pick_place"])
    box_pack = _require_box_pack(base["box_pack"])
    scene_path = resolve_arm_scene(robot, gripper, scene)
    payload = {
        "schema_version": schema_version,
        "robot": robot,
        "gripper": gripper,
        "scene": scene,
        "executor": executor,
        "pick_place": _pick_payload(pick_place),
        "box_pack": _box_pack_payload(box_pack),
    }
    return TaskConfig(
        schema_version, robot, gripper, scene, executor, pick_place, box_pack,
        scene_path, config_sha256(payload), tuple(sources),
    )
