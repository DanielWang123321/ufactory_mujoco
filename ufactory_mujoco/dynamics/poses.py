"""Pose configuration expansion and path-limit helpers for dynamics validation."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

from .report import SafePose
from ..model_paths import CONFIGS_ROOT
from ..robots import ARMS

DYNAMICS_VALIDATION_POSES_YAML = CONFIGS_ROOT / "dynamics_validation_pose.yaml"
SEGMENTS_PER_LEG = 5
ANCHOR_COUNT = 5
NamedPoseTuple = tuple[tuple[str, tuple[float, ...]], ...]

_ALIAS_TO_CANONICAL = {
    "xarm5": "xarm5",
    "xarm5_1305": "xarm5",
    "xarm6": "xarm6",
    "xarm6_1305": "xarm6",
    "xarm7": "xarm7",
    "xarm7_1305": "xarm7",
    "lite6": "lite6",
    "uf850": "uf850",
}


def canonical_robot_name(robot_key: str) -> str:
    """Resolve canonical arm key (e.g. ``xarm6_1305`` -> ``xarm6``)."""
    cleaned = (robot_key or "").strip().lower()
    if cleaned not in _ALIAS_TO_CANONICAL:
        raise KeyError(f"Unknown robot key: {robot_key!r}. Valid: {sorted(_ALIAS_TO_CANONICAL)}")
    return _ALIAS_TO_CANONICAL[cleaned]


def _lerp_deg(a_deg: Sequence[float], b_deg: Sequence[float], t: float) -> np.ndarray:
    a = np.deg2rad(np.asarray(a_deg, dtype=np.float64))
    b = np.deg2rad(np.asarray(b_deg, dtype=np.float64))
    return a + float(t) * (b - a)


def expand_round_trip(
    home_deg: Sequence[float],
    end_deg: Sequence[float],
    *,
    segments: int = SEGMENTS_PER_LEG,
) -> list[np.ndarray]:
    """Expand one home-endpoint round trip into ``2 * segments`` poses (radians)."""
    if segments < 1:
        raise ValueError(f"segments must be >= 1, got {segments}")
    out: list[np.ndarray] = []
    for k in range(1, segments + 1):
        out.append(_lerp_deg(home_deg, end_deg, k / segments))
    for k in range(segments - 1, -1, -1):
        out.append(_lerp_deg(home_deg, end_deg, k / segments))
    return out


def _parse_anchor_rows(rows: Any, *, robot_key: str, dof: int) -> list[list[float]]:
    if not isinstance(rows, list) or len(rows) != ANCHOR_COUNT:
        raise ValueError(
            f"{robot_key}.points: expected {ANCHOR_COUNT} anchor rows, got "
            f"{len(rows) if isinstance(rows, list) else type(rows).__name__}"
        )
    out: list[list[float]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"{robot_key}.points[{i}]: expected list of floats, got {type(row).__name__}")
        if len(row) != dof:
            raise ValueError(f"{robot_key}.points[{i}]: expected {dof} joints, got {len(row)}")
        out.append([float(v) for v in row])
    if out[0] != out[2] or out[0] != out[4]:
        raise ValueError(f"{robot_key}.points: anchors 0, 2, 4 must be identical home poses")
    return out


@lru_cache(maxsize=4)
def _load_yaml_cached(resolved_path: Path) -> dict[str, Any]:
    with resolved_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid dynamics poses YAML: {resolved_path}")
    return data


def _load_yaml(path: Path | None = None) -> dict[str, Any]:
    yaml_path = Path(path or DYNAMICS_VALIDATION_POSES_YAML).resolve()
    return _load_yaml_cached(yaml_path)


def load_dynamics_pose_lists(
    robot_key: str,
    *,
    yaml_path: Path | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return ``(default_qs, stress_qs)`` in radians for ``robot_key``."""
    name = canonical_robot_name(robot_key)
    entry = _load_yaml(yaml_path).get(name)
    if entry is None:
        return [], []
    dof = ARMS[name].dof
    anchors = _parse_anchor_rows(entry.get("points"), robot_key=name, dof=dof)
    home_a, end_a, home_b, end_b, home_c = anchors
    if home_a != home_b or home_a != home_c:
        raise ValueError(f"{name}.points: all home anchors must match")
    default: list[np.ndarray] = []
    default.extend(expand_round_trip(home_a, end_a))
    default.extend(expand_round_trip(home_a, end_b))
    return default, []


def default_configs_named_tuple(
    robot_key: str,
    *,
    yaml_path: Path | None = None,
) -> NamedPoseTuple:
    """Return immutable ``((name, q_tuple), ...)`` default poses."""
    default, _ = load_dynamics_pose_lists(robot_key, yaml_path=yaml_path)
    if not default:
        return ()
    return tuple((str(i), tuple(float(v) for v in q)) for i, q in enumerate(default))


def stress_configs_for_robot(
    robot_key: str,
    *,
    yaml_path: Path | None = None,
) -> NamedPoseTuple:
    """Return immutable stress poses if configured."""
    default, stress = load_dynamics_pose_lists(robot_key, yaml_path=yaml_path)
    if not stress:
        return ()
    base = len(default)
    return tuple((str(base + j), tuple(float(v) for v in q)) for j, q in enumerate(stress))


def merge_test_configs(
    base_configs: Sequence[tuple[str, np.ndarray]],
    extra_configs: Sequence[tuple[str, np.ndarray]] | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Merge config lists, deduplicating by name (first wins)."""
    merged: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    for configs in (base_configs, extra_configs or ()):
        for name, q in configs:
            if name in seen:
                continue
            seen.add(name)
            merged.append((name, np.asarray(q, dtype=np.float64)))
    return merged


def dynamics_default_configs(
    robot_key: str = "xarm6",
    *,
    include_stress: bool = False,
    yaml_path: Path | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Return named validation poses ``("0", q0) ... ("19", q19)``."""
    default, stress = load_dynamics_pose_lists(robot_key, yaml_path=yaml_path)
    configs = [(str(i), q) for i, q in enumerate(default)]
    if include_stress and stress:
        base = len(default)
        stress_list = [(str(base + j), q) for j, q in enumerate(stress)]
        configs = merge_test_configs(configs, stress_list)
    return configs


def xarm6_default_dynamics_configs(*, include_stress: bool = False) -> list[tuple[str, np.ndarray]]:
    """Convenience helper returning xArm6's 20 expanded poses."""
    return dynamics_default_configs("xarm6", include_stress=include_stress)


def filter_safe_configs(
    configs: Sequence[tuple[str, np.ndarray]],
    ee_z_mm_by_name: dict[str, float],
    z_min_mm: float,
    *,
    colliding_names: set[str] | None = None,
) -> tuple[list[SafePose], list[tuple[str, float]]]:
    """Filter poses by minimum flange Z height and optional collision exclusions."""
    safe: list[SafePose] = []
    rejected: list[tuple[str, float]] = []
    blocked = colliding_names or set()
    for name, q in configs:
        if name not in ee_z_mm_by_name:
            rejected.append((name, float("nan")))
            continue
        ee_z = ee_z_mm_by_name[name]
        if name in blocked:
            rejected.append((name, ee_z))
            continue
        if ee_z >= z_min_mm:
            safe.append(SafePose(name=name, q=np.asarray(q, dtype=np.float64), ee_z_mm=ee_z))
        else:
            rejected.append((name, ee_z))
    return safe, rejected


def parse_joint_limits(
    model_or_path: mujoco.MjModel | str | Path,
    joint_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Read lower/upper joint limits (rad) from an MJCF file or loaded ``MjModel``."""
    model = (
        model_or_path
        if isinstance(model_or_path, mujoco.MjModel)
        else mujoco.MjModel.from_xml_path(str(model_or_path))
    )
    lower = np.full(len(joint_names), -np.inf, dtype=np.float64)
    upper = np.full(len(joint_names), np.inf, dtype=np.float64)
    for i, name in enumerate(joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0 or not bool(model.jnt_limited[jid]):
            continue
        lower[i] = float(model.jnt_range[jid, 0])
        upper[i] = float(model.jnt_range[jid, 1])
    return lower, upper


def check_joint_limit_path(
    start_q: Sequence[float],
    target_q: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    margin_rad: float = 0.02,
    steps: int = 25,
) -> list[str]:
    """Verify linear joint-space interpolation stays within ``[lower+margin, upper-margin]``."""
    start = np.asarray(start_q, dtype=np.float64)
    target = np.asarray(target_q, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64) + margin_rad
    hi = np.asarray(upper, dtype=np.float64) - margin_rad
    reasons: list[str] = []
    for i in range(steps + 1):
        alpha = i / steps
        q = (1.0 - alpha) * start + alpha * target
        for idx in np.where(q < lo)[0]:
            reasons.append(f"path step {i}: J{idx + 1} below limit margin ({q[idx]:.4f} < {lo[idx]:.4f})")
        for idx in np.where(q > hi)[0]:
            reasons.append(f"path step {i}: J{idx + 1} above limit margin ({q[idx]:.4f} > {hi[idx]:.4f})")
        if reasons:
            break
    return reasons
