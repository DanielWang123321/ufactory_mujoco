"""Pose configuration expansion and path-limit helpers for dynamics validation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

from .report import SafePose
from ..robots import ARMS, CONFIGS_ROOT, MODELS_ROOT, ROBOT_KEY_ALIASES

DYNAMICS_VALIDATION_POSES_YAML = CONFIGS_ROOT / "dynamics_validation_pose.yaml"
SEGMENTS_PER_LEG = 5
ANCHOR_COUNT = 5
NamedPoseTuple = tuple[tuple[str, tuple[float, ...]], ...]

_ALIAS_TO_CANONICAL = {**{arm: arm for arm in ARMS}, **ROBOT_KEY_ALIASES}


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


def load_dynamics_poses(robot_key: str, *, yaml_path: Path | None = None) -> list[np.ndarray]:
    """Return the validation poses in radians for ``robot_key``."""
    name = canonical_robot_name(robot_key)
    entry = _load_yaml(yaml_path).get(name)
    if entry is None:
        return []
    dof = ARMS[name].dof
    anchors = _parse_anchor_rows(entry.get("points"), robot_key=name, dof=dof)
    home_a, end_a, home_b, end_b, home_c = anchors
    if home_a != home_b or home_a != home_c:
        raise ValueError(f"{name}.points: all home anchors must match")
    default: list[np.ndarray] = []
    default.extend(expand_round_trip(home_a, end_a))
    default.extend(expand_round_trip(home_a, end_b))
    return default


def default_configs_named_tuple(
    robot_key: str,
    *,
    yaml_path: Path | None = None,
) -> NamedPoseTuple:
    """Return immutable ``((name, q_tuple), ...)`` default poses."""
    default = load_dynamics_poses(robot_key, yaml_path=yaml_path)
    if not default:
        return ()
    return tuple((str(i), tuple(float(v) for v in q)) for i, q in enumerate(default))


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
    yaml_path: Path | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Return named validation poses ``("0", q0) ... ("19", q19)``."""
    return [(str(i), q) for i, q in enumerate(load_dynamics_poses(robot_key, yaml_path=yaml_path))]



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


# ----------------------------------------------------------------------
# Runtime profiles
# ----------------------------------------------------------------------

FloatTuple = tuple[float, ...]

# ≈ 40°/s; hard ceiling aligned with ufactory_genesis real-robot dynamics validation.
MAX_DYNAMICS_MOVE_SPEED_DEG_S = 40.0
MAX_DYNAMICS_MOVE_SPEED_RAD_S = math.radians(MAX_DYNAMICS_MOVE_SPEED_DEG_S)
DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S = MAX_DYNAMICS_MOVE_SPEED_RAD_S
ABS_ERR_FRACTION = 0.10


def validate_dynamics_move_speed(speed_rad_s: float) -> float:
    """Validate and enforce the ``<= 40 deg/s`` dynamics motion speed ceiling."""
    val = float(speed_rad_s)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"dynamics joint speed must be finite and > 0 rad/s, got {speed_rad_s!r}")
    if val > MAX_DYNAMICS_MOVE_SPEED_RAD_S + 1e-9:
        raise ValueError(
            f"dynamics joint speed {val:.4f} rad/s ({math.degrees(val):.2f} deg/s) exceeds "
            f"safety ceiling {MAX_DYNAMICS_MOVE_SPEED_RAD_S:.4f} rad/s ({MAX_DYNAMICS_MOVE_SPEED_DEG_S:.1f} deg/s)"
        )
    return min(val, MAX_DYNAMICS_MOVE_SPEED_RAD_S)

_PROFILE_KEYS = {
    "xarm5": "xarm5_1305",
    "xarm6": "xarm6_1305",
    "xarm7": "xarm7_1305",
    "lite6": "lite6",
    "uf850": "uf850",
}


@dataclass(frozen=True)
class RobotModelSpec:
    """Identity and default MJCF paths for one UFACTORY robot family."""

    key: str
    robot_name: str
    dof: int
    ee_site: str
    model_path: Path
    scene_path: Path


@dataclass(frozen=True)
class ArmControlParams:
    """Arm control parameters extracted directly from the authoritative MJCF."""

    joint_names: tuple[str, ...]
    ee_site: str
    home_qpos: FloatTuple
    default_qpos: FloatTuple
    kp: FloatTuple
    kv: FloatTuple
    effort_limits: FloatTuple


@dataclass(frozen=True)
class DynamicsValidationParams:
    """Per-robot L1/L2/L3 validation tolerances and safe motion defaults."""

    default_configs: NamedPoseTuple
    abs_err_limits: FloatTuple = ()
    l2_err_limit: float = 5.0
    rel_err_limit: float = 0.15
    default_z_min_mm: float = 50.0
    default_move_speed_rad_s: float = DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S
    supports_hardware_validation: bool = True
    sim_internal_abs_limits: FloatTuple = ()
    sim_internal_l2_limit: float = 0.8
    pd_vs_ref_abs_limits: FloatTuple = ()
    pd_vs_ref_l2_limit: float = 5.0
    mass_rel_fro_limit_warn: float = 0.05
    mass_rel_fro_limit_fail: float = 0.15
    ref_vs_real_abs_scale: float = 1.5


@dataclass(frozen=True)
class RobotRuntimeProfile:
    """Resolved profile combining model metadata, MJCF gains/limits, and dynamics tolerances."""

    model: RobotModelSpec
    arm: ArmControlParams
    dynamics: DynamicsValidationParams = field(default_factory=lambda: DynamicsValidationParams(default_configs=()))


def robot_runtime_cli_choices() -> list[str]:
    """Canonical robot names exposed on CLI ``--robot``."""
    return ["xarm5", "xarm6", "xarm7", "lite6", "uf850"]


def abs_err_limits_from_efforts(
    efforts: FloatTuple,
    *,
    fraction: float = ABS_ERR_FRACTION,
) -> FloatTuple:
    """Per-joint absolute torque error limits (Nm) as a fraction of actuator effort limits."""
    return tuple(float(e) * fraction for e in efforts)


def _l2_err_limit_from_abs(abs_limits: FloatTuple) -> float:
    return max(2.0, math.sqrt(sum(v * v for v in abs_limits)))


def _default_qpos(robot_name: str, dof: int) -> FloatTuple:
    if robot_name == "xarm6":
        return (0.0, -0.5, 0.0, 0.0, 0.5, 0.0)
    if robot_name == "lite6":
        return (0.0, -0.6, 0.0, 0.0, 0.6, 0.0)
    base = (0.0, -0.5, 0.0, 0.0, 0.5, 0.0, 0.0)
    return base[:dof] if len(base) >= dof else (*base, *([0.0] * (dof - len(base))))


def _extract_mjcf_arm_params(robot_name: str, model_path: Path, dof: int, ee_site: str) -> ArmControlParams:
    m = mujoco.MjModel.from_xml_path(str(model_path))
    efforts = tuple(float(abs(v)) for v in m.actuator_forcerange[:dof, 1])
    kp = tuple(float(v) for v in m.actuator_gainprm[:dof, 0])
    kv = tuple(float(-v) for v in m.actuator_biasprm[:dof, 2])
    return ArmControlParams(
        joint_names=tuple(f"joint{i}" for i in range(1, dof + 1)),
        ee_site=ee_site,
        home_qpos=tuple(0.0 for _ in range(dof)),
        default_qpos=_default_qpos(robot_name, dof),
        kp=kp,
        kv=kv,
        effort_limits=efforts,
    )


@lru_cache(maxsize=8)
def _build_runtime_profile(robot_name: str) -> RobotRuntimeProfile:
    spec = ARMS[robot_name]
    model_path = MODELS_ROOT / robot_name / f"{robot_name}.xml"
    scene_path = MODELS_ROOT / robot_name / "scene.xml"
    model_spec = RobotModelSpec(
        key=_PROFILE_KEYS[robot_name],
        robot_name=robot_name,
        dof=spec.dof,
        ee_site=spec.tcp_site,
        model_path=model_path,
        scene_path=scene_path,
    )
    arm = _extract_mjcf_arm_params(robot_name, model_path, spec.dof, spec.tcp_site)
    abs_limits = abs_err_limits_from_efforts(arm.effort_limits)
    l2_limit = _l2_err_limit_from_abs(abs_limits)
    dyn = DynamicsValidationParams(
        default_configs=default_configs_named_tuple(robot_name),
        abs_err_limits=abs_limits,
        l2_err_limit=l2_limit,
        rel_err_limit=0.15,
        default_z_min_mm=50.0,
        default_move_speed_rad_s=DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S,
        supports_hardware_validation=True,
        sim_internal_abs_limits=tuple(0.5 for _ in arm.effort_limits),
        sim_internal_l2_limit=0.8,
        pd_vs_ref_abs_limits=abs_limits,
        pd_vs_ref_l2_limit=l2_limit,
    )
    return RobotRuntimeProfile(model=model_spec, arm=arm, dynamics=dyn)


def get_robot_runtime_profile(robot_key: str) -> RobotRuntimeProfile:
    """Resolve a robot name or firmware variant key to a typed runtime profile."""
    return _build_runtime_profile(canonical_robot_name(robot_key))
