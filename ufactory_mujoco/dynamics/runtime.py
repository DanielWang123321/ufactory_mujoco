"""Typed runtime profiles for MuJoCo dynamics validation across all 5 UFACTORY arms."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
from pathlib import Path

import mujoco

from .poses import (
    NamedPoseTuple,
    canonical_robot_name,
    default_configs_named_tuple,
    stress_configs_for_robot,
)
from ..model_paths import MODELS_ROOT
from ..robots import ARMS

FloatTuple = tuple[float, ...]

# ≈ 40°/s; matches ufactory_genesis real-robot dynamics validation move speed.
DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S = math.radians(40.0)
ABS_ERR_FRACTION = 0.10

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
    force_lower: FloatTuple
    force_upper: FloatTuple
    effort_limits: FloatTuple


@dataclass(frozen=True)
class DynamicsValidationParams:
    """Per-robot L1/L2/L3 validation tolerances and safe motion defaults."""

    default_configs: NamedPoseTuple
    stress_configs: NamedPoseTuple = ()
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
        force_lower=tuple(-abs(e) for e in efforts),
        force_upper=efforts,
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
        stress_configs=stress_configs_for_robot(robot_name),
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
