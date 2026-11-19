"""Layered static dynamics analysis (L2/L3) + L1 torque classification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from .backends import armature_aligned_mass
from .report import (
    L2_ERR_LIMIT,
    POS_ERR_TOL,
    REL_ERR_LIMIT,
    SATURATION_MARGIN,
    DynamicsSample,
    ModelDynamicsIssue,
    SafePose,
    SimDynamicsSample,
    ValidationStatus,
    _abs_err_limits,
    _effort_limits,
)
from .profiles import get_robot_runtime_profile

if TYPE_CHECKING:
    from .profiles import RobotRuntimeProfile

LAYER_L2A = "L2a"
LAYER_L2B = "L2b"
LAYER_L3A = "L3a"
LAYER_L3B = "L3b"
STATIC_LAYERS = (LAYER_L2A, LAYER_L2B, LAYER_L3A, LAYER_L3B)

PD_TRACKING_POS_ERR_RAD = 0.02
PD_TRACKING_REF_GRAVITY_FRACTION = 0.25


@dataclass(frozen=True)
class PdHoldGateDecision:
    """Gate deciding whether simulation PD-hold anomalies should block hardware."""

    block_hardware: bool
    reason: str
    tracking_limited_joints: tuple[int, ...] = ()
    ref_gravity_theory: np.ndarray | None = None


def evaluate_pd_hold_gate(
    sim_sample: SimDynamicsSample,
    target_q: np.ndarray,
    *,
    runtime_profile: RobotRuntimeProfile,
    reference: Any | None = None,
) -> PdHoldGateDecision:
    """Return whether simulation PD-hold failure should block real-robot motion."""
    if sim_sample.settled and not sim_sample.saturated:
        return PdHoldGateDecision(block_hardware=False, reason="ok")

    target = np.asarray(target_q, dtype=np.float64).reshape(-1)
    limits = np.asarray(runtime_profile.arm.effort_limits, dtype=np.float64)
    tau = np.asarray(sim_sample.pd_hold_tau, dtype=np.float64).reshape(-1)
    q_actual = np.asarray(sim_sample.q_actual, dtype=np.float64).reshape(-1)
    per_joint_pos_err = np.abs(q_actual - target)
    saturated_mask = np.abs(tau) >= limits * SATURATION_MARGIN
    if not saturated_mask.any() or reference is None:
        reason = "not_settled" if not sim_sample.settled else "saturated"
        return PdHoldGateDecision(block_hardware=True, reason=reason)

    try:
        ref_at_actual = np.asarray(reference.gravity(q_actual), dtype=np.float64).reshape(-1)
        ref_at_target = np.asarray(reference.gravity(target), dtype=np.float64).reshape(-1)
    except Exception:
        reason = "not_settled" if not sim_sample.settled else "saturated"
        return PdHoldGateDecision(block_hardware=True, reason=reason)

    tracking_limited: list[int] = []
    for i in range(len(limits)):
        if not saturated_mask[i]:
            continue
        if (
            per_joint_pos_err[i] > PD_TRACKING_POS_ERR_RAD
            and abs(ref_at_actual[i]) < limits[i] * PD_TRACKING_REF_GRAVITY_FRACTION
        ):
            tracking_limited.append(i)

    saturated_indices = {i for i in range(len(limits)) if saturated_mask[i]}
    if saturated_indices and saturated_indices == set(tracking_limited):
        worst_idx = int(np.argmax(per_joint_pos_err))
        if worst_idx in tracking_limited and per_joint_pos_err[worst_idx] > POS_ERR_TOL:
            return PdHoldGateDecision(
                block_hardware=False,
                reason="pd_tracking_saturation",
                tracking_limited_joints=tuple(tracking_limited),
                ref_gravity_theory=ref_at_target,
            )

    reason = "not_settled" if not sim_sample.settled else "saturated"
    return PdHoldGateDecision(block_hardware=True, reason=reason)


def sim_sample_for_torque_compare(
    sim_sample: SimDynamicsSample,
    gate: PdHoldGateDecision,
) -> SimDynamicsSample:
    """Use ``ref_G(q_target)`` as theory torque when PD saturation is tracking-limited."""
    if gate.ref_gravity_theory is None:
        return sim_sample
    return replace(
        sim_sample,
        pd_hold_tau=np.asarray(gate.ref_gravity_theory, dtype=np.float64).copy(),
        settled=True,
        saturated=False,
    )



@dataclass(frozen=True)
class StaticLayerResult:
    """Result for one L2/L3 static diagnostic layer at a single pose."""

    layer: str
    passed: bool
    severity: str  # pass | warn | fail
    l2_err: float
    abs_err: np.ndarray
    signed_err: np.ndarray
    notes: str = ""


@dataclass
class StaticPoseAnalysis:
    """Aggregated L2/L3 diagnostics for one pose."""

    pose: str
    layers: list[StaticLayerResult] = field(default_factory=list)
    clamp_slack_est: np.ndarray | None = None
    armature: np.ndarray | None = None
    reference_mass_matrix: np.ndarray | None = None
    mass_rel_fro: float | None = None
    warnings: list[str] = field(default_factory=list)


def format_torque_row(values: Sequence[float] | np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):8.3f}" for v in values) + "]"


def parse_strict_static_layers(value: str | None) -> frozenset[str]:
    """Parse ``--strict-static-layers`` tokens into canonical layer IDs."""
    if not value or value.strip().lower() in {"all", "*"}:
        return frozenset(STATIC_LAYERS)
    layers: set[str] = set()
    alias = {
        "l2a": LAYER_L2A,
        "l2b": LAYER_L2B,
        "l3a": LAYER_L3A,
        "l3b": LAYER_L3B,
    }
    for token in value.split(","):
        t = token.strip().lower()
        if not t:
            continue
        if t == "l2":
            layers.update({LAYER_L2A, LAYER_L2B})
        elif t == "l3":
            layers.update({LAYER_L3A, LAYER_L3B})
        elif t in alias:
            layers.add(alias[t])
    return frozenset(layers or STATIC_LAYERS)


def estimate_pd_residual(
    q_target: Sequence[float],
    q_actual: Sequence[float],
    qvel: Sequence[float],
    *,
    kp: Sequence[float],
    kv: Sequence[float],
) -> np.ndarray:
    """Estimate PD feedback torque: ``Kp*(q* - q) - Kv*qdot``."""
    target = np.asarray(q_target, dtype=np.float64).reshape(-1)
    actual = np.asarray(q_actual, dtype=np.float64).reshape(-1)
    vel = np.asarray(qvel, dtype=np.float64).reshape(-1)
    kp_arr = np.asarray(kp, dtype=np.float64).reshape(-1)
    kv_arr = np.asarray(kv, dtype=np.float64).reshape(-1)
    n = min(len(target), len(actual), len(vel), len(kp_arr), len(kv_arr))
    return kp_arr[:n] * (target[:n] - actual[:n]) - kv_arr[:n] * vel[:n]


def _layer_result(
    layer: str,
    signed_err: np.ndarray,
    *,
    abs_limits: np.ndarray,
    l2_limit: float,
    rel_limit: float,
    effort_limits: np.ndarray,
    warn_scale: float = 1.0,
    fail_scale: float = 1.0,
    notes: str = "",
) -> StaticLayerResult:
    abs_err = np.abs(signed_err)
    rel_err = abs_err / np.maximum(effort_limits[: len(abs_err)], 1e-9)
    l2_err = float(np.linalg.norm(abs_err))
    warn_abs = abs_limits * warn_scale
    fail_abs = abs_limits * fail_scale
    passed = bool((abs_err <= warn_abs).all() and (rel_err <= rel_limit).all() and l2_err <= l2_limit * warn_scale)
    if passed:
        severity = "pass"
    elif bool((abs_err <= fail_abs).all() and l2_err <= l2_limit * fail_scale):
        severity = "warn"
    else:
        severity = "fail"
    return StaticLayerResult(
        layer=layer,
        passed=passed,
        severity=severity,
        l2_err=l2_err,
        abs_err=abs_err,
        signed_err=np.asarray(signed_err, dtype=np.float64),
        notes=notes,
    )


def analyze_sim_internal(
    sample: Any,
    *,
    runtime_profile: RobotRuntimeProfile,
) -> StaticLayerResult:
    """L2a: simulation actuator PD-hold torque vs total generalized DOF force."""
    dyn = runtime_profile.dynamics
    limits = np.asarray(runtime_profile.arm.effort_limits, dtype=np.float64)
    signed = sample.pd_hold_tau - sample.actual_dof_force
    return _layer_result(
        LAYER_L2A,
        signed,
        abs_limits=np.asarray(dyn.sim_internal_abs_limits, dtype=np.float64),
        l2_limit=dyn.sim_internal_l2_limit,
        rel_limit=dyn.rel_err_limit,
        effort_limits=limits,
        warn_scale=1.0,
        fail_scale=1.5,
        notes="pd_hold_tau vs actual_dof_force",
    )



def analyze_pd_vs_reference(
    pd_hold_tau: np.ndarray,
    q_actual: Sequence[float],
    *,
    reference: Any,
    runtime_profile: RobotRuntimeProfile,
) -> StaticLayerResult:
    """L2b: closed-loop simulation PD-hold torque vs analytical gravity oracle ``G(q_actual)``."""
    dyn = runtime_profile.dynamics
    limits = np.asarray(runtime_profile.arm.effort_limits, dtype=np.float64)
    ref_gravity = np.asarray(reference.gravity(np.asarray(q_actual, dtype=np.float64)), dtype=np.float64)
    signed = np.asarray(pd_hold_tau, dtype=np.float64) - ref_gravity
    return _layer_result(
        LAYER_L2B,
        signed,
        abs_limits=np.asarray(dyn.pd_vs_ref_abs_limits, dtype=np.float64),
        l2_limit=dyn.pd_vs_ref_l2_limit,
        rel_limit=dyn.rel_err_limit,
        effort_limits=limits,
        warn_scale=1.0,
        fail_scale=1.5,
        notes="pd_hold_tau vs ref_G",
    )



def analyze_mass_matrices(
    mass_sim: np.ndarray,
    mass_ref: np.ndarray,
    *,
    runtime_profile: RobotRuntimeProfile,
    armature: Sequence[float] | np.ndarray | None = None,
) -> tuple[StaticLayerResult, float]:
    """L3a: armature-aligned mass matrix relative Frobenius norm error."""
    m_sim = np.asarray(mass_sim, dtype=np.float64)
    m_ref = np.asarray(mass_ref, dtype=np.float64)
    if m_sim.shape != m_ref.shape:
        n = min(m_sim.shape[0], m_ref.shape[0])
        m_sim = m_sim[:n, :n]
        m_ref = m_ref[:n, :n]
    arm_note = ""
    if armature is not None:
        arm_arr = np.asarray(armature, dtype=np.float64).reshape(-1)
        k = min(len(arm_arr), m_sim.shape[0])
        if k > 0:
            m_sim = armature_aligned_mass(m_sim, arm_arr[:k])
            arm_note = f" armature={arm_arr[:k].tolist()}"
    diff = m_sim - m_ref
    ref_norm = float(np.linalg.norm(m_ref))
    rel_fro = float(np.linalg.norm(diff) / max(ref_norm, 1e-9))
    dyn = runtime_profile.dynamics
    if rel_fro <= dyn.mass_rel_fro_limit_warn:
        severity = "pass"
        passed = True
    elif rel_fro <= dyn.mass_rel_fro_limit_fail:
        severity = "warn"
        passed = False
    else:
        severity = "fail"
        passed = False
    layer = StaticLayerResult(
        layer=LAYER_L3A,
        passed=passed,
        severity=severity,
        l2_err=rel_fro,
        abs_err=np.abs(diff).reshape(-1)[:6] if diff.size else np.zeros(1),
        signed_err=diff.reshape(-1)[:6] if diff.size else np.zeros(1),
        notes=f"mass Frobenius rel err={rel_fro:.4f}{arm_note}",
    )
    return layer, rel_fro


def analyze_ref_vs_real(
    ref_gravity: np.ndarray,
    tau_real: np.ndarray,
    *,
    runtime_profile: RobotRuntimeProfile,
) -> StaticLayerResult:
    """L3b: analytical gravity oracle ``G(q)`` vs hardware SDK torque."""
    dyn = runtime_profile.dynamics
    limits = np.asarray(runtime_profile.arm.effort_limits, dtype=np.float64)
    base_abs = np.asarray(dyn.abs_err_limits, dtype=np.float64)
    ref_abs = base_abs * dyn.ref_vs_real_abs_scale
    signed = ref_gravity - tau_real
    return _layer_result(
        LAYER_L3B,
        signed,
        abs_limits=ref_abs,
        l2_limit=dyn.l2_err_limit * dyn.ref_vs_real_abs_scale,
        rel_limit=dyn.rel_err_limit,
        effort_limits=limits,
        warn_scale=1.0,
        fail_scale=1.0,
        notes="ref_G vs tau_real",
    )



def reference_static_at_q(reference: Any, q: Sequence[float]) -> dict[str, Any]:
    """Evaluate gravity, zero-velocity RNEA consistency, and CRBA mass matrix."""
    q_arr = np.asarray(q, dtype=np.float64)
    ref_g = reference.gravity(q_arr)
    rnea_g = reference.rnea(q_arr, np.zeros_like(q_arr), np.zeros_like(q_arr))
    mass_ref = reference.mass_matrix(q_arr)
    rnea_delta = float(np.linalg.norm(ref_g - rnea_g))
    return {
        "ref_gravity": ref_g,
        "rnea_static": rnea_g,
        "rnea_delta": rnea_delta,
        "mass_ref": mass_ref,
    }


def build_static_pose_analysis(
    pose_name: str,
    q_target: Sequence[float],
    sim_sample: Any,
    *,
    runtime_profile: RobotRuntimeProfile,
    reference: Any | None = None,
    tau_real: np.ndarray | None = None,
) -> StaticPoseAnalysis:
    """Compute L2a/L2b/L3a/L3b static analysis layers for one pose."""
    analysis = StaticPoseAnalysis(pose=pose_name)
    pd_residual = estimate_pd_residual(
        q_target,
        sim_sample.q_actual,
        sim_sample.qvel,
        kp=runtime_profile.arm.kp,
        kv=runtime_profile.arm.kv,
    )
    analysis.clamp_slack_est = np.asarray(sim_sample.pd_hold_tau, dtype=np.float64) - pd_residual
    armature = getattr(sim_sample, "armature", None)
    if armature is not None:
        analysis.armature = np.asarray(armature, dtype=np.float64)

    if not sim_sample.settled:
        analysis.warnings.append("not settled: static layers may be unreliable")
        return analysis

    analysis.layers.append(analyze_sim_internal(sim_sample, runtime_profile=runtime_profile))

    if reference is None:
        analysis.warnings.append("reference dynamics backend unavailable")
        return analysis

    static_ref = reference_static_at_q(reference, sim_sample.q_actual)
    ref_g = static_ref["ref_gravity"]
    if static_ref["rnea_delta"] > 1e-4:
        analysis.warnings.append(f"ref rnea(q,0,0) delta={static_ref['rnea_delta']:.2e}")

    analysis.layers.append(
        analyze_pd_vs_reference(
            sim_sample.pd_hold_tau,
            sim_sample.q_actual,
            reference=reference,
            runtime_profile=runtime_profile,
        )
    )
    mass_layer, rel_fro = analyze_mass_matrices(
        sim_sample.mass_matrix,
        static_ref["mass_ref"],
        runtime_profile=runtime_profile,
        armature=analysis.armature,
    )
    analysis.layers.append(mass_layer)
    analysis.mass_rel_fro = rel_fro
    analysis.reference_mass_matrix = static_ref["mass_ref"]

    if tau_real is not None and np.isfinite(np.asarray(tau_real, dtype=np.float64)).all():
        analysis.layers.append(
            analyze_ref_vs_real(ref_g, np.asarray(tau_real, dtype=np.float64), runtime_profile=runtime_profile)
        )

    for layer in analysis.layers:
        if layer.severity == "warn":
            analysis.warnings.append(f"{layer.layer}: {layer.notes} l2={layer.l2_err:.3f}")
        elif layer.severity == "fail":
            analysis.warnings.append(f"{layer.layer}: FAIL {layer.notes} l2={layer.l2_err:.3f}")

    return analysis


def summarize_static_layers(
    analyses: Sequence[StaticPoseAnalysis],
    *,
    strict_layers: frozenset[str] | None = None,
) -> tuple[dict[str, dict[str, int]], bool]:
    """Count pass/warn/fail per static layer and check strict-layer failure."""
    counts = {layer: {"pass": 0, "warn": 0, "fail": 0} for layer in STATIC_LAYERS}
    strict = strict_layers or frozenset()
    strict_fail = False
    for analysis in analyses:
        for layer in analysis.layers:
            bucket = counts.get(layer.layer)
            if bucket is None:
                continue
            bucket[layer.severity] = bucket.get(layer.severity, 0) + 1
            if layer.layer in strict and layer.severity in {"warn", "fail"}:
                strict_fail = True
    return counts, strict_fail


def static_layer_l2(analysis: StaticPoseAnalysis, layer_id: str) -> float | None:
    for layer in analysis.layers:
        if layer.layer == layer_id:
            return layer.l2_err
    return None


def compare_torques(
    tau_sim: np.ndarray,
    tau_real: np.ndarray,
    *,
    runtime_profile: RobotRuntimeProfile | None = None,
    effort_limits: np.ndarray | None = None,
    abs_limits: np.ndarray | None = None,
    l2_limit: float = L2_ERR_LIMIT,
    rel_limit: float = REL_ERR_LIMIT,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Compute per-joint absolute/relative error, L2 residual, and L1 pass status."""
    runtime = runtime_profile or get_robot_runtime_profile("xarm6")
    limits = effort_limits if effort_limits is not None else _effort_limits(runtime)
    abs_lims = abs_limits if abs_limits is not None else _abs_err_limits(runtime)

    tau_s = np.asarray(tau_sim, dtype=np.float64).reshape(-1)
    tau_r = np.asarray(tau_real, dtype=np.float64).reshape(-1)
    abs_err = np.abs(tau_s - tau_r)
    rel_err = abs_err / limits
    l2_err = float(np.linalg.norm(abs_err))
    passed = bool((abs_err <= abs_lims).all() and (rel_err <= rel_limit).all() and l2_err <= l2_limit)
    return abs_err, rel_err, l2_err, passed


def classify_torque_result(
    *,
    settled: bool,
    saturated: bool,
    tau_real: np.ndarray | None,
    abs_err: np.ndarray | None = None,
    rel_err: np.ndarray | None = None,
    l2_err: float | None = None,
    runtime_profile: RobotRuntimeProfile | None = None,
    abs_limits: np.ndarray | None = None,
    l2_limit: float = L2_ERR_LIMIT,
    rel_limit: float = REL_ERR_LIMIT,
) -> ValidationStatus:
    """Classify torque comparison outcome into ``ValidationStatus``."""
    if not settled:
        return ValidationStatus.NOT_SETTLED
    if saturated:
        return ValidationStatus.SATURATED
    if tau_real is None or not np.isfinite(np.asarray(tau_real, dtype=np.float64)).all():
        return ValidationStatus.INSUFFICIENT_DATA
    if abs_err is None or rel_err is None or l2_err is None:
        return ValidationStatus.INSUFFICIENT_DATA

    runtime = runtime_profile or get_robot_runtime_profile("xarm6")
    limits = abs_limits if abs_limits is not None else _abs_err_limits(runtime)
    if bool((abs_err <= limits).all() and (rel_err <= rel_limit).all() and l2_err <= l2_limit):
        return ValidationStatus.PASS
    failed = np.where(abs_err > limits)[0]
    if len(failed) == 1 and l2_err <= l2_limit * 1.25:
        return ValidationStatus.FAIL_BIAS
    return ValidationStatus.FAIL_MODEL


def _apply_static_analysis_fields(
    sample: DynamicsSample,
    analysis: StaticPoseAnalysis,
    *,
    reference_gravity_tau: np.ndarray | None = None,
) -> None:
    sample.clamp_slack_est = analysis.clamp_slack_est
    sample.armature = analysis.armature
    sample.reference_mass_matrix = analysis.reference_mass_matrix
    sample.static_layers = list(analysis.layers)
    sample.static_warnings = list(analysis.warnings)
    sample.l2a_l2_err = static_layer_l2(analysis, LAYER_L2A)
    sample.l2b_l2_err = static_layer_l2(analysis, LAYER_L2B)
    sample.l3a_mass_rel_fro = analysis.mass_rel_fro
    sample.l3b_l2_err = static_layer_l2(analysis, LAYER_L3B)
    if analysis.clamp_slack_est is not None:
        sample.clamp_slack_l2 = float(np.linalg.norm(analysis.clamp_slack_est))
    ref_g = reference_gravity_tau
    if ref_g is not None:
        sample.pin_G_l2 = float(np.linalg.norm(np.asarray(ref_g, dtype=np.float64)))
    elif sample.reference_gravity_tau is not None:
        sample.pin_G_l2 = float(np.linalg.norm(np.asarray(sample.reference_gravity_tau, dtype=np.float64)))


def build_dynamics_sample(
    pose: SafePose,
    sim_sample: SimDynamicsSample,
    *,
    runtime_profile: RobotRuntimeProfile | None = None,
    tau_real: np.ndarray | None = None,
    tau_real_median: np.ndarray | None = None,
    tau_real_std: np.ndarray | None = None,
    tau_real_min: np.ndarray | None = None,
    tau_real_max: np.ndarray | None = None,
    tau_direct: np.ndarray | None = None,
    n_real_samples: int = 0,
    reference_gravity_tau: np.ndarray | None = None,
    reference: Any | None = None,
    skip_reason: str = "",
    notes: list[str] | None = None,
) -> DynamicsSample:
    """Assemble ``DynamicsSample`` with L1 verdict and L2/L3 diagnostics."""
    runtime = runtime_profile or get_robot_runtime_profile("xarm6")
    abs_err = rel_err = signed_err = None
    l2_err: float | None = None
    if tau_real is not None and np.isfinite(np.asarray(tau_real, dtype=np.float64)).all():
        signed_err = sim_sample.pd_hold_tau - np.asarray(tau_real, dtype=np.float64)
        abs_err, rel_err, l2_err, _ = compare_torques(
            sim_sample.pd_hold_tau,
            tau_real,
            runtime_profile=runtime,
            l2_limit=runtime.dynamics.l2_err_limit,
            rel_limit=runtime.dynamics.rel_err_limit,
        )

    status = classify_torque_result(
        settled=sim_sample.settled,
        saturated=sim_sample.saturated,
        tau_real=tau_real,
        abs_err=abs_err,
        rel_err=rel_err,
        l2_err=l2_err,
        runtime_profile=runtime,
        l2_limit=runtime.dynamics.l2_err_limit,
        rel_limit=runtime.dynamics.rel_err_limit,
    )
    if skip_reason == "not settled":
        status = ValidationStatus.NOT_SETTLED
    if skip_reason == "dry-run" and status == ValidationStatus.INSUFFICIENT_DATA:
        notes = [*(notes or []), "dry-run: no real robot torque data"]

    sample = DynamicsSample(
        pose=pose.name,
        q=pose.q,
        ee_z_mm=pose.ee_z_mm,
        status=status,
        settled=sim_sample.settled,
        saturated=sim_sample.saturated,
        q_actual=sim_sample.q_actual,
        qvel=sim_sample.qvel,
        pd_hold_tau=sim_sample.pd_hold_tau,
        actual_dof_force=sim_sample.actual_dof_force,
        mass_matrix=sim_sample.mass_matrix,
        reference_gravity_tau=reference_gravity_tau,
        tau_real=tau_real,
        tau_real_median=tau_real_median,
        tau_real_std=tau_real_std,
        tau_real_min=tau_real_min,
        tau_real_max=tau_real_max,
        tau_direct=tau_direct,
        abs_err=abs_err,
        rel_err=rel_err,
        signed_err=signed_err,
        l2_err=l2_err,
        n_real_samples=n_real_samples,
        skip_reason=skip_reason,
        notes=notes or [],
    )
    static_analysis = build_static_pose_analysis(
        pose.name,
        pose.q,
        sim_sample,
        runtime_profile=runtime,
        reference=reference,
        tau_real=tau_real,
    )
    _apply_static_analysis_fields(sample, static_analysis, reference_gravity_tau=reference_gravity_tau)
    if reference_gravity_tau is None and reference is not None:
        try:
            sample.reference_gravity_tau = reference.gravity(sim_sample.q_actual)
            sample.pin_G_l2 = float(np.linalg.norm(sample.reference_gravity_tau))
        except Exception:
            pass
    return sample


def validate_mjcf_dynamics(
    model_or_path: mujoco.MjModel | str | Path,
    *,
    dof: int | None = None,
    com_abs_limit_m: float = 2.0,
) -> list[ModelDynamicsIssue]:
    """Static MJCF dynamics checks on body inertials, actuators, and joint parameters."""
    model = (
        model_or_path
        if isinstance(model_or_path, mujoco.MjModel)
        else mujoco.MjModel.from_xml_path(str(model_or_path))
    )
    issues: list[ModelDynamicsIssue] = []

    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        mass = float(model.body_mass[body_id])
        if mass <= 0.0:
            issues.append(ModelDynamicsIssue("ERROR", name, "mass must be positive", mass))
            continue

        ipos = np.asarray(model.body_ipos[body_id], dtype=np.float64)
        if float(np.max(np.abs(ipos))) > com_abs_limit_m:
            issues.append(ModelDynamicsIssue("WARN", name, "COM magnitude looks too large", ipos.tolist()))

        inertia = np.asarray(model.body_inertia[body_id], dtype=np.float64)
        if float(np.min(inertia)) <= 0.0:
            issues.append(ModelDynamicsIssue("ERROR", name, "principal inertia must be positive", inertia.tolist()))
        elif not (
            inertia[0] + inertia[1] >= inertia[2]
            and inertia[0] + inertia[2] >= inertia[1]
            and inertia[1] + inertia[2] >= inertia[0]
        ):
            issues.append(
                ModelDynamicsIssue("ERROR", name, "principal inertia triangle inequality failed", inertia.tolist())
            )

    n_joints = int(dof if dof is not None else min(model.njnt, model.nu))
    for j in range(n_joints):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        if not bool(model.jnt_limited[j]):
            issues.append(ModelDynamicsIssue("WARN", jname, "joint is not limited"))
        if float(model.dof_damping[j]) < 0.0:
            issues.append(
                ModelDynamicsIssue("ERROR", jname, "dof_damping must be non-negative", float(model.dof_damping[j]))
            )
        if float(model.dof_frictionloss[j]) < 0.0:
            issues.append(
                ModelDynamicsIssue(
                    "ERROR", jname, "dof_frictionloss must be non-negative", float(model.dof_frictionloss[j])
                )
            )

    for a in range(min(model.nu, n_joints)):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"act_{a}"
        if not bool(model.actuator_forcelimited[a]):
            issues.append(ModelDynamicsIssue("ERROR", aname, "actuator forcelimit disabled"))
        lo, hi = float(model.actuator_forcerange[a, 0]), float(model.actuator_forcerange[a, 1])
        if lo >= 0.0 or hi <= 0.0:
            issues.append(ModelDynamicsIssue("ERROR", aname, "actuator forcerange must straddle zero", [lo, hi]))

    return issues

