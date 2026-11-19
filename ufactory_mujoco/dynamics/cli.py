"""CLI entry points for MuJoCo dynamics validation aligned with ufactory_genesis."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .analysis import (
    STATIC_LAYERS,
    StaticPoseAnalysis,
    build_dynamics_sample,
    build_static_pose_analysis,
    evaluate_pd_hold_gate,
    format_torque_row,
    parse_strict_static_layers,
    sim_sample_for_torque_compare,
    summarize_static_layers,
    validate_mjcf_dynamics,
)
from .profiles import (
    check_joint_limit_path,
    dynamics_default_configs,
    filter_safe_configs,
    merge_test_configs,
    parse_joint_limits,
)
from .backends import (
    load_reference_backend,
    build_mujoco_scene,
    check_mujoco_path_safety,
    compute_ee_z_table_from_sim,
    find_colliding_configs,
    mujoco_pd_hold_torque_at_q,
    read_joint_frictionloss,
)
from .report import (
    write_torque_plot,
    DEFAULT_REPORT_ROOT,
    DynamicsSample,
    SafePose,
    ValidationStatus,
    compare_report_records,
    dynamics_report_paths,
    joint_torque_rows,
    make_run_config,
    now_stamp,
    read_report_records,
    torque_summary,
    write_csv_report,
    write_jsonl_report,
)
from .profiles import (
    RobotRuntimeProfile,
    get_robot_runtime_profile,
    robot_runtime_cli_choices,
    validate_dynamics_move_speed,
)
from ..calibration import (
    KinematicsCalibration,
    export_kinematics_calibration,
    fetch_robot_sn_from_ip,
    get_robot_sn,
    has_per_unit_kinematics_calibration,
    kinematics_suffix_from_sn,
    load_kinematics_calibration,
    parse_sn_model_code,
    validate_ipv4_address,
)
from ..ft_sensor import is_hardware_ft_sensor_enabled, read_hardware_ft_sensor

from ..safety import resolve_xarm_ip

DEFAULT_CALIBRATIONS_DIR = Path("calibrations")
# Per-unit calibrations committed at the repository root (source checkouts only).
REPO_CALIBRATIONS_DIR = Path(__file__).resolve().parents[2] / "calibrations"
DEFAULT_KINEMATICS_SUFFIX_ENV = "XARM_KINEMATICS_SUFFIX"
_PD_HOLD_TORQUE_DEF = "In MuJoCo simulation, joint torques computed by the PD actuator to hold target joint angles."


def _select_poses(
    configs: Sequence[tuple[str, np.ndarray]],
    names_csv: str | None,
) -> tuple[list[tuple[str, np.ndarray]], set[str]]:
    if not names_csv:
        return list(configs), set()
    wanted = {p.strip() for p in names_csv.split(",") if p.strip()}
    selected = [(name, q) for name, q in configs if name in wanted]
    missing = wanted - {name for name, _ in selected}
    return selected, missing


def print_safe_pose_table(safe_poses: Sequence[SafePose], rejected: Sequence[tuple[str, float]]) -> None:
    print(f"\n{'Name':<20} {'EE z (mm)':>12}  Status")
    print("-" * 48)
    for pose in safe_poses:
        print(f"{pose.name:<20} {pose.ee_z_mm:>12.2f}  SAFE")
    for name, ee_z in rejected:
        z_str = f"{ee_z:>12.2f}" if np.isfinite(ee_z) else f"{'n/a':>12}"
        print(f"{name:<20} {z_str}  REJECTED")


def _display_float(value: object, width: int = 10, precision: int = 3) -> str:
    try:
        v = float(value)  # type: ignore[arg-type]
    except Exception:
        return f"{'n/a':>{width}}"
    if not np.isfinite(v):
        return f"{'n/a':>{width}}"
    return f"{v:{width}.{precision}f}"


def print_compare_table(
    results: Sequence[DynamicsSample],
    *,
    runtime_profile: RobotRuntimeProfile | None = None,
) -> None:
    print("\n--- Torque validation summary ---")
    print("torque_l2_err_nm = sqrt(sum((mujoco_tau_nm - sdk_tau_mean_nm)^2))")
    print(
        f"{'Pose':<20} {'EE z mm':>8} {'Status':<18} {'torque_l2':>10} "
        f"{'Worst':>6} {'abs Nm':>9} {'rel':>7} {'N':>4} {'Warn':>4}"
    )
    print("-" * 96)
    for r in results:
        summary = torque_summary(r, runtime_profile=runtime_profile)
        print(
            f"{r.pose:<20} {r.ee_z_mm:8.1f} {summary['status']:<18} "
            f"{_display_float(summary['torque_l2_err_nm'], width=10)} "
            f"{str(summary['worst_joint'] or '-'):>6} "
            f"{_display_float(summary['worst_abs_err_nm'], width=9)} "
            f"{_display_float(summary['worst_rel_err'], width=7, precision=4)} "
            f"{summary['n_real_samples']:4d} {summary['warning_count']:4d}"
        )

    failures = [
        r
        for r in results
        if torque_summary(r, runtime_profile=runtime_profile)["status"] != ValidationStatus.PASS.value
    ]
    if not failures:
        return

    print("\n--- Non-PASS pose details ---")
    for r in failures:
        summary = torque_summary(r, runtime_profile=runtime_profile)
        reason = summary["status_reason"] or "non-PASS"
        print(f"\n[{r.pose}] {summary['status']}: {reason}")
        print(
            f"{'Joint':<5} {'mujoco_tau_nm':>15} {'sdk_tau_mean_nm':>16} "
            f"{'sdk_tau_std_nm':>15} {'abs_err_nm':>12} {'rel_err':>9} "
            f"{'abs_limit_nm':>13} {'rel_limit':>9}"
        )
        print("-" * 104)
        rows = joint_torque_rows(r, runtime_profile=runtime_profile)
        printed = False
        for row in rows:
            if row["joint_status"] == "PASS" and summary["status"] in {
                ValidationStatus.FAIL_BIAS.value,
                ValidationStatus.FAIL_MODEL.value,
            }:
                continue
            printed = True
            print(
                f"{row['joint']:<5} "
                f"{_display_float(row['mujoco_tau_nm'], width=15, precision=6)} "
                f"{_display_float(row['sdk_tau_mean_nm'], width=16, precision=6)} "
                f"{_display_float(row['sdk_tau_std_nm'], width=15, precision=6)} "
                f"{_display_float(row['abs_err_nm'], width=12, precision=6)} "
                f"{_display_float(row['rel_err'], width=9, precision=4)} "
                f"{_display_float(row['abs_limit_nm'], width=13, precision=3)} "
                f"{_display_float(row['rel_limit'], width=9, precision=3)}"
            )
        if not printed and r.notes:
            for note in r.notes[:3]:
                print(f"  note: {note}")


def _static_analyses_from_results(results: Sequence[DynamicsSample]) -> list[StaticPoseAnalysis]:
    analyses: list[StaticPoseAnalysis] = []
    for result in results:
        if not result.static_layers and not result.static_warnings:
            continue
        analyses.append(
            StaticPoseAnalysis(
                pose=result.pose,
                layers=list(result.static_layers),
                clamp_slack_est=result.clamp_slack_est,
                armature=result.armature,
                reference_mass_matrix=result.reference_mass_matrix,
                mass_rel_fro=result.l3a_mass_rel_fro,
                warnings=list(result.static_warnings),
            )
        )
    return analyses


def print_static_layer_summary(
    results: Sequence[DynamicsSample],
    *,
    strict_layers: frozenset[str] | None = None,
) -> tuple[dict[str, dict[str, int]], bool]:
    analyses = _static_analyses_from_results(results)
    if not analyses:
        print("\n--- Static layer summary (L2/L3) ---")
        print("  (no static layer data)")
        return {layer: {"pass": 0, "warn": 0, "fail": 0} for layer in STATIC_LAYERS}, False

    counts, strict_fail = summarize_static_layers(analyses, strict_layers=strict_layers or frozenset())
    print("\n--- Static layer summary (L2/L3) ---")
    print("  L2a/L2b/L3a/L3b are diagnostics; they affect Overall only with --strict-static.")
    print("  L2a/L2b/L3b are Nm L2 residuals; L3a is mass-matrix relative Frobenius error.")
    for layer in STATIC_LAYERS:
        bucket = counts.get(layer, {"pass": 0, "warn": 0, "fail": 0})
        print(f"  {layer}: pass={bucket.get('pass', 0)} warn={bucket.get('warn', 0)} fail={bucket.get('fail', 0)}")
    if strict_layers:
        print(f"  strict layers: {','.join(sorted(strict_layers))} -> {'FAIL' if strict_fail else 'PASS'}")
    return counts, strict_fail


def _fetch_robot_sn_from_ip(ip: str) -> str:
    return fetch_robot_sn_from_ip(ip)


def _find_kinematics_yaml(
    suffix: str,
    robot_name: str,
    kinematics_yaml_dir: str | None = None,
) -> Path | None:
    candidates: list[Path] = []
    if kinematics_yaml_dir:
        candidates.append(Path(kinematics_yaml_dir).expanduser())
    candidates.extend([DEFAULT_CALIBRATIONS_DIR, REPO_CALIBRATIONS_DIR])
    filename = f"{robot_name}_kinematics_{suffix}.yaml"
    seen: set[Path] = set()
    for root in candidates:
        resolved_root = root.resolve()
        if resolved_root in seen:
            continue
        seen.add(resolved_root)
        candidate = resolved_root / filename
        if candidate.is_file():
            return candidate
    return None


def _resolve_calibration(
    *,
    runtime: RobotRuntimeProfile,
    ip: str | None = None,
    kinematics_yaml: str | None = None,
    kinematics_suffix: str | None = None,
    kinematics_yaml_dir: str | None = None,
    nominal_model: bool = False,
    force_kinematics: bool = False,
) -> tuple[KinematicsCalibration | None, str | None, str | None]:
    """Resolve optional per-unit kinematics calibration and SN."""
    robot_name = runtime.model.robot_name
    resolved_sn: str | None = None
    if nominal_model:
        if ip:
            resolved_sn = _fetch_robot_sn_from_ip(ip)
        return None, None, resolved_sn

    if kinematics_yaml is not None:
        yaml_path = Path(kinematics_yaml).expanduser().resolve()
        if ip:
            resolved_sn = _fetch_robot_sn_from_ip(ip)
            _validate_sn_calibration(resolved_sn, robot_name, yaml_path, None, force_kinematics)
        calib = load_kinematics_calibration(yaml_path, robot=robot_name, serial_number=resolved_sn)
        return calib, str(yaml_path), resolved_sn

    suffix = (kinematics_suffix or "").strip() or (os.environ.get(DEFAULT_KINEMATICS_SUFFIX_ENV) or "").strip() or None
    if ip:
        resolved_sn = _fetch_robot_sn_from_ip(ip)
        if suffix is None:
            sn_suffix = kinematics_suffix_from_sn(resolved_sn)
            if has_per_unit_kinematics_calibration(resolved_sn, robot_name) or _find_kinematics_yaml(
                sn_suffix, robot_name, kinematics_yaml_dir
            ):
                suffix = sn_suffix
                print(f"kinematics_suffix: {suffix} (auto from SN {resolved_sn})")

    if suffix is None:
        return None, None, resolved_sn

    yaml_path = _find_kinematics_yaml(suffix, robot_name, kinematics_yaml_dir)
    if yaml_path is None:
        raise FileNotFoundError(
            f"Cannot find kinematics YAML '{robot_name}_kinematics_{suffix}.yaml'. "
            f"Run: python -m ufactory_mujoco.dynamics export-calibration {ip or '<robot_ip>'} --output-dir calibrations"
        )
    if resolved_sn is not None:
        _validate_sn_calibration(resolved_sn, robot_name, yaml_path, suffix, force_kinematics)
    calib = load_kinematics_calibration(yaml_path, robot=robot_name, serial_number=resolved_sn)
    return calib, str(yaml_path), resolved_sn


def _validate_sn_calibration(
    sn: str,
    robot_name: str,
    yaml_path: Path | None,
    suffix: str | None,
    allow_sn_override: bool,
) -> None:
    if has_per_unit_kinematics_calibration(sn, robot_name):
        return
    code = parse_sn_model_code(sn)
    if allow_sn_override:
        print(
            f"[WARN] SN model code {code} normally has no factory compensation; "
            f"using {yaml_path} via --force-kinematics"
        )
        return
    if suffix and suffix == kinematics_suffix_from_sn(sn) and yaml_path and yaml_path.is_file():
        print(f"[WARN] SN model code {code} normally has no factory compensation; using user YAML {yaml_path}")
        return
    raise ValueError(
        f"SN {sn} (model code {code}) has no factory kinematics compensation; "
        "pass --nominal-model or --force-kinematics"
    )


def _run_sim_samples(
    ctx,
    safe_poses: Sequence[SafePose],
    reference: Any,
    runtime: RobotRuntimeProfile,
    *,
    init_at_target: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    print(f"  pd_hold_tau = PD hold torque ({_PD_HOLD_TORQUE_DEF})")
    for pose in safe_poses:
        sample = mujoco_pd_hold_torque_at_q(
            ctx,
            pose.q,
            runtime_profile=runtime,
            init_at_target=init_at_target,
        )
        out[pose.name] = sample
        status = "settled" if sample.settled else "NOT settled"
        sat = " SATURATED" if sample.saturated else ""
        print(f"  [{pose.name}] pd_hold_tau={format_torque_row(sample.pd_hold_tau)}  ({status}{sat})")
        if reference is not None:
            try:
                g_ref = reference.gravity(sample.q_actual)
                print(f"    ref_G      ={format_torque_row(g_ref)}")
                static = build_static_pose_analysis(
                    pose.name,
                    pose.q,
                    sample,
                    runtime_profile=runtime,
                    reference=reference,
                )
                parts = [
                    f"{layer.layer}={layer.severity.upper()}({layer.l2_err:.3f})"
                    for layer in static.layers
                    if layer.severity != "pass"
                ]
                if parts:
                    print(f"    static: {' '.join(parts)}")
            except Exception as exc:
                print(f"    [WARN] Reference gravity failed: {exc}")
    return out


def _sdk_path_z_reasons(
    session, start_q: Sequence[float], target_q: Sequence[float], z_min_mm: float, steps: int = 10
) -> list[str]:
    reasons: list[str] = []
    start = np.asarray(start_q, dtype=np.float64)
    target = np.asarray(target_q, dtype=np.float64)
    for i in range(steps + 1):
        alpha = i / steps
        q = (1.0 - alpha) * start + alpha * target
        code, sdk_pose = session.arm.get_forward_kinematics(
            angles=q.tolist(),
            input_is_radian=True,
            return_is_radian=True,
        )
        if code != 0:
            reasons.append(f"SDK FK failed at path step {i}: code={code}")
            break
        z_mm = float(sdk_pose[2])
        if z_mm < z_min_mm:
            reasons.append(f"SDK path step {i}: EE z {z_mm:.2f} mm < z_min {z_min_mm:.2f} mm")
            break
    return reasons


def _check_controller_tcp_offset_z(
    session,
    safe_poses: Sequence[SafePose],
    home_qpos: Sequence[float],
) -> list[str]:
    tcp_offset = list(getattr(session.arm, "tcp_offset", None) or [])
    if not any(abs(float(v)) > 1e-6 for v in tcp_offset):
        return []
    reasons: list[str] = []
    seen: set[str] = set()
    check_targets = [("home", np.asarray(home_qpos, dtype=np.float64)), *((p.name, p.q) for p in safe_poses)]
    for name, q in check_targets:
        q_arr = np.asarray(q, dtype=np.float64)
        key = ",".join(f"{float(v):.6f}" for v in q_arr)
        if key in seen:
            continue
        seen.add(key)
        code, sdk_pose = session.arm.get_forward_kinematics(
            angles=q_arr.tolist(),
            input_is_radian=True,
            return_is_radian=True,
        )
        if code != 0:
            reasons.append(f"{name}: SDK FK failed (code={code})")
            continue
        z_mm = float(sdk_pose[2])
        if z_mm <= 0.0:
            reasons.append(f"{name}: EE z {z_mm:.2f} mm <= 0.00 mm")
    return reasons


def _hardware_path_reasons_by_waypoint(
    *,
    session,
    ctx,
    start_q: Sequence[float],
    target_q: Sequence[float],
    joint_lower: Sequence[float],
    joint_upper: Sequence[float],
    z_min_mm: float,
    move_strategy: str,
) -> list[str]:
    from .session import build_motion_waypoints

    waypoints = build_motion_waypoints(start_q, target_q, strategy=move_strategy)
    reasons: list[str] = []
    segment_start = np.asarray(start_q, dtype=np.float64)
    for i, waypoint in enumerate(waypoints, start=1):
        segment_reasons: list[str] = []
        segment_reasons.extend(check_joint_limit_path(segment_start, waypoint, joint_lower, joint_upper))
        segment_reasons.extend(check_mujoco_path_safety(ctx, segment_start, waypoint, z_min_mm=z_min_mm))
        segment_reasons.extend(_sdk_path_z_reasons(session, segment_start, waypoint, z_min_mm))
        if segment_reasons:
            reasons.extend(f"waypoint {i}/{len(waypoints)} [{move_strategy}]: {r}" for r in segment_reasons)
            break
        segment_start = waypoint
    return reasons


def _add_shared_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--robot", default="xarm6", choices=robot_runtime_cli_choices())
    parser.add_argument("--ip", type=str, default=None, help="xArm IP")
    parser.add_argument("--kinematics-suffix", type=str, default=None)
    parser.add_argument("--kinematics-yaml", type=str, default=None)
    parser.add_argument("--kinematics-yaml-dir", type=str, default=None)
    parser.add_argument(
        "--force-kinematics",
        action="store_true",
        help="Use kinematics YAML even when SN rules out factory compensation",
    )
    parser.add_argument(
        "--nominal-model",
        "--nominal-urdf",
        dest="nominal_model",
        action="store_true",
        help="Use nominal MJCF only (skip per-unit kinematics YAML)",
    )
    parser.add_argument("--z-min-mm", type=float, default=None, help="Minimum EE z (mm)")
    parser.add_argument(
        "--reference-backend",
        choices=("auto", "mujoco-rnea", "pinocchio", "none"),
        default="auto",
        help="Analytical reference backend (default: auto -> mujoco-rnea; optional pinocchio)",
    )
    parser.add_argument("--urdf-path", type=str, default=None, help="Optional URDF path for Pinocchio backend")
    parser.add_argument("--require-reference", action="store_true", help="Fail if reference backend is unavailable")
    parser.add_argument(
        "--zero-joint-frictionloss",
        action="store_true",
        help="Set per-joint frictionloss to 0 for this run",
    )
    parser.add_argument("--strict-static", action="store_true", help="Treat L2/L3 warnings as FAIL")
    parser.add_argument("--strict-static-layers", type=str, default=None, help="l2a,l2b,l3a,l3b or l2,l3,all")
    parser.add_argument("--report", type=str, default=None, help="CSV report output path")
    parser.add_argument("--jsonl-report", type=str, default=None, help="JSONL report output path")


def _skipped_dynamics_sample(
    pose: Any,
    sim_sample: Any,
    *,
    status: ValidationStatus,
    settled: bool = False,
    skip_reason: str = "",
    notes: list[str] | None = None,
) -> DynamicsSample:
    return DynamicsSample(
        pose=pose.name,
        q=pose.q,
        ee_z_mm=pose.ee_z_mm,
        status=status,
        settled=settled,
        saturated=sim_sample.saturated,
        q_actual=sim_sample.q_actual,
        qvel=sim_sample.qvel,
        pd_hold_tau=sim_sample.pd_hold_tau,
        actual_dof_force=sim_sample.actual_dof_force,
        mass_matrix=sim_sample.mass_matrix,
        skip_reason=skip_reason,
        notes=list(notes) if notes is not None else [],
    )


def _run_hold_current_only(session: Any, reference: Any) -> int:
    session.print_config()
    raw_load = getattr(session.arm, "tcp_load", None)
    if isinstance(raw_load, (list, tuple)) and len(raw_load) >= 1:
        try:
            if abs(float(raw_load[0])) > 0.05:
                print(
                    f"[NOTE] Controller reports non-zero tcp_load={raw_load}; "
                    "bare-arm ref_G reflects flange-only model without end-effector mass."
                )
        except (TypeError, ValueError):
            pass
    if is_hardware_ft_sensor_enabled(session.arm):
        print(
            "[NOTE] Controller reports an enabled F/T sensor; "
            "bare-arm ref_G reflects flange-only model without sensor mass."
        )
        try:
            ft_reading = read_hardware_ft_sensor(session.arm)
        except Exception as exc:
            print(f"[NOTE] F/T sensor read unavailable: {exc}")
        else:
            print(f"ft_raw (F/T)       : {format_torque_row(ft_reading.raw)}")
            print(f"ft_ext (F/T)       : {format_torque_row(ft_reading.ext)}")
    q, qvel, tau = session.get_joint_states()
    tau_direct = session.get_joints_torque()
    print(f"\nCurrent q          : {format_torque_row(q)}")
    print(f"Current dq         : {format_torque_row(qvel)}")
    print(f"joint_states effort: {format_torque_row(tau)}")
    if tau_direct is not None:
        print(f"get_joints_torque  : {format_torque_row(tau_direct)}")
    if reference is not None:
        g_cur = reference.gravity(q)
        print(f"ref_G(current q)   : {format_torque_row(g_cur)}")
        print(f"L2(ref_G - tau)    : {float(np.linalg.norm(g_cur - tau)):.4f} Nm")
    return 0


def _validate_hardware_preconditions(
    session: Any,
    args: argparse.Namespace,
    safe_poses: Sequence[Any],
    home_qpos: np.ndarray,
) -> bool:
    if is_hardware_ft_sensor_enabled(session.arm):
        robot_hint = getattr(args, "robot", None) or "xarm6"
        session.print_config()
        print(
            "\n[FAIL] Controller reports an enabled F/T sensor; "
            "bare-arm hardware check refuses to move without F/T sensor collision & mass modeling."
        )
        print(
            f"[HINT] Use `ufactory-mujoco dynamics --robot {robot_hint} "
            "--gripper ft_sensor --mode sdk-sim ...` for F/T-equipped validation."
        )
        return False
    raw_load = getattr(session.arm, "tcp_load", None)
    load_kg = 0.0
    if isinstance(raw_load, (list, tuple)) and len(raw_load) >= 1:
        try:
            load_kg = float(raw_load[0])
        except (TypeError, ValueError):
            load_kg = 0.0
    if load_kg > 0.05 and not args.clear_tcp_load:
        session.print_config()
        print(
            f"\n[FAIL] Controller reports non-zero tcp_load ({load_kg:.3f} kg > 0.050 kg); "
            "bare-arm hardware check refuses to move or zero payload with an attached end effector."
        )
        print(
            "[HINT] Detach the end effector and set tcp_load=0, or pass --clear-tcp-load "
            "if bare flange is verified."
        )
        return False
    if not args.clear_tcp_offset:
        offset_z_issues = _check_controller_tcp_offset_z(session, safe_poses, home_qpos)
        if offset_z_issues:
            session.print_config()
            print(
                "\n[FAIL] Controller has non-zero tcp_offset and failed EE z > 0 precheck; "
                "real robot will not move:"
            )
            for issue in offset_z_issues[:5]:
                print(f"  - {issue}")
            print("[HINT] Clear controller tcp_offset or pass --clear-tcp-offset if bare flange is verified.")
            return False
    return True


def _sample_hardware_poses(
    *,
    session: Any,
    ctx: Any,
    safe_poses: Sequence[Any],
    sim_data: dict[str, Any],
    runtime: Any,
    reference: Any,
    args: argparse.Namespace,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    motion_error_cls: type[BaseException],
    results: list[DynamicsSample],
) -> tuple[bool, bool]:
    motion_started = False
    hardware_abort = False
    for pose in safe_poses:
        if hardware_abort:
            break
        sim_sample = sim_data[pose.name]
        for repeat_i in range(args.repeats):
            try:
                q_now, _, _ = session.get_joint_states()
            except RuntimeError as exc:
                print(f"  [{pose.name}] repeat {repeat_i + 1}: UNSAFE; telemetry failed: {exc}")
                results.append(
                    _skipped_dynamics_sample(
                        pose,
                        sim_sample,
                        status=ValidationStatus.UNSAFE,
                        skip_reason="telemetry_error",
                        notes=[str(exc)],
                    )
                )
                hardware_abort = True
                break

            unsafe = _hardware_path_reasons_by_waypoint(
                session=session,
                ctx=ctx,
                start_q=q_now,
                target_q=pose.q,
                joint_lower=joint_lower,
                joint_upper=joint_upper,
                z_min_mm=args.z_min_mm,
                move_strategy=args.move_strategy,
            )
            if unsafe:
                print(f"  [{pose.name}] repeat {repeat_i + 1}: UNSAFE; " + "; ".join(unsafe[:2]))
                results.append(
                    _skipped_dynamics_sample(
                        pose, sim_sample, status=ValidationStatus.UNSAFE, skip_reason="unsafe", notes=unsafe
                    )
                )
                continue

            gate = evaluate_pd_hold_gate(sim_sample, pose.q, runtime_profile=runtime, reference=reference)
            if gate.block_hardware:
                status = ValidationStatus.NOT_SETTLED if not sim_sample.settled else ValidationStatus.SATURATED
                print(
                    f"  [{pose.name}] repeat {repeat_i + 1}: {status.value}; "
                    "MuJoCo PD hold check failed; not moving hardware"
                )
                results.append(
                    _skipped_dynamics_sample(
                        pose,
                        sim_sample,
                        status=status,
                        settled=sim_sample.settled,
                        skip_reason=status.value.lower(),
                    )
                )
                continue

            sim_for_compare = sim_sample_for_torque_compare(sim_sample, gate)
            if gate.reason == "pd_tracking_saturation":
                print(
                    f"  [{pose.name}] MuJoCo PD tracking saturation at joints "
                    f"{list(gate.tracking_limited_joints)}; ref_G(target) used for theory torque"
                )

            print(f"  Moving to [{pose.name}] repeat {repeat_i + 1}/{args.repeats} ...")
            try:
                motion_started = True
                real = session.sample_at_hold(
                    pose.q,
                    speed_rad_s=args.speed,
                    move_strategy=args.move_strategy,
                    sample_duration_s=args.sample_duration,
                    sample_poll_s=args.sample_poll,
                )
            except (motion_error_cls, RuntimeError) as exc:
                print(f"  [{pose.name}] repeat {repeat_i + 1}: UNSAFE; motion failed: {exc}")
                results.append(
                    _skipped_dynamics_sample(
                        pose,
                        sim_sample,
                        status=ValidationStatus.UNSAFE,
                        skip_reason="motion_error",
                        notes=[str(exc)],
                    )
                )
                hardware_abort = True
                break

            ref_g = reference.gravity(real.q) if reference is not None else None
            sample = build_dynamics_sample(
                pose,
                sim_for_compare,
                runtime_profile=runtime,
                tau_real=real.tau,
                tau_real_median=real.tau_median,
                tau_real_std=real.tau_std,
                tau_real_min=real.tau_min,
                tau_real_max=real.tau_max,
                tau_direct=real.tau_direct,
                n_real_samples=real.n_samples,
                reference_gravity_tau=ref_g,
                reference=reference,
                skip_reason="" if real.settled else "not settled",
            )
            if not real.settled and sample.status == ValidationStatus.INSUFFICIENT_DATA:
                sample.status = ValidationStatus.NOT_SETTLED
            print(f"  [{pose.name}] tau_real_mean={format_torque_row(real.tau)}  [{sample.status.value}]")
            results.append(sample)
    return motion_started, hardware_abort


def _finalize_hardware_session(
    *,
    session: Any,
    args: argparse.Namespace,
    ctx: Any,
    runtime: Any,
    motion_started: bool,
    completed_without_interrupt: bool,
    hardware_abort: bool,
    joint_lower: np.ndarray | None,
    joint_upper: np.ndarray | None,
) -> None:
    if session is None:
        return
    if (
        not args.hold_current_only
        and motion_started
        and completed_without_interrupt
        and not hardware_abort
        and not args.no_return_home
    ):
        try:
            q_end_now, _, _ = session.get_joint_states()
            home_reasons = (
                _hardware_path_reasons_by_waypoint(
                    session=session,
                    ctx=ctx,
                    start_q=q_end_now,
                    target_q=runtime.arm.home_qpos,
                    joint_lower=joint_lower,
                    joint_upper=joint_upper,
                    z_min_mm=args.z_min_mm,
                    move_strategy=args.move_strategy,
                )
                if joint_lower is not None and joint_upper is not None
                else []
            )
            if home_reasons:
                print("\n[WARN] Skipping return_home due to unsafe path: " + "; ".join(home_reasons[:2]))
            else:
                print("\n--- Returning to home ---")
                session.return_home(speed_rad_s=args.speed, move_strategy=args.move_strategy)
        except Exception as exc:
            print(f"[WARN] return_home failed: {exc}")
    elif motion_started and not completed_without_interrupt:
        session.stop()
    restore_fn = getattr(session, "restore_controller_config", None)
    if callable(restore_fn):
        for err in restore_fn():
            print(f"[WARN] {err}")
    session.disconnect()


def _parse_hardware_check_args(
    argv: Sequence[str] | None,
    move_strategies: Sequence[str],
    move_strategy_direct: str,
) -> tuple[argparse.Namespace, Any, frozenset[str]]:
    parser = argparse.ArgumentParser(
        description=f"UFACTORY MuJoCo vs real static torque validation. {_PD_HOLD_TORQUE_DEF}"
    )
    _add_shared_model_args(parser)
    parser.add_argument("--dry-run", action="store_true", help="MuJoCo torques + safe poses only")
    parser.add_argument(
        "--confirm-real",
        action="store_true",
        help="Required before hardware controller motion or configuration changes",
    )
    parser.add_argument(
        "--hold-current-only",
        action="store_true",
        help="Read-only check of real torque at current pose without enabling motors or altering config",
    )
    parser.add_argument("--clear-tcp-offset", action="store_true", help="Zero controller tcp_offset before checks")
    parser.add_argument(
        "--clear-tcp-load",
        action="store_true",
        help="Allow temporarily zeroing non-zero controller tcp_load during bare-arm check",
    )
    parser.add_argument(
        "--no-return-home",
        action="store_true",
        help="Stay at the last evaluated pose instead of returning to home_qpos",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Joint speed in rad/s (default & max: 40 deg/s = 0.6981 rad/s)",
    )
    parser.add_argument("--move-strategy", choices=move_strategies, default=move_strategy_direct)
    parser.add_argument("--sample-duration", type=float, default=3.0, help="Seconds of hold samples per pose")
    parser.add_argument("--sample-poll", type=float, default=0.1, help="Seconds between hold samples")
    parser.add_argument("--repeats", type=int, default=1, help="Hardware repeats per pose")
    parser.add_argument("--poses", type=str, default=None, help="Comma-separated pose names to run")
    args = parser.parse_args(argv)

    if args.dry_run and args.hold_current_only:
        parser.error("--dry-run cannot be combined with --hold-current-only")
    if args.hold_current_only and (args.clear_tcp_offset or args.clear_tcp_load):
        parser.error(
            "--hold-current-only is strictly read-only and cannot be combined with "
            "--clear-tcp-offset or --clear-tcp-load"
        )
    if not np.isfinite(args.sample_duration) or args.sample_duration <= 0.0:
        parser.error("--sample-duration must be finite and positive")
    if not np.isfinite(args.sample_poll) or args.sample_poll <= 0.0:
        parser.error("--sample-poll must be finite and positive")
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    runtime = get_robot_runtime_profile(args.robot)
    strict_layers = parse_strict_static_layers(args.strict_static_layers) if args.strict_static else frozenset()
    if args.z_min_mm is None:
        args.z_min_mm = runtime.dynamics.default_z_min_mm
    if args.speed is None:
        args.speed = runtime.dynamics.default_move_speed_rad_s
    try:
        args.speed = validate_dynamics_move_speed(args.speed)
    except ValueError as exc:
        parser.error(str(exc))

    if not args.dry_run:
        args.ip = resolve_xarm_ip(args.ip)
    if not args.dry_run and not args.ip:
        parser.error("--ip is required unless --dry-run (or set XARM_IP)")
    if not args.dry_run and not args.hold_current_only and not args.confirm_real:
        parser.error("--confirm-real is required before hardware controller motion or configuration changes")

    if args.z_min_mm < 50.0:
        print(f"[WARN] z_min={args.z_min_mm:.1f} mm is a low-margin hardware mode")
    return args, runtime, strict_layers


def _emit_hardware_check_reports(
    *,
    results: list[DynamicsSample],
    args: argparse.Namespace,
    ctx: Any,
    runtime: Any,
    run_config: Any,
    model_issues: list[Any],
    strict_layers: frozenset[str],
    interrupted: bool,
) -> int:
    if interrupted:
        if results:
            stamp = now_stamp()
            csv_path, jsonl_path, _ = dynamics_report_paths(
                identity=run_config.robot_sn or runtime.model.key,
                report=args.report,
                jsonl_report=args.jsonl_report,
                report_root=DEFAULT_REPORT_ROOT,
                stamp=stamp,
            )
            run_config.joint_frictionloss = read_joint_frictionloss(ctx).tolist()
            write_csv_report(results, csv_path, runtime_profile=runtime)
            write_jsonl_report(
                results, jsonl_path, run_config=run_config, model_issues=model_issues, runtime_profile=runtime
            )
            print(f"Partial CSV report  : {csv_path}")
            print(f"Partial JSONL report: {jsonl_path}")
        return 130

    print_compare_table(results, runtime_profile=runtime)
    _, strict_static_fail = print_static_layer_summary(results, strict_layers=strict_layers)

    stamp = now_stamp()
    csv_path, jsonl_path, plot_path = dynamics_report_paths(
        identity=run_config.robot_sn or runtime.model.key,
        report=args.report,
        jsonl_report=args.jsonl_report,
        report_root=DEFAULT_REPORT_ROOT,
        stamp=stamp,
    )
    run_config.joint_frictionloss = read_joint_frictionloss(ctx).tolist()
    write_csv_report(results, csv_path, runtime_profile=runtime)
    write_jsonl_report(results, jsonl_path, run_config=run_config, model_issues=model_issues, runtime_profile=runtime)
    plot_written = write_torque_plot(results, plot_path, runtime_profile=runtime)
    print(f"\nCSV report  : {csv_path}")
    print(f"JSONL report: {jsonl_path}")
    if plot_written:
        print(f"Torque plot : {plot_path}")
    else:
        print("Torque plot : skipped (no SDK torque data or matplotlib unavailable)")

    if args.dry_run:
        print("\n[OK] Dry-run complete (MuJoCo quantities only)")
        if strict_static_fail:
            print("[FAIL] --strict-static: one or more L2/L3 layers exceeded thresholds")
            return 1
        return 0

    eval_results = [
        r
        for r in results
        if r.status not in {ValidationStatus.NOT_SETTLED, ValidationStatus.SATURATED, ValidationStatus.UNSAFE}
    ]
    n_pass = sum(1 for r in eval_results if r.status == ValidationStatus.PASS)
    n_total = len(eval_results)
    n_unsafe = sum(1 for r in results if r.status == ValidationStatus.UNSAFE)
    all_passed = n_total > 0 and n_pass == n_total and n_unsafe == 0
    print("\n" + "=" * 78)
    print(f"SUMMARY: {n_pass}/{n_total} evaluated poses passed")
    if n_unsafe:
        print(f"UNSAFE: {n_unsafe} poses skipped/aborted")
    print(f"Overall: {'PASS' if all_passed and not strict_static_fail else 'FAIL'}")
    if strict_static_fail:
        print("STATIC: FAIL (--strict-static L2/L3 thresholds exceeded)")
    print("=" * 78)
    return 0 if all_passed and not strict_static_fail else 1


def _prepare_hardware_check_scene(
    args: argparse.Namespace,
    runtime: Any,
    calibration: Any,
    kinematics_yaml_path: Any,
) -> tuple[Any, list[Any], list[Any], Any, dict[str, Any]] | None:
    print("=" * 78)
    print(f"{runtime.model.key} Static Dynamics Validation: MuJoCo PD hold vs Real Robot")
    print("=" * 78)
    print(f"Scene: {runtime.model.scene_path}")
    if kinematics_yaml_path:
        print(f"Calib: {kinematics_yaml_path}")
    print(
        f"z_min: {args.z_min_mm:.1f} mm  speed: {args.speed:.4f} rad/s  move_strategy: {args.move_strategy}  "
        f"zero_joint_frictionloss: {args.zero_joint_frictionloss}"
    )

    ctx = build_mujoco_scene(
        runtime.model.robot_name,
        runtime_profile=runtime,
        calibration=calibration,
        zero_joint_frictionloss=args.zero_joint_frictionloss,
    )
    model_issues = validate_mjcf_dynamics(ctx.model, dof=runtime.model.dof)
    errors = [i for i in model_issues if i.severity == "ERROR"]
    if model_issues:
        print("\n--- MJCF dynamics static checks ---")
        for issue in model_issues:
            print(f"  [{issue.severity}] {issue.item}: {issue.message}")
    if errors:
        print("[FAIL] MJCF dynamics static checks contain errors")
        return None

    configs = dynamics_default_configs(runtime.model.robot_name)
    configs, missing = _select_poses(configs, args.poses)
    if missing:
        print(f"[WARN] Requested poses not found: {sorted(missing)}")
    if not configs:
        print("[FAIL] No poses selected")
        return None

    ee_z_table = compute_ee_z_table_from_sim(ctx, configs)
    colliding = find_colliding_configs(ctx, configs)
    safe_poses, rejected = filter_safe_configs(configs, ee_z_table, args.z_min_mm, colliding_names=colliding)

    print("\n--- Safe pose filter (MuJoCo attachment_site z + collision) ---")
    print_safe_pose_table(safe_poses, rejected)
    if not safe_poses:
        print("[FAIL] No safe poses after z/collision filter")
        return None

    reference = load_reference_backend(
        runtime.model.robot_name,
        backend=args.reference_backend,
        model_path=runtime.model.model_path,
        urdf_path=args.urdf_path,
        calibration=calibration,
        required=args.require_reference,
    )
    if reference is None:
        print("[WARN] Reference backend disabled/unavailable; continuing without L2b/L3a/L3b")

    print("\n--- MuJoCo PD hold torques ---")
    sim_data = _run_sim_samples(ctx, safe_poses, reference, runtime)
    return ctx, model_issues, safe_poses, reference, sim_data


def cli_hardware_check(argv: Sequence[str] | None = None) -> int:
    from .session import (
        MOVE_STRATEGIES,
        MOVE_STRATEGY_DIRECT,
        RealRobotSession,
        RobotMotionError,
    )

    args, runtime, strict_layers = _parse_hardware_check_args(
        argv, MOVE_STRATEGIES, MOVE_STRATEGY_DIRECT
    )

    calibration, kinematics_yaml_path, resolved_sn = _resolve_calibration(
        runtime=runtime,
        ip=args.ip,
        kinematics_yaml=args.kinematics_yaml,
        kinematics_suffix=args.kinematics_suffix,
        kinematics_yaml_dir=args.kinematics_yaml_dir,
        nominal_model=args.nominal_model,
        force_kinematics=args.force_kinematics,
    )

    prepared = _prepare_hardware_check_scene(args, runtime, calibration, kinematics_yaml_path)
    if prepared is None:
        return 1
    ctx, model_issues, safe_poses, reference, sim_data = prepared

    session: RealRobotSession | None = None
    motion_started = False
    hardware_abort = False
    completed_without_interrupt = False
    interrupted = False
    joint_lower: np.ndarray | None = None
    joint_upper: np.ndarray | None = None
    results: list[DynamicsSample] = []
    settle_steps = max(1, int(round(5.0 / float(ctx.model.opt.timestep))))
    run_config = make_run_config(
        robot_key=runtime.model.key,
        model_path=str(runtime.model.model_path),
        kinematics_yaml_path=kinematics_yaml_path,
        sim_dt=float(ctx.model.opt.timestep),
        settle_steps=settle_steps,
        reference_backend=getattr(reference, "backend_name", None),
        mode="dry-run" if args.dry_run else ("hold-current-only" if args.hold_current_only else "hardware"),
    )
    run_config.robot_sn = resolved_sn or run_config.robot_sn

    try:
        if args.hold_current_only:
            session = RealRobotSession(args.ip, dof=runtime.model.dof, home_qpos=runtime.arm.home_qpos)
            return _run_hold_current_only(session, reference)

        if args.dry_run:
            for pose in safe_poses:
                ref_g = reference.gravity(sim_data[pose.name].q_actual) if reference is not None else None
                results.append(
                    build_dynamics_sample(
                        pose,
                        sim_data[pose.name],
                        runtime_profile=runtime,
                        reference_gravity_tau=ref_g,
                        reference=reference,
                        skip_reason="dry-run",
                    )
                )
            completed_without_interrupt = True
        else:
            session = RealRobotSession(args.ip, dof=runtime.model.dof, home_qpos=runtime.arm.home_qpos)
            if not _validate_hardware_preconditions(session, args, safe_poses, runtime.arm.home_qpos):
                return 1
            session.configure_for_dynamics(clear_tcp_offset=args.clear_tcp_offset)
            session.print_config()
            run_config = make_run_config(
                robot_key=runtime.model.key,
                model_path=str(runtime.model.model_path),
                kinematics_yaml_path=kinematics_yaml_path,
                sim_dt=float(ctx.model.opt.timestep),
                settle_steps=settle_steps,
                reference_backend=getattr(reference, "backend_name", None),
                mode="hardware",
                session=session,
            )

            sn = get_robot_sn(session.arm)
            run_config.robot_sn = sn or run_config.robot_sn
            if calibration is not None and sn != calibration.serial_number:
                raise RuntimeError(f"Connected SN {sn!r} != calibration SN {calibration.serial_number!r}")

            joint_lower, joint_upper = parse_joint_limits(ctx.model, runtime.arm.joint_names)
            print("\n--- Hardware sampling ---")
            motion_started, hardware_abort = _sample_hardware_poses(
                session=session,
                ctx=ctx,
                safe_poses=safe_poses,
                sim_data=sim_data,
                runtime=runtime,
                reference=reference,
                args=args,
                joint_lower=joint_lower,
                joint_upper=joint_upper,
                motion_error_cls=RobotMotionError,
                results=results,
            )
            completed_without_interrupt = True
    except KeyboardInterrupt:
        interrupted = True
        print("\n[ABORT] Interrupted by user (Ctrl-C); stopping controller immediately without return_home")
        if session is not None:
            session.stop()
    finally:
        _finalize_hardware_session(
            session=session,
            args=args,
            ctx=ctx,
            runtime=runtime,
            motion_started=motion_started,
            completed_without_interrupt=completed_without_interrupt,
            hardware_abort=hardware_abort,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
        )

    return _emit_hardware_check_reports(
        results=results,
        args=args,
        ctx=ctx,
        runtime=runtime,
        run_config=run_config,
        model_issues=model_issues,
        strict_layers=strict_layers,
        interrupted=interrupted,
    )


def cli_sim_check(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"UFACTORY MuJoCo dynamics simulation regression. {_PD_HOLD_TORQUE_DEF}"
    )
    _add_shared_model_args(parser)
    parser.add_argument("--random-count", type=int, default=100)
    args = parser.parse_args(argv)

    runtime = get_robot_runtime_profile(args.robot)
    strict_layers = parse_strict_static_layers(args.strict_static_layers) if args.strict_static else frozenset()
    if args.z_min_mm is None:
        args.z_min_mm = runtime.dynamics.default_z_min_mm

    calibration, kinematics_yaml_path, resolved_sn = _resolve_calibration(
        runtime=runtime,
        ip=args.ip,
        kinematics_yaml=args.kinematics_yaml,
        kinematics_suffix=args.kinematics_suffix,
        kinematics_yaml_dir=args.kinematics_yaml_dir,
        nominal_model=args.nominal_model,
        force_kinematics=args.force_kinematics,
    )

    ctx = build_mujoco_scene(
        runtime.model.robot_name,
        runtime_profile=runtime,
        calibration=calibration,
        zero_joint_frictionloss=args.zero_joint_frictionloss,
    )
    issues = validate_mjcf_dynamics(ctx.model, dof=runtime.model.dof)
    errors = [i for i in issues if i.severity == "ERROR"]
    if errors:
        for issue in errors:
            print(f"[ERROR] {issue.item}: {issue.message}")
        return 1

    lower, upper = parse_joint_limits(ctx.model, runtime.arm.joint_names)
    lower = np.where(np.isfinite(lower), lower, -1.0)
    upper = np.where(np.isfinite(upper), upper, 1.0)
    rng = np.random.default_rng(42)
    random_configs = [
        (f"random_{i:03d}", rng.uniform(lower + 0.05, upper - 0.05).astype(np.float64))
        for i in range(max(0, args.random_count))
    ]
    calibrated_names = {name for name, _ in dynamics_default_configs(runtime.model.robot_name)}
    configs = merge_test_configs(dynamics_default_configs(runtime.model.robot_name), random_configs)

    ee_z_table = compute_ee_z_table_from_sim(ctx, configs)
    colliding = find_colliding_configs(ctx, configs)
    safe_poses, rejected = filter_safe_configs(configs, ee_z_table, args.z_min_mm, colliding_names=colliding)
    print_safe_pose_table(safe_poses[:25], rejected[:10])
    reference = load_reference_backend(
        runtime.model.robot_name,
        backend=args.reference_backend,
        model_path=runtime.model.model_path,
        urdf_path=args.urdf_path,
        calibration=calibration,
        required=args.require_reference,
    )

    results: list[DynamicsSample] = []
    for pose in safe_poses:
        init_at_target = pose.name not in calibrated_names
        sim_sample = mujoco_pd_hold_torque_at_q(
            ctx,
            pose.q,
            runtime_profile=runtime,
            init_at_target=init_at_target,
        )
        ref_g = reference.gravity(sim_sample.q_actual) if reference is not None else None
        results.append(
            build_dynamics_sample(
                pose,
                sim_sample,
                runtime_profile=runtime,
                reference_gravity_tau=ref_g,
                reference=reference,
                skip_reason="sim-only",
            )
        )

    print_compare_table(results[:25], runtime_profile=runtime)
    _, strict_static_fail = print_static_layer_summary(results, strict_layers=strict_layers)

    stamp = now_stamp()
    settle_steps = max(1, int(round(5.0 / float(ctx.model.opt.timestep))))
    run_config = make_run_config(
        robot_key=runtime.model.key,
        model_path=str(runtime.model.model_path),
        kinematics_yaml_path=kinematics_yaml_path,
        sim_dt=float(ctx.model.opt.timestep),
        settle_steps=settle_steps,
        reference_backend=getattr(reference, "backend_name", None),
        mode="sim",
    )
    run_config.robot_sn = resolved_sn or run_config.robot_sn
    csv_path, jsonl_path, plot_path = dynamics_report_paths(
        identity=run_config.robot_sn or runtime.model.key,
        report=args.report,
        jsonl_report=args.jsonl_report,
        report_root=DEFAULT_REPORT_ROOT,
        stamp=stamp,
    )
    run_config.joint_frictionloss = read_joint_frictionloss(ctx).tolist()
    write_csv_report(results, csv_path, runtime_profile=runtime)
    write_jsonl_report(results, jsonl_path, run_config=run_config, model_issues=issues, runtime_profile=runtime)
    plot_written = write_torque_plot(results, plot_path, runtime_profile=runtime)
    n_bad = sum(1 for r in results if r.status in {ValidationStatus.NOT_SETTLED, ValidationStatus.SATURATED})
    print(f"Simulation samples: {len(results)}, unstable/saturated: {n_bad}")
    print(f"CSV report  : {csv_path}")
    print(f"JSONL report: {jsonl_path}")
    if plot_written:
        print(f"Torque plot : {plot_path}")
    else:
        print("Torque plot : skipped (no SDK torque data or matplotlib unavailable)")
    if strict_static_fail:
        print("[FAIL] --strict-static: one or more L2/L3 layers exceeded thresholds")
        return 1
    return 0 if n_bad == 0 else 1


@dataclass
class SimCollisionResult:
    """Single pose result from controller internal simulation self-collision check."""

    pose_name: str
    passed: bool
    error_code: int | None = None
    waypoint_index: int | None = None
    message: str = ""


def run_sim_collision_chain(
    session,
    poses: Sequence[tuple[str, np.ndarray]],
    *,
    speed_rad_s: float,
    move_strategy: str,
) -> list[SimCollisionResult]:
    """Move through poses in controller simulation mode without returning home in between."""
    from .session import RobotMotionError

    results: list[SimCollisionResult] = []
    for name, q in poses:
        print(f"\n--- [{name}] ---")
        try:
            session.move_to(q, speed_rad_s=speed_rad_s, wait=True, move_strategy=move_strategy)
            error_code = int(session.arm.error_code)
            if error_code == 22:
                results.append(
                    SimCollisionResult(
                        pose_name=name,
                        passed=False,
                        error_code=error_code,
                        message="self-collision (error_code=22) after move",
                    )
                )
                print("  FAIL: self-collision error_code=22")
                continue
            results.append(SimCollisionResult(pose_name=name, passed=True))
            print("  PASS")
        except RobotMotionError as exc:
            error_code = int(exc.code)
            results.append(
                SimCollisionResult(
                    pose_name=name,
                    passed=False,
                    error_code=error_code,
                    waypoint_index=exc.waypoint_index,
                    message=str(exc),
                )
            )
            print(f"  FAIL: {exc}")
        except RuntimeError as exc:
            raw_err = int(getattr(session.arm, "error_code", 0) or 0)
            results.append(
                SimCollisionResult(
                    pose_name=name,
                    passed=False,
                    error_code=raw_err if raw_err != 0 else None,
                    message=str(exc),
                )
            )
            print(f"  FAIL: {exc}")
    return results


def write_sim_collision_report(results: Sequence[SimCollisionResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["pose", "passed", "error_code", "waypoint_index", "message"])
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "pose": row.pose_name,
                    "passed": row.passed,
                    "error_code": row.error_code if row.error_code is not None else "",
                    "waypoint_index": row.waypoint_index if row.waypoint_index is not None else "",
                    "message": row.message,
                }
            )


_SN_PREFIX_TO_ROBOT = {
    "XF": "xarm5",
    "XI": "xarm6",
    "XS": "xarm7",
    "LI": "lite6",
    "FX": "uf850",
}


def cli_sim_collision_check(argv: Sequence[str] | None = None) -> int:
    from .session import MOVE_STRATEGIES, MOVE_STRATEGY_DIRECT, RealRobotSession

    parser = argparse.ArgumentParser(
        description="xArm controller simulation-mode chained self-collision check for dynamics poses",
    )
    parser.add_argument("--ip", type=str, default=None, help="xArm IP (or XARM_IP)")
    parser.add_argument("--robot", default="xarm6", choices=robot_runtime_cli_choices())
    parser.add_argument(
        "--confirm-real",
        action="store_true",
        help="Required before enabling motors and configuring controller simulation mode",
    )
    parser.add_argument("--clear-tcp-offset", action="store_true", help="Zero controller tcp_offset before checks")
    parser.add_argument("--poses", type=str, default=None, help="Comma-separated pose names to run")
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Joint speed in rad/s (default & max: 40 deg/s = 0.6981 rad/s)",
    )
    parser.add_argument("--move-strategy", choices=MOVE_STRATEGIES, default=MOVE_STRATEGY_DIRECT)
    parser.add_argument("--report", type=str, default=None, help="CSV report output path")
    args = parser.parse_args(argv)

    args.ip = resolve_xarm_ip(args.ip)
    if not args.ip:
        parser.error("--ip or XARM_IP is required")

    runtime = get_robot_runtime_profile(args.robot)
    if args.speed is None:
        args.speed = runtime.dynamics.default_move_speed_rad_s
    try:
        args.speed = validate_dynamics_move_speed(args.speed)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.confirm_real:
        parser.error("--confirm-real is required before controller simulation configuration")

    configs = dynamics_default_configs(runtime.model.robot_name)
    configs, missing = _select_poses(configs, args.poses)
    if missing:
        print(f"[WARN] Requested poses not found: {sorted(missing)}")
    if not configs:
        print("[FAIL] No poses selected")
        return 1

    print("=" * 78)
    print(f"{runtime.model.key} Simulation Self-Collision Chain Check")
    print("=" * 78)
    print(f"IP             : {args.ip}")
    print(f"poses          : {len(configs)}")
    print(f"speed          : {args.speed:.4f} rad/s")
    print(f"move_strategy  : {args.move_strategy}")

    session = RealRobotSession(args.ip, dof=runtime.model.dof, home_qpos=runtime.arm.home_qpos)
    cleanup_errors: list[str] = []
    results: list[SimCollisionResult] = []
    physical_drift = float("nan")
    sim_entered = False
    sim_completed = False
    q_physical_before: np.ndarray | None = None
    try:
        axis = int(getattr(session.arm, "axis", runtime.model.dof) or runtime.model.dof)
        if axis != runtime.model.dof:
            print(f"[FAIL] Connected controller DOF ({axis}) does not match {runtime.model.key} ({runtime.model.dof})")
            return 1
        try:
            sn = get_robot_sn(session.arm)
            expected_robot = _SN_PREFIX_TO_ROBOT.get(sn[:2].upper())
            if expected_robot is not None and expected_robot != runtime.model.robot_name:
                print(
                    f"[FAIL] Connected SN {sn!r} ({expected_robot}) does not match "
                    f"--robot {runtime.model.robot_name!r}"
                )
                return 1
        except Exception:
            pass
        q_physical_before, _, _ = session.get_joint_states()
        sim_entered = True
        session.configure_for_simulation_collision_check(clear_tcp_offset=args.clear_tcp_offset)
        session.print_config()
        print("\n--- Moving virtual arm to home ---")
        session.move_to(
            runtime.arm.home_qpos,
            speed_rad_s=args.speed,
            wait=True,
            move_strategy=args.move_strategy,
        )
        results = run_sim_collision_chain(
            session,
            configs,
            speed_rad_s=args.speed,
            move_strategy=args.move_strategy,
        )
        sim_completed = True
    finally:
        if sim_entered:
            cleanup_errors = session.disable_simulation_robot(restore_ready=sim_completed)
            if q_physical_before is not None:
                try:
                    q_physical_after, _, _ = session.get_joint_states()
                    physical_drift = float(np.max(np.abs(q_physical_after - q_physical_before)))
                except Exception as exc:
                    cleanup_errors.append(f"failed to verify post-sim physical joints: {exc}")
        restore_fn = getattr(session, "restore_controller_config", None)
        if callable(restore_fn):
            for err in restore_fn():
                cleanup_errors.append(f"restore_controller_config: {err}")
        session.disconnect()

    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass
    drift_ok = np.isfinite(physical_drift) and physical_drift <= 1e-3
    print("\n" + "=" * 78)
    print(f"SUMMARY: {n_pass}/{len(results)} poses passed, {n_fail} failed")
    print(f"physical_joint_drift_rad: {physical_drift:.6f} rad (limit <= 0.001000)")
    for row in results:
        if not row.passed:
            wp = f" waypoint={row.waypoint_index}" if row.waypoint_index is not None else ""
            code = f" error_code={row.error_code}" if row.error_code is not None else ""
            print(f"  FAIL [{row.pose_name}]{wp}{code}: {row.message[:120]}")
    for err in cleanup_errors:
        print(f"  [CLEANUP ERROR] {err}")
    print("=" * 78)

    stamp = now_stamp()
    report_path = (
        Path(args.report) if args.report else DEFAULT_REPORT_ROOT / f"dynamics_sim_collision_{stamp}.csv"
    )
    write_sim_collision_report(results, report_path)
    print(f"CSV report: {report_path}")

    return 0 if (n_fail == 0 and not cleanup_errors and drift_ok) else 1


def cli_report_compare(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two dynamics validation reports")
    parser.add_argument("old_report")
    parser.add_argument("new_report")
    args = parser.parse_args(argv)

    old_records = read_report_records(args.old_report)
    new_records = read_report_records(args.new_report)
    try:
        stats = compare_report_records(old_records, new_records)
    except ValueError as exc:
        print(f"Report compare error: {exc}", file=sys.stderr)
        return 2
    print(
        f"{'Joint':>5} {'old_bias':>10} {'new_bias':>10} {'d_bias':>10} "
        f"{'old_rmse':>10} {'new_rmse':>10} {'d_rmse':>10}"
    )
    for row in stats:
        print(
            f"J{int(row['joint']):<4} "
            f"{row['old_bias']:10.4f} {row['new_bias']:10.4f} {row['bias_delta']:10.4f} "
            f"{row['old_rmse']:10.4f} {row['new_rmse']:10.4f} {row['rmse_delta']:10.4f}"
        )
    return 0


def _parse_ipv4(value: str) -> str:
    try:
        return validate_ipv4_address(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def cli_export_calibration(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export per-robot kinematics YAML from xArm port 502")
    parser.add_argument("robot_ip", type=_parse_ipv4, help="xArm controller IPv4 address")
    parser.add_argument("--suffix", default=None, help="Filename suffix (default: last 6 chars of SN)")
    parser.add_argument("--output-dir", default=str(DEFAULT_CALIBRATIONS_DIR), help="Target directory")
    parser.add_argument("--force", action="store_true", help="Export even if SN has nominal kinematics")
    args = parser.parse_args(argv)
    return export_kinematics_calibration(
        args.robot_ip,
        args.output_dir,
        suffix=args.suffix,
        force=args.force,
    )


def main(argv: Sequence[str] | None = None) -> int:
    commands = {
        "hardware-check": cli_hardware_check,
        "sim-check": cli_sim_check,
        "sim-collision-check": cli_sim_collision_check,
        "report-compare": cli_report_compare,
        "export-calibration": cli_export_calibration,
    }
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print("Usage: python -m ufactory_mujoco.dynamics <subcommand> [options]")
        print(f"Subcommands: {', '.join(commands)}")
        return 0
    subcommand = args[0]
    handler = commands.get(subcommand)
    if handler is None:
        print(f"Unknown subcommand: {subcommand!r}. Choose from: {', '.join(commands)}", file=sys.stderr)
        return 2
    return handler(args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
