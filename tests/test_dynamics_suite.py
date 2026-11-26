"""Tests for ufactory_mujoco.dynamics validation suite, Schema v4 reporting, and safety gates."""

from __future__ import annotations

import builtins
import json
import math
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from ufactory_mujoco.dynamics import (
    DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S,
    MAX_DYNAMICS_MOVE_SPEED_DEG_S,
    MAX_DYNAMICS_MOVE_SPEED_RAD_S,
    REPORT_SCHEMA_VERSION,
    ArmDynamics,
    DynamicsRunConfig,
    SafePose,
    SimDynamicsSample,
    ValidationStatus,
    attach_payload,
    validate_dynamics_move_speed,
)
from ufactory_mujoco.dynamics import cli as dyn_cli
from ufactory_mujoco.dynamics.analysis import (
    LAYER_L2A,
    LAYER_L2B,
    LAYER_L3A,
    LAYER_L3B,
    StaticPoseAnalysis,
    analyze_mass_matrices,
    analyze_ref_vs_real,
    analyze_sim_internal,
    build_dynamics_sample,
    build_static_pose_analysis,
    classify_torque_result,
    estimate_pd_residual,
    evaluate_pd_hold_gate,
    parse_strict_static_layers,
    sim_sample_for_torque_compare,
    summarize_static_layers,
    validate_mjcf_dynamics,
)
from ufactory_mujoco.dynamics.cli import cli_sim_check, print_compare_table
from ufactory_mujoco.dynamics.report import torque_plot_layout, write_torque_plot
from ufactory_mujoco.dynamics.profiles import (
    DYNAMICS_VALIDATION_POSES_YAML,
    check_joint_limit_path,
    dynamics_default_configs,
    expand_round_trip,
    filter_safe_configs,
    load_dynamics_poses,
    parse_joint_limits,
)
from ufactory_mujoco.dynamics.backends import default_urdf_for_robot, load_reference_backend
from ufactory_mujoco.dynamics.report import (
    ABS_ERR_LIMITS,
    DEFAULT_REPORT_ROOT,
    compare_report_records,
    dynamics_report_paths,
    joint_torque_rows,
    read_report_records,
    sanitize_report_identity,
    torque_summary,
    write_csv_report,
    write_jsonl_report,
)
from ufactory_mujoco.dynamics.profiles import get_robot_runtime_profile
from ufactory_mujoco.dynamics.session import RealRobotSession, RobotMotionError

ROOT = Path(__file__).resolve().parent.parent
ALL_ROBOTS = ("xarm5", "xarm6", "xarm7", "lite6", "uf850")


def _sim_sample(
    dof: int = 6,
    *,
    pd_hold: np.ndarray | None = None,
    dof_force: np.ndarray | None = None,
    settled: bool = True,
) -> SimDynamicsSample:
    pd = np.asarray(pd_hold if pd_hold is not None else np.ones(dof), dtype=np.float64)
    dof_arr = np.asarray(dof_force if dof_force is not None else pd, dtype=np.float64)
    return SimDynamicsSample(
        q_actual=np.zeros(dof),
        qvel=np.zeros(dof),
        pd_hold_tau=pd,
        actual_dof_force=dof_arr,
        mass_matrix=np.eye(dof),
        settled=settled,
        saturated=False,
        pos_err=0.0,
        vel_mag=0.0,
    )


def _load_jsonl(path: Path):
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("type") == "sample":
                out.append(item["data"])
            elif item.get("type") == "run_config":
                out.append(("run_config", item["data"]))
    return out


# ---------------------------------------------------------------------------
# Packaging, backward compatibility, and CLI entry points
# ---------------------------------------------------------------------------


def test_package_exports_and_pyproject_scripts():
    import ufactory_mujoco

    assert ufactory_mujoco.ArmDynamics is ArmDynamics
    assert ufactory_mujoco.attach_payload is attach_payload
    assert DYNAMICS_VALIDATION_POSES_YAML.is_file()
    assert DEFAULT_REPORT_ROOT == Path("reports")

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    scripts = project["scripts"]
    expected = {
        "dynamics-sim-check": "ufactory_mujoco.dynamics.cli:cli_sim_check",
        "dynamics-sim-collision-check": "ufactory_mujoco.dynamics.cli:cli_sim_collision_check",
        "dynamics-hardware-check": "ufactory_mujoco.dynamics.cli:cli_hardware_check",
        "dynamics-report-compare": "ufactory_mujoco.dynamics.cli:cli_report_compare",
    }
    for name, target in expected.items():
        assert scripts.get(name) == target


def test_module_cli_help_and_entrypoint():
    assert dyn_cli.main(["--help"]) == 0
    proc = subprocess.run(
        [sys.executable, "-m", "ufactory_mujoco.dynamics", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "sim-check" in proc.stdout
    assert "hardware-check" in proc.stdout


# ---------------------------------------------------------------------------
# Poses and runtime configuration
# ---------------------------------------------------------------------------


def test_all_robots_have_twenty_expanded_poses():
    for robot_key in ALL_ROBOTS:
        configs = dynamics_default_configs(robot_key)
        assert len(configs) == 20
        assert [name for name, _ in configs] == [str(i) for i in range(20)]


def test_expand_round_trip_xarm6_first_leg():
    home = [0, 0, 0, 0, 0, 0]
    end = [90, -90, -60, 170, 160, -179]
    leg = expand_round_trip(home, end)
    assert len(leg) == 10
    np.testing.assert_allclose(leg[0], np.deg2rad(np.asarray(home) + 0.2 * (np.asarray(end) - np.asarray(home))))
    np.testing.assert_allclose(leg[4], np.deg2rad(end))
    np.testing.assert_allclose(leg[9], np.zeros(6))


def test_xarm6_second_leg_starts_after_first():
    default = load_dynamics_poses("xarm6")
    assert len(default) == 20
    np.testing.assert_allclose(default[9], np.zeros(6))
    np.testing.assert_allclose(
        default[10],
        np.deg2rad(np.asarray([-90, -90, -60, -170, -20, 179]) * 0.2),
    )


@pytest.mark.parametrize(
    ("robot_key", "expected_dof"),
    [("xarm5", 5), ("xarm6", 6), ("xarm7", 7), ("lite6", 6), ("uf850", 6)],
)
def test_runtime_profile_matches_mjcf(robot_key: str, expected_dof: int):
    runtime = get_robot_runtime_profile(robot_key)
    assert runtime.dynamics.supports_hardware_validation
    assert runtime.model.dof == expected_dof
    assert len(runtime.dynamics.default_configs) == 20
    assert len(runtime.arm.effort_limits) == expected_dof
    assert len(runtime.dynamics.abs_err_limits) == expected_dof
    lo, hi = parse_joint_limits(runtime.model.model_path, runtime.arm.joint_names)
    assert lo.shape == (expected_dof,)
    assert np.all(hi > lo)


def test_filter_safe_configs_and_joint_limit_path():
    configs = [("a", np.zeros(6)), ("b", np.ones(6) * 0.2), ("c", np.ones(6) * 0.3)]
    safe, rejected = filter_safe_configs(
        configs,
        {"a": 100.0, "b": 20.0, "c": 120.0},
        z_min_mm=50.0,
        colliding_names={"c"},
    )
    assert [p.name for p in safe] == ["a"]
    assert [name for name, _ in rejected] == ["b", "c"]

    lo = np.full(6, -1.0)
    hi = np.full(6, 1.0)
    assert check_joint_limit_path(np.zeros(6), np.full(6, 0.5), lo, hi) == []
    assert check_joint_limit_path(np.zeros(6), np.full(6, 0.99), lo, hi, margin_rad=0.02)


@pytest.mark.parametrize("robot_key", ALL_ROBOTS)
def test_kinematics_validation_default_poses_within_joint_limits(robot_key: str):
    from ufactory_mujoco.kinematics_validation import default_validation_poses

    runtime = get_robot_runtime_profile(robot_key)
    lo, hi = parse_joint_limits(runtime.model.model_path, runtime.arm.joint_names)
    for name, q in default_validation_poses(robot_key):
        assert q.shape == (runtime.model.dof,), (robot_key, name)
        assert np.all(q >= lo) and np.all(q <= hi), f"{robot_key}.{name} out of joint limits: {q}"


def test_uf850_default_validation_ready_pose_aligns_with_runtime():
    from ufactory_mujoco.kinematics_validation import default_validation_poses

    runtime = get_robot_runtime_profile("uf850")
    poses = dict(default_validation_poses("uf850"))
    np.testing.assert_allclose(poses["ready"], np.asarray(runtime.arm.default_qpos, dtype=np.float64))


# ---------------------------------------------------------------------------
# Static MJCF checks, L1 classification, L2/L3 diagnostics, PD hold gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("robot_key", ALL_ROBOTS)
def test_mjcf_dynamics_static_checks_have_no_errors(robot_key: str):
    runtime = get_robot_runtime_profile(robot_key)
    issues = validate_mjcf_dynamics(runtime.model.scene_path, dof=runtime.model.dof)
    assert [issue for issue in issues if issue.severity == "ERROR"] == []


def test_estimate_pd_residual_matches_formula():
    kp = np.array([100.0, 200.0])
    kv = np.array([10.0, 20.0])
    residual = estimate_pd_residual([0.1, 0.2], [0.0, 0.0], [0.05, 0.1], kp=kp, kv=kv)
    expected = np.array([100.0 * 0.1 - 10.0 * 0.05, 200.0 * 0.2 - 20.0 * 0.1])
    np.testing.assert_allclose(residual, expected)


def test_torque_classification_distinguishes_bias_and_model_failure():
    abs_err = np.array([0.1, 0.2, 0.1, 0.1, 0.1, ABS_ERR_LIMITS[5] + 0.05])
    rel_err = abs_err / (ABS_ERR_LIMITS / 0.10)
    status = classify_torque_result(
        settled=True,
        saturated=False,
        tau_real=np.zeros(6),
        abs_err=abs_err,
        rel_err=rel_err,
        l2_err=float(np.linalg.norm(abs_err)),
    )
    assert status == ValidationStatus.FAIL_BIAS

    big_err = np.full(6, 50.0)
    status2 = classify_torque_result(
        settled=True,
        saturated=False,
        tau_real=np.zeros(6),
        abs_err=big_err,
        rel_err=big_err / (ABS_ERR_LIMITS / 0.10),
        l2_err=float(np.linalg.norm(big_err)),
    )
    assert status2 == ValidationStatus.FAIL_MODEL


def test_l2a_and_l3a_and_l3b_layers():
    runtime = get_robot_runtime_profile("xarm6")
    sample = _sim_sample()
    assert analyze_sim_internal(sample, runtime_profile=runtime).severity == "pass"
    bad_sample = _sim_sample(pd_hold=np.ones(6), dof_force=np.zeros(6))
    assert analyze_sim_internal(bad_sample, runtime_profile=runtime).severity in {"warn", "fail"}

    layer, rel = analyze_mass_matrices(np.eye(6), np.eye(6) * 1.02, runtime_profile=runtime)
    assert layer.layer == LAYER_L3A and rel > 0.0 and layer.passed

    tau_real = np.zeros(6)
    tau_real[0] = runtime.dynamics.abs_err_limits[0] * runtime.dynamics.ref_vs_real_abs_scale + 0.01
    l3b = analyze_ref_vs_real(np.zeros(6), tau_real, runtime_profile=runtime)
    assert l3b.layer == LAYER_L3B and l3b.severity == "fail"


def test_build_static_pose_analysis_and_l1_isolation():
    pose = SafePose("0", np.zeros(6), 108.0)
    runtime = get_robot_runtime_profile("xarm6")
    gravity = np.zeros(6, dtype=np.float64)
    gravity[0] = 1.0 + float(runtime.dynamics.abs_err_limits[0]) + 5.0
    reference = MagicMock()
    reference.gravity.return_value = gravity
    reference.rnea.return_value = gravity
    reference.mass_matrix.return_value = np.eye(6) * 2.0

    sample = build_dynamics_sample(
        pose,
        _sim_sample(),
        runtime_profile=runtime,
        tau_real=np.ones(6),
        reference=reference,
    )
    assert sample.status == ValidationStatus.PASS
    l2b = next(layer for layer in sample.static_layers if layer.layer == LAYER_L2B)
    assert l2b.severity in {"warn", "fail"}

    analysis = build_static_pose_analysis("0", np.zeros(6), _sim_sample(), runtime_profile=runtime, reference=None)
    assert [layer.layer for layer in analysis.layers] == [LAYER_L2A]


def test_strict_static_summary_and_aliases():
    warn_layer = SimpleNamespace(layer=LAYER_L3B, severity="warn")
    _, strict_fail = summarize_static_layers(
        [StaticPoseAnalysis(pose="0", layers=[warn_layer])],
        strict_layers=frozenset({LAYER_L3B}),
    )
    assert strict_fail is True
    assert "L2a" in parse_strict_static_layers("l2")
    assert len(parse_strict_static_layers("all")) == 4


def test_pd_hold_gate_tracking_saturation_bypass():
    runtime = get_robot_runtime_profile("xarm5")
    target_q = np.deg2rad([90.0, -90.0, -60.0, 160.0, -90.0])
    q_actual = np.deg2rad([90.0, -90.216, -59.898, 155.574, -90.005])
    j4_effort = float(runtime.arm.effort_limits[3])
    sample = SimDynamicsSample(
        q_actual=q_actual,
        qvel=np.zeros(5),
        pd_hold_tau=np.array([0.0006, 11.318, -3.549, j4_effort, 0.0824]),
        actual_dof_force=np.zeros(5),
        mass_matrix=np.eye(5),
        settled=False,
        saturated=True,
        pos_err=float(np.abs(q_actual - target_q).max()),
        vel_mag=0.0,
    )
    reference = MagicMock()
    reference.gravity.return_value = np.array([0.0, 11.317, -3.548, -0.897, 0.0], dtype=np.float64)

    gate = evaluate_pd_hold_gate(sample, target_q, runtime_profile=runtime, reference=reference)
    assert gate.block_hardware is False
    assert gate.reason == "pd_tracking_saturation"
    assert gate.tracking_limited_joints == (3,)

    compare_sample = sim_sample_for_torque_compare(sample, gate)
    assert compare_sample.settled is True
    assert compare_sample.saturated is False
    assert compare_sample.pd_hold_tau[3] == pytest.approx(-0.897, abs=0.05)


# ---------------------------------------------------------------------------
# Reporting, cross-repo compatibility, and torque plots
# ---------------------------------------------------------------------------


def test_report_jsonl_and_csv_roundtrip(tmp_path: Path):
    runtime = get_robot_runtime_profile("xarm6")
    pose = SafePose("0", np.zeros(6), 108.0)
    sample = build_dynamics_sample(pose, _sim_sample(), runtime_profile=runtime, tau_real=np.ones(6), n_real_samples=3)
    run_config = DynamicsRunConfig(robot_key="xarm6_1305", model_path="models/xarm6/xarm6.xml", mode="unit")

    jsonl = tmp_path / "report.jsonl"
    csv_file = tmp_path / "report.csv"
    write_jsonl_report([sample], jsonl, run_config=run_config, runtime_profile=runtime)
    write_csv_report([sample], csv_file, runtime_profile=runtime)

    assert read_report_records(jsonl)[0]["pose"] == "0"
    csv_recs = read_report_records(csv_file)
    assert csv_recs[0]["pose"] == "0"
    assert csv_recs[0]["schema_version"] == REPORT_SCHEMA_VERSION
    csv_text = csv_file.read_text(encoding="utf-8")
    assert "mujoco_tau_J1_nm" in csv_text
    assert "sdk_tau_mean_J1_nm" in csv_text
    assert "l3a_mass_rel_fro" in csv_text


def test_default_dynamics_report_paths_and_explicit_precedence(tmp_path: Path):
    csv_path, jsonl_path, plot_path = dynamics_report_paths(
        identity="XI1305/ABC 123",
        report_root=tmp_path,
        stamp="20261116_170001",
    )
    assert sanitize_report_identity("XI1305/ABC 123") == "XI1305_ABC_123"
    assert (
        csv_path
        == tmp_path / "dyn_ver_XI1305_ABC_123" / "20261116_1700" / "dyn_ver_XI1305_ABC_123_20261116_170001.csv"
    )
    assert jsonl_path == csv_path.with_suffix(".jsonl")
    assert plot_path == csv_path.with_name("dyn_ver_XI1305_ABC_123_20261116_170001_torque.png")

    c2, j2, p2 = dynamics_report_paths(
        identity="XI1305",
        report=tmp_path / "manual.csv",
        jsonl_report=tmp_path / "manual-report.jsonl",
    )
    assert c2 == tmp_path / "manual.csv"
    assert j2 == tmp_path / "manual-report.jsonl"
    assert p2 == tmp_path / "manual_torque.png"


def test_read_report_records_cross_repo_genesis_and_legacy_csv(tmp_path: Path):
    genesis_csv = tmp_path / "genesis_v4.csv"
    genesis_csv.write_text(
        "schema_version,pose,status,torque_l2_err_nm,q_cmd_J1_rad,genesis_tau_J1_nm,sdk_tau_mean_J1_nm,abs_err_J1_nm\n"
        "4,0,PASS,1.0,0.0,3.5,2.0,1.5\n",
        encoding="utf-8",
    )
    rec = read_report_records(genesis_csv)[0]
    assert rec["schema_version"] == "4"
    assert rec["signed_err"] == [1.5]

    legacy_csv = tmp_path / "legacy_v2.csv"
    legacy_csv.write_text(
        "schema_version,pose,status,l2_err,q1,pd_hold_tau1,tau_real1,abs_err1\n"
        "2,home,PASS,1.0,0.0,3.0,1.0,2.0\n",
        encoding="utf-8",
    )
    rec2 = read_report_records(legacy_csv)[0]
    assert rec2["signed_err"] == [2.0]


def test_compare_report_records_and_failure_detail_print(capsys):
    old = [{"signed_err": [1, 2, 3, 4, 5, 6, 7]}]
    new = [{"signed_err": [1, 2, 3, 4, 5, 6, 9]}]
    stats = compare_report_records(old, new)
    assert len(stats) == 7
    assert stats[6]["bias_delta"] == 2

    runtime7 = get_robot_runtime_profile("xarm7")
    pose = SafePose("5", np.zeros(7), 500.0)
    sim = _sim_sample(
        7,
        pd_hold=np.array([0.0, 20.824854, 0.0, 0.0, 0.0, 0.0, 0.0]),
        dof_force=np.zeros(7),
    )
    sample = build_dynamics_sample(
        pose,
        sim,
        runtime_profile=runtime7,
        tau_real=np.array([0.0, 40.824854, 0.0, 0.0, 0.0, 0.0, 0.0]),
        tau_real_std=np.array([0.0, 0.298257, 0.0, 0.0, 0.0, 0.0, 0.0]),
        n_real_samples=31,
    )
    summary = torque_summary(sample, runtime_profile=runtime7)
    assert summary["worst_joint"] == "J2"
    rows = joint_torque_rows(sample, runtime_profile=runtime7)
    assert rows[1]["mujoco_tau_nm"] == 20.824854
    print_compare_table([sample], runtime_profile=runtime7)
    out = capsys.readouterr().out
    assert "mujoco_tau_nm" in out
    assert "20.824854" in out


def test_torque_plot_layouts_generation_and_optional_matplotlib_skip(tmp_path: Path, monkeypatch):
    assert torque_plot_layout(5) == (2, 3)
    assert torque_plot_layout(6) == (2, 3)
    assert torque_plot_layout(7) == (3, 3)

    runtime5 = get_robot_runtime_profile("xarm5")
    pose5 = SafePose("0", np.zeros(5), 112.0)
    sample5 = build_dynamics_sample(
        pose5, _sim_sample(5), runtime_profile=runtime5, tau_real=np.ones(5), n_real_samples=3
    )

    runtime6 = get_robot_runtime_profile("xarm6")
    pose6 = SafePose("0", np.zeros(6), 100.0)
    sim6 = _sim_sample()
    sample6 = build_dynamics_sample(pose6, sim6, runtime_profile=runtime6, tau_real=np.ones(6), n_real_samples=3)

    dry_sample = build_dynamics_sample(pose6, sim6, runtime_profile=runtime6, skip_reason="dry-run")
    assert not write_torque_plot([dry_sample], tmp_path / "dry.png", runtime_profile=runtime6)

    orig_import = builtins.__import__

    def _block_matplotlib(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("blocked for test")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_matplotlib)
    assert not write_torque_plot([sample6], tmp_path / "no_mpl.png", runtime_profile=runtime6)
    monkeypatch.undo()

    pytest.importorskip("matplotlib")
    out5 = tmp_path / "xarm5_torque.png"
    assert write_torque_plot([sample5], out5, runtime_profile=runtime5)
    assert out5.exists() and out5.stat().st_size > 0

    out6 = tmp_path / "xarm6_torque.png"
    assert write_torque_plot([sample6], out6, runtime_profile=runtime6)
    assert out6.exists() and out6.stat().st_size > 0


# ---------------------------------------------------------------------------
# Mocked hardware safety gates and RealRobotSession
# ---------------------------------------------------------------------------


def test_check_controller_tcp_offset_z_blocks_nonzero_offset_below_table():
    session = object.__new__(RealRobotSession)
    arm = MagicMock()
    arm.tcp_offset = [0.0, 0.0, 172.0, 0.0, 0.0, 0.0]

    def fk(angles, input_is_radian=True, return_is_radian=True):
        del input_is_radian, return_is_radian
        if np.allclose(angles, np.zeros(6)):
            return 0, [207.0, 0.0, -60.0, 3.14, 0.0, 0.0]
        return 0, [300.0, 0.0, 200.0, 3.14, 0.0, 0.0]

    arm.get_forward_kinematics.side_effect = fk
    session.arm = arm

    poses = [
        SafePose("0", np.array([0.5, -0.5, 0.0, 0.0, 0.0, 0.0]), 200.0),
        SafePose("9", np.zeros(6), 112.0),
    ]
    reasons = dyn_cli._check_controller_tcp_offset_z(session, poses, np.zeros(6))
    assert reasons == ["home: EE z -60.00 mm <= 0.00 mm"]

    arm.tcp_offset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert dyn_cli._check_controller_tcp_offset_z(session, poses, np.zeros(6)) == []


def test_cli_hardware_check_aborts_before_motion_when_tcp_offset_fails_z_precheck(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(dyn_cli, "_fetch_robot_sn_from_ip", lambda ip: "XI130505L56A01")
    fake_session = MagicMock()
    fake_session.arm.tcp_offset = [0.0, 0.0, 172.0, 0.0, 0.0, 0.0]
    fake_session.arm.get_forward_kinematics.return_value = (0, [207.0, 0.0, -60.0, 3.14, 0.0, 0.0])
    monkeypatch.setattr("ufactory_mujoco.dynamics.session.RealRobotSession", lambda *args, **kwargs: fake_session)

    rc = dyn_cli.cli_hardware_check(
        [
            "--robot",
            "xarm6",
            "--ip",
            "192.168.1.60",
            "--confirm-real",
            "--nominal-model",
            "--poses",
            "0,9",
            "--report",
            str(tmp_path / "abort.csv"),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "Controller has non-zero tcp_offset and failed EE z > 0 precheck" in out
    fake_session.configure_for_dynamics.assert_not_called()
    fake_session.sample_at_hold.assert_not_called()
    fake_session.return_home.assert_not_called()
    fake_session.disconnect.assert_called_once()


def test_configure_for_dynamics_re_enables_after_gravity_and_load():
    arm = MagicMock()
    arm.error_code = 0
    arm.warn_code = 0
    arm.state = 2
    arm.get_err_warn_code.return_value = (0, [0, 0])
    arm.get_state.return_value = (0, 2)
    arm.motion_enable.return_value = 0
    arm.set_mode.return_value = 0
    arm.set_state.return_value = 0
    arm.set_gravity_direction.return_value = 0
    arm.set_tcp_load.return_value = 0
    arm.set_report_tau_or_i.return_value = 0

    session = object.__new__(RealRobotSession)
    session.arm = arm
    session.dof = 6
    session.home_qpos = np.zeros(6)
    session._motion_mode = 0

    session.configure_for_dynamics()
    arm.set_gravity_direction.assert_called_once_with([0.0, 0.0, -1.0])
    arm.set_tcp_load.assert_called_once_with(0, [0, 0, 0])
    arm.set_report_tau_or_i.assert_called_once_with(0)
    assert arm.motion_enable.call_count == 3


def test_ensure_ready_refuses_faulted_controller():
    arm = MagicMock()
    arm.get_err_warn_code.return_value = (0, [22, 0])
    arm.get_state.return_value = (0, 4)

    session = object.__new__(RealRobotSession)
    session.arm = arm
    session._motion_mode = 0
    with pytest.raises(RuntimeError, match="controller is not healthy"):
        session.ensure_ready()
    arm.clean_error.assert_not_called()


def test_real_robot_collect_hold_samples_uses_statistics():
    session = object.__new__(RealRobotSession)
    arm = MagicMock()
    arm.get_joint_states.return_value = (0, [np.zeros(7).tolist(), np.zeros(7).tolist(), np.ones(7).tolist()])
    arm.get_joints_torque.return_value = (0, np.ones(7).tolist())
    session.arm = arm
    session.dof = 6
    session.home_qpos = np.zeros(6)

    sample = session.collect_hold_samples(duration_s=0.0)
    assert sample.n_samples == 1
    np.testing.assert_allclose(sample.tau, np.ones(6))
    np.testing.assert_allclose(sample.tau_direct, np.ones(6))


def test_run_sim_collision_chain_records_order_and_failure():
    arm = MagicMock()
    arm.error_code = 0
    session = object.__new__(RealRobotSession)
    session.arm = arm
    session.dof = 6

    def move_to(q, *, speed_rad_s, wait, move_strategy):
        del speed_rad_s, wait, move_strategy
        if float(np.asarray(q)[0]) == 0.2:
            raise RobotMotionError(
                "collision",
                code=22,
                waypoint_index=1,
                waypoint_count=1,
                target_q=q,
                waypoint_q=q,
            )
        return 0

    session.move_to = move_to
    poses = [
        ("0", np.zeros(6)),
        ("bad", np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("next", np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])),
    ]
    results = dyn_cli.run_sim_collision_chain(
        session,
        poses,
        speed_rad_s=DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S,
        move_strategy="direct",
    )
    assert [r.passed for r in results] == [True, False, True]
    assert results[1].error_code == 22


def test_validate_dynamics_move_speed_and_runtime_defaults():
    assert MAX_DYNAMICS_MOVE_SPEED_DEG_S == 40.0
    assert math.isclose(MAX_DYNAMICS_MOVE_SPEED_RAD_S, math.radians(40.0), rel_tol=0.0, abs_tol=1e-12)
    assert validate_dynamics_move_speed(0.1) == pytest.approx(0.1)
    assert validate_dynamics_move_speed(math.radians(40.0)) <= MAX_DYNAMICS_MOVE_SPEED_RAD_S
    for robot_key in ALL_ROBOTS:
        speed = get_robot_runtime_profile(robot_key).dynamics.default_move_speed_rad_s
        assert 0.0 < speed <= MAX_DYNAMICS_MOVE_SPEED_RAD_S
    for bad_speed in (0.0, -0.1, float("nan"), float("inf"), 0.70, math.radians(40.01)):
        with pytest.raises(ValueError):
            validate_dynamics_move_speed(bad_speed)


def test_cli_and_session_enforce_40deg_speed_ceiling():
    with pytest.raises(SystemExit) as exc_hw:
        dyn_cli.cli_hardware_check(["--robot", "xarm7", "--dry-run", "--speed", "0.70"])
    assert exc_hw.value.code == 2

    with pytest.raises(SystemExit) as exc_simcol:
        dyn_cli.cli_sim_collision_check(["--robot", "xarm7", "--ip", "192.0.2.1", "--speed", "0.70"])
    assert exc_simcol.value.code == 2

    arm = MagicMock()
    session = object.__new__(RealRobotSession)
    session.arm = arm
    session.dof = 6
    session.home_qpos = np.zeros(6)
    session._motion_mode = 0
    with pytest.raises(ValueError, match="40.0 deg/s"):
        session.move_to(np.zeros(6), speed_rad_s=0.70)
    arm.set_servo_angle.assert_not_called()


def test_xarm_robot_caps_joint_speed_when_configured():
    from ufactory_mujoco.safety import authorize_real_motion
    from ufactory_mujoco.xarm_robot import XArmRobot

    calib_path = Path(__file__).resolve().parents[1] / "calibrations" / "xarm6_kinematics_L56A01.yaml"
    arm = MagicMock()
    arm.connected = True
    arm.error_code = 0
    arm.warn_code = 0
    arm.state = 2
    arm.axis = 6
    arm.get_err_warn_code.return_value = (0, [0, 0])
    arm.get_state.return_value = (0, 2)
    arm.get_robot_sn.return_value = (0, "XI130505L56A01")
    arm.get_servo_angle.return_value = (0, [0.0] * 7)
    arm.motion_enable.return_value = 0
    arm.set_mode.return_value = 0
    arm.set_state.return_value = 0
    arm.set_servo_angle.return_value = 0

    approval = authorize_real_motion(
        arm, robot="xarm6", calibration=calib_path, preflight_ok=True, confirm_real=True, cycles=1
    )
    with XArmRobot(
        robot="xarm6",
        calibration=calib_path,
        read_only=False,
        approval=approval,
        arm_api=arm,
        max_joint_speed_rad_s=MAX_DYNAMICS_MOVE_SPEED_RAD_S,
    ) as robot:
        robot.move_joints(np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0]), duration=0.5, wait=True)

    assert arm.set_servo_angle.call_count == 1
    cmd_speed = arm.set_servo_angle.call_args.kwargs["speed"]
    assert cmd_speed <= MAX_DYNAMICS_MOVE_SPEED_RAD_S


# ---------------------------------------------------------------------------
# Five-arm simulation regression and optional Pinocchio URDF cross-check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("robot_key", ALL_ROBOTS)
def test_dynamics_sim_regression_all_robots(tmp_path: Path, robot_key: str):
    runtime = get_robot_runtime_profile(robot_key)
    n_dof = runtime.model.dof
    csv_path = tmp_path / f"sim_{robot_key}.csv"
    jsonl_path = tmp_path / f"sim_{robot_key}.jsonl"
    rc = cli_sim_check(
        [
            "--robot",
            robot_key,
            "--random-count",
            "0",
            "--require-reference",
            "--z-min-mm",
            "0",
            "--report",
            str(csv_path),
            "--jsonl-report",
            str(jsonl_path),
        ]
    )
    assert rc == 0

    records = _load_jsonl(jsonl_path)
    run_config = next(r[1] for r in records if isinstance(r, tuple) and r[0] == "run_config")
    samples = [r for r in records if not isinstance(r, tuple)]
    assert len(samples) == 20
    assert run_config["version"] == "4"
    assert run_config.get("joint_frictionloss") is not None

    bad_poses = [s for s in samples if s["status"] in {"NOT_SETTLED", "SATURATED"}]
    assert bad_poses == []

    for s in samples:
        tau = np.asarray(s["pd_hold_tau"], dtype=np.float64)
        ref_g = np.asarray(s["reference_gravity_tau"], dtype=np.float64)
        mass = np.asarray(s["mass_matrix"], dtype=np.float64)
        assert np.isfinite(tau).all()
        assert np.isfinite(ref_g).all()
        assert mass.shape == (n_dof, n_dof)
        assert np.allclose(mass, mass.T, atol=1e-6)
        assert (np.linalg.eigvalsh(mass) > 0.0).all()
        assert s["l2b_l2_err"] < 0.05
        assert s["l3a_mass_rel_fro"] < 1e-5


def _body_inertia_tensor(mj_model, body_id: int) -> np.ndarray:
    """Reconstruct the body 3x3 inertia tensor in the body frame from MuJoCo body_iquat and body_inertia."""
    w, x, y, z = (float(v) for v in mj_model.body_iquat[body_id])
    rot = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rot @ np.diag(mj_model.body_inertia[body_id]) @ rot.T


def _sync_urdf_link_inertials(urdf_path: Path, mj_model, dof: int, out_path: Path) -> Path:
    """Copy a URDF into ``out_path`` with ``link1..link{dof}`` inertials synced from ``mj_model``."""
    import xml.etree.ElementTree as ET
    import mujoco

    tree = ET.parse(str(urdf_path))
    links = {el.get("name"): el for el in tree.getroot().findall("link")}
    for i in range(1, dof + 1):
        lname = f"link{i}"
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, lname)
        tensor = _body_inertia_tensor(mj_model, bid)
        inert = links[lname].find("inertial")
        assert inert is not None
        origin = inert.find("origin")
        assert origin is not None
        origin.set("xyz", " ".join(f"{float(v):.16g}" for v in mj_model.body_ipos[bid]))
        origin.set("rpy", "0 0 0")
        mass_el = inert.find("mass")
        assert mass_el is not None
        mass_el.set("value", f"{float(mj_model.body_mass[bid]):.16g}")
        iel = inert.find("inertia")
        assert iel is not None
        iel.set("ixx", f"{float(tensor[0, 0]):.16g}")
        iel.set("ixy", f"{float(tensor[0, 1]):.16g}")
        iel.set("ixz", f"{float(tensor[0, 2]):.16g}")
        iel.set("iyy", f"{float(tensor[1, 1]):.16g}")
        iel.set("iyz", f"{float(tensor[1, 2]):.16g}")
        iel.set("izz", f"{float(tensor[2, 2]):.16g}")
    tree.write(out_path, encoding="utf-8")
    return out_path


@pytest.mark.parametrize("robot_key", ALL_ROBOTS)
def test_mujoco_rnea_matches_pinocchio_urdf_reference(robot_key: str, tmp_path: Path):
    pytest.importorskip("pinocchio")
    urdf = default_urdf_for_robot(robot_key)
    if urdf is None:
        pytest.skip(f"Sibling URDF not found for {robot_key}")

    mj_ref = load_reference_backend(robot_key, backend="mujoco-rnea", required=True)
    assert mj_ref is not None
    # Sibling ufactory_genesis URDFs still ship legacy HT_BR2 inertials; sync xArm links in a temp copy.
    if robot_key in {"xarm5", "xarm6", "xarm7"}:
        dof = get_robot_runtime_profile(robot_key).model.dof
        urdf = _sync_urdf_link_inertials(urdf, mj_ref.dyn.model, dof, tmp_path / f"{robot_key}.urdf")
    pin_ref = load_reference_backend(robot_key, backend="pinocchio", urdf_path=urdf, required=True)
    assert pin_ref is not None

    for _pose_name, q in dynamics_default_configs(robot_key)[:6]:
        np.testing.assert_allclose(mj_ref.gravity(q), pin_ref.gravity(q), atol=1e-3)
        m_mj = mj_ref.mass_matrix(q)
        m_pin = pin_ref.mass_matrix(q)
        rel_fro = float(np.linalg.norm(m_mj - m_pin) / np.linalg.norm(m_pin))
        assert rel_fro < 1e-4


@pytest.mark.parametrize(
    ("robot_key", "expected_masses", "expected_coms", "expected_total_mass"),
    [
        (
            "xarm5",
            (2.527, 2.242, 2.24, 1.345, 0.173),
            (
                (-0.000164, 0.03594, -0.0065),
                (0.04084, -0.2403, 0.03254),
                (0.072, 0.2363, 0.00083),
                (0.06824, 0.03342, 0.00264),
                (0.0008, -0.00359, -0.01326),
            ),
            8.5270,
        ),
        (
            "xarm6",
            (2.531, 2.242, 2.019, 1.361, 1.345, 0.173),
            (
                (-0.00017, 0.03575, -0.00647),
                (0.04084, -0.2403, 0.0324),
                (0.0714, 0.1268, 0.0109),
                (0.000185, 0.01466, -0.02197),
                (0.0682, 0.03342, 0.00263),
                (0.00084, -0.00359, -0.01326),
            ),
            9.6710,
        ),
        (
            "xarm7",
            (2.507, 2.2812, 2.0, 2.1212, 1.36, 1.345, 0.17),
            (
                (0.0002, 0.03578, -0.006443),
                (-0.00022, -0.13322, 0.02018),
                (0.0464, -0.022, -0.00804),
                (0.0703, -0.12318, 0.0131),
                (0.000192, 0.014597, -0.02198),
                (0.06824, 0.033418, 0.00264),
                (0.0, -0.0036, -0.01325),
            ),
            11.7844,
        ),
    ],
)
def test_xarm_1305_uj_br_2403_link_inertials(
    robot_key: str,
    expected_masses: tuple[float, ...],
    expected_coms: tuple[tuple[float, float, float], ...],
    expected_total_mass: float,
):
    import mujoco

    runtime = get_robot_runtime_profile(robot_key)
    model = mujoco.MjModel.from_xml_path(str(runtime.model.model_path))
    dof = runtime.model.dof
    assert dof == len(expected_masses)
    masses = []
    for i, (exp_m, exp_com) in enumerate(zip(expected_masses, expected_coms), start=1):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"link{i}")
        assert bid >= 0
        m_val = float(model.body_mass[bid])
        masses.append(m_val)
        assert math.isclose(m_val, exp_m, rel_tol=0.0, abs_tol=1e-12)
        np.testing.assert_allclose(model.body_ipos[bid], exp_com, rtol=0.0, atol=1e-12)
    assert math.isclose(sum(masses), expected_total_mass, rel_tol=0.0, abs_tol=1e-9)
    issues = validate_mjcf_dynamics(model, dof=dof)
    assert [issue for issue in issues if issue.severity == "ERROR"] == []


_XARM_BASE_INERTIAL = (
    2.788411,
    (-0.002587, -0.000269, 0.103569),
    (0.008738440348, 0.008801398397, 0.003241739845, -1.7243198e-05, -0.000169824247, -4.797104e-06),
    (3236492.910e-9, 8739003.273e-9, 8806082.405e-9),
)
_UF850_BASE_INERTIAL = (
    5.869631,
    (-0.000762, -4.4e-05, 0.11761),
    (0.032711870245, 0.03295126767, 0.013458380527, 8.0733473e-05, -0.000343329636, -1.0328638e-05),
    (13452256.142e-9, 32692678.345e-9, 32976583.954e-9),
)


@pytest.mark.parametrize(
    ("robot_key", "base_inertial", "expected_bare_total_mass"),
    [
        ("xarm5", _XARM_BASE_INERTIAL, 11.315411),
        ("xarm6", _XARM_BASE_INERTIAL, 12.459411),
        ("xarm7", _XARM_BASE_INERTIAL, 14.572811),
        ("uf850", _UF850_BASE_INERTIAL, 19.844631),
    ],
)
def test_xarm_and_uf850_link_base_inertials(
    robot_key: str,
    base_inertial: tuple[float, tuple[float, float, float], tuple[float, ...], tuple[float, ...]],
    expected_bare_total_mass: float,
):
    """xArm5/6/7 (Xarm+_P01) and UF850 (Arm850_P01) share SolidWorks link_base inertials."""
    import mujoco

    expected_mass, expected_com, fi, expected_principal = base_inertial
    ixx, iyy, izz, ixy, ixz, iyz = fi
    expected_tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float64)

    runtime = get_robot_runtime_profile(robot_key)
    model = mujoco.MjModel.from_xml_path(str(runtime.model.model_path))
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_base")
    assert bid >= 0
    assert math.isclose(float(model.body_mass[bid]), expected_mass, rel_tol=0.0, abs_tol=1e-12)
    np.testing.assert_allclose(model.body_ipos[bid], expected_com, rtol=0.0, atol=1e-12)

    tensor = _body_inertia_tensor(model, bid)
    np.testing.assert_allclose(tensor, expected_tensor, rtol=0.0, atol=1e-12)
    eig = np.linalg.eigvalsh(tensor)
    assert eig.min() > 0.0 and eig[0] + eig[1] > eig[2]
    np.testing.assert_allclose(eig, expected_principal, rtol=0.0, atol=1e-11)

    total_mass = float(np.sum(model.body_mass))
    assert math.isclose(total_mass, expected_bare_total_mass, rel_tol=0.0, abs_tol=1e-9)
    issues = validate_mjcf_dynamics(model, dof=runtime.model.dof)
    assert [issue for issue in issues if issue.severity == "ERROR"] == []


def test_port502_calibration_decode_and_xarm5_degeneracy_skip(tmp_path: Path, monkeypatch, capsys):
    import struct
    import yaml

    from ufactory_mujoco.calibration import (
        CalibrationError,
        DegenerateCalibrationError,
        decode_kinematics_payload,
        export_kinematics_calibration,
        load_kinematics_calibration,
    )

    sn = "XF130500TEST01"

    # Valid port-502 payloads are preserved directly in meters/radians without heuristic /= 1000 scaling.
    valid_flat = [0.0] * 42
    valid_flat[0 * 6 + 2] = 0.267
    valid_flat[1 * 6 + 3] = -1.5708
    valid_flat[2 * 6 + 0] = 0.0535
    valid_flat[2 * 6 + 1] = -0.2845
    valid_flat[3 * 6 + 0] = 0.0775
    valid_flat[3 * 6 + 1] = 1.25  # > 1.0 m and <= 2.0 m must remain unscaled (in meters)
    valid_flat[4 * 6 + 0] = 0.076
    valid_flat[4 * 6 + 1] = 0.097
    valid_flat[4 * 6 + 3] = -1.5708
    valid_pkt = bytes([0, 1, 0, 2, 0, 173, 8, 16, 1, 5, 5]) + struct.pack("<42f", *valid_flat)
    robot_name, payload = decode_kinematics_payload(valid_pkt, serial_number=sn)
    assert robot_name == "xarm5"
    assert payload["joints"]["joint4"]["y"] == pytest.approx(1.25)

    yaml_path = tmp_path / "xarm5_kinematics_TEST01.yaml"
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    loaded = load_kinematics_calibration(yaml_path, robot="xarm5", serial_number=sn)
    assert loaded.serial_number == sn
    assert loaded.joints["joint4"]["y"] == pytest.approx(1.25)

    # Degenerate port-502 parallel-axis payload (~31.9 m) is rejected as DegenerateCalibrationError.
    bad_flat = [0.0] * 42
    bad_flat[6 + 1] = -31.9366
    bad_flat[12 + 2] = 31.9377
    bad_pkt = bytes([0, 1, 0, 2, 0, 173, 8, 16, 1, 5, 5]) + struct.pack("<42f", *bad_flat)
    with pytest.raises(DegenerateCalibrationError, match="--nominal-model") as exc_info:
        decode_kinematics_payload(bad_pkt, serial_number=sn)
    assert isinstance(exc_info.value, CalibrationError)

    # export_kinematics_calibration skips with exit code 2 and writes no YAML file.
    monkeypatch.setattr("ufactory_mujoco.calibration.fetch_robot_sn_from_ip", lambda _ip: sn)
    monkeypatch.setattr("ufactory_mujoco.calibration.fetch_kinematics_bytes", lambda _ip, timeout=10.0: bad_pkt)
    out_dir = tmp_path / "out_cal"
    capsys.readouterr()
    rc = export_kinematics_calibration("192.0.2.76", out_dir)
    captured = capsys.readouterr()
    assert rc == 2
    assert "[Skipped]" in captured.err
    assert "--nominal-model" in captured.err
    assert not out_dir.exists() or list(out_dir.iterdir()) == []


def test_hold_current_only_is_strictly_read_only_and_rejects_conflicting_flags(monkeypatch):
    with pytest.raises(SystemExit) as exc_dry:
        dyn_cli.cli_hardware_check(["--robot", "xarm6", "--dry-run", "--hold-current-only"])
    assert exc_dry.value.code == 2

    with pytest.raises(SystemExit) as exc_clear:
        dyn_cli.cli_hardware_check(
            ["--robot", "xarm6", "--ip", "192.0.2.1", "--hold-current-only", "--clear-tcp-offset"]
        )
    assert exc_clear.value.code == 2

    monkeypatch.setattr(dyn_cli, "_fetch_robot_sn_from_ip", lambda ip: "XI130505L56A01")
    fake_session = MagicMock()
    fake_session.arm.tcp_load = [0.82, [0.0, 0.0, 48.0]]
    fake_session.get_joint_states.return_value = (np.zeros(6), np.zeros(6), np.zeros(6))
    fake_session.get_joints_torque.return_value = np.zeros(6)
    monkeypatch.setattr("ufactory_mujoco.dynamics.session.RealRobotSession", lambda *args, **kwargs: fake_session)

    rc = dyn_cli.cli_hardware_check(
        ["--robot", "xarm6", "--ip", "192.0.2.1", "--hold-current-only", "--nominal-model", "--poses", "0"]
    )
    assert rc == 0
    fake_session.configure_for_dynamics.assert_not_called()
    fake_session.disconnect.assert_called_once()


def test_cli_hardware_check_blocks_nonzero_tcp_load_and_session_restores_config(monkeypatch, tmp_path):
    monkeypatch.setattr(dyn_cli, "_fetch_robot_sn_from_ip", lambda ip: "XI130505L56A01")
    fake_session = MagicMock()
    fake_session.arm.tcp_load = [0.82, [0.0, 0.0, 48.0]]
    monkeypatch.setattr("ufactory_mujoco.dynamics.session.RealRobotSession", lambda *args, **kwargs: fake_session)

    rc = dyn_cli.cli_hardware_check(
        [
            "--robot", "xarm6", "--ip", "192.0.2.1", "--confirm-real",
            "--nominal-model", "--poses", "0", "--report", str(tmp_path / "load_block.csv"),
        ]
    )
    assert rc == 1
    fake_session.configure_for_dynamics.assert_not_called()
    fake_session.sample_at_hold.assert_not_called()

    # Verify RealRobotSession.restore_controller_config restores captured load/offset/collision
    arm = MagicMock()
    arm.error_code = 0
    arm.warn_code = 0
    arm.state = 2
    arm.tcp_load = [0.82, [0.0, 0.0, 48.0]]
    arm.tcp_offset = [0.0, 0.0, 172.0, 0.0, 0.0, 0.0]
    arm.gravity_direction = [0.0, 0.0, -1.0]
    arm.self_collision_params = [False, 0, [0] * 6]
    arm.get_report_tau_or_i.return_value = (0, 1)
    arm.get_err_warn_code.return_value = (0, [0, 0])
    arm.get_state.return_value = (0, 2)
    arm.motion_enable.return_value = 0
    arm.set_mode.return_value = 0
    arm.set_state.return_value = 0
    arm.set_tcp_offset.return_value = 0
    arm.set_gravity_direction.return_value = 0
    arm.set_tcp_load.return_value = 0
    arm.set_report_tau_or_i.return_value = 0
    arm.set_simulation_robot.return_value = 0
    arm.is_simulation_robot = True
    arm.set_self_collision_detection.return_value = 0

    session = object.__new__(RealRobotSession)
    session.arm = arm
    session.dof = 6
    session.home_qpos = np.zeros(6)
    session._motion_mode = 0
    session._config_snapshot = None
    session._modified_dynamics = False
    session._modified_tcp_offset = False
    session._modified_self_collision = False

    session.configure_for_simulation_collision_check(clear_tcp_offset=True)
    arm.set_tcp_load.reset_mock()
    arm.set_tcp_offset.reset_mock()
    arm.set_self_collision_detection.reset_mock()
    arm.set_report_tau_or_i.reset_mock()

    errors = session.restore_controller_config()
    assert errors == []
    arm.set_tcp_load.assert_called_once_with(0.82, [0.0, 0.0, 48.0])
    arm.set_tcp_offset.assert_called_once_with([0.0, 0.0, 172.0, 0.0, 0.0, 0.0])
    arm.set_self_collision_detection.assert_called_once_with(False)
    arm.set_report_tau_or_i.assert_called_once_with(1)


def test_cli_hardware_check_ctrl_c_stops_without_return_home_and_writes_partial(monkeypatch, tmp_path):
    from ufactory_mujoco.dynamics.session import RealRobotSample

    monkeypatch.setattr(dyn_cli, "_fetch_robot_sn_from_ip", lambda ip: "XI130505L56A01")
    fake_session = MagicMock()
    fake_session.arm.tcp_load = [0.0, [0.0, 0.0, 0.0]]
    fake_session.arm.tcp_offset = [0.0] * 6
    fake_session.arm.get_robot_sn.return_value = (0, "XI130505L56A01")
    fake_session.arm.get_forward_kinematics.return_value = (0, [300.0, 0.0, 300.0, 3.14, 0.0, 0.0])
    fake_session.get_joint_states.return_value = (np.zeros(6), np.zeros(6), np.zeros(6))
    fake_session.restore_controller_config.return_value = []

    calls = {"n": 0}

    def sample_at_hold(*args, **kwargs):
        del args, kwargs
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return RealRobotSample(q=np.zeros(6), qvel=np.zeros(6), tau=np.zeros(6), settled=True, n_samples=5)

    fake_session.sample_at_hold.side_effect = sample_at_hold
    monkeypatch.setattr("ufactory_mujoco.dynamics.session.RealRobotSession", lambda *args, **kwargs: fake_session)

    csv_path = tmp_path / "partial.csv"
    jsonl_path = tmp_path / "partial.jsonl"
    rc = dyn_cli.cli_hardware_check(
        [
            "--robot", "xarm6", "--ip", "192.0.2.1", "--confirm-real", "--nominal-model",
            "--poses", "0,1", "--report", str(csv_path), "--jsonl-report", str(jsonl_path),
        ]
    )
    assert rc == 130
    fake_session.stop.assert_called()
    fake_session.return_home.assert_not_called()
    assert csv_path.is_file() and jsonl_path.is_file()


def test_sim_collision_check_requires_confirm_real_and_catches_faulted_runtime_error():
    with pytest.raises(SystemExit) as exc_gate:
        dyn_cli.cli_sim_collision_check(["--robot", "xarm6", "--ip", "192.0.2.1"])
    assert exc_gate.value.code == 2

    arm = MagicMock()
    arm.error_code = 0
    session = object.__new__(RealRobotSession)
    session.arm = arm
    session.dof = 6

    def move_to(q, *, speed_rad_s, wait, move_strategy):
        del speed_rad_s, wait, move_strategy
        val = float(np.asarray(q)[0])
        if val == 0.2:
            arm.error_code = 22
            raise RobotMotionError(
                "collision", code=22, waypoint_index=1, waypoint_count=1, target_q=q, waypoint_q=q
            )
        if arm.error_code != 0:
            raise RuntimeError("controller is not healthy (error_code=22)")
        return 0

    session.move_to = move_to
    poses = [
        ("0", np.zeros(6)),
        ("bad", np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("after_fault", np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])),
    ]
    results = dyn_cli.run_sim_collision_chain(
        session, poses, speed_rad_s=DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S, move_strategy="direct"
    )
    assert [r.passed for r in results] == [True, False, False]
    assert results[2].error_code == 22


def test_compare_report_records_rejects_mismatched_joint_counts(tmp_path: Path):
    from ufactory_mujoco.dynamics.report import compare_report_records

    rec6 = [{"signed_err": [0.1] * 6}]
    rec7 = [{"signed_err": [0.1] * 7}]
    with pytest.raises(ValueError, match="mismatched joint counts"):
        compare_report_records(rec6, rec7)

    import json

    f6 = tmp_path / "r6.jsonl"
    f7 = tmp_path / "r7.jsonl"
    f6.write_text(json.dumps({"type": "sample", "data": {"signed_err": [0.1] * 6}}) + "\n", encoding="utf-8")
    f7.write_text(json.dumps({"type": "sample", "data": {"signed_err": [0.1] * 7}}) + "\n", encoding="utf-8")
    assert dyn_cli.cli_report_compare([str(f6), str(f7)]) == 2


def test_cli_hardware_check_blocks_enabled_ft_sensor_and_hold_only_reads_ft(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(dyn_cli, "_fetch_robot_sn_from_ip", lambda ip: "XI130505L56A01")
    fake_session = MagicMock()
    fake_session.arm.tcp_load = [0.0, [0.0, 0.0, 0.0]]
    fake_session.arm.tcp_offset = [0.0] * 6
    fake_session.arm.ft_sensor_is_enable = True
    fake_session.arm.get_err_warn_code.return_value = (0, (0, 0))
    fake_session.arm.get_ft_sensor_config.return_value = (
        0,
        [0, 1, 1, 1, 200, 0.17, 0.0, [0.0, 0.0, 0.0042], [0.0] * 6],
    )
    fake_session.arm.get_ft_sensor_data.side_effect = (
        lambda is_raw=False: (
            0,
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3] if is_raw else [0.01, -0.02, 0.03, 0.0, 0.0, 0.0],
        )
    )
    fake_session.get_joint_states.return_value = (np.zeros(6), np.zeros(6), np.zeros(6))
    fake_session.get_joints_torque.return_value = np.zeros(6)
    monkeypatch.setattr("ufactory_mujoco.dynamics.session.RealRobotSession", lambda *args, **kwargs: fake_session)

    # Hold-current-only succeeds strictly read-only and prints F/T telemetry
    rc_hold = dyn_cli.cli_hardware_check(
        ["--robot", "xarm6", "--ip", "192.0.2.1", "--hold-current-only", "--nominal-model", "--poses", "0"]
    )
    assert rc_hold == 0
    out_hold = capsys.readouterr().out
    assert "enabled F/T sensor" in out_hold
    assert "ft_raw (F/T)" in out_hold and "ft_ext (F/T)" in out_hold
    fake_session.configure_for_dynamics.assert_not_called()

    # Non-hold bare-arm check refuses motion with --gripper ft_sensor hint
    rc_block = dyn_cli.cli_hardware_check(
        [
            "--robot",
            "xarm6",
            "--ip",
            "192.0.2.1",
            "--confirm-real",
            "--nominal-model",
            "--poses",
            "0",
            "--report",
            str(tmp_path / "ft_block.csv"),
        ]
    )
    assert rc_block == 1
    out_block = capsys.readouterr().out
    assert "--gripper ft_sensor" in out_block
    fake_session.sample_at_hold.assert_not_called()
    fake_session.arm.get_ft_sensor_error.assert_not_called()
