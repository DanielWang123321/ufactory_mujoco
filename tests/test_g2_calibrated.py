"""Unit and closed-loop MuJoCo tests for the calibrated Gripper G2 model and controller."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import unittest

import mujoco
import numpy as np

from ufactory_mujoco.g2_controller import G2Controller
from ufactory_mujoco.g2_model import (
    G2_ARMATURE,
    G2_DAMPING,
    G2_FRICTIONLOSS,
    G2_JOINT_NAMES,
    G2_KP,
    G2_KV,
    G2_LINKAGE_GEOMETRY,
    G2_MIMIC_SOLIMP,
    G2_MIMIC_SOLREF,
    GRIPPER_G2_SIM_CLOSE_DRIVE,
    GRIPPER_G2_SIM_OPEN_DRIVE,
    REFERENCE_GAP_MM,
    REFERENCE_RAW_POSITION,
    G2Model,
    gripper_g2_gap_m_to_sim_drive,
    gripper_g2_sim_drive_to_gap_m,
    sdk_position_counts,
    sdk_position_mm,
    sdk_speed_rpm,
    speed_mm_s,
)
from ufactory_mujoco.model_paths import MODELS_ROOT
from ufactory_mujoco.sim_robot import SimRobot
from ufactory_mujoco.task_geometry import GRIP_DOWN_QUAT, tcp_task_heights

try:
    import pytest
except ImportError as exc:
    raise unittest.SkipTest("pytest is required for test_g2_calibrated") from exc

REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "g2_controller_reference.json").read_text())
FIELDS = REFERENCE["step_fields"]
OUTPUTS = ("position_rad", "velocity_rad_s", "reference_gap_mm", "integral_torque_nm", "torque_limit_nm")


@pytest.fixture(scope="module")
def model() -> G2Model:
    return G2Model.load()


# ---------------------------------------------------------------------------
# 1. Calibrated G2Model & G2LinkageGeometry tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "percent,mm_s,rpm",
    [(0, 0, 0), (1, 15, 577), (5, 15, 577), (10, 22.5, 691), (50, 112.5, 2057), (90, 202.5, 3423), (100, 225, 3765)],
)
def test_percent_speed_and_sdk_register_truncation(percent, mm_s, rpm):
    assert speed_mm_s(percent) == mm_s
    assert sdk_speed_rpm(mm_s) == rpm


def test_reference_retains_sdk_raw_count_and_inverse():
    assert REFERENCE_RAW_POSITION == 230
    assert REFERENCE_GAP_MM == pytest.approx(24.15587200800565, abs=1e-12)
    assert sdk_position_counts(0) == 0
    assert sdk_position_counts(84) == 850


@pytest.mark.parametrize("position", [0, 10, 24.155872, 30, 60, 84])
def test_sdk_position_quantization_is_at_most_one_count(position):
    raw = sdk_position_counts(position)
    restored = sdk_position_mm(raw)
    assert abs(restored - position) < 0.106


@pytest.mark.parametrize(
    "function,value",
    [
        (speed_mm_s, -1),
        (speed_mm_s, 101),
        (speed_mm_s, math.nan),
        (sdk_speed_rpm, 226),
        (sdk_position_counts, -0.1),
        (sdk_position_counts, 84.1),
        (sdk_position_counts, math.inf),
    ],
)
def test_invalid_units_are_rejected(function, value):
    with pytest.raises(ValueError):
        function(value)


def test_curve_retains_plateau_and_sharp_force_change(model):
    assert model.curve.lookup(3000, 1).force_n == model.curve.lookup(3000, 10).force_n
    assert model.curve.lookup(0, 55).force_n - model.curve.lookup(0, 50).force_n > 8
    assert model.curve.maximum_force_n == 63.79


def test_piecewise_current_and_speed_interpolation(model):
    a = model.curve.lookup(3000, 45).force_n
    b = model.curve.lookup(3000, 50).force_n
    assert model.curve.lookup(3000, 47.5).force_n == pytest.approx((a + b) / 2)
    at_3250 = model.curve.lookup(3250, 47.5).force_n
    assert model.curve.lookup(3125, 47.5).force_n == pytest.approx((a + b + 2 * at_3250) / 4)


@pytest.mark.parametrize("gap_mm", [0, 10, REFERENCE_GAP_MM, 30, 60, 84])
def test_physical_endpoints_inverse_and_gap_jacobian(model, gap_mm):
    linkage = model.linkage
    q = linkage.drive_for_gap_m(gap_mm / 1000)
    assert linkage.pad_gap_m(q) == pytest.approx(gap_mm / 1000, abs=1e-12)
    step = 1e-6
    numerical = (linkage.pad_gap_m(q - step) - linkage.pad_gap_m(q + step)) / (2 * step)
    assert linkage.gap_jacobian_m_per_rad(q) == pytest.approx(numerical, rel=1e-8)


def test_public_geometry_matches_the_urdf_collision_pads(model):
    assert asdict(model.linkage) == asdict(G2_LINKAGE_GEOMETRY)
    assert model.linkage.pad_gap_m(GRIPPER_G2_SIM_OPEN_DRIVE) * 1000 == pytest.approx(84, abs=1e-6)
    assert model.linkage.pad_gap_m(GRIPPER_G2_SIM_CLOSE_DRIVE) * 1000 == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("gap_mm", [0, 10, REFERENCE_GAP_MM, 30, 60, 84])
def test_public_scalar_conversion_matches_pad_geometry(gap_mm):
    q = gripper_g2_gap_m_to_sim_drive(gap_mm / 1000)
    assert G2_LINKAGE_GEOMETRY.pad_gap_m(q) * 1000 == pytest.approx(gap_mm, abs=1e-6)
    assert gripper_g2_sim_drive_to_gap_m(q) * 1000 == pytest.approx(gap_mm, abs=1e-6)


def test_scalar_conversion_is_monotonic_and_clamps_to_nominal_stroke():
    gaps = (0, 10, REFERENCE_GAP_MM, 30, 60, 84)
    angles = [gripper_g2_gap_m_to_sim_drive(gap / 1000) for gap in gaps]
    assert all(a > b for a, b in zip(angles, angles[1:]))
    assert gripper_g2_gap_m_to_sim_drive(-0.01) == pytest.approx(GRIPPER_G2_SIM_CLOSE_DRIVE)
    assert gripper_g2_gap_m_to_sim_drive(0.1) == pytest.approx(GRIPPER_G2_SIM_OPEN_DRIVE)
    assert gripper_g2_sim_drive_to_gap_m(0) == pytest.approx(0.084)
    assert gripper_g2_sim_drive_to_gap_m(1) == pytest.approx(0)


@pytest.mark.parametrize("speed,force", [(10, 10), (50, 30), (50, 50), (90, 100)])
def test_two_pad_virtual_work_at_reference_gap(model, speed, force):
    lookup = model.curve.lookup(sdk_speed_rpm(speed_mm_s(speed)), force)
    linkage = model.linkage
    q = linkage.drive_for_gap_m(REFERENCE_GAP_MM / 1000)
    step = 1e-6
    full_gap_travel = linkage.pad_gap_m(q - step) - linkage.pad_gap_m(q + step)
    two_pad_work = 2 * lookup.force_n * full_gap_travel / 2
    joint_work = model.torque_limit_nm(lookup) * 2 * step
    assert joint_work == pytest.approx(two_pad_work, rel=1e-8)
    assert model.torque_limit_nm(lookup) == pytest.approx(lookup.force_n * model.reference_jacobian_m_rad, rel=1e-12)


@pytest.mark.parametrize("rpm,force,wide_gap_mm,narrow_gap_mm", [(2057, 50, 80, 20), (3500, 100, 70, 10)])
def test_fixed_torque_changes_force_with_gap(model, rpm, force, wide_gap_mm, narrow_gap_mm):
    lookup = model.curve.lookup(rpm, force)
    assert model.static_force_n(lookup, wide_gap_mm) > model.static_force_n(lookup, narrow_gap_mm)


def test_rated_torque_does_not_clip_curve(model):
    lookup = model.curve.lookup(3500, 100)
    tau = model.torque_limit_nm(lookup)
    assert model.rated_motor_torque_nm == 0.22
    assert model.motor_torque_estimate_nm(tau) > model.rated_motor_torque_nm
    assert model.motor_torque_estimate_nm(tau) <= model.peak_motor_torque_nm
    maximum = model.curve.lookup(0, 100)
    assert model.motor_torque_estimate_nm(model.torque_limit_nm(maximum)) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 2. G2Controller reference parity & vectorized tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype,tolerance_mm", [(np.float64, 1e-9), (np.float32, 2e-5)])
def test_array_geometry_matches_scalar_conversions(model, dtype, tolerance_mm):
    gaps = np.array([0, 10, REFERENCE_GAP_MM, 30, 60, 84], dtype=dtype)
    controller = G2Controller(model, num_envs=len(gaps), dtype=dtype)
    expected = np.array([gripper_g2_gap_m_to_sim_drive(g / 1000) for g in gaps.tolist()], dtype=dtype)
    q = controller.drive_for_gap_mm(gaps)
    np.testing.assert_allclose(q, expected, rtol=0, atol=tolerance_mm / 80)
    np.testing.assert_allclose(controller.gap_mm(q), gaps, rtol=0, atol=tolerance_mm)
    scalar_gaps = np.array([gripper_g2_sim_drive_to_gap_m(float(a)) for a in q], dtype=dtype)
    np.testing.assert_allclose(controller.gap_mm(q) / 1000, scalar_gaps, rtol=0, atol=tolerance_mm / 1000)
    assert bool((q[:-1] > q[1:]).all())


def _replay_reference(model, case, dtype, tolerance):
    controller = G2Controller(model, initial_position_mm=case["initial_position_mm"], dtype=dtype)
    if case["reset"] is not None:
        controller.reset(**case["reset"])
    for event in case["commands"]:
        controller.command(*event["command"])
        assert float(controller.requested_speed_mm_s[0]) == pytest.approx(event["requested_speed_mm_s"], abs=1e-9)
        assert int(controller.requested_speed_rpm[0]) == event["requested_speed_rpm"]
        for step in event["steps"]:
            row = dict(zip(FIELDS, step, strict=True))
            output = controller.update(
                row["q"], row["velocity"], REFERENCE["dt"], measured_gap_mm=row["measured_gap_mm"]
            )
            for field in OUTPUTS:
                assert float(getattr(output, field)[0]) == pytest.approx(row[field], abs=tolerance), field
            lookup = controller.force_lookup()
            assert lookup.force_n == pytest.approx(row["force_n"], abs=1e-12)
            assert lookup.calibration_rpm == row["calibration_rpm"]
            assert lookup.outside_calibration == bool(row["outside_calibration"])


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=[case["name"] for case in REFERENCE["cases"]])
def test_controller_matches_recorded_scalar_reference(model, case):
    _replay_reference(model, case, np.float64, 1e-10)


@pytest.mark.parametrize("name", ["speed_force_grid", "reverse_mid_stroke"])
def test_float32_controller_tracks_recorded_scalar_reference(model, name):
    case = next(case for case in REFERENCE["cases"] if case["name"] == name)
    _replay_reference(model, case, np.float32, 2e-4)


@pytest.mark.parametrize("dtype,tolerance", [(np.float64, 0.0), (np.float32, 0.0)])
def test_batch_rows_match_independent_single_environment_controllers(model, dtype, tolerance):
    batch = G2Controller(model, num_envs=4, dtype=dtype)
    single = [G2Controller(model, dtype=dtype) for _ in range(4)]
    q = np.array([model.linkage.drive_for_gap_m(g / 1000) for g in (24, 30, 50, 70)], dtype=dtype)
    velocity = np.array([0, 1, -0.5, 0.01], dtype=dtype)
    gaps = [23, 29, 49, 69]
    for position, speeds, forces in [
        ([0, 10, 20, 30], [10, 50, 90, 100], [10, 30, 50, 100]),
        ([0, 10, 20, 30], [0, 0, 0, 0], [10, 30, 80, 100]),
        ([84, 84, 0, 0], [0, 50, 90, 100], [80, 30, 50, 10]),
        ([84, 84, 84, 84], [50, 90, 10, 50], [80, 50, 100, 30]),
    ]:
        batch.command(position, speeds, forces)
        for i, c in enumerate(single):
            c.command(position[i], speeds[i], forces[i])
        for _ in range(12):
            actual = batch.update(q, velocity, measured_gap_mm=gaps)
            expected = [
                c.update(q[i : i + 1], velocity[i : i + 1], measured_gap_mm=gaps[i]) for i, c in enumerate(single)
            ]
            for field in OUTPUTS:
                np.testing.assert_allclose(
                    getattr(actual, field),
                    np.concatenate([getattr(out, field) for out in expected]),
                    rtol=0,
                    atol=tolerance,
                )


def test_reset_is_independent_and_preheld_reset_seeds_torque(model):
    batch = G2Controller(model, num_envs=4)
    batch.command(0, [10, 50, 90, 100], [10, 30, 50, 100])
    batch.update(np.zeros(4), np.zeros(4))
    untouched = {
        key: value[[1, 3]].copy()
        for key, value in vars(batch).items()
        if isinstance(value, np.ndarray) and value.shape == (4,)
    }
    batch.reset([0, 2], position_mm=[30, 40], target_mm=[0, 10], prime=True)
    for key, value in untouched.items():
        np.testing.assert_array_equal(getattr(batch, key)[[1, 3]], value)
    np.testing.assert_array_equal(batch.integral_torque_nm[[0, 2]], batch.torque_limit_nm[[0, 2]])
    batch.reset([2], position_mm=84)
    assert batch.integral_torque_nm[2] == 0
    assert batch.reference_gap_mm[2] == batch.position_mm[2] == 84


def test_independent_torque_limits_and_atomic_invalid_batch_command(model):
    batch = G2Controller(model, num_envs=4, dtype=np.float32)
    batch.command(0, [10, 50, 90, 100], [10, 30, 80, 100])
    assert batch.torque_limit_nm.tolist() == pytest.approx(
        [model.torque_limit_nm(batch.force_lookup(i)) for i in range(4)], rel=1e-6
    )
    before = batch.position_mm.copy()
    with pytest.raises(ValueError):
        batch.command([84, 84, 85, 84], 50, 50)
    np.testing.assert_array_equal(before, batch.position_mm)


def test_num_envs_is_keyword_only(model):
    with pytest.raises(TypeError):
        G2Controller(model, 50)


def _value(arr) -> float:
    return float(arr[0])


def test_speed_above_chart_clamps_force_only(model):
    controller = G2Controller(model)
    controller.command(0, 100, 80)
    assert _value(controller.requested_speed_mm_s) == 225
    assert _value(controller.requested_speed_rpm) == 3765
    lookup = controller.force_lookup()
    assert lookup.calibration_rpm == 3500
    assert lookup.outside_calibration
    assert lookup.force_n == model.curve.lookup(3500, 80).force_n


def test_reference_advances_at_command_speed_without_overshooting_target(model):
    controller = G2Controller(model)
    controller.command(80, 10, 50)
    q = model.linkage.drive_for_gap_m(0.084)
    for _ in range(20):
        previous = _value(controller.reference_gap_mm)
        output = controller.update(q, 0)
        assert 0 <= previous - _value(output.reference_gap_mm) <= 0.450000001
        assert _value(output.reference_gap_mm) >= 80
    assert _value(controller.reference_gap_mm) == 80


def test_pause_holds_preload_and_last_nonzero_calibration_then_resumes(model):
    c = G2Controller(model, initial_position_mm=0)
    c.command(0, 90, 80)
    q = model.linkage.drive_for_gap_m(REFERENCE_GAP_MM / 1000)
    c.update(q, 0)
    before = c.force_lookup()
    c.command(0, 0, 80)
    paused = c.update(q, 0)
    assert _value(paused.reference_gap_mm) == pytest.approx(REFERENCE_GAP_MM)
    assert _value(paused.velocity_rad_s) == 0
    assert _value(paused.integral_torque_nm) == pytest.approx(_value(paused.torque_limit_nm))
    assert c.force_lookup() == before
    assert _value(c.requested_speed_rpm) == 0
    for _ in range(20):
        assert _value(c.update(q, 0).reference_gap_mm) == _value(paused.reference_gap_mm)
    c.command(0, 90, 80)
    resumed = c.update(q, 0)
    assert _value(resumed.reference_gap_mm) < _value(paused.reference_gap_mm)
    assert _value(resumed.velocity_rad_s) > 0


def test_paused_force_change_updates_saturated_holding_effort(model):
    c = G2Controller(model, initial_position_mm=0)
    q = model.linkage.drive_for_gap_m(REFERENCE_GAP_MM / 1000)
    c.command(0, 50, 30)
    c.update(q, 0)
    c.command(0, 0, 30)
    before = c.update(q, 0)
    c.command(0, 0, 80)
    after = c.update(q, 0)
    assert _value(after.torque_limit_nm) > _value(before.torque_limit_nm)
    assert _value(after.integral_torque_nm) == pytest.approx(_value(after.torque_limit_nm))
    compressed_q = q + 0.005
    for _ in range(500):
        held = c.update(compressed_q, 0)
    assert _value(held.integral_torque_nm) == pytest.approx(_value(held.torque_limit_nm))
    assert c.kp * (_value(held.position_rad) - compressed_q) == pytest.approx(_value(held.torque_limit_nm))


def test_paused_new_position_keeps_grasp_until_nonzero_speed(model):
    c = G2Controller(model, initial_position_mm=0)
    q = model.linkage.drive_for_gap_m(REFERENCE_GAP_MM / 1000)
    c.command(0, 50, 80)
    c.update(q, 0)
    c.command(84, 0, 80)
    paused = c.update(q, 0)
    assert _value(paused.reference_gap_mm) == pytest.approx(REFERENCE_GAP_MM)
    assert _value(paused.integral_torque_nm) == pytest.approx(_value(paused.torque_limit_nm))
    c.command(84, 50, 80)
    opened = c.update(q, 0)
    assert _value(opened.reference_gap_mm) > REFERENCE_GAP_MM
    assert _value(opened.velocity_rad_s) < 0
    assert _value(opened.integral_torque_nm) <= 0


def test_moving_pause_uses_actual_pad_position_but_stopped_pause_keeps_joint_posture(model):
    c = G2Controller(model, initial_position_mm=50)
    q = model.linkage.drive_for_gap_m(0.050)
    c.command(0, 0, 80)
    assert _value(c.update(q, 1.0, measured_gap_mm=47.0).reference_gap_mm) == 47.0
    c = G2Controller(model, initial_position_mm=0)
    c.command(0, 50, 80)
    c.update(q, 0)
    c.command(0, 0, 80)
    assert _value(c.update(q, 0, measured_gap_mm=47.0).reference_gap_mm) == pytest.approx(50.0)


def test_conditional_integration_stops_windup_and_allows_unwinding(model):
    c = G2Controller(model, initial_position_mm=0)
    c.command(0, 50, 10)
    q = model.linkage.drive_for_gap_m(REFERENCE_GAP_MM / 1000)
    for _ in range(500):
        c.update(q, 0)
    assert _value(c.integral_torque_nm) == 0
    c = G2Controller(model, initial_position_mm=30)
    limit = model.torque_limit_nm(c.force_lookup())
    c.integral_torque_nm.fill(limit)
    narrower_q = model.linkage.drive_for_gap_m(0.029)
    c.update(narrower_q, -2)
    assert 0 < _value(c.integral_torque_nm) < limit


def test_reversal_clears_integral_and_rebases_to_observed_position(model):
    c = G2Controller(model, initial_position_mm=0)
    c.command(0, 50, 80)
    c.integral_torque_nm.fill(model.torque_limit_nm(c.force_lookup()))
    q = model.linkage.drive_for_gap_m(REFERENCE_GAP_MM / 1000)
    c.command(84, 50, 80)
    output = c.update(q, 0)
    assert _value(output.reference_gap_mm) == pytest.approx(REFERENCE_GAP_MM + 2.25)
    assert _value(output.integral_torque_nm) <= 0
    assert _value(output.velocity_rad_s) < 0


@pytest.mark.parametrize(
    "command", [(-1, 50, 50), (85, 50, 50), (0, -1, 50), (0, 101, 50), (0, 50, 0), (0, 50, 101), (0, math.nan, 50)]
)
def test_invalid_command_is_atomic(model, command):
    c = G2Controller(model)
    before = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in vars(c).items()}
    with pytest.raises(ValueError):
        c.command(*command)
    for key, value in vars(c).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, before[key])
        else:
            assert value == before[key], key


# ---------------------------------------------------------------------------
# 3. MJCF model parameter & closed-loop MuJoCo speed/force control tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name",
    [
        "gripper_g2",
        "xarm5_with_gripper",
        "xarm6_with_gripper",
        "xarm7_with_gripper",
        "uf850_with_gripper",
    ],
)
def test_mjcf_g2_models_match_calibrated_parameters(model_name):
    xml_path = MODELS_ROOT / model_name / f"{model_name}.xml"
    m = mujoco.MjModel.from_xml_path(str(xml_path))

    for jnt_name in G2_JOINT_NAMES:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
        assert jid >= 0, f"Missing joint {jnt_name} in {model_name}"
        dof_id = int(m.jnt_dofadr[jid])
        np.testing.assert_allclose(
            m.jnt_range[jid],
            [GRIPPER_G2_SIM_OPEN_DRIVE, GRIPPER_G2_SIM_CLOSE_DRIVE],
            atol=1e-12,
        )
        assert float(m.dof_armature[dof_id]) == pytest.approx(G2_ARMATURE, abs=1e-12)
        assert float(m.dof_damping[dof_id]) == pytest.approx(G2_DAMPING, abs=1e-12)
        assert float(m.dof_frictionloss[dof_id]) == pytest.approx(G2_FRICTIONLOSS, abs=1e-12)
        assert float(m.jnt_stiffness[jid]) == pytest.approx(0.0, abs=1e-12)

    assert m.neq == 5
    for eq_id in range(m.neq):
        np.testing.assert_allclose(m.eq_solref[eq_id], G2_MIMIC_SOLREF, atol=1e-12)
        np.testing.assert_allclose(m.eq_solimp[eq_id], G2_MIMIC_SOLIMP, atol=1e-12)

    act_name = "gripper_motor" if model_name == "gripper_g2" else "gripper"
    act_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
    assert act_id >= 0
    assert float(m.actuator_gainprm[act_id, 0]) == pytest.approx(G2_KP, abs=1e-12)
    assert float(m.actuator_biasprm[act_id, 1]) == pytest.approx(-G2_KP, abs=1e-12)
    assert float(m.actuator_biasprm[act_id, 2]) == pytest.approx(-G2_KV, abs=1e-12)


def test_mujoco_closed_loop_speed_control():
    scene = MODELS_ROOT / "gripper_g2" / "scene.xml"

    def steps_to_reach_gap(speed_pct: float, target_gap_m: float = 0.015) -> int:
        with SimRobot(scene) as robot:
            robot.set_gripper_command(0.0, speed_percent=speed_pct, force_percent=80.0)
            for step in range(1, 2500):
                robot._step()
                if robot.get_measured_gripper_gap() <= target_gap_m:
                    return step
        return 2500

    steps_slow = steps_to_reach_gap(15.0)
    steps_mid = steps_to_reach_gap(50.0)
    steps_fast = steps_to_reach_gap(100.0)
    assert steps_fast < steps_mid < steps_slow
    assert steps_slow > 2 * steps_fast


def test_mujoco_closed_loop_force_control_monotonicity():
    from ufactory_mujoco.config import load_task_config
    from ufactory_mujoco.resource_paths import CONFIGS_ROOT

    cfg = load_task_config(CONFIGS_ROOT / "pick_place" / "xarm6.yaml")
    pp = cfg.pick_place

    forces_measured = []
    torque_limits = []

    for force_pct in (10.0, 50.0, 100.0):
        with SimRobot(cfg.scene_path) as robot:
            # Use a 1 kg load-cell block mass so MuJoCo's mass-scaled contact
            # impedance stays stiff under up to ~62 N pinch load across versions.
            cube_id = robot.model.body("cube").id
            scale = 1.0 / float(robot.model.body_mass[cube_id])
            robot.model.body_mass[cube_id] = 1.0
            robot.model.body_inertia[cube_id] *= scale
            mujoco.mj_setConst(robot.model, robot.data)

            heights, q_home = tcp_task_heights(robot, pp)
            robot.data.qpos[: robot.n_arm] = q_home[: robot.n_arm]
            robot.data.ctrl[: robot.n_arm] = q_home[: robot.n_arm]
            mujoco.mj_forward(robot.model, robot.data)
            q_grasp, ok = robot.solve_tcp(
                np.array([pp.cube_xy[0], pp.cube_xy[1], heights["grasp_z"]]),
                GRIP_DOWN_QUAT,
                q_init=robot.data.qpos.copy(),
            )
            assert ok
            robot.data.qpos[: robot.n_arm] = q_grasp[: robot.n_arm]
            robot.data.ctrl[: robot.n_arm] = q_grasp[: robot.n_arm]
            mujoco.mj_forward(robot.model, robot.data)

            robot.set_gripper_command(0.0, speed_percent=50.0, force_percent=force_pct)
            robot.settle(1.0)
            out = robot.get_gripper_control_output()
            assert out is not None
            torque_limits.append(float(out.torque_limit_nm[0]))
            f_left, f_right = robot.get_gripper_contact_forces("cube")
            forces_measured.append(0.5 * (f_left + f_right))

    assert torque_limits[0] < torque_limits[1] < torque_limits[2]
    assert forces_measured[0] < forces_measured[1] < forces_measured[2]
    assert forces_measured[0] > 15.0
    assert forces_measured[2] > 55.0

