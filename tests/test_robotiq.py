"""Robotiq 2F-85, Robotiq 2F-140, and 11 mm robotiq_mounting adapter tests."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import dataclasses
import importlib
import io
from pathlib import Path
import struct
import sys
from unittest.mock import patch

import mujoco
import numpy as np
import pytest

from ufactory_mujoco import (
    MODELS_ROOT,
    RobotiqController,
    SimRobot,
    XArmRobot,
    compose_model,
    compose_spec,
    resolve_arm_scene,
    resolve_model_path,
)
from ufactory_mujoco.config import ConfigError
from ufactory_mujoco.grippers import (
    gripper_object_contact_forces_n,
    identify_gripper,
    measured_robotiq_pad_gap_mm,
)
from ufactory_mujoco.robotiq import (
    ROBOTIQ_2F_85_JOINT_NAMES,
    ROBOTIQ_2F_140_JOINT_NAMES,
    ROBOTIQ_ADAPTERS,
    ROBOTIQ_ARMATURE,
    ROBOTIQ_DAMPING,
    RobotiqModel,
    get_robotiq_spec,
    robotiq_gap_m_to_register,
    robotiq_gap_m_to_sim_drive,
    robotiq_percent_to_register,
    robotiq_register_to_gap_m,
    robotiq_sim_drive_to_gap_m,
    robotiq_speed_mm_s,
)
from ufactory_mujoco.robots import ConfigError as RobotsConfigError
from ufactory_mujoco.safety import authorize_real_motion

gripper_demo = importlib.import_module("examples.simulation.04_gripper_control")

ROOT = Path(__file__).resolve().parent.parent
CALIB_PATH = ROOT / "calibrations" / "xarm6_kinematics_L56A01.yaml"
SN = "XI130505L56A01"


def _stl_vertices(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    tri_count = struct.unpack_from("<I", raw, 80)[0]
    assert len(raw) == 84 + 50 * tri_count
    tri_dtype = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    return np.frombuffer(raw, dtype=tri_dtype, offset=84, count=tri_count)["v"].reshape(-1, 3).astype(np.float64)


class _StubXArmAPI:
    """Minimal connected xArm SDK stub for fail-closed XArmRobot Robotiq tests."""

    def __init__(self, *, dof: int = 6, sn: str = SN) -> None:
        self.connected = True
        self.axis = dof
        self.sn = sn
        self.state = 5
        self.mode = 0
        self.error_code = 0
        self.warn_code = 0
        self.tcp_offset = [0.0] * 6
        self.tcp_load = [0.0, [0.0, 0.0, 0.0]]
        self.ft_sensor_is_enable = False
        self.physical_q = np.zeros(dof, dtype=float)

    def get_robot_sn(self):
        return 0, self.sn

    def get_err_warn_code(self):
        return 0, (self.error_code, self.warn_code)

    def get_state(self):
        return 0, self.state

    def get_servo_angle(self, is_radian: bool = True):
        del is_radian
        return 0, self.physical_q.tolist()

    def set_state(self, state: int = 0):
        self.state = int(state)
        return 0

    def disconnect(self):
        self.connected = False
        return 0


def test_robotiq_mounting_inertials_sites_hulls_and_appearance():
    """Verify measured 0.118 kg inertial, 12 CAD sites, 2 STL hulls, and frosted aluminum appearance."""
    model_dir = MODELS_ROOT / "robotiq_mounting"
    col_dir = model_dir / "assets" / "robotiq_mounting" / "collision"

    for col_name, exp_tris, z_lo_mm, z_hi_mm in (
        ("robotiq_mounting_col_00.stl", 1260, 0.0, 13.9),
        ("robotiq_mounting_col_01.stl", 944, -5.6, 0.0),
    ):
        raw = (col_dir / col_name).read_bytes()
        tri_count = struct.unpack_from("<I", raw, 80)[0]
        assert tri_count == exp_tris
        verts_mm = _stl_vertices(col_dir / col_name) * 1000.0
        assert float(verts_mm[:, 2].min()) == pytest.approx(z_lo_mm, abs=0.05)
        assert float(verts_mm[:, 2].max()) == pytest.approx(z_hi_mm, abs=0.05)

    for scene in (False, True):
        m = mujoco.MjModel.from_xml_path(str(resolve_model_path("robotiq_mounting", scene=scene)))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        assert d.ncon == 0
        np.testing.assert_allclose(d.body("robotiq_mounting").xpos, [0.0, 0.0, 0.0056], atol=1e-12)

        body = m.body("robotiq_mounting")
        assert float(body.mass[0]) == pytest.approx(0.118, abs=1e-12)
        np.testing.assert_allclose(body.ipos, [0.0, -0.0007996449433, 0.004455899167], atol=1e-12)

        # Rotate diag inertia back into the body frame using body.iquat.
        r_mat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(r_mat, body.iquat)
        r = r_mat.reshape(3, 3)
        tensor = r @ np.diag(body.inertia) @ r.T
        expected = np.array([
            [4.427070717e-05, 0.0, 0.0],
            [0.0, 4.997953153e-05, 7.208106518e-08],
            [0.0, 7.208106518e-08, 9.018403314e-05],
        ])
        np.testing.assert_allclose(tensor, expected, atol=1e-11)

        expected_sites = {
            "robotiq_mounting_flange_mount": [0.0, 0.0, 0.0],
            "robotiq_mounting_tool_mount": [0.0, 0.0, 0.011],
            "robotiq_mounting_xarm_pin": [0.0, -0.025, 0.0],
            "robotiq_mounting_robotiq_pin": [0.0, -0.0305, 0.011],
            "robotiq_mounting_xarm_hole_0": [0.01767766953, 0.01767766953, 0.0],
            "robotiq_mounting_xarm_hole_1": [-0.01767766953, 0.01767766953, 0.0],
            "robotiq_mounting_xarm_hole_2": [-0.01767766953, -0.01767766953, 0.0],
            "robotiq_mounting_xarm_hole_3": [0.01767766953, -0.01767766953, 0.0],
            "robotiq_mounting_robotiq_hole_0": [0.016, 0.027, 0.011],
            "robotiq_mounting_robotiq_hole_1": [-0.016, 0.027, 0.011],
            "robotiq_mounting_robotiq_hole_2": [-0.016, -0.027, 0.011],
            "robotiq_mounting_robotiq_hole_3": [0.016, -0.027, 0.011],
        }
        for s_name, s_pos in expected_sites.items():
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s_name)
            assert sid >= 0, f"Missing site {s_name}"
            assert int(m.site_group[sid]) == 4
            np.testing.assert_allclose(m.site_pos[sid], s_pos, atol=1e-9)

        alu_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "robotiq_mounting_aluminum")
        assert alu_id >= 0
        np.testing.assert_allclose(m.mat_rgba[alu_id], [0.60, 0.62, 0.64, 1.0], atol=1e-6)
        assert float(m.mat_metallic[alu_id]) == pytest.approx(0.8, abs=1e-6)
        assert float(m.mat_roughness[alu_id]) == pytest.approx(0.64, abs=1e-6)
        if scene:
            np.testing.assert_allclose(m.vis.headlight.specular, [0.35, 0.35, 0.38], atol=1e-6)


@pytest.mark.parametrize(
    "variant,base_z,tcp_z,expected_mass,joint_names",
    [
        ("robotiq_2f_85", 0.22, 0.162253, 0.92086427, ROBOTIQ_2F_85_JOINT_NAMES),
        ("robotiq_2f_140", 0.29, 0.234466, 1.000000, ROBOTIQ_2F_140_JOINT_NAMES),
    ],
)
def test_robotiq_grippers_standalone_specs_and_geometry(variant, base_z, tcp_z, expected_mass, joint_names):
    """Verify standalone pose, masses, linkage/pad match, stiff equalities, and SDK registers."""
    spec_info = get_robotiq_spec(variant)
    assert len(joint_names) == 6
    col_dir = MODELS_ROOT / variant / "assets" / variant / "collision"
    assert len(list(col_dir.glob("*.stl"))) == 11

    for scene in (False, True):
        m = mujoco.MjModel.from_xml_path(str(resolve_model_path(variant, scene=scene)))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        assert d.ncon == 0
        base = d.body(f"{variant}_base_link")
        np.testing.assert_allclose(base.xpos, [0.0, 0.0, base_z], atol=1e-12)
        np.testing.assert_allclose(base.xmat.reshape(3, 3), np.diag([1.0, -1.0, -1.0]), atol=1e-12)
        np.testing.assert_allclose(d.site("link_tcp").xpos, [0.0, 0.0, base_z - tcp_z], atol=1e-12)
        if scene:
            np.testing.assert_allclose(m.vis.headlight.specular, [0.35, 0.35, 0.38], atol=1e-6)

        total_mass = float(m.body_mass[1:].sum())
        assert total_mass == pytest.approx(expected_mass, abs=1e-6)
        assert m.neq == 5
        for eq_i in range(m.neq):
            np.testing.assert_allclose(m.eq_solref[eq_i], [0.004, 1.0], atol=1e-9)
            np.testing.assert_allclose(m.eq_solimp[eq_i][:2], [0.9999, 0.9999], atol=1e-9)

        for jname in joint_names:
            j_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            assert j_id >= 0
            assert float(m.dof_armature[int(m.jnt_dofadr[j_id])]) == pytest.approx(ROBOTIQ_ARMATURE, abs=1e-9)
            assert float(m.dof_damping[int(m.jnt_dofadr[j_id])]) == pytest.approx(ROBOTIQ_DAMPING, abs=1e-9)

        for side in ("left", "right"):
            pad_gid = mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_MESH, f"{variant}_{side}_finger_pad"
            )
            assert pad_gid >= 0
            geom_ids = [i for i in range(m.ngeom) if m.geom_dataid[i] == pad_gid]
            assert len(geom_ids) == 1
            assert int(m.geom_priority[geom_ids[0]]) == 1

    # Verify pad STL surface separation agrees with linkage pad_gap_m across the full stroke.
    m = mujoco.MjModel.from_xml_path(str(resolve_model_path(variant, scene=False)))
    d = mujoco.MjData(m)
    pad_verts = {
        side: _stl_vertices(col_dir / f"{variant}_{side}_finger_pad.stl")
        for side in ("left", "right")
    }
    linkage = spec_info.linkage
    for gap_mm in (0.0, 20.0, spec_info.open_gap_mm * 0.5, spec_info.open_gap_mm):
        q = linkage.drive_for_gap_m(gap_mm / 1000.0)
        # Robotiq 2F-85 URDF 8-digit joint origins yield 84.999989 mm at q=0 (11 nm below 85.0).
        assert linkage.pad_gap_m(q) * 1000.0 == pytest.approx(gap_mm, abs=2e-5)
        assert robotiq_gap_m_to_sim_drive(variant, gap_mm / 1000.0) == pytest.approx(q, abs=1e-12)
        assert robotiq_sim_drive_to_gap_m(variant, q) * 1000.0 == pytest.approx(gap_mm, abs=2e-5)
        step = 1e-6
        num_jac = (linkage.pad_gap_m(q - step) - linkage.pad_gap_m(q + step)) / (2.0 * step)
        assert linkage.gap_jacobian_m_per_rad(q) == pytest.approx(num_jac, rel=1e-7)

        for jname in joint_names:
            d.joint(jname).qpos[0] = q
        mujoco.mj_forward(m, d)
        assert measured_robotiq_pad_gap_mm(m, d, variant) == pytest.approx(gap_mm, abs=0.01)

        base = d.body(f"{variant}_base_link")
        b_pos, b_mat = base.xpos, base.xmat.reshape(3, 3)
        pad_y = {}
        for side in ("left", "right"):
            fb = d.body(f"{variant}_{side}_finger")
            world = pad_verts[side] @ fb.xmat.reshape(3, 3).T + fb.xpos
            pad_y[side] = (world - b_pos) @ b_mat
        stl_gap_mm = float((pad_y["left"][:, 1].min() - pad_y["right"][:, 1].max()) * 1000.0)
        assert stl_gap_mm == pytest.approx(gap_mm, abs=0.01)

    assert robotiq_gap_m_to_register(variant, spec_info.open_gap_m) == 0
    assert robotiq_gap_m_to_register(variant, 0.0) == 255
    assert robotiq_register_to_gap_m(variant, 0) == pytest.approx(spec_info.open_gap_m, abs=1e-9)
    assert robotiq_register_to_gap_m(variant, 255) == pytest.approx(0.0, abs=1e-9)
    assert robotiq_percent_to_register(0.0) == 0
    assert robotiq_percent_to_register(100.0) == 255


@pytest.mark.parametrize("variant", ["robotiq_2f_85", "robotiq_2f_140"])
def test_robotiq_controller_unit(variant):
    """Verify RobotiqModel, batched float64/float32 RobotiqController, pause preload, reversal, and validation."""
    rmodel = RobotiqModel.load(variant)
    spec = rmodel.spec
    assert robotiq_speed_mm_s(spec, 0.0) == 0.0
    assert robotiq_speed_mm_s(spec, 100.0) == pytest.approx(spec.speed_max_mm_s)

    low_lookup = rmodel.curve.lookup(spec.speed_min_mm_s, 1.0)
    high_lookup = rmodel.curve.lookup(spec.speed_max_mm_s, 100.0)
    assert low_lookup.force_n == pytest.approx(spec.force_min_n)
    assert high_lookup.force_n == pytest.approx(spec.force_max_n)
    assert rmodel.torque_limit_nm(high_lookup) == pytest.approx(spec.force_max_n * rmodel.reference_jacobian_m_rad)
    assert rmodel.motor_torque_estimate_nm(rmodel.torque_limit_nm(high_lookup)) == pytest.approx(1.0)

    for dtype, atol in ((np.float64, 1e-10), (np.float32, 2e-4)):
        batch = RobotiqController(rmodel, num_envs=4, dtype=dtype)
        single = [RobotiqController(rmodel, dtype=dtype) for _ in range(4)]
        mid_mm = spec.open_gap_mm * 0.5
        q = np.full(4, rmodel.linkage.drive_for_gap_m(mid_mm / 1000.0), dtype=dtype)
        vel = np.array([0.0, 0.5, -0.3, 0.01], dtype=dtype)

        for pos, spd, frc in (
            ([0.0, 10.0, 20.0, 30.0], [10.0, 50.0, 90.0, 100.0], [10.0, 30.0, 50.0, 100.0]),
            ([0.0, 10.0, 20.0, 30.0], [0.0, 0.0, 0.0, 0.0], [10.0, 30.0, 80.0, 100.0]),
            ([spec.open_gap_mm] * 2 + [0.0, 0.0], [0.0, 50.0, 90.0, 100.0], [80.0, 30.0, 50.0, 10.0]),
        ):
            batch.command(pos, spd, frc)
            for i, c in enumerate(single):
                c.command(pos[i], spd[i], frc[i])
            for _ in range(8):
                actual = batch.update(q, vel, 0.02)
                expected = [c.update(q[i : i + 1], vel[i : i + 1], 0.02) for i, c in enumerate(single)]
                for f in dataclasses.fields(actual):
                    np.testing.assert_allclose(
                        getattr(actual, f.name),
                        np.concatenate([getattr(out, f.name) for out in expected]),
                        rtol=0,
                        atol=atol,
                    )

    # Pause holds saturated preload & adapts to lowered force limit, then resumes cleanly.
    c = RobotiqController(rmodel, initial_position_mm=spec.open_gap_mm * 0.5)
    q_mid = rmodel.linkage.drive_for_gap_m(spec.open_gap_mm * 0.5 / 1000.0)
    c.command(0.0, 80.0, 80.0)
    for _ in range(15):
        c.update(q_mid, 0.0, 0.02)
    c.command(0.0, 0.0, 80.0)
    paused = c.update(q_mid, 0.0, 0.02)
    assert float(paused.velocity_rad_s[0]) == 0.0
    assert float(paused.integral_torque_nm[0]) == pytest.approx(float(paused.torque_limit_nm[0]))

    c.command(0.0, 0.0, 20.0)
    lowered = c.update(q_mid, 0.0, 0.02)
    assert float(lowered.torque_limit_nm[0]) < float(paused.torque_limit_nm[0])
    assert float(lowered.integral_torque_nm[0]) == pytest.approx(float(lowered.torque_limit_nm[0]))

    c.command(0.0, 80.0, 20.0)
    resumed = c.update(q_mid, 0.0, 0.02)
    assert float(resumed.velocity_rad_s[0]) > 0.0

    # Prime reset and atomic validation on invalid inputs.
    batch = RobotiqController(rmodel, num_envs=2)
    batch.reset([1], position_mm=40.0, target_mm=10.0, prime=True)
    assert float(batch.integral_torque_nm[1]) == pytest.approx(float(batch.torque_limit_nm[1]))
    before = batch.position_mm.copy()
    with pytest.raises(ValueError):
        batch.command([10.0, spec.open_gap_mm + 1.0], 50.0, 50.0)
    np.testing.assert_array_equal(before, batch.position_mm)


def test_robotiq_mounted_arm_assemblies_and_guards():
    """Verify xArm5/6/7 and UF850 Robotiq assemblies, init keyframe rules, lite6/ft_sensor guards, and XArmRobot."""
    flange_links = {"xarm5": "link5", "xarm6": "link6", "xarm7": "link7", "uf850": "link6"}
    for arm, flange_link in flange_links.items():
        for variant, tcp_z in (("robotiq_2f_85", 0.162253), ("robotiq_2f_140", 0.234466)):
            combo = f"{arm}_with_{variant}"
            spec = compose_spec(combo, scene=True)
            m = spec.compile()
            d = mujoco.MjData(m)
            key_name = "init" if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "init") >= 0 else "home"
            if arm == "uf850":
                assert (key_name == "init") == (variant == "robotiq_2f_140")
            else:
                assert key_name == "init"

            mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key_name))
            mujoco.mj_forward(m, d)
            assert d.ncon == 0

            # Auto-inserted 11 mm robotiq_mounting adapter preserves its 12 CAD sites.
            flange = d.body(flange_link)
            flange_pos, flange_rot = flange.xpos, flange.xmat.reshape(3, 3)
            adapter = d.body("robotiq_mounting")
            tool = d.body(f"{variant}_base_link")
            np.testing.assert_allclose(flange_rot.T @ (adapter.xpos - flange_pos), [0.0, 0.0, 0.0], atol=1e-12)
            np.testing.assert_allclose(flange_rot.T @ (tool.xpos - flange_pos), [0.0, 0.0, 0.011], atol=1e-12)
            np.testing.assert_allclose(
                flange_rot.T @ (d.site("attachment_site").xpos - flange_pos),
                [0.0, 0.0, 0.011 + tcp_z],
                atol=1e-12,
            )
            assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robotiq_mounting_tool_mount") >= 0
            assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robotiq_mounting_robotiq_hole_3") >= 0

            # Sweep the full gripper stroke at ready without self-collision.
            spec_info = get_robotiq_spec(variant)
            ready_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "ready")
            mujoco.mj_resetDataKeyframe(m, d, ready_id)
            for frac in (0.0, 0.5, 1.0):
                q = spec_info.linkage.drive_for_gap_m(frac * spec_info.open_gap_m)
                for jname in spec_info.joint_names:
                    d.joint(jname).qpos[0] = q
                mujoco.mj_forward(m, d)
                assert d.ncon == 0

    # Unsupported compositions and no Hand-E support.
    for variant in ROBOTIQ_ADAPTERS:
        with pytest.raises((ConfigError, RobotsConfigError)):
            resolve_arm_scene("lite6", variant, "scene")
        with pytest.raises(ValueError):
            compose_model(arm="lite6", gripper=variant)
        with pytest.raises(ValueError):
            compose_model(arm="xarm6", ft_sensor=True, gripper=variant)
    with pytest.raises(ValueError):
        compose_model(arm="xarm6", gripper="robotiq_hand_e")
    with pytest.raises(FileNotFoundError):
        resolve_model_path("robotiq_hand_e")

    # XArmRobot fails closed on Robotiq grippers in both read-only and unlocked states.
    for variant in ("robotiq_2f_85", "robotiq_2f_140"):
        stub = _StubXArmAPI()
        robot = XArmRobot(robot="xarm6", gripper=variant, read_only=True, arm_api=stub)
        try:
            assert robot.has_gripper
            assert robot.gripper_kind == "jaw"
            with pytest.raises(NotImplementedError):
                robot.get_gripper_position()
            with pytest.raises(NotImplementedError):
                robot.get_gripper_gap()
            if CALIB_PATH.is_file():
                robot_rw = XArmRobot(
                    robot="xarm6",
                    gripper=variant,
                    calibration=CALIB_PATH,
                    read_only=True,
                    arm_api=_StubXArmAPI(),
                )
                try:
                    approval = authorize_real_motion(
                        robot_rw._arm,
                        robot="xarm6",
                        calibration=CALIB_PATH,
                        preflight_ok=True,
                        confirm_real=True,
                    )
                    robot_rw.unlock_motion(approval)
                    with pytest.raises(NotImplementedError):
                        robot_rw.set_gripper(0.5)
                    with pytest.raises(NotImplementedError):
                        robot_rw.set_gripper_gap(0.04)
                finally:
                    robot_rw.close()
        finally:
            robot.close()


def test_robotiq_camera_stand_and_four_cameras():
    """Verify xarm6 + camera_stand + all 4 depth cameras + Robotiq 2F-85 / 2F-140 stacking and clearances."""
    for variant, tcp_z in (("robotiq_2f_85", 0.162253), ("robotiq_2f_140", 0.234466)):
        for cam in ("d405", "d435", "gemini_305", "zed_x_nano"):
            m = compose_model(arm="xarm6", gripper=variant, camera=cam, scene=True)
            d = mujoco.MjData(m)
            ready_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "ready")
            mujoco.mj_resetDataKeyframe(m, d, ready_id)
            mujoco.mj_forward(m, d)
            assert d.ncon == 0

            flange = d.body("link6")
            flange_pos, flange_rot = flange.xpos, flange.xmat.reshape(3, 3)
            adapter = d.body("robotiq_mounting")
            tool = d.body(f"{variant}_base_link")
            # 3 mm camera_stand + 11 mm robotiq_mounting = 14 mm tool lift.
            np.testing.assert_allclose(flange_rot.T @ (adapter.xpos - flange_pos), [0.0, 0.0, 0.003], atol=1e-12)
            np.testing.assert_allclose(flange_rot.T @ (tool.xpos - flange_pos), [0.0, 0.0, 0.014], atol=1e-12)
            np.testing.assert_allclose(
                flange_rot.T @ (d.site("attachment_site").xpos - flange_pos),
                [0.0, 0.0, 0.014 + tcp_z],
                atol=1e-12,
            )

            cam_geoms = [
                i for i in range(m.ngeom) if m.geom_group[i] == 3 and m.geom_bodyid[i] == m.body(f"{cam}_body").id
            ]
            rq_geoms = [
                i
                for i in range(m.ngeom)
                if m.geom_group[i] == 3
                and "robotiq" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[i])) or "")
            ]
            min_clear_mm = min(
                mujoco.mj_geomDistance(m, d, a, b, 0.5, None) for a in cam_geoms for b in rq_geoms
            ) * 1000.0
            assert min_clear_mm > 8.0


@pytest.mark.parametrize(
    "name,variant,loadcell_z",
    [
        ("robotiq_2f_85", "robotiq_2f_85", 0.145),
        ("xarm6_with_robotiq_2f_85", "robotiq_2f_85", 0.145),
        ("robotiq_2f_140", "robotiq_2f_140", 0.215),
        ("xarm6_with_robotiq_2f_140", "robotiq_2f_140", 0.215),
    ],
)
def test_robotiq_closed_loop_open_close_and_loadcell_force(name, variant, loadcell_z):
    """Verify full-stroke closed-loop tracking and monotonic force-limited loadcell grasps on SimRobot."""
    open_gap_m = get_robotiq_spec(variant).open_gap_m
    with SimRobot(compose_spec(name, scene=False)) as robot:
        assert robot.robotiq_controller is not None
        robot.set_gripper_gap(0.0, speed_percent=100.0, force_percent=80.0)
        robot.settle(robot.gripper_wait_duration() + 0.3)
        assert robot.get_gripper_gap(measured=True) < 0.003

        robot.set_gripper_gap(open_gap_m, speed_percent=100.0, force_percent=80.0)
        robot.settle(robot.gripper_wait_duration() + 0.3)
        assert robot.get_gripper_gap(measured=True) > open_gap_m - 0.003

        robot.set_gripper_gap(open_gap_m * 0.5, speed_percent=80.0, force_percent=80.0)
        robot.settle(robot.gripper_wait_duration() + 0.3)
        assert robot.get_gripper_gap(measured=True) == pytest.approx(open_gap_m * 0.5, abs=0.003)

    spec = compose_spec(name, scene=False)
    loadcell = spec.body(f"{variant}_base_link").add_body(name="loadcell", pos=[0, 0, loadcell_z])
    loadcell.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.015, 0.020, 0.015], friction=[1.0, 1.0, 0.005])
    forces, limits = [], []
    with SimRobot(spec) as robot:
        gripper_spec = identify_gripper(robot.model)
        assert gripper_spec is not None and gripper_spec.name == variant
        for percent in (10.0, 50.0, 100.0):
            robot.set_gripper_gap(open_gap_m, speed_percent=100.0, force_percent=percent)
            robot.settle(robot.gripper_wait_duration() + 0.2)
            robot.set_gripper_gap(0.0, speed_percent=80.0, force_percent=percent)
            robot.settle(robot.gripper_wait_duration() + 0.3)

            assert robot.bilateral_finger_contact("loadcell")
            assert robot.get_gripper_gap(measured=True) == pytest.approx(0.040, abs=0.0015)
            left, right = gripper_object_contact_forces_n(robot.model, robot.data, gripper_spec, "loadcell")
            assert min(left, right) > 5.0
            forces.append((left + right) / 2.0)
            out = robot.gripper_control_output
            assert out is not None
            limits.append(float(out.torque_limit_nm[0]))

    assert forces[0] < forces[1] < forces[2]
    assert limits[0] < limits[1] < limits[2]


def test_robotiq_04_example_headless_smoke():
    """Run example 04 headless on xarm6_with_robotiq_2f_85 / robotiq_2f_140 and reject robotiq_mounting."""
    for model_name in ("xarm6_with_robotiq_2f_85", "robotiq_2f_140"):
        argv = [
            "04_gripper_control.py",
            "--model",
            model_name,
            "--headless",
            "--cycles",
            "1",
            "--speed-percent",
            "100",
            "--force-percent",
            "80",
        ]
        buf = io.StringIO()
        with patch.object(sys, "argv", argv), redirect_stdout(buf):
            gripper_demo.main()
        assert "Gripper demonstration PASS" in buf.getvalue()

    with patch.object(sys, "argv", ["04_gripper_control.py", "--model", "robotiq_mounting", "--headless"]):
        with redirect_stderr(io.StringIO()), pytest.raises(SystemExit) as caught:
            gripper_demo.main()
        assert caught.value.code == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
