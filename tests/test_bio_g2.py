"""Bio Gripper G2 URDF model regression and closed-loop controller physics tests."""

from __future__ import annotations

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from ufactory_mujoco import MODELS_ROOT, compose_model, resolve_model_path
import copy
from ufactory_mujoco.bio_g2 import (
    BioGripperG2Controller,
    MuJoCoBioG2Adapter,
)
from ufactory_mujoco.bio_g2 import (
    BASE_INERTIA_SCALE,
    DEFAULT_SPEED_MM_S,
    GLB_FINGER_VISUAL_Y_OFFSET_M,
    LEFT_JOINT as LEFT,
    MAX_SPEED_MM_S,
    MIN_SPEED_MM_S,
    RIGHT_JOINT as RIGHT,
    SPEEDS_MM_S,
    bio_g2_speed_mm_s_from_percent,
    bio_g2_speed_percent_from_mm_s,
    drive_limit_n,
    finger_force_n,
    gap_to_q,
    q_to_gap,
)
from ufactory_mujoco.grippers import GripperSpec, gripper_object_contact_forces_n
from ufactory_mujoco.sim_robot import SimRobot

# ==========================================================================
# From test_bio_model.py
# ==========================================================================

ROOT = Path(__file__).resolve().parent.parent


def _bio_link_masses(path: Path) -> dict[str, float]:
    """Masses of the ``bio_gripper_g2_*`` links in an MJCF or URDF file."""
    root = ET.parse(path).getroot()
    if root.tag == "mujoco":
        return {body.get("name"): float(body.find("inertial").get("mass")) for body in root.iter("body")
                if (body.get("name") or "").startswith("bio_gripper_g2_")
                and body.find("inertial") is not None and "mass" in body.find("inertial").attrib}
    return {link.get("name"): float(link.find("inertial/mass").get("value", "0")) for link in root.findall("link")
            if (link.get("name") or "").startswith("bio_gripper_g2_") and link.find("inertial/mass") is not None}

REFERENCE = ROOT / "tests/fixtures/bio_gripper_g2_movable_visual.urdf"

MODEL_NAMES = (
    "bio_gripper_g2",
    "xarm5_with_bio_gripper",
    "xarm6_with_bio_gripper",
    "xarm7_with_bio_gripper",
    "uf850_with_bio_gripper",
)

class BioModelTest(unittest.TestCase):
    def test_mass_com_and_compiled_full_inertia_match_reference(self):
        reference = ET.parse(REFERENCE).getroot()
        for name in MODEL_NAMES:
            model = mujoco.MjModel.from_xml_path(str(resolve_model_path(name, scene=False)))
            total_bio_mass = 0.0
            for link in reference.findall("link"):
                inertial = link.find("inertial")
                if inertial is None:
                    continue
                with self.subTest(model=name, body=link.get("name")):
                    body = model.body(link.get("name"))
                    expected_mass = float(inertial.find("mass").get("value"))
                    total_bio_mass += expected_mass
                    self.assertAlmostEqual(body.mass[0], expected_mass, places=12)
                    com = np.fromstring(inertial.find("origin").get("xyz"), sep=" ")
                    np.testing.assert_allclose(body.ipos, com, rtol=0, atol=1e-12)
                    inertia = inertial.find("inertia")
                    xx, yy, zz, xy, xz, yz = [
                        float(inertia.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
                    ]
                    expected = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
                    rotation = np.empty(9)
                    mujoco.mju_quat2Mat(rotation, body.iquat)
                    rotation = rotation.reshape(3, 3)
                    compiled = rotation @ np.diag(body.inertia) @ rotation.T
                    np.testing.assert_allclose(compiled, expected, rtol=0, atol=1e-11)
            self.assertAlmostEqual(total_bio_mass, 0.79, places=12)

    def test_tcp_visual_and_collision_poses_match_reference(self):
        reference = ET.parse(REFERENCE).getroot()
        tcp = reference.find("joint[@name='bio_gripper_g2_tcp_joint']/origin")
        for name in MODEL_NAMES:
            path = resolve_model_path(name, scene=False)
            xml = ET.parse(path).getroot()
            model = mujoco.MjModel.from_xml_path(str(path))
            site_name = "link_tcp" if name == "bio_gripper_g2" else "attachment_site"
            np.testing.assert_allclose(
                model.site(site_name).pos,
                np.fromstring(tcp.get("xyz"), sep=" "),
                rtol=0,
                atol=1e-12,
            )
            self.assertEqual(
                model.body(int(model.site(site_name).bodyid[0])).name,
                "bio_gripper_g2_base_link",
            )
            for link in reference.findall("link"):
                link_name = link.get("name") or ""
                collisions = link.findall("collision")
                if not collisions:
                    continue
                with self.subTest(model=name, body=link_name):
                    element = xml.find(f".//body[@name='{link_name}']")
                    col_geoms = [g for g in element.findall("geom") if "collision" in g.get("class", "")]
                    self.assertEqual(len(col_geoms), len(collisions))
                    for col_node, geom in zip(collisions, col_geoms):
                        expected = np.fromstring(col_node.find("origin").get("xyz", "0 0 0"), sep=" ")
                        np.testing.assert_allclose(
                            np.fromstring(geom.get("pos", "0 0 0"), sep=" "),
                            expected,
                            rtol=0,
                            atol=1e-12,
                        )
                    body_id = model.body(link_name).id
                    geom_ids = np.flatnonzero((model.geom_bodyid == body_id) & (model.geom_group == 3))
                    self.assertEqual(len(geom_ids), len(collisions))
                    for gid in geom_ids:
                        mesh_id = model.geom_dataid[gid]
                        np.testing.assert_allclose(
                            model.geom_pos[gid] - model.mesh_pos[mesh_id],
                            [0.0, 0.0, 0.0],
                            rtol=0,
                            atol=1e-12,
                        )
                    vis_origin = link.find("visual/origin")
                    if vis_origin is not None:
                        expected_vis = np.fromstring(vis_origin.get("xyz", "0 0 0"), sep=" ")
                        vis_geoms = [g for g in element.findall("geom") if "visual" in g.get("class", "")]
                        for vg in vis_geoms:
                            np.testing.assert_allclose(
                                np.fromstring(vg.get("pos", "0 0 0"), sep=" "),
                                expected_vis,
                                rtol=0,
                                atol=1e-12,
                            )

    def test_standalone_gripper_closes_and_opens(self):
        for filename in ("bio_gripper_g2.xml", "scene.xml"):
            with self.subTest(file=filename):
                path = MODELS_ROOT / "bio_gripper_g2" / filename
                model = mujoco.MjModel.from_xml_path(str(path))
                data = mujoco.MjData(model)
                for target in (0.0395, 0.0, 0.01975, 0.0395, 0.0):
                    data.ctrl[0] = target
                    for _ in range(1000):
                        mujoco.mj_step(model, data)
                    np.testing.assert_allclose(data.qpos, [-target, target], rtol=0, atol=5e-5)
                    self.assertTrue(all(contact.dist >= -1e-6 for contact in data.contact))

    def test_standalone_orientation_flange_up_tcp_down(self):
        """Standalone bio_gripper_g2 flips flange up (+Z) and TCP down without changing mounted combos."""
        for filename in ("bio_gripper_g2.xml", "scene.xml"):
            with self.subTest(file=filename):
                m = mujoco.MjModel.from_xml_path(str(MODELS_ROOT / "bio_gripper_g2" / filename))
                d = mujoco.MjData(m)
                mujoco.mj_forward(m, d)
                base_pos = d.body("bio_gripper_g2_base_link").xpos
                base_mat = d.body("bio_gripper_g2_base_link").xmat.reshape(3, 3)
                tcp_pos = d.site("link_tcp").xpos
                np.testing.assert_allclose(base_pos, [0.0, 0.0, 0.17], atol=1e-12)
                np.testing.assert_allclose(base_mat, np.diag([1.0, -1.0, -1.0]), atol=1e-12)
                np.testing.assert_allclose(tcp_pos, [0.135, 0.0, 0.115], atol=1e-12)
                self.assertLess(tcp_pos[2], base_pos[2])
                self.assertGreater(tcp_pos[2], 0.10)
                self.assertEqual(d.ncon, 0)

        flange_links = {
            "xarm5": "link5",
            "xarm6": "link6",
            "xarm7": "link7",
            "uf850": "link6",
        }
        for arm, flange_link in flange_links.items():
            name = f"{arm}_with_bio_gripper"
            with self.subTest(model=name):
                m = compose_model(name, scene=True)
                d = mujoco.MjData(m)
                mujoco.mj_forward(m, d)
                flange_pos = d.body(flange_link).xpos
                flange_rot = d.body(flange_link).xmat.reshape(3, 3)
                tcp_in_flange = flange_rot.T @ (d.site("attachment_site").xpos - flange_pos)
                np.testing.assert_allclose(tcp_in_flange * 1000.0, [157.64, -0.03, 57.52], atol=0.1)

    def test_anodized_aluminum_material_and_scene_specular(self):
        """Bio G2 fingers and metallic base use frosted anodized aluminum with soft scene specular."""
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                m = mujoco.MjModel.from_xml_path(str(resolve_model_path(name, scene=True)))
                mat_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "bio_g2_anodized_aluminum")
                self.assertGreaterEqual(mat_id, 0)
                np.testing.assert_allclose(m.mat_rgba[mat_id], [0.60, 0.62, 0.64, 1.0], atol=1e-6)
                self.assertAlmostEqual(float(m.mat_specular[mat_id]), 0.72, places=6)
                self.assertAlmostEqual(float(m.mat_shininess[mat_id]), 0.16, places=6)
                self.assertAlmostEqual(float(m.mat_reflectance[mat_id]), 0.03, places=6)
                self.assertAlmostEqual(float(m.mat_metallic[mat_id]), 0.8, places=6)
                self.assertAlmostEqual(float(m.mat_roughness[mat_id]), 0.64, places=6)

                for body_name in ("bio_gripper_g2_left_finger", "bio_gripper_g2_right_finger"):
                    body_id = m.body(body_name).id
                    vis_geoms = np.flatnonzero((m.geom_bodyid == body_id) & (m.geom_group == 2))
                    self.assertGreater(len(vis_geoms), 0)
                    for gid in vis_geoms:
                        self.assertEqual(int(m.geom_matid[gid]), mat_id)

                base_id = m.body("bio_gripper_g2_base_link").id
                base_vis = np.flatnonzero((m.geom_bodyid == base_id) & (m.geom_group == 2))
                base_mats = {
                    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[gid])): int(m.geom_matid[gid])
                    for gid in base_vis
                }
                white_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "white")
                self.assertEqual(base_mats["base_silver_vis"], mat_id)
                self.assertEqual(base_mats["base_white_vis"], white_id)
                np.testing.assert_allclose(m.vis.headlight.specular, [0.35, 0.35, 0.38], atol=1e-6)

# ==========================================================================
# From test_bio_g2_controller.py
# ==========================================================================

ROOT = Path(__file__).resolve().parent.parent

URDF_FIXTURE = ROOT / "tests/fixtures/bio_gripper_g2_movable_visual.urdf"

ASSETS_DIR = MODELS_ROOT / "bio_gripper_g2/assets/bio_gripper_g2"

class FakeAdapter:
    def __init__(self):
        self.q = gap_to_q(150.0)
        self.v = 0.0
        self.resets = 0
        self.calls: list[tuple] = []

    def state(self) -> tuple[float, float, float, float]:
        return self.q, self.v, -self.q, -self.v

    def set_force_limit(self, limit: float) -> None:
        self.calls.append(("limit", limit))

    def servo(self, q: float, v: float) -> None:
        self.calls.append(("servo", q, v))

    def initialize(self, q: float) -> None:
        self.q, self.v = q, 0.0
        self.resets += 1

class BioG2ControllerTest(unittest.TestCase):
    def test_units_and_virtual_work(self):
        for gap in (71, 80, 110, 140, 150):
            with self.subTest(gap=gap):
                q = gap_to_q(gap)
                self.assertAlmostEqual(q_to_gap(q, -q), gap, places=10)
                self.assertAlmostEqual(q_to_gap(q), gap, places=10)
        for force in (1, 25, 50, 75, 100):
            dq = 0.001
            self.assertAlmostEqual(
                drive_limit_n(force) * dq,
                finger_force_n(force) * 2.0 * dq,
                places=12,
            )
        self.assertEqual(SPEEDS_MM_S, (500 / 12, 1000 / 12, 2000 / 12, 3000 / 12, 4000 / 12))

    def test_speed_percent_and_mm_s_conversions(self):
        self.assertEqual(bio_g2_speed_mm_s_from_percent(0), 0.0)
        self.assertAlmostEqual(bio_g2_speed_mm_s_from_percent(10), MIN_SPEED_MM_S, places=10)
        self.assertAlmostEqual(bio_g2_speed_mm_s_from_percent(50), DEFAULT_SPEED_MM_S, places=10)
        self.assertAlmostEqual(bio_g2_speed_mm_s_from_percent(100), MAX_SPEED_MM_S, places=10)
        self.assertEqual(bio_g2_speed_percent_from_mm_s(0.0), 0.0)
        self.assertAlmostEqual(bio_g2_speed_percent_from_mm_s(DEFAULT_SPEED_MM_S), 50.0, places=10)
        self.assertAlmostEqual(bio_g2_speed_percent_from_mm_s(MAX_SPEED_MM_S), 100.0, places=10)

    def test_mass_inertia_and_mimic_across_mjcf_and_urdf(self):
        mjcf_masses = _bio_link_masses(MODELS_ROOT / "bio_gripper_g2" / "bio_gripper_g2.xml")
        urdf_masses = _bio_link_masses(URDF_FIXTURE)
        self.assertAlmostEqual(sum(mjcf_masses.values()), 0.79, places=12)
        self.assertAlmostEqual(sum(urdf_masses.values()), 0.79, places=12)
        self.assertEqual(mjcf_masses, urdf_masses)
        self.assertAlmostEqual(BASE_INERTIA_SCALE, 1.9719005072482427, places=12)

        robot = ET.parse(URDF_FIXTURE).getroot()
        for node in robot.findall("link/inertial/inertia"):
            a = {k: float(v) for k, v in node.attrib.items()}
            matrix = [
                [a["ixx"], a["ixy"], a["ixz"]],
                [a["ixy"], a["iyy"], a["iyz"]],
                [a["ixz"], a["iyz"], a["izz"]],
            ]
            self.assertGreater(float(np.linalg.eigvalsh(matrix).min()), 0.0)
        mimic = robot.find(f"joint[@name='{LEFT}']/mimic")
        self.assertIsNotNone(mimic)
        self.assertEqual(mimic.get("joint"), RIGHT)
        self.assertEqual(float(mimic.get("multiplier")), -1.0)

    def test_actual_visual_and_collision_surfaces(self):
        try:
            import trimesh
        except ImportError:
            self.skipTest("trimesh not installed in this environment")

        for gap in (71, 80, 110, 140, 150):
            with self.subTest(gap=gap):
                bounds = {}
                for side in ("left", "right"):
                    name = f"bio_gripper_g2_{side}_finger"
                    q = gap_to_q(gap) * (-1.0 if side == "left" else 1.0)
                    visual = trimesh.load(ASSETS_DIR / f"visual/{name}.obj", force="mesh")
                    visual.apply_translation([0.059, GLB_FINGER_VISUAL_Y_OFFSET_M[name] + q, 0.027])
                    collision = trimesh.load(ASSETS_DIR / f"collision/{name}_convex_0.stl", force="mesh")
                    collision.apply_translation([0.059, q, 0.027])
                    for surface, mesh in (("visual", visual), ("collision", collision)):
                        tip = mesh.vertices[mesh.vertices[:, 0] > 0.09]
                        bounds[(side, surface)] = float(
                            tip[:, 1].max() if side == "left" else tip[:, 1].min()
                        )
                    self.assertAlmostEqual(bounds[(side, "visual")], bounds[(side, "collision")], delta=5e-5)
                    for idx in range(5):
                        hull = trimesh.load(ASSETS_DIR / f"collision/{name}_convex_{idx}.stl", force="mesh")
                        self.assertTrue(hull.is_convex and hull.is_watertight, f"{name}_convex_{idx}")
                base_hull = trimesh.load(ASSETS_DIR / "collision/bio_gripper_g2_base_link_convex_0.stl", force="mesh")
                self.assertTrue(base_hull.is_convex and base_hull.is_watertight)
                self.assertAlmostEqual(
                    (bounds[("right", "collision")] - bounds[("left", "collision")]) * 1000.0,
                    gap,
                    delta=1e-4,
                )

    def test_invalid_commands_and_dt_are_atomic(self):
        for field in ("position_mm", "speed_mm_s", "force_percent"):
            for value in (float("nan"), float("inf"), -float("inf"), -1, 0, 10000):
                with self.subTest(field=field, value=value):
                    adapter = FakeAdapter()
                    controller = BioGripperG2Controller(adapter)
                    controller.command(110)
                    before = copy.deepcopy({k: v for k, v in vars(controller).items() if k != "adapter"})
                    calls = copy.deepcopy(adapter.calls)
                    args = dict(position_mm=110, speed_mm_s=500 / 3, force_percent=50)
                    args[field] = value
                    with self.assertRaises(ValueError):
                        controller.command(**args)
                    self.assertEqual({k: v for k, v in vars(controller).items() if k != "adapter"}, before)
                    self.assertEqual(adapter.calls, calls)

        for dt in (0, -1, float("nan"), float("inf")):
            with self.subTest(dt=dt):
                adapter = FakeAdapter()
                controller = BioGripperG2Controller(adapter)
                before = copy.deepcopy({k: v for k, v in vars(controller).items() if k != "adapter"})
                calls = copy.deepcopy(adapter.calls)
                with self.assertRaises(ValueError):
                    controller.apply(dt)
                self.assertEqual({k: v for k, v in vars(controller).items() if k != "adapter"}, before)
                self.assertEqual(adapter.calls, calls)

    def test_rate_limit_pause_force_reversal_and_reset(self):
        adapter = FakeAdapter()
        controller = BioGripperG2Controller(adapter)
        controller.command(71, speed_mm_s=500 / 12, force_percent=100)
        old_q = controller.reference_q_m
        controller.apply(0.02)
        self.assertAlmostEqual(old_q - controller.reference_q_m, (500 / 12) * 0.02 / 1000, places=12)

        controller.integral_n = -30.0
        controller.pause()
        reference = controller.reference_q_m
        controller.command(150, force_percent=1)
        self.assertTrue(controller.paused)
        self.assertLessEqual(abs(controller.integral_n), 8.0)
        self.assertEqual(adapter.calls[-2], ("limit", 8.0))
        for _ in range(5):
            controller.apply(0.02)
        self.assertEqual(controller.reference_q_m, reference)
        self.assertEqual(adapter.resets, 0)

        controller.resume()
        self.assertEqual(controller.integral_n, 0.0)
        controller.apply(0.02)
        self.assertEqual(adapter.resets, 0)

        for gap in (71, 110, 150, 110):
            controller.reset(gap)
            self.assertAlmostEqual(q_to_gap(adapter.q), gap, places=10)
            self.assertEqual(controller.integral_n, 0.0)
            self.assertFalse(controller.paused)
        self.assertEqual(adapter.resets, 4)

        controller.command(71, speed_percent=0, force_percent=50)
        self.assertTrue(controller.paused)
        controller.command(71, speed_percent=50, force_percent=50)
        self.assertFalse(controller.paused)
        self.assertAlmostEqual(controller.speed_mm_s, DEFAULT_SPEED_MM_S, places=10)

    def test_conditional_integral_does_not_wind_up_at_saturation(self):
        adapter = FakeAdapter()
        controller = BioGripperG2Controller(adapter)
        controller.command(71, force_percent=1)
        for _ in range(200):
            controller.apply(0.02)
        before = controller.integral_n
        for _ in range(100):
            controller.apply(0.02)
        self.assertEqual(controller.integral_n, before)
        self.assertLessEqual(abs(controller.integral_n), 8.0)

    def test_mujoco_closed_loop_contact_force_and_speed_calibration(self):
        """Closed-loop MuJoCo grasping of an 85 mm rigid plate tracks 4..20 N force and rate limits."""
        spec_xml = mujoco.MjSpec.from_file(str(MODELS_ROOT / "bio_gripper_g2/bio_gripper_g2.xml"))
        plate = spec_xml.worldbody.add_body(name="test_plate", pos=[0.128, 0.0, 0.120])
        plate.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.020, 0.0425, 0.015],
            group=3,
            solref=[0.005, 1.0],
            solimp=[0.99, 0.99, 0.001, 0.5, 2.0],
        )
        model = spec_xml.compile()
        data = mujoco.MjData(model)
        adapter = MuJoCoBioG2Adapter(model, data, 0)
        controller = BioGripperG2Controller(adapter)
        gripper_spec = GripperSpec(
            "jaw",
            "bio_gripper_g2",
            "bio_gripper_g2_left_finger",
            "bio_gripper_g2_right_finger",
            open_gap_m=0.150,
            closed_gap_m=0.071,
        )

        for force_pct in (1.0, 25.0, 50.0, 100.0):
            with self.subTest(force_percent=force_pct):
                controller.reset(120.0)
                mujoco.mj_forward(model, data)
                controller.command(71.0, speed_percent=100.0, force_percent=force_pct)
                for _ in range(800):
                    controller.apply(model.opt.timestep)
                    mujoco.mj_step(model, data)
                fl, fr = gripper_object_contact_forces_n(model, data, gripper_spec, "test_plate")
                expected_f = finger_force_n(force_pct)
                mean_f = 0.5 * (fl + fr)
                self.assertAlmostEqual(mean_f, expected_f, delta=0.5)
                self.assertLessEqual(abs(fl - expected_f), 0.15 * expected_f)
                self.assertLessEqual(abs(fr - expected_f), 0.15 * expected_f)
                self.assertLess(controller.telemetry()["sync_error_mm"], 0.05)

        # Verify rate-limited speed tracking in free space (10% vs 100% speed)
        from ufactory_mujoco import resolve_model_path
        with SimRobot(resolve_model_path("xarm6_with_bio_gripper")) as robot:
            self.assertIsNotNone(robot.bio_g2_controller)
            robot.set_gripper_gap(0.071, speed_percent=10.0, force_percent=50.0)
            robot.settle(0.30)
            gap_slow = robot.get_gripper_gap(measured=True) * 1000.0
            # At 10% speed (83.33 mm/s bilateral), 0.30 s moves ~25 mm from 150 mm -> ~125 mm
            self.assertGreater(gap_slow, 118.0)
            self.assertLess(gap_slow, 132.0)

            robot.open_gripper(wait=True, speed_percent=100.0, force_percent=100.0)
            self.assertAlmostEqual(robot.get_gripper_gap(measured=True), 0.150, places=3)
            robot.set_gripper_gap(0.071, speed_percent=100.0, force_percent=80.0)
            robot.settle(0.30)
            gap_fast = robot.get_gripper_gap(measured=True) * 1000.0
            self.assertAlmostEqual(gap_fast, 71.0, delta=0.5)
            out = robot.gripper_control_output
            self.assertIsNotNone(out)
            self.assertAlmostEqual(out.force_limit_n, drive_limit_n(80.0), places=6)


_PCD = 25.0 * np.sqrt(0.5)
_LUG = {"+y": (15.01, 43.07), "-y": (15.04, -43.02)}
_PIN = (0.044, -24.990)
_ADAPTER_CIRCLES = {
    "rim": ((0.0, 0.0), 37.59, 0.0, (1.6, 3.4), 1, ((0, 42), (90, 270), (318, 360))),
    "bore": ((0.0, 0.0), 15.97, 0.0, (1.6, 3.4), -1, None),
    "M6 hole 45": ((_PCD, _PCD), 3.13, 0.0, (0.2, 0.9), -1, None),
    "M6 hole 135": ((-_PCD, _PCD), 3.13, 0.0, (0.2, 0.9), -1, None),
    "M6 hole -135": ((-_PCD, -_PCD), 3.13, 0.0, (0.2, 0.9), -1, None),
    "M6 hole -45": ((_PCD, -_PCD), 3.13, 0.0, (0.2, 0.9), -1, None),
    "lug arc +y": (_LUG["+y"], 8.02, 0.0, (1.6, 3.4), 1, ((10, 170),)),
    "lug arc -y": (_LUG["-y"], 8.02, 0.0, (1.6, 3.4), 1, ((190, 350),)),
    "lug arc fillet +y": ((2.28, 42.27), 4.74, 0.0, (1.1, 3.9), -1, ((277, 353),)),
    "lug arc fillet -y": ((2.31, -42.26), 4.74, 0.0, (1.1, 3.9), -1, ((7, 83),)),
    "lug flank fillet +y": ((36.26, 35.69), 13.27, 0.0, (1.1, 3.9), -1, ((185, 219),)),
    "lug flank fillet -y": ((36.30, -35.65), 13.27, 0.0, (1.1, 3.9), -1, ((141, 175),)),
    "lug countersink +y": (_LUG["+y"], 7.01, -1.0, (0.4, 3.4), -1, None),
    "lug countersink -y": (_LUG["-y"], 7.01, -1.0, (0.4, 3.4), -1, None),
    "lug screw head +y": (_LUG["+y"], 5.68, 0.0, (0.15, 0.6), 1, None),
    "lug screw head -y": (_LUG["-y"], 5.68, 0.0, (0.15, 0.6), 1, None),
    "pin counterbore": (_PIN, 2.54, 0.0, (0.25, 3.1), -1, None),
    "pin hole": (_PIN, 1.04, 0.0, (2.9, 5.1), -1, None),
}
_ADAPTER_RIMS = {
    "bore": ((0.031, 0.026), 16.972, 16.972),
    "M6 hole 45": ((17.688, 17.695), 3.129, 7.025),
    "M6 hole 135": ((-17.657, 17.678), 3.129, 7.025),
    "M6 hole -135": ((-17.647, -17.670), 3.129, 7.025),
    "M6 hole -45": ((17.700, -17.656), 3.129, 7.025),
    "pin hole": (_PIN, 2.739, 1.040),
}


def _load_bio_g2_adapter_silver_half() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse first-half geometry of bio_gripper_g2_base_silver.obj into adapter frame (mm)."""
    path = ASSETS_DIR / "visual" / "bio_gripper_g2_base_silver.obj"
    verts: list[list[float]] = []
    norms: list[list[float]] = []
    faces_v: list[list[int]] = []
    faces_n: list[list[int]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("v "):
                verts.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("vn "):
                norms.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("f "):
                fv, fn = [], []
                for tok in line.split()[1:]:
                    parts = tok.split("/")
                    v_idx = int(parts[0]) - 1
                    n_idx = int(parts[2]) - 1 if len(parts) > 2 and parts[2] else v_idx
                    fv.append(v_idx)
                    fn.append(n_idx)
                faces_v.append(fv)
                faces_n.append(fn)

    half_faces = len(faces_v) // 2
    half_norms = len(norms) // 2
    v_arr = np.asarray(verts, dtype=np.float64)
    n_arr = np.asarray(norms[:half_norms], dtype=np.float64)
    fv_arr = np.asarray(faces_v[:half_faces], dtype=np.int64)
    fn_arr = np.asarray(faces_n[:half_faces], dtype=np.int64)
    if n_arr.shape != v_arr.shape or not np.array_equal(fv_arr, fn_arr):
        # Remap vertex normals if OBJ face normal indices differ from vertex indices.
        mapped = np.zeros_like(v_arr)
        mapped[fv_arr] = n_arr[fn_arr]
        n_arr = mapped

    rotation = Rotation.from_euler("xyz", np.radians([0.031, 0.254, 0.360])).as_matrix()
    points_mm = (v_arr * 1000.0 - np.array([-22.636, -0.956, -2.516])) @ rotation
    normals = n_arr @ rotation
    return points_mm, normals, fv_arr


class BioG2AdapterGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.points, cls.normals, cls.faces = _load_bio_g2_adapter_silver_half()

    def test_adapter_plate_circles_and_normals(self) -> None:
        """Adapter plate curved faces are analytic surfaces of revolution with analytic normals."""
        points, normals = self.points, self.normals
        z = points[:, 2]
        for feature, (centre, nominal, slope, (z_low, z_high), side, arcs) in _ADAPTER_CIRCLES.items():
            with self.subTest(feature=feature):
                c = np.asarray(centre, dtype=np.float64)
                d = points[:, :2] - c
                r = np.linalg.norm(d, axis=1)
                u = d / np.maximum(r[:, None], 1e-12)
                equivalent = r - slope * z
                facing = side * (normals[:, :2] * u).sum(1) > 0.3
                surface = (z > z_low) & (z < z_high) & (np.abs(equivalent - nominal) < 0.6) & facing
                if arcs is not None:
                    azimuth = np.degrees(np.arctan2(u[:, 1], u[:, 0])) % 360
                    surface &= np.any([(azimuth >= lo) & (azimuth <= hi) for lo, hi in arcs], axis=0)
                self.assertGreater(int(surface.sum()), 30)
                xy = c + equivalent[surface, None] * u[surface]
                fx, fy, k = np.linalg.lstsq(np.c_[2 * xy, np.ones(len(xy))], (xy**2).sum(1), rcond=None)[0]
                fitted = float(np.sqrt(k + fx**2 + fy**2))
                residual = np.hypot(xy[:, 0] - fx, xy[:, 1] - fy) - fitted
                d2 = points[:, :2] - np.array([fx, fy])
                u2 = d2 / np.maximum(np.linalg.norm(d2, axis=1)[:, None], 1e-12)
                expected = side * np.c_[u2, np.full(len(u2), -slope)] / np.hypot(1.0, slope)
                normal_error = np.degrees(np.arccos(np.clip((normals * expected).sum(1), -1.0, 1.0)))[surface]
                self.assertLess(float(np.abs(residual).max()), 0.03)
                self.assertLess(float(normal_error.max()), 1.0)
                self.assertAlmostEqual(fitted, nominal, delta=0.05)

    def test_adapter_plate_through_holes_no_protruding_screws(self) -> None:
        """Adapter plate has open M6 holes at +/-135 deg without protruding screw tips."""
        points = self.points
        self.assertGreater(float(points[:, 2].min()), -0.1)
        for azimuth in (135.0, -135.0):
            axis = 25.0 * np.array([np.cos(np.radians(azimuth)), np.sin(np.radians(azimuth))])
            self.assertFalse(
                bool((np.hypot(*(points[:, :2] - axis).T) < 2.5).any()),
                f"Unexpected geometry inside {azimuth:+.0f} deg M6 hole",
            )

    def test_adapter_plate_flat_faces_stop_at_hole_rims(self) -> None:
        """Contact face (z=0) and plate top (z=5) end cleanly at every hole rim."""
        points, normals, faces = self.points, self.normals, self.faces
        normal_z = normals[faces][:, :, 2]
        tri = points[faces]
        deepest: dict[str, float] = {}
        for hole, (centre, rim_bottom, rim_top) in _ADAPTER_RIMS.items():
            c = np.asarray(centre, dtype=np.float64)
            for z_face, sign, rim in ((0.0, -1.0, rim_bottom), (5.0, 1.0, rim_top)):
                flat = (sign * normal_z > 0.99).all(axis=1) & (np.abs(tri[:, :, 2] - z_face) < 0.03).all(axis=1)
                t = tri[flat][:, :, :2]
                t = t[np.linalg.norm(t - c, axis=2).min(axis=1) < rim + 1.0]
                self.assertGreater(len(t), 0, f"No flat faces near {hole} at z={z_face}")
                dist = np.full(len(t), np.inf)
                for i in range(3):
                    a = t[:, i]
                    d = t[:, (i + 1) % 3] - a
                    s = np.clip(((c - a) * d).sum(1) / np.maximum((d * d).sum(1), 1e-30), 0.0, 1.0)
                    dist = np.minimum(dist, np.linalg.norm(a + s[:, None] * d - c, axis=1))
                e = [t[:, (i + 1) % 3] - t[:, i] for i in range(3)]
                w = [c - t[:, i] for i in range(3)]
                side = [np.sign(e[i][:, 0] * w[i][:, 1] - e[i][:, 1] * w[i][:, 0]) for i in range(3)]
                dist[(side[0] == side[1]) & (side[1] == side[2])] = 0.0
                deepest[f"{hole} z{z_face:.0f}"] = float(rim - dist.min())
        worst = max(deepest, key=deepest.get)
        self.assertLess(deepest[worst], 0.025, f"Flat faces reach {deepest[worst]:.3f} mm into {worst} rim")


if __name__ == "__main__":
    unittest.main()
