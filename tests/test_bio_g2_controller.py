"""Unit and closed-loop MuJoCo physics tests for Bio Gripper G2 (bio_g2_calibrated_v1)."""

from __future__ import annotations

import copy
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from ufactory_mujoco.bio_g2_controller import (
    BioGripperG2Controller,
    GripperController,
    MuJoCoBioG2Adapter,
)
from ufactory_mujoco.bio_g2_model import (
    BASE_INERTIA_SCALE,
    DEFAULT_SPEED_MM_S,
    GLB_FINGER_VISUAL_Y_OFFSET_M,
    LEFT_JOINT as LEFT,
    MAX_SPEED_MM_S,
    MIN_SPEED_MM_S,
    RIGHT_JOINT as RIGHT,
    SPEEDS_MM_S,
    BioGripperG2Model,
    bio_g2_physics_summary,
    bio_g2_speed_mm_s_from_percent,
    bio_g2_speed_percent_from_mm_s,
    drive_limit_n,
    finger_force_n,
    gap_to_q,
    q_to_gap,
)
from ufactory_mujoco.grippers import GripperSpec, gripper_object_contact_forces_n
from ufactory_mujoco.sim_robot import SimRobot

ROOT = Path(__file__).resolve().parent.parent
URDF_FIXTURE = ROOT / "tests/fixtures/bio_gripper_g2_movable_visual.urdf"
ASSETS_DIR = ROOT / "models/bio_gripper_g2/assets/bio_gripper_g2"


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
        mjcf_model = BioGripperG2Model.load()
        urdf_model = BioGripperG2Model.load(URDF_FIXTURE)
        summary = bio_g2_physics_summary()
        self.assertAlmostEqual(mjcf_model.total_mass_kg, 0.79, places=12)
        self.assertAlmostEqual(urdf_model.total_mass_kg, 0.79, places=12)
        self.assertEqual(mjcf_model.link_masses_kg, urdf_model.link_masses_kg)
        self.assertAlmostEqual(summary["total_mass_kg"], 0.79, places=12)
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
                    self.assertTrue(collision.is_convex and collision.is_watertight)
                    root_hull = trimesh.load(ASSETS_DIR / f"collision/{name}_convex_1.stl", force="mesh")
                    self.assertTrue(root_hull.is_convex and root_hull.is_watertight)
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
                    controller = GripperController(adapter)
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
                controller = GripperController(adapter)
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
        controller = GripperController(adapter)
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
        spec_xml = mujoco.MjSpec.from_file(str(ROOT / "models/bio_gripper_g2/bio_gripper_g2.xml"))
        plate = spec_xml.worldbody.add_body(name="test_plate", pos=[0.128, 0.0, 0.050])
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
        with SimRobot(ROOT / "models/xarm6_with_bio_gripper/scene.xml") as robot:
            self.assertIsNotNone(robot.bio_g2_controller)
            robot.set_gripper_command(71.0, speed_percent=10.0, force_percent=50.0)
            robot.settle(0.30)
            gap_slow = robot.get_measured_gripper_gap() * 1000.0
            # At 10% speed (83.33 mm/s bilateral), 0.30 s moves ~25 mm from 150 mm -> ~125 mm
            self.assertGreater(gap_slow, 118.0)
            self.assertLess(gap_slow, 132.0)

            robot.open_gripper(wait=True, speed_percent=100.0, force_percent=100.0)
            self.assertAlmostEqual(robot.get_measured_gripper_gap(), 0.150, places=3)
            robot.set_gripper_command(71.0, speed_percent=100.0, force_percent=80.0)
            robot.settle(0.30)
            gap_fast = robot.get_measured_gripper_gap() * 1000.0
            self.assertAlmostEqual(gap_fast, 71.0, delta=0.5)
            out = robot.get_gripper_control_output()
            self.assertIsNotNone(out)
            self.assertAlmostEqual(out.force_limit_n, drive_limit_n(80.0), places=6)


if __name__ == "__main__":
    unittest.main()
