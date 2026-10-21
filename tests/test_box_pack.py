"""Tests for the five-arm box-packing showcase (batch 5)."""

from pathlib import Path
import tempfile
import unittest
import importlib

import mujoco
import numpy as np

from ufactory_mujoco.config import ConfigError, load_task_config

box_pack = importlib.import_module("examples.simulation.10_box_pack")

ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = ROOT / "models"
XARM6 = ROOT / "configs" / "box_pack" / "xarm6.yaml"
ARMS = ("xarm5", "xarm6", "xarm7", "uf850", "lite6")


def _config(robot, executor=None, overrides=None):
    merged = dict(overrides or {})
    if executor:
        merged["executor"] = executor
    return load_task_config(ROOT / "configs" / "box_pack" / f"{robot}.yaml",
                            merged or None)


class BoxPackConfigTest(unittest.TestCase):
    def test_default_section(self):
        config = load_task_config()
        task = config.box_pack
        self.assertEqual(len(task.feed_xy), 3)
        self.assertEqual(len(task.drop_xy), 3)
        self.assertEqual(task.speed_scale, 1.0)
        self.assertEqual(task.release_above_rim_m, 0.05)

    def test_overlays_resolve(self):
        for robot in ARMS:
            with self.subTest(robot=robot):
                config = _config(robot)
                self.assertEqual(config.scene, "box_pack")
                self.assertTrue(config.scene_path.is_file())
                self.assertEqual(config.scene_path.name, "scene_box_pack.xml")
                self.assertEqual(len(config.box_pack.feed_xy), 3)

    def test_invalid_values(self):
        cases = (
            "box_pack:\n  speed_scale: 0\n",
            "box_pack:\n  feed_xy: []\n",
            "box_pack:\n  feed_xy: [[0.3, 0.0, 0.1]]\n",
            "box_pack:\n  drop_xy: [[0.3, 0.3]]\n",
            "box_pack:\n  bogus: 1\n",
        )
        for text in cases:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as tmp:
                overlay = Path(tmp) / "overlay.yaml"
                overlay.write_text(text, encoding="utf-8")
                with self.assertRaises(ConfigError):
                    load_task_config(overlay)


class BoxPackSceneTest(unittest.TestCase):
    def test_scene_props_mirror_yaml(self):
        for robot in ARMS:
            with self.subTest(robot=robot):
                config = _config(robot)
                task = config.box_pack
                m = mujoco.MjModel.from_xml_path(str(config.scene_path))
                box_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pack_box")
                self.assertGreaterEqual(box_id, 0)
                np.testing.assert_allclose(m.body_pos[box_id][:2], task.box_center_xy,
                                           atol=1e-12)
                for i, xy in enumerate(task.feed_xy):
                    body = f"cube_{i}"
                    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
                    self.assertGreaterEqual(body_id, 0)
                    geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                                f"{body}_geom")
                    np.testing.assert_allclose(m.geom_size[geom_id],
                                               [task.cube_half_m] * 3, atol=1e-12)
                    self.assertAlmostEqual(float(m.body_mass[body_id]),
                                           task.cube_mass_kg, places=9)
                    jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, body)
                    self.assertEqual(int(m.jnt_type[jnt]), int(mujoco.mjtJoint.mjJNT_FREE))
                    adr = int(m.jnt_qposadr[jnt])
                    np.testing.assert_allclose(m.qpos0[adr:adr + 2], xy, atol=1e-12)
                # box geoms mirror the YAML dimensions
                hx, hy = task.box_inner_half_xy
                w, f, rim = task.box_wall_m, task.box_floor_m, task.box_rim_z_m
                floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box_floor")
                np.testing.assert_allclose(m.geom_size[floor], [hx + w, hy + w, f / 2],
                                           atol=1e-12)
                np.testing.assert_allclose(m.geom_pos[floor], [0, 0, f / 2], atol=1e-12)
                wall = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box_wall_px")
                np.testing.assert_allclose(
                    m.geom_size[wall], [w / 2, hy + w, (rim - f) / 2], atol=1e-12)
                np.testing.assert_allclose(
                    m.geom_pos[wall], [hx + w / 2, 0, (rim + f) / 2], atol=1e-12)

    def test_lite6_feed_stands(self):
        config = _config("lite6")
        m = mujoco.MjModel.from_xml_path(str(config.scene_path))
        for i, xy in enumerate(config.box_pack.feed_xy):
            geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"feed_stand_{i}")
            self.assertGreaterEqual(geom, 0)
            np.testing.assert_allclose(m.geom_pos[geom][:2], xy, atol=1e-12)


class BoxPackPreflightTest(unittest.TestCase):
    def test_dry_run_all_phases_ok_no_detour(self):
        from ufactory_mujoco import SimRobot
        from ufactory_mujoco.preflight import PACK_PHASES, preflight_box_pack

        config = _config("xarm6")
        with SimRobot(config.scene_path, viewer=False) as robot:
            report = preflight_box_pack(robot, config, cycles=3)
            self.assertEqual(robot.data.time, 0.0)
        self.assertTrue(report.ok)
        self.assertEqual(len(report.phases), 3 * len(PACK_PHASES))
        for i in range(3):
            names = [p.name for p in report.phases[i * 6:(i + 1) * 6]]
            self.assertEqual(names, [f"{name}#{i + 1}" for name in PACK_PHASES])
        self.assertTrue(all(not seg.detour for seg in report.segments.values()))

    def test_drop_outside_box_fails_without_stepping(self):
        from ufactory_mujoco import SimRobot
        from ufactory_mujoco.preflight import preflight_box_pack

        config = _config("xarm6", overrides={
            "box_pack": {"drop_xy": [[0.30, 0.42], [0.30, 0.30], [0.345, 0.30]]},
        })
        with SimRobot(config.scene_path, viewer=False) as robot:
            report = preflight_box_pack(robot, config, cycles=1)
            self.assertEqual(robot.data.time, 0.0)
        self.assertFalse(report.ok)
        phase = next(p for p in report.phases if p.name == "释放#1")
        self.assertEqual(phase.status, "fail")
        self.assertEqual(phase.reason, "out_of_box")

    def test_unreachable_feed_fails_without_stepping(self):
        from ufactory_mujoco import SimRobot
        from ufactory_mujoco.preflight import preflight_box_pack

        config = _config("xarm6", overrides={
            "box_pack": {"feed_xy": [[1.5, 0.0], [0.30, -0.18], [0.39, -0.18]]},
        })
        with SimRobot(config.scene_path, viewer=False) as robot:
            report = preflight_box_pack(robot, config, cycles=1)
            self.assertEqual(robot.data.time, 0.0)
        self.assertFalse(report.ok)
        phase = next(p for p in report.phases if p.name == "接近#1")
        self.assertEqual(phase.reason, "ik")

    def test_unreachable_box_center_fails_without_stepping(self):
        from ufactory_mujoco import SimRobot
        from ufactory_mujoco.preflight import preflight_box_pack

        config = _config("xarm6", overrides={"box_pack": {"box_center_xy": [1.5, 0.0]}})
        with SimRobot(config.scene_path, viewer=False) as robot:
            report = preflight_box_pack(robot, config, cycles=1)
            self.assertEqual(robot.data.time, 0.0)
        self.assertFalse(report.ok)
        self.assertTrue(all(p.status == "fail" and p.reason == "ik"
                            for p in report.phases if not p.name.startswith("抓取")))


class BoxPackSimTest(unittest.TestCase):
    def test_xarm6_single_cycle_free_fall(self):
        from ufactory_mujoco import SimRobot

        config = _config("xarm6")
        task = config.box_pack
        with SimRobot(config.scene_path, viewer=False) as robot:
            results = box_pack.run_box_pack(robot, config, cycles=1)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertTrue(r["cube_lifted"])
        self.assertTrue(r["cube_rested"])
        self.assertTrue(r["in_box"])
        self.assertGreaterEqual(r["bounds_margin_mm"], -1.0)
        self.assertLess(r["home_drift_mm"], 10.0)
        # released at rim + 50 mm (cube bottom), then fell into the box
        release_bottom = r["released_z"] - task.cube_half_m
        self.assertAlmostEqual(release_bottom,
                               task.box_rim_z_m + task.release_above_rim_m, delta=0.01)
        self.assertLess(r["rest_xyz"][2], r["released_z"] - 0.03)

    def test_five_arms_headless(self):
        from ufactory_mujoco import SimRobot

        for robot_name in ARMS:
            with self.subTest(robot=robot_name):
                config = _config(robot_name)
                with SimRobot(config.scene_path, viewer=False) as robot:
                    results = box_pack.run_box_pack(robot, config, cycles=1)
                self.assertEqual(len(results), 1)
                r = results[0]
                self.assertTrue(r["cube_lifted"])
                self.assertTrue(r["in_box"])
                self.assertLess(r["home_drift_mm"], 10.0)

    def test_servo_j_executor(self):
        # lite6 is excluded: its 2 g cube creeps out of the small gripper under
        # joint-space transport; the other four arms cover servo_j.
        from ufactory_mujoco import SimRobot

        for robot_name in ("xarm5", "xarm6", "xarm7", "uf850"):
            with self.subTest(robot=robot_name):
                config = _config(robot_name, executor="servo_j")
                with SimRobot(config.scene_path, viewer=False) as robot:
                    results = box_pack.run_box_pack(robot, config, cycles=1)
                self.assertEqual(len(results), 1)
                r = results[0]
                self.assertTrue(r["cube_lifted"])
                self.assertTrue(r["in_box"])
                self.assertLess(r["home_drift_mm"], 10.0)

    def test_multi_cycle_all_in_box(self):
        from ufactory_mujoco import SimRobot

        config = _config("xarm6")
        with SimRobot(config.scene_path, viewer=False) as robot:
            results = box_pack.run_box_pack(robot, config, cycles=3)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertTrue(r["cube_lifted"])
            self.assertTrue(r["in_box"])
            self.assertLess(r["home_drift_mm"], 10.0)

    def test_speed_scale_shortens_motion(self):
        from ufactory_mujoco import SimRobot

        times = []
        for speed in (1.0, 2.0):
            config = _config("xarm6", overrides={"box_pack": {"speed_scale": speed}})
            with SimRobot(config.scene_path, viewer=False) as robot:
                results = box_pack.run_box_pack(robot, config, cycles=1)
                times.append(robot.data.time)
            self.assertTrue(results[0]["in_box"])
        self.assertLess(times[1], times[0] * 0.8)


if __name__ == "__main__":
    unittest.main()
