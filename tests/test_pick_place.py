"""Tests for the xArm 6 + Gripper G2 cube pick-and-place example (roadmap 0.0.4)."""

from pathlib import Path
import unittest
import importlib

import mujoco
import numpy as np

from ufactory_mujoco.config import ConfigError, load_task_config

pick_place = importlib.import_module("examples.simulation.05_pick_place")

ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = ROOT / "models"
SCENE = MODELS_ROOT / "xarm6_with_gripper" / "scene_pick_place.xml"
XARM6 = ROOT / "configs" / "pick_place" / "xarm6.yaml"


class PickPlaceSceneTest(unittest.TestCase):
    def test_scene_props(self):
        """scene_pick_place.xml: 30 mm / 17 g cube free body + visual-only target marker."""
        m = mujoco.MjModel.from_xml_path(str(SCENE))
        cube_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.assertGreaterEqual(cube_id, 0)
        geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.assertAlmostEqual(m.body_mass[cube_id], 0.017, places=6)
        np.testing.assert_allclose(m.geom_size[geom_id], [0.015, 0.015, 0.015], atol=1e-12)
        # cube is a free body
        jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube")
        self.assertEqual(int(m.jnt_type[jnt_id]), int(mujoco.mjtJoint.mjJNT_FREE))
        # target marker: visual only, at (0.30, 0.30, 0.015)
        marker_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target_marker")
        self.assertGreaterEqual(marker_id, 0)
        self.assertEqual(m.geom_contype[marker_id], 0)
        self.assertEqual(m.geom_conaffinity[marker_id], 0)
        np.testing.assert_allclose(m.geom_pos[marker_id], [0.30, 0.30, 0.015], atol=1e-12)

    def test_simrobot_init_keeps_cube_at_authored_pose(self):
        """Keyframe state init must not drop the free-joint cube to the origin
        (MuJoCo < 3.15 pads partial keyframe qpos vectors with zeros)."""
        from ufactory_mujoco import SimRobot

        with SimRobot(SCENE, viewer=False) as robot:
            cube_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
            np.testing.assert_allclose(robot.data.body(cube_id).xpos,
                                       [0.30, 0.0, 0.015], atol=1e-9)

    def test_resolve_scene(self):
        config = load_task_config(XARM6)
        self.assertEqual(config.scene_path, SCENE)
        with self.assertRaises(ConfigError):
            load_task_config(XARM6, {"executor": "servo"})


class IkJointLimitTest(unittest.TestCase):
    def test_solve_ik_respects_joint_limits(self):
        """DLS IK must clamp hinge/slide joints to their ranges (numpy-scalar enum
        comparison silently skipped the clamp before, allowing multi-turn escapes)."""
        from ufactory_mujoco.ik import solve_ik

        m = mujoco.MjModel.from_xml_path(str(SCENE))
        d = mujoco.MjData(m)
        q, ok = solve_ik(m, d, "attachment_site", np.array([0.30, 0.0, 0.45]),
                         pick_place.GRIP_DOWN_QUAT, q_init=np.zeros(m.nq))
        self.assertTrue(ok)
        for j in range(m.njnt):
            if int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                adr = m.jnt_qposadr[j]
                lo, hi = m.jnt_range[j]
                # multi-turn equivalents (e.g. -396 deg for -36 deg) must be folded into range
                self.assertGreaterEqual(q[adr], lo - 1e-9, f"joint {m.joint(j).name} below range")
                self.assertLessEqual(q[adr], hi + 1e-9, f"joint {m.joint(j).name} above range")


class PickPlaceEndToEndTest(unittest.TestCase):
    def test_open_gripper_refuses_lift(self):
        """Fingers start open and clear of the cube, so the lift gate must refuse."""
        from ufactory_mujoco import SimRobot
        from ufactory_mujoco.grippers import GraspError

        with SimRobot(SCENE, viewer=False) as robot:
            cube_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
            z_before = float(robot.data.body(cube_id).xpos[2])
            with self.assertRaises(GraspError):
                robot.require_bilateral_contact("cube")
            self.assertAlmostEqual(float(robot.data.body(cube_id).xpos[2]), z_before)

    def test_pick_place_headless(self):
        """Full pick-and-place sequence: cube lifted, placed within 50 mm, home within 10 mm."""
        from ufactory_mujoco import SimRobot

        config = load_task_config(XARM6)
        with SimRobot(config.scene_path, viewer=False) as robot:
            results = pick_place.run_pick_place(robot, config, cycles=1)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertTrue(np.isfinite(r["place_error_mm"]))
        self.assertTrue(r["cube_lifted"], "cube was not lifted after closing the gripper")
        self.assertLess(r["place_error_mm"], 50.0, "place_error_mm above the 50 mm success threshold")
        self.assertLess(r["home_drift_mm"], 10.0, "home_drift_mm above the 10 mm success threshold")
        self.assertEqual(r["robot"], "xarm6")
        self.assertEqual(r["executor"], "servo_cartesian")
        self.assertLess(r["place_error_mm"], 15.0)


class MultiArmPickPlaceTests(unittest.TestCase):
    ARMS = ("xarm5", "xarm6", "xarm7", "uf850", "lite6")

    def _config(self, robot, executor):
        return load_task_config(ROOT / "configs" / "pick_place" / f"{robot}.yaml",
                                {"executor": executor})

    def test_overlays_resolve_and_lite6_gap_fits(self):
        for robot in self.ARMS:
            with self.subTest(robot=robot):
                config = self._config(robot, "servo_cartesian")
                self.assertEqual(config.scene, "pick_place")
                self.assertTrue(config.scene_path.is_file())
                self.assertEqual(config.scene_path.name, "scene_pick_place.xml")
        lite6 = self._config("lite6", "servo_cartesian")
        task = lite6.pick_place
        self.assertLessEqual(task.grasp_gap_m, 0.0178)
        self.assertLess(task.grasp_gap_m, 2 * task.cube_half_m)
        self.assertLess(task.cube_half_m, 0.0089)

    def test_grasp_and_place_ik(self):
        from ufactory_mujoco import SimRobot

        for robot_name in self.ARMS:
            config = self._config(robot_name, "servo_j")
            task = config.pick_place
            with self.subTest(robot=robot_name):
                with SimRobot(config.scene_path, viewer=False) as robot:
                    heights, q_home = pick_place.tcp_task_heights(robot, task)
                    for xy, z in (
                        (task.cube_xy, heights["grasp_z"]),
                        (task.target_xy, heights["lift_z"]),
                    ):
                        q, ok = robot.solve_tcp(
                            [xy[0], xy[1], z], pick_place.GRIP_DOWN_QUAT, q_init=q_home)
                        self.assertTrue(ok, f"IK failed at {xy} z={z}")
                        self.assertEqual(len(q), robot.model.nq)

    def test_headless_both_executors(self):
        from ufactory_mujoco import SimRobot

        for robot_name in self.ARMS:
            for executor in ("servo_cartesian", "servo_j"):
                with self.subTest(robot=robot_name, executor=executor):
                    config = self._config(robot_name, executor)
                    with SimRobot(config.scene_path, viewer=False) as robot:
                        results = pick_place.run_pick_place(robot, config, cycles=1)
                    self.assertEqual(len(results), 1)
                    report = results[0]
                    self.assertEqual(report["robot"], robot_name)
                    self.assertEqual(report["executor"], executor)
                    self.assertTrue(report["cube_lifted"])
                    self.assertLess(report["place_error_mm"], 50.0)
                    self.assertLess(report["home_drift_mm"], 10.0)


if __name__ == "__main__":
    unittest.main()
