"""Tests for the xArm 6 + Gripper G2 cube pick-and-place example (roadmap 0.0.4)."""

from pathlib import Path
import unittest
import importlib

import mujoco
import numpy as np

pick_place = importlib.import_module("examples.05_pick_place")

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"
SCENE = MODELS_ROOT / "xarm6_with_gripper" / "scene_pick_place.xml"


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

    def test_resolve_scene(self):
        self.assertEqual(pick_place.resolve_scene("xarm6_with_gripper"), SCENE)
        self.assertEqual(pick_place.resolve_scene(str(SCENE)), SCENE)
        with self.assertRaises(FileNotFoundError):
            pick_place.resolve_scene("xarm6")


class IkJointLimitTest(unittest.TestCase):
    def test_solve_ik_respects_joint_limits(self):
        """DLS IK must clamp hinge/slide joints to their ranges (numpy-scalar enum
        comparison silently skipped the clamp before, allowing multi-turn escapes)."""
        from src.ik import solve_ik

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
    def test_pick_place_headless(self):
        """Full pick-and-place sequence: cube lifted, placed within 50 mm, home within 10 mm."""
        from src import SimRobot

        with SimRobot(SCENE, viewer=False) as robot:
            results = pick_place.run_pick_place(robot, cycles=1)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertTrue(np.isfinite(r["place_error_mm"]))
        self.assertTrue(r["cube_lifted"], "cube was not lifted after closing the gripper")
        self.assertLess(r["place_error_mm"], 50.0, "place_error_mm above the 50 mm success threshold")
        self.assertLess(r["home_drift_mm"], 10.0, "home_drift_mm above the 10 mm success threshold")


if __name__ == "__main__":
    unittest.main()
