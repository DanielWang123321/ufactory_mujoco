"""Unit tests for CollisionChecker (roadmap 0.0.6 M2).

Scene constants are calibrated against the planar 2-joint arm in
``tests/fixtures/planar_arm.xml``: q=[0, 0] drives both links through the disk
(penetration ~0.12 m), folded configurations are free, and q=[0.2, 0.4]
passes the disk with ~17 mm gap.
"""

import unittest
from pathlib import Path

import mujoco
import numpy as np

from src.collision import CollisionChecker, arm_joint_limits

ROOT = Path(__file__).resolve().parent.parent
PLANAR_XML = ROOT / "tests" / "fixtures" / "planar_arm.xml"

COLLIDING = np.array([0.0, 0.0])
NEAR_MISS = np.array([0.2, 0.4])          # ~17 mm clearance to the disk
FREE_FOLDED = np.array([1.2, -0.6])
FREE_OTHER = np.array([-1.2, 0.6])

GRIPPER_BOUNDARY_XML = """
<mujoco model="boundary">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.2">
      <joint name="j1" type="hinge" axis="0 0 1" range="-1 1"/>
      <geom type="box" size="0.05 0.05 0.1"/>
      <body name="link" pos="0 0 0.2">
        <joint name="j2" type="hinge" axis="0 0 1" range="-2 2"/>
        <geom type="box" size="0.05 0.05 0.1"/>
        <body name="finger" pos="0 0 0.2">
          <joint name="slide" type="slide" axis="0 0 1" range="0 0.04"/>
          <geom type="box" size="0.02 0.02 0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <general name="act1" joint="j1" gainprm="1" biasprm="0 -1 0"/>
    <general name="act2" joint="j2" gainprm="1" biasprm="0 -1 0"/>
    <general name="gripper" joint="slide" gainprm="1" biasprm="0 -1 0"/>
  </actuator>
</mujoco>
"""


class ArmJointLimitsTests(unittest.TestCase):
    def test_synthetic_arm_limits(self):
        n, lo, hi = arm_joint_limits(mujoco.MjModel.from_xml_path(str(PLANAR_XML)))
        self.assertEqual(n, 2)
        np.testing.assert_allclose(lo, [-3.14, -3.14])
        np.testing.assert_allclose(hi, [3.14, 3.14])

    def test_gripper_actuator_excluded(self):
        n, lo, hi = arm_joint_limits(mujoco.MjModel.from_xml_string(GRIPPER_BOUNDARY_XML))
        self.assertEqual(n, 2)  # slide joint driven by 'gripper' is not arm
        np.testing.assert_allclose(lo, [-1.0, -2.0])
        np.testing.assert_allclose(hi, [1.0, 2.0])

    def test_xarm6_limits(self):
        n, lo, hi = arm_joint_limits(
            mujoco.MjModel.from_xml_path(str(ROOT / "models" / "xarm6" / "xarm6.xml")))
        self.assertEqual(n, 6)
        np.testing.assert_allclose(lo[1], -2.059)   # joint2 range from the MJCF
        np.testing.assert_allclose(hi[1], 2.0944)


class CollisionCheckerTests(unittest.TestCase):
    def setUp(self):
        self.checker = CollisionChecker(PLANAR_XML, safety_margin=0.0)

    def test_known_states(self):
        self.assertTrue(self.checker.in_collision(COLLIDING))
        self.assertLess(self.checker.min_distance(COLLIDING), 0.0)
        for q in (FREE_FOLDED, FREE_OTHER, NEAR_MISS):
            self.assertFalse(self.checker.in_collision(q), q)
            self.assertGreater(self.checker.min_distance(q), 0.0, q)

    def test_min_distance_decreases_when_approaching(self):
        d_far = self.checker.min_distance(np.array([0.25, 0.4]))
        d_near = self.checker.min_distance(NEAR_MISS)
        self.assertGreater(d_far, d_near)
        self.assertGreater(d_near, self.checker.min_distance(COLLIDING))

    def test_safety_margin_catches_near_miss(self):
        self.assertFalse(self.checker.in_collision(NEAR_MISS))
        self.assertTrue(self.checker.in_collision(NEAR_MISS, safety_margin=0.05))
        strict = CollisionChecker(PLANAR_XML, safety_margin=0.05)
        self.assertTrue(strict.in_collision(NEAR_MISS))
        self.assertFalse(strict.in_collision(np.array([0.3, 0.4])))  # > 5 cm gap

    def test_ignore_pairs(self):
        ignored = CollisionChecker(
            PLANAR_XML, ignore_pairs=[("base", "disk"), ("elbow", "disk")])
        self.assertFalse(ignored.in_collision(COLLIDING))

    def test_repeated_queries_are_deterministic(self):
        d1 = self.checker.min_distance(NEAR_MISS)
        d2 = self.checker.min_distance(NEAR_MISS)
        self.assertEqual(d1, d2)
        self.assertEqual(self.checker.in_collision(COLLIDING),
                         self.checker.in_collision(COLLIDING))

    def test_checker_does_not_touch_simulation_model(self):
        scene = ROOT / "models" / "xarm6" / "scene.xml"
        robot_model = mujoco.MjModel.from_xml_path(str(scene))
        robot_data = mujoco.MjData(robot_model)
        mujoco.mj_forward(robot_model, robot_data)
        ncon_before = robot_data.ncon

        checker = CollisionChecker(scene)
        checker.in_collision(np.zeros(6))

        np.testing.assert_array_equal(robot_model.geom_margin, 0.0)
        mujoco.mj_forward(robot_model, robot_data)
        self.assertEqual(robot_data.ncon, ncon_before)

    def test_xarm6_ready_pose_is_free(self):
        checker = CollisionChecker(ROOT / "models" / "xarm6" / "scene.xml")
        self.assertFalse(checker.in_collision(np.zeros(6)))

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            self.checker.in_collision(np.zeros(3))
        with self.assertRaises(ValueError):
            self.checker.in_collision(np.full(2, np.nan))
        with self.assertRaises(ValueError):
            CollisionChecker(PLANAR_XML, safety_margin=0.1, query_margin=0.05)
        with self.assertRaises(ValueError):
            CollisionChecker(PLANAR_XML, safety_margin=-0.1)
        with self.assertRaises(ValueError):
            CollisionChecker(PLANAR_XML, ignore_pairs=[("base", "nope")])
        with self.assertRaises(TypeError):
            CollisionChecker(42)


if __name__ == "__main__":
    unittest.main()
