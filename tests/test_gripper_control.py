"""Unit tests verifying Gripper G2 and Bio Gripper G2 control on SimRobot."""

import importlib
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

from src.sim_robot import SimRobot

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"
gripper_demo = importlib.import_module("examples.simulation.04_gripper_control")


class GripperControlTest(unittest.TestCase):
    def test_robot_without_gripper(self):
        """Standard xarm6 has no gripper; gripper methods raise NotImplementedError."""
        scene = MODELS_ROOT / "xarm6" / "scene.xml"
        robot = SimRobot(scene)
        self.assertFalse(robot.has_gripper)
        self.assertEqual(len(robot.get_joint_positions()), 6)

        with self.assertRaises(NotImplementedError):
            robot.get_gripper_position()

        with self.assertRaises(NotImplementedError):
            robot.set_gripper(0.5)

        with self.assertRaises(NotImplementedError):
            robot.open_gripper(wait=False)

        with self.assertRaises(NotImplementedError):
            robot.close_gripper(wait=False)

    def test_xarm6_with_gripper_g2_properties(self):
        """xarm6_with_gripper detects gripper, isolates arm and gripper channels."""
        scene = MODELS_ROOT / "xarm6_with_gripper" / "scene.xml"
        robot = SimRobot(scene)
        self.assertTrue(robot.has_gripper)

        # Arm joints should be exactly 6
        q_arm = robot.get_joint_positions()
        self.assertEqual(len(q_arm), 6)

        # Initial gripper position at keyframe 0 (home: 0 ctrl -> 0.0 normalized)
        pos = robot.get_gripper_position()
        self.assertAlmostEqual(pos, 0.0, places=3)

    def test_xarm6_with_gripper_g2_open_close(self):
        """Test open and close for Gripper G2."""
        scene = MODELS_ROOT / "xarm6_with_gripper" / "scene.xml"
        robot = SimRobot(scene)

        # Close gripper
        robot.close_gripper(wait=True)
        self.assertAlmostEqual(robot.get_gripper_position(), 1.0, places=3)
        # Gripper joints should be contracted (> 0.5 rad)
        gripper_qpos = robot.data.qpos[6:]
        self.assertTrue(np.all(gripper_qpos > 0.5), f"Fingers not closed: {gripper_qpos}")

        # Open gripper
        robot.open_gripper(wait=True)
        self.assertAlmostEqual(robot.get_gripper_position(), 0.0, places=3)
        gripper_qpos = robot.data.qpos[6:]
        self.assertTrue(np.all(gripper_qpos < 0.15), f"Fingers not open: {gripper_qpos}")

    def test_xarm6_with_bio_gripper_properties(self):
        """xarm6_with_bio_gripper detects gripper and isolates channels."""
        scene = MODELS_ROOT / "xarm6_with_bio_gripper" / "scene.xml"
        robot = SimRobot(scene)
        self.assertTrue(robot.has_gripper)

        # Arm joints should be exactly 6
        q_arm = robot.get_joint_positions()
        self.assertEqual(len(q_arm), 6)

        # Keyframe home has 0 ctrl -> normalized 0.0
        pos = robot.get_gripper_position()
        self.assertAlmostEqual(pos, 0.0, places=3)

    def test_xarm6_with_bio_gripper_open_close(self):
        """Test open and close for Bio Gripper G2."""
        scene = MODELS_ROOT / "xarm6_with_bio_gripper" / "scene.xml"
        robot = SimRobot(scene)

        # Close bio gripper (slide joints move inwards to close)
        robot.close_gripper(wait=True)
        self.assertAlmostEqual(robot.get_gripper_position(), 1.0, places=3)
        # Left finger ctrlrange is [-0.0395, 0]; close is -0.0395
        left_finger_pos = robot.data.qpos[6]
        right_finger_pos = robot.data.qpos[7]
        self.assertLess(left_finger_pos, -0.03, f"Left finger not closed: {left_finger_pos}")
        self.assertGreater(right_finger_pos, 0.03, f"Right finger not closed: {right_finger_pos}")

        # Open bio gripper
        robot.open_gripper(wait=True)
        self.assertAlmostEqual(robot.get_gripper_position(), 0.0, places=3)
        left_finger_pos = robot.data.qpos[6]
        right_finger_pos = robot.data.qpos[7]
        self.assertGreater(left_finger_pos, -0.005, f"Left finger not open: {left_finger_pos}")
        self.assertLess(right_finger_pos, 0.005, f"Right finger not open: {right_finger_pos}")

    def test_arm_gripper_motion_isolation(self):
        """Arm motion does not alter gripper state, gripper motion does not move arm."""
        scene = MODELS_ROOT / "xarm6_with_gripper" / "scene.xml"
        robot = SimRobot(scene)

        # Close gripper
        robot.close_gripper(wait=True)
        gripper_closed_pos = robot.get_gripper_position()
        self.assertAlmostEqual(gripper_closed_pos, 1.0, places=3)

        # Move arm to a ready-like goal
        q_arm_init = robot.get_joint_positions()
        q_arm_goal = np.array([0.2, -0.5, -0.8, 0.5, 0.2, 0.1])
        robot.move_joints(q_arm_goal, duration=0.5, wait=True)

        # Arm should reach goal
        q_arm_final = robot.get_joint_positions()
        np.testing.assert_allclose(q_arm_final, q_arm_goal, atol=0.01)

        # Gripper should remain closed
        self.assertAlmostEqual(robot.get_gripper_position(), 1.0, places=3)

        # Open gripper: arm should stay at q_arm_goal
        robot.open_gripper(wait=True)
        np.testing.assert_allclose(robot.get_joint_positions(), q_arm_goal, atol=0.01)
        self.assertAlmostEqual(robot.get_gripper_position(), 0.0, places=3)

    def test_lite6_with_gripper(self):
        """The Lite 6 example moves to the factory pose and grips without collisions."""
        scene = MODELS_ROOT / "lite6_with_gripper" / "scene.xml"
        factory_q = np.deg2rad([0, 9.9, 31.8, 0, 21.9, 0])
        with SimRobot(scene) as robot:
            self.assertTrue(robot.has_gripper)
            self.assertEqual(len(robot.get_joint_positions()), 6)

            penetrating_contacts = set()
            original_step = robot._step

            def check_step():
                original_step()
                for contact in robot.data.contact:
                    if contact.dist < -1e-6:  # Ignore numerical contact noise below 1 micrometre.
                        bodies = tuple(sorted(
                            robot.model.body(int(robot.model.geom_bodyid[g])).name
                            for g in (contact.geom1, contact.geom2)
                        ))
                        penetrating_contacts.add(bodies)

            with patch.object(robot, "_step", side_effect=check_step):
                gripper_demo.move_arm_to_ready(robot)
                np.testing.assert_allclose(robot.get_joint_positions(), factory_q, atol=0.01)

                robot.close_gripper(wait=True)
                self.assertAlmostEqual(robot.get_gripper_position(), 1.0, places=2)
                gripper_qpos = robot.data.qpos[6:8]
                self.assertTrue(np.all(gripper_qpos > 0.007), f"Fingers not closed: {gripper_qpos}")

                robot.open_gripper(wait=True)
                self.assertAlmostEqual(robot.get_gripper_position(), 0.0, places=2)
                gripper_qpos = robot.data.qpos[6:8]
                self.assertTrue(np.all(gripper_qpos < 0.002), f"Fingers not open: {gripper_qpos}")

                robot.set_gripper(0.5)
                robot.settle(0.5)
                np.testing.assert_allclose(robot.data.qpos[6:8], [0.00445, 0.00445], atol=0.0002)
                robot.open_gripper(wait=True)
                np.testing.assert_allclose(robot.get_joint_positions(), factory_q, atol=0.001)

            self.assertFalse(penetrating_contacts, f"Unexpected contacts: {penetrating_contacts}")


if __name__ == "__main__":
    unittest.main()


