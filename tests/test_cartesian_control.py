"""Cartesian pose interpolation and TCP orientation tracking regressions."""

import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from src import SimRobot


class CartesianControlTests(unittest.TestCase):
    START = np.array([0.2, -0.8, -1.2, 0.5, 0.3, -0.2])

    def _set_start(self, robot):
        robot.data.qpos[:] = self.START
        robot.data.ctrl[:] = self.START
        mujoco.mj_forward(robot.model, robot.data)

    def _pose_at(self, robot, joints):
        robot.eval_data.qpos[:] = joints
        mujoco.mj_forward(robot.model, robot.eval_data)
        pos = robot.eval_data.site_xpos[robot._site].copy()
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, robot.eval_data.site_xmat[robot._site])
        return pos, quat

    def _rotation(self, quat):
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, quat)
        return matrix.reshape(3, 3)

    def _reference_poses(self, robot, pos, quat):
        """Inspect the Cartesian references before IK or servo tracking."""
        joint_path = np.tile(self.START, (50, 1))
        with patch("src.sim_robot.solve_ik_trajectory",
                   return_value=(joint_path, True)) as solve_path, patch.object(
                       robot, "move_joints"
                   ):
            robot.move_tcp(pos, quat, duration=2.5, wait=False)
        args = solve_path.call_args.args
        return args[3], args[4]

    def test_tool_axis_rotation_waypoints_match_fk(self):
        """A J6-only rotation follows the tool Z axis at every reference point."""
        with SimRobot(viewer=False) as robot:
            self._set_start(robot)
            for delta in (0.2, -0.2):
                goal = self.START.copy()
                goal[5] += delta
                pos, quat = self._pose_at(robot, goal)
                for sign in (1.0, -1.0):
                    with self.subTest(delta=delta, quaternion_sign=sign):
                        positions, orientations = self._reference_poses(
                            robot, pos, sign * quat
                        )
                        self.assertEqual(orientations.shape, (50, 4))
                        for i, orientation in enumerate(orientations):
                            joints = self.START.copy()
                            joints[5] += delta * (i + 1) / len(orientations)
                            expected_pos, expected_quat = self._pose_at(robot, joints)
                            np.testing.assert_allclose(
                                positions[i], expected_pos, rtol=0, atol=1e-12,
                            )
                            np.testing.assert_allclose(
                                self._rotation(orientation), self._rotation(expected_quat),
                                rtol=0, atol=1e-12,
                            )

    def test_interpolation_endpoint_matches_general_pose_goal(self):
        """A noncommuting pose change still ends at the commanded orientation."""
        with SimRobot(viewer=False) as robot:
            self._set_start(robot)
            goal = self.START + np.array([0.1, -0.05, 0.03, 0.07, -0.06, 0.2])
            pos, quat = self._pose_at(robot, goal)
            positions, orientations = self._reference_poses(robot, pos, quat)
            np.testing.assert_allclose(positions[-1], pos, rtol=0, atol=1e-12)
            np.testing.assert_allclose(
                self._rotation(orientations[-1]), self._rotation(quat),
                rtol=0, atol=1e-12,
            )

    def test_tcp_rotation_tracks_goal_headless(self):
        """The reported J6 rotation reaches its Cartesian goal after simulation."""
        with SimRobot(viewer=False) as robot:
            self._set_start(robot)
            goal = self.START.copy()
            goal[5] = 0.0
            pos, quat = self._pose_at(robot, goal)
            robot.move_tcp(pos, quat, duration=2.5)
            mujoco.mj_forward(robot.model, robot.data)
            actual_pos, actual_quat = robot.get_tcp_pose()
            orientation_error = np.empty(3)
            mujoco.mju_subQuat(orientation_error, quat, actual_quat)
            self.assertLess(np.linalg.norm(actual_pos - pos) * 1000.0, 1.0)
            self.assertLess(np.rad2deg(np.linalg.norm(orientation_error)), 0.5)


if __name__ == "__main__":
    unittest.main()
