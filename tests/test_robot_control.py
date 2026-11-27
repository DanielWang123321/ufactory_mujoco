"""Joint and Cartesian control regressions for SimRobot."""

import unittest
from unittest.mock import patch
import mujoco
import numpy as np
from ufactory_mujoco import SimRobot
import importlib
from types import SimpleNamespace
from ufactory_mujoco import resolve_model_path
from ufactory_mujoco.robots import ARMS
from ufactory_mujoco.viewer import configure_low_power_rendering

# ==========================================================================
# From test_cartesian_control.py
# ==========================================================================

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
        with patch("ufactory_mujoco.sim_robot.solve_ik_trajectory",
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

# ==========================================================================
# From test_joint_control.py
# ==========================================================================

joint_control = importlib.import_module("examples.simulation.02_joint_control")

class JointControlTests(unittest.TestCase):
    def test_dynamics_pose_sequence_tracks_without_penetration(self):
        home, targets = joint_control.get_dynamics_targets()
        self.assertEqual(len(targets), 20)
        np.testing.assert_allclose(targets[4][1], np.deg2rad([90, -90, -60, 170, 160, -179]))
        np.testing.assert_allclose(targets[14][1], np.deg2rad([-90, -90, -60, -170, -20, 179]))
        for index in (9, 19):
            np.testing.assert_array_equal(targets[index][1], home)

        for speed in (90, 120):
            with self.subTest(speed=speed):
                self._check_dynamics_motion(targets, speed)

    def _check_dynamics_motion(self, targets, speed_deg_s):
        with SimRobot() as robot:
            original_step = robot._step
            last_ctrl = robot.data.ctrl.copy()

            def checked_step():
                nonlocal last_ctrl
                ctrl = robot.data.ctrl.copy()
                speed = np.max(np.abs(ctrl - last_ctrl)) / robot.model.opt.timestep
                self.assertLessEqual(speed, np.deg2rad(speed_deg_s) + 1e-8)
                last_ctrl = ctrl
                original_step()
                self.assertTrue(np.isfinite(robot.data.qpos).all())
                for contact in robot.data.contact:
                    self.assertGreaterEqual(contact.dist, -1e-6)

            robot._step = checked_step
            for name, q in targets:
                with self.subTest(pose=name):
                    duration = joint_control.move_duration(
                        robot.data.ctrl[:robot._n_arm_actuators], q, speed_deg_s,
                        robot.model.opt.timestep,
                    )
                    robot.move_joints(q, duration)
                    self.assertLess(np.max(np.abs(robot.get_joint_positions() - q)), 0.01)

    def test_demo_presets_keep_every_arm_and_combination_above_the_floor(self):
        """ready -> demo of example 02 must not push a link or a mounted tool into the floor."""
        from ufactory_mujoco import available_models
        names = sorted(n for n in available_models() if n.split("_with_")[0] in ARMS and "_with_al0700" not in n)
        self.assertGreaterEqual(len(names), 23)
        for name in names:
            with self.subTest(model=name):
                model = mujoco.MjModel.from_xml_path(str(resolve_model_path(name)))
                data = mujoco.MjData(model)
                n = ARMS[name.split("_with_")[0]].dof
                ready, demo = joint_control.get_arm_targets(SimpleNamespace(model=model, n_arm=n), name)
                key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
                np.testing.assert_allclose(ready, model.key(key).qpos[:n], atol=1e-9)
                for q in (ready, demo):
                    self.assertTrue(np.all(q >= model.jnt_range[:n, 0]) and np.all(q <= model.jnt_range[:n, 1]))
                floors = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
                moving = [g for g in range(model.ngeom)
                          if model.body_weldid[model.geom_bodyid[g]] != 0
                          and (model.geom_contype[g] or model.geom_conaffinity[g])]
                self.assertTrue(floors and moving)
                for s in np.linspace(0.0, 1.0, 11):
                    mujoco.mj_resetDataKeyframe(model, data, key)
                    data.qpos[:n] = (1.0 - s) * ready + s * demo
                    mujoco.mj_forward(model, data)
                    clearance = min(mujoco.mj_geomDistance(model, data, g, f, 1.0, None)
                                    for g in moving for f in floors)
                    self.assertGreater(clearance, 0.001, f"path fraction {s:.1f}")
                    for contact in data.contact[: data.ncon]:
                        self.assertGreaterEqual(contact.dist, -1e-4, f"penetration at path fraction {s:.1f}")

    def test_fractional_duration_reaches_exact_command(self):
        with SimRobot() as robot:
            q = robot.get_joint_positions()
            q[0] += 0.05
            dt = robot.model.opt.timestep
            t0 = robot.data.time
            robot.move_joints(q, duration=2.5 * dt, wait=False)
            np.testing.assert_allclose(robot.data.ctrl, q, atol=1e-12)
            self.assertAlmostEqual(robot.data.time - t0, 3 * dt)

    def test_viewer_sync_rate_preserves_physics_rate(self):
        now = [0.0]
        sync_times = []

        def sleep(duration):
            now[0] += duration

        viewer = SimpleNamespace(
            sync=lambda: sync_times.append(now[0]), is_running=lambda: True,
            close=lambda: None,
        )
        with patch("ufactory_mujoco.sim_robot.PassiveViewer", return_value=viewer), patch(
            "ufactory_mujoco.sim_robot.time.monotonic", side_effect=lambda: now[0]
        ), patch("ufactory_mujoco.sim_robot.time.sleep", side_effect=sleep):
            with SimRobot(viewer=True, viewer_hz=30) as robot:
                robot.settle(0.1)
                self.assertAlmostEqual(robot.data.time, 0.1)
                self.assertAlmostEqual(now[0], 0.1)
                # Two periodic updates plus a forced update at the final state.
                self.assertEqual(len(sync_times), 3)
                self.assertAlmostEqual(sync_times[-1], robot.data.time)

    def test_closed_viewer_stops_settling(self):
        viewer = SimpleNamespace(is_running=lambda: False, close=lambda: None)
        with patch("ufactory_mujoco.sim_robot.PassiveViewer", return_value=viewer):
            with SimRobot(viewer=True) as robot:
                robot.settle(0.5)
                self.assertEqual(robot.data.time, 0.0)

    def test_render_quality_preserves_physics(self):
        xml = """<mujoco><asset><material name="floor" reflectance="0.2"/></asset>
          <worldbody><light castshadow="true"/><body pos="0 0 1">
            <joint type="hinge"/><geom type="box" size="0.1 0.1 0.1"/>
          </body></worldbody></mujoco>"""
        full = mujoco.MjModel.from_xml_string(xml)
        fast = mujoco.MjModel.from_xml_string(xml)
        configure_low_power_rendering(fast)
        self.assertFalse(np.any(fast.light_castshadow))
        self.assertFalse(np.any(fast.mat_reflectance))
        full_data, fast_data = mujoco.MjData(full), mujoco.MjData(fast)
        for data in (full_data, fast_data):
            data.qpos[:] = 0.3
            data.qfrc_applied[:] = 0.2
        mujoco.mj_step(full, full_data, nstep=100)
        mujoco.mj_step(fast, fast_data, nstep=100)
        np.testing.assert_array_equal(fast_data.qpos, full_data.qpos)
        np.testing.assert_array_equal(fast_data.qvel, full_data.qvel)

    def test_n_arm_follows_the_loaded_model(self):
        for name, n in (("xarm5", 5), ("xarm6", 6), ("xarm7", 7)):
            with self.subTest(model=name):
                with SimRobot(resolve_model_path(name)) as robot:
                    self.assertEqual(robot.n_arm, n)
                    self.assertEqual(robot.get_joint_positions().shape, (n,))

    def test_move_joints_rejects_wrong_length(self):
        with SimRobot() as robot:
            ctrl = robot.data.ctrl.copy()
            with self.assertRaises(ValueError):
                robot.move_joints(np.zeros(robot.n_arm - 1), duration=0.1)
            np.testing.assert_array_equal(robot.data.ctrl, ctrl)

    def test_arm_commands_reject_nan_and_out_of_limits(self):
        with SimRobot() as robot:
            ctrl = robot.data.ctrl.copy()
            q_nan = np.zeros(robot.n_arm)
            q_nan[0] = np.nan
            for bad_q in (q_nan, np.full(robot.n_arm, 10.0), np.full(robot.n_arm, -10.0)):
                with self.assertRaises(ValueError):
                    robot.set_joint_positions(bad_q)
                with self.assertRaises(ValueError):
                    robot.move_joints(bad_q, duration=0.2)
            np.testing.assert_array_equal(robot.data.ctrl, ctrl)


class StartPoseTests(unittest.TestCase):
    """Where SimRobot starts: the model's initial keyframe, or ``start``."""

    def test_default_start_is_zero_unless_the_model_has_init(self):
        for name, zero in (("xarm6", True), ("xarm6_with_gripper", False), ("xarm5_with_vacuum_gripper", False),
                           ("uf850_with_gripper", True), ("lite6_with_gripper", True)):
            with self.subTest(model=name), SimRobot(name) as robot:
                q = robot.get_joint_positions()
                np.testing.assert_array_equal(robot.data.ctrl[:robot.n_arm], q)
                if zero:
                    np.testing.assert_array_equal(q, np.zeros(robot.n_arm))
                else:
                    np.testing.assert_array_equal(q, robot.model.key("init").qpos[:robot.n_arm])
                    self.assertEqual(robot.data.ncon, 0)

    def test_start_joints_and_tcp_pose(self):
        q_start = np.array([0.1, -0.5, -0.5, 0.0, 1.0, 0.0])
        with SimRobot("xarm6_with_gripper", start=q_start) as robot:
            np.testing.assert_array_equal(robot.get_joint_positions(), q_start)
            np.testing.assert_array_equal(robot.data.ctrl[:6], q_start)
            np.testing.assert_array_equal(robot.data.qvel, 0.0)
        down = (0.0, 1.0, 0.0, 0.0)
        with SimRobot("lite6_with_gripper", start=((0.25, 0.05, 0.25), down)) as robot:
            pos, quat = robot.get_tcp_pose()
            np.testing.assert_allclose(pos, [0.25, 0.05, 0.25], atol=5e-4)
            self.assertAlmostEqual(abs(float(np.dot(quat, down))), 1.0, places=6)

    def test_bad_start_is_rejected(self):
        for start in (np.zeros(5), np.full(6, 10.0), ((2.0, 0.0, 0.3), (0.0, 1.0, 0.0, 0.0))):
            with self.subTest(start=start), self.assertRaises(ValueError):
                SimRobot("xarm6_with_gripper", start=start)


if __name__ == "__main__":
    unittest.main()
