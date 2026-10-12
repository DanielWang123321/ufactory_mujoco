"""Dynamics pose tracking and independence of physics from viewer updates."""

import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from src import SimRobot
from src.viewer import configure_low_power_rendering

joint_control = importlib.import_module("examples.02_joint_control")


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
        with patch("src.sim_robot.PassiveViewer", return_value=viewer), patch(
            "src.sim_robot.time.monotonic", side_effect=lambda: now[0]
        ), patch("src.sim_robot.time.sleep", side_effect=sleep):
            with SimRobot(viewer=True, viewer_hz=30) as robot:
                robot.settle(0.1)
                self.assertAlmostEqual(robot.data.time, 0.1)
                self.assertAlmostEqual(now[0], 0.1)
                # Two periodic updates plus a forced update at the final state.
                self.assertEqual(len(sync_times), 3)
                self.assertAlmostEqual(sync_times[-1], robot.data.time)

    def test_closed_viewer_stops_settling(self):
        viewer = SimpleNamespace(is_running=lambda: False, close=lambda: None)
        with patch("src.sim_robot.PassiveViewer", return_value=viewer):
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


if __name__ == "__main__":
    unittest.main()
