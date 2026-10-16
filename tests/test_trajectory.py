"""Unit tests for trajectory planning, time parameterization, and execution."""

import unittest
from pathlib import Path

import mujoco
import numpy as np

from ufactory_mujoco import (
    JointTrajectory,
    SimRobot,
    plan_cartesian_line,
    plan_joint_waypoints,
    plan_ptp_lspb,
    plan_ptp_quintic,
)

ROOT = Path(__file__).resolve().parent.parent
XARM6_XML = ROOT / "models" / "xarm6" / "xarm6.xml"
READY = np.array([0.0, -0.8, -1.2, 1.0, 0.3, 0.0])
TARGET = np.array([0.3, -1.2, -1.0, 0.6, 0.6, 0.5])


class JointTrajectoryDataClassTests(unittest.TestCase):
    def test_properties_and_sampling(self):
        t = np.array([0.0, 0.5, 1.0])
        q = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 2.0]])
        qd = np.zeros_like(q)
        qdd = np.zeros_like(q)

        traj = JointTrajectory(times=t, q=q, qd=qd, qdd=qdd)
        self.assertAlmostEqual(traj.duration, 1.0)
        self.assertAlmostEqual(traj.dt, 0.5)
        self.assertEqual(traj.n_steps, 3)
        self.assertEqual(traj.n_joints, 2)

        q_mid, _, _ = traj.sample(0.5)
        np.testing.assert_allclose(q_mid, [0.5, 1.0])

        q_interp, _, _ = traj.sample(0.25)
        np.testing.assert_allclose(q_interp, [0.25, 0.5])

    def test_concat(self):
        traj1 = plan_ptp_quintic([0.0], [1.0], max_vel=1.0, max_acc=2.0, dt=0.01)
        traj2 = plan_ptp_quintic([1.0], [0.5], max_vel=1.0, max_acc=2.0, dt=0.01)

        combined = traj1.concat(traj2)
        self.assertAlmostEqual(combined.times[0], 0.0)
        self.assertAlmostEqual(combined.q[0, 0], 0.0)
        self.assertAlmostEqual(combined.q[-1, 0], 0.5)
        self.assertGreater(combined.duration, traj1.duration)


class PlanPTPQuinticTests(unittest.TestCase):
    def test_boundary_conditions_and_limits(self):
        max_vel = np.array([0.5, 1.0, 1.5, 0.8, 1.2, 2.0])
        max_acc = np.array([2.0, 3.0, 4.0, 2.5, 3.5, 5.0])
        dt = 0.002

        traj = plan_ptp_quintic(READY, TARGET, max_vel=max_vel, max_acc=max_acc, dt=dt)

        # Boundaries
        np.testing.assert_allclose(traj.q[0], READY, atol=1e-10)
        np.testing.assert_allclose(traj.q[-1], TARGET, atol=1e-10)
        np.testing.assert_allclose(traj.qd[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(traj.qd[-1], 0.0, atol=1e-10)
        np.testing.assert_allclose(traj.qdd[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(traj.qdd[-1], 0.0, atol=1e-10)

        # Limits (with a tiny numerical tolerance)
        max_measured_vel = np.max(np.abs(traj.qd), axis=0)
        max_measured_acc = np.max(np.abs(traj.qdd), axis=0)

        self.assertTrue(np.all(max_measured_vel <= max_vel + 1e-6))
        self.assertTrue(np.all(max_measured_acc <= max_acc + 1e-6))

        # Check derivative consistency
        num_qd = np.gradient(traj.q, dt, axis=0)
        np.testing.assert_allclose(traj.qd[10:-10], num_qd[10:-10], atol=1e-3)

    def test_zero_displacement(self):
        traj = plan_ptp_quintic(READY, READY, max_vel=1.0, max_acc=2.0, dt=0.01)
        self.assertGreaterEqual(traj.duration, 0.01)
        np.testing.assert_allclose(traj.q[0], READY)
        np.testing.assert_allclose(traj.q[-1], READY)
        np.testing.assert_allclose(traj.qd, 0.0)


class PlanPTPLSPBTests(unittest.TestCase):
    def test_boundary_conditions_and_limits(self):
        max_vel = 1.0
        max_acc = 2.0
        dt = 0.002

        traj = plan_ptp_lspb(READY, TARGET, max_vel=max_vel, max_acc=max_acc, dt=dt)

        # Boundaries
        np.testing.assert_allclose(traj.q[0], READY, atol=1e-10)
        np.testing.assert_allclose(traj.q[-1], TARGET, atol=1e-10)
        np.testing.assert_allclose(traj.qd[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(traj.qd[-1], 0.0, atol=1e-10)

        # Limits
        max_measured_vel = np.max(np.abs(traj.qd))
        max_measured_acc = np.max(np.abs(traj.qdd))
        self.assertLessEqual(max_measured_vel, max_vel + 1e-6)
        self.assertLessEqual(max_measured_acc, max_acc + 1e-6)


class PlanJointWaypointsTests(unittest.TestCase):
    def test_spline_passes_through_waypoints_and_respects_limits(self):
        w1 = READY
        w2 = READY + np.array([0.1, -0.2, 0.15, -0.1, 0.2, -0.1])
        w3 = TARGET
        waypoints = np.array([w1, w2, w3])

        max_vel = 0.8
        max_acc = 2.5
        dt = 0.002

        traj = plan_joint_waypoints(waypoints, max_vel=max_vel, max_acc=max_acc, dt=dt, method="spline")

        # Boundary velocities zero
        np.testing.assert_allclose(traj.qd[0], 0.0, atol=1e-7)
        np.testing.assert_allclose(traj.qd[-1], 0.0, atol=1e-7)

        # Respect limits
        self.assertLessEqual(np.max(np.abs(traj.qd)), max_vel + 1e-6)
        self.assertLessEqual(np.max(np.abs(traj.qdd)), max_acc + 1e-6)

        # Start and end match
        np.testing.assert_allclose(traj.q[0], w1, atol=1e-7)
        np.testing.assert_allclose(traj.q[-1], w3, atol=1e-7)

        # Middle waypoint w2 is passed through (within one discrete dt step distance)
        dist_to_w2 = np.min(np.linalg.norm(traj.q - w2, axis=1))
        self.assertLess(dist_to_w2, 1e-3)

    def test_stop_method_connects_waypoints(self):
        waypoints = np.array([READY, READY + 0.1, TARGET])
        traj = plan_joint_waypoints(waypoints, max_vel=1.0, max_acc=2.0, dt=0.01, method="stop")
        np.testing.assert_allclose(traj.q[0], waypoints[0], atol=1e-7)
        np.testing.assert_allclose(traj.q[-1], waypoints[-1], atol=1e-7)


class PlanCartesianLineTests(unittest.TestCase):
    def test_cartesian_straight_line(self):
        model = mujoco.MjModel.from_xml_path(str(XARM6_XML))
        data = mujoco.MjData(model)
        data.qpos[:6] = READY
        mujoco.mj_forward(model, data)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        p0 = data.site_xpos[site_id].copy()
        quat0 = np.empty(4)
        mujoco.mju_mat2Quat(quat0, data.site_xmat[site_id])

        p1 = p0 + np.array([0.08, 0.05, -0.04])
        traj, ok = plan_cartesian_line(
            p0, quat0, p1, quat0, model, site_name="attachment_site",
            max_lin_vel=0.15, max_lin_acc=0.8, q_init=data.qpos.copy(), dt=0.005,
        )
        self.assertTrue(ok, "Cartesian line IK converged")
        self.assertEqual(traj.q.shape[1], model.nq)

        # Validate start and end TCP via FK
        data.qpos[:] = traj.q[0]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.site_xpos[site_id], p0, atol=5e-4)

        data.qpos[:] = traj.q[-1]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.site_xpos[site_id], p1, atol=5e-4)


class SimRobotExecuteTrajectoryTests(unittest.TestCase):
    def test_execute_trajectory_tracking(self):
        with SimRobot() as robot:
            robot.set_joint_positions(READY)
            robot.settle(0.5)

            traj = robot.plan_joint_trajectory(TARGET, max_vel=1.5, max_acc=3.0)
            robot.execute_trajectory(traj, wait=True)

            final_q = robot.get_joint_positions()
            err = np.max(np.abs(final_q - TARGET))
            self.assertLess(err, 0.005, f"Target tracking error {err:.5f} rad exceeds 0.005 rad")

    def test_execute_cartesian_trajectory(self):
        with SimRobot() as robot:
            robot.move_joints(READY, duration=1.0)
            p0, quat0 = robot.get_tcp_pose()
            p_target = p0 + np.array([0.05, 0.05, 0.0])

            traj, ok = robot.plan_cartesian_trajectory(p_target, quat0, max_lin_vel=0.15)
            self.assertTrue(ok)
            robot.execute_trajectory(traj, wait=True)

            p_final, _ = robot.get_tcp_pose()
            err_mm = np.linalg.norm(p_final - p_target) * 1000.0
            self.assertLess(err_mm, 2.0, f"Cartesian error {err_mm:.2f} mm exceeds 2 mm")


if __name__ == "__main__":
    unittest.main()
