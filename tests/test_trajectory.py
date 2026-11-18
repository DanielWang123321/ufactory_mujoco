"""Unit tests for trajectory planning, time parameterization, and execution."""

import unittest
from pathlib import Path

import mujoco
import numpy as np

from ufactory_mujoco import (
    JointTrajectory,
    SimRobot,
    fk_pose,
    plan_cartesian_line,
    plan_joint_waypoints,
    plan_ptp_lspb,
    plan_ptp_quintic,
)
from ufactory_mujoco.kinematics import quat_error

ROOT = Path(__file__).resolve().parent.parent
XARM6_XML = ROOT / "models" / "xarm6" / "xarm6.xml"
XARM6_GRIPPER_SCENE = ROOT / "models" / "xarm6_with_gripper" / "scene.xml"
XARM6_FT_GRIPPER_SCENE = ROOT / "models" / "xarm6_with_ft_sensor_and_gripper" / "scene.xml"
READY = np.array([0.0, -0.8, -1.2, 1.0, 0.3, 0.0])
TARGET = np.array([0.3, -1.2, -1.0, 0.6, 0.6, 0.5])


def _world_compound_rotate(quat0: np.ndarray, rot_z: float, rot_x: float) -> np.ndarray:
    """Apply world-frame X then Z rotations to ``quat0``."""
    qz = np.empty(4)
    qx = np.empty(4)
    qzx = np.empty(4)
    out = np.empty(4)
    mujoco.mju_axisAngle2Quat(qz, np.array([0.0, 0.0, 1.0]), rot_z)
    mujoco.mju_axisAngle2Quat(qx, np.array([1.0, 0.0, 0.0]), rot_x)
    mujoco.mju_mulQuat(qzx, qz, qx)
    mujoco.mju_mulQuat(out, qzx, quat0)
    return out


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

    def test_validation_rejects_empty_nonfinite_or_nonmonotonic(self):
        z = np.zeros((2, 2))
        with self.assertRaises(ValueError):
            JointTrajectory(times=np.array([]), q=np.zeros((0, 2)), qd=np.zeros((0, 2)), qdd=np.zeros((0, 2)))
        with self.assertRaises(ValueError):
            JointTrajectory(times=np.array([0.0, np.nan]), q=z, qd=z, qdd=z)
        with self.assertRaises(ValueError):
            JointTrajectory(times=np.array([0.0, 0.1]), q=np.full((2, 2), np.inf), qd=z, qdd=z)
        with self.assertRaises(ValueError):
            JointTrajectory(times=np.array([0.2, 0.1]), q=z, qd=z, qdd=z)


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

    def test_infeasible_duration_and_invalid_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError, "shorter than minimum feasible"):
            plan_ptp_quintic(READY, TARGET, max_vel=0.5, max_acc=1.0, duration=0.1, dt=0.002)
        for bad_kwargs in (
            {"duration": 0.0},
            {"duration": -1.0},
            {"max_vel": 0.0},
            {"max_vel": -0.5},
            {"max_acc": 0.0},
            {"dt": 0.0},
            {"dt": float("nan")},
        ):
            with self.assertRaises(ValueError):
                plan_ptp_quintic(READY, TARGET, **bad_kwargs)
        q_nan = READY.copy()
        q_nan[0] = np.nan
        with self.assertRaises(ValueError):
            plan_ptp_quintic(q_nan, TARGET)


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

    def test_infeasible_duration_and_invalid_inputs_rejected(self):
        # Previously duration < 2*sqrt(d/a) produced disc < 0 and a midpoint position jump
        with self.assertRaisesRegex(ValueError, "shorter than minimum feasible"):
            plan_ptp_lspb(READY, TARGET, max_vel=0.5, max_acc=1.0, duration=0.1, dt=0.002)
        for bad_kwargs in (
            {"duration": 0.0},
            {"max_vel": -1.0},
            {"max_acc": 0.0},
            {"dt": -0.002},
        ):
            with self.assertRaises(ValueError):
                plan_ptp_lspb(READY, TARGET, **bad_kwargs)


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

    def test_cartesian_line_compound_rotation_and_non_mutating_quats(self):
        model = mujoco.MjModel.from_xml_path(str(XARM6_XML))
        data = mujoco.MjData(model)
        p0, quat0 = fk_pose(model, data, "attachment_site", READY)
        quat1 = _world_compound_rotate(quat0, rot_z=0.45, rot_x=0.25)
        p1 = p0 + np.array([0.06, 0.07, 0.04])

        # Pass scaled copies to verify caller arrays are never mutated in-place
        quat0_in = quat0 * 2.0
        quat1_in = quat1 * 0.5
        quat0_snap = quat0_in.copy()
        quat1_snap = quat1_in.copy()

        traj, ok = plan_cartesian_line(
            p0,
            quat0_in,
            p1,
            quat1_in,
            model,
            site_name="attachment_site",
            max_lin_vel=0.12,
            max_lin_acc=0.6,
            max_rot_vel=0.8,
            q_init=READY.copy(),
            dt=0.005,
        )
        np.testing.assert_array_equal(quat0_in, quat0_snap)
        np.testing.assert_array_equal(quat1_in, quat1_snap)
        self.assertTrue(ok, "Compound-rotation Cartesian line IK converged continuously")

        p_end, quat_end = fk_pose(model, data, "attachment_site", traj.q[-1])
        pos_err_m = float(np.linalg.norm(p_end - p1))
        rot_err_rad = float(np.linalg.norm(quat_error(quat_end, quat1)))
        self.assertLess(pos_err_m, 5e-4)
        self.assertLess(rot_err_rad, 2e-3)

        # Straight-line lateral deviation across intermediate waypoints
        line_dir = (p1 - p0) / np.linalg.norm(p1 - p0)
        for q_step in traj.q[::10]:
            p_step, _ = fk_pose(model, data, "attachment_site", q_step)
            rel = p_step - p0
            lateral = np.linalg.norm(rel - np.dot(rel, line_dir) * line_dir)
            self.assertLess(lateral, 5e-4)

    def test_cartesian_line_validates_inputs(self):
        model = mujoco.MjModel.from_xml_path(str(XARM6_XML))
        p0 = np.array([0.3, 0.0, 0.3])
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        for kwargs in (
            {"dt": 0.0},
            {"max_lin_vel": 0.0},
            {"max_lin_acc": -1.0},
            {"max_rot_vel": 0.0},
        ):
            with self.assertRaises(ValueError):
                plan_cartesian_line(p0, q0, p0, q0, model, q_init=READY, **kwargs)
        with self.assertRaises(ValueError):
            plan_cartesian_line(p0, np.zeros(4), p0, q0, model, q_init=READY)
        with self.assertRaises(ValueError):
            plan_cartesian_line(np.array([np.nan, 0.0, 0.0]), q0, p0, q0, model, q_init=READY)


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

    def test_gripper_and_custom_site_cartesian_trajectory_with_rotation(self):
        with SimRobot(XARM6_GRIPPER_SCENE, viewer=False) as robot:
            self.assertEqual(robot.model.nq, 12)
            self.assertEqual(robot.n_arm, 6)
            robot.move_joints(READY, duration=1.0)
            p0, quat0 = robot.get_tcp_pose()
            p_target = p0 + np.array([0.0, 0.06, 0.04])
            quat_target = _world_compound_rotate(quat0, rot_z=0.35, rot_x=0.15)

            traj, ok = robot.plan_cartesian_trajectory(
                p_target, quat_target, max_lin_vel=0.10, max_lin_acc=0.5, max_rot_vel=0.5
            )
            self.assertTrue(ok)
            self.assertEqual(traj.n_joints, 6)
            robot.execute_trajectory(traj, wait=True)

            p_final, quat_final = robot.get_tcp_pose()
            self.assertLess(np.linalg.norm(p_final - p_target) * 1000.0, 2.0)
            self.assertLess(np.linalg.norm(quat_error(quat_final, quat_target)), 0.01)

        with SimRobot(XARM6_FT_GRIPPER_SCENE, viewer=False) as robot:
            robot.move_joints(READY, duration=1.0)
            p_ft0, quat_ft0 = fk_pose(robot.model, robot.eval_data, "ft_sensor_site", robot.data.qpos.copy())
            p_tcp0, _ = robot.get_tcp_pose()
            self.assertGreater(np.linalg.norm(p_ft0 - p_tcp0), 0.05)
            p_ft_goal = p_ft0 + np.array([0.03, -0.04, 0.02])
            traj_ft, ok_ft = robot.plan_cartesian_trajectory(
                p_ft_goal, quat_ft0, max_lin_vel=0.10, site_name="ft_sensor_site"
            )
            self.assertTrue(ok_ft)
            self.assertEqual(traj_ft.n_joints, 6)
            q_end_full = robot.data.qpos.copy()
            q_end_full[:6] = traj_ft.q[-1]
            p_ft_end, _ = fk_pose(robot.model, robot.eval_data, "ft_sensor_site", q_end_full)
            np.testing.assert_allclose(p_ft_end, p_ft_goal, atol=5e-4)


if __name__ == "__main__":
    unittest.main()
