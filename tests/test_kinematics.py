"""FK / geometric Jacobian / velocity-level IK regressions."""

import unittest
from pathlib import Path

import mujoco
import numpy as np

from src import SimRobot
from src.ik import solve_ik
from src.kinematics import (
    condition_number,
    fk_pose,
    manipulability,
    pose_error,
    pose_jacobian,
    quat_error,
    site_jacobian,
    solve_joint_velocity,
)

_Q0 = np.array([0.2, -0.8, -1.2, 0.5, 0.3, -0.2])
_PERTURB = 1e-5


class KinematicsTests(unittest.TestCase):
    def setUp(self):
        self.robot = SimRobot(viewer=False)
        self.model, self.data = self.robot.model, self.robot.eval_data
        self.site = self.model.site(self.robot._site).name

    def tearDown(self):
        self.robot.close()

    # -- helpers ---------------------------------------------------------
    def _fk(self, q):
        return fk_pose(self.model, self.data, self.site, q)

    def _fd_jacobian(self, q, h=_PERTURB):
        """Central-difference Jacobian via mj_integratePos (quaternion-joint safe)."""
        cols_p, cols_r = [], []
        for i in range(self.model.nv):
            dv = np.zeros(self.model.nv)
            dv[i] = h
            qp, qm = q.copy(), q.copy()
            mujoco.mj_integratePos(self.model, qp, dv, 1.0)
            mujoco.mj_integratePos(self.model, qm, dv, -1.0)
            pos_p, quat_p = self._fk(qp)
            pos_m, quat_m = self._fk(qm)
            cols_p.append((pos_p - pos_m) / (2 * h))
            cols_r.append(quat_error(quat_m, quat_p) / (2 * h))
        return np.column_stack(cols_p), np.column_stack(cols_r)

    # -- FK --------------------------------------------------------------
    def test_fk_pose_matches_forward_kinematics(self):
        pos, quat = self._fk(_Q0)
        quat2 = np.empty(4)
        mujoco.mju_mat2Quat(quat2, self.data.site_xmat[self.robot._site])
        np.testing.assert_allclose(pos, self.data.site_xpos[self.robot._site], atol=1e-12)
        np.testing.assert_allclose(quat, quat2, atol=1e-12)
        # same result as SimRobot.get_tcp_pose when fed the live qpos
        live_pos, live_quat = self.robot.get_tcp_pose()
        pos3, quat3 = self._fk(self.robot.data.qpos.copy())
        np.testing.assert_allclose(pos3, live_pos, atol=1e-12)
        np.testing.assert_allclose(quat3, live_quat, atol=1e-12)

    def test_fk_pose_rejects_wrong_q_shape(self):
        with self.assertRaises(ValueError):
            self._fk(_Q0[:-1])

    # -- Jacobian --------------------------------------------------------
    def test_position_jacobian_matches_finite_difference(self):
        jacp, _ = site_jacobian(self.model, self.data, self.site, _Q0)
        fd_p, _ = self._fd_jacobian(_Q0)
        np.testing.assert_allclose(jacp, fd_p, rtol=1e-4, atol=1e-5)

    def test_rotational_jacobian_matches_finite_difference(self):
        _, jacr = site_jacobian(self.model, self.data, self.site, _Q0)
        _, fd_r = self._fd_jacobian(_Q0)
        np.testing.assert_allclose(jacr, fd_r, rtol=1e-4, atol=1e-5)

    def test_pose_jacobian_is_stacked_and_simrobot_wrapper_agrees(self):
        J = pose_jacobian(self.model, self.data, self.site, _Q0)
        jacp, jacr = site_jacobian(self.model, self.data, self.site, _Q0)
        np.testing.assert_allclose(J, np.vstack([jacp, jacr]), atol=0)
        np.testing.assert_allclose(self.robot.tcp_jacobian(_Q0), J, atol=0)

    # -- pose error / IK consistency -------------------------------------
    def test_pose_error_vanishes_at_fk_pose(self):
        pos, quat = self._fk(_Q0)
        np.testing.assert_allclose(
            pose_error(pos, quat, pos, quat), np.zeros(6), atol=1e-12)
        # q and -q are the same orientation -> zero rotation error
        np.testing.assert_allclose(quat_error(quat, -quat), np.zeros(3), atol=1e-12)

    def test_pose_error_zero_after_ik_convergence(self):
        target_pos = np.array([0.45, 0.15, 0.35])
        target_quat = np.array([0.0, 1.0, 0.0, 0.0])
        q, ok = solve_ik(self.model, self.data, self.site,
                         target_pos, target_quat, q_init=_Q0)
        self.assertTrue(ok)
        pos, quat = self._fk(q)
        err = pose_error(pos, quat, target_pos, target_quat)
        self.assertLess(np.linalg.norm(err[:3]), 1e-4)
        self.assertLess(np.linalg.norm(err[3:]), 1e-3)

    # -- velocity-level IK -----------------------------------------------
    def test_solve_joint_velocity_reproduces_twist(self):
        twist = np.array([0.1, -0.05, 0.08, 0.3, -0.2, 0.1])
        qd = solve_joint_velocity(self.model, self.data, self.site, twist,
                                  _Q0, damping=1e-9)
        J = pose_jacobian(self.model, self.data, self.site, _Q0)
        np.testing.assert_allclose(J @ qd, twist, atol=1e-6)

    def test_solve_joint_velocity_is_minimum_norm(self):
        twist = np.array([0.1, -0.05, 0.08, 0.3, -0.2, 0.1])
        qd = solve_joint_velocity(self.model, self.data, self.site, twist,
                                  _Q0, damping=0.0)
        J = pose_jacobian(self.model, self.data, self.site, _Q0)
        qd_lstsq = np.linalg.lstsq(J, twist, rcond=None)[0]
        np.testing.assert_allclose(qd, qd_lstsq, rtol=1e-8, atol=1e-10)

    def test_solve_joint_velocity_validates_inputs(self):
        with self.assertRaises(ValueError):
            solve_joint_velocity(self.model, self.data, self.site,
                                 np.zeros(5), _Q0)
        with self.assertRaises(ValueError):
            solve_joint_velocity(self.model, self.data, self.site,
                                 np.zeros(6), _Q0, damping=-1.0)

    def test_simrobot_velocity_ik_wrapper(self):
        twist = np.array([0.05, 0.02, -0.03, 0.1, 0.0, -0.1])
        qd = self.robot.solve_tcp_velocity(twist, _Q0, damping=1e-9)
        np.testing.assert_allclose(self.robot.tcp_jacobian(_Q0) @ qd,
                                   twist, atol=1e-6)

    # -- singularity metrics ---------------------------------------------
    def test_manipulability_matches_analytic_and_detects_singularity(self):
        J = pose_jacobian(self.model, self.data, self.site, _Q0)
        w = manipulability(self.model, self.data, self.site, _Q0)
        np.testing.assert_allclose(
            w, np.sqrt(np.linalg.det(J @ J.T)), rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(
            self.robot.tcp_manipulability(_Q0), w, atol=0)

        q_sing = np.zeros(6)  # fully stretched: axes align -> true singularity
        w_sing = manipulability(self.model, self.data, self.site, q_sing)
        self.assertLess(w_sing, 1e-9 * w)

    def test_condition_number_matches_numpy(self):
        J = pose_jacobian(self.model, self.data, self.site, _Q0)
        np.testing.assert_allclose(
            condition_number(self.model, self.data, self.site, _Q0),
            np.linalg.cond(J), rtol=1e-9)
        q_sing = np.zeros(6)
        self.assertGreater(
            condition_number(self.model, self.data, self.site, q_sing),
            1e6 * condition_number(self.model, self.data, self.site, _Q0))


class FreeJointModelTests(unittest.TestCase):
    """nq != nv (free-joint scene): full-q FK/Jacobian must still work."""

    def setUp(self):
        scene = (Path(__file__).resolve().parent.parent / "models"
                 / "xarm6_with_gripper" / "scene_pick_place.xml")
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)

    def test_fk_and_jacobian_with_full_qpos(self):
        self.assertGreater(self.model.nq, self.model.nv)
        q = self.model.qpos0.copy()
        pos, quat = fk_pose(self.model, self.data, "attachment_site", q)
        self.assertEqual(pos.shape, (3,))
        self.assertEqual(quat.shape, (4,))
        J = pose_jacobian(self.model, self.data, "attachment_site", q)
        self.assertEqual(J.shape, (6, self.model.nv))
        with self.assertRaises(ValueError):
            fk_pose(self.model, self.data, "attachment_site", q[:-1])


if __name__ == "__main__":
    unittest.main()
