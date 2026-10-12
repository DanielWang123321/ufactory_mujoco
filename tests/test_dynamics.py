"""Unit tests for src/dynamics.py (offline, no real robot required)."""

import unittest
from pathlib import Path

import mujoco
import numpy as np

from src.dynamics import ArmDynamics, attach_payload

ROOT = Path(__file__).resolve().parent.parent
XARM6 = ROOT / "models" / "xarm6" / "xarm6.xml"
READY = np.array([0, -0.8, -1.2, 1.0, 0.3, 0])
PAYLOAD = (3.0, [0.0, 0.0, 0.021])

# Measured on the real robot 2026-10-10 (static holding torque, sim side):
# ready pose with 3 kg @ [0,0,21] mm -> J2 -5.46 Nm, J3 -21.10 Nm.


class AttachPayloadTest(unittest.TestCase):
    def test_payload_adds_body_and_mass(self):
        base = mujoco.MjModel.from_xml_path(str(XARM6))
        loaded = attach_payload(XARM6, PAYLOAD[0], PAYLOAD[1])
        self.assertEqual(loaded.nbody, base.nbody + 1)
        self.assertAlmostEqual(float(np.sum(loaded.body_mass)) - float(np.sum(base.body_mass)),
                               PAYLOAD[0], places=6)
        body_id = mujoco.mj_name2id(loaded, mujoco.mjtObj.mjOBJ_BODY, "tcp_payload")
        self.assertGreaterEqual(body_id, 0)

    def test_payload_com_position_in_flange_frame(self):
        loaded = attach_payload(XARM6, PAYLOAD[0], PAYLOAD[1])
        body_id = mujoco.mj_name2id(loaded, mujoco.mjtObj.mjOBJ_BODY, "tcp_payload")
        np.testing.assert_allclose(loaded.body_pos[body_id], PAYLOAD[1], atol=1e-9)


class ArmDynamicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dyn = ArmDynamics(XARM6)
        cls.dyn_pl = ArmDynamics(XARM6, payload=PAYLOAD)

    def test_mass_matrix_symmetric_positive_definite(self):
        for q in (np.zeros(6), READY, np.array([0.5, -1.0, -2.0, 1.5, 2.0, -1.0])):
            M = self.dyn.mass_matrix(q)
            self.assertEqual(M.shape, (6, 6))
            np.testing.assert_allclose(M, M.T, atol=1e-12)
            self.assertTrue(np.all(np.linalg.eigvalsh(M) > 0),
                            f"M not positive definite at {q}")

    def test_mass_matrix_matches_solveM(self):
        q = np.array([0.3, -1.2, -0.8, 0.7, 1.4, 0.2])
        M = self.dyn.mass_matrix(q, full=True)  # leaves qM unfactored
        b = np.arange(1.0, self.dyn.nv + 1.0)
        x_dense = np.linalg.solve(M, b)
        mujoco.mj_factorM(self.dyn.model, self.dyn.data)
        b2 = b.reshape(1, -1)  # binding expects (nvec, nv)
        x_sparse = np.empty_like(b2)
        mujoco.mj_solveM(self.dyn.model, self.dyn.data, x_sparse, b2)
        np.testing.assert_allclose(x_dense, x_sparse.ravel(), atol=1e-10)

    def test_gravity_matches_qfrc_bias_at_rest(self):
        q = np.array([0.2, -0.9, -1.5, 0.4, 1.1, -0.3])
        g = self.dyn.gravity(q)
        self.dyn._set(q)
        mujoco.mj_forward(self.dyn.model, self.dyn.data)
        np.testing.assert_allclose(g, self.dyn.data.qfrc_bias[:6], atol=1e-12)

    def test_gravity_payload_matches_real_robot_measurement(self):
        g = self.dyn_pl.gravity(READY)
        # tolerance 5% around the 2026-10-10 real-robot static comparison
        self.assertAlmostEqual(g[1], -5.46, delta=abs(-5.46) * 0.05)
        self.assertAlmostEqual(g[2], -21.10, delta=abs(-21.10) * 0.05)

    def test_coriolis_zero_at_rest(self):
        q = np.array([0.1, -1.0, -1.8, 0.9, 0.5, 0.0])
        np.testing.assert_allclose(self.dyn.coriolis(q, np.zeros(6)), 0.0, atol=1e-12)

    def test_coriolis_plus_gravity_equals_bias(self):
        q = np.array([0.1, -1.0, -1.8, 0.9, 0.5, 0.0])
        qd = np.array([0.4, -0.6, 0.8, -0.3, 0.9, -0.5])
        self.dyn._set(q, qd)
        mujoco.mj_forward(self.dyn.model, self.dyn.data)
        bias = self.dyn.data.qfrc_bias[:6].copy()
        np.testing.assert_allclose(self.dyn.coriolis(q, qd) + self.dyn.gravity(q),
                                   bias, atol=1e-10)

    def test_inverse_forward_roundtrip(self):
        rng = np.random.default_rng(7)
        lo = np.array([-3.0, -2.0, -3.9, -3.0, -1.6, -3.0])
        hi = np.array([3.0, 2.0, 0.19, 3.0, 3.1, 3.0])
        for _ in range(5):
            q = lo + (hi - lo) * rng.random(6)
            qd = 1.5 * rng.random(6) - 0.75
            qdd = 2.0 * rng.random(6) - 1.0
            tau = self.dyn_pl.inverse_dynamics(q, qd, qdd)
            qdd2 = self.dyn_pl.forward_dynamics(q, qd, tau)
            np.testing.assert_allclose(qdd2, qdd, atol=1e-8)

    def test_inverse_dynamics_linear_in_acceleration(self):
        q = np.array([0.0, -0.8, -1.2, 1.0, 0.3, 0.0])
        qd = np.array([0.2, 0.1, -0.3, 0.1, 0.2, -0.1])
        M = self.dyn.mass_matrix(q)
        tau0 = self.dyn.inverse_dynamics(q, qd, np.zeros(6))
        qdd = np.array([0.5, -0.4, 0.6, -0.2, 0.3, 0.1])
        tau1 = self.dyn.inverse_dynamics(q, qd, qdd)
        np.testing.assert_allclose(tau1 - tau0, M @ qdd, atol=1e-10)

    def test_feedforward_traj_endpoints_equal_gravity(self):
        T, dt = 2.0, 0.002
        t = np.arange(0, T + dt, dt)
        s = 10 * (t / T) ** 3 - 15 * (t / T) ** 4 + 6 * (t / T) ** 5
        q_traj = s[:, None] * READY[None, :]
        tau_ff = self.dyn_pl.feedforward_traj(q_traj, dt)
        self.assertEqual(tau_ff.shape, (len(t), 6))
        # endpoints are rest-to-rest, so tau -> g(q); np.gradient edge
        # one-sided differences leave ~0.13 rad/s^2 accel error there
        g0 = self.dyn_pl.gravity(np.zeros(6))
        g1 = self.dyn_pl.gravity(READY)
        np.testing.assert_allclose(tau_ff[0], g0, atol=0.3)
        np.testing.assert_allclose(tau_ff[-1], g1, atol=0.3)
        # interior point: compare against analytic quintic derivatives
        i = len(t) // 2
        si = t[i] / T
        sd = (30 * si ** 2 - 60 * si ** 3 + 30 * si ** 4) / T
        sdd = (60 * si - 180 * si ** 2 + 120 * si ** 3) / T ** 2
        tau_ref = self.dyn_pl.inverse_dynamics(q_traj[i], sd * READY, sdd * READY)
        np.testing.assert_allclose(tau_ff[i], tau_ref, atol=1e-2)

    def test_feedforward_traj_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            self.dyn.feedforward_traj(np.zeros((10, 4)), 0.002)


if __name__ == "__main__":
    unittest.main()
