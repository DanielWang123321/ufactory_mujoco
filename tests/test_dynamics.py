"""Unit tests for src/dynamics.py (offline, no real robot required)."""

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import mujoco
import numpy as np

from src.dynamics import ArmDynamics, attach_payload

ROOT = Path(__file__).resolve().parent.parent
XARM6 = ROOT / "models" / "xarm6" / "xarm6.xml"
READY = np.array([0, -0.8, -1.2, 1.0, 0.3, 0])
PAYLOAD = (3.0, [0.0, 0.0, 0.021])

PAYLOAD_FIXTURE = """<mujoco>
  <compiler angle="degree"/>
  <worldbody>
    <body name="arm" pos="0.1 -0.2 0.3" euler="20 -30 15">
      <joint axis="0 1 0"/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.03 0.04 0.05"/>
      <body name="tool" pos="0.1 0.2 -0.1" euler="10 15 30">
        <joint axis="0 0 1"/>
        <inertial pos="0 0 0" mass="2" diaginertia="0.02 0.03 0.04"/>
        <site name="tcp" pos="0.2 -0.1 0.05" {orientation}/>
      </body>
    </body>
  </worldbody>
</mujoco>"""

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

    def test_payload_com_relative_to_translated_and_rotated_site(self):
        com = np.array([0.03, -0.02, 0.07])
        orientations = ("", 'quat="0.7071067811865476 0 0 0.7071067811865476"',
                        'euler="25 -35 60"', 'xyaxes="0 1 0 -1 0 0"')
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "arm.xml"
            for orientation in orientations:
                with self.subTest(orientation=orientation):
                    path.write_text(PAYLOAD_FIXTURE.format(orientation=orientation))
                    base = mujoco.MjModel.from_xml_path(str(path))
                    loaded = attach_payload(path, 2.5, com, site="tcp")
                    for q in ([0, 0], [0.4, -0.7]):
                        data = mujoco.MjData(loaded)
                        data.qpos[:] = q
                        mujoco.mj_forward(loaded, data)
                        tcp = data.site("tcp")
                        expected = tcp.xpos + tcp.xmat.reshape(3, 3) @ com
                        np.testing.assert_allclose(data.body("tcp_payload").xipos,
                                                   expected, rtol=0, atol=1e-12)
                    self.assertEqual(loaded.nbody, base.nbody + 1)

    def test_payload_handles_includes_frames_and_static_fusion(self):
        com = np.array([0.02, -0.01, 0.03])
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "scene.xml"
            child = Path(tmp) / "included.xml"
            child.write_text('''<mujoco><worldbody><body name="arm">
              <joint axis="0 1 0"/>
              <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
              <body name="fixed_tool" pos="0.1 0.2 0.3">
                <frame pos="0.2 -0.1 0" euler="0 0 90">
                  <site euler="0 0 90" pos="0.1 0 0" name="tcp"></site>
                </frame>
              </body></body></worldbody></mujoco>''')
            for fused in ("false", "true"):
                with self.subTest(fusestatic=fused):
                    path.write_text(f'<mujoco><compiler angle="degree" fusestatic="{fused}"/>'
                                    '<include file="included.xml"/></mujoco>')
                    base = mujoco.MjModel.from_xml_path(str(path))
                    loaded = attach_payload(path, 2, com, site="tcp")
                    d0, d1 = mujoco.MjData(base), mujoco.MjData(loaded)
                    d0.qpos[:] = d1.qpos[:] = 0.4
                    mujoco.mj_forward(base, d0)
                    mujoco.mj_forward(loaded, d1)
                    world_com = d0.site("tcp").xpos + d0.site("tcp").xmat.reshape(3, 3) @ com
                    jacp, jacr = np.zeros((3, base.nv)), np.zeros((3, base.nv))
                    mujoco.mj_jac(base, d0, jacp, jacr, world_com,
                                  int(base.site("tcp").bodyid[0]))
                    dyn0, dyn1 = ArmDynamics(base), ArmDynamics(loaded)
                    np.testing.assert_allclose(dyn1.mass_matrix([0.4]) - dyn0.mass_matrix([0.4]),
                                               2 * jacp.T @ jacp, rtol=0, atol=1e-12)
                    np.testing.assert_allclose(dyn1.gravity([0.4]) - dyn0.gravity([0.4]),
                                               -2 * jacp.T @ base.opt.gravity, rtol=0, atol=1e-12)

    def test_payload_com_on_combined_scene_without_chdir(self):
        cwd = Path.cwd()
        for name in ("xarm6_with_gripper", "xarm6_with_bio_gripper", "lite6_with_gripper"):
            with self.subTest(model=name), patch("os.chdir", side_effect=AssertionError("chdir")):
                path = ROOT / "models" / name / "scene.xml"
                loaded = attach_payload(path, *PAYLOAD)
                data = mujoco.MjData(loaded)
                data.qpos[:] = loaded.key("ready").qpos
                mujoco.mj_forward(loaded, data)
                tcp = data.site("attachment_site")
                expected = tcp.xpos + tcp.xmat.reshape(3, 3) @ PAYLOAD[1]
                np.testing.assert_allclose(data.body("tcp_payload").xipos,
                                           expected, rtol=0, atol=1e-12)
        self.assertEqual(Path.cwd(), cwd)

    def test_point_mass_inertia_and_gravity_match_analytic_jacobian(self):
        mass, com = 2.5, [0.03, -0.02, 0.07]
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "arm.xml"
            path.write_text(PAYLOAD_FIXTURE.format(orientation='euler="25 -35 60"'))
            base = mujoco.MjModel.from_xml_path(str(path))
            loaded = attach_payload(path, mass, com, site="tcp")
            np.testing.assert_array_equal(loaded.body("tcp_payload").inertia, [0, 0, 0])
            dyn0, dyn1 = ArmDynamics(base), ArmDynamics(loaded)
            for q in ([0.2, -0.3], [0.7, 0.4]):
                data = mujoco.MjData(base)
                data.qpos[:] = q
                mujoco.mj_forward(base, data)
                tcp = data.site("tcp")
                position = tcp.xpos + tcp.xmat.reshape(3, 3) @ com
                jacp, jacr = np.zeros((3, base.nv)), np.zeros((3, base.nv))
                mujoco.mj_jac(base, data, jacp, jacr, position, int(base.site("tcp").bodyid[0]))
                np.testing.assert_allclose(dyn1.mass_matrix(q) - dyn0.mass_matrix(q),
                                           mass * jacp.T @ jacp, rtol=0, atol=1e-12)
                np.testing.assert_allclose(dyn1.gravity(q) - dyn0.gravity(q),
                                           -mass * jacp.T @ base.opt.gravity, rtol=0, atol=1e-12)

    def test_zero_payload_leaves_model_and_dynamics_unchanged(self):
        for name in ("xarm6", "xarm6_with_gripper", "xarm6_with_bio_gripper", "lite6_with_gripper"):
            with self.subTest(model=name):
                path = ROOT / "models" / name / "scene.xml"
                base = ArmDynamics(path)
                zero = ArmDynamics(path, payload=(0, [0.03, -0.02, 0.07]))
                self.assertEqual(base.model.nbody, zero.model.nbody)
                self.assertEqual(mujoco.mj_name2id(zero.model, mujoco.mjtObj.mjOBJ_BODY,
                                                 "tcp_payload"), -1)
                q = base.model.key("ready").qpos
                np.testing.assert_array_equal(base.mass_matrix(q, full=True), zero.mass_matrix(q, full=True))
                np.testing.assert_array_equal(base.gravity(q, full=True), zero.gravity(q, full=True))

    def test_payload_rejects_invalid_mass_com_and_missing_site(self):
        for mass in (-1, np.nan, np.inf, -np.inf):
            with self.subTest(mass=mass), self.assertRaisesRegex(ValueError, "mass"):
                attach_payload(XARM6, mass, [0, 0, 0])
        for com in ([], [0, 0], [0, 0, 0, 0], [[0, 0, 0]], [np.nan, 0, 0], [0, np.inf, 0]):
            with self.subTest(com=com), self.assertRaisesRegex(ValueError, "COM"):
                attach_payload(XARM6, 1, com)
        for mass in (0, 1):
            with self.assertRaisesRegex(ValueError, "site.*not found"):
                attach_payload(XARM6, mass, [0, 0, 0], site="missing")


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


class CombinedDynamicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        names = ("xarm5", "xarm6", "xarm7", "uf850", "lite6", "xarm6_with_gripper",
                 "xarm6_with_bio_gripper", "lite6_with_gripper", "lite6_with_vacuum_gripper")
        cls.models = {name: ArmDynamics(ROOT / "models" / name / f"{name}.xml")
                      for name in names}

    @staticmethod
    def position(dyn, opening):
        q = dyn.model.key("ready").qpos.copy()
        for j in range(dyn.model.njnt):
            adr = int(dyn.model.jnt_qposadr[j])
            if adr >= dyn.n_arm:
                lo, hi = dyn.model.jnt_range[j]
                q[adr] = opening * (hi if abs(hi) > abs(lo) else lo)
        return q

    def test_arm_roundtrip_at_different_gripper_openings(self):
        rng = np.random.default_rng(17)
        for name, dyn in self.models.items():
            self.assertEqual(dyn.n_arm, 5 if name == "xarm5" else 7 if name == "xarm7" else 6)
            for opening in (0.0, 0.45, 1.0):
                with self.subTest(model=name, opening=opening):
                    q = self.position(dyn, opening)
                    qd, qdd = rng.uniform(-0.6, 0.6, (2, dyn.n_arm))
                    tau = dyn.inverse_dynamics(q, qd, qdd)
                    result = dyn.forward_dynamics(q, qd, tau)
                    self.assertEqual(result.shape, (dyn.n_arm,))
                    np.testing.assert_allclose(result, qdd, rtol=0, atol=1e-9)
                    result_full = dyn.forward_dynamics(q, qd, tau, full=True)
                    np.testing.assert_allclose(result_full[:dyn.n_arm], qdd, rtol=0, atol=1e-9)
                    np.testing.assert_array_equal(result_full[dyn.n_arm:], 0)

    def test_full_roundtrip_and_output_slicing(self):
        rng = np.random.default_rng(27)
        for name, dyn in self.models.items():
            for opening in (0, 0.6, 1):
                with self.subTest(model=name, opening=opening):
                    q = self.position(dyn, opening)
                    qd, qdd = rng.uniform(-0.5, 0.5, (2, dyn.nv))
                    tau = dyn.inverse_dynamics(q, qd, qdd, full=True)
                    self.assertEqual(tau.shape, (dyn.nv,))
                    np.testing.assert_allclose(dyn.forward_dynamics(q, qd, tau, full=True),
                                               qdd, rtol=0, atol=1e-9)
                    np.testing.assert_allclose(dyn.forward_dynamics(q, qd, tau),
                                               qdd[:dyn.n_arm], rtol=0, atol=1e-9)

    def test_arm_torque_solves_arm_block_with_zero_gripper_acceleration(self):
        for name, dyn in self.models.items():
            with self.subTest(model=name):
                q = self.position(dyn, 0.45)
                qd = np.linspace(-0.2, 0.3, dyn.n_arm)
                tau = np.linspace(-2, 3, dyn.n_arm)
                mass = dyn.mass_matrix(q)
                dyn._set(q, qd)
                mujoco.mj_forward(dyn.model, dyn.data)
                expected = np.linalg.solve(mass, tau - dyn.data.qfrc_bias[:dyn.n_arm]
                                           + dyn.data.qfrc_passive[:dyn.n_arm])
                np.testing.assert_allclose(dyn.forward_dynamics(q, qd, tau),
                                           expected, rtol=0, atol=1e-9)

    def test_inverse_dynamics_requires_full_for_gripper_acceleration(self):
        for name, dyn in self.models.items():
            if dyn.nv == dyn.n_arm:
                continue
            with self.subTest(model=name):
                q = self.position(dyn, 0.5)
                qdd = np.zeros(dyn.nv)
                qdd[dyn.n_arm] = 0.1
                with self.assertRaisesRegex(ValueError, "full=True"):
                    dyn.inverse_dynamics(q, np.zeros(dyn.n_arm), qdd)
                tau = dyn.inverse_dynamics(q, np.zeros(dyn.n_arm), qdd, full=True)
                np.testing.assert_allclose(dyn.forward_dynamics(q, np.zeros(dyn.n_arm), tau, full=True),
                                           qdd, rtol=0, atol=1e-9)
                qdd[dyn.n_arm] = 0
                self.assertEqual(dyn.inverse_dynamics(q, np.zeros(dyn.n_arm), qdd).shape,
                                 (dyn.n_arm,))

    def test_arm_only_state_pads_gripper_with_zeros(self):
        for name, dyn in self.models.items():
            with self.subTest(model=name):
                q = self.position(dyn, 0)
                qd, qdd = np.ones(dyn.n_arm) * 0.1, np.ones(dyn.n_arm) * 0.2
                np.testing.assert_array_equal(dyn.inverse_dynamics(q[:dyn.n_arm], qd, qdd),
                                              dyn.inverse_dynamics(q, qd, qdd))
                np.testing.assert_array_equal(dyn.data.qvel[dyn.n_arm:], 0)

    def test_feedforward_with_fixed_and_moving_gripper(self):
        for name in ("xarm6_with_gripper", "xarm6_with_bio_gripper", "lite6_with_gripper"):
            dyn = self.models[name]
            with self.subTest(model=name):
                q = self.position(dyn, 0.45)
                trajectory = np.tile(q, (5, 1))
                trajectory[:, 0] += 0.001 * np.arange(5) ** 2
                tau = dyn.feedforward_traj(trajectory, 0.01)
                self.assertEqual(tau.shape, (5, dyn.n_arm))
                trajectory[:, dyn.n_arm] += 0.0001 * np.arange(5) ** 2
                with self.assertRaisesRegex(ValueError, "full=True"):
                    dyn.feedforward_traj(trajectory, 0.01)
                self.assertEqual(dyn.feedforward_traj(trajectory, 0.01, full=True).shape,
                                 (5, dyn.nv))


if __name__ == "__main__":
    unittest.main()
