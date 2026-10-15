"""Unit tests for the TCP reaching RL environment (roadmap 0.0.5, 0.0.7).

Covers the full UFACTORY lineup (lite6 / xarm5 / xarm6 / xarm7 / uf850).
Gymnasium / stable-baselines3 specifics are skipped when those packages are
absent (e.g. the py314 project default), mirroring the trimesh skip pattern.
"""

import importlib.util
import unittest
from pathlib import Path

import mujoco
import numpy as np

from src.ik import solve_ik
from src.rl import ROBOT_PROFILES, ReachEnv, XArm6ReachEnv

ROOT = Path(__file__).resolve().parent.parent
_HAS_GYM = importlib.util.find_spec("gymnasium") is not None
_HAS_SB3 = importlib.util.find_spec("stable_baselines3") is not None


def _greedy_action(env: ReachEnv) -> np.ndarray:
    """Action driving the arm toward the IK solution for the current target.

    When close (8 cm), the IK target is extended 2x past the current TCP
    (ILC-style overcommand) to cancel the position servo's gravity droop
    (~10 mrad / ~8 mm steady state), which would otherwise stall above the
    5 mm success standard (ufactory_genesis).
    """
    target = env.target_position
    err = target - env.tcp_position()
    ik_target = target + 2.0 * err if np.linalg.norm(err) < 0.08 else target
    q_ik, _ = solve_ik(env.model, env._ik_data, "attachment_site",
                       ik_target, None, q_init=env.joint_positions(),
                       pos_tol=1e-3)
    q = env.joint_positions()
    return np.clip((q_ik - q) / env._delta, -1.0, 1.0)


def _max_penetration(env: XArm6ReachEnv) -> float:
    data = env.data
    return max((0.0, *[float(-data.contact[i].dist) for i in range(data.ncon)]))


class ReachEnvContractTests(unittest.TestCase):
    def setUp(self):
        self.env = XArm6ReachEnv()

    def tearDown(self):
        self.env.close()

    def test_spaces_shapes_and_dtypes(self):
        self.assertEqual(self.env.observation_space.shape, (21,))
        self.assertEqual(self.env.action_space.shape, (6,))
        self.assertEqual(self.env.observation_space.dtype, np.float32)
        self.assertEqual(self.env.action_space.dtype, np.float32)

    def test_reset_seed_reproducible(self):
        obs_a, info_a = self.env.reset(seed=123)
        obs_b, info_b = self.env.reset(seed=123)
        np.testing.assert_array_equal(obs_a, obs_b)
        np.testing.assert_array_equal(info_a["target"], info_b["target"])
        obs_c, info_c = self.env.reset(seed=456)
        self.assertFalse(np.allclose(info_a["target"], info_c["target"]))

    def test_obs_relative_vector_matches_fk(self):
        obs, info = self.env.reset(seed=7)
        np.testing.assert_allclose(
            obs[:3], self.env.tcp_position() - info["target"], atol=1e-9)
        np.testing.assert_allclose(
            self.env.data.mocap_pos[self.env._mocap_id], info["target"], atol=1e-12)

    def test_step_returns_standard_tuple(self):
        self.env.reset(seed=0)
        out = self.env.step(np.zeros(6))
        self.assertEqual(len(out), 5)
        obs, reward, terminated, truncated, info = out
        self.assertEqual(obs.shape, (21,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("dist", info)

    def test_random_rollout_stays_finite_and_within_limits(self):
        self.env.reset(seed=42)
        rng = np.random.default_rng(42)
        for _ in range(1000):
            obs, reward, terminated, truncated, _ = self.env.step(
                rng.uniform(-3.0, 3.0, 6))  # deliberately out of range
            self.assertTrue(np.all(np.isfinite(obs)))
            self.assertTrue(np.isfinite(reward))
            q = self.env.joint_positions()
            self.assertTrue(np.all(q >= self.env._q_lo - 1e-9))
            self.assertTrue(np.all(q <= self.env._q_hi + 1e-9))
            if terminated or truncated:
                self.env.reset(seed=int(rng.integers(1 << 31)))
        self.assertTrue(np.all(np.isfinite(self.env.data.qpos)))

    def test_commands_clamp_to_joint_limits(self):
        env = XArm6ReachEnv(q0_noise=0.0, validate_targets=False)
        try:
            env.reset(seed=0, options={"target": [0.35, 0.0, 0.20]})
            q_before = env.joint_positions()
            env.step(np.full(6, 5.0))
            # action is clipped to [-1, 1] and the position command to joint limits
            np.testing.assert_allclose(
                env.data.ctrl[env._act_ids],
                np.clip(q_before + env._delta, env._q_lo, env._q_hi), atol=1e-12)
            for _ in range(20):
                env.step(np.full(6, 5.0))
            q = env.joint_positions()
            self.assertTrue(np.all(q <= env._q_hi + 1e-9))
            self.assertTrue(np.all(q >= env._q_lo - 1e-9))
        finally:
            env.close()

    def test_fixed_target_option(self):
        target = np.array([0.32, -0.10, 0.18])
        _, info = self.env.reset(seed=1, options={"target": target})
        np.testing.assert_allclose(info["target"], target)

    def test_unreachable_target_rejected(self):
        with self.assertRaises(ValueError):
            self.env.reset(seed=1, options={"target": [1.5, 0.0, 0.2]})


class ReachEnvRewardTests(unittest.TestCase):
    def setUp(self):
        self.env = XArm6ReachEnv(q0_noise=0.0, max_steps=200)
        self.target = np.array([0.35, 0.15, 0.25])

    def tearDown(self):
        self.env.close()

    def test_reward_positive_toward_target_negative_away(self):
        self.env.reset(seed=0, options={"target": self.target})
        greedy = _greedy_action(self.env)
        _, reward_toward, *_ = self.env.step(greedy)

        self.env.reset(seed=0, options={"target": self.target})
        _, reward_away, *_ = self.env.step(-greedy)
        self.assertGreater(reward_toward, 0.0)
        self.assertLess(reward_away, 0.0)
        self.assertGreater(reward_toward, reward_away)

    def test_ik_greedy_policy_reaches_target(self):
        _, info = self.env.reset(seed=3, options={"target": self.target})
        start_dist = info["dist"]
        self.assertGreater(start_dist, 0.05)
        for _ in range(self.env._max_steps):
            obs, _, terminated, truncated, info = self.env.step(_greedy_action(self.env))
            if terminated or truncated:
                break
        self.assertTrue(info["success"], f"final dist {info['dist']:.4f} m")
        self.assertLess(info["dist"], 0.005)   # ufactory_genesis success standard
        self.assertLess(info["dist"], start_dist)
        self.assertLessEqual(_max_penetration(self.env), 1e-3)

    def test_success_terminates_episode(self):
        self.env.reset(seed=0, options={"target": self.target})
        for _ in range(self.env._max_steps):
            _, _, terminated, truncated, info = self.env.step(_greedy_action(self.env))
            if terminated:
                break
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["success"])

    def test_timeout_truncates_episode(self):
        env = XArm6ReachEnv(q0_noise=0.0, max_steps=5, validate_targets=False)
        try:
            env.reset(seed=0, options={"target": [0.4, 0.2, 0.3]})
            for _ in range(4):
                _, _, terminated, truncated, _ = env.step(np.zeros(6))
                self.assertFalse(truncated)
            _, _, terminated, truncated, info = env.step(np.zeros(6))
            self.assertFalse(info["success"] or terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["steps"], 5)
        finally:
            env.close()


@unittest.skipUnless(_HAS_GYM, "gymnasium not installed")
class ReachEnvGymTests(unittest.TestCase):
    def test_gymnasium_env_checker(self):
        from gymnasium.utils.env_checker import check_env
        env = XArm6ReachEnv()
        try:
            check_env(env, skip_render_check=True)
        finally:
            env.close()

    def test_registered_spaces_are_gymnasium_boxes(self):
        from gymnasium import spaces
        env = XArm6ReachEnv()
        try:
            self.assertIsInstance(env.observation_space, spaces.Box)
            self.assertIsInstance(env.action_space, spaces.Box)
        finally:
            env.close()


@unittest.skipUnless(_HAS_SB3, "stable-baselines3 not installed")
class ReachEnvTrainingSmokeTests(unittest.TestCase):
    def test_ppo_learns_and_predicts(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env

        vec_env = make_vec_env(XArm6ReachEnv, n_envs=1, seed=0)
        try:
            model = PPO("MlpPolicy", vec_env, n_steps=64, batch_size=64,
                        n_epochs=2, verbose=0, seed=0, device="cpu")
            model.learn(total_timesteps=128)
            obs = vec_env.reset()
            action, _ = model.predict(obs, deterministic=True)
            self.assertEqual(np.asarray(action).shape[-1], 6)
        finally:
            vec_env.close()


# ---------------------------------------------------------------------------
# 0.0.7: multi-robot coverage (docs/plan_0.0.7_rl.md §6)

ALL_ROBOTS = ("lite6", "xarm5", "xarm6", "xarm7", "uf850")


class ReachEnvMultiRobotTests(unittest.TestCase):
    """Contract tests parameterized over the full UFACTORY lineup."""

    def test_spaces_match_joint_count(self):
        for robot in ALL_ROBOTS:
            env = ReachEnv(robot=robot)
            try:
                n = ROBOT_PROFILES[robot].n_joints
                self.assertEqual(env.n_joints, n, robot)
                self.assertEqual(env.observation_space.shape, (3 + 3 * n,), robot)
                self.assertEqual(env.action_space.shape, (n,), robot)
            finally:
                env.close()

    def test_reset_and_step_contract(self):
        for robot in ALL_ROBOTS:
            env = ReachEnv(robot=robot)
            try:
                n = env.n_joints
                obs, info = env.reset(seed=3)
                self.assertEqual(obs.shape, (3 + 3 * n,), robot)
                self.assertIn("target", info)
                self.assertGreater(info["dist"], 0.0, robot)
                obs, reward, term, trunc, info = env.step(np.zeros(n))
                self.assertEqual(obs.shape, (3 + 3 * n,), robot)
                self.assertTrue(np.all(np.isfinite(obs)), robot)
                self.assertIsInstance(reward, float)
            finally:
                env.close()

    def test_ik_greedy_teacher_reaches_target_on_every_robot(self):
        """The IK-greedy teacher must reach the target on all five arms."""
        for robot in ALL_ROBOTS:
            env = ReachEnv(robot=robot, q0_noise=0.0, max_steps=300)
            try:
                env.reset(seed=11)
                done, success = False, False
                while not done:
                    _, _, term, trunc, info = env.step(_greedy_action(env))
                    success = bool(info["success"])
                    done = term or trunc
                self.assertTrue(success, f"{robot}: greedy teacher failed, "
                                f"final dist {info['dist']:.4f} m")
            finally:
                env.close()

    def test_lite6_actuator_order_fallback(self):
        """lite6 ships unnamed actuators: ids must resolve by model order."""
        env = ReachEnv(robot="lite6")
        try:
            np.testing.assert_array_equal(env._act_ids, np.arange(6))
        finally:
            env.close()

    def test_named_actuators_resolve_by_name(self):
        for robot in ("xarm5", "xarm6", "xarm7", "uf850"):
            env = ReachEnv(robot=robot)
            try:
                expected = [mujoco.mj_name2id(
                    env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act{i + 1}")
                    for i in range(env.n_joints)]
                np.testing.assert_array_equal(env._act_ids, expected)
            finally:
                env.close()

    def test_unknown_robot_rejected(self):
        with self.assertRaises(ValueError):
            ReachEnv(robot="ur5")


class SceneReachAssetTests(unittest.TestCase):
    def test_scene_reach_assets_load(self):
        for robot in ALL_ROBOTS:
            path = ROOT / "models" / robot / "scene_reach.xml"
            model = mujoco.MjModel.from_xml_path(str(path))
            n = ROBOT_PROFILES[robot].n_joints
            self.assertEqual(model.nu, n, robot)
            self.assertGreaterEqual(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                  "attachment_site"), 0, robot)
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                        "target_mocap")
            self.assertGreaterEqual(body_id, 0, robot)
            self.assertGreaterEqual(int(model.body("target_mocap").mocapid[0]), 0, robot)


class RobotProfileCalibrationTests(unittest.TestCase):
    """Anti-rot: frozen ROBOT_PROFILES must stay valid for each scene."""

    def test_target_box_covered_by_radial_band(self):
        for robot in ALL_ROBOTS:
            p = ROBOT_PROFILES[robot]
            low = np.array(p.target_low)
            high = np.array(p.target_high)
            shoulder = np.array(p.shoulder)
            r_min = float(np.linalg.norm(shoulder - np.clip(shoulder, low, high)))
            corners = np.array([[x, y, z]
                                for x in (low[0], high[0])
                                for y in (low[1], high[1])
                                for z in (low[2], high[2])])
            r_max = float(np.max(np.linalg.norm(corners - shoulder, axis=1)))
            self.assertLessEqual(p.reach_min, r_min + 1e-9, robot)
            self.assertGreaterEqual(p.reach_max, r_max - 1e-9, robot)

    def test_target_box_ik_reachability(self):
        """>= 90% of uniform box samples pass the env's reachability IK."""
        rng = np.random.default_rng(7)
        for robot in ALL_ROBOTS:
            p = ROBOT_PROFILES[robot]
            model = mujoco.MjModel.from_xml_path(
                str(ROOT / "models" / robot / "scene_reach.xml"))
            data = mujoco.MjData(model)
            low = np.array(p.target_low)
            high = np.array(p.target_high)
            targets = rng.uniform(low, high, size=(50, 3))
            ok = 0
            for t in targets:
                data.qpos[:] = 0.0
                _, converged = solve_ik(model, data, "attachment_site", t, None,
                                        q_init=np.zeros(model.nq),
                                        pos_tol=2e-3, max_iter=80)
                ok += int(bool(converged))
            self.assertGreaterEqual(ok / len(targets), 0.9, robot)


if __name__ == "__main__":
    unittest.main()
