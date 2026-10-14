"""Unit tests for the xArm 6 TCP reaching RL environment (roadmap 0.0.5).

Gymnasium / stable-baselines3 specifics are skipped when those packages are
absent (e.g. the py314 project default), mirroring the trimesh skip pattern.
"""

import importlib.util
import unittest
from pathlib import Path

import mujoco
import numpy as np

from src.ik import solve_ik
from src.rl import XArm6ReachEnv

ROOT = Path(__file__).resolve().parent.parent
_HAS_GYM = importlib.util.find_spec("gymnasium") is not None
_HAS_SB3 = importlib.util.find_spec("stable_baselines3") is not None


def _greedy_action(env: XArm6ReachEnv) -> np.ndarray:
    """Action driving the arm toward the IK solution for the current target."""
    q_ik, _ = solve_ik(env.model, env._ik_data, "attachment_site",
                       env.target_position, None, q_init=env.joint_positions(),
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
        self.assertLess(info["dist"], 0.015)
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


if __name__ == "__main__":
    unittest.main()
