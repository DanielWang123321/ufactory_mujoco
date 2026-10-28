"""Unit and integration tests for Pick-and-Place RL environment, artifacts, and evaluation (Batch 8)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from examples.rl.pick_place.evaluate import (
    acceptance_passed,
    checkpoint_score,
    evaluate_policy,
)
from ufactory_mujoco.config import load_task_config
from ufactory_mujoco.rl import (
    PICK_PLACE_ACTION_CONTRACT,
    PICK_PLACE_NUM_ACTIONS,
    PICK_PLACE_NUM_OBS,
    ArtifactError,
    PickPlaceEnv,
    PickPlaceExpert,
    PickPlaceMetrics,
    XArm6PickPlaceEnv,
    build_pick_place_task_configs,
    load_training_recipe,
    load_validated_ppo,
    validate_checkpoint_artifacts,
)

ROOT = Path(__file__).resolve().parent.parent
RECIPE_PATH = ROOT / "examples" / "rl" / "pick_place" / "recipe.yaml"
RUNTIME_CFG_PATH = ROOT / "configs" / "pick_place" / "xarm6.yaml"
_HAS_GYM = importlib.util.find_spec("gymnasium") is not None
_HAS_SB3 = importlib.util.find_spec("stable_baselines3") is not None


class PickPlaceEnvContractTests(unittest.TestCase):
    def setUp(self):
        self.env = XArm6PickPlaceEnv()

    def tearDown(self):
        self.env.close()

    def test_spaces_shapes_and_dtypes(self):
        self.assertEqual(self.env.observation_space.shape, (PICK_PLACE_NUM_OBS,))
        self.assertEqual(self.env.action_space.shape, (PICK_PLACE_NUM_ACTIONS,))
        self.assertEqual(self.env.observation_space.dtype, np.float32)
        self.assertEqual(self.env.action_space.dtype, np.float32)

    def test_reuses_batch3_scene_and_g2_kinematics(self):
        task_cfg = load_task_config(RUNTIME_CFG_PATH)
        self.assertEqual(self.env.model.nu, 7)
        self.assertEqual(self.env._gripper_spec.name, "gripper_g2")
        obs, info = self.env.reset(seed=5)
        self.assertEqual(obs.shape, (PICK_PLACE_NUM_OBS,))
        np.testing.assert_allclose(self.env.default_spawn_xy, task_cfg.pick_place.cube_xy)
        np.testing.assert_allclose(self.env.target_xy, task_cfg.pick_place.target_xy)
        self.assertAlmostEqual(self.env.measured_gap(), self.env.pre_grasp_gap_m, places=2)
        self.assertFalse(info["success"])

    def test_zero_action_holds_setpoint_and_gap(self):
        self.env.reset(seed=0)
        sp_before = self.env.ee_setpoint.copy()
        gap_before = float(self.env.gap_cmd)
        obs, reward, terminated, truncated, info = self.env.step(np.zeros(PICK_PLACE_NUM_ACTIONS))
        self.assertEqual(obs.shape, (PICK_PLACE_NUM_OBS,))
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        np.testing.assert_allclose(self.env.ee_setpoint, sp_before, atol=1e-12)
        self.assertAlmostEqual(self.env.gap_cmd, gap_before, places=12)
        self.assertEqual(info["steps"], 1)

    def test_expert_completes_pick_and_place_within_strict_thresholds(self):
        expert = PickPlaceExpert(self.env)
        obs, info = self.env.reset(seed=0)
        done = False
        while not done:
            action = expert(obs)
            obs, _reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)
        self.assertTrue(info["success"], f"expert failed: {info}")
        self.assertTrue(info["ever_lifted_clear"])
        self.assertTrue(info["release_confirmed"])
        self.assertTrue(info["touchdown_seen"])
        self.assertLessEqual(info["press_mm"], 1.0)
        self.assertLessEqual(info["release_height_mm"], 3.0)
        self.assertLessEqual(info["max_drag_mm"], 10.0)
        self.assertLessEqual(info["max_tilt_deg"], 15.0)
        self.assertLessEqual(info["target_error_mm"], 20.0)
        self.assertGreaterEqual(info["hold_steps"], 10)

    def test_metrics_detect_press_and_high_release_failures(self):
        metrics = PickPlaceMetrics(
            spawn_xy=(0.35, -0.10),
            target_xy=(0.35, 0.10),
            cube_half_m=0.02,
        )
        metrics.reset((0.35, -0.10), ee_z=0.128)
        for _ in range(3):
            metrics.update(
                np.array([0.35, -0.10, 0.10]),
                np.array([1.0, 0.0, 0.0, 0.0]),
                np.array([5.0, 5.0]),
                np.zeros(3),
                0.04,
                ee_z=0.10,
                table_force=0.0,
                robot_table_force=0.0,
            )
        self.assertTrue(metrics.ever_lifted_clear)
        # Simulate high release at z=0.08 (clearance 60 mm > 3 mm)
        for _ in range(3):
            status = metrics.update(
                np.array([0.35, 0.10, 0.08]),
                np.array([1.0, 0.0, 0.0, 0.0]),
                np.array([0.0, 0.0]),
                np.zeros(3),
                0.06,
                ee_z=0.08,
                table_force=0.0,
                robot_table_force=0.0,
            )
        self.assertTrue(status["high_release"])
        self.assertTrue(metrics.release_confirmed)


class PickPlaceArtifactAndEvaluateTests(unittest.TestCase):
    def test_pick_place_recipe_and_task_config_binding(self):
        task_cfg = load_task_config(RUNTIME_CFG_PATH)
        env_cfg, reward_cfg, train_cfg = build_pick_place_task_configs(
            "xarm6",
            recipe_path=RECIPE_PATH,
            config_path=RUNTIME_CFG_PATH,
        )
        self.assertEqual(env_cfg["task_name"], "pick_place")
        self.assertEqual(env_cfg["robot"], "xarm6")
        self.assertEqual(env_cfg["gripper"], "gripper_g2")
        self.assertEqual(env_cfg["control"], PICK_PLACE_ACTION_CONTRACT)
        self.assertEqual(env_cfg["runtime_config_sha256"], task_cfg.sha256)
        self.assertIn("lift_potential", reward_cfg)
        self.assertEqual(train_cfg["algorithm"], "PPO")

    def test_recipe_rejects_physical_geometry_duplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_recipe = Path(tmp) / "bad_recipe.yaml"
            text = RECIPE_PATH.read_text(encoding="utf-8") + "  cube_xy: [0.35, -0.10]\n"
            # Insert cube_xy under environment section
            text = RECIPE_PATH.read_text(encoding="utf-8").replace(
                "  env_version: 1\n",
                "  env_version: 1\n  cube_xy: [0.35, -0.10]\n",
            )
            bad_recipe.write_text(text, encoding="utf-8")
            with self.assertRaises(ArtifactError):
                load_training_recipe(bad_recipe)

    def test_evaluate_expert_and_zero_policy(self):
        env = PickPlaceEnv()
        try:
            expert_summary, _ = evaluate_policy(env, None, episodes=2, seed=17000)
            self.assertTrue(acceptance_passed(expert_summary))
            self.assertEqual(expert_summary["success_rate"], 1.0)
            self.assertLessEqual(expert_summary["press_mm"]["max"], 1.0)

            zero_summary, _ = evaluate_policy(
                env,
                lambda _obs: np.zeros(PICK_PLACE_NUM_ACTIONS, dtype=np.float32),
                episodes=2,
                seed=17000,
            )
            self.assertEqual(zero_summary["success_rate"], 0.0)
            self.assertEqual(zero_summary["timeout_rate"], 1.0)
            self.assertGreater(checkpoint_score(expert_summary), checkpoint_score(zero_summary))
        finally:
            env.close()

    @unittest.skipUnless(_HAS_GYM, "gymnasium not installed")
    def test_gymnasium_env_checker(self):
        from gymnasium.utils.env_checker import check_env

        env = PickPlaceEnv()
        try:
            check_env(env, skip_render_check=True)
        finally:
            env.close()

    @unittest.skipUnless(_HAS_SB3, "stable-baselines3 not installed")
    def test_trained_pick_place_checkpoint_validates_and_evaluates(self):
        ckpt = ROOT / "build" / "rl" / "pick_place" / "xarm6" / "ppo_pick_place.zip"
        if not ckpt.is_file():
            self.skipTest("trained pick_place checkpoint not present")
        env_cfg, reward_cfg, _ = build_pick_place_task_configs("xarm6", recipe_path=RECIPE_PATH)
        model, artifact, manifest = load_validated_ppo(
            ckpt,
            device="cpu",
            expected_task="pick_place",
            expected_robot="xarm6",
            expected_runtime_config_sha256=env_cfg["runtime_config_sha256"],
            expected_runtime_env=env_cfg,
        )
        self.assertEqual(manifest.action_contract, PICK_PLACE_ACTION_CONTRACT)
        self.assertEqual(artifact["task"], "pick_place")
        env = PickPlaceEnv(reward_scales=reward_cfg)
        try:
            summary, _ = evaluate_policy(
                env,
                lambda obs: model.predict(obs, deterministic=True)[0],
                episodes=3,
                seed=17000,
            )
            self.assertTrue(acceptance_passed(summary), f"trained checkpoint failed eval: {summary}")
        finally:
            env.close()

    @unittest.skipUnless(_HAS_SB3, "stable-baselines3 not installed")
    def test_train_pick_place_smoke_without_warmstart(self):
        from scripts.train_pick_place_rl import main as train_main

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "pick_place_smoke"
            train_main(
                [
                    "--total-steps",
                    "64",
                    "--n-envs",
                    "1",
                    "--n-steps",
                    "32",
                    "--batch-size",
                    "32",
                    "--n-epochs",
                    "1",
                    "--warmstart-episodes",
                    "0",
                    "--eval-episodes",
                    "1",
                    "--out-dir",
                    str(out_dir),
                ]
            )
            ckpt = out_dir / "ppo_pick_place.zip"
            self.assertTrue(ckpt.is_file())
            self.assertTrue((out_dir / "config.yaml").is_file())
            self.assertTrue((out_dir / "ppo_pick_place.checkpoint_manifest.json").is_file())
            self.assertTrue((out_dir / "provenance.json").is_file())
            self.assertTrue((out_dir / "metrics.json").is_file())
            env_cfg, _, _ = build_pick_place_task_configs("xarm6", recipe_path=RECIPE_PATH)
            validate_checkpoint_artifacts(
                ckpt,
                expected_task="pick_place",
                expected_robot="xarm6",
                expected_runtime_config_sha256=env_cfg["runtime_config_sha256"],
                expected_runtime_env=env_cfg,
            )


if __name__ == "__main__":
    unittest.main()
