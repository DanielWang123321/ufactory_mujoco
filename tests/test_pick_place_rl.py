"""Unit and integration tests for pure-PPO Pick-and-Place RL environment, artifacts, and evaluation."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from examples.rl.pick_place.evaluate import (
    acceptance_passed,
    checkpoint_score,
    evaluate_policy,
    make_spawns,
)
from ufactory_mujoco.config import load_task_config
from ufactory_mujoco.ik import solve_ik
from ufactory_mujoco.rl import (
    PICK_PLACE_ACTION_CONTRACT,
    PICK_PLACE_NUM_ACTIONS,
    PICK_PLACE_NUM_OBS,
    SPAWN_XY_LOWER,
    SPAWN_XY_UPPER,
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
from ufactory_mujoco.rl.pick_place_env import TCP_SITE
from ufactory_mujoco.task_geometry import GRIP_DOWN_QUAT

ROOT = Path(__file__).resolve().parent.parent
RECIPE_PATH = ROOT / "examples" / "rl" / "pick_place" / "recipe.yaml"
RUNTIME_CFG_PATH = ROOT / "configs" / "pick_place" / "xarm6.yaml"
TRAIN_SCRIPT_PATH = ROOT / "scripts" / "train_pick_place_rl.py"
_HAS_GYM = importlib.util.find_spec("gymnasium") is not None
_HAS_SB3 = importlib.util.find_spec("stable_baselines3") is not None


class PickPlaceEnvContractTests(unittest.TestCase):
    def setUp(self):
        self.env = XArm6PickPlaceEnv()

    def tearDown(self):
        self.env.close()

    def test_spaces_shapes_and_dtypes(self):
        self.assertEqual(PICK_PLACE_NUM_OBS, 70)
        self.assertEqual(self.env.observation_space.shape, (70,))
        self.assertEqual(self.env.action_space.shape, (PICK_PLACE_NUM_ACTIONS,))
        self.assertEqual(self.env.observation_space.dtype, np.float32)
        self.assertEqual(self.env.action_space.dtype, np.float32)

    def test_uses_calibrated_g2_controller_without_legacy_v1_overrides_or_phase(self):
        self.assertFalse(hasattr(self.env, "_configure_env_v1_gripper_model"))
        self.assertFalse(hasattr(self.env, "_phase"))
        self.assertEqual(self.env.model.nu, 7)
        self.assertEqual(self.env._gripper_spec.name, "gripper_g2")
        forcerange = self.env.model.actuator_forcerange[self.env._gripper_act_id]
        self.assertLessEqual(float(np.max(np.abs(forcerange))), 7.01)
        self.assertGreater(float(np.max(np.abs(forcerange))), 1.0)

        task_cfg = load_task_config(RUNTIME_CFG_PATH)
        obs, info = self.env.reset(seed=5)
        self.assertEqual(obs.shape, (PICK_PLACE_NUM_OBS,))
        np.testing.assert_allclose(self.env.default_spawn_xy, task_cfg.pick_place.cube_xy)
        np.testing.assert_allclose(self.env.target_xy, task_cfg.pick_place.target_xy)
        self.assertAlmostEqual(self.env.measured_gap(), self.env.pre_grasp_gap_m, places=2)
        self.assertFalse(info["success"])

    def test_random_spawn_bounds_and_deterministic_make_spawns(self):
        spawns_a = make_spawns(16, seed=17000)
        spawns_b = make_spawns(16, seed=17000)
        np.testing.assert_allclose(spawns_a, spawns_b)
        self.assertEqual(spawns_a.shape, (16, 2))
        self.assertTrue(np.all(spawns_a >= np.asarray(SPAWN_XY_LOWER)))
        self.assertTrue(np.all(spawns_a <= np.asarray(SPAWN_XY_UPPER)))

        sampled = []
        for s in range(12):
            self.env.reset(seed=100 + s)
            xy = self.env.cube_position()[:2]
            self.assertTrue(np.all(xy >= self.env.spawn_xy_lower - 1e-9))
            self.assertTrue(np.all(xy <= self.env.spawn_xy_upper + 1e-9))
            sampled.append(xy)
        sampled_arr = np.asarray(sampled)
        self.assertGreater(float(np.ptp(sampled_arr[:, 0])), 0.03)
        self.assertGreater(float(np.ptp(sampled_arr[:, 1])), 0.05)

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

    def test_pre_lift_and_near_table_safety_guardrails(self):
        self.env.reset(seed=0, options={"cube_xy": [0.30, 0.00]})
        # Drive straight down repeatedly before lift: fingertips must never collide with floor
        for _ in range(45):
            _obs, _rew, _term, _trunc, info = self.env.step(np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32))
        self.assertFalse(info["collision_failed"])
        self.assertEqual(self.env.robot_table_force, 0.0)

    def test_expert_completes_pick_and_place_across_multiple_spawns(self):
        spawns = np.array([[0.30, 0.00], [0.32, 0.16], [0.48, -0.18]], dtype=float)
        self.env.set_evaluation_spawns(spawns)
        try:
            expert = PickPlaceExpert(self.env)
            for idx in range(len(spawns)):
                obs, info = self.env.reset(seed=idx)
                done = False
                while not done:
                    action = expert(obs)
                    obs, _reward, terminated, truncated, info = self.env.step(action)
                    done = bool(terminated or truncated)
                self.assertTrue(info["success"], f"expert failed on spawn {spawns[idx].tolist()}: {info}")
                self.assertTrue(info["ever_lifted_clear"])
                self.assertTrue(info["release_confirmed"])
                self.assertFalse(info["high_release"])
                self.assertFalse(info["collision_failed"])
                self.assertLessEqual(info["press_mm"], 1.0)
                self.assertLessEqual(info["release_height_mm"], 3.0)
                self.assertLessEqual(info["max_drag_mm"], 10.0)
                self.assertLessEqual(info["max_tilt_deg"], 15.0)
                self.assertLessEqual(info["target_error_mm"], 20.0)
                self.assertGreaterEqual(info["hold_steps"], 10)
        finally:
            self.env.set_evaluation_spawns(None)

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


class PickPlaceEnvFastPathTests(unittest.TestCase):
    """Per-step fast paths must match the reference computations bit for bit."""

    def setUp(self):
        self.env = PickPlaceEnv()

    def tearDown(self):
        self.env.close()

    def test_grip_down_ik_matches_solve_ik(self):
        env = self.env
        env.reset(seed=3)
        ref_data = mujoco.MjData(env.model)
        rng = np.random.default_rng(7)
        outcomes = set()
        for i in range(60):
            q_init = env.data.qpos.copy()
            q_init[env._qpos_adr] = env._home_q + rng.normal(0.0, 0.2, env.n_joints)
            if i % 3:
                target = env.home_tcp + rng.normal(0.0, 0.05, 3)
            else:  # mostly unreachable: covers non-convergence and rejected line searches
                target = rng.uniform(env.workspace_lower - 0.3, env.workspace_upper + 0.3)
            q_ref, ok_ref = solve_ik(
                env.model, ref_data, TCP_SITE, target, GRIP_DOWN_QUAT, q_init=q_init,
                pos_tol=1e-3, rot_tol=1e-2, max_iter=20, damping=0.01,
            )
            q_fast, ok_fast = env._ik.solve(target, q_init)
            self.assertEqual(q_fast.tobytes(), q_ref.tobytes())
            self.assertEqual(ok_fast, ok_ref)
            outcomes.add(ok_ref)
        self.assertEqual(outcomes, {True, False})

    def test_cached_kinematics_and_potential_match_fresh_values(self):
        env = self.env
        expert = PickPlaceExpert(env)
        rng = np.random.default_rng(11)
        reused = 0
        for policy in ("random", "expert"):
            for episode in range(2):
                obs, _ = env.reset(seed=200 + episode)
                self.assertIsNone(env._cached_potential)
                self._assert_snapshot_is_fresh()
                done = False
                while not done:
                    if env._cached_potential is not None:
                        self.assertEqual(env._cached_potential.hex(), env._potential().hex())
                        reused += 1
                    action = expert(obs) if policy == "expert" else rng.uniform(-1.0, 1.0, PICK_PLACE_NUM_ACTIONS)
                    obs, _reward, terminated, truncated, _info = env.step(action)
                    self._assert_snapshot_is_fresh()
                    done = terminated or truncated
        self.assertGreater(reused, 100)

    def _assert_snapshot_is_fresh(self):
        env = self.env
        left, right = env.pad_centers()
        self.assertEqual(env._tcp.tobytes(), env.tcp_position().tobytes())
        self.assertEqual(env._left_pad.tobytes(), left.tobytes())
        self.assertEqual(env._right_pad.tobytes(), right.tobytes())
        self.assertEqual(env._gap.hex(), env.measured_gap().hex())


class PickPlaceArtifactAndEvaluateTests(unittest.TestCase):
    def test_train_script_is_pure_ppo_without_expert_or_warmstart(self):
        source = TRAIN_SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(TRAIN_SCRIPT_PATH))
        identifiers = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr)
            elif isinstance(node, ast.alias):
                identifiers.add(node.name)
                if node.asname:
                    identifiers.add(node.asname)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                identifiers.add(node.name)
        self.assertNotIn("PickPlaceExpert", identifiers)
        for ident in identifiers:
            self.assertNotIn("warmstart", ident.lower())
        self.assertNotIn("warmstart", source.lower())

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
        self.assertEqual(env_cfg["num_obs"], 70)
        self.assertEqual(env_cfg["spawn_xy_lower"], [0.30, -0.20])
        self.assertEqual(env_cfg["spawn_xy_upper"], [0.50, 0.20])
        self.assertEqual(env_cfg["runtime_config_sha256"], task_cfg.sha256)
        self.assertIn("lift_potential", reward_cfg)
        self.assertIn("first_carry", reward_cfg)
        self.assertEqual(train_cfg["algorithm"], "PPO")
        self.assertNotIn("warmstart_episodes", train_cfg)

    def test_recipe_rejects_physical_geometry_duplication_and_warmstart_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_geom = Path(tmp) / "bad_geom.yaml"
            bad_geom.write_text(
                RECIPE_PATH.read_text(encoding="utf-8").replace(
                    "  env_version: 2\n",
                    "  env_version: 2\n  cube_xy: [0.35, -0.10]\n",
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ArtifactError):
                load_training_recipe(bad_geom)

            bad_ws = Path(tmp) / "bad_ws.yaml"
            bad_ws.write_text(
                RECIPE_PATH.read_text(encoding="utf-8") + "  warmstart_episodes: 16\n",
                encoding="utf-8",
            )
            with self.assertRaises(ArtifactError):
                load_training_recipe(bad_ws)

    def test_evaluate_expert_and_zero_policy(self):
        env = PickPlaceEnv()
        try:
            expert_summary, _ = evaluate_policy(env, None, episodes=2, seed=17000)
            self.assertTrue(acceptance_passed(expert_summary))
            self.assertEqual(expert_summary["success_rate"], 1.0)
            self.assertLessEqual(expert_summary["press_mm"]["max"], 1.0)
            self.assertIn("spawn_sha256", expert_summary)

            zero_summary, _ = evaluate_policy(
                env,
                lambda _obs: np.zeros(PICK_PLACE_NUM_ACTIONS, dtype=np.float32),
                episodes=2,
                seed=17000,
            )
            self.assertEqual(zero_summary["success_rate"], 0.0)
            self.assertEqual(zero_summary["timeout_rate"], 1.0)
            self.assertEqual(expert_summary["spawn_sha256"], zero_summary["spawn_sha256"])
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
        ckpt = ROOT / "build" / "rl" / "pick_place" / "xarm6_g2_v2" / "ppo_pick_place.zip"
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
        env = PickPlaceEnv(
            max_steps=int(env_cfg["max_steps"]),
            decimation=int(env_cfg["decimation"]),
            action_scale=float(env_cfg["action_scale"]),
            action_clip=float(env_cfg["action_clip"]),
            gap_action_scale=float(env_cfg["gap_action_scale"]),
            setpoint_leash_m=float(env_cfg["setpoint_leash_m"]),
            pre_grasp_gap_m=float(env_cfg["pre_grasp_gap_m"]),
            close_floor_gap_m=float(env_cfg["close_floor_gap_m"]),
            workspace_lower=env_cfg["workspace_lower"],
            workspace_upper=env_cfg["workspace_upper"],
            spawn_xy_lower=env_cfg["spawn_xy_lower"],
            spawn_xy_upper=env_cfg["spawn_xy_upper"],
            g2_speed_percent=float(env_cfg["g2_speed_percent"]),
            g2_force_percent=float(env_cfg["g2_force_percent"]),
            reward_gamma=float(env_cfg["reward_gamma"]),
            place_clearance_m=float(env_cfg["place_clearance_m"]),
            release_max_height_m=float(env_cfg["release_max_height_m"]),
            press_limit_m=float(env_cfg["press_limit_m"]),
            near_table_m=float(env_cfg["near_table_m"]),
            max_near_table_descent_m_s=float(env_cfg["max_near_table_descent_m_s"]),
            table_overload_margin_n=float(env_cfg["table_overload_margin_n"]),
            place_success_dist_m=float(env_cfg["place_success_dist_m"]),
            drag_limit_m=float(env_cfg["drag_limit_m"]),
            tilt_limit_deg=float(env_cfg["tilt_limit_deg"]),
            success_hold_steps=int(env_cfg["success_hold_steps"]),
            reward_scales=reward_cfg,
            expected_runtime_model_sha256=env_cfg["runtime_model_sha256"],
        )
        try:
            summary, _ = evaluate_policy(
                env,
                lambda obs: model.predict(obs, deterministic=True)[0],
                episodes=10,
                seed=17000,
            )
            self.assertTrue(
                acceptance_passed(summary, min_success_rate=0.90),
                f"trained checkpoint failed eval: {summary}",
            )
        finally:
            env.close()

    @unittest.skipUnless(_HAS_SB3, "stable-baselines3 not installed")
    def test_train_pick_place_pure_ppo_smoke(self):
        from scripts.train_pick_place_rl import main as train_main

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "pick_place_smoke"
            train_main(
                [
                    "--total-steps",
                    "128",
                    "--n-envs",
                    "4",
                    "--n-workers",
                    "2",
                    "--n-steps",
                    "32",
                    "--batch-size",
                    "64",
                    "--n-epochs",
                    "1",
                    "--eval-interval",
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
