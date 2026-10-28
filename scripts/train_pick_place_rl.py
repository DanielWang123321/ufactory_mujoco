#!/usr/bin/env python3
"""Train a PPO policy for the xArm6 + Gripper G2 tabletop pick-and-place task (Batch 8).

Reads defaults from examples/rl/pick_place/recipe.yaml (--recipe) and writes:
- <out-dir>/ppo_pick_place.zip
- <out-dir>/config.yaml
- <out-dir>/ppo_pick_place.checkpoint_manifest.json
- <out-dir>/provenance.json
- <out-dir>/metrics.json

Usage:
    python scripts/train_pick_place_rl.py
    python scripts/train_pick_place_rl.py --total-steps 50000 --n-envs 4
    python scripts/train_pick_place_rl.py --warmstart-episodes 0 --total-steps 256 --n-envs 1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from examples.rl.pick_place.evaluate import (  # noqa: E402
    acceptance_passed,
    checkpoint_score,
    evaluate_policy,
)
from ufactory_mujoco.rl import (  # noqa: E402
    PICK_PLACE_ACTION_CONTRACT,
    PickPlaceEnv,
    PickPlaceExpert,
    build_pick_place_task_configs,
    load_training_config,
    validate_checkpoint_artifacts,
    write_checkpoint_manifest,
    write_run_provenance,
    write_training_config,
)

DEFAULT_RECIPE = ROOT / "examples" / "rl" / "pick_place" / "recipe.yaml"
ENV_MEMORY_MB = 400


def _available_memory_mb() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def auto_n_envs(cpu_count: int | None = None, avail_mb: int | None = None, reserve_mb: int = 1800) -> int:
    cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    n = min(cpus, 8)
    avail = avail_mb if avail_mb is not None else _available_memory_mb()
    if avail is not None:
        n = min(n, max(1, (avail - reserve_mb) // ENV_MEMORY_MB))
    return max(1, n)


def _env_kwargs(
    robot: str,
    runtime_config: Path | None,
    env_cfg: dict[str, Any],
    reward_cfg: dict[str, float],
) -> dict[str, Any]:
    return {
        "robot": robot,
        "config_path": runtime_config,
        "max_steps": int(env_cfg["max_steps"]),
        "decimation": int(env_cfg["decimation"]),
        "action_scale": float(env_cfg["action_scale"]),
        "action_clip": float(env_cfg["action_clip"]),
        "gap_action_scale": float(env_cfg["gap_action_scale"]),
        "setpoint_leash_m": float(env_cfg["setpoint_leash_m"]),
        "pre_grasp_gap_m": float(env_cfg["pre_grasp_gap_m"]),
        "close_floor_gap_m": float(env_cfg["close_floor_gap_m"]),
        "workspace_lower": env_cfg["workspace_lower"],
        "workspace_upper": env_cfg["workspace_upper"],
        "reward_gamma": float(env_cfg["reward_gamma"]),
        "place_clearance_m": float(env_cfg["place_clearance_m"]),
        "release_max_height_m": float(env_cfg["release_max_height_m"]),
        "press_limit_m": float(env_cfg["press_limit_m"]),
        "near_table_m": float(env_cfg["near_table_m"]),
        "max_near_table_descent_m_s": float(env_cfg["max_near_table_descent_m_s"]),
        "table_overload_margin_n": float(env_cfg["table_overload_margin_n"]),
        "place_success_dist_m": float(env_cfg["place_success_dist_m"]),
        "drag_limit_m": float(env_cfg["drag_limit_m"]),
        "tilt_limit_deg": float(env_cfg["tilt_limit_deg"]),
        "success_hold_steps": int(env_cfg["success_hold_steps"]),
        "reward_scales": reward_cfg,
    }


def _provenance_sources(recipe_path: Path, runtime_config_path: Path) -> list[Path]:
    return [
        Path(__file__).resolve(),
        recipe_path.resolve(),
        runtime_config_path.resolve(),
        ROOT / "ufactory_mujoco" / "rl" / "pick_place_env.py",
        ROOT / "ufactory_mujoco" / "rl" / "artifacts.py",
        ROOT / "ufactory_mujoco" / "grippers.py",
    ]


def collect_expert_dataset(
    env: PickPlaceEnv,
    *,
    episodes: int,
    seed: int,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect clean and mildly perturbed expert rollouts (DAgger-style) with discounted returns."""
    expert = PickPlaceExpert(env)
    rng = np.random.default_rng(seed)
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    ret_list: list[np.ndarray] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_obs: list[np.ndarray] = []
        ep_act: list[np.ndarray] = []
        ep_rew: list[float] = []
        # First 60% episodes clean, remaining 40% with gentle action perturbations for recovery coverage
        noise_std = 0.0 if ep < max(1, int(0.6 * episodes)) else 0.04
        while not done:
            target_action = expert(obs)
            if noise_std > 0.0:
                applied_action = np.clip(
                    target_action + rng.normal(0.0, noise_std, size=target_action.shape).astype(np.float32),
                    -1.0,
                    1.0,
                )
            else:
                applied_action = target_action
            next_obs, reward, terminated, truncated, info = env.step(applied_action)
            ep_obs.append(obs.copy())
            ep_act.append(target_action.copy())
            ep_rew.append(float(reward))
            obs = next_obs
            done = bool(terminated or truncated)

        # Only include successful rollouts in behavior cloning dataset
        if not info.get("success", False) and obs_list:
            continue

        returns = np.zeros(len(ep_rew), dtype=np.float32)
        running = 0.0
        for idx in range(len(ep_rew) - 1, -1, -1):
            running = ep_rew[idx] + gamma * running
            returns[idx] = running

        obs_list.extend(ep_obs)
        act_list.extend(ep_act)
        ret_list.append(returns)

    return (
        np.asarray(obs_list, dtype=np.float32),
        np.asarray(act_list, dtype=np.float32),
        np.concatenate(ret_list, axis=0).astype(np.float32),
    )


def warmstart_ppo_policy(
    model: Any,
    env: PickPlaceEnv,
    *,
    episodes: int,
    epochs: int,
    batch_size: int,
    seed: int,
    gamma: float,
) -> dict[str, Any]:
    """Warmstart PPO actor and critic networks from PickPlaceExpert rollouts."""
    import torch

    t0 = time.perf_counter()
    obs_arr, act_arr, ret_arr = collect_expert_dataset(
        env,
        episodes=episodes,
        seed=seed + 5000,
        gamma=gamma,
    )
    device = model.device
    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(act_arr, dtype=torch.float32, device=device)
    ret_t = torch.as_tensor(ret_arr, dtype=torch.float32, device=device).unsqueeze(-1)

    policy = model.policy
    policy.set_training_mode(True)
    n_epochs = max(1, epochs)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-4)
    rng = np.random.default_rng(seed)
    n_samples = len(obs_arr)
    final_act_loss = 0.0
    final_val_loss = 0.0

    for _epoch in range(n_epochs):
        perm = rng.permutation(n_samples)
        act_losses = []
        val_losses = []
        for start_idx in range(0, n_samples, batch_size):
            batch_idx = perm[start_idx : start_idx + batch_size]
            b_obs = obs_t[batch_idx]
            b_act = act_t[batch_idx]
            b_ret = ret_t[batch_idx]

            features = policy.extract_features(b_obs)
            if policy.share_features_extractor:
                latent_pi, latent_vf = policy.mlp_extractor(features)
            else:
                pi_features, vf_features = features
                latent_pi = policy.mlp_extractor.forward_actor(pi_features)
                latent_vf = policy.mlp_extractor.forward_critic(vf_features)
            mean_actions = policy.action_net(latent_pi)
            values = policy.value_net(latent_vf)

            act_loss = torch.mean((mean_actions - b_act) ** 2)
            val_loss = torch.mean((values - b_ret) ** 2)
            loss = act_loss + 1e-4 * val_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            act_losses.append(float(act_loss.detach().cpu()))
            val_losses.append(float(val_loss.detach().cpu()))
        scheduler.step()
        final_act_loss = float(np.mean(act_losses))
        final_val_loss = float(np.mean(val_losses))

    if hasattr(policy, "log_std") and isinstance(policy.log_std, torch.nn.Parameter):
        with torch.no_grad():
            policy.log_std.fill_(-4.5)
    policy.set_training_mode(False)

    return {
        "episodes": int(episodes),
        "epochs": int(epochs),
        "transitions": int(n_samples),
        "final_action_mse": round(final_act_loss, 6),
        "final_value_mse": round(final_val_loss, 4),
        "seconds": round(time.perf_counter() - t0, 2),
    }


def write_pick_place_artifacts(
    out_dir: Path,
    *,
    robot: str,
    recipe_path: Path,
    runtime_config: Path | None = None,
    train_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Any]:
    """Write config.yaml, ppo_pick_place.checkpoint_manifest.json, and provenance.json."""
    env_cfg, reward_cfg, train_cfg = build_pick_place_task_configs(
        robot,
        recipe_path=recipe_path,
        config_path=runtime_config,
    )
    if train_overrides:
        train_cfg.update(train_overrides)
    config_path = out_dir / "config.yaml"
    write_training_config(
        config_path,
        task="pick_place",
        robot=robot,
        env=env_cfg,
        reward=reward_cfg,
        train=train_cfg,
    )
    training_config = load_training_config(config_path)
    ckpt_path = out_dir / "ppo_pick_place.zip"
    manifest_path = write_checkpoint_manifest(
        ckpt_path,
        training_config=training_config,
        action_contract=PICK_PLACE_ACTION_CONTRACT,
    )
    runtime_cfg_path = (
        Path(runtime_config)
        if runtime_config is not None
        else (ROOT / "configs" / "pick_place" / f"{robot}.yaml")
    )
    write_run_provenance(
        out_dir / "provenance.json",
        training_config=training_config,
        source_paths=_provenance_sources(recipe_path, runtime_cfg_path),
    )
    validated, manifest = validate_checkpoint_artifacts(
        ckpt_path,
        config_path,
        manifest_path,
        expected_task="pick_place",
        expected_robot=robot,
        expected_runtime_config_sha256=env_cfg["runtime_config_sha256"],
        expected_runtime_env=env_cfg,
    )
    return validated, manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--robot", dest="model", default="xarm6", choices=["xarm6"],
                        help="robot model (default: xarm6)")
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE,
                        help="path to pick_place recipe.yaml")
    parser.add_argument("--runtime-config", type=Path, default=None,
                        help="optional override for configs/pick_place/xarm6.yaml")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="PPO training timesteps (default from recipe)")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="parallel envs (default: auto from CPU cores and free memory)")
    parser.add_argument("--n-steps", type=int, default=None,
                        help="PPO rollout steps per env (default from recipe)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="minibatch size (default from recipe)")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="PPO epochs per rollout (default from recipe)")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="PPO learning rate (default from recipe)")
    parser.add_argument("--ent-coef", type=float, default=None,
                        help="PPO entropy coefficient (default from recipe)")
    parser.add_argument("--net-arch", default=None,
                        help="comma-separated MLP layer sizes (default from recipe: 256,256)")
    parser.add_argument("--warmstart-episodes", type=int, default=None,
                        help="expert warmstart episodes before PPO (0 disables warmstart)")
    parser.add_argument("--warmstart-epochs", type=int, default=None,
                        help="behavior cloning epochs for warmstart")
    parser.add_argument("--eval-episodes", type=int, default=None,
                        help="deterministic evaluation episodes (default from recipe)")
    parser.add_argument("--save-interval", type=int, default=None,
                        help="checkpoint save interval in env steps")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default: build/rl/pick_place/<model>)")
    args = parser.parse_args(argv)

    try:
        import gymnasium  # noqa: E402
        import stable_baselines3  # noqa: E402
        from stable_baselines3 import PPO  # noqa: E402
        from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
        from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402
    except ImportError as exc:  # pragma: no cover
        sys.exit(f"training needs gymnasium + stable-baselines3 (requirements-rl.txt): {exc}")

    env_cfg, reward_cfg, train_cfg = build_pick_place_task_configs(
        args.model,
        recipe_path=args.recipe,
        config_path=args.runtime_config,
    )

    total_steps = int(args.total_steps if args.total_steps is not None else train_cfg.get("total_steps", 200_000))
    n_steps = int(args.n_steps if args.n_steps is not None else train_cfg.get("n_steps", 512))
    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 256))
    n_epochs = int(args.n_epochs if args.n_epochs is not None else train_cfg.get("n_epochs", 10))
    learning_rate = float(args.learning_rate if args.learning_rate is not None else train_cfg.get("learning_rate", 3e-4))
    ent_coef = float(args.ent_coef if args.ent_coef is not None else train_cfg.get("ent_coef", 0.005))
    warmstart_episodes = int(
        args.warmstart_episodes if args.warmstart_episodes is not None else train_cfg.get("warmstart_episodes", 32)
    )
    warmstart_epochs = int(
        args.warmstart_epochs if args.warmstart_epochs is not None else train_cfg.get("warmstart_epochs", 15)
    )
    eval_episodes = int(args.eval_episodes if args.eval_episodes is not None else train_cfg.get("eval_episodes", 20))
    save_interval = int(args.save_interval if args.save_interval is not None else train_cfg.get("save_interval", 50_000))
    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 1))
    net_arch = (
        [int(x) for x in args.net_arch.split(",")]
        if args.net_arch is not None
        else [int(x) for x in train_cfg.get("net_arch", [256, 256])]
    )
    gamma = float(train_cfg.get("gamma", env_cfg["reward_gamma"]))
    gae_lambda = float(train_cfg.get("gae_lambda", 0.95))

    n_envs = int(args.n_envs if args.n_envs is not None else auto_n_envs())
    if total_steps > 0:
        n_steps = max(16, min(n_steps, max(16, total_steps // max(n_envs, 1))))
        batch_size = min(batch_size, n_steps * max(n_envs, 1))

    out_dir = Path(args.out_dir or (ROOT / "build" / "rl" / "pick_place" / args.model))
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = _env_kwargs(args.model, args.runtime_config, env_cfg, reward_cfg)
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    vec_env = make_vec_env(
        PickPlaceEnv,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_cls,
    )

    ppo_lr = min(learning_rate * 0.05, 1.5e-5) if warmstart_episodes > 0 else learning_rate
    ppo_ent = 0.0 if warmstart_episodes > 0 else ent_coef
    ppo_clip = 0.05 if warmstart_episodes > 0 else 0.2
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        device=args.device,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=ppo_lr,
        clip_range=ppo_clip,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ppo_ent,
        policy_kwargs={"net_arch": net_arch},
    )

    eval_env = PickPlaceEnv(**env_kwargs)
    candidates: list[tuple[tuple[float, float, float, float, float], Path, dict[str, Any]]] = []
    warmstart_info: dict[str, Any] | None = None

    start = time.perf_counter()
    try:
        if warmstart_episodes > 0:
            print(
                f"[warmstart] {args.model}: collecting {warmstart_episodes} expert episodes "
                f"and training {warmstart_epochs} epochs..."
            )
            warmstart_info = warmstart_ppo_policy(
                model,
                eval_env,
                episodes=warmstart_episodes,
                epochs=warmstart_epochs,
                batch_size=min(128, batch_size),
                seed=seed,
                gamma=gamma,
            )
            ws_ckpt = ckpt_dir / "ppo_pick_place_warmstart.zip"
            model.save(ws_ckpt)
            ws_summary, _ = evaluate_policy(
                eval_env,
                lambda obs: model.predict(obs, deterministic=True)[0],
                episodes=eval_episodes,
                seed=17_000 + seed,
            )
            candidates.append((checkpoint_score(ws_summary), ws_ckpt, ws_summary))
            print(
                f"[warmstart] eval success_rate={ws_summary['success_rate']:.2%} "
                f"press_max={ws_summary['press_mm']['max']:.3f}mm "
                f"target_err_mean={ws_summary['target_error_mm']['mean']:.2f}mm"
            )

        if total_steps > 0:
            checkpoint_cb = CheckpointCallback(
                save_freq=max(save_interval // max(n_envs, 1), 1),
                save_path=str(ckpt_dir),
                name_prefix="ppo_pick_place",
            )
            print(f"[train] {args.model}: {total_steps} PPO steps, {n_envs} envs, device={args.device}")
            model.learn(total_timesteps=total_steps, callback=checkpoint_cb)

        final_ckpt = ckpt_dir / "ppo_pick_place_final.zip"
        model.save(final_ckpt)
        final_summary, _ = evaluate_policy(
            eval_env,
            lambda obs: model.predict(obs, deterministic=True)[0],
            episodes=eval_episodes,
            seed=17_000 + seed,
        )
        candidates.append((checkpoint_score(final_summary), final_ckpt, final_summary))

        # Also score any intermediate checkpoints written by CheckpointCallback
        for step_ckpt in sorted(ckpt_dir.glob("ppo_pick_place_*_steps.zip")):
            step_model = PPO.load(step_ckpt, device=args.device)
            step_summary, _ = evaluate_policy(
                eval_env,
                lambda obs, m=step_model: m.predict(obs, deterministic=True)[0],
                episodes=eval_episodes,
                seed=17_000 + seed,
            )
            candidates.append((checkpoint_score(step_summary), step_ckpt, step_summary))
    finally:
        eval_env.close()
        vec_env.close()

    train_seconds = time.perf_counter() - start
    best_score, best_ckpt_path, best_summary = max(candidates, key=lambda item: item[0])
    target_ckpt = out_dir / "ppo_pick_place.zip"
    shutil.copy2(best_ckpt_path, target_ckpt)

    training_config, manifest = write_pick_place_artifacts(
        out_dir,
        robot=args.model,
        recipe_path=args.recipe,
        runtime_config=args.runtime_config,
        train_overrides={
            "total_steps": total_steps,
            "n_envs": n_envs,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "net_arch": net_arch,
            "warmstart_episodes": warmstart_episodes,
            "warmstart_epochs": warmstart_epochs,
            "eval_episodes": eval_episodes,
            "save_interval": save_interval,
            "seed": seed,
            "device": args.device,
        },
    )

    metrics = dict(best_summary)
    metrics.update(
        {
            "passed": acceptance_passed(best_summary),
            "selected_checkpoint": best_ckpt_path.name,
            "checkpoint_score": list(best_score),
            "warmstart": warmstart_info,
            "config": {
                "model": args.model,
                "total_steps": total_steps,
                "n_envs": n_envs,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "learning_rate": learning_rate,
                "ent_coef": ent_coef,
                "net_arch": net_arch,
                "warmstart_episodes": warmstart_episodes,
                "warmstart_epochs": warmstart_epochs,
                "seed": seed,
                "device": args.device,
                "config_sha256": training_config["config_sha256"],
                "checkpoint_sha256": manifest.checkpoint_sha256,
                "runtime_config_sha256": env_cfg["runtime_config_sha256"],
            },
            "train_seconds": round(train_seconds, 1),
            "train_steps_per_second": round(total_steps / max(train_seconds, 1e-9), 1),
            "versions": {
                "python": platform.python_version(),
                "mujoco": __import__("mujoco").__version__,
                "numpy": np.__version__,
                "gymnasium": gymnasium.__version__,
                "stable_baselines3": stable_baselines3.__version__,
            },
        }
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"\n[train] {args.model} pick-and-place evaluation ({best_summary['episodes']} episodes):")
    print(f"  selected_checkpoint : {best_ckpt_path.name}")
    print(f"  success_rate        : {best_summary['success_rate']:.2%}")
    print(f"  max_press_mm        : {best_summary['press_mm']['max']:.3f}")
    print(f"  mean_target_err_mm  : {best_summary['target_error_mm']['mean']:.2f}")
    print(f"  mean_return         : {best_summary['return']['mean']:.2f}")
    print(f"[train] saved {target_ckpt}, config.yaml, checkpoint_manifest.json, and metrics.json")


if __name__ == "__main__":
    main()
