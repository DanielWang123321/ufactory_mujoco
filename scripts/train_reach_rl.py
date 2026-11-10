#!/usr/bin/env python3
"""Train a PPO policy for the TCP reaching task.

Supports lite6 / xarm5 / xarm6 / xarm7 / uf850 (--model). Reads defaults from
examples/rl/reach/recipe.yaml (--recipe) and writes:
- <out-dir>/ppo_reach.zip
- <out-dir>/config.yaml
- <out-dir>/ppo_reach.checkpoint_manifest.json
- <out-dir>/provenance.json
- <out-dir>/metrics.json

Usage:
    python scripts/train_reach_rl.py                        # xArm 6
    python scripts/train_reach_rl.py --model lite6          # another arm
    python scripts/train_reach_rl.py --total-steps 1000000  # override steps
    python scripts/train_reach_rl.py --backfill-manifest    # generate manifests for existing build/rl/*/ppo_reach.zip
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ufactory_mujoco.rl import (  # noqa: E402
    REACH_ACTION_CONTRACT,
    ROBOT_PROFILES,
    ReachEnv,
    build_reach_task_configs,
    load_training_config,
    validate_checkpoint_artifacts,
    write_checkpoint_manifest,
    write_run_provenance,
    write_training_config,
)

DEFAULT_RECIPE = ROOT / "examples" / "rl" / "reach" / "recipe.yaml"


def evaluate(model, robot: str, episodes: int, seed: int, max_steps: int = 200) -> dict:
    """Deterministic-policy evaluation on a fresh single env."""
    env = ReachEnv(robot=robot, max_steps=max_steps)
    successes, dists, returns, hit_steps = 0, [], [], []
    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            done, total = False, 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                done = terminated or truncated
            successes += int(info["success"])
            dists.append(float(info["dist"]))
            returns.append(total)
            if info["success"]:
                hit_steps.append(int(info["steps"]))
    finally:
        env.close()
    dists_arr = np.asarray(dists, dtype=float)
    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "mean_final_dist_m": float(np.mean(dists_arr)),
        "median_final_dist_m": float(np.median(dists_arr)),
        "p90_final_dist_m": float(np.percentile(dists_arr, 90)),
        "p99_final_dist_m": float(np.percentile(dists_arr, 99)),
        "max_final_dist_m": float(np.max(dists_arr)),
        "mean_return": float(np.mean(returns)),
        "mean_steps_to_success": float(np.mean(hit_steps)) if hit_steps else None,
    }


ENV_MEMORY_MB = 350  # measured peak RSS per SubprocVecEnv worker (lite visuals + torch) ~310 MB


def _available_memory_mb() -> int | None:
    """MemAvailable from /proc/meminfo (Linux); None elsewhere."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def auto_n_envs(cpu_count: int | None = None, avail_mb: int | None = None,
                reserve_mb: int = 1500) -> int:
    """Parallel env count bounded by CPU cores and available memory (min 1, max 16)."""
    import os
    cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    n = min(cpus, 16)
    avail = avail_mb if avail_mb is not None else _available_memory_mb()
    if avail is not None:
        n = min(n, max(1, (avail - reserve_mb) // ENV_MEMORY_MB))
    return max(1, n)


def _provenance_sources(recipe_path: Path) -> list[Path]:
    return [
        Path(__file__).resolve(),
        recipe_path.resolve(),
        ROOT / "ufactory_mujoco" / "rl" / "reach_env.py",
        ROOT / "ufactory_mujoco" / "rl" / "artifacts.py",
        ROOT / "ufactory_mujoco" / "robots.py",
    ]


def write_reach_artifacts(
    out_dir: Path,
    *,
    robot: str,
    recipe_path: Path,
    train_overrides: dict | None = None,
) -> tuple[dict, object]:
    """Write config.yaml, ppo_reach.checkpoint_manifest.json, and provenance.json."""
    env_cfg, reward_cfg, train_cfg = build_reach_task_configs(robot, recipe_path=recipe_path)
    if train_overrides:
        train_cfg.update(train_overrides)
    config_path = out_dir / "config.yaml"
    write_training_config(
        config_path,
        task="reach",
        robot=robot,
        env=env_cfg,
        reward=reward_cfg,
        train=train_cfg,
    )
    training_config = load_training_config(config_path)
    ckpt_path = out_dir / "ppo_reach.zip"
    manifest_path = write_checkpoint_manifest(
        ckpt_path,
        training_config=training_config,
        action_contract=REACH_ACTION_CONTRACT,
    )
    write_run_provenance(
        out_dir / "provenance.json",
        training_config=training_config,
        source_paths=_provenance_sources(recipe_path),
    )
    validated, manifest = validate_checkpoint_artifacts(
        ckpt_path,
        config_path,
        manifest_path,
        expected_task="reach",
        expected_robot=robot,
        expected_runtime_env=env_cfg,
    )
    return validated, manifest


def backfill_manifests(models: list[str], recipe_path: Path, base_dir: Path | None = None) -> None:
    """Generate config.yaml and checkpoint manifests for existing ppo_reach.zip files."""
    root_dir = base_dir if base_dir is not None else (ROOT / "build" / "rl")
    for robot in models:
        out_dir = root_dir / robot
        ckpt = out_dir / "ppo_reach.zip"
        if not ckpt.is_file():
            raise FileNotFoundError(f"cannot backfill manifest; missing checkpoint: {ckpt}")
        overrides = {}
        metrics_file = out_dir / "metrics.json"
        if metrics_file.is_file():
            saved = json.loads(metrics_file.read_text(encoding="utf-8"))
            cfg = saved.get("config", {})
            for key in ("total_steps", "n_envs", "n_steps", "seed", "device"):
                if key in cfg:
                    overrides[key] = cfg[key]
        _, manifest = write_reach_artifacts(
            out_dir,
            robot=robot,
            recipe_path=recipe_path,
            train_overrides=overrides,
        )
        print(f"[backfill] {robot}: config={manifest.config_sha256[:12]} ckpt={manifest.checkpoint_sha256[:12]}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train PPO on TCP reaching.")
    parser.add_argument("--model", default="xarm6", choices=[*sorted(ROBOT_PROFILES), "all"],
                        help="robot model (default: xarm6; 'all' allowed with --backfill-manifest)")
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE,
                        help="path to reach recipe.yaml")
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=None,
                        help="parallel envs (default: auto from CPU cores and free memory)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    parser.add_argument("--out-dir", default=None,
                        help="output directory (default: build/rl/<model>)")
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=None,
                        help="checkpoint every N total steps")
    parser.add_argument("--n-steps", type=int, default=None, help="PPO rollout per env")
    parser.add_argument("--net-arch", default="64,64",
                        help="MLP layer sizes, comma separated (default: 64,64)")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                        help="PPO entropy coefficient (default: 0.0)")
    parser.add_argument("--backfill-manifest", action="store_true",
                        help="generate config.yaml and checkpoint_manifest.json for existing ppo_reach.zip")
    args = parser.parse_args(argv)

    if args.backfill_manifest:
        models = sorted(ROBOT_PROFILES) if args.model == "all" else [args.model]
        backfill_manifests(models, args.recipe, Path(args.out_dir) if args.out_dir else None)
        return

    if args.model == "all":
        parser.error("--model all is only supported with --backfill-manifest")

    try:
        import gymnasium  # noqa: E402
        import stable_baselines3  # noqa: E402
        from stable_baselines3 import PPO  # noqa: E402
        from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
        from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402
    except ImportError as exc:  # pragma: no cover
        sys.exit(f"training needs gymnasium + stable-baselines3 (requirements-rl.txt): {exc}")

    env_cfg, _reward_cfg, train_cfg = build_reach_task_configs(args.model, recipe_path=args.recipe)
    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 0))
    n_steps = int(args.n_steps if args.n_steps is not None else train_cfg.get("n_steps", 2048))
    save_interval = int(
        args.save_interval if args.save_interval is not None else train_cfg.get("save_interval", 50_000)
    )
    net_arch = [int(x) for x in args.net_arch.split(",")]

    if args.n_envs is None:
        args.n_envs = auto_n_envs()
    out_dir = Path(args.out_dir or ROOT / "build" / "rl" / args.model)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    vec_env = make_vec_env(
        ReachEnv,
        n_envs=args.n_envs,
        seed=seed,
        env_kwargs={
            "robot": args.model,
            "max_steps": int(env_cfg["max_steps"]),
            "decimation": int(env_cfg["decimation"]),
            "delta": float(env_cfg["delta"]),
            "success_dist": float(env_cfg["success_dist"]),
            "q0_noise": float(env_cfg["q0_noise"]),
            "validate_targets": bool(env_cfg["validate_targets"]),
        },
        vec_env_cls=vec_cls,
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        device=args.device,
        n_steps=n_steps,
        batch_size=int(train_cfg.get("batch_size", 256)),
        n_epochs=int(train_cfg.get("n_epochs", 10)),
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        gamma=float(train_cfg.get("gamma", 0.99)),
        gae_lambda=float(train_cfg.get("gae_lambda", 0.95)),
        ent_coef=args.ent_coef,
        policy_kwargs={"net_arch": net_arch},
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=max(save_interval // max(args.n_envs, 1), 1),
        save_path=str(out_dir / "checkpoints"),
        name_prefix="ppo_reach",
    )

    print(f"[train] {args.model}: {args.total_steps} steps, {args.n_envs} envs, device={args.device}")
    start = time.perf_counter()
    model.learn(total_timesteps=args.total_steps, callback=checkpoint_cb)
    train_seconds = time.perf_counter() - start

    model.save(out_dir / "ppo_reach.zip")
    vec_env.close()

    training_config, manifest = write_reach_artifacts(
        out_dir,
        robot=args.model,
        recipe_path=args.recipe,
        train_overrides={
            "total_steps": args.total_steps,
            "n_envs": args.n_envs,
            "n_steps": n_steps,
            "seed": seed,
            "device": args.device,
            "net_arch": net_arch,
            "ent_coef": float(args.ent_coef),
            "save_interval": save_interval,
            "eval_episodes": args.eval_episodes,
        },
    )

    metrics = evaluate(model, args.model, args.eval_episodes, seed=10_000 + seed, max_steps=int(env_cfg["max_steps"]))
    metrics.update({
        "config": {
            "model": args.model,
            "total_steps": args.total_steps,
            "n_envs": args.n_envs,
            "n_steps": n_steps,
            "seed": seed,
            "device": args.device,
            "config_sha256": training_config["config_sha256"],
            "checkpoint_sha256": manifest.checkpoint_sha256,
        },
        "train_seconds": round(train_seconds, 1),
        "train_steps_per_second": round(args.total_steps / max(train_seconds, 1e-9), 1),
        "versions": {
            "python": platform.python_version(),
            "mujoco": __import__("mujoco").__version__,
            "numpy": np.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
        },
    })
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"\n[train] {args.model} evaluation ({metrics['episodes']} episodes, deterministic policy):")
    print(f"  success_rate        : {metrics['success_rate']:.2%}")
    print(f"  mean_final_dist_m   : {metrics['mean_final_dist_m']:.4f}")
    print(f"  max_final_dist_m    : {metrics['max_final_dist_m']:.4f}")
    print(f"  mean_return         : {metrics['mean_return']:.2f}")
    print(f"  mean_steps_to_success: {metrics['mean_steps_to_success']}")
    print(f"[train] saved {out_dir / 'ppo_reach.zip'}, config.yaml, checkpoint_manifest.json, and metrics.json")


if __name__ == "__main__":
    main()
