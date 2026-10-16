#!/usr/bin/env python3
"""Train a PPO policy for the TCP reaching task (roadmap 0.0.5, 0.0.7).

Supports lite6 / xarm5 / xarm6 / xarm7 / uf850 (--model). Requires the RL
dependencies (requirements-rl.txt: gymnasium + stable-baselines3). The trained
policy is saved to <out-dir>/ppo_reach.zip together with an evaluation report
<out-dir>/metrics.json (default out-dir: build/rl/<model>/).

Usage:
    python scripts/train_reach_rl.py                        # xArm 6, 300k steps
    python scripts/train_reach_rl.py --model lite6          # another arm
    python scripts/train_reach_rl.py --total-steps 1000000  # longer run
    python scripts/train_reach_rl.py --n-envs 4 --device cpu
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

try:
    import gymnasium  # noqa: E402
    import stable_baselines3  # noqa: E402
    from stable_baselines3 import PPO  # noqa: E402
    from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
    from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402
except ImportError as exc:  # pragma: no cover
    sys.exit(f"training needs gymnasium + stable-baselines3 (requirements-rl.txt): {exc}")

from src.rl import ROBOT_PROFILES, ReachEnv  # noqa: E402


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
    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "mean_final_dist_m": float(np.mean(dists)),
        "max_final_dist_m": float(np.max(dists)),
        "mean_return": float(np.mean(returns)),
        "mean_steps_to_success": float(np.mean(hit_steps)) if hit_steps else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on TCP reaching.")
    parser.add_argument("--model", default="xarm6", choices=sorted(ROBOT_PROFILES),
                        help="robot model (default: xarm6)")
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    parser.add_argument("--out-dir", default=None,
                        help="output directory (default: build/rl/<model>)")
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=50_000,
                        help="checkpoint every N total steps")
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO rollout per env")
    parser.add_argument("--net-arch", default="64,64",
                        help="MLP layer sizes, comma separated (default: 64,64)")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                        help="PPO entropy coefficient (default: 0.0)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir or ROOT / "build" / "rl" / args.model)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    vec_env = make_vec_env(ReachEnv, n_envs=args.n_envs, seed=args.seed,
                           env_kwargs={"robot": args.model, "max_steps": 200},
                           vec_env_cls=vec_cls)
    model = PPO(
        "MlpPolicy", vec_env, verbose=1, seed=args.seed, device=args.device,
        n_steps=args.n_steps, batch_size=256, n_epochs=10,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, ent_coef=args.ent_coef,
        policy_kwargs={"net_arch": [int(x) for x in args.net_arch.split(",")]},
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.save_interval // max(args.n_envs, 1), 1),
        save_path=str(out_dir / "checkpoints"), name_prefix="ppo_reach")

    print(f"[train] {args.model}: {args.total_steps} steps, {args.n_envs} envs, "
          f"device={args.device}")
    start = time.perf_counter()
    model.learn(total_timesteps=args.total_steps, callback=checkpoint_cb)
    train_seconds = time.perf_counter() - start

    model.save(out_dir / "ppo_reach.zip")
    metrics = evaluate(model, args.model, args.eval_episodes, seed=10_000 + args.seed)
    metrics.update({
        "config": {
            "model": args.model,
            "total_steps": args.total_steps, "n_envs": args.n_envs,
            "n_steps": args.n_steps, "seed": args.seed, "device": args.device,
        },
        "train_seconds": round(train_seconds, 1),
        "train_steps_per_second": round(args.total_steps / train_seconds, 1),
        "versions": {
            "python": platform.python_version(),
            "mujoco": __import__("mujoco").__version__,
            "numpy": np.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
        },
    })
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"\n[train] {args.model} evaluation "
          f"({metrics['episodes']} episodes, deterministic policy):")
    print(f"  success_rate        : {metrics['success_rate']:.2%}")
    print(f"  mean_final_dist_m   : {metrics['mean_final_dist_m']:.4f}")
    print(f"  max_final_dist_m    : {metrics['max_final_dist_m']:.4f}")
    print(f"  mean_return         : {metrics['mean_return']:.2f}")
    print(f"  mean_steps_to_success: {metrics['mean_steps_to_success']}")
    print(f"[train] saved {out_dir / 'ppo_reach.zip'} and metrics.json")
    vec_env.close()


if __name__ == "__main__":
    main()
