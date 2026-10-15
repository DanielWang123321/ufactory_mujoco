#!/usr/bin/env python3
"""RL example 01: TCP reaching with a trained PPO policy (roadmap 0.0.5).

Loads the policy trained by scripts/train_reach_rl.py and rolls out episodes
in the XArm6ReachEnv task: the TCP chases a randomly sampled target point.
Use --random for the untrained baseline (works without stable-baselines3).

Usage:
    python examples/rl/01_reach.py                      # GUI, 10 episodes
    python examples/rl/01_reach.py --headless           # metrics only
    python examples/rl/01_reach.py --policy build/rl/ppo_reach.zip --episodes 20
    python examples/rl/01_reach.py --random             # untrained baseline
    python examples/rl/01_reach.py --target 0.35 0.15 0.25   # fixed target
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

from src.rl import XArm6ReachEnv  # noqa: E402

DEFAULT_POLICY = Path(__file__).resolve().parent.parent.parent / "build" / "rl" / "ppo_reach.zip"


def make_policy(args):
    """Return (policy_fn, description). policy_fn(obs) -> action."""
    if args.random:
        rng = np.random.default_rng(args.seed)
        return (lambda obs: rng.uniform(-1.0, 1.0, 6)), "random baseline"
    try:
        from stable_baselines3 import PPO
    except ImportError:
        sys.exit("stable-baselines3 not installed (requirements-rl.txt); "
                 "use --random for the untrained baseline")
    policy_path = Path(args.policy)
    if not policy_path.is_file():
        sys.exit(f"policy not found: {policy_path} -- "
                 "train first with scripts/train_reach_rl.py or use --random")
    model = PPO.load(str(policy_path), device=args.device)
    return (lambda obs: model.predict(obs, deterministic=True)[0]), \
        f"PPO policy {policy_path}"


def run_episodes(env: XArm6ReachEnv, policy_fn, episodes: int, seed: int,
                 pace: bool, options: dict | None = None) -> list[dict]:
    period = env._decimation * float(env.model.opt.timestep)
    results = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep, options=options)
        total, done = 0.0, False
        while not done:
            t0 = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(policy_fn(obs))
            total += float(reward)
            done = terminated or truncated
            if pace:
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
        result = {
            "dist_mm": float(info["dist"]) * 1000.0,
            "success": bool(info["success"]),
            "steps": int(info["steps"]),
            "return": total,
        }
        results.append(result)
        print(f"  episode {ep + 1}/{episodes}: dist={result['dist_mm']:.1f} mm  "
              f"success={result['success']}  steps={result['steps']}  "
              f"return={result['return']:.1f}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="RL TCP reaching with a trained PPO policy.")
    ap.add_argument("--policy", default=str(DEFAULT_POLICY), help="path to ppo_reach.zip")
    ap.add_argument("--episodes", type=int, default=10, help="evaluation episodes (default: 10)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    ap.add_argument("--headless", action="store_true", help="no viewer, print metrics")
    ap.add_argument("--random", action="store_true", help="untrained random baseline")
    ap.add_argument("--target", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="fixed base-frame target instead of random sampling")
    ap.add_argument("--min-success-rate", type=float, default=0.7,
                    help="PASS threshold over the episodes (default: 0.7)")
    args = ap.parse_args()

    policy_fn, description = make_policy(args)
    options = {"target": np.asarray(args.target)} if args.target is not None else None
    env = XArm6ReachEnv(render_mode=None if args.headless else "human")
    print(f"Reaching with {description} ({args.episodes} episodes)")
    try:
        results = run_episodes(env, policy_fn, args.episodes, args.seed,
                               pace=not args.headless, options=options)
    finally:
        env.close()

    success_rate = sum(r["success"] for r in results) / len(results)
    mean_dist = float(np.mean([r["dist_mm"] for r in results]))
    print(f"\n  success_rate={success_rate:.2%}  mean_final_dist={mean_dist:.1f} mm  "
          f"mean_steps={np.mean([r['steps'] for r in results]):.1f}")

    ok = success_rate >= args.min_success_rate
    if args.random:
        print("\nRL reaching BASELINE (untrained policy, not judged)")
        sys.exit(0)
    print("\nRL reaching " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
