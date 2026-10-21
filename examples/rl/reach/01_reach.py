#!/usr/bin/env python3
"""RL example 01: TCP reaching with a trained PPO policy (roadmap 0.0.5, 0.0.7).

Loads the policy trained by scripts/train_reach_rl.py and rolls out episodes
in the ReachEnv task: the TCP chases a randomly sampled target point.
Supports lite6 / xarm5 / xarm6 / xarm7 / uf850 (--model).
Use --random for the untrained baseline (works without stable-baselines3).

Usage:
    python examples/rl/reach/01_reach.py                      # xArm 6, GUI, 10 episodes
    python examples/rl/reach/01_reach.py --model lite6        # another arm
    python examples/rl/reach/01_reach.py --headless           # metrics only
    python examples/rl/reach/01_reach.py --policy build/rl/xarm6/ppo_reach.zip --episodes 20
    python examples/rl/reach/01_reach.py --random             # untrained baseline
    python examples/rl/reach/01_reach.py --target 0.35 0.15 0.25   # fixed target
"""

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from ufactory_mujoco.rl import ROBOT_PROFILES, ReachEnv  # noqa: E402

_BUILD = _ROOT / "build" / "rl"


def make_policy(args):
    """Return (policy_fn, description). policy_fn(obs) -> action."""
    if args.random:
        rng = np.random.default_rng(args.seed)
        return ((lambda obs: rng.uniform(-1.0, 1.0, ROBOT_PROFILES[args.model].n_joints)),
                "random baseline")
    try:
        from stable_baselines3 import PPO
    except ImportError:
        sys.exit("stable-baselines3 not installed (requirements-rl.txt); "
                 "use --random for the untrained baseline")
    policy_path = Path(args.policy)
    if not policy_path.is_file():
        sys.exit(f"policy not found: {policy_path} -- "
                 f"train first with scripts/train_reach_rl.py --model {args.model} "
                 "or use --random")
    model = PPO.load(str(policy_path), device=args.device)
    return (lambda obs: model.predict(obs, deterministic=True)[0]), \
        f"PPO policy {policy_path}"


def run_episodes(env: ReachEnv, policy_fn, episodes: int, seed: int,
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
    ap.add_argument("--model", default="xarm6", choices=sorted(ROBOT_PROFILES),
                    help="robot model (default: xarm6)")
    ap.add_argument("--policy", default=None,
                    help="path to ppo_reach.zip (default: build/rl/<model>/ppo_reach.zip)")
    ap.add_argument("--episodes", type=int, default=10, help="evaluation episodes (default: 10)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    ap.add_argument("--headless", action="store_true", help="no viewer, print metrics")
    ap.add_argument("--random", action="store_true", help="untrained random baseline")
    ap.add_argument("--target", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="fixed base-frame target instead of random sampling")
    ap.add_argument("--min-success-rate", type=float, default=0.95,
                    help="PASS success-rate threshold (default: 0.95, "
                         "ufactory_genesis acceptance standard)")
    ap.add_argument("--max-mean-dist-mm", type=float, default=10.0,
                    help="PASS mean final distance bound in mm (default: 10.0, "
                         "ufactory_genesis acceptance standard)")
    args = ap.parse_args()
    if args.policy is None:
        args.policy = str(_BUILD / args.model / "ppo_reach.zip")

    policy_fn, description = make_policy(args)
    options = {"target": np.asarray(args.target)} if args.target is not None else None
    env = ReachEnv(robot=args.model, render_mode=None if args.headless else "human")
    print(f"Reaching on {args.model} with {description} ({args.episodes} episodes)")
    try:
        results = run_episodes(env, policy_fn, args.episodes, args.seed,
                               pace=not args.headless, options=options)
    finally:
        env.close()

    success_rate = sum(r["success"] for r in results) / len(results)
    mean_dist = float(np.mean([r["dist_mm"] for r in results]))
    print(f"\n  success_rate={success_rate:.2%}  mean_final_dist={mean_dist:.1f} mm  "
          f"mean_steps={np.mean([r['steps'] for r in results]):.1f}")

    ok = success_rate >= args.min_success_rate and mean_dist <= args.max_mean_dist_mm
    if args.random:
        print("\nRL reaching BASELINE (untrained policy, not judged)")
        sys.exit(0)
    print("\nRL reaching " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
