#!/usr/bin/env python3
"""Strict checkpoint evaluation for the five-arm TCP reaching task.

Validates the checkpoint manifest and config hashes before loading the SB3 PPO
policy, evaluates on deterministic seeds, and writes a structured JSON summary.

Usage:
    python examples/rl/reach/evaluate.py --model xarm6 --episodes 50
    python examples/rl/reach/evaluate.py --model xarm6 --episodes 512 --seed 17000 \
        --min-success-rate 0.95 --max-mean-dist-mm 10.0 --summary-json build/rl/xarm6/eval.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from ufactory_mujoco.rl import (  # noqa: E402
    ROBOT_PROFILES,
    ReachEnv,
    build_reach_task_configs,
    load_validated_ppo,
)
from ufactory_mujoco.rl.evaluation import (  # noqa: E402
    evaluate_reach_policy,
    summarize_reach_reports,
)

DEFAULT_RECIPE = Path(__file__).resolve().with_name("recipe.yaml")
_BUILD = _ROOT / "build" / "rl"


def _evaluate_single_robot(
    robot: str,
    *,
    checkpoint: Path | None,
    recipe: Path,
    episodes: int,
    seed: int,
    device: str,
    min_success_rate: float,
    max_mean_m: float,
    quiet_episodes: bool,
) -> dict[str, Any]:
    ckpt_path = checkpoint or (_BUILD / robot / "ppo_reach.zip")
    expected_env, _reward, _train = build_reach_task_configs(robot, recipe_path=recipe)
    model, artifact, manifest = load_validated_ppo(
        ckpt_path,
        device=device,
        expected_task="reach",
        expected_robot=robot,
        expected_runtime_env=expected_env,
    )

    env = ReachEnv(
        robot=robot,
        max_steps=int(expected_env["max_steps"]),
        decimation=int(expected_env["decimation"]),
        delta=float(expected_env["delta"]),
        success_dist=float(expected_env["success_dist"]),
        q0_noise=float(expected_env["q0_noise"]),
        validate_targets=bool(expected_env["validate_targets"]),
    )
    try:
        reports = evaluate_reach_policy(
            env,
            lambda obs: model.predict(obs, deterministic=True)[0],
            episodes=episodes,
            seed=seed,
        )
    finally:
        env.close()

    summary = summarize_reach_reports(reports)
    passed = bool(summary["success_rate"] >= min_success_rate and summary["mean_m"] <= max_mean_m)
    if not quiet_episodes:
        for r in reports:
            print(
                f"robot={robot} target={r['index']:04d} success={int(r['success'])} "
                f"distance={r['dist_mm']:.1f} mm time={r['elapsed_s']:.2f} s"
            )
    print(
        f"[{robot}] episodes={summary['episodes']} "
        f"success={summary['success_rate']:.3f} "
        f"mean={summary['mean_m'] * 1000:.1f} mm "
        f"median={summary['median_m'] * 1000:.1f} mm "
        f"p90={summary['p90_m'] * 1000:.1f} mm "
        f"p99={summary['p99_m'] * 1000:.1f} mm"
    )
    return {
        "schema_version": 1,
        "task": "reach",
        "robot": robot,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "config_sha256": artifact["config_sha256"],
        "seed": seed,
        "thresholds": {
            "min_success_rate": min_success_rate,
            "max_mean_distance_m": max_mean_m,
        },
        "passed": passed,
        "summary": summary,
        "episodes": reports,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Reach PPO checkpoint under strict artifact contracts.")
    parser.add_argument("--model", "--robot", dest="model", default="xarm6", choices=[*sorted(ROBOT_PROFILES), "all"],
                        help="robot model (default: xarm6, or 'all')")
    parser.add_argument("--checkpoint", "--policy", dest="checkpoint", type=Path, default=None,
                        help="path to ppo_reach.zip (default: build/rl/<model>/ppo_reach.zip)")
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE,
                        help="path to reach recipe.yaml")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-success-rate", type=float, default=0.95)
    parser.add_argument("--max-mean-dist-mm", type=float, default=10.0)
    parser.add_argument("--max-mean-distance-m", "--max-mean-dist-m", dest="max_mean_distance_m", type=float,
                        default=None,
                        help="optional bound in meters (overrides --max-mean-dist-mm when given)")
    parser.add_argument("--summary-json", "--report", "--out", dest="summary_json", type=Path, default=None,
                        help="save evaluation summary, episode records, and artifact hashes to JSON")
    parser.add_argument("--quiet-episodes", action="store_true",
                        help="only print the aggregate summary")
    args = parser.parse_args(argv)

    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.min_success_rate is not None and not (0.0 <= args.min_success_rate <= 1.0):
        parser.error("--min-success-rate must be between 0 and 1")
    if args.model == "all" and args.checkpoint is not None:
        parser.error("--checkpoint cannot be used with --model all")
    max_mean_m = (
        float(args.max_mean_distance_m)
        if args.max_mean_distance_m is not None
        else float(args.max_mean_dist_mm) / 1000.0
    )
    if not math.isfinite(max_mean_m) or max_mean_m <= 0.0:
        parser.error("maximum mean distance must be finite and positive")

    if args.model == "all":
        results = {}
        all_passed = True
        for robot in sorted(ROBOT_PROFILES):
            res = _evaluate_single_robot(
                robot,
                checkpoint=None,
                recipe=args.recipe,
                episodes=args.episodes,
                seed=args.seed,
                device=args.device,
                min_success_rate=args.min_success_rate,
                max_mean_m=max_mean_m,
                quiet_episodes=args.quiet_episodes,
            )
            results[robot] = res
            all_passed = all_passed and bool(res["passed"])
        payload: dict[str, Any] = {
            "schema_version": 1,
            "task": "reach",
            "passed": all_passed,
            "robots": results,
        }
        passed = all_passed
    else:
        payload = _evaluate_single_robot(
            args.model,
            checkpoint=args.checkpoint,
            recipe=args.recipe,
            episodes=args.episodes,
            seed=args.seed,
            device=args.device,
            min_success_rate=args.min_success_rate,
            max_mean_m=max_mean_m,
            quiet_episodes=args.quiet_episodes,
        )
        passed = bool(payload["passed"])

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    if not passed:
        print("Checkpoint failed the requested success rate or mean terminal distance threshold")
        sys.exit(1)


if __name__ == "__main__":
    main()
