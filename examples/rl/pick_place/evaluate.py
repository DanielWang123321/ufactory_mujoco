#!/usr/bin/env python3
"""Strict tabletop pick-and-place evaluation for PPO checkpoints, the scripted expert, or baselines.

Usage:
    python examples/rl/pick_place/evaluate.py --expert --episodes 10 --quiet-episodes
    python examples/rl/pick_place/evaluate.py --zero-policy --episodes 5 --quiet-episodes
    python examples/rl/pick_place/evaluate.py --checkpoint build/rl/pick_place/xarm6_g2_v2/ppo_pick_place.zip \
        --episodes 20 --min-success-rate 0.90 --max-press-mm 1.0
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from ufactory_mujoco.rl import (  # noqa: E402
    PICK_PLACE_NUM_ACTIONS,
    PickPlaceEnv,
    build_pick_place_task_configs,
    load_validated_ppo,
)
from ufactory_mujoco.rl.evaluation import (  # noqa: E402,F401  (re-exported for callers of this script)
    acceptance_passed,
    checkpoint_score,
    evaluate_policy,
    make_spawns,
    pick_place_env_kwargs,
    save_results,
    spawn_sha256,
    summarize,
)

DEFAULT_RECIPE = Path(__file__).resolve().with_name("recipe.yaml")
DEFAULT_CHECKPOINT = _ROOT / "build" / "rl" / "pick_place" / "xarm6_g2_v2" / "ppo_pick_place.zip"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", action="store_true", help="run the independent scripted physics baseline")
    parser.add_argument("--zero-policy", action="store_true", help="run an all-zero policy baseline")
    parser.add_argument("--random", action="store_true", help="run a uniform random policy baseline")
    parser.add_argument("--checkpoint", "--ckpt", "--policy", dest="checkpoint", type=Path, default=None,
                        help="path to trained ppo_pick_place.zip")
    parser.add_argument("--robot", default="xarm6")
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--runtime-config", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-success-rate", type=float, default=None)
    parser.add_argument("--max-target-error-mm", type=float, default=20.0)
    parser.add_argument("--max-press-mm", type=float, default=1.0)
    parser.add_argument("--summary-json", "--out", "--report", dest="summary_json", type=Path, default=None,
                        help="output JSON/CSV path")
    parser.add_argument("--quiet-episodes", action="store_true")
    args = parser.parse_args(argv)

    mode_count = sum(bool(x) for x in (args.expert, args.zero_policy, args.random, args.checkpoint is not None))
    if mode_count > 1:
        parser.error("Specify at most one of --expert, --zero-policy, --random, or --checkpoint")
    if mode_count == 0:
        args.checkpoint = DEFAULT_CHECKPOINT
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.max_press_mm < 0 or (args.min_success_rate is not None and not (0.0 <= args.min_success_rate <= 1.0)):
        parser.error("require --max-press-mm >= 0 and --min-success-rate in [0, 1]")

    expected_env, expected_reward, _train = build_pick_place_task_configs(
        args.robot,
        recipe_path=args.recipe,
        config_path=args.runtime_config,
    )

    extra_meta: dict[str, Any] = {"task": "pick_place", "robot": args.robot}
    policy_fn: Callable[[np.ndarray], np.ndarray] | None = None

    if args.zero_policy:
        extra_meta["mode"] = "zero_policy"

        def _zero_policy(_obs: np.ndarray) -> np.ndarray:
            return np.zeros(PICK_PLACE_NUM_ACTIONS, dtype=np.float32)

        policy_fn = _zero_policy
    elif args.random:
        extra_meta["mode"] = "random"
        rng = np.random.default_rng(args.seed)

        def _random_policy(_obs: np.ndarray) -> np.ndarray:
            return rng.uniform(-1.0, 1.0, PICK_PLACE_NUM_ACTIONS).astype(np.float32)

        policy_fn = _random_policy
    elif args.expert:
        extra_meta["mode"] = "expert"
        policy_fn = None
    else:
        assert args.checkpoint is not None
        model, artifact, manifest = load_validated_ppo(
            args.checkpoint,
            device=args.device,
            expected_task="pick_place",
            expected_robot=args.robot,
            expected_runtime_config_sha256=expected_env["runtime_config_sha256"],
            expected_runtime_env=expected_env,
        )
        extra_meta.update(
            {
                "mode": "checkpoint",
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": manifest.checkpoint_sha256,
                "config_sha256": artifact["config_sha256"],
                "runtime_config_sha256": expected_env["runtime_config_sha256"],
            }
        )

        def _ppo_policy(obs: np.ndarray) -> np.ndarray:
            return model.predict(obs, deterministic=True)[0]

        policy_fn = _ppo_policy

    env = PickPlaceEnv(**pick_place_env_kwargs(
        expected_env, expected_reward, robot=args.robot, config_path=args.runtime_config))
    try:
        summary, rows = evaluate_policy(env, policy_fn, episodes=args.episodes, seed=args.seed)
    finally:
        env.close()

    default_sr = 0.0 if (args.zero_policy or args.random) else 0.90
    min_sr = args.min_success_rate if args.min_success_rate is not None else default_sr
    passed = True if (args.zero_policy or args.random) else acceptance_passed(
        summary,
        min_success_rate=min_sr,
        max_press_mm=args.max_press_mm,
        max_target_error_mm=args.max_target_error_mm,
    )
    summary["passed"] = passed
    if args.summary_json is not None:
        save_results(args.summary_json, summary, rows, extra_metadata=extra_meta)

    if not args.quiet_episodes:
        for row in rows:
            print(
                f"scenario={int(row['scenario_id']):04d} success={int(bool(row['success']))} "
                f"target_err={float(row['target_error_mm']):.2f}mm "
                f"drag={float(row['max_drag_mm']):.2f}mm tilt={float(row['max_tilt_deg']):.2f}deg "
                f"press={float(row['press_mm']):.3f}mm release={float(row['release_height_mm']):.2f}mm "
                f"steps={int(row['steps'])}"
            )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()

