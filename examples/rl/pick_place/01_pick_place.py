#!/usr/bin/env python3
"""RL example 02: Tabletop cube pick-and-place on xArm6 + Gripper G2 (Batch 8).

Loads a trained PPO policy (default: build/rl/pick_place/xarm6/ppo_pick_place.zip),
the independent scripted expert (--expert), or an untrained baseline (--random /
--zero-policy), and rolls out episodes in PickPlaceEnv.

Usage:
    python examples/rl/pick_place/01_pick_place.py                      # GUI, trained policy
    python examples/rl/pick_place/01_pick_place.py --headless           # headless evaluation
    python examples/rl/pick_place/01_pick_place.py --expert --headless  # scripted expert baseline
    python examples/rl/pick_place/01_pick_place.py --random --headless  # untrained random baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from ufactory_mujoco.rl import (  # noqa: E402
    PICK_PLACE_NUM_ACTIONS,
    PickPlaceEnv,
    PickPlaceExpert,
    build_pick_place_task_configs,
    load_validated_ppo,
)
from examples.rl.pick_place.evaluate import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_RECIPE,
    acceptance_passed,
    evaluate_policy,
    save_results,
)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="RL tabletop cube pick-and-place on xArm6 + Gripper G2.")
    ap.add_argument("--policy", "--checkpoint", dest="policy", type=Path, default=None,
                    help="path to ppo_pick_place.zip (default: build/rl/pick_place/xarm6/ppo_pick_place.zip)")
    ap.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE,
                    help="path to pick-place recipe.yaml")
    ap.add_argument("--episodes", type=int, default=5, help="evaluation episodes (default: 5)")
    ap.add_argument("--seed", type=int, default=17000)
    ap.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    ap.add_argument("--headless", action="store_true", help="no viewer, print metrics")
    ap.add_argument("--expert", action="store_true", help="run independent scripted expert baseline")
    ap.add_argument("--random", action="store_true", help="run untrained random baseline")
    ap.add_argument("--zero-policy", action="store_true", help="run all-zero command-hold baseline")
    ap.add_argument("--min-success-rate", type=float, default=0.90,
                    help="PASS success-rate threshold (default: 0.90)")
    ap.add_argument("--max-press-mm", type=float, default=1.0,
                    help="PASS max touchdown pressing bound in mm (default: 1.0)")
    ap.add_argument("--max-target-error-mm", type=float, default=20.0,
                    help="PASS mean target placement error bound in mm (default: 20.0)")
    ap.add_argument("--report", "--summary-json", "--out", dest="report", type=Path, default=None,
                    help="optional output JSON/CSV report path")
    args = ap.parse_args(argv)

    expected_env, expected_reward, _ = build_pick_place_task_configs("xarm6", recipe_path=args.recipe)
    env = PickPlaceEnv(
        robot="xarm6",
        render_mode=None if args.headless else "human",
        max_steps=int(expected_env["max_steps"]),
        decimation=int(expected_env["decimation"]),
        action_scale=float(expected_env["action_scale"]),
        action_clip=float(expected_env["action_clip"]),
        gap_action_scale=float(expected_env["gap_action_scale"]),
        setpoint_leash_m=float(expected_env["setpoint_leash_m"]),
        pre_grasp_gap_m=float(expected_env["pre_grasp_gap_m"]),
        close_floor_gap_m=float(expected_env["close_floor_gap_m"]),
        workspace_lower=expected_env["workspace_lower"],
        workspace_upper=expected_env["workspace_upper"],
        reward_gamma=float(expected_env["reward_gamma"]),
        place_clearance_m=float(expected_env["place_clearance_m"]),
        release_max_height_m=float(expected_env["release_max_height_m"]),
        press_limit_m=float(expected_env["press_limit_m"]),
        near_table_m=float(expected_env["near_table_m"]),
        max_near_table_descent_m_s=float(expected_env["max_near_table_descent_m_s"]),
        table_overload_margin_n=float(expected_env["table_overload_margin_n"]),
        place_success_dist_m=float(expected_env["place_success_dist_m"]),
        drag_limit_m=float(expected_env["drag_limit_m"]),
        tilt_limit_deg=float(expected_env["tilt_limit_deg"]),
        success_hold_steps=int(expected_env["success_hold_steps"]),
        reward_scales=expected_reward,
    )

    extra_meta = {"task": "pick_place", "robot": "xarm6"}
    try:
        if args.random:
            rng = np.random.default_rng(args.seed)

            def policy_fn(_obs: np.ndarray) -> np.ndarray:
                return rng.uniform(-1.0, 1.0, PICK_PLACE_NUM_ACTIONS).astype(np.float32)

            description = "random baseline"
            extra_meta["mode"] = "random"
        elif args.zero_policy:

            def policy_fn(_obs: np.ndarray) -> np.ndarray:
                return np.zeros(PICK_PLACE_NUM_ACTIONS, dtype=np.float32)

            description = "zero policy baseline"
            extra_meta["mode"] = "zero_policy"
        elif args.expert:
            policy_fn = PickPlaceExpert(env)
            description = "scripted expert baseline"
            extra_meta["mode"] = "expert"
        else:
            policy_path = args.policy or DEFAULT_CHECKPOINT
            if not policy_path.is_file():
                sys.exit(
                    f"policy not found: {policy_path} -- "
                    "train first with scripts/train_pick_place_rl.py "
                    "or use --expert / --random"
                )
            try:
                model, artifact, manifest = load_validated_ppo(
                    policy_path,
                    device=args.device,
                    expected_task="pick_place",
                    expected_robot="xarm6",
                    expected_runtime_config_sha256=expected_env["runtime_config_sha256"],
                    expected_runtime_env=expected_env,
                )
            except ImportError:
                sys.exit(
                    "stable-baselines3 not installed (requirements-rl.txt); "
                    "use --expert or --random for baselines"
                )

            def policy_fn(obs: np.ndarray) -> np.ndarray:
                return model.predict(obs, deterministic=True)[0]

            description = f"PPO policy {policy_path}"
            extra_meta.update(
                {
                    "mode": "policy",
                    "checkpoint": str(policy_path),
                    "checkpoint_sha256": manifest.checkpoint_sha256,
                    "config_sha256": artifact["config_sha256"],
                    "runtime_config_sha256": expected_env["runtime_config_sha256"],
                }
            )

        print(f"Pick-and-place on xarm6+gripper_g2 with {description} ({args.episodes} episodes)")
        summary, rows = evaluate_policy(
            env,
            policy_fn,
            episodes=args.episodes,
            seed=args.seed,
            pace=not args.headless,
        )
    finally:
        env.close()

    for r in rows:
        print(
            f"  episode {int(r['scenario_id']) + 1}/{args.episodes}: "
            f"success={r['success']}  lifted={r['ever_lifted_clear']}  "
            f"release_ok={r['release_confirmed']}  target_err={r['target_error_mm']:.1f} mm  "
            f"press={r['press_mm']:.3f} mm  steps={r['steps']}  return={r['return']:.1f}"
        )

    print(
        f"\n  success_rate={summary['success_rate']:.2%}  "
        f"lifted_rate={summary['ever_lifted_clear_rate']:.2%}  "
        f"release_confirmed_rate={summary['release_confirmed_rate']:.2%}  "
        f"mean_target_error={summary['target_error_mm']['mean']:.1f} mm  "
        f"max_press={summary['press_mm']['max']:.3f} mm"
    )

    ok = acceptance_passed(
        summary,
        min_success_rate=args.min_success_rate,
        max_press_mm=args.max_press_mm,
        max_target_error_mm=args.max_target_error_mm,
    )
    summary["passed"] = True if (args.random or args.zero_policy) else ok
    if args.report is not None:
        save_results(args.report, summary, rows, extra_metadata=extra_meta)

    if args.random or args.zero_policy:
        print("\nRL pick-place BASELINE (untrained policy, not judged)")
        sys.exit(0)
    print("\nRL pick-place " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

