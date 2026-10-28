#!/usr/bin/env python3
"""Strict tabletop pick-and-place evaluation for PPO checkpoints, expert, or baselines (Batch 8).

Usage:
    python examples/rl/pick_place/evaluate.py --expert --episodes 10 --quiet-episodes
    python examples/rl/pick_place/evaluate.py --zero-policy --episodes 5 --quiet-episodes
    python examples/rl/pick_place/evaluate.py --checkpoint build/rl/pick_place/xarm6/ppo_pick_place.zip \
        --episodes 20 --min-success-rate 0.90 --max-press-mm 1.0 --out build/rl/pick_place/xarm6/eval.json
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

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

DEFAULT_RECIPE = Path(__file__).resolve().with_name("recipe.yaml")
DEFAULT_CHECKPOINT = _ROOT / "build" / "rl" / "pick_place" / "xarm6" / "ppo_pick_place.zip"


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate boolean outcome rates and continuous metric distributions."""
    if not records:
        raise RuntimeError("evaluation finished no episodes")
    summary: dict[str, Any] = {"episodes": len(records)}
    for key in (
        "success",
        "drag_failed",
        "tilt_failed",
        "dropped",
        "press_failed",
        "high_release",
        "collision_failed",
        "timeout",
        "release_confirmed",
        "ever_carried",
        "ever_lifted_clear",
        "touchdown_seen",
    ):
        count = sum(int(bool(r[key])) for r in records)
        summary[key] = count
        summary[f"{key}_rate"] = float(count / len(records))
    for key in (
        "max_drag_mm",
        "max_tilt_deg",
        "target_error_mm",
        "steps",
        "press_mm",
        "release_height_mm",
        "max_held_clearance_mm",
        "max_table_force_n",
        "max_place_table_force_n",
        "max_held_table_force_n",
        "max_robot_table_force_n",
        "return",
    ):
        values = [float(r[key]) for r in records]
        summary[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
    return summary


def checkpoint_score(summary: dict[str, Any]) -> tuple[float, float, float, float, float]:
    """Rank checkpoints by strict success, low release, touchdown, low press, and landing error."""
    no_high = 1.0 - float(summary.get("high_release_rate", 0.0))
    low_release = float(summary.get("release_confirmed_rate", 0.0)) * no_high
    target_touchdown = (
        float(summary.get("touchdown_seen_rate", 1.0))
        * no_high
        * float(summary["target_error_mm"]["mean"] <= 20.0)
    )
    return (
        float(summary["success_rate"]),
        float(low_release),
        float(target_touchdown),
        -float(summary["press_mm"]["max"]),
        -float(summary["target_error_mm"]["mean"]),
    )


def acceptance_passed(
    summary: dict[str, Any],
    *,
    min_success_rate: float = 0.90,
    max_press_mm: float = 1.0,
    max_target_error_mm: float | None = 20.0,
) -> bool:
    ok = bool(summary["success_rate"] >= min_success_rate and summary["press_mm"]["max"] <= max_press_mm)
    if max_target_error_mm is not None:
        ok = ok and bool(summary["target_error_mm"]["mean"] <= max_target_error_mm)
    return ok


def evaluate_policy(
    env: PickPlaceEnv,
    policy_fn: Callable[[np.ndarray], np.ndarray] | None,
    *,
    episodes: int = 10,
    seed: int = 17000,
    pace: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if policy_fn is None:
        expert = PickPlaceExpert(env)
        policy_fn = expert

    rows: list[dict[str, Any]] = []
    start = time.monotonic()
    total_steps = 0
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            t0 = time.perf_counter()
            action = policy_fn(obs)
            obs, _reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            done = bool(terminated or truncated)
            if pace:
                time.sleep(max(0.0, env.ctrl_dt - (time.perf_counter() - t0)))
        row = dict(info)
        row["scenario_id"] = ep
        row["seed"] = seed + ep
        rows.append(row)

    summary = summarize(rows)
    elapsed = max(1e-9, time.monotonic() - start)
    summary["seconds"] = round(elapsed, 3)
    summary["transitions_per_second"] = round(total_steps / elapsed, 1)
    summary["seed"] = int(seed)
    summary["env_version"] = int(env.env_version)
    return summary, rows


def save_results(
    path: str | Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(summary)
    if extra_metadata:
        payload.update(extra_metadata)
    payload["episodes_detail"] = rows
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if rows:
        csv_rows = [
            {k: (json.dumps(v) if isinstance(v, list | dict) else v) for k, v in row.items()}
            for row in rows
        ]
        with target.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)


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

    env = PickPlaceEnv(
        robot=args.robot,
        config_path=args.runtime_config,
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
    try:
        summary, rows = evaluate_policy(env, policy_fn, episodes=args.episodes, seed=args.seed)
    finally:
        env.close()

    min_sr = args.min_success_rate if args.min_success_rate is not None else (0.0 if (args.zero_policy or args.random) else 0.90)
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

