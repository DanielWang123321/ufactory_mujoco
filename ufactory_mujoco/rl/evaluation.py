"""Evaluation helpers shared by the RL training scripts, examples and tests.

Pick-and-place: deterministic spawn sampling, rollout, summary statistics,
checkpoint ranking and acceptance gate. Reach: per-episode rollout records and
their summary. Only mujoco + numpy are needed at import time.
"""

from __future__ import annotations

from collections.abc import Callable
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .model_hash import verify_runtime_model
from .pick_place_env import SPAWN_XY_LOWER, SPAWN_XY_UPPER, PickPlaceEnv, PickPlaceExpert
from .reach_env import ReachEnv

__all__ = [
    "acceptance_passed",
    "checkpoint_score",
    "evaluate_policy",
    "evaluate_reach_policy",
    "make_spawns",
    "pick_place_env_kwargs",
    "save_results",
    "spawn_sha256",
    "summarize",
    "summarize_reach_reports",
]


def pick_place_env_kwargs(
    env_cfg: dict[str, Any],
    reward_cfg: dict[str, float],
    *,
    robot: str = "xarm6",
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build ``PickPlaceEnv`` keyword arguments from a resolved recipe environment and reward config."""
    return {
        "robot": robot,
        "config_path": config_path,
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
        "spawn_xy_lower": env_cfg["spawn_xy_lower"],
        "spawn_xy_upper": env_cfg["spawn_xy_upper"],
        "g2_speed_percent": float(env_cfg["g2_speed_percent"]),
        "g2_force_percent": float(env_cfg["g2_force_percent"]),
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
        "expected_runtime_model_sha256": env_cfg["runtime_model_sha256"],
    }


def make_spawns(
    episodes: int,
    *,
    seed: int = 17000,
    lower: tuple[float, float] | list[float] | np.ndarray = SPAWN_XY_LOWER,
    upper: tuple[float, float] | list[float] | np.ndarray = SPAWN_XY_UPPER,
) -> np.ndarray:
    """Deterministically sample ``(episodes, 2)`` cube XY spawns independent of policy RNG."""
    if int(episodes) < 1:
        raise ValueError("episodes must be positive")
    lo = np.asarray(lower, dtype=np.float64).reshape(2)
    hi = np.asarray(upper, dtype=np.float64).reshape(2)
    if np.any(lo > hi):
        raise ValueError("spawn lower bounds must not exceed upper bounds")
    rng = np.random.default_rng(int(seed))
    return lo + rng.random((int(episodes), 2)) * (hi - lo)


def spawn_sha256(spawns: np.ndarray) -> str:
    """Stable SHA-256 digest of an evaluation spawn matrix rounded to 1e-8 m."""
    arr = np.ascontiguousarray(np.round(np.asarray(spawns, dtype=np.float64), 8))
    return hashlib.sha256(arr.tobytes()).hexdigest()


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


def checkpoint_score(summary: dict[str, Any]) -> tuple[float, ...]:
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
        float(summary.get("ever_lifted_clear_rate", 0.0)),
        float(summary.get("ever_carried_rate", 0.0)),
        float(summary["return"]["mean"]),
    )


def acceptance_passed(
    summary: dict[str, Any],
    *,
    min_success_rate: float = 0.90,
    max_press_mm: float = 1.0,
    max_target_error_mm: float | None = 20.0,
) -> bool:
    """Return True when success rate, maximum press depth and mean landing error meet the thresholds."""
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
    """Roll out ``episodes`` on deterministic spawns; ``policy_fn=None`` runs the scripted expert."""
    verify_runtime_model(env.scene_path, env.runtime_model_sha256)
    if policy_fn is None:
        expert = PickPlaceExpert(env)
        policy_fn = expert

    spawns = make_spawns(
        episodes,
        seed=seed,
        lower=env.spawn_xy_lower,
        upper=env.spawn_xy_upper,
    )
    env.set_evaluation_spawns(spawns)
    rows: list[dict[str, Any]] = []
    start = time.monotonic()
    total_steps = 0
    try:
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
    finally:
        env.set_evaluation_spawns(None)

    summary = summarize(rows)
    elapsed = max(1e-9, time.monotonic() - start)
    summary["seconds"] = round(elapsed, 3)
    summary["transitions_per_second"] = round(total_steps / elapsed, 1)
    summary["seed"] = int(seed)
    summary["spawn_sha256"] = spawn_sha256(spawns)
    summary["spawn_xy_lower"] = env.spawn_xy_lower.tolist()
    summary["spawn_xy_upper"] = env.spawn_xy_upper.tolist()
    summary["env_version"] = int(env.env_version)
    summary["runtime_model_sha256"] = verify_runtime_model(env.scene_path, env.runtime_model_sha256)
    return summary, rows


def save_results(
    path: str | Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Write the summary plus per-episode rows to JSON, and the rows to a sibling CSV."""
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


def summarize_reach_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate reach episode records into success rate and terminal-distance statistics."""
    if not reports:
        raise RuntimeError("evaluation finished no episodes")
    dists_m = np.asarray([float(r["terminal_distance_m"]) for r in reports], dtype=float)
    steps = np.asarray([int(r["steps"]) for r in reports], dtype=float)
    elapsed = np.asarray([float(r["elapsed_s"]) for r in reports], dtype=float)
    successes = [r for r in reports if r["success"]]
    return {
        "episodes": len(reports),
        "successes": len(successes),
        "success_rate": float(len(successes) / len(reports)),
        "mean_m": float(np.mean(dists_m)),
        "median_m": float(np.median(dists_m)),
        "p90_m": float(np.percentile(dists_m, 90)),
        "p99_m": float(np.percentile(dists_m, 99)),
        "max_m": float(np.max(dists_m)),
        "mean_mm": float(np.mean(dists_m) * 1000.0),
        "mean_steps": float(np.mean(steps)),
        "mean_elapsed_s": float(np.mean(elapsed)),
        "mean_success_elapsed_s": (
            float(np.mean([float(r["elapsed_s"]) for r in successes])) if successes else None
        ),
    }


def evaluate_reach_policy(
    env: ReachEnv,
    policy_fn,
    *,
    episodes: int,
    seed: int,
    options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Roll out reach episodes with seeds ``seed + i`` and return one record per episode."""
    ctrl_dt = env._decimation * float(env.model.opt.timestep)
    reports: list[dict[str, Any]] = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep, options=options)
        done = False
        total_return = 0.0
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            done = bool(terminated or truncated)
        reports.append(
            {
                "index": ep,
                "seed": seed + ep,
                "target_m": [float(x) for x in info["target"]],
                "success": bool(info["success"]),
                "timeout": not bool(info["success"]),
                "terminal_distance_m": float(info["dist"]),
                "dist_mm": float(info["dist"]) * 1000.0,
                "steps": int(info["steps"]),
                "elapsed_s": round(int(info["steps"]) * ctrl_dt, 4),
                "return": float(total_return),
            }
        )
    return reports
