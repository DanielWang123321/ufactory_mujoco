#!/usr/bin/env python3
"""Train a pure-PPO policy from scratch for xArm6 + Gripper G2 tabletop pick-and-place.

Reads defaults from examples/rl/pick_place/recipe.yaml (--recipe) and writes:
- <out-dir>/ppo_pick_place.zip
- <out-dir>/config.yaml
- <out-dir>/ppo_pick_place.checkpoint_manifest.json
- <out-dir>/provenance.json
- <out-dir>/metrics.json
- <out-dir>/eval_history.json

Usage:
    python scripts/train_pick_place_rl.py
    python scripts/train_pick_place_rl.py --n-envs 128 --n-workers 28 --device cuda
    python scripts/train_pick_place_rl.py --total-steps 256 --n-envs 1
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

# Prevent OpenMP / BLAS thread oversubscription across parallel MuJoCo workers
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from examples.rl.pick_place.evaluate import (  # noqa: E402
    acceptance_passed,
    checkpoint_score,
    evaluate_policy,
)
from ufactory_mujoco.config import load_task_config  # noqa: E402
from ufactory_mujoco.resource_paths import CONFIGS_ROOT  # noqa: E402
from ufactory_mujoco.rl import (  # noqa: E402
    PICK_PLACE_ACTION_CONTRACT,
    PickPlaceEnv,
    build_pick_place_task_configs,
    load_training_config,
    validate_checkpoint_artifacts,
    write_checkpoint_manifest,
    write_run_provenance,
    write_training_config,
)
from ufactory_mujoco.rl.model_hash import verify_runtime_model  # noqa: E402

DEFAULT_RECIPE = ROOT / "examples" / "rl" / "pick_place" / "recipe.yaml"
ENV_MEMORY_MB = 180


def _available_memory_mb() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def auto_n_envs(cpu_count: int | None = None, avail_mb: int | None = None, reserve_mb: int = 2000) -> int:
    cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    n = min(cpus, 32)
    avail = avail_mb if avail_mb is not None else _available_memory_mb()
    if avail is not None:
        n = min(n, max(1, (avail - reserve_mb) // ENV_MEMORY_MB))
    return max(1, n)


def _parse_xy_pair(raw: str | None, flag: str) -> list[float] | None:
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"{flag} expects 'x,y', got {raw!r}")
    vals = [float(parts[0]), float(parts[1])]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{flag} values must be finite")
    return vals


def _env_kwargs(
    robot: str,
    runtime_config: Path | None,
    env_cfg: dict[str, Any],
    reward_cfg: dict[str, float],
    *,
    fast_info: bool = False,
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
        "fast_info": bool(fast_info),
    }


def _provenance_sources(recipe_path: Path, runtime_config_path: Path) -> list[Path]:
    return [
        Path(__file__).resolve(),
        recipe_path.resolve(),
        runtime_config_path.resolve(),
        ROOT / "ufactory_mujoco" / "rl" / "pick_place_env.py",
        ROOT / "ufactory_mujoco" / "rl" / "artifacts.py",
        ROOT / "ufactory_mujoco" / "g2_model.py",
        ROOT / "ufactory_mujoco" / "g2_controller.py",
        ROOT / "ufactory_mujoco" / "grippers.py",
        ROOT / "ufactory_mujoco" / "ik.py",
    ]


def _grouped_worker_loop(
    remote: Any,
    parent_remote: Any,
    env_kwargs: dict[str, Any],
    seeds: list[int],
) -> None:
    """Worker process hosting a batch of PickPlaceEnv instances to amortize IPC overhead."""
    parent_remote.close()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    envs = [PickPlaceEnv(**env_kwargs) for _ in seeds]
    t_starts = [time.perf_counter() for _ in seeds]
    pending_seeds: list[int | None] = list(seeds)

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                if data is not None:
                    pending_seeds = list(data)
                obs_batch = []
                for idx, env in enumerate(envs):
                    s = pending_seeds[idx]
                    obs, _ = env.reset(seed=s)
                    pending_seeds[idx] = None
                    t_starts[idx] = time.perf_counter()
                    obs_batch.append(obs)
                remote.send(np.stack(obs_batch, axis=0))
            elif cmd == "step":
                actions = np.asarray(data, dtype=np.float32)
                obs_batch = []
                rew_batch = np.empty(len(envs), dtype=np.float32)
                done_batch = np.empty(len(envs), dtype=bool)
                infos: list[dict[str, Any]] = []
                for idx, env in enumerate(envs):
                    obs, reward, terminated, truncated, info = env.step(actions[idx])
                    done = bool(terminated or truncated)
                    rew_batch[idx] = float(reward)
                    done_batch[idx] = done
                    if done:
                        # time_remaining is part of obs, so finite-horizon timeout is Markovian without value bootstrap
                        step_info: dict[str, Any] = {
                            "terminal_observation": obs.copy(),
                            "TimeLimit.truncated": False,
                            "episode": {
                                "r": round(float(info.get("return", env.ep_return)), 6),
                                "l": int(info.get("steps", env.step_count)),
                                "t": round(time.perf_counter() - t_starts[idx], 4),
                            },
                            "success": bool(info.get("success", False)),
                        }
                        obs, _ = env.reset()
                        t_starts[idx] = time.perf_counter()
                        infos.append(step_info)
                    else:
                        infos.append({})
                    obs_batch.append(obs)
                remote.send((np.stack(obs_batch, axis=0), rew_batch, done_batch, infos))
            elif cmd == "seed":
                pending_seeds = list(data)
                remote.send(pending_seeds)
            elif cmd == "get_spaces":
                remote.send((envs[0].observation_space, envs[0].action_space))
            elif cmd == "get_attr":
                remote.send([getattr(env, data) for env in envs])
            elif cmd == "set_attr":
                attr_name, value = data
                for env in envs:
                    setattr(env, attr_name, value)
                remote.send(None)
            elif cmd == "close":
                remote.close()
                break
            else:
                raise NotImplementedError(f"unsupported worker command: {cmd!r}")
    except EOFError:
        pass
    finally:
        for env in envs:
            env.close()


def _make_grouped_vec_env(
    *,
    n_envs: int,
    n_workers: int,
    seed: int,
    env_kwargs: dict[str, Any],
) -> Any:
    """Build either a single-process DummyVecEnv or a multi-env-per-worker GroupedSubprocVecEnv."""
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv

    if n_envs <= 1 or n_workers <= 1:
        def _make_single(rank: int):
            def _init():
                kw = dict(env_kwargs, fast_info=False)
                env = PickPlaceEnv(**kw)
                env.reset(seed=seed + rank)
                return Monitor(env)

            return _init

        return DummyVecEnv([_make_single(i) for i in range(n_envs)])

    actual_workers = min(int(n_workers), int(n_envs))
    counts = [n_envs // actual_workers + (1 if w < (n_envs % actual_workers) else 0) for w in range(actual_workers)]
    slices: list[tuple[int, int]] = []
    cursor = 0
    for count in counts:
        slices.append((cursor, cursor + count))
        cursor += count

    class GroupedSubprocVecEnv(VecEnv):
        def __init__(self) -> None:
            start_method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
            ctx = mp.get_context(start_method)
            self.waiting = False
            self.closed = False
            self._slices = slices
            self.remotes = []
            self.work_remotes = []
            self.processes = []

            worker_kw = dict(env_kwargs, fast_info=True)
            for start_idx, end_idx in self._slices:
                remote, work_remote = ctx.Pipe()
                self.remotes.append(remote)
                self.work_remotes.append(work_remote)
                worker_seeds = [int(seed + idx) for idx in range(start_idx, end_idx)]
                proc = ctx.Process(
                    target=_grouped_worker_loop,
                    args=(work_remote, remote, worker_kw, worker_seeds),
                    daemon=True,
                )
                proc.start()
                work_remote.close()
                self.processes.append(proc)

            self.remotes[0].send(("get_spaces", None))
            observation_space, action_space = self.remotes[0].recv()
            super().__init__(n_envs, observation_space, action_space)

        def step_async(self, actions: np.ndarray) -> None:
            for remote, (start_idx, end_idx) in zip(self.remotes, self._slices, strict=True):
                remote.send(("step", actions[start_idx:end_idx]))
            self.waiting = True

        def step_wait(self):
            results = [remote.recv() for remote in self.remotes]
            self.waiting = False
            obs_parts, rew_parts, done_parts, info_lists = zip(*results, strict=True)
            infos: list[dict[str, Any]] = []
            for part in info_lists:
                infos.extend(part)
            return (
                np.concatenate(obs_parts, axis=0),
                np.concatenate(rew_parts, axis=0),
                np.concatenate(done_parts, axis=0),
                infos,
            )

        def reset(self):
            for remote in self.remotes:
                remote.send(("reset", None))
            obs_parts = [remote.recv() for remote in self.remotes]
            return np.concatenate(obs_parts, axis=0)

        def seed(self, seed: int | None = None):
            base = 0 if seed is None else int(seed)
            out = []
            for remote, (start_idx, end_idx) in zip(self.remotes, self._slices, strict=True):
                s_list = [base + i for i in range(start_idx, end_idx)]
                remote.send(("seed", s_list))
                out.extend(remote.recv())
            return out

        def close(self) -> None:
            if self.closed:
                return
            if self.waiting:
                for remote in self.remotes:
                    remote.recv()
            for remote in self.remotes:
                remote.send(("close", None))
            for proc in self.processes:
                proc.join(timeout=5.0)
            self.closed = True

        def get_attr(self, attr_name: str, indices=None):
            target_indices = self._get_indices(indices)
            all_vals = []
            for remote in self.remotes:
                remote.send(("get_attr", attr_name))
                all_vals.extend(remote.recv())
            return [all_vals[i] for i in target_indices]

        def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
            for remote in self.remotes:
                remote.send(("set_attr", (attr_name, value)))
            for remote in self.remotes:
                remote.recv()

        def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
            raise NotImplementedError("env_method is not used by PPO training")

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False for _ in self._get_indices(indices)]

    return GroupedSubprocVecEnv()


def _strict_gate_passed(
    summary: dict[str, Any],
    *,
    min_success_rate: float,
    max_press_mm: float = 1.0,
    max_target_error_mm: float = 20.0,
) -> bool:
    """Require strict acceptance thresholds plus zero collision, drag, tilt, or high-release failures."""
    if not acceptance_passed(
        summary,
        min_success_rate=min_success_rate,
        max_press_mm=max_press_mm,
        max_target_error_mm=max_target_error_mm,
    ):
        return False
    return (
        int(summary.get("collision_failed", 0)) == 0
        and int(summary.get("drag_failed", 0)) == 0
        and int(summary.get("tilt_failed", 0)) == 0
        and int(summary.get("high_release", 0)) == 0
    )


def write_pick_place_artifacts(
    out_dir: Path,
    *,
    robot: str,
    recipe_path: Path,
    runtime_config: Path | None = None,
    env_overrides: dict[str, Any] | None = None,
    train_overrides: dict[str, Any] | None = None,
    expected_model_sha256: str,
    expected_env: dict[str, Any],
    expected_reward: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    """Write config.yaml, ppo_pick_place.checkpoint_manifest.json, and provenance.json."""
    env_cfg, reward_cfg, train_cfg = build_pick_place_task_configs(
        robot,
        recipe_path=recipe_path,
        config_path=runtime_config,
    )
    if env_overrides:
        env_cfg.update(env_overrides)
    if env_cfg["runtime_model_sha256"] != expected_model_sha256:
        raise ValueError("runtime model changed during training; refusing to save artifacts")
    if env_cfg != expected_env or reward_cfg != expected_reward:
        raise ValueError("environment or reward configuration changed during training")
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
    parser.add_argument("--n-workers", type=int, default=None,
                        help="worker subprocesses for GroupedSubprocVecEnv (default: min(n_envs, cpu_count))")
    parser.add_argument("--n-steps", type=int, default=None,
                        help="PPO rollout steps per env (default from recipe)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="minibatch size (default from recipe)")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="PPO epochs per rollout (default from recipe)")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="PPO learning rate (default from recipe)")
    parser.add_argument("--clip-range", type=float, default=None,
                        help="PPO clip range (default from recipe)")
    parser.add_argument("--vf-coef", type=float, default=None,
                        help="PPO value function loss coefficient (default from recipe)")
    parser.add_argument("--ent-coef", type=float, default=None,
                        help="PPO entropy coefficient (default from recipe)")
    parser.add_argument("--max-grad-norm", type=float, default=None,
                        help="PPO gradient clipping norm (default from recipe)")
    parser.add_argument("--net-arch", default=None,
                        help="comma-separated MLP layer sizes (default from recipe: 256,256)")
    parser.add_argument("--activation", default=None, choices=["elu", "tanh", "relu"],
                        help="policy/value MLP activation (default from recipe: elu)")
    parser.add_argument("--init-std", type=float, default=None,
                        help="initial Gaussian policy exploration std (default from recipe: 0.50)")
    parser.add_argument("--spawn-xy-lower", default=None,
                        help="optional comma-separated override 'x,y' for spawn_xy_lower")
    parser.add_argument("--spawn-xy-upper", default=None,
                        help="optional comma-separated override 'x,y' for spawn_xy_upper")
    parser.add_argument("--eval-interval", type=int, default=None,
                        help="periodic deterministic evaluation interval in env steps")
    parser.add_argument("--eval-episodes", type=int, default=None,
                        help="deterministic evaluation episodes (default from recipe)")
    parser.add_argument("--early-stop-evaluations", type=int, default=None,
                        help="stop early after N consecutive passing evaluations (0 disables)")
    parser.add_argument("--min-success-rate", type=float, default=None,
                        help="minimum success rate for evaluation pass / early stop (default from recipe)")
    parser.add_argument("--save-interval", type=int, default=None,
                        help="checkpoint save interval in env steps")
    parser.add_argument("--resume", type=Path, default=None,
                        help="optional existing SB3 PPO .zip checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default: build/rl/pick_place/xarm6_g2_v2)")
    args = parser.parse_args(argv)

    try:
        import gymnasium  # noqa: E402
        import stable_baselines3  # noqa: E402
        from stable_baselines3 import PPO  # noqa: E402
        from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
        import torch  # noqa: E402
    except ImportError as exc:  # pragma: no cover
        sys.exit(f"training needs gymnasium + stable-baselines3 + torch (requirements-rl.txt): {exc}")

    env_cfg, reward_cfg, train_cfg = build_pick_place_task_configs(
        args.model,
        recipe_path=args.recipe,
        config_path=args.runtime_config,
    )
    env_overrides: dict[str, Any] = {}
    sp_lo = _parse_xy_pair(args.spawn_xy_lower, "--spawn-xy-lower")
    sp_hi = _parse_xy_pair(args.spawn_xy_upper, "--spawn-xy-upper")
    if sp_lo is not None:
        env_overrides["spawn_xy_lower"] = sp_lo
    if sp_hi is not None:
        env_overrides["spawn_xy_upper"] = sp_hi
    if env_overrides:
        env_cfg.update(env_overrides)

    scene_path = load_task_config(args.runtime_config or CONFIGS_ROOT / "pick_place" / "xarm6.yaml").scene_path
    initial_model_hash = env_cfg["runtime_model_sha256"]
    verify_runtime_model(scene_path, initial_model_hash)

    total_steps = int(args.total_steps if args.total_steps is not None else train_cfg.get("total_steps", 50_000_000))
    n_steps = int(args.n_steps if args.n_steps is not None else train_cfg.get("n_steps", 256))
    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 4096))
    n_epochs = int(args.n_epochs if args.n_epochs is not None else train_cfg.get("n_epochs", 5))
    learning_rate = float(
        args.learning_rate if args.learning_rate is not None else train_cfg.get("learning_rate", 3e-4)
    )
    clip_range = float(args.clip_range if args.clip_range is not None else train_cfg.get("clip_range", 0.2))
    vf_coef = float(args.vf_coef if args.vf_coef is not None else train_cfg.get("vf_coef", 1.0))
    ent_coef = float(args.ent_coef if args.ent_coef is not None else train_cfg.get("ent_coef", 0.005))
    max_grad_norm = float(args.max_grad_norm if args.max_grad_norm is not None else train_cfg.get("max_grad_norm", 1.0))
    activation_name = str(
        args.activation if args.activation is not None else train_cfg.get("activation", "elu")
    ).lower()
    init_std = float(args.init_std if args.init_std is not None else train_cfg.get("init_std", 0.50))
    if init_std <= 0.0 or not math.isfinite(init_std):
        raise ValueError("init_std must be finite and positive")

    eval_interval = int(
        args.eval_interval if args.eval_interval is not None else train_cfg.get("eval_interval", 1_000_000)
    )
    eval_episodes = int(args.eval_episodes if args.eval_episodes is not None else train_cfg.get("eval_episodes", 64))
    early_stop_evals = int(
        args.early_stop_evaluations
        if args.early_stop_evaluations is not None
        else train_cfg.get("early_stop_evaluations", 2)
    )
    min_success_rate = float(
        args.min_success_rate if args.min_success_rate is not None else train_cfg.get("min_success_rate", 0.95)
    )
    save_interval = int(
        args.save_interval if args.save_interval is not None else train_cfg.get("save_interval", 1_000_000)
    )
    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 1))
    net_arch = (
        [int(x) for x in args.net_arch.split(",")]
        if args.net_arch is not None
        else [int(x) for x in train_cfg.get("net_arch", [256, 256])]
    )
    gamma = float(train_cfg.get("gamma", env_cfg["reward_gamma"]))
    gae_lambda = float(train_cfg.get("gae_lambda", 0.95))

    n_envs = int(args.n_envs if args.n_envs is not None else auto_n_envs())
    default_workers = min(n_envs, max(1, (os.cpu_count() or 1) - 2)) if n_envs > 1 else 1
    n_workers = int(args.n_workers if args.n_workers is not None else default_workers)
    n_workers = max(1, min(n_workers, n_envs))

    if total_steps > 0:
        n_steps = max(16, min(n_steps, max(16, total_steps // max(n_envs, 1))))
        batch_size = min(batch_size, n_steps * max(n_envs, 1))

    activation_map = {
        "elu": torch.nn.ELU,
        "tanh": torch.nn.Tanh,
        "relu": torch.nn.ReLU,
    }
    if activation_name not in activation_map:
        raise ValueError(f"unsupported activation: {activation_name!r}")

    out_dir = Path(args.out_dir or (ROOT / "build" / "rl" / "pick_place" / "xarm6_g2_v2"))
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_history_path = out_dir / "eval_history.json"

    env_kwargs = _env_kwargs(args.model, args.runtime_config, env_cfg, reward_cfg)
    vec_env = _make_grouped_vec_env(
        n_envs=n_envs,
        n_workers=n_workers,
        seed=seed,
        env_kwargs=env_kwargs,
    )

    if args.resume is not None:
        if not args.resume.is_file():
            vec_env.close()
            raise FileNotFoundError(f"resume checkpoint not found: {args.resume}")
        from stable_baselines3.common.utils import FloatSchedule, get_schedule_fn

        model = PPO.load(
            str(args.resume),
            env=vec_env,
            device=args.device,
            custom_objects={
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "learning_rate": learning_rate,
                "lr_schedule": FloatSchedule(get_schedule_fn(learning_rate)),
                "clip_range": FloatSchedule(get_schedule_fn(clip_range)),
                "clip_range_vf": FloatSchedule(get_schedule_fn(clip_range)),
            },
        )
    else:
        policy_kwargs = {
            "net_arch": {"pi": list(net_arch), "vf": list(net_arch)},
            "activation_fn": activation_map[activation_name],
            "log_std_init": math.log(init_std),
        }
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            seed=seed,
            device=args.device,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            clip_range=clip_range,
            clip_range_vf=clip_range,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
            gae_lambda=gae_lambda,
            policy_kwargs=policy_kwargs,
        )

    eval_env = PickPlaceEnv(**env_kwargs)
    candidates: list[tuple[tuple[float, float, float, float, float], Path, dict[str, Any]]] = []
    eval_history: list[dict[str, Any]] = []
    best_ckpt_file = ckpt_dir / "ppo_pick_place_best.zip"

    start = time.perf_counter()
    actual_steps = 0
    early_stopped = False
    try:
        if total_steps > 0:
            class PeriodicStrictEvalCallback(BaseCallback):
                def __init__(self) -> None:
                    super().__init__(verbose=0)
                    self.next_save_steps = max(1, save_interval)
                    self.next_eval_steps = max(1, eval_interval) if eval_interval > 0 else 0
                    self.best_score: tuple[float, float, float, float, float] | None = None
                    self.consecutive_passes = 0

                def _on_step(self) -> bool:
                    nonlocal early_stopped
                    if save_interval > 0 and self.num_timesteps >= self.next_save_steps:
                        verify_runtime_model(scene_path, initial_model_hash)
                        step_path = ckpt_dir / f"ppo_pick_place_{self.num_timesteps}_steps.zip"
                        self.model.save(step_path)
                        while self.num_timesteps >= self.next_save_steps:
                            self.next_save_steps += save_interval

                    if self.next_eval_steps > 0 and self.num_timesteps >= self.next_eval_steps:
                        verify_runtime_model(scene_path, initial_model_hash)
                        summary, _ = evaluate_policy(
                            eval_env,
                            lambda obs: self.model.predict(obs, deterministic=True)[0],
                            episodes=eval_episodes,
                            seed=17_000 + seed,
                        )
                        score = checkpoint_score(summary)
                        gate_ok = _strict_gate_passed(
                            summary,
                            min_success_rate=min_success_rate,
                            max_press_mm=1.0,
                            max_target_error_mm=20.0,
                        )
                        self.consecutive_passes = (self.consecutive_passes + 1) if gate_ok else 0

                        elapsed = max(1e-9, time.perf_counter() - start)
                        fps = self.num_timesteps / elapsed
                        record = {
                            "timesteps": int(self.num_timesteps),
                            "elapsed_seconds": round(elapsed, 1),
                            "fps": round(fps, 1),
                            "gate_passed": bool(gate_ok),
                            "consecutive_passes": int(self.consecutive_passes),
                            "checkpoint_score": list(score),
                            "success_rate": float(summary["success_rate"]),
                            "ever_carried_rate": float(summary["ever_carried_rate"]),
                            "ever_lifted_clear_rate": float(summary["ever_lifted_clear_rate"]),
                            "touchdown_seen_rate": float(summary["touchdown_seen_rate"]),
                            "release_confirmed_rate": float(summary["release_confirmed_rate"]),
                            "max_press_mm": float(summary["press_mm"]["max"]),
                            "max_release_height_mm": float(summary["release_height_mm"]["max"]),
                            "mean_target_error_mm": float(summary["target_error_mm"]["mean"]),
                            "mean_return": float(summary["return"]["mean"]),
                            "collision_failed": int(summary["collision_failed"]),
                            "drag_failed": int(summary["drag_failed"]),
                            "tilt_failed": int(summary["tilt_failed"]),
                            "press_failed": int(summary["press_failed"]),
                            "high_release": int(summary["high_release"]),
                        }
                        eval_history.append(record)
                        eval_history_path.write_text(
                            json.dumps(eval_history, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                            encoding="utf-8",
                        )

                        if self.best_score is None or score > self.best_score:
                            self.best_score = score
                            self.model.save(best_ckpt_file)
                            candidates.append((score, best_ckpt_file, summary))
                        elif gate_ok:
                            pass_ckpt = ckpt_dir / f"ppo_pick_place_pass_{self.num_timesteps}.zip"
                            self.model.save(pass_ckpt)
                            candidates.append((score, pass_ckpt, summary))

                        print(
                            f"[eval @ {self.num_timesteps:,} steps | {fps:,.0f} fps] "
                            f"sr={summary['success_rate']:.2%} "
                            f"carry={summary['ever_carried_rate']:.2%} "
                            f"lift={summary['ever_lifted_clear_rate']:.2%} "
                            f"rel={summary['release_confirmed_rate']:.2%} "
                            f"press_max={summary['press_mm']['max']:.3f}mm "
                            f"err_mean={summary['target_error_mm']['mean']:.2f}mm "
                            f"ret={summary['return']['mean']:.1f} "
                            f"passes={self.consecutive_passes}/{early_stop_evals}",
                            flush=True,
                        )

                        while self.num_timesteps >= self.next_eval_steps:
                            self.next_eval_steps += eval_interval

                        if early_stop_evals > 0 and self.consecutive_passes >= early_stop_evals:
                            early_stopped = True
                            print(
                                f"[early-stop] Reached {self.consecutive_passes} consecutive strict evaluation passes "
                                f"at {self.num_timesteps:,} steps.",
                                flush=True,
                            )
                            return False
                    return True

            eval_cb = PeriodicStrictEvalCallback()
            print(
                f"[train] {args.model} pure PPO from scratch: budget={total_steps:,} steps, "
                f"n_envs={n_envs} ({n_workers} workers), n_steps={n_steps}, batch_size={batch_size}, "
                f"device={args.device}, seed={seed}",
                flush=True,
            )
            model.learn(total_timesteps=total_steps, callback=eval_cb)
            actual_steps = int(model.num_timesteps)

        final_ckpt = ckpt_dir / "ppo_pick_place_final.zip"
        verify_runtime_model(scene_path, initial_model_hash)
        model.save(final_ckpt)
        final_summary, _ = evaluate_policy(
            eval_env,
            lambda obs: model.predict(obs, deterministic=True)[0],
            episodes=eval_episodes,
            seed=17_000 + seed,
        )
        candidates.append((checkpoint_score(final_summary), final_ckpt, final_summary))
    finally:
        eval_env.close()
        vec_env.close()

    train_seconds = time.perf_counter() - start
    completed_steps = actual_steps if actual_steps > 0 else total_steps
    best_score, best_ckpt_path, best_summary = max(candidates, key=lambda item: item[0])
    target_ckpt = out_dir / "ppo_pick_place.zip"
    verify_runtime_model(scene_path, initial_model_hash)
    shutil.copy2(best_ckpt_path, target_ckpt)

    training_config, manifest = write_pick_place_artifacts(
        out_dir,
        robot=args.model,
        recipe_path=args.recipe,
        runtime_config=args.runtime_config,
        env_overrides=env_overrides or None,
        expected_model_sha256=initial_model_hash,
        expected_env=env_cfg,
        expected_reward=reward_cfg,
        train_overrides={
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "early_stopped": early_stopped,
            "n_envs": n_envs,
            "n_workers": n_workers,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "learning_rate": learning_rate,
            "clip_range": clip_range,
            "vf_coef": vf_coef,
            "ent_coef": ent_coef,
            "max_grad_norm": max_grad_norm,
            "net_arch": net_arch,
            "activation": activation_name,
            "init_std": init_std,
            "eval_interval": eval_interval,
            "eval_episodes": eval_episodes,
            "early_stop_evaluations": early_stop_evals,
            "min_success_rate": min_success_rate,
            "save_interval": save_interval,
            "seed": seed,
            "device": args.device,
        },
    )

    metrics = dict(best_summary)
    metrics.update(
        {
            "passed": _strict_gate_passed(
                best_summary,
                min_success_rate=min_success_rate,
                max_press_mm=1.0,
                max_target_error_mm=20.0,
            ),
            "selected_checkpoint": best_ckpt_path.name,
            "checkpoint_score": list(best_score),
            "early_stopped": early_stopped,
            "completed_steps": completed_steps,
            "config": {
                "model": args.model,
                "total_steps": total_steps,
                "completed_steps": completed_steps,
                "n_envs": n_envs,
                "n_workers": n_workers,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "learning_rate": learning_rate,
                "clip_range": clip_range,
                "vf_coef": vf_coef,
                "ent_coef": ent_coef,
                "max_grad_norm": max_grad_norm,
                "net_arch": net_arch,
                "activation": activation_name,
                "init_std": init_std,
                "seed": seed,
                "device": args.device,
                "config_sha256": training_config["config_sha256"],
                "checkpoint_sha256": manifest.checkpoint_sha256,
                "runtime_config_sha256": env_cfg["runtime_config_sha256"],
            },
            "train_seconds": round(train_seconds, 1),
            "train_steps_per_second": round(completed_steps / max(train_seconds, 1e-9), 1),
            "versions": {
                "python": platform.python_version(),
                "mujoco": __import__("mujoco").__version__,
                "numpy": np.__version__,
                "gymnasium": gymnasium.__version__,
                "stable_baselines3": stable_baselines3.__version__,
                "torch": torch.__version__,
            },
        }
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"\n[train] {args.model} pick-and-place evaluation ({best_summary['episodes']} episodes):")
    print(f"  selected_checkpoint : {best_ckpt_path.name}")
    print(f"  completed_steps     : {completed_steps:,} (early_stopped={early_stopped})")
    print(f"  success_rate        : {best_summary['success_rate']:.2%}")
    print(f"  max_press_mm        : {best_summary['press_mm']['max']:.3f}")
    print(f"  mean_target_err_mm  : {best_summary['target_error_mm']['mean']:.2f}")
    print(f"  mean_return         : {best_summary['return']['mean']:.2f}")
    print(f"[train] saved {target_ckpt}, config.yaml, checkpoint_manifest.json, and metrics.json")


if __name__ == "__main__":
    main()
