#!/usr/bin/env python3
"""Run the fixed three-phase seed sensitivity experiment on the 17 server.

All state and outputs stay below --out-dir. Restart the same command to retry
interrupted training from scratch and continue completed training's evaluations.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "scripts/train_pick_place_rl.py"
EVALUATE = ROOT / "examples/rl/pick_place/evaluate.py"
RECIPE = ROOT / "examples/rl/pick_place/recipe.yaml"
PHASES = ("legacy", "deterministic", "new-long")
THREAD_ENV = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                      "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def worklog(message: str) -> None:
    with (ROOT / "worklog.md").open("a", encoding="utf-8") as stream:
        stream.write(f"- 种子敏感性实验（{now()}）：{message}\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_matrix() -> list[dict[str, Any]]:
    tasks = []
    rng = random.Random(20261106)
    for phase in PHASES:
        seeds = ([10001 + 1000 * i for i in range(20)] if phase == "new-long"
                 else [1, 1001, 2001, 3001, 4001])
        for rep in range(1, 4):
            shuffled = seeds.copy()
            rng.shuffle(shuffled)
            for seed in shuffled:
                tasks.append({"id": f"{phase}_seed{seed}_rep{rep}", "phase": phase, "seed": seed,
                              "replicate": rep, "budget": 50_000_000 if phase == "new-long" else 10_000_000,
                              "checkpoints": [10_000_000, 20_000_000, 50_000_000] if phase == "new-long"
                                             else [10_000_000],
                              "eval_seeds": [50_001, 51_001, 52_001, 53_001, 54_001]
                                            if phase == "new-long" else [30_001, 31_001, 32_001, 33_001, 34_001]})
    return tasks


def source_paths() -> list[Path]:
    paths = [TRAIN, EVALUATE, RECIPE, Path(__file__).resolve()]
    for folder in (ROOT / "ufactory_mujoco", ROOT / "configs", ROOT / "models/xarm6_with_gripper"):
        paths.extend(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                     and (folder.name == "xarm6_with_gripper" or p.suffix in (".py", ".yaml", ".yml")))
    return sorted(set(paths), key=lambda p: str(p.relative_to(ROOT)))


def stat(path: Path) -> tuple[int, int, int, int]:
    s = path.stat()
    return s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino


def dependencies() -> dict[str, str]:
    return dict(sorted((d.metadata.get("Name", "unknown").lower(), d.version)
                       for d in importlib.metadata.distributions()))


def resource_sample(running: dict[str, dict[str, Any]], previous_cpu: dict[str, list[int]]) -> tuple[dict, dict]:
    row: dict[str, Any] = {"utc": now(), "training": {}}
    current = {}
    for line in Path("/proc/stat").read_text().splitlines():
        fields = line.split()
        if fields and fields[0].startswith("cpu"):
            current[fields[0]] = [int(x) for x in fields[1:]]
    row["cpu_percent"] = {}
    for name, values in current.items():
        if name not in previous_cpu:
            continue
        delta = [a - b for a, b in zip(values, previous_cpu[name])]
        total = sum(delta[:8])
        if total > 0:
            row["cpu_percent"][name] = 100 * (1 - (delta[3] + delta[4]) / total)
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemTotal:", "MemAvailable:")):
            fields = line.split()
            row[fields[0].rstrip(":") + "_kib"] = int(fields[1])
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3, check=True)
        row["gpu"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        row["gpu_error"] = str(exc)
    for task_id, info in running.items():
        if info["kind"] == "train":
            progress = info["progress"]
            try:
                row["training"][task_id] = json.loads(progress.read_text())
            except (OSError, json.JSONDecodeError):
                row["training"][task_id] = {"status": "starting"}
    return row, current


def summarize_phase(out: Path, phase: str, tasks: list[dict], state: dict) -> None:
    results = []
    for task in (item for item in tasks if item["phase"] == phase):
        item = state["tasks"][task["id"]]
        for budget in task["checkpoints"]:
            reports = [json.loads((out / item["reports"][f"{budget}_{seed}"]).read_text())
                       for seed in task["eval_seeds"]]
            rows = [episode for report in reports for episode in report["episodes_detail"]]
            if len(rows) != 200 * len(task["eval_seeds"]):
                raise ValueError(f"incorrect evaluation episode count for {task['id']} at {budget}")
            n = len(rows)
            success = sum(bool(row["success"]) for row in rows) / n
            press = max(float(row["press_mm"]) for row in rows)
            error = sum(float(row["target_error_mm"]) for row in rows) / n
            failure_keys = ("drag_failed", "tilt_failed", "dropped", "press_failed", "high_release",
                            "collision_failed", "timeout")
            result = {"id": task["id"], "seed": task["seed"], "replicate": task["replicate"],
                      "budget": budget, "success_rate": success, "max_press_mm": press,
                      "mean_target_error_mm": error, "passed": success >= 0.95 and press <= 1 and error <= 20,
                      "failures": {key: sum(bool(row[key]) for row in rows) for key in failure_keys},
                      "policy_state_sha256": item["checkpoints"][str(budget)]["policy_state_sha256"],
                      "episode_outcome_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True,
                                                                             separators=(",", ":")).encode()).hexdigest()}
            results.append(result)
    payload: dict[str, Any] = {"phase": phase, "generated_utc": now(), "models": results}
    manifest = json.loads((out / "manifest.json").read_text())
    reference = manifest.get("historical_reference")
    if phase == "legacy" and reference:
        historical = json.loads((ROOT / reference).read_text())
        comparison = []
        for row in historical["runs"]:
            seed = row["train_seed"]
            group = [r["success_rate"] for r in results if r["seed"] == seed]
            if group:
                comparison.append({"seed": seed, "historical_success_rate": row["success_rate"],
                                   "server_success_rates": group, "server_mean_success_rate": sum(group) / len(group)})
        payload["historical_reference"] = {"path": reference, "comparison": comparison}
    if phase == "deterministic":
        checks = {}
        for seed in sorted({r["seed"] for r in results}):
            group = [r for r in results if r["seed"] == seed]
            checks[str(seed)] = {"identical_policy_hashes": len({r["policy_state_sha256"] for r in group}) == 1,
                                 "identical_episode_records": len({r["episode_outcome_sha256"] for r in group}) == 1}
        payload["repeat_comparison"] = checks
    if phase == "new-long":
        payload["statistics"] = calculate_statistics(results)
    write_json(out / f"summary_{phase}.json", payload)
    if phase == "new-long":
        plot_results(out, results)
    worklog(f"{phase} 阶段评估完成，{len(results)} 个模型；汇总 {out / f'summary_{phase}.json'}。")


def calculate_statistics(results: list[dict]) -> dict:
    import numpy as np

    rng = np.random.default_rng(20261106)
    summaries = {}
    for budget in (10_000_000, 20_000_000, 50_000_000):
        selected = [r for r in results if r["budget"] == budget]
        seeds = sorted({r["seed"] for r in selected})
        lookup = {(r["seed"], r["replicate"]): r["success_rate"] for r in selected}
        y = np.array([[lookup[(seed, rep)] for rep in (1, 2, 3)] for seed in seeds])

        def components(x):
            within = float(np.sum((x - x.mean(axis=1, keepdims=True)) ** 2) / (x.shape[0] * (x.shape[1] - 1)))
            between = max(0.0, (float(x.shape[1] * np.var(x.mean(axis=1), ddof=1)) - within) / x.shape[1])
            return np.sqrt(between), np.sqrt(within)

        between, within = components(y)
        observed = float(np.var(y.mean(axis=1), ddof=1))
        boot = np.empty((10000, 3))
        for i in range(len(boot)):
            sampled = y[rng.integers(0, len(seeds), len(seeds))]
            boot[i] = (sampled.mean(), *components(sampled))
        p_value = None
        if budget in (10_000_000, 50_000_000):
            count = 0
            for _ in range(99999):
                perm = np.column_stack([rng.permutation(y[:, rep]) for rep in range(3)])
                count += float(np.var(perm.mean(axis=1), ddof=1)) >= observed - 1e-15
            p_value = (count + 1) / 100000
        summaries[str(budget)] = {
            "mean_success_rate": float(y.mean()), "between_seed_sd_adjusted": float(between),
            "within_seed_repeat_sd": float(within),
            "bootstrap_95_ci": {key: [float(v) for v in np.percentile(boot[:, i], [2.5, 97.5])]
                                for i, key in enumerate(("mean_success_rate", "between_seed_sd_adjusted",
                                                         "within_seed_repeat_sd"))},
            "permutation_p": p_value,
            # between/within are NumPy scalars; cast so json.dumps accepts the flag.
            "marked_seed_sensitivity": bool(p_value < 0.005 and between >= 0.10 and between >= 3 * within)
                                       if p_value is not None else None,
        }
    return {"method": "seed bootstrap (10000); within-replicate label permutation (99999); "
                      "ANOVA variance components, truncated at zero", "budgets": summaries}


def plot_results(out: Path, results: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for seed in sorted({r["seed"] for r in results}):
        group = [r for r in results if r["seed"] == seed]
        xs = [10, 20, 50]
        ys = [sum(r["success_rate"] for r in group if r["budget"] == x * 1_000_000) / 3 for x in xs]
        ax.plot(xs, ys, alpha=0.55, marker=".", linewidth=1)
    ax.set(xlabel="Training steps (millions)", ylabel="Success rate", ylim=(-0.03, 1.03),
           title="Seed means across three independent training runs")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "seed_trends.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, default=None,
                        help="historical local summary, copied inside the server project")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if not out.is_relative_to(ROOT):
        parser.error("--out-dir must be inside the project")
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "controller.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another scheduler holds {out / 'controller.lock'}") from None
    manifest_path, state_path = out / "manifest.json", out / "state.json"
    tasks = task_matrix()
    reference = args.reference_summary.resolve() if args.reference_summary else None
    if reference and (not reference.is_relative_to(ROOT) or not reference.is_file()):
        parser.error("--reference-summary must be an existing file inside the project")

    def frozen_paths() -> list[Path]:
        return sorted(set(source_paths() + ([reference] if reference else [])),
                      key=lambda path: str(path.relative_to(ROOT)))

    paths = frozen_paths()
    input_hashes = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    versions = dependencies()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["source_sha256"] != input_hashes or manifest["dependencies"] != versions
                or manifest["tasks"] != tasks or manifest["python"] != sys.executable
                or manifest.get("historical_reference") != (str(reference.relative_to(ROOT)) if reference else None)):
            raise SystemExit("frozen inputs, dependencies, interpreter or task matrix changed; refusing to resume")
    else:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        manifest = {"created_utc": now(), "git_revision": revision, "python": sys.executable,
                    "python_version": sys.version, "source_sha256": input_hashes, "dependencies": versions,
                    "tasks": tasks, "layout": [{"slot": i, "train_cpu": 2 * i + 1,
                    "worker_cpus": [2 * i, 16 + 2 * i, 17 + 2 * i], "worker_envs": [64, 32, 32]}
                    for i in range(8)], "deterministic": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "torch_deterministic_algorithms": True, "cudnn_benchmark": False},
                    "bootstrap_samples": 10000, "permutations": 99999, "alpha": 0.005}
        manifest["historical_reference"] = str(reference.relative_to(ROOT)) if reference else None
        manifest["gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout.strip()
        write_json(manifest_path, manifest)
        (out / "recipe.yaml").write_bytes(RECIPE.read_bytes())
    if args.dry_run:
        print(json.dumps({"out_dir": str(out), "tasks": dict(Counter(t["phase"] for t in tasks)),
                          "checkpoints": 210, "evaluations": 1050, "episodes": 210000}, indent=2))
        return 0
    available = os.sched_getaffinity(0)
    if not set(range(32)).issubset(available):
        raise SystemExit(f"CPUs 0–31 must be available; affinity is {sorted(available)}")
    state = json.loads(state_path.read_text()) if state_path.exists() else {"tasks": {}}
    for item in state.get("active_evaluations", {}).values():
        try:
            os.kill(item["pid"], 0)
        except ProcessLookupError:
            continue
        raise SystemExit(f"previous evaluation PID {item['pid']} is still alive; refusing a duplicate run")
    state["active_evaluations"] = {}
    for task in tasks:
        item = state["tasks"].setdefault(task["id"], {"status": "pending", "attempts": [], "reports": {}})
        if item["status"] == "training":
            pid = item["attempts"][-1]["pid"]
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise SystemExit(f"previous training PID {pid} is still alive; refusing a duplicate run")
            item["attempts"][-1].update({"status": "interrupted", "finished_utc": now()})
            item["status"] = "pending"
    state["status"] = "running"
    state["started_utc"] = state.get("started_utc", now())
    write_json(state_path, state)
    worklog(f"调度器启动或恢复，输出目录 {out}。")
    metadata = {str(p.relative_to(ROOT)): stat(p) for p in paths}
    running: dict[str, dict[str, Any]] = {}
    occupied: dict[int, str] = {}
    previous_cpu: dict[str, list[int]] = {}
    last_monitor = last_check = last_deps = 0.0
    interrupted = False

    def stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)

    def launch_train(task: dict, slot: int) -> None:
        item = state["tasks"][task["id"]]
        attempt_num = len(item["attempts"]) + 1
        attempt = out / task["id"] / f"attempt_{attempt_num:02d}"
        attempt.mkdir(parents=True, exist_ok=False)
        progress = attempt / "progress.json"
        train_dir = attempt / "train"
        cpus = [2 * slot, 16 + 2 * slot, 17 + 2 * slot]
        cmd = [sys.executable, str(TRAIN), "--seed", str(task["seed"]), "--total-steps", str(task["budget"]),
               "--n-envs", "128", "--n-workers", "3", "--worker-env-counts", "64,32,32",
               "--worker-cpu-sets", ";".join(map(str, cpus)), "--torch-threads", "1", "--device", "cuda",
               "--save-interval", "0", "--eval-interval", "0", "--early-stop-evaluations", "0",
               "--disable-periodic-eval", "--no-final-eval", "--progress-json", str(progress),
               "--out-dir", str(train_dir)]
        if task["phase"] == "new-long":
            cmd.extend(["--checkpoint-steps", ",".join(map(str, task["checkpoints"]))])
        if task["phase"] == "deterministic":
            cmd.append("--deterministic")
        env = dict(os.environ, **THREAD_ENV)
        if task["phase"] == "deterministic":
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            env["PYTHONHASHSEED"] = str(task["seed"])
        else:
            env.pop("CUBLAS_WORKSPACE_CONFIG", None)
        stream = (attempt / "train.log").open("w")
        proc = subprocess.Popen(["taskset", "-c", str(2 * slot + 1), *cmd], cwd=ROOT, env=env,
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        item["attempts"].append({"number": attempt_num, "started_utc": now(), "status": "running", "slot": slot,
                                 "pid": proc.pid, "path": str(attempt.relative_to(out)), "command": cmd})
        item["status"] = "training"
        running[task["id"]] = {"kind": "train", "task": task, "slot": slot, "process": proc,
                               "stream": stream, "progress": progress, "attempt": attempt, "cpus": cpus,
                               "started_monotonic": time.monotonic()}
        occupied.update({cpu: task["id"] for cpu in cpus})
        write_json(state_path, state)
        print(f"{now()} training {task['id']} slot={slot} pid={proc.pid}", flush=True)

    def finish_train(info: dict, rc: int) -> None:
        import yaml

        task, attempt = info["task"], info["attempt"]
        item = state["tasks"][task["id"]]
        item["attempts"][-1].update({"finished_utc": now(), "returncode": rc,
                                      "status": "failed" if rc else "complete"})
        if rc:
            if len(item["attempts"]) >= 3:
                raise RuntimeError(f"{task['id']} training failed three times; see {attempt / 'train.log'}")
            item["status"] = "pending"
            return
        train_dir = attempt / "train"
        metrics = json.loads((train_dir / "metrics.json").read_text())
        config = yaml.safe_load((train_dir / "config.yaml").read_text())
        recipe = yaml.safe_load(RECIPE.read_text())
        for key in ("n_steps", "batch_size", "n_epochs", "learning_rate", "gamma", "gae_lambda",
                    "clip_range", "vf_coef", "ent_coef", "max_grad_norm", "net_arch", "activation", "init_std"):
            if config["train"][key] != recipe["train"][key]:
                raise ValueError(f"training parameter mismatch for {task['id']}: {key}")
        actual = metrics["config"]
        expected_config = {"seed": task["seed"], "n_envs": 128, "n_workers": 3,
                           "worker_env_counts": [64, 32, 32], "worker_cpu_ids": [[cpu] for cpu in info["cpus"]],
                           "deterministic": task["phase"] == "deterministic", "torch_threads": 1, "device": "cuda"}
        if any(actual.get(key) != value for key, value in expected_config.items()):
            raise ValueError(f"training runtime settings mismatch for {task['id']}")
        contract = {key: config["env"][key] for key in ("runtime_config_sha256", "runtime_model_sha256")}
        if "runtime_contract" in state and state["runtime_contract"] != contract:
            raise ValueError(f"runtime model or config mismatch for {task['id']}")
        state["runtime_contract"] = contract
        expected = ((task["budget"] + 16383) // 16384) * 16384
        if metrics["completed_steps"] != expected:
            raise ValueError(f"unexpected completed steps for {task['id']}")
        checkpoints = {}
        for budget in task["checkpoints"]:
            if task["phase"] == "new-long":
                matches = list((train_dir / "checkpoints" / f"budget_{budget}").glob(
                    f"ppo_pick_place_{budget}_requested_*_steps.zip"))
                if len(matches) != 1:
                    raise ValueError(f"missing/ambiguous checkpoint {budget} for {task['id']}")
                checkpoint = matches[0]
                details = json.loads(checkpoint.with_suffix(".state.json").read_text())
                if details["actual_steps"] != ((budget + 16383) // 16384) * 16384:
                    raise ValueError(f"unexpected checkpoint steps for {task['id']} {budget}")
            else:
                checkpoint = train_dir / "ppo_pick_place.zip"
                details = {"actual_steps": metrics["completed_steps"],
                           "policy_state_sha256": metrics["config"]["policy_state_sha256"]}
            manifest = json.loads(checkpoint.with_suffix(".checkpoint_manifest.json").read_text())
            checkpoint_config = yaml.safe_load((checkpoint.parent / "config.yaml").read_text())
            checkpoint_sha = sha(checkpoint)
            if (manifest["checkpoint_sha256"] != checkpoint_sha
                    or manifest["config_sha256"] != checkpoint_config["config_sha256"]
                    or checkpoint_config["train"]["completed_steps"] != details["actual_steps"]):
                raise ValueError(f"checkpoint manifest/config mismatch: {checkpoint}")
            if budget == task["budget"] and details["policy_state_sha256"] != actual["policy_state_sha256"]:
                raise ValueError(f"final policy differs from final checkpoint: {checkpoint}")
            details.update({"path": str(checkpoint.relative_to(out)), "sha256": checkpoint_sha})
            checkpoints[str(budget)] = details
        wall_seconds = time.monotonic() - info["started_monotonic"]
        item.update({"status": "trained", "checkpoints": checkpoints, "train_seconds": metrics["train_seconds"],
                     "wall_seconds": wall_seconds})
        same_budget = [state["tasks"][t["id"]]["wall_seconds"] for t in tasks
                       if t["budget"] == task["budget"] and "wall_seconds" in state["tasks"][t["id"]]]
        mean_seconds = sum(same_budget) / len(same_budget)
        remaining_steps = sum(t["budget"] for t in tasks if state["tasks"][t["id"]]["status"] == "pending")
        for active in running.values():
            if active["kind"] != "train" or active is info:
                continue
            try:
                progress_steps = int(json.loads(active["progress"].read_text())["timesteps"])
            except (OSError, json.JSONDecodeError, KeyError):
                progress_steps = 0
            remaining_steps += max(0, active["task"]["budget"] - progress_steps)
        nominal_fps = expected / mean_seconds
        calibration = {"samples": len(same_budget), "mean_wall_seconds": mean_seconds,
                       "per_slot_fps": nominal_fps, "remaining_steps_estimate": remaining_steps,
                       "ideal_remaining_training_hours_at_eight_slots": remaining_steps / (8 * nominal_fps * 3600)}
        state.setdefault("calibration", {})[str(task["budget"])] = calibration
        if len(same_budget) == 1:
            worklog(f"首个 {task['budget']} 步训练完成，用时 {wall_seconds:.1f} 秒；"
                    f"八槽理想剩余训练时间估计 {calibration['ideal_remaining_training_hours_at_eight_slots']:.1f} 小时。")
        print(f"{now()} trained {task['id']} wall_seconds={wall_seconds:.1f}", flush=True)

    def launch_eval(task: dict, budget: int, eval_seed: int, cpu: int) -> None:
        item = state["tasks"][task["id"]]
        report_key = f"{budget}_{eval_seed}"
        folder = out / task["id"]
        attempt = 1
        while (folder / f"eval_{report_key}_attempt_{attempt}.log").exists():
            attempt += 1
        report = folder / f"eval_{report_key}_attempt_{attempt}.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_info = item["checkpoints"][str(budget)]
        if sha(out / checkpoint_info["path"]) != checkpoint_info["sha256"]:
            raise ValueError(f"checkpoint changed: {checkpoint_info['path']}")
        cmd = [sys.executable, str(EVALUATE), "--checkpoint", str(out / item["checkpoints"][str(budget)]["path"]),
               "--episodes", "200", "--seed", str(eval_seed), "--device", "cpu", "--min-success-rate", "0.95",
               "--max-press-mm", "1", "--max-target-error-mm", "20", "--quiet-episodes", "--summary-json", str(report)]
        stream = (report.parent / f"eval_{report_key}.log").open("w")
        proc = subprocess.Popen(["taskset", "-c", str(cpu), *cmd], cwd=ROOT, env=dict(os.environ, **THREAD_ENV),
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        key = f"{task['id']}:{report_key}"
        running[key] = {"kind": "eval", "task": task, "budget": budget, "eval_seed": eval_seed,
                        "cpu": cpu, "process": proc, "stream": stream, "report": report}
        occupied[cpu] = key
        state["active_evaluations"][key] = {"pid": proc.pid, "cpu": cpu,
                                            "report": str(report.relative_to(out)), "started_utc": now()}
        write_json(state_path, state)

    try:
        while True:
            tick = time.monotonic()
            if interrupted:
                raise InterruptedError("received termination signal")
            if tick - last_check >= 5:
                if [str(p.relative_to(ROOT)) for p in frozen_paths()] != list(input_hashes):
                    raise RuntimeError("frozen input file list changed")
                for path in paths:
                    name = str(path.relative_to(ROOT))
                    current_stat = stat(path)
                    if current_stat != metadata[name]:
                        if sha(path) != input_hashes[name]:
                            raise RuntimeError(f"frozen input changed: {name}")
                        metadata[name] = current_stat
                if sha(out / "recipe.yaml") != sha(RECIPE):
                    raise RuntimeError("frozen recipe copy changed")
                last_check = tick
            if tick - last_deps >= 60:
                if dependencies() != versions:
                    raise RuntimeError("installed dependencies changed")
                last_deps = tick
            if tick - last_monitor >= 5:
                sample, previous_cpu = resource_sample(running, previous_cpu)
                with (out / "resources.jsonl").open("a") as stream:
                    stream.write(json.dumps(sample, allow_nan=False) + "\n")
                last_monitor = tick
            for key, info in list(running.items()):
                rc = info["process"].poll()
                if rc is None:
                    continue
                info["stream"].close()
                if info["kind"] == "train":
                    for cpu in info["cpus"]:
                        occupied.pop(cpu, None)
                    finish_train(info, rc)
                else:
                    occupied.pop(info["cpu"], None)
                    state["active_evaluations"].pop(key, None)
                    report = info["report"]
                    if rc not in (0, 1) or not report.is_file():
                        raise RuntimeError(f"evaluation failed ({rc}): {report.with_suffix('.log')}")
                    data = json.loads(report.read_text())
                    if data.get("episodes") != 200 or len(data.get("episodes_detail", [])) != 200:
                        raise ValueError(f"incomplete evaluation: {report}")
                    item = state["tasks"][info["task"]["id"]]
                    expected_checkpoint = item["checkpoints"][str(info["budget"])]["sha256"]
                    if data.get("checkpoint_sha256") != expected_checkpoint or data.get("seed") != info["eval_seed"]:
                        raise ValueError(f"evaluation metadata mismatch: {report}")
                    spawn_hashes = state.setdefault("evaluation_spawn_sha256", {})
                    seed_key = str(info["eval_seed"])
                    if seed_key in spawn_hashes and spawn_hashes[seed_key] != data["spawn_sha256"]:
                        raise ValueError(f"evaluation scenario mismatch: {report}")
                    spawn_hashes[seed_key] = data["spawn_sha256"]
                    item["reports"][f"{info['budget']}_{info['eval_seed']}"] = str(report.relative_to(out))
                    if len(item["reports"]) == len(info["task"]["checkpoints"]) * len(info["task"]["eval_seeds"]):
                        item["status"] = "complete"
                        item["finished_utc"] = now()
                        print(f"{now()} completed {info['task']['id']}", flush=True)
                        worklog(f"{info['task']['id']} 训练与全部固定场景评估完成，结果位于 {report.parent}。")
                del running[key]
                write_json(state_path, state)
            phase = next((p for p in PHASES if any(state["tasks"][t["id"]]["status"] != "complete"
                                                    for t in tasks if t["phase"] == p)), None)
            if phase is None:
                for p in PHASES:
                    if not (out / f"summary_{p}.json").exists():
                        summarize_phase(out, p, tasks, state)
                state.update({"status": "complete", "finished_utc": now()})
                write_json(state_path, state)
                worklog(f"三个阶段全部完成，统计结果位于 {out}。")
                print(f"{now()} experiment complete", flush=True)
                return 0
            for prior in PHASES[:PHASES.index(phase)]:
                if not (out / f"summary_{prior}.json").exists():
                    summarize_phase(out, prior, tasks, state)
            state["phase"] = phase
            pending = [t for t in tasks if t["phase"] == phase and state["tasks"][t["id"]]["status"] == "pending"]
            for slot in range(8):
                cpus = [2 * slot, 16 + 2 * slot, 17 + 2 * slot]
                if pending and not any(cpu in occupied for cpu in cpus):
                    launch_train(pending.pop(0), slot)
            # Only run evaluations on cores belonging to slots with no training.
            for slot in range(8):
                cpus = [2 * slot, 16 + 2 * slot, 17 + 2 * slot]
                if any(info["kind"] == "train" and info["slot"] == slot for info in running.values()):
                    continue
                for cpu in cpus:
                    if cpu in occupied:
                        continue
                    chosen = None
                    for task in tasks:
                        item = state["tasks"][task["id"]]
                        if task["phase"] != phase or item["status"] != "trained":
                            continue
                        for budget in task["checkpoints"]:
                            for eval_seed in task["eval_seeds"]:
                                report_key = f"{budget}_{eval_seed}"
                                if report_key not in item["reports"] and f"{task['id']}:{report_key}" not in running:
                                    chosen = (task, budget, eval_seed)
                                    break
                            if chosen:
                                break
                        if chosen:
                            break
                    if chosen:
                        launch_eval(*chosen, cpu)
            write_json(state_path, state)
            time.sleep(1)
    except (Exception, KeyboardInterrupt) as exc:
        state.update({"status": "stopped", "stopped_utc": now(), "reason": repr(exc)})
        worklog(f"队列停止，原因 {exc!r}；状态文件 {state_path}。")
        print(f"{now()} stopped: {exc!r}", file=sys.stderr, flush=True)
        for info in running.values():
            try:
                os.killpg(info["process"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for info in running.values():
            try:
                info["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(info["process"].pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                info["process"].wait()
            info["stream"].close()
            if info["kind"] == "train":
                item = state["tasks"][info["task"]["id"]]
                item["attempts"][-1].update({"status": "interrupted", "finished_utc": now()})
                item["status"] = "pending"
        state["active_evaluations"] = {}
        write_json(state_path, state)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
