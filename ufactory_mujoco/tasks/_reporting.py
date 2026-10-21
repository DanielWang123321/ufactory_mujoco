"""Strict JSON reports shared by installed commands and library tasks."""

import json
from pathlib import Path


def write_report(path, payload):
    if path is None:
        return
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text + "\n", encoding="utf-8")


def task_report(preflight, config, dry_run, requested_cycles, cycles=None):
    payload = preflight.as_dict("dry-run" if dry_run else "sim")
    payload.update(robot=config.robot, gripper=config.gripper, executor=config.executor,
                   config_sha256=config.sha256, preflight_ok=preflight.ok,
                   requested_cycles=requested_cycles)
    completed = preflight.ok if dry_run else len(cycles or []) == requested_cycles
    success = preflight.ok and completed
    if not dry_run:
        payload["cycles"] = cycles or []
        success = success and all(
            r["cube_lifted"] and r["bilateral_contact"] and r["home_drift_mm"] < 10.0
            and (r["place_error_mm"] < 50.0 if "place_error_mm" in r else r["in_box"])
            for r in payload["cycles"])
    payload.update(ok=bool(success), success=bool(success), completed=bool(completed))
    return payload
