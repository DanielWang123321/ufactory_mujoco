#!/usr/bin/env python3
"""Example 12: AL0700 700 mm linear motor simulation with drag-chain deformation.

Demonstrates independent AL0700 position control, optional xArm 5/6/7 mounting
and end-effector/sensor/camera composition, drag-chain visual linkage, and
optional JSON motion reporting.

Usage:
    python examples/simulation/12_linear_motor.py
    python examples/simulation/12_linear_motor.py --model al0700 --position-mm 350
    python examples/simulation/12_linear_motor.py --model xarm6_with_al0700 --round-trips 2 --joint joint1=10
    python examples/simulation/12_linear_motor.py --arm xarm6 --gripper gripper_g2 --ft-sensor --camera d435
    python examples/simulation/12_linear_motor.py --model xarm6_with_al0700 --headless
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from ufactory_mujoco import RailSimulation, create_rail_viewer, move_to_rail_position
from ufactory_mujoco.robots import RAIL_ARMS


def _parse_joint(spec: str) -> tuple[str, float]:
    if "=" not in spec:
        raise ValueError(f"Invalid --joint format {spec!r}; expected JOINT=DEG")
    name, deg_text = spec.split("=", 1)
    if not name:
        raise ValueError(f"Invalid --joint name in {spec!r}")
    return name, float(np.deg2rad(float(deg_text)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        default=None,
        help="AL0700 model name or XML path (default: xarm6_with_al0700 when --arm is omitted)",
    )
    parser.add_argument(
        "--arm",
        choices=RAIL_ARMS,
        default=None,
        help="compose AL0700 with this arm and optional accessories",
    )
    parser.add_argument("--gripper", default=None, help="optional gripper when --arm is set")
    parser.add_argument("--ft-sensor", action="store_true", help="mount AI1500 F/T sensor when --arm is set")
    parser.add_argument("--camera", default=None, help="optional camera when --arm is set")
    parser.add_argument("--position-mm", type=float, default=350.0, help="target rail position in mm (0..700)")
    parser.add_argument("--speed-mm-s", type=float, default=100.0, help="peak commanded rail speed in mm/s")
    parser.add_argument("--round-trips", type=int, default=0, help="0 for single target, N for N 700->0 mm trips")
    parser.add_argument("--joint", action="append", default=[], metavar="JOINT=DEG", help="independent arm target")
    parser.add_argument("--headless", action="store_true", help="no viewer, print motion metrics")
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=None,
        help="viewer hold after motion; default until closed",
    )
    parser.add_argument("--report", type=Path, default=None, help="optional JSON output path relative to CWD")
    args = parser.parse_args(argv)

    has_combo_args = any((args.arm is not None, args.gripper is not None, args.ft_sensor, args.camera is not None))
    if args.model is not None and has_combo_args:
        print("Error: --model cannot be combined with --arm/--gripper/--ft-sensor/--camera", file=sys.stderr)
        return 2
    if not args.arm and (args.gripper is not None or args.ft_sensor or args.camera is not None):
        print("Error: --arm is required with --gripper/--ft-sensor/--camera", file=sys.stderr)
        return 2
    if args.round_trips < 0:
        print("Error: --round-trips must be nonnegative", file=sys.stderr)
        return 2
    if args.hold_seconds is not None and (not math.isfinite(args.hold_seconds) or args.hold_seconds < 0.0):
        print("Error: --hold-seconds must be finite and nonnegative", file=sys.stderr)
        return 2

    viewer = None
    report: dict = {"status": "failed", "moves": []}
    exit_code = 1
    try:
        try:
            joint_targets = [_parse_joint(item) for item in args.joint]
            if args.arm is not None:
                sim = RailSimulation(
                    arm=args.arm,
                    gripper=args.gripper,
                    ft_sensor=args.ft_sensor,
                    camera=args.camera,
                )
            else:
                model_arg = "xarm6_with_al0700" if args.model is None else args.model
                sim = RailSimulation(model_arg)
            sim.rail.set_target_mm(args.position_mm, args.speed_mm_s)
            for name, rad in joint_targets:
                sim.set_arm_target_rad(name, rad)
        except (ValueError, FileNotFoundError) as exc:
            exit_code = 2
            raise ValueError(str(exc)) from exc

        report["model"] = sim.name
        if not args.headless and sys.platform.startswith("linux") and not os.getenv("DISPLAY"):
            raise RuntimeError("No DISPLAY; use --headless or run from a desktop terminal")

        if not args.headless:
            viewer = create_rail_viewer(sim)

        targets = [700.0, 0.0] * args.round_trips if args.round_trips else [args.position_mm]
        for target in targets:
            measurement = move_to_rail_position(sim, target, args.speed_mm_s, viewer)
            report["moves"].append(measurement)
            print(json.dumps(measurement), flush=True)

        report.update(
            status="passed",
            physics="estimated AL0700 mass/inertia (20 kg total, CAD height 107.02 mm); not calibrated",
        )
        print(f"AL0700 demonstration PASS ({sim.name}, {len(targets)} move(s))")
        exit_code = 0

        if viewer:
            end = None if args.hold_seconds is None else time.monotonic() + args.hold_seconds
            while viewer.is_running() and (end is None or time.monotonic() < end):
                with viewer.lock():
                    sim.step()
                viewer.update_chain()
                time.sleep(sim.model.opt.timestep)
    except InterruptedError as exc:
        report.update(status="interrupted", error=str(exc))
        print(f"Interrupted: {exc}", file=sys.stderr)
        exit_code = 1
    except (ValueError, RuntimeError, OSError) as exc:
        report["error"] = str(exc)
        print(f"Error: {exc}", file=sys.stderr)
    finally:
        if viewer is not None:
            viewer.close()
        if args.report is not None:
            try:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            except OSError as exc:
                print(f"Error writing report {args.report}: {exc}", file=sys.stderr)
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
