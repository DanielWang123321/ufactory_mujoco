"""Argument validation and failure reporting for manipulation commands."""

import argparse
import importlib
import math
import sys

from ufactory_mujoco.config import ConfigError, EXECUTORS, ModelResourceError
from ufactory_mujoco.model_paths import MODELS_ROOT
from ufactory_mujoco.robots import ARMS
from ._reporting import write_report


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def manipulation_main(command, argv=None):
    task = importlib.import_module(f"ufactory_mujoco.tasks.{command.replace('-', '_')}")
    parser = argparse.ArgumentParser(prog=f"ufactory-mujoco {command}", description=task.__doc__)
    parser.add_argument("--robot", choices=tuple(ARMS), default="xarm6")
    parser.add_argument("--gripper", help="override the overlay gripper")
    parser.add_argument("--executor", choices=EXECUTORS)
    parser.add_argument("--config", help="overlay YAML instead of the robot default")
    parser.add_argument("--headless", action="store_true", help="run without a viewer")
    parser.add_argument("--dry-run", action="store_true", help="preflight without physics steps")
    parser.add_argument("--report", help="write JSON relative to the current directory")
    parser.add_argument("--cycles", type=positive_int, default=1)
    if command == "box-pack":
        parser.add_argument("--speed", type=positive_float, help="motion speed multiplier")
    args = parser.parse_args(argv)
    code = 1
    payload = None
    try:
        if not MODELS_ROOT.is_dir():
            raise FileNotFoundError(f"Model resource directory not found: {MODELS_ROOT}")
        load_args = [args.robot, args.gripper, args.executor, args.config]
        if command == "box-pack":
            load_args.append(args.speed)
        config = task._load(*load_args)
        if command == "box-pack" and args.cycles > len(config.box_pack.feed_xy):
            raise ConfigError(f"cycles must be in 1..{len(config.box_pack.feed_xy)}")
    except ModelResourceError as exc:
        payload = {"error": str(exc)}
    except (ConfigError, ValueError, TypeError, KeyError) as exc:
        code = 2
        payload = {"error": str(exc)}
    except Exception as exc:
        payload = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        try:
            with task.SimRobot(config.scene_path, viewer=not args.headless and not args.dry_run) as robot:
                run = task.run_pick_place if command == "pick-place" else task.run_box_pack
                outcome = run(robot, config, cycles=args.cycles,
                              dry_run=args.dry_run, report_path=args.report)
            ok = outcome.ok if args.dry_run else len(outcome) == args.cycles and all(
                r["cube_lifted"] and r["bilateral_contact"] and r["home_drift_mm"] < 10.0
                and (r["place_error_mm"] < 50.0 if command == "pick-place" else r["in_box"])
                for r in outcome)
            code = 0 if ok else 1
        except Exception as exc:
            payload = {"error": f"{type(exc).__name__}: {exc}"}
    if payload is not None:
        print(payload["error"], file=sys.stderr)
        payload.update(ok=False, success=False, completed=False,
                       mode="dry-run" if args.dry_run else "sim", exit_code=code)
        try:
            write_report(args.report, payload)
        except (OSError, ValueError, TypeError) as exc:
            print(f"Report failed: {exc}", file=sys.stderr)
            code = 1
    print(f"\n{command}: {'PASS' if code == 0 else 'FAIL'}")
    return code
