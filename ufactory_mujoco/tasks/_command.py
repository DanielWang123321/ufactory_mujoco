"""Argument validation, hardware safety gate, and reporting for manipulation commands."""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys

import numpy as np

from ufactory_mujoco.config import ConfigError, EXECUTORS, ModelResourceError
from ufactory_mujoco.model_paths import MODELS_ROOT
from ufactory_mujoco.preflight import (
    make_box_pack_checker,
    make_checker,
    preflight_box_pack,
    preflight_pick_place,
)
from ufactory_mujoco.robots import ARMS
from ufactory_mujoco.safety import (
    SafetyGateError,
    authorize_real_motion,
    enforce_single_cycle,
    run_sdk_simulation,
    stream_sha256,
)
from ufactory_mujoco.trajectory import plan_ptp_quintic
from ufactory_mujoco.xarm_robot import XArmRobot
from ._reporting import write_report

MODES = ("sim", "dry-run", "sdk-sim", "real")


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


def _build_shadow_stream(
    report,
    n_arm: int,
    *,
    max_vel_rad_s: float = 0.8,
    min_seg_duration: float = 0.4,
    dt: float = 0.02,
) -> np.ndarray:
    """Build a velocity-bounded joint trajectory array ``(N, n_arm)`` from preflight segments."""
    chunks: list[np.ndarray] = []
    for seg in report.segments.values():
        w = np.asarray(seg.waypoints, dtype=np.float64)
        for i in range(len(w) - 1):
            q0 = w[i, :n_arm]
            q1 = w[i + 1, :n_arm]
            disp = float(np.max(np.abs(q1 - q0)))
            duration = max(min_seg_duration, disp * 1.875 / max_vel_rad_s)
            traj = plan_ptp_quintic(q0, q1, duration=duration, dt=dt)
            chunks.append(traj.q if not chunks else traj.q[1:])
    if not chunks:
        return np.zeros((1, n_arm), dtype=np.float64)
    return np.vstack(chunks)


def _run_hardware_manipulation(command: str, args, config) -> tuple[int, dict]:
    """Execute sdk-sim or real mode with fail-closed calibration, preflight, and cycle gates."""
    mode = args.mode
    enforce_single_cycle(args.cycles, mode)
    ip = args.ip or os.environ.get("XARM_IP")
    if not ip:
        raise SafetyGateError(f"mode={mode!r} requires --ip or XARM_IP")
    if not args.calibration:
        raise SafetyGateError(f"mode={mode!r} requires --calibration YAML")

    with XArmRobot(
        ip,
        robot=config.robot,
        gripper=config.gripper,
        scene=config.scene,
        model_path=config.scene_path,
        calibration=args.calibration,
        read_only=True,
    ) as robot:
        if command == "pick-place":
            report = preflight_pick_place(robot, config)
            checker = make_checker(config.scene_path)
        else:
            report = preflight_box_pack(robot, config, cycles=1)
            checker = make_box_pack_checker(config.scene_path)

        shadow_q = _build_shadow_stream(report, robot.n_arm)
        prog_hash = stream_sha256(shadow_q)

        if mode == "sdk-sim":
            evidence = run_sdk_simulation(
                robot._arm,
                robot=config.robot,
                calibration=robot.calibration,
                shadow_joint_stream_rad=shadow_q,
                checker=checker,
                preflight_ok=report.ok,
                cycles=args.cycles,
                config_sha256=config.sha256,
                program_sha256=prog_hash,
                step_dt_s=0.02,
            )
            ok = bool(report.ok and evidence.passed)
            payload = report.as_dict("sdk-sim")
            payload.update(
                robot=config.robot,
                gripper=config.gripper,
                executor=config.executor,
                serial_number=robot.serial_number,
                calibration_sha256=robot.calibration.sha256,
                config_sha256=config.sha256,
                program_sha256=prog_hash,
                preflight_ok=report.ok,
                requested_cycles=args.cycles,
                sdk_sim=evidence.as_dict(),
                ok=ok,
                success=ok,
                completed=True,
            )
            return (0 if ok else 1), payload

        # mode == "real": require explicit --confirm-real and issue MotionApproval
        approval = authorize_real_motion(
            robot._arm,
            robot=config.robot,
            calibration=robot.calibration,
            preflight_ok=report.ok,
            confirm_real=bool(args.confirm_real),
            cycles=args.cycles,
            config_sha256=config.sha256,
            program_sha256=prog_hash,
        )
        robot.unlock_motion(approval)
        for seg in report.segments.values():
            w = np.asarray(seg.waypoints, dtype=np.float64)
            for i in range(1, len(w)):
                robot.move_joints(w[i, : robot.n_arm], duration=1.5, wait=True)
        q_end = robot.get_joint_positions()
        home_err_rad = float(np.max(np.abs(q_end - shadow_q[-1])))
        ok = bool(report.ok and home_err_rad < 0.05)
        payload = report.as_dict("real")
        payload.update(
            robot=config.robot,
            gripper=config.gripper,
            executor=config.executor,
            serial_number=robot.serial_number,
            calibration_sha256=robot.calibration.sha256,
            config_sha256=config.sha256,
            program_sha256=prog_hash,
            preflight_ok=report.ok,
            requested_cycles=1,
            confirm_real=True,
            final_joint_error_rad=home_err_rad,
            ok=ok,
            success=ok,
            completed=True,
        )
        return (0 if ok else 1), payload


def manipulation_main(command, argv=None):
    task = importlib.import_module(f"ufactory_mujoco.tasks.{command.replace('-', '_')}")
    parser = argparse.ArgumentParser(prog=f"ufactory-mujoco {command}", description=task.__doc__)
    parser.add_argument("--robot", choices=tuple(ARMS), default="xarm6")
    parser.add_argument("--gripper", help="override the overlay gripper")
    parser.add_argument("--executor", choices=EXECUTORS)
    parser.add_argument("--config", help="overlay YAML instead of the robot default")
    parser.add_argument("--mode", choices=MODES, default=None,
                        help="execution mode: sim (default), dry-run, sdk-sim, or real")
    parser.add_argument("--ip", default=None, help="xArm controller IPv4 address (or XARM_IP)")
    parser.add_argument("--calibration", help="robot-bound kinematic calibration YAML")
    parser.add_argument("--confirm-real", action="store_true",
                        help="explicitly authorize single-cycle real robot motion")
    parser.add_argument("--headless", action="store_true", help="run without a viewer")
    parser.add_argument("--dry-run", action="store_true", help="preflight without physics steps")
    parser.add_argument("--report", help="write JSON relative to the current directory")
    parser.add_argument("--cycles", type=positive_int, default=1)
    if command == "box-pack":
        parser.add_argument("--speed", type=positive_float, help="motion speed multiplier")
    args = parser.parse_args(argv)

    if args.dry_run and args.mode not in (None, "dry-run"):
        parser.error("--dry-run cannot be combined with --mode " + str(args.mode))
    effective_mode = "dry-run" if args.dry_run else (args.mode or "sim")
    args.mode = effective_mode
    args.dry_run = effective_mode == "dry-run"

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
        if effective_mode in ("sdk-sim", "real"):
            try:
                code, hw_report = _run_hardware_manipulation(command, args, config)
                try:
                    write_report(args.report, hw_report)
                except (OSError, ValueError, TypeError) as exc:
                    print(f"Report failed: {exc}", file=sys.stderr)
                    code = 1
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
                       mode=effective_mode, exit_code=code)
        try:
            write_report(args.report, payload)
        except (OSError, ValueError, TypeError) as exc:
            print(f"Report failed: {exc}", file=sys.stderr)
            code = 1
    print(f"\n{command}: {'PASS' if code == 0 else 'FAIL'}")
    return code
