"""Shared reporting, motion, config overlay, and CLI scaffolding for simulation tasks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np

from ..config import ConfigError, EXECUTORS, ModelResourceError, _read_yaml, load_task_config
from ..preflight import (
    GRIP_DOWN_QUAT,
    make_box_pack_checker,
    make_checker,
    preflight_box_pack,
    preflight_pick_place,
)
from ..robots import ARMS, MODELS_ROOT
from ..safety import (
    resolve_xarm_ip,
    SafetyGateError,
    authorize_real_motion,
    enforce_single_cycle,
    run_sdk_simulation,
    stream_sha256,
)
from ..trajectory import JointTrajectory, plan_joint_waypoints, plan_ptp_quintic
from ..xarm_robot import XArmRobot

MODES = ("sim", "dry-run", "sdk-sim", "real")

# Simulated pick-place and box-pack start every arm with the TCP at [300, 0, 300] mm,
# RPY [180, 0, 0] deg. Hardware modes start from the arm's measured pose.
TASK_START = ((0.30, 0.0, 0.30), tuple(float(v) for v in GRIP_DOWN_QUAT))


def write_report(path, payload):
    """Write ``payload`` as strict JSON to ``path``; ``None`` skips writing."""
    if path is None:
        return
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text + "\n", encoding="utf-8")


def task_report(preflight, config, dry_run, requested_cycles, cycles=None):
    """Build the task report from preflight results, config and per-cycle results."""
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
    payload.update(ok=success, success=success, completed=completed)
    return payload


def set_cube(robot, joint_name: str, xy, scene_kind: str) -> None:
    """Place a free-joint cube at ``xy`` on its authored height, upright and at rest."""
    joint_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Scene has no {joint_name!r} joint; load a {scene_kind} scene")
    adr = int(robot.model.jnt_qposadr[joint_id])
    authored_z = float(robot.model.qpos0[adr + 2])
    robot.data.qpos[adr:adr + 3] = (xy[0], xy[1], authored_z)
    robot.data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
    dof = int(robot.model.jnt_dofadr[joint_id])
    robot.data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(robot.model, robot.data)


@dataclass(frozen=True)
class Mover:
    """Executor and already-scaled Cartesian/detour limits for one task run."""

    robot: Any
    executor: str
    max_lin_vel: float
    max_lin_acc: float
    detour_max_vel: float
    detour_max_acc: float

    def duration(self, dist: float) -> float:
        """Trapezoidal segment time for distance ``dist`` under the Cartesian limits."""
        return dist / self.max_lin_vel + self.max_lin_vel / self.max_lin_acc + 0.1

    def move(self, pos: np.ndarray, duration: float, q_seed: np.ndarray, *,
             wait: bool = True) -> np.ndarray:
        """Move the TCP to ``pos`` with a grip-down orientation and return the resulting joint seed.

        ``wait=False`` skips the final settle, for a motion that the next one continues.
        """
        robot = self.robot
        if self.executor == "servo_j":
            traj, ok = robot.plan_cartesian_trajectory(
                pos, GRIP_DOWN_QUAT, max_lin_vel=self.max_lin_vel, max_lin_acc=self.max_lin_acc)
            if ok:
                n = robot.n_arm
                robot.execute_trajectory(JointTrajectory(
                    traj.times, traj.q[:, :n], traj.qd[:, :n], traj.qdd[:, :n]), wait=wait)
                return robot.data.qpos.copy()
            q, ok = robot.solve_tcp(pos, GRIP_DOWN_QUAT, q_init=q_seed)
            if not ok:
                raise RuntimeError(f"IK failed for joint move to {pos}")
            robot.move_joints(q[:robot.n_arm], duration, wait=wait)
            return q
        robot.move_tcp(pos, GRIP_DOWN_QUAT, duration=duration, wait=wait)
        return robot.data.qpos.copy()

    def segment(self, name: str, pos: np.ndarray, duration: float, q_seed: np.ndarray,
                report, *, wait: bool = True) -> np.ndarray:
        """Run preflight segment ``name``: a joint-space detour when preflight planned one, else ``move``."""
        segment = report.segments.get(name)
        if segment is not None and segment.detour:
            traj = plan_joint_waypoints(
                segment.waypoints, max_vel=self.detour_max_vel, max_acc=self.detour_max_acc,
                dt=self.robot.model.opt.timestep)
            self.robot.execute_trajectory(traj, wait=wait)
            return self.robot.data.qpos.copy()
        return self.move(pos, duration, q_seed, wait=wait)


def print_phases(report) -> None:
    """Print each preflight phase status and its failure reason."""
    print("Preflight:")
    for phase in report.phases:
        reason = f" ({phase.reason})" if phase.reason else ""
        print(f"  {phase.name}: {phase.status}{reason}")


def load_overlay(task_name: str, robot: str | None, gripper: str | None, executor: str | None,
                 config: str | None, extra: dict[str, Any] | None = None):
    """Load ``config`` (or built-in ``(task_name, robot)`` preset) with optional CLI overrides."""
    overrides: dict[str, Any] = {}
    if gripper:
        overrides["gripper"] = gripper
    if executor:
        overrides["executor"] = executor
    if extra:
        overrides.update(extra)

    if config is None:
        effective_robot = robot or "xarm6"
        return load_task_config(preset=(task_name, effective_robot), overrides=overrides or None)

    config_path = Path(config)
    if robot is None:
        return load_task_config(config_path, overrides or None)

    custom_raw = _read_yaml(config_path)
    if "robot" in custom_raw and custom_raw["robot"] != robot:
        raise ConfigError(
            f"--robot {robot!r} conflicts with robot {custom_raw['robot']!r} in {config_path}"
        )
    return load_task_config(config_path, overrides or None, preset=(task_name, robot))


def positive_int(value):
    """argparse type for positive integers."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_float(value):
    """argparse type for finite positive floats."""
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def _build_shadow_stream(
    report,
    n_arm: int,
    *,
    max_vel_rad_s: float = 0.8,
    max_acc_rad_s2: float = 4.0,
    min_seg_duration: float = 0.4,
    dt: float = 0.02,
) -> np.ndarray:
    """Build a velocity- and acceleration-bounded joint trajectory array ``(N, n_arm)`` from preflight segments."""
    chunks: list[np.ndarray] = []
    for seg in report.segments.values():
        w = np.asarray(seg.waypoints, dtype=np.float64)
        for i in range(len(w) - 1):
            q0 = w[i, :n_arm]
            q1 = w[i + 1, :n_arm]
            disp = float(np.max(np.abs(q1 - q0)))
            t_v = disp * 1.875 / max_vel_rad_s
            t_a = math.sqrt(5.7735 * disp / max_acc_rad_s2)
            duration = max(min_seg_duration, t_v, t_a)
            traj = plan_ptp_quintic(q0, q1, duration=duration, dt=dt)
            chunks.append(traj.q if not chunks else traj.q[1:])
    if not chunks:
        return np.zeros((1, n_arm), dtype=np.float64)
    return np.vstack(chunks)


def _run_hardware_manipulation(command: str, args, config) -> tuple[int, dict]:
    """Execute sdk-sim or real mode with fail-closed calibration, preflight, and cycle gates."""
    mode = args.mode
    enforce_single_cycle(args.cycles, mode)
    ip = resolve_xarm_ip(args.ip)
    if not ip:
        raise SafetyGateError(f"mode={mode!r} requires --ip or XARM_IP")
    if not args.calibration:
        raise SafetyGateError(f"mode={mode!r} requires --calibration YAML")

    with XArmRobot(
        ip,
        robot=config.robot,
        gripper=config.gripper,
        scene=config.scene,
        model_path=config.scene_spec(),
        calibration=args.calibration,
        read_only=True,
    ) as robot:
        if command == "pick-place":
            report = preflight_pick_place(robot, config)
            checker = make_checker(config.scene_spec())
            task_cfg = config.pick_place
        else:
            report = preflight_box_pack(robot, config, cycles=1)
            checker = make_box_pack_checker(config.scene_spec())
            task_cfg = config.box_pack

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

        q_live = robot.get_joint_positions()
        live_drift = float(np.max(np.abs(q_live - shadow_q[0])))
        if live_drift > 0.02:
            raise SafetyGateError(
                f"joint positions drifted by {live_drift:.4f} rad since preflight (> 0.0200 rad); "
                "refusing real motion"
            )
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
        grasp_gap_m = float(task_cfg.grasp_gap_m)
        gripper_actuated = False
        measured_grasp_gap_m: float | None = None
        if robot.gripper_kind == "jaw":
            robot.open_gripper(wait=True)
            robot.settle(0.3)
            gripper_actuated = True
        for seg_name, seg in report.segments.items():
            w = np.asarray(seg.waypoints, dtype=np.float64)
            for i in range(1, len(w)):
                robot.move_joints(w[i, : robot.n_arm], duration=1.5, wait=True)
            if robot.gripper_kind == "jaw":
                if seg_name == "approach_grasp" or seg_name.startswith("approach_grasp#"):
                    robot.set_gripper_gap(grasp_gap_m)
                    robot.settle(0.5)
                    measured_grasp_gap_m = float(robot.get_gripper_gap(measured=True))
                elif seg_name == "放置" or seg_name.startswith("释放#"):
                    robot.open_gripper(wait=True)
                    robot.settle(0.5)
        q_end = robot.get_joint_positions()
        home_err_rad = float(np.max(np.abs(q_end - shadow_q[-1])))
        ok = bool(report.ok and home_err_rad < 0.05)
        object_grasped_inferred = (
            bool(measured_grasp_gap_m > grasp_gap_m + 0.002)
            if measured_grasp_gap_m is not None
            else False
        )
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
            gripper_actuated=gripper_actuated,
            commanded_grasp_gap_m=grasp_gap_m if gripper_actuated else None,
            measured_grasp_gap_m=measured_grasp_gap_m,
            object_grasped_inferred=object_grasped_inferred,
            final_joint_error_rad=home_err_rad,
            ok=ok,
            success=ok,
            completed=True,
        )
        return (0 if ok else 1), payload


def manipulation_main(command, argv=None):
    """Shared CLI for ``pick-place`` and ``box-pack``: parse arguments, run the mode, return the exit code."""
    task = importlib.import_module(f"ufactory_mujoco.tasks.{command.replace('-', '_')}")
    parser = argparse.ArgumentParser(prog=f"ufactory-mujoco {command}", description=task.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", choices=tuple(ARMS), default=None,
                        help="arm key (defaults to xarm6 when --config is omitted)")
    parser.add_argument("--gripper", help="override the overlay gripper")
    parser.add_argument("--executor", choices=EXECUTORS,
                        help="servo_cartesian (default) tracks TCP waypoints; servo_j moves the joints through IK")
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
    parser.add_argument("--cycles", type=positive_int, default=1,
                        help="number of cycles (default: 1; box-pack: up to one per feed cube; "
                             "sdk-sim and real: 1)")
    if command == "box-pack":
        parser.add_argument("--speed", type=positive_float,
                            help="motion speed multiplier (default: the preset speed_scale, 1.0)")
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
        expected_scene = "pick_place" if command == "pick-place" else "box_pack"
        if config.scene != expected_scene:
            raise ConfigError(
                f"{command} requires scene={expected_scene!r}, got {config.scene!r}"
            )
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
                with task.SimRobot(config.scene_spec(), viewer=not args.headless and not args.dry_run,
                                   start=TASK_START) as robot:
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
