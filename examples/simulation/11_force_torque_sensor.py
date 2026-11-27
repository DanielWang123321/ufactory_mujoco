#!/usr/bin/env python3
"""Example 11: 6-axis force/torque sensor (AI1500) reading and wrench verification.

Demonstrates ``SimRobot``'s calibrated 200 Hz ``AI1500Reader`` integration
(dual ``raw`` / ``ext`` channels, ``+`` = applied external wrench onto the
output face, 2-body mass/CoM split, IEEE-754 float32 quantization) across
standalone ``ft_sensor`` and mounted arm/gripper combinations, plus strictly
read-only live hardware F/T sampling via ``XArmRobot(..., read_only=True)``.

Usage:
    python examples/simulation/11_force_torque_sensor.py
    python examples/simulation/11_force_torque_sensor.py --model ft_sensor --force 0 0 -5
    python examples/simulation/11_force_torque_sensor.py --model xarm6_with_ft_sensor_and_vacuum_gripper
    python examples/simulation/11_force_torque_sensor.py --headless   # no viewer, prints metrics
    python examples/simulation/11_force_torque_sensor.py --ip <ROBOT_IP> --robot xarm6 --gripper ft_sensor
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import mujoco
import numpy as np

from ufactory_mujoco import SimRobot, XArmRobot, apply_ft_sensor_wrench, clear_ft_sensor_wrench
from ufactory_mujoco import resolve_model_path


def _positive_float(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def _move_arm_to_ready(robot: SimRobot) -> None:
    if robot.n_arm == 0:
        return
    ready_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    if ready_id < 0:
        return
    ready_q = robot.model.key(ready_id).qpos[: robot.n_arm].copy()
    print(f"Positioning arm in ready pose (deg): {np.round(np.rad2deg(ready_q), 1)}")
    robot.move_joints(ready_q, duration=1.5, wait=False)
    robot.settle(0.5)


def _run_simulation(args: argparse.Namespace) -> int:
    force = np.asarray(args.force, dtype=float)
    torque = np.asarray(args.torque, dtype=float)
    if not np.isfinite(force).all() or not np.isfinite(torque).all():
        raise SystemExit("--force and --torque must be finite")

    model_path = resolve_model_path(args.model)

    try:
        sim_ctx = SimRobot(model_path, viewer=not args.headless)
    except ValueError as exc:
        if "AL0700" in str(exc):
            print(f"{exc} Use examples/simulation/12_linear_motor.py instead.", file=sys.stderr)
            return 2
        raise

    with sim_ctx as robot:
        if not robot.has_ft_sensor:
            print(
                f"Model '{args.model}' has no force/torque sensor "
                "(supported: 'ft_sensor' and '*_with_ft_sensor*' combinations).",
                file=sys.stderr,
            )
            return 2

        print(f"Loaded: {args.model} (dt={robot.model.opt.timestep:.3f} s, 200 Hz fp32 sampling)")
        _move_arm_to_ready(robot)
        if robot.n_arm == 0:
            robot.settle(0.1)

        baseline = robot.get_ft_sensor_reading()
        print(f"Baseline raw (floating head + downstream load): {np.round(baseline.raw, 4)}")
        print(f"Baseline ext (gravity-compensated free space):  {np.round(baseline.ext, 4)}")

        n_steps = max(1, round(args.duration / robot.model.opt.timestep))
        stopped_early = False
        for _ in range(n_steps):
            if robot.has_viewer and not robot.viewer_is_running:
                stopped_early = True
                break
            apply_ft_sensor_wrench(robot.model, robot.data, force, torque)
            robot.step()

        loaded = robot.get_ft_sensor_reading()
        clear_ft_sensor_wrench(robot.model, robot.data)
        mujoco.mj_forward(robot.model, robot.data)

        if stopped_early:
            print("Viewer closed; stopped early.")
            return 0

        net_delta = loaded.raw - baseline.raw
        sampled_delta = loaded.sampled - baseline.sampled
        expected_delta = np.concatenate((force, torque))
        force_err = float(np.linalg.norm(net_delta[:3] - expected_delta[:3]))
        torque_err = float(np.linalg.norm(net_delta[3:] - expected_delta[3:]))
        ext_force_err = float(np.linalg.norm(loaded.ext[:3] - expected_delta[:3]))
        ext_torque_err = float(np.linalg.norm(loaded.ext[3:] - expected_delta[3:]))
        sampled_force_err = float(np.linalg.norm(sampled_delta[:3] - expected_delta[:3]))
        sampled_torque_err = float(np.linalg.norm(sampled_delta[3:] - expected_delta[3:]))
        base_ext_force = float(np.linalg.norm(baseline.ext[:3]))
        base_ext_torque = float(np.linalg.norm(baseline.ext[3:]))
        fp32_exact = np.allclose(loaded.sampled, loaded.sampled.astype(np.float32).astype(np.float64), atol=0, rtol=0)

        print(f"Applied local wrench [F, T]:      {np.round(expected_delta, 4)}")
        print(f"Loaded raw wrench:                {np.round(loaded.raw, 4)}")
        print(f"Loaded ext wrench:                {np.round(loaded.ext, 4)}")
        print(f"Loaded 200 Hz sampled (@{loaded.sample_time_s:.3f}s): {np.round(loaded.sampled, 4)}")
        print(f"Net wrench delta (raw - base):    {np.round(net_delta, 4)}")
        print(f"Expected net delta (+applied):    {np.round(expected_delta, 4)}")
        print(f"Overrange axes: {loaded.overrange.tolist()}  Overload axes: {loaded.overload.tolist()}")
        print(
            f"Errors: raw_delta=({force_err:.4f} N, {torque_err:.4f} Nm), "
            f"ext=({ext_force_err:.4f} N, {ext_torque_err:.4f} Nm)"
        )

        ok = (
            base_ext_force < 0.05
            and base_ext_torque < 0.005
            and force_err < 0.5
            and torque_err < 0.05
            and ext_force_err < 0.5
            and ext_torque_err < 0.05
            and sampled_force_err < 0.5
            and sampled_torque_err < 0.05
            and fp32_exact
            and np.isfinite(loaded.raw).all()
            and np.isfinite(loaded.ext).all()
            and np.isfinite(loaded.sampled).all()
        )
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1


def _run_hardware(args: argparse.Namespace) -> int:
    gripper = None if args.gripper == "none" else args.gripper
    print(f"Connecting read-only to {args.ip} (robot={args.robot}, gripper={gripper})")
    with XArmRobot(
        args.ip,
        robot=args.robot,
        gripper=gripper,
        calibration=args.calibration,
        read_only=True,
    ) as robot:
        if not robot.has_ft_sensor:
            print("Controller does not report an enabled force/torque sensor.", file=sys.stderr)
            return 2
        cfg = robot.get_ft_sensor_config()
        print(
            f"HW F/T config: freq={cfg['ft_freq_hz']} Hz, mass={cfg['ft_mass_kg']:.3f} kg, "
            f"centroid_m={np.round(cfg['ft_centroid_m'], 5).tolist()}"
        )
        raw_samples: list[np.ndarray] = []
        ext_samples: list[np.ndarray] = []
        overrange_any = np.zeros(6, dtype=bool)
        overload_any = np.zeros(6, dtype=bool)
        deadline = time.monotonic() + args.duration
        while True:
            reading = robot.get_ft_sensor_reading()
            raw_samples.append(reading.raw)
            ext_samples.append(reading.ext)
            overrange_any |= reading.overrange
            overload_any |= reading.overload
            if time.monotonic() >= deadline:
                break
            time.sleep(0.005)

    raw_arr = np.stack(raw_samples, axis=0)
    ext_arr = np.stack(ext_samples, axis=0)
    print(f"Collected {len(raw_samples)} read-only samples over {args.duration:.2f} s")
    print(f"HW raw mean: {np.round(raw_arr.mean(axis=0), 4)}  std: {np.round(raw_arr.std(axis=0), 4)}")
    print(f"HW ext mean: {np.round(ext_arr.mean(axis=0), 4)}  std: {np.round(ext_arr.std(axis=0), 4)}")
    print(f"Overrange axes: {overrange_any.tolist()}  Overload axes: {overload_any.tolist()}")
    ok = (
        len(raw_samples) >= 1
        and np.isfinite(raw_arr).all()
        and np.isfinite(ext_arr).all()
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Demonstrate AI1500 6-axis force/torque sensor readings.")
    ap.add_argument(
        "--model",
        default="xarm6_with_ft_sensor_and_gripper",
        help="Model name (e.g. 'ft_sensor', 'xarm6_with_ft_sensor', "
        "'xarm6_with_ft_sensor_and_gripper', 'xarm6_with_ft_sensor_and_vacuum_gripper')",
    )
    ap.add_argument(
        "--force",
        nargs=3,
        type=float,
        default=(0.0, 0.0, -10.0),
        metavar=("FX", "FY", "FZ"),
        help="local force (N) applied onto the sensor (default: 0 0 -10)",
    )
    ap.add_argument(
        "--torque",
        nargs=3,
        type=float,
        default=(0.0, 0.2, 0.0),
        metavar=("TX", "TY", "TZ"),
        help="local torque (N*m) applied onto the sensor (default: 0 0.2 0)",
    )
    ap.add_argument(
        "--duration",
        type=_positive_float,
        default=1.0,
        help="wrench application or hardware sampling duration in seconds (default: 1.0)",
    )
    ap.add_argument("--ip", default=None, help="xArm controller IP for read-only live F/T sampling")
    ap.add_argument("--robot", default="xarm6", help="robot model for --ip hardware mode (default: xarm6)")
    ap.add_argument("--gripper", default="ft_sensor", help="attached assembly for --ip mode (default: ft_sensor)")
    ap.add_argument("--calibration", type=Path, default=None, help="optional calibration YAML for --ip mode")
    ap.add_argument("--headless", action="store_true", help="no viewer, print metrics")
    args = ap.parse_args()

    code = _run_hardware(args) if args.ip else _run_simulation(args)
    if code != 0:
        sys.exit(code)


if __name__ == "__main__":
    main()
