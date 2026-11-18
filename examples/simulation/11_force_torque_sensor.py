#!/usr/bin/env python3
"""Example 11: 6-axis force/torque sensor (AI1500) reading and wrench verification.

Demonstrates SimRobot's automatic 200 Hz AI1500Reader integration across standalone
``ft_sensor`` and mounted arm/gripper combinations. Moves a mounted arm to its
``ready`` keyframe, records the static tool-gravity baseline, applies a local
wrench at ``ft_sensor_site``, and verifies the net wrench delta.

Usage:
    python examples/simulation/11_force_torque_sensor.py
    python examples/simulation/11_force_torque_sensor.py --headless
    python examples/simulation/11_force_torque_sensor.py --model ft_sensor --force 0 0 -5 --headless
    python examples/simulation/11_force_torque_sensor.py --model xarm6_with_ft_sensor_and_vacuum_gripper --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from ufactory_mujoco import SimRobot, apply_ft_sensor_wrench, clear_ft_sensor_wrench  # noqa: E402
from ufactory_mujoco.ft_sensor import RESOLUTION  # noqa: E402
from ufactory_mujoco.model_paths import resolve_model_path  # noqa: E402


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
        help="local force (N) applied at ft_sensor_site (default: 0 0 -10)",
    )
    ap.add_argument(
        "--torque",
        nargs=3,
        type=float,
        default=(0.0, 0.2, 0.0),
        metavar=("TX", "TY", "TZ"),
        help="local torque (N*m) applied at ft_sensor_site (default: 0 0.2 0)",
    )
    ap.add_argument(
        "--duration",
        type=_positive_float,
        default=1.0,
        help="wrench application duration in seconds (default: 1.0)",
    )
    ap.add_argument("--headless", action="store_true", help="no viewer, print metrics")
    args = ap.parse_args()

    force = np.asarray(args.force, dtype=float)
    torque = np.asarray(args.torque, dtype=float)
    if not np.isfinite(force).all() or not np.isfinite(torque).all():
        ap.error("--force and --torque must be finite")

    model_path = resolve_model_path(args.model)

    with SimRobot(model_path, viewer=not args.headless) as robot:
        if not robot.has_ft_sensor:
            print(
                f"Model '{args.model}' has no force/torque sensor "
                "(supported: 'ft_sensor' and '*_with_ft_sensor*' combinations).",
                file=sys.stderr,
            )
            sys.exit(2)

        print(f"Loaded: {args.model} (dt={robot.model.opt.timestep:.3f} s, 200 Hz sampling)")
        _move_arm_to_ready(robot)
        if robot.n_arm == 0:
            robot.settle(0.1)

        baseline = robot.get_ft_sensor_reading()
        print(f"Baseline wrench (static tool load): {np.round(baseline.raw, 4)}")

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
            return

        net_delta = loaded.raw - baseline.raw
        sampled_delta = loaded.sampled - baseline.sampled
        expected_delta = -np.concatenate((force, torque))
        force_err = float(np.linalg.norm(net_delta[:3] - expected_delta[:3]))
        torque_err = float(np.linalg.norm(net_delta[3:] - expected_delta[3:]))
        sampled_force_err = float(np.linalg.norm(sampled_delta[:3] - expected_delta[:3]))
        sampled_torque_err = float(np.linalg.norm(sampled_delta[3:] - expected_delta[3:]))
        quant_force_tol = float(0.5 + np.linalg.norm(RESOLUTION[:3]))
        quant_torque_tol = float(0.05 + np.linalg.norm(RESOLUTION[3:]))

        print(f"Applied local wrench [F, T]:      {np.round(np.concatenate((force, torque)), 4)}")
        print(f"Loaded raw wrench:                {np.round(loaded.raw, 4)}")
        print(f"Loaded 200 Hz sampled (@{loaded.sample_time_s:.3f}s): {np.round(loaded.sampled, 4)}")
        print(f"Net wrench delta (raw - base):    {np.round(net_delta, 4)}")
        print(f"Expected net delta (-applied):    {np.round(expected_delta, 4)}")
        print(f"Overrange axes: {loaded.overrange.tolist()}  Overload axes: {loaded.overload.tolist()}")
        print(f"Net error: force={force_err:.4f} N (< 0.5), torque={torque_err:.4f} Nm (< 0.05)")

        ok = (
            force_err < 0.5
            and torque_err < 0.05
            and sampled_force_err < quant_force_tol
            and sampled_torque_err < quant_torque_tol
            and np.isfinite(loaded.raw).all()
            and np.isfinite(loaded.sampled).all()
        )
        print("PASS" if ok else "FAIL")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
