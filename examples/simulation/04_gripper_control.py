#!/usr/bin/env python3
"""Example 04: Gripper open/close position control.

Demonstrates controlling Gripper G2, Bio Gripper G2, and the Lite 6 gripper
(mounted or standalone) using position commands [0.0 = open, 1.0 = closed].

Usage:
    # xArm 6 + Gripper G2
    python examples/simulation/04_gripper_control.py --model xarm6_with_gripper

    # xArm 6 + Bio Gripper G2
    python examples/simulation/04_gripper_control.py --model xarm6_with_bio_gripper

    # Lite 6 + parallel gripper (Studio factory ready pose)
    python examples/simulation/04_gripper_control.py --model lite6_with_gripper

    # Standalone Grippers
    python examples/simulation/04_gripper_control.py --model gripper_g2
    python examples/simulation/04_gripper_control.py --model bio_gripper_g2

    # Headless mode (metrics only)
    python examples/simulation/04_gripper_control.py --model xarm6_with_gripper --headless
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import mujoco
import numpy as np

from ufactory_mujoco import SimRobot
from ufactory_mujoco import resolve_model_path


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def _speed_percent(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or not (0.0 <= number <= 100.0):
        raise argparse.ArgumentTypeError("must be finite in [0, 100]")
    return number


def _force_percent(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or not (1.0 <= number <= 100.0):
        raise argparse.ArgumentTypeError("must be finite in [1, 100]")
    return number


def move_arm_to_ready(robot: SimRobot) -> None:
    """Move a mounted arm to its own model's ready keyframe."""
    if robot.n_arm == 0:
        return
    ready_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    if ready_id < 0:
        print("No ready keyframe; keeping the initial arm pose.")
        return
    ready_pose = robot.model.key(ready_id).qpos[:robot.n_arm].copy()
    print(f"Positioning arm in ready pose (deg): {np.round(np.rad2deg(ready_pose), 1)}")
    robot.move_joints(ready_pose, duration=1.5, wait=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Demonstrate Gripper G2 / Bio Gripper G2 / Lite 6 gripper motion.")
    ap.add_argument(
        "--model",
        default="xarm6_with_gripper",
        help="Model name (e.g. 'xarm6_with_gripper', 'xarm6_with_bio_gripper', 'lite6_with_gripper', "
             "'gripper_g2', 'bio_gripper_g2', 'lite6_gripper')",
    )
    ap.add_argument("--headless", action="store_true", help="no viewer, print metrics")
    ap.add_argument("--cycles", type=_positive_int, default=3, help="number of open/close cycles (default: 3)")
    ap.add_argument(
        "--speed-percent",
        type=_speed_percent,
        default=100.0,
        help="Gripper G2 speed percent [0..100] (default: 100)",
    )
    ap.add_argument(
        "--force-percent",
        type=_force_percent,
        default=100.0,
        help="Gripper G2 force percent [1..100] (default: 100)",
    )
    args = ap.parse_args()

    model_path = resolve_model_path(args.model)

    with SimRobot(model_path, viewer=not args.headless) as robot:
        if not robot.has_gripper:
            print(f"Model '{args.model}' has no gripper actuator "
                  "(supported: gripper_g2, bio_gripper_g2, lite6_gripper and their arm combos).",
                  file=sys.stderr)
            sys.exit(2)

        print(f"Loaded: {args.model} (gripper actuator id: {robot.gripper_actuator_id})")
        print(f"Gripper ctrl range: {robot.gripper_ctrl_range}")
        print("Starting open/close motion demonstration...")

        move_arm_to_ready(robot)

        ok = True
        for cycle in range(1, args.cycles + 1):
            if robot.has_viewer and not robot.viewer_is_running:
                return

            print(f"\n--- Cycle {cycle}/{args.cycles} ---")

            # 1. Close gripper
            print(
                f"  Action: Close gripper (target=1.0, speed={args.speed_percent:.0f}%, "
                f"force={args.force_percent:.0f}%)..."
            )
            robot.close_gripper(
                wait=True,
                speed_percent=args.speed_percent,
                force_percent=args.force_percent,
            )
            pos_close = robot.get_gripper_position()
            gap_close = robot.get_gripper_gap(measured=True)
            out_close = robot.gripper_control_output
            if out_close is not None and hasattr(out_close, "torque_limit_nm"):
                tau_str = f", torque_limit={float(out_close.torque_limit_nm[0]):.3f} Nm"
            elif out_close is not None and hasattr(out_close, "force_limit_n"):
                tau_str = f", force_limit={float(out_close.force_limit_n):.2f} N"
            else:
                tau_str = ""
            print(
                f"  Closed state: normalized={pos_close:.3f}, "
                f"measured_gap={gap_close * 1000:.2f} mm, "
                f"gripper_qpos={np.round(robot.gripper_qpos, 4)}{tau_str}"
            )

            # 2. Open gripper
            print("  Action: Open gripper (target=0.0)...")
            robot.open_gripper(
                wait=True,
                speed_percent=args.speed_percent,
                force_percent=args.force_percent,
            )
            pos_open = robot.get_gripper_position()
            gap_open = robot.get_gripper_gap(measured=True)
            print(
                f"  Open state:   normalized={pos_open:.3f}, "
                f"measured_gap={gap_open * 1000:.2f} mm, "
                f"gripper_qpos={np.round(robot.gripper_qpos, 4)}"
            )

            # 3. Half open
            print("  Action: Half grip (target=0.5)...")
            robot.set_gripper(
                0.5,
                speed_percent=args.speed_percent,
                force_percent=args.force_percent,
            )
            robot.settle(robot.gripper_wait_duration())
            pos_half = robot.get_gripper_position()
            gap_half = robot.get_gripper_gap(measured=True)
            print(
                f"  Half state:   normalized={pos_half:.3f}, "
                f"measured_gap={gap_half * 1000:.2f} mm, "
                f"gripper_qpos={np.round(robot.gripper_qpos, 4)}"
            )

            # Return to open
            robot.open_gripper(
                wait=True,
                speed_percent=args.speed_percent,
                force_percent=args.force_percent,
            )
            if args.speed_percent > 0.0 and not (gap_close + 0.002 < gap_half < gap_open - 0.002):
                ok = False

        print("\nGripper demonstration " + ("PASS" if ok else "FAIL"))
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
