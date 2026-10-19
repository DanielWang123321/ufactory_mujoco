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


import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from ufactory_mujoco import SimRobot  # noqa: E402
from ufactory_mujoco.model_paths import resolve_model_path  # noqa: E402


def move_arm_to_ready(robot: SimRobot) -> None:
    """Move a mounted arm to its own model's ready keyframe."""
    if robot._n_arm_qpos == 0:
        return
    ready_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    if ready_id < 0:
        print("No ready keyframe; keeping the initial arm pose.")
        return
    ready_pose = robot.model.key(ready_id).qpos[:robot._n_arm_qpos].copy()
    print(f"Positioning arm in ready pose (deg): {np.round(np.rad2deg(ready_pose), 1)}")
    robot.move_joints(ready_pose, duration=1.5, wait=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Demonstrate Gripper G2 / Bio Gripper G2 / Lite 6 gripper motion.")
    ap.add_argument(
        "--model",
        default="xarm6_with_gripper",
        help="Model name (e.g. 'xarm6_with_gripper', 'xarm6_with_bio_gripper', 'lite6_with_gripper', 'gripper_g2', 'bio_gripper_g2', 'lite6_gripper')",
    )
    ap.add_argument("--headless", action="store_true", help="no viewer, print metrics")
    ap.add_argument("--cycles", type=int, default=3, help="number of open/close cycles (default: 3)")
    args = ap.parse_args()

    model_path = resolve_model_path(args.model)

    with SimRobot(model_path, viewer=not args.headless) as robot:
        if not robot.has_gripper:
            print(f"Model '{args.model}' has no gripper actuator "
                  "(supported: gripper_g2, bio_gripper_g2, lite6_gripper and their arm combos).",
                  file=sys.stderr)
            sys.exit(2)

        print(f"Loaded: {args.model} (gripper actuator id: {robot._gripper_actuator_id})")
        print(f"Gripper ctrl range: {robot._gripper_ctrl_range}")
        print("Starting open/close motion demonstration...")

        move_arm_to_ready(robot)

        for cycle in range(1, args.cycles + 1):
            if robot._viewer is not None and not robot._viewer.is_running():
                break

            print(f"\n--- Cycle {cycle}/{args.cycles} ---")

            # 1. Close gripper
            print("  Action: Close gripper (target=1.0)...")
            robot.close_gripper(wait=True)
            pos_close = robot.get_gripper_position()
            print(f"  Closed state: normalized={pos_close:.3f}, gripper_qpos={np.round(robot.data.qpos[robot._n_arm_qpos:], 4)}")

            # 2. Open gripper
            print("  Action: Open gripper (target=0.0)...")
            robot.open_gripper(wait=True)
            pos_open = robot.get_gripper_position()
            print(f"  Open state:   normalized={pos_open:.3f}, gripper_qpos={np.round(robot.data.qpos[robot._n_arm_qpos:], 4)}")

            # 3. Half open
            print("  Action: Half grip (target=0.5)...")
            robot.set_gripper(0.5)
            robot.settle(0.5)
            pos_half = robot.get_gripper_position()
            print(f"  Half state:   normalized={pos_half:.3f}, gripper_qpos={np.round(robot.data.qpos[robot._n_arm_qpos:], 4)}")

            # Return to open
            robot.open_gripper(wait=True)

        print("\nGripper demonstration completed successfully.")


if __name__ == "__main__":
    main()
