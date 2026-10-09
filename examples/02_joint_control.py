#!/usr/bin/env python3
"""Example 02: joint-space position control with quintic interpolation.

Moves ready -> demo -> ready and reports final positioning error.

Usage:
    python examples/02_joint_control.py            # with interactive viewer
    python examples/02_joint_control.py --headless # offscreen, prints metrics
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src import SimRobot  # noqa: E402

READY = [0, -0.8, -1.2, 1.0, 0.3, 0]  # ready pose (keyframe 1)
DEMO  = [0.3, -1.2, -1.0, 0.6, 0.6, 0.5]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="no viewer, print metrics")
    args = ap.parse_args()

    with SimRobot(viewer=not args.headless) as robot:
        print("start joints:", np.round(robot.get_joint_positions(), 4))

        targets = [READY, DEMO, READY]
        for q in targets:
            robot.move_joints(np.array(q), duration=2.5)
            if robot._viewer is not None and not robot._viewer.is_running():
                break

        if args.headless:
            err = np.abs(robot.get_joint_positions() - READY)
            print(f"final joints: {np.round(robot.get_joint_positions(), 4)}")
            print(f"max |error| per joint (rad): {np.round(err, 5)}")
            ok = err.max() < 0.01
            print("PASS" if ok else "FAIL", f"(max error {err.max():.5f} rad < 0.01)")


if __name__ == "__main__":
    main()
