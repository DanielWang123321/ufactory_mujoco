#!/usr/bin/env python3
"""Example 03: Cartesian-space control via DLS inverse kinematics.

From the ready pose: straight line out and back (move_tcp), then a circle in
the X-Z plane tracked by IK waypoints. Reports TCP tracking error.

Usage:
    python examples/simulation/03_cartesian_ik.py            # with interactive viewer
    python examples/simulation/03_cartesian_ik.py --headless # offscreen, prints metrics
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from ufactory_mujoco import SimRobot

READY = [0, -0.8, -1.2, 1.0, 0.3, 0]
CIRCLE_RADIUS = 0.07
CIRCLE_POINTS = 72


def circle_waypoints(center: np.ndarray, r: float, n: int) -> np.ndarray:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.tile(center, (n, 1))
    pts[:, 0] += r * np.cos(ang)
    pts[:, 2] += r * np.sin(ang)
    return pts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="no viewer, print metrics")
    args = ap.parse_args()

    with SimRobot(viewer=not args.headless) as robot:
        robot.move_joints(np.array(READY), duration=2.0)
        p0, q0 = robot.get_tcp_pose()
        print(f"start TCP: pos={np.round(p0, 4)} quat={np.round(q0, 4)}")

        # --- 1) straight line: +0.15 m along X, then back (pose fully constrained)
        robot.move_tcp(p0 + np.array([0.15, 0, 0]), q0, duration=2.5)
        p1, _ = robot.get_tcp_pose()
        line_err_mm = float(np.linalg.norm(p1 - (p0 + [0.15, 0, 0])) * 1e3)
        robot.move_tcp(p0, q0, duration=2.5)
        print(f"line segment done, max deviation from line goal: {line_err_mm:.2f} mm")

        # --- 2) circle in X-Z plane, orientation locked, IK waypoint tracking
        waypoints = circle_waypoints(p0, CIRCLE_RADIUS, CIRCLE_POINTS)
        q_traj, ik_ok = robot.ik_trajectory(waypoints, q0, pos_tol=5e-5)
        if not ik_ok:
            print("WARNING: some circle waypoints did not fully converge")

        for q in q_traj:
            robot.move_joints(q, duration=4 * robot.model.opt.timestep, wait=False)
            if robot.has_viewer and not robot.viewer_is_running:
                return
        robot.settle(0.5)

        # --- tracking evaluation: FK replay of the solved joint path
        fk = robot.fk_positions(q_traj)
        errs = np.linalg.norm(fk - waypoints, axis=1) * 1e3
        print(f"circle tracking error (FK vs command): mean={errs.mean():.3f} mm  "
              f"max={errs.max():.3f} mm")
        ok = bool(ik_ok and line_err_mm < 2.0 and errs.max() < 1.0)
        print("PASS" if ok else "FAIL", f"(line {line_err_mm:.2f} mm < 2 mm, circle max {errs.max():.3f} mm < 1 mm)")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
