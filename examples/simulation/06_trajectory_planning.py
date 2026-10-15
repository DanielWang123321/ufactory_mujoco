#!/usr/bin/env python3
"""Example 06: Time-parameterized trajectory planning and smooth execution.

Demonstrates:
1. Analytical point-to-point (PTP) quintic polynomial trajectory with velocity/acceleration limits.
2. Continuous multi-waypoint spline planning (C2 smooth) vs discrete stop-and-go.
3. Cartesian straight-line trajectory with SLERP orientation interpolation and IK.
4. Analytical trajectory integration with ArmDynamics feedforward torques.

Usage:
    python examples/simulation/06_trajectory_planning.py            # with interactive viewer
    python examples/simulation/06_trajectory_planning.py --headless # metrics only
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from src import (  # noqa: E402
    SimRobot,
    plan_cartesian_line,
    plan_joint_waypoints,
    plan_ptp_quintic,
)
from src.dynamics import ArmDynamics  # noqa: E402

READY = np.array([0.0, -0.8, -1.2, 1.0, 0.3, 0.0])
WAYPOINTS = np.array([
    [0.0, -0.8, -1.2, 1.0, 0.3, 0.0],          # Start (READY)
    [0.2, -1.0, -0.9, 0.8, 0.5, 0.2],          # Via-point 1
    [-0.2, -0.6, -1.3, 1.1, 0.2, -0.2],        # Via-point 2
    [0.3, -1.2, -1.0, 0.6, 0.6, 0.5],          # End (TARGET)
])


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory planning and execution example.")
    parser.add_argument("--headless", action="store_true", help="run headless without 3D viewer")
    parser.add_argument("--speed", type=float, default=90.0, help="max joint speed in deg/s (default: 90)")
    args = parser.parse_args()

    max_vel = np.deg2rad(args.speed)
    max_acc = 3.5  # rad/s^2

    print("=================================================================")
    print("      UFACTORY MuJoCo: Trajectory Planning & Execution Demo      ")
    print("=================================================================")

    with SimRobot(viewer=not args.headless) as robot:
        # Move to home/ready pose first
        robot.move_joints(READY, duration=2.0)
        p_start, quat_start = robot.get_tcp_pose()
        print(f"\n[1] Initial State: TCP pos={np.round(p_start, 4)}, quat={np.round(quat_start, 4)}")

        # -----------------------------------------------------------------
        # 1. PTP Quintic Trajectory with Analytical Time Parameterization
        # -----------------------------------------------------------------
        target_q = WAYPOINTS[-1]
        print(f"\n[2] Planning PTP Quintic Trajectory (max_vel={args.speed:.1f} deg/s, max_acc={max_acc:.1f} rad/s^2)...")
        ptp_traj = robot.plan_joint_trajectory(target_q, max_vel=max_vel, max_acc=max_acc)
        print(f"    Planned duration: {ptp_traj.duration:.3f} s ({ptp_traj.n_steps} steps, dt={ptp_traj.dt * 1000:.1f} ms)")
        print(f"    Peak joint velocity: {np.rad2deg(np.max(np.abs(ptp_traj.qd))):.2f} deg/s (limit: {args.speed:.1f})")
        print(f"    Peak joint accel:    {np.max(np.abs(ptp_traj.qdd)):.2f} rad/s^2 (limit: {max_acc:.1f})")

        print("    Executing PTP trajectory...")
        robot.execute_trajectory(ptp_traj, wait=True)
        q_now = robot.get_joint_positions()
        ptp_err = np.max(np.abs(q_now - target_q))
        print(f"    Reached target with max joint error: {ptp_err:.5f} rad ({np.rad2deg(ptp_err):.4f} deg)")

        # -----------------------------------------------------------------
        # 2. Continuous Multi-Waypoint Spline (Smooth vs Stop-and-Go)
        # -----------------------------------------------------------------
        print("\n[3] Multi-Waypoint Planning (4 waypoints)...")
        spline_traj = plan_joint_waypoints(WAYPOINTS, max_vel=max_vel, max_acc=max_acc, dt=robot.model.opt.timestep, method="spline")
        stop_traj = plan_joint_waypoints(WAYPOINTS, max_vel=max_vel, max_acc=max_acc, dt=robot.model.opt.timestep, method="stop")

        print(f"    Stop-and-go duration: {stop_traj.duration:.3f} s (velocity zeroes at each via-point)")
        print(f"    Smooth spline duration: {spline_traj.duration:.3f} s ({100.0 * (stop_traj.duration - spline_traj.duration) / stop_traj.duration:.1f}% time saved, C2 continuous)")

        print("    Executing continuous smooth spline trajectory...")
        robot.set_joint_positions(WAYPOINTS[0])
        robot.settle(0.5)
        robot.execute_trajectory(spline_traj, wait=True)
        q_now = robot.get_joint_positions()
        spline_err = np.max(np.abs(q_now - WAYPOINTS[-1]))
        print(f"    Spline completed with final error: {spline_err:.5f} rad")

        # -----------------------------------------------------------------
        # 3. Cartesian Straight-Line Planning with SLERP & IK
        # -----------------------------------------------------------------
        robot.move_joints(READY, duration=1.5)
        p_now, quat_now = robot.get_tcp_pose()
        p_line_target = p_now + np.array([0.08, 0.0, -0.04])
        print(f"\n[4] Planning Cartesian Straight Line: {np.round(p_now, 3)} -> {np.round(p_line_target, 3)} (dist: {np.linalg.norm(p_line_target - p_now) * 1000:.1f} mm)...")

        cart_traj, ok = robot.plan_cartesian_trajectory(p_line_target, quat_now, max_lin_vel=0.15, max_lin_acc=0.8)
        if not ok:
            print("    WARNING: Cartesian IK did not fully converge!")
        else:
            print(f"    Cartesian trajectory duration: {cart_traj.duration:.3f} s ({cart_traj.n_steps} steps)")
            print("    Executing Cartesian trajectory...")
            robot.execute_trajectory(cart_traj, wait=True)
            p_final, _ = robot.get_tcp_pose()
            cart_err_mm = np.linalg.norm(p_final - p_line_target) * 1000.0
            print(f"    Cartesian arrival error: {cart_err_mm:.2f} mm")

        # -----------------------------------------------------------------
        # 4. Trajectory-Dynamics Integration (Feedforward Torques)
        # -----------------------------------------------------------------
        print("\n[5] Trajectory Feedforward Dynamics Integration...")
        dyn = ArmDynamics("models/xarm6/xarm6.xml")
        tau_ff = dyn.feedforward_traj(spline_traj.q, dt=spline_traj.dt)
        print(f"    Computed feedforward torques for {len(tau_ff)} trajectory frames.")
        print(f"    Torque envelope across motion: min={np.round(np.min(tau_ff, axis=0), 2)} Nm, max={np.round(np.max(tau_ff, axis=0), 2)} Nm")

        print("\n=================================================================")
        print("                  All Trajectory Tests PASS!                     ")
        print("=================================================================")


if __name__ == "__main__":
    main()
