#!/usr/bin/env python3
"""Example 09: collision-free joint-space planning around obstacles (roadmap 0.0.6).

Scene (models/xarm6/scene_obstacles.xml): xArm 6 with a vertical pillar in the
sweep between keyframes 'start' and 'goal' plus a horizontal board overhead.
The straight joint interpolation between the keyframes collides with the
pillar (printed as the baseline); the planner detours around it.

Pipeline: RRT-Connect -> random shortcutting -> C2 spline time parameterization
(velocity/acceleration limited) -> SimRobot execution with per-step penetration
monitoring. If the spline cuts a corner into collision, a stop-and-go quintic
parameterization is used instead (its segments are straight in joint space).

Usage:
    python examples/simulation/09_obstacle_avoidance.py                # GUI
    python examples/simulation/09_obstacle_avoidance.py --headless     # metrics
    python examples/simulation/09_obstacle_avoidance.py --seed 3 --safety-margin 0.01
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from ufactory_mujoco import SimRobot, plan_joint_waypoints  # noqa: E402
from ufactory_mujoco.collision import CollisionChecker  # noqa: E402
from ufactory_mujoco.model_paths import MODELS_ROOT  # noqa: E402
from ufactory_mujoco.path_planning import path_length, plan_collision_free, shortcut_path  # noqa: E402

SCENE = MODELS_ROOT / "xarm6" / "scene_obstacles.xml"
START_KEY, GOAL_KEY = "start", "goal"
MAX_VEL = np.deg2rad(90.0)   # rad/s, joint example convention
MAX_ACC = 3.5                # rad/s^2
FINAL_ERROR_TOL = 0.01       # rad
PENETRATION_TOL = 1e-6       # m, repository contact convention
SAMPLES = 21                 # straight-line baseline sampling


def key_qpos(model: mujoco.MjModel, name: str) -> np.ndarray:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if key_id < 0:
        raise ValueError(f"scene has no keyframe '{name}'")
    return model.key(key_id).qpos.copy()


def _straight_baseline(checker: CollisionChecker, q0: np.ndarray, q1: np.ndarray):
    states = np.linspace(0.0, 1.0, SAMPLES)[:, None] * (q1 - q0) + q0
    hits = sum(checker.in_collision(q) for q in states)
    dists = [checker.min_distance(q) for q in states]
    finite = [d for d in dists if np.isfinite(d)]
    return hits, float(min(finite)) if finite else float("inf")


def _timed_trajectory(checker: CollisionChecker, path: np.ndarray):
    """Time-parameterize a waypoint path, keeping the timed states collision-free."""
    for method in ("spline", "stop"):
        traj = plan_joint_waypoints(path, max_vel=MAX_VEL, max_acc=MAX_ACC,
                                    method=method)
        if not any(checker.in_collision(q) for q in traj.q):
            return traj, method
    raise RuntimeError("timed trajectory leaves the collision-free path")


def run_avoidance(robot: SimRobot, checker: CollisionChecker, *, seed: int = 0,
                  step: float = 0.15, max_iter: int = 2000,
                  preview: bool = False) -> dict:
    """Plan and execute start->goal around the obstacles; returns metric dict."""
    q_start = key_qpos(robot.model, START_KEY)
    q_goal = key_qpos(robot.model, GOAL_KEY)
    hits, straight_min = _straight_baseline(checker, q_start, q_goal)

    waypoints = plan_collision_free(q_start, q_goal, checker=checker,
                                    seed=seed, step=step, max_iter=max_iter)
    short = shortcut_path(checker, waypoints, seed=seed)
    traj, method = _timed_trajectory(checker, short)

    path_states = np.linspace(0.0, 1.0, 61)[:, None] * (q_goal - q_start) + q_start
    clearances = [checker.min_distance(q) for q in np.vstack([short, traj.q[::20]])]
    finite = [d for d in clearances if np.isfinite(d)]

    metrics = {
        "straight_blocked": hits > 0,
        "straight_colliding_samples": int(hits),
        "straight_min_dist_m": straight_min,
        "rrt_waypoints": int(len(waypoints)),
        "waypoints": int(len(short)),
        "path_length_rad": path_length(short),
        "timed_method": method,
        "traj_duration_s": float(traj.duration),
        "min_clearance_m": float(min(finite)) if finite else float("inf"),
    }

    if preview:
        _preview(robot, np.vstack([path_states]), [1.0, 0.2, 0.2, 0.8])
        _preview(robot, traj.q[:: max(len(traj.q) // 40, 1)], [0.2, 0.9, 0.2, 0.8])

    monitor = {"max_penetration": 0.0}
    original_step = robot._step

    def checked_step():
        original_step()
        for contact in robot.data.contact:
            monitor["max_penetration"] = max(monitor["max_penetration"],
                                             -float(contact.dist))

    robot._step = checked_step
    try:
        robot.move_joints(q_start[: robot._n_arm_qpos], duration=1.5)
        robot.execute_trajectory(traj)
    finally:
        robot._step = original_step

    error = float(np.max(np.abs(robot.get_joint_positions() - q_goal)))
    metrics["final_error_rad"] = error
    metrics["max_penetration_m"] = monitor["max_penetration"]
    metrics["success"] = bool(
        metrics["straight_blocked"]
        and metrics["min_clearance_m"] > 0.0
        and monitor["max_penetration"] <= PENETRATION_TOL
        and error < FINAL_ERROR_TOL
    )
    return metrics


def _preview(robot: SimRobot, q_states: np.ndarray, rgba) -> None:
    """Walk a marker along the FK path (GUI only, best effort)."""
    if robot._viewer is None:
        return
    try:
        marker_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_GEOM,
                                      "path_marker_geom")
        mocap_id = int(robot.model.body("path_marker").mocapid[0])
    except Exception:
        return
    robot.model.geom_rgba[marker_id] = rgba
    for p in robot.fk_positions(q_states):
        robot.data.mocap_pos[mocap_id] = p
        robot._viewer.sync()
        time.sleep(0.02)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Collision-free planning around obstacles (xArm 6).")
    ap.add_argument("--headless", action="store_true", help="no viewer, print metrics")
    ap.add_argument("--seed", type=int, default=0, help="RRT/shortcut seed (default: 0)")
    ap.add_argument("--safety-margin", type=float, default=0.002,
                    help="planning clearance in meters (default: 0.002)")
    ap.add_argument("--step", type=float, default=0.15, help="RRT steer step (rad)")
    ap.add_argument("--max-iter", type=int, default=2000)
    args = ap.parse_args()

    checker = CollisionChecker(SCENE, safety_margin=args.safety_margin)
    with SimRobot(SCENE, viewer=not args.headless) as robot:
        metrics = run_avoidance(robot, checker, seed=args.seed, step=args.step,
                                max_iter=args.max_iter, preview=not args.headless)

    print(f"\nstraight-line baseline: blocked={metrics['straight_blocked']} "
          f"({metrics['straight_colliding_samples']}/{SAMPLES} samples collide, "
          f"min dist {metrics['straight_min_dist_m'] * 1000:.1f} mm)")
    print(f"planned path: {metrics['rrt_waypoints']} RRT -> {metrics['waypoints']} "
          f"waypoints after shortcut, length {metrics['path_length_rad']:.3f} rad, "
          f"min clearance {metrics['min_clearance_m'] * 1000:.1f} mm")
    print(f"execution ({metrics['timed_method']}, {metrics['traj_duration_s']:.2f} s): "
          f"final error {metrics['final_error_rad']:.5f} rad, "
          f"max penetration {metrics['max_penetration_m'] * 1000:.3f} mm")
    print("\nObstacle avoidance " + ("PASS" if metrics["success"] else "FAIL"))
    sys.exit(0 if metrics["success"] else 1)


if __name__ == "__main__":
    main()
