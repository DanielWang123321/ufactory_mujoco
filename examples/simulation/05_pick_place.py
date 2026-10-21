#!/usr/bin/env python3
"""Example 05: cube pick-and-place for five arm and gripper pairs.

Geometry, speed and grasp gap come from configs/pick_place/<robot>.yaml.
servo_cartesian tracks TCP waypoints. servo_j converts each waypoint with IK
and then moves the joints. The arm lifts only after both finger pads touch
the cube.

Success: the cube is lifted, place error < 50 mm, and the TCP returns within
10 mm of the lift pose.

Usage:
    python examples/simulation/05_pick_place.py --headless
    python examples/simulation/05_pick_place.py --robot lite6 --executor servo_j --headless
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from ufactory_mujoco import SimRobot, plan_joint_waypoints  # noqa: E402
from ufactory_mujoco.config import EXECUTORS, load_task_config  # noqa: E402
from ufactory_mujoco.preflight import preflight_pick_place  # noqa: E402
from ufactory_mujoco.trajectory import JointTrajectory  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
PICK_CONFIGS = ROOT / "configs" / "pick_place"
GRIP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # TCP z-axis points down


def _close_until_contact(robot, object_body: str, gap_m: float) -> float:
    """Command ``gap_m``, then shrink it until both pads touch. Do not lift otherwise."""
    gap = gap_m
    robot.set_gripper_gap(gap)
    robot.settle(2.0)
    robot.settle(0.5)
    while not robot.bilateral_finger_contact(object_body) and gap > 0.0:
        gap = max(0.0, gap - 0.002)
        robot.set_gripper_gap(gap)
        robot.settle(0.3)
    robot.require_bilateral_contact(object_body)
    return gap


def _cart_duration(dist: float, task) -> float:
    """Trapezoidal segment time from the configured Cartesian limits."""
    return dist / task.cart_v_m_s + task.cart_v_m_s / task.cart_a_m_s2 + 0.1


def _set_cube(robot: SimRobot, task) -> None:
    joint_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, "cube")
    if joint_id < 0:
        raise ValueError("Scene has no 'cube' joint; load a pick-place scene")
    adr = int(robot.model.jnt_qposadr[joint_id])
    # Keep the authored height so a raised cube stays on its support.
    authored_z = float(robot.model.qpos0[adr + 2])
    robot.data.qpos[adr:adr + 3] = (task.cube_xy[0], task.cube_xy[1], authored_z)
    robot.data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
    dof = int(robot.model.jnt_dofadr[joint_id])
    robot.data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(robot.model, robot.data)


def tcp_task_heights(robot: SimRobot, task) -> tuple[dict[str, float], np.ndarray]:
    """Flange-frame task heights shifted by the measured flange-to-TCP offset."""
    cube_x, cube_y = task.cube_xy
    q_seed, ok = robot.solve_tcp(
        [cube_x, cube_y, task.lift_z_m + 0.15], GRIP_DOWN_QUAT)
    if not ok:
        raise RuntimeError("IK failed while reaching the seed pose")
    q, ok = robot.solve_tcp(
        [cube_x, cube_y, task.lift_z_m], GRIP_DOWN_QUAT, q_init=q_seed)
    if not ok:
        raise RuntimeError("IK failed while measuring the flange->TCP offset")
    robot.eval_data.qpos[:] = q
    mujoco.mj_forward(robot.model, robot.eval_data)
    site_id = robot._site
    z_axis = robot.eval_data.site_xmat[site_id].reshape(3, 3)[:, 2]
    if float(z_axis @ np.array([0.0, 0.0, -1.0])) < 0.999:
        raise RuntimeError(f"TCP is not gripper-down (site z-axis {z_axis})")
    tcp_z = float(robot.eval_data.site_xpos[site_id][2])
    link_name = f"link{robot.n_arm}"
    link_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, link_name)
    flange_z = float(robot.eval_data.body(link_id).xpos[2])
    off = tcp_z - flange_z
    heights = {
        "grasp_z": task.flange_grasp_z_m + off,
        "pre_grasp_z": task.flange_grasp_z_m + task.approach_m + off,
        "lift_z": task.lift_z_m + off,
    }
    q_home, ok = robot.solve_tcp(
        [cube_x, cube_y, heights["lift_z"]], GRIP_DOWN_QUAT, q_init=q)
    if not ok:
        raise RuntimeError("IK failed while solving the deployment pose")
    return heights, q_home


def _move_pose(robot: SimRobot, config, pos: np.ndarray, duration: float,
               q_seed: np.ndarray) -> np.ndarray:
    if config.executor == "servo_j":
        task = config.pick_place
        traj, ok = robot.plan_cartesian_trajectory(
            pos, GRIP_DOWN_QUAT, max_lin_vel=task.cart_v_m_s, max_lin_acc=task.cart_a_m_s2)
        if ok:
            n = robot.n_arm
            robot.execute_trajectory(JointTrajectory(
                traj.times, traj.q[:, :n], traj.qd[:, :n], traj.qdd[:, :n]))
            return robot.data.qpos.copy()
        q, ok = robot.solve_tcp(pos, GRIP_DOWN_QUAT, q_init=q_seed)
        if not ok:
            raise RuntimeError(f"IK failed for joint move to {pos}")
        robot.move_joints(q[:robot.n_arm], duration)
        return q
    robot.move_tcp(pos, GRIP_DOWN_QUAT, duration=duration)
    return robot.data.qpos.copy()


def _print_phases(report) -> None:
    print("Preflight:")
    for phase in report.phases:
        reason = f" ({phase.reason})" if phase.reason else ""
        print(f"  {phase.name}: {phase.status}{reason}")


def _execute_segment(robot: SimRobot, config, name: str, pos: np.ndarray,
                     duration: float, q_seed: np.ndarray, report) -> np.ndarray:
    segment = report.segments.get(name)
    if segment is not None and segment.detour:
        traj = plan_joint_waypoints(
            segment.waypoints, max_vel=np.deg2rad(90.0), max_acc=3.5,
            dt=robot.model.opt.timestep)
        robot.execute_trajectory(traj)
        return robot.data.qpos.copy()
    return _move_pose(robot, config, pos, duration, q_seed)


def run_pick_place(robot: SimRobot, config, cycles: int = 1, *,
                   dry_run: bool = False, report_path: str | None = None):
    """Run the pick-and-place sequence and return one report dict per cycle.

    ``dry_run`` only preflights. A failed preflight does not step the simulation.
    """
    report = preflight_pick_place(robot, config)
    _print_phases(report)
    if dry_run or not report.ok:
        payload = report.as_dict("dry-run" if dry_run else "sim")
        if report_path:
            Path(report_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
        return report if dry_run else []

    task = config.pick_place
    cube_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    if cube_id < 0:
        raise ValueError("Scene has no 'cube' body; load a pick-place scene")
    if robot.gripper_kind != "jaw":
        raise ValueError("Pick-and-place requires a jaw gripper")

    _set_cube(robot, task)
    h, q_seed = tcp_task_heights(robot, task)
    cube_joint = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, "cube")
    cube_z = float(robot.data.qpos[int(robot.model.jnt_qposadr[cube_joint]) + 2])
    target3 = np.array([task.target_xy[0], task.target_xy[1], cube_z])

    def viewer_closed() -> bool:
        return robot._viewer is not None and not robot._viewer.is_running()

    results = []
    for cycle in range(1, cycles + 1):
        if viewer_closed():
            break
        cube_xy = robot.data.body(cube_id).xpos[:2].copy()
        print(f"\n--- Cycle {cycle}/{cycles}: {config.robot} {config.gripper} "
              f"{config.executor}, cube at ({cube_xy[0]:.3f}, {cube_xy[1]:.3f}) ---")

        home = np.array([cube_xy[0], cube_xy[1], h["lift_z"]])
        pre_grasp = np.array([cube_xy[0], cube_xy[1], h["pre_grasp_z"]])
        grasp = np.array([cube_xy[0], cube_xy[1], h["grasp_z"]])
        above_target = np.array([task.target_xy[0], task.target_xy[1], h["lift_z"]])
        place = np.array([task.target_xy[0], task.target_xy[1], h["grasp_z"]])

        print("  Open gripper and deploy above cube ...")
        robot.open_gripper(wait=True)
        deploy_seg = report.segments.get("approach_deploy")
        if deploy_seg is not None and deploy_seg.detour:
            q_seed = _execute_segment(robot, config, "approach_deploy", home, 3.0, q_seed, report)
        else:
            robot.move_joints(q_seed[:robot.n_arm], duration=3.0)
            q_seed = robot.data.qpos.copy()
        deploy = np.array([task.cube_xy[0], task.cube_xy[1], h["lift_z"]])
        q_seed = _move_pose(robot, config, home, _cart_duration(
            float(np.linalg.norm(home - deploy)), task), q_seed)
        if viewer_closed():
            break
        print("  Approach pre-grasp and descend ...")
        q_seed = _execute_segment(robot, config, "approach_pre", pre_grasp, _cart_duration(
            h["lift_z"] - h["pre_grasp_z"], task), q_seed, report)
        q_seed = _execute_segment(robot, config, "approach_grasp", grasp, _cart_duration(
            h["pre_grasp_z"] - h["grasp_z"], task), q_seed, report)
        if viewer_closed():
            break

        print(f"  Close gripper to gap {task.grasp_gap_m * 1000:.0f} mm ...")
        gap = _close_until_contact(robot, "cube", task.grasp_gap_m)
        print(f"  Bilateral pad contact at {gap * 1000:.0f} mm; lifting ...")

        q_seed = _execute_segment(
            robot, config, "抬起", np.array([cube_xy[0], cube_xy[1], h["lift_z"]]),
            _cart_duration(h["lift_z"] - h["grasp_z"], task), q_seed, report)
        cube_lift_z = float(robot.data.body(cube_id).xpos[2])
        cube_lifted = cube_lift_z > task.cube_half_m + 0.05
        q_seed = _execute_segment(robot, config, "运送", above_target, _cart_duration(
            float(np.linalg.norm(above_target - home)), task), q_seed, report)
        if viewer_closed():
            break

        print("  Descend and release ...")
        q_seed = _execute_segment(robot, config, "放置", place, _cart_duration(
            h["lift_z"] - h["grasp_z"], task), q_seed, report)
        robot.open_gripper(wait=True)
        robot.settle(1.0)

        print("  Retreat and return home ...")
        q_seed = _execute_segment(robot, config, "return_up", above_target, _cart_duration(
            h["lift_z"] - h["grasp_z"], task), q_seed, report)
        _execute_segment(robot, config, "return_home", home, _cart_duration(
            float(np.linalg.norm(home - above_target)), task), q_seed, report)
        robot.settle(0.5)

        cube_final = robot.data.body(cube_id).xpos.copy()
        tcp_final, _ = robot.get_tcp_pose()
        place_error_mm = float(np.linalg.norm(cube_final - target3) * 1000.0)
        home_drift_mm = float(np.linalg.norm(tcp_final - home) * 1000.0)
        result = {
            "robot": config.robot,
            "gripper": config.gripper,
            "executor": config.executor,
            "grasp_gap_m": gap,
            "cycle": cycle,
            "cube_lifted": bool(cube_lifted),
            "place_error_mm": place_error_mm,
            "home_drift_mm": home_drift_mm,
        }
        results.append(result)
        print(f"  cube_lifted={result['cube_lifted']} (z={cube_lift_z:.3f})  "
              f"place_error={place_error_mm:.1f} mm  home_drift={home_drift_mm:.1f} mm")
    if report_path:
        payload = report.as_dict("sim")
        payload["cycles"] = results
        Path(report_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    return results


def _load(robot: str, gripper: str | None, executor: str | None, config: str | None):
    overlay = config or (PICK_CONFIGS / f"{robot}.yaml")
    overrides = {}
    if gripper:
        overrides["gripper"] = gripper
    if executor:
        overrides["executor"] = executor
    return load_task_config(overlay, overrides or None)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cube pick-and-place for a configured arm and gripper.")
    ap.add_argument("--robot", default="xarm6",
                    help="arm key; selects configs/pick_place/<robot>.yaml")
    ap.add_argument("--gripper", default=None, help="override the overlay gripper")
    ap.add_argument("--executor", default=None, choices=EXECUTORS)
    ap.add_argument("--config", default=None, help="overlay YAML instead of the robot default")
    ap.add_argument("--headless", action="store_true", help="no viewer, print metrics")
    ap.add_argument("--dry-run", action="store_true",
                    help="preflight only; do not step the simulation")
    ap.add_argument("--report", default=None, help="write the preflight or cycle report as JSON")
    ap.add_argument("--cycles", type=int, default=1)
    args = ap.parse_args()

    config = _load(args.robot, args.gripper, args.executor, args.config)
    with SimRobot(config.scene_path, viewer=not args.headless and not args.dry_run) as robot:
        outcome = run_pick_place(robot, config, cycles=args.cycles,
                                 dry_run=args.dry_run, report_path=args.report)

    if args.dry_run:
        ok = bool(outcome.ok)
    else:
        ok = bool(outcome) and all(
            r["cube_lifted"] and r["place_error_mm"] < 50.0 and r["home_drift_mm"] < 10.0
            for r in outcome
        )
    print("\nPick-and-place " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
