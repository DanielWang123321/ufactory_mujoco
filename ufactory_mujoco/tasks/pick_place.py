#!/usr/bin/env python3
"""Cube pick-and-place for five arm and gripper pairs.

Geometry, speed and grasp gap come from the pick_place presets in configs/default.yaml;
all five arms share the cube spot, the target and the TCP lift height. The simulated
task start (``TASK_START``) points the TCP down above the cube, so the arm descends to
the grasp in one straight motion; from any other pose it first deploys above the cube.
servo_cartesian tracks TCP waypoints. servo_j converts each waypoint with IK
and then moves the joints. The arm lifts only after both finger pads touch
the cube.

Success: the cube is lifted, place error < 50 mm, and the TCP returns within
10 mm of the lift pose.

Usage:
    ufactory-mujoco pick-place
    ufactory-mujoco pick-place --robot lite6 --executor servo_j
"""

import mujoco
import numpy as np

from ufactory_mujoco import SimRobot
from ufactory_mujoco.preflight import (
    GRIP_DOWN_QUAT,
    _close_until_contact,
    preflight_pick_place,
    tcp_task_heights,
)
from . import _common
from ._common import manipulation_main, task_report, write_report


def _points_down_above(pos: np.ndarray, quat: np.ndarray, xy, min_z: float) -> bool:
    """True when a TCP pose points down within 2 mm of ``xy`` and at least ``min_z`` high."""
    tilt = 2.0 * np.arccos(min(1.0, abs(float(np.dot(quat, GRIP_DOWN_QUAT)))))
    return (float(np.linalg.norm(pos[:2] - np.asarray(xy))) < 2e-3 and float(pos[2]) >= min_z
            and tilt < np.deg2rad(1.0))


def run_pick_place(robot: SimRobot, config, cycles: int = 1, *,
                   dry_run: bool = False, report_path: str | None = None):
    """Run the pick-and-place sequence and return one report dict per cycle.

    ``dry_run`` only preflights. A failed preflight does not step the simulation.
    """
    if type(cycles) is not int or cycles <= 0:
        raise ValueError("cycles must be a positive integer")
    report = preflight_pick_place(robot, config)
    _common.print_phases(report)
    if dry_run or not report.ok:
        write_report(report_path, task_report(report, config, dry_run, cycles))
        return report if dry_run else []

    task = config.pick_place
    cube_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    if cube_id < 0:
        raise ValueError("Scene has no 'cube' body; load a pick-place scene")
    if robot.gripper_kind != "jaw":
        raise ValueError("Pick-and-place requires a jaw gripper")
    mover = _common.Mover(robot, config.executor, task.cart_v_m_s, task.cart_a_m_s2,
                          np.deg2rad(90.0), 3.5)

    _common.set_cube(robot, "cube", task.cube_xy, "pick-place")
    h, q_seed = tcp_task_heights(robot, task)
    cube_joint = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, "cube")
    cube_z = float(robot.data.qpos[int(robot.model.jnt_qposadr[cube_joint]) + 2])
    target3 = np.array([task.target_xy[0], task.target_xy[1], cube_z])

    def viewer_closed() -> bool:
        return robot.has_viewer and not robot.viewer_is_running

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

        # The approach runs without settles: each motion hands over to the next one.
        robot.open_gripper(wait=True)
        approach = [report.segments.get(name)
                    for name in ("approach_deploy", "approach_pre", "approach_grasp")]
        detour = any(seg is not None and seg.detour for seg in approach)
        pos, quat = robot.get_tcp_pose()
        if not detour and _points_down_above(pos, quat, cube_xy, h["pre_grasp_z"]):
            # The task start pose is above the cube: one straight descent to the grasp pose.
            print("  Descend to the grasp pose ...")
            q_seed = mover.move(grasp, mover.duration(float(pos[2]) - h["grasp_z"]), q_seed,
                                wait=False)
        else:
            print("  Deploy above cube ...")
            deploy_seg = approach[0]
            if deploy_seg is not None and deploy_seg.detour:
                q_seed = mover.segment("approach_deploy", home, 3.0, q_seed, report, wait=False)
            else:
                robot.move_joints(q_seed[:robot.n_arm], duration=3.0, wait=False)
                q_seed = robot.data.qpos.copy()
            deploy = np.array([task.cube_xy[0], task.cube_xy[1], h["lift_z"]])
            shift = float(np.linalg.norm(home - deploy))
            if shift > 1e-3:  # the cube is not on its preset spot
                q_seed = mover.move(home, mover.duration(shift), q_seed)
            if viewer_closed():
                break
            print("  Descend to the grasp pose ...")
            if any(seg is not None and seg.detour for seg in approach[1:]):
                q_seed = mover.segment("approach_pre", pre_grasp, mover.duration(
                    h["lift_z"] - h["pre_grasp_z"]), q_seed, report, wait=False)
                q_seed = mover.segment("approach_grasp", grasp, mover.duration(
                    h["pre_grasp_z"] - h["grasp_z"]), q_seed, report, wait=False)
            else:
                # pre_grasp lies on the straight line down to grasp: one motion, no stop.
                q_seed = mover.move(grasp, mover.duration(h["lift_z"] - h["grasp_z"]), q_seed,
                                    wait=False)
        # Close as soon as the lagging arm reaches the grasp height, not after a fixed settle.
        # The lag is along the descent; a static sideways offset (1.4 mm on the UF850) remains.
        def at_grasp() -> bool:
            err = robot.get_tcp_pose()[0] - grasp
            return abs(float(err[2])) < 1e-3 and float(np.linalg.norm(err[:2])) < 3e-3

        robot.settle_until(at_grasp, 1.0)
        if viewer_closed():
            break

        print(f"  Close gripper to gap {task.grasp_gap_m * 1000:.0f} mm ...")
        gap = _close_until_contact(robot, "cube", task.grasp_gap_m)
        print(f"  Bilateral pad contact at {gap * 1000:.0f} mm; lifting ...")

        q_seed = mover.segment(
            "抬起", np.array([cube_xy[0], cube_xy[1], h["lift_z"]]),
            mover.duration(h["lift_z"] - h["grasp_z"]), q_seed, report)
        cube_lift_z = float(robot.data.body(cube_id).xpos[2])
        cube_lifted = cube_lift_z > task.cube_half_m + 0.05
        q_seed = mover.segment("运送", above_target, mover.duration(
            float(np.linalg.norm(above_target - home))), q_seed, report)
        if viewer_closed():
            break

        print("  Descend and release ...")
        q_seed = mover.segment("放置", place, mover.duration(
            h["lift_z"] - h["grasp_z"]), q_seed, report)
        robot.open_gripper(wait=True)
        robot.settle(1.0)

        print("  Retreat and return home ...")
        q_seed = mover.segment("return_up", above_target, mover.duration(
            h["lift_z"] - h["grasp_z"]), q_seed, report)
        mover.segment("return_home", home, mover.duration(
            float(np.linalg.norm(home - above_target))), q_seed, report)
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
            "bilateral_contact": True,
            "place_error_mm": place_error_mm,
            "home_drift_mm": home_drift_mm,
        }
        results.append(result)
        print(f"  cube_lifted={result['cube_lifted']} (z={cube_lift_z:.3f})  "
              f"place_error={place_error_mm:.1f} mm  home_drift={home_drift_mm:.1f} mm")
    write_report(report_path, task_report(report, config, False, cycles, results))
    return results


def _load(robot: str | None, gripper: str | None, executor: str | None, config: str | None):
    return _common.load_overlay("pick_place", robot, gripper, executor, config)


def main(argv=None) -> None:
    """``ufactory-mujoco pick-place`` command line entry point."""
    raise SystemExit(manipulation_main("pick-place", argv))


if __name__ == "__main__":
    main()
