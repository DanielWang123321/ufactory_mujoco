#!/usr/bin/env python3
"""Example 10: box-packing showcase for five arm and gripper pairs.

Geometry, speed and grasp gap come from configs/box_pack/<robot>.yaml.
Each cycle picks one feed cube, transports it above the box, and releases it
release_above_rim_m above the rim; the cube free-falls and may rebound.
speed_scale shortens commanded durations; free-fall physics is not scaled.

Success: every released cube is at rest with its collision bounds inside the
box (1 mm contact tolerance) and the TCP returns within 10 mm of home.

Usage:
    python examples/simulation/10_box_pack.py --headless
    python examples/simulation/10_box_pack.py --robot lite6 --cycles 3 --speed 2.0 --headless
    python examples/simulation/10_box_pack.py --dry-run
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from ufactory_mujoco import SimRobot, plan_joint_waypoints  # noqa: E402
from ufactory_mujoco.config import EXECUTORS, load_task_config  # noqa: E402
from ufactory_mujoco.preflight import preflight_box_pack  # noqa: E402
from ufactory_mujoco.trajectory import JointTrajectory  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
PACK_CONFIGS = ROOT / "configs" / "box_pack"
GRIP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # TCP z-axis points down
IN_BOX_TOL_M = 1e-3
_REST_LIN_VEL = 2e-3
_REST_ANG_VEL = 2e-2
_REST_HOLD_S = 0.15

_close_until_contact = importlib.import_module(
    "examples.simulation.05_pick_place")._close_until_contact


def _set_cube(robot: SimRobot, cube_name: str, xy) -> None:
    joint_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, cube_name)
    if joint_id < 0:
        raise ValueError(f"Scene has no {cube_name!r} joint; load a box-pack scene")
    adr = int(robot.model.jnt_qposadr[joint_id])
    # Keep the authored height so a raised cube stays on its support.
    authored_z = float(robot.model.qpos0[adr + 2])
    robot.data.qpos[adr:adr + 3] = (xy[0], xy[1], authored_z)
    robot.data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
    dof = int(robot.model.jnt_dofadr[joint_id])
    robot.data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(robot.model, robot.data)


def _cube_rest_z(robot: SimRobot, cube_name: str = "cube_0") -> float:
    joint_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, cube_name)
    if joint_id < 0:
        raise ValueError(f"Scene has no {cube_name!r} joint; load a box-pack scene")
    return float(robot.model.qpos0[int(robot.model.jnt_qposadr[joint_id]) + 2])


def box_task_heights(robot: SimRobot, task) -> tuple[dict[str, float], np.ndarray]:
    """Flange-frame task heights shifted by the measured flange-to-TCP offset."""
    cx, cy = task.box_center_xy
    q_seed, ok = robot.solve_tcp([cx, cy, task.lift_z_m + 0.15], GRIP_DOWN_QUAT)
    if not ok:
        raise RuntimeError("IK failed while reaching the seed pose")
    q, ok = robot.solve_tcp([cx, cy, task.lift_z_m], GRIP_DOWN_QUAT, q_init=q_seed)
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
    grasp_z = task.flange_grasp_z_m + off
    # Held-cube center rides (grasp_z - cube_rest_z) below the TCP; releasing at
    # release_z leaves the cube bottom release_above_rim_m above the rim.
    heights = {
        "grasp_z": grasp_z,
        "pre_grasp_z": grasp_z + task.approach_m,
        "lift_z": task.lift_z_m + off,
        "release_z": (task.box_rim_z_m + task.release_above_rim_m + task.cube_half_m
                      + (grasp_z - _cube_rest_z(robot))),
    }
    q_home, ok = robot.solve_tcp([cx, cy, heights["lift_z"]], GRIP_DOWN_QUAT, q_init=q)
    if not ok:
        raise RuntimeError("IK failed while solving the deployment pose")
    return heights, q_home


def _scaled_vel_acc(task) -> tuple[float, float]:
    s = task.speed_scale
    return task.cart_v_m_s * s, task.cart_a_m_s2 * s * s


def _cart_duration(task, dist: float) -> float:
    """Trapezoidal segment time from the configured Cartesian limits."""
    v, a = _scaled_vel_acc(task)
    return dist / v + v / a + 0.1


def _wait(task, seconds: float) -> float:
    return seconds / task.speed_scale


def _move_pose(robot: SimRobot, config, pos: np.ndarray, duration: float,
               q_seed: np.ndarray) -> np.ndarray:
    if config.executor == "servo_j":
        task = config.box_pack
        v, a = _scaled_vel_acc(task)
        traj, ok = robot.plan_cartesian_trajectory(pos, GRIP_DOWN_QUAT,
                                                   max_lin_vel=v, max_lin_acc=a)
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


def _execute_segment(robot: SimRobot, config, name: str, pos: np.ndarray,
                     duration: float, q_seed: np.ndarray, report) -> np.ndarray:
    segment = report.segments.get(name)
    if segment is not None and segment.detour:
        s = config.box_pack.speed_scale
        traj = plan_joint_waypoints(
            segment.waypoints, max_vel=np.deg2rad(90.0) * s, max_acc=3.5 * s,
            dt=robot.model.opt.timestep)
        robot.execute_trajectory(traj)
        return robot.data.qpos.copy()
    return _move_pose(robot, config, pos, duration, q_seed)


def _settle_cube(robot: SimRobot, cube_name: str, timeout: float) -> bool:
    """Free-fall settle: step until the cube is at rest or the timeout hits."""
    joint_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, cube_name)
    dof = int(robot.model.jnt_dofadr[joint_id])
    state = {"since": None}

    def at_rest() -> bool:
        vel = robot.data.qvel[dof:dof + 6]
        if (np.linalg.norm(vel[:3]) < _REST_LIN_VEL
                and np.linalg.norm(vel[3:]) < _REST_ANG_VEL):
            if state["since"] is None:
                state["since"] = robot.data.time
            return robot.data.time - state["since"] >= _REST_HOLD_S
        state["since"] = None
        return False

    return robot.settle_until(at_rest, timeout)


def cube_box_margin(task, pos: np.ndarray, quat: np.ndarray) -> float:
    """Signed clearance (m) of the cube's collision bounds inside the box."""
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, np.asarray(quat, dtype=float))
    ext = np.abs(rot.reshape(3, 3)) @ np.full(3, task.cube_half_m)
    cx, cy = task.box_center_xy
    hx, hy = task.box_inner_half_xy
    return min(
        hx - (abs(pos[0] - cx) + ext[0]),
        hy - (abs(pos[1] - cy) + ext[1]),
        (pos[2] - ext[2]) - task.box_floor_m,
        task.box_rim_z_m - (pos[2] + ext[2]),
    )


def _print_phases(report) -> None:
    print("Preflight:")
    for phase in report.phases:
        reason = f" ({phase.reason})" if phase.reason else ""
        print(f"  {phase.name}: {phase.status}{reason}")


def run_box_pack(robot: SimRobot, config, cycles: int = 1, *,
                 dry_run: bool = False, report_path: str | None = None):
    """Run the box-pack sequence and return one report dict per cycle.

    ``dry_run`` only preflights. A failed preflight does not step the simulation.
    """
    report = preflight_box_pack(robot, config, cycles)
    _print_phases(report)
    if dry_run or not report.ok:
        payload = report.as_dict("dry-run" if dry_run else "sim")
        if report_path:
            Path(report_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
        return report if dry_run else []

    task = config.box_pack
    if robot.gripper_kind != "jaw":
        raise ValueError("Box-packing requires a jaw gripper")
    h, q_seed = box_task_heights(robot, task)
    n_arm = robot.n_arm

    def viewer_closed() -> bool:
        return robot._viewer is not None and not robot._viewer.is_running()

    results = []
    for i in range(cycles):
        tag = f"#{i + 1}"
        cube = f"cube_{i}"
        feed = np.asarray(task.feed_xy[i], dtype=float)
        drop = np.asarray(task.drop_xy[i], dtype=float)
        _set_cube(robot, cube, feed)
        cube_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, cube)
        cube_z0 = _cube_rest_z(robot, cube)
        if viewer_closed():
            break
        print(f"\n--- Cycle {i + 1}/{cycles}: {config.robot} {config.gripper} "
              f"{config.executor} x{task.speed_scale:g}, {cube} at "
              f"({feed[0]:.3f}, {feed[1]:.3f}) ---")

        home = np.array([feed[0], feed[1], h["lift_z"]])
        pre_grasp = np.array([feed[0], feed[1], h["pre_grasp_z"]])
        grasp = np.array([feed[0], feed[1], h["grasp_z"]])
        above_box = np.array([drop[0], drop[1], h["lift_z"]])
        release = np.array([drop[0], drop[1], h["release_z"]])

        q_home, ok = robot.solve_tcp(home, GRIP_DOWN_QUAT, q_init=q_seed)
        if not ok:
            raise RuntimeError(f"IK failed for the deployment pose above {feed}")

        print("  Open gripper and deploy above the feed cube ...")
        robot.open_gripper(wait=False)
        robot.settle(_wait(task, 0.5))
        deploy_seg = report.segments.get(f"approach_deploy{tag}")
        if deploy_seg is not None and deploy_seg.detour:
            q_seed = _execute_segment(robot, config, f"approach_deploy{tag}", home,
                                      _wait(task, 3.0), q_home, report)
        else:
            robot.move_joints(q_home[:n_arm], duration=_wait(task, 3.0))
            q_seed = robot.data.qpos.copy()
        print("  Approach pre-grasp and descend ...")
        q_seed = _execute_segment(robot, config, f"approach_pre{tag}", pre_grasp,
                                  _cart_duration(task, h["lift_z"] - h["pre_grasp_z"]),
                                  q_seed, report)
        q_seed = _execute_segment(robot, config, f"approach_grasp{tag}", grasp,
                                  _cart_duration(task, h["pre_grasp_z"] - h["grasp_z"]),
                                  q_seed, report)
        if viewer_closed():
            break

        print(f"  Close gripper to gap {task.grasp_gap_m * 1000:.0f} mm ...")
        gap = _close_until_contact(robot, cube, task.grasp_gap_m)
        print(f"  Bilateral pad contact at {gap * 1000:.0f} mm; lifting ...")

        q_seed = _execute_segment(robot, config, f"抬起{tag}", home,
                                  _cart_duration(task, h["lift_z"] - h["grasp_z"]),
                                  q_seed, report)
        cube_lift_z = float(robot.data.body(cube_id).xpos[2])
        cube_lifted = cube_lift_z > cube_z0 + 0.05
        q_seed = _execute_segment(robot, config, f"运送{tag}", above_box,
                                  _cart_duration(task, float(np.linalg.norm(
                                      above_box - home))), q_seed, report)
        if viewer_closed():
            break

        print(f"  Release {task.release_above_rim_m * 1000:.0f} mm above the rim ...")
        q_seed = _execute_segment(robot, config, f"释放{tag}", release,
                                  _cart_duration(task, h["lift_z"] - h["release_z"]),
                                  q_seed, report)
        released_z = float(robot.data.body(cube_id).xpos[2])
        robot.open_gripper(wait=False)
        rested = _settle_cube(robot, cube, task.settle_timeout_s)

        print("  Retreat and return home ...")
        q_seed = _execute_segment(robot, config, f"return_up{tag}", above_box,
                                  _cart_duration(task, h["lift_z"] - h["release_z"]),
                                  q_seed, report)
        _execute_segment(robot, config, f"return_home{tag}", home,
                         _cart_duration(task, float(np.linalg.norm(home - above_box))),
                         q_seed, report)
        robot.settle(_wait(task, 0.5))

        rest_pos = robot.data.body(cube_id).xpos.copy()
        rest_quat = robot.data.body(cube_id).xquat.copy()
        margin_m = cube_box_margin(task, rest_pos, rest_quat)
        in_box = bool(rested) and bool(margin_m >= -IN_BOX_TOL_M)
        tcp_final, _ = robot.get_tcp_pose()
        home_drift_mm = float(np.linalg.norm(tcp_final - home) * 1000.0)
        result = {
            "robot": config.robot,
            "gripper": config.gripper,
            "executor": config.executor,
            "speed_scale": task.speed_scale,
            "cycle": i + 1,
            "cube": cube,
            "grasp_gap_m": gap,
            "cube_lifted": bool(cube_lifted),
            "released_z": released_z,
            "cube_rested": bool(rested),
            "rest_xyz": [float(v) for v in rest_pos],
            "bounds_margin_mm": float(margin_m * 1000.0),
            "in_box": in_box,
            "home_drift_mm": home_drift_mm,
        }
        results.append(result)
        print(f"  cube_lifted={result['cube_lifted']} (z={cube_lift_z:.3f})  "
              f"in_box={in_box} (margin={result['bounds_margin_mm']:.1f} mm)  "
              f"home_drift={home_drift_mm:.1f} mm")
    if report_path:
        payload = report.as_dict("sim")
        payload["cycles"] = results
        Path(report_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    return results


def _load(robot: str, gripper: str | None, executor: str | None, config: str | None,
          speed: float | None):
    overlay = config or (PACK_CONFIGS / f"{robot}.yaml")
    overrides = {}
    if gripper:
        overrides["gripper"] = gripper
    if executor:
        overrides["executor"] = executor
    if speed is not None:
        overrides["box_pack"] = {"speed_scale": speed}
    return load_task_config(overlay, overrides or None)


def main() -> None:
    ap = argparse.ArgumentParser(description="Box-packing showcase for a configured arm and gripper.")
    ap.add_argument("--robot", default="xarm6",
                    help="arm key; selects configs/box_pack/<robot>.yaml")
    ap.add_argument("--gripper", default=None, help="override the overlay gripper")
    ap.add_argument("--executor", default=None, choices=EXECUTORS)
    ap.add_argument("--config", default=None, help="overlay YAML instead of the robot default")
    ap.add_argument("--headless", action="store_true", help="no viewer, print metrics")
    ap.add_argument("--dry-run", action="store_true",
                    help="preflight only; do not step the simulation")
    ap.add_argument("--report", default=None, help="write the preflight or cycle report as JSON")
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--speed", type=float, default=None,
                    help="motion speed multiplier (>1 is faster; free fall is not scaled)")
    args = ap.parse_args()

    config = _load(args.robot, args.gripper, args.executor, args.config, args.speed)
    with SimRobot(config.scene_path, viewer=not args.headless and not args.dry_run) as robot:
        outcome = run_box_pack(robot, config, cycles=args.cycles,
                               dry_run=args.dry_run, report_path=args.report)

    if args.dry_run:
        ok = bool(outcome.ok)
    else:
        ok = bool(outcome) and all(
            r["cube_lifted"] and r["in_box"] and r["home_drift_mm"] < 10.0
            for r in outcome
        )
    print("\nBox-pack " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
