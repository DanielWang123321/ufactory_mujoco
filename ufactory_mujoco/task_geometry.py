"""Shared task geometry and bilateral grasp checks; no physics steps in geometry probes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from .sim_robot import SimRobot

GRIP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])


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

