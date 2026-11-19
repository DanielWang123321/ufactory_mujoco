"""Pick-and-place and box-packing task geometry, preflight checks, and detours.

Waypoint IK and height probes use ``robot.eval_data`` and never step physics;
``_close_until_contact`` steps the simulation while closing the gripper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from .calibration import apply_calibration_to_model
from .planning import CollisionChecker, PlanningError, plan_collision_free

if TYPE_CHECKING:
    from .sim_robot import SimRobot

GRIP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])

# Phase names are report data keys (approach, close/grasp, lift, transport, place/release,
# return); they stay unchanged so existing reports and tooling keep matching.
PHASES = ("接近", "闭合", "抬起", "运送", "放置", "回位")
_IGNORED_BODIES = ("cube", "pick_stand", "place_stand")
_BOX_PACK_PREFIXES = ("cube", "feed_stand", "pick_stand", "place_stand")
_BASE_BODIES = ("link_base", "xarm_base_plate", "uf850_base_plate", "lite6_base_plate")
_FINGER_PAIR = ("uflite_finger1", "uflite_finger2")
_CHECK_RES = 0.02


def _close_until_contact(robot, object_body: str, gap_m: float) -> float:
    """Command ``gap_m``, then shrink it until both pads touch. Do not lift otherwise."""
    spec = getattr(robot, "_gripper_spec", None)
    min_gap = float(spec.closed_gap_m) if spec is not None else 0.0
    gap = gap_m
    robot.set_gripper_gap(gap)
    robot.settle(2.0)
    robot.settle(0.5)
    while not robot.bilateral_finger_contact(object_body) and gap > min_gap:
        gap = max(min_gap, gap - 0.002)
        robot.set_gripper_gap(gap)
        robot.settle(0.3)
    robot.require_bilateral_contact(object_body)
    return gap


def _flange_tcp_offset(robot: SimRobot, x: float, y: float, lift_z: float) -> tuple[float, np.ndarray]:
    """Solve a grip-down pose above ``(x, y)`` and return the TCP-minus-flange z offset and its joints."""
    q_seed, ok = robot.solve_tcp([x, y, lift_z + 0.15], GRIP_DOWN_QUAT, q_init=robot.model.qpos0.copy())
    if not ok:
        raise RuntimeError("IK failed while reaching the seed pose")
    q, ok = robot.solve_tcp([x, y, lift_z], GRIP_DOWN_QUAT, q_init=q_seed)
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
    return tcp_z - flange_z, q


def _deploy_pose(robot: SimRobot, x: float, y: float, lift_z: float, q: np.ndarray) -> np.ndarray:
    """Joint pose with the TCP at ``(x, y, lift_z)``, seeded from ``q``."""
    q_home, ok = robot.solve_tcp([x, y, lift_z], GRIP_DOWN_QUAT, q_init=q)
    if not ok:
        raise RuntimeError("IK failed while solving the deployment pose")
    return q_home


def tcp_task_heights(robot: SimRobot, task) -> tuple[dict[str, float], np.ndarray]:
    """Flange-frame task heights shifted by the measured flange-to-TCP offset."""
    cube_x, cube_y = task.cube_xy
    off, q = _flange_tcp_offset(robot, cube_x, cube_y, task.lift_z_m)
    heights = {
        "grasp_z": task.flange_grasp_z_m + off,
        "pre_grasp_z": task.flange_grasp_z_m + task.approach_m + off,
        "lift_z": task.lift_z_m + off,
    }
    return heights, _deploy_pose(robot, cube_x, cube_y, heights["lift_z"], q)


def _cube_rest_z(robot: SimRobot, cube_name: str = "cube_0") -> float:
    joint_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, cube_name)
    if joint_id < 0:
        raise ValueError(f"Scene has no {cube_name!r} joint; load a box-pack scene")
    return float(robot.model.qpos0[int(robot.model.jnt_qposadr[joint_id]) + 2])


def box_task_heights(robot: SimRobot, task) -> tuple[dict[str, float], np.ndarray]:
    """Flange-frame task heights shifted by the measured flange-to-TCP offset."""
    cx, cy = task.box_center_xy
    off, q = _flange_tcp_offset(robot, cx, cy, task.lift_z_m)
    grasp_z = task.flange_grasp_z_m + off
    heights = {
        "grasp_z": grasp_z,
        "pre_grasp_z": grasp_z + task.approach_m,
        "lift_z": task.lift_z_m + off,
        "release_z": (task.box_rim_z_m + task.release_above_rim_m + task.cube_half_m
                      + (grasp_z - _cube_rest_z(robot))),
    }
    return heights, _deploy_pose(robot, cx, cy, heights["lift_z"], q)


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


@dataclass
class PhaseResult:
    """Preflight result for one task phase: name, ``ok``/``fail`` status and failure reason."""
    name: str
    status: str
    reason: str | None = None


@dataclass
class SegmentPlan:
    """One arm motion. ``detour`` is true when RRT replaced the straight line."""

    phase: str
    waypoints: np.ndarray
    detour: bool
    reason: str | None = None


@dataclass
class PreflightReport:
    """Overall preflight verdict, per-phase results and the planned arm segments."""
    ok: bool
    phases: list[PhaseResult]
    segments: dict[str, SegmentPlan] = field(default_factory=dict)

    def as_dict(self, mode: str) -> dict:
        """JSON-ready summary of the verdict and phases for report ``mode``."""
        return {
            "mode": mode,
            "ok": self.ok,
            "phases": [
                {"name": phase.name, "status": phase.status, "reason": phase.reason}
                for phase in self.phases
            ],
        }


def _within_limits(q: np.ndarray, checker: CollisionChecker) -> bool:
    return bool(np.all(q >= checker.q_lo - 1e-9) and np.all(q <= checker.q_hi + 1e-9))


def _segment_blocked(checker: CollisionChecker, q0: np.ndarray, q1: np.ndarray) -> bool:
    dist = float(np.linalg.norm(q1 - q0))
    n = max(int(np.ceil(dist / _CHECK_RES)), 1)
    for t in np.linspace(0.0, 1.0, n + 1):
        if checker.in_collision(q0 + t * (q1 - q0)):
            return True
    return False


def plan_segment(q0: np.ndarray, q1: np.ndarray, checker: CollisionChecker, *,
                 seed: int = 0) -> tuple[np.ndarray, str | None]:
    """Return arm waypoints and a failure reason."""
    q0 = np.asarray(q0, dtype=float).reshape(-1)
    q1 = np.asarray(q1, dtype=float).reshape(-1)
    ends = np.vstack((q0, q1))
    if not _within_limits(q0, checker) or not _within_limits(q1, checker):
        return ends, "joint_limit"
    if checker.in_collision(q0) or checker.in_collision(q1):
        return ends, "collision"
    if not _segment_blocked(checker, q0, q1):
        return ends, None
    try:
        path = plan_collision_free(q0, q1, checker=checker, seed=seed)
    except PlanningError:
        return ends, "collision"
    return path, None


def _finger_pairs(names: set[str]) -> list[tuple[str, str]]:
    if all(name in names for name in _FINGER_PAIR):
        return [_FINGER_PAIR]
    return []


def _base_world_pairs(names: set[str]) -> list[tuple[str, str]]:
    return [(base_name, "world") for base_name in _BASE_BODIES if base_name in names]


def make_checker(scene) -> CollisionChecker:
    """Penetration-only checker that checks floor contacts while ignoring base-floor, cube, and finger pair."""
    probe = CollisionChecker(scene, safety_margin=0.0, query_margin=0.01)
    names = {probe.model.body(i).name for i in range(probe.model.nbody)}
    return CollisionChecker(
        scene, safety_margin=0.0, query_margin=0.01,
        ignore_pairs=[*_finger_pairs(names), *_base_world_pairs(names)],
        ignore_bodies=_IGNORED_BODIES,
    )


def make_box_pack_checker(scene) -> CollisionChecker:
    """Penetration-only checker: feed props and base-floor ignored, floor and pack box are checked."""
    probe = CollisionChecker(scene, safety_margin=0.0, query_margin=0.01)
    names = {probe.model.body(i).name for i in range(probe.model.nbody)}
    ignored = [n for n in names if n.startswith(_BOX_PACK_PREFIXES)]
    return CollisionChecker(
        scene, safety_margin=0.0, query_margin=0.01,
        ignore_pairs=[*_finger_pairs(names), *_base_world_pairs(names)],
        ignore_bodies=tuple(ignored),
    )


def _ik_full(robot, pos: np.ndarray, q_init: np.ndarray) -> np.ndarray | None:
    q, ok = robot.solve_tcp(pos, GRIP_DOWN_QUAT, q_init=q_init)
    if not ok:
        return None
    return np.asarray(q, dtype=float)


def _phase(name: str, reasons: list[str | None]) -> PhaseResult:
    for reason in reasons:
        if reason is not None:
            return PhaseResult(name, "fail", reason)
    return PhaseResult(name, "ok", None)


def _connect(phase: str, pairs: list[tuple[str, np.ndarray, np.ndarray]],
             checker: CollisionChecker, segments: dict[str, SegmentPlan]) -> list[str | None]:
    reasons = []
    for name, q0, q1 in pairs:
        waypoints, reason = plan_segment(q0, q1, checker)
        segments[name] = SegmentPlan(phase, waypoints, waypoints.shape[0] > 2, reason)
        reasons.append(reason)
    return reasons


def preflight_pick_place(robot, config) -> PreflightReport:
    """Plan one pick-and-place cycle and check it without stepping physics."""
    task = config.pick_place
    checker = make_checker(config.scene_path)
    calib = getattr(robot, "calibration", None)
    if calib is not None:
        apply_calibration_to_model(checker.model, calib, checker.data)
    heights, q_home_full = tcp_task_heights(robot, task)
    q_home = np.asarray(q_home_full[:robot.n_arm], dtype=float)
    q_now = np.asarray(robot.get_joint_positions(), dtype=float)

    def pose(xy, z, seed_full):
        solved = _ik_full(robot, np.array([xy[0], xy[1], z], dtype=float), seed_full)
        if solved is None:
            return None
        return solved

    cube_xy = np.asarray(task.cube_xy, dtype=float)
    target_xy = np.asarray(task.target_xy, dtype=float)
    segments: dict[str, SegmentPlan] = {}
    phases: list[PhaseResult] = []

    pre = pose(cube_xy, heights["pre_grasp_z"], q_home_full)
    grasp = pose(cube_xy, heights["grasp_z"], q_home_full if pre is None else pre)
    lift = pose(cube_xy, heights["lift_z"], q_home_full)
    above = pose(target_xy, heights["lift_z"], q_home_full)
    place = pose(target_xy, heights["grasp_z"], q_home_full if above is None else above)

    approach_reasons: list[str | None] = []
    if pre is None or grasp is None:
        approach_reasons.append("ik")
    else:
        approach_reasons.extend(_connect("接近", [
            ("approach_deploy", q_now, q_home),
            ("approach_pre", q_home, pre[:robot.n_arm]),
            ("approach_grasp", pre[:robot.n_arm], grasp[:robot.n_arm]),
        ], checker, segments))
    phases.append(_phase("接近", approach_reasons))

    spec = robot._gripper_spec
    if spec is None or spec.kind != "jaw":
        phases.append(PhaseResult("闭合", "fail", "ik"))
    elif not spec.closed_gap_m - 1e-9 <= task.grasp_gap_m <= spec.open_gap_m + 1e-9:
        phases.append(PhaseResult("闭合", "fail", "joint_limit"))
    else:
        phases.append(PhaseResult("闭合", "ok", None))

    def motion(name: str, q0, q1) -> None:
        if q0 is None or q1 is None:
            phases.append(PhaseResult(name, "fail", "ik"))
            return
        reasons = _connect(name, [(name, q0, q1)], checker, segments)
        phases.append(_phase(name, reasons))

    motion("抬起", None if grasp is None else grasp[:robot.n_arm],
           None if lift is None else lift[:robot.n_arm])
    motion("运送", None if lift is None else lift[:robot.n_arm],
           None if above is None else above[:robot.n_arm])
    motion("放置", None if above is None else above[:robot.n_arm],
           None if place is None else place[:robot.n_arm])

    return_reasons: list[str | None] = []
    if place is None or above is None:
        return_reasons.append("ik")
    else:
        return_reasons.extend(_connect("回位", [
            ("return_up", place[:robot.n_arm], above[:robot.n_arm]),
            ("return_home", above[:robot.n_arm], q_home),
        ], checker, segments))
    phases.append(_phase("回位", return_reasons))

    ordered = []
    by_name = {phase.name: phase for phase in phases}
    for name in PHASES:
        ordered.append(by_name[name])
    return PreflightReport(all(phase.status == "ok" for phase in ordered), ordered, segments)


def _drop_in_box(task, xy: np.ndarray) -> bool:
    """Drop point must place the cube footprint inside the box interior."""
    cx, cy = task.box_center_xy
    hx, hy = task.box_inner_half_xy
    return (abs(float(xy[0]) - cx) + task.cube_half_m <= hx + 1e-9
            and abs(float(xy[1]) - cy) + task.cube_half_m <= hy + 1e-9)


def preflight_box_pack(robot, config, cycles: int = 1) -> PreflightReport:
    """Plan ``cycles`` box-pack cycles and check them without stepping physics."""
    task = config.box_pack
    n_blocks = len(task.feed_xy)
    if not 1 <= cycles <= n_blocks:
        raise ValueError(f"cycles must be within 1..{n_blocks}, got {cycles}")
    checker = make_box_pack_checker(config.scene_path)
    calib = getattr(robot, "calibration", None)
    if calib is not None:
        apply_calibration_to_model(checker.model, calib, checker.data)
    try:
        heights, q_home_full = box_task_heights(robot, task)
    except RuntimeError:
        heights, q_home_full = None, None
    n_arm = robot.n_arm

    def pose(xy, z, seed_full):
        if heights is None:
            return None
        return _ik_full(robot, np.array([xy[0], xy[1], z], dtype=float), seed_full)

    segments: dict[str, SegmentPlan] = {}
    phases: list[PhaseResult] = []
    spec = robot._gripper_spec
    q_prev = np.asarray(robot.get_joint_positions(), dtype=float)

    for i in range(cycles):
        tag = f"#{i + 1}"
        feed = np.asarray(task.feed_xy[i], dtype=float)
        drop = np.asarray(task.drop_xy[i], dtype=float)
        if heights is None:
            home = pre = grasp = above = release = None
        else:
            home = pose(feed, heights["lift_z"], q_home_full)
            pre = pose(feed, heights["pre_grasp_z"], q_home_full if home is None else home)
            grasp = pose(feed, heights["grasp_z"], q_home_full if pre is None else pre)
            above = pose(drop, heights["lift_z"], q_home_full)
            release = pose(drop, heights["release_z"],
                           q_home_full if above is None else above)

        approach_reasons: list[str | None] = []
        if home is None or pre is None or grasp is None:
            approach_reasons.append("ik")
        else:
            approach_reasons.extend(_connect("接近", [
                (f"approach_deploy{tag}", q_prev, home[:n_arm]),
                (f"approach_pre{tag}", home[:n_arm], pre[:n_arm]),
                (f"approach_grasp{tag}", pre[:n_arm], grasp[:n_arm]),
            ], checker, segments))
        phases.append(_phase(f"接近{tag}", approach_reasons))

        if spec is None or spec.kind != "jaw":
            phases.append(PhaseResult(f"抓取{tag}", "fail", "ik"))
        elif not spec.closed_gap_m - 1e-9 <= task.grasp_gap_m <= spec.open_gap_m + 1e-9:
            phases.append(PhaseResult(f"抓取{tag}", "fail", "joint_limit"))
        else:
            phases.append(PhaseResult(f"抓取{tag}", "ok", None))

        def motion(name: str, q0, q1) -> None:
            if q0 is None or q1 is None:
                phases.append(PhaseResult(f"{name}{tag}", "fail", "ik"))
                return
            reasons = _connect(name, [(f"{name}{tag}", q0, q1)], checker, segments)
            phases.append(_phase(f"{name}{tag}", reasons))

        motion("抬起", None if grasp is None else grasp[:n_arm],
               None if home is None else home[:n_arm])
        motion("运送", None if home is None else home[:n_arm],
               None if above is None else above[:n_arm])

        release_reasons: list[str | None] = []
        if above is None or release is None:
            release_reasons.append("ik")
        else:
            release_reasons.extend(_connect("释放", [
                (f"释放{tag}", above[:n_arm], release[:n_arm]),
            ], checker, segments))
        if not _drop_in_box(task, drop):
            release_reasons.append("out_of_box")
        phases.append(_phase(f"释放{tag}", release_reasons))

        return_reasons: list[str | None] = []
        if release is None or above is None or home is None:
            return_reasons.append("ik")
        else:
            return_reasons.extend(_connect("回位", [
                (f"return_up{tag}", release[:n_arm], above[:n_arm]),
                (f"return_home{tag}", above[:n_arm], home[:n_arm]),
            ], checker, segments))
        phases.append(_phase(f"回位{tag}", return_reasons))

        if home is not None:
            q_prev = home[:n_arm]

    return PreflightReport(all(phase.status == "ok" for phase in phases), phases, segments)
