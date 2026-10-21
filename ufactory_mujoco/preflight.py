"""Pick-and-place preflight: joint limits, penetration checks, and detours.

Waypoint IK uses the robot's evaluation data and does not step physics.
Expected table, cube, stand, and Lite6 finger-finger contacts are ignored.
A straight arm segment that still hits another body is replaced by RRT-Connect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .collision import CollisionChecker
from .path_planning import PlanningError, plan_collision_free
from .task_geometry import tcp_task_heights, box_task_heights

PHASES = ("接近", "闭合", "抬起", "运送", "放置", "回位")
PACK_PHASES = ("接近", "抓取", "抬起", "运送", "释放", "回位")
_IGNORED_BODIES = ("world", "cube", "pick_stand", "place_stand")
_BOX_PACK_PREFIXES = ("cube", "feed_stand", "pick_stand", "place_stand")
_FINGER_PAIR = ("uflite_finger1", "uflite_finger2")
_CHECK_RES = 0.08


@dataclass
class PhaseResult:
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
    ok: bool
    phases: list[PhaseResult]
    segments: dict[str, SegmentPlan] = field(default_factory=dict)

    def as_dict(self, mode: str) -> dict:
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
    """Return arm waypoints and a failure reason.

    A free straight line is the two endpoints. A blocked segment is replaced
    by RRT-Connect. Limit violations and a penetrating endpoint fail closed.
    """
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
    # Finger-finger is a pair, not a body to drop entirely.
    if all(name in names for name in _FINGER_PAIR):
        return [_FINGER_PAIR]
    return []


def make_checker(scene) -> CollisionChecker:
    """Penetration-only checker that ignores the table, cube, and finger pair."""
    probe = CollisionChecker(scene, safety_margin=0.0, query_margin=0.01)
    names = {probe.model.body(i).name for i in range(probe.model.nbody)}
    return CollisionChecker(
        scene, safety_margin=0.0, query_margin=0.01,
        ignore_pairs=_finger_pairs(names), ignore_bodies=_IGNORED_BODIES,
    )


def make_box_pack_checker(scene) -> CollisionChecker:
    """Penetration-only checker: feed props ignored, the pack box is not."""
    probe = CollisionChecker(scene, safety_margin=0.0, query_margin=0.01)
    names = {probe.model.body(i).name for i in range(probe.model.nbody)}
    ignored = ["world"]
    ignored.extend(n for n in names if n.startswith(_BOX_PACK_PREFIXES))
    return CollisionChecker(
        scene, safety_margin=0.0, query_margin=0.01,
        ignore_pairs=_finger_pairs(names), ignore_bodies=tuple(ignored),
    )


def _grip_down() -> np.ndarray:
    return np.array([0.0, 1.0, 0.0, 0.0])


def _ik_full(robot, pos: np.ndarray, q_init: np.ndarray) -> np.ndarray | None:
    q, ok = robot.solve_tcp(pos, _grip_down(), q_init=q_init)
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
        raise ValueError(f"cycles 必须在 1..{n_blocks} 之间, 得到 {cycles}")
    checker = make_box_pack_checker(config.scene_path)
    # An unreachable probe point fails every phase instead of raising.
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
