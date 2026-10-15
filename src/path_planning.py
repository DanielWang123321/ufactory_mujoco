"""Sampling-based collision-free path planning on arm joint space.

``plan_collision_free`` runs RRT-Connect between two arm configurations and
returns collision-free waypoints (start and goal inclusive). Time
parameterization is delegated to :mod:`src.trajectory` (e.g.
``plan_joint_waypoints``) and execution to ``SimRobot.execute_trajectory``.

Determinism: pass a ``seed`` for reproducible search trees and shortcuts.
"""

from __future__ import annotations

import numpy as np

from .collision import CollisionChecker


class PlanningError(RuntimeError):
    """No collision-free path found (or the endpoints are in collision)."""


def path_length(waypoints: np.ndarray) -> float:
    """Sum of Euclidean segment lengths between consecutive waypoints."""
    w = np.asarray(waypoints, dtype=float)
    return float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum())


def _edge_free(checker: CollisionChecker, qa: np.ndarray, qb: np.ndarray,
               check_res: float, safety_margin: float | None) -> bool:
    """True when sampled states along the segment qa->qb (endpoints included)
    are collision-free at the given resolution."""
    dist = float(np.linalg.norm(qb - qa))
    n = max(int(np.ceil(dist / check_res)), 1)
    for t in np.linspace(0.0, 1.0, n + 1):
        if checker.in_collision(qa + t * (qb - qa), safety_margin=safety_margin):
            return False
    return True


def _trace(nodes: list[np.ndarray], parents: list[int], idx: int) -> list[np.ndarray]:
    """Waypoints from tree root to node ``idx``."""
    chain = []
    while idx != -1:
        chain.append(nodes[idx])
        idx = parents[idx]
    return chain[::-1]


def _validate_q(q: np.ndarray, checker: CollisionChecker, name: str) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.shape != (checker.n_arm_qpos,):
        raise ValueError(f"{name} must have shape ({checker.n_arm_qpos},), got {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError(f"{name} must be finite")
    if np.any(q < checker.q_lo - 1e-9) or np.any(q > checker.q_hi + 1e-9):
        raise ValueError(f"{name} violates joint limits")
    return q


def plan_collision_free(q_start, q_goal, *, checker: CollisionChecker,
                        max_iter: int = 2000, step: float = 0.1,
                        check_res: float | None = None, goal_bias: float = 0.1,
                        seed: int | None = None,
                        safety_margin: float | None = None) -> np.ndarray:
    """RRT-Connect waypoints (K, n_arm) from q_start to q_goal.

    A straight-line fast path is returned when the direct segment is free.
    New edges are collision-checked at a resolution of at most ``check_res``
    (default ``step / 2``). Raises PlanningError when the endpoints collide or
    no path is found within ``max_iter`` tree extensions.
    """
    if step <= 0 or not np.isfinite(step):
        raise ValueError("step must be finite and positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if not 0.0 <= goal_bias <= 1.0:
        raise ValueError("goal_bias must be in [0, 1]")
    check_res = step / 2.0 if check_res is None else float(check_res)
    if check_res <= 0 or not np.isfinite(check_res):
        raise ValueError("check_res must be finite and positive")

    q_start = _validate_q(q_start, checker, "q_start")
    q_goal = _validate_q(q_goal, checker, "q_goal")
    for name, q in (("start", q_start), ("goal", q_goal)):
        if checker.in_collision(q, safety_margin=safety_margin):
            raise PlanningError(f"{name} state is in collision")

    if _edge_free(checker, q_start, q_goal, check_res, safety_margin):
        return np.vstack([q_start, q_goal])

    rng = np.random.default_rng(seed)
    nodes_a, parents_a = [q_start], [-1]   # grows from start
    nodes_b, parents_b = [q_goal], [-1]    # grows from goal

    def _extend(nodes, parents, q_target) -> int | None:
        i_near = int(np.argmin([np.linalg.norm(q - q_target) for q in nodes]))
        q_near = nodes[i_near]
        delta = q_target - q_near
        dist = float(np.linalg.norm(delta))
        q_new = q_target.copy() if dist <= step else q_near + step * delta / dist
        q_new = np.clip(q_new, checker.q_lo, checker.q_hi)
        if np.allclose(q_new, q_near):
            return None
        if not _edge_free(checker, q_near, q_new, check_res, safety_margin):
            return None
        nodes.append(q_new)
        parents.append(i_near)
        return len(nodes) - 1

    def _connect(nodes, parents, q_target) -> int | None:
        """Extend toward q_target until reached or blocked."""
        while True:
            idx = _extend(nodes, parents, q_target)
            if idx is None:
                return None
            if np.linalg.norm(nodes[idx] - q_target) <= 1e-12:
                return idx

    forward = True
    for _ in range(max_iter):
        if forward:
            grow, grow_p, other, other_p = nodes_a, parents_a, nodes_b, parents_b
            q_bias = q_goal
        else:
            grow, grow_p, other, other_p = nodes_b, parents_b, nodes_a, parents_a
            q_bias = q_start
        q_rand = q_bias if rng.random() < goal_bias \
            else rng.uniform(checker.q_lo, checker.q_hi)

        idx_new = _extend(grow, grow_p, q_rand)
        if idx_new is not None:
            idx_link = _connect(other, other_p, grow[idx_new])
            if idx_link is not None:
                ia = idx_new if forward else idx_link
                ib = idx_link if forward else idx_new
                path_a = _trace(nodes_a, parents_a, ia)   # start -> connect
                path_b = _trace(nodes_b, parents_b, ib)   # goal  -> connect
                return np.array(path_a + path_b[::-1][1:])
        forward = not forward

    raise PlanningError(f"no collision-free path found within {max_iter} iterations")


def shortcut_path(checker: CollisionChecker, waypoints, *, iterations: int = 200,
                  check_res: float = 0.05, seed: int | None = None,
                  safety_margin: float | None = None) -> np.ndarray:
    """Random shortcutting: drop intermediate waypoints while segments stay free.

    Never increases the number of waypoints or the path length; the returned
    array keeps the original first and last waypoints exactly.
    """
    path = [_validate_q(w, checker, "waypoint") for w in np.asarray(waypoints, dtype=float)]
    if len(path) < 2:
        raise ValueError("need at least two waypoints")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        if len(path) <= 2:
            break
        i, j = sorted(int(k) for k in rng.choice(len(path), 2, replace=False))
        if j - i < 2:
            continue
        if _edge_free(checker, path[i], path[j], check_res, safety_margin):
            path = path[:i + 1] + path[j:]
    return np.array(path)
