"""Collision queries and RRT-Connect collision-free planning on arm joint space."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .grippers import GRIPPER_ACTUATOR_NAMES


def arm_joint_limits(model: mujoco.MjModel) -> tuple[int, np.ndarray, np.ndarray]:
    """(n_arm_qpos, q_lo, q_hi) for the arm joints, mirroring SimRobot boundaries."""
    n_arm_actuators = model.nu
    for i in range(model.nu):
        if model.actuator(i).name in GRIPPER_ACTUATOR_NAMES:
            n_arm_actuators = i
            break
    joints: list[int] = []
    for i in range(n_arm_actuators):
        if int(model.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_JOINT):
            joints.append(int(model.actuator_trnid[i, 0]))
    if not joints:
        raise ValueError("model has no arm actuators driving joints")
    joints.sort(key=lambda j: int(model.jnt_qposadr[j]))
    qpos_adr = [int(model.jnt_qposadr[j]) for j in joints]
    if max(qpos_adr) + 1 != len(qpos_adr):
        raise ValueError("arm joint qpos entries are not the leading qpos block")
    lo = np.array([model.jnt_range[j, 0] for j in joints])
    hi = np.array([model.jnt_range[j, 1] for j in joints])
    return len(qpos_adr), lo, hi


class CollisionChecker:
    """Arm-joint collision queries against a private copy of a scene model."""

    def __init__(self, scene: str | Path | mujoco.MjSpec, *, safety_margin: float = 0.0,
                 query_margin: float = 0.05, ignore_pairs=(), ignore_bodies=()):
        if not np.isfinite(safety_margin) or safety_margin < 0:
            raise ValueError("safety_margin must be finite and non-negative")
        if not np.isfinite(query_margin) or query_margin < safety_margin or query_margin <= 0:
            raise ValueError("query_margin must be finite, positive and >= safety_margin")
        if isinstance(scene, mujoco.MjSpec):
            self.model = scene.compile()
        elif isinstance(scene, str) and scene.lstrip().startswith("<"):
            self.model = mujoco.MjModel.from_xml_string(scene)
        elif isinstance(scene, (str, Path)):
            self.model = mujoco.MjModel.from_xml_path(str(scene))
        else:
            raise TypeError("scene must be an MJCF path, XML text or mujoco.MjSpec")
        self.data = mujoco.MjData(self.model)
        self.safety_margin = float(safety_margin)
        self.query_margin = float(query_margin)

        self.n_arm_qpos, self.q_lo, self.q_hi = arm_joint_limits(self.model)

        # inflate collision geoms (and the floor) so contacts appear before penetration
        active = (self.model.geom_contype.astype(bool)
                  | self.model.geom_conaffinity.astype(bool))
        self.model.geom_margin[active] = query_margin

        self._ignore: set[frozenset[int]] = set()
        for pair in ignore_pairs:
            b1, b2 = pair
            id1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b1)
            id2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b2)
            if id1 < 0 or id2 < 0:
                raise ValueError(f"unknown body in ignore_pairs: {pair}")
            self._ignore.add(frozenset((id1, id2)))

        self._ignore_bodies: set[int] = set()
        for name in ignore_bodies:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                self._ignore_bodies.add(body_id)

    def _contacts(self, q: np.ndarray) -> list[tuple[int, int, float]]:
        """Filtered contacts at state q: (body1, body2, dist) tuples."""
        self._set_q(q)
        out = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[c.geom[0]])
            b2 = int(self.model.geom_bodyid[c.geom[1]])
            if b1 in self._ignore_bodies or b2 in self._ignore_bodies:
                continue
            if frozenset((b1, b2)) in self._ignore:
                continue
            out.append((b1, b2, float(c.dist)))
        return out

    def _set_q(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape != (self.n_arm_qpos,):
            raise ValueError(f"q must have shape ({self.n_arm_qpos},), got {q.shape}")
        if not np.all(np.isfinite(q)):
            raise ValueError("q must be finite")
        self.data.qpos[:self.n_arm_qpos] = q
        mujoco.mj_forward(self.model, self.data)

    def in_collision(self, q: np.ndarray, *, safety_margin: float | None = None) -> bool:
        """True when any non-ignored contact is closer than the safety margin."""
        margin = self.safety_margin if safety_margin is None else float(safety_margin)
        return any(dist < margin for _, _, dist in self._contacts(q))

    def min_distance(self, q: np.ndarray) -> float:
        """Closest contact distance (m); inf when nothing within query_margin."""
        contacts = self._contacts(q)
        return min((dist for _, _, dist in contacts), default=float("inf"))


class PlanningError(RuntimeError):
    """No collision-free path found (or the endpoints are in collision)."""


def _edge_free(checker: CollisionChecker, qa: np.ndarray, qb: np.ndarray,
               check_res: float, safety_margin: float | None) -> bool:
    """True when sampled states along the segment qa->qb (endpoints included) are collision-free."""
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
    """RRT-Connect waypoints (K, n_arm) from q_start to q_goal."""
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
    nodes_a, parents_a = [q_start], [-1]
    nodes_b, parents_b = [q_goal], [-1]

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
                path_a = _trace(nodes_a, parents_a, ia)
                path_b = _trace(nodes_b, parents_b, ib)
                return np.array(path_a + path_b[::-1][1:])
        forward = not forward

    raise PlanningError(f"no collision-free path found within {max_iter} iterations")


def shortcut_path(checker: CollisionChecker, waypoints, *, iterations: int = 200,
                  check_res: float = 0.05, seed: int | None = None,
                  safety_margin: float | None = None) -> np.ndarray:
    """Random shortcutting: drop intermediate waypoints while segments stay free."""
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
