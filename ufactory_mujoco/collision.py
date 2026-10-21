"""Collision queries for motion planning on UFACTORY MJCF models.

Collision semantics follow the model's own contact setup (contact excludes and
geom contype/conaffinity filters apply). With zero margins MuJoCo generates
contacts on actual penetration only; the checker therefore inflates geom
margins in a **private model copy** so that near-miss distances become visible
without touching the simulation's physics.

The query state ``q`` covers the arm joints in the SimRobot convention: the
first ``n_arm_qpos`` qpos entries (gripper joints excluded).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .sim_robot import _GRIPPER_ACTUATOR_NAMES  # shared gripper-actuator convention


def arm_joint_limits(model: mujoco.MjModel) -> tuple[int, np.ndarray, np.ndarray]:
    """(n_arm_qpos, q_lo, q_hi) for the arm joints, mirroring SimRobot boundaries."""
    n_arm_actuators = model.nu
    for i in range(model.nu):
        if model.actuator(i).name in _GRIPPER_ACTUATOR_NAMES:
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
    """Arm-joint collision queries against a private copy of a scene model.

    Parameters
    ----------
    scene:
        Path to an MJCF scene/model file, MJCF XML text, or a ``mujoco.MjSpec``
        (compiled here, so the caller's model is never mutated).
    safety_margin:
        Default clearance threshold: a state counts as colliding when any
        contact distance is below this value (0.0 = penetration only).
    query_margin:
        Contact detection radius: geom margins are inflated by this amount in
        the private copy so contacts (and true surface distances) appear
        before penetration. Must be >= safety_margin.
    ignore_pairs:
        Iterable of body-name pairs whose contacts are ignored on top of the
        model's own ``<contact><exclude>`` list (e.g. a base resting on the
        floor).
    """

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

    # ------------------------------------------------------------------ queries
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
        """Closest contact distance (m); inf when nothing within query_margin.

        Negative values indicate penetration. Adjacent links listed in the
        model's ``<contact><exclude>`` pairs are not reported.
        """
        contacts = self._contacts(q)
        return min((dist for _, _, dist in contacts), default=float("inf"))
