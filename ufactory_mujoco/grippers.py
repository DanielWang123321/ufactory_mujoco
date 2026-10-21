"""Jaw and vacuum gripper specs recognized from MJCF body names.

Commanded jaw gap is a linear map onto the existing 0–1 gripper position.
It is not a measured pad-to-pad distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

_CONTACT_DIST_M = 1e-4


class GraspError(RuntimeError):
    """The gripper is not holding the object, so the arm must not lift."""


@dataclass(frozen=True)
class GripperSpec:
    """One mounted end effector. ``kind`` is ``jaw`` or ``vacuum``."""

    kind: str
    name: str
    left_body: str = ""
    right_body: str = ""
    cup_body: str = ""
    open_gap_m: float = 0.0
    closed_gap_m: float = 0.0


_JAWS = (
    GripperSpec("jaw", "gripper_g2", "left_finger", "right_finger",
                open_gap_m=0.084, closed_gap_m=0.0),
    GripperSpec("jaw", "bio_gripper_g2",
                "bio_gripper_g2_left_finger", "bio_gripper_g2_right_finger",
                open_gap_m=0.079, closed_gap_m=0.0),
    GripperSpec("jaw", "lite6_gripper", "uflite_finger1", "uflite_finger2",
                open_gap_m=0.0178, closed_gap_m=0.0),
)
_VACUUM = GripperSpec("vacuum", "lite6_vacuum_gripper", cup_body="uflite_vacuum_gripper_link")


def _body_names(model: mujoco.MjModel) -> set[str]:
    return {model.body(i).name for i in range(model.nbody)}


def identify_gripper(model: mujoco.MjModel) -> GripperSpec | None:
    """Return the mounted gripper, or None for a bare arm."""
    names = _body_names(model)
    for spec in _JAWS:
        if spec.left_body in names and spec.right_body in names:
            return spec
    if _VACUUM.cup_body in names:
        return _VACUUM
    return None


def gap_to_position(spec: GripperSpec, gap_m: float) -> float:
    """Map a commanded gap onto 0 (open) .. 1 (closed)."""
    if spec.kind != "jaw":
        raise NotImplementedError("真空吸盘没有开合间隙")
    gap = float(gap_m)
    span = spec.open_gap_m - spec.closed_gap_m
    if (not np.isfinite(gap) or span <= 0.0
            or gap < spec.closed_gap_m or gap > spec.open_gap_m):
        raise ValueError(
            f"gap must be within [{spec.closed_gap_m}, {spec.open_gap_m}] m, got {gap_m}"
        )
    return (spec.open_gap_m - gap) / span


def position_to_gap(spec: GripperSpec, position: float) -> float:
    """Inverse of ``gap_to_position`` for the current command."""
    if spec.kind != "jaw":
        raise NotImplementedError("真空吸盘没有开合间隙")
    pos = float(np.clip(position, 0.0, 1.0))
    return spec.open_gap_m - pos * (spec.open_gap_m - spec.closed_gap_m)


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"body {name!r} not found")
    return body_id


def bodies_in_contact(model: mujoco.MjModel, data: mujoco.MjData,
                      body_a: str, body_b: str, *, max_dist: float = _CONTACT_DIST_M) -> bool:
    """True when a contact pair joins the two bodies and is not separated."""
    id_a = _body_id(model, body_a)
    id_b = _body_id(model, body_b)
    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.dist > max_dist:
            continue
        pair = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
        if pair == {id_a, id_b}:
            return True
    return False


def bilateral_finger_contact(model: mujoco.MjModel, data: mujoco.MjData,
                             spec: GripperSpec, object_body: str) -> bool:
    """Both finger bodies touch ``object_body``."""
    if spec.kind != "jaw":
        raise NotImplementedError("真空吸盘没有双侧指垫")
    return (bodies_in_contact(model, data, spec.left_body, object_body)
            and bodies_in_contact(model, data, spec.right_body, object_body))


def require_bilateral_contact(model: mujoco.MjModel, data: mujoco.MjData,
                              spec: GripperSpec, object_body: str) -> None:
    if not bilateral_finger_contact(model, data, spec, object_body):
        raise GraspError(f"无双侧指垫接触，拒绝抬起: {object_body}")
