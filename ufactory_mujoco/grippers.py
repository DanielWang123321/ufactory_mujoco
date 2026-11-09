"""Jaw and vacuum gripper specs recognized from MJCF body names.

Includes calibrated 4-bar linkage geometry conversions and contact-force helpers
for Gripper G2.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .g2_model import (
    G2_LINKAGE_GEOMETRY,
    GRIPPER_G2_SIM_CLOSE_DRIVE,
    GRIPPER_G2_SIM_OPEN_DRIVE,
    G2LinkageGeometry,
    gripper_g2_sim_drive_to_gap_m,
)

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
    drive_range_rad: tuple[float, float] = (0.0, 0.0)


_JAWS = (
    GripperSpec(
        "jaw",
        "gripper_g2",
        "left_finger",
        "right_finger",
        open_gap_m=0.084,
        closed_gap_m=0.0,
        drive_range_rad=(GRIPPER_G2_SIM_OPEN_DRIVE, GRIPPER_G2_SIM_CLOSE_DRIVE),
    ),
    GripperSpec(
        "jaw",
        "bio_gripper_g2",
        "bio_gripper_g2_left_finger",
        "bio_gripper_g2_right_finger",
        open_gap_m=0.079,
        closed_gap_m=0.0,
        drive_range_rad=(0.0, 0.0395),
    ),
    GripperSpec(
        "jaw",
        "lite6_gripper",
        "uflite_finger1",
        "uflite_finger2",
        open_gap_m=0.0178,
        closed_gap_m=0.0,
        drive_range_rad=(0.0, 0.0089),
    ),
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
    if (
        not np.isfinite(gap)
        or span <= 0.0
        or gap < spec.closed_gap_m
        or gap > spec.open_gap_m
    ):
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


def drive_to_gap(
    spec: GripperSpec,
    drive_qpos: float,
    *,
    open_ctrl: float | None = None,
    close_ctrl: float | None = None,
) -> float:
    """Map a drive joint/actuator coordinate back to the physical jaw gap (m)."""
    if spec.kind != "jaw":
        raise NotImplementedError("真空吸盘没有开合间隙")
    if spec.name == "gripper_g2":
        return gripper_g2_sim_drive_to_gap_m(float(drive_qpos))
    lo = spec.drive_range_rad[0] if open_ctrl is None else float(open_ctrl)
    hi = spec.drive_range_rad[1] if close_ctrl is None else float(close_ctrl)
    span = hi - lo
    if abs(span) < 1e-12:
        return spec.open_gap_m
    norm_pos = float(np.clip((float(drive_qpos) - lo) / span, 0.0, 1.0))
    return position_to_gap(spec, norm_pos)


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"body {name!r} not found")
    return body_id


def g2_pad_centers(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geometry: G2LinkageGeometry = G2_LINKAGE_GEOMETRY,
    *,
    left_body: str = "left_finger",
    right_body: str = "right_finger",
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame inner pad centers of Gripper G2."""
    left_id = _body_id(model, left_body)
    right_id = _body_id(model, right_body)
    l_pos = np.asarray(data.body(left_id).xpos, dtype=float)
    l_mat = np.asarray(data.body(left_id).xmat, dtype=float).reshape(3, 3)
    r_pos = np.asarray(data.body(right_id).xpos, dtype=float)
    r_mat = np.asarray(data.body(right_id).xmat, dtype=float).reshape(3, 3)
    l_local = np.array([0.0, geometry.pad_inner_y_m, geometry.pad_center_z_m], dtype=float)
    r_local = np.array([0.0, -geometry.pad_inner_y_m, geometry.pad_center_z_m], dtype=float)
    return l_pos + l_mat @ l_local, r_pos + r_mat @ r_local


def measured_g2_pad_gap_mm(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geometry: G2LinkageGeometry = G2_LINKAGE_GEOMETRY,
    *,
    left_body: str = "left_finger",
    right_body: str = "right_finger",
) -> float:
    """Actual pad separation (mm) along the Gripper G2 closing normal in world frame."""
    left_center, right_center = g2_pad_centers(
        model, data, geometry, left_body=left_body, right_body=right_body
    )
    left_id = _body_id(model, left_body)
    l_mat = np.asarray(data.body(left_id).xmat, dtype=float).reshape(3, 3)
    normal = l_mat @ np.array([0.0, 1.0, 0.0], dtype=float)
    return float(abs(np.dot(right_center - left_center, normal)) * 1000.0)


def gripper_object_contact_forces_n(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec: GripperSpec,
    object_body: str,
) -> tuple[float, float]:
    """Return net contact force magnitudes (N) carried by (left_finger, right_finger) on ``object_body``."""
    if spec.kind != "jaw":
        raise NotImplementedError("真空吸盘没有双侧指垫")
    left_id = _body_id(model, spec.left_body)
    right_id = _body_id(model, spec.right_body)
    obj_id = _body_id(model, object_body)

    left_net = np.zeros(3, dtype=float)
    right_net = np.zeros(3, dtype=float)
    wrench = np.zeros(6, dtype=float)

    for i in range(data.ncon):
        c = data.contact[i]
        b1 = int(model.geom_bodyid[c.geom[0]])
        b2 = int(model.geom_bodyid[c.geom[1]])
        if obj_id not in (b1, b2):
            continue
        other = b2 if b1 == obj_id else b1
        if other not in (left_id, right_id):
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        frame = np.asarray(c.frame, dtype=float).reshape(3, 3)
        # wrench[:3] is force from geom1 onto geom2 in contact frame (rows of c.frame are x, y, z axes)
        force_world = frame.T @ wrench[:3]
        applied_to_finger = force_world if other == b2 else -force_world
        if other == left_id:
            left_net += applied_to_finger
        else:
            right_net += applied_to_finger

    return float(np.linalg.norm(left_net)), float(np.linalg.norm(right_net))


def bodies_in_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_a: str,
    body_b: str,
    *,
    max_dist: float = _CONTACT_DIST_M,
) -> bool:
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


def bilateral_finger_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec: GripperSpec,
    object_body: str,
) -> bool:
    """Both finger bodies touch ``object_body``."""
    if spec.kind != "jaw":
        raise NotImplementedError("真空吸盘没有双侧指垫")
    return bodies_in_contact(model, data, spec.left_body, object_body) and bodies_in_contact(
        model, data, spec.right_body, object_body
    )


def require_bilateral_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec: GripperSpec,
    object_body: str,
) -> None:
    if not bilateral_finger_contact(model, data, spec, object_body):
        raise GraspError(f"无双侧指垫接触，拒绝抬起: {object_body}")
