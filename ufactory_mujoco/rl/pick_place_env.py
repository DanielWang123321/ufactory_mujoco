"""Gymnasium environment, strict metrics, and independent scripted baseline for
xArm6 + Gripper G2 tabletop cube pick-and-place.

Reuses project assets and calibrated Gripper G2 interfaces:
- Scene: ``models/xarm6_with_gripper/scene_pick_place.xml`` via ``configs/pick_place/xarm6.yaml``
- Calibrated Gripper G2 linkage & servo: ``ufactory_mujoco.g2_model`` / ``g2_controller``
  (``ScalarG2Controller``, bit-identical to ``G2Controller`` for one environment)
- Kinematics/IK: ``ufactory_mujoco.ik.solve_ik`` (per-step IK runs the same algorithm with
  precomputed constants in ``_GripDownIK``) and ``ufactory_mujoco.task_geometry``

Requires only ``mujoco`` + ``numpy`` at runtime; ``gymnasium`` is optional
(fallback base class for core-only installs and baseline rollouts), and ``stable-baselines3``
is required only for PPO training/checkpoint loading.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..config import load_task_config
from ..g2_controller import ScalarG2Controller
from ..g2_model import (
    G2_LINKAGE_GEOMETRY,
    G2Model,
    gripper_g2_gap_m_to_sim_drive,
)
from ..grippers import identify_gripper
from ..ik import solve_ik
from ..resource_paths import CONFIGS_ROOT
from ..task_geometry import GRIP_DOWN_QUAT
from .artifacts import (
    PICK_PLACE_ACTION_CONTRACT,
    PICK_PLACE_ENV_VERSION,
    PICK_PLACE_NUM_ACTIONS,
    PICK_PLACE_NUM_OBS,
)
from .model_hash import runtime_model_sha256, verify_runtime_model
from .reach_env import _ENV_BASE, _BoxStub, gym, load_reach_model, spaces

TCP_SITE = "attachment_site"

# Default task and physical constants aligned with ufactory_genesis pick_place
SPAWN_X = 0.300
SPAWN_Y = 0.000
SPAWN_XY_LOWER = (0.300, -0.200)
SPAWN_XY_UPPER = (0.500, 0.200)
TARGET_X = 0.300
TARGET_Y = 0.300
CUBE_HALF_M = 0.015
CUBE_SIZE = 2.0 * CUBE_HALF_M
CUBE_REST_Z = 0.015
SAFE_LIFT_Z = 0.080

TARGET_TOL_M = 0.020
DRAG_LIMIT_M = 0.010
TILT_LIMIT_DEG = 15.0
SUCCESS_HOLD_STEPS = 10
CONTACT_CONFIRM_STEPS = 3
CONTACT_FORCE_N = 0.2
RELEASE_FORCE_N = 0.05
SUPPORT_FORCE_N = 0.05
RELEASE_MAX_HEIGHT_M = 0.003
PRESS_LIMIT_M = 0.001
PLACE_CLEARANCE_M = 0.002

# Minimum TCP (attachment_site) Z setpoint before lift: keeps closed fingertips >= 2.7 mm
# above the z=0 table while placing fingertip pad centers inside the cube's grasp band.
_PRE_LIFT_MIN_TCP_Z = 0.0055

# Local coordinates of the calibrated G2 inner fingertip pad centers on left_finger / right_finger
_LEFT_PAD_LOCAL = np.array(
    [0.0, G2_LINKAGE_GEOMETRY.pad_inner_y_m, G2_LINKAGE_GEOMETRY.pad_center_z_m],
    dtype=float,
)
_RIGHT_PAD_LOCAL = np.array(
    [0.0, -G2_LINKAGE_GEOMETRY.pad_inner_y_m, G2_LINKAGE_GEOMETRY.pad_center_z_m],
    dtype=float,
)
_PAD_NORMAL_LOCAL = np.array([0.0, 1.0, 0.0], dtype=float)


def cube_clearance(pos: np.ndarray, quat: np.ndarray, cube_half_m: float = CUBE_HALF_M) -> float:
    """Lowest cube point above the z=0 tabletop, using a WXYZ quaternion."""
    pos_arr = np.asarray(pos, dtype=float).reshape(-1)
    q = np.asarray(quat, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm > 0.0:
        q = q / norm
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    r0 = abs(2.0 * (x * z - w * y))
    r1 = abs(2.0 * (y * z + w * x))
    r2 = abs(1.0 - 2.0 * (x * x + y * y))
    return float(pos_arr[2] - float(cube_half_m) * (r0 + r1 + r2))


def grasp_distance(pad_center: np.ndarray, obj_pos: np.ndarray, cube_half_m: float = CUBE_HALF_M) -> float:
    """Distance from fingertip pad midpoint to the cube's graspable vertical band."""
    rel = np.asarray(pad_center, dtype=float).reshape(3) - np.asarray(obj_pos, dtype=float).reshape(3)
    band = 0.8 * float(cube_half_m)
    z_excess = max(0.0, abs(float(rel[2])) - band)
    return float(math.sqrt(float(rel[0]) ** 2 + float(rel[1]) ** 2 + z_excess * z_excess))


class PickPlaceMetrics:
    """Track grasp, lift, touchdown pressing, release height, and settled table support."""

    def __init__(
        self,
        *,
        ctrl_dt: float = 0.02,
        target_tol: float = TARGET_TOL_M,
        drag_limit: float = DRAG_LIMIT_M,
        tilt_limit: float = TILT_LIMIT_DEG,
        hold_steps: int = SUCCESS_HOLD_STEPS,
        place_clearance: float = PLACE_CLEARANCE_M,
        release_max_height: float = RELEASE_MAX_HEIGHT_M,
        press_limit: float = PRESS_LIMIT_M,
        spawn_xy: tuple[float, float] = (SPAWN_X, SPAWN_Y),
        target_xy: tuple[float, float] = (TARGET_X, TARGET_Y),
        cube_half_m: float = CUBE_HALF_M,
    ) -> None:
        self.ctrl_dt = float(ctrl_dt)
        self.target_tol = float(target_tol)
        self.drag_limit = float(drag_limit)
        self.tilt_limit = float(tilt_limit)
        self.place_clearance = float(place_clearance)
        self.release_max_height = float(release_max_height)
        self.press_limit = float(press_limit)
        self.cube_half_m = float(cube_half_m)
        self.required_hold_steps = max(int(hold_steps), math.ceil(0.2 / self.ctrl_dt))
        self.spawn_xy = np.asarray(spawn_xy, dtype=float).reshape(2).copy()
        self.target_xy = np.asarray(target_xy, dtype=float).reshape(2).copy()
        self.finger_contact = np.zeros(2, dtype=bool)
        self.reset(spawn_xy=self.spawn_xy)

    def reset(
        self,
        spawn_xy: np.ndarray | tuple[float, float] | None = None,
        ee_z: float = 0.1266,
    ) -> None:
        """Clear all episode metrics for a new cube spawn."""
        if spawn_xy is not None:
            self.spawn_xy = np.asarray(spawn_xy, dtype=float).reshape(2).copy()
        self.finger_contact[:] = False
        self.max_drag = 0.0
        self.max_tilt_deg = 0.0
        self.max_held_z = CUBE_REST_Z
        self.target_error = float(np.linalg.norm(self.spawn_xy - self.target_xy))
        self.clearance = 0.0
        self.max_press = 0.0
        self.touchdown_ee_z = 0.0
        self.min_ee_z = 0.0
        self.release_height = 0.0
        self.release_candidate_height = 0.0
        self.max_table_force = 0.0
        self.max_place_table_force = 0.0
        self.max_held_table_force = 0.0
        self.max_robot_table_force = 0.0
        self.prev_ee_z = float(ee_z)
        self.prev_clearance = 0.0

        self.ever_carried = False
        self.ever_lifted_clear = False
        self.touchdown_seen = False
        self.release_confirmed = False
        self.high_release = False
        self.prev_any_contact = False

        self.contact_steps = 0
        self.hold_steps = 0
        self.loss_steps = 0
        self.clear_steps = 0

    def update(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        forces: np.ndarray,
        obj_vel: np.ndarray,
        gap: float,
        *,
        ee_z: float,
        table_force: float,
        robot_table_force: float,
    ) -> dict[str, Any]:
        """Update drag, tilt, press, release and contact metrics from one control step; returns the per-step flags."""
        pos = np.asarray(pos, dtype=float).reshape(3)
        quat = np.asarray(quat, dtype=float).reshape(4)
        forces = np.asarray(forces, dtype=float).reshape(2)
        obj_vel = np.asarray(obj_vel, dtype=float).reshape(3)
        gap = float(gap)
        ee_z = float(ee_z)
        table_force = float(table_force)
        robot_table_force = float(robot_table_force)

        self.clearance = cube_clearance(pos, quat, self.cube_half_m)
        height = max(0.0, self.clearance)
        q_norm = float(np.linalg.norm(quat))
        q = quat / q_norm if q_norm > 0.0 else np.array([1.0, 0.0, 0.0, 0.0])
        up_z = float(np.clip(1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]), -1.0, 1.0))
        tilt = float(np.rad2deg(math.acos(up_z)))

        on_table = self.clearance < 0.015
        before_lift = (not self.ever_carried) and on_table
        drag = float(math.hypot(float(pos[0] - self.spawn_xy[0]), float(pos[1] - self.spawn_xy[1])))
        if before_lift:
            self.max_drag = max(self.max_drag, drag)
            self.max_tilt_deg = max(self.max_tilt_deg, tilt)
        quality_ok = (self.max_drag <= self.drag_limit) and (self.max_tilt_deg <= self.tilt_limit)

        for i in range(2):
            if forces[i] > CONTACT_FORCE_N:
                self.finger_contact[i] = True
            elif forces[i] < RELEASE_FORCE_N:
                self.finger_contact[i] = False
        touching = bool(self.finger_contact[0] and self.finger_contact[1])
        any_contact = bool(forces[0] >= RELEASE_FORCE_N or forces[1] >= RELEASE_FORCE_N)

        recontact = self.release_confirmed and any_contact
        if recontact:
            self.release_confirmed = False
            self.touchdown_seen = False

        self.contact_steps = self.contact_steps + 1 if touching else 0
        stable_contact = self.contact_steps >= CONTACT_CONFIRM_STEPS
        carried = stable_contact and (height > 0.02)
        first_carry = carried and (not self.ever_carried) and quality_ok
        self.ever_carried = self.ever_carried or carried
        if stable_contact:
            self.max_held_z = max(self.max_held_z, height + CUBE_REST_Z)
        self.loss_steps = self.loss_steps + 1 if (self.ever_carried and not touching) else 0
        lifted = carried and (height >= SAFE_LIFT_Z - CUBE_REST_Z) and quality_ok
        first_lift_clear = lifted and (not self.ever_lifted_clear)
        self.ever_lifted_clear = self.ever_lifted_clear or lifted

        supported = (table_force > SUPPORT_FORCE_N) and (self.clearance <= self.place_clearance)
        first_touchdown = (
            self.ever_lifted_clear
            and supported
            and (not self.touchdown_seen)
            and (not self.release_confirmed)
            and (any_contact or self.prev_any_contact)
        )
        if first_touchdown:
            self.touchdown_ee_z = max(self.prev_ee_z, ee_z)
            self.min_ee_z = ee_z
            self.touchdown_seen = True
        tracking = self.touchdown_seen and (not self.release_confirmed) and (any_contact or self.prev_any_contact)
        if tracking:
            self.min_ee_z = min(self.min_ee_z, ee_z)
        old_press = self.max_press
        if tracking:
            self.max_press = max(self.max_press, max(0.0, self.touchdown_ee_z - self.min_ee_z))

        both_clear = bool(forces[0] < RELEASE_FORCE_N and forces[1] < RELEASE_FORCE_N)
        first_clear = self.ever_carried and both_clear and (self.clear_steps == 0) and (not self.release_confirmed)
        if first_clear:
            self.release_candidate_height = max(max(0.0, self.prev_clearance), height)
        self.clear_steps = self.clear_steps + 1 if (both_clear and self.ever_carried) else 0
        first_release = (self.clear_steps >= CONTACT_CONFIRM_STEPS) and (not self.release_confirmed)
        if first_release:
            self.release_height = max(self.release_height, self.release_candidate_height)
            if self.release_height > self.release_max_height + 1e-7:
                self.high_release = True
            self.release_confirmed = True

        valid_release = (
            self.release_confirmed
            and (not self.high_release)
            and (self.release_height <= self.release_max_height + 1e-7)
        )
        released = both_clear and (gap >= 0.035) and valid_release

        self.max_table_force = max(self.max_table_force, table_force)
        if self.ever_carried:
            self.max_place_table_force = max(self.max_place_table_force, table_force)
        if any_contact or self.prev_any_contact:
            self.max_held_table_force = max(self.max_held_table_force, table_force)
        self.max_robot_table_force = max(self.max_robot_table_force, robot_table_force)

        self.target_error = float(math.hypot(float(pos[0] - self.target_xy[0]), float(pos[1] - self.target_xy[1])))
        in_target_xy = self.target_error <= self.target_tol
        at_target_table = self.ever_lifted_clear and in_target_xy and supported and (tilt <= self.tilt_limit)
        press_failed = self.max_press > self.press_limit + 1e-7
        collision_failed = self.max_robot_table_force > CONTACT_FORCE_N
        settled = (
            at_target_table
            and released
            and quality_ok
            and (not press_failed)
            and (not collision_failed)
            and (float(np.linalg.norm(obj_vel)) < 0.03)
        )
        self.hold_steps = self.hold_steps + 1 if settled else 0
        self.prev_ee_z = ee_z
        self.prev_clearance = self.clearance
        self.prev_any_contact = any_contact

        return {
            "touching": touching,
            "any_contact": any_contact,
            "stable_contact": stable_contact,
            "quality_ok": quality_ok,
            "first_carry": first_carry,
            "first_lift_clear": first_lift_clear,
            "first_release": first_release,
            "first_touchdown": first_touchdown,
            "recontact": recontact,
            "in_target_xy": in_target_xy,
            "at_target_table": at_target_table,
            "released": released,
            "supported": supported,
            "settled": settled,
            "success": self.hold_steps >= self.required_hold_steps,
            "drag_failed": self.max_drag > self.drag_limit,
            "tilt_failed": self.max_tilt_deg > self.tilt_limit,
            "dropped": self.ever_carried and on_table and (not in_target_xy) and (self.loss_steps >= 15),
            "press_failed": press_failed,
            "high_release": self.high_release,
            "collision_failed": collision_failed,
            "press_increment": self.max_press - old_press,
            "up_z": up_z,
        }


def _extract_contact_loads(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    cube_body_id: int,
    left_finger_id: int,
    right_finger_id: int,
    floor_geom_id: int,
    base_body_id: int,
) -> tuple[np.ndarray, float, float]:
    """Return ``(finger_forces[2], table_support_force_n, robot_table_force_n)`` using net 3D force vectors."""
    left_net = np.zeros(3, dtype=float)
    right_net = np.zeros(3, dtype=float)
    table_net = np.zeros(3, dtype=float)
    robot_table_force = 0.0
    wrench = np.zeros(6, dtype=float)

    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.dist > 1e-4:
            continue
        g1, g2 = int(contact.geom1), int(contact.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        is_floor = g1 == floor_geom_id or g2 == floor_geom_id
        has_cube = b1 == cube_body_id or b2 == cube_body_id
        if not (is_floor or has_cube):
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        if not np.all(np.isfinite(wrench[:3])):
            continue
        frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
        force_world = frame.T @ wrench[:3]

        if has_cube and not is_floor:
            other = b2 if b1 == cube_body_id else b1
            if other == left_finger_id:
                left_net += force_world if other == b2 else -force_world
            elif other == right_finger_id:
                right_net += force_world if other == b2 else -force_world
        elif is_floor:
            other = b2 if g1 == floor_geom_id else b1
            if other == cube_body_id:
                table_net += force_world if other == b2 else -force_world
            elif other not in (0, base_body_id):
                robot_table_force += float(np.linalg.norm(force_world))

    f_left = float(np.linalg.norm(left_net))
    f_right = float(np.linalg.norm(right_net))
    table_force = float(np.linalg.norm(table_net))
    return np.array([f_left, f_right], dtype=float), table_force, robot_table_force


class _GripDownIK:
    """``solve_ik(model, data, site, target, GRIP_DOWN_QUAT, q_init, pos_tol=1e-3, rot_tol=1e-2,
    max_iter=20, damping=0.01)`` with constants and buffers built once.

    Same iterations, line search, and floating-point operations as ``ufactory_mujoco.ik.solve_ik``
    (``tests/test_pick_place_rl.py`` compares both bit for bit). Joint ranges must not change after
    construction.
    """

    POS_TOL = 1e-3
    ROT_TOL = 1e-2
    MAX_ITER = 20
    DAMPING = 0.01

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> None:
        self.model = model
        self.data = data
        self.site_id = model.site(site_name).id
        self.tol_vec = np.array([self.POS_TOL] * 3 + [self.ROT_TOL] * 3)
        jnt_types = model.jnt_type
        clamp_mask = (jnt_types == int(mujoco.mjtJoint.mjJNT_HINGE)) | (
            jnt_types == int(mujoco.mjtJoint.mjJNT_SLIDE)
        )
        self.clamp_adrs = model.jnt_qposadr[clamp_mask]
        self.clamp_lo = model.jnt_range[clamp_mask, 0]
        self.clamp_hi = model.jnt_range[clamp_mask, 1]
        self.has_clamp = self.clamp_adrs.size > 0
        self.eye_m = (self.DAMPING**2) * np.eye(6)
        # Rows 0-2 = jacp, rows 3-5 = jacr: the same array np.vstack([jacp, jacr]) builds.
        self.jac = np.zeros((6, model.nv))
        self._quat = np.empty(4)
        self._quat_conj = np.empty(4)
        self._quat_err = np.empty(4)
        self._err = np.empty(6)
        self._err_trial = np.empty(6)

    def _pose_error(self, target_pos: np.ndarray, out: np.ndarray) -> None:
        """``concatenate([target_pos - site_pos, quat_error(site_quat, GRIP_DOWN_QUAT)])`` into ``out``."""
        data, site_id = self.data, self.site_id
        np.subtract(target_pos, data.site_xpos[site_id], out=out[:3])
        quat, conj = self._quat, self._quat_conj
        mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
        conj[0] = quat[0]
        np.negative(quat[1:], out=conj[1:])
        mujoco.mju_mulQuat(self._quat_err, GRIP_DOWN_QUAT, conj)
        mujoco.mju_quat2Vel(out[3:], self._quat_err, 1.0)

    def solve(self, target_pos: np.ndarray, q_init: np.ndarray) -> tuple[np.ndarray, bool]:
        model, data, site_id = self.model, self.data, self.site_id
        tol_vec, jac = self.tol_vec, self.jac
        err, err_trial = self._err, self._err_trial
        q = q_init.astype(float).copy()
        step = 0.5
        for _ in range(self.MAX_ITER):
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            self._pose_error(target_pos, err)
            if np.all(np.abs(err) < tol_vec):
                break
            err_scaled = err / tol_vec
            mujoco.mj_comPos(model, data)
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
            jjt = jac @ jac.T + self.eye_m
            dq = jac.T @ np.linalg.solve(jjt, err)
            v_cur = float(np.sum(np.maximum(np.abs(err_scaled) - 1.0, 0.0)))
            accepted = False
            for _ in range(8):
                q_trial = q.copy()
                mujoco.mj_integratePos(model, q_trial, dq, step)
                if self.has_clamp:
                    q_trial[self.clamp_adrs] = np.clip(q_trial[self.clamp_adrs], self.clamp_lo, self.clamp_hi)
                data.qpos[:] = q_trial
                mujoco.mj_kinematics(model, data)
                self._pose_error(target_pos, err_trial)
                if float(np.sum(np.maximum(np.abs(err_trial / tol_vec) - 1.0, 0.0))) < v_cur:
                    q = q_trial
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
            step = min(step * 1.5, 0.5)

        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        self._pose_error(target_pos, err)
        converged = bool(np.all(np.abs(err[:3]) < self.POS_TOL)) and bool(np.all(np.abs(err[3:]) < self.ROT_TOL))
        return q.copy(), converged


class PickPlaceEnv(_ENV_BASE):
    """xArm6 + calibrated Gripper G2 tabletop pick-and-place Gymnasium environment.

    State changes only through ``reset`` and ``step``. TCP, pad centers, and measured gap are
    cached once per physics state for internal use (the public accessors always read ``data``),
    so writing ``env.data`` directly between steps is not supported.
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        robot: str = "xarm6",
        config_path: str | Path | None = None,
        render_mode: str | None = None,
        max_steps: int = 400,
        decimation: int = 10,
        action_scale: float = 0.005,
        action_clip: float = 1.0,
        gap_action_scale: float = 0.003,
        setpoint_leash_m: float = 0.025,
        pre_grasp_gap_m: float = 0.050,
        close_floor_gap_m: float = 0.012,
        workspace_lower: tuple[float, float, float] | list[float] = (0.10, -0.35, -0.02),
        workspace_upper: tuple[float, float, float] | list[float] = (0.65, 0.35, 0.55),
        spawn_xy_lower: tuple[float, float] | list[float] = SPAWN_XY_LOWER,
        spawn_xy_upper: tuple[float, float] | list[float] = SPAWN_XY_UPPER,
        g2_speed_percent: float = 50.0,
        g2_force_percent: float = 50.0,
        reward_gamma: float = 0.995,
        place_clearance_m: float = PLACE_CLEARANCE_M,
        release_max_height_m: float = RELEASE_MAX_HEIGHT_M,
        press_limit_m: float = PRESS_LIMIT_M,
        near_table_m: float = 0.020,
        max_near_table_descent_m_s: float = 0.020,
        table_overload_margin_n: float = 1.0,
        place_success_dist_m: float = TARGET_TOL_M,
        drag_limit_m: float = DRAG_LIMIT_M,
        tilt_limit_deg: float = TILT_LIMIT_DEG,
        success_hold_steps: int = SUCCESS_HOLD_STEPS,
        reward_scales: dict[str, float] | None = None,
        lite_visuals: bool | None = None,
        expected_runtime_model_sha256: str | None = None,
        fast_info: bool = False,
    ) -> None:
        if robot != "xarm6":
            raise ValueError(f"pick-place RL only supports xArm6, got {robot!r}")
        if render_mode is not None and render_mode != "human":
            raise ValueError(f"unsupported render_mode: {render_mode!r}")
        for name, val in (
            ("max_steps", max_steps),
            ("decimation", decimation),
            ("action_scale", action_scale),
            ("action_clip", action_clip),
            ("gap_action_scale", gap_action_scale),
            ("setpoint_leash_m", setpoint_leash_m),
            ("pre_grasp_gap_m", pre_grasp_gap_m),
            ("close_floor_gap_m", close_floor_gap_m),
            ("g2_speed_percent", g2_speed_percent),
            ("g2_force_percent", g2_force_percent),
        ):
            if not np.isfinite(val) or val <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if g2_speed_percent > 100.0 or g2_force_percent > 100.0:
            raise ValueError("g2_speed_percent and g2_force_percent must be <= 100")

        overlay = config_path if config_path is not None else (CONFIGS_ROOT / "pick_place" / "xarm6.yaml")
        self.task_config = load_task_config(overlay)
        if self.task_config.robot != "xarm6" or self.task_config.gripper != "gripper_g2":
            raise ValueError("pick-place RL requires robot=xarm6 and gripper=gripper_g2")
        pp = self.task_config.pick_place

        self.robot = "xarm6"
        self.env_version = PICK_PLACE_ENV_VERSION
        self.action_contract = PICK_PLACE_ACTION_CONTRACT
        self.runtime_config_sha256 = self.task_config.sha256
        self.render_mode = render_mode
        self.fast_info = bool(fast_info)
        self._max_steps = int(max_steps)
        self._decimation = int(decimation)
        self.action_scale = float(action_scale)
        self.action_clip = float(action_clip)
        self.gap_action_scale = float(gap_action_scale)
        self.setpoint_leash_m = float(setpoint_leash_m)
        self.pre_grasp_gap_m = float(pre_grasp_gap_m)
        self.close_floor_gap_m = float(close_floor_gap_m)
        self.workspace_lower = np.asarray(workspace_lower, dtype=float).reshape(3)
        self.workspace_upper = np.asarray(workspace_upper, dtype=float).reshape(3)
        self.spawn_xy_lower = np.asarray(spawn_xy_lower, dtype=float).reshape(2)
        self.spawn_xy_upper = np.asarray(spawn_xy_upper, dtype=float).reshape(2)
        if (
            np.any(self.spawn_xy_lower > self.spawn_xy_upper)
            or np.any(self.spawn_xy_lower < self.workspace_lower[:2])
            or np.any(self.spawn_xy_upper > self.workspace_upper[:2])
        ):
            raise ValueError("spawn_xy bounds must be ordered and within workspace XY bounds")
        self.g2_speed_percent = float(g2_speed_percent)
        self.g2_force_percent = float(g2_force_percent)
        self.reward_gamma = float(reward_gamma)
        self.near_table_m = float(near_table_m)
        self.max_near_table_descent_m_s = float(max_near_table_descent_m_s)
        self.table_overload_margin_n = float(table_overload_margin_n)
        self.reward_scales = dict(reward_scales or {})

        self.default_spawn_xy = np.asarray(pp.cube_xy, dtype=float).reshape(2)
        self.target_xy = np.asarray(pp.target_xy, dtype=float).reshape(2)
        self.cube_half_m = float(pp.cube_half_m)
        self.object_mass_kg = float(pp.cube_mass_kg)
        self.grasp_gap_m = float(pp.grasp_gap_m)

        scene_path = Path(model_path) if model_path is not None else self.task_config.scene_path
        if not scene_path.is_file():
            raise FileNotFoundError(f"scene model not found: {scene_path}")
        self.scene_path = scene_path
        self.runtime_model_sha256 = runtime_model_sha256(scene_path)
        if expected_runtime_model_sha256 is not None:
            verify_runtime_model(scene_path, expected_runtime_model_sha256)
        if lite_visuals is None:
            lite_visuals = render_mode is None
        self.model = load_reach_model(scene_path, lite_visuals=lite_visuals)
        verify_runtime_model(scene_path, self.runtime_model_sha256)
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)
        self._ik = _GripDownIK(self.model, self._ik_data, TCP_SITE)

        self._gripper_spec = identify_gripper(self.model)
        if self._gripper_spec is None or self._gripper_spec.name != "gripper_g2":
            raise ValueError("scene must mount gripper_g2")
        self.open_gap_m = float(self._gripper_spec.open_gap_m)

        self._n_arm = 6
        self._arm_qfrc_applied = self.data.qfrc_applied[: self._n_arm]
        self._arm_qfrc_bias = self.data.qfrc_bias[: self._n_arm]
        self._joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i + 1}") for i in range(self._n_arm)]
        )
        if (self._joint_ids < 0).any():
            raise ValueError("scene must define joints joint1..joint6")
        self._act_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act{i + 1}") for i in range(self._n_arm)]
        )
        self._gripper_act_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper"))
        self._driver_joint_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint"))
        self._driver_qpos_adr = int(self.model.jnt_qposadr[self._driver_joint_id])
        self._driver_dof_adr = int(self.model.jnt_dofadr[self._driver_joint_id])
        self._gripper_qpos_adrs = np.array(
            [
                int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)])
                for name in (
                    "left_driver_joint",
                    "left_finger_joint",
                    "left_inner_knuckle_joint",
                    "right_driver_joint",
                    "right_finger_joint",
                    "right_inner_knuckle_joint",
                )
            ]
        )

        # Calibrated Gripper G2 servo (matches SimRobot & ufactory_genesis), single-env scalar form
        self.model.actuator_ctrllimited[self._gripper_act_id] = 0
        self.g2_model = G2Model.load()
        self.g2_controller = ScalarG2Controller(
            self.g2_model,
            initial_position_mm=self.pre_grasp_gap_m * 1000.0,
        )
        self.g2_controller.command(
            self.pre_grasp_gap_m * 1000.0,
            self.g2_speed_percent,
            self.g2_force_percent,
        )
        self._g2_kv_over_kp = self.g2_controller.kv / self.g2_controller.kp
        init_limit = self.g2_controller.torque_limit_nm
        self.model.actuator_forcerange[self._gripper_act_id] = (-init_limit, init_limit)

        self._qpos_adr = self.model.jnt_qposadr[self._joint_ids]
        self._qvel_adr = self.model.jnt_dofadr[self._joint_ids]
        self._q_lo = self.model.jnt_range[self._joint_ids, 0].copy()
        self._q_hi = self.model.jnt_range[self._joint_ids, 1].copy()

        self._site_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE))
        self._cube_body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube"))
        self._cube_joint_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube"))
        self._cube_qpos_adr = int(self.model.jnt_qposadr[self._cube_joint_id])
        self._cube_dof_adr = int(self.model.jnt_dofadr[self._cube_joint_id])
        self._left_finger_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"))
        self._right_finger_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"))
        self._base_body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link_base"))
        self._floor_geom_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
        self._configure_contact_physics()

        self.ctrl_dt = self._decimation * float(self.model.opt.timestep)
        self._home_q, self._heights = self._compute_home_pose(pp)
        self.home_tcp = np.array([SPAWN_X, SPAWN_Y, self._heights["lift_z"]], dtype=float)

        self.metrics = PickPlaceMetrics(
            ctrl_dt=self.ctrl_dt,
            target_tol=place_success_dist_m,
            drag_limit=drag_limit_m,
            tilt_limit=tilt_limit_deg,
            hold_steps=success_hold_steps,
            place_clearance=place_clearance_m,
            release_max_height=release_max_height_m,
            press_limit=press_limit_m,
            spawn_xy=tuple(self.default_spawn_xy),
            target_xy=tuple(self.target_xy),
            cube_half_m=self.cube_half_m,
        )

        self._rng = np.random.default_rng()
        self.fixed_spawns: np.ndarray | None = None
        self.fixed_cursor = 0
        self.scenario_id = -1
        self._step_count = 0
        self.episode_id = -1
        self.ee_setpoint = self.home_tcp.copy()
        self.q_target = self._home_q.copy()
        self.gap_cmd = self.pre_grasp_gap_m
        self.forces = np.zeros(2, dtype=float)
        self.table_force = 0.0
        self.prev_raw_table_force = 0.0
        self.robot_table_force = 0.0
        self.last_action = np.zeros(PICK_PLACE_NUM_ACTIONS, dtype=np.float32)
        self.contact_rewarded = False
        self.touchdown_rewarded = False
        self.place_rewarded = False
        self.ik_rejects = 0
        self.ep_return = 0.0
        self._prev_tcp_pos = self.home_tcp.copy()
        self._viewer = None
        self._cached_potential: float | None = None  # potential of the current state, set at the end of step()
        self._refresh_kinematics()

        obs_low = np.full(PICK_PLACE_NUM_OBS, -10.0, dtype=np.float32)
        obs_high = np.full(PICK_PLACE_NUM_OBS, 10.0, dtype=np.float32)
        if spaces is not None:
            self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (PICK_PLACE_NUM_ACTIONS,), dtype=np.float32)
        else:
            self.observation_space = _BoxStub(obs_low, obs_high, (PICK_PLACE_NUM_OBS,), np.float32)
            self.action_space = _BoxStub(-1.0, 1.0, (PICK_PLACE_NUM_ACTIONS,), np.float32)

    def _configure_contact_physics(self) -> None:
        """Configure rigid Signorini-like contact impedance for calibrated G2 pinch forces on the 17 g cube."""
        m = self.model
        m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        timeconst = max(0.004, 2.0 * float(m.opt.timestep))
        contact_bodies = {self._cube_body_id, self._left_finger_id, self._right_finger_id}
        finger_bodies = {self._left_finger_id, self._right_finger_id}
        for gid in range(m.ngeom):
            body_id = int(m.geom_bodyid[gid])
            if gid == self._floor_geom_id or body_id in contact_bodies:
                m.geom_solref[gid] = (timeconst, 1.0)
                m.geom_solimp[gid] = (0.99, 0.999, 0.001, 0.5, 2.0)
            if body_id in finger_bodies:
                m.geom_friction[gid, 0] = max(float(m.geom_friction[gid, 0]), 1.2)

    def _compute_home_pose(self, pp) -> tuple[np.ndarray, dict[str, float]]:
        """Solve fixed home deployment pose at (0.30, 0.00) with flange at ~0.30 m."""
        mujoco.mj_resetDataKeyframe(self.model, self._ik_data, 0)
        mujoco.mj_forward(self.model, self._ik_data)
        q_seed, ok = solve_ik(
            self.model,
            self._ik_data,
            TCP_SITE,
            np.array([SPAWN_X, SPAWN_Y, float(pp.lift_z_m)]),
            GRIP_DOWN_QUAT,
            q_init=self._ik_data.qpos.copy(),
        )
        if not ok:
            raise RuntimeError("IK failed while solving seed pose")
        self._ik_data.qpos[:] = q_seed
        mujoco.mj_forward(self.model, self._ik_data)
        tcp_z = float(self._ik_data.site_xpos[self._site_id][2])
        link6_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link6"))
        flange_z = float(self._ik_data.body(link6_id).xpos[2])
        off = tcp_z - flange_z
        heights = {
            "grasp_z": max(_PRE_LIFT_MIN_TCP_Z, float(pp.flange_grasp_z_m) + off),
            "pre_grasp_z": float(pp.flange_grasp_z_m) + float(pp.approach_m) + off,
            "lift_z": 0.30 + off,
        }
        q_home, ok = solve_ik(
            self.model,
            self._ik_data,
            TCP_SITE,
            np.array([SPAWN_X, SPAWN_Y, heights["lift_z"]]),
            GRIP_DOWN_QUAT,
            q_init=q_seed,
        )
        if not ok:
            raise RuntimeError("IK failed while solving home deployment pose")
        return q_home[: self._n_arm].copy(), heights

    def set_evaluation_spawns(self, xy: Any) -> None:
        """Install deterministic (N, 2) evaluation cube XY spawns; pass None to clear."""
        if xy is None:
            self.fixed_spawns = None
            self.fixed_cursor = 0
            return
        arr = np.asarray(xy, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 1:
            raise ValueError(f"evaluation spawns must have shape (N, 2), got {arr.shape}")
        self.fixed_spawns = arr.copy()
        self.fixed_cursor = 0

    # ------------------------------------------------------------------ state
    @property
    def n_joints(self) -> int:
        """Number of arm joints."""
        return self._n_arm

    @property
    def step_count(self) -> int:
        """Control steps taken in the current episode."""
        return self._step_count

    def tcp_position(self) -> np.ndarray:
        """TCP site position in the world frame (m)."""
        return self.data.site_xpos[self._site_id].copy()

    def tcp_velocity(self) -> np.ndarray:
        """TCP velocity over the last control step (m/s)."""
        return (self.tcp_position() - self._prev_tcp_pos) / self.ctrl_dt

    def cube_position(self) -> np.ndarray:
        """Cube position in the world frame (m)."""
        return self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3].copy()

    def cube_quat(self) -> np.ndarray:
        """Cube orientation quaternion (w, x, y, z)."""
        return self.data.qpos[self._cube_qpos_adr + 3 : self._cube_qpos_adr + 7].copy()

    def cube_velocity(self) -> np.ndarray:
        """Cube linear velocity (m/s)."""
        return self.data.qvel[self._cube_dof_adr : self._cube_dof_adr + 3].copy()

    def cube_angular_velocity(self) -> np.ndarray:
        """Cube angular velocity (rad/s)."""
        return self.data.qvel[self._cube_dof_adr + 3 : self._cube_dof_adr + 6].copy()

    def pad_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """World positions of the left and right finger-pad centers."""
        l_pos = self.data.xpos[self._left_finger_id]
        l_mat = self.data.xmat[self._left_finger_id].reshape(3, 3)
        r_pos = self.data.xpos[self._right_finger_id]
        r_mat = self.data.xmat[self._right_finger_id].reshape(3, 3)
        return l_pos + l_mat @ _LEFT_PAD_LOCAL, r_pos + r_mat @ _RIGHT_PAD_LOCAL

    def measured_gap(self) -> float:
        """Measured fingertip pad separation in meters via 4-bar linkage geometry."""
        return self._pad_gap(*self.pad_centers())

    def _pad_gap(self, left_center: np.ndarray, right_center: np.ndarray) -> float:
        l_mat = self.data.xmat[self._left_finger_id].reshape(3, 3)
        normal = l_mat @ _PAD_NORMAL_LOCAL
        return float(abs(np.dot(right_center - left_center, normal)))

    def _refresh_kinematics(self) -> None:
        """Cache TCP, pad centers, and measured gap of the current physics state for internal use."""
        self._tcp = self.tcp_position()
        self._left_pad, self._right_pad = self.pad_centers()
        self._gap = self._pad_gap(self._left_pad, self._right_pad)

    def _solve_ik(self, target_pos: np.ndarray) -> tuple[np.ndarray, bool]:
        q_init = self.data.qpos.copy()
        q_init[self._qpos_adr] = self.q_target
        q_sol, converged = self._ik.solve(target_pos, q_init)
        q_arm = q_sol[self._qpos_adr].copy()
        reject = (
            (not converged)
            or (not np.all(np.isfinite(q_arm)))
            or bool(np.max(np.abs(q_arm - self.q_target)) > 0.5)
            or bool(np.any(q_arm < self._q_lo - 1e-5))
            or bool(np.any(q_arm > self._q_hi + 1e-5))
        )
        if reject:
            self.ik_rejects += 1
            return self.q_target.copy(), True
        return q_arm, False

    def _apply_action(self, action: np.ndarray, current_measured_gap: float) -> None:
        """Integrate 4D increments [dx, dy, dz, dgap] with near-table safety guardrails."""
        act = np.asarray(action, dtype=float).reshape(PICK_PLACE_NUM_ACTIONS).copy()
        ee = self._tcp
        base_setpoint = self.ee_setpoint.copy()

        obj_pos = self.cube_position()
        height = max(0.0, cube_clearance(obj_pos, self.cube_quat(), self.cube_half_m))
        in_target_xy = (
            math.hypot(float(obj_pos[0] - self.target_xy[0]), float(obj_pos[1] - self.target_xy[1]))
            <= self.metrics.target_tol
        )
        at_release_table = (
            self.metrics.ever_lifted_clear
            and in_target_xy
            and (height <= self.metrics.release_max_height)
            and (
                self.metrics.touchdown_seen
                or self.metrics.release_confirmed
                or (height <= self.metrics.place_clearance)
            )
        )

        near_place = False
        table_sp_z: float | None = None
        if not self.metrics.ever_lifted_clear:
            gripping_pre_lift = (self.metrics.contact_steps >= CONTACT_CONFIRM_STEPS) or self.metrics.ever_carried
            pre_lift_min_sp_z = (
                max(base_setpoint[2], ee[2], _PRE_LIFT_MIN_TCP_Z) if gripping_pre_lift else _PRE_LIFT_MIN_TCP_Z
            )
            base_setpoint[2] = max(base_setpoint[2], pre_lift_min_sp_z)
            if gripping_pre_lift:
                act[2] = max(0.0, float(act[2]))
        elif height <= self.near_table_m:
            near_place = True
            # Allow ~0.4 mm gentle table contact settlement while physically bounding press < 1.0 mm
            ref_table_ee_z = max(self.metrics.touchdown_ee_z, float(ee[2]) - height)
            table_sp_z = ref_table_ee_z - 0.0004
            base_setpoint[2] = max(base_setpoint[2], max(table_sp_z, float(ee[2]) - 0.0040))
            near_dz_limit = self.max_near_table_descent_m_s * self.ctrl_dt
            dz_cmd = max(float(act[2]) * self.action_scale, -near_dz_limit) / self.action_scale
            if at_release_table:
                opening_started = (
                    self.gap_cmd >= 2.0 * self.cube_half_m - 0.0005
                    or self.metrics.clear_steps > 0
                    or self.metrics.release_confirmed
                )
                dz_cmd = min(dz_cmd, near_dz_limit / self.action_scale)
                if (self.metrics.touchdown_seen or opening_started) and float(act[2]) <= 0.3:
                    dz_cmd = 0.0
                if opening_started:
                    if current_measured_gap < 0.038:
                        act[:2] = 0.0
                    else:
                        act[:2] *= 0.25
                    act[3] = max(0.0, float(act[3]))
                else:
                    act[:2] *= 0.25
            act[2] = dz_cmd

        moving = bool(np.any(act[:3] != 0.0)) or bool(np.any(base_setpoint != self.ee_setpoint))
        if moving:
            target = base_setpoint + act[:3] * self.action_scale
            if not self.metrics.ever_lifted_clear:
                target[2] = max(target[2], _PRE_LIFT_MIN_TCP_Z)
            if near_place and table_sp_z is not None:
                target[2] = max(target[2], table_sp_z)
            target = np.clip(target, self.workspace_lower, self.workspace_upper)
            delta = target - ee
            dist = float(np.linalg.norm(delta))
            if dist > self.setpoint_leash_m:
                target = ee + delta * (self.setpoint_leash_m / max(dist, 1e-9))
            q_new, reject = self._solve_ik(target)
            if not reject:
                self.ee_setpoint = target
                self.q_target = q_new
            else:
                self.ee_setpoint = base_setpoint

        self.data.ctrl[self._act_ids] = self.q_target

        base_gap = self.gap_cmd
        if at_release_table and float(act[3]) > 0.0 and self.gap_cmd < current_measured_gap:
            base_gap = current_measured_gap
        self.gap_cmd = float(
            np.clip(base_gap + float(act[3]) * self.gap_action_scale, self.close_floor_gap_m, self.open_gap_m)
        )
        self.g2_controller.command(self.gap_cmd * 1000.0, self.g2_speed_percent, self.g2_force_percent)
        q_drv = float(self.data.qpos[self._driver_qpos_adr])
        v_drv = float(self.data.qvel[self._driver_dof_adr])
        out = self.g2_controller.update(
            q_drv,
            v_drv,
            self.ctrl_dt,
            measured_gap_mm=current_measured_gap * 1000.0,
        )
        limit = out.torque_limit_nm
        self.model.actuator_forcerange[self._gripper_act_id] = (-limit, limit)
        self.data.ctrl[self._gripper_act_id] = out.position_rad + self._g2_kv_over_kp * out.velocity_rad_s

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Gymnasium reset.

        ``options={'cube_xy': ...}`` fixes the spawn; otherwise evaluation spawns or a random spawn are used.
        """
        if gym is not None:
            super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if options is not None and "cube_xy" in options:
            spawn_xy = np.asarray(options["cube_xy"], dtype=float).reshape(2).copy()
            self.scenario_id = int(options.get("scenario_id", -1))
        elif self.fixed_spawns is not None and len(self.fixed_spawns) > 0:
            idx = self.fixed_cursor % len(self.fixed_spawns)
            spawn_xy = self.fixed_spawns[idx].copy()
            self.scenario_id = self.fixed_cursor
            self.fixed_cursor += 1
        else:
            u = self._rng.random(2)
            spawn_xy = self.spawn_xy_lower + u * (self.spawn_xy_upper - self.spawn_xy_lower)
            self.scenario_id = -1

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        self.data.qpos[self._qpos_adr] = self._home_q
        self.data.ctrl[self._act_ids] = self._home_q

        drv_rad = gripper_g2_gap_m_to_sim_drive(self.pre_grasp_gap_m)
        self.data.qpos[self._gripper_qpos_adrs] = drv_rad
        self.g2_controller.reset(position_mm=self.pre_grasp_gap_m * 1000.0)
        self.g2_controller.command(
            self.pre_grasp_gap_m * 1000.0,
            self.g2_speed_percent,
            self.g2_force_percent,
        )
        limit = self.g2_controller.torque_limit_nm
        self.model.actuator_forcerange[self._gripper_act_id] = (-limit, limit)
        self.data.ctrl[self._gripper_act_id] = drv_rad

        self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3] = [
            spawn_xy[0],
            spawn_xy[1],
            CUBE_REST_Z,
        ]
        self.data.qpos[self._cube_qpos_adr + 3 : self._cube_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        self._refresh_kinematics()
        self._cached_potential = None

        self._step_count = 0
        self.episode_id += 1
        self.q_target = self._home_q.copy()
        self.ee_setpoint = self._tcp.copy()
        self._prev_tcp_pos = self.ee_setpoint.copy()
        self.gap_cmd = self.pre_grasp_gap_m
        self.forces = np.zeros(2, dtype=float)
        self.table_force = 0.0
        self.prev_raw_table_force = 0.0
        self.robot_table_force = 0.0
        self.last_action = np.zeros(PICK_PLACE_NUM_ACTIONS, dtype=np.float32)
        self.contact_rewarded = False
        self.touchdown_rewarded = False
        self.place_rewarded = False
        self.ik_rejects = 0
        self.ep_return = 0.0

        self.metrics.reset(spawn_xy=spawn_xy, ee_z=float(self.ee_setpoint[2]))
        obs = self._get_obs()
        info = {} if self.fast_info else self._build_info(success=False, failed=False, timeout=False)
        if self.render_mode == "human":
            self._ensure_viewer()
            self._viewer.sync()
        return obs, info

    def _action_penalty(self, raw_action: np.ndarray) -> float:
        bounded = np.clip(raw_action, -self.action_clip, self.action_clip)
        excess = np.clip(raw_action - bounded, -5.0, 5.0)
        s = self.reward_scales
        return float(s.get("action_penalty", 0.001)) * float(bounded @ bounded) + float(
            s.get("action_saturation_penalty", 0.1)
        ) * float(excess @ excess)

    def _potential(self) -> float:
        obj = self.cube_position()
        quat = self.cube_quat()
        left, right = self._left_pad, self._right_pad
        pad_mid = 0.5 * (left + right)
        distance = grasp_distance(pad_mid, obj, self.cube_half_m)
        pad_xy_err = float(math.hypot(float(pad_mid[0] - obj[0]), float(pad_mid[1] - obj[1])))
        dz_band = max(0.0, abs(float(pad_mid[2] - obj[2])) - CUBE_SIZE * 0.4)
        height = max(0.0, cube_clearance(obj, quat, self.cube_half_m))
        target_error = float(math.hypot(float(obj[0] - self.target_xy[0]), float(obj[1] - self.target_xy[1])))
        current_gap = self._gap
        grasp_near = math.exp(-distance / 0.025)
        placed = self.place_rewarded

        min_force = float(min(self.forces[0], self.forces[1]))
        if self.metrics.ever_lifted_clear:
            grip_force = min(1.0, max(0.0, min_force / 2.0))
        else:
            grip_force = 0.75 * min(1.0, max(0.0, min_force / 2.0)) + 0.25 * min(1.0, max(0.0, min_force / 20.0))
        grip = grip_force * grasp_near * (0.0 if placed else 1.0)

        any_force = bool(self.forces[0] > 0.2 or self.forces[1] > 0.2)
        outside_xy = max(0.0, pad_xy_err - 0.015)
        z_floor = float(obj[2]) + CUBE_SIZE * 0.4 + 0.038 * (1.0 - math.exp(-((outside_xy / 0.015) ** 2)))
        if self.metrics.ever_lifted_clear or any_force:
            below_floor = 0.0
            near_cube = 1.0
        else:
            below_floor = max(0.0, z_floor - float(pad_mid[2]))
            near_cube = math.exp(-((distance / 0.012) ** 2))
        floor_gate = math.exp(-below_floor / 0.015)

        at_table = self.metrics.ever_lifted_clear and (
            (height <= self.metrics.release_max_height)
            or self.metrics.touchdown_seen
            or self.metrics.release_confirmed
        )
        at_target_table = at_table and (target_error <= self.metrics.target_tol)
        if at_target_table:
            over_open = 0.0
        else:
            over_open = 0.65 * max(0.0, current_gap - self.pre_grasp_gap_m) + 0.35 * max(
                0.0, self.gap_cmd - self.pre_grasp_gap_m
            )
        gap_span_floor = max(1e-6, self.pre_grasp_gap_m - self.close_floor_gap_m)
        gap_span_cube = max(1e-6, self.pre_grasp_gap_m - CUBE_SIZE)
        early_close = min(1.0, max(0.0, (self.pre_grasp_gap_m - min(current_gap, self.gap_cmd)) / gap_span_floor))
        cmd_close = min(1.0, max(0.0, (self.pre_grasp_gap_m - self.gap_cmd) / gap_span_floor))
        meas_close = min(1.0, max(0.0, (self.pre_grasp_gap_m - current_gap) / gap_span_cube))
        if at_target_table:
            close_progress = 1.0
        elif self.metrics.ever_lifted_clear:
            close_progress = meas_close
        else:
            close_progress = 0.65 * meas_close + 0.35 * cmd_close
        pregrasp_gap = (
            0.62 * (1.0 - ((1.0 - near_cube) ** 2) * early_close) + 0.38 * near_cube * close_progress
        ) * math.exp(-over_open / 0.010)

        approach_kernel = (
            0.35 * math.exp(-pad_xy_err / 0.22 - dz_band / 0.06) + 0.65 * math.exp(-distance / 0.045)
        ) * floor_gate
        pregrasp_kernel = 1.125 * math.exp(-distance / 0.040) * pregrasp_gap * floor_gate
        target_near = math.exp(-target_error / 0.04)
        place_approach = 0.75 * target_near + 0.25 * math.exp(-target_error / 0.18)
        target_zone = math.exp(-max(0.0, target_error - 0.015) / 0.04)
        lifted = 1.0 if self.metrics.ever_lifted_clear else 0.0
        place_height_err = max(0.0, height - self.metrics.place_clearance)
        smooth_clear = math.exp(-max(float(self.forces[0]), float(self.forces[1])) / 10.0)
        meas_open = min(1.0, max(0.0, (current_gap - CUBE_SIZE) / gap_span_cube))

        ee_z = float(self._tcp[2])
        ref_ee_z = self.metrics.touchdown_ee_z if self.metrics.touchdown_ee_z > 0.0 else self._heights["grasp_z"]
        has_td_ref = (self.metrics.touchdown_ee_z > 0.0 or self.metrics.release_confirmed) and at_table
        ee_sink = max(0.0, ref_ee_z - ee_z - 0.0025) if has_td_ref else 0.0
        post_td_pose = math.exp(-ee_sink / 0.0015)
        table_guard = post_td_pose * (smooth_clear if placed else 1.0)
        place_zone = lifted * target_zone * math.exp(-place_height_err / 0.005) * table_guard
        descent_funnel = target_zone * math.exp(-place_height_err / 0.025) * post_td_pose
        cmd_open = min(1.0, max(0.0, (self.gap_cmd - self.close_floor_gap_m) / gap_span_floor))
        action_open = min(1.0, max(0.0, 0.5 * (float(self.last_action[3]) + 1.0)))
        open_drive = 0.85 * cmd_open + 0.15 * action_open
        release_zone = place_zone * max(open_drive, meas_open)

        approach_term = max(
            approach_kernel * (table_guard if at_table else 1.0),
            release_zone,
        )
        pregrasp = max(
            pregrasp_kernel * (table_guard if at_table else 1.0),
            1.125 * release_zone,
        )
        grasp_or_place = max(
            grip * (post_td_pose if at_table else 1.0),
            place_zone,
        )
        hold_progress = min(1.0, max(0.0, self.metrics.hold_steps / self.metrics.required_hold_steps))
        init_target_dist = float(
            math.hypot(
                float(self.metrics.spawn_xy[0] - self.target_xy[0]),
                float(self.metrics.spawn_xy[1] - self.target_xy[1]),
            )
        )
        lift_cap = SAFE_LIFT_Z - CUBE_REST_Z
        max_lift = min(lift_cap, max(0.0, self.metrics.max_held_z - CUBE_REST_Z))
        current_lift = (
            min(lift_cap, max(0.0, height))
            if (self.forces[0] > RELEASE_FORCE_N and self.forces[1] > RELEASE_FORCE_N)
            else 0.0
        )
        lift_progress = max_lift if self.metrics.ever_lifted_clear else (0.75 * current_lift + 0.25 * max_lift)
        transit_xy_progress = max(-0.05, init_target_dist - target_error)
        excess_height = max(0.0, height - lift_cap)

        s = self.reward_scales
        return float(
            float(s.get("approach_potential", 8.0)) * approach_term
            + float(s.get("pregrasp_potential", 4.0)) * pregrasp
            + float(s.get("grasp_potential", 4.0)) * grasp_or_place
            + float(s.get("lift_potential", 40.0)) * lift_progress
            + float(s.get("transit_potential", 8.0)) * lifted * (transit_xy_progress - 0.8 * excess_height)
            + float(s.get("place_potential", 16.0))
            * lifted
            * (0.4 * place_approach / (1.0 + place_height_err / 0.10) + 0.6 * descent_funnel)
            + float(s.get("release_potential", 4.0)) * place_zone * open_drive
            + float(s.get("hold_potential", 6.0)) * (0.5 * place_zone * meas_open + 0.5 * hold_progress)
        )

    def step(self, action):
        """Gymnasium step with a 4-D action: Cartesian TCP increment (3) and G2 gap increment (1), all in [-1, 1]."""
        raw_action = np.asarray(action, dtype=float).reshape(PICK_PLACE_NUM_ACTIONS)
        if not np.all(np.isfinite(raw_action)):
            raise ValueError("action must be finite")
        act_pen = self._action_penalty(raw_action)
        clipped = np.clip(raw_action, -self.action_clip, self.action_clip)

        old_potential = self._potential() if self._cached_potential is None else self._cached_potential
        self._prev_tcp_pos = self._tcp.copy()
        self._apply_action(clipped, self._gap)

        qfrc_applied, qfrc_bias = self._arm_qfrc_applied, self._arm_qfrc_bias
        for _ in range(self._decimation):
            qfrc_applied[:] = qfrc_bias
            mujoco.mj_step(self.model, self.data)
        self.data.qfrc_applied[:] = 0.0
        self._refresh_kinematics()

        self._step_count += 1
        self.last_action = clipped.astype(np.float32)

        ee = self._tcp
        obj = self.cube_position()
        obj_quat = self.cube_quat()
        obj_vel = self.cube_velocity()
        self.forces, raw_table_force, self.robot_table_force = _extract_contact_loads(
            self.model,
            self.data,
            cube_body_id=self._cube_body_id,
            left_finger_id=self._left_finger_id,
            right_finger_id=self._right_finger_id,
            floor_geom_id=self._floor_geom_id,
            base_body_id=self._base_body_id,
        )
        substep_alias = (
            raw_table_force == 0.0
            and self.prev_raw_table_force > SUPPORT_FORCE_N
            and cube_clearance(obj, obj_quat, self.cube_half_m) <= 0.0
            and abs(float(obj_vel[2])) < 0.01
        )
        self.table_force = self.prev_raw_table_force if substep_alias else raw_table_force
        self.prev_raw_table_force = raw_table_force

        status = self.metrics.update(
            obj,
            obj_quat,
            self.forces,
            obj_vel,
            self._gap,
            ee_z=float(ee[2]),
            table_force=self.table_force,
            robot_table_force=self.robot_table_force,
        )

        success = bool(status["success"])
        failed = bool(
            status["drag_failed"]
            or status["tilt_failed"]
            or status["dropped"]
            or status["press_failed"]
            or status["high_release"]
            or status["collision_failed"]
        )
        timeout = bool(self._step_count >= self._max_steps and not (success or failed))
        terminated = bool(success or failed)
        truncated = bool(timeout)

        s = self.reward_scales
        left, right = self._left_pad, self._right_pad
        pad_mid_z = 0.5 * float(left[2] + right[2])
        in_vertical_grasp_band = abs(pad_mid_z - float(obj[2])) <= CUBE_SIZE * 0.4 + 1e-4
        first_contact = bool(
            status["stable_contact"]
            and (not self.contact_rewarded)
            and status["quality_ok"]
            and in_vertical_grasp_band
        )
        self.contact_rewarded = self.contact_rewarded or first_contact
        first_target_td = bool(
            self.metrics.ever_lifted_clear
            and status["in_target_xy"]
            and (self.metrics.touchdown_seen or self.metrics.clearance <= self.metrics.place_clearance)
            and (not self.touchdown_rewarded)
        )
        self.touchdown_rewarded = self.touchdown_rewarded or first_target_td
        first_place = bool(
            status["first_release"]
            and status["in_target_xy"]
            and (not self.metrics.high_release)
            and (not self.place_rewarded)
        )
        self.place_rewarded = self.place_rewarded or first_place

        done = terminated or truncated
        # Zero terminal potential on failure so failing early can never harvest shaping;
        # keep terminal potential on timeout/success so finite-horizon PBRS telescopes cleanly.
        new_potential = 0.0 if failed else self._potential()
        # Its inputs stay unchanged until the next step(), which reuses it as old_potential; reset() clears it.
        self._cached_potential = None if failed else new_potential
        reward = self.reward_gamma * new_potential - old_potential - float(s.get("step_penalty", 0.005))
        if first_contact:
            reward += float(s.get("first_contact", 2.0))
        if status["first_carry"]:
            reward += float(s.get("first_carry", 3.0))
        if status["first_lift_clear"]:
            reward += float(s.get("first_lift", 6.0))
        if first_target_td:
            reward += float(s.get("first_touchdown", 6.0))
        if first_place:
            reward += float(s.get("first_place", 10.0))

        at_release_table = (
            (self.metrics.clearance <= 0.0005)
            or self.metrics.touchdown_seen
            or self.metrics.release_confirmed
        )
        near_held = (
            self.metrics.ever_lifted_clear
            and (self.metrics.clearance < self.near_table_m)
            and (status["any_contact"] or at_release_table)
        )
        if near_held:
            ee_vel = (self._tcp - self._prev_tcp_pos) / self.ctrl_dt
            descent_limit = 0.005 if at_release_table else (self.max_near_table_descent_m_s + 0.010)
            fast_descent = max(0.0, -float(ee_vel[2]) - descent_limit)
            lateral_excess = max(0.0, float(np.linalg.norm(ee_vel[:2])) - 0.02) if at_release_table else 0.0
            reward -= float(s.get("fast_descent_penalty", 10.0)) * (fast_descent + lateral_excess)
            overload = max(0.0, self.table_force - self.object_mass_kg * 9.81 - self.table_overload_margin_n)
            reward -= float(s.get("table_overload_penalty", 0.1)) * overload * self.ctrl_dt

        reward -= float(s.get("press_penalty", 3000.0)) * float(status["press_increment"])
        if status["recontact"]:
            reward -= float(s.get("recontact_penalty", 5.0))
        if status["first_release"] and self.metrics.high_release:
            reward -= float(s.get("high_release_penalty", 10.0))
        reward -= float(s.get("robot_table_penalty", 1.0)) * self.robot_table_force * self.ctrl_dt
        reward -= act_pen
        if success:
            reward += float(s.get("success", 40.0))
        elif failed:
            reward -= float(s.get("failure_penalty", 10.0))

        self.ep_return += float(reward)
        obs = self._get_obs()
        if self.fast_info and not done:
            info: dict[str, Any] = {}
        else:
            info = self._build_info(success=success, failed=failed, timeout=timeout, status=status)
        if self.render_mode == "human":
            self._ensure_viewer()
            self._viewer.sync()
        return obs, float(reward), terminated, truncated, info

    def _build_info(
        self,
        *,
        success: bool,
        failed: bool,
        timeout: bool,
        status: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        obj = self.cube_position()
        m = self.metrics
        return {
            "success": bool(success),
            "failed": bool(failed),
            "timeout": bool(timeout),
            "drag_failed": bool(status["drag_failed"]) if status else (m.max_drag > m.drag_limit),
            "tilt_failed": bool(status["tilt_failed"]) if status else (m.max_tilt_deg > m.tilt_limit),
            "dropped": bool(status["dropped"]) if status else False,
            "press_failed": bool(status["press_failed"]) if status else (m.max_press > m.press_limit + 1e-7),
            "high_release": bool(m.high_release),
            "collision_failed": (
                bool(status["collision_failed"]) if status else (m.max_robot_table_force > CONTACT_FORCE_N)
            ),
            "max_drag_mm": float(m.max_drag * 1000.0),
            "max_tilt_deg": float(m.max_tilt_deg),
            "target_error_mm": float(m.target_error * 1000.0),
            "release_height_mm": float(m.release_height * 1000.0),
            "release_confirmed": bool(m.release_confirmed),
            "ever_carried": bool(m.ever_carried),
            "ever_lifted_clear": bool(m.ever_lifted_clear),
            "max_held_clearance_mm": float((m.max_held_z - CUBE_REST_Z) * 1000.0),
            "touchdown_seen": bool(m.touchdown_seen),
            "press_mm": float(m.max_press * 1000.0),
            "max_table_force_n": float(m.max_table_force),
            "max_place_table_force_n": float(m.max_place_table_force),
            "max_held_table_force_n": float(m.max_held_table_force),
            "max_robot_table_force_n": float(m.max_robot_table_force),
            "clearance_mm": float(m.clearance * 1000.0),
            "obj_pos": obj.tolist(),
            "spawn_xy": m.spawn_xy.tolist(),
            "scenario_id": int(self.scenario_id),
            "hold_steps": int(m.hold_steps),
            "steps": int(self._step_count),
            "return": float(self.ep_return),
            "ik_rejects": int(self.ik_rejects),
        }

    def _get_obs(self) -> np.ndarray:
        """70 policy inputs aligned with ufactory_genesis: measured state, contacts, and safety margins."""
        q = self.data.qpos[self._qpos_adr] / math.pi  # 6
        qd = self.data.qvel[self._qvel_adr] / 2.0  # 6
        ee = self._tcp
        ee_vel = ((self._tcp - self._prev_tcp_pos) / self.ctrl_dt) / 0.2  # 3
        obj = self.cube_position()
        quat = self.cube_quat()
        q_norm = float(np.linalg.norm(quat))
        quat_unit = quat / q_norm if q_norm > 0.0 else np.array([1.0, 0.0, 0.0, 0.0])  # 4
        obj_vel = self.cube_velocity() / 0.2  # 3
        obj_ang = self.cube_angular_velocity() / 2.0  # 3
        left, right = self._left_pad, self._right_pad
        meas_gap = self._gap
        m = self.metrics

        if m.ever_lifted_clear:
            ee_obs = np.array(
                [
                    (ee[0] - self.target_xy[0]) / 0.3,
                    (ee[1] - self.target_xy[1]) / 0.3,
                    ee[2],
                ],
                dtype=float,
            )
            at_release = m.touchdown_seen or m.release_confirmed
            target_obs = np.array(
                [
                    (self.target_xy[0] - obj[0]) / 0.3,
                    (self.target_xy[1] - obj[1]) / 0.3,
                    0.0 if at_release else (CUBE_REST_Z - obj[2]) / 0.3,
                ],
                dtype=float,
            )
        else:
            ee_obs = np.array([0.0, 0.0, ee[2]], dtype=float)
            target_obs = np.zeros(3, dtype=float)

        obj_obs = np.array([obj[0] - ee[0], obj[1] - ee[1], obj[2]], dtype=float)  # 3
        left_rel = (left - obj) / 0.1  # 3
        right_rel = (right - obj) / 0.1  # 3
        gaps = np.array([self.gap_cmd / self.open_gap_m, meas_gap / self.open_gap_m], dtype=float)  # 2
        setpoint_err = (self.ee_setpoint - ee) / self.setpoint_leash_m  # 3
        forces_obs = np.clip(self.forces / 20.0, 0.0, 1.0)  # 2
        contact_flags = m.finger_contact.astype(float)  # 2
        table_loads = np.array(
            [
                min(1.0, max(0.0, self.table_force / 20.0)) if m.ever_lifted_clear else 0.0,
                min(1.0, max(0.0, self.robot_table_force / 20.0)),
            ],
            dtype=float,
        )  # 2
        quality_hold = np.array(
            [
                m.max_drag / max(m.drag_limit, 1e-9),
                m.max_tilt_deg / max(m.tilt_limit, 1e-9),
                m.hold_steps / max(m.required_hold_steps, 1),
            ],
            dtype=float,
        )  # 3
        lift_flags = np.array([float(m.ever_carried), float(m.ever_lifted_clear)], dtype=float)  # 2
        press_obs = np.array(
            [
                m.max_press / max(m.press_limit, 1e-9),
                float(m.touchdown_seen),
                ((m.touchdown_ee_z - ee[2]) / 0.01) if m.touchdown_seen else 0.0,
            ],
            dtype=float,
        )  # 3
        release_obs = np.array(
            [
                m.release_height / max(m.release_max_height, 1e-9),
                float(m.release_confirmed),
                float(m.high_release),
            ],
            dtype=float,
        )  # 3
        reward_flags = np.array([float(self.contact_rewarded), float(self.place_rewarded)], dtype=float)  # 2
        tail = np.array(
            [
                1.0 - float(self._step_count) / max(self._max_steps, 1),
                min(1.0, max(0.0, (m.max_held_z - CUBE_REST_Z) / (SAFE_LIFT_Z - CUBE_REST_Z))),
            ],
            dtype=float,
        )  # 2

        obs = np.concatenate(
            [
                q,  # 6
                qd,  # 6
                ee_obs,  # 3
                ee_vel,  # 3
                obj_obs,  # 3
                quat_unit,  # 4
                obj_vel,  # 3
                obj_ang,  # 3
                target_obs,  # 3
                left_rel,  # 3
                right_rel,  # 3
                gaps,  # 2
                setpoint_err,  # 3
                forces_obs,  # 2
                contact_flags,  # 2
                table_loads,  # 2
                self.last_action,  # 4
                quality_hold,  # 3
                lift_flags,  # 2
                press_obs,  # 3
                release_obs,  # 3
                reward_flags,  # 2
                tail,  # 2
            ]
        ).astype(np.float32)
        if obs.shape != (PICK_PLACE_NUM_OBS,):
            raise RuntimeError(f"pick-place observation contract mismatch: {obs.shape}")
        return np.clip(obs, -10.0, 10.0)

    def render(self):
        """Sync the passive viewer when ``render_mode='human'``."""
        if self.render_mode == "human":
            self._ensure_viewer()
            self._viewer.sync()
        return None

    def close(self) -> None:
        """Close the viewer if one is open."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def _ensure_viewer(self) -> None:
        if self._viewer is None:
            from ..viewer import PassiveViewer

            self._viewer = PassiveViewer(self.model, self.data)


class PickPlaceExpert:
    """Independent closed-loop scripted policy for physics checks; never used for PPO training."""

    def __init__(self, env: PickPlaceEnv) -> None:
        self.env = env
        self.stage = 0
        self.steps = 0
        self.episode_id = -1

    def __call__(self, _obs: np.ndarray | None = None) -> np.ndarray:
        env = self.env
        if self.episode_id != env.episode_id:
            self.stage = 0
            self.steps = 0
            self.episode_id = env.episode_id

        ee = env.tcp_position()
        ee_vel = env.tcp_velocity()
        obj = env.cube_position()
        left, right = env.pad_centers()
        center = 0.5 * (left + right)
        height = max(0.0, cube_clearance(obj, env.cube_quat(), env.cube_half_m))

        # Open fingers have pad centers ~20 mm above fingertip bottoms; leave fingertip clearance above table
        goal = ee + (obj - center)
        goal[2] += 0.009 + (0.050 if self.stage == 0 else 0.0)
        goal[2] = max(goal[2], _PRE_LIFT_MIN_TCP_Z)

        aligned = (float(np.linalg.norm(goal - env.ee_setpoint)) < 0.0025) and (
            float(np.linalg.norm(ee_vel)) < 0.025
        )
        target_error = float(np.linalg.norm(obj[:2] - env.target_xy))
        advance = (
            (self.stage <= 1 and aligned and self.steps > 5)
            or (self.stage == 2 and env.metrics.contact_steps >= 6 and self.steps >= 16)
            or (self.stage == 3 and obj[2] >= 0.090)
            or (self.stage == 4 and target_error < 0.010 and float(np.linalg.norm(ee_vel[:2])) < 0.04)
            or (self.stage == 5 and (env.metrics.touchdown_seen or height <= 0.0018))
        )
        if advance:
            self.stage += 1
            self.steps = 0
        else:
            self.steps += 1

        action = np.zeros(PICK_PLACE_NUM_ACTIONS, dtype=np.float32)
        if self.stage <= 1:
            action[:3] = (goal - env.ee_setpoint) / env.action_scale
        elif self.stage == 2:
            action[3] = -1.0
        elif self.stage == 3:
            action[2] = 0.90
        elif self.stage == 4:
            xy_goal = ee[:2] + (env.target_xy - obj[:2])
            action[:2] = np.clip((xy_goal - env.ee_setpoint[:2]) / env.action_scale, -1.0, 1.0)
        elif self.stage == 5:
            xy_goal = ee[:2] + (env.target_xy - obj[:2])
            action[:2] = np.clip((xy_goal - env.ee_setpoint[:2]) / env.action_scale, -0.5, 0.5)
            speed = 0.20 if height > 0.020 else 0.015
            descent = -min(max(0.0, height - 0.0010), speed * env.ctrl_dt) / env.action_scale
            action[2] = float(descent)
        else:
            action[3] = 1.0
        return np.clip(action, -1.0, 1.0)
