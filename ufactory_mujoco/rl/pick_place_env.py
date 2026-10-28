"""Gymnasium environment, strict metrics, and scripted expert for xArm6 + Gripper G2
tabletop cube pick-and-place (Batch 8).

Reuses Batch 2-3 assets and interfaces:
- Scene: ``models/xarm6_with_gripper/scene_pick_place.xml`` via ``configs/pick_place/xarm6.yaml``
- Gripper G2 gap and contact checks: ``ufactory_mujoco.grippers``
- Kinematics/IK: ``ufactory_mujoco.ik.solve_ik`` and ``ufactory_mujoco.task_geometry``

Requires only ``mujoco`` + ``numpy`` at runtime; ``gymnasium`` is optional
(fallback base class for ``py314`` and baseline rollouts), and ``stable-baselines3``
is required only for PPO training/checkpoint loading.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..config import load_task_config
from ..grippers import gap_to_position, identify_gripper, position_to_gap
from ..ik import solve_ik
from ..resource_paths import CONFIGS_ROOT
from ..task_geometry import GRIP_DOWN_QUAT
from .artifacts import (
    PICK_PLACE_ACTION_CONTRACT,
    PICK_PLACE_ENV_VERSION,
    PICK_PLACE_NUM_ACTIONS,
    PICK_PLACE_NUM_OBS,
)
from .reach_env import _BoxStub, _ENV_BASE, gym, load_reach_model, spaces

TCP_SITE = "attachment_site"

# Default task and physical constants aligned with configs/pick_place/xarm6.yaml
SPAWN_X = 0.30
SPAWN_Y = 0.00
TARGET_X = 0.30
TARGET_Y = 0.30
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

# Local coordinates of the inner fingertip pad centers on left_finger / right_finger
_LEFT_PAD_LOCAL = np.array([0.0, -0.02568, 0.0567], dtype=float)
_RIGHT_PAD_LOCAL = np.array([0.0, 0.02568, 0.0567], dtype=float)


def cube_clearance(pos: np.ndarray, quat: np.ndarray, cube_half_m: float = CUBE_HALF_M) -> float:
    """Lowest cube point above the z=0 tabletop, using a WXYZ quaternion."""
    pos_arr = np.asarray(pos, dtype=float).reshape(-1)
    q = np.asarray(quat, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm > 0.0:
        q = q / norm
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    row_z = np.array(
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
        dtype=float,
    )
    return float(pos_arr[2] - float(cube_half_m) * float(np.sum(np.abs(row_z))))


def grasp_distance(pad_center: np.ndarray, obj_pos: np.ndarray, cube_half_m: float = CUBE_HALF_M) -> float:
    """Distance from fingertip pad midpoint to the cube's graspable vertical band."""
    rel = np.asarray(pad_center, dtype=float).reshape(3) - np.asarray(obj_pos, dtype=float).reshape(3)
    band = 0.8 * float(cube_half_m)
    z_excess = max(0.0, abs(float(rel[2])) - band)
    return float(np.linalg.norm([rel[0], rel[1], z_excess]))


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
        ee_z: float = 0.128,
    ) -> None:
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
        q_norm = np.linalg.norm(quat)
        q = quat / q_norm if q_norm > 0.0 else np.array([1.0, 0.0, 0.0, 0.0])
        up_z = float(np.clip(1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]), -1.0, 1.0))
        tilt = float(np.rad2deg(np.arccos(up_z)))

        on_table = self.clearance < 0.015
        before_lift = (not self.ever_carried) and on_table
        drag = float(np.linalg.norm(pos[:2] - self.spawn_xy))
        if before_lift:
            self.max_drag = max(self.max_drag, drag)
            self.max_tilt_deg = max(self.max_tilt_deg, tilt)
        quality_ok = (self.max_drag <= self.drag_limit) and (self.max_tilt_deg <= self.tilt_limit)

        for i in range(2):
            if forces[i] > CONTACT_FORCE_N:
                self.finger_contact[i] = True
            elif forces[i] < RELEASE_FORCE_N:
                self.finger_contact[i] = False
        touching = bool(np.all(self.finger_contact))
        any_contact = bool(np.any(forces >= RELEASE_FORCE_N))

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

        both_clear = bool(np.all(forces < RELEASE_FORCE_N))
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

        self.target_error = float(np.linalg.norm(pos[:2] - self.target_xy))
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
    """Return ``(finger_forces[2], table_support_force_n, robot_table_force_n)``."""
    f_left = 0.0
    f_right = 0.0
    table_force = 0.0
    robot_table_force = 0.0
    f6 = np.zeros(6, dtype=float)

    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.dist > 1e-4:
            continue
        g1, g2 = int(contact.geom1), int(contact.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        mujoco.mj_contactForce(model, data, i, f6)
        force_mag = float(np.linalg.norm(f6[:3]))
        if not math.isfinite(force_mag):
            continue
        pair = {b1, b2}
        if pair == {cube_body_id, left_finger_id}:
            f_left += force_mag
        elif pair == {cube_body_id, right_finger_id}:
            f_right += force_mag
        elif g1 == floor_geom_id or g2 == floor_geom_id:
            other_body = b2 if g1 == floor_geom_id else b1
            if other_body == cube_body_id:
                table_force += force_mag
            elif other_body not in (0, base_body_id):
                robot_table_force += force_mag

    return np.array([f_left, f_right], dtype=float), table_force, robot_table_force


class PickPlaceEnv(_ENV_BASE):
    """xArm6 + Gripper G2 tabletop pick-and-place Gymnasium environment."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        robot: str = "xarm6",
        config_path: str | Path | None = None,
        render_mode: str | None = None,
        max_steps: int = 380,
        decimation: int = 10,
        action_scale: float = 0.005,
        action_clip: float = 1.0,
        gap_action_scale: float = 0.003,
        setpoint_leash_m: float = 0.025,
        pre_grasp_gap_m: float = 0.050,
        close_floor_gap_m: float = 0.018,
        workspace_lower: tuple[float, float, float] | list[float] = (0.10, -0.35, -0.02),
        workspace_upper: tuple[float, float, float] | list[float] = (0.65, 0.35, 0.55),
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
        ):
            if not np.isfinite(val) or val <= 0:
                raise ValueError(f"{name} must be finite and positive")

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
        if lite_visuals is None:
            lite_visuals = render_mode is None
        self.model = load_reach_model(scene_path, lite_visuals=lite_visuals)
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)

        self._gripper_spec = identify_gripper(self.model)
        if self._gripper_spec is None or self._gripper_spec.name != "gripper_g2":
            raise ValueError("scene must mount gripper_g2")
        self.open_gap_m = float(self._gripper_spec.open_gap_m)

        self._n_arm = 6
        self._joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i + 1}") for i in range(self._n_arm)]
        )
        if (self._joint_ids < 0).any():
            raise ValueError("scene must define joints joint1..joint6")
        self._act_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act{i + 1}") for i in range(self._n_arm)]
        )
        self._gripper_act_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper"))
        self._gripper_ctrl_max = float(self.model.actuator(self._gripper_act_id).ctrlrange[1])
        self._driver_joint_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint"))
        self._driver_qpos_adr = int(self.model.jnt_qposadr[self._driver_joint_id])
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

        self._qpos_adr = self.model.jnt_qposadr[self._joint_ids]
        self._qvel_adr = self.model.jnt_dofadr[self._joint_ids]
        self._q_lo = self.model.jnt_range[self._joint_ids, 0].copy()
        self._q_hi = self.model.jnt_range[self._joint_ids, 1].copy()
        self._q_mid = 0.5 * (self._q_lo + self._q_hi)
        self._q_half = 0.5 * (self._q_hi - self._q_lo)

        self._site_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE))
        self._cube_body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube"))
        self._cube_joint_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube"))
        self._cube_qpos_adr = int(self.model.jnt_qposadr[self._cube_joint_id])
        self._cube_dof_adr = int(self.model.jnt_dofadr[self._cube_joint_id])
        self._left_finger_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"))
        self._right_finger_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"))
        self._base_body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link_base"))
        self._floor_geom_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"))

        self.ctrl_dt = self._decimation * float(self.model.opt.timestep)
        self._home_q, self._heights = self._compute_home_pose(pp)
        self.home_tcp = np.array([self.default_spawn_xy[0], self.default_spawn_xy[1], self._heights["lift_z"]])

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
        self._step_count = 0
        self.episode_id = -1
        self._phase = 0
        self._phase_steps = 0
        self.ee_setpoint = self.home_tcp.copy()
        self.q_target = self._home_q.copy()
        self.gap_cmd = self.pre_grasp_gap_m
        self.forces = np.zeros(2, dtype=float)
        self.table_force = 0.0
        self.robot_table_force = 0.0
        self.last_action = np.zeros(PICK_PLACE_NUM_ACTIONS, dtype=np.float32)
        self.contact_rewarded = False
        self.place_rewarded = False
        self.ik_rejects = 0
        self.ep_return = 0.0
        self._prev_tcp_pos = self.home_tcp.copy()
        self._viewer = None

        obs_low = np.full(PICK_PLACE_NUM_OBS, -10.0, dtype=np.float32)
        obs_high = np.full(PICK_PLACE_NUM_OBS, 10.0, dtype=np.float32)
        if spaces is not None:
            self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (PICK_PLACE_NUM_ACTIONS,), dtype=np.float32)
        else:
            self.observation_space = _BoxStub(obs_low, obs_high, (PICK_PLACE_NUM_OBS,), np.float32)
            self.action_space = _BoxStub(-1.0, 1.0, (PICK_PLACE_NUM_ACTIONS,), np.float32)

    def _compute_home_pose(self, pp) -> tuple[np.ndarray, dict[str, float]]:
        """Solve the deployment pose above the cube using flange-to-TCP offset."""
        mujoco.mj_resetDataKeyframe(self.model, self. _ik_data, 0)
        mujoco.mj_forward(self.model, self._ik_data)
        cx, cy = float(pp.cube_xy[0]), float(pp.cube_xy[1])
        q_seed, ok = solve_ik(
            self.model,
            self._ik_data,
            TCP_SITE,
            np.array([cx, cy, float(pp.lift_z_m)]),
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
            "grasp_z": float(pp.flange_grasp_z_m) + off,
            "pre_grasp_z": float(pp.flange_grasp_z_m) + float(pp.approach_m) + off,
            "lift_z": float(pp.lift_z_m) + off,
        }
        q_home, ok = solve_ik(
            self.model,
            self._ik_data,
            TCP_SITE,
            np.array([cx, cy, heights["lift_z"]]),
            GRIP_DOWN_QUAT,
            q_init=q_seed,
        )
        if not ok:
            raise RuntimeError("IK failed while solving home deployment pose")
        return q_home[: self._n_arm].copy(), heights

    # ------------------------------------------------------------------ state
    @property
    def n_joints(self) -> int:
        return self._n_arm

    @property
    def step_count(self) -> int:
        return self._step_count

    def tcp_position(self) -> np.ndarray:
        return self.data.site_xpos[self._site_id].copy()

    def tcp_velocity(self) -> np.ndarray:
        return (self.tcp_position() - self._prev_tcp_pos) / self.ctrl_dt

    def cube_position(self) -> np.ndarray:
        return self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3].copy()

    def cube_quat(self) -> np.ndarray:
        return self.data.qpos[self._cube_qpos_adr + 3 : self._cube_qpos_adr + 7].copy()

    def cube_velocity(self) -> np.ndarray:
        return self.data.qvel[self._cube_dof_adr : self._cube_dof_adr + 3].copy()

    def pad_centers(self) -> tuple[np.ndarray, np.ndarray]:
        l_pos = self.data.body(self._left_finger_id).xpos
        l_mat = self.data.body(self._left_finger_id).xmat.reshape(3, 3)
        r_pos = self.data.body(self._right_finger_id).xpos
        r_mat = self.data.body(self._right_finger_id).xmat.reshape(3, 3)
        return l_pos + l_mat @ _LEFT_PAD_LOCAL, r_pos + r_mat @ _RIGHT_PAD_LOCAL

    def measured_gap(self) -> float:
        q_drv = float(self.data.qpos[self._driver_qpos_adr])
        norm_pos = float(np.clip(q_drv / max(self._gripper_ctrl_max, 1e-9), 0.0, 1.0))
        return float(position_to_gap(self._gripper_spec, norm_pos))

    def _gap_to_ctrl(self, gap_m: float) -> float:
        pos = gap_to_position(self._gripper_spec, float(np.clip(gap_m, 0.0, self.open_gap_m)))
        return float(pos * self._gripper_ctrl_max)

    def _solve_ik(self, target_pos: np.ndarray) -> tuple[np.ndarray, bool]:
        q_init = self.data.qpos.copy()
        q_init[self._qpos_adr] = self.q_target
        q_sol, converged = solve_ik(
            self.model,
            self._ik_data,
            TCP_SITE,
            target_pos,
            GRIP_DOWN_QUAT,
            q_init=q_init,
            pos_tol=1e-3,
            rot_tol=1e-2,
            max_iter=20,
        )
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
        """Integrate 4D increments [dx, dy, dz, dgap]; zero actions hold commands."""
        action = np.asarray(action, dtype=float).reshape(PICK_PLACE_NUM_ACTIONS)
        moving = bool(np.any(action[:3] != 0.0))
        if moving:
            target = np.clip(
                self.ee_setpoint + action[:3] * self.action_scale,
                self.workspace_lower,
                self.workspace_upper,
            )
            ee = self.tcp_position()
            delta = target - ee
            dist = float(np.linalg.norm(delta))
            if dist > self.setpoint_leash_m:
                target = ee + delta * (self.setpoint_leash_m / max(dist, 1e-9))
            q_new, reject = self._solve_ik(target)
            if not reject:
                self.ee_setpoint = target
                self.q_target = q_new

        self.data.ctrl[self._act_ids] = self.q_target

        base_gap = self.gap_cmd
        at_release_table = self.metrics.ever_lifted_clear and (
            self.metrics.touchdown_seen or self.metrics.release_confirmed
        )
        if at_release_table and action[3] > 0.0 and self.gap_cmd < current_measured_gap:
            base_gap = current_measured_gap
        self.gap_cmd = float(
            np.clip(base_gap + action[3] * self.gap_action_scale, self.close_floor_gap_m, self.open_gap_m)
        )
        self.data.ctrl[self._gripper_act_id] = self._gap_to_ctrl(self.gap_cmd)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if gym is not None:
            super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        spawn_xy = self.default_spawn_xy.copy()
        if options is not None and "cube_xy" in options:
            spawn_xy = np.asarray(options["cube_xy"], dtype=float).reshape(2).copy()

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        self.data.qpos[self._qpos_adr] = self._home_q
        self.data.ctrl[self._act_ids] = self._home_q

        drv_ctrl = self._gap_to_ctrl(self.pre_grasp_gap_m)
        self.data.qpos[self._gripper_qpos_adrs] = drv_ctrl
        self.data.ctrl[self._gripper_act_id] = drv_ctrl

        self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3] = [
            spawn_xy[0],
            spawn_xy[1],
            CUBE_REST_Z,
        ]
        self.data.qpos[self._cube_qpos_adr + 3 : self._cube_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self.episode_id += 1
        self._phase = 0
        self._phase_steps = 0
        self.q_target = self._home_q.copy()
        self.ee_setpoint = self.tcp_position()
        self._prev_tcp_pos = self.ee_setpoint.copy()
        self.gap_cmd = self.pre_grasp_gap_m
        self.forces = np.zeros(2, dtype=float)
        self.table_force = 0.0
        self.robot_table_force = 0.0
        self.last_action = np.zeros(PICK_PLACE_NUM_ACTIONS, dtype=np.float32)
        self.contact_rewarded = False
        self.place_rewarded = False
        self.ik_rejects = 0
        self.ep_return = 0.0

        self.metrics.reset(spawn_xy=spawn_xy, ee_z=float(self.ee_setpoint[2]))
        obs = self._get_obs()
        info = self._build_info(success=False, failed=False, timeout=False)
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
        left, right = self.pad_centers()
        pad_mid = 0.5 * (left + right)
        distance = grasp_distance(pad_mid, obj, self.cube_half_m)
        pad_xy_err = float(np.linalg.norm((pad_mid - obj)[:2]))
        height = max(0.0, cube_clearance(obj, quat, self.cube_half_m))
        target_error = float(np.linalg.norm(obj[:2] - self.target_xy))
        current_gap = self.measured_gap()
        grasp_near = math.exp(-distance / 0.025)
        placed = self.place_rewarded

        min_force = float(np.min(self.forces))
        if self.metrics.ever_lifted_clear:
            grip_force = float(np.clip(min_force / 2.0, 0.0, 1.0))
        else:
            grip_force = 0.75 * float(np.clip(min_force / 2.0, 0.0, 1.0)) + 0.25 * float(
                np.clip(min_force / 20.0, 0.0, 1.0)
            )
        grip = grip_force * grasp_near * (0.0 if placed else 1.0)

        desired_gap = self.pre_grasp_gap_m - 0.025 * math.exp(-distance / 0.012)
        gap_alignment = math.exp(-abs(current_gap - desired_gap) / 0.012)
        target_near = math.exp(-target_error / 0.04)
        target_zone = math.exp(-max(0.0, target_error - 0.015) / 0.04)
        lifted = 1.0 if self.metrics.ever_lifted_clear else 0.0
        place_height_err = max(0.0, height - self.metrics.place_clearance)
        both_clear = 1.0 if bool(np.all(self.forces < 0.1)) else 0.0
        released_clear = (1.0 if self.metrics.release_confirmed else 0.0) * both_clear
        meas_open = float(np.clip((current_gap - CUBE_SIZE) / (self.pre_grasp_gap_m - CUBE_SIZE), 0.0, 1.0))
        at_table = self.metrics.ever_lifted_clear and (
            (height <= self.metrics.release_max_height)
            or self.metrics.touchdown_seen
            or self.metrics.release_confirmed
        )
        pad_xy_drift = max(0.0, pad_xy_err - 0.008)
        ee_z = float(self.tcp_position()[2])
        ref_ee_z = self.metrics.touchdown_ee_z if self.metrics.touchdown_ee_z > 0.0 else self._heights["grasp_z"]
        has_td_ref = (self.metrics.touchdown_ee_z > 0.0 or self.metrics.release_confirmed) and at_table
        ee_sink = max(0.0, ref_ee_z - ee_z - 0.0025) if has_td_ref else 0.0
        post_td_pose = math.exp(-pad_xy_drift / 0.015) * math.exp(-ee_sink / 0.0015)
        table_guard = post_td_pose * (released_clear if placed else 1.0)
        place_zone = lifted * target_zone * math.exp(-place_height_err / 0.005) * table_guard
        cmd_open = float(
            np.clip((self.gap_cmd - self.close_floor_gap_m) / (self.pre_grasp_gap_m - self.close_floor_gap_m), 0.0, 1.0)
        )
        action_open = float(np.clip(0.5 * (float(self.last_action[3]) + 1.0), 0.0, 1.0))
        open_drive = 0.85 * cmd_open + 0.15 * action_open
        release_zone = place_zone * max(open_drive, meas_open)

        approach_term = max(
            math.exp(-distance / 0.06) * (table_guard if at_table else 1.0),
            release_zone,
        )
        pregrasp = max(
            math.exp(-distance / 0.04) * gap_alignment * (table_guard if at_table else 1.0),
            release_zone,
        )
        grasp_or_place = max(
            grip * (post_td_pose if at_table else 1.0),
            place_zone,
        )
        hold_progress = float(np.clip(self.metrics.hold_steps / self.metrics.required_hold_steps, 0.0, 1.0))
        lift_progress = float(np.clip(self.metrics.max_held_z - CUBE_REST_Z, 0.0, SAFE_LIFT_Z - CUBE_REST_Z))
        s = self.reward_scales
        return float(
            float(s.get("approach_potential", 8.0)) * approach_term
            + float(s.get("pregrasp_potential", 4.0)) * pregrasp
            + float(s.get("grasp_potential", 4.0)) * grasp_or_place
            + float(s.get("lift_potential", 40.0)) * lift_progress
            + float(s.get("transit_potential", 8.0)) * lifted * (0.3 - target_error)
            + float(s.get("place_potential", 6.0)) * lifted * target_near / (1.0 + place_height_err / 0.10)
            + float(s.get("release_potential", 4.0)) * place_zone * open_drive
            + float(s.get("hold_potential", 6.0)) * (0.5 * place_zone * meas_open + 0.5 * hold_progress)
        )

    def step(self, action):
        raw_action = np.asarray(action, dtype=float).reshape(PICK_PLACE_NUM_ACTIONS)
        if not np.all(np.isfinite(raw_action)):
            raise ValueError("action must be finite")
        act_pen = self._action_penalty(raw_action)
        clipped = np.clip(raw_action, -self.action_clip, self.action_clip)

        old_potential = self._potential()
        self._prev_tcp_pos = self.tcp_position()
        self._apply_action(clipped, self.measured_gap())

        for _ in range(self._decimation):
            self.data.qfrc_applied[: self._n_arm] = self.data.qfrc_bias[: self._n_arm]
            mujoco.mj_step(self.model, self.data)
        self.data.qfrc_applied[:] = 0.0

        self._step_count += 1
        self.last_action = clipped.astype(np.float32)

        ee = self.tcp_position()
        obj = self.cube_position()
        self.forces, self.table_force, self.robot_table_force = _extract_contact_loads(
            self.model,
            self.data,
            cube_body_id=self._cube_body_id,
            left_finger_id=self._left_finger_id,
            right_finger_id=self._right_finger_id,
            floor_geom_id=self._floor_geom_id,
            base_body_id=self._base_body_id,
        )
        status = self.metrics.update(
            obj,
            self.cube_quat(),
            self.forces,
            self.cube_velocity(),
            self.measured_gap(),
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
        first_contact = bool(status["stable_contact"] and (not self.contact_rewarded) and status["quality_ok"])
        self.contact_rewarded = self.contact_rewarded or first_contact
        first_place = bool(
            status["first_release"]
            and status["in_target_xy"]
            and (not self.metrics.high_release)
            and (not self.place_rewarded)
        )
        self.place_rewarded = self.place_rewarded or first_place

        done = terminated or truncated
        new_potential = 0.0 if (done and not success) else self._potential()
        reward = self.reward_gamma * new_potential - old_potential - float(s.get("step_penalty", 0.005))
        if first_contact:
            reward += float(s.get("first_contact", 2.0))
        if status["first_lift_clear"]:
            reward += float(s.get("first_lift", 6.0))
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
            ee_vel = self.tcp_velocity()
            descent_limit = 0.0 if at_release_table else self.max_near_table_descent_m_s
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
        self._update_phase()
        obs = self._get_obs()
        info = self._build_info(success=success, failed=failed, timeout=timeout, status=status)
        if self.render_mode == "human":
            self._ensure_viewer()
            self._viewer.sync()
        return obs, float(reward), terminated, truncated, info

    def _update_phase(self) -> None:
        ee = self.tcp_position()
        obj = self.cube_position()
        height = max(0.0, cube_clearance(obj, self.cube_quat(), self.cube_half_m))
        grasp_z = self._heights["grasp_z"]
        goal = np.array([obj[0], obj[1], grasp_z + (0.045 if self._phase == 0 else 0.0)])
        aligned = float(np.linalg.norm(goal - ee)) < 0.004
        target_error = float(np.linalg.norm(obj[:2] - self.target_xy))
        ee_xy_speed = float(np.linalg.norm(self.tcp_velocity()[:2]))

        advance = (
            (self._phase <= 1 and aligned and self._phase_steps > 3)
            or (self._phase == 2 and self.metrics.contact_steps >= 4 and self._phase_steps >= 12)
            or (self._phase == 3 and height >= 0.070)
            or (self._phase == 4 and target_error < 0.003 and ee_xy_speed < 0.02)
            or (self._phase == 5 and height <= 0.0018)
        )
        if advance:
            self._phase += 1
            self._phase_steps = 0
        else:
            self._phase_steps += 1

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
            "collision_failed": bool(status["collision_failed"]) if status else (m.max_robot_table_force > CONTACT_FORCE_N),
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
            "hold_steps": int(m.hold_steps),
            "steps": int(self._step_count),
            "return": float(self.ep_return),
            "ik_rejects": int(self.ik_rejects),
        }

    def _get_obs(self) -> np.ndarray:
        q = self.data.qpos[self._qpos_adr]
        q_norm = np.clip((q - self._q_mid) / self._q_half, -1.5, 1.5)
        qd = np.clip(self.data.qvel[self._qvel_adr] / 2.0, -5.0, 5.0)
        ee = self.tcp_position()
        obj = self.cube_position()
        quat = self.cube_quat()
        obj_vel = np.clip(self.cube_velocity() / 0.2, -5.0, 5.0)
        target = np.array([self.target_xy[0], self.target_xy[1], CUBE_REST_Z])
        target_rel = np.clip((target - obj) / 0.3, -3.0, 3.0)
        left, right = self.pad_centers()
        pad_mid = 0.5 * (left + right)
        pad_mid_rel = np.clip((pad_mid - obj) / 0.1, -5.0, 5.0)
        gaps = np.array(
            [
                self.gap_cmd / self.open_gap_m,
                self.measured_gap() / self.open_gap_m,
            ],
            dtype=float,
        )
        setpoint_err = np.clip((self.ee_setpoint - ee) / self.setpoint_leash_m, -2.0, 2.0)
        contact_state = np.array(
            [
                min(5.0, max(0.0, self.forces[0] / 10.0)),
                min(5.0, max(0.0, self.forces[1] / 10.0)),
                1.0 if self.metrics.finger_contact[0] else 0.0,
                1.0 if self.metrics.finger_contact[1] else 0.0,
            ],
            dtype=float,
        )
        progress = np.array(
            [
                min(1.0, float(self._phase) / 3.0),
                max(0.0, float(self._phase - 3) / 3.0),
            ],
            dtype=float,
        )
        obs = np.concatenate(
            [
                q_norm,  # 6
                qd,  # 6
                ee,  # 3
                obj,  # 3
                quat,  # 4
                obj_vel,  # 3
                target_rel,  # 3
                pad_mid_rel,  # 3
                gaps,  # 2
                setpoint_err,  # 3
                contact_state,  # 4
                self.last_action,  # 4
                progress,  # 2
            ]
        ).astype(np.float32)
        if obs.shape != (PICK_PLACE_NUM_OBS,):
            raise RuntimeError(f"pick-place observation contract mismatch: {obs.shape}")
        return np.clip(obs, -10.0, 10.0)

    def render(self):
        if self.render_mode == "human":
            self._ensure_viewer()
            self._viewer.sync()
        return None

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def _ensure_viewer(self) -> None:
        if self._viewer is None:
            from ..viewer import PassiveViewer

            self._viewer = PassiveViewer(self.model, self.data)


class PickPlaceExpert:
    """Independent closed-loop scripted expert for physics and evaluation checks."""

    def __init__(self, env: PickPlaceEnv) -> None:
        self.env = env
        self.stage = 0
        self.steps = 0
        self.episode_id = -1

    def __call__(self, _obs: np.ndarray | None = None) -> np.ndarray:
        env = self.env
        self.stage = env._phase
        self.steps = env._phase_steps
        self.episode_id = env.episode_id

        ee = env.tcp_position()
        obj = env.cube_position()
        height = max(0.0, cube_clearance(obj, env.cube_quat(), env.cube_half_m))
        grasp_z = env._heights["grasp_z"]
        goal = np.array([obj[0], obj[1], grasp_z + (0.045 if self.stage == 0 else 0.0)])

        action = np.zeros(PICK_PLACE_NUM_ACTIONS, dtype=np.float32)
        if self.stage <= 1:
            action[:3] = (goal - env.ee_setpoint) / env.action_scale
        elif self.stage == 2:
            action[3] = -1.0
        elif self.stage == 3:
            action[2] = 0.8
        elif self.stage == 4:
            xy_goal = ee[:2] + (env.target_xy - obj[:2])
            action[:2] = np.clip((xy_goal - env.ee_setpoint[:2]) / env.action_scale, -0.8, 0.8)
        elif self.stage == 5:
            xy_goal = ee[:2] + (env.target_xy - obj[:2])
            action[:2] = np.clip((xy_goal - env.ee_setpoint[:2]) / env.action_scale, -0.5, 0.5)
            speed = 0.10 if height > 0.025 else 0.015
            descent = -min(max(0.0, height - 0.0015), speed * env.ctrl_dt) / env.action_scale
            action[2] = float(descent)
        else:
            action[3] = 1.0
        return np.clip(action, -1.0, 1.0)


XArm6PickPlaceEnv = PickPlaceEnv
