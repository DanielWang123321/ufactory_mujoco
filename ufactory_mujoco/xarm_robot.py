"""Real xArm hardware backend implementing RobotInterface with a fail-closed safety gate.

By default, XArmRobot opens a strictly read-only session:
  - Observations (get_joint_positions, get_tcp_pose, get_joint_states, read_state)
    only query controller telemetry and never enable motors or change modes.
  - Any motion or gripper command raises SafetyGateError unless the session was
    bound to a verified KinematicsCalibration AND unlocked with an unforgeable
    MotionApproval token (requiring preflight PASS + --confirm-real + cycles == 1).
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import mujoco
import numpy as np

from .bio_g2_model import (
    MIN_SPEED_RPM as BIO_G2_MIN_SPEED_RPM,
    bio_g2_speed_mm_s_from_percent,
    speed_rpm_from_mm_s as bio_g2_speed_rpm_from_mm_s,
)
from .calibration import (
    CalibrationError,
    KinematicsCalibration,
    apply_calibration_to_model,
    canonical_robot_key,
    get_robot_sn,
    read_tcp_offset,
    tcp_mm_rpy_to_flange_m_quat,
)
from .g2_model import checked, sdk_position_counts, sdk_speed_rpm, speed_mm_s
from .grippers import (
    GRIPPER_ACTUATOR_NAMES,
    GripperSpec,
    gap_to_position,
    identify_gripper,
    position_to_gap,
)
from .ik import solve_ik
from .robot_interface import RobotInterface
from .robots import ARMS, resolve_arm_scene
from .safety import (
    MotionApproval,
    REPORT_STATE_STOPPING,
    SafetyGateError,
    assert_controller_healthy,
    read_arm_joint_positions,
    verify_hardware_identity_and_calibration,
)
from .trajectory import JointTrajectory, plan_cartesian_line, plan_ptp_quintic


def _connect_xarm_sdk(ip: str) -> Any:
    try:
        from xarm.wrapper import XArmAPI
    except ImportError as exc:
        raise SafetyGateError(
            "xArm-Python-SDK ('xarm') is required to connect to a real robot"
        ) from exc
    arm = XArmAPI(ip, is_radian=True)
    if not bool(getattr(arm, "connected", False)):
        connect_fn = getattr(arm, "connect", None)
        if callable(connect_fn):
            connect_fn()
    if not bool(getattr(arm, "connected", False)):
        raise SafetyGateError(f"cannot connect to xArm controller at {ip}")
    time.sleep(0.35)
    return arm


class XArmRobot(RobotInterface):
    """Real xArm hardware adapter implementing ``RobotInterface``."""

    def __init__(
        self,
        ip: str = "192.168.1.60",
        *,
        robot: str = "xarm6",
        gripper: str | None = None,
        scene: str = "scene",
        model_path: str | Path | None = None,
        calibration: KinematicsCalibration | str | Path | None = None,
        read_only: bool = True,
        approval: MotionApproval | None = None,
        arm_api: Any | None = None,
    ) -> None:
        self.ip = str(ip)
        self.robot_key = canonical_robot_key(robot)
        self.gripper_name = gripper
        self.read_only = bool(read_only)
        self._approval: MotionApproval | None = None
        self._calibration: KinematicsCalibration | None = None
        self._owns_connection = arm_api is None
        self._arm = _connect_xarm_sdk(self.ip) if arm_api is None else arm_api
        self._motion_prepared_mode: int | None = None

        try:
            if not self.read_only and calibration is None:
                raise SafetyGateError(
                    "non-read-only XArmRobot session requires --calibration YAML"
                )
            if calibration is not None:
                sn, calib = verify_hardware_identity_and_calibration(
                    self._arm, robot=self.robot_key, calibration_path=calibration
                )
                self.serial_number = sn
                self._calibration = calib
            else:
                assert_controller_healthy(self._arm)
                try:
                    self.serial_number = get_robot_sn(self._arm)
                except CalibrationError as exc:
                    raise SafetyGateError(str(exc)) from exc

            resolved_xml = (
                Path(model_path)
                if model_path is not None
                else resolve_arm_scene(self.robot_key, gripper, scene)
            )
            self.model = mujoco.MjModel.from_xml_path(str(resolved_xml))
            self.data = mujoco.MjData(self.model)
            self.eval_data = mujoco.MjData(self.model)
            if self._calibration is not None:
                apply_calibration_to_model(self.model, self._calibration, self.data)
                mujoco.mj_setConst(self.model, self.eval_data)
                mujoco.mj_forward(self.model, self.eval_data)

            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
            if site_id < 0:
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "link_tcp")
            self._site = site_id

            self._n_arm_qpos = ARMS[self.robot_key].dof
            self._gripper_actuator_id: int | None = None
            for i in range(self.model.nu):
                if self.model.actuator(i).name in GRIPPER_ACTUATOR_NAMES:
                    self._gripper_actuator_id = i
                    break
            self._gripper_spec: GripperSpec | None = identify_gripper(self.model)
            self._commanded_gripper_pos: float = 0.0

            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            q_init = read_arm_joint_positions(self._arm, self._n_arm_qpos)
            self.data.qpos[: self._n_arm_qpos] = q_init
            self.eval_data.qpos[: self._n_arm_qpos] = q_init
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_forward(self.model, self.eval_data)

            if approval is not None:
                self.unlock_motion(approval)
            elif not self.read_only:
                raise SafetyGateError(
                    "non-read-only XArmRobot session requires a valid MotionApproval token"
                )
        except Exception:
            if self._owns_connection:
                try:
                    self._arm.disconnect()
                except Exception:
                    pass
            raise

    # -- properties & gate --------------------------------------------------
    @property
    def n_arm(self) -> int:
        return self._n_arm_qpos

    @property
    def calibration(self) -> KinematicsCalibration | None:
        """Kinematic calibration bound to this session, if any."""
        return self._calibration

    @property
    def motion_unlocked(self) -> bool:
        """True only when the session is writable and holds a verified ``MotionApproval``."""
        return (not self.read_only) and (self._approval is not None)

    @property
    def tcp_offset(self) -> np.ndarray:
        """Controller TCP offset as reported by the SDK."""
        return read_tcp_offset(self._arm)

    def unlock_motion(self, approval: MotionApproval) -> None:
        """Unlock motion on this session after verifying the MotionApproval token."""
        if not isinstance(approval, MotionApproval):
            raise SafetyGateError("unlock_motion requires a valid MotionApproval token")
        if self._calibration is None:
            raise SafetyGateError("cannot unlock motion on an uncalibrated XArmRobot session")
        if approval.robot != self.robot_key:
            raise SafetyGateError(
                f"approval robot {approval.robot!r} does not match session {self.robot_key!r}"
            )
        if approval.serial_number != self.serial_number:
            raise SafetyGateError(
                f"approval serial_number {approval.serial_number!r} != {self.serial_number!r}"
            )
        if approval.calibration_sha256 != self._calibration.sha256:
            raise SafetyGateError("approval calibration_sha256 does not match loaded calibration")
        if approval.cycles != 1:
            raise SafetyGateError(f"approval cycles must be 1, got {approval.cycles}")
        assert_controller_healthy(self._arm)
        self._approval = approval
        self.read_only = False

    def _require_motion_unlocked(self, action: str) -> None:
        if self.read_only or self._approval is None:
            raise SafetyGateError(
                f"{action} refused: session is read-only or lacks MotionApproval (--confirm-real)"
            )
        assert_controller_healthy(self._arm)

    def _prepare_mode(self, mode: int) -> None:
        self._require_motion_unlocked(f"set_mode({mode})")
        if self._motion_prepared_mode == mode:
            return
        for label, fn in (
            ("motion_enable", lambda: self._arm.motion_enable(enable=True)),
            (f"set_mode({mode})", lambda: self._arm.set_mode(mode)),
            ("set_state(0)", lambda: self._arm.set_state(0)),
        ):
            code = int(fn())
            if code != 0:
                self._arm.set_state(REPORT_STATE_STOPPING)
                raise SafetyGateError(f"{label} failed with SDK code {code}; controller stopped")
        self._motion_prepared_mode = mode

    def _require_arm_q(self, q: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(q, dtype=np.float64)
        if arr.shape != (self.n_arm,) or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be a finite array of shape ({self.n_arm},), got {arr.shape}")
        return arr

    def _tcp_site_name(self) -> str:
        return self.model.site(self._site).name

    # -- read-only observations ---------------------------------------------
    def get_joint_positions(self) -> np.ndarray:
        """Read current arm joint positions in radians without issuing motion commands."""
        q = read_arm_joint_positions(self._arm, self.n_arm)
        self.data.qpos[: self.n_arm] = q
        mujoco.mj_forward(self.model, self.data)
        return q

    def get_joint_states(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read ``(q_rad, qd_rad_s, tau_nm)`` of shape ``(n_arm,)``."""
        code, states = self._arm.get_joint_states(is_radian=True)
        if int(code) != 0:
            raise SafetyGateError(f"get_joint_states failed with SDK code {code}")
        pos, vel, _eff = states
        code_t, tau = self._arm.get_joints_torque()
        if int(code_t) != 0:
            raise SafetyGateError(f"get_joints_torque failed with SDK code {code_t}")
        q = np.asarray(pos[: self.n_arm], dtype=np.float64)
        qd = np.asarray(vel[: self.n_arm], dtype=np.float64)
        torques = np.asarray(tau[: self.n_arm], dtype=np.float64)
        self.data.qpos[: self.n_arm] = q
        mujoco.mj_forward(self.model, self.data)
        return q, qd, torques

    def get_tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return calibrated MuJoCo site pose ``(pos_m (3,), quat_wxyz (4,))`` at current joints."""
        self.get_joint_positions()
        pos = self.data.site_xpos[self._site].copy()
        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self._site])
        return pos, quat

    def get_controller_tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Read controller ``get_position(is_radian=True)`` converted to flange ``(pos_m, quat_wxyz)``."""
        code, pose = self._arm.get_position(is_radian=True)
        if int(code) != 0:
            raise SafetyGateError(f"get_position failed with SDK code {code}")
        return tcp_mm_rpy_to_flange_m_quat(pose[:6], self.tcp_offset)

    def read_state(self) -> dict[str, Any]:
        """Read a structured, side-effect-free telemetry snapshot."""
        q = self.get_joint_positions()
        err, warn = 0, 0
        getter = getattr(self._arm, "get_err_warn_code", None)
        if callable(getter):
            res = getter()
            if isinstance(res, tuple) and len(res) == 2 and int(res[0]) == 0:
                err, warn = int(res[1][0]), int(res[1][1])
        state_val = int(getattr(self._arm, "state", 0) or 0)
        state_getter = getattr(self._arm, "get_state", None)
        if callable(state_getter):
            res = state_getter()
            if isinstance(res, tuple) and len(res) == 2 and int(res[0]) == 0:
                state_val = int(res[1])
        load = getattr(self._arm, "tcp_load", [0.0, [0.0, 0.0, 0.0]])
        return {
            "robot": self.robot_key,
            "serial_number": self.serial_number,
            "connected": bool(getattr(self._arm, "connected", False)),
            "read_only": self.read_only,
            "motion_unlocked": self.motion_unlocked,
            "calibration_sha256": None if self._calibration is None else self._calibration.sha256,
            "q_rad": q.tolist(),
            "tcp_offset": self.tcp_offset.tolist(),
            "tcp_load": load,
            "state": state_val,
            "mode": int(getattr(self._arm, "mode", 0) or 0),
            "error_code": err,
            "warn_code": warn,
        }

    # -- motion commands (gated) --------------------------------------------
    def set_joint_positions(self, q: np.ndarray) -> None:
        self._require_motion_unlocked("set_joint_positions")
        q_target = self._require_arm_q(q, "q")
        self._prepare_mode(1)
        code = int(
            self._arm.set_servo_angle_j(
                angles=q_target.tolist(),
                speed=0,
                mvacc=0,
                mvtime=0,
                is_radian=True,
            )
        )
        if code != 0:
            self._arm.set_state(REPORT_STATE_STOPPING)
            raise SafetyGateError(f"set_servo_angle_j failed with SDK code {code}")
        self.data.qpos[: self.n_arm] = q_target
        mujoco.mj_forward(self.model, self.data)

    def move_joints(self, q_goal: np.ndarray, duration: float, wait: bool = True) -> None:
        self._require_motion_unlocked("move_joints")
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive")
        q_target = self._require_arm_q(q_goal, "q_goal")
        q_now = self.get_joint_positions()
        disp = float(np.max(np.abs(q_target - q_now)))
        speed = float(np.clip(max(disp / duration * 1.5, 0.1), 0.05, 1.0))
        self._prepare_mode(0)
        code = int(
            self._arm.set_servo_angle(
                angle=q_target.tolist(),
                speed=speed,
                mvacc=2.0,
                wait=bool(wait),
                is_radian=True,
            )
        )
        if code != 0:
            self._arm.set_state(REPORT_STATE_STOPPING)
            raise SafetyGateError(f"set_servo_angle failed with SDK code {code}")
        self.get_joint_positions()

    def move_tcp(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        duration: float,
        wait: bool = True,
    ) -> None:
        self._require_motion_unlocked("move_tcp")
        q_sol, ok = self.solve_tcp(pos, quat, q_init=self.data.qpos.copy())
        if not ok:
            raise RuntimeError(f"IK did not converge for target pos={pos} quat={quat}")
        self.move_joints(q_sol[: self.n_arm], duration=duration, wait=wait)

    def execute_trajectory(self, traj: JointTrajectory, wait: bool = True) -> None:
        self._require_motion_unlocked("execute_trajectory")
        if traj.n_joints != self.n_arm:
            raise ValueError(f"trajectory n_joints must be {self.n_arm}, got {traj.n_joints}")
        if traj.n_steps == 0:
            return
        self._prepare_mode(1)
        dt = max(traj.dt, 0.004)
        t_end = float(traj.times[-1])
        cur_t = float(traj.times[0])
        while cur_t <= t_end + 0.5 * dt:
            q_cmd, _, _ = traj.sample(cur_t)
            code = int(
                self._arm.set_servo_angle_j(
                    angles=q_cmd.tolist(),
                    speed=0,
                    mvacc=0,
                    mvtime=0,
                    is_radian=True,
                )
            )
            if code != 0:
                self._arm.set_state(REPORT_STATE_STOPPING)
                raise SafetyGateError(f"set_servo_angle_j failed with SDK code {code}")
            time.sleep(dt)
            cur_t += dt
        if wait:
            self.settle(0.2)
        self.get_joint_positions()

    # -- planning helpers ---------------------------------------------------
    def solve_tcp(self, pos, quat=None, q_init=None):
        """Solve IK for a TCP pose on the local (calibrated) model; returns ``(q, converged)``."""
        seed = self.data.qpos.copy() if q_init is None else np.asarray(q_init, dtype=float)
        return solve_ik(
            self.model,
            self.eval_data,
            self._tcp_site_name(),
            np.asarray(pos, dtype=float),
            None if quat is None else np.asarray(quat, dtype=float),
            q_init=seed,
        )

    def plan_joint_trajectory(
        self,
        q_goal: np.ndarray,
        max_vel: float | np.ndarray = 1.57,
        max_acc: float | np.ndarray = 4.0,
        duration: float | None = None,
    ) -> JointTrajectory:
        """Plan a quintic joint move from the current measured joints."""
        q0 = self.get_joint_positions()
        return plan_ptp_quintic(
            q0, q_goal, max_vel=max_vel, max_acc=max_acc, duration=duration, dt=self.model.opt.timestep
        )

    def plan_cartesian_trajectory(
        self,
        pos: np.ndarray,
        quat: np.ndarray | None = None,
        max_lin_vel: float = 0.20,
        max_lin_acc: float = 1.0,
        site_name: str | None = None,
    ) -> tuple[JointTrajectory, bool]:
        """Plan a straight TCP line from the current pose; returns ``(trajectory, ok)``."""
        site = site_name if site_name is not None else self._tcp_site_name()
        p0, quat0 = self.get_tcp_pose()
        return plan_cartesian_line(
            p0,
            quat0,
            pos,
            quat,
            self.model,
            site_name=site,
            max_lin_vel=max_lin_vel,
            max_lin_acc=max_lin_acc,
            q_init=self.data.qpos.copy(),
            dt=self.model.opt.timestep,
            eval_data=self.eval_data,
        )

    # -- gripper control (gated) --------------------------------------------
    @property
    def has_gripper(self) -> bool:
        return self._gripper_actuator_id is not None

    @property
    def gripper_kind(self) -> str:
        if self._gripper_spec is None:
            return "none"
        return self._gripper_spec.kind

    def _require_jaw(self) -> GripperSpec:
        if self.gripper_kind == "vacuum":
            raise NotImplementedError("vacuum gripper has no jaw gap")
        if self._gripper_spec is None or self._gripper_spec.kind != "jaw":
            raise NotImplementedError("This robot has no jaw gripper")
        return self._gripper_spec

    def get_gripper_position(self) -> float:
        if not self.has_gripper:
            raise NotImplementedError("This robot has no gripper")
        if self._gripper_spec is not None and self._gripper_spec.name == "bio_gripper_g2":
            bio_getter = getattr(self._arm, "get_bio_gripper_g2_position", None)
            if callable(bio_getter):
                code, gap_mm = bio_getter()
                if int(code) == 0 and gap_mm is not None:
                    gap_m = float(np.clip(
                        float(gap_mm) / 1000.0,
                        self._gripper_spec.closed_gap_m,
                        self._gripper_spec.open_gap_m,
                    ))
                    return gap_to_position(self._gripper_spec, gap_m)
            return self._commanded_gripper_pos
        if self._gripper_spec is not None and self._gripper_spec.name == "lite6_gripper":
            return self._commanded_gripper_pos
        getter = getattr(self._arm, "get_gripper_position", None)
        if callable(getter):
            code, pos = getter()
            if int(code) == 0 and pos is not None:
                # xArm Gripper G2 SDK scale: 850 (open) .. 0 (closed)
                return float(np.clip(1.0 - float(pos) / 850.0, 0.0, 1.0))
        return self._commanded_gripper_pos

    def set_gripper(
        self,
        pos: float,
        *,
        speed_percent: float | None = None,
        force_percent: float | None = None,
    ) -> None:
        self._require_motion_unlocked("set_gripper")
        if not self.has_gripper:
            raise NotImplementedError("This robot has no gripper")
        pos_clamped = float(np.clip(pos, 0.0, 1.0))

        if self._gripper_spec is not None and self._gripper_spec.name == "bio_gripper_g2":
            err_fn = getattr(self._arm, "get_bio_gripper_error", None)
            if callable(err_fn):
                code, err = err_fn()
                if int(code) != 0 or int(err) != 0:
                    raise SafetyGateError(f"Bio Gripper G2 error check failed (code={code}, err={err})")
            bio_enable = getattr(self._arm, "set_bio_gripper_enable", None)
            if callable(bio_enable):
                code = int(bio_enable(True))
                if code != 0:
                    raise SafetyGateError(f"set_bio_gripper_enable failed with SDK code {code}")
            gap_mm = float(position_to_gap(self._gripper_spec, pos_clamped) * 1000.0)
            spd_pct = 50.0 if speed_percent is None else checked(speed_percent, 0.0, 100.0, "speed_percent")
            frc_pct = 50.0 if force_percent is None else checked(force_percent, 1.0, 100.0, "force_percent")
            spd_mm = bio_g2_speed_mm_s_from_percent(spd_pct)
            spd_rpm = (
                int(round(BIO_G2_MIN_SPEED_RPM))
                if spd_mm <= 0.0
                else int(round(bio_g2_speed_rpm_from_mm_s(spd_mm)))
            )
            bio_pos_fn = getattr(self._arm, "set_bio_gripper_g2_position", None)
            if callable(bio_pos_fn):
                code = int(
                    bio_pos_fn(
                        int(round(gap_mm)),
                        speed=spd_rpm,
                        force=int(round(frc_pct)),
                        wait=True,
                    )
                )
                if code != 0:
                    raise SafetyGateError(f"set_bio_gripper_g2_position failed with SDK code {code}")
                self._commanded_gripper_pos = pos_clamped
                return
            fallback_fn = getattr(
                self._arm,
                "open_bio_gripper" if pos_clamped < 0.5 else "close_bio_gripper",
                None,
            )
            if callable(fallback_fn):
                code = int(fallback_fn(speed=spd_rpm, wait=True))
                if code != 0:
                    raise SafetyGateError(f"Bio Gripper SDK command failed with code {code}")
            self._commanded_gripper_pos = pos_clamped
            return

        if (
            self._gripper_spec is not None
            and self._gripper_spec.name == "lite6_gripper"
            and callable(getattr(self._arm, "open_lite6_gripper", None))
            and callable(getattr(self._arm, "close_lite6_gripper", None))
        ):
            method_name = "open_lite6_gripper" if pos_clamped <= 0.5 else "close_lite6_gripper"
            code = int(getattr(self._arm, method_name)(sync=True))
            if code != 0:
                raise SafetyGateError(f"{method_name} failed with SDK code {code}")
            self._commanded_gripper_pos = 0.0 if pos_clamped <= 0.5 else 1.0
            return

        enable_fn = getattr(self._arm, "set_gripper_enable", None)
        if callable(enable_fn):
            code = int(enable_fn(True))
            if code != 0:
                raise SafetyGateError(f"set_gripper_enable failed with SDK code {code}")

        if (
            self._gripper_spec is not None
            and self._gripper_spec.name == "gripper_g2"
            and (speed_percent is not None or force_percent is not None)
        ):
            gap_mm = float(position_to_gap(self._gripper_spec, pos_clamped) * 1000.0)
            spd_pct = 50.0 if speed_percent is None else checked(speed_percent, 0.0, 100.0, "speed_percent")
            frc_pct = 50.0 if force_percent is None else checked(force_percent, 1.0, 100.0, "force_percent")
            spd_mm = speed_mm_s(spd_pct)
            g2_fn = getattr(self._arm, "set_gripper_g2_position", None)
            if callable(g2_fn):
                code = int(
                    g2_fn(
                        int(round(gap_mm)),
                        speed=int(round(spd_mm)),
                        force=int(round(frc_pct)),
                        wait=True,
                    )
                )
                if code != 0:
                    raise SafetyGateError(f"set_gripper_g2_position failed with SDK code {code}")
                self._commanded_gripper_pos = pos_clamped
                return
            speed_fn = getattr(self._arm, "set_gripper_speed", None)
            if callable(speed_fn) and spd_mm > 0:
                code = int(speed_fn(sdk_speed_rpm(spd_mm)))
                if code != 0:
                    raise SafetyGateError(f"set_gripper_speed failed with SDK code {code}")
            set_pos_fn = getattr(self._arm, "set_gripper_position", None)
            if callable(set_pos_fn):
                code = int(set_pos_fn(sdk_position_counts(gap_mm), wait=True))
                if code != 0:
                    raise SafetyGateError(f"set_gripper_position failed with SDK code {code}")
            self._commanded_gripper_pos = pos_clamped
            return

        set_pos_fn = getattr(self._arm, "set_gripper_position", None)
        if callable(set_pos_fn):
            sdk_pulse = float((1.0 - pos_clamped) * 850.0)
            code = int(set_pos_fn(sdk_pulse, wait=True))
            if code != 0:
                raise SafetyGateError(f"set_gripper_position failed with SDK code {code}")
        self._commanded_gripper_pos = pos_clamped

    def get_gripper_gap(self) -> float:
        spec = self._require_jaw()
        return position_to_gap(spec, self.get_gripper_position())

    def set_gripper_gap(
        self,
        gap_m: float,
        *,
        speed_percent: float | None = None,
        force_percent: float | None = None,
    ) -> None:
        self._require_motion_unlocked("set_gripper_gap")
        spec = self._require_jaw()
        norm_pos = gap_to_position(spec, gap_m)
        if speed_percent is None and force_percent is None:
            self.set_gripper(norm_pos)
        else:
            self.set_gripper(norm_pos, speed_percent=speed_percent, force_percent=force_percent)

    def bilateral_finger_contact(self, object_body: str) -> bool:
        """No pad contact sensing on hardware: returns True only while motion is unlocked."""
        del object_body
        _ = self._require_jaw()
        # Real hardware verified via gripper position/stall feedback once unlocked
        return self.motion_unlocked

    def settle(self, duration: float = 1.0) -> None:
        """Sleep up to 0.5 s while motion is unlocked; read-only sessions return immediately."""
        if self.motion_unlocked and duration > 0:
            time.sleep(min(float(duration), 0.5))

    def close(self) -> None:
        """Disconnect the SDK session if this object opened it."""
        if self._owns_connection and self._arm is not None:
            try:
                self._arm.disconnect()
            except Exception:
                pass

    def __enter__(self) -> XArmRobot:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

