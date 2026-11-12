"""Rate-limited position/velocity servo with conditional integral anti-windup for Bio Gripper G2.

Pure Python/NumPy implementation matching ``ufactory_genesis`` ``BioGripperG2Controller``,
with a native MuJoCo right-finger servo adapter ``MuJoCoBioG2Adapter``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import mujoco

from .bio_g2_model import (
    BIO_G2_KI,
    BIO_G2_KP,
    BIO_G2_KV,
    DEFAULT_SPEED_MM_S,
    LEFT_JOINT,
    MAX_SPEED_MM_S,
    MIN_SPEED_MM_S,
    RIGHT_JOINT,
    bio_g2_speed_mm_s_from_percent,
    bio_g2_speed_percent_from_mm_s,
    bounded,
    drive_limit_n,
    gap_to_q,
    q_to_gap,
    speed_mm_s_from_rpm,
)


@dataclass(frozen=True)
class ServoParameters:
    kp_n_m: float = BIO_G2_KP
    kv_n_s_m: float = BIO_G2_KV
    ki_n_m_s: float = BIO_G2_KI


@dataclass(frozen=True)
class BioG2ControlOutput:
    position_m: float
    velocity_m_s: float
    reference_q_m: float
    reference_gap_mm: float
    integral_n: float
    force_limit_n: float
    predicted_force_n: float


class Adapter(Protocol):
    def state(self) -> tuple[float, float, float, float]: ...
    def set_force_limit(self, limit_n: float) -> None: ...
    def servo(self, position_m: float, velocity_m_s: float) -> None: ...
    def initialize(self, q_m: float) -> None: ...


class MuJoCoBioG2Adapter:
    """MuJoCo adapter driving ``bio_gripper_g2_right_finger_joint`` with velocity feedforward bias."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        actuator_id: int,
        parameters: ServoParameters = ServoParameters(),
    ) -> None:
        self.model = model
        self.data = data
        self.actuator_id = int(actuator_id)
        self.parameters = parameters

        right_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, RIGHT_JOINT)
        left_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, LEFT_JOINT)
        if right_jnt < 0 or left_jnt < 0:
            raise ValueError(f"Bio Gripper G2 joints ({RIGHT_JOINT!r}, {LEFT_JOINT!r}) not found in MjModel")

        self.right_qpos_adr = int(model.jnt_qposadr[right_jnt])
        self.right_dof_adr = int(model.jnt_dofadr[right_jnt])
        self.left_qpos_adr = int(model.jnt_qposadr[left_jnt])
        self.left_dof_adr = int(model.jnt_dofadr[left_jnt])

        # Allow integral preload + velocity feedforward bias beyond nominal [0, 0.0395] m stroke.
        self.model.actuator_ctrllimited[self.actuator_id] = 0
        self.last_servo = (float(self.data.qpos[self.right_qpos_adr]), 0.0)

    def state(self) -> tuple[float, float, float, float]:
        qr = float(self.data.qpos[self.right_qpos_adr])
        vr = float(self.data.qvel[self.right_dof_adr])
        ql = float(self.data.qpos[self.left_qpos_adr])
        vl = float(self.data.qvel[self.left_dof_adr])
        return qr, vr, ql, vl

    def set_force_limit(self, limit_n: float) -> None:
        lim = float(limit_n)
        self.model.actuator_forcerange[self.actuator_id] = (-lim, lim)

    def servo(self, position_m: float, velocity_m_s: float) -> None:
        pos = float(position_m)
        vel = float(velocity_m_s)
        self.last_servo = (pos, vel)
        vel_bias = (self.parameters.kv_n_s_m / self.parameters.kp_n_m) * vel
        self.data.ctrl[self.actuator_id] = pos + vel_bias

    def initialize(self, q_m: float) -> None:
        q = float(q_m)
        self.data.qpos[self.right_qpos_adr] = q
        self.data.qpos[self.left_qpos_adr] = -q
        self.data.qvel[self.right_dof_adr] = 0.0
        self.data.qvel[self.left_dof_adr] = 0.0
        self.data.ctrl[self.actuator_id] = q


class BioGripperG2Controller:
    """Calibrated Bio Gripper G2 servo controller."""

    def __init__(self, adapter: Adapter, parameters: ServoParameters = ServoParameters()) -> None:
        self.adapter, self.parameters = adapter, parameters
        self.position_mm = 150.0
        self.speed_mm_s = DEFAULT_SPEED_MM_S
        self.speed_percent = 50.0
        self.force_percent = 50.0
        self.paused = False
        self._paused_by_zero_speed = False
        self._clear_integral_on_resume = False
        self.reference_q_m = adapter.state()[0]
        self.servo_reference_q_m = self.reference_q_m
        self.reference_velocity_m_s = 0.0
        self.integral_n = 0.0
        self.force_limit_n = drive_limit_n(self.force_percent)
        self.predicted_force_n = 0.0
        adapter.set_force_limit(self.force_limit_n)

    def command(
        self,
        position_mm: float,
        speed_mm_s: float | None = None,
        force_percent: float = 50.0,
        *,
        speed_percent: float | None = None,
        speed_rpm: float | None = None,
    ) -> None:
        """Issue a validated position/speed/force command (atomic on validation failure)."""
        provided = sum(v is not None for v in (speed_mm_s, speed_percent, speed_rpm))
        if provided > 1:
            raise ValueError("Specify at most one of speed_mm_s, speed_percent, or speed_rpm")
        q = gap_to_q(position_mm)
        force = bounded(force_percent, 1.0, 100.0, "force_percent")
        zero_speed_pause = False
        if speed_percent is not None:
            pct = bounded(speed_percent, 0.0, 100.0, "speed_percent")
            if pct == 0.0:
                zero_speed_pause = True
                speed = self.speed_mm_s
            else:
                speed = bio_g2_speed_mm_s_from_percent(pct)
        elif speed_rpm is not None:
            speed = speed_mm_s_from_rpm(speed_rpm)
            pct = bio_g2_speed_percent_from_mm_s(speed)
        else:
            raw_speed = DEFAULT_SPEED_MM_S if speed_mm_s is None else speed_mm_s
            speed = bounded(raw_speed, MIN_SPEED_MM_S, MAX_SPEED_MM_S, "speed_mm_s")
            pct = bio_g2_speed_percent_from_mm_s(speed)

        if zero_speed_pause and not self.paused:
            self.pause()
            self._paused_by_zero_speed = True
        elif not zero_speed_pause and self._paused_by_zero_speed:
            self._paused_by_zero_speed = False
            self.resume()

        new_limit = drive_limit_n(force)
        old_direction = gap_to_q(self.position_mm) - self.reference_q_m
        new_direction = q - self.reference_q_m
        reversal = (
            old_direction * new_direction < 0
            or new_direction * self.integral_n < 0
            or (new_direction == 0 and old_direction * self.integral_n > 0)
        )
        if self.paused:
            self._clear_integral_on_resume = bool(reversal)
        elif reversal:
            self.integral_n = 0.0
        self.position_mm = float(position_mm)
        self.speed_mm_s = float(speed)
        self.speed_percent = float(pct)
        self.force_percent = float(force)
        self.force_limit_n = new_limit
        self.integral_n = max(-new_limit, min(new_limit, self.integral_n))
        self.adapter.set_force_limit(new_limit)
        # Re-issue the current servo input so lowered limits take effect while paused, too.
        self._servo()

    def pause(self) -> None:
        self._paused_by_zero_speed = False
        if self.paused:
            return
        q, v, _, _ = self.adapter.state()
        preload = self.parameters.kp_n_m * (self.reference_q_m - q) + self.integral_n
        stalled = abs(v) < 0.001 and abs(preload) >= 0.9 * self.force_limit_n
        self.integral_n = max(-self.force_limit_n, min(self.force_limit_n, preload)) if stalled else 0.0
        self.reference_q_m = q
        self.servo_reference_q_m = q
        self.reference_velocity_m_s = 0.0
        self.paused = True
        self._servo()

    def resume(self) -> None:
        self._paused_by_zero_speed = False
        if self._clear_integral_on_resume:
            self.integral_n = 0.0
            self._clear_integral_on_resume = False
        self.paused = False

    def reset(self, position_mm: float = 150.0) -> None:
        q = gap_to_q(position_mm)
        self.adapter.initialize(q)
        self.position_mm = float(position_mm)
        self.speed_mm_s = DEFAULT_SPEED_MM_S
        self.speed_percent = 50.0
        self.force_percent = 50.0
        self.force_limit_n = drive_limit_n(50.0)
        self.reference_q_m, self.reference_velocity_m_s = q, 0.0
        self.servo_reference_q_m = q
        self.integral_n, self.predicted_force_n = 0.0, 0.0
        self.paused = False
        self._paused_by_zero_speed = False
        self._clear_integral_on_resume = False
        self.adapter.set_force_limit(self.force_limit_n)
        self._servo()

    def _servo(self) -> None:
        self.adapter.servo(
            self.servo_reference_q_m + self.integral_n / self.parameters.kp_n_m,
            self.reference_velocity_m_s,
        )

    def apply(self, dt: float = 0.02) -> BioG2ControlOutput:
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        q, v, _, _ = self.adapter.state()
        previous_reference = self.reference_q_m
        if not self.paused:
            delta = gap_to_q(self.position_mm) - self.reference_q_m
            step = max(-self.speed_mm_s * dt / 1000.0, min(self.speed_mm_s * dt / 1000.0, delta))
            self.reference_q_m += step
            # Midpoint reference cancels the phase lead of a zero-order-held position/velocity command.
            self.servo_reference_q_m = previous_reference + step / 2.0
            self.reference_velocity_m_s = step / dt
        error = previous_reference - q
        p = self.parameters
        force = (
            p.kp_n_m * (self.servo_reference_q_m - q) + p.kv_n_s_m * (self.reference_velocity_m_s - v) + self.integral_n
        )
        increment = p.ki_n_m_s * error * dt
        # Pause holds the captured preload; don't integrate a new motion command during pause.
        if not self.paused and (abs(force) < self.force_limit_n or force * increment < 0):
            self.integral_n = max(-self.force_limit_n, min(self.force_limit_n, self.integral_n + increment))
        self.predicted_force_n = max(-self.force_limit_n, min(self.force_limit_n, force))
        self._servo()
        return BioG2ControlOutput(
            position_m=self.servo_reference_q_m + self.integral_n / p.kp_n_m,
            velocity_m_s=self.reference_velocity_m_s,
            reference_q_m=self.reference_q_m,
            reference_gap_mm=q_to_gap(self.reference_q_m),
            integral_n=self.integral_n,
            force_limit_n=self.force_limit_n,
            predicted_force_n=self.predicted_force_n,
        )

    def telemetry(self) -> dict:
        qr, vr, ql, vl = self.adapter.state()
        return {
            "gap_mm": q_to_gap(qr, ql),
            "speed_mm_s": (vr - vl) * 500.0,
            "gap_speed_mm_s": (vr - vl) * 1000.0,
            "right_q_m": qr,
            "left_q_m": ql,
            "sync_error_mm": abs(qr + ql) * 1000.0,
            "target_mm": self.position_mm,
            "command_speed_mm_s": self.speed_mm_s,
            "command_gap_speed_mm_s": 2.0 * self.speed_mm_s,
            "speed_percent": self.speed_percent,
            "force_percent": self.force_percent,
            "drive_limit_n": self.force_limit_n,
            "reference_gap_mm": q_to_gap(self.reference_q_m),
            "integral_n": self.integral_n,
            "paused": self.paused,
        }


GripperController = BioGripperG2Controller
