"""Rate-limited position/velocity servo with conditional PI anti-windup for Gripper G2.

Pure NumPy implementation matching ``ufactory_genesis``'s ``G2Controller`` reference
trajectory within ``1e-10`` in ``float64``. Supports both single-environment
(``num_envs=1``) and vectorized batch (``(num_envs,)``) execution without requiring PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .g2_model import (
    G2_KI,
    G2_KP,
    G2_KV,
    G2_LINKAGE_GEOMETRY,
    GRIPPER_G2_OPEN_GAP_M,
    ForceLookup,
    G2LinkageGeometry,
    G2Model,
    checked,
)


def drive_for_gap_m(
    gap_m: np.ndarray,
    geometry: G2LinkageGeometry = G2_LINKAGE_GEOMETRY,
) -> np.ndarray:
    g = geometry
    clamped_gap = np.clip(gap_m, 0.0, GRIPPER_G2_OPEN_GAP_M)
    cosine = (clamped_gap / 2.0 - g.base_joint_y_m - g.pad_inner_y_m) / g.radius_m
    angles = np.arccos(np.clip(cosine, -1.0, 1.0)) - g.phase_rad
    return np.clip(angles, g.drive_lower_rad, g.drive_upper_rad).astype(gap_m.dtype, copy=False)


def pad_gap_m(
    drive_rad: np.ndarray,
    geometry: G2LinkageGeometry = G2_LINKAGE_GEOMETRY,
) -> np.ndarray:
    g = geometry
    gap = 2.0 * (
        g.base_joint_y_m
        + g.finger_joint_y_m * np.cos(drive_rad)
        - g.finger_joint_z_m * np.sin(drive_rad)
        + g.pad_inner_y_m
    )
    return np.asarray(gap, dtype=drive_rad.dtype)


@dataclass(frozen=True)
class ControlOutput:
    position_rad: np.ndarray
    velocity_rad_s: np.ndarray
    reference_gap_mm: np.ndarray
    integral_torque_nm: np.ndarray
    torque_limit_nm: np.ndarray


class G2Controller:
    """Calibrated G2 servo; every state and output is a ``(num_envs,)`` NumPy array."""

    def __init__(
        self,
        model: G2Model | None = None,
        *,
        num_envs: int = 1,
        dtype: Any = np.float64,
        initial_position_mm: float = 84.0,
        kp: float = G2_KP,
        kv: float = G2_KV,
        ki: float = G2_KI,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.model = G2Model.load() if model is None else model
        self.num_envs = int(num_envs)
        self.dtype = np.dtype(dtype)
        self.kp = checked(kp, 1e-9, 1e6, "kp")
        self.kv = checked(kv, 0, 1e6, "kv")
        self.ki = checked(ki, 0, 1e6, "ki")
        initial = checked(initial_position_mm, 0, 84, "initial_position_mm")
        self.reference_gap_mm = self._full(initial)
        self.position_mm = self._full(initial)
        self.speed_percent = self._full(50.0)
        self.force_percent = self._full(50.0)
        self.last_nonzero_speed_percent = self._full(50.0)
        self.integral_torque_nm = self._full(0.0)
        self._pause_pending = np.zeros(self.num_envs, dtype=bool)
        self._reverse_pending = np.zeros(self.num_envs, dtype=bool)
        self._pause_start_limit_nm = self._full(0.0)
        self._pause_start_integral_nm = self._full(0.0)
        self._paused_hold_sign = self._full(0.0)
        self._speeds = np.asarray(self.model.curve.speeds, dtype=self.dtype)
        currents = sorted({c for points in self.model.curve.points.values() for c in points})
        self._currents = np.asarray(currents, dtype=self.dtype)
        self._forces = np.asarray(
            [[self.model.curve._at_speed(rpm, current) for current in currents] for rpm in self.model.curve.speeds],
            dtype=self.dtype,
        )

    def _full(self, value: float) -> np.ndarray:
        return np.full((self.num_envs,), float(value), dtype=self.dtype)

    def _vector(
        self,
        value: Any,
        name: str,
        *,
        size: int | None = None,
        bounds: tuple[float, float] | None = None,
    ) -> np.ndarray:
        size = self.num_envs if size is None else size
        arr = np.asarray(value, dtype=self.dtype)
        if arr.ndim == 0:
            arr = np.broadcast_to(arr, (size,)).copy()
        else:
            arr = arr.copy()
        if arr.shape != (size,) or not bool(np.isfinite(arr).all()):
            raise ValueError(f"{name} must be finite with shape ({size},) or scalar")
        if bounds is not None and not bool(((arr >= bounds[0]) & (arr <= bounds[1])).all()):
            raise ValueError(f"{name} must be within {bounds}")
        return arr

    @staticmethod
    def speed_mm_s(percent: np.ndarray) -> np.ndarray:
        return np.where(percent > 0, np.maximum(2.25 * percent, 15.0), 0.0).astype(percent.dtype, copy=False)

    @staticmethod
    def sdk_speed_rpm(speed_mm_s: np.ndarray) -> np.ndarray:
        return np.where(speed_mm_s > 0, np.trunc((speed_mm_s * 60.0 / 9.88235 + 140.0) / 0.4), 0.0).astype(
            speed_mm_s.dtype, copy=False
        )

    @property
    def requested_speed_mm_s(self) -> np.ndarray:
        return self.speed_mm_s(self.speed_percent)

    @property
    def requested_speed_rpm(self) -> np.ndarray:
        return self.sdk_speed_rpm(self.requested_speed_mm_s)

    def _force_rpm(self) -> np.ndarray:
        return self.sdk_speed_rpm(self.speed_mm_s(self.last_nonzero_speed_percent))

    def force_n(self) -> np.ndarray:
        rpm = np.minimum(self._force_rpm(), self.dtype.type(3500.0))
        hi = np.clip(np.searchsorted(self._speeds, rpm, side="left"), 1, len(self._speeds) - 1)
        lo = hi - 1
        ratio = (rpm - self._speeds[lo]) / (self._speeds[hi] - self._speeds[lo])
        chi = np.clip(np.searchsorted(self._currents, self.force_percent, side="left"), 1, len(self._currents) - 1)
        clo = chi - 1
        cr = (self.force_percent - self._currents[clo]) / (self._currents[chi] - self._currents[clo])
        a = self._forces[lo, clo] + cr * (self._forces[lo, chi] - self._forces[lo, clo])
        b = self._forces[hi, clo] + cr * (self._forces[hi, chi] - self._forces[hi, clo])
        return (a + ratio * (b - a)).astype(self.dtype, copy=False)

    def force_lookup(self, env: int = 0) -> ForceLookup:
        """Calibration lookup of one environment, for telemetry."""
        return self.model.curve.lookup(float(self._force_rpm()[env]), float(self.force_percent[env]))

    @property
    def torque_limit_nm(self) -> np.ndarray:
        return (self.force_n() * self.model.reference_jacobian_m_rad).astype(self.dtype, copy=False)

    def drive_for_gap_mm(self, gap: Any) -> np.ndarray:
        arr = np.asarray(gap, dtype=self.dtype)
        return drive_for_gap_m(arr / self.dtype.type(1000.0), self.model.linkage)

    def gap_mm(self, q: Any) -> np.ndarray:
        arr = np.asarray(q, dtype=self.dtype)
        return (self.dtype.type(1000.0) * pad_gap_m(arr, self.model.linkage)).astype(self.dtype, copy=False)

    def command(
        self,
        position_mm: Any,
        speed_percent: Any = 50.0,
        force_percent: Any = 50.0,
        *,
        envs_idx: Any = None,
    ) -> None:
        idx = slice(None) if envs_idx is None else envs_idx
        size = int(self.position_mm[idx].size)
        position = self._vector(position_mm, "position_mm", size=size, bounds=(0.0, 84.0))
        speed = self._vector(speed_percent, "speed_percent", size=size, bounds=(0.0, 100.0))
        force = self._vector(force_percent, "force_percent", size=size, bounds=(1.0, 100.0))
        old_limit = self.torque_limit_nm[idx]
        old_direction = self.position_mm[idx] - self.reference_gap_mm[idx]
        new_direction = position - self.reference_gap_mm[idx]
        reverse = (speed > 0) & (
            (old_direction * new_direction < 0) | (new_direction * self.integral_torque_nm[idx] > 0)
        )
        pause = (speed == 0) & (self.speed_percent[idx] > 0)
        self._reverse_pending[idx] = self._reverse_pending[idx] | reverse
        self.integral_torque_nm[idx] = np.where(reverse, 0.0, self.integral_torque_nm[idx])
        self._pause_pending[idx] = self._pause_pending[idx] | pause
        self._pause_start_limit_nm[idx] = np.where(pause, old_limit, self._pause_start_limit_nm[idx])
        self._pause_start_integral_nm[idx] = np.where(
            pause, self.integral_torque_nm[idx], self._pause_start_integral_nm[idx]
        )
        self.last_nonzero_speed_percent[idx] = np.where(speed > 0, speed, self.last_nonzero_speed_percent[idx])
        self._paused_hold_sign[idx] = np.where(speed > 0, 0.0, self._paused_hold_sign[idx])
        self.position_mm[idx], self.speed_percent[idx], self.force_percent[idx] = position, speed, force
        limit = self.torque_limit_nm[idx]
        self.integral_torque_nm[idx] = np.clip(self.integral_torque_nm[idx], -limit, limit)

    def reset(
        self,
        envs_idx: Any = None,
        *,
        position_mm: Any = 84.0,
        target_mm: Any = None,
        prime: bool = False,
    ) -> None:
        """Reset the reference; optionally seed a saturated pre-held grasp."""
        idx = slice(None) if envs_idx is None else envs_idx
        size = int(self.position_mm[idx].size)
        position = self._vector(position_mm, "position_mm", size=size, bounds=(0.0, 84.0))
        target = position if target_mm is None else self._vector(target_mm, "target_mm", size=size, bounds=(0.0, 84.0))
        self.reference_gap_mm[idx], self.position_mm[idx] = position, target
        self.integral_torque_nm[idx] = -np.sign(target - position) * self.torque_limit_nm[idx] if prime else 0.0
        self._pause_pending[idx] = False
        self._reverse_pending[idx] = False
        self._pause_start_limit_nm[idx] = 0.0
        self._pause_start_integral_nm[idx] = 0.0
        self._paused_hold_sign[idx] = 0.0

    def update(
        self,
        position_rad: Any,
        velocity_rad_s: Any,
        dt: float = 0.02,
        *,
        measured_gap_mm: Any = None,
    ) -> ControlOutput:
        q = self._vector(position_rad, "position_rad")
        velocity = self._vector(velocity_rad_s, "velocity_rad_s")
        dt_val = self.dtype.type(checked(dt, 1e-12, 1e6, "dt"))
        measured = self.gap_mm(q) if measured_gap_mm is None else self._vector(measured_gap_mm, "measured_gap_mm")
        measured = np.clip(measured, 0.0, 84.0)
        limit = self.torque_limit_nm
        old_q_ref = self.drive_for_gap_mm(self.reference_gap_mm)
        self.reference_gap_mm = np.where(self._reverse_pending, measured, self.reference_gap_mm)
        self.integral_torque_nm = np.where(self._reverse_pending, 0.0, self.integral_torque_nm)
        self._reverse_pending.fill(False)
        preload = self._pause_start_integral_nm + self.kp * (old_q_ref - q)
        self.integral_torque_nm = np.where(self._pause_pending, preload, self.integral_torque_nm)
        hold_sign = np.where(
            (np.abs(velocity) <= 0.02) & (np.abs(preload) >= 0.95 * self._pause_start_limit_nm),
            np.sign(preload),
            0.0,
        )
        self._paused_hold_sign = np.where(self._pause_pending, hold_sign, self._paused_hold_sign)
        paused_gap = np.where(np.abs(velocity) > 0.02, measured, np.clip(self.gap_mm(q), 0.0, 84.0))
        self.reference_gap_mm = np.where(self._pause_pending, paused_gap, self.reference_gap_mm)
        self._pause_pending.fill(False)
        previous_q_ref = self.drive_for_gap_mm(self.reference_gap_mm)
        travel = self.speed_mm_s(self.speed_percent) * dt_val
        self.reference_gap_mm = self.reference_gap_mm + np.clip(
            self.position_mm - self.reference_gap_mm,
            -travel,
            travel,
        )
        q_ref = self.drive_for_gap_mm(self.reference_gap_mm)
        v_ref = (q_ref - previous_q_ref) / dt_val
        error = q_ref - q
        integral = np.clip(self.integral_torque_nm, -limit, limit)
        candidate = np.clip(integral + self.ki * error * dt_val, -limit, limit)
        candidate_total = self.kp * error + self.kv * (v_ref - velocity) + candidate
        integral = np.where((np.abs(candidate_total) <= limit) | (error * candidate_total <= 0), candidate, integral)
        holding = (self.speed_percent == 0) & (self._paused_hold_sign != 0)
        self.integral_torque_nm = np.where(holding, self._paused_hold_sign * limit, integral).astype(
            self.dtype, copy=False
        )
        target = (np.where(holding, q, q_ref) + self.integral_torque_nm / self.kp).astype(self.dtype, copy=False)
        vel_out = np.where(holding, 0.0, v_ref).astype(self.dtype, copy=False)
        return ControlOutput(
            target,
            vel_out,
            self.reference_gap_mm.copy(),
            self.integral_torque_nm.copy(),
            limit.copy(),
        )

