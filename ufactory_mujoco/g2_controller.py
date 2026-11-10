"""Rate-limited position/velocity servo with conditional PI anti-windup for Gripper G2.

Pure NumPy implementation matching ``ufactory_genesis``'s ``G2Controller`` reference
trajectory within ``1e-10`` in ``float64``. Supports both single-environment
(``num_envs=1``) and vectorized batch (``(num_envs,)``) execution without requiring PyTorch.
``ScalarG2Controller`` is a faster single-environment version with identical ``float64`` results.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
import math
from typing import Any, NamedTuple

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
    """Vectorized inverse linkage: pad gap (m, clipped to 0..84 mm) to drive angle (rad, clipped to the joint range)."""
    g = geometry
    clamped_gap = np.clip(gap_m, 0.0, GRIPPER_G2_OPEN_GAP_M)
    cosine = (clamped_gap / 2.0 - g.base_joint_y_m - g.pad_inner_y_m) / g.radius_m
    angles = np.arccos(np.clip(cosine, -1.0, 1.0)) - g.phase_rad
    return np.clip(angles, g.drive_lower_rad, g.drive_upper_rad).astype(gap_m.dtype, copy=False)


def pad_gap_m(
    drive_rad: np.ndarray,
    geometry: G2LinkageGeometry = G2_LINKAGE_GEOMETRY,
) -> np.ndarray:
    """Vectorized forward linkage: drive angle (rad) to pad gap (m)."""
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
    """One controller step per environment.

    Position and velocity targets, reference gap, integral torque and torque limit.
    """
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
        """Map speed percent to jaw speed in mm/s (0 stays 0, otherwise at least 15 mm/s)."""
        return np.where(percent > 0, np.maximum(2.25 * percent, 15.0), 0.0).astype(percent.dtype, copy=False)

    @staticmethod
    def sdk_speed_rpm(speed_mm_s: np.ndarray) -> np.ndarray:
        """Map jaw speed in mm/s to the SDK motor speed in rpm (truncated)."""
        return np.where(speed_mm_s > 0, np.trunc((speed_mm_s * 60.0 / 9.88235 + 140.0) / 0.4), 0.0).astype(
            speed_mm_s.dtype, copy=False
        )

    @property
    def requested_speed_mm_s(self) -> np.ndarray:
        """Jaw speed in mm/s for the current speed command."""
        return self.speed_mm_s(self.speed_percent)

    @property
    def requested_speed_rpm(self) -> np.ndarray:
        """SDK motor speed in rpm for the current speed command."""
        return self.sdk_speed_rpm(self.requested_speed_mm_s)

    def _force_rpm(self) -> np.ndarray:
        return self.sdk_speed_rpm(self.speed_mm_s(self.last_nonzero_speed_percent))

    def force_n(self) -> np.ndarray:
        """Static pinch force (N) from the calibration table at the last non-zero speed and the force command."""
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
        """Drive torque limit (N*m): calibrated force times the reference linkage Jacobian."""
        return (self.force_n() * self.model.reference_jacobian_m_rad).astype(self.dtype, copy=False)

    def drive_for_gap_mm(self, gap: Any) -> np.ndarray:
        """Drive angle (rad) for a pad gap given in mm."""
        arr = np.asarray(gap, dtype=self.dtype)
        return drive_for_gap_m(arr / self.dtype.type(1000.0), self.model.linkage)

    def gap_mm(self, q: Any) -> np.ndarray:
        """Pad gap (mm) for a drive angle in rad."""
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
        """Set target gap (mm), speed (%) and force (%) for all or selected environments.

        Reversing direction or pausing (speed 0) resets the integral term.
        """
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
        """Advance the rate-limited reference by ``dt`` and return PI+PD position targets for the drive joint."""
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



class ScalarControlOutput(NamedTuple):
    """``ControlOutput`` of one environment, as Python floats."""

    position_rad: float
    velocity_rad_s: float
    reference_gap_mm: float
    integral_torque_nm: float
    torque_limit_nm: float


def _clip(value: float, lower: float, upper: float) -> float:
    """``np.clip`` on one ``float64``: NaN in any argument propagates and ties keep ``value``."""
    if value != value:
        return value
    if lower != lower:
        return lower
    if upper != upper:
        return upper
    if value < lower:
        value = lower
    if value > upper:
        value = upper
    return value


def _sign(value: float) -> float:
    """``np.sign`` on one ``float64`` (``-0.0`` -> ``0.0``, NaN -> NaN)."""
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0 if value == 0.0 else value


class ScalarG2Controller:
    """Single-environment ``float64`` G2 servo with Python-float state.

    Same formulas, operation order, and input validation as ``G2Controller(num_envs=1)``, so
    outputs and state match bit for bit. ``arccos``/``cos``/``sin`` run through the same NumPy
    ufuncs on one-element arrays, keeping results identical where NumPy uses SIMD math libraries.
    """

    def __init__(
        self,
        model: G2Model | None = None,
        *,
        initial_position_mm: float = 84.0,
        kp: float = G2_KP,
        kv: float = G2_KV,
        ki: float = G2_KI,
    ) -> None:
        self.model = G2Model.load() if model is None else model
        self.kp = checked(kp, 1e-9, 1e6, "kp")
        self.kv = checked(kv, 0, 1e6, "kv")
        self.ki = checked(ki, 0, 1e6, "ki")
        initial = checked(initial_position_mm, 0, 84, "initial_position_mm")
        self.reference_gap_mm = initial
        self.position_mm = initial
        self.speed_percent = 50.0
        self.force_percent = 50.0
        self.last_nonzero_speed_percent = 50.0
        self.integral_torque_nm = 0.0
        self._pause_pending = False
        self._reverse_pending = False
        self._pause_start_limit_nm = 0.0
        self._pause_start_integral_nm = 0.0
        self._paused_hold_sign = 0.0

        # Force table built exactly as in G2Controller, then held as Python floats.
        currents = sorted({c for points in self.model.curve.points.values() for c in points})
        self._speeds = np.asarray(self.model.curve.speeds, dtype=np.float64).tolist()
        self._currents = np.asarray(currents, dtype=np.float64).tolist()
        self._forces = np.asarray(
            [[self.model.curve._at_speed(rpm, current) for current in currents] for rpm in self.model.curve.speeds],
            dtype=np.float64,
        ).tolist()
        self._force_key: tuple[float, float] | None = None
        self._force_value = 0.0

        g = self.model.linkage
        self._base_y, self._pad_y = g.base_joint_y_m, g.pad_inner_y_m
        self._finger_y, self._finger_z = g.finger_joint_y_m, g.finger_joint_z_m
        self._radius, self._phase = g.radius_m, g.phase_rad
        self._drive_lower, self._drive_upper = g.drive_lower_rad, g.drive_upper_rad
        self._reference_jacobian = self.model.reference_jacobian_m_rad
        self._ufunc_in = np.empty(1, dtype=np.float64)
        self._ufunc_out = np.empty(1, dtype=np.float64)

    @staticmethod
    def _value(value: Any, name: str, bounds: tuple[float, float] | None = None) -> float:
        if type(value) is float:
            result = value
        else:
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape not in ((), (1,)):
                raise ValueError(f"{name} must be finite with shape (1,) or scalar")
            result = arr.item()
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite with shape (1,) or scalar")
        if bounds is not None and not bounds[0] <= result <= bounds[1]:
            raise ValueError(f"{name} must be within {bounds}")
        return result

    def _ufunc(self, func: np.ufunc, value: float) -> float:
        self._ufunc_in[0] = value
        func(self._ufunc_in, out=self._ufunc_out)
        return self._ufunc_out.item(0)

    @staticmethod
    def speed_mm_s(percent: float) -> float:
        """Map speed percent to jaw speed in mm/s (0 stays 0, otherwise at least 15 mm/s)."""
        return max(2.25 * percent, 15.0) if percent > 0 else 0.0

    @staticmethod
    def sdk_speed_rpm(speed_mm_s: float) -> float:
        """Map jaw speed in mm/s to the SDK motor speed in rpm (truncated)."""
        return float(math.trunc((speed_mm_s * 60.0 / 9.88235 + 140.0) / 0.4)) if speed_mm_s > 0 else 0.0

    def force_n(self) -> float:
        """Static pinch force (N) from the calibration table; cached per speed/force command."""
        key = (self.last_nonzero_speed_percent, self.force_percent)
        if key != self._force_key:
            speed_percent, force_percent = key
            rpm = min(self.sdk_speed_rpm(self.speed_mm_s(speed_percent)), 3500.0)
            speeds, currents, forces = self._speeds, self._currents, self._forces
            hi = min(max(bisect.bisect_left(speeds, rpm), 1), len(speeds) - 1)
            lo = hi - 1
            ratio = (rpm - speeds[lo]) / (speeds[hi] - speeds[lo])
            chi = min(max(bisect.bisect_left(currents, force_percent), 1), len(currents) - 1)
            clo = chi - 1
            cr = (force_percent - currents[clo]) / (currents[chi] - currents[clo])
            a = forces[lo][clo] + cr * (forces[lo][chi] - forces[lo][clo])
            b = forces[hi][clo] + cr * (forces[hi][chi] - forces[hi][clo])
            self._force_key, self._force_value = key, a + ratio * (b - a)
        return self._force_value

    @property
    def torque_limit_nm(self) -> float:
        """Drive torque limit (N*m): calibrated force times the reference linkage Jacobian."""
        return self.force_n() * self._reference_jacobian

    def drive_for_gap_mm(self, gap_mm: float) -> float:
        """Drive angle (rad) for a pad gap given in mm."""
        clamped_gap = _clip(gap_mm / 1000.0, 0.0, GRIPPER_G2_OPEN_GAP_M)
        cosine = (clamped_gap / 2.0 - self._base_y - self._pad_y) / self._radius
        angle = self._ufunc(np.arccos, _clip(cosine, -1.0, 1.0)) - self._phase
        return _clip(angle, self._drive_lower, self._drive_upper)

    def gap_mm(self, q: float) -> float:
        """Pad gap (mm) for a drive angle in rad."""
        cos_q = self._ufunc(np.cos, q)
        sin_q = self._ufunc(np.sin, q)
        return 1000.0 * (2.0 * (self._base_y + self._finger_y * cos_q - self._finger_z * sin_q + self._pad_y))

    def command(self, position_mm: Any, speed_percent: Any = 50.0, force_percent: Any = 50.0) -> None:
        """Set target gap (mm), speed (%) and force (%); reversing or pausing resets the integral term."""
        position = self._value(position_mm, "position_mm", (0.0, 84.0))
        speed = self._value(speed_percent, "speed_percent", (0.0, 100.0))
        force = self._value(force_percent, "force_percent", (1.0, 100.0))
        old_limit = self.torque_limit_nm
        old_direction = self.position_mm - self.reference_gap_mm
        new_direction = position - self.reference_gap_mm
        reverse = speed > 0 and (old_direction * new_direction < 0 or new_direction * self.integral_torque_nm > 0)
        pause = speed == 0 and self.speed_percent > 0
        self._reverse_pending = self._reverse_pending or reverse
        if reverse:
            self.integral_torque_nm = 0.0
        self._pause_pending = self._pause_pending or pause
        if pause:
            self._pause_start_limit_nm = old_limit
            self._pause_start_integral_nm = self.integral_torque_nm
        if speed > 0:
            self.last_nonzero_speed_percent = speed
            self._paused_hold_sign = 0.0
        self.position_mm, self.speed_percent, self.force_percent = position, speed, force
        limit = self.torque_limit_nm
        self.integral_torque_nm = _clip(self.integral_torque_nm, -limit, limit)

    def reset(self, *, position_mm: Any = 84.0, target_mm: Any = None, prime: bool = False) -> None:
        """Reset the reference; optionally seed a saturated pre-held grasp."""
        position = self._value(position_mm, "position_mm", (0.0, 84.0))
        target = position if target_mm is None else self._value(target_mm, "target_mm", (0.0, 84.0))
        self.reference_gap_mm, self.position_mm = position, target
        self.integral_torque_nm = -_sign(target - position) * self.torque_limit_nm if prime else 0.0
        self._pause_pending = False
        self._reverse_pending = False
        self._pause_start_limit_nm = 0.0
        self._pause_start_integral_nm = 0.0
        self._paused_hold_sign = 0.0

    def update(
        self,
        position_rad: Any,
        velocity_rad_s: Any,
        dt: float = 0.02,
        *,
        measured_gap_mm: Any = None,
    ) -> ScalarControlOutput:
        """Advance the rate-limited reference by ``dt`` and return the PI+PD position target for the drive joint."""
        q = self._value(position_rad, "position_rad")
        velocity = self._value(velocity_rad_s, "velocity_rad_s")
        dt_val = checked(dt, 1e-12, 1e6, "dt")
        measured = self.gap_mm(q) if measured_gap_mm is None else self._value(measured_gap_mm, "measured_gap_mm")
        measured = _clip(measured, 0.0, 84.0)
        limit = self.torque_limit_nm
        old_q_ref = self.drive_for_gap_mm(self.reference_gap_mm)
        if self._reverse_pending:
            self.reference_gap_mm = measured
            self.integral_torque_nm = 0.0
        self._reverse_pending = False
        if self._pause_pending:
            preload = self._pause_start_integral_nm + self.kp * (old_q_ref - q)
            self.integral_torque_nm = preload
            hold = abs(velocity) <= 0.02 and abs(preload) >= 0.95 * self._pause_start_limit_nm
            self._paused_hold_sign = _sign(preload) if hold else 0.0
            self.reference_gap_mm = measured if abs(velocity) > 0.02 else _clip(self.gap_mm(q), 0.0, 84.0)
        self._pause_pending = False
        previous_q_ref = self.drive_for_gap_mm(self.reference_gap_mm)
        travel = self.speed_mm_s(self.speed_percent) * dt_val
        self.reference_gap_mm = self.reference_gap_mm + _clip(
            self.position_mm - self.reference_gap_mm, -travel, travel
        )
        q_ref = self.drive_for_gap_mm(self.reference_gap_mm)
        v_ref = (q_ref - previous_q_ref) / dt_val
        error = q_ref - q
        integral = _clip(self.integral_torque_nm, -limit, limit)
        candidate = _clip(integral + self.ki * error * dt_val, -limit, limit)
        candidate_total = self.kp * error + self.kv * (v_ref - velocity) + candidate
        if abs(candidate_total) <= limit or error * candidate_total <= 0:
            integral = candidate
        holding = self.speed_percent == 0 and self._paused_hold_sign != 0
        self.integral_torque_nm = self._paused_hold_sign * limit if holding else integral
        target = (q if holding else q_ref) + self.integral_torque_nm / self.kp
        return ScalarControlOutput(
            target, 0.0 if holding else v_ref, self.reference_gap_mm, self.integral_torque_nm, limit
        )
