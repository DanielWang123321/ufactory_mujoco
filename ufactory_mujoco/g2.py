"""Gripper G2 calibrated linkage geometry, SDK unit conversions, and force curve model."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from functools import lru_cache
import math
from typing import Any

import numpy as np

G2_JOINT_NAMES: tuple[str, ...] = (
    "left_driver_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_driver_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
)

GRIPPER_G2_CLOSED_GAP_MM = 0.0
GRIPPER_G2_OPEN_GAP_MM = 84.0
GRIPPER_G2_OPEN_GAP_M = GRIPPER_G2_OPEN_GAP_MM / 1000.0

GRIPPER_G2_SIM_OPEN_DRIVE = 0.057210812728026743
GRIPPER_G2_SIM_CLOSE_DRIVE = 0.86509464599328212

G2_KP, G2_KV, G2_KI = 40.0, 5.0, 40.0
G2_ARMATURE = 0.01
G2_DAMPING = 0.05
G2_FRICTIONLOSS = 0.0
G2_MIMIC_SOLREF: tuple[float, float] = (0.02, 1.0)
G2_MIMIC_SOLIMP: tuple[float, float, float, float, float] = (0.9, 0.95, 0.001, 0.5, 2.0)

REFERENCE_RAW_POSITION = 230

FORCE_CURVE_CURRENTS_PERCENT: tuple[float, ...] = tuple(float(c) for c in range(0, 101, 5))
FORCE_CURVE_TABLE_N: dict[float, tuple[float, ...]] = {
    0.0: (
        15.04,
        15.23,
        15.44,
        16.38,
        20.38,
        24.24,
        25.18,
        31.43,
        32.73,
        34.0,
        35.13,
        43.56,
        44.61,
        45.79,
        46.78,
        47.59,
        48.49,
        58.49,
        59.71,
        60.55,
        63.79,
    ),
    250.0: (
        15.04,
        15.23,
        15.44,
        15.89,
        19.73,
        23.84,
        24.59,
        31.43,
        32.73,
        34.28,
        36.25,
        44.45,
        44.61,
        45.79,
        46.78,
        47.78,
        48.68,
        58.49,
        59.71,
        60.55,
        63.79,
    ),
    500.0: (
        14.93,
        15.04,
        15.14,
        16.06,
        20.28,
        23.76,
        24.59,
        31.43,
        32.73,
        33.89,
        35.33,
        43.56,
        44.61,
        45.79,
        46.78,
        47.78,
        48.68,
        58.72,
        59.71,
        60.55,
        62.95,
    ),
    750.0: (
        14.83,
        14.82,
        15.03,
        15.98,
        22.42,
        24.06,
        25.25,
        31.43,
        32.73,
        33.89,
        35.33,
        43.72,
        44.61,
        45.79,
        46.88,
        48.14,
        49.04,
        58.72,
        59.71,
        60.55,
        62.95,
    ),
    1000.0: (
        17.67,
        17.32,
        17.75,
        17.75,
        23.14,
        23.84,
        25.22,
        31.59,
        32.81,
        33.89,
        35.53,
        43.66,
        44.61,
        45.79,
        46.88,
        48.01,
        48.83,
        59.04,
        59.71,
        60.55,
        62.95,
    ),
    1250.0: (
        17.67,
        17.32,
        17.75,
        17.75,
        23.14,
        23.59,
        24.2,
        31.59,
        32.81,
        33.89,
        35.73,
        43.76,
        44.61,
        45.79,
        46.88,
        48.01,
        48.83,
        59.04,
        59.71,
        60.95,
        61.53,
    ),
    1500.0: (
        16.2,
        16.3,
        16.31,
        16.25,
        22.62,
        23.68,
        24.98,
        32.45,
        33.22,
        34.03,
        36.04,
        44.06,
        44.61,
        45.79,
        46.88,
        48.01,
        48.83,
        59.04,
        59.71,
        60.95,
        61.53,
    ),
    1750.0: (
        16.41,
        16.41,
        16.41,
        16.56,
        22.73,
        23.85,
        24.55,
        31.92,
        33.22,
        34.03,
        36.04,
        44.06,
        45.1,
        46.22,
        47.21,
        48.61,
        58.36,
        59.23,
        60.16,
        60.6,
        61.41,
    ),
    2000.0: (
        16.62,
        16.82,
        16.62,
        16.88,
        22.99,
        23.76,
        23.9,
        31.92,
        33.22,
        34.13,
        36.04,
        44.06,
        45.1,
        46.22,
        47.21,
        48.61,
        58.36,
        59.23,
        60.16,
        61.82,
        61.58,
    ),
    2250.0: (
        17.22,
        17.62,
        17.42,
        17.49,
        22.91,
        23.66,
        28.65,
        31.94,
        33.09,
        34.13,
        43.38,
        44.34,
        45.1,
        46.22,
        47.21,
        48.61,
        58.36,
        59.23,
        60.16,
        61.01,
        61.01,
    ),
    2500.0: (
        18.02,
        18.02,
        18.02,
        18.24,
        23.11,
        23.63,
        24.41,
        32.35,
        33.65,
        34.56,
        43.38,
        44.34,
        45.44,
        46.54,
        47.21,
        48.61,
        58.36,
        59.13,
        60.23,
        61.66,
        62.8,
    ),
    2750.0: (
        22.64,
        22.64,
        22.64,
        22.54,
        23.24,
        23.43,
        23.71,
        32.39,
        33.54,
        34.39,
        43.15,
        44.23,
        45.22,
        46.41,
        47.38,
        48.46,
        58.16,
        59.14,
        60.14,
        62.18,
        63.21,
    ),
    3000.0: (
        22.42,
        22.42,
        22.42,
        22.39,
        23.01,
        23.02,
        23.1,
        32.08,
        33.36,
        34.18,
        42.91,
        44.11,
        45.22,
        46.41,
        47.38,
        48.46,
        58.16,
        59.09,
        60.15,
        60.49,
        60.6,
    ),
    3250.0: (
        23.04,
        23.03,
        23.04,
        22.81,
        22.81,
        22.82,
        22.95,
        31.73,
        32.7,
        33.76,
        43.22,
        43.6,
        45.14,
        46.14,
        47.14,
        48.37,
        57.93,
        58.68,
        60.12,
        61.78,
        63.08,
    ),
    3500.0: (
        22.82,
        23.02,
        23.42,
        23.22,
        23.22,
        23.22,
        23.25,
        32.05,
        32.82,
        34.24,
        42.64,
        44.01,
        44.81,
        46.01,
        47.01,
        48.03,
        57.83,
        59.01,
        59.8,
        60.81,
        63.21,
    ),
}


@dataclass(frozen=True)
class G2LinkageGeometry:
    """Parallel-pad 4-bar linkage dimensions from the calibrated G2 URDF and collision mesh."""

    base_joint_y_m: float = 0.035
    finger_joint_y_m: float = 0.035465
    finger_joint_z_m: float = 0.042039
    pad_inner_y_m: float = -0.02600320242345333
    pad_center_z_m: float = 0.041003202088177204
    drive_lower_rad: float = GRIPPER_G2_SIM_OPEN_DRIVE
    drive_upper_rad: float = GRIPPER_G2_SIM_CLOSE_DRIVE

    @property
    def radius_m(self) -> float:
        """Distance from the base joint to the finger joint (m)."""
        return math.hypot(self.finger_joint_y_m, self.finger_joint_z_m)

    @property
    def phase_rad(self) -> float:
        """Angle of the finger joint offset in the linkage plane (rad)."""
        return math.atan2(self.finger_joint_z_m, self.finger_joint_y_m)

    def pad_gap_m(self, drive_rad: float) -> float:
        """Pad gap (m) for a drive angle (rad)."""
        q = float(drive_rad)
        return 2.0 * (
            self.base_joint_y_m
            + self.finger_joint_y_m * math.cos(q)
            - self.finger_joint_z_m * math.sin(q)
            + self.pad_inner_y_m
        )

    def gap_jacobian_m_per_rad(self, drive_rad: float) -> float:
        """Absolute derivative of the full pad gap with respect to joint angle."""
        q = float(drive_rad)
        return abs(2.0 * (self.finger_joint_y_m * math.sin(q) + self.finger_joint_z_m * math.cos(q)))

    def drive_for_gap_m(self, gap_m: float) -> float:
        """Drive angle (rad) for a pad gap (m); raises if the gap is outside the physical range."""
        gap = float(gap_m)
        if not math.isfinite(gap) or not self.pad_gap_m(self.drive_upper_rad) <= gap <= self.pad_gap_m(
            self.drive_lower_rad
        ):
            raise ValueError("calibration gap is outside the collision pads' physical range")
        cosine = (gap / 2.0 - self.base_joint_y_m - self.pad_inner_y_m) / self.radius_m
        q = math.acos(max(-1.0, min(1.0, cosine))) - self.phase_rad
        return max(self.drive_lower_rad, min(self.drive_upper_rad, q))


G2_LINKAGE_GEOMETRY = G2LinkageGeometry()


def clamp_gripper_g2_gap_m(gap_m: float) -> float:
    """Clamp a two-finger gap in metres to the Gripper G2 physical range."""
    return max(GRIPPER_G2_CLOSED_GAP_MM / 1000.0, min(GRIPPER_G2_OPEN_GAP_M, float(gap_m)))


def gripper_g2_gap_m_to_sim_drive(gap_m: float) -> float:
    """Map a clamped pad gap in metres to the actual drive joint angle in radians."""
    return G2_LINKAGE_GEOMETRY.drive_for_gap_m(clamp_gripper_g2_gap_m(gap_m))


def gripper_g2_sim_drive_to_gap_m(drive: float) -> float:
    """Map the actual joint angle in radians to a clamped pad gap in metres."""
    clipped = max(GRIPPER_G2_SIM_OPEN_DRIVE, min(GRIPPER_G2_SIM_CLOSE_DRIVE, float(drive)))
    return clamp_gripper_g2_gap_m(G2_LINKAGE_GEOMETRY.pad_gap_m(clipped))


def checked(value: float, low: float, high: float, name: str) -> float:
    """Return ``value`` as float, raising ValueError unless it is a finite number within ``[low, high]``."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and within {low:g}–{high:g}")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and within {low:g}–{high:g}") from exc
    if not math.isfinite(numeric) or not low <= numeric <= high:
        raise ValueError(f"{name} must be finite and within {low:g}–{high:g}, got {numeric}")
    return numeric


def sdk_position_counts(position_mm: float) -> int:
    """Convert a gap in mm to the SDK raw position counts."""
    position = checked(position_mm, 0, 84, "position_mm")
    return int((math.degrees(math.asin((position - 16) / 110)) + 8.33) * 18.28)


def sdk_position_mm(raw_position: float) -> float:
    """Convert SDK raw position counts to a gap in mm."""
    raw_position = float(raw_position)
    if not math.isfinite(raw_position):
        raise ValueError("raw_position must be finite")
    return math.sin(math.radians(raw_position / 18.28 - 8.33)) * 110 + 16


REFERENCE_GAP_MM = sdk_position_mm(REFERENCE_RAW_POSITION)


def speed_mm_s(speed_percent: float) -> float:
    """Map speed percent (0..100) to jaw speed in mm/s (0 stays 0, otherwise at least 15 mm/s)."""
    speed = checked(speed_percent, 0, 100, "speed_percent")
    return max(15.0, 225.0 * speed / 100.0) if speed else 0.0


def sdk_speed_rpm(speed_mm_s_val: float) -> int:
    """Map jaw speed in mm/s (0..225) to the SDK motor speed in rpm."""
    speed = checked(speed_mm_s_val, 0, 225, "speed_mm_s")
    return int(((speed * 60) / 9.88235 + 140) / 0.4) if speed else 0


@dataclass(frozen=True)
class ForceLookup:
    """Force-curve result: force (N), requested and table speeds (rpm), and whether the request exceeded the table."""
    force_n: float
    requested_rpm: float
    calibration_rpm: float
    outside_calibration: bool


class ForceCurve:
    """Piecewise linear in current, then speed; no fitted or smoothed curve."""

    def __init__(self, points: dict[float, dict[float, float]] | None = None):
        if points is None:
            points = {
                float(rpm): dict(zip(FORCE_CURVE_CURRENTS_PERCENT, forces, strict=True))
                for rpm, forces in FORCE_CURVE_TABLE_N.items()
            }
        self.points: dict[float, dict[float, float]] = {}
        for rpm, curve in points.items():
            rpm = float(rpm)
            if not math.isfinite(rpm):
                raise ValueError("invalid force calibration speed")
            row: dict[float, float] = {}
            for current, force in curve.items():
                current, force = float(current), float(force)
                if not math.isfinite(current) or not math.isfinite(force) or force <= 0:
                    raise ValueError("invalid force calibration point")
                row[current] = force
            self.points[rpm] = row
        self.speeds = sorted(self.points)
        if len(self.speeds) < 2 or self.speeds[0] != 0 or self.speeds[-1] != 3500:
            raise ValueError("force curves must cover 0–3500 rpm")
        for curve in self.points.values():
            if len(curve) < 2 or min(curve) != 0 or max(curve) != 100:
                raise ValueError("each force curve must cover 0–100% current")

    @property
    def maximum_force_n(self) -> float:
        """Largest calibrated pinch force (N)."""
        return max(force for curve in self.points.values() for force in curve.values())

    @staticmethod
    def _bracket(knots: list[float], value: float) -> tuple[float, float, float]:
        if value in knots:
            return value, value, 0.0
        upper = bisect_right(knots, value)
        lo, hi = knots[upper - 1], knots[upper]
        return lo, hi, (value - lo) / (hi - lo)

    def _at_speed(self, rpm: float, current: float) -> float:
        points = self.points[rpm]
        lo, hi, ratio = self._bracket(sorted(points), current)
        a, b = points[lo], points[hi]
        return a + ratio * (b - a)

    def lookup(self, rpm: float, force_percent: float) -> ForceLookup:
        """Interpolate the pinch force at ``rpm`` and ``force_percent``; speeds above the table are clamped."""
        current = checked(force_percent, 1, 100, "force_percent")
        rpm = checked(rpm, 0, 1e9, "rpm")
        calibration_rpm = min(rpm, self.speeds[-1])
        lo, hi, ratio = self._bracket(self.speeds, calibration_rpm)
        a, b = self._at_speed(lo, current), self._at_speed(hi, current)
        return ForceLookup(
            a + ratio * (b - a),
            rpm,
            calibration_rpm,
            rpm > self.speeds[-1],
        )


@dataclass(frozen=True)
class G2Model:
    """Calibrated G2 linkage and force curve, referred to the drive joint.

    A force sensor between two symmetric pads reads the load at either pad, so
    virtual work gives tau_drive = F_sensor * abs(dgap/dq) without an extra
    factor of two. The calibrated torque gain combines reduction and efficiency;
    it is not an identified mechanical gear ratio.
    """

    linkage: G2LinkageGeometry = G2_LINKAGE_GEOMETRY
    curve: ForceCurve = field(default_factory=ForceCurve)
    rated_motor_torque_nm: float = 0.22
    peak_motor_torque_nm: float = 0.5
    reference_raw_position: int = REFERENCE_RAW_POSITION

    @classmethod
    @lru_cache(maxsize=1)
    def load(cls) -> G2Model:
        """Shared default model (cached)."""
        return cls(G2_LINKAGE_GEOMETRY, ForceCurve())

    @property
    def reference_gap_mm(self) -> float:
        """Gap (mm) at the calibration reference position."""
        return sdk_position_mm(self.reference_raw_position)

    @property
    def reference_jacobian_m_rad(self) -> float:
        """|d gap / d q| (m/rad) at the calibration reference position."""
        q = self.linkage.drive_for_gap_m(self.reference_gap_mm / 1000)
        return self.linkage.gap_jacobian_m_per_rad(q)

    @property
    def effective_torque_gain(self) -> float:
        """Drive-torque per motor-torque gain implied by the peak force and peak motor torque."""
        return self.curve.maximum_force_n * self.reference_jacobian_m_rad / self.peak_motor_torque_nm

    def torque_limit_nm(self, lookup: ForceLookup) -> float:
        """Drive torque limit (N*m) for a force lookup."""
        return lookup.force_n * self.reference_jacobian_m_rad

    def static_force_n(self, lookup: ForceLookup, gap_mm: float) -> float:
        """Static pinch force (N) at ``gap_mm`` for the torque limit of ``lookup``."""
        q = self.linkage.drive_for_gap_m(checked(gap_mm, 0, 84, "gap_mm") / 1000)
        return self.torque_limit_nm(lookup) / self.linkage.gap_jacobian_m_per_rad(q)

    def motor_torque_estimate_nm(self, drive_torque_nm: float) -> float:
        """Estimate motor torque (N*m) from a drive-joint torque."""
        return float(drive_torque_nm) / self.effective_torque_gain


# ----------------------------------------------------------------------
# Servo controller
# ----------------------------------------------------------------------

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
