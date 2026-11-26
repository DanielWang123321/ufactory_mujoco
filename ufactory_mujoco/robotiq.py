"""Robotiq 2F-85 and 2F-140 linkage geometry, units, force curve, and PI+PD servo controller.

Pure Python/NumPy port of ``ufactory_genesis`` Robotiq modules (``robotiq.py``,
``robotiq_model.py``, and ``robotiq_controller.py``), with zero PyTorch or URDF dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any

import numpy as np

from .g2 import ControlOutput, checked

ROBOTIQ_2F_85_PREFIX = "robotiq_2f_85"
ROBOTIQ_2F_140_PREFIX = "robotiq_2f_140"
ROBOTIQ_ADAPTERS = frozenset((ROBOTIQ_2F_85_PREFIX, ROBOTIQ_2F_140_PREFIX))

ROBOTIQ_JOINT_SUFFIXES = (
    "drive_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
)


def robotiq_joint_names(prefix: str) -> tuple[str, ...]:
    """Return the 6 active + mimic joint names for a prefixed Robotiq gripper."""
    return tuple(f"{prefix}_{suffix}" for suffix in ROBOTIQ_JOINT_SUFFIXES)


ROBOTIQ_2F_85_JOINT_NAMES = robotiq_joint_names(ROBOTIQ_2F_85_PREFIX)
ROBOTIQ_2F_140_JOINT_NAMES = robotiq_joint_names(ROBOTIQ_2F_140_PREFIX)

ROBOTIQ_2F_85_CLOSED_GAP_MM = 0.0
ROBOTIQ_2F_85_OPEN_GAP_MM = 85.0
ROBOTIQ_2F_85_OPEN_GAP_M = ROBOTIQ_2F_85_OPEN_GAP_MM / 1000.0
ROBOTIQ_2F_85_SIM_OPEN_DRIVE = 0.0
ROBOTIQ_2F_85_SIM_CLOSE_DRIVE = 0.8014334346423151

ROBOTIQ_2F_140_CLOSED_GAP_MM = 0.0
ROBOTIQ_2F_140_OPEN_GAP_MM = 140.0
ROBOTIQ_2F_140_OPEN_GAP_M = ROBOTIQ_2F_140_OPEN_GAP_MM / 1000.0
ROBOTIQ_2F_140_SIM_OPEN_DRIVE = 0.0038593285902723817
ROBOTIQ_2F_140_SIM_CLOSE_DRIVE = 0.7665940156296729

ROBOTIQ_KP = 120.0
ROBOTIQ_KV = 10.0
ROBOTIQ_KI = 160.0
ROBOTIQ_ARMATURE = 0.02
ROBOTIQ_DAMPING = 0.1


@dataclass(frozen=True)
class RobotiqLinkageGeometry:
    """Four-bar parallel-pad linkage dimensions from the Robotiq URDF and pad mesh."""

    base_joint_y_m: float
    finger_joint_y_m: float
    finger_joint_z_m: float
    pad_inner_y_m: float
    pad_center_z_m: float
    drive_lower_rad: float
    drive_upper_rad: float

    @property
    def radius_m(self) -> float:
        return math.hypot(self.finger_joint_y_m, self.finger_joint_z_m)

    @property
    def phase_rad(self) -> float:
        return math.atan2(self.finger_joint_z_m, self.finger_joint_y_m)

    def pad_gap_m(self, drive_rad: float) -> float:
        """Full two-pad separation (m) for a drive joint angle (rad)."""
        q = float(drive_rad)
        return 2.0 * (
            self.base_joint_y_m
            + self.finger_joint_y_m * math.cos(q)
            - self.finger_joint_z_m * math.sin(q)
            + self.pad_inner_y_m
        )

    def gap_jacobian_m_per_rad(self, drive_rad: float) -> float:
        """Absolute derivative ``|d gap / d q|`` (m/rad)."""
        q = float(drive_rad)
        return abs(2.0 * (self.finger_joint_y_m * math.sin(q) + self.finger_joint_z_m * math.cos(q)))

    def drive_for_gap_m(self, gap_m: float) -> float:
        """Inverse linkage: pad gap (m) -> drive joint angle (rad)."""
        gap = float(gap_m)
        lo = self.pad_gap_m(self.drive_upper_rad) - 1e-6
        hi = self.pad_gap_m(self.drive_lower_rad) + 1e-6
        if not math.isfinite(gap) or not lo <= gap <= hi:
            raise ValueError("calibration gap is outside the collision pads' physical range")
        cosine = (gap / 2.0 - self.base_joint_y_m - self.pad_inner_y_m) / self.radius_m
        q = math.acos(max(-1.0, min(1.0, cosine))) - self.phase_rad
        return max(self.drive_lower_rad, min(self.drive_upper_rad, q))


ROBOTIQ_2F_85_LINKAGE = RobotiqLinkageGeometry(
    base_joint_y_m=0.03060114,
    finger_joint_y_m=0.0371575,
    finger_joint_z_m=0.04342168,
    pad_inner_y_m=-0.02525864541530609,
    pad_center_z_m=0.032010000000000004,
    drive_lower_rad=ROBOTIQ_2F_85_SIM_OPEN_DRIVE,
    drive_upper_rad=ROBOTIQ_2F_85_SIM_CLOSE_DRIVE,
)

ROBOTIQ_2F_140_LINKAGE = RobotiqLinkageGeometry(
    base_joint_y_m=0.030601,
    finger_joint_y_m=0.066855,
    finger_joint_z_m=0.074366,
    pad_inner_y_m=-0.02716850000000001,
    pad_center_z_m=0.04772499999999999,
    drive_lower_rad=ROBOTIQ_2F_140_SIM_OPEN_DRIVE,
    drive_upper_rad=ROBOTIQ_2F_140_SIM_CLOSE_DRIVE,
)


@dataclass(frozen=True)
class RobotiqSpec:
    """Variant specification and conversions for one Robotiq 2F adaptive gripper."""

    adapter: str
    display_name: str
    prefix: str
    closed_gap_mm: float
    open_gap_mm: float
    speed_min_mm_s: float
    speed_max_mm_s: float
    force_min_n: float
    force_max_n: float
    linkage: RobotiqLinkageGeometry

    @property
    def closed_gap_m(self) -> float:
        return self.closed_gap_mm / 1000.0

    @property
    def open_gap_m(self) -> float:
        return self.open_gap_mm / 1000.0

    @property
    def joint_names(self) -> tuple[str, ...]:
        return robotiq_joint_names(self.prefix)

    @property
    def drive_joint_name(self) -> str:
        return f"{self.prefix}_drive_joint"

    def clamp_gap_m(self, gap_m: float) -> float:
        value = float(gap_m)
        if not math.isfinite(value):
            raise ValueError("gap_m must be finite")
        return max(self.closed_gap_m, min(self.open_gap_m, value))

    def gap_m_to_sim_drive(self, gap_m: float) -> float:
        return self.linkage.drive_for_gap_m(self.clamp_gap_m(gap_m))

    def sim_drive_to_gap_m(self, drive: float) -> float:
        value = float(drive)
        if not math.isfinite(value):
            raise ValueError("drive must be finite")
        clipped = max(self.linkage.drive_lower_rad, min(self.linkage.drive_upper_rad, value))
        return self.clamp_gap_m(self.linkage.pad_gap_m(clipped))

    def gap_m_to_register(self, gap_m: float) -> int:
        """Map pad gap in metres to xArm SDK Robotiq register ``[0=open .. 255=closed]``."""
        clamped = self.clamp_gap_m(gap_m)
        ratio_closed = 1.0 - (clamped - self.closed_gap_m) / (self.open_gap_m - self.closed_gap_m)
        return int(max(0, min(255, round(ratio_closed * 255.0))))

    def register_to_gap_m(self, register: int) -> float:
        """Map xArm SDK Robotiq register ``[0=open .. 255=closed]`` to pad gap in metres."""
        clamped = max(0, min(255, int(register)))
        ratio_open = 1.0 - float(clamped) / 255.0
        return self.closed_gap_m + ratio_open * (self.open_gap_m - self.closed_gap_m)

    def speed_percent_to_mm_s(self, speed_percent: float) -> float:
        pct = max(0.0, min(100.0, float(speed_percent)))
        return self.speed_min_mm_s + (pct / 100.0) * (self.speed_max_mm_s - self.speed_min_mm_s)

    def force_percent_to_n(self, force_percent: float) -> float:
        pct = max(1.0, min(100.0, float(force_percent)))
        return self.force_min_n + ((pct - 1.0) / 99.0) * (self.force_max_n - self.force_min_n)


ROBOTIQ_2F_85_SPEC = RobotiqSpec(
    adapter="robotiq_2f_85",
    display_name="Robotiq 2F-85",
    prefix=ROBOTIQ_2F_85_PREFIX,
    closed_gap_mm=ROBOTIQ_2F_85_CLOSED_GAP_MM,
    open_gap_mm=ROBOTIQ_2F_85_OPEN_GAP_MM,
    speed_min_mm_s=20.0,
    speed_max_mm_s=150.0,
    force_min_n=20.0,
    force_max_n=235.0,
    linkage=ROBOTIQ_2F_85_LINKAGE,
)

ROBOTIQ_2F_140_SPEC = RobotiqSpec(
    adapter="robotiq_2f_140",
    display_name="Robotiq 2F-140",
    prefix=ROBOTIQ_2F_140_PREFIX,
    closed_gap_mm=ROBOTIQ_2F_140_CLOSED_GAP_MM,
    open_gap_mm=ROBOTIQ_2F_140_OPEN_GAP_MM,
    speed_min_mm_s=30.0,
    speed_max_mm_s=250.0,
    force_min_n=10.0,
    force_max_n=125.0,
    linkage=ROBOTIQ_2F_140_LINKAGE,
)

ROBOTIQ_SPECS: dict[str, RobotiqSpec] = {
    "robotiq_2f_85": ROBOTIQ_2F_85_SPEC,
    "robotiq_2f_140": ROBOTIQ_2F_140_SPEC,
}


def get_robotiq_spec(variant: str) -> RobotiqSpec:
    """Return the Robotiq variant spec for ``robotiq_2f_85`` or ``robotiq_2f_140``."""
    try:
        return ROBOTIQ_SPECS[str(variant)]
    except KeyError as exc:
        raise ValueError(f"unknown Robotiq variant: {variant}") from exc


def robotiq_percent_to_register(percent: float) -> int:
    """Convert a percentage ``[0..100]`` to a Robotiq 8-bit register value ``[0..255]``."""
    value = float(percent)
    if not math.isfinite(value):
        raise ValueError("percent must be finite")
    return int(max(0, min(255, round(max(0.0, min(100.0, value)) * 255.0 / 100.0))))


def robotiq_gap_m_to_sim_drive(variant: str, gap_m: float) -> float:
    return get_robotiq_spec(variant).gap_m_to_sim_drive(gap_m)


def robotiq_sim_drive_to_gap_m(variant: str, drive: float) -> float:
    return get_robotiq_spec(variant).sim_drive_to_gap_m(drive)


def robotiq_gap_m_to_register(variant: str, gap_m: float) -> int:
    return get_robotiq_spec(variant).gap_m_to_register(gap_m)


def robotiq_register_to_gap_m(variant: str, register: int) -> float:
    return get_robotiq_spec(variant).register_to_gap_m(register)


def robotiq_speed_mm_s(spec: RobotiqSpec, speed_percent: float) -> float:
    """Map speed percentage ``[0..100]`` to Robotiq jaw speed (mm/s)."""
    speed = checked(speed_percent, 0.0, 100.0, "speed_percent")
    return spec.speed_percent_to_mm_s(speed) if speed > 0.0 else 0.0


# ----------------------------------------------------------------------
# Force curve and calibrated model
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class RobotiqForceLookup:
    force_n: float
    requested_speed_mm_s: float
    force_percent: float
    outside_calibration: bool = False


class RobotiqForceCurve:
    """Linear datasheet steady-force curve across ``force_percent`` [1..100]."""

    def __init__(self, force_min_n: float, force_max_n: float) -> None:
        self.force_min_n = checked(force_min_n, 1e-6, 1e4, "force_min_n")
        self.force_max_n = checked(force_max_n, self.force_min_n, 1e4, "force_max_n")

    @property
    def maximum_force_n(self) -> float:
        return self.force_max_n

    def lookup(self, speed_mm_s: float, force_percent: float) -> RobotiqForceLookup:
        requested_speed = checked(speed_mm_s, 0.0, 1e4, "speed_mm_s")
        pct = checked(force_percent, 1.0, 100.0, "force_percent")
        ratio = (pct - 1.0) / 99.0
        force_n = self.force_min_n + ratio * (self.force_max_n - self.force_min_n)
        return RobotiqForceLookup(
            force_n=force_n,
            requested_speed_mm_s=requested_speed,
            force_percent=pct,
            outside_calibration=False,
        )


@dataclass(frozen=True)
class RobotiqModel:
    """Calibrated Robotiq 2F linkage and force curve, referred to the drive joint."""

    spec: RobotiqSpec
    linkage: RobotiqLinkageGeometry
    curve: RobotiqForceCurve
    reference_gap_mm: float

    @classmethod
    @lru_cache(maxsize=2)
    def load(cls, variant: str = "robotiq_2f_85") -> RobotiqModel:
        spec = get_robotiq_spec(variant)
        return cls(
            spec=spec,
            linkage=spec.linkage,
            curve=RobotiqForceCurve(spec.force_min_n, spec.force_max_n),
            reference_gap_mm=spec.open_gap_mm * 0.5,
        )

    @property
    def reference_jacobian_m_rad(self) -> float:
        q = self.linkage.drive_for_gap_m(self.reference_gap_mm / 1000.0)
        return self.linkage.gap_jacobian_m_per_rad(q)

    def torque_limit_nm(self, lookup: RobotiqForceLookup) -> float:
        return lookup.force_n * self.reference_jacobian_m_rad

    def static_force_n(self, lookup: RobotiqForceLookup, gap_mm: float) -> float:
        q = self.linkage.drive_for_gap_m(checked(gap_mm, 0.0, self.spec.open_gap_mm, "gap_mm") / 1000.0)
        return self.torque_limit_nm(lookup) / self.linkage.gap_jacobian_m_per_rad(q)

    def motor_torque_estimate_nm(self, drive_torque_nm: float) -> float:
        peak_nm = self.curve.maximum_force_n * self.reference_jacobian_m_rad
        if peak_nm <= 0.0 or not math.isfinite(peak_nm):
            return float(drive_torque_nm)
        return float(drive_torque_nm) / peak_nm


# ----------------------------------------------------------------------
# Vectorized NumPy controller
# ----------------------------------------------------------------------

def robotiq_drive_for_gap_m(
    gap_m: np.ndarray,
    geometry: RobotiqLinkageGeometry,
    max_gap_m: float,
) -> np.ndarray:
    """Vectorized inverse linkage: pad gap (m) -> drive joint angle (rad)."""
    g = geometry
    clamped_gap = np.clip(gap_m, 0.0, max_gap_m)
    cosine = (clamped_gap / 2.0 - g.base_joint_y_m - g.pad_inner_y_m) / g.radius_m
    angles = np.arccos(np.clip(cosine, -1.0, 1.0)) - g.phase_rad
    return np.clip(angles, g.drive_lower_rad, g.drive_upper_rad).astype(gap_m.dtype, copy=False)


def robotiq_pad_gap_m(drive_rad: np.ndarray, geometry: RobotiqLinkageGeometry) -> np.ndarray:
    """Vectorized forward linkage: drive joint angle (rad) -> pad gap (m)."""
    g = geometry
    gap = 2.0 * (
        g.base_joint_y_m
        + g.finger_joint_y_m * np.cos(drive_rad)
        - g.finger_joint_z_m * np.sin(drive_rad)
        + g.pad_inner_y_m
    )
    return np.asarray(gap, dtype=drive_rad.dtype)


class RobotiqController:
    """Calibrated Robotiq 2F-85 / 2F-140 servo; every state and output is ``(num_envs,)``."""

    def __init__(
        self,
        model: RobotiqModel | str | None = None,
        *,
        num_envs: int = 1,
        dtype: Any = np.float64,
        initial_position_mm: float | None = None,
        kp: float = ROBOTIQ_KP,
        kv: float = ROBOTIQ_KV,
        ki: float = ROBOTIQ_KI,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if model is None:
            resolved = RobotiqModel.load("robotiq_2f_85")
        elif isinstance(model, str):
            resolved = RobotiqModel.load(model)
        else:
            resolved = model
        self.model = resolved
        self.num_envs = int(num_envs)
        self.dtype = np.dtype(dtype)
        self.kp = checked(kp, 1e-9, 1e6, "kp")
        self.kv = checked(kv, 0.0, 1e6, "kv")
        self.ki = checked(ki, 0.0, 1e6, "ki")
        self.open_gap_mm = float(resolved.spec.open_gap_mm)
        self.open_gap_m = float(resolved.spec.open_gap_m)
        initial_raw = self.open_gap_mm if initial_position_mm is None else initial_position_mm
        initial = checked(initial_raw, 0.0, self.open_gap_mm, "initial_position_mm")
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

    def speed_mm_s(self, percent: np.ndarray) -> np.ndarray:
        """Map speed percent ``[0..100]`` to jaw speed (mm/s)."""
        lo = self.dtype.type(self.model.spec.speed_min_mm_s)
        hi = self.dtype.type(self.model.spec.speed_max_mm_s)
        scaled = lo + (percent / self.dtype.type(100.0)) * (hi - lo)
        return np.where(percent > 0.0, scaled, 0.0).astype(self.dtype, copy=False)

    @property
    def requested_speed_mm_s(self) -> np.ndarray:
        return self.speed_mm_s(self.speed_percent)

    def force_n(self) -> np.ndarray:
        lo = self.dtype.type(self.model.spec.force_min_n)
        hi = self.dtype.type(self.model.spec.force_max_n)
        scaled = lo + ((self.force_percent - self.dtype.type(1.0)) / self.dtype.type(99.0)) * (hi - lo)
        return scaled.astype(self.dtype, copy=False)

    def force_lookup(self, env: int = 0) -> RobotiqForceLookup:
        """Calibration lookup of one environment, for telemetry."""
        speed = float(self.speed_mm_s(self.last_nonzero_speed_percent)[env])
        return self.model.curve.lookup(speed, float(self.force_percent[env]))

    @property
    def torque_limit_nm(self) -> np.ndarray:
        ref_jac = self.dtype.type(self.model.reference_jacobian_m_rad)
        return (self.force_n() * ref_jac).astype(self.dtype, copy=False)

    def drive_for_gap_mm(self, gap: Any) -> np.ndarray:
        arr = np.asarray(gap, dtype=self.dtype)
        return robotiq_drive_for_gap_m(arr / self.dtype.type(1000.0), self.model.linkage, self.open_gap_m)

    def gap_mm(self, q: Any) -> np.ndarray:
        arr = np.asarray(q, dtype=self.dtype)
        return (self.dtype.type(1000.0) * robotiq_pad_gap_m(arr, self.model.linkage)).astype(self.dtype, copy=False)

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
        position = self._vector(position_mm, "position_mm", size=size, bounds=(0.0, self.open_gap_mm))
        speed = self._vector(speed_percent, "speed_percent", size=size, bounds=(0.0, 100.0))
        force = self._vector(force_percent, "force_percent", size=size, bounds=(1.0, 100.0))
        old_limit = self.torque_limit_nm[idx]
        old_direction = self.position_mm[idx] - self.reference_gap_mm[idx]
        new_direction = position - self.reference_gap_mm[idx]
        reverse = (speed > 0.0) & (
            (old_direction * new_direction < 0.0) | (new_direction * self.integral_torque_nm[idx] > 0.0)
        )
        pause = (speed == 0.0) & (self.speed_percent[idx] > 0.0)
        self._reverse_pending[idx] = self._reverse_pending[idx] | reverse
        self.integral_torque_nm[idx] = np.where(reverse, 0.0, self.integral_torque_nm[idx])
        self._pause_pending[idx] = self._pause_pending[idx] | pause
        self._pause_start_limit_nm[idx] = np.where(pause, old_limit, self._pause_start_limit_nm[idx])
        self._pause_start_integral_nm[idx] = np.where(
            pause, self.integral_torque_nm[idx], self._pause_start_integral_nm[idx]
        )
        self.last_nonzero_speed_percent[idx] = np.where(speed > 0.0, speed, self.last_nonzero_speed_percent[idx])
        self._paused_hold_sign[idx] = np.where(speed > 0.0, 0.0, self._paused_hold_sign[idx])
        self.position_mm[idx], self.speed_percent[idx], self.force_percent[idx] = position, speed, force
        limit = self.torque_limit_nm[idx]
        self.integral_torque_nm[idx] = np.clip(self.integral_torque_nm[idx], -limit, limit)

    def reset(
        self,
        envs_idx: Any = None,
        *,
        position_mm: Any = None,
        target_mm: Any = None,
        prime: bool = False,
    ) -> None:
        """Reset the reference; optionally seed a saturated pre-held grasp."""
        idx = slice(None) if envs_idx is None else envs_idx
        size = int(self.position_mm[idx].size)
        pos_val = self.open_gap_mm if position_mm is None else position_mm
        position = self._vector(pos_val, "position_mm", size=size, bounds=(0.0, self.open_gap_mm))
        target = (
            position
            if target_mm is None
            else self._vector(target_mm, "target_mm", size=size, bounds=(0.0, self.open_gap_mm))
        )
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
        measured = np.clip(measured, 0.0, self.open_gap_mm).astype(self.dtype, copy=False)
        limit = self.torque_limit_nm
        old_q_ref = self.drive_for_gap_mm(self.reference_gap_mm)
        self.reference_gap_mm = np.where(self._reverse_pending, measured, self.reference_gap_mm).astype(
            self.dtype, copy=False
        )
        self.integral_torque_nm = np.where(self._reverse_pending, 0.0, self.integral_torque_nm).astype(
            self.dtype, copy=False
        )
        self._reverse_pending.fill(False)
        preload = self._pause_start_integral_nm + self.dtype.type(self.kp) * (old_q_ref - q)
        self.integral_torque_nm = np.where(self._pause_pending, preload, self.integral_torque_nm).astype(
            self.dtype, copy=False
        )
        hold_sign = np.where(
            (np.abs(velocity) <= 0.02) & (np.abs(preload) >= 0.95 * self._pause_start_limit_nm),
            np.sign(preload),
            0.0,
        )
        self._paused_hold_sign = np.where(self._pause_pending, hold_sign, self._paused_hold_sign).astype(
            self.dtype, copy=False
        )
        paused_gap = np.where(np.abs(velocity) > 0.02, measured, np.clip(self.gap_mm(q), 0.0, self.open_gap_mm))
        self.reference_gap_mm = np.where(self._pause_pending, paused_gap, self.reference_gap_mm).astype(
            self.dtype, copy=False
        )
        self._pause_pending.fill(False)
        previous_q_ref = self.drive_for_gap_mm(self.reference_gap_mm)
        travel = self.speed_mm_s(self.speed_percent) * dt_val
        self.reference_gap_mm = (
            self.reference_gap_mm + np.clip(self.position_mm - self.reference_gap_mm, -travel, travel)
        ).astype(self.dtype, copy=False)
        q_ref = self.drive_for_gap_mm(self.reference_gap_mm)
        v_ref = (q_ref - previous_q_ref) / dt_val
        error = q_ref - q
        integral = np.clip(self.integral_torque_nm, -limit, limit)
        candidate = np.clip(integral + self.dtype.type(self.ki) * error * dt_val, -limit, limit)
        candidate_total = (
            self.dtype.type(self.kp) * error + self.dtype.type(self.kv) * (v_ref - velocity) + candidate
        )
        integral = np.where(
            (np.abs(candidate_total) <= limit) | (error * candidate_total <= 0.0),
            candidate,
            integral,
        )
        holding = (self.speed_percent == 0.0) & (self._paused_hold_sign != 0.0)
        self.integral_torque_nm = np.where(holding, self._paused_hold_sign * limit, integral).astype(
            self.dtype, copy=False
        )
        target = (np.where(holding, q, q_ref) + self.integral_torque_nm / self.dtype.type(self.kp)).astype(
            self.dtype, copy=False
        )
        vel_out = np.where(holding, 0.0, v_ref).astype(self.dtype, copy=False)
        return ControlOutput(
            target,
            vel_out,
            self.reference_gap_mm.copy(),
            self.integral_torque_nm.copy(),
            limit.copy(),
        )
