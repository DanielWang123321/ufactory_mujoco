"""Per-unit kinematic calibration loader, MuJoCo model patching, and TCP transforms.

Schema v1 YAML format (aligned with ufactory_genesis):
  schema_version: 1
  robot_key: xarm6_1305   # or canonical xarm6 / lite6 / xarm5 / xarm7 / uf850
  serial_number: XI130505L56A01
  units:
    position: m
    angle: rad
  joints:
    joint1..N: {x, y, z, roll, pitch, yaw}
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

from .config import ConfigError
from .robots import ARMS

_CALIBRATION_ROOT_KEYS = frozenset({"schema_version", "robot_key", "serial_number", "units", "joints"})
_JOINT_AXIS_KEYS = frozenset({"x", "y", "z", "roll", "pitch", "yaw"})
_FIRMWARE_KEY_MAP = {
    "xarm5_1305": "xarm5",
    "xarm6_1305": "xarm6",
    "xarm7_1305": "xarm7",
}

# SN positions 3-6 (1-based) = four-digit model code, e.g. XI130505L56A01 -> 1305
XARM_KINEMATICS_MIN_SN_MODEL_CODE = 1304
LITE6_KINEMATICS_MIN_SN_MODEL_CODE = 1006


class CalibrationError(ConfigError):
    """Raised when kinematic calibration is missing, invalid, or mismatches hardware."""


@dataclass(frozen=True)
class KinematicsCalibration:
    """Strictly validated, robot-bound kinematic calibration."""

    schema_version: int
    robot_key: str
    raw_robot_key: str
    serial_number: str
    position_unit: str
    angle_unit: str
    joints: dict[str, dict[str, float]]
    source_path: Path
    sha256: str


def canonical_robot_key(raw_key: str) -> str:
    """Map firmware/URDF variant keys (e.g. ``xarm6_1305``) to ``ARMS`` keys."""
    if not isinstance(raw_key, str) or not raw_key.strip():
        raise CalibrationError("calibration robot_key must be a non-empty string")
    cleaned = raw_key.strip()
    mapped = _FIRMWARE_KEY_MAP.get(cleaned, cleaned)
    if mapped not in ARMS:
        raise CalibrationError(f"unknown calibration robot_key: {raw_key!r}")
    return mapped


def parse_sn_model_code(sn: str) -> int | None:
    """Extract 4-digit model code from SN characters [2:6], or None if non-numeric."""
    text = (sn or "").strip()
    if len(text) >= 6 and text[2:6].isdigit():
        return int(text[2:6])
    return None


def has_per_unit_kinematics_calibration(sn: str, robot_name: str) -> bool:
    """Whether factory firmware stores per-unit kinematic compensation for this SN."""
    key = canonical_robot_key(robot_name)
    if key == "uf850":
        return True
    code = parse_sn_model_code(sn)
    if code is None:
        return False
    if key == "lite6":
        return code >= LITE6_KINEMATICS_MIN_SN_MODEL_CODE
    if key in ("xarm5", "xarm6", "xarm7"):
        return code >= XARM_KINEMATICS_MIN_SN_MODEL_CODE
    return False


def kinematics_suffix_from_sn(sn: str) -> str:
    """Return the last 6 characters of a valid robot serial number."""
    cleaned = (sn or "").strip()
    if len(cleaned) < 8:
        raise CalibrationError(f"invalid or incomplete serial number: {sn!r}")
    return cleaned[-6:]


def get_robot_sn(arm: Any) -> str:
    """Read and validate the full serial number from an XArmAPI instance."""
    getter = getattr(arm, "get_robot_sn", None)
    if not callable(getter):
        raise CalibrationError("controller API does not provide get_robot_sn()")
    res = getter()
    if isinstance(res, tuple) and len(res) == 2:
        code, sn = res
    else:
        code, sn = 0, res
    sn_text = str(sn or "").strip().rstrip("\x00")
    if int(code) != 0 or len(sn_text) < 8:
        raise CalibrationError(f"failed to read valid serial number from controller (code={code}, sn={sn!r})")
    return sn_text


def read_tcp_offset(arm: Any) -> np.ndarray:
    """Read 6-DOF controller TCP offset ``[x_mm, y_mm, z_mm, roll_rad, pitch_rad, yaw_rad]``."""
    raw = getattr(arm, "tcp_offset", None)
    if raw is None:
        return np.zeros(6, dtype=np.float64)
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.shape != (6,) or not np.all(np.isfinite(arr)):
        raise CalibrationError(f"invalid controller tcp_offset: {raw!r}")
    return arr.copy()


def _validate_joints(
    joints: Mapping[str, Any],
    *,
    dof: int,
    source: str,
) -> dict[str, dict[str, float]]:
    expected = tuple(f"joint{i}" for i in range(1, dof + 1))
    actual = set(map(str, joints.keys()))
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise CalibrationError(
            f"{source}: calibration joint set mismatch; missing={missing}, extra={extra}"
        )
    validated: dict[str, dict[str, float]] = {}
    for name in expected:
        entry = joints[name]
        if not isinstance(entry, Mapping):
            raise CalibrationError(f"{source}: {name} must be a mapping")
        keys = set(map(str, entry.keys()))
        if keys != _JOINT_AXIS_KEYS:
            raise CalibrationError(
                f"{source}: {name} keys must be {sorted(_JOINT_AXIS_KEYS)}, got {sorted(keys)}"
            )
        axis_map: dict[str, float] = {}
        for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
            val = entry[axis]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise CalibrationError(f"{source}: {name}.{axis} must be numeric")
            fval = float(val)
            if not math.isfinite(fval):
                raise CalibrationError(f"{source}: {name}.{axis} must be finite")
            limit = 2.0 if axis in ("x", "y", "z") else 2.0 * math.pi
            if abs(fval) > limit:
                raise CalibrationError(
                    f"{source}: {name}.{axis}={fval} exceeds bound ±{limit}"
                )
            axis_map[axis] = fval
        validated[name] = axis_map
    return validated


def load_kinematics_calibration(
    yaml_path: str | Path,
    *,
    robot: str | None = None,
    serial_number: str | None = None,
) -> KinematicsCalibration:
    """Load and validate a schema-v1 kinematic calibration YAML file."""
    if not yaml_path:
        raise CalibrationError("calibration YAML path is required")
    path = Path(yaml_path).expanduser().resolve()
    if not path.is_file():
        raise CalibrationError(f"calibration YAML not found: {path}")
    try:
        raw_bytes = path.read_bytes()
        data = yaml.safe_load(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise CalibrationError(f"failed to read calibration YAML {path}: {exc}") from exc

    if not isinstance(data, Mapping):
        raise CalibrationError(f"{path}: calibration YAML root must be a mapping")
    root_keys = set(map(str, data.keys()))
    if root_keys != _CALIBRATION_ROOT_KEYS:
        raise CalibrationError(
            f"{path}: root fields must be {sorted(_CALIBRATION_ROOT_KEYS)}, got {sorted(root_keys)}"
        )
    version = data["schema_version"]
    if type(version) is not int or version != 1:
        raise CalibrationError(f"{path}: schema_version must be integer 1")

    raw_robot_key = str(data["robot_key"]).strip()
    robot_key = canonical_robot_key(raw_robot_key)
    if robot is not None:
        expected_robot = canonical_robot_key(robot)
        if robot_key != expected_robot:
            raise CalibrationError(
                f"{path}: calibration robot_key {robot_key!r} does not match requested {expected_robot!r}"
            )

    file_sn = str(data["serial_number"]).strip()
    if file_sn != "NOMINAL" and len(file_sn) < 8:
        raise CalibrationError(f"{path}: serial_number must be complete (>= 8 chars), got {file_sn!r}")
    if serial_number is not None:
        expected_sn = str(serial_number).strip()
        if file_sn != expected_sn:
            raise CalibrationError(
                f"{path}: calibration serial_number {file_sn!r} does not match robot {expected_sn!r}"
            )

    units = data["units"]
    if not isinstance(units, Mapping) or set(map(str, units.keys())) != {"position", "angle"}:
        raise CalibrationError(f"{path}: units must contain position and angle")
    if units["position"] != "m" or units["angle"] != "rad":
        raise CalibrationError(f"{path}: units must be position='m' and angle='rad'")

    joints_raw = data["joints"]
    if not isinstance(joints_raw, Mapping):
        raise CalibrationError(f"{path}: joints must be a mapping")
    dof = ARMS[robot_key].dof
    joints = _validate_joints(joints_raw, dof=dof, source=str(path))
    digest = hashlib.sha256(raw_bytes).hexdigest()

    return KinematicsCalibration(
        schema_version=1,
        robot_key=robot_key,
        raw_robot_key=raw_robot_key,
        serial_number=file_sn,
        position_unit="m",
        angle_unit="rad",
        joints=joints,
        source_path=path,
        sha256=digest,
    )


def rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert extrinsic XYZ (URDF/xArm) roll-pitch-yaw (rad) to quaternion ``[w, x, y, z]``."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    quat = np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def quat_wxyz_to_rpy(quat: Sequence[float]) -> np.ndarray:
    """Convert quaternion ``[w, x, y, z]`` to extrinsic XYZ roll-pitch-yaw (rad)."""
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    q = q / np.linalg.norm(q)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi * 0.5, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert roll-pitch-yaw (rad) to a 3x3 rotation matrix ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def matrix_to_rpy(rot: np.ndarray) -> np.ndarray:
    """Extract roll-pitch-yaw (rad) from a 3x3 rotation matrix."""
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(rot, dtype=np.float64).reshape(9))
    return quat_wxyz_to_rpy(quat)


def _pose_to_se3(pose_mm_rad: Sequence[float]) -> np.ndarray:
    arr = np.asarray(pose_mm_rad, dtype=np.float64).reshape(6)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rpy_to_matrix(float(arr[3]), float(arr[4]), float(arr[5]))
    mat[:3, 3] = arr[:3]
    return mat


def _se3_to_pose(mat: np.ndarray) -> np.ndarray:
    rpy = matrix_to_rpy(mat[:3, :3])
    return np.array([mat[0, 3], mat[1, 3], mat[2, 3], rpy[0], rpy[1], rpy[2]], dtype=np.float64)


def pose_flange_to_tcp(
    flange_pose_mm_rad: Sequence[float],
    tcp_offset_mm_rad: Sequence[float] | None = None,
) -> np.ndarray:
    """Transform flange pose ``[x_mm, y_mm, z_mm, r, p, y]`` to controller TCP frame."""
    if tcp_offset_mm_rad is None:
        return np.asarray(flange_pose_mm_rad, dtype=np.float64).reshape(6).copy()
    t_flange = _pose_to_se3(flange_pose_mm_rad)
    t_offset = _pose_to_se3(tcp_offset_mm_rad)
    return _se3_to_pose(t_flange @ t_offset)


def pose_tcp_to_flange(
    tcp_pose_mm_rad: Sequence[float],
    tcp_offset_mm_rad: Sequence[float] | None = None,
) -> np.ndarray:
    """Transform controller TCP pose ``[x_mm, y_mm, z_mm, r, p, y]`` back to flange frame."""
    if tcp_offset_mm_rad is None:
        return np.asarray(tcp_pose_mm_rad, dtype=np.float64).reshape(6).copy()
    t_tcp = _pose_to_se3(tcp_pose_mm_rad)
    t_offset = _pose_to_se3(tcp_offset_mm_rad)
    t_offset_inv = np.eye(4, dtype=np.float64)
    r_T = t_offset[:3, :3].T
    t_offset_inv[:3, :3] = r_T
    t_offset_inv[:3, 3] = -r_T @ t_offset[:3, 3]
    return _se3_to_pose(t_tcp @ t_offset_inv)


def flange_m_quat_to_tcp_mm_rpy(
    pos_m: Sequence[float],
    quat_wxyz: Sequence[float],
    tcp_offset_mm_rad: Sequence[float] | None = None,
) -> np.ndarray:
    """Convert MuJoCo ``attachment_site`` ``(pos_m, quat_wxyz)`` to controller TCP ``[mm, rad]``."""
    pos_mm = np.asarray(pos_m, dtype=np.float64).reshape(3) * 1000.0
    rpy = quat_wxyz_to_rpy(quat_wxyz)
    flange = np.concatenate([pos_mm, rpy])
    return pose_flange_to_tcp(flange, tcp_offset_mm_rad)


def tcp_mm_rpy_to_flange_m_quat(
    tcp_pose_mm_rad: Sequence[float],
    tcp_offset_mm_rad: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert controller TCP ``[mm, rad]`` to MuJoCo ``attachment_site`` ``(pos_m, quat_wxyz)``."""
    flange = pose_tcp_to_flange(tcp_pose_mm_rad, tcp_offset_mm_rad)
    pos_m = flange[:3] / 1000.0
    quat = rpy_to_quat_wxyz(float(flange[3]), float(flange[4]), float(flange[5]))
    return pos_m, quat


def apply_calibration_to_model(
    model: mujoco.MjModel,
    calibration: KinematicsCalibration,
    data: mujoco.MjData | None = None,
) -> None:
    """Patch a MuJoCo model in-place with per-unit joint origin calibration."""
    dof = ARMS[calibration.robot_key].dof
    for idx in range(1, dof + 1):
        joint_name = f"joint{idx}"
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise CalibrationError(f"MuJoCo model is missing calibrated joint {joint_name!r}")
        body_id = int(model.jnt_bodyid[joint_id])
        params = calibration.joints[joint_name]
        model.body_pos[body_id] = (params["x"], params["y"], params["z"])
        model.body_quat[body_id] = rpy_to_quat_wxyz(
            params["roll"], params["pitch"], params["yaw"]
        )
    target_data = data if data is not None else mujoco.MjData(model)
    mujoco.mj_forward(model, target_data)

