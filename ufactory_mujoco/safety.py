"""Fail-closed hardware safety gate, single-cycle enforcement, and SDK-sim validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any

import mujoco
import numpy as np

from .calibration import (
    CalibrationError,
    KinematicsCalibration,
    apply_calibration_to_model,
    canonical_robot_key,
    get_robot_sn,
    load_kinematics_calibration,
    read_tcp_offset,
)
from .collision import CollisionChecker
from .kinematics import fk_pose
from .robots import ARMS

_ISSUER = object()

# Controller reported states (xArm API)
REPORT_STATE_IN_MOTION = 1
REPORT_STATE_SLEEPING = 2
REPORT_STATE_PAUSED = 3
REPORT_STATE_STOPPING = 4
REPORT_STATE_READY = 5


class SafetyGateError(RuntimeError):
    """Raised when a hardware safety gate condition fails (fail-closed)."""


@dataclass(frozen=True, init=False)
class MotionApproval:
    """Unforgeable token authorizing a single-cycle real-robot motion."""

    robot: str
    serial_number: str
    calibration_sha256: str
    config_sha256: str
    program_sha256: str
    cycles: int
    issued_at_unix_s: float

    def __init__(
        self,
        *,
        robot: str,
        serial_number: str,
        calibration_sha256: str,
        config_sha256: str,
        program_sha256: str,
        cycles: int,
        issued_at_unix_s: float,
        _issuer: object = None,
    ) -> None:
        if _issuer is not _ISSUER:
            raise SafetyGateError("MotionApproval can only be issued by authorize_real_motion()")
        object.__setattr__(self, "robot", robot)
        object.__setattr__(self, "serial_number", serial_number)
        object.__setattr__(self, "calibration_sha256", calibration_sha256)
        object.__setattr__(self, "config_sha256", config_sha256)
        object.__setattr__(self, "program_sha256", program_sha256)
        object.__setattr__(self, "cycles", int(cycles))
        object.__setattr__(self, "issued_at_unix_s", float(issued_at_unix_s))


@dataclass(frozen=True)
class SdkSimEvidence:
    """Validation evidence from controller internal simulation (sdk-sim)."""

    schema_version: int
    passed: bool
    robot: str
    serial_number: str
    calibration_sha256: str
    config_sha256: str
    program_sha256: str
    shadow_stream_sha256: str
    firmware_stream_sha256: str
    samples: int
    max_joint_error_rad: float
    max_ee_error_m: float
    min_distance_m: float
    physical_joint_drift_rad: float
    created_at_unix_s: float
    failures: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready copy of the evidence."""
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "robot": self.robot,
            "serial_number": self.serial_number,
            "calibration_sha256": self.calibration_sha256,
            "config_sha256": self.config_sha256,
            "program_sha256": self.program_sha256,
            "shadow_stream_sha256": self.shadow_stream_sha256,
            "firmware_stream_sha256": self.firmware_stream_sha256,
            "samples": self.samples,
            "max_joint_error_rad": self.max_joint_error_rad,
            "max_ee_error_m": self.max_ee_error_m,
            "min_distance_m": self.min_distance_m,
            "physical_joint_drift_rad": self.physical_joint_drift_rad,
            "created_at_unix_s": self.created_at_unix_s,
            "failures": list(self.failures),
        }


def stream_sha256(stream: np.ndarray) -> str:
    """Compute canonical SHA-256 of a float64 trajectory array."""
    values = np.asarray(stream, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(json.dumps(list(values.shape)).encode("utf-8"))
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def enforce_single_cycle(cycles: int, mode: str) -> None:
    """Reject multi-cycle requests in sdk-sim or real modes."""
    if type(cycles) is not int or cycles != 1:
        raise SafetyGateError(
            f"mode={mode!r} enforces a single-cycle cap (cycles must be 1, got {cycles})"
        )


def assert_controller_healthy(arm: Any) -> None:
    """Check controller connection, error/warning codes, and stop state without auto-clearing."""
    if not bool(getattr(arm, "connected", False)):
        raise SafetyGateError("controller is not connected")
    err_code = int(getattr(arm, "error_code", 0) or 0)
    warn_code = int(getattr(arm, "warn_code", 0) or 0)
    getter = getattr(arm, "get_err_warn_code", None)
    if callable(getter):
        res = getter()
        if isinstance(res, tuple) and len(res) == 2 and int(res[0]) == 0:
            err_code, warn_code = int(res[1][0]), int(res[1][1])
    if err_code != 0:
        raise SafetyGateError(
            f"controller has active error_code={err_code}; inspect and recover manually in xArm Studio"
        )
    if warn_code != 0:
        raise SafetyGateError(
            f"controller has active warn_code={warn_code}; inspect and recover manually in xArm Studio"
        )
    state = int(getattr(arm, "state", 0) or 0)
    state_getter = getattr(arm, "get_state", None)
    if callable(state_getter):
        res = state_getter()
        if isinstance(res, tuple) and len(res) == 2 and int(res[0]) == 0:
            state = int(res[1])
    if state == REPORT_STATE_STOPPING:
        raise SafetyGateError(
            "controller is in STOP state (state=4); release physical E-stop or enable in xArm Studio"
        )


def verify_hardware_identity_and_calibration(
    arm: Any,
    *,
    robot: str,
    calibration_path: str | Any,
) -> tuple[str, KinematicsCalibration]:
    """Verify controller health, DOF, serial number, and matching calibration file."""
    robot_key = canonical_robot_key(robot)
    if not calibration_path:
        raise SafetyGateError(
            "missing --calibration YAML; hardware connection requires a robot-bound calibration file"
        )
    assert_controller_healthy(arm)
    try:
        sn = get_robot_sn(arm)
    except CalibrationError as exc:
        raise SafetyGateError(str(exc)) from exc

    expected_dof = ARMS[robot_key].dof
    axis = int(getattr(arm, "axis", expected_dof) or expected_dof)
    if axis != expected_dof:
        raise SafetyGateError(
            f"connected robot DOF ({axis}) does not match {robot_key} ({expected_dof})"
        )

    try:
        if isinstance(calibration_path, KinematicsCalibration):
            calibration = calibration_path
            if calibration.robot_key != robot_key:
                raise CalibrationError(
                    f"calibration robot_key {calibration.robot_key!r} != {robot_key!r}"
                )
            if calibration.serial_number != sn:
                raise CalibrationError(
                    f"calibration serial_number {calibration.serial_number!r} != {sn!r}"
                )
        else:
            calibration = load_kinematics_calibration(
                calibration_path, robot=robot_key, serial_number=sn
            )
    except CalibrationError as exc:
        raise SafetyGateError(f"calibration check failed: {exc}") from exc

    return sn, calibration


def authorize_real_motion(
    arm: Any,
    *,
    robot: str,
    calibration: KinematicsCalibration | str | Any,
    preflight_ok: bool,
    confirm_real: bool,
    cycles: int = 1,
    config_sha256: str = "",
    program_sha256: str = "",
    sdk_evidence: SdkSimEvidence | None = None,
) -> MotionApproval:
    """Issue a MotionApproval token only when every safety gate condition passes."""
    enforce_single_cycle(cycles, "real")
    sn, calib = verify_hardware_identity_and_calibration(
        arm, robot=robot, calibration_path=calibration
    )
    if not preflight_ok:
        raise SafetyGateError("preflight failed; real motion is forbidden")
    if sdk_evidence is not None and not sdk_evidence.passed:
        raise SafetyGateError("sdk-sim validation failed; real motion is forbidden")
    if not confirm_real:
        raise SafetyGateError(
            "real motion requires explicit confirmation (--confirm-real); refusing to move"
        )
    if not config_sha256:
        config_sha256 = hashlib.sha256(f"default:{calib.robot_key}".encode("utf-8")).hexdigest()
    if not program_sha256:
        program_sha256 = hashlib.sha256(f"program:{calib.sha256}".encode("utf-8")).hexdigest()
    return MotionApproval(
        robot=calib.robot_key,
        serial_number=sn,
        calibration_sha256=calib.sha256,
        config_sha256=config_sha256,
        program_sha256=program_sha256,
        cycles=cycles,
        issued_at_unix_s=time.time(),
        _issuer=_ISSUER,
    )


def read_arm_joint_positions(arm: Any, dof: int) -> np.ndarray:
    """Read ``(dof,)`` joint angles in radians without issuing any motion command."""
    code, angles = arm.get_servo_angle(is_radian=True)
    if int(code) != 0:
        raise SafetyGateError(f"get_servo_angle failed with SDK code {code}")
    q = np.asarray(angles, dtype=np.float64).reshape(-1)[:dof]
    if q.shape != (dof,) or not np.all(np.isfinite(q)):
        raise SafetyGateError(f"invalid joint feedback shape {q.shape} for dof={dof}")
    return q.copy()


def _wait_for_servo_ready(arm: Any, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        assert_controller_healthy(arm)
        code, state = arm.get_state()
        mode_val = getattr(arm, "mode", 1)
        if (
            int(code) == 0
            and int(state) in (REPORT_STATE_IN_MOTION, REPORT_STATE_SLEEPING, REPORT_STATE_READY)
            and int(mode_val if mode_val is not None else 1) in (0, 1)
        ):
            if int(mode_val if mode_val is not None else 1) == 1:
                return
        time.sleep(0.03)
    raise SafetyGateError(f"controller did not enter servo-ready state within {timeout_s}s")


def run_sdk_simulation(
    arm: Any,
    *,
    robot: str,
    calibration: KinematicsCalibration | str | Any,
    shadow_joint_stream_rad: np.ndarray,
    checker: CollisionChecker,
    preflight_ok: bool = True,
    cycles: int = 1,
    config_sha256: str = "",
    program_sha256: str = "",
    step_dt_s: float = 0.01,
    max_joint_error_tol_rad: float = 0.05,
    max_ee_error_tol_m: float = 0.005,
    physical_drift_tol_rad: float = 1e-3,
) -> SdkSimEvidence:
    """Replay a joint trajectory in controller internal simulation (physical joints stationary)."""
    enforce_single_cycle(cycles, "sdk-sim")
    sn, calib = verify_hardware_identity_and_calibration(
        arm, robot=robot, calibration_path=calibration
    )
    if not preflight_ok:
        raise SafetyGateError("preflight failed; sdk-sim replay is forbidden")

    dof = ARMS[calib.robot_key].dof
    shadow = np.asarray(shadow_joint_stream_rad, dtype=np.float64)
    if shadow.ndim != 2 or shadow.shape[1] != dof or len(shadow) == 0:
        raise SafetyGateError(f"shadow_joint_stream_rad must have shape (N, {dof}) with N >= 1")
    if not np.all(np.isfinite(shadow)):
        raise SafetyGateError("shadow_joint_stream_rad contains non-finite values")

    # Apply calibration to checker's model and an FK evaluation model
    apply_calibration_to_model(checker.model, calib, checker.data)
    fk_model = checker.model
    fk_data = mujoco.MjData(fk_model)
    site_name = ARMS[calib.robot_key].tcp_site
    _ = read_tcp_offset(arm)

    # Record physical joint positions BEFORE switching to simulation mode
    q_physical_before = read_arm_joint_positions(arm, dof)
    start_q = shadow[0].copy()

    def _require_ok(label: str, code: int) -> None:
        if int(code) != 0:
            raise SafetyGateError(f"sdk-sim {label} failed with SDK code {code}")

    simulation_enabled = bool(getattr(arm, "is_simulation_robot", False))
    firmware_rows: list[np.ndarray] = []
    try:
        if not simulation_enabled:
            _require_ok("motion_enable", arm.motion_enable(enable=True))
            _require_ok("set_mode(0)", arm.set_mode(0))
            _require_ok("set_state(0)", arm.set_state(0))
            _require_ok("set_simulation_robot(True)", arm.set_simulation_robot(True))
            simulation_enabled = True

        deadline = time.monotonic() + 2.0
        while not bool(getattr(arm, "is_simulation_robot", False)) and time.monotonic() < deadline:
            time.sleep(0.05)
        if not bool(getattr(arm, "is_simulation_robot", False)):
            raise SafetyGateError("controller did not confirm simulation mode (is_simulation_robot=False)")

        _require_ok("set_mode(0)", arm.set_mode(0))
        _require_ok("set_state(0)", arm.set_state(0))
        _require_ok(
            "virtual preposition",
            arm.set_servo_angle(
                angle=start_q.tolist(),
                speed=0.5,
                mvacc=2.0,
                wait=True,
                is_radian=True,
            ),
        )
        virtual_start = read_arm_joint_positions(arm, dof)
        start_err = float(np.max(np.abs(virtual_start - start_q)))
        if start_err > 0.02:
            raise SafetyGateError(
                f"sdk-sim virtual start error {start_err:.4f} rad exceeds 0.0200 rad"
            )

        _require_ok("set_mode(1)", arm.set_mode(1))
        _require_ok("set_state(0)", arm.set_state(0))
        _wait_for_servo_ready(arm)

        firmware_rows.append(virtual_start)
        for q_cmd in shadow[1:]:
            code = arm.set_servo_angle_j(
                angles=q_cmd.tolist(),
                speed=0,
                mvacc=0,
                mvtime=0,
                is_radian=True,
            )
            if int(code) != 0:
                arm.set_state(REPORT_STATE_STOPPING)
                raise SafetyGateError(f"sdk-sim set_servo_angle_j failed with SDK code {code}")
            if step_dt_s > 0.0:
                time.sleep(step_dt_s)
            firmware_rows.append(read_arm_joint_positions(arm, dof))
    finally:
        if simulation_enabled:
            cleanup_errors: list[str] = []
            pause_code = int(arm.set_state(REPORT_STATE_PAUSED))
            if pause_code != 0:
                cleanup_errors.append(f"set_state(3) returned {pause_code}")
            disable_code = int(arm.set_simulation_robot(False))
            if disable_code != 0:
                cleanup_errors.append(f"set_simulation_robot(False) returned {disable_code}")
            deadline = time.monotonic() + 2.0
            while bool(getattr(arm, "is_simulation_robot", False)) and time.monotonic() < deadline:
                time.sleep(0.05)
            if bool(getattr(arm, "is_simulation_robot", False)):
                cleanup_errors.append("controller still reports is_simulation_robot=True")
            arm.set_mode(0)
            arm.set_state(0)
            if cleanup_errors:
                raise SafetyGateError("sdk-sim cleanup failed: " + "; ".join(cleanup_errors))

    # Give controller report stream a brief moment to refresh real encoder values
    if step_dt_s > 0.0:
        time.sleep(0.2)
    q_physical_after = read_arm_joint_positions(arm, dof)
    physical_drift = float(np.max(np.abs(q_physical_after - q_physical_before)))

    firmware = np.stack(firmware_rows, axis=0)
    failures: list[str] = []
    if firmware.shape != shadow.shape:
        failures.append(f"firmware shape {firmware.shape} != shadow shape {shadow.shape}")

    max_joint_err = float(np.max(np.abs(firmware - shadow))) if firmware.shape == shadow.shape else math.inf
    if max_joint_err > max_joint_error_tol_rad:
        failures.append(
            f"max joint error {max_joint_err:.6g} rad exceeds {max_joint_error_tol_rad:.6g} rad"
        )

    ee_errors: list[float] = []
    distances: list[float] = []
    for idx, (q_sh, q_fw) in enumerate(zip(shadow, firmware)):
        fk_data.qpos[:dof] = q_sh
        pos_sh, _ = fk_pose(fk_model, fk_data, site_name)
        fk_data.qpos[:dof] = q_fw
        pos_fw, _ = fk_pose(fk_model, fk_data, site_name)
        ee_errors.append(float(np.linalg.norm(pos_sh - pos_fw)))
        if np.any(q_fw < checker.q_lo - 1e-6) or np.any(q_fw > checker.q_hi + 1e-6):
            failures.append(f"sample {idx}: firmware joint position outside limits")
        dist = float(checker.min_distance(q_fw))
        distances.append(dist)
        if checker.in_collision(q_fw):
            failures.append(f"sample {idx}: firmware pose in collision (min_distance={dist:.6g} m)")

    max_ee_err = max(ee_errors, default=math.inf)
    if max_ee_err > max_ee_error_tol_m:
        failures.append(
            f"max EE error {max_ee_err:.6g} m exceeds {max_ee_error_tol_m:.6g} m"
        )
    min_dist = min(distances, default=math.inf)

    if physical_drift > physical_drift_tol_rad:
        failures.append(
            f"physical joint drift {physical_drift:.6g} rad exceeds {physical_drift_tol_rad:.6g} rad"
        )

    sh_hash = stream_sha256(shadow)
    fw_hash = stream_sha256(firmware)
    cfg_hash = config_sha256 or hashlib.sha256(f"default:{calib.robot_key}".encode("utf-8")).hexdigest()
    prog_hash = program_sha256 or sh_hash

    return SdkSimEvidence(
        schema_version=1,
        passed=not failures,
        robot=calib.robot_key,
        serial_number=sn,
        calibration_sha256=calib.sha256,
        config_sha256=cfg_hash,
        program_sha256=prog_hash,
        shadow_stream_sha256=sh_hash,
        firmware_stream_sha256=fw_hash,
        samples=int(len(firmware)),
        max_joint_error_rad=max_joint_err,
        max_ee_error_m=max_ee_err,
        min_distance_m=min_dist,
        physical_joint_drift_rad=physical_drift,
        created_at_unix_s=time.time(),
        failures=tuple(failures),
    )
