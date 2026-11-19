"""Dataclasses, Schema v4 report I/O, and report comparison for dynamics validation."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import re
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .analysis import StaticLayerResult
    from .profiles import RobotRuntimeProfile

SIM_DT = 0.002
SETTLE_DURATION_S = 5.0
SETTLE_STEPS = 2500
POS_ERR_TOL = 0.05  # rad
VEL_TOL = 0.01  # rad/s
SATURATION_MARGIN = 0.995

# Per-joint absolute error limits (Nm); xArm 6 1305 fallback (10% of official max torque).
ABS_ERR_LIMITS = np.array((17.94445, 17.94445, 9.2, 8.1558, 3.06, 3.06), dtype=np.float64)
L2_ERR_LIMIT = float(np.linalg.norm(ABS_ERR_LIMITS))
REL_ERR_LIMIT = 0.15  # fraction of effort limit
REPORT_SCHEMA_VERSION = "4"
REPORT_STEM = "dyn_ver"
DEFAULT_REPORT_ROOT = Path("reports")


class ValidationStatus(str, Enum):
    """Pose-level L1 dynamics validation outcome."""

    PASS = "PASS"
    FAIL_MODEL = "FAIL_MODEL"
    FAIL_BIAS = "FAIL_BIAS"
    NOT_SETTLED = "NOT_SETTLED"
    SATURATED = "SATURATED"
    UNSAFE = "UNSAFE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class SafePose:
    """Validated joint target passing EE height and collision filters."""

    name: str
    q: np.ndarray
    ee_z_mm: float


@dataclass
class ModelDynamicsIssue:
    """Static MJCF/URDF inertial or joint-dynamics validation issue."""

    severity: str
    item: str
    message: str
    value: Any = None


UrdfDynamicsIssue = ModelDynamicsIssue


@dataclass
class DynamicsRunConfig:
    """Metadata header for a Schema v4 JSONL dynamics report."""

    robot_key: str
    model_path: str
    model_sha256: str | None = None
    kinematics_yaml_path: str | None = None
    kinematics_yaml_sha256: str | None = None
    mujoco_version: str | None = None
    reference_backend: str | None = None
    sim_dt: float = SIM_DT
    settle_steps: int = SETTLE_STEPS
    sdk_version: str | None = None
    firmware: str | None = None
    robot_sn: str | None = None
    tcp_offset: list[float] | None = None
    tcp_load: list[Any] | None = None
    gravity_direction: list[float] | None = None
    joint_frictionloss: list[float] | None = None
    git_sha: str | None = None
    version: str = REPORT_SCHEMA_VERSION
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    mode: str = "dry-run"


@dataclass
class SimDynamicsSample:
    """Closed-loop MuJoCo PD-hold measurement at one target configuration."""

    q_actual: np.ndarray
    qvel: np.ndarray
    pd_hold_tau: np.ndarray
    actual_dof_force: np.ndarray
    mass_matrix: np.ndarray
    settled: bool
    saturated: bool
    pos_err: float
    vel_mag: float
    armature: np.ndarray | None = None
    joint_frictionloss: np.ndarray | None = None
    self_contacts: list[dict[str, Any]] | None = None


GenesisDynamicsSample = SimDynamicsSample


@dataclass
class DynamicsSample:
    """Combined simulation, analytical reference, and hardware record for one pose."""

    pose: str
    q: np.ndarray
    ee_z_mm: float
    status: ValidationStatus
    settled: bool = False
    saturated: bool = False
    q_actual: np.ndarray | None = None
    qvel: np.ndarray | None = None
    pd_hold_tau: np.ndarray | None = None
    actual_dof_force: np.ndarray | None = None
    mass_matrix: np.ndarray | None = None
    reference_gravity_tau: np.ndarray | None = None
    reference_mass_matrix: np.ndarray | None = None
    clamp_slack_est: np.ndarray | None = None
    armature: np.ndarray | None = None
    static_layers: list[StaticLayerResult] = field(default_factory=list)
    static_warnings: list[str] = field(default_factory=list)
    l2a_l2_err: float | None = None
    l2b_l2_err: float | None = None
    l3a_mass_rel_fro: float | None = None
    l3b_l2_err: float | None = None
    pin_G_l2: float | None = None
    clamp_slack_l2: float | None = None
    tau_real: np.ndarray | None = None
    tau_real_median: np.ndarray | None = None
    tau_real_std: np.ndarray | None = None
    tau_real_min: np.ndarray | None = None
    tau_real_max: np.ndarray | None = None
    tau_direct: np.ndarray | None = None
    abs_err: np.ndarray | None = None
    rel_err: np.ndarray | None = None
    signed_err: np.ndarray | None = None
    l2_err: float | None = None
    n_real_samples: int = 0
    skip_reason: str = ""
    notes: list[str] = field(default_factory=list)


def sha256_file(path: str | Path | None) -> str | None:
    """Return hex SHA-256 of ``path``, or ``None`` if absent."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_sha(root: str | Path | None = None) -> str | None:
    """Return short HEAD SHA for provenance metadata."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root or Path(__file__).resolve().parents[2]),
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return out.strip() or None


def package_version(name: str) -> str | None:
    """Return installed metadata version for ``name`` if available."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def minute_stamp(stamp: str | None = None) -> str:
    value = stamp or now_stamp()
    return value[:13] if len(value) >= 13 else value


def sanitize_report_identity(identity: str | None) -> str:
    value = str(identity or "").strip()
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    value = value.strip("_-")
    return value or "unknown"


def dynamics_report_paths(
    *,
    identity: str | None,
    report: str | Path | None = None,
    jsonl_report: str | Path | None = None,
    report_root: str | Path = DEFAULT_REPORT_ROOT,
    stamp: str | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve CSV/JSONL/plot paths for one dynamics validation run."""
    if report is not None or jsonl_report is not None:
        csv_path = Path(report) if report is not None else Path(jsonl_report).with_suffix(".csv")
        jsonl_path = Path(jsonl_report) if jsonl_report is not None else csv_path.with_suffix(".jsonl")
        plot_path = csv_path.with_name(f"{csv_path.stem}_torque.png")
        return csv_path, jsonl_path, plot_path

    file_stamp = stamp or now_stamp()
    safe_identity = sanitize_report_identity(identity)
    run_dir = Path(report_root) / f"{REPORT_STEM}_{safe_identity}" / minute_stamp(file_stamp)
    base = run_dir / f"{REPORT_STEM}_{safe_identity}_{file_stamp}"
    return base.with_suffix(".csv"), base.with_suffix(".jsonl"), base.with_name(f"{base.name}_torque.png")


def _effort_limits(runtime: RobotRuntimeProfile) -> np.ndarray:
    return np.asarray(runtime.arm.effort_limits, dtype=np.float64)


def _abs_err_limits(runtime: RobotRuntimeProfile) -> np.ndarray:
    return np.asarray(runtime.dynamics.abs_err_limits, dtype=np.float64)


def _sample_dof(sample: DynamicsSample) -> int:
    arrays = [
        sample.q,
        sample.q_actual,
        sample.pd_hold_tau,
        sample.tau_real,
        sample.abs_err,
        sample.rel_err,
    ]
    return max((int(np.asarray(a).reshape(-1).size) for a in arrays if a is not None), default=0)


def _fit_float_tuple(values: Sequence[float] | np.ndarray | None, n: int, fill: float = np.nan) -> np.ndarray:
    if values is None:
        return np.full(n, fill, dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size >= n:
        return arr[:n].astype(np.float64)
    out = np.full(n, fill, dtype=np.float64)
    out[: arr.size] = arr
    return out


def _runtime_abs_limits(runtime_profile: RobotRuntimeProfile | None, n: int) -> np.ndarray:
    if runtime_profile is not None:
        return _fit_float_tuple(runtime_profile.dynamics.abs_err_limits, n, fill=np.nan)
    return _fit_float_tuple(ABS_ERR_LIMITS, n, fill=float(ABS_ERR_LIMITS[-1]))


def _runtime_effort_limits(runtime_profile: RobotRuntimeProfile | None, n: int) -> np.ndarray:
    if runtime_profile is not None:
        return _fit_float_tuple(runtime_profile.arm.effort_limits, n, fill=np.nan)
    fallback = np.asarray((179.4445, 179.4445, 92.0, 81.558, 30.6, 30.6), dtype=np.float64)
    return _fit_float_tuple(fallback, n, fill=float(fallback[-1]))


def _runtime_l2_limit(runtime_profile: RobotRuntimeProfile | None) -> float:
    if runtime_profile is not None:
        return float(runtime_profile.dynamics.l2_err_limit)
    return float(L2_ERR_LIMIT)


def _runtime_rel_limit(runtime_profile: RobotRuntimeProfile | None) -> float:
    if runtime_profile is not None:
        return float(runtime_profile.dynamics.rel_err_limit)
    return float(REL_ERR_LIMIT)


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def has_sdk_torque(sample: DynamicsSample) -> bool:
    """Return True when ``sample`` carries finite SDK real-robot torque data."""
    tau = sample.tau_real
    if tau is None:
        return False
    arr = np.asarray(tau, dtype=np.float64).reshape(-1)
    return bool(arr.size and np.isfinite(arr).any())


def status_reason(
    sample: DynamicsSample,
    *,
    runtime_profile: RobotRuntimeProfile | None = None,
) -> str:
    """Return human-readable reason for non-PASS samples."""
    try:
        status = sample.status if isinstance(sample.status, ValidationStatus) else ValidationStatus(str(sample.status))
    except ValueError:
        return str(sample.status)
    if status == ValidationStatus.PASS:
        return ""
    if status == ValidationStatus.NOT_SETTLED:
        return "pose did not settle"
    if status == ValidationStatus.SATURATED:
        return "Simulation torque saturated"
    if status == ValidationStatus.UNSAFE:
        notes = getattr(sample, "notes", []) or []
        skip = getattr(sample, "skip_reason", "")
        return "; ".join(str(n) for n in notes[:2]) or skip or "hardware path rejected by safety checks"
    if status == ValidationStatus.INSUFFICIENT_DATA:
        return "insufficient SDK torque data"
    rows = joint_torque_rows(sample, runtime_profile=runtime_profile)
    worst = max((row for row in rows if row["abs_err_nm"] is not None), key=lambda row: row["abs_err_nm"], default=None)
    joint = worst["joint"] if worst else "unknown joint"
    if status == ValidationStatus.FAIL_BIAS:
        return f"{joint} exceeds torque error limit"
    if status == ValidationStatus.FAIL_MODEL:
        return "torque error exceeds model limits"
    return str(status)


def joint_torque_rows(
    sample: DynamicsSample,
    *,
    runtime_profile: RobotRuntimeProfile | None = None,
) -> list[dict[str, Any]]:
    """Expand per-joint simulation, SDK, and error metrics for CSV/JSONL/printing."""
    n = _sample_dof(sample)
    if n <= 0:
        return []
    q_cmd = _fit_float_tuple(sample.q, n)
    q_actual = _fit_float_tuple(sample.q_actual, n)
    sim_tau = _fit_float_tuple(sample.pd_hold_tau, n)
    sdk_mean = _fit_float_tuple(sample.tau_real, n)
    sdk_median = _fit_float_tuple(sample.tau_real_median, n)
    sdk_std = _fit_float_tuple(sample.tau_real_std, n)
    sdk_min = _fit_float_tuple(sample.tau_real_min, n)
    sdk_max = _fit_float_tuple(sample.tau_real_max, n)
    abs_err = _fit_float_tuple(sample.abs_err, n)
    rel_err = _fit_float_tuple(sample.rel_err, n)

    abs_limits = _runtime_abs_limits(runtime_profile, n)
    effort_limits = _runtime_effort_limits(runtime_profile, n)
    rel_limit = _runtime_rel_limit(runtime_profile)
    rows: list[dict[str, Any]] = []
    for j in range(n):
        joint_ok = (
            math.isfinite(abs_err[j])
            and math.isfinite(rel_err[j])
            and math.isfinite(abs_limits[j])
            and abs_err[j] <= abs_limits[j]
            and rel_err[j] <= rel_limit
        )
        joint_status = "" if not has_sdk_torque(sample) else ("PASS" if joint_ok else "FAIL")
        tau_val = _finite_or_none(sim_tau[j])
        rows.append(
            {
                "joint": f"J{j + 1}",
                "joint_index": j + 1,
                "q_cmd_rad": _finite_or_none(q_cmd[j]),
                "q_actual_rad": _finite_or_none(q_actual[j]),
                "mujoco_tau_nm": tau_val,
                "theory_tau_nm": tau_val,
                "sdk_tau_mean_nm": _finite_or_none(sdk_mean[j]),
                "sdk_tau_median_nm": _finite_or_none(sdk_median[j]),
                "sdk_tau_std_nm": _finite_or_none(sdk_std[j]),
                "sdk_tau_min_nm": _finite_or_none(sdk_min[j]),
                "sdk_tau_max_nm": _finite_or_none(sdk_max[j]),
                "abs_err_nm": _finite_or_none(abs_err[j]),
                "rel_err": _finite_or_none(rel_err[j]),
                "abs_limit_nm": _finite_or_none(abs_limits[j]),
                "effort_limit_nm": _finite_or_none(effort_limits[j]),
                "rel_limit": rel_limit,
                "joint_status": joint_status,
            }
        )
    return rows


def torque_summary(
    sample: DynamicsSample,
    *,
    runtime_profile: RobotRuntimeProfile | None = None,
) -> dict[str, Any]:
    """Summarize L1 status and worst joint residual for one pose sample."""
    rows = joint_torque_rows(sample, runtime_profile=runtime_profile)
    finite_rows = [row for row in rows if row["abs_err_nm"] is not None]
    worst = max(finite_rows, key=lambda row: row["abs_err_nm"], default=None)
    status = sample.status.value
    l2_err = _finite_or_none(sample.l2_err)
    summary = {
        "status": status,
        "torque_l2_err_nm": l2_err,
        "torque_l2_limit_nm": _runtime_l2_limit(runtime_profile),
        "worst_joint": worst["joint"] if worst else "",
        "worst_abs_err_nm": worst["abs_err_nm"] if worst else None,
        "worst_rel_err": worst["rel_err"] if worst else None,
        "n_real_samples": int(sample.n_real_samples),
        "warning_count": len(sample.static_warnings),
        "has_sdk_torque": has_sdk_torque(sample),
    }
    summary["status_reason"] = status_reason(sample, runtime_profile=runtime_profile) if status != "PASS" else ""
    return summary


def sample_json_record(
    sample: DynamicsSample,
    *,
    runtime_profile: RobotRuntimeProfile | None = None,
) -> dict[str, Any]:
    """Serialize one sample to Schema v4 JSON record dict."""
    data = _jsonable(sample)
    summary = torque_summary(sample, runtime_profile=runtime_profile)
    rows = joint_torque_rows(sample, runtime_profile=runtime_profile)
    data["schema_version"] = REPORT_SCHEMA_VERSION
    data.update(summary)
    data["torque_compare"] = {
        "source_theory": "mujoco_pd_hold_tau",
        "source_sdk": "get_joint_states_effort_mean",
        **summary,
        "joints": rows,
    }
    return data


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def make_run_config(
    *,
    robot_key: str,
    model_path: str,
    kinematics_yaml_path: str | None = None,
    sim_dt: float = SIM_DT,
    settle_steps: int = SETTLE_STEPS,
    reference_backend: str | None = None,
    mode: str,
    session: Any | None = None,
) -> DynamicsRunConfig:
    """Build ``DynamicsRunConfig`` populated with environment and controller metadata."""
    mujoco_version = package_version("mujoco")
    try:
        import mujoco

        mujoco_version = getattr(mujoco, "__version__", mujoco_version)
    except Exception:
        pass

    firmware = robot_sn = None
    tcp_offset = tcp_load = gravity_direction = None
    if session is not None:
        arm = session.arm
        firmware = str(getattr(arm, "version", "")) or None
        robot_sn = str(getattr(arm, "sn", "")) or None
        for attr, target in (
            ("tcp_offset", "tcp_offset"),
            ("tcp_load", "tcp_load"),
            ("gravity_direction", "gravity_direction"),
        ):
            try:
                val = list(getattr(arm, attr))
            except Exception:
                val = None
            if target == "tcp_offset":
                tcp_offset = val
            elif target == "tcp_load":
                tcp_load = val
            else:
                gravity_direction = val

    return DynamicsRunConfig(
        robot_key=robot_key,
        model_path=str(model_path),
        model_sha256=sha256_file(model_path),
        kinematics_yaml_path=str(kinematics_yaml_path) if kinematics_yaml_path else None,
        kinematics_yaml_sha256=sha256_file(kinematics_yaml_path),
        mujoco_version=mujoco_version,
        reference_backend=reference_backend,
        sim_dt=float(sim_dt),
        settle_steps=int(settle_steps),
        sdk_version=package_version("xarm-python-sdk"),
        firmware=firmware,
        robot_sn=robot_sn,
        tcp_offset=tcp_offset,
        tcp_load=tcp_load,
        gravity_direction=gravity_direction,
        git_sha=current_git_sha(Path(__file__).resolve().parents[2]),
        mode=mode,
    )


def write_jsonl_report(
    results: Sequence[DynamicsSample],
    path: str | Path,
    *,
    run_config: DynamicsRunConfig,
    model_issues: Sequence[ModelDynamicsIssue] | None = None,
    runtime_profile: RobotRuntimeProfile | None = None,
) -> None:
    """Write Schema v4 JSONL report."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "run_config", "data": _jsonable(run_config)}, sort_keys=True) + "\n")
        for issue in model_issues or ():
            f.write(json.dumps({"type": "model_issue", "data": _jsonable(issue)}, sort_keys=True) + "\n")
        for result in results:
            data = sample_json_record(result, runtime_profile=runtime_profile)
            f.write(json.dumps({"type": "sample", "data": data}, sort_keys=True) + "\n")


def _csv_float(value: Any, digits: int = 6) -> str:
    value = _finite_or_none(value)
    return f"{value:.{digits}f}" if value is not None else ""


def write_csv_report(
    results: Sequence[DynamicsSample],
    path: str | Path,
    *,
    runtime_profile: RobotRuntimeProfile | None = None,
) -> None:
    """Write Schema v4 CSV report."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    dof = 0
    for item in results:
        dof = max(dof, _sample_dof(item))
    if dof == 0:
        dof = len(runtime_profile.arm.joint_names) if runtime_profile is not None else 6
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "schema_version",
            "pose",
            "ee_z_mm",
            "settled",
            "status",
            "passed",
            "skip_reason",
            "status_reason",
            "torque_l2_err_nm",
            "torque_l2_limit_nm",
            "worst_joint",
            "worst_abs_err_nm",
            "worst_rel_err",
            "n_real_samples",
            "static_warn_count",
            "l2a_l2_err_nm",
            "l2b_l2_err_nm",
            "l3a_mass_rel_fro",
            "l3b_l2_err_nm",
            "pin_G_l2_nm",
            "clamp_slack_l2_nm",
            "static_warns",
        ]
        for i in range(1, dof + 1):
            header.extend(
                [
                    f"q_cmd_J{i}_rad",
                    f"q_actual_J{i}_rad",
                    f"mujoco_tau_J{i}_nm",
                    f"sdk_tau_mean_J{i}_nm",
                    f"sdk_tau_median_J{i}_nm",
                    f"sdk_tau_std_J{i}_nm",
                    f"sdk_tau_min_J{i}_nm",
                    f"sdk_tau_max_J{i}_nm",
                    f"abs_err_J{i}_nm",
                    f"rel_err_J{i}",
                    f"abs_limit_J{i}_nm",
                    f"effort_limit_J{i}_nm",
                    f"joint_status_J{i}",
                ]
            )
        writer.writerow(header)
        for result in results:
            summary = torque_summary(result, runtime_profile=runtime_profile)
            row = [
                REPORT_SCHEMA_VERSION,
                result.pose,
                f"{result.ee_z_mm:.3f}",
                int(result.settled),
                summary["status"],
                int(summary["status"] == ValidationStatus.PASS.value),
                result.skip_reason,
                summary["status_reason"],
                _csv_float(summary["torque_l2_err_nm"]),
                _csv_float(summary["torque_l2_limit_nm"]),
                summary["worst_joint"],
                _csv_float(summary["worst_abs_err_nm"]),
                _csv_float(summary["worst_rel_err"]),
                summary["n_real_samples"],
                summary["warning_count"],
                _csv_float(getattr(result, "l2a_l2_err", None)),
                _csv_float(getattr(result, "l2b_l2_err", None)),
                _csv_float(getattr(result, "l3a_mass_rel_fro", None)),
                _csv_float(getattr(result, "l3b_l2_err", None)),
                _csv_float(getattr(result, "pin_G_l2", None)),
                _csv_float(getattr(result, "clamp_slack_l2", None)),
                ";".join(getattr(result, "static_warnings", []) or []),
            ]
            joint_rows = joint_torque_rows(result, runtime_profile=runtime_profile)
            for j in range(dof):
                joint = joint_rows[j] if j < len(joint_rows) else {}
                row.extend(
                    [
                        _csv_float(joint.get("q_cmd_rad")),
                        _csv_float(joint.get("q_actual_rad")),
                        _csv_float(joint.get("mujoco_tau_nm")),
                        _csv_float(joint.get("sdk_tau_mean_nm")),
                        _csv_float(joint.get("sdk_tau_median_nm")),
                        _csv_float(joint.get("sdk_tau_std_nm")),
                        _csv_float(joint.get("sdk_tau_min_nm")),
                        _csv_float(joint.get("sdk_tau_max_nm")),
                        _csv_float(joint.get("abs_err_nm")),
                        _csv_float(joint.get("rel_err")),
                        _csv_float(joint.get("abs_limit_nm")),
                        _csv_float(joint.get("effort_limit_nm")),
                        joint.get("joint_status", ""),
                    ]
                )
            writer.writerow(row)


def read_report_records(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL or CSV reports (both ``mujoco_tau_*`` and ``genesis_tau_*`` plus v1-v3)."""
    p = Path(path)
    if p.suffix == ".jsonl":
        records = []
        schema_version = "1"
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item.get("type") == "run_config" and isinstance(item.get("data"), dict):
                    schema_version = str(item["data"].get("version", schema_version))
                elif item.get("type") == "sample":
                    data = item["data"]
                    data.setdefault("_schema_version", schema_version)
                    records.append(data)
        return records

    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    records = []
    for row in rows:
        schema_version = row.get("schema_version") or "1"
        l2_value = row.get("torque_l2_err_nm") if schema_version in {"3", "4"} else row.get("l2_err")
        rec: dict[str, Any] = {
            "pose": row.get("pose"),
            "status": row.get("status") or ("PASS" if row.get("passed") == "1" else "FAIL_MODEL"),
            "l2_err": float(l2_value) if l2_value else None,
            "torque_l2_err_nm": float(l2_value) if l2_value else None,
            "schema_version": schema_version,
            "abs_err": [],
            "signed_err": [],
        }
        v3_indices = []
        for key in row:
            match = re.fullmatch(r"q_cmd_J(\d+)_rad", key)
            if match:
                v3_indices.append(int(match.group(1)))
        if v3_indices:
            for i in sorted(v3_indices):
                abs_key = f"abs_err_J{i}_nm"
                tau_sim_val = row.get(f"mujoco_tau_J{i}_nm") or row.get(f"genesis_tau_J{i}_nm")
                tau_r_val = row.get(f"sdk_tau_mean_J{i}_nm")
                rec["abs_err"].append(float(row[abs_key]) if row.get(abs_key) else None)
                if tau_sim_val and tau_r_val:
                    rec["signed_err"].append(float(tau_sim_val) - float(tau_r_val))
                else:
                    rec["signed_err"].append(None)
            records.append(rec)
            continue

        indices = sorted(int(key.replace("q", "")) for key in row if key.startswith("q") and key[1:].isdigit())
        for i in indices:
            abs_key = f"abs_err{i}"
            tau_r_key = f"tau_real{i}"
            tau_g_key = f"pd_hold_tau{i}" if row.get(f"pd_hold_tau{i}") not in (None, "") else f"tau_g{i}"
            rec["abs_err"].append(float(row[abs_key]) if row.get(abs_key) else None)
            if row.get(tau_g_key) and row.get(tau_r_key):
                rec["signed_err"].append(float(row[tau_g_key]) - float(row[tau_r_key]))
            else:
                rec["signed_err"].append(None)
        records.append(rec)
    return records


def compare_report_records(
    old_records: Sequence[dict[str, Any]],
    new_records: Sequence[dict[str, Any]],
) -> list[dict[str, float]]:
    """Compare per-joint signed residual distributions between two reports."""
    out: list[dict[str, float]] = []
    dofs = {
        len(record["signed_err"])
        for record in (*old_records, *new_records)
        if record.get("signed_err")
    }
    if len(dofs) > 1:
        raise ValueError(
            f"cannot compare reports with mismatched joint counts: {sorted(dofs)}"
        )
    dof = next(iter(dofs), 0)
    for j in range(dof):
        old_vals = [
            float(r["signed_err"][j]) for r in old_records if r.get("signed_err") and r["signed_err"][j] is not None
        ]
        new_vals = [
            float(r["signed_err"][j]) for r in new_records if r.get("signed_err") and r["signed_err"][j] is not None
        ]
        old_arr = np.asarray(old_vals, dtype=np.float64)
        new_arr = np.asarray(new_vals, dtype=np.float64)
        old_rmse = float(np.sqrt(np.mean(old_arr**2))) if old_arr.size else float("nan")
        new_rmse = float(np.sqrt(np.mean(new_arr**2))) if new_arr.size else float("nan")
        old_bias = float(np.mean(old_arr)) if old_arr.size else float("nan")
        new_bias = float(np.mean(new_arr)) if new_arr.size else float("nan")
        out.append(
            {
                "joint": float(j + 1),
                "old_bias": old_bias,
                "new_bias": new_bias,
                "bias_delta": new_bias - old_bias if np.isfinite(old_bias) and np.isfinite(new_bias) else float("nan"),
                "old_rmse": old_rmse,
                "new_rmse": new_rmse,
                "rmse_delta": new_rmse - old_rmse if np.isfinite(old_rmse) and np.isfinite(new_rmse) else float("nan"),
            }
        )
    return out


# ----------------------------------------------------------------------
# Torque plotting
# ----------------------------------------------------------------------

def torque_plot_layout(dof: int) -> tuple[int, int]:
    """Select subplot grid rows/cols for ``dof`` joints."""
    if dof == 6:
        return 2, 3
    if dof == 7:
        return 3, 3
    return max(1, int(math.ceil(dof / 3))), 3


def write_torque_plot(
    results: Sequence[DynamicsSample],
    path: str | Path,
    *,
    runtime_profile: Any | None = None,
) -> bool:
    """Write per-joint MuJoCo theory vs SDK mean±std torque plot.

    Returns ``False`` when no sample carries SDK torque data (dry-run / sim-only).
    """
    samples = [r for r in results if has_sdk_torque(r)]
    if not samples:
        return False

    dof = max((len(joint_torque_rows(r, runtime_profile=runtime_profile)) for r in samples), default=0)
    if dof <= 0:
        return False

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    rows, cols = torque_plot_layout(dof)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.6, rows * 3.0), squeeze=False)
    pose_names = [r.pose for r in samples]
    x = np.arange(len(samples), dtype=np.float64)

    for joint_idx in range(dof):
        ax = axes[joint_idx // cols][joint_idx % cols]
        sim_vals: list[float] = []
        sdk_vals: list[float] = []
        sdk_std_vals: list[float] = []
        abs_err_vals: list[float] = []
        for sample in samples:
            joint_rows = joint_torque_rows(sample, runtime_profile=runtime_profile)
            row = joint_rows[joint_idx] if joint_idx < len(joint_rows) else {}
            sim_vals.append(float(row["mujoco_tau_nm"]) if row.get("mujoco_tau_nm") is not None else np.nan)
            sdk_vals.append(float(row["sdk_tau_mean_nm"]) if row.get("sdk_tau_mean_nm") is not None else np.nan)
            sdk_std_vals.append(float(row["sdk_tau_std_nm"]) if row.get("sdk_tau_std_nm") is not None else np.nan)
            abs_err_vals.append(float(row["abs_err_nm"]) if row.get("abs_err_nm") is not None else np.nan)

        sim_arr = np.asarray(sim_vals, dtype=np.float64)
        sdk = np.asarray(sdk_vals, dtype=np.float64)
        sdk_std = np.nan_to_num(np.asarray(sdk_std_vals, dtype=np.float64), nan=0.0)
        abs_err = np.asarray(abs_err_vals, dtype=np.float64)

        ax.plot(x, sim_arr, marker="o", linewidth=1.4, label="MuJoCo theory")
        ax.errorbar(x, sdk, yerr=sdk_std, marker="s", linewidth=1.2, capsize=2.0, label="SDK mean")
        max_err = float(np.nanmax(abs_err)) if np.isfinite(abs_err).any() else float("nan")
        title = f"J{joint_idx + 1}"
        if math.isfinite(max_err):
            title += f"  max |err|={max_err:.3f} Nm"
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Torque (Nm)")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(pose_names, rotation=80, ha="right", fontsize=7)
        if joint_idx == 0:
            ax.legend(fontsize=8)

    for unused_idx in range(dof, rows * cols):
        axes[unused_idx // cols][unused_idx % cols].axis("off")

    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return True
