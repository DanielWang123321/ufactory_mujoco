"""Read-only FK/IK cross-validation between calibrated MuJoCo models and xArm controller."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np

from .calibration import (
    KinematicsCalibration,
    apply_calibration_to_model,
    canonical_robot_key,
    flange_m_quat_to_tcp_mm_rpy,
    read_tcp_offset,
    tcp_mm_rpy_to_flange_m_quat,
)
from .ik import solve_ik
from .kinematics import fk_pose
from .model_paths import MODELS_ROOT
from .robots import ARMS
from .safety import (
    SafetyGateError,
    read_arm_joint_positions,
    verify_hardware_identity_and_calibration,
)
from .xarm_robot import _connect_xarm_sdk

PASS_POS_MM = 1.0
PASS_RPY_DEG = 0.5


def angle_diff_deg(a_rad: float, b_rad: float) -> float:
    """Shortest angular difference in degrees between two angles in radians."""
    diff = (float(a_rad) - float(b_rad) + math.pi) % (2.0 * math.pi) - math.pi
    return abs(diff) * 180.0 / math.pi


def default_validation_poses(
    robot: str,
    current_q: np.ndarray | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Standard joint configurations (rad) for FK/IK verification."""
    key = canonical_robot_key(robot)
    dof = ARMS[key].dof
    ready_map = {
        "lite6": np.deg2rad([0.0, 9.9, 31.8, 0.0, 21.9, 0.0]),
        "xarm5": np.array([0.0, -0.5, -0.8, 0.5, 0.0]),
        "xarm6": np.array([0.0, -0.8, -1.2, 1.0, 0.3, 0.0]),
        "xarm7": np.array([0.0, -0.5, 0.0, 0.6, 0.0, 0.5, 0.0]),
        "uf850": np.array([0.0, -0.5, -0.8, 0.0, 0.5, 0.0]),
    }
    poses: list[tuple[str, np.ndarray]] = [
        ("home", np.zeros(dof, dtype=np.float64)),
        ("ready", np.asarray(ready_map[key], dtype=np.float64)),
        (
            "pose_a",
            np.asarray([0.5, -0.3, 0.0, 0.0, 0.3, *([0.0] * max(0, dof - 5))], dtype=np.float64)[:dof],
        ),
    ]
    if dof >= 6:
        poses.append(
            (
                "pose_b",
                np.asarray([0.0, -0.5, -0.1, 0.5, 0.5, 0.0, *([0.0] * max(0, dof - 6))], dtype=np.float64)[:dof],
            )
        )
    if current_q is not None:
        poses.append(("current", np.asarray(current_q, dtype=np.float64).reshape(dof).copy()))
    return poses


def validate_fk_ik_with_controller(
    arm: Any,
    *,
    robot: str = "xarm6",
    calibration: KinematicsCalibration | str | Path,
    pos_tol_mm: float = PASS_POS_MM,
    rpy_tol_deg: float = PASS_RPY_DEG,
) -> dict[str, Any]:
    """Compare calibrated MuJoCo FK/IK against controller SDK FK/IK (strictly read-only)."""
    robot_key = canonical_robot_key(robot)
    sn, calib = verify_hardware_identity_and_calibration(
        arm, robot=robot_key, calibration_path=calibration
    )
    dof = ARMS[robot_key].dof
    site_name = ARMS[robot_key].tcp_site
    xml_path = MODELS_ROOT / robot_key / f"{robot_key}.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    eval_data = mujoco.MjData(model)
    apply_calibration_to_model(model, calib, data)
    mujoco.mj_setConst(model, eval_data)
    mujoco.mj_forward(model, eval_data)

    tcp_offset = read_tcp_offset(arm)
    current_q = read_arm_joint_positions(arm, dof)
    poses = default_validation_poses(robot_key, current_q=current_q)

    fk_results: list[dict[str, Any]] = []
    ik_results: list[dict[str, Any]] = []
    failures: list[str] = []

    for name, q in poses:
        code_fk, sdk_pose_raw = arm.get_forward_kinematics(
            q.tolist(), input_is_radian=True, return_is_radian=True
        )
        if int(code_fk) != 0:
            raise SafetyGateError(f"controller get_forward_kinematics({name}) failed: code={code_fk}")
        sdk_tcp = np.asarray(sdk_pose_raw[:6], dtype=np.float64)

        mj_pos_m, mj_quat = fk_pose(model, data, site_name, q)
        mj_tcp = flange_m_quat_to_tcp_mm_rpy(mj_pos_m, mj_quat, tcp_offset)

        pos_err_mm = float(np.linalg.norm(mj_tcp[:3] - sdk_tcp[:3]))
        rpy_err_deg = float(max(angle_diff_deg(a, b) for a, b in zip(mj_tcp[3:6], sdk_tcp[3:6])))
        fk_ok = bool(pos_err_mm < pos_tol_mm and rpy_err_deg < rpy_tol_deg)
        if not fk_ok:
            failures.append(
                f"FK[{name}]: pos_err={pos_err_mm:.4f} mm (tol {pos_tol_mm}), "
                f"rpy_err={rpy_err_deg:.4f} deg (tol {rpy_tol_deg})"
            )
        fk_results.append({
            "pose": name,
            "q_rad": q.tolist(),
            "sdk_tcp_mm_rad": sdk_tcp.tolist(),
            "mujoco_tcp_mm_rad": mj_tcp.tolist(),
            "pos_error_mm": pos_err_mm,
            "rpy_error_deg": rpy_err_deg,
            "passed": fk_ok,
        })

        # IK cross-check around each pose (perturbed seed)
        target_flange_m, target_quat = tcp_mm_rpy_to_flange_m_quat(sdk_tcp, tcp_offset)
        seed_q = np.clip(q + 0.05, model.jnt_range[:dof, 0], model.jnt_range[:dof, 1])
        q_ik, ik_converged = solve_ik(
            model, eval_data, site_name, target_flange_m, target_quat, q_init=seed_q
        )
        q_ik = np.asarray(q_ik[:dof], dtype=np.float64)
        ik_pos_m, ik_quat = fk_pose(model, data, site_name, q_ik)
        ik_mj_tcp = flange_m_quat_to_tcp_mm_rpy(ik_pos_m, ik_quat, tcp_offset)

        code_fk2, ik_sdk_raw = arm.get_forward_kinematics(
            q_ik.tolist(), input_is_radian=True, return_is_radian=True
        )
        if int(code_fk2) != 0:
            raise SafetyGateError(
                f"controller get_forward_kinematics(IK[{name}]) failed: code={code_fk2}"
            )
        ik_sdk_tcp = np.asarray(ik_sdk_raw[:6], dtype=np.float64)

        ik_pos_err_mm = float(np.linalg.norm(ik_sdk_tcp[:3] - sdk_tcp[:3]))
        ik_rpy_err_deg = float(
            max(angle_diff_deg(a, b) for a, b in zip(ik_sdk_tcp[3:6], sdk_tcp[3:6]))
        )
        # Also query controller IK if available
        sdk_ik_code = None
        ik_getter = getattr(arm, "get_inverse_kinematics", None)
        if callable(ik_getter):
            res_ik = ik_getter(sdk_tcp.tolist(), input_is_radian=True, return_is_radian=True)
            if isinstance(res_ik, tuple) and len(res_ik) == 2:
                sdk_ik_code = int(res_ik[0])

        ik_ok = bool(ik_converged and ik_pos_err_mm < pos_tol_mm and ik_rpy_err_deg < rpy_tol_deg)
        if not ik_ok:
            failures.append(
                f"IK[{name}]: converged={ik_converged}, pos_err={ik_pos_err_mm:.4f} mm, "
                f"rpy_err={ik_rpy_err_deg:.4f} deg"
            )
        ik_results.append({
            "pose": name,
            "q_target_rad": q.tolist(),
            "q_solved_rad": q_ik.tolist(),
            "converged": bool(ik_converged),
            "sdk_ik_code": sdk_ik_code,
            "mujoco_reproj_tcp_mm_rad": ik_mj_tcp.tolist(),
            "controller_reproj_tcp_mm_rad": ik_sdk_tcp.tolist(),
            "pos_error_mm": ik_pos_err_mm,
            "rpy_error_deg": ik_rpy_err_deg,
            "passed": ik_ok,
        })

    max_fk_pos = max((r["pos_error_mm"] for r in fk_results), default=0.0)
    max_fk_rpy = max((r["rpy_error_deg"] for r in fk_results), default=0.0)
    max_ik_pos = max((r["pos_error_mm"] for r in ik_results), default=0.0)
    max_ik_rpy = max((r["rpy_error_deg"] for r in ik_results), default=0.0)
    passed = not failures

    return {
        "schema_version": 1,
        "mode": "read-only-fk-ik",
        "passed": passed,
        "success": passed,
        "completed": True,
        "robot": robot_key,
        "serial_number": sn,
        "calibration_sha256": calib.sha256,
        "tcp_offset_mm_rad": tcp_offset.tolist(),
        "thresholds": {
            "pos_tol_mm": pos_tol_mm,
            "rpy_tol_deg": rpy_tol_deg,
        },
        "max_fk_pos_error_mm": max_fk_pos,
        "max_fk_rpy_error_deg": max_fk_rpy,
        "max_ik_pos_error_mm": max_ik_pos,
        "max_ik_rpy_error_deg": max_ik_rpy,
        "fk": fk_results,
        "ik": ik_results,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ufactory_mujoco.kinematics_validation",
        description=__doc__,
    )
    parser.add_argument("--robot", choices=tuple(ARMS), default="xarm6")
    parser.add_argument("--ip", default=os.environ.get("XARM_IP"), help="controller IPv4 address")
    parser.add_argument("--calibration", type=Path, help="robot-bound calibration YAML")
    parser.add_argument("--report", type=Path, help="output JSON report path")
    args = parser.parse_args(argv)

    if not args.ip:
        print("Error: --ip or XARM_IP is required", file=sys.stderr)
        return 2
    if args.calibration is None:
        print("Error: --calibration is required", file=sys.stderr)
        return 2

    arm = None
    try:
        arm = _connect_xarm_sdk(args.ip)
        report = validate_fk_ik_with_controller(
            arm, robot=args.robot, calibration=args.calibration
        )
        for fk_row, ik_row in zip(report["fk"], report["ik"]):
            print(
                f"  [{fk_row['pose']:<7}] FK err: {fk_row['pos_error_mm']:.4f} mm / "
                f"{fk_row['rpy_error_deg']:.4f} deg | "
                f"IK err: {ik_row['pos_error_mm']:.4f} mm / {ik_row['rpy_error_deg']:.4f} deg"
            )
        print(
            f"FK/IK validation ({report['robot']} {report['serial_number']}): "
            f"{'PASS' if report['passed'] else 'FAIL'}"
        )
        code = 0 if report["passed"] else 1
    except Exception as exc:
        print(f"FK/IK validation failed: {exc}", file=sys.stderr)
        report = {
            "schema_version": 1,
            "mode": "read-only-fk-ik",
            "passed": False,
            "success": False,
            "completed": False,
            "robot": args.robot,
            "error": f"{type(exc).__name__}: {exc}",
        }
        code = 1
    finally:
        if arm is not None and getattr(arm, "connected", False):
            try:
                arm.disconnect()
            except Exception:
                pass

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
