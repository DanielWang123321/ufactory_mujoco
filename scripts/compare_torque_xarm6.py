#!/usr/bin/env python3
"""Compare real xArm6 (192.168.1.60) holding torque vs MuJoCo static torque.

Protocol:
1. Read TCP load (mass + CoM) and TCP offset from the robot and replicate the
   payload in the MuJoCo xarm6 model at attachment_site (flange frame).
2. If --calibration is provided, verify SN + SHA-256 and patch the MuJoCo model.
3. Record the robot's measured joint angles and estimated joint torques at the
   current pose (strictly read-only when --no-move is passed).
4. Moving to READY requires --calibration, preflight PASS, and --confirm-real.
5. Print a per-joint comparison plus an FK cross-check of the TCP position.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mujoco
import numpy as np

from ufactory_mujoco.calibration import (  # noqa: E402
    apply_calibration_to_model,
    flange_m_quat_to_tcp_mm_rpy,
    pose_tcp_to_flange,
    read_tcp_offset,
)
from ufactory_mujoco.collision import CollisionChecker  # noqa: E402
from ufactory_mujoco.dynamics import ArmDynamics  # noqa: E402
from ufactory_mujoco.safety import (  # noqa: E402
    authorize_real_motion,
    verify_hardware_identity_and_calibration,
)
from ufactory_mujoco.trajectory import plan_ptp_quintic  # noqa: E402

IP = "192.168.1.60"
READY = [0.0, -0.8, -1.2, 1.0, 0.3, 0.0]
ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "xarm6" / "xarm6.xml"
SCENE = ROOT / "models" / "xarm6" / "scene.xml"


def record(arm, label: str) -> dict:
    code_a, angles = arm.get_servo_angle(is_radian=True)
    code_t, torques = arm.get_joints_torque()
    code_p, pose = arm.get_position(is_radian=True)
    if code_a or code_t or code_p:
        raise RuntimeError(f"read failed at {label}: angle={code_a} torque={code_t} pose={code_p}")
    q = np.array(angles[:6], dtype=float)
    tau = np.array(torques[:6], dtype=float)
    tcp = np.array(pose[:3], dtype=float)
    rpy = np.array(pose[3:6], dtype=float)
    print(f"[{label}] q = {np.round(q, 4).tolist()}")
    print(f"[{label}] tcp(mm) = {np.round(tcp, 2).tolist()} rpy = {np.round(pose[3:6], 4).tolist()}")
    print(f"[{label}] tau_real = {np.round(tau, 3).tolist()}")
    return {"label": label, "q": q, "tau": tau, "tcp": tcp, "rpy": rpy}


def _preflight_segment(q0: np.ndarray, q1: np.ndarray, calib=None) -> bool:
    checker = CollisionChecker(SCENE, ignore_pairs=[("link_base", "world")])
    if calib is not None:
        apply_calibration_to_model(checker.model, calib, checker.data)
    traj = plan_ptp_quintic(q0, q1, duration=2.0, dt=0.05)
    for q in traj.q:
        if np.any(q < checker.q_lo - 1e-6) or np.any(q > checker.q_hi + 1e-6):
            return False
        if checker.in_collision(q):
            return False
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default=IP, help="xArm controller IPv4 address")
    parser.add_argument("--no-move", action="store_true",
                        help="only record the current pose (skip the move to READY)")
    parser.add_argument("--calibration", type=Path, default=None,
                        help="robot-bound kinematic calibration YAML (required when moving)")
    parser.add_argument("--confirm-real", action="store_true",
                        help="explicitly authorize moving the real robot to READY")
    parser.add_argument("--report", type=Path, default=None,
                        help="optional JSON output report path")
    args = parser.parse_args(argv)

    from xarm.wrapper import XArmAPI

    arm = XArmAPI(args.ip, is_radian=True)
    if type(arm).__module__.startswith("xarm"):
        time.sleep(0.35)
    records = []
    calib = None
    sn = None
    try:
        load = arm.tcp_load  # [mass_kg, [com_x, com_y, com_z]] mm in flange frame
        tcp_offset = read_tcp_offset(arm)
        print(f"robot tcp_load = {load}")
        print(f"robot tcp_offset = {tcp_offset.tolist()}")
        code_e, (err, warn) = arm.get_err_warn_code()
        print(f"err_warn = ({err}, {warn}), state = {arm.get_state()}, mode = {arm.mode}")
        if err:
            raise RuntimeError(f"robot in error state: {err}")

        if args.calibration is not None:
            sn, calib = verify_hardware_identity_and_calibration(
                arm, robot="xarm6", calibration_path=args.calibration
            )

        current_rec = record(arm, "current")
        records.append(current_rec)

        if not args.no_move:
            preflight_ok = _preflight_segment(
                current_rec["q"], np.asarray(READY, dtype=float), calib=calib
            )
            approval = authorize_real_motion(
                arm,
                robot="xarm6",
                calibration=calib if calib is not None else args.calibration,
                preflight_ok=preflight_ok,
                confirm_real=bool(args.confirm_real),
                cycles=1,
            )
            calib = calib or verify_hardware_identity_and_calibration(
                arm, robot="xarm6", calibration_path=args.calibration
            )[1]
            sn = approval.serial_number
            arm.motion_enable(True)
            arm.set_mode(0)
            arm.set_state(0)
            code = arm.set_servo_angle(angle=READY, speed=0.6, acc=4, is_radian=True, wait=True)
            if code != 0:
                raise RuntimeError(f"set_servo_angle failed: {code}")
            time.sleep(1.5)
            records.append(record(arm, "ready"))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps({"ok": False, "success": False, "error": str(exc)}, indent=2) + "\n",
                encoding="utf-8",
            )
        return 1
    finally:
        arm.disconnect()

    m, com = float(load[0]), [float(c) / 1000.0 for c in load[1]]
    payload = (m, com)
    dyn = ArmDynamics(MODEL, payload=payload)
    model = dyn.model
    data = mujoco.MjData(model)
    if calib is not None:
        apply_calibration_to_model(model, calib, data)
    site_id = model.site("attachment_site").id
    print(f"\nMuJoCo model: nq={model.nq}, payload {m} kg @ {com} m (flange frame)")

    print(f"\n{'pose':<8} {'joint':<6} {'real(Nm)':>9} {'mujoco(Nm)':>11} {'delta':>8} {'err%':>7}")
    report_poses = []
    for rec in records:
        data.qpos[:6] = rec["q"]
        data.qvel[:6] = 0.0
        mujoco.mj_forward(model, data)
        tau_sim = data.qfrc_bias[:6].copy()

        quat_sim = np.empty(4)
        mujoco.mju_mat2Quat(quat_sim, data.site_xmat[site_id])
        tcp_sim_pose = flange_m_quat_to_tcp_mm_rpy(data.site_xpos[site_id], quat_sim, tcp_offset)
        flange_real_pose = pose_tcp_to_flange(
            np.concatenate([rec["tcp"], rec["rpy"]]), tcp_offset
        )
        flange_sim_mm = data.site_xpos[site_id] * 1000.0
        tcp_err_mm = float(np.linalg.norm(tcp_sim_pose[:3] - rec["tcp"]))

        print(
            f"{rec['label']:<8} flange(mm) sim={np.round(flange_sim_mm, 2).tolist()} "
            f"real={np.round(flange_real_pose[:3], 2).tolist()}"
        )
        print(
            f"{rec['label']:<8} tcp(mm)    sim={np.round(tcp_sim_pose[:3], 2).tolist()} "
            f"real={np.round(rec['tcp'], 2).tolist()} (err={tcp_err_mm:.3f} mm)"
        )

        joints_out = []
        for i in range(6):
            d = float(rec["tau"][i] - tau_sim[i])
            pct = abs(d) / max(abs(float(rec["tau"][i])), 1.0) * 100.0
            print(f"{rec['label']:<8} J{i + 1:<3} {rec['tau'][i]:>9.3f} {tau_sim[i]:>11.3f} {d:>8.3f} {pct:>6.1f}%")
            joints_out.append({
                "joint": i + 1,
                "tau_real_nm": float(rec["tau"][i]),
                "tau_mujoco_nm": float(tau_sim[i]),
                "delta_nm": d,
                "rel_error_pct": pct,
            })
        print()
        report_poses.append({
            "label": rec["label"],
            "q_rad": rec["q"].tolist(),
            "tcp_real_mm": rec["tcp"].tolist(),
            "tcp_mujoco_mm": tcp_sim_pose[:3].tolist(),
            "tcp_pos_error_mm": tcp_err_mm,
            "joints": joints_out,
        })

    if args.report is not None:
        out_payload = {
            "schema_version": 1,
            "mode": "read-only" if args.no_move else "real",
            "ok": True,
            "success": True,
            "serial_number": sn,
            "calibration_sha256": None if calib is None else calib.sha256,
            "payload": {"mass_kg": m, "com_m": com},
            "tcp_offset_mm_rad": tcp_offset.tolist(),
            "poses": report_poses,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(out_payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
