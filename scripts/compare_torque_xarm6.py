"""Compare real xArm6 (192.168.1.60) holding torque vs MuJoCo static torque.

Protocol:
1. Read TCP load (mass + CoM) from the robot and replicate it in the MuJoCo
   xarm6 model as a payload body attached at attachment_site (flange frame).
2. Record the robot's *measured* joint angles and estimated joint torques
   (get_joints_torque) at the current pose.
3. Move the robot to READY [0,-0.8,-1.2,1.0,0.3,0] and record again.
4. For each recorded pose, set the MuJoCo qpos to the measured angles, run
   mj_forward, and read qfrc_bias (gravity + bias torque; pure gravity at rest).
5. Print a per-joint comparison plus an FK cross-check of the TCP position.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dynamics import ArmDynamics  # noqa: E402

IP = "192.168.1.60"
READY = [0, -0.8, -1.2, 1.0, 0.3, 0]
ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "xarm6" / "xarm6.xml"


def record(arm, label):
    code_a, angles = arm.get_servo_angle()
    code_t, torques = arm.get_joints_torque()
    code_p, pose = arm.get_position(is_radian=True)
    if code_a or code_t or code_p:
        raise RuntimeError(f"read failed at {label}: angle={code_a} torque={code_t} pose={code_p}")
    q = np.array(angles[:6], dtype=float)
    tau = np.array(torques[:6], dtype=float)
    tcp = np.array(pose[:3], dtype=float)
    rpy = np.array(pose[3:6], dtype=float)
    print(f"[{label}] q = {np.round(q, 4).tolist()}")
    print(f"[{label}] tcp(mm) = {np.round(tcp, 2).tolist()} rpy = {np.round(pose[3:], 4).tolist()}")
    print(f"[{label}] tau_real = {np.round(tau, 3).tolist()}")
    return {"label": label, "q": q, "tau": tau, "tcp": tcp, "rpy": rpy}


def main():
    from xarm.wrapper import XArmAPI

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-move", action="store_true",
                        help="only record the current pose (skip the move to READY)")
    args = parser.parse_args()

    arm = XArmAPI(IP, is_radian=True)
    records = []
    try:
        load = arm.tcp_load  # [mass_kg, [com_x, com_y, com_z]] mm in flange frame
        print(f"robot tcp_load = {load}")
        try:
            print(f"robot tcp_offset = {arm.tcp_offset}")
        except Exception:  # noqa: BLE001
            pass
        code_e, (err, warn) = arm.get_err_warn_code()
        print(f"err_warn = ({err}, {warn}), state = {arm.get_state()}, mode = {arm.mode}")
        if err:
            raise RuntimeError(f"robot in error state: {err}")

        records.append(record(arm, "current"))

        if not args.no_move:
            # move to READY (speed 0.6 rad/s, acc 4 rad/s^2)
            arm.motion_enable(True)
            arm.set_mode(0)
            arm.set_state(0)
            code = arm.set_servo_angle(angle=READY, speed=0.6, acc=4, is_radian=True, wait=True)
            if code != 0:
                raise RuntimeError(f"set_servo_angle failed: {code}")
            time.sleep(1.5)
            records.append(record(arm, "ready"))
    finally:
        arm.disconnect()

    # --- MuJoCo model with identical TCP payload ---
    m, com = load[0], load[1]
    payload = (m, [c / 1000 for c in com])
    model = ArmDynamics(MODEL, payload=payload).model
    data = mujoco.MjData(model)
    site_id = model.site("attachment_site").id
    print(f"\nMuJoCo model: nq={model.nq}, payload {m} kg @ {[c / 1000 for c in com]} m (flange frame)")

    print(f"\n{'pose':<8} {'joint':<6} {'real(Nm)':>9} {'mujoco(Nm)':>11} {'delta':>8} {'err%':>7}")
    for rec in records:
        data.qpos[:] = rec["q"]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        tau_sim = data.qfrc_bias[:6].copy()

        # FK cross-check: real pose is the TCP frame (robot tcp_offset=42mm
        # along flange z); sim flange = attachment_site.
        r, p, y = rec["rpy"]
        cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
        R_real = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ])
        flange_real = rec["tcp"] - R_real @ np.array([0, 0, 42.0])
        R_sim = data.site_xmat[site_id].reshape(3, 3)
        tcp_sim = data.site_xpos[site_id] * 1000 + R_sim @ np.array([0, 0, 42.0])
        print(f"{rec['label']:<8} flange(mm) sim={np.round(data.site_xpos[site_id] * 1000, 2).tolist()} real={np.round(flange_real, 2).tolist()}")
        print(f"{rec['label']:<8} tcp(mm)    sim={np.round(tcp_sim, 2).tolist()} real={np.round(rec['tcp'], 2).tolist()}")

        for i in range(6):
            d = rec["tau"][i] - tau_sim[i]
            pct = abs(d) / max(abs(rec["tau"][i]), 1.0) * 100
            print(f"{rec['label']:<8} J{i + 1:<3} {rec['tau'][i]:>9.3f} {tau_sim[i]:>11.3f} {d:>8.3f} {pct:>6.1f}%")
        print()


if __name__ == "__main__":
    main()
