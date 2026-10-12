#!/usr/bin/env python3
"""Dynamic cross-validation of the xarm6 MuJoCo model against the real robot.

Two phases:

1. RECORD (needs the real robot): read tcp_load, execute a few slow joint-space
   sweeps while sampling (q, qd, tau_est) from the xarm SDK, and save a CSV.
2. REPLAY (offline): smooth the recorded trajectory, differentiate to qdd,
   run MuJoCo inverse dynamics with the same payload, and report per-joint
   RMSE / max / correlation of real vs simulated torque — for full inverse
   dynamics and for a gravity-only baseline.

Usage:
    python scripts/compare_dynamics_xarm6.py                      # record + replay
    python scripts/compare_dynamics_xarm6.py --no-move            # static only
    python scripts/compare_dynamics_xarm6.py --replay build/dynamics/run_xxx.csv
    python scripts/compare_dynamics_xarm6.py --payload 3 0 0 21   # mm, flange frame
"""
import argparse
import csv
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dynamics import ArmDynamics  # noqa: E402

IP = "192.168.1.60"
ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "xarm6" / "xarm6.xml"
OUT_DIR = ROOT / "build" / "dynamics"

# Joint sweeps: slow enough that qdd from smoothed finite differences stays
# meaningful, large enough that gravity and velocity terms vary.
SEGMENTS = [
    [0.0, -0.5, -0.8, 0.0, 0.6, 0.0],
    [0.0, -1.2, -0.3, 0.5, 1.2, 0.0],
    [0.3, -0.3, -1.5, -0.5, 0.2, 0.8],
]
SPEED = 0.5    # rad/s, conservative for a 3 kg payload
ACC = 2.0      # rad/s^2


def sample(arm):
    code, states = arm.get_joint_states()
    if code != 0:
        raise RuntimeError(f"get_joint_states failed: {code}")
    pos, vel, _eff = states
    code_t, tau = arm.get_joints_torque()
    if code_t != 0:
        raise RuntimeError(f"get_joints_torque failed: {code_t}")
    return np.asarray(pos[:6]), np.asarray(vel[:6]), np.asarray(tau[:6])


def record(arm, payload, hz=100.0):
    """Sample in a background thread while the main thread drives the sweeps.

    The real robot's get_joint_states reports zero velocity, so the replay
    derives qd/qdd from smoothed positions instead of trusting that column.
    """
    rows, t0 = [], time.monotonic()
    stop = threading.Event()

    def sampler():
        period = 1.0 / hz
        while not stop.is_set():
            try:
                q, qd, tau = sample(arm)
                rows.append((time.monotonic() - t0, *q, *qd, *tau))
            except RuntimeError:
                pass  # transient read failure: skip this tick
            stop.wait(period)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    try:
        for target in SEGMENTS:
            code = arm.set_servo_angle(angle=target, speed=SPEED, acc=ACC,
                                       is_radian=True, wait=True)
            if code != 0:
                raise RuntimeError(f"set_servo_angle failed: {code}")
            time.sleep(0.5)  # settle; sampler keeps recording the tail
    finally:
        stop.set()
        th.join()
    return np.asarray(rows)


def smooth(x, win):
    """Centered moving average with edge replication (window odd)."""
    if win <= 1:
        return x.copy()
    pad = win // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    ker = np.ones(win) / win
    return np.apply_along_axis(lambda c: np.convolve(c, ker, mode="valid"), 0, xp)


def replay(rows, payload, label=""):
    """Offline inverse-dynamics replay of recorded (t, q, tau) rows.

    The real robot reports zero joint velocity, so qd/qdd are derived from
    smoothed positions. Spikes (communication glitches, motion gaps) are
    despiked before differentiation.
    """
    t = rows[:, 0] - rows[0, 0]
    q = rows[:, 1:7]
    tau_real = rows[:, 13:19]

    # despike: at <=0.5 rad/s and ~100 Hz, |dq| > 0.05 rad is a glitch
    dq = np.abs(np.diff(q, axis=0)).max(axis=1)
    bad = np.concatenate([[False], dq > 0.05])
    n_bad = int(bad.sum())
    t, q, tau_real = t[~bad], q[~bad], tau_real[~bad]
    dt = float(np.median(np.diff(t)))
    print(f"\n[{label}] samples={len(t)}  dt={dt * 1e3:.2f} ms  "
          f"duration={t[-1]:.2f} s  despiked={n_bad}  "
          f"payload={payload[0]} kg @ {payload[1]} m")

    # smooth + differentiate; qdd from smoothed position derivative
    qs = smooth(q, 5)
    qds = np.gradient(qs, t, axis=0, edge_order=2)
    qdd = np.gradient(qds, t, axis=0, edge_order=2)

    dyn = ArmDynamics(MODEL, payload=payload)
    tau_id = np.array([dyn.inverse_dynamics(qi, qdi, qddi)
                       for qi, qdi, qddi in zip(qs, qds, qdd)])
    tau_rne = tau_id + np.array([dyn._passive(qi, qdi)[:dyn.n_arm]
                                 for qi, qdi in zip(qs, qds)])  # add passive back
    tau_g = np.array([dyn.gravity(qi) for qi in qs])

    print(f"{'joint':<6} {'RMSE_id':>9} {'RMSE_rne':>9} {'RMSE_g':>9} {'max_id':>9} "
          f"{'mean_id':>9} {'corr_id':>8} {'|tau|max':>9}")
    summary = []
    for j in range(6):
        e_id = tau_real[:, j] - tau_id[:, j]
        e_rne = tau_real[:, j] - tau_rne[:, j]
        e_g = tau_real[:, j] - tau_g[:, j]
        rmse_id = float(np.sqrt(np.mean(e_id ** 2)))
        rmse_rne = float(np.sqrt(np.mean(e_rne ** 2)))
        rmse_g = float(np.sqrt(np.mean(e_g ** 2)))
        if np.std(tau_real[:, j]) < 1e-9 or np.std(tau_id[:, j]) < 1e-9:
            corr = float("nan")  # constant column (e.g. J1 carries no load here)
        else:
            corr = float(np.corrcoef(tau_real[:, j], tau_id[:, j])[0, 1])
        row = (j + 1, rmse_id, rmse_rne, rmse_g, float(np.max(np.abs(e_id))),
               float(np.mean(e_id)), corr, float(np.max(np.abs(tau_real[:, j]))))
        summary.append(row)
        corr_s = f"{corr:>8.3f}" if np.isfinite(corr) else "     n/a"
        print(f"J{j + 1:<4} {rmse_id:>9.3f} {rmse_rne:>9.3f} {rmse_g:>9.3f} "
              f"{np.max(np.abs(e_id)):>9.3f} {np.mean(e_id):>9.3f} {corr_s} "
              f"{np.max(np.abs(tau_real[:, j])):>9.3f}")
    print("(RMSE_id: full ID incl. passive-damping subtraction; RMSE_rne: pure "
          "M qdd + C qd + g without passive)")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ip", default=IP)
    parser.add_argument("--no-move", action="store_true", help="skip dynamic sweeps")
    parser.add_argument("--replay", metavar="CSV", help="replay a previously recorded CSV")
    parser.add_argument("--payload", type=float, nargs=4, metavar=("M", "X", "Y", "Z"),
                        help="payload kg + CoM mm in flange frame (default: read from robot)")
    args = parser.parse_args()

    if args.replay:
        rows = np.loadtxt(args.replay, delimiter=",", skiprows=1)
        payload = tuple(args.payload[:1] + [np.asarray(args.payload[1:]) / 1000.0]) \
            if args.payload else (3.0, [0.0, 0.0, 0.021])
        replay(rows, payload, label=Path(args.replay).stem)
        return

    from xarm.wrapper import XArmAPI
    arm = XArmAPI(args.ip, is_radian=True)
    try:
        load = arm.tcp_load
        payload = (args.payload[0], list(np.asarray(args.payload[1:]) / 1000.0)) \
            if args.payload else (load[0], [c / 1000.0 for c in load[1]])
        print(f"robot tcp_load = {load} -> payload {payload[0]} kg @ {payload[1]} m")
        code_e, (err, warn) = arm.get_err_warn_code()
        if err:
            raise RuntimeError(f"robot in error state: {err}")
        arm.motion_enable(True)
        arm.set_mode(0)
        arm.set_state(0)

        q, _, tau = sample(arm)
        dyn = ArmDynamics(MODEL, payload=payload)
        g = dyn.gravity(q)
        print("\nstatic baseline (current pose):")
        print(f"{'joint':<6} {'real':>9} {'gravity':>9} {'delta':>8}")
        for j in range(6):
            print(f"J{j + 1:<4} {tau[j]:>9.3f} {g[j]:>9.3f} {tau[j] - g[j]:>8.3f}")

        if args.no_move:
            return
        rows = record(arm, payload)
    finally:
        arm.disconnect()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"run_{stamp}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + [f"q{j}" for j in range(6)]
                   + [f"qd{j}" for j in range(6)] + [f"tau{j}" for j in range(6)])
        w.writerows(rows)
    print(f"\nsaved {csv_path}")
    replay(rows, payload, label=csv_path.stem)


if __name__ == "__main__":
    main()
