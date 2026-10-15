#!/usr/bin/env python3
"""Example 08: rigid-body dynamics toolkit on the xArm 6 (roadmap 0.0.6).

Demonstrates src/dynamics.py:
1. Mass matrix symmetry and positive definiteness.
2. Gravity / Coriolis decomposition (zero velocity => zero Coriolis term).
3. Inverse/forward dynamics round trip (strictly reciprocal).
4. TCP payload injection cross-checked against the 2026-10-10 real-robot
   static holding torques (3 kg at [0, 0, 21] mm -> J2 -5.46 Nm, J3 -21.10 Nm).
5. Feedforward torque envelope along a quintic joint trajectory vs the
   official UFACTORY actuator torque limits.

Usage:
    python examples/simulation/08_dynamics.py            # with interactive viewer
    python examples/simulation/08_dynamics.py --headless # metrics only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from src.dynamics import ArmDynamics  # noqa: E402
from src.model_paths import MODELS_ROOT  # noqa: E402
from src.sim_robot import SimRobot  # noqa: E402
from src.trajectory import plan_ptp_quintic  # noqa: E402

XARM6 = MODELS_ROOT / "xarm6" / "xarm6.xml"
SCENE = MODELS_ROOT / "xarm6" / "scene.xml"
READY = np.array([0.0, -0.8, -1.2, 1.0, 0.3, 0.0])
TARGET = np.array([0.5, -1.2, -1.0, 0.8, 0.6, 0.4])
PAYLOAD = (3.0, [0.0, 0.0, 0.021])            # kg, site-frame COM (m)
REAL_J2, REAL_J3 = -5.46, -21.10              # Nm, real robot 2026-10-10
REAL_TOL = 0.05                               # 5 %

rng = np.random.default_rng(0)


def check_mass_matrix(dyn: ArmDynamics) -> tuple[float, float]:
    """(asymmetry, min eigenvalue) of M(READY)."""
    M = dyn.mass_matrix(READY)
    return float(np.max(np.abs(M - M.T))), float(np.min(np.linalg.eigvalsh(M)))


def check_decomposition(dyn: ArmDynamics) -> float:
    """Max |Coriolis term| at zero velocity (must vanish)."""
    return float(np.max(np.abs(dyn.coriolis(READY, np.zeros(6)))))


def check_round_trip(dyn: ArmDynamics) -> float:
    """Max |FD(ID(q,qd,qdd)) - qdd| over random states."""
    err = 0.0
    for _ in range(5):
        q = rng.uniform(-1.0, 1.0, 6)
        qd = rng.uniform(-1.0, 1.0, 6)
        qdd = rng.uniform(-2.0, 2.0, 6)
        tau = dyn.inverse_dynamics(q, qd, qdd)
        err = max(err, float(np.max(np.abs(dyn.forward_dynamics(q, qd, tau) - qdd))))
    return err


def check_payload_against_real_robot() -> tuple[float, float]:
    """(J2, J3) gravity torques with the 3 kg payload at READY."""
    dyn = ArmDynamics(XARM6, payload=PAYLOAD)
    g = dyn.gravity(READY)
    return float(g[1]), float(g[2])


def check_feedforward_envelope(dyn: ArmDynamics) -> tuple[np.ndarray, np.ndarray, float]:
    """(max |tau| per joint, actuator limits, worst utilization)."""
    traj = plan_ptp_quintic(READY, TARGET, max_vel=1.0, max_acc=2.0, dt=0.01)
    tau = dyn.feedforward_traj(traj.q, 0.01)
    peak = np.max(np.abs(tau), axis=0)
    model = mujoco.MjModel.from_xml_path(str(XARM6))
    limits = np.array([model.actuator(i).forcerange[1] for i in range(6)])
    return peak, limits, float(np.max(peak / limits))


def run_checks() -> dict:
    dyn = ArmDynamics(XARM6, payload=PAYLOAD)
    asym, min_eig = check_mass_matrix(dyn)
    corr = check_decomposition(dyn)
    round_trip = check_round_trip(dyn)
    j2, j3 = check_payload_against_real_robot()
    peak, limits, utilization = check_feedforward_envelope(dyn)

    metrics = {
        "mass_asymmetry": asym,
        "mass_min_eigenvalue": min_eig,
        "coriolis_at_rest": corr,
        "id_fd_round_trip": round_trip,
        "payload_j2_nm": j2,
        "payload_j3_nm": j3,
        "ff_peak_tau": peak,
        "actuator_limits": limits,
        "ff_utilization": utilization,
    }
    metrics["success"] = bool(
        asym < 1e-12
        and min_eig > 0.0
        and corr < 1e-9
        and round_trip < 1e-8
        and abs(j2 - REAL_J2) <= abs(REAL_J2) * REAL_TOL
        and abs(j3 - REAL_J3) <= abs(REAL_J3) * REAL_TOL
        and utilization < 1.0
    )
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Rigid-body dynamics demo (xArm 6).")
    ap.add_argument("--headless", action="store_true", help="no viewer, prints metrics")
    args = ap.parse_args()

    metrics = run_checks()

    print(f"\nmass matrix      : asymmetry {metrics['mass_asymmetry']:.1e}, "
          f"min eigenvalue {metrics['mass_min_eigenvalue']:.4f} kg*m^2")
    print(f"coriolis at rest : {metrics['coriolis_at_rest']:.1e} Nm")
    print(f"ID/FD round trip : {metrics['id_fd_round_trip']:.1e} rad/s^2")
    print(f"payload gravity  : J2 {metrics['payload_j2_nm']:.2f} Nm "
          f"(real {REAL_J2}), J3 {metrics['payload_j3_nm']:.2f} Nm (real {REAL_J3})")
    print("feedforward peak : " + ", ".join(
        f"J{i + 1} {t:.1f}/{l:.1f} Nm"
        for i, (t, l) in enumerate(zip(metrics["ff_peak_tau"], metrics["actuator_limits"])))
        + f"  (worst {metrics['ff_utilization']:.1%})")

    if not args.headless:
        with SimRobot(SCENE, viewer=True) as robot:
            robot.move_joints(READY, duration=1.5)
            robot.move_joints(TARGET, duration=3.0)  # the envelope's motion
            robot.move_joints(READY, duration=3.0)

    print("\nDynamics demo " + ("PASS" if metrics["success"] else "FAIL"))
    sys.exit(0 if metrics["success"] else 1)


if __name__ == "__main__":
    main()
