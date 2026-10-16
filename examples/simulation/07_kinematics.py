#!/usr/bin/env python3
"""Example 07: kinematics toolkit on the xArm 6 (roadmap 0.0.6).

Demonstrates ufactory_mujoco/kinematics.py / SimRobot kinematics helpers:
1. Forward kinematics cross-checked against the SimRobot FK/pose API.
2. Geometric Jacobian cross-checked against central finite differences
   (mj_integratePos perturbations, quaternion joints handled safely).
3. Velocity-level damped least-squares IK: joint velocity for a commanded
   Cartesian twist, verified by re-multiplying the Jacobian.
4. Manipulability / condition number: a generic pose vs the all-zero pose --
   a true singularity of the xArm 6 (non-spherical wrist with J4/J6 offsets,
   so there is no closed-form IK; DLS is the right tool).

Usage:
    python examples/simulation/07_kinematics.py            # with interactive viewer
    python examples/simulation/07_kinematics.py --headless # metrics only
"""

import argparse
import sys
from pathlib import Path


import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from ufactory_mujoco import SimRobot  # noqa: E402
from ufactory_mujoco.kinematics import (  # noqa: E402
    condition_number,
    fk_pose,
    pose_jacobian,
    quat_error,
    solve_joint_velocity,
)
from ufactory_mujoco.model_paths import MODELS_ROOT  # noqa: E402

SCENE = MODELS_ROOT / "xarm6" / "scene.xml"
SITE = "attachment_site"
READY = np.array([0.0, -0.8, -1.2, 1.0, 0.3, 0.0])
GENERIC = np.array([0.5, -0.8, -1.2, 1.0, 0.3, 0.3])
ZERO = np.zeros(6)
TWIST = np.array([0.05, -0.03, 0.02, 0.10, -0.15, 0.20])  # [m/s; rad/s]

FD_STEP = 1e-6
JAC_TOL = 1e-3        # m/s or rad/s per unit joint velocity
TWIST_TOL = 1e-3      # m/s, twist reproduction residual


def check_fk(robot: SimRobot, q: np.ndarray) -> float:
    """FK through ufactory_mujoco.kinematics vs the SimRobot pose API; returns max |pos diff|."""
    pos, _quat = fk_pose(robot.model, robot.eval_data, SITE, q)
    return float(np.max(np.abs(pos - robot.fk_positions(q[None, :])[0])))


def check_jacobian(robot: SimRobot, q: np.ndarray, h: float = FD_STEP) -> float:
    """Max element difference between the analytic and central-difference Jacobian."""
    J = pose_jacobian(robot.model, robot.eval_data, SITE, q)
    n = robot.model.nv
    jac_num = np.zeros((6, n))
    for i in range(n):
        dq = np.zeros(n)
        dq[i] = h
        q_plus = np.zeros(robot.model.nq)
        q_plus[: robot._n_arm_qpos] = q
        q_minus = q_plus.copy()
        mujoco.mj_integratePos(robot.model, q_plus, dq, 1.0)
        mujoco.mj_integratePos(robot.model, q_minus, dq, -1.0)
        pos_p, quat_p = fk_pose(robot.model, robot.eval_data, SITE, q_plus)
        pos_m, quat_m = fk_pose(robot.model, robot.eval_data, SITE, q_minus)
        jac_num[:3, i] = (pos_p - pos_m) / (2 * h)
        jac_num[3:, i] = quat_error(quat_m, quat_p) / (2 * h)
    return float(np.max(np.abs(J - jac_num)))


def check_velocity_ik(robot: SimRobot, q: np.ndarray) -> float:
    """Residual ||J @ qd - twist||_inf of the damped least-squares solution."""
    qd = robot.solve_tcp_velocity(TWIST, q, damping=1e-4)
    J = pose_jacobian(robot.model, robot.eval_data, SITE, q)
    return float(np.max(np.abs(J @ qd - TWIST)))


def check_singularity(robot: SimRobot) -> dict:
    w_generic = robot.tcp_manipulability(GENERIC)
    w_zero = robot.tcp_manipulability(ZERO)
    c_generic = condition_number(robot.model, robot.eval_data, SITE, GENERIC)
    c_zero = condition_number(robot.model, robot.eval_data, SITE, ZERO)
    return {
        "manipulability_generic": w_generic,
        "manipulability_zero": w_zero,
        "condition_generic": c_generic,
        "condition_zero": c_zero,
    }


def run_checks(robot: SimRobot) -> dict:
    metrics = {
        "fk_pos_diff_m": check_fk(robot, READY),
        "jacobian_max_diff": check_jacobian(robot, READY),
        "velocity_ik_residual": check_velocity_ik(robot, READY),
    }
    metrics.update(check_singularity(robot))
    metrics["success"] = bool(
        metrics["fk_pos_diff_m"] < 1e-9
        and metrics["jacobian_max_diff"] < JAC_TOL
        and metrics["velocity_ik_residual"] < TWIST_TOL
        and metrics["condition_zero"] > 1e6                 # true singularity
        and metrics["manipulability_zero"] < 1e-6 * metrics["manipulability_generic"]
        and metrics["condition_generic"] < 1e3              # regular pose
    )
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Kinematics toolkit demo (xArm 6).")
    ap.add_argument("--headless", action="store_true", help="no viewer, print metrics")
    args = ap.parse_args()

    with SimRobot(SCENE, viewer=not args.headless) as robot:
        metrics = run_checks(robot)

    print(f"\nFK vs SimRobot pose API       : {metrics['fk_pos_diff_m']:.2e} m")
    print(f"analytic vs numeric Jacobian  : {metrics['jacobian_max_diff']:.2e}")
    print(f"velocity-IK twist residual    : {metrics['velocity_ik_residual']:.2e}")
    print(f"manipulability (generic/zero) : {metrics['manipulability_generic']:.4e}"
          f" / {metrics['manipulability_zero']:.2e}")
    print(f"condition number (generic/zero): {metrics['condition_generic']:.1f}"
          f" / {metrics['condition_zero']:.2e}")
    print("\nKinematics demo " + ("PASS" if metrics["success"] else "FAIL"))
    sys.exit(0 if metrics["success"] else 1)


if __name__ == "__main__":
    main()
