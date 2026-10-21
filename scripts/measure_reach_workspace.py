#!/usr/bin/env python3
"""Calibrate per-robot reach-workspace profiles for the RL reach task (0.0.7 M0).

For every supported arm this derives the values frozen in
ufactory_mujoco/robots.py::ROBOT_PROFILES:

1. shoulder position  - world anchor of joint2 at the zero configuration
2. FK point cloud     - 20k joint-space samples (seed 0) pushed through FK
3. target box         - front-sector quantile box, shrunk until the
                        environment's reachability IK accepts >= 90% of
                        uniform samples (same solve_ik settings as the env)
4. radial band        - [min, max] distance shoulder<->box, +-10% margin

Run from the repository root; profiles are then pasted into ROBOT_PROFILES and
guarded by tests/test_reach_env.py::RobotProfileCalibrationTests.

Usage:
    python scripts/measure_reach_workspace.py             # all robots
    python scripts/measure_reach_workspace.py lite6 uf850 # subset
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from ufactory_mujoco.ik import solve_ik  # noqa: E402

ROBOTS = ("lite6", "xarm5", "xarm6", "xarm7", "uf850")
N_CLOUD = 20_000
N_IK_SAMPLES = 200
IK_RATE_MIN = 0.90
SEED = 0
MARGIN = 0.02          # keep samples 2% inside joint limits
TCP_SITE = "attachment_site"


def build(robot: str) -> tuple[mujoco.MjModel, list[int], np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / robot / "scene.xml"))
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i + 1}")
              for i in range(_n_joints(robot))]
    if any(j < 0 for j in joints):
        raise RuntimeError(f"{robot}: joint naming mismatch")
    lo = model.jnt_range[joints, 0].copy()
    hi = model.jnt_range[joints, 1].copy()
    return model, joints, lo, hi


def _n_joints(robot: str) -> int:
    return {"xarm5": 5, "xarm7": 7}.get(robot, 6)


def fk_cloud(model: mujoco.MjModel, joints: list[int], lo: np.ndarray,
             hi: np.ndarray, n: int = N_CLOUD) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    margin = MARGIN * (hi - lo)
    q = rng.uniform(lo + margin, hi - margin, size=(n, len(joints)))
    data = mujoco.MjData(model)
    site_id = model.site(TCP_SITE).id
    adr = model.jnt_qposadr[joints]
    pts = np.empty((n, 3))
    for k in range(n):
        data.qpos[adr] = q[k]
        mujoco.mj_forward(model, data)
        pts[k] = data.site_xpos[site_id]
    return pts


def shoulder_of(model: mujoco.MjModel) -> np.ndarray:
    j2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint2")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return data.xanchor[j2].copy()


def candidate_box(pts: np.ndarray) -> np.ndarray:
    """(2,3) box from front-sector quantiles of the FK cloud."""
    front = pts[pts[:, 0] > 0.0]
    x_lo, x_hi = np.quantile(front[:, 0], [0.45, 0.80])
    y_abs = np.quantile(np.abs(front[:, 1]), 0.75)
    z_lo, z_hi = np.quantile(front[:, 2], [0.25, 0.60])
    return np.array([[x_lo, -y_abs, z_lo], [x_hi, y_abs, z_hi]])


def ik_rate(model: mujoco.MjModel, box: np.ndarray, n: int = N_IK_SAMPLES) -> float:
    rng = np.random.default_rng(SEED + 1)
    data = mujoco.MjData(model)
    site_id = model.site(TCP_SITE).id
    targets = rng.uniform(box[0], box[1], size=(n, 3))
    ok = 0
    for t in targets:
        data.qpos[:] = 0.0
        _, converged = solve_ik(model, data, TCP_SITE, t, None,
                                q_init=np.zeros(model.nq), pos_tol=2e-3, max_iter=80)
        ok += int(bool(converged))
    return ok / n


def radial_band(box: np.ndarray, shoulder: np.ndarray) -> tuple[float, float]:
    closest = np.clip(shoulder, box[0], box[1])
    r_min = float(np.linalg.norm(shoulder - closest))
    corners = np.array([[x, y, z] for x in (box[0][0], box[1][0])
                        for y in (box[0][1], box[1][1])
                        for z in (box[0][2], box[1][2])])
    r_max = float(np.max(np.linalg.norm(corners - shoulder, axis=1)))
    return r_min * 0.9, r_max * 1.1


def calibrate(robot: str) -> dict:
    model, joints, lo, hi = build(robot)
    pts = fk_cloud(model, joints, lo, hi)
    shoulder = shoulder_of(model)
    box = candidate_box(pts)
    rate = ik_rate(model, box)
    shrink = 0.85
    while rate < IK_RATE_MIN:
        center = box.mean(axis=0)
        box = center + shrink * (box - center)
        rate = ik_rate(model, box)
    reach_min, reach_max = radial_band(box, shoulder)
    return {
        "robot": robot,
        "n_joints": len(joints),
        "target_low": np.round(box[0], 4),
        "target_high": np.round(box[1], 4),
        "shoulder": np.round(shoulder, 4),
        "reach_min": round(reach_min, 4),
        "reach_max": round(reach_max, 4),
        "ik_rate": round(rate, 3),
        "cloud_z": (round(float(pts[:, 2].min()), 3), round(float(pts[:, 2].max()), 3)),
    }


def main() -> None:
    robots = tuple(sys.argv[1:]) or ROBOTS
    for robot in robots:
        r = calibrate(robot)
        print(f"{r['robot']}:")
        print(f"  n_joints    : {r['n_joints']}")
        print(f"  target_low  : {r['target_low'].tolist()}")
        print(f"  target_high : {r['target_high'].tolist()}")
        print(f"  shoulder    : {r['shoulder'].tolist()}")
        print(f"  reach band  : [{r['reach_min']}, {r['reach_max']}]")
        print(f"  ik rate     : {r['ik_rate']}   cloud z: {r['cloud_z']}")


if __name__ == "__main__":
    main()
