"""Damped least-squares (DLS) inverse kinematics on a MuJoCo model.

Position + orientation error, joint-limit clamping, joint velocity damping.
"""

import mujoco
import numpy as np

_POS_TOL = 1e-4      # m
_ROT_TOL = 1e-3      # rad
_MAX_ITER = 100
_DAMPING = 1e-3


def _quat_error(quat_cur: np.ndarray, quat_target: np.ndarray) -> np.ndarray:
    """Orientation error as world-frame rotation vector: e = target * conj(cur)."""
    q_conj = np.array([quat_cur[0], -quat_cur[1], -quat_cur[2], -quat_cur[3]])
    q_err = np.empty(4)
    mujoco.mju_mulQuat(q_err, quat_target, q_conj)
    rot = np.empty(3)
    mujoco.mju_quat2Vel(rot, q_err, 1.0)
    return rot


def solve_ik(model: mujoco.MjModel, data: mujoco.MjData, site_name: str,
             target_pos: np.ndarray, target_quat: np.ndarray | None,
             q_init: np.ndarray | None = None, *,
             pos_tol: float = _POS_TOL, rot_tol: float = _ROT_TOL,
             max_iter: int = _MAX_ITER, damping: float = _DAMPING,
             nullspace_ref: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
    """Solve IK for the named site. Returns (q, converged).

    quat is wxyz. When target_quat is None only position is tracked.
    nullspace_ref: optional secondary-task posture the solution stays close to.
    """
    site_id = model.site(site_name).id
    q = data.qpos.copy() if q_init is None else q_init.astype(float).copy()
    lo, hi = model.jnt_range[:, 0], model.jnt_range[:, 1]

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(max_iter):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)  # FK (kinematics only needed, but forward is fine)

        pos_err = target_pos - data.site_xpos[site_id]
        err = pos_err
        if target_quat is not None:
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
            rot_err = _quat_error(quat, target_quat)
            err = np.concatenate([pos_err, rot_err])
            tol = np.array([pos_tol] * 3 + [rot_tol] * 3)
            if np.all(np.abs(err) < tol):
                break
        elif np.linalg.norm(pos_err) < pos_tol:
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = jacp if target_quat is None else np.vstack([jacp, jacr])
        JJt = J @ J.T + (damping ** 2) * np.eye(J.shape[0])
        dq = J.T @ np.linalg.solve(JJt, err)

        # nullspace bias towards a reference posture
        if nullspace_ref is not None:
            dq += (np.eye(model.nv) - J.T @ np.linalg.solve(JJt, J)) @ (nullspace_ref - q) * 0.05

        step = 0.5
        q += step * dq
        np.clip(q, lo, hi, out=q)

    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    pos_err = np.linalg.norm(target_pos - data.site_xpos[site_id])
    if target_quat is not None:
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
        rot_err = np.linalg.norm(_quat_error(quat, target_quat))
        converged = pos_err < pos_tol and rot_err < rot_tol
    else:
        rot_err = 0.0
        converged = pos_err < pos_tol
    return q.copy(), converged


def solve_ik_trajectory(model: mujoco.MjModel, data: mujoco.MjData, site_name: str,
                        waypoints_pos: np.ndarray, quat: np.ndarray | None,
                        q_start: np.ndarray, *,
                        pos_tol: float = 2e-4, **kwargs) -> tuple[np.ndarray, bool]:
    """Sequential IK along Cartesian waypoints. Returns (q_traj (N,6), all_converged)."""
    qs = np.empty((len(waypoints_pos), model.nv))
    ok = True
    q = q_start.astype(float).copy()
    quat_arr = None if quat is None else np.asarray(quat, dtype=float)
    if quat_arr is not None and quat_arr.ndim == 1:
        quat_arr = np.broadcast_to(quat_arr, (len(waypoints_pos), 4))
    for i, p in enumerate(waypoints_pos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        q, conv = solve_ik(model, data, site_name, p,
                           None if quat_arr is None else quat_arr[i], q_init=q,
                           pos_tol=pos_tol, **kwargs)
        qs[i] = q
        ok &= conv
    return qs, ok
