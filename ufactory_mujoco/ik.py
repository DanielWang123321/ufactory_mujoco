"""Damped least-squares (DLS) inverse kinematics on a MuJoCo model.

Position + orientation error, joint-limit clamping, joint velocity damping.
"""

import mujoco
import numpy as np

from .kinematics import quat_error as _quat_error

_POS_TOL = 1e-4      # m
_ROT_TOL = 1e-3      # rad
_MAX_ITER = 100
_DAMPING = 1e-3


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
    tol_vec = np.array([pos_tol] * 3 + ([rot_tol] * 3 if target_quat is not None else []))

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    def _clamp_hinge_slide(qvec: np.ndarray) -> None:
        # jnt_type is an int array; compare as ints (enum tuple `in` fails on numpy scalars)
        for j in range(model.njnt):
            if int(model.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                adr = model.jnt_qposadr[j]
                qvec[adr] = np.clip(qvec[adr], model.jnt_range[j, 0], model.jnt_range[j, 1])

    step = 0.5
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
            if np.all(np.abs(err) < tol_vec):
                break
        elif np.linalg.norm(pos_err) < pos_tol:
            break
        err_scaled = err / tol_vec

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = jacp if target_quat is None else np.vstack([jacp, jacr])
        JJt = J @ J.T + (damping ** 2) * np.eye(J.shape[0])
        dq = J.T @ np.linalg.solve(JJt, err)

        # nullspace bias towards a reference posture
        if nullspace_ref is not None:
            dq += (np.eye(model.nv) - J.T @ np.linalg.solve(JJt, J)) @ (nullspace_ref - q) * 0.05

        # Backtracking line search: a fixed large step oscillates around the
        # tolerance boundary without converging, so accept a trial only when
        # the total tolerance violation strictly decreases. Within-tolerance
        # fluctuations (e.g. rotation while polishing position) must not
        # reject an otherwise improving step.
        def _violation(scaled_err: np.ndarray) -> float:
            return float(np.sum(np.maximum(np.abs(scaled_err) - 1.0, 0.0)))

        v_cur = _violation(err_scaled)
        accepted = False
        for _ in range(8):
            q_trial = q.copy()
            mujoco.mj_integratePos(model, q_trial, dq, step)
            _clamp_hinge_slide(q_trial)
            data.qpos[:] = q_trial
            mujoco.mj_forward(model, data)
            pos_trial = target_pos - data.site_xpos[site_id]
            if target_quat is not None:
                quat_trial = np.empty(4)
                mujoco.mju_mat2Quat(quat_trial, data.site_xmat[site_id])
                err_trial = np.concatenate([pos_trial, _quat_error(quat_trial, target_quat)])
            else:
                err_trial = pos_trial
            if _violation(err_trial / tol_vec) < v_cur:
                q = q_trial
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break  # cannot improve: singular or local minimum
        step = min(step * 1.5, 0.5)

    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    pos_err_vec = target_pos - data.site_xpos[site_id]
    converged = bool(np.all(np.abs(pos_err_vec) < pos_tol))  # per-component, like the loop
    pos_err = float(np.linalg.norm(pos_err_vec))
    if target_quat is not None:
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
        rot_err_vec = _quat_error(quat, target_quat)
        converged = converged and bool(np.all(np.abs(rot_err_vec) < rot_tol))
        rot_err = float(np.linalg.norm(rot_err_vec))
    else:
        rot_err = 0.0
    return q.copy(), converged


def solve_ik_trajectory(model: mujoco.MjModel, data: mujoco.MjData, site_name: str,
                        waypoints_pos: np.ndarray, quat: np.ndarray | None,
                        q_start: np.ndarray, *,
                        pos_tol: float = 2e-4, **kwargs) -> tuple[np.ndarray, bool]:
    """Sequential IK along Cartesian waypoints. Returns (q_traj (N,nq), all_converged)."""
    qs = np.empty((len(waypoints_pos), model.nq))
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
