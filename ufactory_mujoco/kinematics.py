"""Forward kinematics, geometric Jacobian, velocity IK, and damped least-squares (DLS) IK.

Jacobian convention follows MuJoCo's ``mj_jacSite`` (world frame):
    site point velocity   = jacp @ qd
    site angular velocity = jacr @ qd
"""

from __future__ import annotations

import mujoco
import numpy as np

_POS_TOL = 1e-4      # m
_ROT_TOL = 1e-3      # rad
_MAX_ITER = 100
_DAMPING = 1e-3


def _set_qpos(model: mujoco.MjModel, data: mujoco.MjData,
              q: np.ndarray | None) -> None:
    """Optionally set qpos (validated against nq, handles quaternion joints) and forward."""
    if q is not None:
        q = np.asarray(q, dtype=float)
        if q.shape != (model.nq,):
            raise ValueError(f"q must have shape ({model.nq},), got {q.shape}")
        if not np.all(np.isfinite(q)):
            raise ValueError("q must contain only finite values")
        data.qpos[:] = q
    mujoco.mj_forward(model, data)


def fk_pose(model: mujoco.MjModel, data: mujoco.MjData, site_name: str,
            q: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Forward kinematics of any site -> (pos (3,), quat wxyz (4,)).

    q=None uses the current data.qpos.
    """
    site_id = model.site(site_name).id
    _set_qpos(model, data, q)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
    return data.site_xpos[site_id].copy(), quat


def quat_error(quat_cur: np.ndarray, quat_target: np.ndarray) -> np.ndarray:
    """Orientation error as world-frame rotation vector: e = target * conj(cur)."""
    q_conj = np.array([quat_cur[0], -quat_cur[1], -quat_cur[2], -quat_cur[3]])
    q_err = np.empty(4)
    mujoco.mju_mulQuat(q_err, quat_target, q_conj)
    rot = np.empty(3)
    mujoco.mju_quat2Vel(rot, q_err, 1.0)
    return rot


def site_jacobian(model: mujoco.MjModel, data: mujoco.MjData, site_name: str,
                  q: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """World-frame Jacobians (jacp (3, nv), jacr (3, nv)) of the named site."""
    site_id = model.site(site_name).id
    _set_qpos(model, data, q)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp, jacr


def pose_jacobian(model: mujoco.MjModel, data: mujoco.MjData, site_name: str,
                  q: np.ndarray | None = None) -> np.ndarray:
    """Stacked (6, nv) Jacobian [[jacp]; [jacr]]; twist = J @ qd."""
    jacp, jacr = site_jacobian(model, data, site_name, q)
    return np.vstack([jacp, jacr])


def solve_joint_velocity(model: mujoco.MjModel, data: mujoco.MjData, site_name: str,
                         twist: np.ndarray, q: np.ndarray | None = None, *,
                         damping: float = _DAMPING) -> np.ndarray:
    """Velocity-level IK: damped least-squares joint velocity for a site twist.

    twist is (6,): [linear (m/s); angular (rad/s)] in the world frame.
    Returns qd (nv,), the minimum-norm solution of J @ qd = twist
    (qd = J^T (J J^T + damping^2 I)^-1 twist).
    """
    twist = np.asarray(twist, dtype=float)
    if twist.shape != (6,):
        raise ValueError(f"twist must have shape (6,), got {twist.shape}")
    if not np.isfinite(damping) or damping < 0:
        raise ValueError("damping must be finite and non-negative")
    J = pose_jacobian(model, data, site_name, q)
    JJt = J @ J.T + (damping ** 2) * np.eye(6)
    return J.T @ np.linalg.solve(JJt, twist)


def _site_chain_dof_indices(model: mujoco.MjModel, site_id: int) -> np.ndarray:
    """DOF column indices belonging to joints on the ancestor kinematic chain of ``site_id``."""
    chain_jnts: list[int] = []
    body_id = int(model.site_bodyid[site_id])
    while body_id > 0:
        jnt_adr = int(model.body_jntadr[body_id])
        jnt_num = int(model.body_jntnum[body_id])
        for j in range(jnt_adr, jnt_adr + jnt_num):
            chain_jnts.append(j)
        body_id = int(model.body_parentid[body_id])
    if not chain_jnts:
        return np.empty(0, dtype=int)
    return np.flatnonzero(np.isin(model.dof_jntid, chain_jnts))


def manipulability(model: mujoco.MjModel, data: mujoco.MjData, site_name: str,
                   q: np.ndarray | None = None, *,
                   include_rotation: bool = True) -> float:
    """Yoshikawa manipulability across the kinematic chain driving ``site_name``.

    Equals ``sqrt(det(J J^T))`` for >=6-DOF arms (``sqrt(det(J^T J))`` for
    underactuated 5-DOF arms), excluding downstream gripper finger DOFs and
    unrelated free-joint scene props. Near zero at (or close to) a singularity.
    ``include_rotation=False`` uses the 3-row positional Jacobian only.
    """
    site_id = model.site(site_name).id
    J = pose_jacobian(model, data, site_name, q) if include_rotation \
        else site_jacobian(model, data, site_name, q)[0]
    dof_idx = _site_chain_dof_indices(model, site_id)
    if dof_idx.size == 0:
        return 0.0
    J_chain = J[:, dof_idx]
    return float(np.prod(np.linalg.svd(J_chain, compute_uv=False)))


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

    jnt_types = model.jnt_type
    clamp_mask = (jnt_types == int(mujoco.mjtJoint.mjJNT_HINGE)) | (
        jnt_types == int(mujoco.mjtJoint.mjJNT_SLIDE)
    )
    clamp_adrs = model.jnt_qposadr[clamp_mask]
    clamp_lo = model.jnt_range[clamp_mask, 0]
    clamp_hi = model.jnt_range[clamp_mask, 1]
    has_clamp = clamp_adrs.size > 0
    quat = np.empty(4)
    quat_trial = np.empty(4)
    eye_m = (damping ** 2) * np.eye(6 if target_quat is not None else 3)

    def _clamp_hinge_slide(qvec: np.ndarray) -> None:
        if has_clamp:
            qvec[clamp_adrs] = np.clip(qvec[clamp_adrs], clamp_lo, clamp_hi)

    step = 0.5
    for _ in range(max_iter):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)

        pos_err = target_pos - data.site_xpos[site_id]
        err = pos_err
        if target_quat is not None:
            mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
            rot_err = quat_error(quat, target_quat)
            err = np.concatenate([pos_err, rot_err])
            if np.all(np.abs(err) < tol_vec):
                break
        elif np.linalg.norm(pos_err) < pos_tol:
            break
        err_scaled = err / tol_vec

        mujoco.mj_comPos(model, data)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = jacp if target_quat is None else np.vstack([jacp, jacr])
        JJt = J @ J.T + eye_m
        dq = J.T @ np.linalg.solve(JJt, err)

        # nullspace bias towards a reference posture
        if nullspace_ref is not None:
            dq += (np.eye(model.nv) - J.T @ np.linalg.solve(JJt, J)) @ (nullspace_ref - q) * 0.05

        def _violation(scaled_err: np.ndarray) -> float:
            return float(np.sum(np.maximum(np.abs(scaled_err) - 1.0, 0.0)))

        v_cur = _violation(err_scaled)
        accepted = False
        for _ in range(8):
            q_trial = q.copy()
            mujoco.mj_integratePos(model, q_trial, dq, step)
            _clamp_hinge_slide(q_trial)
            data.qpos[:] = q_trial
            mujoco.mj_kinematics(model, data)
            pos_trial = target_pos - data.site_xpos[site_id]
            if target_quat is not None:
                mujoco.mju_mat2Quat(quat_trial, data.site_xmat[site_id])
                err_trial = np.concatenate([pos_trial, quat_error(quat_trial, target_quat)])
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
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    pos_err_vec = target_pos - data.site_xpos[site_id]
    converged = bool(np.all(np.abs(pos_err_vec) < pos_tol))  # per-component, like the loop
    if target_quat is not None:
        mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
        rot_err_vec = quat_error(quat, target_quat)
        converged = converged and bool(np.all(np.abs(rot_err_vec) < rot_tol))
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
