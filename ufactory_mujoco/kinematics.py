"""Forward kinematics, geometric Jacobian, and velocity-level inverse kinematics.

Jacobian convention follows MuJoCo's ``mj_jacSite`` (world frame):
    site point velocity   = jacp @ qd
    site angular velocity = jacr @ qd
"""

import mujoco
import numpy as np

_DEFAULT_DAMPING = 1e-3


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
                         damping: float = _DEFAULT_DAMPING) -> np.ndarray:
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
