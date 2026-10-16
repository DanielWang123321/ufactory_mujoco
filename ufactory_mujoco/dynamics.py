"""Rigid-body dynamics for UFACTORY arms (MuJoCo backend).

Pure-computation module, independent of the viewer and the real robot:

    dyn = ArmDynamics("models/xarm6/xarm6.xml", payload=(3.0, [0, 0, 0.021]))
    M = dyn.mass_matrix(q)                # dense mass matrix M(q)
    g = dyn.gravity(q)                    # gravity term g(q)
    c = dyn.coriolis(q, qd)               # velocity-product term C(q,qd)qd
    tau = dyn.inverse_dynamics(q, qd, qdd)  # RNE inverse dynamics
    qdd = dyn.forward_dynamics(q, qd, tau)  # fixed gripper for arm-only torque
    tau_ff = dyn.feedforward_traj(q_traj, dt)  # torques along a trajectory

Conventions match the standard manipulator equation
    tau = M(q) qdd + C(q,qd) qd + g(q) - qfrc_passive(q,qd)
where inverse_dynamics / forward_dynamics return the *actuator* torque
(viscous damping is subtracted because it acts on the joints without motor
effort). Dry friction (frictionloss) is excluded from both — xArms
compensate friction in firmware — which keeps forward_dynamics the exact
inverse of inverse_dynamics. The individual terms gravity()/coriolis() are
pure RNE terms. Arm-only velocity and acceleration inputs pad gripper DOFs
with zeros; a full position vector selects the gripper opening. Arm-only
torque inputs solve the arm mass-matrix block with gripper acceleration
fixed at zero. Full torque inputs solve all DOFs. The `full` flag controls
only the output dimensions. These calculations exclude equality-constraint
forces, including gripper closed-loop reactions.

Not thread-safe: each ArmDynamics owns one MjData and mutates it on every
call (same eval_data isolation pattern as SimRobot — keep it off the
render thread).
"""

from pathlib import Path

import mujoco
import numpy as np

_GRIPPER_ACTUATOR_NAMES = ("gripper", "gripper_motor", "finger_actuator")


def attach_payload(model_path: str | Path, mass: float, com,
                   site: str = "attachment_site") -> mujoco.MjModel:
    """Load an MJCF arm model with a point-mass payload at the named site.

    mass is finite and nonnegative (kg); com contains three finite coordinates
    in meters relative to the specified site's origin and axes. Convert xArm
    tcp_load COM from mm before calling. A zero mass returns the base model.
    The payload has zero rotational inertia about its own COM.
    """
    mass = float(mass)
    if not np.isfinite(mass) or mass < 0:
        raise ValueError("payload mass must be finite and nonnegative (kg)")
    com = np.asarray(com, dtype=float)
    if com.shape != (3,) or not np.all(np.isfinite(com)):
        raise ValueError("payload COM must contain three finite coordinates (m)")

    spec = mujoco.MjSpec.from_file(str(Path(model_path).resolve()))
    site_spec = spec.site(site)
    if site_spec is None:
        raise ValueError(f'site "{site}" not found in {model_path}')
    if mass == 0:
        return spec.compile()

    # Compilation resolves frames and alternative orientation attributes, but
    # also edits the spec. Probe a copy, retaining the site's original parent
    # even when the source enables static-body fusion.
    probe = spec.copy()
    probe.compiler.fusestatic = False
    base = probe.compile()
    site_id = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_SITE, site)
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, base.site_quat[site_id])
    position = base.site_pos[site_id] + rotation.reshape(3, 3) @ com
    site_spec.parent.add_body(name="tcp_payload", pos=position, mass=mass,
                              ipos=[0, 0, 0], inertia=[0, 0, 0],
                              explicitinertial=True)
    return spec.compile()


def _detect_arm_nv(model: mujoco.MjModel) -> int:
    """Number of arm DOFs: DOFs before the gripper actuator's joint targets."""
    for i in range(model.nu):
        if model.actuator(i).name in _GRIPPER_ACTUATOR_NAMES:
            dofs = [model.jnt_dofadr[model.actuator_trnid[j, 0]] + 1
                    for j in range(i)
                    if model.actuator_trntype[j] == mujoco.mjtTrn.mjTRN_JOINT]
            return max(dofs) if dofs else model.nv
    return model.nv


class ArmDynamics:
    """Rigid-body dynamics for arms, optionally with a fixed-opening gripper."""

    def __init__(self, model: str | Path | mujoco.MjModel,
                 payload: tuple[float, list] | None = None,
                 site: str = "attachment_site"):
        if isinstance(model, mujoco.MjModel):
            if payload is not None:
                raise ValueError("payload injection requires an MJCF path, not an MjModel")
            self.model = model
        elif payload is not None:
            mass, com = payload
            self.model = attach_payload(model, mass, com, site=site)
        else:
            self.model = mujoco.MjModel.from_xml_path(str(model))
        self.data = mujoco.MjData(self.model)
        self.nv = self.model.nv
        self.n_arm = _detect_arm_nv(self.model)

    # -- internal state helpers ---------------------------------------------
    def _full(self, v: np.ndarray, n: int) -> np.ndarray:
        """Pad an arm-only vector to full length (gripper DOFs zero)."""
        v = np.asarray(v, dtype=float)
        if v.ndim != 1:
            raise ValueError("expected a one-dimensional joint vector")
        if v.size == n:
            return v
        if v.size == self.n_arm:
            out = np.zeros(n)
            out[:self.n_arm] = v
            return out
        raise ValueError(f"expected {n} (full) or {self.n_arm} (arm) elements, got {v.shape[0]}")

    def _set(self, q, qd=None, qdd=None) -> None:
        self.data.qpos[:] = self._full(q, self.model.nq)
        self.data.qvel[:] = self._full(qd, self.nv) if qd is not None else 0.0
        self.data.qacc[:] = self._full(qdd, self.nv) if qdd is not None else 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.ctrl[:] = 0.0

    @staticmethod
    def _slice(out: np.ndarray, n_arm: int, full: bool) -> np.ndarray:
        return out if full else out[:n_arm]

    # -- dynamics terms -------------------------------------------------------
    def mass_matrix(self, q, full: bool = False) -> np.ndarray:
        """Dense symmetric positive-definite mass matrix M(q), shape (nv, nv).

        Leaves data.qM as the unfactored inertia (mj_crb output), so callers
        may mj_factorM + mj_solveM against it directly afterwards.
        """
        self._set(q)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_crb(self.model, self.data)
        M = np.zeros((self.nv, self.nv))
        mujoco.mj_fullM(self.model, self.data, M)
        n = self.nv if full else self.n_arm
        return M[:n, :n]

    def _bias(self, q, qd) -> np.ndarray:
        """qfrc_bias = C(q,qd)qd + g(q) (passive forces excluded)."""
        self._set(q, qd)
        mujoco.mj_forward(self.model, self.data)
        return self.data.qfrc_bias.copy()

    def gravity(self, q, full: bool = False) -> np.ndarray:
        """Gravity term g(q): bias force at zero velocity."""
        return self._slice(self._bias(q, np.zeros(self.nv)), self.n_arm, full)

    def coriolis(self, q, qd, full: bool = False) -> np.ndarray:
        """Velocity-product term C(q,qd)qd = qfrc_bias - g(q)."""
        qd = self._full(qd, self.nv)
        return self._slice(self._bias(q, qd) - self._bias(q, np.zeros(self.nv)),
                           self.n_arm, full)

    # -- inverse / forward dynamics -------------------------------------------
    def _passive(self, q, qd) -> np.ndarray:
        """qfrc_passive at (q, qd), including damping but not frictionloss."""
        self._set(q, qd)
        mujoco.mj_forward(self.model, self.data)
        return self.data.qfrc_passive.copy()

    def inverse_dynamics(self, q, qd, qdd, full: bool = False) -> np.ndarray:
        """Actuator torque for a desired motion:
        tau = M(q) qdd + C(q,qd) qd + g(q) - qfrc_passive(q,qd)
        i.e. the torque the motors must generate; viscous damping assists
        (passive is subtracted). Dry friction (frictionloss) is excluded —
        xArms compensate friction in firmware, and excluding it keeps
        forward_dynamics the exact inverse. Nonzero gripper acceleration
        requires full=True so the necessary gripper torques are returned."""
        qdd = self._full(qdd, self.nv)
        if not full and np.any(qdd[self.n_arm:] != 0):
            raise ValueError("nonzero gripper acceleration requires full=True")
        M = self.mass_matrix(q, full=True)
        bias = self._bias(q, self._full(qd, self.nv))
        passive = self._passive(q, self._full(qd, self.nv))
        tau = M @ qdd + bias - passive
        return self._slice(tau, self.n_arm, full)

    def forward_dynamics(self, q, qd, tau, full: bool = False) -> np.ndarray:
        """Solve M qdd = tau - bias + passive.

        Arm-only tau holds gripper acceleration at zero and solves M_aa.
        Full tau solves the whole rigid-body system without constraint forces.
        full controls the returned dimensions, not which system is solved.
        """
        tau = np.asarray(tau, dtype=float)
        self._full(tau, self.nv)  # validate shape before evaluating the model
        M = self.mass_matrix(q, full=True)
        bias = self._bias(q, self._full(qd, self.nv))
        passive = self._passive(q, self._full(qd, self.nv))
        if tau.size == self.nv:
            qdd = np.linalg.solve(M, tau - bias + passive)
        else:
            qdd = np.zeros(self.nv)
            n = self.n_arm
            qdd[:n] = np.linalg.solve(M[:n, :n], tau - bias[:n] + passive[:n])
        return self._slice(qdd, self.n_arm, full)

    def feedforward_traj(self, q_traj, dt: float, full: bool = False) -> np.ndarray:
        """Inverse-dynamics torques along a joint trajectory.

        q_traj: (N, n_arm) or (N, nq) positions; velocities/accelerations are
        obtained by second-order central differences. Constant columns have
        exactly zero derivatives. Moving grippers with nonzero acceleration
        require full=True. Returns (N, n_arm), or (N, nv) when full=True.
        """
        q = np.asarray(q_traj, dtype=float)
        if q.ndim != 2 or q.shape[1] not in (self.n_arm, self.model.nq):
            raise ValueError(f"q_traj must be (N, {self.n_arm}) or (N, {self.model.nq})")
        qd = np.gradient(q, dt, axis=0, edge_order=2)
        qdd = np.gradient(qd, dt, axis=0, edge_order=2)
        stationary = np.all(q == q[0], axis=0)
        qd[:, stationary] = qdd[:, stationary] = 0.0
        tau = np.empty((len(q), self.nv if full else self.n_arm))
        for i in range(len(q)):
            tau[i] = self.inverse_dynamics(q[i], qd[i], qdd[i], full=full)
        return tau
