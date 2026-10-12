"""Rigid-body dynamics for UFACTORY arms (MuJoCo backend).

Pure-computation module, independent of the viewer and the real robot:

    dyn = ArmDynamics("models/xarm6/xarm6.xml", payload=(3.0, [0, 0, 0.021]))
    M = dyn.mass_matrix(q)                # dense mass matrix M(q)
    g = dyn.gravity(q)                    # gravity term g(q)
    c = dyn.coriolis(q, qd)               # velocity-product term C(q,qd)qd
    tau = dyn.inverse_dynamics(q, qd, qdd)  # RNE inverse dynamics
    qdd = dyn.forward_dynamics(q, qd, tau)  # M^-1 (tau - bias)
    tau_ff = dyn.feedforward_traj(q_traj, dt)  # torques along a trajectory

Conventions match the standard manipulator equation
    tau = M(q) qdd + C(q,qd) qd + g(q) - qfrc_passive(q,qd)
where inverse_dynamics / forward_dynamics return the *actuator* torque
(viscous damping is subtracted because it acts on the joints without motor
effort). Dry friction (frictionloss) is excluded from both — xArms
compensate friction in firmware — which keeps forward_dynamics the exact
inverse of inverse_dynamics. The individual terms gravity()/coriolis() are
pure RNE terms. Use a pure arm model (e.g. xarm6.xml, or a payload-augmented
one via `attach_payload`); combined arm+gripper models work for the arm
slice but inverse dynamics ignores equality constraints.

Not thread-safe: each ArmDynamics owns one MjData and mutates it on every
call (same eval_data isolation pattern as SimRobot — keep it off the
render thread).
"""

import os
import re
from pathlib import Path

import mujoco
import numpy as np

_PAYLOAD_XML = ('<body name="tcp_payload" pos="{x} {y} {z}">\n'
                '<inertial pos="0 0 0" mass="{m}" diaginertia="0.001 0.001 0.001"/>\n'
                '</body>')

_GRIPPER_ACTUATOR_NAMES = ("gripper", "gripper_motor", "finger_actuator")


def attach_payload(model_path: str | Path, mass: float, com,
                   site: str = "attachment_site") -> mujoco.MjModel:
    """Load an MJCF arm model with a point-mass payload at the named site.

    mass in kg, com in meters expressed in the site (flange) frame — the same
    quantities reported by xarm tcp_load ([kg, [x, y, z] in mm]).
    """
    model_path = Path(model_path)
    xml = model_path.read_text()
    site_el = re.search(r'<site name="%s"[^>]*/>' % re.escape(site), xml)
    if site_el is None:
        raise ValueError(f'site "{site}" not found in {model_path}')
    payload = _PAYLOAD_XML.format(x=com[0], y=com[1], z=com[2], m=mass)
    xml = xml[:site_el.end()] + "\n" + payload + xml[site_el.end():]
    # from_xml_string resolves relative mesh paths against the CWD; save/restore
    cwd = os.getcwd()
    os.chdir(model_path.parent)
    try:
        return mujoco.MjModel.from_xml_string(xml)
    finally:
        os.chdir(cwd)


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
    """Standard rigid-body dynamics on a pure MuJoCo arm model."""

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
        if v.shape[0] == n:
            return v
        if v.shape[0] == self.n_arm:
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
        """qfrc_passive at (q, qd) (damping + frictionloss; state-dependent)."""
        self._set(q, qd)
        mujoco.mj_forward(self.model, self.data)
        return self.data.qfrc_passive.copy()

    def inverse_dynamics(self, q, qd, qdd, full: bool = False) -> np.ndarray:
        """Actuator torque for a desired motion:
        tau = M(q) qdd + C(q,qd) qd + g(q) - qfrc_passive(q,qd)
        i.e. the torque the motors must generate; viscous damping assists
        (passive is subtracted). Dry friction (frictionloss) is excluded —
        xArms compensate friction in firmware, and excluding it keeps
        forward_dynamics the exact inverse."""
        M = self.mass_matrix(q, full=True)
        bias = self._bias(q, self._full(qd, self.nv))
        passive = self._passive(q, self._full(qd, self.nv))
        tau = M @ self._full(qdd, self.nv) + bias - passive
        return self._slice(tau, self.n_arm, full)

    def forward_dynamics(self, q, qd, tau, full: bool = False) -> np.ndarray:
        """qdd = M(q)^-1 (tau + qfrc_passive - C(q,qd)qd - g(q)); exact
        inverse of inverse_dynamics."""
        M = self.mass_matrix(q, full=True)
        bias = self._bias(q, self._full(qd, self.nv))
        passive = self._passive(q, self._full(qd, self.nv))
        qdd = np.linalg.solve(M, self._full(tau, self.nv) - bias + passive)
        return self._slice(qdd, self.n_arm, full)

    def feedforward_traj(self, q_traj, dt: float, full: bool = False) -> np.ndarray:
        """Inverse-dynamics torques along a joint trajectory.

        q_traj: (N, n_arm) arm joint angles; velocities/accelerations are
        obtained by second-order central differences of q_traj.
        Returns (N, n_arm) torque rows matching the trajectory length.
        """
        q = np.asarray(q_traj, dtype=float)
        if q.ndim != 2 or q.shape[1] not in (self.n_arm, self.model.nq):
            raise ValueError(f"q_traj must be (N, {self.n_arm}) or (N, {self.model.nq})")
        qd = np.gradient(q, dt, axis=0, edge_order=2)
        qdd = np.gradient(qd, dt, axis=0, edge_order=2)
        tau = np.empty((len(q), self.nv if full else self.n_arm))
        for i in range(len(q)):
            tau[i] = self.inverse_dynamics(q[i], qd[i], qdd[i], full=full)
        return tau
