"""Lightweight numerical trajectory planning and time parameterization.

Provides:
- JointTrajectory: Container for time-parameterized joint trajectories (t, q, qd, qdd).
- plan_ptp_quintic: Smooth point-to-point quintic polynomial planning with analytical limits.
- plan_ptp_lspb: Trapezoidal velocity profile (LSPB) with parabolic blends.
- plan_joint_waypoints: Smooth multi-waypoint via-point planning (C2 clamped cubic spline).
- plan_cartesian_line: Cartesian straight-line planning with SLERP/rotation integration and IK.

Zero heavy dependencies (uses NumPy, SciPy, and MuJoCo).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline

from .ik import solve_ik, solve_ik_trajectory

_EPS = 1e-12

# Standard defaults for UFACTORY arms (rad/s, rad/s^2, m/s, m/s^2)
DEFAULT_MAX_JOINT_VEL = 1.57     # ~90 deg/s
DEFAULT_MAX_JOINT_ACC = 4.0      # ~230 deg/s^2
DEFAULT_MAX_LIN_VEL = 0.20       # 200 mm/s
DEFAULT_MAX_LIN_ACC = 1.0        # 1000 mm/s^2
DEFAULT_MAX_ROT_VEL = 1.57       # ~90 deg/s


@dataclass
class JointTrajectory:
    """Time-parameterized joint trajectory with analytical kinematics.

    times: (N,) timestamps starting at 0.0, in seconds.
    q:     (N, n_joints) joint positions, in radians.
    qd:    (N, n_joints) joint velocities, in rad/s.
    qdd:   (N, n_joints) joint accelerations, in rad/s^2.
    """

    times: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=np.float64)
        self.q = np.asarray(self.q, dtype=np.float64)
        self.qd = np.asarray(self.qd, dtype=np.float64)
        self.qdd = np.asarray(self.qdd, dtype=np.float64)

        if not (self.times.ndim == 1 and self.q.ndim == 2):
            raise ValueError(f"Shape mismatch: times {self.times.shape}, q {self.q.shape}")
        n = len(self.times)
        if self.q.shape[0] != n or self.qd.shape[0] != n or self.qdd.shape[0] != n:
            raise ValueError(
                f"Length mismatch: times({n}), q({self.q.shape[0]}), "
                f"qd({self.qd.shape[0]}), qdd({self.qdd.shape[0]})"
            )

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        return float(self.times[-1] - self.times[0]) if len(self.times) > 0 else 0.0

    @property
    def dt(self) -> float:
        """Nominal timestep in seconds."""
        return float(self.times[1] - self.times[0]) if len(self.times) > 1 else 0.0

    @property
    def n_steps(self) -> int:
        return len(self.times)

    @property
    def n_joints(self) -> int:
        return self.q.shape[1]

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample (q, qd, qdd) at continuous time t."""
        t_clamped = float(np.clip(t, self.times[0], self.times[-1]))
        q_val = np.empty(self.n_joints)
        qd_val = np.empty(self.n_joints)
        qdd_val = np.empty(self.n_joints)
        for j in range(self.n_joints):
            q_val[j] = np.interp(t_clamped, self.times, self.q[:, j])
            qd_val[j] = np.interp(t_clamped, self.times, self.qd[:, j])
            qdd_val[j] = np.interp(t_clamped, self.times, self.qdd[:, j])
        return q_val, qd_val, qdd_val

    def concat(self, other: JointTrajectory) -> JointTrajectory:
        """Concatenate another trajectory sequentially."""
        if self.n_joints != other.n_joints:
            raise ValueError(f"Joint count mismatch: {self.n_joints} vs {other.n_joints}")
        t_offset = self.times[-1] + self.dt
        other_times = other.times - other.times[0] + t_offset
        return JointTrajectory(
            times=np.concatenate([self.times, other_times]),
            q=np.vstack([self.q, other.q]),
            qd=np.vstack([self.qd, other.qd]),
            qdd=np.vstack([self.qdd, other.qdd]),
        )


def _compute_quintic_duration(
    delta_q: np.ndarray,
    max_vel: np.ndarray,
    max_acc: np.ndarray,
) -> float:
    """Compute minimal time T for unit quintic s(t) = 10u^3 - 15u^4 + 6u^5.

    Peak velocity:     s'(u)_max = 1.875 / T
    Peak acceleration: s''(u)_max = 10 / (sqrt(3) * T^2) ~= 5.7735 / T^2
    """
    abs_d = np.abs(delta_q)
    if np.max(abs_d) < _EPS:
        return 0.0

    t_v = np.max(1.875 * abs_d / np.maximum(max_vel, _EPS))
    t_a = np.max(np.sqrt(5.7735 * abs_d / np.maximum(max_acc, _EPS)))
    return float(max(t_v, t_a))


def plan_ptp_quintic(
    q0: np.ndarray,
    q1: np.ndarray,
    max_vel: float | np.ndarray = DEFAULT_MAX_JOINT_VEL,
    max_acc: float | np.ndarray = DEFAULT_MAX_JOINT_ACC,
    duration: float | None = None,
    dt: float = 0.002,
) -> JointTrajectory:
    """Plan a point-to-point quintic polynomial joint trajectory.

    Continuous position, velocity, and acceleration; zero vel and acc at endpoints.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    if q0.shape != q1.shape:
        raise ValueError(f"q0 shape {q0.shape} != q1 shape {q1.shape}")

    n_joints = q0.size
    v_lim = np.broadcast_to(np.asarray(max_vel, dtype=np.float64), (n_joints,))
    a_lim = np.broadcast_to(np.asarray(max_acc, dtype=np.float64), (n_joints,))

    dq = q1 - q0
    if duration is None:
        t_min = _compute_quintic_duration(dq, v_lim, a_lim)
        dur = max(t_min, dt)
    else:
        dur = max(float(duration), dt)

    n_steps = max(1, int(math.ceil(dur / dt)))
    actual_duration = n_steps * dt
    t = np.linspace(0.0, actual_duration, n_steps + 1)
    u = t / actual_duration

    # Analytical quintic polynomial and derivatives
    s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    sd = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / actual_duration
    sdd = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / (actual_duration**2)

    q = q0 + s[:, None] * dq
    qd = sd[:, None] * dq
    qdd = sdd[:, None] * dq

    return JointTrajectory(times=t, q=q, qd=qd, qdd=qdd)


def _lspb_duration_scalar(dist: float, v_max: float, a_max: float) -> float:
    """Time-optimal LSPB duration for scalar displacement."""
    d = abs(float(dist))
    v = max(float(v_max), _EPS)
    a = max(float(a_max), _EPS)
    if d < _EPS:
        return 0.0
    d_a = v * v / (2.0 * a)
    if d >= 2.0 * d_a:
        t_a = v / a
        t_c = (d - 2.0 * d_a) / v
        return 2.0 * t_a + t_c
    return 2.0 * math.sqrt(d / a)


def plan_ptp_lspb(
    q0: np.ndarray,
    q1: np.ndarray,
    max_vel: float | np.ndarray = DEFAULT_MAX_JOINT_VEL,
    max_acc: float | np.ndarray = DEFAULT_MAX_JOINT_ACC,
    duration: float | None = None,
    dt: float = 0.002,
) -> JointTrajectory:
    """Plan a trapezoidal velocity (LSPB) point-to-point joint trajectory.

    All joints are synchronized to reach the destination simultaneously.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    if q0.shape != q1.shape:
        raise ValueError(f"q0 shape {q0.shape} != q1 shape {q1.shape}")

    n_joints = q0.size
    v_lim = np.broadcast_to(np.asarray(max_vel, dtype=np.float64), (n_joints,))
    a_lim = np.broadcast_to(np.asarray(max_acc, dtype=np.float64), (n_joints,))
    dq = q1 - q0

    durations = [_lspb_duration_scalar(dq[i], v_lim[i], a_lim[i]) for i in range(n_joints)]
    t_opt = max(durations) if durations else 0.0
    dur = max(t_opt if duration is None else float(duration), dt)

    n_steps = max(1, int(math.ceil(dur / dt)))
    actual_duration = n_steps * dt
    t = np.linspace(0.0, actual_duration, n_steps + 1)

    q = np.empty((n_steps + 1, n_joints), dtype=np.float64)
    qd = np.empty((n_steps + 1, n_joints), dtype=np.float64)
    qdd = np.empty((n_steps + 1, n_joints), dtype=np.float64)

    for i in range(n_joints):
        d_i = abs(dq[i])
        if d_i < _EPS:
            q[:, i] = q0[i]
            qd[:, i] = 0.0
            qdd[:, i] = 0.0
            continue

        sgn = 1.0 if dq[i] >= 0.0 else -1.0
        a = a_lim[i]
        # Solve for blend time t_a fitting displacement d_i into actual_duration:
        # a * t_a * (T - t_a) >= d_i => a * t_a^2 - a * T * t_a + d_i = 0
        disc = actual_duration**2 - 4.0 * d_i / a
        if disc >= 0.0:
            t_a = 0.5 * (actual_duration - math.sqrt(disc))
            t_a = max(_EPS, min(0.5 * actual_duration, t_a))
        else:
            t_a = 0.5 * actual_duration

        # Cruise velocity
        v_cruise = d_i / (actual_duration - t_a)
        t_dec = actual_duration - t_a

        for k, tk in enumerate(t):
            if tk <= t_a:
                pos = 0.5 * a * (tk**2)
                vel = a * tk
                acc = a
            elif tk <= t_dec:
                pos = 0.5 * a * (t_a**2) + v_cruise * (tk - t_a)
                vel = v_cruise
                acc = 0.0
            else:
                tau = actual_duration - tk
                pos = d_i - 0.5 * a * (tau**2)
                vel = a * tau
                acc = -a

            q[k, i] = q0[i] + sgn * pos
            qd[k, i] = sgn * vel
            qdd[k, i] = sgn * acc

    return JointTrajectory(times=t, q=q, qd=qd, qdd=qdd)


def plan_joint_waypoints(
    waypoints: np.ndarray,
    max_vel: float | np.ndarray = DEFAULT_MAX_JOINT_VEL,
    max_acc: float | np.ndarray = DEFAULT_MAX_JOINT_ACC,
    dt: float = 0.002,
    method: str = "spline",
) -> JointTrajectory:
    """Plan a smooth joint trajectory through a sequence of waypoints.

    waypoints: (M, n_joints) waypoints with M >= 2.
    method:
      - 'spline': C2 continuous clamped cubic spline (zero endpoint velocities).
                  Times are automatically scaled to strictly respect velocity and acceleration limits.
      - 'stop': Stop-and-go sequence connecting waypoints via quintic PTP segments.
    """
    w = np.asarray(waypoints, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] < 2:
        raise ValueError(f"Waypoints must have shape (M, n_joints) with M >= 2, got {w.shape}")

    m_points, n_joints = w.shape
    v_lim = np.broadcast_to(np.asarray(max_vel, dtype=np.float64), (n_joints,))
    a_lim = np.broadcast_to(np.asarray(max_acc, dtype=np.float64), (n_joints,))

    if method == "stop":
        segments = []
        for i in range(m_points - 1):
            seg = plan_ptp_quintic(w[i], w[i + 1], max_vel=v_lim, max_acc=a_lim, dt=dt)
            segments.append(seg)
        result = segments[0]
        for seg in segments[1:]:
            result = result.concat(seg)
        return result

    if method != "spline":
        raise ValueError(f"Unknown planning method: {method}")

    # Estimate chord durations between adjacent waypoints
    delta_w = np.abs(np.diff(w, axis=0))
    seg_durations = np.empty(m_points - 1, dtype=np.float64)
    for k in range(m_points - 1):
        t_v = np.max(delta_w[k] / np.maximum(v_lim, _EPS))
        t_a = np.max(np.sqrt(delta_w[k] / np.maximum(a_lim, _EPS)))
        seg_durations[k] = max(t_v, t_a, 0.1)

    knot_times = np.concatenate([[0.0], np.cumsum(seg_durations)])

    # Clamped cubic spline with zero velocity at start and end
    cs = CubicSpline(knot_times, w, axis=0, bc_type=((1, np.zeros(n_joints)), (1, np.zeros(n_joints))))

    # Sample densely to check velocity and acceleration limits
    test_t = np.linspace(0.0, knot_times[-1], max(100, int(knot_times[-1] / dt)))
    test_qd = cs(test_t, 1)
    test_qdd = cs(test_t, 2)

    v_ratio = np.max(np.abs(test_qd) / v_lim[None, :])
    a_ratio = np.max(np.abs(test_qdd) / a_lim[None, :])
    scale = max(1.0, v_ratio, math.sqrt(a_ratio))

    # Scale knot times to satisfy all limits and align with discrete dt
    knot_times *= scale
    total_dur = float(knot_times[-1])
    n_steps = max(1, int(math.ceil(total_dur / dt)))
    final_time = n_steps * dt
    if total_dur > _EPS:
        knot_times = knot_times * (final_time / total_dur)

    cs_scaled = CubicSpline(
        knot_times, w, axis=0, bc_type=((1, np.zeros(n_joints)), (1, np.zeros(n_joints)))
    )

    t = np.linspace(0.0, final_time, n_steps + 1)
    q = cs_scaled(t, 0)
    qd = cs_scaled(t, 1)
    qdd = cs_scaled(t, 2)

    # Enforce boundary zero velocities
    qd[0] = 0.0
    qd[-1] = 0.0

    return JointTrajectory(times=t, q=q, qd=qd, qdd=qdd)


def plan_cartesian_line(
    p0: np.ndarray,
    quat0: np.ndarray,
    p1: np.ndarray,
    quat1: np.ndarray | None,
    model: mujoco.MjModel,
    site_name: str = "attachment_site",
    max_lin_vel: float = DEFAULT_MAX_LIN_VEL,
    max_lin_acc: float = DEFAULT_MAX_LIN_ACC,
    max_rot_vel: float = DEFAULT_MAX_ROT_VEL,
    q_init: np.ndarray | None = None,
    dt: float = 0.002,
    eval_data: mujoco.MjData | None = None,
) -> tuple[JointTrajectory, bool]:
    """Plan a Cartesian straight-line trajectory with SLERP orientation and IK.

    Returns (JointTrajectory, ik_converged).
    """
    p0 = np.asarray(p0, dtype=np.float64).reshape(3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    quat0 = np.asarray(quat0, dtype=np.float64).reshape(4)
    quat0 /= np.linalg.norm(quat0)

    lin_dist = float(np.linalg.norm(p1 - p0))

    # Rotation distance
    rot_dist = 0.0
    vel_full = np.zeros(3)
    if quat1 is not None:
        quat1 = np.asarray(quat1, dtype=np.float64).reshape(4)
        quat1 /= np.linalg.norm(quat1)
        q0_conj = np.array([quat0[0], -quat0[1], -quat0[2], -quat0[3]])
        q_err = np.empty(4)
        mujoco.mju_mulQuat(q_err, quat1, q0_conj)
        mujoco.mju_quat2Vel(vel_full, q_err, 1.0)
        rot_dist = float(np.linalg.norm(vel_full))

    # Time parameterization for Cartesian motion
    t_lin_v = 1.875 * lin_dist / max(max_lin_vel, _EPS)
    t_lin_a = math.sqrt(5.7735 * lin_dist / max(max_lin_acc, _EPS))
    t_rot = 1.875 * rot_dist / max(max_rot_vel, _EPS)
    dur = max(t_lin_v, t_lin_a, t_rot, dt)

    n_steps = max(1, int(math.ceil(dur / dt)))
    actual_dur = n_steps * dt
    t = np.linspace(0.0, actual_dur, n_steps + 1)
    u = t / actual_dur

    # Quintic scalar progress
    s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5

    waypoints_pos = p0[None, :] + s[:, None] * (p1 - p0)[None, :]
    waypoints_quat = None
    if quat1 is not None:
        waypoints_quat = np.empty((n_steps + 1, 4), dtype=np.float64)
        for i in range(n_steps + 1):
            waypoints_quat[i] = quat0.copy()
            mujoco.mju_quatIntegrate(waypoints_quat[i], vel_full, float(s[i]))

    data = eval_data if eval_data is not None else mujoco.MjData(model)
    q_start = data.qpos.copy() if q_init is None else np.asarray(q_init, dtype=np.float64).copy()

    q_traj, ok = solve_ik_trajectory(
        model, data, site_name, waypoints_pos, waypoints_quat, q_start
    )

    # Differentiate joint path to compute qd and qdd
    qd = np.gradient(q_traj, dt, axis=0)
    qdd = np.gradient(qd, dt, axis=0)

    # Force zero boundary velocities/accelerations
    qd[0] = 0.0
    qd[-1] = 0.0
    qdd[0] = 0.0
    qdd[-1] = 0.0

    traj = JointTrajectory(times=t, q=q_traj, qd=qd, qdd=qdd)
    return traj, ok
