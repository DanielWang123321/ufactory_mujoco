"""Real xArm hardware session helpers for static torque and sim-collision checks.

Unit policy:
  - joint angles q: rad
  - joint speed: rad/s
Aligns with ``XArmAPI(is_radian=True)`` and ``set_servo_angle(..., is_radian=True)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from .profiles import DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S, validate_dynamics_move_speed

MODE_POSITION = 0
REPORT_STATE_PAUSED = 3
REPORT_STATE_STOPPING = 4

DEFAULT_GRAVITY_DIRECTION = [0.0, 0.0, -1.0]
MOVE_STRATEGY_AXIS_SEQUENTIAL = "axis-sequential"
MOVE_STRATEGY_DIRECT = "direct"
MOVE_STRATEGIES = (MOVE_STRATEGY_AXIS_SEQUENTIAL, MOVE_STRATEGY_DIRECT)

VEL_TOL_RAD_S = 0.01
SETTLE_TIMEOUT_S = 5.0
SETTLE_POLL_S = 0.1
SIMULATION_SWITCH_TIMEOUT_S = 10.0


def _check_config_code(label: str, code: Any) -> None:
    """Raise when a controller configuration command is rejected."""
    if code is not None and int(code) != 0:
        raise RuntimeError(f"{label} failed with code {code}")


def format_arm_status(arm: Any) -> str:
    """Human-readable controller status summary."""
    parts = [
        f"state={getattr(arm, 'state', '?')}",
        f"mode={getattr(arm, 'mode', '?')}",
        f"error_code={getattr(arm, 'error_code', '?')}",
        f"warn_code={getattr(arm, 'warn_code', '?')}",
    ]
    getter = getattr(arm, "get_state", None)
    if callable(getter):
        res = getter()
        if isinstance(res, tuple) and len(res) == 2 and int(res[0]) == 0:
            parts.append(f"get_state()={res[1]}")
    return ", ".join(parts)


def _read_err_warn_and_state(arm: Any) -> tuple[int, int, int]:
    err_code = int(getattr(arm, "error_code", 0) or 0)
    warn_code = int(getattr(arm, "warn_code", 0) or 0)
    ew_getter = getattr(arm, "get_err_warn_code", None)
    if callable(ew_getter):
        res = ew_getter()
        if isinstance(res, tuple) and len(res) == 2 and int(res[0]) == 0:
            err_code, warn_code = int(res[1][0]), int(res[1][1])

    state = int(getattr(arm, "state", 4) or 4)
    st_getter = getattr(arm, "get_state", None)
    if callable(st_getter):
        res = st_getter()
        if isinstance(res, tuple) and len(res) == 2 and int(res[0]) == 0:
            state = int(res[1])
    return err_code, warn_code, state



@dataclass
class RealRobotSample:
    """Aggregated hardware hold statistics at one pose."""

    q: np.ndarray
    qvel: np.ndarray
    tau: np.ndarray
    settled: bool
    tau_median: np.ndarray | None = None
    tau_std: np.ndarray | None = None
    tau_min: np.ndarray | None = None
    tau_max: np.ndarray | None = None
    tau_direct: np.ndarray | None = None
    tau_direct_std: np.ndarray | None = None
    n_samples: int = 1
    duration_s: float = 0.0


class RobotMotionError(RuntimeError):
    """Raised when a real robot motion command fails."""

    def __init__(
        self,
        message: str,
        *,
        code: int,
        waypoint_index: int,
        waypoint_count: int,
        target_q: Sequence[float],
        waypoint_q: Sequence[float],
        actual_q: Sequence[float] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.waypoint_index = waypoint_index
        self.waypoint_count = waypoint_count
        self.target_q = np.asarray(target_q, dtype=np.float64)
        self.waypoint_q = np.asarray(waypoint_q, dtype=np.float64)
        self.actual_q = None if actual_q is None else np.asarray(actual_q, dtype=np.float64)


def build_axis_sequential_waypoints(
    start_q: Sequence[float],
    target_q: Sequence[float],
    *,
    atol: float = 1e-9,
) -> list[np.ndarray]:
    """Build waypoints that move J1..Jn one axis at a time."""
    start = np.asarray(start_q, dtype=np.float64).reshape(-1)
    target = np.asarray(target_q, dtype=np.float64).reshape(-1)
    if start.shape != target.shape:
        raise ValueError(f"start_q shape {start.shape} does not match target_q shape {target.shape}")

    current = start.copy()
    waypoints: list[np.ndarray] = []
    for idx in range(target.size):
        if abs(float(current[idx] - target[idx])) <= atol:
            continue
        current = current.copy()
        current[idx] = target[idx]
        waypoints.append(current.copy())
    return waypoints


def build_motion_waypoints(
    start_q: Sequence[float],
    target_q: Sequence[float],
    *,
    strategy: str = MOVE_STRATEGY_DIRECT,
) -> list[np.ndarray]:
    """Build real-robot motion waypoints for ``direct`` or ``axis-sequential`` strategy."""
    target = np.asarray(target_q, dtype=np.float64).reshape(-1)
    if strategy == MOVE_STRATEGY_AXIS_SEQUENTIAL:
        return build_axis_sequential_waypoints(start_q, target)
    if strategy == MOVE_STRATEGY_DIRECT:
        return [target.copy()]
    raise ValueError(f"Unknown move strategy: {strategy}")


@dataclass(frozen=True)
class ControllerConfigSnapshot:
    """Original controller settings captured before dynamics/sim configuration."""

    tcp_load: tuple[float, list[float]] | None = None
    tcp_offset: list[float] | None = None
    gravity_direction: list[float] | None = None
    report_tau_or_i: int | None = None
    self_collision_detection: bool | None = None


def _capture_controller_snapshot(arm: Any) -> ControllerConfigSnapshot:
    tcp_load: tuple[float, list[float]] | None = None
    raw_load = getattr(arm, "tcp_load", None)
    if (
        isinstance(raw_load, (list, tuple))
        and len(raw_load) >= 2
        and isinstance(raw_load[1], (list, tuple))
        and len(raw_load[1]) >= 3
    ):
        try:
            tcp_load = (float(raw_load[0]), [float(x) for x in raw_load[1][:3]])
        except (TypeError, ValueError):
            tcp_load = None

    tcp_offset: list[float] | None = None
    raw_offset = getattr(arm, "tcp_offset", None)
    if isinstance(raw_offset, (list, tuple)) and len(raw_offset) >= 6:
        try:
            tcp_offset = [float(x) for x in raw_offset[:6]]
        except (TypeError, ValueError):
            tcp_offset = None

    gravity_direction: list[float] | None = None
    raw_grav = getattr(arm, "gravity_direction", None)
    if isinstance(raw_grav, (list, tuple)) and len(raw_grav) >= 3:
        try:
            gravity_direction = [float(x) for x in raw_grav[:3]]
        except (TypeError, ValueError):
            gravity_direction = None

    report_tau_or_i: int | None = None
    tau_getter = getattr(arm, "get_report_tau_or_i", None)
    if callable(tau_getter):
        try:
            res = tau_getter()
            if isinstance(res, tuple) and len(res) == 2 and int(res[0]) == 0:
                report_tau_or_i = int(res[1])
        except Exception:
            report_tau_or_i = None

    self_collision: bool | None = None
    raw_sc = getattr(arm, "self_collision_params", None)
    if isinstance(raw_sc, (list, tuple)) and len(raw_sc) >= 1:
        self_collision = bool(raw_sc[0])

    return ControllerConfigSnapshot(
        tcp_load=tcp_load,
        tcp_offset=tcp_offset,
        gravity_direction=gravity_direction,
        report_tau_or_i=report_tau_or_i,
        self_collision_detection=self_collision,
    )


class RealRobotSession:
    """Controlled xArm SDK session for slow position holds and torque sampling."""

    def __init__(
        self,
        ip: str,
        *,
        dof: int = 6,
        home_qpos: Sequence[float] | None = None,
        is_radian: bool = True,
        motion_mode: int = MODE_POSITION,
    ) -> None:
        try:
            from xarm.wrapper import XArmAPI
        except ImportError as exc:
            raise RuntimeError("xarm-python-sdk ('xarm') is required to connect to an xArm controller") from exc

        self.ip = ip
        self.dof = int(dof)
        self.home_qpos = np.asarray(
            home_qpos if home_qpos is not None else np.zeros(self.dof, dtype=np.float64),
            dtype=np.float64,
        )
        self._motion_mode = motion_mode
        self._config_snapshot: ControllerConfigSnapshot | None = None
        self._modified_dynamics = False
        self._modified_tcp_offset = False
        self._modified_self_collision = False
        self.arm = XArmAPI(ip, is_radian=is_radian)
        time.sleep(0.5)
        if not self.arm.connected:
            try:
                self.arm.disconnect()
            except Exception:
                pass
            raise ConnectionError(f"Failed to connect to xArm at {ip}")

    def ensure_ready(self, mode: int | None = None) -> None:
        """Enable an already healthy controller without clearing or resetting faults."""
        arm = self.arm
        err_code, warn_code, state = _read_err_warn_and_state(arm)
        if err_code != 0 or warn_code != 0 or state == REPORT_STATE_STOPPING:
            raise RuntimeError(
                f"controller is not healthy ({format_arm_status(arm)}); "
                "inspect and recover manually in UFACTORY Studio"
            )
        selected_mode = mode if mode is not None else self._motion_mode
        operations = (
            ("motion_enable", lambda: arm.motion_enable(enable=True)),
            ("set_mode", lambda: arm.set_mode(selected_mode)),
            ("set_state", lambda: arm.set_state(0)),
        )
        for label, operation in operations:
            code = operation()
            if int(code) != 0:
                arm.set_state(REPORT_STATE_STOPPING)
                raise RuntimeError(f"{label} failed with code {code}; controller stopped without automatic recovery")

    def configure_for_dynamics(self, *, clear_tcp_offset: bool = False) -> None:
        """Position mode, bare flange (tcp_load=0), and torque reporting in Nm."""
        if getattr(self, "_config_snapshot", None) is None:
            self._config_snapshot = _capture_controller_snapshot(self.arm)
        self._motion_mode = MODE_POSITION
        self.ensure_ready(MODE_POSITION)
        arm = self.arm
        if clear_tcp_offset:
            _check_config_code("set_tcp_offset(0)", arm.set_tcp_offset([0, 0, 0, 0, 0, 0]))
            self._modified_tcp_offset = True
            self.ensure_ready(MODE_POSITION)
        _check_config_code("set_gravity_direction", arm.set_gravity_direction(DEFAULT_GRAVITY_DIRECTION))
        # Controller leaves ready state after gravity-direction / load updates on xArm firmware.
        self.ensure_ready(MODE_POSITION)
        _check_config_code("set_tcp_load(0)", arm.set_tcp_load(0, [0, 0, 0]))
        self.ensure_ready(MODE_POSITION)
        _check_config_code("set_report_tau_or_i(0)", arm.set_report_tau_or_i(0))
        self._modified_dynamics = True

    def configure_for_simulation_collision_check(
        self,
        *,
        clear_tcp_offset: bool = False,
        simulation_timeout_s: float = SIMULATION_SWITCH_TIMEOUT_S,
    ) -> None:
        """Dynamics setup plus controller simulation mode and self-collision detection."""
        self.configure_for_dynamics(clear_tcp_offset=clear_tcp_offset)
        arm = self.arm
        _check_config_code("set_simulation_robot(True)", arm.set_simulation_robot(on_off=True))
        self.wait_for_simulation_robot(True, timeout_s=simulation_timeout_s)
        _check_config_code("set_self_collision_detection(True)", arm.set_self_collision_detection(True))
        self._modified_self_collision = True
        self.ensure_ready(MODE_POSITION)

    def restore_controller_config(self) -> list[str]:
        """Restore controller load/offset/gravity/collision settings captured before configuration."""
        snap: ControllerConfigSnapshot | None = getattr(self, "_config_snapshot", None)
        if snap is None:
            return []
        mod_dyn = bool(getattr(self, "_modified_dynamics", False))
        mod_off = bool(getattr(self, "_modified_tcp_offset", False))
        mod_sc = bool(getattr(self, "_modified_self_collision", False))
        if not (mod_dyn or mod_off or mod_sc):
            return []
        errors: list[str] = []
        arm = self.arm
        err_code, warn_code, state = _read_err_warn_and_state(arm)
        if err_code != 0 or warn_code != 0 or state == REPORT_STATE_STOPPING:
            errors.append(
                f"skipped controller config restore because controller is not ready ({format_arm_status(arm)})"
            )
            return errors

        def _try_restore(label: str, fn) -> None:
            try:
                code = fn()
                if code is not None and int(code) != 0:
                    errors.append(f"{label} returned {code}")
            except Exception as exc:
                errors.append(f"{label} raised {exc}")

        if mod_sc and snap.self_collision_detection is not None:
            _try_restore(
                f"set_self_collision_detection({snap.self_collision_detection})",
                lambda: arm.set_self_collision_detection(snap.self_collision_detection),
            )
        if mod_off and snap.tcp_offset is not None:
            _try_restore("restore set_tcp_offset", lambda: arm.set_tcp_offset(list(snap.tcp_offset)))
            try:
                self.ensure_ready(MODE_POSITION)
            except Exception as exc:
                errors.append(f"ensure_ready after set_tcp_offset failed: {exc}")
        if mod_dyn:
            if snap.gravity_direction is not None:
                _try_restore(
                    "restore set_gravity_direction",
                    lambda: arm.set_gravity_direction(list(snap.gravity_direction)),
                )
                try:
                    self.ensure_ready(MODE_POSITION)
                except Exception as exc:
                    errors.append(f"ensure_ready after set_gravity_direction failed: {exc}")
            if snap.tcp_load is not None:
                weight, com = snap.tcp_load
                _try_restore("restore set_tcp_load", lambda: arm.set_tcp_load(weight, list(com)))
                try:
                    self.ensure_ready(MODE_POSITION)
                except Exception as exc:
                    errors.append(f"ensure_ready after set_tcp_load failed: {exc}")
            if snap.report_tau_or_i is not None:
                _try_restore(
                    f"restore set_report_tau_or_i({snap.report_tau_or_i})",
                    lambda: arm.set_report_tau_or_i(snap.report_tau_or_i),
                )
        self._modified_dynamics = False
        self._modified_tcp_offset = False
        self._modified_self_collision = False
        return errors

    def disable_simulation_robot(
        self,
        *,
        simulation_timeout_s: float = SIMULATION_SWITCH_TIMEOUT_S,
        restore_ready: bool = True,
    ) -> list[str]:
        """Flush queued virtual motion, leave simulation mode, and optionally restore ready state."""
        errors: list[str] = []
        arm = self.arm
        try:
            stop_code = int(arm.set_state(REPORT_STATE_STOPPING))
            if stop_code != 0:
                errors.append(f"set_state(4) returned {stop_code}")
        except Exception as exc:
            errors.append(f"set_state(4) raised {exc}")
        try:
            disable_code = int(arm.set_simulation_robot(on_off=False))
            if disable_code != 0:
                errors.append(f"set_simulation_robot(False) returned {disable_code}")
            self.wait_for_simulation_robot(False, timeout_s=simulation_timeout_s)
        except Exception as exc:
            errors.append(str(exc))
        if restore_ready and not errors:
            try:
                arm.motion_enable(enable=True)
                arm.set_mode(MODE_POSITION)
                arm.set_state(0)
            except Exception as exc:
                errors.append(f"post-sim ready restore failed: {exc}")
        time.sleep(0.2)
        return errors

    def wait_for_simulation_robot(
        self,
        enabled: bool,
        *,
        timeout_s: float = SIMULATION_SWITCH_TIMEOUT_S,
        poll_s: float = SETTLE_POLL_S,
    ) -> None:
        """Wait until the controller report stream confirms ``is_simulation_robot``."""
        deadline = time.monotonic() + max(0.0, timeout_s)
        while bool(getattr(self.arm, "is_simulation_robot", False)) != bool(enabled):
            if time.monotonic() >= deadline:
                self.arm.set_state(REPORT_STATE_STOPPING)
                raise RuntimeError(
                    f"simulation robot did not switch to {bool(enabled)} within {timeout_s:.1f} s; controller stopped"
                )
            time.sleep(poll_s)

    def print_config(self) -> None:
        arm = self.arm
        print(f"SDK firmware   : {arm.version}")
        code, reported_state = arm.get_state()
        gs = f"{reported_state}" if code == 0 else f"err({code})"
        print(f"robot state    : {arm.state} (report; 4=stop)  get_state()={gs}")
        print(f"robot mode     : {arm.mode} (report; 0=position)")
        print(f"error/warn     : {arm.error_code} / {arm.warn_code}")
        print(f"tcp_offset     : {list(arm.tcp_offset)}")
        print(f"tcp_load       : {arm.tcp_load}")
        print(f"gravity_dir    : {list(arm.gravity_direction)}")
        code, tau_or_i = arm.get_report_tau_or_i()
        if code == 0:
            print(f"report_tau_or_i: {tau_or_i} (0=torque Nm)")

    def get_joint_states(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        code, states = self.arm.get_joint_states(is_radian=True, num=3)
        if code != 0:
            raise RuntimeError(f"get_joint_states failed with code {code}")
        pos, vel, effort = states
        return (
            np.asarray(pos[: self.dof], dtype=np.float64),
            np.asarray(vel[: self.dof], dtype=np.float64),
            np.asarray(effort[: self.dof], dtype=np.float64),
        )

    def get_joints_torque(self) -> np.ndarray | None:
        """Read direct joint torque API when available."""
        getter = getattr(self.arm, "get_joints_torque", None)
        if getter is None:
            return None
        code, tau = getter()
        if code != 0:
            return None
        return np.asarray(tau[: self.dof], dtype=np.float64)

    def get_joint_currents(self) -> np.ndarray:
        """Read per-joint servo current from the SDK report stream."""
        return np.asarray(self.arm.currents[: self.dof], dtype=np.float64)

    def wait_until_settled(
        self,
        *,
        vel_tol: float = VEL_TOL_RAD_S,
        timeout_s: float = SETTLE_TIMEOUT_S,
        poll_s: float = SETTLE_POLL_S,
    ) -> RealRobotSample:
        deadline = time.monotonic() + timeout_s
        last_sample: RealRobotSample | None = None
        while time.monotonic() < deadline:
            q, qvel, tau = self.get_joint_states()
            last_sample = RealRobotSample(q=q, qvel=qvel, tau=tau, settled=False)
            if float(np.abs(qvel).max()) <= vel_tol:
                last_sample.settled = True
                return last_sample
            time.sleep(poll_s)
        if last_sample is None:
            raise RuntimeError("No joint state samples received")
        return last_sample

    def collect_hold_samples(
        self,
        *,
        duration_s: float = 3.0,
        poll_s: float = SETTLE_POLL_S,
        vel_tol: float = VEL_TOL_RAD_S,
    ) -> RealRobotSample:
        """Collect steady-state samples over ``duration_s`` and return mean/median/std/min/max."""
        deadline = time.monotonic() + max(0.0, duration_s)
        qs: list[np.ndarray] = []
        qvels: list[np.ndarray] = []
        efforts: list[np.ndarray] = []
        direct_torques: list[np.ndarray] = []

        while True:
            q, qvel, tau = self.get_joint_states()
            qs.append(q)
            qvels.append(qvel)
            efforts.append(tau)
            tau_direct = self.get_joints_torque()
            if tau_direct is not None:
                direct_torques.append(tau_direct)

            if time.monotonic() >= deadline:
                break
            time.sleep(poll_s)

        q_arr = np.vstack(qs)
        qvel_arr = np.vstack(qvels)
        tau_arr = np.vstack(efforts)
        settled = bool(float(np.abs(qvel_arr).max()) <= vel_tol)
        tau_direct_mean = tau_direct_std = None
        if direct_torques:
            direct_arr = np.vstack(direct_torques)
            tau_direct_mean = direct_arr.mean(axis=0)
            tau_direct_std = direct_arr.std(axis=0)

        return RealRobotSample(
            q=q_arr.mean(axis=0),
            qvel=qvel_arr.mean(axis=0),
            tau=tau_arr.mean(axis=0),
            settled=settled,
            tau_median=np.median(tau_arr, axis=0),
            tau_std=tau_arr.std(axis=0),
            tau_min=tau_arr.min(axis=0),
            tau_max=tau_arr.max(axis=0),
            tau_direct=tau_direct_mean,
            tau_direct_std=tau_direct_std,
            n_samples=len(efforts),
            duration_s=duration_s,
        )

    def move_to(
        self,
        q: Sequence[float],
        *,
        speed_rad_s: float = DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S,
        wait: bool = True,
        move_strategy: str = MOVE_STRATEGY_DIRECT,
    ) -> int:
        checked_speed = validate_dynamics_move_speed(speed_rad_s)
        target = np.asarray(q, dtype=np.float64).reshape(-1)
        q_start, _, _ = self.get_joint_states()
        waypoints = build_motion_waypoints(q_start, target, strategy=move_strategy)
        if not waypoints:
            return 0

        for i, waypoint in enumerate(waypoints, start=1):
            self.ensure_ready(MODE_POSITION)
            print(
                f"    waypoint {i}/{len(waypoints)} [{move_strategy}] "
                f"speed={checked_speed:.4f} rad/s "
                f"q=[{', '.join(f'{v:.4f}' for v in waypoint)}]"
            )
            code = self.arm.set_servo_angle(
                angle=waypoint.tolist(),
                speed=checked_speed,
                mvtime=0,
                wait=wait,
                is_radian=True,
                radius=None,
            )
            if code != 0:
                try:
                    q_now, _, _ = self.get_joint_states()
                except Exception:
                    q_now = None
                actual = "unknown" if q_now is None else list(np.asarray(q_now, dtype=np.float64))
                raise RobotMotionError(
                    f"set_servo_angle failed code={code} waypoint {i}/{len(waypoints)} "
                    f"move_strategy={move_strategy} target_q={target.tolist()} "
                    f"waypoint_q={waypoint.tolist()} actual_q={actual} {format_arm_status(self.arm)}",
                    code=code,
                    waypoint_index=i,
                    waypoint_count=len(waypoints),
                    target_q=target,
                    waypoint_q=waypoint,
                    actual_q=q_now,
                )
        return 0

    def sample_at_hold(
        self,
        q: Sequence[float],
        *,
        speed_rad_s: float = DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S,
        move_strategy: str = MOVE_STRATEGY_DIRECT,
        post_move_wait_s: float = 1.5,
        sample_duration_s: float = 3.0,
        sample_poll_s: float = SETTLE_POLL_S,
    ) -> RealRobotSample:
        self.move_to(q, speed_rad_s=speed_rad_s, wait=True, move_strategy=move_strategy)
        time.sleep(post_move_wait_s)
        settled_sample = self.wait_until_settled()
        stats = self.collect_hold_samples(duration_s=sample_duration_s, poll_s=sample_poll_s)
        stats.settled = bool(settled_sample.settled and stats.settled)
        return stats

    def stop(self) -> None:
        """Send software STOP (state=4) to immediately halt motion and flush queued commands."""
        try:
            self.arm.set_state(REPORT_STATE_STOPPING)
        except Exception:
            pass

    def return_home(
        self,
        *,
        speed_rad_s: float = DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S,
        move_strategy: str = MOVE_STRATEGY_DIRECT,
    ) -> None:
        try:
            self.move_to(self.home_qpos, speed_rad_s=speed_rad_s, wait=True, move_strategy=move_strategy)
            time.sleep(1.0)
        except RuntimeError as exc:
            print(f"[WARN] return_home failed: {exc}")

    def disconnect(self) -> None:
        self.arm.disconnect()
