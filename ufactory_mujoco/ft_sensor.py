"""AI1500 6-axis force/torque sensor reader and deterministic 200 Hz sampling."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

TIMESTEP_S = 0.001
SAMPLE_STEPS = 5
RESOLUTION = np.array([0.4, 0.4, 0.4, 0.01, 0.01, 0.01], dtype=float)
RANGE = np.array([400.0, 400.0, 400.0, 20.0, 20.0, 20.0], dtype=float)
OVERLOAD = 1.5 * RANGE

_SENSOR_PAIRS: tuple[tuple[str, str], ...] = (
    ("ft_sensor_force", "ft_sensor_torque"),
    ("ai1500_force", "ai1500_torque"),
)
_WRENCH_TARGETS: tuple[tuple[str, str], ...] = (
    ("ft_sensor_output", "ft_sensor_site"),
    ("ai1500_output", "ai1500_measurement"),
)


@dataclass(frozen=True)
class SensorReading:
    """Instantaneous and last sampled local-frame wrench and per-axis states."""

    raw: np.ndarray
    sampled: np.ndarray
    sample_time_s: float
    overrange: np.ndarray
    overload: np.ndarray

    def as_dict(self) -> dict:
        """Convert the reading into JSON-serializable plain Python values."""
        return {
            "raw": self.raw.tolist(),
            "sampled": self.sampled.tolist(),
            "sample_time_s": self.sample_time_s,
            "overrange": self.overrange.tolist(),
            "overload": self.overload.tolist(),
        }


def _find_sensor_names(model: mujoco.MjModel) -> tuple[str, str] | None:
    for force_name, torque_name in _SENSOR_PAIRS:
        f_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, force_name)
        t_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, torque_name)
        if (
            f_idx >= 0
            and t_idx >= 0
            and model.sensor_dim[f_idx] == 3
            and model.sensor_dim[t_idx] == 3
            and int(model.sensor_type[f_idx]) == int(mujoco.mjtSensor.mjSENS_FORCE)
            and int(model.sensor_type[t_idx]) == int(mujoco.mjtSensor.mjSENS_TORQUE)
        ):
            return force_name, torque_name
    return None


def has_ft_sensor(model: mujoco.MjModel) -> bool:
    """Return True if ``model`` contains valid AI1500 force and torque sensors."""
    return _find_sensor_names(model) is not None


def apply_ft_sensor_wrench(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    force: np.ndarray | tuple[float, float, float] | list[float],
    torque: np.ndarray | tuple[float, float, float] | list[float],
) -> None:
    """Apply local-frame force (N) and torque (N*m) at the sensor output site.

    Converts ``(force, torque)`` from the current ``ft_sensor_site`` frame into
    world-frame ``data.xfrc_applied`` on the sensor output body. When the arm is
    moving/rotating, call this helper before each physics step so the world
    wrench tracks the rotating site frame, and call ``clear_ft_sensor_wrench``
    afterwards. Note that MuJoCo force/torque sensors report internal reaction
    wrenches, so an applied external wrench produces ``net_delta ~= -applied``.
    """
    f_local = np.asarray(force, dtype=float)
    t_local = np.asarray(torque, dtype=float)
    if f_local.shape != (3,) or t_local.shape != (3,):
        raise ValueError("force and torque must each have shape (3,)")
    if not np.isfinite(f_local).all() or not np.isfinite(t_local).all():
        raise ValueError("force and torque must be finite")

    body_id = -1
    site_id = -1
    for body_name, site_name in _WRENCH_TARGETS:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        s = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if b >= 0 and s >= 0:
            body_id, site_id = b, s
            break
    if body_id < 0 or site_id < 0:
        raise ValueError("FT sensor output body or measurement site is missing")

    mujoco.mj_forward(model, data)
    rotation = data.site_xmat[site_id].reshape(3, 3)
    world_force = rotation @ f_local
    world_torque = rotation @ t_local
    data.xfrc_applied[body_id, :3] = world_force
    data.xfrc_applied[body_id, 3:] = world_torque + np.cross(
        data.site_xpos[site_id] - data.xipos[body_id], world_force
    )


def clear_ft_sensor_wrench(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Clear any external wrench applied to the FT sensor output body."""
    for body_name, _site_name in _WRENCH_TARGETS:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if b >= 0:
            data.xfrc_applied[b] = 0.0
            return
    raise ValueError("FT sensor output body is missing")


class AI1500Reader:
    """Sample an AI1500 F/T sensor once per 1 ms physics step, holding for 5 ms.

    Call ``update`` at the initial state and after every ``mj_step``. Repeated
    calls at the same simulation time refresh the raw value without resampling.
    After resetting ``mjData`` or changing simulation time, call ``reset`` first.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        if not np.isclose(model.opt.timestep, TIMESTEP_S, rtol=0.0, atol=1e-12):
            raise ValueError("AI1500Reader requires a 0.001 s simulation timestep")
        names = _find_sensor_names(model)
        if names is None:
            raise ValueError("Missing or invalid FT sensor force/torque pair in model")
        self.model = model
        self.data = data
        self._force = self._sensor_address(names[0], mujoco.mjtSensor.mjSENS_FORCE)
        self._torque = self._sensor_address(names[1], mujoco.mjtSensor.mjSENS_TORQUE)
        self._last_time: float | None = None
        self._steps: int = 0
        self._sampled: np.ndarray | None = None
        self._sample_time: float | None = None
        self._overrange: np.ndarray | None = None
        self._overload: np.ndarray | None = None
        self.reset()

    def _sensor_address(self, name: str, sensor_type: int) -> int:
        index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if (
            index < 0
            or self.model.sensor_dim[index] != 3
            or int(self.model.sensor_type[index]) != int(sensor_type)
        ):
            raise ValueError(f"Missing or invalid sensor: {name}")
        return int(self.model.sensor_adr[index])

    def reset(self) -> None:
        """Forget the sample schedule; the next ``update`` samples immediately."""
        self._last_time = None
        self._steps = 0
        self._sampled = None
        self._sample_time = None
        self._overrange = None
        self._overload = None

    def update(self) -> SensorReading:
        """Refresh the MuJoCo sensor and return instantaneous and held readings."""
        now = float(self.data.time)
        if not np.isfinite(now):
            raise ValueError("Simulation time must be finite")
        sample_due = self._last_time is None
        next_steps = self._steps
        if self._last_time is not None:
            elapsed = now - self._last_time
            if abs(elapsed) <= 1e-9:
                pass
            elif abs(elapsed - TIMESTEP_S) <= 1e-9:
                next_steps += 1
                sample_due = next_steps % SAMPLE_STEPS == 0
            else:
                raise ValueError(
                    "Call update after each 1 ms step; call reset after resetting simulation time"
                )

        # mj_step advances time after computing sensordata; recompute at current state.
        mujoco.mj_forward(self.model, self.data)
        raw = np.concatenate(
            (
                self.data.sensordata[self._force : self._force + 3],
                self.data.sensordata[self._torque : self._torque + 3],
            )
        ).copy()
        if not np.isfinite(raw).all():
            raise ValueError("Non-finite AI1500 wrench")
        if sample_due:
            self._sampled = np.rint(raw / RESOLUTION) * RESOLUTION
            self._sample_time = now
            self._overrange = np.abs(raw) > RANGE
            self._overload = np.abs(raw) >= OVERLOAD
        assert self._sampled is not None
        assert self._sample_time is not None
        assert self._overrange is not None
        assert self._overload is not None
        self._steps = next_steps
        self._last_time = now
        return SensorReading(
            raw,
            self._sampled.copy(),
            self._sample_time,
            self._overrange.copy(),
            self._overload.copy(),
        )


FTSensorReader = AI1500Reader
