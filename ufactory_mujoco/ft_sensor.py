"""AI1500 6-axis force/torque sensor reader, 200 Hz sampling, and hardware bridge."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import time
from typing import Any

import mujoco
import numpy as np

TIMESTEP_S = 0.001
SAMPLE_STEPS = 5
ACCURACY_SPEC = np.array([0.4, 0.4, 0.4, 0.01, 0.01, 0.01], dtype=float)
RESOLUTION = ACCURACY_SPEC  # backwards-compatible alias for datasheet accuracy spec
RAW_NOISE_STD = np.array([0.065029, 0.065496, 0.048908, 0.001037, 0.001056, 0.001398], dtype=float)
EXT_NOISE_STD = np.array([0.030548, 0.028681, 0.024360, 0.000480, 0.000512, 0.000649], dtype=float)
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


def _as_vector6(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (6,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite 6-element vector")
    return arr.copy()


@dataclass(frozen=True)
class SensorReading:
    """Instantaneous and sampled local-frame wrench (raw & ext) and per-axis states."""

    raw: np.ndarray
    sampled: np.ndarray
    sample_time_s: float
    overrange: np.ndarray
    overload: np.ndarray
    ext: np.ndarray
    source: str = "sim"

    def as_dict(self) -> dict:
        """Convert the reading into JSON-serializable plain Python values."""
        return {
            "raw": self.raw.tolist(),
            "ext": self.ext.tolist(),
            "sampled": self.sampled.tolist(),
            "sample_time_s": self.sample_time_s,
            "overrange": self.overrange.tolist(),
            "overload": self.overload.tolist(),
            "source": self.source,
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


def _find_wrench_target(model: mujoco.MjModel) -> tuple[int, int] | None:
    for body_name, site_name in _WRENCH_TARGETS:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if body_id >= 0 and site_id >= 0:
            return int(body_id), int(site_id)
    return None


def has_ft_sensor(model: mujoco.MjModel) -> bool:
    """Return True if ``model`` contains valid AI1500 force and torque sensors."""
    return _find_sensor_names(model) is not None


def apply_ft_sensor_wrench(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    force: np.ndarray | tuple[float, float, float] | list[float],
    torque: np.ndarray | tuple[float, float, float] | list[float],
    *,
    at_output_face: bool = False,
) -> None:
    """Apply local-frame external force (N) and torque (N*m) onto the sensor.

    Converts ``(force, torque)`` from the current ``ft_sensor_site`` frame
    (or the ``+58.6 mm`` ``ft_sensor_output`` face when ``at_output_face=True``)
    into world-frame ``data.xfrc_applied`` on the sensor output body. Matches real
    AI1500 polarity: an applied external wrench produces ``net_delta ~= +applied``.
    When the arm is moving/rotating, call this helper before each physics step
    and call ``clear_ft_sensor_wrench`` afterwards.
    """
    f_local = np.asarray(force, dtype=float)
    t_local = np.asarray(torque, dtype=float)
    if f_local.shape != (3,) or t_local.shape != (3,):
        raise ValueError("force and torque must each have shape (3,)")
    if not np.isfinite(f_local).all() or not np.isfinite(t_local).all():
        raise ValueError("force and torque must be finite")

    target = _find_wrench_target(model)
    if target is None:
        raise ValueError("FT sensor output body or measurement site is missing")
    body_id, site_id = target

    mujoco.mj_forward(model, data)
    if at_output_face:
        rotation = data.xmat[body_id].reshape(3, 3)
        point = data.xpos[body_id]
    else:
        rotation = data.site_xmat[site_id].reshape(3, 3)
        point = data.site_xpos[site_id]
    world_force = rotation @ f_local
    world_torque = rotation @ t_local
    data.xfrc_applied[body_id, :3] = world_force
    data.xfrc_applied[body_id, 3:] = world_torque + np.cross(point - data.xipos[body_id], world_force)


def clear_ft_sensor_wrench(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Clear any external wrench applied to the FT sensor output body."""
    for body_name, _site_name in _WRENCH_TARGETS:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if b >= 0:
            data.xfrc_applied[b] = 0.0
            return
    raise ValueError("FT sensor output body is missing")


class AI1500Reader:
    """Sample an AI1500 F/T sensor at 1 kHz physics and output 200 Hz fp32 readings.

    Produces real-hardware polarity (``+`` = external wrench applied onto the
    output face) and dual channels: ``raw`` (with floating head + downstream
    gravity + optional noise) and ``ext`` (subtracting ``zero_offset`` and
    pose-dependent static downstream subtree gravity). Call ``update`` at the
    initial state and after every 1 ms ``mj_step``.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        sample_steps: int = SAMPLE_STEPS,
        zero_offset: Sequence[float] | np.ndarray | None = None,
        noise_std: Sequence[float] | str | None = None,
        seed: int | None = None,
    ):
        if int(sample_steps) < 1:
            raise ValueError("sample_steps must be >= 1")
        if not np.isclose(model.opt.timestep, TIMESTEP_S, rtol=0.0, atol=1e-12):
            raise ValueError("AI1500Reader requires a 0.001 s simulation timestep")
        names = _find_sensor_names(model)
        if names is None:
            raise ValueError("Missing or invalid FT sensor force/torque pair in model")
        self.model = model
        self.data = data
        self.sample_steps = int(sample_steps)
        self._zero_offset = (
            np.zeros(6, dtype=float)
            if zero_offset is None
            else _as_vector6(zero_offset, "zero_offset")
        )
        self._raw_noise_std, self._ext_noise_std = self._resolve_noise(noise_std)
        self._rng = np.random.default_rng(seed)

        self._force, f_idx = self._sensor_address(names[0], mujoco.mjtSensor.mjSENS_FORCE)
        self._torque, _t_idx = self._sensor_address(names[1], mujoco.mjtSensor.mjSENS_TORQUE)
        self._site_id, self._output_body_id = self._resolve_site_and_body(f_idx)
        self._last_time: float | None = None
        self._steps: int = 0
        self._sampled: np.ndarray | None = None
        self._sample_time: float | None = None
        self._overrange: np.ndarray | None = None
        self._overload: np.ndarray | None = None
        self.reset()

    @staticmethod
    def _resolve_noise(noise_std: Sequence[float] | str | None) -> tuple[np.ndarray | None, np.ndarray | None]:
        if noise_std is None:
            return None, None
        if isinstance(noise_std, str):
            if noise_std == "calibrated":
                return RAW_NOISE_STD.copy(), EXT_NOISE_STD.copy()
            raise ValueError(f"Unsupported noise_std preset: {noise_std!r}")
        custom = _as_vector6(noise_std, "noise_std")
        if (custom < 0.0).any():
            raise ValueError("noise_std must be non-negative")
        return custom, custom.copy()

    def _sensor_address(self, name: str, sensor_type: int) -> tuple[int, int]:
        index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if (
            index < 0
            or self.model.sensor_dim[index] != 3
            or int(self.model.sensor_type[index]) != int(sensor_type)
        ):
            raise ValueError(f"Missing or invalid sensor: {name}")
        return int(self.model.sensor_adr[index]), int(index)

    def _resolve_site_and_body(self, force_sensor_idx: int) -> tuple[int, int]:
        if int(self.model.sensor_objtype[force_sensor_idx]) == int(mujoco.mjtObj.mjOBJ_SITE):
            site_id = int(self.model.sensor_objid[force_sensor_idx])
            if site_id >= 0:
                return site_id, int(self.model.site_bodyid[site_id])
        target = _find_wrench_target(self.model)
        if target is None:
            raise ValueError("FT sensor output body or measurement site is missing")
        return target[1], target[0]

    def _gravity_wrench(self) -> np.ndarray:
        mass_sub = float(self.model.body_subtreemass[self._output_body_id])
        rotation = self.data.site_xmat[self._site_id].reshape(3, 3)
        g_local = rotation.T @ self.model.opt.gravity
        r_local = rotation.T @ (self.data.subtree_com[self._output_body_id] - self.data.site_xpos[self._site_id])
        f_grav = mass_sub * g_local
        return np.concatenate((f_grav, np.cross(r_local, f_grav)))

    def _raw_ideal_and_gravity(self) -> tuple[np.ndarray, np.ndarray]:
        # mj_step advances time after computing sensordata; recompute at current state.
        mujoco.mj_forward(self.model, self.data)
        sens = np.concatenate(
            (
                self.data.sensordata[self._force : self._force + 3],
                self.data.sensordata[self._torque : self._torque + 3],
            )
        )
        if not np.isfinite(sens).all():
            raise ValueError("Non-finite AI1500 wrench")
        return -sens, self._gravity_wrench()

    def zero(self) -> np.ndarray:
        """Capture current non-gravitational wrench offset so ``ext`` reads zero."""
        raw_ideal, w_grav = self._raw_ideal_and_gravity()
        self._zero_offset = (raw_ideal - w_grav).copy()
        return self._zero_offset.copy()

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
                sample_due = next_steps % self.sample_steps == 0
            else:
                raise ValueError(
                    "Call update after each 1 ms step; call reset after resetting simulation time"
                )

        raw_ideal, w_grav = self._raw_ideal_and_gravity()
        raw = raw_ideal.copy()
        ext = (raw_ideal - self._zero_offset) - w_grav
        if self._raw_noise_std is not None:
            raw = raw + self._rng.normal(0.0, self._raw_noise_std)
        if self._ext_noise_std is not None:
            ext = ext + self._rng.normal(0.0, self._ext_noise_std)

        if sample_due:
            self._sampled = raw.astype(np.float32).astype(np.float64)
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
            raw=raw,
            sampled=self._sampled.copy(),
            sample_time_s=self._sample_time,
            overrange=self._overrange.copy(),
            overload=self._overload.copy(),
            ext=ext,
            source="sim",
        )


def is_hardware_ft_sensor_enabled(arm: Any) -> bool:
    """Return True if the hardware controller has an F/T sensor enabled.

    Avoids ``get_ft_sensor_error()`` / ``SERVO_R16B`` queries which drop
    controller state from ``2`` back to ``5`` on firmware v2.8.0.
    """
    raw_flag = getattr(arm, "ft_sensor_is_enable", None)
    if raw_flag is None:
        raw_flag = getattr(getattr(arm, "_arm", None), "ft_sensor_is_enable", False)
    enabled = isinstance(raw_flag, (bool, int, np.integer)) and bool(int(raw_flag))
    getter = getattr(arm, "get_ft_sensor_config", None)
    if not callable(getter):
        return enabled
    try:
        code, cfg = getter()
    except Exception:
        return enabled
    if (
        not isinstance(code, (bool, int, np.integer))
        or int(code) not in (0, 1, 2)
        or not isinstance(cfg, (list, tuple))
        or len(cfg) < 2
    ):
        return enabled
    return (isinstance(cfg[1], (bool, int, np.integer)) and bool(int(cfg[1]))) or enabled


def read_hardware_ft_sensor_config(arm: Any) -> dict[str, Any]:
    """Read controller F/T sensor configuration non-intrusively via ``get_ft_sensor_config``."""
    from .safety import SafetyGateError

    getter = getattr(arm, "get_ft_sensor_config", None)
    if not callable(getter):
        raise SafetyGateError("Controller handle does not implement get_ft_sensor_config()")
    code, cfg = getter()
    if int(code) not in (0, 2) or not isinstance(cfg, (list, tuple)) or len(cfg) < 9:
        raise SafetyGateError(f"get_ft_sensor_config() failed (code={code}, cfg={cfg!r})")
    return {
        "ft_app_status": int(cfg[0]),
        "ft_is_started": bool(int(cfg[1])),
        "ft_type": int(cfg[2]),
        "ft_id": int(cfg[3]),
        "ft_freq_hz": int(cfg[4]),
        "ft_mass_kg": float(cfg[5]),
        "ft_dir_bias": float(cfg[6]),
        "ft_centroid_m": [float(v) for v in cfg[7][:3]],
        "ft_zero": [float(v) for v in cfg[8][:6]],
    }


def read_hardware_ft_sensor(arm: Any) -> SensorReading:
    """Read raw and compensated external 6-axis wrench from real hardware safely.

    Checks ``get_err_warn_code()`` first (raising ``SafetyGateError`` on active
    controller error) and avoids ``get_ft_sensor_error()`` so controller state
    remains intact.
    """
    from .safety import SafetyGateError

    err_getter = getattr(arm, "get_err_warn_code", None)
    if callable(err_getter):
        err_code, err_warn = err_getter()
        if int(err_code) not in (0, 2) or int(err_warn[0]) != 0:
            raise SafetyGateError(
                f"Hardware F/T read blocked by controller error (code={err_code}, err_warn={err_warn})"
            )

    data_getter = getattr(arm, "get_ft_sensor_data", None)
    if not callable(data_getter):
        raise SafetyGateError("Controller handle does not implement get_ft_sensor_data()")

    code_raw, raw_vals = data_getter(is_raw=True)
    code_ext, ext_vals = data_getter(is_raw=False)
    if (
        int(code_raw) not in (0, 2)
        or int(code_ext) not in (0, 2)
        or not isinstance(raw_vals, (list, tuple, np.ndarray))
        or not isinstance(ext_vals, (list, tuple, np.ndarray))
        or len(raw_vals) < 6
        or len(ext_vals) < 6
    ):
        raise SafetyGateError(
            f"Hardware get_ft_sensor_data() failed (code_raw={code_raw}, code_ext={code_ext})"
        )

    raw = _as_vector6(raw_vals[:6], "hardware raw wrench")
    ext = _as_vector6(ext_vals[:6], "hardware ext wrench")
    sampled = raw.astype(np.float32).astype(np.float64)
    return SensorReading(
        raw=raw,
        sampled=sampled,
        sample_time_s=float(time.monotonic()),
        overrange=np.abs(ext) > RANGE,
        overload=np.abs(ext) >= OVERLOAD,
        ext=ext,
        source="hardware",
    )


FTSensorReader = AI1500Reader
