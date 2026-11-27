"""AL0700 700 mm Direct-Drive Linear Motor control, drag-chain deformation, and viewer support."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
import re
import time

import mujoco
import numpy as np

from . import viewer as viewer_mod
from .compose import compose_spec, initial_keyframe, load_model
from .ft_sensor import AI1500Reader, SensorReading, has_ft_sensor
from .grippers import GRIPPER_ACTUATOR_NAMES, GripperSpec, identify_gripper

AL0700_STROKE_MM = 700.0
AL0700_SLIDE_JOINT = "al0700_slide"
AL0700_ACTUATOR = "al0700_position"
AL0700_DRAG_CHAIN_MESH = "al0700_part_35"

AL0700_ROTATION = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
AL0700_ORIGIN = np.array([0.3435, -0.03824, 0.042], dtype=float)
AL0700_MOUNT_ROTATION = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)

# CAD-frame drag chain landmarks from ufactory_robot_viewer/src/rail-drag-chain.ts (metres).
_BEND_END_Z = 0.12437742
_STRAIGHT_END_Z = 0.290
_BEND_CENTER_Y = 0.0104962
_ZERO_EXTENSION = 0.220
_ARM_JOINT_RE = re.compile(r"joint\d+")


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


@dataclass(frozen=True)
class RailConfig:
    """Validated simulation, trajectory, and servo tolerances for AL0700."""

    total_mass_kg: float = 20.0
    carriage_mass_kg: float | None = None
    kp_n_per_m: float = 20000.0
    dampratio: float = 1.0
    max_force_n: float = 160.0
    max_speed_mm_s: float = 1000.0
    default_speed_mm_s: float = 100.0
    acceleration_mm_s2: float = 200.0
    position_tolerance_mm: float = 0.1
    velocity_tolerance_mm_s: float = 0.5
    settle_timeout_s: float = 3.0
    timestep_s: float = 0.001
    limit_timeconstant_s: float = 0.002

    def __post_init__(self) -> None:
        fields = (
            ("total_mass_kg", self.total_mass_kg),
            ("kp_n_per_m", self.kp_n_per_m),
            ("dampratio", self.dampratio),
            ("max_force_n", self.max_force_n),
            ("max_speed_mm_s", self.max_speed_mm_s),
            ("default_speed_mm_s", self.default_speed_mm_s),
            ("acceleration_mm_s2", self.acceleration_mm_s2),
            ("position_tolerance_mm", self.position_tolerance_mm),
            ("velocity_tolerance_mm_s", self.velocity_tolerance_mm_s),
            ("settle_timeout_s", self.settle_timeout_s),
            ("timestep_s", self.timestep_s),
            ("limit_timeconstant_s", self.limit_timeconstant_s),
        )
        for name, raw in fields:
            val = _finite(raw, name)
            if val <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.carriage_mass_kg is not None:
            carriage = _finite(self.carriage_mass_kg, "carriage_mass_kg")
            if carriage <= 0.0 or carriage >= self.total_mass_kg:
                raise ValueError("carriage_mass_kg must be in (0, total_mass_kg)")
        if self.default_speed_mm_s > self.max_speed_mm_s:
            raise ValueError("default_speed_mm_s exceeds max_speed_mm_s")
        if self.limit_timeconstant_s < 2.0 * self.timestep_s:
            raise ValueError("limit_timeconstant_s must be at least twice timestep_s")


def to_world(vertices: np.ndarray) -> np.ndarray:
    """Transform AL0700 CAD coordinates into MuJoCo world coordinates (metres)."""
    return np.asarray(vertices, dtype=float) @ AL0700_ROTATION.T + AL0700_ORIGIN


def deform_drag_chain(vertices: np.ndarray, travel_m: float) -> np.ndarray:
    """Compute drag-chain CAD vertex positions for a given rail travel (metres)."""
    travel = float(travel_m)
    output = np.array(vertices, dtype=float, copy=True)
    weight = np.clip((output[:, 2] - _BEND_END_Z) / (_STRAIGHT_END_Z - _BEND_END_Z), 0.0, 1.0)
    end_offset = np.where(output[:, 1] > _BEND_CENTER_Y, -travel, 0.0)
    output[:, 2] += (1.0 - weight) * (-_ZERO_EXTENSION - travel / 2.0) + weight * end_offset
    return output


def has_linear_motor(model: mujoco.MjModel) -> bool:
    """Return True if ``model`` contains the AL0700 slide joint and position actuator."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, AL0700_SLIDE_JOINT)
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, AL0700_ACTUATOR)
    if joint_id < 0 or actuator_id < 0:
        return False
    return (
        int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_SLIDE)
        and int(model.actuator_trntype[actuator_id]) == int(mujoco.mjtTrn.mjTRN_JOINT)
        and int(model.actuator_trnid[actuator_id, 0]) == joint_id
    )


class RailController:
    """Write only the named AL0700 rail actuator; the caller owns physics stepping."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: RailConfig | None = None):
        self.model = model
        self.data = data
        self.config = RailConfig() if config is None else config
        self.joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, AL0700_SLIDE_JOINT)
        self.actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, AL0700_ACTUATOR)
        if not has_linear_motor(model):
            raise ValueError("Model has no valid AL0700 slide/position actuator mapping")
        self.qpos_address = int(model.jnt_qposadr[self.joint_id])
        self.dof_address = int(model.jnt_dofadr[self.joint_id])
        self.target_mm = self.get_position_mm()
        self._start_m = self.target_mm / 1000.0
        self._start_time = float(data.time)
        self.duration_s = 0.0

    def get_position_mm(self) -> float:
        """Current carriage travel in millimetres."""
        return float(self.data.qpos[self.qpos_address] * 1000.0)

    def get_velocity_mm_s(self) -> float:
        """Current carriage velocity in millimetres/second."""
        return float(self.data.qvel[self.dof_address] * 1000.0)

    def set_target_mm(self, position_mm: float, speed_mm_s: float | None = None) -> None:
        """Start a rest-to-rest quintic command; ``speed_mm_s`` is its peak, not average."""
        target = _finite(position_mm, "position_mm")
        speed = _finite(
            self.config.default_speed_mm_s if speed_mm_s is None else speed_mm_s,
            "speed_mm_s",
        )
        if not 0.0 <= target <= AL0700_STROKE_MM:
            raise ValueError(f"position_mm must be within 0..{AL0700_STROKE_MM:g}")
        if not 0.0 < speed <= self.config.max_speed_mm_s:
            raise ValueError(f"speed_mm_s must be within (0, {self.config.max_speed_mm_s:g}]")
        current_mm = self.get_position_mm()
        self._start_m = current_mm / 1000.0
        self.target_mm = target
        self._start_time = float(self.data.time)
        distance_mm = abs(target - current_mm)
        # Quintic maxima: max(f') = 15/8, max(abs(f'')) = 10/sqrt(3).
        self.duration_s = max(
            1.875 * distance_mm / speed,
            math.sqrt((10.0 / math.sqrt(3.0)) * distance_mm / self.config.acceleration_mm_s2),
        )

    @property
    def trajectory_complete(self) -> bool:
        """True once the quintic command duration has elapsed."""
        return float(self.data.time) - self._start_time >= self.duration_s

    def settled(self) -> bool:
        """True when trajectory duration elapsed and position/velocity are within tolerances."""
        return (
            self.trajectory_complete
            and abs(self.get_position_mm() - self.target_mm) <= self.config.position_tolerance_mm
            and abs(self.get_velocity_mm_s()) <= self.config.velocity_tolerance_mm_s
        )

    def update(self, dt: float | None = None) -> None:
        """Advance the position command for the upcoming physics step (metres)."""
        step_dt = float(self.model.opt.timestep) if dt is None else _finite(dt, "dt")
        if step_dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.duration_s == 0.0:
            fraction = 1.0
        else:
            fraction = float(np.clip((float(self.data.time) + step_dt - self._start_time) / self.duration_s, 0.0, 1.0))
        smooth = fraction**3 * (10.0 + fraction * (-15.0 + 6.0 * fraction))
        self.data.ctrl[self.actuator_id] = self._start_m + (self.target_mm / 1000.0 - self._start_m) * smooth


class DragChain:
    """Deform the visual-only AL0700 drag-chain mesh in its compiler-centred frame without drift."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, AL0700_DRAG_CHAIN_MESH)
        if self.mesh_id < 0:
            raise ValueError("Missing AL0700 drag-chain mesh")
        start = int(model.mesh_vertadr[self.mesh_id])
        count = int(model.mesh_vertnum[self.mesh_id])
        self.vertex_slice = slice(start, start + count)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.mesh_quat[self.mesh_id])
        self.rotation = rotation.reshape(3, 3)
        self.offset = model.mesh_pos[self.mesh_id].copy()
        compiled = model.mesh_vert[self.vertex_slice].astype(float).copy()
        world = compiled @ self.rotation.T + self.offset
        self.original_cad = (world - AL0700_ORIGIN) @ AL0700_ROTATION
        self.geom_ids = np.flatnonzero(model.geom_dataid == self.mesh_id)
        if any(model.geom_contype[g] or model.geom_conaffinity[g] for g in self.geom_ids):
            raise ValueError("Drag-chain mesh must be visual only")
        self._position: float | None = None
        self.update(0.0)

    def update(self, position_mm: float) -> bool:
        """Update CPU vertices and culling bounds; return True if a GPU mesh upload is needed."""
        position = min(AL0700_STROKE_MM, max(0.0, _finite(position_mm, "position_mm")))
        if self._position == position:
            return False
        world = to_world(deform_drag_chain(self.original_cad, position / 1000.0))
        local = (world - self.offset) @ self.rotation
        self.model.mesh_vert[self.vertex_slice] = local
        lower, upper = local.min(axis=0), local.max(axis=0)
        rbound = float(np.linalg.norm(local, axis=1).max())
        center = (lower + upper) / 2.0
        half = (upper - lower) / 2.0
        for geom in self.geom_ids:
            self.model.geom_rbound[geom] = rbound
            self.model.geom_aabb[geom, :3] = center
            self.model.geom_aabb[geom, 3:] = half
        self._position = position
        return True

    def cad_vertices(self) -> np.ndarray:
        """Recover current drag-chain vertices in untransformed CAD coordinates (metres)."""
        vertices = self.model.mesh_vert[self.vertex_slice].astype(float)
        return (vertices @ self.rotation.T + self.offset - AL0700_ORIGIN) @ AL0700_ROTATION

    def upload(self, renderer: mujoco.Renderer) -> None:
        """Upload deformed vertices to an offscreen renderer's OpenGL context."""
        renderer._gl_context.make_current()
        mujoco.mjr_uploadMesh(self.model, renderer._mjr_context, self.mesh_id)


def _infer_model_name(source: str | Path | mujoco.MjSpec | mujoco.MjModel) -> str:
    if isinstance(source, mujoco.MjSpec):
        return source.modelname.removesuffix(" scene") or "al0700"
    if isinstance(source, mujoco.MjModel):
        return "al0700_model"
    path = Path(source)
    if path.is_file():
        return path.parent.name if path.name == "scene.xml" else path.stem
    return str(source)


class RailSimulation:
    """MuJoCo simulation world with named rail, arm, optional gripper, and optional F/T sensor."""

    def __init__(
        self,
        model: str | Path | mujoco.MjSpec | mujoco.MjModel | None = None,
        *,
        arm: str | None = None,
        gripper: str | None = None,
        ft_sensor: bool = False,
        camera: str | None = None,
        config: RailConfig | None = None,
    ):
        has_components = any((arm is not None, gripper is not None, ft_sensor, camera is not None))
        if model is not None and has_components:
            raise ValueError("Specify either 'model' or composition arguments, not both")
        self.config = RailConfig() if config is None else config

        if has_components:
            if arm is None:
                raise ValueError("'arm' is required when specifying gripper, ft_sensor, or camera")
            spec = compose_spec(
                arm=arm,
                linear_motor=True,
                gripper=gripper,
                ft_sensor=ft_sensor,
                camera=camera,
                scene=True,
            )
            self.name = spec.modelname.removesuffix(" scene")
            self.model = spec.compile()
        else:
            target: str | Path | mujoco.MjSpec | mujoco.MjModel = "al0700" if model is None else model
            self.name = _infer_model_name(target)
            self.model = load_model(target, scene=True)

        if not has_linear_motor(self.model):
            raise ValueError(f"Model {self.name!r} does not contain the AL0700 linear motor")

        self.data = mujoco.MjData(self.model)
        key = initial_keyframe(self.model)
        if key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.rail = RailController(self.model, self.data, self.config)
        self.chain = DragChain(self.model)

        self.arm_actuators: dict[str, int] = {}
        self.gripper_actuator_id: int | None = None
        for actuator in range(self.model.nu):
            if actuator == self.rail.actuator_id:
                continue
            act_name = self.model.actuator(actuator).name or ""
            if act_name in GRIPPER_ACTUATOR_NAMES:
                if self.gripper_actuator_id is not None:
                    raise ValueError("Multiple gripper actuators are not supported")
                self.gripper_actuator_id = actuator
                continue
            if int(self.model.actuator_trntype[actuator]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                raise ValueError(f"Unsupported non-joint actuator: {act_name!r}")
            joint_id = int(self.model.actuator_trnid[actuator, 0])
            joint_name = self.model.joint(joint_id).name or ""
            if not _ARM_JOINT_RE.fullmatch(joint_name):
                raise ValueError(f"Unexpected arm joint actuator target: {joint_name!r}")
            self.arm_actuators[joint_name] = actuator

        self.gripper_spec: GripperSpec | None = identify_gripper(self.model)
        self.ft_sensor: AI1500Reader | None = None
        self._ft_reading: SensorReading | None = None
        if has_ft_sensor(self.model):
            self.ft_sensor = AI1500Reader(self.model, self.data)
            self._ft_reading = self.ft_sensor.update()

    def __enter__(self) -> RailSimulation:
        return self

    def __exit__(self, *_exc) -> None:
        return None

    def set_arm_target_rad(self, joint: str, value: float) -> None:
        """Set one arm actuator target independently of the rail (radians)."""
        if joint not in self.arm_actuators:
            raise ValueError(f"Unknown arm joint: {joint}")
        angle = _finite(value, "joint target")
        actuator = self.arm_actuators[joint]
        low, high = self.model.actuator_ctrlrange[actuator]
        if not low <= angle <= high:
            raise ValueError(f"{joint} target outside [{low}, {high}] rad")
        self.data.ctrl[actuator] = angle

    def get_arm_positions_rad(self) -> dict[str, float]:
        """Return current arm joint angles keyed by joint name (radians)."""
        positions: dict[str, float] = {}
        for joint_name, actuator in self.arm_actuators.items():
            joint_id = int(self.model.actuator_trnid[actuator, 0])
            positions[joint_name] = float(self.data.qpos[int(self.model.jnt_qposadr[joint_id])])
        return positions

    def set_gripper_ctrl(self, value: float) -> None:
        """Set the mounted jaw gripper actuator command within its MJCF ctrlrange."""
        if self.gripper_actuator_id is None:
            raise ValueError("Model has no gripper actuator")
        target = _finite(value, "gripper ctrl")
        low, high = self.model.actuator_ctrlrange[self.gripper_actuator_id]
        if not low <= target <= high:
            raise ValueError(f"gripper ctrl outside [{low}, {high}]")
        self.data.ctrl[self.gripper_actuator_id] = target

    def get_ft_sensor_reading(self) -> SensorReading:
        """Return the latest 200 Hz AI1500 force/torque reading when an F/T sensor is mounted."""
        if self.ft_sensor is None or self._ft_reading is None:
            raise ValueError("Model has no AI1500 force/torque sensor")
        return self._ft_reading

    def step(self) -> None:
        """Update the rail command and step physics by one timestep."""
        self.rail.update()
        mujoco.mj_step(self.model, self.data)
        if not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
            raise RuntimeError("Non-finite simulation state")
        if any(warning.number for warning in self.data.warning):
            raise RuntimeError("MuJoCo reported a simulation warning")
        if self.ft_sensor is not None:
            last_time = self.ft_sensor._last_time
            needs_reset = (
                last_time is not None
                and abs(self.data.time - last_time - self.model.opt.timestep) > 1e-9
                and abs(self.data.time - last_time) > 1e-9
            )
            if needs_reset:
                self.ft_sensor.reset()
            self._ft_reading = self.ft_sensor.update()

    def refresh_visuals(self) -> bool:
        """Run forward kinematics and update the drag-chain mesh vertices for current rail travel."""
        mujoco.mj_forward(self.model, self.data)
        return self.chain.update(self.rail.get_position_mm())

    def move_to(
        self,
        target_mm: float,
        speed_mm_s: float | None = None,
        viewer: RailViewer | None = None,
    ) -> dict[str, float | bool]:
        """Run one bounded move to ``target_mm`` and return measured motion metrics."""
        return move_to_rail_position(self, target_mm, speed_mm_s=speed_mm_s, viewer=viewer)


class RailViewer(viewer_mod.PassiveViewer):
    """Passive MuJoCo viewer configured for AL0700 with thread-safe drag-chain mesh uploads."""

    def __init__(self, simulation: RailSimulation):
        viewer_mod.configure_backend()
        super().__init__(simulation.model, simulation.data)
        self._simulation = simulation
        with self.lock():
            self._handle.opt.geomgroup[3] = False

    def lock(self):
        """Return the MuJoCo passive viewer mutex context manager."""
        return self._handle.lock()

    def update_chain(self) -> None:
        """Deform CPU drag-chain vertices under lock, upload outside lock, and sync viewer."""
        with self.lock():
            changed = self._simulation.refresh_visuals()
        # update_mesh locks internally: calling it while holding lock deadlocks.
        if changed:
            self._handle.update_mesh(self._simulation.chain.mesh_id)
        self.sync()


def create_rail_viewer(simulation: RailSimulation) -> RailViewer:
    """Create a ``RailViewer`` bound to ``simulation``."""
    return RailViewer(simulation)


def move_to_rail_position(
    simulation: RailSimulation,
    target_mm: float,
    speed_mm_s: float | None = None,
    viewer: RailViewer | None = None,
) -> dict[str, float | bool]:
    """Run one bounded AL0700 move, validating settling, travel limits, force, and contacts."""
    rail = simulation.rail
    with viewer.lock() if viewer else nullcontext():
        rail.set_target_mm(target_mm, speed_mm_s)
    deadline = float(simulation.data.time) + rail.duration_s + rail.config.settle_timeout_s
    start_time = float(simulation.data.time)
    peak_force = 0.0
    peak_velocity = 0.0
    pos0 = rail.get_position_mm()
    min_position, max_position = pos0, pos0
    worst_contact = 0.0
    next_frame = float(simulation.data.time)
    start_wall = time.monotonic()

    while float(simulation.data.time) < deadline:
        if viewer and not viewer.is_running():
            raise InterruptedError("Viewer closed before move completed")
        with viewer.lock() if viewer else nullcontext():
            simulation.step()
            peak_force = max(peak_force, abs(float(simulation.data.actuator_force[rail.actuator_id])))
            peak_velocity = max(peak_velocity, abs(rail.get_velocity_mm_s()))
            position = rail.get_position_mm()
            min_position = min(min_position, position)
            max_position = max(max_position, position)
            for contact in simulation.data.contact[: simulation.data.ncon]:
                worst_contact = min(worst_contact, float(contact.dist))
        if viewer and float(simulation.data.time) >= next_frame:
            viewer.update_chain()
            next_frame = float(simulation.data.time) + 1.0 / 30.0
            remaining = start_wall + float(simulation.data.time) - start_time - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
        if rail.settled():
            break

    if viewer:
        viewer.update_chain()
    else:
        simulation.refresh_visuals()

    final_pos = rail.get_position_mm()
    report: dict[str, float | bool] = {
        "target_mm": float(rail.target_mm),
        "position_mm": final_pos,
        "error_mm": abs(final_pos - float(rail.target_mm)),
        "velocity_mm_s": rail.get_velocity_mm_s(),
        "duration_s": float(simulation.data.time) - start_time,
        "trajectory_s": rail.duration_s,
        "peak_force_n": peak_force,
        "min_position_mm": min_position,
        "max_position_mm": max_position,
        "peak_velocity_mm_s": peak_velocity,
        "worst_contact_m": worst_contact,
        "settled": rail.settled(),
    }
    if not report["settled"]:
        raise RuntimeError(f"Rail failed to settle within {rail.config.settle_timeout_s} s: {report}")
    if min_position < -0.1 or max_position > AL0700_STROKE_MM + 0.1:
        raise RuntimeError(f"Rail exceeded travel tolerance: {report}")
    if worst_contact < -0.00005:
        raise RuntimeError(f"Unexpected penetration: {report}")
    if peak_force > rail.config.max_force_n + 1e-6:
        raise RuntimeError(f"Rail exceeded force limit: {report}")
    return report
