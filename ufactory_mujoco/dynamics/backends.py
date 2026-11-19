"""Independent rigid-body dynamics reference backends (MuJoCo RNEA & Pinocchio)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .arm import ArmDynamics
from .profiles import RobotRuntimeProfile, canonical_robot_name, get_robot_runtime_profile
from .report import (
    POS_ERR_TOL,
    SATURATION_MARGIN,
    SETTLE_DURATION_S,
    VEL_TOL,
    SimDynamicsSample,
    _effort_limits,
)
from ..calibration import KinematicsCalibration, apply_calibration_to_model
from ..planning import CollisionChecker
from ..robots import MODELS_ROOT

DEFAULT_GRAVITY_VECTOR = (0.0, 0.0, -9.81)


class MujocoRneaReference:
    """Isolated zero-armature, zero-damping, zero-friction MuJoCo RNEA/CRBA oracle."""

    backend_name = "mujoco_rnea"

    def __init__(
        self,
        model_path: str | Path,
        *,
        calibration: KinematicsCalibration | None = None,
        gravity_vector: Sequence[float] | None = None,
    ) -> None:
        model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
        if calibration is not None:
            data = mujoco.MjData(model)
            apply_calibration_to_model(model, calibration, data)
            mujoco.mj_setConst(model, data)
            mujoco.mj_forward(model, data)
        model.dof_armature[:] = 0.0
        model.dof_damping[:] = 0.0
        model.dof_frictionloss[:] = 0.0
        g = np.asarray(
            gravity_vector if gravity_vector is not None else DEFAULT_GRAVITY_VECTOR,
            dtype=np.float64,
        )
        model.opt.gravity[:] = g
        self.dyn = ArmDynamics(model)

    def mass_matrix(self, q: Sequence[float]) -> np.ndarray:
        mat = np.asarray(self.dyn.mass_matrix(q), dtype=np.float64)
        return np.asarray((mat + mat.T) * 0.5, dtype=np.float64)

    def gravity(self, q: Sequence[float]) -> np.ndarray:
        return np.asarray(self.dyn.gravity(q), dtype=np.float64)

    def rnea(self, q: Sequence[float], qd: Sequence[float], qdd: Sequence[float]) -> np.ndarray:
        return np.asarray(self.dyn.inverse_dynamics(q, qd, qdd), dtype=np.float64)


class PinocchioReference:
    """Optional independent Pinocchio CRBA/RNEA reference backend from URDF."""

    backend_name = "pinocchio"

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        calibration: KinematicsCalibration | None = None,
        gravity_vector: Sequence[float] | None = None,
    ) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise ImportError(
                "Pinocchio reference backend is unavailable in this environment."
            ) from exc

        self.pin = pin
        resolved = Path(urdf_path).expanduser().resolve()
        if calibration is not None:
            xml_text = _patch_urdf_xml_with_calibration(resolved, calibration)
            self.model = pin.buildModelFromXML(xml_text)
        else:
            self.model = pin.buildModelFromUrdf(str(resolved))
        self.data = self.model.createData()
        g = np.asarray(
            gravity_vector if gravity_vector is not None else DEFAULT_GRAVITY_VECTOR,
            dtype=np.float64,
        )
        self.model.gravity.linear = g

    def mass_matrix(self, q: Sequence[float]) -> np.ndarray:
        q_np = np.asarray(q, dtype=np.float64)
        mat = self.pin.crba(self.model, self.data, q_np)
        return np.asarray((mat + mat.T) * 0.5, dtype=np.float64)

    def gravity(self, q: Sequence[float]) -> np.ndarray:
        q_np = np.asarray(q, dtype=np.float64)
        return np.asarray(self.pin.computeGeneralizedGravity(self.model, self.data, q_np), dtype=np.float64)

    def rnea(self, q: Sequence[float], qd: Sequence[float], qdd: Sequence[float]) -> np.ndarray:
        return np.asarray(
            self.pin.rnea(
                self.model,
                self.data,
                np.asarray(q, dtype=np.float64),
                np.asarray(qd, dtype=np.float64),
                np.asarray(qdd, dtype=np.float64),
            ),
            dtype=np.float64,
        )


def _patch_urdf_xml_with_calibration(urdf_path: Path, calibration: KinematicsCalibration) -> str:
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    for jname, cfg in calibration.joints.items():
        target = joints.get(jname)
        if target is None:
            raise ValueError(f"{urdf_path}: missing calibrated joint {jname}")
        origin = target.find("origin")
        if origin is None:
            origin = ET.Element("origin")
            target.insert(0, origin)
        origin.set("xyz", f"{cfg['x']} {cfg['y']} {cfg['z']}")
        origin.set("rpy", f"{cfg['roll']} {cfg['pitch']} {cfg['yaw']}")
    return ET.tostring(root, encoding="unicode")


def default_urdf_for_robot(robot_key: str) -> Path | None:
    """Locate sibling or UFACTORY_GENESIS_ROOT URDF if available on this machine."""
    name = canonical_robot_name(robot_key)
    env_root = (os.environ.get("UFACTORY_GENESIS_ROOT") or "").strip()
    genesis_root = (
        Path(env_root).expanduser()
        if env_root
        else Path(__file__).resolve().parents[3] / "ufactory_genesis"
    )
    urdf_name = f"{name}_1305.urdf" if name in {"xarm5", "xarm6", "xarm7"} else f"{name}.urdf"
    candidate = genesis_root / "assets" / "urdf" / name / urdf_name
    return candidate if candidate.is_file() else None


def load_reference_backend(
    robot_key: str,
    *,
    backend: str = "auto",
    model_path: str | Path | None = None,
    urdf_path: str | Path | None = None,
    calibration: KinematicsCalibration | None = None,
    required: bool = False,
    gravity_vector: Sequence[float] | None = None,
) -> MujocoRneaReference | PinocchioReference | None:
    """Load the requested independent reference dynamics backend."""
    choice = (backend or "auto").strip().lower()
    if choice == "none":
        return None

    name = canonical_robot_name(robot_key)
    resolved_model = Path(model_path) if model_path is not None else MODELS_ROOT / name / f"{name}.xml"

    if choice == "auto":
        if urdf_path is not None:
            choice = "pinocchio"
        else:
            choice = "mujoco-rnea"

    if choice == "mujoco-rnea":
        return MujocoRneaReference(
            resolved_model,
            calibration=calibration,
            gravity_vector=gravity_vector,
        )

    if choice == "pinocchio":
        resolved_urdf = Path(urdf_path) if urdf_path is not None else default_urdf_for_robot(name)
        if resolved_urdf is None or not resolved_urdf.is_file():
            if required:
                raise FileNotFoundError(f"URDF not found for Pinocchio reference backend ({name})")
            return None
        try:
            return PinocchioReference(
                resolved_urdf,
                calibration=calibration,
                gravity_vector=gravity_vector,
            )
        except ImportError:
            if required:
                raise
            return None

    raise ValueError(f"Unknown reference backend: {backend!r}")


def armature_aligned_mass(mass_sim: np.ndarray, armature: Sequence[float] | np.ndarray | None) -> np.ndarray:
    """Return ``mass_sim`` with reflected rotor armature ``diag(armature)`` removed."""
    m = np.asarray(mass_sim, dtype=np.float64).copy()
    if armature is None:
        return m
    arm = np.asarray(armature, dtype=np.float64).reshape(-1)
    k = min(len(arm), m.shape[0])
    if k > 0:
        m[:k, :k] -= np.diag(arm[:k])
    return m


# ----------------------------------------------------------------------
# MuJoCo probes
# ----------------------------------------------------------------------

@dataclass
class MujocoProbeContext:
    """Isolated MuJoCo simulation and collision-check context for dynamics probes."""

    scene_path: Path
    model: mujoco.MjModel
    data: mujoco.MjData
    mass_data: mujoco.MjData
    fk_data: mujoco.MjData
    checker: CollisionChecker
    dof: int
    ee_site_id: int
    runtime_profile: RobotRuntimeProfile


def build_mujoco_scene(
    robot_key: str = "xarm6",
    *,
    runtime_profile: RobotRuntimeProfile | None = None,
    scene_path: str | Path | None = None,
    calibration: KinematicsCalibration | None = None,
    zero_joint_frictionloss: bool = False,
) -> MujocoProbeContext:
    """Build a MuJoCo scene + isolated CollisionChecker for closed-loop PD hold probes."""
    runtime = runtime_profile or get_robot_runtime_profile(robot_key)
    resolved_scene = Path(scene_path) if scene_path is not None else runtime.model.scene_path
    model = mujoco.MjModel.from_xml_path(str(resolved_scene))
    data = mujoco.MjData(model)
    mass_data = mujoco.MjData(model)
    fk_data = mujoco.MjData(model)
    checker = CollisionChecker(resolved_scene, ignore_pairs=[("link_base", "world")])

    if calibration is not None:
        for m, d in ((model, data), (checker.model, checker.data)):
            apply_calibration_to_model(m, calibration, d)
            mujoco.mj_setConst(m, d)
            mujoco.mj_forward(m, d)
        mujoco.mj_setConst(model, mass_data)
        mujoco.mj_setConst(model, fk_data)

    dof = runtime.model.dof
    if zero_joint_frictionloss:
        model.dof_frictionloss[:dof] = 0.0

    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, runtime.arm.ee_site)
    if ee_site_id < 0:
        raise KeyError(f"EE site not found: {runtime.arm.ee_site}")

    return MujocoProbeContext(
        scene_path=resolved_scene,
        model=model,
        data=data,
        mass_data=mass_data,
        fk_data=fk_data,
        checker=checker,
        dof=dof,
        ee_site_id=ee_site_id,
        runtime_profile=runtime,
    )


def read_joint_frictionloss(ctx: MujocoProbeContext) -> np.ndarray:
    """Read per-joint frictionloss (Nm) applied in the MuJoCo probe model."""
    return np.asarray(ctx.model.dof_frictionloss[: ctx.dof], dtype=np.float64).copy()


def capture_self_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    min_force_n: float = 1e-6,
) -> list[dict[str, Any]]:
    """Return active non-base-floor contacts at the settled state for diagnostics."""
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_base")
    ignore_pair = frozenset((0, base_id)) if base_id >= 0 else None
    out: list[dict[str, Any]] = []
    force_buf = np.zeros(6, dtype=np.float64)
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = int(model.geom_bodyid[c.geom[0]])
        b2 = int(model.geom_bodyid[c.geom[1]])
        if ignore_pair is not None and frozenset((b1, b2)) == ignore_pair:
            continue
        mujoco.mj_contactForce(model, data, i, force_buf)
        force_mag = float(np.linalg.norm(force_buf[:3]))
        if force_mag < min_force_n and float(c.dist) >= 0.0:
            continue
        out.append(
            {
                "body_a": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or str(b1),
                "body_b": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or str(b2),
                "dist_m": float(c.dist),
                "force_n": force_mag,
            }
        )
    return out


def _compute_mass_matrix_at_q(ctx: MujocoProbeContext, q: np.ndarray) -> np.ndarray:
    d = ctx.mass_data
    d.qpos[: ctx.dof] = q
    d.qvel[:] = 0.0
    d.qacc[:] = 0.0
    mujoco.mj_kinematics(ctx.model, d)
    mujoco.mj_comPos(ctx.model, d)
    mujoco.mj_crb(ctx.model, d)
    full_m = np.zeros((ctx.model.nv, ctx.model.nv), dtype=np.float64)
    mujoco.mj_fullM(ctx.model, d, full_m)
    return full_m[: ctx.dof, : ctx.dof].copy()


def mujoco_pd_hold_torque_at_q(
    ctx: MujocoProbeContext,
    target_q: Sequence[float],
    *,
    runtime_profile: RobotRuntimeProfile | None = None,
    settle_duration_s: float = SETTLE_DURATION_S,
    settle_steps: int | None = None,
    pos_tol: float = POS_ERR_TOL,
    vel_tol: float = VEL_TOL,
    effort_limits: np.ndarray | None = None,
    init_at_target: bool = False,
    capture_contacts: bool = False,
) -> SimDynamicsSample:
    """Hold ``target_q`` under closed-loop MuJoCo PD control and sample physics state."""
    runtime = runtime_profile or ctx.runtime_profile
    dof = ctx.dof
    target = np.asarray(target_q, dtype=np.float64).reshape(dof)
    steps = (
        int(settle_steps)
        if settle_steps is not None
        else max(1, int(round(settle_duration_s / float(ctx.model.opt.timestep))))
    )

    if init_at_target:
        mujoco.mj_resetData(ctx.model, ctx.data)
        ctx.data.qpos[:dof] = target
        ctx.data.qvel[:] = 0.0
        mujoco.mj_forward(ctx.model, ctx.data)

    ctx.data.ctrl[:dof] = target
    for _ in range(steps):
        mujoco.mj_step(ctx.model, ctx.data)

    qpos = ctx.data.qpos[:dof].astype(np.float64).copy()
    qvel = ctx.data.qvel[:dof].astype(np.float64).copy()
    pd_hold_tau = ctx.data.qfrc_actuator[:dof].astype(np.float64).copy()
    actual_dof_force = (
        ctx.data.qfrc_actuator[:dof] + ctx.data.qfrc_passive[:dof] + ctx.data.qfrc_constraint[:dof]
    ).astype(np.float64).copy()
    mass_matrix = _compute_mass_matrix_at_q(ctx, qpos)
    armature = ctx.model.dof_armature[:dof].astype(np.float64).copy()
    frictionloss = ctx.model.dof_frictionloss[:dof].astype(np.float64).copy()

    pos_err = float(np.abs(qpos - target).max())
    vel_mag = float(np.abs(qvel).max())
    settled = pos_err <= pos_tol and vel_mag <= vel_tol
    limits = effort_limits if effort_limits is not None else _effort_limits(runtime)
    saturated = bool(np.any(np.abs(pd_hold_tau) >= np.asarray(limits, dtype=np.float64) * SATURATION_MARGIN))
    contacts = capture_self_contacts(ctx.model, ctx.data) if capture_contacts else None

    return SimDynamicsSample(
        q_actual=qpos,
        qvel=qvel,
        pd_hold_tau=pd_hold_tau,
        actual_dof_force=actual_dof_force,
        mass_matrix=mass_matrix,
        settled=settled,
        saturated=saturated,
        pos_err=pos_err,
        vel_mag=vel_mag,
        armature=armature,
        joint_frictionloss=frictionloss,
        self_contacts=contacts,
    )


def mujoco_ee_xyz_mm_at_q(ctx: MujocoProbeContext, q: Sequence[float]) -> tuple[float, float, float]:
    """Return flange ``attachment_site`` world (x, y, z) in mm at ``q``."""
    ctx.fk_data.qpos[: ctx.dof] = np.asarray(q, dtype=np.float64).reshape(ctx.dof)
    ctx.fk_data.qvel[:] = 0.0
    mujoco.mj_kinematics(ctx.model, ctx.fk_data)
    pos = ctx.fk_data.site_xpos[ctx.ee_site_id]
    return float(pos[0] * 1000.0), float(pos[1] * 1000.0), float(pos[2] * 1000.0)


def mujoco_ee_z_mm_at_q(ctx: MujocoProbeContext, q: Sequence[float]) -> float:
    """Return flange ``attachment_site`` world Z in mm at ``q``."""
    return mujoco_ee_xyz_mm_at_q(ctx, q)[2]


def compute_ee_z_table_from_sim(
    ctx: MujocoProbeContext,
    configs: Sequence[tuple[str, np.ndarray]],
) -> dict[str, float]:
    """Compute world flange Z (mm) for all named configurations."""
    return {name: mujoco_ee_z_mm_at_q(ctx, q) for name, q in configs}


def find_colliding_configs(
    ctx: MujocoProbeContext,
    configs: Sequence[tuple[str, np.ndarray]],
) -> set[str]:
    """Identify pose names with self-collision or floor penetration in MuJoCo."""
    return {name for name, q in configs if ctx.checker.in_collision(np.asarray(q, dtype=np.float64))}


def check_mujoco_path_safety(
    ctx: MujocoProbeContext,
    start_q: Sequence[float],
    target_q: Sequence[float],
    *,
    z_min_mm: float,
    steps: int = 25,
) -> list[str]:
    """Check interpolated joint path for minimum flange Z and MuJoCo collisions."""
    start = np.asarray(start_q, dtype=np.float64)
    target = np.asarray(target_q, dtype=np.float64)
    reasons: list[str] = []
    for i in range(steps + 1):
        alpha = i / steps
        q = (1.0 - alpha) * start + alpha * target
        z = mujoco_ee_z_mm_at_q(ctx, q)
        if z < z_min_mm:
            reasons.append(f"MuJoCo path step {i}: EE z {z:.2f} mm < z_min {z_min_mm:.2f} mm")
            break
        if ctx.checker.in_collision(q):
            min_d = ctx.checker.min_distance(q)
            reasons.append(f"MuJoCo path step {i}: collision detected (min_dist={min_d:.4f} m)")
            break
    return reasons
