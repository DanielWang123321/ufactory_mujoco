"""Independent rigid-body dynamics reference backends (MuJoCo RNEA & Pinocchio)."""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .arm import ArmDynamics
from .poses import canonical_robot_name
from ..calibration import KinematicsCalibration, apply_calibration_to_model
from ..model_paths import MODELS_ROOT

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
