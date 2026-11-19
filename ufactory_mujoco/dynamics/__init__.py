"""Rigid-body dynamics and ufactory_genesis-aligned validation types."""

from .arm import ArmDynamics, attach_payload
from .report import (
    REPORT_SCHEMA_VERSION,
    DynamicsRunConfig,
    DynamicsSample,
    GenesisDynamicsSample,
    ModelDynamicsIssue,
    SafePose,
    SimDynamicsSample,
    UrdfDynamicsIssue,
    ValidationStatus,
)
from .profiles import (
    DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S,
    MAX_DYNAMICS_MOVE_SPEED_DEG_S,
    MAX_DYNAMICS_MOVE_SPEED_RAD_S,
    validate_dynamics_move_speed,
)

__all__ = [
    "ArmDynamics",
    "attach_payload",
    "DEFAULT_DYNAMICS_MOVE_SPEED_RAD_S",
    "MAX_DYNAMICS_MOVE_SPEED_DEG_S",
    "MAX_DYNAMICS_MOVE_SPEED_RAD_S",
    "REPORT_SCHEMA_VERSION",
    "DynamicsRunConfig",
    "DynamicsSample",
    "GenesisDynamicsSample",
    "ModelDynamicsIssue",
    "SafePose",
    "SimDynamicsSample",
    "UrdfDynamicsIssue",
    "ValidationStatus",
    "validate_dynamics_move_speed",
]
