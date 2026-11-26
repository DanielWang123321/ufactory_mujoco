"""MuJoCo simulation suite for UFACTORY arms, grippers and depth cameras.

The top level exports the user-facing API. Lower-level helpers stay importable
from their submodules (for example ``ufactory_mujoco.preflight``).
"""

from .viewer import configure_backend

# MuJoCo imports GLFW while importing its rendering modules.
configure_backend()

from .bio_g2 import BioGripperG2Controller
from .calibration import CalibrationError, KinematicsCalibration, load_kinematics_calibration
from .compose import (
    COMPOSED_MODELS,
    available_models,
    compose_model,
    compose_spec,
    export_model,
    load_model,
    resolve_model_path,
)
from .config import ConfigError, TaskConfig, load_task_config, resolve_arm_scene
from .dynamics import ArmDynamics, attach_payload
from .ft_sensor import (
    AI1500Reader,
    FTSensorReader,
    SensorReading,
    apply_ft_sensor_wrench,
    clear_ft_sensor_wrench,
    is_hardware_ft_sensor_enabled,
    read_hardware_ft_sensor,
    read_hardware_ft_sensor_config,
)
from .g2 import G2Controller
from .grippers import GraspError, GripperSpec, identify_gripper
from .kinematics import (
    fk_pose,
    manipulability,
    pose_jacobian,
    quat_error,
    site_jacobian,
    solve_ik,
    solve_joint_velocity,
)
from .planning import CollisionChecker, PlanningError, plan_collision_free, shortcut_path
from .robot_interface import RobotInterface
from .robotiq import RobotiqController
from .robots import ARMS, MODELS_ROOT, SCENES, ArmSpec
from .safety import MotionApproval, SafetyGateError
from .sim_robot import SimRobot
from .trajectory import (
    JointTrajectory,
    plan_cartesian_line,
    plan_joint_waypoints,
    plan_ptp_lspb,
    plan_ptp_quintic,
)
from .xarm_robot import XArmRobot

__version__ = "0.1.9"

__all__ = [
    "__version__",
    # robots, models and runtime composition
    "SimRobot",
    "XArmRobot",
    "RobotInterface",
    "ARMS",
    "ArmSpec",
    "SCENES",
    "resolve_arm_scene",
    "MODELS_ROOT",
    "resolve_model_path",
    "load_model",
    "available_models",
    "COMPOSED_MODELS",
    "compose_spec",
    "compose_model",
    "export_model",
    # kinematics
    "fk_pose",
    "solve_ik",
    "quat_error",
    "site_jacobian",
    "pose_jacobian",
    "solve_joint_velocity",
    "manipulability",
    # trajectories
    "JointTrajectory",
    "plan_ptp_quintic",
    "plan_ptp_lspb",
    "plan_joint_waypoints",
    "plan_cartesian_line",
    # dynamics
    "ArmDynamics",
    "attach_payload",
    # force/torque sensor
    "AI1500Reader",
    "FTSensorReader",
    "SensorReading",
    "apply_ft_sensor_wrench",
    "clear_ft_sensor_wrench",
    "is_hardware_ft_sensor_enabled",
    "read_hardware_ft_sensor",
    "read_hardware_ft_sensor_config",
    # collision-free planning
    "CollisionChecker",
    "plan_collision_free",
    "shortcut_path",
    "PlanningError",
    # grippers
    "G2Controller",
    "BioGripperG2Controller",
    "RobotiqController",
    "GripperSpec",
    "identify_gripper",
    "GraspError",
    # task configuration
    "TaskConfig",
    "load_task_config",
    "ConfigError",
    # hardware
    "MotionApproval",
    "SafetyGateError",
    "KinematicsCalibration",
    "load_kinematics_calibration",
    "CalibrationError",
]
