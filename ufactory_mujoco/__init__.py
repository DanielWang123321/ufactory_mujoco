"""MuJoCo simulation suite for UFACTORY arms, grippers and depth cameras.

The top level exports the user-facing API. Lower-level helpers stay importable
from their submodules (for example ``ufactory_mujoco.preflight``).
"""

from .viewer import configure_backend

# MuJoCo imports GLFW while importing its rendering modules.
configure_backend()

from .bio_g2_controller import BioGripperG2Controller
from .calibration import CalibrationError, KinematicsCalibration, load_kinematics_calibration
from .collision import CollisionChecker
from .config import ConfigError, TaskConfig, load_task_config
from .dynamics import ArmDynamics, attach_payload
from .ft_sensor import AI1500Reader, FTSensorReader, SensorReading, apply_ft_sensor_wrench
from .g2_controller import G2Controller
from .grippers import GraspError, GripperSpec, identify_gripper
from .ik import solve_ik
from .kinematics import fk_pose, manipulability, pose_jacobian, site_jacobian, solve_joint_velocity
from .model_paths import MODELS_ROOT, resolve_model_path
from .path_planning import PlanningError, plan_collision_free, shortcut_path
from .robot_interface import RobotInterface
from .robots import ARMS, SCENES, ArmSpec, resolve_arm_scene
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

__all__ = [
    # robots and models
    "SimRobot",
    "XArmRobot",
    "RobotInterface",
    "ARMS",
    "ArmSpec",
    "SCENES",
    "resolve_arm_scene",
    "MODELS_ROOT",
    "resolve_model_path",
    # kinematics
    "fk_pose",
    "solve_ik",
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
    # collision-free planning
    "CollisionChecker",
    "plan_collision_free",
    "shortcut_path",
    "PlanningError",
    # grippers
    "G2Controller",
    "BioGripperG2Controller",
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
