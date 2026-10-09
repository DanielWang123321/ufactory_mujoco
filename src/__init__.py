from .viewer import configure_backend

# MuJoCo imports GLFW while importing its rendering modules.
configure_backend()

from .robot_interface import RobotInterface
from .sim_robot import SimRobot

__all__ = ["RobotInterface", "SimRobot"]
