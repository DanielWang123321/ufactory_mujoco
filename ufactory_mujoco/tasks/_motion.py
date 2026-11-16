"""Motion helpers shared by the pick-and-place and box-packing tasks.

Callers pass already-scaled velocity and acceleration limits, so each task keeps
its own speed policy (box-pack applies ``speed_scale``; pick-place does not).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..config import load_task_config
from ..task_geometry import GRIP_DOWN_QUAT
from ..trajectory import JointTrajectory, plan_joint_waypoints


def set_cube(robot, joint_name: str, xy, scene_kind: str) -> None:
    """Place a free-joint cube at ``xy`` on its authored height, upright and at rest."""
    joint_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Scene has no {joint_name!r} joint; load a {scene_kind} scene")
    adr = int(robot.model.jnt_qposadr[joint_id])
    # Keep the authored height so a raised cube stays on its support.
    authored_z = float(robot.model.qpos0[adr + 2])
    robot.data.qpos[adr:adr + 3] = (xy[0], xy[1], authored_z)
    robot.data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
    dof = int(robot.model.jnt_dofadr[joint_id])
    robot.data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(robot.model, robot.data)


@dataclass(frozen=True)
class Mover:
    """Executor and already-scaled Cartesian/detour limits for one task run."""

    robot: Any
    executor: str
    max_lin_vel: float
    max_lin_acc: float
    detour_max_vel: float
    detour_max_acc: float

    def duration(self, dist: float) -> float:
        """Trapezoidal segment time for distance ``dist`` under the Cartesian limits."""
        return dist / self.max_lin_vel + self.max_lin_vel / self.max_lin_acc + 0.1

    def move(self, pos: np.ndarray, duration: float, q_seed: np.ndarray) -> np.ndarray:
        """Move the TCP to ``pos`` with a grip-down orientation and return the resulting joint seed.

        ``servo_j`` plans a Cartesian line in joint space and falls back to an IK joint move;
        ``servo_cartesian`` tracks TCP waypoints directly.
        """
        robot = self.robot
        if self.executor == "servo_j":
            traj, ok = robot.plan_cartesian_trajectory(
                pos, GRIP_DOWN_QUAT, max_lin_vel=self.max_lin_vel, max_lin_acc=self.max_lin_acc)
            if ok:
                n = robot.n_arm
                robot.execute_trajectory(JointTrajectory(
                    traj.times, traj.q[:, :n], traj.qd[:, :n], traj.qdd[:, :n]))
                return robot.data.qpos.copy()
            q, ok = robot.solve_tcp(pos, GRIP_DOWN_QUAT, q_init=q_seed)
            if not ok:
                raise RuntimeError(f"IK failed for joint move to {pos}")
            robot.move_joints(q[:robot.n_arm], duration)
            return q
        robot.move_tcp(pos, GRIP_DOWN_QUAT, duration=duration)
        return robot.data.qpos.copy()

    def segment(self, name: str, pos: np.ndarray, duration: float, q_seed: np.ndarray,
                report) -> np.ndarray:
        """Run preflight segment ``name``: a joint-space detour when preflight planned one, else ``move``."""
        segment = report.segments.get(name)
        if segment is not None and segment.detour:
            traj = plan_joint_waypoints(
                segment.waypoints, max_vel=self.detour_max_vel, max_acc=self.detour_max_acc,
                dt=self.robot.model.opt.timestep)
            self.robot.execute_trajectory(traj)
            return self.robot.data.qpos.copy()
        return self.move(pos, duration, q_seed)


def print_phases(report) -> None:
    """Print each preflight phase status and its failure reason."""
    print("Preflight:")
    for phase in report.phases:
        reason = f" ({phase.reason})" if phase.reason else ""
        print(f"  {phase.name}: {phase.status}{reason}")


def load_overlay(configs_dir: Path, robot: str, gripper: str | None, executor: str | None,
                 config: str | None, extra: dict[str, Any] | None = None):
    """Load ``config`` (or ``configs_dir/<robot>.yaml``) with optional CLI overrides."""
    overlay = config or (configs_dir / f"{robot}.yaml")
    overrides: dict[str, Any] = {}
    if gripper:
        overrides["gripper"] = gripper
    if executor:
        overrides["executor"] = executor
    if extra:
        overrides.update(extra)
    return load_task_config(overlay, overrides or None)
