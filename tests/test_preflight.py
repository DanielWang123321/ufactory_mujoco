"""Pick-and-place preflight: limits, penetration, and obstacle detours."""

import unittest
from pathlib import Path

import mujoco
import numpy as np

from ufactory_mujoco.collision import CollisionChecker
from ufactory_mujoco.config import load_task_config
from ufactory_mujoco.preflight import PHASES, make_checker, plan_segment, preflight_pick_place
from ufactory_mujoco.sim_robot import SimRobot

ROOT = Path(__file__).resolve().parent.parent
OBSTACLES = ROOT / "models" / "xarm6" / "scene_obstacles.xml"
XARM6 = ROOT / "configs" / "pick_place" / "xarm6.yaml"


def _key(model, name):
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    return model.key(key_id).qpos.copy()


class PreflightTests(unittest.TestCase):
    def test_joint_limit_fails_without_stepping(self):
        config = load_task_config(XARM6)
        with SimRobot(config.scene_path, viewer=False) as robot:
            checker = make_checker(config.scene_path)
            q0 = robot.get_joint_positions()
            q1 = checker.q_hi + 0.2
            _, reason = plan_segment(q0, q1, checker)
            self.assertEqual(reason, "joint_limit")
            self.assertEqual(robot.data.time, 0.0)

    def test_penetrating_goal_fails(self):
        checker = CollisionChecker(OBSTACLES, safety_margin=0.0, query_margin=0.02)
        model = checker.model
        start = _key(model, "start")
        goal = _key(model, "goal")
        blocked = None
        for t in np.linspace(0.0, 1.0, 21):
            q = (1.0 - t) * start + t * goal
            if checker.in_collision(q):
                blocked = q
                break
        self.assertIsNotNone(blocked)
        _, reason = plan_segment(blocked, blocked, checker)
        self.assertEqual(reason, "collision")

    def test_obstacle_segment_detours(self):
        checker = CollisionChecker(OBSTACLES, safety_margin=0.0, query_margin=0.02)
        model = checker.model
        start = _key(model, "start")
        goal = _key(model, "goal")
        path, reason = plan_segment(start, goal, checker, seed=0)
        self.assertIsNone(reason)
        self.assertGreater(path.shape[0], 2)
        for q in path:
            self.assertFalse(checker.in_collision(q))

    def test_free_segment_is_the_endpoints(self):
        config = load_task_config(XARM6)
        checker = make_checker(config.scene_path)
        with SimRobot(config.scene_path, viewer=False) as robot:
            q0 = robot.get_joint_positions()
        q1 = q0.copy()
        q1[5] = np.clip(q1[5] + 0.05, checker.q_lo[5], checker.q_hi[5])
        path, reason = plan_segment(q0, q1, checker)
        self.assertIsNone(reason)
        self.assertEqual(path.shape[0], 2)
        np.testing.assert_allclose(path[0], q0)
        np.testing.assert_allclose(path[1], q1)

    def test_xarm6_dry_run_phases(self):
        import importlib
        pick_place = importlib.import_module("examples.simulation.05_pick_place")
        config = load_task_config(XARM6)
        with SimRobot(config.scene_path, viewer=False) as robot:
            report = pick_place.run_pick_place(robot, config, dry_run=True)
            self.assertEqual(robot.data.time, 0.0)
        self.assertTrue(report.ok)
        self.assertEqual([phase.name for phase in report.phases], list(PHASES))
        self.assertTrue(all(phase.status == "ok" for phase in report.phases))
        self.assertTrue(all(not segment.detour for segment in report.segments.values()))


if __name__ == "__main__":
    unittest.main()
