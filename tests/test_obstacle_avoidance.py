"""Integration tests for the obstacle-avoidance example (roadmap 0.0.6 M3).

Scene: models/xarm6/scene_obstacles.xml (pillar + board). The straight joint
interpolation between keyframes 'start' and 'goal' penetrates the pillar, so a
valid plan must detour; execution is monitored per physics step for
penetration (repository convention: <= 1 micrometre).
"""

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from ufactory_mujoco.collision import CollisionChecker
from ufactory_mujoco.path_planning import plan_collision_free, shortcut_path
from ufactory_mujoco.sim_robot import SimRobot

avoidance = importlib.import_module("examples.simulation.09_obstacle_avoidance")

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "models" / "xarm6" / "scene_obstacles.xml"


class ObstacleSceneTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.checker = CollisionChecker(SCENE, safety_margin=0.0)

    def test_obstacles_are_physical(self):
        for name in ("pillar_geom", "board_geom"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            self.assertGreaterEqual(gid, 0, name)
            self.assertNotEqual(int(self.model.geom_contype[gid]), 0)
            self.assertNotEqual(int(self.model.geom_conaffinity[gid]), 0)

    def test_keyframes_exist_and_are_free(self):
        q0 = avoidance.key_qpos(self.model, "start")
        q1 = avoidance.key_qpos(self.model, "goal")
        self.assertEqual(q0.shape, (6,))
        self.assertFalse(self.checker.in_collision(q0))
        self.assertFalse(self.checker.in_collision(q1))

    def test_straight_line_between_keyframes_is_blocked(self):
        q0 = avoidance.key_qpos(self.model, "start")
        q1 = avoidance.key_qpos(self.model, "goal")
        hits, min_dist = avoidance._straight_baseline(self.checker, q0, q1)
        self.assertGreater(hits, 0, "fixture broken: straight line no longer blocked")
        self.assertLess(min_dist, 0.0)


class AvoidancePlanningTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.checker = CollisionChecker(SCENE, safety_margin=0.002)
        self.q0 = avoidance.key_qpos(self.model, "start")
        self.q1 = avoidance.key_qpos(self.model, "goal")

    def test_detour_waypoints_are_collision_free(self):
        path = plan_collision_free(self.q0, self.q1, checker=self.checker,
                                   seed=0, step=0.15)
        path = shortcut_path(self.checker, path, seed=0)
        np.testing.assert_allclose(path[0], self.q0)
        np.testing.assert_allclose(path[-1], self.q1)
        self.assertGreaterEqual(len(path), 3)
        for qa, qb in zip(path[:-1], path[1:]):
            for t in np.linspace(0.0, 1.0, 21):
                self.assertFalse(self.checker.in_collision(qa + t * (qb - qa)))

    def test_timed_trajectory_stays_on_valid_states(self):
        path = plan_collision_free(self.q0, self.q1, checker=self.checker,
                                   seed=0, step=0.15)
        path = shortcut_path(self.checker, path, seed=0)
        traj, method = avoidance._timed_trajectory(self.checker, path)
        self.assertIn(method, ("spline", "stop"))
        self.assertGreater(traj.duration, 0.0)
        self.assertFalse(any(self.checker.in_collision(q) for q in traj.q))


class AvoidanceExecutionTests(unittest.TestCase):
    def test_end_to_end_execution_without_penetration(self):
        checker = CollisionChecker(SCENE, safety_margin=0.002)
        with SimRobot(SCENE, viewer=False) as robot:
            metrics = avoidance.run_avoidance(robot, checker, seed=0)

        self.assertTrue(metrics["straight_blocked"])
        self.assertGreater(metrics["min_clearance_m"], 0.0)
        self.assertLessEqual(metrics["max_penetration_m"], 1e-6)
        self.assertLess(metrics["final_error_rad"], 0.01)
        self.assertTrue(metrics["success"])

    def test_second_seed_also_succeeds(self):
        checker = CollisionChecker(SCENE, safety_margin=0.002)
        with SimRobot(SCENE, viewer=False) as robot:
            metrics = avoidance.run_avoidance(robot, checker, seed=1)
        self.assertTrue(metrics["success"], metrics)


class AvoidanceExampleSmokeTests(unittest.TestCase):
    def test_headless_main_passes(self):
        with patch.object(sys, "argv", ["09_obstacle_avoidance.py", "--headless"]):
            with self.assertRaises(SystemExit) as ctx:
                avoidance.main()
        self.assertEqual(ctx.exception.code, 0, "example reported FAIL")


if __name__ == "__main__":
    unittest.main()
