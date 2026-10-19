"""Unit tests for RRT-Connect planning and shortcutting (roadmap 0.0.6 M2).

Fixture: tests/fixtures/planar_arm.xml. The straight joint interpolation
between FREE_A=[1.2,-0.6] and FREE_B=[-1.2,0.6] passes through q=[0,0], which
drives the links through the disk, so a valid plan must detour in joint space.
"""

import unittest
from pathlib import Path

import numpy as np

from ufactory_mujoco.collision import CollisionChecker
from ufactory_mujoco.path_planning import (
    PlanningError,
    path_length,
    plan_collision_free,
    shortcut_path,
)
from ufactory_mujoco.trajectory import plan_joint_waypoints

ROOT = Path(__file__).resolve().parent.parent
PLANAR_XML = ROOT / "tests" / "fixtures" / "planar_arm.xml"

FREE_A = np.array([1.2, -0.6])
FREE_B = np.array([-1.2, 0.6])
FREE_C = np.array([2.6, 0.5])
COLLIDING = np.array([0.0, 0.0])


def _segments_free(checker: CollisionChecker, waypoints: np.ndarray,
                   res: float = 0.02) -> bool:
    for qa, qb in zip(waypoints[:-1], waypoints[1:]):
        n = max(int(np.ceil(np.linalg.norm(qb - qa) / res)), 1)
        for t in np.linspace(0.0, 1.0, n + 1):
            if checker.in_collision(qa + t * (qb - qa)):
                return False
    return True


class StraightLineFastPathTests(unittest.TestCase):
    def setUp(self):
        self.checker = CollisionChecker(PLANAR_XML)

    def test_direct_segment_is_used_when_free(self):
        path = plan_collision_free(FREE_A, FREE_C, checker=self.checker, seed=0)
        self.assertEqual(path.shape, (2, 2))
        np.testing.assert_allclose(path[0], FREE_A)
        np.testing.assert_allclose(path[1], FREE_C)

    def test_fixture_endpoints_are_blocked_in_a_straight_line(self):
        self.assertTrue(self.checker.in_collision((FREE_A + FREE_B) / 2.0))


class RRTConnectTests(unittest.TestCase):
    def setUp(self):
        self.checker = CollisionChecker(PLANAR_XML)

    def test_detour_found_around_obstacle(self):
        path = plan_collision_free(FREE_A, FREE_B, checker=self.checker,
                                   seed=0, step=0.2)
        self.assertGreaterEqual(path.shape[0], 3)  # straight line is impossible
        np.testing.assert_allclose(path[0], FREE_A)
        np.testing.assert_allclose(path[-1], FREE_B)
        self.assertTrue(_segments_free(self.checker, path))
        self.assertTrue(np.all(path >= self.checker.q_lo - 1e-9))
        self.assertTrue(np.all(path <= self.checker.q_hi + 1e-9))

    def test_reproducible_with_seed(self):
        p1 = plan_collision_free(FREE_A, FREE_B, checker=self.checker, seed=7, step=0.2)
        p2 = plan_collision_free(FREE_A, FREE_B, checker=self.checker, seed=7, step=0.2)
        np.testing.assert_array_equal(p1, p2)

    def test_goal_in_collision_raises(self):
        with self.assertRaises(PlanningError):
            plan_collision_free(FREE_A, COLLIDING, checker=self.checker, seed=0)

    def test_start_in_collision_raises(self):
        with self.assertRaises(PlanningError):
            plan_collision_free(COLLIDING, FREE_B, checker=self.checker, seed=0)

    def test_iteration_budget_exhaustion_raises(self):
        with self.assertRaises(PlanningError):
            plan_collision_free(FREE_A, FREE_B, checker=self.checker,
                                seed=0, step=0.2, max_iter=1)

    def test_parameter_validation(self):
        for kwargs in ({"step": 0.0}, {"max_iter": 0}, {"goal_bias": 1.5},
                       {"check_res": 0.0}):
            with self.assertRaises(ValueError):
                plan_collision_free(FREE_A, FREE_B, checker=self.checker, **kwargs)
        with self.assertRaises(ValueError):
            plan_collision_free(np.zeros(3), FREE_B, checker=self.checker)
        with self.assertRaises(ValueError):
            plan_collision_free([0.0, 10.0], FREE_B, checker=self.checker)  # over limit


class ShortcutTests(unittest.TestCase):
    def setUp(self):
        self.checker = CollisionChecker(PLANAR_XML)
        self.path = plan_collision_free(FREE_A, FREE_B, checker=self.checker,
                                        seed=0, step=0.2)

    def test_shortcut_stays_valid_and_shortens(self):
        short = shortcut_path(self.checker, self.path, seed=0)
        self.assertLessEqual(len(short), len(self.path))
        self.assertLessEqual(path_length(short), path_length(self.path) + 1e-9)
        np.testing.assert_allclose(short[0], self.path[0])
        np.testing.assert_allclose(short[-1], self.path[-1])
        self.assertTrue(_segments_free(self.checker, short))

    def test_shortcut_reproducible_with_seed(self):
        s1 = shortcut_path(self.checker, self.path, seed=3)
        s2 = shortcut_path(self.checker, self.path, seed=3)
        np.testing.assert_array_equal(s1, s2)

    def test_shortcut_validation(self):
        with self.assertRaises(ValueError):
            shortcut_path(self.checker, [FREE_A])
        with self.assertRaises(ValueError):
            shortcut_path(self.checker, self.path, iterations=-1)


class TimeParameterizationHandoffTests(unittest.TestCase):
    def test_rrt_path_feeds_time_parameterization(self):
        checker = CollisionChecker(PLANAR_XML)
        path = plan_collision_free(FREE_A, FREE_B, checker=checker, seed=0, step=0.2)
        path = shortcut_path(checker, path, seed=0)
        traj = plan_joint_waypoints(path, max_vel=2.0, max_acc=4.0)
        np.testing.assert_allclose(traj.q[0], path[0], atol=1e-9)
        np.testing.assert_allclose(traj.q[-1], path[-1], atol=1e-9)
        self.assertGreater(traj.duration, 0.0)
        self.assertLessEqual(np.abs(traj.qd).max(), 2.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
