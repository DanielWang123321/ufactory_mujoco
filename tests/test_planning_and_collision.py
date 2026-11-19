"""Collision queries, RRT-Connect path planning, and obstacle-avoidance integration tests."""

import unittest
from pathlib import Path
import mujoco
import numpy as np
from ufactory_mujoco.planning import CollisionChecker, arm_joint_limits
from ufactory_mujoco.planning import (
    PlanningError,
    plan_collision_free,
    shortcut_path,
)
from ufactory_mujoco.trajectory import plan_joint_waypoints
import importlib
import sys
from unittest.mock import patch
from ufactory_mujoco.sim_robot import SimRobot

# ==========================================================================
# From test_collision.py
# ==========================================================================

ROOT = Path(__file__).resolve().parent.parent

PLANAR_XML = ROOT / "tests" / "fixtures" / "planar_arm.xml"

COLLIDING = np.array([0.0, 0.0])

NEAR_MISS = np.array([0.2, 0.4])          # ~17 mm clearance to the disk

FREE_FOLDED = np.array([1.2, -0.6])

FREE_OTHER = np.array([-1.2, 0.6])

GRIPPER_BOUNDARY_XML = """
<mujoco model="boundary">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.2">
      <joint name="j1" type="hinge" axis="0 0 1" range="-1 1"/>
      <geom type="box" size="0.05 0.05 0.1"/>
      <body name="link" pos="0 0 0.2">
        <joint name="j2" type="hinge" axis="0 0 1" range="-2 2"/>
        <geom type="box" size="0.05 0.05 0.1"/>
        <body name="finger" pos="0 0 0.2">
          <joint name="slide" type="slide" axis="0 0 1" range="0 0.04"/>
          <geom type="box" size="0.02 0.02 0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <general name="act1" joint="j1" gainprm="1" biasprm="0 -1 0"/>
    <general name="act2" joint="j2" gainprm="1" biasprm="0 -1 0"/>
    <general name="gripper" joint="slide" gainprm="1" biasprm="0 -1 0"/>
  </actuator>
</mujoco>
"""

class ArmJointLimitsTests(unittest.TestCase):
    def test_synthetic_arm_limits(self):
        n, lo, hi = arm_joint_limits(mujoco.MjModel.from_xml_path(str(PLANAR_XML)))
        self.assertEqual(n, 2)
        np.testing.assert_allclose(lo, [-3.14, -3.14])
        np.testing.assert_allclose(hi, [3.14, 3.14])

    def test_gripper_actuator_excluded(self):
        n, lo, hi = arm_joint_limits(mujoco.MjModel.from_xml_string(GRIPPER_BOUNDARY_XML))
        self.assertEqual(n, 2)  # slide joint driven by 'gripper' is not arm
        np.testing.assert_allclose(lo, [-1.0, -2.0])
        np.testing.assert_allclose(hi, [1.0, 2.0])

    def test_xarm6_limits(self):
        n, lo, hi = arm_joint_limits(
            mujoco.MjModel.from_xml_path(str(ROOT / "models" / "xarm6" / "xarm6.xml")))
        self.assertEqual(n, 6)
        np.testing.assert_allclose(lo[1], -2.059)   # joint2 range from the MJCF
        np.testing.assert_allclose(hi[1], 2.0944)

class CollisionCheckerTests(unittest.TestCase):
    def setUp(self):
        self.checker = CollisionChecker(PLANAR_XML, safety_margin=0.0)

    def test_known_states(self):
        self.assertTrue(self.checker.in_collision(COLLIDING))
        self.assertLess(self.checker.min_distance(COLLIDING), 0.0)
        for q in (FREE_FOLDED, FREE_OTHER, NEAR_MISS):
            self.assertFalse(self.checker.in_collision(q), q)
            self.assertGreater(self.checker.min_distance(q), 0.0, q)

    def test_min_distance_decreases_when_approaching(self):
        d_far = self.checker.min_distance(np.array([0.25, 0.4]))
        d_near = self.checker.min_distance(NEAR_MISS)
        self.assertGreater(d_far, d_near)
        self.assertGreater(d_near, self.checker.min_distance(COLLIDING))

    def test_safety_margin_catches_near_miss(self):
        self.assertFalse(self.checker.in_collision(NEAR_MISS))
        self.assertTrue(self.checker.in_collision(NEAR_MISS, safety_margin=0.05))
        strict = CollisionChecker(PLANAR_XML, safety_margin=0.05)
        self.assertTrue(strict.in_collision(NEAR_MISS))
        self.assertFalse(strict.in_collision(np.array([0.3, 0.4])))  # > 5 cm gap

    def test_ignore_pairs(self):
        ignored = CollisionChecker(
            PLANAR_XML, ignore_pairs=[("base", "disk"), ("elbow", "disk")])
        self.assertFalse(ignored.in_collision(COLLIDING))

    def test_repeated_queries_are_deterministic(self):
        d1 = self.checker.min_distance(NEAR_MISS)
        d2 = self.checker.min_distance(NEAR_MISS)
        self.assertEqual(d1, d2)
        self.assertEqual(self.checker.in_collision(COLLIDING),
                         self.checker.in_collision(COLLIDING))

    def test_checker_does_not_touch_simulation_model(self):
        scene = ROOT / "models" / "xarm6" / "scene.xml"
        robot_model = mujoco.MjModel.from_xml_path(str(scene))
        robot_data = mujoco.MjData(robot_model)
        mujoco.mj_forward(robot_model, robot_data)
        ncon_before = robot_data.ncon

        checker = CollisionChecker(scene)
        checker.in_collision(np.zeros(6))

        np.testing.assert_array_equal(robot_model.geom_margin, 0.0)
        mujoco.mj_forward(robot_model, robot_data)
        self.assertEqual(robot_data.ncon, ncon_before)

    def test_xarm6_ready_pose_is_free(self):
        checker = CollisionChecker(ROOT / "models" / "xarm6" / "scene.xml")
        self.assertFalse(checker.in_collision(np.zeros(6)))

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            self.checker.in_collision(np.zeros(3))
        with self.assertRaises(ValueError):
            self.checker.in_collision(np.full(2, np.nan))
        with self.assertRaises(ValueError):
            CollisionChecker(PLANAR_XML, safety_margin=0.1, query_margin=0.05)
        with self.assertRaises(ValueError):
            CollisionChecker(PLANAR_XML, safety_margin=-0.1)
        with self.assertRaises(ValueError):
            CollisionChecker(PLANAR_XML, ignore_pairs=[("base", "nope")])
        with self.assertRaises(TypeError):
            CollisionChecker(42)

# ==========================================================================
# From test_path_planning.py
# ==========================================================================

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

def path_length(waypoints) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(waypoints, dtype=float), axis=0), axis=1).sum())

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

# ==========================================================================
# From test_obstacle_avoidance.py
# ==========================================================================

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
