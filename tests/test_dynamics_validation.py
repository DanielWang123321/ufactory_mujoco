"""Batch 6: five-arm dynamics acceptance and monitored failure paths, offline."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from functools import lru_cache
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np

from ufactory_mujoco.collision import CollisionChecker
from ufactory_mujoco.dynamics import ArmDynamics
from ufactory_mujoco.robots import ARMS, resolve_arm_scene
from ufactory_mujoco.sim_robot import SimRobot

validation = importlib.import_module("ufactory_mujoco.tasks.dynamics")
ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "examples" / "simulation" / "08_dynamics.py"
PLANAR = ROOT / "tests" / "fixtures" / "planar_arm.xml"
PHASES = ("deploy", "deploy_settle", "outbound", "outbound_settle", "return", "return_settle")
BASELINE = np.array([0.0, -5.457423426494502, -21.096553945585573,
                     -1.5295099439534692, -1.570599127710176, 0.0066876600139051945])


@lru_cache(maxsize=1)
def five_arm_report():
    # Exercise the real full sequence once; share immutable results across assertions.
    return validation.run_validation("all", headless=True)


def by_robot():
    return {r["robot"]: r for r in five_arm_report()["robots"]}


def main_exit(arguments):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        try:
            validation.main(arguments)
        except SystemExit as exc:
            return exc.code
    raise AssertionError("CLI did not set an exit code")


class FiveArmValidationTest(unittest.TestCase):
    def test_all_bare_arms_pass_with_registry_identity(self):
        report = five_arm_report()
        self.assertTrue(report["success"], report)
        self.assertTrue(report["completed"])
        self.assertEqual([r["robot"] for r in report["robots"]], list(ARMS))
        for name, result in by_robot().items():
            with self.subTest(robot=name):
                self.assertEqual(result["dof"], ARMS[name].dof)
                self.assertEqual(result["tcp_site"], ARMS[name].tcp_site)
                self.assertEqual(Path(result["scene_path"]), resolve_arm_scene(name, None, "scene"))
                self.assertEqual(Path(result["model_path"]).name, f"{name}.xml")
                model = mujoco.MjModel.from_xml_path(result["model_path"])
                self.assertEqual(model.nq, ARMS[name].dof)
                self.assertEqual(model.nv, ARMS[name].dof)
                self.assertLess(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tcp_payload"), 0)

    def test_mass_gravity_coriolis_and_inverse_forward_at_all_poses(self):
        for name, result in by_robot().items():
            dynamics = result["dynamics"]
            with self.subTest(robot=name):
                self.assertTrue(dynamics["success"], dynamics["failures"])
                self.assertEqual(dynamics["pose_count"], 35)
                self.assertEqual(dynamics["evaluated_pose_count"], 35)
                self.assertEqual([p["pose"] for p in dynamics["poses"][:3]], ["home", "ready", "target"])
                for pose in dynamics["poses"]:
                    self.assertEqual(len(pose["q"]), ARMS[name].dof)
                    self.assertEqual(len(pose["gravity_nm"]), ARMS[name].dof)
                    self.assertTrue(np.all(np.isfinite(pose["gravity_nm"])))
                    self.assertLess(pose["mass_asymmetry"], 1e-12)
                    self.assertGreater(pose["mass_min_eigenvalue"], 0)
                    self.assertLess(pose["coriolis_at_rest_nm"], 1e-9)
                    self.assertLess(pose["id_fd_round_trip_rad_s2"], 1e-8)

    def test_feedforward_finite_and_inside_signed_actuator_ranges(self):
        for name, result in by_robot().items():
            with self.subTest(robot=name):
                ff = result["feedforward"]
                self.assertTrue(ff["success"], ff["failures"])
                ranges = np.asarray(ff["actuator_force_ranges_nm"])
                self.assertEqual(ranges.shape, (ARMS[name].dof, 2))
                self.assertTrue(np.all(np.isfinite(ff["peak_tau_nm"])))
                self.assertTrue(np.all(np.asarray(ff["min_tau_nm"]) >= ranges[:, 0]))
                self.assertTrue(np.all(np.asarray(ff["max_tau_nm"]) <= ranges[:, 1]))
                self.assertLessEqual(ff["utilization"], 1.0)
                self.assertEqual(ff["samples"], 4201)

    def test_random_samples_are_within_limits_and_collisions_are_only_classified(self):
        for name, result in by_robot().items():
            with self.subTest(robot=name):
                checker = validation.make_checker(result["scene_path"])
                expected = np.random.default_rng(0).uniform(checker.q_lo, checker.q_hi, (32, ARMS[name].dof))
                actual = result["collision"]["random_poses"]
                np.testing.assert_array_equal([r["q"] for r in actual], expected)
                np.testing.assert_array_equal([r["q"] for r in result["dynamics"]["poses"][3:]], expected)
                self.assertGreater(result["collision"]["random_collision_count"], 0)
                self.assertTrue(result["success"])
                self.assertTrue(all(r["joint_limit_excess_rad"] == 0 for r in actual))

    def test_preflight_and_monitor_cover_deploy_moves_holds_and_return(self):
        for name, result in by_robot().items():
            with self.subTest(robot=name):
                preflight, motion = result["preflight"], result["motion"]
                self.assertTrue(preflight["success"])
                self.assertEqual(preflight["samples_checked"], 4201)
                self.assertEqual([p["name"] for p in preflight["phases"]], ["initial", *PHASES])
                self.assertTrue(motion["success"], motion["failures"])
                self.assertTrue(motion["completed"])
                self.assertEqual([p["name"] for p in motion["phases"]], list(PHASES))
                self.assertTrue(all(p["completed"] for p in motion["phases"]))
                self.assertEqual(motion["observations"], motion["steps"] + 1)
                self.assertEqual(sum(p["steps"] for p in motion["phases"]), motion["steps"])
                self.assertGreaterEqual(motion["simulated_seconds"], 8.4 - 1e-10)
                self.assertLessEqual(motion["simulated_seconds"], 8.406)
                self.assertLessEqual(motion["max_penetration_m"], 1e-6)
                self.assertLessEqual(motion["max_joint_limit_excess_rad"], 1e-6)
                self.assertFalse(motion["warnings"])
                for i, phase in enumerate(motion["phases"]):
                    self.assertGreaterEqual(phase["steps"], (750, 150, 1500, 150, 1500, 150)[i])
                    if phase["name"].endswith("settle"):
                        self.assertLess(phase["arrival_error_rad"], 0.01)

    def test_repeat_run_resets_rng_and_matches_all_mode(self):
        second = validation.run_checks("xarm6")
        self.assertEqual(second, by_robot()["xarm6"])

    def test_report_metadata_and_strict_json(self):
        report = five_arm_report()
        encoded = json.dumps(report, allow_nan=False)
        self.assertEqual(json.loads(encoded), report)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["sampling"], {"samples": 32, "seed": 0})
        for name in ("python", "executable", "mujoco", "numpy", "platform", "machine"):
            self.assertTrue(report["environment"][name])
        self.assertEqual(report["thresholds"]["arrival_error_rad"], 0.01)

    def test_fixed_six_axis_payload_baseline_and_original_demo(self):
        payload = by_robot()["xarm6"]["payload"]
        self.assertTrue(payload["success"], payload["failures"])
        np.testing.assert_array_equal(payload["baseline_gravity_nm"], BASELINE)
        np.testing.assert_allclose(payload["gravity_nm"], BASELINE, rtol=0, atol=1e-6)
        self.assertLessEqual(payload["max_baseline_error_nm"], 1e-6)
        self.assertTrue(np.all(np.asarray(payload["historical_relative_errors"]) <= 0.05))
        self.assertTrue(payload["feedforward"]["success"])
        self.assertEqual(payload["mass_kg"], 3)
        self.assertEqual(payload["com_m"], [0, 0, 0.021])
        self.assertFalse(payload["new_hardware_validation"])
        for name in ("lite6", "xarm5", "xarm7", "uf850"):
            self.assertIsNone(by_robot()[name]["payload"])


class DynamicsCollisionTest(unittest.TestCase):
    def test_fixed_safe_and_colliding_poses_without_advancing_physics(self):
        indices = {"lite6": 2, "xarm5": 3, "xarm6": 4, "xarm7": 0, "uf850": 2}
        for name, arm in ARMS.items():
            with self.subTest(robot=name):
                checker = validation.make_checker(resolve_arm_scene(name, None, "scene"))
                ready = np.asarray(validation.PRESETS[name][0])
                self.assertFalse(checker.in_collision(ready))
                self.assertTrue(checker.in_collision(validation.COLLISION_POSES[name]))
                self.assertLess(checker.min_distance(validation.COLLISION_POSES[name]), -0.001)
                sample = np.random.default_rng(0).uniform(checker.q_lo, checker.q_hi, (32, arm.dof))
                np.testing.assert_array_equal(sample[indices[name]], validation.COLLISION_POSES[name])
                self.assertTrue(all(checker.min_distance(q) >= -0.001 for q in sample[:indices[name]]))
                self.assertEqual(checker.data.time, 0)

    def test_floor_and_self_collisions_remain_visible(self):
        for name in ARMS:
            with self.subTest(robot=name):
                checker = validation.make_checker(resolve_arm_scene(name, None, "scene"))
                metrics = validation.pose_metrics(checker, validation.COLLISION_POSES[name])
                if name in ("lite6", "xarm5", "uf850"):
                    self.assertTrue(metrics["floor_collision"])
                else:
                    self.assertTrue(metrics["self_collision"])

    def test_safe_endpoints_with_midpath_collision_are_rejected(self):
        checker = CollisionChecker(PLANAR)
        start, goal = np.array([1.2, -0.6]), np.array([-1.2, 0.6])
        self.assertFalse(checker.in_collision(start))
        self.assertFalse(checker.in_collision(goal))
        segments = (validation.MotionSegment("cross_disk", goal, 1.5),)
        with patch.object(mujoco, "mj_step", side_effect=AssertionError("must not step physics")):
            report = validation.preflight_motion(checker, start, segments)
        self.assertFalse(report["success"])
        self.assertTrue(checker.in_collision(report["failed_q"]))
        self.assertIn("cross_disk sample", report["failures"][0])
        self.assertGreater(report["max_penetration_m"], 1e-6)
        self.assertEqual(checker.data.time, 0)

    def test_joint_limit_preflight_rejection(self):
        checker = CollisionChecker(PLANAR)
        start = np.array([1.2, -0.6])
        goal = np.array([checker.q_hi[0] + 0.1, -0.6])
        report = validation.preflight_motion(checker, start, (validation.MotionSegment("over_limit", goal, 1.5),))
        self.assertFalse(report["success"])
        self.assertTrue(any("joint limit" in reason for reason in report["failures"]))
        self.assertEqual(checker.data.time, 0)

    def test_preflight_failure_does_not_construct_or_step_simulation(self):
        failed = {"success": False, "completed": False, "failures": ["outbound sample 42: penetration"]}
        with patch.object(validation, "preflight_motion", return_value=failed):
            with patch.object(validation, "MonitoredSimRobot") as simulation:
                report = validation.run_checks("xarm5", samples=1)
        simulation.assert_not_called()
        self.assertFalse(report["success"])
        self.assertEqual(report["motion"]["steps"], 0)
        self.assertTrue(any("outbound sample 42" in s for s in report["failures"]))


class DynamicsFailureTest(unittest.TestCase):
    def setUp(self):
        self.scene = resolve_arm_scene("xarm6", None, "scene")
        self.segments = validation.motion_segments(validation.READY, validation.TARGET)

    def injected_motion(self, inject, phase="deploy"):
        checker = validation.make_checker(self.scene)
        original_step = SimRobot._step
        injected = False

        def step(robot):
            nonlocal injected
            original_step(robot)
            if not injected and robot.phase["name"] == phase:
                injected = True
                inject(robot)

        with validation.MonitoredSimRobot(self.scene, checker) as robot:
            with patch.object(SimRobot, "_step", step):
                report = validation.execute_motion(robot, self.segments)
        self.assertTrue(injected)
        self.assertFalse(report["success"])
        self.assertFalse(report["completed"])
        self.assertEqual(report["phases"][-1]["name"], phase)
        self.assertEqual(report["phases"][-1]["steps"], 1)
        self.assertFalse(report["phases"][-1]["completed"])
        return report

    def test_collision_in_integrated_state_is_detected_despite_stale_contacts(self):
        def inject(robot):
            # Original mj_step has already computed contacts at the previous pose.
            self.assertEqual(robot.data.ncon, 0)
            robot.data.qpos[:] = validation.COLLISION_POSES["xarm6"]

        report = self.injected_motion(inject)
        self.assertEqual(report["steps"], 1)
        self.assertGreater(report["max_penetration_m"], 0.001)
        self.assertIn("penetration", report["failures"][0])

    def test_joint_limit_violation_during_motion_stops_before_return(self):
        def inject(robot):
            robot.data.qpos[0] = robot.checker.q_hi[0] + 0.01

        report = self.injected_motion(inject, "outbound")
        self.assertGreater(report["max_joint_limit_excess_rad"], 1e-6)
        self.assertIn("joint limit", report["failures"][0])

    def test_nonfinite_velocity_during_return_stops_and_reports(self):
        def inject(robot):
            robot.data.qvel[0] = np.nan

        report = self.injected_motion(inject, "return")
        self.assertIn("non-finite state: qvel", report["failures"][0])
        json.dumps(report, allow_nan=False)

    def test_final_return_settle_is_also_monitored(self):
        def inject(robot):
            robot.data.qpos[:] = validation.COLLISION_POSES["xarm6"]

        report = self.injected_motion(inject, "return_settle")
        self.assertIn("penetration", report["failures"][0])
        self.assertTrue(all(p["completed"] for p in report["phases"][:-1]))

    def test_numerical_warning_is_a_failure(self):
        def inject(robot):
            robot.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number += 1

        report = self.injected_motion(inject, "deploy_settle")
        self.assertEqual(report["warnings"]["mjWARN_BADQACC"], 1)
        self.assertIn("MuJoCo warning", report["failures"][0])

    def test_time_reset_is_a_failure(self):
        def inject(robot):
            robot.data.time = 0.0

        report = self.injected_motion(inject)
        self.assertIn("physics time", report["failures"][0])

    def test_viewer_closure_is_incomplete(self):
        def inject(robot):
            robot._viewer = Mock()
            robot._viewer.is_running.return_value = False

        report = self.injected_motion(inject)
        self.assertIn("viewer closed", report["failures"][0])

    def test_nonfinite_initial_position_is_rejected_before_physics(self):
        checker = validation.make_checker(self.scene)
        with validation.MonitoredSimRobot(self.scene, checker) as robot:
            robot.data.qpos[0] = np.nan
            with patch.object(SimRobot, "_step") as step:
                report = validation.execute_motion(robot, self.segments)
        step.assert_not_called()
        self.assertEqual(report["steps"], 0)
        self.assertIn("non-finite state: qpos", report["failures"][0])

    def test_early_move_and_settle_are_failures(self):
        for method in ("move_joints", "settle"):
            with self.subTest(method=method):
                checker = validation.make_checker(self.scene)
                with validation.MonitoredSimRobot(self.scene, checker) as robot:
                    with patch.object(robot, method, return_value=None):
                        report = validation.execute_motion(robot, self.segments)
                self.assertFalse(report["success"])
                self.assertIn("motion ended early", report["failures"][0])

    def test_arrival_error_is_a_failure(self):
        checker = validation.make_checker(self.scene)
        with validation.MonitoredSimRobot(self.scene, checker) as robot:
            settle = robot.settle

            def wrong_arrival(duration):
                settle(duration)
                robot.data.qpos[0] += 0.02

            with patch.object(robot, "settle", side_effect=wrong_arrival):
                report = validation.execute_motion(robot, self.segments)
        self.assertFalse(report["success"])
        self.assertGreater(report["max_arrival_error_rad"], 0.01)
        self.assertIn("arrival error", report["failures"][0])

    def test_bad_dynamics_shapes_and_nonfinite_outputs_fail(self):
        dyn = ArmDynamics(validation.XARM6)
        cases = (("mass_matrix", np.eye(5)), ("gravity", np.full(6, np.nan)),
                 ("coriolis", np.ones(6)), ("forward_dynamics", np.full(6, np.inf)),
                 ("mass_matrix", -np.eye(6)))
        for method, value in cases:
            with self.subTest(method=method, value=value):
                with patch.object(dyn, method, return_value=value):
                    report = validation.check_dynamics(dyn, [("home", np.zeros(6))], np.random.default_rng(0))
                self.assertFalse(report["success"])
                self.assertTrue(report["failures"])

    def test_nonfinite_and_signed_overload_feedforward_fail(self):
        dyn = ArmDynamics(validation.XARM6)
        q = np.zeros((3, 6))
        with patch.object(dyn, "feedforward_traj", return_value=np.full((3, 6), np.nan)):
            with self.assertRaisesRegex(validation.ValidationFailure, "feedforward"):
                validation.check_feedforward(dyn, q, 0.01)
        for torque in (-1000.0, 1000.0):
            with self.subTest(torque=torque):
                with patch.object(dyn, "feedforward_traj", return_value=np.full((3, 6), torque)):
                    report = validation.check_feedforward(dyn, q, 0.01)
                self.assertFalse(report["success"])
                self.assertGreater(report["utilization"], 1)

    def test_payload_baseline_detects_drift_on_other_than_j2_j3(self):
        changed = BASELINE.copy()
        changed[5] += 2e-6
        with patch.object(ArmDynamics, "gravity", return_value=changed):
            report = validation.check_payload_regression()
        self.assertFalse(report["success"])
        self.assertIn("six-axis gravity baseline", report["failures"][0])
        self.assertTrue(np.all(np.asarray(report["historical_relative_errors"]) < 0.05))


class DynamicsCliTest(unittest.TestCase):
    def test_default_legacy_options_still_select_xarm6_and_full_validation(self):
        report = deepcopy(five_arm_report())
        report["robots"] = [by_robot()["xarm6"]]
        with patch.object(validation, "run_validation", return_value=report) as run:
            self.assertEqual(main_exit(["--headless"]), 0)
        run.assert_called_once_with("xarm6", 32, 0, headless=True)

    def test_all_cli_writes_complete_strict_report(self):
        expected = five_arm_report()
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "nested" / "all.json"
            with patch.object(validation, "run_validation", return_value=expected) as run:
                code = main_exit(["--robot", "all", "--headless", "--report", str(path)])
            self.assertEqual(code, 0)
            run.assert_called_once_with("all", 32, 0, headless=True)
            actual = json.loads(path.read_text(), parse_constant=lambda s: self.fail(s))
            self.assertEqual(actual, expected)

    def test_single_cli_runs_without_xarm_sdk(self):
        # A fresh interpreter blocks all xarm SDK imports before loading the entry point.
        bootstrap = '''
import importlib.abc
import runpy
import sys
class NoSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'xarm':
            raise ImportError('xArm SDK deliberately unavailable')
sys.meta_path.insert(0, NoSDK())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "single.json"
            result = subprocess.run(
                [sys.executable, "-c", bootstrap, str(SCRIPT), "--robot", "xarm5", "--headless",
                 "--samples", "3", "--seed", "7", "--report", str(path)],
                cwd=ROOT, capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(path.read_text())
        self.assertTrue(report["success"])
        self.assertEqual(report["sampling"], {"samples": 3, "seed": 7})
        self.assertEqual([r["robot"] for r in report["robots"]], ["xarm5"])
        self.assertEqual(report["robots"][0]["dynamics"]["pose_count"], 6)
        self.assertGreater(report["robots"][0]["motion"]["steps"], 0)

    def test_illegal_arguments_exit_two_before_validation(self):
        cases = (["--robot", "unknown"], ["--samples", "0"], ["--samples", "-1"],
                 ["--samples", "1.5"], ["--samples", "nan"], ["--seed", "-1"],
                 ["--seed", "1.5"], ["--unknown"])
        for arguments in cases:
            with self.subTest(arguments=arguments), patch.object(validation, "run_validation") as run:
                self.assertEqual(main_exit(arguments), 2)
                run.assert_not_called()

    def test_failed_or_incomplete_run_exits_one_and_serializes_nonfinite_as_null(self):
        report = deepcopy(five_arm_report())
        report["success"] = report["completed"] = False
        report["robots"][0]["success"] = False
        report["robots"][0]["failures"] = ["injected numerical failure"]
        report["unavailable"] = np.array([np.nan, np.inf, -np.inf])
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "failed.json"
            with patch.object(validation, "run_validation", return_value=report):
                self.assertEqual(main_exit(["--headless", "--report", str(path)]), 1)
            actual = json.loads(path.read_text(), parse_constant=lambda s: self.fail(s))
        self.assertFalse(actual["success"])
        self.assertEqual(actual["unavailable"], [None, None, None])
        self.assertTrue(actual["robots"][0]["failures"])

    def test_incomplete_report_cannot_exit_zero(self):
        report = deepcopy(five_arm_report())
        report["completed"] = False
        with patch.object(validation, "run_validation", return_value=report):
            self.assertEqual(main_exit(["--headless"]), 1)

    def test_cli_runtime_failure_retains_other_arm_results_and_exits_one(self):
        original_run, original_step = validation.run_checks, SimRobot._step

        def inject(robot):
            original_step(robot)
            robot.data.qpos[:] = validation.COLLISION_POSES["xarm6"]

        def run(name, samples, seed, **kwargs):
            if name == "xarm6":
                with patch.object(SimRobot, "_step", inject):
                    return original_run(name, samples, seed, **kwargs)
            return deepcopy(by_robot()[name])

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "motion_failed.json"
            with patch.object(validation, "run_checks", side_effect=run) as checked:
                code = main_exit(["--robot", "all", "--headless", "--samples", "1", "--report", str(path)])
            report = json.loads(path.read_text())
        self.assertEqual(code, 1)
        self.assertEqual(checked.call_count, 5)
        for result in report["robots"]:
            if result["robot"] == "xarm6":
                self.assertFalse(result["success"])
                self.assertEqual(result["motion"]["steps"], 1)
                self.assertTrue(any("penetration" in reason for reason in result["failures"]))
            else:
                self.assertTrue(result["success"])

    def test_all_mode_continues_after_single_model_exception(self):
        def run(name, *args, **kwargs):
            if name == "xarm6":
                raise RuntimeError("injected model error")
            return deepcopy(by_robot()[name])

        with patch.object(validation, "run_checks", side_effect=run) as checked:
            report = validation.run_validation("all")
        self.assertEqual(checked.call_count, 5)
        self.assertFalse(report["success"])
        self.assertEqual([r["robot"] for r in report["robots"]], list(ARMS))
        for result in report["robots"]:
            if result["robot"] == "xarm6":
                self.assertIn("injected model error", result["failures"][0])
            else:
                self.assertTrue(result["success"])

    def test_interruption_marks_remaining_models_incomplete(self):
        with patch.object(validation, "run_checks", side_effect=KeyboardInterrupt) as checked:
            report = validation.run_validation("all")
        self.assertEqual(checked.call_count, 1)
        self.assertFalse(report["success"])
        self.assertFalse(report["completed"])
        self.assertEqual(len(report["robots"]), 5)
        self.assertTrue(all(not r["completed"] and r["failures"] for r in report["robots"]))
        json.dumps(report, allow_nan=False)

    def test_report_write_failure_exits_one(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            with patch.object(validation, "run_validation", return_value=five_arm_report()):
                self.assertEqual(main_exit(["--headless", "--report", tmp]), 1)


if __name__ == "__main__":
    unittest.main()
