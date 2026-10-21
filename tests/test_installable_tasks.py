"""Installed-task contracts: argument errors, reports, preflight and compatibility."""

import ast
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

import mujoco

from ufactory_mujoco import SimRobot
from ufactory_mujoco.cli import main
from ufactory_mujoco.config import load_task_config
from ufactory_mujoco.preflight import PreflightReport
from ufactory_mujoco.resource_paths import CONFIGS_ROOT
from ufactory_mujoco.tasks import pick_place, box_pack
from ufactory_mujoco.tasks._reporting import task_report

ROOT = Path(__file__).resolve().parents[1]


def exit_code(args):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        try:
            result = main(args)
        except SystemExit as exc:
            return exc.code
    return result


class CommandTests(unittest.TestCase):
    def test_help_and_unknown_commands(self):
        for args in (["--help"], ["pick-place", "--help"], ["box-pack", "--help"],
                     ["dynamics", "--help"]):
            with self.subTest(args=args):
                self.assertEqual(exit_code(args), 0)
        for args in ([], ["unknown"], ["pick-place", "--bogus"]):
            self.assertEqual(exit_code(args), 2)

    def test_invalid_arguments_rejected_before_robot_creation(self):
        cases = (["--robot", "unknown"], ["--executor", "unknown"], ["--cycles", "0"],
                 ["--cycles", "-1"], ["--cycles", "1.5"], ["--gripper", "unknown"],
                 ["--config", "missing.yaml"])
        for command, module in (("pick-place", pick_place), ("box-pack", box_pack)):
            with patch.object(module, "SimRobot") as robot:
                for args in cases:
                    with self.subTest(command=command, args=args):
                        self.assertEqual(exit_code([command, *args, "--headless"]), 2)
                robot.assert_not_called()
        for speed in ("0", "-1", "nan", "inf"):
            self.assertEqual(exit_code(["box-pack", "--speed", speed, "--headless"]), 2)
        self.assertEqual(exit_code(["box-pack", "--cycles", "4", "--headless"]), 2)

    def test_runtime_value_error_is_failure_not_argument_error(self):
        for command, module, function in (("pick-place", pick_place, "run_pick_place"),
                                          ("box-pack", box_pack, "run_box_pack")):
            with self.subTest(command=command), tempfile.TemporaryDirectory(dir=ROOT) as tmp:
                path = Path(tmp) / "failure.json"
                with patch.object(module, "SimRobot"), patch.object(module, function,
                                                                     side_effect=ValueError("runtime failure")):
                    self.assertEqual(exit_code([command, "--headless", "--report", str(path)]), 1)
                self.assertFalse(json.loads(path.read_text())["success"])

    def test_invalid_model_override_reports_failure(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            for command in ("pick-place", "box-pack", "dynamics"):
                for model_dir in (Path(tmp) / "missing", Path(tmp)):
                    report = Path(tmp) / f"{command}.json"
                    result = subprocess.run(
                        [sys.executable, "-m", "ufactory_mujoco", command, "--headless", "--report", str(report)],
                        cwd=ROOT, env=dict(os.environ, UFACTORY_MUJOCO_MODELS=str(model_dir)),
                        capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertFalse(json.loads(report.read_text())["success"])

    def test_relative_nested_report_and_write_failure(self):
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            try:
                os.chdir(tmp)
                for command in ("pick-place", "box-pack"):
                    with self.subTest(command=command), patch.object(mujoco, "mj_step", side_effect=AssertionError):
                        report = Path(command) / "nested/report.json"
                        self.assertEqual(exit_code([command, "--dry-run", "--report", str(report)]), 0)
                        self.assertTrue(json.loads(report.read_text())["success"])
                        self.assertEqual(exit_code([command, "--dry-run", "--report", command]), 1)
            finally:
                os.chdir(old_cwd)

    def test_preflight_failure_report_matches_exit(self):
        config = load_task_config(CONFIGS_ROOT / "pick_place/xarm6.yaml")
        config = replace(config, pick_place=replace(config.pick_place, target_xy=(9.0, 9.0)))
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            report = Path(tmp) / "failed.json"
            with patch.object(pick_place, "_load", return_value=config), \
                    patch.object(mujoco, "mj_step", side_effect=AssertionError):
                self.assertEqual(exit_code(["pick-place", "--dry-run", "--report", str(report)]), 1)
            payload = json.loads(report.read_text())
            self.assertFalse(payload["ok"])
            self.assertFalse(payload["success"])
            self.assertTrue(any(p["status"] == "fail" for p in payload["phases"]))


class TaskPackagingTests(unittest.TestCase):
    def test_five_arm_dry_runs_never_step_physics(self):
        with patch.object(mujoco, "mj_step", side_effect=AssertionError("dry-run stepped physics")), \
                redirect_stdout(io.StringIO()):
            for section, module in (("pick_place", pick_place), ("box_pack", box_pack)):
                for arm in ("lite6", "xarm5", "xarm6", "xarm7", "uf850"):
                    with self.subTest(task=section, arm=arm):
                        config = load_task_config(CONFIGS_ROOT / section / f"{arm}.yaml")
                        with SimRobot(config.scene_path) as robot:
                            outcome = getattr(module, "run_" + section)(robot, config, dry_run=True)
                            self.assertEqual(robot.data.time, 0)
                            self.assertTrue(outcome.ok)

    def test_cycle_failure_and_incomplete_reports_cannot_pass(self):
        config = load_task_config()
        preflight = PreflightReport(True, [])
        good = {"cube_lifted": True, "bilateral_contact": True, "home_drift_mm": 1., "place_error_mm": 1.}
        self.assertTrue(task_report(preflight, config, False, 1, [good])["ok"])
        for cycles in ([], [good, good], [good | {"place_error_mm": 50.}],
                       [good | {"home_drift_mm": 10.}], [good | {"bilateral_contact": False}]):
            self.assertFalse(task_report(preflight, config, False, 1, cycles)["ok"])

    def test_legacy_imports_share_implementations(self):
        for old, new in (("05_pick_place", "pick_place"), ("08_dynamics", "dynamics"),
                         ("10_box_pack", "box_pack")):
            self.assertIs(importlib.import_module(f"examples.simulation.{old}"),
                          importlib.import_module(f"ufactory_mujoco.tasks.{new}"))

    def test_no_package_imports_examples(self):
        for path in (ROOT / "ufactory_mujoco").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith("examples"), str(path))
                elif isinstance(node, ast.Import):
                    self.assertFalse(any(n.name.startswith("examples") for n in node.names), str(path))
                elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                    self.assertFalse(str(node.args[0].value).startswith("examples.simulation"), str(path))

    def test_requirements_match_pyproject(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        for name, expected in (("requirements.txt", project["dependencies"]),
                               ("requirements-rl.txt", project["optional-dependencies"]["rl"]),
                               ("requirements-assets.txt", project["optional-dependencies"]["assets"])):
            actual = [s.strip() for s in (ROOT / name).read_text().splitlines()
                      if s.strip() and not s.lstrip().startswith("#")]
            self.assertEqual(sorted(actual), sorted(expected))


if __name__ == "__main__":
    unittest.main()
