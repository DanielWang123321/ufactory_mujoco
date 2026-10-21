"""Task YAML: defaults, strict overlay, and a stable content hash."""

import tempfile
import unittest
from pathlib import Path

from ufactory_mujoco.config import ConfigError, load_task_config


class TaskConfigTests(unittest.TestCase):
    def test_default(self):
        config = load_task_config()
        self.assertEqual(config.schema_version, 1)
        self.assertEqual(config.robot, "xarm6")
        self.assertIsNone(config.gripper)
        self.assertEqual(config.scene, "scene")
        self.assertTrue(config.scene_path.is_file())

    def test_unknown_field(self):
        with self._overlay("speed: 1\n"):
            with self.assertRaises(ConfigError) as caught:
                load_task_config(self.overlay_path)
        self.assertIn("未知配置字段", str(caught.exception))

    def test_schema_version_must_be_one(self):
        with self._overlay("schema_version: 2\n"):
            with self.assertRaises(ConfigError):
                load_task_config(self.overlay_path)

    def test_root_must_be_mapping(self):
        with self._overlay("- a\n- b\n"):
            with self.assertRaises(ConfigError) as caught:
                load_task_config(self.overlay_path)
        self.assertIn("mapping", str(caught.exception))

    def test_robot_must_be_string(self):
        with self._overlay("robot: [xarm6]\n"):
            with self.assertRaises(ConfigError):
                load_task_config(self.overlay_path)

    def test_hash_is_stable_and_changes_with_robot(self):
        with self._overlay("robot: lite6\n"):
            first = load_task_config(self.overlay_path)
            second = load_task_config(self.overlay_path)
        self.assertEqual(first.sha256, second.sha256)
        self.assertNotEqual(first.sha256, load_task_config().sha256)
        self.assertEqual(len(first.sha256), 64)

    def _overlay(self, text: str):
        return _Overlay(self, text)


class _Overlay:
    def __init__(self, test: TaskConfigTests, text: str):
        self.test = test
        self.text = text
        self._tmp = None

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        path = Path(self._tmp.name) / "overlay.yaml"
        path.write_text(self.text, encoding="utf-8")
        self.test.overlay_path = path
        return path

    def __exit__(self, exc_type, exc, tb):
        self._tmp.cleanup()
        return False
