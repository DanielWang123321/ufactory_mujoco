"""Regression checks for backend selection, startup failure and render cleanup."""

import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ufactory_mujoco import SimRobot
from ufactory_mujoco.viewer import PassiveViewer, _prepare_viewer, configure_backend

ROOT = Path(__file__).resolve().parent.parent


class ViewerTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux wheel selection")
    def test_wayland_desktop_loads_x11_before_mujoco(self):
        env = os.environ.copy()
        env.update(DISPLAY=":0", XDG_SESSION_TYPE="wayland")
        env.pop("PYGLFW_LIBRARY", None)
        env.pop("PYGLFW_LIBRARY_VARIANT", None)
        result = subprocess.run(
            [sys.executable, "-c",
             "from ufactory_mujoco import SimRobot; import glfw; "
             "print(glfw.get_version_string().decode())"],
            cwd=ROOT, env=env, text=True, capture_output=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("X11", result.stdout)
        self.assertNotIn("Wayland", result.stdout)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux wheel selection")
    def test_wayland_desktop_reloads_x11_even_if_mujoco_imported_first(self):
        env = os.environ.copy()
        env.update(DISPLAY=":0", XDG_SESSION_TYPE="wayland")
        env.pop("PYGLFW_LIBRARY", None)
        env.pop("PYGLFW_LIBRARY_VARIANT", None)
        result = subprocess.run(
            [sys.executable, "-c",
             "import mujoco; from ufactory_mujoco import SimRobot; import glfw; "
             "print(glfw.get_version_string().decode())"],
            cwd=ROOT, env=env, text=True, capture_output=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("X11", result.stdout)
        self.assertNotIn("Wayland", result.stdout)

    def test_backend_keeps_explicit_overrides(self):
        for override in ({"PYGLFW_LIBRARY_VARIANT": "wayland"},
                         {"PYGLFW_LIBRARY": "/custom/libglfw.so"}):
            env = {"DISPLAY": ":0", "XDG_SESSION_TYPE": "wayland", **override}
            with self.subTest(override=override), patch.dict(os.environ, env, clear=True):
                configure_backend()
                for key, value in override.items():
                    self.assertEqual(os.environ[key], value)
                if "PYGLFW_LIBRARY" in override:
                    self.assertNotIn("PYGLFW_LIBRARY_VARIANT", os.environ)

    def test_glfw_initialization_failure_has_actionable_error(self):
        with patch("glfw.init", return_value=0), patch(
            "glfw.get_error", return_value=(65550, b"X11: Failed to open display :0")
        ):
            with self.assertRaisesRegex(RuntimeError, "Failed to open display.*--headless"):
                _prepare_viewer()

    def test_render_startup_failure_reaches_caller(self):
        def fail(*args, **kwargs):
            raise RuntimeError("could not create window")

        with patch("ufactory_mujoco.viewer._prepare_viewer",
                   return_value=SimpleNamespace(_launch_internal=fail)):
            with self.assertRaisesRegex(RuntimeError, "startup failed: could not create window"):
                PassiveViewer(None, None)
        self.assertFalse(any(t.name == "xarm6-viewer" for t in threading.enumerate()))

    def test_close_waits_for_render_cleanup(self):
        exit_requested = threading.Event()
        cleanup_started = threading.Event()
        allow_cleanup = threading.Event()
        cleanup_finished = threading.Event()
        close_finished = threading.Event()
        handle = SimpleNamespace(close=exit_requested.set)

        def render(model, data, *, run_physics_thread, handle_return):
            handle_return.put(handle)
            if exit_requested.wait(2):
                cleanup_started.set()
                allow_cleanup.wait(2)
                cleanup_finished.set()

        with patch("ufactory_mujoco.viewer._prepare_viewer",
                   return_value=SimpleNamespace(_launch_internal=render)):
            viewer = PassiveViewer(None, None)

            def close():
                viewer.close()
                close_finished.set()

            closer = threading.Thread(target=close)
            closer.start()
            try:
                self.assertTrue(cleanup_started.wait(1))
                self.assertFalse(close_finished.is_set())
            finally:
                allow_cleanup.set()
                closer.join(2)
            self.assertTrue(cleanup_finished.is_set())
            self.assertTrue(close_finished.is_set())
            self.assertFalse(closer.is_alive())

    def test_robot_cleans_up_after_exception(self):
        close_calls = []
        viewer = SimpleNamespace(close=lambda: close_calls.append(True))
        with patch("ufactory_mujoco.sim_robot.PassiveViewer", return_value=viewer):
            with self.assertRaisesRegex(ValueError, "control failed"):
                with SimRobot(viewer=True) as robot:
                    raise ValueError("control failed")
        self.assertEqual(close_calls, [True])
        robot.close()
        self.assertEqual(close_calls, [True])


if __name__ == "__main__":
    unittest.main()
