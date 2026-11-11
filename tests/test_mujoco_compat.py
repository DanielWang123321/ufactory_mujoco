"""Guards for the MuJoCo version range and the private viewer hooks this project relies on."""

from importlib import metadata
import inspect
from pathlib import Path
import re
import tomllib
import unittest
from unittest.mock import patch

import mujoco

from ufactory_mujoco.rl.artifacts import installed_version

ROOT = Path(__file__).resolve().parent.parent


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


class MujocoCompatTests(unittest.TestCase):
    def test_installed_mujoco_is_inside_declared_range(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        spec = next(dep for dep in project["dependencies"] if re.match(r"mujoco\b", dep))
        lower = re.search(r">=\s*([\d.]+)", spec).group(1)
        upper = re.search(r"<\s*([\d.]+)", spec).group(1)
        installed = _version_tuple(mujoco.__version__)
        self.assertGreaterEqual(installed, _version_tuple(lower), f"MuJoCo {mujoco.__version__} is below {spec}")
        self.assertLess(installed, _version_tuple(upper), f"MuJoCo {mujoco.__version__} is above {spec}")

    def test_private_viewer_hooks_exist(self):
        # ufactory_mujoco.viewer owns the render thread through these two private entry points.
        import mujoco.viewer

        parameters = inspect.signature(mujoco.viewer._launch_internal).parameters
        self.assertLessEqual({"run_physics_thread", "handle_return"}, set(parameters))
        self.assertTrue(callable(mujoco.viewer._simulate.set_glfw_dlhandle))

    def test_provenance_reports_mujoco_version_for_core_only_install(self):
        def version(name: str) -> str:
            if name == "mujoco-core":
                return "4.0.0"
            raise metadata.PackageNotFoundError(name)

        with patch("ufactory_mujoco.rl.artifacts.metadata.version", side_effect=version):
            self.assertEqual(installed_version("mujoco"), "4.0.0")
            self.assertIsNone(installed_version("numpy"))

    def test_provenance_prefers_the_mujoco_distribution(self):
        with patch("ufactory_mujoco.rl.artifacts.metadata.version", return_value="3.16.0") as version:
            self.assertEqual(installed_version("mujoco"), "3.16.0")
        version.assert_called_once_with("mujoco")


if __name__ == "__main__":
    unittest.main()
