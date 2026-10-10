"""Unit tests verifying all generated MJCF models can load, step, and parse in MuJoCo."""

from pathlib import Path
import unittest
import numpy as np

import mujoco

import importlib

load_and_view = importlib.import_module("examples.01_load_and_view")
resolve_model_path = load_and_view.resolve_model_path

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"
ALL_MODELS = [
    "xarm6",
    "xarm7",
    "xarm5",
    "uf850",
    "lite6",
    "gripper_g2",
    "bio_gripper_g2",
    "lite6_gripper",
    "lite6_vacuum_gripper",
    "xarm6_with_gripper",
]


class AllModelsLoadingTest(unittest.TestCase):
    def test_all_models_exist(self):
        """Ensure all 10 model directories contain scene.xml and their primary XML."""
        for name in ALL_MODELS:
            model_dir = MODELS_ROOT / name
            self.assertTrue(model_dir.is_dir(), f"Model directory {name} does not exist")
            scene_xml = model_dir / "scene.xml"
            primary_xml = model_dir / f"{name}.xml"
            self.assertTrue(scene_xml.is_file(), f"{name}/scene.xml not found")
            self.assertTrue(primary_xml.is_file(), f"{name}/{name}.xml not found")

    def test_all_models_load_and_step(self):
        """Load both primary XML and scene XML into MuJoCo and simulate 100 physics steps."""
        for name in ALL_MODELS:
            model_dir = MODELS_ROOT / name
            for xml_file in [model_dir / f"{name}.xml", model_dir / "scene.xml"]:
                with self.subTest(model=name, xml=xml_file.name):
                    m = mujoco.MjModel.from_xml_path(str(xml_file))
                    d = mujoco.MjData(m)
                    mujoco.mj_forward(m, d)

                    # Simulate 100 steps
                    for _ in range(100):
                        mujoco.mj_step(m, d)

                    self.assertFalse(np.isnan(d.qpos).any(), f"NaN in qpos for {xml_file}")
                    self.assertFalse(np.isinf(d.qpos).any(), f"Inf in qpos for {xml_file}")
                    self.assertFalse(np.isnan(d.qvel).any(), f"NaN in qvel for {xml_file}")
                    self.assertFalse(np.isinf(d.qvel).any(), f"Inf in qvel for {xml_file}")

    def test_resolve_model_path_support(self):
        """Verify examples/01_load_and_view can resolve all model names."""
        for name in ALL_MODELS:
            path = resolve_model_path(name)
            self.assertTrue(path.is_file(), f"Could not resolve model {name}")
            self.assertEqual(path.name, "scene.xml")

        # Nonexistent model should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            resolve_model_path("non_existent_robot_123")

    def test_keyframes_reset_and_step(self):
        """Test keyframe initialization for models that define home or ready poses."""
        for name in ALL_MODELS:
            scene_xml = MODELS_ROOT / name / "scene.xml"
            m = mujoco.MjModel.from_xml_path(str(scene_xml))
            d = mujoco.MjData(m)

            for key_id in range(m.nkey):
                key_name = m.key(key_id).name
                with self.subTest(model=name, keyframe=key_name):
                    mujoco.mj_resetDataKeyframe(m, d, key_id)
                    mujoco.mj_forward(m, d)
                    for _ in range(20):
                        mujoco.mj_step(m, d)
                    self.assertFalse(np.isnan(d.qpos).any(), f"NaN in qpos at keyframe {key_name}")


if __name__ == "__main__":
    unittest.main()
