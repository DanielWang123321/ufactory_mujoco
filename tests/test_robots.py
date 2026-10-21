"""Arm registry: degrees of freedom, scene resolution, gripper compatibility."""

import unittest

import mujoco

from ufactory_mujoco.config import ConfigError
from ufactory_mujoco.model_paths import MODELS_ROOT
from ufactory_mujoco.robots import ARMS, resolve_arm_scene

EXPECTED_DOF = {
    "xarm5": 5,
    "lite6": 6,
    "xarm6": 6,
    "xarm7": 7,
    "uf850": 6,
}


class ArmRegistryTests(unittest.TestCase):
    def test_dof(self):
        self.assertEqual(set(ARMS), set(EXPECTED_DOF))
        for key, dof in EXPECTED_DOF.items():
            self.assertEqual(ARMS[key].dof, dof, key)
            self.assertEqual(ARMS[key].tcp_site, "attachment_site", key)

    def test_bare_scenes_load(self):
        for key, dof in EXPECTED_DOF.items():
            for scene in ("scene", "reach"):
                with self.subTest(robot=key, scene=scene):
                    path = resolve_arm_scene(key, None, scene)
                    self.assertTrue(path.is_file())
                    model = mujoco.MjModel.from_xml_path(str(path))
                    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
                    self.assertGreaterEqual(site, 0)
                    self.assertEqual(model.nu, dof)

    def test_xarm6_gripper_scenes(self):
        scene = resolve_arm_scene("xarm6", "gripper_g2", "scene")
        pick = resolve_arm_scene("xarm6", "gripper_g2", "pick_place")
        self.assertEqual(scene, MODELS_ROOT / "xarm6_with_gripper" / "scene.xml")
        self.assertEqual(pick, MODELS_ROOT / "xarm6_with_gripper" / "scene_pick_place.xml")
        self.assertTrue(scene.is_file())
        self.assertTrue(pick.is_file())

    def test_incompatible_gripper(self):
        with self.assertRaises(ConfigError) as caught:
            resolve_arm_scene("lite6", "gripper_g2", "scene")
        self.assertIn("不兼容", str(caught.exception))

    def test_compatible_gripper_without_mjcf(self):
        with self.assertRaises(ConfigError) as caught:
            resolve_arm_scene("xarm5", "bio_gripper_g2", "scene")
        self.assertIn("尚未构建", str(caught.exception))
        self.assertNotIn("不兼容", str(caught.exception))

    def test_reach_rejects_gripper(self):
        with self.assertRaises(ConfigError):
            resolve_arm_scene("xarm6", "gripper_g2", "reach")
