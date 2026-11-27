"""Arm registry, task YAML configs, runtime MjSpec composition, and export CLI."""

import unittest
from unittest.mock import patch
import mujoco
from ufactory_mujoco import robots as robots_mod
from ufactory_mujoco.config import ConfigError
from ufactory_mujoco import ARMS, resolve_arm_scene
import tempfile
from pathlib import Path
from ufactory_mujoco.config import _canonical_json_sha256, load_task_config

# ==========================================================================
# From test_robots.py
# ==========================================================================

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
            with self.subTest(robot=key):
                path = resolve_arm_scene(key, None, "scene")
                self.assertTrue(path.is_file())
                model = mujoco.MjModel.from_xml_path(str(path))
                site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
                self.assertGreaterEqual(site, 0)
                self.assertEqual(model.nu, dof)

    def test_xarm6_gripper_scenes(self):
        scene = resolve_arm_scene("xarm6", "gripper_g2", "scene")
        pick = resolve_arm_scene("xarm6", "gripper_g2", "pick_place")
        self.assertEqual((scene.parent.name, scene.name), ("xarm6_with_gripper", "scene.xml"))
        self.assertEqual((pick.parent.name, pick.name), ("xarm6_with_gripper", "scene_pick_place.xml"))
        for path in (scene, pick):
            model = mujoco.MjModel.from_xml_path(str(path))
            self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"), 0)

    def test_incompatible_gripper(self):
        for unsupported in ("gripper_g2", "robotiq_2f_85", "robotiq_2f_140"):
            with self.assertRaises(ConfigError) as caught:
                resolve_arm_scene("lite6", unsupported, "scene")
            self.assertIn("does not support", str(caught.exception))
        for ft_combo in ("ft_sensor", "ft_sensor_and_gripper", "ft_sensor_and_vacuum_gripper"):
            with self.assertRaises(ConfigError) as ft_caught:
                resolve_arm_scene("lite6", ft_combo, "scene")
            self.assertIn("does not support", str(ft_caught.exception))
        for gripper in robots_mod.TASK_GRIPPERS.values():
            self.assertNotIn("ft_sensor", gripper)

    def test_all_declared_combos_resolve(self):
        for arm_key, spec in ARMS.items():
            for gripper in spec.grippers:
                with self.subTest(robot=arm_key, gripper=gripper):
                    path = resolve_arm_scene(arm_key, gripper, "scene")
                    self.assertTrue(path.is_file(), f"Missing combo scene: {path}")
                    model = mujoco.MjModel.from_xml_path(str(path))
                    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
                    self.assertGreaterEqual(site, 0)

    def test_compatible_gripper_without_mjcf(self):
        with patch.dict(robots_mod.RECIPES):
            del robots_mod.RECIPES["xarm5_with_bio_gripper"]
            with self.assertRaises(ConfigError) as caught:
                resolve_arm_scene("xarm5", "bio_gripper_g2", "scene")
            self.assertIn("not built yet", str(caught.exception))
            self.assertNotIn("does not support", str(caught.exception))

    def test_unknown_scene_rejected(self):
        with self.assertRaises(ConfigError):
            resolve_arm_scene("xarm6", None, "reach")

# ==========================================================================
# From test_config.py
# ==========================================================================

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
        self.assertIn("unknown config field", str(caught.exception))

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

    def test_booleans_rejected_for_numeric_fields(self):
        for snippet in (
            "pick_place:\n  lift_z_m: true\n",
            "pick_place:\n  grasp_gap_m: false\n",
            "pick_place:\n  cube_xy: [true, 0.1]\n",
            "box_pack:\n  speed_scale: true\n",
        ):
            with self.subTest(snippet=snippet), self._overlay(snippet):
                with self.assertRaises(ConfigError):
                    load_task_config(self.overlay_path)

    def test_hash_is_stable_and_changes_with_robot(self):
        with self._overlay("robot: lite6\n"):
            first = load_task_config(self.overlay_path)
            second = load_task_config(self.overlay_path)
        self.assertEqual(first.sha256, second.sha256)
        self.assertNotEqual(first.sha256, load_task_config().sha256)
        self.assertEqual(len(first.sha256), 64)

    def test_canonical_json_digest_vector_and_key_order(self):
        self.assertEqual(
            _canonical_json_sha256({"b": 1, "a": [1.5, "é", None, True]}),
            "4310c259a6f903aaf441de115a0b417227ea158ea1613ca86eff57e82894d4ba",
        )
        self.assertEqual(
            _canonical_json_sha256({"x": 1, "y": 2}),
            _canonical_json_sha256({"y": 2, "x": 1}),
        )

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

# ==========================================================================
# Runtime MjSpec composition & export-model CLI tests
# ==========================================================================


class ComposeAndExportTests(unittest.TestCase):
    def test_composed_registry_and_in_memory_spec(self):
        from ufactory_mujoco import COMPOSED_MODELS, available_models, compose_model, compose_spec

        self.assertLessEqual(set(COMPOSED_MODELS), set(available_models()))
        spec = compose_spec("xarm6_with_ft_sensor_and_gripper", scene=True)
        model = spec.compile()
        self.assertEqual(model.nsensor, 2)
        self.assertEqual(model.nu, 7)
        custom = compose_model(
            arm="xarm6",
            base_plate=True,
            ft_sensor=True,
            gripper="gripper_g2",
            camera="d435",
            scene=True,
        )
        self.assertGreaterEqual(mujoco.mj_name2id(custom, mujoco.mjtObj.mjOBJ_BODY, "base_plate"), 0)
        self.assertGreaterEqual(mujoco.mj_name2id(custom, mujoco.mjtObj.mjOBJ_BODY, "ft_sensor_output"), 0)
        self.assertGreaterEqual(mujoco.mj_name2id(custom, mujoco.mjtObj.mjOBJ_BODY, "d435_body"), 0)
        self.assertEqual(custom.nsensor, 2)
        self.assertEqual(custom.nu, 7)

        rail_combo = compose_model(
            arm="xarm6",
            linear_motor=True,
            ft_sensor=True,
            gripper="gripper_g2",
            camera="d435",
            scene=True,
        )
        self.assertGreaterEqual(mujoco.mj_name2id(rail_combo, mujoco.mjtObj.mjOBJ_BODY, "al0700_carriage"), 0)
        self.assertGreaterEqual(mujoco.mj_name2id(rail_combo, mujoco.mjtObj.mjOBJ_JOINT, "al0700_slide"), 0)
        self.assertEqual(rail_combo.nsensor, 2)
        self.assertEqual(rail_combo.nu, 8)

        with self.assertRaises(ValueError):
            compose_model(arm="lite6", linear_motor=True)
        with self.assertRaises(ValueError):
            compose_model(arm="xarm6", linear_motor=True, base_plate=True)

    def test_export_model_cli_and_portable_copy(self):
        from contextlib import redirect_stdout
        import io
        from ufactory_mujoco import export_model
        from ufactory_mujoco.cli import main as cli_main

        with tempfile.TemporaryDirectory() as tmp:
            out_ref = Path(tmp) / "ref_export"
            scene_ref = export_model("xarm6_with_ft_sensor_and_gripper", out_ref)
            self.assertTrue(scene_ref.is_file())
            self.assertTrue((out_ref / "xarm6_with_ft_sensor_and_gripper.xml").is_file())
            m1 = mujoco.MjModel.from_xml_path(str(scene_ref))
            self.assertEqual(m1.nsensor, 2)

            out_portable = Path(tmp) / "portable_export"
            with redirect_stdout(io.StringIO()):
                rc = cli_main([
                    "export-model", "xarm6_with_base_plate",
                    "--out-dir", str(out_portable), "--copy-meshes",
                ])
            self.assertEqual(rc, 0)
            self.assertTrue((out_portable / "assets").is_dir())
            m2 = mujoco.MjModel.from_xml_path(str(out_portable / "scene.xml"))
            self.assertGreaterEqual(mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_BODY, "base_plate"), 0)

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc_list = cli_main(["export-model", "--list"])
            self.assertEqual(rc_list, 0)
            self.assertIn("[composed]", buf.getvalue())
            self.assertIn("[static]", buf.getvalue())

            out_rail = Path(tmp) / "rail_export"
            with redirect_stdout(io.StringIO()):
                rc_rail = cli_main([
                    "export-model", "--arm", "xarm6", "--linear-motor",
                    "--out-dir", str(out_rail), "--copy-meshes",
                ])
            self.assertEqual(rc_rail, 0)
            m3 = mujoco.MjModel.from_xml_path(str(out_rail / "scene.xml"))
            self.assertGreaterEqual(mujoco.mj_name2id(m3, mujoco.mjtObj.mjOBJ_JOINT, "al0700_slide"), 0)

if __name__ == "__main__":
    unittest.main()
