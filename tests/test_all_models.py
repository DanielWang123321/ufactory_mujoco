"""Unit tests verifying all generated MJCF models can load, step, and parse in MuJoCo."""

import re
import unittest
import warnings

import numpy as np

import mujoco

from ufactory_mujoco.compose import COMPOSED_MODELS, compose_model, initial_keyframe, is_composed_model
from ufactory_mujoco import MODELS_ROOT, available_models, resolve_arm_scene, resolve_model_path
from ufactory_mujoco.robots import RECIPES, TASK_GRIPPERS, needs_combo_init

ALL_MODELS = [
    "xarm6",
    "xarm7",
    "xarm5",
    "uf850",
    "lite6",
    "gripper_g2",
    "bio_gripper_g2",
    "robotiq_2f_85",
    "robotiq_2f_140",
    "robotiq_mounting",
    "lite6_gripper",
    "lite6_vacuum_gripper",
    "vacuum_gripper",
    "xarm6_with_gripper",
    "xarm5_with_gripper",
    "xarm7_with_gripper",
    "uf850_with_gripper",
    "xarm5_with_bio_gripper",
    "xarm6_with_bio_gripper",
    "xarm7_with_bio_gripper",
    "uf850_with_bio_gripper",
    "xarm5_with_robotiq_2f_85",
    "xarm6_with_robotiq_2f_85",
    "xarm7_with_robotiq_2f_85",
    "uf850_with_robotiq_2f_85",
    "xarm5_with_robotiq_2f_140",
    "xarm6_with_robotiq_2f_140",
    "xarm7_with_robotiq_2f_140",
    "uf850_with_robotiq_2f_140",
    "xarm5_with_vacuum_gripper",
    "xarm6_with_vacuum_gripper",
    "xarm7_with_vacuum_gripper",
    "uf850_with_vacuum_gripper",
    "lite6_with_gripper",
    "lite6_with_vacuum_gripper",
    "lite6_with_d435",
    "lite6_with_gripper_and_d435",
    "d405",
    "d435",
    "gemini_305",
    "zed_x_nano",
    "camera_stand",
    "camera_stand_with_d405",
    "camera_stand_with_d435",
    "camera_stand_with_gemini_305",
    "camera_stand_with_zed_x_nano",
    "xarm6_with_gripper_and_d405",
    "xarm6_with_gripper_and_d435",
    "xarm6_with_gripper_and_gemini_305",
    "xarm6_with_gripper_and_zed_x_nano",
    "xarm6_with_robotiq_2f_85_and_d435",
    "xarm6_with_robotiq_2f_140_and_d435",
    "xarm5_with_d435",
    "xarm6_with_d435",
    "xarm7_with_d435",
    "uf850_with_d435",
    "xarm5_with_gripper_and_d435",
    "xarm7_with_gripper_and_d435",
    "uf850_with_gripper_and_d435",
    "ft_sensor",
    "xarm5_with_ft_sensor",
    "xarm6_with_ft_sensor",
    "xarm7_with_ft_sensor",
    "uf850_with_ft_sensor",
    "xarm5_with_ft_sensor_and_gripper",
    "xarm6_with_ft_sensor_and_gripper",
    "xarm7_with_ft_sensor_and_gripper",
    "uf850_with_ft_sensor_and_gripper",
    "xarm5_with_ft_sensor_and_vacuum_gripper",
    "xarm6_with_ft_sensor_and_vacuum_gripper",
    "xarm7_with_ft_sensor_and_vacuum_gripper",
    "uf850_with_ft_sensor_and_vacuum_gripper",
    "xarm_base_plate",
    "uf850_base_plate",
    "lite6_base_plate",
    "xarm5_with_base_plate",
    "xarm6_with_base_plate",
    "xarm7_with_base_plate",
    "uf850_with_base_plate",
    "lite6_with_base_plate",
]


class AllModelsLoadingTest(unittest.TestCase):
    def test_all_models_exist(self):
        """The model list is the single snapshot of available models; every name resolves."""
        self.assertEqual(sorted(available_models()), sorted(ALL_MODELS))
        self.assertEqual(set(COMPOSED_MODELS), {n for n in ALL_MODELS if not (MODELS_ROOT / n).is_dir()})
        for name in ALL_MODELS:
            if is_composed_model(name):
                scene_xml = resolve_model_path(name, scene=True)
                primary_xml = resolve_model_path(name, scene=False)
            else:
                model_dir = MODELS_ROOT / name
                self.assertTrue(model_dir.is_dir(), f"Model directory {name} does not exist")
                scene_xml = model_dir / "scene.xml"
                primary_xml = model_dir / f"{name}.xml"
            self.assertTrue(scene_xml.is_file(), f"{name}/scene.xml not found")
            self.assertTrue(primary_xml.is_file(), f"{name}/{name}.xml not found")
        # task scenes across all 5 arm+gripper pairs (pick_place, box_pack) and xarm6 obstacles
        for arm, gripper in TASK_GRIPPERS.items():
            for task in ("pick_place", "box_pack"):
                path = resolve_arm_scene(arm, gripper, task)
                self.assertEqual(path.name, f"scene_{task}.xml")
                self.assertTrue(path.is_file(), f"{arm} {task} scene not found")
        self.assertTrue((MODELS_ROOT / "xarm6" / "scene_obstacles.xml").is_file())

    def test_all_models_load_and_step(self):
        """Every model, scene and task scene compiles and runs forward without warnings, then steps 100 times."""
        for name in ALL_MODELS:
            xmls = [resolve_model_path(name, scene=False), resolve_model_path(name, scene=True)]
            if (MODELS_ROOT / name / "scene_obstacles.xml").is_file():
                xmls.append(MODELS_ROOT / name / "scene_obstacles.xml")
            arm = name.removesuffix("_with_gripper")
            if TASK_GRIPPERS.get(arm) and name == f"{arm}_with_gripper":
                xmls += [resolve_arm_scene(arm, TASK_GRIPPERS[arm], task) for task in ("pick_place", "box_pack")]
            for xml_file in xmls:
                with self.subTest(model=name, xml=xml_file.name):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        m = mujoco.MjModel.from_xml_path(str(xml_file))
                    d = mujoco.MjData(m)
                    if m.nkey:
                        mujoco.mj_resetDataKeyframe(m, d, 0)
                    mujoco.mj_forward(m, d)
                    issues = [str(w.message) for w in caught] + [
                        f"{mujoco.mjtWarning(i).name}x{d.warning[i].number}"
                        for i in range(len(d.warning)) if d.warning[i].number
                    ]
                    self.assertEqual(issues, [], f"warnings for {xml_file}")
                    mujoco.mj_resetData(m, d)
                    mujoco.mj_forward(m, d)

                    # Simulate 100 steps
                    for _ in range(100):
                        mujoco.mj_step(m, d)

                    self.assertFalse(np.isnan(d.qpos).any(), f"NaN in qpos for {xml_file}")
                    self.assertFalse(np.isinf(d.qpos).any(), f"Inf in qpos for {xml_file}")
                    self.assertFalse(np.isnan(d.qvel).any(), f"NaN in qvel for {xml_file}")
                    self.assertFalse(np.isinf(d.qvel).any(), f"Inf in qvel for {xml_file}")

    def test_resolve_model_path_support(self):
        """Verify resolve_model_path and in-memory compose_model work across all models."""
        for name in ALL_MODELS:
            path = resolve_model_path(name)
            self.assertTrue(path.is_file(), f"Could not resolve model {name}")
            self.assertEqual(path.name, "scene.xml")
        m_mem = compose_model("xarm6_with_ft_sensor_and_gripper", scene=True)
        self.assertEqual(m_mem.nsensor, 2)

        # Nonexistent model should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            resolve_model_path("non_existent_robot_123")

    def test_keyframes_reset_and_step(self):
        """Test keyframe initialization for models that define home or ready poses."""
        for name in ALL_MODELS:
            scene_xml = resolve_model_path(name)
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

    def test_lite6_studio_factory_ready_pose(self):
        """Lite 6 ready matches Studio joint angles and the 200 mm flange pose."""
        expected_q = np.deg2rad([0, 9.9, 31.8, 0, 21.9, 0])
        for name in (
            "lite6",
            "lite6_with_gripper",
            "lite6_with_vacuum_gripper",
            "lite6_with_d435",
            "lite6_with_gripper_and_d435",
        ):
            with self.subTest(model=name):
                m = mujoco.MjModel.from_xml_path(str(resolve_model_path(name)))
                d = mujoco.MjData(m)
                ready_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "ready")
                self.assertGreaterEqual(ready_id, 0)
                mujoco.mj_resetDataKeyframe(m, d, ready_id)
                mujoco.mj_forward(m, d)
                np.testing.assert_allclose(d.qpos[:6], expected_q, atol=1e-10)
                np.testing.assert_allclose(d.ctrl[:6], expected_q, atol=1e-10)
                # Tool TCP offsets differ; the shared flange pose is the Studio reference.
                np.testing.assert_allclose(d.body("link6").xpos, [0.2, 0, 0.2], atol=0.001)
                np.testing.assert_allclose(
                    d.body("link6").xmat.reshape(3, 3), np.diag([1, -1, -1]), atol=1e-5,
                )

    def test_home_is_all_zero_and_xarm_tool_combos_start_from_init(self):
        """Every arm model's home is all-zero joints; floor-reaching combos add init (= ready)."""
        for name in ALL_MODELS:
            m = mujoco.MjModel.from_xml_path(str(resolve_model_path(name)))
            arm = [int(m.jnt_qposadr[j]) for j in range(m.njnt)
                   if re.fullmatch(r"joint\d+", m.joint(j).name or "")]
            if not arm:
                continue
            with self.subTest(model=name):
                home = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
                self.assertEqual(home, 0, "home must be keyframe 0")
                np.testing.assert_array_equal(m.key(home).qpos[arm], np.zeros(len(arm)))
                np.testing.assert_array_equal(m.key(home).ctrl[:len(arm)], np.zeros(len(arm)))
                recipe = RECIPES.get(name)
                init = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "init")
                self.assertEqual(init >= 0, needs_combo_init(recipe))
                if init < 0:
                    self.assertEqual(initial_keyframe(m), 0)
                    continue
                self.assertEqual(initial_keyframe(m), init)
                ready = m.key("ready")
                np.testing.assert_array_equal(m.key(init).qpos, ready.qpos)
                np.testing.assert_array_equal(m.key(init).ctrl, ready.ctrl)
                d = mujoco.MjData(m)
                mujoco.mj_resetDataKeyframe(m, d, init)
                mujoco.mj_forward(m, d)
                self.assertEqual(d.ncon, 0, f"{name} must start contact-free")

    def test_combo_ready_poses_point_down_at_300mm(self):
        """xArm 5/6/7 and UF850 combination ready poses point downward with G2 TCP at (300, 0, 300) mm."""
        for name in ("xarm5_with_gripper", "xarm6_with_gripper", "xarm7_with_gripper", "uf850_with_gripper"):
            with self.subTest(model=name):
                m = mujoco.MjModel.from_xml_path(str(resolve_model_path(name)))
                d = mujoco.MjData(m)
                ready_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "ready")
                self.assertGreaterEqual(ready_id, 0)
                mujoco.mj_resetDataKeyframe(m, d, ready_id)
                mujoco.mj_forward(m, d)
                site = d.site("attachment_site")
                np.testing.assert_allclose(site.xpos, [0.3, 0.0, 0.3], atol=1e-5)
                np.testing.assert_allclose(site.xmat.reshape(3, 3), np.diag([1, -1, -1]), atol=2e-5)

        last_links = {"xarm5": "link5", "xarm6": "link6", "xarm7": "link7", "uf850": "link6"}
        for arm, last_link in last_links.items():
            for suffix in (
                "gripper",
                "bio_gripper",
                "robotiq_2f_85",
                "robotiq_2f_140",
                "vacuum_gripper",
                "d435",
                "gripper_and_d435",
                "ft_sensor",
                "ft_sensor_and_gripper",
                "ft_sensor_and_vacuum_gripper",
            ):
                name = f"{arm}_with_{suffix}"
                with self.subTest(model=name):
                    m = mujoco.MjModel.from_xml_path(str(resolve_model_path(name)))
                    d = mujoco.MjData(m)
                    ready_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "ready")
                    self.assertGreaterEqual(ready_id, 0)
                    mujoco.mj_resetDataKeyframe(m, d, ready_id)
                    mujoco.mj_forward(m, d)
                    np.testing.assert_allclose(
                        d.body(last_link).xmat.reshape(3, 3), np.diag([1, -1, -1]), atol=2e-5,
                    )

    def test_standalone_remaining_grippers_flange_up_and_mounted_combos_unchanged(self):
        """Standalone vacuum and Lite6 grippers flip flange up (+Z) without altering mounted combos."""
        standalone = (
            ("vacuum_gripper", "xarm_vacuum_gripper_link", 0.20, "suction_site", 0.126),
            ("lite6_gripper", "uflite_gripper_link", 0.18, "link_tcp", 0.0836),
            ("lite6_vacuum_gripper", "uflite_vacuum_gripper_link", 0.17, "suction_site", 0.061),
        )
        for name, root_body, base_z, site_name, tcp_z in standalone:
            for scene in (False, True):
                with self.subTest(model=name, scene=scene):
                    m = mujoco.MjModel.from_xml_path(str(resolve_model_path(name, scene=scene)))
                    d = mujoco.MjData(m)
                    mujoco.mj_forward(m, d)
                    base_pos = d.body(root_body).xpos
                    base_mat = d.body(root_body).xmat.reshape(3, 3)
                    site_pos = d.site(site_name).xpos
                    np.testing.assert_allclose(base_pos, [0.0, 0.0, base_z], atol=1e-12)
                    np.testing.assert_allclose(base_mat, np.diag([1.0, -1.0, -1.0]), atol=1e-12)
                    np.testing.assert_allclose(site_pos, [0.0, 0.0, base_z - tcp_z], atol=1e-12)
                    self.assertLess(site_pos[2], base_pos[2])
                    self.assertEqual(d.ncon, 0)

        combos = (
            ("xarm5_with_vacuum_gripper", "link5", "xarm_vacuum_gripper_link", 0.126),
            ("xarm6_with_vacuum_gripper", "link6", "xarm_vacuum_gripper_link", 0.126),
            ("xarm7_with_vacuum_gripper", "link7", "xarm_vacuum_gripper_link", 0.126),
            ("uf850_with_vacuum_gripper", "link6", "xarm_vacuum_gripper_link", 0.126),
            ("xarm6_with_ft_sensor_and_vacuum_gripper", "ft_sensor_output", "xarm_vacuum_gripper_link", 0.126),
            ("lite6_with_gripper", "link6", "uflite_gripper_link", 0.0836),
            ("lite6_with_vacuum_gripper", "link6", "uflite_vacuum_gripper_link", 0.061),
        )
        for name, parent_body, tool_body, tcp_z in combos:
            with self.subTest(combo=name):
                m = compose_model(name, scene=True)
                d = mujoco.MjData(m)
                mujoco.mj_forward(m, d)
                parent = d.body(parent_body)
                parent_pos = parent.xpos
                parent_rot = parent.xmat.reshape(3, 3)
                tool = d.body(tool_body)
                np.testing.assert_allclose(parent_rot.T @ (tool.xpos - parent_pos), [0.0, 0.0, 0.0], atol=1e-12)
                np.testing.assert_allclose(parent_rot.T @ tool.xmat.reshape(3, 3), np.eye(3), atol=1e-12)
                tcp_in_parent = parent_rot.T @ (d.site("attachment_site").xpos - parent_pos)
                np.testing.assert_allclose(tcp_in_parent, [0.0, 0.0, tcp_z], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
