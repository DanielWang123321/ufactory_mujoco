"""BIO regression against the saved official movable URDF.

Reference: xArm-Developer/ufactory_genesis, commit
c7495bd857881a96a044f6eed21f062efd6d89d5,
assets/urdf/bio_gripper_g2/bio_gripper_g2_movable_visual.urdf.
"""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from scripts import build_all_mjcf

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "tests/fixtures/bio_gripper_g2_movable_visual.urdf"
MODEL_NAMES = ("bio_gripper_g2", "xarm6_with_bio_gripper")


class BioModelTest(unittest.TestCase):
    def test_mass_com_and_compiled_full_inertia_match_reference(self):
        reference = ET.parse(REFERENCE).getroot()
        for name in MODEL_NAMES:
            model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / name / f"{name}.xml"))
            for link in reference.findall("link"):
                inertial = link.find("inertial")
                if inertial is None:
                    continue
                with self.subTest(model=name, body=link.get("name")):
                    body = model.body(link.get("name"))
                    self.assertAlmostEqual(body.mass[0], float(inertial.find("mass").get("value")), places=12)
                    com = np.fromstring(inertial.find("origin").get("xyz"), sep=" ")
                    np.testing.assert_allclose(body.ipos, com, rtol=0, atol=1e-12)
                    inertia = inertial.find("inertia")
                    xx, yy, zz, xy, xz, yz = [float(inertia.get(k))
                                             for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
                    expected = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
                    rotation = np.empty(9)
                    mujoco.mju_quat2Mat(rotation, body.iquat)
                    rotation = rotation.reshape(3, 3)
                    compiled = rotation @ np.diag(body.inertia) @ rotation.T
                    np.testing.assert_allclose(compiled, expected, rtol=0, atol=1e-12)

    def test_tcp_and_collision_poses_match_reference(self):
        reference = ET.parse(REFERENCE).getroot()
        tcp = reference.find("joint[@name='bio_gripper_g2_tcp_joint']/origin")
        for name in MODEL_NAMES:
            path = ROOT / "models" / name / f"{name}.xml"
            xml = ET.parse(path).getroot()
            model = mujoco.MjModel.from_xml_path(str(path))
            site_name = "link_tcp" if name == "bio_gripper_g2" else "attachment_site"
            np.testing.assert_allclose(model.site(site_name).pos,
                                       np.fromstring(tcp.get("xyz"), sep=" "), rtol=0, atol=1e-12)
            self.assertEqual(model.body(int(model.site(site_name).bodyid[0])).name,
                             "bio_gripper_g2_base_link")
            for link in reference.findall("link"):
                collision = link.find("collision/origin")
                if collision is None:
                    continue
                with self.subTest(model=name, body=link.get("name")):
                    expected = np.fromstring(collision.get("xyz"), sep=" ")
                    element = xml.find(f".//body[@name='{link.get('name')}']")
                    geom = next(g for g in element.findall("geom") if "collision" in g.get("class", ""))
                    np.testing.assert_allclose(np.fromstring(geom.get("pos"), sep=" "),
                                               expected, rtol=0, atol=1e-12)
                    body_id = model.body(link.get("name")).id
                    geom_ids = np.flatnonzero((model.geom_bodyid == body_id) & (model.geom_group == 3))
                    self.assertEqual(len(geom_ids), 1)
                    gid = geom_ids[0]
                    # MuJoCo recenters mesh vertices and adds that shift to geom.pos.
                    mesh_id = model.geom_dataid[gid]
                    np.testing.assert_allclose(model.geom_pos[gid] - model.mesh_pos[mesh_id],
                                               expected, rtol=0, atol=1e-12)

    def test_generated_models_match_builder_templates(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            models_dir = Path(tmp)
            with patch.object(build_all_mjcf, "MODELS_DIR", models_dir), contextlib.redirect_stdout(io.StringIO()):
                build_all_mjcf.build_bio_gripper_g2()
                build_all_mjcf.build_xarm6_with_bio_gripper()
            for name in MODEL_NAMES:
                for filename in (f"{name}.xml", "scene.xml"):
                    with self.subTest(model=name, file=filename):
                        self.assertEqual((models_dir / name / filename).read_bytes(),
                                         (ROOT / "models" / name / filename).read_bytes())

    def test_standalone_gripper_closes_and_opens(self):
        for filename in ("bio_gripper_g2.xml", "scene.xml"):
            with self.subTest(file=filename):
                path = ROOT / "models/bio_gripper_g2" / filename
                model = mujoco.MjModel.from_xml_path(str(path))
                data = mujoco.MjData(model)
                for target in (-0.0395, 0, -0.01975, -0.0395, 0):
                    data.ctrl[0] = target
                    for _ in range(1000):
                        mujoco.mj_step(model, data)
                    np.testing.assert_allclose(data.qpos, [target, -target], rtol=0, atol=5e-5)
                    self.assertTrue(all(contact.dist >= -1e-6 for contact in data.contact))


if __name__ == "__main__":
    unittest.main()
