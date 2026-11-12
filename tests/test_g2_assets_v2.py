"""Check the G2 source snapshot, flange frame and 85 CAD reference poses."""

import csv
import hashlib
import json
from pathlib import Path
import re
import struct
import unittest

import mujoco
import numpy as np

from ufactory_mujoco.g2_model import G2_JOINT_NAMES, gripper_g2_gap_m_to_sim_drive
from ufactory_mujoco.grippers import measured_g2_pad_gap_mm
from ufactory_mujoco.sim_robot import SimRobot

try:
    import pytest
except ImportError as exc:
    raise unittest.SkipTest("pytest is required for test_g2_assets_v2") from exc

ROOT = Path(__file__).resolve().parents[1]
G2 = ROOT / "models/gripper_g2"
ASSETS = G2 / "assets/gripper_g2"
SOURCE = G2 / "source/gripper_g2"


def _vertices(path):
    if path.suffix == ".obj":
        with path.open() as stream:
            return np.array([[float(x) for x in line.split()[1:4]] for line in stream if line.startswith("v ")])
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    triangles = np.frombuffer(data, dtype=np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)),
                                                    ("attr", "<u2")]), count=count, offset=84)
    return triangles["vertices"].reshape(-1, 3).astype(float)


def test_snapshot_file_hashes():
    metadata = json.loads((SOURCE / "snapshot.json").read_text())
    for name, expected in metadata["files_sha256"].items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == expected
    reference = G2 / "source/reference_opening_axial_length.csv"
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == metadata["reference_axial_length_csv"]["sha256"]


def test_base_flange_and_mesh_correspondence():
    for name in ("base", "left_finger", "right_finger", "left_inner_knuckle", "right_inner_knuckle",
                 "left_outer_knuckle", "right_outer_knuckle"):
        visual = _vertices(ASSETS / f"visual/{name}.obj")
        if name == "base":
            base_stls = sorted((ASSETS / "collision").glob("base_link_col_*.stl"))
            assert len(base_stls) >= 2
            collision = np.vstack([_vertices(p) for p in base_stls])
        else:
            collision = _vertices(ASSETS / f"collision/{name}.stl")
        bounds_v = np.array([visual.min(axis=0), visual.max(axis=0)])
        bounds_c = np.array([collision.min(axis=0), collision.max(axis=0)])
        # Collision hulls omit decorative details; Y/Z bounds retain the linkage and pad geometry.
        np.testing.assert_allclose(bounds_v[:, 1:], bounds_c[:, 1:], atol=0.00008)
        if name == "base":
            np.testing.assert_allclose(bounds_v.mean(axis=0)[:2], [0, 0], atol=0.00001)
            np.testing.assert_allclose(bounds_v[:, 2], [-0.005, 0.1045], atol=0.00001)
            np.testing.assert_allclose(np.ptp(bounds_v, axis=0)[:2], [0.075, 0.109559], atol=0.00001)


def test_converted_visual_matches_draco_snapshot():
    draco = pytest.importorskip("DracoPy")
    gltf_lib = pytest.importorskip("pygltflib")
    for obj in sorted((ASSETS / "visual").glob("*.obj")):
        glb = gltf_lib.GLTF2().load(str(SOURCE / f"meshes/visual/visual_glb/{obj.stem}.glb"))
        blob = glb.binary_blob()
        points = []
        for mesh in glb.meshes:
            for primitive in mesh.primitives:
                extension = primitive.extensions["KHR_draco_mesh_compression"]
                view = glb.bufferViews[extension["bufferView"]]
                start = view.byteOffset or 0
                points.append(np.asarray(draco.decode(blob[start:start + view.byteLength]).points))
        source = np.vstack(points)
        converted = _vertices(obj)
        assert len(source) == len(converted)
        np.testing.assert_allclose([converted.min(axis=0), converted.max(axis=0)],
                                   [source.min(axis=0), source.max(axis=0)], atol=1e-7)


def test_85_reference_openings_and_axial_lengths():
    rows = list(csv.DictReader((G2 / "source/reference_opening_axial_length.csv").open(encoding="utf-8-sig")))
    assert len(rows) == 85
    model = mujoco.MjModel.from_xml_path(str(G2 / "gripper_g2.xml"))
    data = mujoco.MjData(model)
    for name, pos in [("left_outer_knuckle", [0, 0.035, 0.0606]),
                      ("right_outer_knuckle", [0, -0.035, 0.0606]),
                      ("left_inner_knuckle", [0, 0.02, 0.075602]),
                      ("right_inner_knuckle", [0, -0.02, 0.075602])]:
        np.testing.assert_allclose(model.body(name).pos, pos, atol=1e-12)
    np.testing.assert_allclose(model.site("link_tcp").pos, [0, 0, 0.1735], atol=1e-12)
    vertices = {(side, kind): _vertices(ASSETS / f"{kind}/{side}_finger.{'obj' if kind == 'visual' else 'stl'}")
                for side in ("left", "right") for kind in ("visual", "collision")}
    for row in rows:
        gap = float(row["opening_mm"])
        q = gripper_g2_gap_m_to_sim_drive(gap / 1000)
        assert q == pytest.approx(float(row["drive_angle_rad"]), abs=1e-8)
        for name in G2_JOINT_NAMES:
            data.joint(name).qpos[0] = q
        mujoco.mj_forward(model, data)
        assert measured_g2_pad_gap_mm(model, data) == pytest.approx(gap, abs=0.005)
        for kind in ("visual", "collision"):
            surfaces = []
            for side in ("left", "right"):
                body = data.body(f"{side}_finger")
                surfaces.append(vertices[(side, kind)] @ body.xmat.reshape(3, 3).T + body.xpos)
            mesh_gap = (surfaces[0][:, 1].min() - surfaces[1][:, 1].max()) * 1000
            axial = max(surface[:, 2].max() for surface in surfaces) * 1000
            assert mesh_gap == pytest.approx(gap, abs=0.005)
            assert axial == pytest.approx(float(row["axial_length_mm"]), abs=0.005)


@pytest.mark.parametrize("name", ["gripper_g2", "xarm5_with_gripper", "xarm6_with_gripper",
                                   "xarm7_with_gripper", "uf850_with_gripper"])
def test_open_close_and_contact_force_on_each_model(name, tmp_path):
    source = ROOT / "models" / name / f"{name}.xml"
    with SimRobot(source) as robot:
        robot.set_gripper_command(0, speed_percent=100, force_percent=80)
        robot.settle(1.0)
        assert robot.get_measured_gripper_gap() < 0.003
        robot.set_gripper_command(84, speed_percent=100, force_percent=80)
        robot.settle(1.0)
        assert robot.get_measured_gripper_gap() > 0.081

    # A fixed load cell in the gripper's own frame isolates pinch force from
    # the arm's pose and keeps the contact test identical across all models.
    xml = source.read_text().replace('meshdir="assets"', f'meshdir="{source.parent / "assets"}"')
    xml, count = re.subn(
        r'(<body name="xarm_gripper_base_link"[^>]*>)',
        r'\1\n<body name="loadcell" pos="0 0 0.1563"><geom type="box" '
        r'size="0.012 0.015 0.012" friction="1 1 0.005"/></body>', xml, count=1,
    )
    assert count == 1
    test_scene = tmp_path / f"{name}.xml"
    test_scene.write_text(xml)
    forces, limits = [], []
    with SimRobot(test_scene) as robot:
        for percent in (10, 50, 100):
            robot.set_gripper_command(84, speed_percent=100, force_percent=percent)
            robot.settle(1.0)
            robot.set_gripper_command(0, speed_percent=50, force_percent=percent)
            robot.settle(1.0)
            left, right = robot.get_gripper_contact_forces("loadcell")
            assert min(left, right) > 1
            forces.append((left + right) / 2)
            limits.append(float(robot.get_gripper_control_output().torque_limit_nm[0]))
    assert forces[0] < forces[1] < forces[2]
    assert limits[0] < limits[1] < limits[2]
