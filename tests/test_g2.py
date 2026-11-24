"""Gripper G2 v2 asset geometry, linkage, force curve, and closed-loop controller tests."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import struct
import unittest
import mujoco
import numpy as np
from ufactory_mujoco.g2 import gripper_g2_gap_m_to_sim_drive
from ufactory_mujoco.grippers import gripper_object_contact_forces_n, identify_gripper, measured_g2_pad_gap_mm
from ufactory_mujoco.sim_robot import SimRobot
from ufactory_mujoco import compose_model, compose_spec, resolve_model_path
from dataclasses import asdict
import math
from ufactory_mujoco.g2 import G2Controller
from ufactory_mujoco.g2 import (
    G2_KP,
    G2_KV,
    G2_LINKAGE_GEOMETRY,
    GRIPPER_G2_SIM_CLOSE_DRIVE,
    GRIPPER_G2_SIM_OPEN_DRIVE,
    REFERENCE_GAP_MM,
    REFERENCE_RAW_POSITION,
    G2Model,
    gripper_g2_sim_drive_to_gap_m,
    sdk_position_counts,
    sdk_position_mm,
    sdk_speed_rpm,
    speed_mm_s,
)
from ufactory_mujoco.robots import MODELS_ROOT
from ufactory_mujoco.preflight import GRIP_DOWN_QUAT, tcp_task_heights

# ==========================================================================
# From test_g2_assets_v2.py
# ==========================================================================

try:
    import pytest
except ImportError as exc:
    raise unittest.SkipTest("pytest is required for test_g2_assets_v2") from exc

ROOT = Path(__file__).resolve().parents[1]

# Calibrated Gripper G2 joint parameters that every G2 MJCF must carry.
G2_JOINT_NAMES = ("left_driver_joint", "left_finger_joint", "left_inner_knuckle_joint",
                  "right_driver_joint", "right_finger_joint", "right_inner_knuckle_joint")
G2_ARMATURE, G2_DAMPING, G2_FRICTIONLOSS = 0.01, 0.05, 0.0
G2_MIMIC_SOLREF = (0.004, 1.0)
G2_MIMIC_SOLIMP = (0.9999, 0.9999, 0.001, 0.5, 2.0)

G2 = MODELS_ROOT / "gripper_g2"

ASSETS = G2 / "assets/gripper_g2"

SOURCE = ROOT / "dev/g2_sources/gripper_g2"

REFERENCE_CSV = ROOT / "tests/fixtures/g2_reference_opening_axial_length.csv"

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
    if not SOURCE.is_dir():
        pytest.skip("dev/g2_sources snapshot not present")
    metadata = json.loads((SOURCE / "snapshot.json").read_text())
    for name, expected in metadata["files_sha256"].items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == expected
    assert hashlib.sha256(REFERENCE_CSV.read_bytes()).hexdigest() == metadata["reference_axial_length_csv"]["sha256"]

def _classify_glb_material(pbr: dict) -> str:
    color = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
    metal = float(pbr.get("metallicFactor", 0.0))
    rgb = color[:3]
    if metal > 0.5:
        return "anodized_black" if max(rgb) < 0.3 else "silver"
    if len(color) >= 4 and color[3] < 0.99 and min(rgb) > 0.8:
        return "silicone"
    if max(rgb) < 0.6:
        return "black"
    return "white"


def _load_front_half_obj(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load vertices (mm), front-half vertex normals, and front-half triangle vertex indices."""
    verts, normals, faces = [], [], []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vn "):
                normals.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                tokens = line.split()[1:]
                v_idx = [int(tok.split("/")[0]) - 1 for tok in tokens]
                n_idx = [int(tok.split("/")[2]) - 1 for tok in tokens]
                faces.append((v_idx, n_idx))
    half_faces = faces[: len(faces) // 2]
    half_normals = np.asarray(normals[: len(normals) // 2], dtype=np.float64)
    v_arr = np.asarray(verts, dtype=np.float64) * 1000.0
    fv_arr = np.asarray([fv for fv, _ in half_faces], dtype=np.int64)
    fn_arr = np.asarray([fn for _, fn in half_faces], dtype=np.int64)
    vert_normals = np.zeros_like(v_arr)
    vert_normals[fv_arr] = half_normals[fn_arr]
    return v_arr, vert_normals, fv_arr


def _section_mm(tri_mm: np.ndarray, z_mm: float) -> np.ndarray:
    """Intersect triangles (mm) with horizontal plane z = z_mm and return 2D XY points (mm)."""
    segments = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        pa, pb = tri_mm[:, i], tri_mm[:, j]
        za, zb = pa[:, 2] - z_mm, pb[:, 2] - z_mm
        mask = (za * zb <= 0.0) & (np.abs(zb - za) > 1e-12)
        if np.any(mask):
            t = (-za[mask] / (zb[mask] - za[mask]))[:, None]
            segments.append(pa[mask, :2] + t * (pb[mask, :2] - pa[mask, :2]))
    return np.concatenate(segments, axis=0)


_G2_FLANGE_CIRCLES = (
    ("boss", (0.0, 0.0), 15.75, (-4.4, 1.4), 1, -1.5, None),
    ("flange outer", (0.0, 0.0), 37.50, (0.4, 7.9), 1, 4.5, (117.0, 423.0)),
    ("boss pocket", (0.0, 0.0), 15.75, (0.35, 1.95), -1, 1.15, None),
    ("pin hole", (0.0, -25.0), 3.00, (0.35, 1.95), -1, 1.15, None),
    ("cover outer", (0.0, 0.0), 37.50, (0.1, 8.0), 1, 4.5, (67.0, 113.0)),
    ("cover fillet +x", (5.5, 27.15), 10.00, (0.1, 2.8), 1, 1.45, (272.0, 358.0)),
    ("cover fillet -x", (-5.5, 27.15), 10.00, (0.1, 2.8), 1, 1.45, (182.0, 268.0)),
)


def test_base_flange_and_mesh_correspondence():
    for name in ("base", "left_finger", "right_finger", "left_inner_knuckle", "right_inner_knuckle",
                 "left_outer_knuckle", "right_outer_knuckle"):
        if name == "base":
            base_objs = sorted((ASSETS / "visual").glob("base_*.obj"))
            assert [p.name for p in base_objs] == [
                "base_anodized_black.obj",
                "base_black.obj",
                "base_silver.obj",
                "base_white.obj",
            ]
            visual = np.vstack([_vertices(p) for p in base_objs])
            base_stls = sorted((ASSETS / "collision").glob("base_link_col_*.stl"))
            assert len(base_stls) >= 2
            collision = np.vstack([_vertices(p) for p in base_stls])
        elif name in ("left_finger", "right_finger"):
            finger_objs = sorted((ASSETS / "visual").glob(f"{name}_*.obj"))
            assert [p.name for p in finger_objs] == [f"{name}_black.obj", f"{name}_silicone.obj"]
            visual = np.vstack([_vertices(p) for p in finger_objs])
            collision = _vertices(ASSETS / f"collision/{name}.stl")
        else:
            visual = _vertices(ASSETS / f"visual/{name}.obj")
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
    if not SOURCE.is_dir():
        pytest.skip("dev/g2_sources snapshot not present")
    draco = pytest.importorskip("DracoPy")
    gltf_lib = pytest.importorskip("pygltflib")

    split_parts: dict[str, np.ndarray] = {}
    for stem in ("base", "left_finger", "right_finger"):
        glb = gltf_lib.GLTF2().load(str(SOURCE / f"meshes/visual/visual_glb/{stem}.glb"))
        blob = glb.binary_blob()
        for mesh in glb.meshes:
            for primitive in mesh.primitives:
                mat = glb.materials[primitive.material]
                pbr = mat.pbrMetallicRoughness
                cat = _classify_glb_material(
                    {
                        "baseColorFactor": pbr.baseColorFactor or [1.0, 1.0, 1.0, 1.0],
                        "metallicFactor": pbr.metallicFactor or 0.0,
                    }
                )
                extension = primitive.extensions["KHR_draco_mesh_compression"]
                view = glb.bufferViews[extension["bufferView"]]
                start = view.byteOffset or 0
                split_parts[f"{stem}_{cat}"] = np.asarray(draco.decode(blob[start:start + view.byteLength]).points)

    for obj in sorted((ASSETS / "visual").glob("*.obj")):
        if obj.stem in split_parts:
            source = split_parts[obj.stem]
        else:
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
    rows = list(csv.DictReader(REFERENCE_CSV.open(encoding="utf-8-sig")))
    assert len(rows) == 85
    model = mujoco.MjModel.from_xml_path(str(G2 / "gripper_g2.xml"))
    data = mujoco.MjData(model)
    for name, pos in [("left_outer_knuckle", [0, 0.035, 0.0606]),
                      ("right_outer_knuckle", [0, -0.035, 0.0606]),
                      ("left_inner_knuckle", [0, 0.02, 0.075602]),
                      ("right_inner_knuckle", [0, -0.02, 0.075602])]:
        np.testing.assert_allclose(model.body(name).pos, pos, atol=1e-12)
    np.testing.assert_allclose(model.site("link_tcp").pos, [0, 0, 0.1735], atol=1e-12)
    vertices = {
        (side, kind): (
            np.vstack([_vertices(ASSETS / f"visual/{side}_finger_{part}.obj") for part in ("black", "silicone")])
            if kind == "visual"
            else _vertices(ASSETS / f"collision/{side}_finger.stl")
        )
        for side in ("left", "right") for kind in ("visual", "collision")
    }
    for row in rows:
        gap = float(row["opening_mm"])
        q = gripper_g2_gap_m_to_sim_drive(gap / 1000)
        assert q == pytest.approx(float(row["drive_angle_rad"]), abs=1e-8)
        for name in G2_JOINT_NAMES:
            data.joint(name).qpos[0] = q
        mujoco.mj_forward(model, data)
        assert measured_g2_pad_gap_mm(model, data) == pytest.approx(gap, abs=0.005)
        base = data.body("xarm_gripper_base_link")
        base_pos = base.xpos
        base_mat = base.xmat.reshape(3, 3)
        for kind in ("visual", "collision"):
            surfaces = []
            for side in ("left", "right"):
                body = data.body(f"{side}_finger")
                world = vertices[(side, kind)] @ body.xmat.reshape(3, 3).T + body.xpos
                surfaces.append((world - base_pos) @ base_mat)
            mesh_gap = (surfaces[0][:, 1].min() - surfaces[1][:, 1].max()) * 1000
            axial = max(surface[:, 2].max() for surface in surfaces) * 1000
            assert mesh_gap == pytest.approx(gap, abs=0.005)
            assert axial == pytest.approx(float(row["axial_length_mm"]), abs=0.005)

def test_standalone_orientation_flange_up_tcp_down():
    """Standalone gripper_g2 flips flange up (+Z) and TCP down without changing mounted combos."""
    for filename in ("gripper_g2.xml", "scene.xml"):
        m = mujoco.MjModel.from_xml_path(str(G2 / filename))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        base_pos = d.body("xarm_gripper_base_link").xpos
        base_mat = d.body("xarm_gripper_base_link").xmat.reshape(3, 3)
        tcp_pos = d.site("link_tcp").xpos
        np.testing.assert_allclose(base_pos, [0.0, 0.0, 0.23], atol=1e-12)
        np.testing.assert_allclose(base_mat, np.diag([1.0, -1.0, -1.0]), atol=1e-12)
        np.testing.assert_allclose(tcp_pos, [0.0, 0.0, 0.23 - 0.1735], atol=1e-12)
        assert tcp_pos[2] < base_pos[2]
        assert d.ncon == 0

    flange_links = {
        "xarm5": "link5",
        "xarm6": "link6",
        "xarm7": "link7",
        "uf850": "link6",
    }
    for arm, flange_link in flange_links.items():
        name = f"{arm}_with_gripper"
        m = compose_model(name, scene=True)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        flange_pos = d.body(flange_link).xpos
        flange_rot = d.body(flange_link).xmat.reshape(3, 3)
        base = d.body("xarm_gripper_base_link")
        np.testing.assert_allclose(flange_rot.T @ (base.xpos - flange_pos), [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(flange_rot.T @ base.xmat.reshape(3, 3), np.eye(3), atol=1e-12)
        tcp_in_flange = flange_rot.T @ (d.site("attachment_site").xpos - flange_pos)
        np.testing.assert_allclose(tcp_in_flange, [0.0, 0.0, 0.1735], atol=1e-12)

def test_base_materials_and_scene_specular():
    """G2 base parts bind 4 distinct materials and soft scene specular without breaking F/T scenes."""
    for name in ("gripper_g2", "xarm5_with_gripper", "xarm6_with_gripper", "xarm7_with_gripper", "uf850_with_gripper"):
        m = mujoco.MjModel.from_xml_path(str(resolve_model_path(name, scene=True)))
        alu_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "g2_anodized_aluminum")
        ab_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "g2_anodized_black")
        white_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "white")
        black_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "black")
        assert min(alu_id, ab_id, white_id, black_id) >= 0

        np.testing.assert_allclose(m.mat_rgba[alu_id], [0.60, 0.62, 0.64, 1.0], atol=1e-6)
        assert float(m.mat_specular[alu_id]) == pytest.approx(0.72, abs=1e-6)
        assert float(m.mat_shininess[alu_id]) == pytest.approx(0.16, abs=1e-6)
        assert float(m.mat_reflectance[alu_id]) == pytest.approx(0.03, abs=1e-6)
        assert float(m.mat_metallic[alu_id]) == pytest.approx(0.8, abs=1e-6)
        assert float(m.mat_roughness[alu_id]) == pytest.approx(0.64, abs=1e-6)

        np.testing.assert_allclose(m.mat_rgba[ab_id], [0.12, 0.12, 0.125, 1.0], atol=1e-6)
        assert float(m.mat_specular[ab_id]) == pytest.approx(0.50, abs=1e-6)
        assert float(m.mat_shininess[ab_id]) == pytest.approx(0.18, abs=1e-6)
        assert float(m.mat_reflectance[ab_id]) == pytest.approx(0.02, abs=1e-6)
        assert float(m.mat_metallic[ab_id]) == pytest.approx(0.8, abs=1e-6)
        assert float(m.mat_roughness[ab_id]) == pytest.approx(0.62, abs=1e-6)

        base_id = m.body("xarm_gripper_base_link").id
        vis_geoms = np.flatnonzero((m.geom_bodyid == base_id) & (m.geom_group == 2))
        prefix = "" if name == "gripper_g2" else "g2_"
        mats = {
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[gid])): int(m.geom_matid[gid])
            for gid in vis_geoms
        }
        assert mats == {
            f"{prefix}base_white_vis": white_id,
            f"{prefix}base_silver_vis": alu_id,
            f"{prefix}base_black_vis": black_id,
            f"{prefix}base_anodized_black_vis": ab_id,
        }
        np.testing.assert_allclose(m.vis.headlight.specular, [0.35, 0.35, 0.38], atol=1e-6)

    m_ft = mujoco.MjModel.from_xml_path(str(resolve_model_path("xarm6_with_ft_sensor_and_gripper", scene=True)))
    np.testing.assert_allclose(m_ft.vis.headlight.specular, [0.55, 0.57, 0.6], atol=1e-6)


def test_finger_silicone_pad_material_and_geometry():
    """G2 finger inner pads bind single-sided translucent g2_silicone and match CAD bounds."""
    for name in (
        "gripper_g2",
        "xarm5_with_gripper",
        "xarm6_with_gripper",
        "xarm7_with_gripper",
        "uf850_with_gripper",
        "xarm6_with_ft_sensor_and_gripper",
    ):
        m = mujoco.MjModel.from_xml_path(str(resolve_model_path(name, scene=True)))
        sil_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "g2_silicone")
        black_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "black")
        assert min(sil_id, black_id) >= 0

        np.testing.assert_allclose(m.mat_rgba[sil_id], [0.92, 0.92, 0.90, 0.75], atol=1e-6)
        assert float(m.mat_specular[sil_id]) == pytest.approx(0.45, abs=1e-6)
        assert float(m.mat_shininess[sil_id]) == pytest.approx(0.55, abs=1e-6)
        assert float(m.mat_metallic[sil_id]) == pytest.approx(0.0, abs=1e-6)
        assert float(m.mat_roughness[sil_id]) == pytest.approx(0.30, abs=1e-6)

        prefix = "" if name == "gripper_g2" else "g2_"
        for side in ("left", "right"):
            body_id = m.body(f"{side}_finger").id
            vis_geoms = np.flatnonzero((m.geom_bodyid == body_id) & (m.geom_group == 2))
            mats = {
                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[gid])): int(m.geom_matid[gid])
                for gid in vis_geoms
            }
            assert mats == {
                f"{prefix}{side}_finger_black_vis": black_id,
                f"{prefix}{side}_finger_silicone_vis": sil_id,
            }

    for side in ("left", "right"):
        obj_path = ASSETS / f"visual/{side}_finger_silicone.obj"
        lines = obj_path.read_text(encoding="utf-8").splitlines()
        verts = np.array([[float(x) for x in line.split()[1:4]] for line in lines if line.startswith("v ")]) * 1000.0
        n_faces = sum(line.startswith("f ") for line in lines)
        assert len(verts) == 338
        assert n_faces == 320
        want_y_min, want_y_max = (-26.0, -23.0) if side == "left" else (23.0, 26.0)
        assert verts.min(0) == pytest.approx([-14.75, want_y_min, 22.26], abs=0.05)
        assert verts.max(0) == pytest.approx([14.75, want_y_max, 59.75], abs=0.05)
        inner_y_m = float(verts[:, 1].min() if side == "left" else -verts[:, 1].max()) * 0.001
        assert inner_y_m == pytest.approx(G2_LINKAGE_GEOMETRY.pad_inner_y_m, abs=5e-5)


def test_flange_outer_circle_boss_and_locating_pin_geometry():
    """Base flange outer rim, boss, cover, bearing cover, and slotted round pin match analytic CAD geometry."""
    silver_pts, silver_nrm, silver_faces = _load_front_half_obj(ASSETS / "visual/base_silver.obj")
    black_pts, black_nrm, black_faces = _load_front_half_obj(ASSETS / "visual/base_black.obj")
    bearing_pts, _, _ = _load_front_half_obj(ASSETS / "visual/base_anodized_black.obj")

    assert bearing_pts[:, 0].min() > -14.6 and bearing_pts[:, 0].max() < 14.4
    assert bearing_pts[:, 1].min() > -15.1 and bearing_pts[:, 1].max() < 15.1
    assert bearing_pts[:, 2].min() == pytest.approx(92.94, abs=0.05)
    assert bearing_pts[:, 2].max() == pytest.approx(101.94, abs=0.05)

    # 1. Locating pin at (0, -25) mm: r = 2.98 mm, 2 mm dome to z = -4.50 mm, -7.04 deg 1 mm slot.
    silver_tri = silver_pts[silver_faces]
    pin_axis = np.array([0.0, -25.0])
    pin_r = np.hypot(silver_tri[:, :, 0] - pin_axis[0], silver_tri[:, :, 1] - pin_axis[1])
    pin_tri = silver_tri[(pin_r.max(axis=1) <= 3.1) & (silver_tri[:, :, 2].min(axis=1) < -0.05)]
    assert len(pin_tri) > 200
    pin_pts = pin_tri.reshape(-1, 3)
    assert float(pin_pts[:, 2].min()) == pytest.approx(-4.50, abs=0.02)

    cyl_sec = _section_mm(pin_tri, -1.25)
    cyl_d = cyl_sec - pin_axis
    np.testing.assert_allclose(np.hypot(cyl_d[:, 0], cyl_d[:, 1]), 2.98, atol=0.02)
    pin_angles = np.sort(np.degrees(np.arctan2(cyl_d[:, 1], cyl_d[:, 0])) % 360.0)
    assert float(np.diff(np.r_[pin_angles, pin_angles[0] + 360.0]).max()) < 15.0

    yaw = -0.12283004
    c, s = np.cos(-yaw), np.sin(-yaw)
    rot2 = np.array([[c, -s], [s, c]])
    for z in np.linspace(-4.30, -2.65, 6):
        sec = _section_mm(pin_tri, z)
        assert sec.any() and float(np.hypot(sec[:, 0] - pin_axis[0], sec[:, 1] - pin_axis[1]).max()) <= 3.00
        uv = (sec - pin_axis) @ rot2.T
        assert not (np.abs(uv[:, 0]) < 0.45).any()
        assert (uv[:, 0] > 0.55).any() and (uv[:, 0] < -0.55).any()
    floor_pts = pin_pts[np.abs(pin_pts[:, 2] + 2.50) < 0.02]
    assert (np.abs(((floor_pts[:, :2] - pin_axis) @ rot2.T)[:, 0]) < 0.40).any()

    # 2. Analytic circular walls & vertex normals across flange, boss, pin hole, and connector cover.
    pts_all = np.vstack([silver_pts, black_pts])
    nrm_all = np.vstack([silver_nrm, black_nrm])
    tri_all = np.vstack([silver_tri, black_pts[black_faces]])
    fn = np.cross(tri_all[:, 1] - tri_all[:, 0], tri_all[:, 2] - tri_all[:, 0])
    fn_norm = np.linalg.norm(fn, axis=1)
    valid = fn_norm > 1e-10
    tri_all = tri_all[valid]
    fn = fn[valid] / fn_norm[valid, None]
    centers = tri_all.mean(axis=1)

    for label, xy, radius, (z_lo, z_hi), side, z_sec, arc in _G2_FLANGE_CIRCLES:
        axis = np.asarray(xy)
        dc = centers[:, :2] - axis
        rc = np.hypot(dc[:, 0], dc[:, 1])
        uc = dc / np.maximum(rc[:, None], 1e-12)
        ang_c = np.degrees(np.arctan2(uc[:, 1], uc[:, 0])) % 360.0
        sel = (
            (tri_all[:, :, 2].min(axis=1) >= z_lo - 0.5)
            & (tri_all[:, :, 2].max(axis=1) <= z_hi + 0.5)
            & (np.abs(rc - radius) < 0.8)
            & (side * (fn[:, :2] * uc).sum(axis=1) > 0.85)
            & (np.abs(fn[:, 2]) < 0.1)
        )
        if arc is not None:
            a0, a1 = arc
            sel &= ((ang_c >= a0) | (ang_c <= a1 - 360.0)) if a1 > 360.0 else ((ang_c >= a0) & (ang_c <= a1))
        assert sel.sum() >= 60, f"{label}: too few triangles ({sel.sum()})"
        sec = _section_mm(tri_all[sel], z_sec)
        sec_d = sec - axis
        assert float(np.abs(np.hypot(sec_d[:, 0], sec_d[:, 1]) - radius).max()) < 0.02, label
        sa = np.sort(np.degrees(np.arctan2(sec_d[:, 1], sec_d[:, 0])) % 360.0)
        diffs = (
            np.diff(np.sort((sa - arc[0]) % 360.0))
            if arc is not None and arc[1] > 360.0
            else (np.diff(sa) if arc is not None else np.diff(np.r_[sa, sa[0] + 360.0]))
        )
        assert float(diffs.max()) < 6.0, f"{label} gap {diffs.max():.2f} deg"

        dv = pts_all[:, :2] - axis
        rv = np.hypot(dv[:, 0], dv[:, 1])
        uv = dv / np.maximum(rv[:, None], 1e-12)
        ang_v = np.degrees(np.arctan2(uv[:, 1], uv[:, 0])) % 360.0
        vsel = (
            (pts_all[:, 2] >= z_lo - 0.15)
            & (pts_all[:, 2] <= z_hi + 0.15)
            & (np.abs(rv - radius) < 0.05)
            & (side * (nrm_all[:, :2] * uv).sum(axis=1) > 0.9)
            & (np.abs(nrm_all[:, 2]) < 0.1)
        )
        if arc is not None:
            a0, a1 = arc
            vsel &= ((ang_v >= a0) | (ang_v <= a1 - 360.0)) if a1 > 360.0 else ((ang_v >= a0) & (ang_v <= a1))
        assert vsel.sum() >= 60, f"{label} normals: too few vertices ({vsel.sum()})"
        expected_n = side * np.c_[uv[vsel], np.zeros(vsel.sum())]
        err_deg = np.degrees(np.arccos(np.clip((nrm_all[vsel] * expected_n).sum(axis=1), -1.0, 1.0)))
        assert float(err_deg.max()) < 1.0, f"{label} normal error {err_deg.max():.2f} deg"

    # 3. 360 deg flange outer wall partition between anodized aluminum and +Y black connector cover.
    grid = np.array([(i, j) for i in range(31) for j in range(31 - i)]) / 30.0
    weights = np.c_[grid, 1.0 - grid.sum(axis=1)]

    def outer_degrees(triangles: np.ndarray) -> set[int]:
        centre = triangles.mean(axis=1)
        side = (np.hypot(centre[:, 0], centre[:, 1]) > 37.0) & (centre[:, 2] > 0.5) & (centre[:, 2] < 7.5)
        sampled = np.einsum("sk,nkd->nsd", weights, triangles[side]).reshape(-1, 3)
        return set((np.degrees(np.arctan2(sampled[:, 1], sampled[:, 0])) // 1 % 360).astype(int).tolist())

    metal_deg = outer_degrees(silver_tri)
    cover_deg = outer_degrees(black_pts[black_faces])
    assert metal_deg | cover_deg == set(range(360))
    assert cover_deg <= set(range(64, 116)) and not (metal_deg & set(range(66, 114)))

@pytest.mark.parametrize("name", ["gripper_g2", "xarm5_with_gripper", "xarm6_with_gripper",
                                   "xarm7_with_gripper", "uf850_with_gripper"])
def test_open_close_and_contact_force_on_each_model(name):
    with SimRobot(compose_spec(name, scene=False)) as robot:
        robot.set_gripper_gap(0.0, speed_percent=100, force_percent=80)
        robot.settle(1.0)
        assert robot.get_gripper_gap(measured=True) < 0.003
        robot.set_gripper_gap(0.084, speed_percent=100, force_percent=80)
        robot.settle(1.0)
        assert robot.get_gripper_gap(measured=True) > 0.081

    # A fixed load cell in the gripper's own frame isolates pinch force from
    # the arm's pose and keeps the contact test identical across all models.
    spec = compose_spec(name, scene=False)
    loadcell = spec.body("xarm_gripper_base_link").add_body(name="loadcell", pos=[0, 0, 0.1563])
    loadcell.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.012, 0.015, 0.012], friction=[1, 1, 0.005])
    forces, limits = [], []
    with SimRobot(spec) as robot:
        for percent in (10, 50, 100):
            robot.set_gripper_gap(0.084, speed_percent=100, force_percent=percent)
            robot.settle(1.0)
            robot.set_gripper_gap(0.0, speed_percent=50, force_percent=percent)
            robot.settle(1.0)
            left, right = gripper_object_contact_forces_n(
                robot.model, robot.data, identify_gripper(robot.model), "loadcell")
            assert min(left, right) > 1
            forces.append((left + right) / 2)
            limits.append(float(robot.gripper_control_output.torque_limit_nm[0]))
    assert forces[0] < forces[1] < forces[2]
    assert limits[0] < limits[1] < limits[2]

# ==========================================================================
# From test_g2_calibrated.py
# ==========================================================================

try:
    import pytest
except ImportError as exc:
    raise unittest.SkipTest("pytest is required for test_g2_calibrated") from exc

REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "g2_controller_reference.json").read_text())

FIELDS = REFERENCE["step_fields"]

OUTPUTS = ("position_rad", "velocity_rad_s", "reference_gap_mm", "integral_torque_nm", "torque_limit_nm")

@pytest.fixture(scope="module")
def model() -> G2Model:
    return G2Model.load()

@pytest.mark.parametrize(
    "percent,mm_s,rpm",
    [(0, 0, 0), (1, 15, 577), (5, 15, 577), (10, 22.5, 691), (50, 112.5, 2057), (90, 202.5, 3423), (100, 225, 3765)],
)
def test_percent_speed_and_sdk_register_truncation(percent, mm_s, rpm):
    assert speed_mm_s(percent) == mm_s
    assert sdk_speed_rpm(mm_s) == rpm

def test_reference_retains_sdk_raw_count_and_inverse():
    assert REFERENCE_RAW_POSITION == 230
    assert REFERENCE_GAP_MM == pytest.approx(24.15587200800565, abs=1e-12)
    assert sdk_position_counts(0) == 0
    assert sdk_position_counts(84) == 850

@pytest.mark.parametrize("position", [0, 10, 24.155872, 30, 60, 84])
def test_sdk_position_quantization_is_at_most_one_count(position):
    raw = sdk_position_counts(position)
    restored = sdk_position_mm(raw)
    assert abs(restored - position) < 0.106

@pytest.mark.parametrize(
    "function,value",
    [
        (speed_mm_s, -1),
        (speed_mm_s, 101),
        (speed_mm_s, math.nan),
        (sdk_speed_rpm, 226),
        (sdk_position_counts, -0.1),
        (sdk_position_counts, 84.1),
        (sdk_position_counts, math.inf),
    ],
)
def test_invalid_units_are_rejected(function, value):
    with pytest.raises(ValueError):
        function(value)

def test_curve_retains_plateau_and_sharp_force_change(model):
    assert model.curve.lookup(3000, 1).force_n == model.curve.lookup(3000, 10).force_n
    assert model.curve.lookup(0, 55).force_n - model.curve.lookup(0, 50).force_n > 8
    assert model.curve.maximum_force_n == 63.79

def test_piecewise_current_and_speed_interpolation(model):
    a = model.curve.lookup(3000, 45).force_n
    b = model.curve.lookup(3000, 50).force_n
    assert model.curve.lookup(3000, 47.5).force_n == pytest.approx((a + b) / 2)
    at_3250 = model.curve.lookup(3250, 47.5).force_n
    assert model.curve.lookup(3125, 47.5).force_n == pytest.approx((a + b + 2 * at_3250) / 4)

@pytest.mark.parametrize("gap_mm", [0, 10, REFERENCE_GAP_MM, 30, 60, 84])
def test_physical_endpoints_inverse_and_gap_jacobian(model, gap_mm):
    linkage = model.linkage
    q = linkage.drive_for_gap_m(gap_mm / 1000)
    assert linkage.pad_gap_m(q) == pytest.approx(gap_mm / 1000, abs=1e-12)
    step = 1e-6
    numerical = (linkage.pad_gap_m(q - step) - linkage.pad_gap_m(q + step)) / (2 * step)
    assert linkage.gap_jacobian_m_per_rad(q) == pytest.approx(numerical, rel=1e-8)

def test_public_geometry_matches_the_urdf_collision_pads(model):
    assert asdict(model.linkage) == asdict(G2_LINKAGE_GEOMETRY)
    assert model.linkage.pad_gap_m(GRIPPER_G2_SIM_OPEN_DRIVE) * 1000 == pytest.approx(84, abs=1e-6)
    assert model.linkage.pad_gap_m(GRIPPER_G2_SIM_CLOSE_DRIVE) * 1000 == pytest.approx(0, abs=1e-6)

@pytest.mark.parametrize("gap_mm", [0, 10, REFERENCE_GAP_MM, 30, 60, 84])
def test_public_scalar_conversion_matches_pad_geometry(gap_mm):
    q = gripper_g2_gap_m_to_sim_drive(gap_mm / 1000)
    assert G2_LINKAGE_GEOMETRY.pad_gap_m(q) * 1000 == pytest.approx(gap_mm, abs=1e-6)
    assert gripper_g2_sim_drive_to_gap_m(q) * 1000 == pytest.approx(gap_mm, abs=1e-6)

def test_scalar_conversion_is_monotonic_and_clamps_to_nominal_stroke():
    gaps = (0, 10, REFERENCE_GAP_MM, 30, 60, 84)
    angles = [gripper_g2_gap_m_to_sim_drive(gap / 1000) for gap in gaps]
    assert all(a > b for a, b in zip(angles, angles[1:]))
    assert gripper_g2_gap_m_to_sim_drive(-0.01) == pytest.approx(GRIPPER_G2_SIM_CLOSE_DRIVE)
    assert gripper_g2_gap_m_to_sim_drive(0.1) == pytest.approx(GRIPPER_G2_SIM_OPEN_DRIVE)
    assert gripper_g2_sim_drive_to_gap_m(0) == pytest.approx(0.084)
    assert gripper_g2_sim_drive_to_gap_m(1) == pytest.approx(0)

@pytest.mark.parametrize("speed,force", [(10, 10), (50, 30), (50, 50), (90, 100)])
def test_two_pad_virtual_work_at_reference_gap(model, speed, force):
    lookup = model.curve.lookup(sdk_speed_rpm(speed_mm_s(speed)), force)
    linkage = model.linkage
    q = linkage.drive_for_gap_m(REFERENCE_GAP_MM / 1000)
    step = 1e-6
    full_gap_travel = linkage.pad_gap_m(q - step) - linkage.pad_gap_m(q + step)
    two_pad_work = 2 * lookup.force_n * full_gap_travel / 2
    joint_work = model.torque_limit_nm(lookup) * 2 * step
    assert joint_work == pytest.approx(two_pad_work, rel=1e-8)
    assert model.torque_limit_nm(lookup) == pytest.approx(lookup.force_n * model.reference_jacobian_m_rad, rel=1e-12)

@pytest.mark.parametrize("rpm,force,wide_gap_mm,narrow_gap_mm", [(2057, 50, 80, 20), (3500, 100, 70, 10)])
def test_fixed_torque_changes_force_with_gap(model, rpm, force, wide_gap_mm, narrow_gap_mm):
    lookup = model.curve.lookup(rpm, force)
    assert model.static_force_n(lookup, wide_gap_mm) > model.static_force_n(lookup, narrow_gap_mm)

def test_rated_torque_does_not_clip_curve(model):
    lookup = model.curve.lookup(3500, 100)
    tau = model.torque_limit_nm(lookup)
    assert model.rated_motor_torque_nm == 0.22
    assert model.motor_torque_estimate_nm(tau) > model.rated_motor_torque_nm
    assert model.motor_torque_estimate_nm(tau) <= model.peak_motor_torque_nm
    maximum = model.curve.lookup(0, 100)
    assert model.motor_torque_estimate_nm(model.torque_limit_nm(maximum)) == pytest.approx(0.5)

@pytest.mark.parametrize("dtype,tolerance_mm", [(np.float64, 1e-9), (np.float32, 2e-5)])
def test_array_geometry_matches_scalar_conversions(model, dtype, tolerance_mm):
    gaps = np.array([0, 10, REFERENCE_GAP_MM, 30, 60, 84], dtype=dtype)
    controller = G2Controller(model, num_envs=len(gaps), dtype=dtype)
    expected = np.array([gripper_g2_gap_m_to_sim_drive(g / 1000) for g in gaps.tolist()], dtype=dtype)
    q = controller.drive_for_gap_mm(gaps)
    np.testing.assert_allclose(q, expected, rtol=0, atol=tolerance_mm / 80)
    np.testing.assert_allclose(controller.gap_mm(q), gaps, rtol=0, atol=tolerance_mm)
    scalar_gaps = np.array([gripper_g2_sim_drive_to_gap_m(float(a)) for a in q], dtype=dtype)
    np.testing.assert_allclose(controller.gap_mm(q) / 1000, scalar_gaps, rtol=0, atol=tolerance_mm / 1000)
    assert bool((q[:-1] > q[1:]).all())

def _replay_reference(model, case, dtype, tolerance):
    controller = G2Controller(model, initial_position_mm=case["initial_position_mm"], dtype=dtype)
    if case["reset"] is not None:
        controller.reset(**case["reset"])
    for event in case["commands"]:
        controller.command(*event["command"])
        assert float(controller.requested_speed_mm_s[0]) == pytest.approx(event["requested_speed_mm_s"], abs=1e-9)
        assert int(controller.requested_speed_rpm[0]) == event["requested_speed_rpm"]
        for step in event["steps"]:
            row = dict(zip(FIELDS, step, strict=True))
            output = controller.update(
                row["q"], row["velocity"], REFERENCE["dt"], measured_gap_mm=row["measured_gap_mm"]
            )
            for field in OUTPUTS:
                assert float(getattr(output, field)[0]) == pytest.approx(row[field], abs=tolerance), field
            lookup = controller.force_lookup()
            assert lookup.force_n == pytest.approx(row["force_n"], abs=1e-12)
            assert lookup.calibration_rpm == row["calibration_rpm"]
            assert lookup.outside_calibration == bool(row["outside_calibration"])

@pytest.mark.parametrize("case", REFERENCE["cases"], ids=[case["name"] for case in REFERENCE["cases"]])
def test_controller_matches_recorded_scalar_reference(model, case):
    _replay_reference(model, case, np.float64, 1e-10)

@pytest.mark.parametrize("name", ["speed_force_grid", "reverse_mid_stroke"])
def test_float32_controller_tracks_recorded_scalar_reference(model, name):
    case = next(case for case in REFERENCE["cases"] if case["name"] == name)
    _replay_reference(model, case, np.float32, 2e-4)

@pytest.mark.parametrize("dtype,tolerance", [(np.float64, 0.0), (np.float32, 0.0)])
def test_batch_rows_match_independent_single_environment_controllers(model, dtype, tolerance):
    batch = G2Controller(model, num_envs=4, dtype=dtype)
    single = [G2Controller(model, dtype=dtype) for _ in range(4)]
    q = np.array([model.linkage.drive_for_gap_m(g / 1000) for g in (24, 30, 50, 70)], dtype=dtype)
    velocity = np.array([0, 1, -0.5, 0.01], dtype=dtype)
    gaps = [23, 29, 49, 69]
    for position, speeds, forces in [
        ([0, 10, 20, 30], [10, 50, 90, 100], [10, 30, 50, 100]),
        ([0, 10, 20, 30], [0, 0, 0, 0], [10, 30, 80, 100]),
        ([84, 84, 0, 0], [0, 50, 90, 100], [80, 30, 50, 10]),
        ([84, 84, 84, 84], [50, 90, 10, 50], [80, 50, 100, 30]),
    ]:
        batch.command(position, speeds, forces)
        for i, c in enumerate(single):
            c.command(position[i], speeds[i], forces[i])
        for _ in range(12):
            actual = batch.update(q, velocity, measured_gap_mm=gaps)
            expected = [
                c.update(q[i : i + 1], velocity[i : i + 1], measured_gap_mm=gaps[i]) for i, c in enumerate(single)
            ]
            for field in OUTPUTS:
                np.testing.assert_allclose(
                    getattr(actual, field),
                    np.concatenate([getattr(out, field) for out in expected]),
                    rtol=0,
                    atol=tolerance,
                )

def test_reset_is_independent_and_preheld_reset_seeds_torque(model):
    batch = G2Controller(model, num_envs=4)
    batch.command(0, [10, 50, 90, 100], [10, 30, 50, 100])
    batch.update(np.zeros(4), np.zeros(4))
    untouched = {
        key: value[[1, 3]].copy()
        for key, value in vars(batch).items()
        if isinstance(value, np.ndarray) and value.shape == (4,)
    }
    batch.reset([0, 2], position_mm=[30, 40], target_mm=[0, 10], prime=True)
    for key, value in untouched.items():
        np.testing.assert_array_equal(getattr(batch, key)[[1, 3]], value)
    np.testing.assert_array_equal(batch.integral_torque_nm[[0, 2]], batch.torque_limit_nm[[0, 2]])
    batch.reset([2], position_mm=84)
    assert batch.integral_torque_nm[2] == 0
    assert batch.reference_gap_mm[2] == batch.position_mm[2] == 84

def test_independent_torque_limits_and_atomic_invalid_batch_command(model):
    batch = G2Controller(model, num_envs=4, dtype=np.float32)
    batch.command(0, [10, 50, 90, 100], [10, 30, 80, 100])
    assert batch.torque_limit_nm.tolist() == pytest.approx(
        [model.torque_limit_nm(batch.force_lookup(i)) for i in range(4)], rel=1e-6
    )
    before = batch.position_mm.copy()
    with pytest.raises(ValueError):
        batch.command([84, 84, 85, 84], 50, 50)
    np.testing.assert_array_equal(before, batch.position_mm)

def test_num_envs_is_keyword_only(model):
    with pytest.raises(TypeError):
        G2Controller(model, 50)

def _value(arr) -> float:
    return float(arr[0])

def test_speed_above_chart_clamps_force_only(model):
    controller = G2Controller(model)
    controller.command(0, 100, 80)
    assert _value(controller.requested_speed_mm_s) == 225
    assert _value(controller.requested_speed_rpm) == 3765
    lookup = controller.force_lookup()
    assert lookup.calibration_rpm == 3500
    assert lookup.outside_calibration
    assert lookup.force_n == model.curve.lookup(3500, 80).force_n

def test_reference_advances_at_command_speed_without_overshooting_target(model):
    controller = G2Controller(model)
    controller.command(80, 10, 50)
    q = model.linkage.drive_for_gap_m(0.084)
    for _ in range(20):
        previous = _value(controller.reference_gap_mm)
        output = controller.update(q, 0)
        assert 0 <= previous - _value(output.reference_gap_mm) <= 0.450000001
        assert _value(output.reference_gap_mm) >= 80
    assert _value(controller.reference_gap_mm) == 80

def test_pause_holds_preload_and_last_nonzero_calibration_then_resumes(model):
    c = G2Controller(model, initial_position_mm=0)
    c.command(0, 90, 80)
    q = model.linkage.drive_for_gap_m(REFERENCE_GAP_MM / 1000)
    c.update(q, 0)
    before = c.force_lookup()
    c.command(0, 0, 80)
    paused = c.update(q, 0)
    assert _value(paused.reference_gap_mm) == pytest.approx(REFERENCE_GAP_MM)
    assert _value(paused.velocity_rad_s) == 0
    assert _value(paused.integral_torque_nm) == pytest.approx(_value(paused.torque_limit_nm))
    assert c.force_lookup() == before
    assert _value(c.requested_speed_rpm) == 0
    for _ in range(20):
        assert _value(c.update(q, 0).reference_gap_mm) == _value(paused.reference_gap_mm)
    c.command(0, 90, 80)
    resumed = c.update(q, 0)
    assert _value(resumed.reference_gap_mm) < _value(paused.reference_gap_mm)
    assert _value(resumed.velocity_rad_s) > 0

def test_paused_force_change_updates_saturated_holding_effort(model):
    c = G2Controller(model, initial_position_mm=0)
    q = model.linkage.drive_for_gap_m(REFERENCE_GAP_MM / 1000)
    c.command(0, 50, 30)
    c.update(q, 0)
    c.command(0, 0, 30)
    before = c.update(q, 0)
    c.command(0, 0, 80)
    after = c.update(q, 0)
    assert _value(after.torque_limit_nm) > _value(before.torque_limit_nm)
    assert _value(after.integral_torque_nm) == pytest.approx(_value(after.torque_limit_nm))
    compressed_q = q + 0.005
    for _ in range(500):
        held = c.update(compressed_q, 0)
    assert _value(held.integral_torque_nm) == pytest.approx(_value(held.torque_limit_nm))
    assert c.kp * (_value(held.position_rad) - compressed_q) == pytest.approx(_value(held.torque_limit_nm))

def test_paused_new_position_keeps_grasp_until_nonzero_speed(model):
    c = G2Controller(model, initial_position_mm=0)
    q = model.linkage.drive_for_gap_m(REFERENCE_GAP_MM / 1000)
    c.command(0, 50, 80)
    c.update(q, 0)
    c.command(84, 0, 80)
    paused = c.update(q, 0)
    assert _value(paused.reference_gap_mm) == pytest.approx(REFERENCE_GAP_MM)
    assert _value(paused.integral_torque_nm) == pytest.approx(_value(paused.torque_limit_nm))
    c.command(84, 50, 80)
    opened = c.update(q, 0)
    assert _value(opened.reference_gap_mm) > REFERENCE_GAP_MM
    assert _value(opened.velocity_rad_s) < 0
    assert _value(opened.integral_torque_nm) <= 0

def test_moving_pause_uses_actual_pad_position_but_stopped_pause_keeps_joint_posture(model):
    c = G2Controller(model, initial_position_mm=50)
    q = model.linkage.drive_for_gap_m(0.050)
    c.command(0, 0, 80)
    assert _value(c.update(q, 1.0, measured_gap_mm=47.0).reference_gap_mm) == 47.0
    c = G2Controller(model, initial_position_mm=0)
    c.command(0, 50, 80)
    c.update(q, 0)
    c.command(0, 0, 80)
    assert _value(c.update(q, 0, measured_gap_mm=47.0).reference_gap_mm) == pytest.approx(50.0)

def test_conditional_integration_stops_windup_and_allows_unwinding(model):
    c = G2Controller(model, initial_position_mm=0)
    c.command(0, 50, 10)
    q = model.linkage.drive_for_gap_m(REFERENCE_GAP_MM / 1000)
    for _ in range(500):
        c.update(q, 0)
    assert _value(c.integral_torque_nm) == 0
    c = G2Controller(model, initial_position_mm=30)
    limit = model.torque_limit_nm(c.force_lookup())
    c.integral_torque_nm.fill(limit)
    narrower_q = model.linkage.drive_for_gap_m(0.029)
    c.update(narrower_q, -2)
    assert 0 < _value(c.integral_torque_nm) < limit

def test_reversal_clears_integral_and_rebases_to_observed_position(model):
    c = G2Controller(model, initial_position_mm=0)
    c.command(0, 50, 80)
    c.integral_torque_nm.fill(model.torque_limit_nm(c.force_lookup()))
    q = model.linkage.drive_for_gap_m(REFERENCE_GAP_MM / 1000)
    c.command(84, 50, 80)
    output = c.update(q, 0)
    assert _value(output.reference_gap_mm) == pytest.approx(REFERENCE_GAP_MM + 2.25)
    assert _value(output.integral_torque_nm) <= 0
    assert _value(output.velocity_rad_s) < 0

@pytest.mark.parametrize(
    "command", [(-1, 50, 50), (85, 50, 50), (0, -1, 50), (0, 101, 50), (0, 50, 0), (0, 50, 101), (0, math.nan, 50)]
)
def test_invalid_command_is_atomic(model, command):
    c = G2Controller(model)
    before = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in vars(c).items()}
    with pytest.raises(ValueError):
        c.command(*command)
    for key, value in vars(c).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, before[key])
        else:
            assert value == before[key], key

@pytest.mark.parametrize(
    "model_name",
    [
        "gripper_g2",
        "xarm5_with_gripper",
        "xarm6_with_gripper",
        "xarm7_with_gripper",
        "uf850_with_gripper",
    ],
)
def test_mjcf_g2_models_match_calibrated_parameters(model_name):
    m = compose_model(model_name, scene=False)

    for jnt_name in G2_JOINT_NAMES:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
        assert jid >= 0, f"Missing joint {jnt_name} in {model_name}"
        dof_id = int(m.jnt_dofadr[jid])
        np.testing.assert_allclose(
            m.jnt_range[jid],
            [GRIPPER_G2_SIM_OPEN_DRIVE, GRIPPER_G2_SIM_CLOSE_DRIVE],
            atol=1e-12,
        )
        assert float(m.dof_armature[dof_id]) == pytest.approx(G2_ARMATURE, abs=1e-12)
        assert float(m.dof_damping[dof_id]) == pytest.approx(G2_DAMPING, abs=1e-12)
        assert float(m.dof_frictionloss[dof_id]) == pytest.approx(G2_FRICTIONLOSS, abs=1e-12)
        assert float(m.jnt_stiffness[jid]) == pytest.approx(0.0, abs=1e-12)

    assert m.neq == 5
    for eq_id in range(m.neq):
        np.testing.assert_allclose(m.eq_solref[eq_id], G2_MIMIC_SOLREF, atol=1e-12)
        np.testing.assert_allclose(m.eq_solimp[eq_id], G2_MIMIC_SOLIMP, atol=1e-12)

    act_name = "gripper_motor" if model_name == "gripper_g2" else "gripper"
    act_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
    assert act_id >= 0
    assert float(m.actuator_gainprm[act_id, 0]) == pytest.approx(G2_KP, abs=1e-12)
    assert float(m.actuator_biasprm[act_id, 1]) == pytest.approx(-G2_KP, abs=1e-12)
    assert float(m.actuator_biasprm[act_id, 2]) == pytest.approx(-G2_KV, abs=1e-12)

def test_mujoco_closed_loop_speed_control():
    scene = MODELS_ROOT / "gripper_g2" / "scene.xml"

    def steps_to_reach_gap(speed_pct: float, target_gap_m: float = 0.015) -> int:
        with SimRobot(scene) as robot:
            robot.set_gripper_gap(0.0, speed_percent=speed_pct, force_percent=80.0)
            for step in range(1, 2500):
                robot._step()
                if robot.get_gripper_gap(measured=True) <= target_gap_m:
                    return step
        return 2500

    steps_slow = steps_to_reach_gap(15.0)
    steps_mid = steps_to_reach_gap(50.0)
    steps_fast = steps_to_reach_gap(100.0)
    assert steps_fast < steps_mid < steps_slow
    assert steps_slow > 2 * steps_fast

def test_mujoco_closed_loop_force_control_monotonicity():
    from ufactory_mujoco.config import load_task_config

    cfg = load_task_config(preset=("pick_place", "xarm6"))
    pp = cfg.pick_place

    forces_measured = []
    torque_limits = []

    for force_pct in (10.0, 50.0, 100.0):
        with SimRobot(cfg.scene_path) as robot:
            # Use a 1 kg load-cell block mass so MuJoCo's mass-scaled contact
            # impedance stays stiff under up to ~62 N pinch load across versions.
            cube_id = robot.model.body("cube").id
            scale = 1.0 / float(robot.model.body_mass[cube_id])
            robot.model.body_mass[cube_id] = 1.0
            robot.model.body_inertia[cube_id] *= scale
            mujoco.mj_setConst(robot.model, robot.data)

            heights, q_home = tcp_task_heights(robot, pp)
            robot.data.qpos[: robot.n_arm] = q_home[: robot.n_arm]
            robot.data.ctrl[: robot.n_arm] = q_home[: robot.n_arm]
            mujoco.mj_forward(robot.model, robot.data)
            q_grasp, ok = robot.solve_tcp(
                np.array([pp.cube_xy[0], pp.cube_xy[1], heights["grasp_z"]]),
                GRIP_DOWN_QUAT,
                q_init=robot.data.qpos.copy(),
            )
            assert ok
            robot.data.qpos[: robot.n_arm] = q_grasp[: robot.n_arm]
            robot.data.ctrl[: robot.n_arm] = q_grasp[: robot.n_arm]
            mujoco.mj_forward(robot.model, robot.data)

            robot.set_gripper_gap(0.0, speed_percent=50.0, force_percent=force_pct)
            robot.settle(1.0)
            out = robot.gripper_control_output
            assert out is not None
            torque_limits.append(float(out.torque_limit_nm[0]))
            f_left, f_right = gripper_object_contact_forces_n(
                robot.model, robot.data, identify_gripper(robot.model), "cube")
            forces_measured.append(0.5 * (f_left + f_right))

    assert torque_limits[0] < torque_limits[1] < torque_limits[2]
    assert forces_measured[0] < forces_measured[1] < forces_measured[2]
    assert forces_measured[0] > 15.0
    assert forces_measured[2] > 55.0

if __name__ == "__main__":
    unittest.main()
