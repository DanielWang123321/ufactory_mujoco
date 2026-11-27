#!/usr/bin/env python3
"""Automated tests validating all converted UFACTORY visual OBJs and collision STLs.

Tests verify:
1. All target asset directories exist with expected OBJ and STL counts.
2. Every visual OBJ loads successfully into MuJoCo without colocated-vertex or convex-hull errors.
3. Every collision STL loads successfully into MuJoCo.
4. Mesh statistics (vertices, faces) are non-zero.
5. Total unique visual OBJ disk footprint stays within the full-precision band
   (guards against both bloat and re-decimation).
6. No visual mesh loses smooth shading to the classic renderer's normal flattening.
7. Listed double-sided OBJs carry reversed twin faces and every MJCF referencing
   them declares ``inertia="shell"``.
"""

from __future__ import annotations

import re
import struct
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from ufactory_mujoco import MODELS_ROOT

MODELS_DIR = MODELS_ROOT

# Full-precision visual OBJs total about 469 MiB (after unifying xArm5/xArm7 on xArm6); decimated 0.1.3 was ~95 MiB.
VISUAL_OBJ_MIN_MIB = 450
VISUAL_OBJ_MAX_MIB = 650

# mjr_uploadMesh (classic renderer, meshes without texcoords) replaces a corner
# normal by the face normal when their dot product is below 0.8 (36.9 degrees).
FLAT_NORMAL_DOT = 0.8
FLAT_AREA_MAX = 0.005

# Thin-walled visual OBJs carrying reversed twin faces; MJCF must load with inertia="shell".
DOUBLE_SIDED_COUNT = 73


def _is_double_sided(path: Path) -> bool:
    """True if the second half of faces/normals is the reversed/negated first half."""
    normals, faces = [], []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("vn "):
                normals.append(tuple(float(t) for t in line.split()[1:4]))
            elif line.startswith("f "):
                faces.append([(tok.split("/") + ["", ""])[:3] for tok in line.split()[1:]])
    nf, nn = len(faces), len(normals)
    if nf == 0 or nf % 2 or nn % 2:
        return False
    hf, hn = nf // 2, nn // 2
    for a, b in zip(faces[:hf], faces[hf:]):
        twin = [a[0]] + a[:0:-1]
        if len(twin) != len(b) or [c[0] for c in b] != [c[0] for c in twin]:
            return False
        if any(x[2] and (not y[2] or int(y[2]) != int(x[2]) + hn) for x, y in zip(twin, b)):
            return False
    return all(abs(p + q) <= 1e-6 for na, nb in zip(normals[:hn], normals[hn:]) for p, q in zip(na, nb))
# Measured on the source GLB normals (identical to ufactory_robot_viewer).
FLAT_AREA_ALLOW = {
    "bio_gripper_g2/assets/bio_gripper_g2/visual/bio_gripper_g2_left_finger.obj": 0.015,
    "bio_gripper_g2/assets/bio_gripper_g2/visual/bio_gripper_g2_right_finger.obj": 0.015,
}


def _mesh_model(obj_path: Path) -> mujoco.MjModel:
    xml = f"""<mujoco>
  <asset>
    <mesh name="m" file="{obj_path}" inertia="shell"/>
  </asset>
  <worldbody>
    <geom type="mesh" mesh="m" contype="0" conaffinity="0" group="2"/>
  </worldbody>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def _flat_normal_area_ratio(model: mujoco.MjModel) -> float:
    vert = model.mesh_vert.reshape(-1, 3)
    face = model.mesh_face.reshape(-1, 3)
    normal = model.mesh_normal.reshape(-1, 3)
    facenormal = model.mesh_facenormal.reshape(-1, 3)
    tri = vert[face]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1)
    keep = area > 0
    unit = cross[keep] / area[keep, None]
    corner = normal[facenormal[keep]]
    corner = corner / np.maximum(np.linalg.norm(corner, axis=2, keepdims=True), 1e-12)
    flat = (np.einsum("fkj,fj->fk", corner, unit) < FLAT_NORMAL_DOT).any(axis=1)
    return float(area[keep][flat].sum() / area[keep].sum())

EXPECTED_TARGETS = {
    "xarm6": {"min_objs": 15, "min_stls": 22},
    "xarm7": {"min_objs": 12, "min_stls": 23},
    "uf850": {"min_objs": 11, "min_stls": 27},
    "lite6": {"min_objs": 12, "min_stls": 19},
    "gripper_g2": {"min_objs": 12, "min_stls": 16},
    "bio_gripper_g2": {"min_objs": 4, "min_stls": 11},
    "lite6_gripper": {"min_objs": 7, "min_stls": 1},
    "lite6_vacuum_gripper": {"min_objs": 1, "min_stls": 1},
    "vacuum_gripper": {"min_objs": 3, "min_stls": 33},
    "robotiq_mounting": {"min_objs": 1, "min_stls": 2},
    "robotiq_2f_85": {"min_objs": 14, "min_stls": 11},
    "robotiq_2f_140": {"min_objs": 12, "min_stls": 11},
    "al0700": {"min_objs": 51, "min_stls": 9},
}


class ConvertedAssetsTest(unittest.TestCase):
    def test_all_target_directories_exist(self) -> None:
        """Verify each target has visual and collision asset folders."""
        for target, exp in EXPECTED_TARGETS.items():
            vis_dir = MODELS_DIR / target / "assets" / target / "visual"
            col_dir = MODELS_DIR / target / "assets" / target / "collision"
            self.assertTrue(vis_dir.is_dir(), f"Missing visual dir for {target}: {vis_dir}")
            self.assertTrue(col_dir.is_dir(), f"Missing collision dir for {target}: {col_dir}")

            objs = list(vis_dir.glob("*.obj"))
            stls = list(col_dir.glob("*.stl"))
            self.assertGreaterEqual(
                len(objs), exp["min_objs"],
                f"{target} has {len(objs)} OBJs, expected >= {exp['min_objs']}",
            )
            self.assertGreaterEqual(
                len(stls), exp["min_stls"],
                f"{target} has {len(stls)} STLs, expected >= {exp['min_stls']}",
            )

    def test_mujoco_loads_all_visual_objs(self) -> None:
        """Verify each OBJ mesh can be parsed and loaded into MuJoCo without error."""
        all_objs = sorted({p.resolve() for p in MODELS_DIR.glob("*/assets/*/visual/*.obj")})
        self.assertGreater(len(all_objs), 80, "Expected at least 80 converted OBJs")

        for obj_path in all_objs:
            try:
                model = _mesh_model(obj_path)
            except Exception as e:
                self.fail(f"MuJoCo failed to load visual OBJ {obj_path}: {e}")

            self.assertGreater(model.nmeshvert, 0, f"{obj_path} has 0 vertices in MuJoCo")
            self.assertGreater(model.nmeshface, 0, f"{obj_path} has 0 faces in MuJoCo")

            rel = obj_path.relative_to(MODELS_DIR.resolve()).as_posix()
            limit = FLAT_AREA_ALLOW.get(rel, FLAT_AREA_MAX)
            flat = _flat_normal_area_ratio(model)
            self.assertLessEqual(
                flat, limit,
                f"{rel}: {flat:.2%} of the area is drawn with flattened normals (limit {limit:.2%}); "
                "visual meshes must keep full tessellation",
            )

    def test_mujoco_loads_all_collision_stls(self) -> None:
        """Verify each STL collision mesh can be loaded into MuJoCo without error."""
        all_stls = sorted({p.resolve() for p in MODELS_DIR.glob("*/assets/*/collision/*.stl")})
        self.assertGreater(len(all_stls), 200, "Expected at least 200 unique collision STLs")

        for stl_path in all_stls:
            xml = f"""<mujoco>
  <asset>
    <mesh name="m" file="{stl_path}"/>
  </asset>
  <worldbody>
    <geom type="mesh" mesh="m" group="3"/>
  </worldbody>
</mujoco>"""
            try:
                model = mujoco.MjModel.from_xml_string(xml)
            except Exception as e:
                self.fail(f"MuJoCo failed to load collision STL {stl_path}: {e}")

            self.assertGreater(model.nmeshvert, 0, f"{stl_path} has 0 vertices in MuJoCo")
            self.assertGreater(model.nmeshface, 0, f"{stl_path} has 0 faces in MuJoCo")

    def test_visual_obj_footprint_within_budget(self) -> None:
        """Verify unique visual OBJs stay in the full-precision size band."""
        all_objs = sorted({p.resolve() for p in MODELS_DIR.glob("*/assets/*/visual/*.obj")})
        total_mib = sum(p.stat().st_size for p in all_objs) / (1024 * 1024)
        self.assertGreaterEqual(
            total_mib, VISUAL_OBJ_MIN_MIB,
            f"Total visual OBJ size {total_mib:.1f} MiB < {VISUAL_OBJ_MIN_MIB} MiB; meshes were decimated",
        )
        self.assertLessEqual(
            total_mib, VISUAL_OBJ_MAX_MIB,
            f"Total visual OBJ size {total_mib:.1f} MiB exceeds {VISUAL_OBJ_MAX_MIB} MiB",
        )

    def test_double_sided_visuals(self) -> None:
        """OBJs loaded with inertia="shell" have reversed twin faces, and every reference to them uses shell."""
        mesh_re = re.compile(r"<mesh\b[^>]*>")
        comp_re = re.compile(r"<compiler\b[^>]*>")
        tags = []
        for xml in sorted(MODELS_DIR.glob("*/*.xml")):
            text = xml.read_text(encoding="utf-8")
            comp = comp_re.search(text)
            meshdir_match = re.search(r'meshdir="([^"]+)"', comp.group(0)) if comp else None
            meshdir = xml.parent / (meshdir_match.group(1) if meshdir_match else "assets")
            for tag in mesh_re.findall(text):
                file_match = re.search(r'file="([^"]+)"', tag)
                if file_match:
                    tags.append((xml.name, (meshdir / file_match.group(1)).resolve(), tag))
        # Camera glass STLs also use shell inertia (thin panels); only OBJs carry twin faces.
        shell_paths = {path for _, path, tag in tags if path.suffix == ".obj" and 'inertia="shell"' in tag}
        self.assertEqual(len(shell_paths), DOUBLE_SIDED_COUNT)
        self.assertTrue(
            {
                (MODELS_DIR / "gripper_g2/assets/gripper_g2/visual" / f"{n}.obj").resolve()
                for n in (
                    "base_anodized_black", "base_black", "base_silver", "base_white",
                    "left_finger_black", "left_inner_knuckle", "left_outer_knuckle",
                    "right_finger_black", "right_inner_knuckle", "right_outer_knuckle",
                )
            } <= shell_paths,
        )
        for path in sorted(shell_paths):
            self.assertTrue(path.is_file(), path)
            self.assertTrue(_is_double_sided(path), f"{path} is not double-sided")
        for xml_name, path, tag in tags:
            if path in shell_paths:
                self.assertIn('inertia="shell"', tag, f"{xml_name}: {tag}")

    def test_lite6_link6_three_hull_geometry_and_wrist_clearance(self) -> None:
        """Verify Lite6 link6 uses 3 tight convex hulls and detects near-limit link4/link6 contact."""
        col_dir = MODELS_DIR / "lite6" / "assets" / "lite6" / "collision"
        col_files = sorted(col_dir.glob("lite6_link6_col_*.stl"))
        self.assertEqual([p.name for p in col_files], [f"lite6_link6_col_{i:02d}.stl" for i in range(3)])

        tri_dtype = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        expected_specs = ((324, -20.0, 0.0), (184, -24.5, -20.0), (92, -14.0, -4.0))
        hulls: list[tuple[np.ndarray, np.ndarray]] = []
        for path, (exp_tris, exp_z_min_mm, exp_z_max_mm) in zip(col_files, expected_specs, strict=True):
            raw = path.read_bytes()
            count = struct.unpack_from("<I", raw, 80)[0]
            self.assertEqual(count, exp_tris, path.name)
            self.assertEqual(len(raw), 84 + 50 * count)
            tris = np.frombuffer(raw, dtype=tri_dtype, offset=84, count=count)["v"].astype(np.float64)
            z_mm = tris.reshape(-1, 3)[:, 2] * 1000.0
            self.assertAlmostEqual(float(z_mm.min()), exp_z_min_mm, delta=0.01)
            self.assertAlmostEqual(float(z_mm.max()), exp_z_max_mm, delta=0.01)
            normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            hulls.append((tris, normals))

        vis_dir = MODELS_DIR / "lite6" / "assets" / "lite6" / "visual"
        vis_pts = np.asarray(
            [
                [float(tok) for tok in line.split()[1:4]]
                for name in ("link6_silver.obj", "link6_seal.obj")
                for line in (vis_dir / name).open(encoding="utf-8")
                if line.startswith("v ")
            ],
            dtype=np.float64,
        )
        min_outside = np.full(len(vis_pts), np.inf, dtype=np.float64)
        for tris, normals in hulls:
            outside = np.max(vis_pts @ normals.T - np.sum(tris[:, 0] * normals, axis=1)[None, :], axis=1)
            min_outside = np.minimum(min_outside, outside)
        self.assertLess(float(min_outside.max()) * 1000.0, 0.25)

        model = mujoco.MjModel.from_xml_path(str(MODELS_DIR / "lite6" / "lite6.xml"))
        data = mujoco.MjData(model)
        l4_geoms = [
            i
            for i in range(model.ngeom)
            if model.geom_group[i] == 3
            and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])) or "") == "link4"
        ]
        l6_geoms = [
            i
            for i in range(model.ngeom)
            if model.geom_group[i] == 3
            and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])) or "") == "link6"
        ]
        self.assertEqual(len(l6_geoms), 3)

        def link4_link6_min_mm(q5: float, q6: float) -> float:
            data.qpos[:] = [0.0, 0.0, 0.0, 0.0, q5, q6]
            mujoco.mj_forward(model, data)
            min_d = min(mujoco.mj_geomDistance(model, data, a, b, 0.2, None) for a in l4_geoms for b in l6_geoms)
            return min_d * 1000.0

        normal_min_mm = min(
            link4_link6_min_mm(q5, q6)
            for q5 in (-1.8, -1.0, 0.0, 0.2742, 1.0, 1.8)
            for q6 in np.linspace(-np.pi, np.pi, 13)
        )
        self.assertGreater(normal_min_mm, 5.8)

        limit_min_mm = min(link4_link6_min_mm(2.1642, q6) for q6 in np.linspace(-np.pi, np.pi, 13))
        self.assertLess(limit_min_mm, 0.5)

    def test_xarm_bases_use_production_version_a_and_shared_meshes(self) -> None:
        """Verify xArm 5/6/7 share xarm6 base/flange meshes with production version A dimensions."""
        x6_vis = MODELS_DIR / "xarm6" / "assets" / "xarm6" / "visual"
        expected_base_vis = (x6_vis / "link_base.obj").resolve()
        expected_flange_vis = {
            (x6_vis / f"link6_{part}.obj").resolve() for part in ("white", "silver", "black")
        }
        expected_cols = {
            (MODELS_DIR / "xarm6" / "assets" / "xarm6" / "collision" / f"xarm6_link_base_col_{i:02d}.stl").resolve()
            for i in range(2)
        }
        for arm, flange_prefix in (("xarm5", "link6"), ("xarm6", "link6"), ("xarm7", "link7")):
            xml_path = MODELS_DIR / arm / f"{arm}.xml"
            root = ET.parse(xml_path).getroot()
            comp = root.find("compiler")
            meshdir = xml_path.parent / (comp.get("meshdir", "assets") if comp is not None else "assets")
            resolved = {m.get("name"): (meshdir / m.get("file", "")).resolve() for m in root.findall("./asset/mesh")}
            self.assertEqual(resolved["link_base_vis"], expected_base_vis)
            self.assertEqual(
                {resolved["xarm6_link_base_col_00"], resolved["xarm6_link_base_col_01"]},
                expected_cols,
            )
            self.assertEqual(
                {resolved[f"{flange_prefix}_{part}_vis"] for part in ("white", "silver", "black")},
                expected_flange_vis,
            )

        self.assertFalse((MODELS_DIR / "xarm5" / "assets").exists())
        self.assertFalse((MODELS_DIR / "xarm7" / "assets" / "xarm7" / "visual" / "link_base.obj").exists())

        def read_obj_verts_mm(path: Path) -> np.ndarray:
            with path.open(encoding="utf-8") as fh:
                verts = [[float(t) * 1000.0 for t in line.split()[1:4]] for line in fh if line.startswith("v ")]
            return np.asarray(verts, dtype=np.float64)

        base_vis_mm = read_obj_verts_mm(expected_base_vis)
        z_min = float(base_vis_mm[:, 2].min())
        z_max = float(base_vis_mm[:, 2].max())
        self.assertTrue(-0.05 <= z_min <= 0.05, f"base visual z_min={z_min:.4f} mm")
        self.assertTrue(150.25 <= z_max <= 150.45, f"base visual z_max={z_max:.4f} mm")
        flange_d_mm = 2.0 * float(np.hypot(base_vis_mm[:, 0], base_vis_mm[:, 1]).max())
        self.assertTrue(125.5 <= flange_d_mm <= 127.0, f"base flange D={flange_d_mm:.3f} mm")

        r_base = np.hypot(base_vis_mm[:, 0], base_vis_mm[:, 1])
        cyl_mask = (base_vis_mm[:, 2] >= 60.0) & (base_vis_mm[:, 2] <= 100.0)
        self.assertTrue(95.0 <= float(2.0 * np.median(r_base[cyl_mask])) <= 96.0)
        cyl_fwd_mask = cyl_mask & (base_vis_mm[:, 0] > -20.0)
        self.assertLess(float(2.0 * r_base[cyl_fwd_mask].max()), 96.5)

        tri_dtype = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        col_chunks: list[np.ndarray] = []
        for col_path in sorted(expected_cols):
            raw = col_path.read_bytes()
            tri_cnt = struct.unpack_from("<I", raw, 80)[0]
            col_chunks.append(
                np.frombuffer(raw, dtype=tri_dtype, offset=84, count=tri_cnt)["v"].reshape(-1, 3).astype(np.float64)
                * 1000.0
            )
        col_pts_mm = np.concatenate(col_chunks, axis=0)
        self.assertTrue(-0.05 <= float(col_pts_mm[:, 2].min()) <= 0.05)
        self.assertTrue(150.25 <= float(col_pts_mm[:, 2].max()) <= 150.45)

        x6_root = ET.parse(MODELS_DIR / "xarm6" / "xarm6.xml").getroot()
        link1_el = x6_root.find(".//body[@name='link1']")
        assert link1_el is not None
        link1_z_mm = float(link1_el.get("pos", "0 0 0").split()[2]) * 1000.0
        l1_white_mm = read_obj_verts_mm(x6_vis / "link1_white.obj")
        l1_white_mm[:, 2] += link1_z_mm
        r_l1 = np.hypot(l1_white_mm[:, 0], l1_white_mm[:, 1])
        outer_mask = r_l1 > 45.0
        self.assertGreater(float(l1_white_mm[outer_mask, 2].min()) - z_max, 0.3)
        skirt_mask = outer_mask & (l1_white_mm[:, 2] >= 151.0) & (l1_white_mm[:, 2] <= 185.0)
        self.assertGreater(2.0 * float(np.median(r_l1[skirt_mask])), 97.0)

if __name__ == "__main__":
    unittest.main()

