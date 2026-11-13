#!/usr/bin/env python3
"""Automated tests validating all converted UFACTORY visual OBJs and collision STLs.

Tests verify:
1. All 10 target directories exist with expected OBJ and STL counts.
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
import unittest
from pathlib import Path

import mujoco
import numpy as np

from scripts.make_double_sided_objs import (
    DOUBLE_SIDED_VISUALS,
    double_sided_paths,
    is_double_sided,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Full-precision visual OBJs total about 592 MiB (0.1.5); the decimated 0.1.3 set was about 95 MiB.
VISUAL_OBJ_MIN_MIB = 450
VISUAL_OBJ_MAX_MIB = 650

# mjr_uploadMesh (classic renderer, meshes without texcoords) replaces a corner
# normal by the face normal when their dot product is below 0.8 (36.9 degrees).
FLAT_NORMAL_DOT = 0.8
FLAT_AREA_MAX = 0.005
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
    "xarm5": {"min_objs": 12, "min_stls": 19},
    "xarm7": {"min_objs": 16, "min_stls": 26},
    "uf850": {"min_objs": 11, "min_stls": 27},
    "lite6": {"min_objs": 12, "min_stls": 17},
    "gripper_g2": {"min_objs": 7, "min_stls": 16},
    "bio_gripper_g2": {"min_objs": 4, "min_stls": 5},
    "lite6_gripper": {"min_objs": 7, "min_stls": 3},
    "lite6_vacuum_gripper": {"min_objs": 1, "min_stls": 1},
    "vacuum_gripper": {"min_objs": 3, "min_stls": 33},
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
        """Listed OBJs have reversed twin faces; every referencing MJCF uses inertia="shell"."""
        self.assertTrue(DOUBLE_SIDED_VISUALS)
        tails = set()
        for path in double_sided_paths():
            self.assertTrue(path.is_file(), path)
            self.assertTrue(is_double_sided(path), f"{path} is not double-sided")
            tails.add(path.relative_to(MODELS_DIR).as_posix().split("/assets/", 1)[1])
        self.assertTrue(
            {f"gripper_g2/visual/{n}.obj" for n in (
                "base", "left_finger", "left_inner_knuckle", "left_outer_knuckle",
                "right_finger", "right_inner_knuckle", "right_outer_knuckle",
            )} <= tails,
        )
        mesh_re = re.compile(r"<mesh\b[^>]*>")
        referenced = set()
        for xml in sorted(MODELS_DIR.glob("*/*.xml")):
            for tag in mesh_re.findall(xml.read_text(encoding="utf-8")):
                file_match = re.search(r'file="([^"]+)"', tag)
                if not file_match or file_match.group(1) not in tails:
                    continue
                referenced.add(file_match.group(1))
                self.assertIn('inertia="shell"', tag, f"{xml.name}: {tag}")
        self.assertEqual(referenced, tails)


if __name__ == "__main__":
    unittest.main()

