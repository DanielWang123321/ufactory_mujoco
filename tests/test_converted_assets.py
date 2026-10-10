#!/usr/bin/env python3
"""Automated tests validating all converted UFACTORY visual OBJs and collision STLs.

Tests verify:
1. All 9 target directories exist with expected OBJ and STL counts.
2. Every visual OBJ loads successfully into MuJoCo without colocated-vertex or convex-hull errors.
3. Every collision STL loads successfully into MuJoCo.
4. Mesh statistics (vertices, faces) are non-zero.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import mujoco

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

EXPECTED_TARGETS = {
    "xarm6": {"min_objs": 15, "min_stls": 7},
    "xarm5": {"min_objs": 12, "min_stls": 6},
    "xarm7": {"min_objs": 16, "min_stls": 8},
    "uf850": {"min_objs": 11, "min_stls": 7},
    "lite6": {"min_objs": 12, "min_stls": 7},
    "gripper_g2": {"min_objs": 7, "min_stls": 7},
    "bio_gripper_g2": {"min_objs": 4, "min_stls": 3},
    "lite6_gripper": {"min_objs": 7, "min_stls": 3},
    "lite6_vacuum_gripper": {"min_objs": 1, "min_stls": 1},
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
        all_objs = list(MODELS_DIR.glob("*/assets/*/visual/*.obj"))
        self.assertGreater(len(all_objs), 80, "Expected at least 80 converted OBJs")

        for obj_path in all_objs:
            xml = f"""<mujoco>
  <asset>
    <mesh name="m" file="{obj_path.resolve()}"/>
  </asset>
  <worldbody>
    <geom type="mesh" mesh="m" contype="0" conaffinity="0" group="2"/>
  </worldbody>
</mujoco>"""
            try:
                model = mujoco.MjModel.from_xml_string(xml)
            except Exception as e:
                self.fail(f"MuJoCo failed to load visual OBJ {obj_path}: {e}")

            self.assertGreater(model.nmeshvert, 0, f"{obj_path} has 0 vertices in MuJoCo")
            self.assertGreater(model.nmeshface, 0, f"{obj_path} has 0 faces in MuJoCo")

    def test_mujoco_loads_all_collision_stls(self) -> None:
        """Verify each STL collision mesh can be loaded into MuJoCo without error."""
        all_stls = list(MODELS_DIR.glob("*/assets/*/collision/*.stl"))
        self.assertGreater(len(all_stls), 45, "Expected at least 45 collision STLs")

        for stl_path in all_stls:
            xml = f"""<mujoco>
  <asset>
    <mesh name="m" file="{stl_path.resolve()}"/>
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


if __name__ == "__main__":
    unittest.main()

