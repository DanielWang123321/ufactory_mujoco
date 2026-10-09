#!/usr/bin/env python3
"""Rebuild the xArm 6 URDF draft from the official xarm_description package and
convert it into an MJCF draft via MuJoCo.

The final model is models/xarm6/xarm6.xml (hand-tuned actuators/scene based on
this draft). Run this script only when the upstream URDF changes.

Usage:
    python scripts/build_model.py [--keep]

Requires: xacro, mujoco. Clones xArm-Developer/xarm_ros (shallow) into /tmp.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/xArm-Developer/xarm_ros.git"
BUILD_DIR = Path("/tmp/xarm_build")
PROJECT = Path(__file__).resolve().parent.parent
MIN_XACRO = """<?xml version="1.0"?>
<robot xmlns:xacro="http://ros.org/wiki/xacro" name="xarm6_standalone">
  <xacro:include filename="{build}/urdf/xarm_device_macro.xacro" />
  <xacro:xarm_device dof="6" robot_type="xarm" load_gazebo_plugin="false"
    add_gripper="false" add_vacuum_gripper="false" add_bio_gripper="false"
    add_realsense_d435i="false" add_other_geometry="false" />
</robot>
"""


def run() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep intermediate URDF drafts")
    args = ap.parse_args()

    # 1. shallow clone xarm_description
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True)
    print(f"[1/4] cloning {REPO_URL} (shallow)...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", REPO_URL, "/tmp/xarm_ros_clone"],
        check=True,
    )
    src = Path("/tmp/xarm_ros_clone/xarm_description")
    for d in ("urdf", "config", "meshes"):
        shutil.copytree(src / d, BUILD_DIR / d)

    # 2. replace $(find xarm_description) with absolute paths (no ROS needed)
    print("[2/4] patching xacro paths...")
    for f in (BUILD_DIR / "urdf").rglob("*.xacro"):
        f.write_text(f.read_text().replace("$(find xarm_description)", str(BUILD_DIR)))
    (BUILD_DIR / "xarm6_min.urdf.xacro").write_text(MIN_XACRO.format(build=BUILD_DIR))

    # 3. expand xacro -> URDF, strip package:// prefix, inject meshdir
    print("[3/4] expanding xacro...")
    raw = BUILD_DIR / "xarm6_raw.urdf"
    subprocess.run(
        [sys.executable, "-c",
         "import sys; from xacro import process_file, open_output;"
         "doc = process_file(sys.argv[1]); open_output(sys.argv[2]).write(doc.toprettyxml())",
         str(BUILD_DIR / "xarm6_min.urdf.xacro"), str(raw)],
        check=True,
    )
    text = raw.read_text().replace("package://xarm_description/meshes/", "")
    m = re.search(r"(<robot[^>]*>)", text)
    text = (text[: m.end()]
            + '\n<mujoco>\n  <compiler meshdir="' + str(BUILD_DIR / "meshes")
            + '" balanceinertia="true" discardvisual="false"/>\n</mujoco>'
            + text[m.end():])
    (BUILD_DIR / "xarm6_mj.urdf").write_text(text)

    # 4. load in MuJoCo, export MJCF draft
    print("[4/4] converting via MuJoCo...")
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(BUILD_DIR / "xarm6_mj.urdf"))
    out = BUILD_DIR / "xarm6_converted.xml"
    mujoco.mj_saveLastXML(str(out), model)
    print(f"nq={model.nq} nv={model.nv} nbody={model.nbody} ngeom={model.ngeom}")
    print(f"MJCF draft saved: {out}")
    if not args.keep:
        shutil.rmtree(BUILD_DIR)
        print("intermediates removed (use --keep to inspect)")


if __name__ == "__main__":
    run()
