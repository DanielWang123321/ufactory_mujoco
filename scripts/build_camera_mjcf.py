#!/usr/bin/env python3
"""Build standardized MJCF models and scenes for depth cameras and camera stand.

Generates 13 model directories (26 XML files) under ``models/``:
- 4 standalone cameras: ``d405``, ``d435``, ``gemini_305``, ``zed_x_nano``
- 1 standalone bracket: ``camera_stand``
- 4 bracket + camera assemblies: ``camera_stand_with_{d405,d435,gemini_305,zed_x_nano}``
- 4 xArm 6 + Gripper G2 + bracket + camera assemblies:
  ``xarm6_with_gripper_and_{d405,d435,gemini_305,zed_x_nano}``

Usage:
    python scripts/build_camera_mjcf.py --sync-dev-meshes
    python scripts/build_camera_mjcf.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

try:
    from scripts.make_double_sided_objs import apply_shell_inertia
except ImportError:  # run as ``python scripts/build_camera_mjcf.py``
    from make_double_sided_objs import apply_shell_inertia

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DEV_DIR = ROOT / "dev"
XARM6_G2_XML = MODELS_DIR / "xarm6_with_gripper" / "xarm6_with_gripper.xml"


@dataclass(frozen=True)
class CameraSpec:
    """Native-frame geometry, mounting sites, and collision boxes for a camera."""

    key: str
    label: str
    scale: float
    mount: tuple[float, float, float]
    mount_quat: tuple[float, float, float, float]
    holes: tuple[tuple[float, float, float], tuple[float, float, float]]
    collision: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]
    finish: str
    mount_clearance_m: float = 0.0


CAMERAS: dict[str, CameraSpec] = {
    "d405": CameraSpec(
        key="d405",
        label="RealSense D405",
        scale=0.001,
        mount=(0.0, 0.0, -0.023),
        mount_quat=(0.0, 1.0, 0.0, 0.0),
        holes=((-0.010, 0.0, -0.023), (0.010, 0.0, -0.023)),
        collision=(((-0.000045, 0.0, -0.0115), (0.021045, 0.021, 0.0115)),),
        finish="silver",
    ),
    "d435": CameraSpec(
        key="d435",
        label="RealSense D435",
        scale=1.0,
        mount=(0.0, 0.0, -0.025),
        mount_quat=(0.0, 1.0, 0.0, 0.0),
        holes=((-0.0225, 0.0, -0.025), (0.0225, 0.0, -0.025)),
        collision=(((0.0, 0.0, -0.0125), (0.045, 0.0125, 0.0125)),),
        finish="silver",
        # D435 ROS mesh extends 0.0547 mm past its nominal rear mounting plane.
        mount_clearance_m=0.000055,
    ),
    "gemini_305": CameraSpec(
        key="gemini_305",
        label="Orbbec Gemini 305",
        scale=0.001,
        mount=(0.018348755, 0.023076703, -0.017177002),
        mount_quat=(0.707106781, -0.707106781, 0.0, 0.0),
        holes=((0.008348767, 0.023076703, -0.017190216), (0.028348746, 0.023076703, -0.017161645)),
        collision=(((0.018345856, 0.011576704, -0.017181074), (0.021014405, 0.0115, 0.0210143)),),
        finish="silver",
    ),
    "zed_x_nano": CameraSpec(
        key="zed_x_nano",
        label="Stereolabs ZED X Nano",
        scale=0.001,
        mount=(0.0, 0.0, 0.0),
        mount_quat=(0.0, 1.0, 0.0, 0.0),
        holes=((-0.010, 0.0, 0.0), (0.010, 0.0, 0.0)),
        collision=(
            ((0.0, 0.00366, 0.01525), (0.021, 0.0169, 0.01525)),
            ((0.0, 0.02593, 0.01525), (0.0101, 0.00537, 0.00675)),
        ),
        finish="black",
    ),
}

# Columns of R are camera-native axes expressed in camera_stand coordinates.
R_STANDARD = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=float)
R_GEMINI = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
STAND_MOUNT_TARGET = np.array([0.0675, 0.003, 0.0], dtype=float)

SCENE_TEMPLATE = """<mujoco model="{name} scene">
  <include file="{xml_file}"/>

  <statistic center="{center}" extent="{extent}"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-140" elevation="-25" offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
      rgb2="0.1 0.15 0.2" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor"{floor_pos} size="0 0 0.05" type="plane" material="groundplane"/>
    <camera name="front" pos="{front_pos}" {front_orient}/>
    <camera name="side" pos="{side_pos}" {side_orient}/>
  </worldbody>
</mujoco>
"""


def _vec(values: tuple[float, ...] | list[float] | np.ndarray) -> str:
    return " ".join(f"{float(x):.10g}" for x in values)


def _quat_from_matrix(matrix: np.ndarray) -> str:
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return _vec([w, x, y, z])


def _elem(parent: ET.Element, tag: str, **kwargs: str) -> ET.Element:
    return ET.SubElement(parent, tag, {k: str(v) for k, v in kwargs.items()})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_verified_stl(src: Path, dst: Path) -> None:
    raw = src.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"Invalid binary STL (too short): {src}")
    tri_count = int.from_bytes(raw[80:84], "little")
    if len(raw) != 84 + 50 * tri_count or not (0 < tri_count <= 200_000):
        raise ValueError(f"Invalid binary STL header or face count ({tri_count}) in {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    if _sha256(src) != _sha256(dst):
        raise RuntimeError(f"SHA-256 mismatch after copying {src} -> {dst}")


def sync_dev_meshes() -> int:
    """Copy canonical camera and camera_stand STLs from dev/ into models/*/assets/."""
    copied = 0
    for cam in CAMERAS:
        src_dir = DEV_DIR / "camera" / cam / "meshes"
        dst_dir = MODELS_DIR / cam / "assets" / cam / "visual"
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        files = sorted(src_dir.glob("visual_*.stl"))
        if not files:
            raise FileNotFoundError(f"No visual_*.stl found in {src_dir}")
        for src in files:
            _copy_verified_stl(src, dst_dir / src.name)
            copied += 1

    stand_src = DEV_DIR / "camera_stand" / "meshes"
    stand_vis_dst = MODELS_DIR / "camera_stand" / "assets" / "camera_stand" / "visual"
    stand_col_dst = MODELS_DIR / "camera_stand" / "assets" / "camera_stand" / "collision"
    for d in (stand_vis_dst, stand_col_dst):
        if d.exists():
            shutil.rmtree(d)
    _copy_verified_stl(stand_src / "stand_mm.stl", stand_vis_dst / "stand_mm.stl")
    copied += 1

    pieces = sorted((stand_src / "collision").glob("piece_*.stl"))
    if len(pieces) != 175:
        raise RuntimeError(f"Expected 175 collision STLs in {stand_src / 'collision'}, got {len(pieces)}")
    for src in pieces:
        _copy_verified_stl(src, stand_col_dst / src.name)
        copied += 1
    return copied


def _ensure_symlinks(model_dir: Path, parts: tuple[str, ...]) -> None:
    assets_dir = model_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    expected = set(parts)
    for child in list(assets_dir.iterdir()):
        if child.name not in expected or not child.is_symlink():
            if child.is_symlink() or child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)
    for part in parts:
        link = assets_dir / part
        target = Path("..") / ".." / part / "assets" / part
        if link.is_symlink():
            if os.readlink(link) == target.as_posix():
                continue
            link.unlink()
        link.symlink_to(target)


def _new_model_root(model_name: str, comment: str) -> ET.Element:
    root = ET.Element("mujoco", {"model": model_name})
    root.append(ET.Comment(f" {comment} "))
    _elem(root, "compiler", angle="radian", autolimits="true", meshdir="assets")
    _elem(root, "option", integrator="implicitfast")
    _elem(root, "asset")
    _elem(root, "worldbody")
    return root


def _append_camera(
    root: ET.Element,
    parent: ET.Element,
    cam_key: str,
    *,
    pos: np.ndarray | None = None,
    orientation: np.ndarray | None = None,
) -> ET.Element:
    spec = CAMERAS[cam_key]
    vis_dir = MODELS_DIR / cam_key / "assets" / cam_key / "visual"
    vis_files = sorted(vis_dir.glob("visual_*.stl"))
    if not vis_files:
        raise FileNotFoundError(f"Missing canonical visual STLs in {vis_dir}; run with --sync-dev-meshes first.")

    asset_el = root.find("asset")
    assert asset_el is not None
    housing_mat = f"{cam_key}_anodized_aluminum"
    glass_mat = f"{cam_key}_optical_glass"
    housing_rgba = "0.68 0.70 0.73 1" if spec.finish == "silver" else "0.15 0.16 0.17 1"
    _elem(asset_el, "material", name=housing_mat, rgba=housing_rgba, specular="0.55", shininess="0.35")
    _elem(asset_el, "material", name=glass_mat, rgba="0.025 0.03 0.035 1", specular="0.8", shininess="0.75")

    body_attrs: dict[str, str] = {"name": f"{cam_key}_body"}
    if pos is not None:
        body_attrs["pos"] = _vec(pos)
    if orientation is not None:
        body_attrs["quat"] = _quat_from_matrix(orientation)
    body = _elem(parent, "body", **body_attrs)

    scale_str = _vec([spec.scale] * 3)
    for idx, stl_path in enumerate(vis_files):
        mesh_name = f"{cam_key}_visual_mesh_{idx:02d}"
        is_glass = stl_path.name.startswith("visual_glass_")
        extra = {"inertia": "shell"} if is_glass else {}
        _elem(
            asset_el,
            "mesh",
            name=mesh_name,
            file=f"{cam_key}/visual/{stl_path.name}",
            scale=scale_str,
            **extra,
        )
        _elem(
            body,
            "geom",
            name=f"{cam_key}_visual_{idx:02d}",
            type="mesh",
            mesh=mesh_name,
            group="2",
            contype="0",
            conaffinity="0",
            mass="0",
            material=glass_mat if is_glass else housing_mat,
        )

    for idx, (box_pos, box_half) in enumerate(spec.collision):
        _elem(
            body,
            "geom",
            name=f"{cam_key}_collision_{idx:02d}",
            type="box",
            pos=_vec(box_pos),
            size=_vec(box_half),
            group="3",
            mass="0",
            rgba="0.8 0.15 0.1 0.3",
        )

    _elem(
        body,
        "site",
        name=f"{cam_key}_mount",
        pos=_vec(spec.mount),
        quat=_vec(spec.mount_quat),
        size="0.002",
        rgba="1 0.4 0 1",
        group="4",
    )
    for idx, hole_pos in enumerate(spec.holes):
        _elem(
            body,
            "site",
            name=f"{cam_key}_hole_{idx}",
            pos=_vec(hole_pos),
            quat=_vec(spec.mount_quat),
            size="0.001",
            group="4",
        )
    return body


def _append_bracket(
    root: ET.Element,
    parent: ET.Element,
    *,
    with_camera: str | None = None,
    on_arm_flange: bool = False,
) -> ET.Element:
    stand_root = MODELS_DIR / "camera_stand" / "assets" / "camera_stand"
    vis_stl = stand_root / "visual" / "stand_mm.stl"
    col_stls = sorted((stand_root / "collision").glob("piece_*.stl"))
    if not vis_stl.is_file() or len(col_stls) != 175:
        raise FileNotFoundError(
            f"Missing canonical camera_stand STLs in {stand_root}; run with --sync-dev-meshes first."
        )

    asset_el = root.find("asset")
    assert asset_el is not None
    _elem(
        asset_el,
        "material",
        name="stand_anodized_aluminum",
        rgba="0.72 0.74 0.76 1",
        specular="0.5",
        shininess="0.35",
    )
    _elem(
        asset_el,
        "mesh",
        name="stand_visual_mesh",
        file="camera_stand/visual/stand_mm.stl",
        scale="0.001 0.001 0.001",
    )

    body_attrs: dict[str, str] = {"name": "camera_stand"}
    if on_arm_flange:
        body_attrs["quat"] = _vec([0.707106781, 0.707106781, 0.0, 0.0])
    body = _elem(parent, "body", **body_attrs)
    _elem(
        body,
        "geom",
        name="stand_visual",
        type="mesh",
        mesh="stand_visual_mesh",
        group="2",
        contype="0",
        conaffinity="0",
        mass="0",
        material="stand_anodized_aluminum",
    )

    for idx, stl_path in enumerate(col_stls):
        mesh_name = f"stand_collision_mesh_{idx:03d}"
        _elem(asset_el, "mesh", name=mesh_name, file=f"camera_stand/collision/{stl_path.name}")
        _elem(
            body,
            "geom",
            name=f"stand_collision_{idx:03d}",
            type="mesh",
            mesh=mesh_name,
            group="3",
            mass="0",
            rgba="0.9 0.2 0.1 0.25",
        )

    _elem(
        body,
        "site",
        name="stand_flange",
        pos="0 0 0",
        quat="0.707106781 0.707106781 0 0",
        size="0.002",
        rgba="0.2 0.7 1 1",
        group="4",
    )
    _elem(
        body,
        "site",
        name="stand_camera_mount",
        pos=_vec(STAND_MOUNT_TARGET),
        quat="0.707106781 -0.707106781 0 0",
        size="0.002",
        rgba="1 0.4 0 1",
        group="4",
    )
    for half_spacing in (0.010, 0.0225):
        spacing_mm = int(round(half_spacing * 2000))
        for idx, sign in enumerate((-1, 1)):
            _elem(
                body,
                "site",
                name=f"stand_hole_{spacing_mm}_{idx}",
                pos=_vec([0.0675, 0.003, sign * half_spacing]),
                quat="0.707106781 -0.707106781 0 0",
                size="0.001",
                group="4",
            )

    if with_camera is not None:
        spec = CAMERAS[with_camera]
        rot = R_GEMINI if with_camera == "gemini_305" else R_STANDARD
        cam_pos = STAND_MOUNT_TARGET + np.array([0.0, spec.mount_clearance_m, 0.0]) - rot @ np.array(spec.mount)
        _append_camera(root, body, with_camera, pos=cam_pos, orientation=rot)

    return body


def _write_xml(root: ET.Element, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    apply_shell_inertia(root)
    ET.indent(root, space="  ")
    text = ET.tostring(root, encoding="unicode")
    out_path.write_text(text + "\n", encoding="utf-8")


def _write_scene(
    model_dir: Path,
    name: str,
    *,
    kind: str,
) -> Path:
    scene_path = model_dir / "scene.xml"
    if kind == "arm":
        content = SCENE_TEMPLATE.format(
            name=name,
            xml_file=f"{name}.xml",
            center="0.3 0 0.3",
            extent="0.9",
            floor_pos="",
            front_pos="1.2 0.8 0.9",
            front_orient='quat="0.72 0 0.19 0.67"',
            side_pos="-0.3 1.6 0.8",
            side_orient='quat="0.66 0.24 0.24 0.67"',
        )
    elif kind == "camera":
        content = SCENE_TEMPLATE.format(
            name=name,
            xml_file=f"{name}.xml",
            center="0 0 0",
            extent="0.16",
            floor_pos=' pos="0 0 -0.05"',
            front_pos="0.16 0.14 0.10",
            front_orient='xyaxes="-0.66 0.75 0 -0.36 -0.32 0.88"',
            side_pos="-0.04 0.22 0.09",
            side_orient='xyaxes="-0.98 -0.18 0 0.07 -0.38 0.92"',
        )
    else:
        content = SCENE_TEMPLATE.format(
            name=name,
            xml_file=f"{name}.xml",
            center="0.04 0.01 0",
            extent="0.24",
            floor_pos=' pos="0 0 -0.05"',
            front_pos="0.24 0.20 0.14",
            front_orient='xyaxes="-0.64 0.77 0 -0.37 -0.31 0.88"',
            side_pos="0.04 0.28 0.12",
            side_orient='xyaxes="-1 0 0 0 -0.39 0.92"',
        )
    scene_path.write_text(content, encoding="utf-8")
    return scene_path


def build_all(*, verify: bool = True) -> list[Path]:
    """Generate all 13 camera/bracket/arm-combo model directories and return XML paths."""
    generated: list[Path] = []

    # 1. Four standalone cameras
    for cam_key, spec in CAMERAS.items():
        model_dir = MODELS_DIR / cam_key
        xml_path = model_dir / f"{cam_key}.xml"
        root = _new_model_root(
            cam_key,
            f"{spec.label} depth camera (visual, mount sites, box collision; "
            f"mass=0, no optical sensor; Roadmap 0.1.2).",
        )
        world = root.find("worldbody")
        assert world is not None
        _append_camera(root, world, cam_key)
        _write_xml(root, xml_path)
        generated.extend([xml_path, _write_scene(model_dir, cam_key, kind="camera")])

    # 2. Standalone camera_stand
    stand_dir = MODELS_DIR / "camera_stand"
    stand_xml = stand_dir / "camera_stand.xml"
    stand_root = _new_model_root(
        "camera_stand",
        "UFACTORY camera mounting bracket (visual + 175-piece convex collision; mass=0; Roadmap 0.1.2).",
    )
    world = stand_root.find("worldbody")
    assert world is not None
    _append_bracket(stand_root, world)
    _write_xml(stand_root, stand_xml)
    generated.extend([stand_xml, _write_scene(stand_dir, "camera_stand", kind="stand")])

    # 3. Four camera_stand_with_<cam> assemblies
    for cam_key, spec in CAMERAS.items():
        combo_name = f"camera_stand_with_{cam_key}"
        combo_dir = MODELS_DIR / combo_name
        _ensure_symlinks(combo_dir, ("camera_stand", cam_key))
        xml_path = combo_dir / f"{combo_name}.xml"
        root = _new_model_root(
            combo_name,
            f"UFACTORY camera mounting bracket assembled with {spec.label} (mass=0; Roadmap 0.1.2).",
        )
        world = root.find("worldbody")
        assert world is not None
        _append_bracket(root, world, with_camera=cam_key)
        _write_xml(root, xml_path)
        generated.extend([xml_path, _write_scene(combo_dir, combo_name, kind="stand")])

    # 4. Four xarm6_with_gripper_and_<cam> assemblies
    for cam_key, spec in CAMERAS.items():
        combo_name = f"xarm6_with_gripper_and_{cam_key}"
        combo_dir = MODELS_DIR / combo_name
        _ensure_symlinks(combo_dir, ("xarm6", "gripper_g2", "camera_stand", cam_key))
        xml_path = combo_dir / f"{combo_name}.xml"

        tree = ET.parse(XARM6_G2_XML)
        root = tree.getroot()
        root.set("model", combo_name)
        root.insert(
            0,
            ET.Comment(
                f" UFACTORY xArm 6 with Gripper G2, camera stand, and {spec.label} "
                f"(G2 shifted +3 mm along link6 +Z; Roadmap 0.1.2). "
            ),
        )
        link6 = root.find(".//body[@name='link6']")
        assert link6 is not None
        gripper_base = link6.find("./body[@name='xarm_gripper_base_link']")
        assert gripper_base is not None
        gripper_base.set("pos", "0 0 0.003")
        _append_bracket(root, link6, with_camera=cam_key, on_arm_flange=True)

        _write_xml(root, xml_path)
        generated.extend([xml_path, _write_scene(combo_dir, combo_name, kind="arm")])

    if verify:
        for xml_path in generated:
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            data = mujoco.MjData(model)
            if model.nkey > 0:
                mujoco.mj_resetDataKeyframe(model, data, 0)
            mujoco.mj_forward(model, data)

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sync-dev-meshes",
        action="store_true",
        help="Copy canonical STL meshes from dev/ into models/*/assets/ before generating XMLs.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip compiling generated XMLs with MuJoCo.",
    )
    args = parser.parse_args()

    if args.sync_dev_meshes:
        copied = sync_dev_meshes()
        print(f"Synced and SHA-256 verified {copied} canonical STL meshes from dev/.")

    generated = build_all(verify=not args.no_verify)
    print(f"Generated and verified {len(generated)} MJCF XML files across {len(generated) // 2} model directories.")


if __name__ == "__main__":
    main()
