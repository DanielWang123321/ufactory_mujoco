"""Runtime MJCF / MuJoCo 4.x ``MjSpec`` composer for long-tail robot and accessory assemblies.

The repository ships 19 atomic component directories (5 bare arms, 5 end effectors,
4 depth cameras, 1 camera bracket, 1 AI1500 F/T sensor, 3 base plates) plus the
5 primary task-scene arm+gripper assemblies statically under ``models/``. All 41
long-tail combinations (and arbitrary custom stacks) are composed on demand from
those atomic XMLs without duplicating combo XML directories or symlinks.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .robots import MODELS_ROOT, RESOURCE_ROOT

_ARMS = ("lite6", "xarm5", "xarm6", "xarm7", "uf850")
_XARM_FAMILY = ("xarm5", "xarm6", "xarm7", "uf850")
_CAMERAS = ("d405", "d435", "gemini_305", "zed_x_nano")

_FLANGE_BODY: dict[str, str] = {
    "lite6": "link6",
    "xarm5": "link5",
    "xarm6": "link6",
    "xarm7": "link7",
    "uf850": "link6",
}

# Elevated home/ready keyframes used by mounted end-effector / sensor / camera combos
# so downstream hardware stays clear of the floor in a tool-down orientation.
_COMBO_KEYFRAMES: dict[str, tuple[str, str]] = {
    "xarm5": (
        "0 -0.533 -1 0.5 0.3",
        "0 -0.63088336 -0.837867545 1.468750905 0",
    ),
    "xarm6": (
        "0 -0.631529917 -0.833150486 0 1.464680403 0",
        "0 -0.631529917 -0.833150486 0 1.464680403 0",
    ),
    "xarm7": (
        "0 -0.8 0 0 0 0 0",
        "0 -0.612697406 0 0.831413847 0 1.444111253 0",
    ),
    "uf850": (
        "0 0 0 0 0 0",
        "0 0.305871495 -0.534714507 0 -0.840586002 0",
    ),
}

_BIO_GRIPPER_MOUNT: dict[str, tuple[str, str | None]] = {
    "xarm5": ("0.026044 -0.000867 -0.000697", "0.999993824591 0 0 -0.00351436776576"),
    "xarm6": ("0.027018 -0.000715 -0.000692", "0.999993767297 0 0 -0.00353063266485"),
    "xarm7": ("0.026931 -0.000882 -0.00076", "0.99999230379 0 0 -0.0039233099351"),
    "uf850": ("0.024657 0 0", None),
}

# Base plate type, thickness (mm), and least-squares mounting fit of link_base on z=0.
_BASE_PLATE_THICKNESS_MM: dict[str, float] = {
    "xarm": 8.0,
    "uf850": 10.0,
    "lite6": 5.0,
}

_BASE_PLATE_MOUNT: dict[str, tuple[str, tuple[float, float, float], float]] = {
    "xarm5": ("xarm", (-0.00012433, -0.00038722, -0.00001016), 0.000002356),
    "xarm6": ("xarm", (0.00007243, 0.00002863, -0.00004715), -0.00726035),
    "xarm7": ("xarm", (-0.00057017, 0.00006319, 0.00001328), -0.00725905),
    "uf850": ("uf850", (-0.00001021, 0.00000350, 0.00000420), 0.0),
    "lite6": ("lite6", (0.00298920, 0.00001614, 0.00171732), 0.000000298),
}

# Camera mounting transforms in camera_stand coordinates.
_STAND_MOUNT_TARGET = np.array([0.0675, 0.003, 0.0], dtype=float)
_R_STANDARD = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=float)
_R_GEMINI = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
_CAMERA_MOUNT: dict[str, tuple[tuple[float, float, float], float]] = {
    "d405": ((0.0, 0.0, -0.023), 0.0),
    "d435": ((0.0, 0.0, -0.025), 0.000055),
    "gemini_305": ((0.018348755, 0.023076703, -0.017177002), 0.0),
    "zed_x_nano": ((0.0, 0.0, 0.0), 0.0),
}

_FT_TIMESTEP = "0.001"
_FT_HEADLIGHT_SPECULAR = "0.55 0.57 0.6"

_ARM_SCENE_TEMPLATE = """<mujoco model="{name} scene">
  <include file="{xml_file}"/>

  <statistic center="{center}" extent="{extent}"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="{specular}"/>
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
    <camera name="front" pos="1.2 0.8 0.9" quat="0.72 0 0.19 0.67"/>
    <camera name="side" pos="-0.3 1.6 0.8" quat="0.66 0.24 0.24 0.67"/>
  </worldbody>
</mujoco>
"""

COMPOSED_MODELS: tuple[str, ...] = tuple(
    sorted(
        [f"camera_stand_with_{cam}" for cam in _CAMERAS]
        + [f"{arm}_with_base_plate" for arm in _ARMS]
        + ["lite6_with_vacuum_gripper"]
        + [f"{arm}_with_bio_gripper" for arm in _XARM_FAMILY]
        + [f"{arm}_with_vacuum_gripper" for arm in _XARM_FAMILY]
        + [f"{arm}_with_d435" for arm in _XARM_FAMILY]
        + [f"{arm}_with_gripper_and_d435" for arm in _XARM_FAMILY]
        + [f"xarm6_with_gripper_and_{cam}" for cam in _CAMERAS if cam != "d435"]
        + [f"{arm}_with_ft_sensor" for arm in _XARM_FAMILY]
        + [f"{arm}_with_ft_sensor_and_gripper" for arm in _XARM_FAMILY]
        + [f"{arm}_with_ft_sensor_and_vacuum_gripper" for arm in _XARM_FAMILY]
    )
)
_COMPOSED_SET = frozenset(COMPOSED_MODELS)


def is_composed_model(name: str) -> bool:
    """Return True if ``name`` is one of the 41 dynamically composed named assemblies."""
    return name in _COMPOSED_SET


def _vec(values: tuple[float, ...] | list[float] | np.ndarray) -> str:
    return " ".join(f"{float(x):.10g}" for x in values)


def _quat_from_matrix(matrix: np.ndarray) -> str:
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return _vec((w, x, y, z))


def _resolve_mesh_file(rel_file: str, *, models_root: Path, resource_root: Path) -> Path:
    """Resolve a canonical mesh path across editable checkouts and wheel installs."""
    rel_path = Path(rel_file)
    if rel_path.is_absolute():
        return rel_path
    staged = resource_root / "meshes" / rel_file
    if staged.is_file():
        return staged.resolve()
    part = rel_file.split("/", 1)[0]
    editable = models_root / part / "assets" / rel_file
    if editable.is_file():
        return editable.resolve()
    # Allow overrides via UFACTORY_MUJOCO_MODELS pointing to a staged tree with sibling meshes/.
    alt_staged = models_root.parent / "meshes" / rel_file
    if alt_staged.is_file():
        return alt_staged.resolve()
    raise FileNotFoundError(f"Canonical mesh not found for {rel_file!r} under {models_root}")


def _rewrite_mesh_paths(
    root: ET.Element,
    *,
    models_root: Path,
    resource_root: Path,
    copy_to_assets_dir: Path | None = None,
) -> None:
    """Resolve every ``<mesh file=...>`` to an absolute canonical path or staged export dir."""
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler", angle="radian", autolimits="true")
        root.insert(0, compiler)
    if copy_to_assets_dir is not None:
        compiler.set("meshdir", "assets")
    else:
        compiler.attrib.pop("meshdir", None)

    for mesh in root.findall("./asset/mesh"):
        raw = mesh.attrib.get("file")
        if not raw:
            continue
        src = _resolve_mesh_file(raw, models_root=models_root, resource_root=resource_root)
        if copy_to_assets_dir is not None:
            dst = copy_to_assets_dir / raw
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copyfile(src, dst)
            mesh.set("file", raw)
        else:
            mesh.set("file", src.as_posix())


def _load_xml(models_root: Path, model_name: str, filename: str | None = None) -> ET.Element:
    xml_name = filename or f"{model_name}.xml"
    path = models_root / model_name / xml_name
    if not path.is_file():
        raise FileNotFoundError(f"Atomic MJCF not found: {path}")
    return ET.parse(path).getroot()


def _merge_sections(dst: ET.Element, src: ET.Element) -> None:
    """Merge assets, defaults, contacts, equalities, actuators, and sensors from ``src`` into ``dst``."""
    dst_asset = dst.find("asset")
    src_asset = src.find("asset")
    if src_asset is not None:
        if dst_asset is None:
            dst_asset = ET.SubElement(dst, "asset")
        existing_mat = {el.attrib.get("name") for el in dst_asset.findall("material")}
        existing_mesh = {el.attrib.get("name") for el in dst_asset.findall("mesh")}
        existing_tex = {el.attrib.get("name") for el in dst_asset.findall("texture")}
        for child in src_asset:
            name = child.attrib.get("name")
            if child.tag == "material" and name in existing_mat:
                continue
            if child.tag == "mesh" and name in existing_mesh:
                continue
            if child.tag == "texture" and name in existing_tex:
                continue
            dst_asset.append(copy.deepcopy(child))
            if child.tag == "material":
                existing_mat.add(name)
            elif child.tag == "mesh":
                existing_mesh.add(name)
            elif child.tag == "texture":
                existing_tex.add(name)

    src_default = src.find("default")
    if src_default is not None:
        dst_default = dst.find("default")
        if dst_default is None:
            dst_default = ET.SubElement(dst, "default")
        existing_cls = {el.attrib.get("class") for el in dst_default.findall(".//default") if "class" in el.attrib}
        for child in src_default:
            cls = child.attrib.get("class")
            if cls is not None and cls in existing_cls:
                continue
            dst_default.append(copy.deepcopy(child))
            if cls is not None:
                existing_cls.add(cls)

    src_contact = src.find("contact")
    if src_contact is not None:
        dst_contact = dst.find("contact")
        if dst_contact is None:
            dst_contact = ET.SubElement(dst, "contact")
        existing_pairs = {
            (e.attrib.get("body1"), e.attrib.get("body2"))
            for e in dst_contact.findall("exclude")
        }
        for child in src_contact:
            if child.tag == "exclude":
                pair = (child.attrib.get("body1"), child.attrib.get("body2"))
                if pair in existing_pairs:
                    continue
                existing_pairs.add(pair)
            dst_contact.append(copy.deepcopy(child))

    for tag in ("equality", "actuator", "sensor"):
        src_sec = src.find(tag)
        if src_sec is not None:
            dst_sec = dst.find(tag)
            if dst_sec is None:
                dst_sec = ET.SubElement(dst, tag)
            for child in src_sec:
                dst_sec.append(copy.deepcopy(child))


def _add_exclude(root: ET.Element, body1: str, body2: str) -> None:
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for ex in contact.findall("exclude"):
        if ex.attrib.get("body1") == body1 and ex.attrib.get("body2") == body2:
            return
    ET.SubElement(contact, "exclude", body1=body1, body2=body2)


def _set_keyframes(root: ET.Element, home_qpos: str, ready_qpos: str, home_ctrl: str, ready_ctrl: str) -> None:
    old = root.find("keyframe")
    if old is not None:
        root.remove(old)
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", name="home", qpos=home_qpos, ctrl=home_ctrl)
    ET.SubElement(kf, "key", name="ready", qpos=ready_qpos, ctrl=ready_ctrl)


def _attach_camera_to_stand(
    root: ET.Element,
    stand_body: ET.Element,
    cam_key: str,
    *,
    models_root: Path,
) -> None:
    cam_root = _load_xml(models_root, cam_key)
    _merge_sections(root, cam_root)
    cam_body = cam_root.find(f"./worldbody/body[@name='{cam_key}_body']")
    if cam_body is None:
        raise ValueError(f"Missing {cam_key}_body in {cam_key}.xml")
    cam_copy = copy.deepcopy(cam_body)
    mount, clearance = _CAMERA_MOUNT[cam_key]
    rot = _R_GEMINI if cam_key == "gemini_305" else _R_STANDARD
    cam_pos = _STAND_MOUNT_TARGET + np.array([0.0, clearance, 0.0]) - rot @ np.array(mount)
    cam_copy.set("pos", _vec(cam_pos))
    cam_copy.set("quat", _quat_from_matrix(rot))
    stand_body.append(cam_copy)


def _attach_stand_and_camera_to_flange(
    root: ET.Element,
    flange: ET.Element,
    cam_key: str,
    *,
    models_root: Path,
) -> None:
    stand_root = _load_xml(models_root, "camera_stand")
    _merge_sections(root, stand_root)
    stand_body = stand_root.find("./worldbody/body[@name='camera_stand']")
    if stand_body is None:
        raise ValueError("Missing camera_stand body in camera_stand.xml")
    stand_copy = copy.deepcopy(stand_body)
    stand_copy.set("quat", _vec((0.707106781, 0.707106781, 0.0, 0.0)))
    _attach_camera_to_stand(root, stand_copy, cam_key, models_root=models_root)
    flange.append(stand_copy)


def _attach_bio_gripper(root: ET.Element, arm: str, *, models_root: Path) -> None:
    flange_name = _FLANGE_BODY[arm]
    flange = root.find(f".//worldbody//body[@name='{flange_name}']")
    if flange is None:
        raise ValueError(f"Missing flange body {flange_name!r} in {arm}.xml")
    att = flange.find("site[@name='attachment_site']")
    if att is not None:
        flange.remove(att)

    bio_root = _load_xml(models_root, "bio_gripper_g2")
    _merge_sections(root, bio_root)
    bio_body = bio_root.find("./worldbody/body[@name='bio_gripper_g2_base_link']")
    if bio_body is None:
        raise ValueError("Missing bio_gripper_g2_base_link in bio_gripper_g2.xml")
    bio_copy = copy.deepcopy(bio_body)
    pos, quat = _BIO_GRIPPER_MOUNT[arm]
    bio_copy.set("pos", pos)
    if quat is not None:
        bio_copy.set("quat", quat)
    tcp = bio_copy.find("site[@name='link_tcp']")
    if tcp is not None:
        tcp.set("name", "attachment_site")
        tcp.set("size", "0.001")
        tcp.set("rgba", "1 0 0 1")
        tcp.set("group", "4")
    flange.append(bio_copy)
    _add_exclude(root, flange_name, "bio_gripper_g2_base_link")

    home_q, ready_q = _COMBO_KEYFRAMES[arm]
    _set_keyframes(
        root,
        f"{home_q} -0.0395 0.0395",
        f"{ready_q} -0.0395 0.0395",
        f"{home_q} 0.0395",
        f"{ready_q} 0.0395",
    )


def _attach_vacuum_gripper(root: ET.Element, arm: str, *, models_root: Path) -> None:
    flange_name = _FLANGE_BODY[arm]
    flange = root.find(f".//worldbody//body[@name='{flange_name}']")
    if flange is None:
        raise ValueError(f"Missing flange body {flange_name!r} in {arm}.xml")
    att = flange.find("site[@name='attachment_site']")
    if att is not None:
        flange.remove(att)

    is_lite = arm == "lite6"
    vg_name = "lite6_vacuum_gripper" if is_lite else "vacuum_gripper"
    vg_body_name = "uflite_vacuum_gripper_link" if is_lite else "xarm_vacuum_gripper_link"
    att_z = "0.061" if is_lite else "0.126"

    vg_root = _load_xml(models_root, vg_name)
    _merge_sections(root, vg_root)
    vg_body = vg_root.find(f"./worldbody/body[@name='{vg_body_name}']")
    if vg_body is None:
        raise ValueError(f"Missing {vg_body_name!r} in {vg_name}.xml")
    vg_copy = copy.deepcopy(vg_body)
    ET.SubElement(vg_copy, "site", name="attachment_site", pos=f"0 0 {att_z}", size="0.001", rgba="1 0 0 1", group="4")
    flange.append(vg_copy)

    prev_link = f"link{int(flange_name.removeprefix('link')) - 1}"
    _add_exclude(root, flange_name, vg_body_name)
    _add_exclude(root, prev_link, vg_body_name)
    if is_lite:
        _add_exclude(root, "link4", "link6")
        _add_exclude(root, "link4", vg_body_name)
    else:
        home_q, ready_q = _COMBO_KEYFRAMES[arm]
        _set_keyframes(root, home_q, ready_q, home_q, ready_q)


def _attach_ft_sensor(root: ET.Element, arm: str, tool: str | None, *, models_root: Path) -> None:
    option = root.find("option")
    if option is None:
        option = ET.Element("option", integrator="implicitfast")
        root.insert(1, option)
    option.set("timestep", _FT_TIMESTEP)

    flange_name = _FLANGE_BODY[arm]
    flange = root.find(f".//worldbody//body[@name='{flange_name}']")
    if flange is None:
        raise ValueError(f"Missing flange body {flange_name!r} in {arm}.xml")

    ft_root = _load_xml(models_root, "ft_sensor")
    _merge_sections(root, ft_root)
    ft_base = ft_root.find("./worldbody/body[@name='ft_sensor_base']")
    if ft_base is None:
        raise ValueError("Missing ft_sensor_base in ft_sensor.xml")
    ft_copy = copy.deepcopy(ft_base)
    ft_copy.attrib.pop("pos", None)
    ft_output = ft_copy.find("./body[@name='ft_sensor_output']")
    if ft_output is None:
        raise ValueError("Missing ft_sensor_output in ft_sensor.xml")

    if tool is None:
        flange_att = flange.find("site[@name='attachment_site']")
        if flange_att is not None:
            flange.remove(flange_att)
        flange.append(ft_copy)
        home_q, ready_q = _COMBO_KEYFRAMES[arm]
        _set_keyframes(root, home_q, ready_q, home_q, ready_q)
    else:
        tool_body_name = "xarm_gripper_base_link" if tool == "gripper_g2" else "xarm_vacuum_gripper_link"
        tool_body = flange.find(f"body[@name='{tool_body_name}']")
        if tool_body is None:
            raise ValueError(f"Missing tool body {tool_body_name!r} on {flange_name!r}")
        flange.remove(tool_body)
        out_att = ft_output.find("site[@name='attachment_site']")
        if out_att is not None:
            ft_output.remove(out_att)
        ft_output.append(tool_body)
        flange.append(ft_copy)
        _add_exclude(root, "ft_sensor_base", tool_body_name)


def _attach_base_plate(root: ET.Element, scene_root: ET.Element, arm: str, *, models_root: Path) -> None:
    plate, pos, yaw = _BASE_PLATE_MOUNT[arm]
    plate_model = f"{plate}_base_plate"
    floor_z = f"{-_BASE_PLATE_THICKNESS_MM[plate] / 1000.0:.6g}"

    world = root.find("worldbody")
    if world is None:
        raise ValueError(f"Missing worldbody in {arm}.xml")
    link_base = world.find("body[@name='link_base']")
    if link_base is None:
        raise ValueError(f"Missing link_base in {arm}.xml")
    link_base.set("pos", _vec(pos))
    if yaw != 0.0:
        link_base.set("quat", _vec((math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))))
    else:
        link_base.attrib.pop("quat", None)

    plate_root = _load_xml(models_root, plate_model)
    _merge_sections(root, plate_root)
    plate_body = plate_root.find("./worldbody/body[@name='base_plate']")
    if plate_body is None:
        raise ValueError(f"Missing base_plate in {plate_model}.xml")
    world.insert(0, copy.deepcopy(plate_body))

    floor = scene_root.find(".//geom[@name='floor']")
    if floor is not None:
        floor.set("pos", f"0 0 {floor_z}")


def _recipe(
    *,
    arm: str | None = None,
    base_plate: bool = False,
    ft_sensor: bool = False,
    gripper: str | None = None,
    camera: str | None = None,
    camera_stand: bool = False,
) -> dict:
    return {
        "arm": arm,
        "base_plate": base_plate,
        "ft_sensor": ft_sensor,
        "gripper": gripper,
        "camera": camera,
        "camera_stand": camera_stand,
    }


def parse_model_recipe(name: str) -> dict | None:
    """Parse any of the 41 named long-tail combinations into structured composition options."""
    if name not in _COMPOSED_SET:
        return None
    if name.startswith("camera_stand_with_"):
        return _recipe(camera=name.removeprefix("camera_stand_with_"), camera_stand=True)
    arm, rest = name.split("_with_", 1)
    if rest == "base_plate":
        return _recipe(arm=arm, base_plate=True)
    if rest == "bio_gripper":
        return _recipe(arm=arm, gripper="bio_gripper_g2")
    if rest == "vacuum_gripper":
        gripper = "lite6_vacuum_gripper" if arm == "lite6" else "vacuum_gripper"
        return _recipe(arm=arm, gripper=gripper)
    if rest in _CAMERAS:
        return _recipe(arm=arm, camera=rest, camera_stand=True)
    if rest.startswith("gripper_and_"):
        return _recipe(arm=arm, gripper="gripper_g2", camera=rest.removeprefix("gripper_and_"), camera_stand=True)
    if rest == "ft_sensor":
        return _recipe(arm=arm, ft_sensor=True)
    if rest == "ft_sensor_and_gripper":
        return _recipe(arm=arm, ft_sensor=True, gripper="gripper_g2")
    if rest == "ft_sensor_and_vacuum_gripper":
        return _recipe(arm=arm, ft_sensor=True, gripper="vacuum_gripper")
    return None


def _indent_xml(root: ET.Element) -> str:
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def compose_xml(
    name: str | None = None,
    *,
    arm: str | None = None,
    base_plate: bool = False,
    ft_sensor: bool = False,
    gripper: str | None = None,
    camera: str | None = None,
    camera_stand: bool = False,
    models_root: Path | None = None,
    resource_root: Path | None = None,
    copy_to_assets_dir: Path | None = None,
) -> tuple[str, str]:
    """Compose ``(model_xml, scene_xml)`` strings from atomic MJCF assets.

    Pass either a known model ``name`` (any of the 41 composed names or 24 static names)
    or explicit component arguments (``arm``, ``base_plate``, ``ft_sensor``, ``gripper``,
    ``camera``).
    """
    mroot = Path(models_root) if models_root is not None else MODELS_ROOT
    rroot = Path(resource_root) if resource_root is not None else RESOURCE_ROOT

    if name is not None:
        recipe = parse_model_recipe(name)
        if recipe is None:
            static_dir = mroot / name
            if (static_dir / f"{name}.xml").is_file() and (static_dir / "scene.xml").is_file():
                root = _load_xml(mroot, name)
                scene_root = _load_xml(mroot, name, "scene.xml")
                _rewrite_mesh_paths(root, models_root=mroot, resource_root=rroot, copy_to_assets_dir=copy_to_assets_dir)
                return _indent_xml(root), _indent_xml(scene_root)
            raise ValueError(f"Unknown model or recipe: {name!r}")
        arm = recipe["arm"]
        base_plate = recipe["base_plate"]
        ft_sensor = recipe["ft_sensor"]
        gripper = recipe["gripper"]
        camera = recipe["camera"]
        camera_stand = recipe["camera_stand"]
        model_name = name
    else:
        parts = []
        if arm:
            parts.append(arm)
        elif camera_stand:
            parts.append("camera_stand")
        if base_plate:
            parts.append("base_plate")
        if ft_sensor:
            parts.append("ft_sensor")
        if gripper:
            parts.append(gripper)
        if camera:
            parts.append(camera)
        if not parts:
            raise ValueError("Specify either a model name or at least one component")
        model_name = "_with_".join([parts[0], "_and_".join(parts[1:])]) if len(parts) > 1 else parts[0]

    # Case 1: camera_stand + camera (no arm)
    if arm is None:
        if not (camera_stand and camera in _CAMERAS):
            raise ValueError(f"Unsupported non-arm composition: camera_stand={camera_stand}, camera={camera}")
        root = _load_xml(mroot, "camera_stand")
        root.set("model", model_name)
        stand_body = root.find("./worldbody/body[@name='camera_stand']")
        assert stand_body is not None
        _attach_camera_to_stand(root, stand_body, camera, models_root=mroot)
        scene_root = _load_xml(mroot, "camera_stand", "scene.xml")
        scene_root.set("model", f"{model_name} scene")
        inc = scene_root.find("include")
        assert inc is not None
        inc.set("file", f"{model_name}.xml")
        _rewrite_mesh_paths(root, models_root=mroot, resource_root=rroot, copy_to_assets_dir=copy_to_assets_dir)
        return _indent_xml(root), _indent_xml(scene_root)

    if arm not in _ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {_ARMS}")

    # Choose base XML: reuse static <arm>_with_gripper when Gripper G2 is included.
    if gripper in ("gripper_g2", "lite6_gripper"):
        base_name = f"{arm}_with_gripper"
        root = _load_xml(mroot, base_name)
        scene_root = _load_xml(mroot, base_name, "scene.xml")
    else:
        root = _load_xml(mroot, arm)
        scene_root = _load_xml(mroot, arm, "scene.xml")
        if gripper == "bio_gripper_g2":
            _attach_bio_gripper(root, arm, models_root=mroot)
            scene_root = ET.fromstring(
                _ARM_SCENE_TEMPLATE.format(
                    name=model_name, xml_file=f"{model_name}.xml",
                    center="0.3 0 0.3", extent="0.9", specular="0 0 0", floor_pos="",
                )
            )
        elif gripper in ("vacuum_gripper", "lite6_vacuum_gripper"):
            _attach_vacuum_gripper(root, arm, models_root=mroot)
            if arm == "lite6":
                center, extent = "0.25 0 0.25", "0.7"
            else:
                center, extent = "0.3 0 0.3", "0.9"
            scene_root = ET.fromstring(
                _ARM_SCENE_TEMPLATE.format(
                    name=model_name, xml_file=f"{model_name}.xml",
                    center=center, extent=extent, specular="0 0 0", floor_pos="",
                )
            )
        elif gripper is not None:
            raise ValueError(f"Unsupported gripper for composition: {gripper!r}")

    root.set("model", model_name)

    if camera is not None:
        if camera not in _CAMERAS:
            raise ValueError(f"Unknown camera {camera!r}; expected one of {_CAMERAS}")
        flange_name = _FLANGE_BODY[arm]
        flange = root.find(f".//worldbody//body[@name='{flange_name}']")
        assert flange is not None
        if gripper == "gripper_g2":
            g2_base = flange.find("./body[@name='xarm_gripper_base_link']")
            if g2_base is None:
                raise ValueError(f"Missing xarm_gripper_base_link on {flange_name}")
            g2_base.set("pos", "0 0 0.003")
        else:
            att = flange.find("./site[@name='attachment_site']")
            if att is not None:
                att.set("pos", "0 0 0.003")
            home_q, ready_q = _COMBO_KEYFRAMES[arm]
            _set_keyframes(root, home_q, ready_q, home_q, ready_q)
        _attach_stand_and_camera_to_flange(root, flange, camera, models_root=mroot)
        scene_root = ET.fromstring(
            _ARM_SCENE_TEMPLATE.format(
                name=model_name, xml_file=f"{model_name}.xml",
                center="0.3 0 0.3", extent="0.9", specular="0 0 0", floor_pos="",
            )
        )

    if ft_sensor:
        if arm not in _XARM_FAMILY:
            raise ValueError(f"ft_sensor is not supported on {arm!r}")
        if gripper not in (None, "gripper_g2", "vacuum_gripper"):
            raise ValueError(f"ft_sensor composition does not support gripper {gripper!r}")
        _attach_ft_sensor(root, arm, gripper, models_root=mroot)
        hl = scene_root.find("./visual/headlight")
        if hl is not None:
            hl.set("specular", _FT_HEADLIGHT_SPECULAR)

    if base_plate:
        _attach_base_plate(root, scene_root, arm, models_root=mroot)

    scene_root.set("model", f"{model_name} scene")
    inc = scene_root.find("include")
    if inc is not None:
        inc.set("file", f"{model_name}.xml")

    _rewrite_mesh_paths(root, models_root=mroot, resource_root=rroot, copy_to_assets_dir=copy_to_assets_dir)
    return _indent_xml(root), _indent_xml(scene_root)


def _merge_scene_for_memory(model_xml: str, scene_xml: str) -> str:
    """Inline ``scene_xml`` elements into ``model_xml`` so ``MjSpec.from_string`` needs no disk include."""
    m_root = ET.fromstring(model_xml)
    s_root = ET.fromstring(scene_xml)
    m_root.set("model", s_root.attrib.get("model", m_root.attrib.get("model", "scene")))
    for child in s_root:
        if child.tag == "include":
            continue
        if child.tag in ("statistic", "visual"):
            old = m_root.find(child.tag)
            if old is not None:
                m_root.remove(old)
            m_root.append(copy.deepcopy(child))
        elif child.tag == "asset":
            dst = m_root.find("asset")
            if dst is None:
                dst = ET.SubElement(m_root, "asset")
            for item in child:
                dst.append(copy.deepcopy(item))
        elif child.tag == "worldbody":
            dst = m_root.find("worldbody")
            if dst is None:
                dst = ET.SubElement(m_root, "worldbody")
            for idx, item in enumerate(child):
                dst.insert(idx, copy.deepcopy(item))
    return _indent_xml(m_root)


def compose_spec(
    name: str | None = None,
    *,
    arm: str | None = None,
    base_plate: bool = False,
    ft_sensor: bool = False,
    gripper: str | None = None,
    camera: str | None = None,
    camera_stand: bool = False,
    scene: bool = True,
    models_root: Path | None = None,
    resource_root: Path | None = None,
) -> mujoco.MjSpec:
    """Return an in-memory MuJoCo 4.x ``MjSpec`` for a composed robot/accessory model."""
    model_xml, scene_xml = compose_xml(
        name,
        arm=arm,
        base_plate=base_plate,
        ft_sensor=ft_sensor,
        gripper=gripper,
        camera=camera,
        camera_stand=camera_stand,
        models_root=models_root,
        resource_root=resource_root,
    )
    xml_text = _merge_scene_for_memory(model_xml, scene_xml) if scene else model_xml
    return mujoco.MjSpec.from_string(xml_text)


def compose_model(
    name: str | None = None,
    *,
    arm: str | None = None,
    base_plate: bool = False,
    ft_sensor: bool = False,
    gripper: str | None = None,
    camera: str | None = None,
    camera_stand: bool = False,
    scene: bool = True,
    models_root: Path | None = None,
    resource_root: Path | None = None,
) -> mujoco.MjModel:
    """Compile and return a ``mujoco.MjModel`` directly in memory via ``MjSpec``."""
    return compose_spec(
        name,
        arm=arm,
        base_plate=base_plate,
        ft_sensor=ft_sensor,
        gripper=gripper,
        camera=camera,
        camera_stand=camera_stand,
        scene=scene,
        models_root=models_root,
        resource_root=resource_root,
    ).compile()


def _atomic_write(path: Path, text: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _cache_root() -> Path:
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return Path(tempfile.gettempdir()) / f"ufactory_mujoco_composed_{uid}"


def resolve_composed_path(
    name: str,
    *,
    scene: bool = True,
    models_root: Path | None = None,
    resource_root: Path | None = None,
) -> Path:
    """Materialize a composed model into a deterministic writable cache dir and return its XML path."""
    mroot = Path(models_root) if models_root is not None else MODELS_ROOT
    rroot = Path(resource_root) if resource_root is not None else RESOURCE_ROOT
    model_xml, scene_xml = compose_xml(name, models_root=mroot, resource_root=rroot)
    digest = hashlib.sha256((model_xml + "\0" + scene_xml).encode("utf-8")).hexdigest()[:12]
    target_dir = _cache_root() / f"{name}_{digest}"
    model_path = target_dir / f"{name}.xml"
    scene_path = target_dir / "scene.xml"
    _atomic_write(model_path, model_xml)
    _atomic_write(scene_path, scene_xml)
    return scene_path if scene else model_path


def export_model(
    name: str | None,
    out_dir: str | Path,
    *,
    arm: str | None = None,
    base_plate: bool = False,
    ft_sensor: bool = False,
    gripper: str | None = None,
    camera: str | None = None,
    camera_stand: bool = False,
    copy_meshes: bool = False,
    models_root: Path | None = None,
    resource_root: Path | None = None,
) -> Path:
    """Export a standalone loadable ``<name>.xml`` and ``scene.xml`` to ``out_dir`` for MuJoCo ``simulate``."""
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    assets_dir = out / "assets" if copy_meshes else None
    model_xml, scene_xml = compose_xml(
        name,
        arm=arm,
        base_plate=base_plate,
        ft_sensor=ft_sensor,
        gripper=gripper,
        camera=camera,
        camera_stand=camera_stand,
        models_root=models_root,
        resource_root=resource_root,
        copy_to_assets_dir=assets_dir,
    )
    model_root = ET.fromstring(model_xml)
    xml_stem = model_root.attrib["model"]
    model_path = out / f"{xml_stem}.xml"
    scene_path = out / "scene.xml"
    _atomic_write(model_path, model_xml)
    _atomic_write(scene_path, scene_xml)
    return scene_path
