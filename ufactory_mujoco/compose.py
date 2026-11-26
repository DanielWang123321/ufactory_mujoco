"""Assemble composite models from the atomic MJCF parts with MuJoCo's native ``MjSpec.attach``.

Every arm + end effector, F/T sensor, camera stand + camera and base plate combination, and
every pick-place / box-pack scene, is built in memory from the parts under ``models/`` and
the data in ``robots``. Each build is serialized once with ``MjSpec.to_xml()``; in-memory
loads, exported files and cached paths all come from that XML.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import warnings

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from . import robots
from .robots import ARMS, MODELS_ROOT, RECIPES, TOOLS, ConfigError, ModelResourceError, Recipe

COMPOSED_MODELS: tuple[str, ...] = tuple(RECIPES)
_EMPTY_PREFIX = "Attaching a spec with an empty prefix"
_DEFAULT_SITE_RGBA = (0.5, 0.5, 0.5, 1.0)
_STUB = b"<mujoco><worldbody/></mujoco>"


def is_composed_model(name: str) -> bool:
    """Return True if ``name`` is a composite built at runtime."""
    return name in RECIPES


def _round(values) -> list[float]:
    """Values as 10 significant digits, the precision of the validated MJCF text."""
    return [float(f"{float(v):.10g}") for v in values]


def _quat_from_matrix(matrix) -> list[float]:
    x, y, z, w = Rotation.from_matrix(np.asarray(matrix, dtype=float)).as_quat()
    return _round((w, x, y, z))


def _on_mujoco_warning(message: str) -> None:
    if not message.startswith(_EMPTY_PREFIX):
        warnings.warn(f"MuJoCo: {message}", RuntimeWarning, stacklevel=2)


@contextlib.contextmanager
def _attaching():
    """Keep names unprefixed: drop only MuJoCo's empty-prefix notice (stderr and MUJOCO_LOG.TXT)."""
    previous = mujoco.get_mju_user_warning()
    mujoco.set_mju_user_warning(_on_mujoco_warning)
    try:
        yield
    finally:
        mujoco.set_mju_user_warning(previous)


def _load(path: Path, include: dict[str, bytes] | None = None) -> mujoco.MjSpec:
    """Load an MJCF file with mesh paths made absolute, so attached parts keep their assets."""
    if not path.is_file():
        raise FileNotFoundError(f"MJCF not found: {path}")
    spec = mujoco.MjSpec.from_file(str(path), include=include)
    meshdir = Path(spec.modelfiledir) / spec.meshdir
    for mesh in spec.meshes:
        if mesh.file and not os.path.isabs(mesh.file):
            mesh.file = str((meshdir / mesh.file).resolve())
    spec.meshdir = ""
    return spec


def _part(name: str, root: Path) -> mujoco.MjSpec:
    return _load(root / name / f"{name}.xml")


def _env(name: str, root: Path) -> mujoco.MjSpec:
    """The part's ``scene.xml`` without the part: visual, statistic, lights, floor, cameras."""
    return _load(root / name / "scene.xml", include={f"{name}.xml": _STUB})


def _attach(parent: mujoco.MjSpec, child: mujoco.MjSpec, *, site_rgba=None, **where) -> None:
    """Attach ``child`` without its keyframes; same-named materials resolve to the parent's.

    ``site_rgba`` recolors the child's sites that use MuJoCo's default color, as parts placed
    on an arm take the arm's site style.
    """
    for key in list(child.keys):
        child.delete(key)
    if site_rgba is not None:
        for child_site in child.sites:
            if tuple(child_site.rgba) == _DEFAULT_SITE_RGBA:
                child_site.rgba = list(site_rgba)
    taken = {material.name for material in parent.materials}
    for material in list(child.materials):
        if material.name in taken:  # the parent's material of the same name wins
            child.delete(material)
    if child.option.timestep < parent.option.timestep:
        parent.option.timestep = child.option.timestep
    with _attaching():
        parent.attach(child, prefix="", **where)


def _mount(parent: mujoco.MjSpec, child: mujoco.MjSpec, root: str, *, site: str = "attachment_site",
           pos=(0, 0, 0), quat=None, site_rgba=None) -> None:
    """Attach ``child`` with body ``root`` placed at ``site``, which is then removed."""
    body = child.body(root)
    body.pos = list(pos)
    body.quat = list(quat) if quat is not None else [1, 0, 0, 0]
    anchor = parent.site(site)
    anchor.name = "_mount"
    _attach(parent, child, site=anchor, site_rgba=site_rgba)
    parent.delete(parent.site("_mount"))


def _open_key(part: mujoco.MjSpec) -> tuple[list[float], list[float]]:
    key = next((k for k in part.keys if k.name == "open"), None)
    return (list(key.qpos), list(key.ctrl)) if key is not None else ([], [])


def _mount_tool(spec: mujoco.MjSpec, tool_key: str, root: Path, *, site: str, flange: str, prev: str,
                site_style, pos=(0, 0, 0), quat=None) -> tuple[list[float], list[float]]:
    tool = TOOLS[tool_key]
    part = _part(tool_key, root)
    tail = _open_key(part)
    if tool.mesh_prefix:
        renamed = {mesh.name: tool.mesh_prefix + mesh.name for mesh in part.meshes}
        for mesh in part.meshes:
            mesh.name = renamed[mesh.name]
        for geom in part.geoms:
            if geom.meshname in renamed:
                geom.meshname = renamed[geom.meshname]
    if tool.actuator:
        part.actuators[0].name = tool.actuator
    if tool.drop_root_excludes:
        for exclude in list(part.excludes):
            if tool.root in (exclude.bodyname1, exclude.bodyname2):
                part.delete(exclude)
    _mount(spec, part, tool.root, site=site, pos=pos, quat=quat, site_rgba=site_style[1])
    if tool.tcp:
        tcp = spec.site(tool.tcp)
        tcp.name = "attachment_site"
    else:
        tcp = spec.body(tool.root).add_site(name="attachment_site", pos=list(spec.site("suction_site").pos))
    tcp.size, tcp.rgba, tcp.group = site_style
    names = {"{flange}": flange, "{prev}": prev, "{root}": tool.root}
    for body1, body2 in tool.excludes:
        spec.add_exclude(bodyname1=names.get(body1, body1), bodyname2=names.get(body2, body2))
    return tail


def _mount_camera(stand: mujoco.MjSpec, camera: str, root: Path) -> None:
    offset, clearance, rotation = robots.CAMERA_MOUNT[camera]
    part = _part(camera, root)
    target = np.array(robots.STAND_MOUNT_TARGET) + np.array([0.0, clearance, 0.0])
    body = part.body(f"{camera}_body")
    body.pos = _round(target - np.asarray(rotation, dtype=float) @ np.asarray(offset))
    body.quat = _quat_from_matrix(rotation)
    _attach(stand, part, frame=stand.body("camera_stand").add_frame())


def _build_robot(recipe: Recipe, name: str, root: Path) -> tuple[mujoco.MjSpec, dict[str, Any]]:
    """Model spec of ``recipe`` and the scene adjustments it needs."""
    scene: dict[str, Any] = {"env": recipe.arm or "camera_stand"}
    if recipe.arm is None:
        spec = _part("camera_stand", root)
        _mount_camera(spec, recipe.camera, root)
        spec.modelname = name
        return spec, scene

    arm = recipe.arm
    if recipe.base_plate:
        plate, pos, yaw = robots.BASE_PLATE_MOUNT[arm]
        spec = _part(plate, root)
        arm_part = _part(arm, root)
        link_base = arm_part.body("link_base")
        link_base.pos = _round(pos)
        if yaw:
            link_base.quat = _round((math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)))
        keys = [(k.name, list(k.qpos), list(k.ctrl)) for k in arm_part.keys]
        _attach(spec, arm_part, frame=spec.worldbody.add_frame())
        for key_name, qpos, ctrl in keys:
            spec.add_key(name=key_name, qpos=qpos, ctrl=ctrl)
        scene["floor_z"] = _env(plate, root).geom("floor").pos[2]
    else:
        spec = _part(arm, root)
    spec.modelname = name
    arm_keys = [(k.name, list(k.qpos), list(k.ctrl)) for k in spec.keys]

    anchor = spec.site("attachment_site")
    flange = anchor.parent.name
    prev = spec.body(flange).parent.name
    site_style = (list(anchor.size), list(anchor.rgba), anchor.group)
    tails: list[tuple[list[float], list[float]]] = []
    tool_site = "attachment_site"
    if recipe.ft_sensor:
        if arm not in robots.COMBO_READY:
            raise ValueError(f"ft_sensor is not supported on {arm!r}")
        _mount(spec, _part("ft_sensor", root), "ft_sensor_base", site_rgba=site_style[1])
        scene["headlight_specular"] = list(_env("ft_sensor", root).visual.headlight.specular)
    if recipe.tool is not None:
        tool_cfg = TOOLS[recipe.tool]
        if tool_cfg.adapter is not None:
            adapter_pos = (0, 0, robots.STAND_FLANGE_LIFT if recipe.camera is not None else 0.0)
            _mount(spec, _part(tool_cfg.adapter, root), tool_cfg.adapter, site=tool_site,
                   pos=adapter_pos, quat=None, site_rgba=site_style[1])
            spec.add_exclude(bodyname1=flange, bodyname2=tool_cfg.adapter)
            if recipe.camera is not None:
                spec.add_exclude(bodyname1="camera_stand", bodyname2=tool_cfg.adapter)
            spec.body(tool_cfg.adapter).add_site(name="_adapter_tool_mount", pos=[0, 0, robots.ROBOTIQ_MOUNTING_LIFT])
            scene.setdefault("headlight_specular", list(_env(recipe.tool, root).visual.headlight.specular))
            tails.append(_mount_tool(spec, recipe.tool, root, site="_adapter_tool_mount", flange=tool_cfg.adapter,
                                     prev=flange, site_style=site_style, pos=(0, 0, 0), quat=None))
        else:
            pos, quat = (0, 0, 0), None
            if recipe.tool == "bio_gripper_g2":
                pos, quat = robots.BIO_GRIPPER_MOUNT[arm]
                scene.setdefault("headlight_specular", list(_env("bio_gripper_g2", root).visual.headlight.specular))
            else:
                if recipe.tool == "gripper_g2":
                    scene.setdefault("headlight_specular", list(_env("gripper_g2", root).visual.headlight.specular))
                if recipe.camera is not None:
                    pos = (0, 0, robots.STAND_FLANGE_LIFT)
            tails.append(_mount_tool(spec, recipe.tool, root, site=tool_site, flange=flange, prev=prev,
                                     site_style=site_style, pos=pos, quat=quat))
        if recipe.ft_sensor:
            spec.add_exclude(bodyname1="ft_sensor_base", bodyname2=tool_cfg.root)
    if recipe.camera is not None:
        scene.setdefault("headlight_specular", list(_env("camera_stand", root).visual.headlight.specular))
        if recipe.tool is None:
            spec.site("attachment_site").pos = [0, 0, robots.STAND_FLANGE_LIFT]
        stand = _part("camera_stand", root)
        stand.body("camera_stand").quat = list(robots.STAND_FLANGE_QUAT)
        _mount_camera(stand, recipe.camera, root)
        _attach(spec, stand, frame=spec.body(flange).add_frame(), site_rgba=site_style[1])
    if recipe.tool is not None or recipe.camera is not None:
        scene["statistic"] = robots.COMBO_STATISTIC[arm]

    mounted = recipe.tool is not None or recipe.camera is not None or recipe.ft_sensor
    if mounted and arm in robots.COMBO_READY:
        ready = [float(v) for v in robots.COMBO_READY[arm]]
        zero = [0.0] * len(ready)
        arm_keys = [("home", zero, zero), ("ready", ready, ready)]
        if robots.needs_combo_init(recipe):
            arm_keys.append(("init", ready, ready))
    if mounted:
        for key in list(spec.keys):
            spec.delete(key)
        qtail = [v for qpos, _ in tails for v in qpos]
        ctail = [v for _, ctrl in tails for v in ctrl]
        for key_name, qpos, ctrl in arm_keys:
            spec.add_key(name=key_name, qpos=qpos + qtail, ctrl=ctrl + ctail)
    return spec, scene


def _add_env(spec: mujoco.MjSpec, env: mujoco.MjSpec) -> None:
    """Attach ``env``'s world (lights, floor, cameras, props) and adopt its visual and statistic."""
    for group in ("global_", "quality", "headlight", "map", "scale", "rgba"):
        source, target = getattr(env.visual, group), getattr(spec.visual, group)
        for attr in dir(source):
            if not attr.startswith("_") and not callable(getattr(source, attr)):
                setattr(target, attr, getattr(source, attr))
    for attr in ("center", "extent", "meaninertia", "meanmass", "meansize"):
        setattr(spec.stat, attr, getattr(env.stat, attr))
    _attach(spec, env, frame=spec.worldbody.add_frame())


def _scene_env(scene: dict[str, Any], root: Path) -> mujoco.MjSpec:
    env = _env(scene["env"], root)
    if "statistic" in scene:
        center, extent = scene["statistic"]
        env.stat.center, env.stat.extent = list(center), extent
    if "headlight_specular" in scene:
        env.visual.headlight.specular = scene["headlight_specular"]
    if "floor_z" in scene:
        env.geom("floor").pos = [0, 0, scene["floor_z"]]
    return env


def _to_xml(spec: mujoco.MjSpec) -> str:
    # Write the authored values exactly; MuJoCo's default writes compiled values with 6 digits.
    spec.compiler.savecompiled = False
    spec.compiler.savecanonical = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_EMPTY_PREFIX, category=UserWarning)
        return spec.to_xml()


def _recipe_for(name: str | None, arm=None, base_plate=False, ft_sensor=False, gripper=None, camera=None,
                camera_stand=False) -> tuple[str, Recipe]:
    if name is not None:
        recipe = RECIPES.get(name)
        if recipe is None:
            raise ValueError(f"Unknown model or recipe: {name!r}")
        return name, recipe
    if not (arm or camera_stand or base_plate or ft_sensor or gripper or camera):
        raise ValueError("Specify either a model name or at least one component")
    if arm is None:
        if not (camera_stand and camera in robots.CAMERAS) or base_plate or ft_sensor or gripper:
            raise ValueError(f"Unsupported non-arm composition: camera_stand={camera_stand}, camera={camera}")
    elif arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {tuple(ARMS)}")
    if camera is not None and camera not in robots.CAMERAS:
        raise ValueError(f"Unknown camera {camera!r}; expected one of {robots.CAMERAS}")
    if gripper is not None and gripper not in TOOLS:
        raise ValueError(f"Unsupported gripper for composition: {gripper!r}")
    if arm is not None and gripper is not None and gripper not in ARMS[arm].grippers:
        raise ValueError(f"{arm!r} does not support gripper {gripper!r}")
    if gripper == "bio_gripper_g2" and camera is not None:
        raise ValueError("camera_stand does not fit with 'bio_gripper_g2'")
    if ft_sensor and gripper not in (None, "gripper_g2", "vacuum_gripper"):
        raise ValueError(f"ft_sensor composition does not support gripper {gripper!r}")
    parts = [p for p, on in (("base_plate", base_plate), ("ft_sensor", ft_sensor)) if on]
    parts += [p for p in (gripper, camera) if p]
    model_name = f"{arm or 'camera_stand'}_with_{'_and_'.join(parts)}" if parts else arm
    return model_name, Recipe(arm, gripper, ft_sensor, camera, base_plate)


@functools.lru_cache(maxsize=256)
def _composite_xml(name: str, recipe: Recipe, scene: bool, root: Path) -> str:
    spec, adjust = _build_robot(recipe, name, root)
    if scene:
        spec.modelname = f"{name} scene"
        _add_env(spec, _scene_env(adjust, root))
    return _to_xml(spec)


@functools.lru_cache(maxsize=64)
def _static_xml(name: str, scene: bool, root: Path) -> str:
    return _to_xml(_load(root / name / ("scene.xml" if scene else f"{name}.xml")))


def _xml(name: str | None, scene: bool, models_root: Path | None, **components) -> str:
    root = Path(models_root) if models_root is not None else MODELS_ROOT
    static = name if name is not None and name not in RECIPES else None
    if static is None and name is None and components.get("arm") and not any(
            v for k, v in components.items() if k != "arm"):
        static = components["arm"]
    if static is not None and (root / static / f"{static}.xml").is_file() and (root / static / "scene.xml").is_file():
        return _static_xml(static, scene, root)
    model_name, recipe = _recipe_for(name, **components)
    return _composite_xml(model_name, recipe, scene, root)


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
) -> mujoco.MjSpec:
    """Return an ``MjSpec`` for a named model (static or composite) or explicit components."""
    xml = _xml(name, scene, models_root, arm=arm, base_plate=base_plate, ft_sensor=ft_sensor,
               gripper=gripper, camera=camera, camera_stand=camera_stand)
    return mujoco.MjSpec.from_string(xml)


def compose_model(name: str | None = None, *, scene: bool = True, **components) -> mujoco.MjModel:
    """Compile ``compose_spec(name, ...)`` into an ``MjModel``."""
    return compose_spec(name, scene=scene, **components).compile()


def task_scene_spec(robot: str, task: str, params: Any, models_root: Path | None = None) -> mujoco.MjSpec:
    """Pick-place or box-pack scene of ``robot`` with props placed from task ``params``."""
    root = Path(models_root) if models_root is not None else MODELS_ROOT
    return mujoco.MjSpec.from_string(_task_xml(robot, task, params, root))


def task_scene_path(robot: str, task: str, params: Any, models_root: Path | None = None) -> Path:
    """Export ``task_scene_spec(...)`` to the cache and return its path."""
    root = Path(models_root) if models_root is not None else MODELS_ROOT
    name = robots.combo_name(robot, robots.TASK_GRIPPERS[robot])
    return cached_xml_path(name, _task_xml(robot, task, params, root), f"scene_{task}.xml", root)


# MuJoCo's soft friction lets a held prop creep out of the fingers; the noslip pass stops it.
_TASK_NOSLIP_ITERATIONS = 3


@functools.lru_cache(maxsize=32)
def _task_xml(robot: str, task: str, params: Any, root: Path) -> str:
    name = robots.combo_name(robot, robots.TASK_GRIPPERS[robot])
    spec, _ = _build_robot(RECIPES[name], name, root)
    spec.modelname = f"{name} {task} scene"
    env = _env(robot, root)
    env.stat.center = list(robots.TASK_STATISTIC_CENTER[(robot, task)])
    env.stat.extent = robots.TASK_STATISTIC_EXTENT
    _add_task_props(env.worldbody, robots.TASK_STANDS.get(robot, {}).get(task), task, params)
    _add_env(spec, env)
    spec.option.noslip_iterations = _TASK_NOSLIP_ITERATIONS
    return _to_xml(spec)


_CUBE_RGBA = ((0.9, 0.1, 0.1, 1), (0.1, 0.8, 0.2, 1), (0.2, 0.4, 0.9, 1))
_PROP_FRICTION = (1, 1, 0.005)
_GRAY = (0.4, 0.4, 0.4, 1)
_BOX = mujoco.mjtGeom.mjGEOM_BOX


def _add_task_props(world, stand, task: str, p: Any) -> None:
    lift = 2 * stand[2] if stand else 0.0
    cube_z = _round([p.cube_half_m + lift])[0]

    def cube(name: str, xy, rgba) -> None:
        body = world.add_body(name=name, pos=[*xy, cube_z])
        body.add_freejoint(name=name)
        body.add_geom(name=f"{name}_geom", type=_BOX, size=[p.cube_half_m] * 3, mass=p.cube_mass_kg,
                      rgba=list(rgba), friction=list(_PROP_FRICTION))

    def stand_at(name: str, xy) -> None:
        world.add_geom(name=name, type=_BOX, size=list(stand), pos=[*xy, stand[2]], rgba=list(_GRAY))

    if task == "pick_place":
        if stand:
            stand_at("pick_stand", p.cube_xy)
            stand_at("place_stand", p.target_xy)
        cube("cube", p.cube_xy, _CUBE_RGBA[0])
        world.add_geom(name="target_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0],
                       pos=[*p.target_xy, cube_z], rgba=[0.1, 0.9, 0.1, 1], contype=0, conaffinity=0)
        return
    if stand:
        for i, xy in enumerate(p.feed_xy):
            stand_at(f"feed_stand_{i}", xy)
    for i, xy in enumerate(p.feed_xy):
        cube(f"cube_{i}", xy, _CUBE_RGBA[i % len(_CUBE_RGBA)])
    hx, hy = p.box_inner_half_xy
    wall, floor, rim = p.box_wall_m, p.box_floor_m, p.box_rim_z_m
    wall_hz, wall_z = (rim - floor) / 2, floor + (rim - floor) / 2
    box = world.add_body(name="pack_box", pos=[*p.box_center_xy, 0])
    walls = (
        ("box_floor", (hx + wall, hy + wall, floor / 2), (0, 0, floor / 2), (0.55, 0.4, 0.25, 1)),
        ("box_wall_px", (wall / 2, hy + wall, wall_hz), (hx + wall / 2, 0, wall_z), (0.6, 0.45, 0.3, 1)),
        ("box_wall_nx", (wall / 2, hy + wall, wall_hz), (-(hx + wall / 2), 0, wall_z), (0.6, 0.45, 0.3, 1)),
        ("box_wall_py", (hx, wall / 2, wall_hz), (0, hy + wall / 2, wall_z), (0.6, 0.45, 0.3, 1)),
        ("box_wall_ny", (hx, wall / 2, wall_hz), (0, -(hy + wall / 2), wall_z), (0.6, 0.45, 0.3, 1)),
    )
    for geom_name, size, pos, rgba in walls:
        box.add_geom(name=geom_name, type=_BOX, size=_round(size), pos=_round(pos), rgba=list(rgba),
                     friction=list(_PROP_FRICTION))


# ----------------------------------------------------------------------------
# Names, paths and loading
# ----------------------------------------------------------------------------

def available_models(models_root: Path | None = None) -> list[str]:
    """Return sorted names of all static and composite models."""
    root = Path(models_root) if models_root is not None else MODELS_ROOT
    static = {d.name for d in root.iterdir() if d.is_dir() and (d / "scene.xml").is_file()} if root.is_dir() else set()
    return sorted(static | set(COMPOSED_MODELS)) if static else []


def _atomic_write(path: Path, text: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _cache_dir(root: Path) -> Path:
    uid = os.getuid() if hasattr(os, "getuid") else 0
    key = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:10]
    return Path(tempfile.gettempdir()) / f"ufactory_mujoco_composed_{uid}" / key


def cached_xml_path(name: str, xml: str, filename: str, root: Path | None = None) -> Path:
    """Write ``xml`` to the per-installation cache (same name overwrites) and return its path."""
    path = _cache_dir(Path(root) if root is not None else MODELS_ROOT) / name / filename
    _atomic_write(path, xml)
    return path


def resolve_model_path(model_arg: str | Path, *, scene: bool = True) -> Path:
    """Resolve a model name or XML path to a loadable file; composites are exported to a cache."""
    path = Path(model_arg)
    if path.is_file():
        return path
    name = str(model_arg)
    candidates = [MODELS_ROOT / name / "scene.xml", MODELS_ROOT / name / f"{name}.xml"]
    if not scene:
        candidates.reverse()
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if is_composed_model(name):
        return cached_xml_path(name, _xml(name, scene, None), "scene.xml" if scene else f"{name}.xml")
    raise FileNotFoundError(
        f"Model '{model_arg}' not found. Searched:\n  - {path}\n  - {candidates[0]}\n  - {candidates[1]}\n"
        f"Available models: {available_models()}"
    )


def load_model(model: str | Path | mujoco.MjSpec | mujoco.MjModel, *, scene: bool = True) -> mujoco.MjModel:
    """Compile a model given by name, MJCF path, ``MjSpec`` or ``MjModel`` (returned as is)."""
    if isinstance(model, mujoco.MjModel):
        return model
    if isinstance(model, mujoco.MjSpec):
        return model.compile()
    if isinstance(model, str) and is_composed_model(model) and not Path(model).is_file():
        return compose_model(model, scene=scene)
    return mujoco.MjModel.from_xml_path(str(resolve_model_path(model, scene=scene)))


def initial_keyframe(model: mujoco.MjModel) -> int:
    """Keyframe a simulation starts from: ``init`` when the model has one, else keyframe 0, else -1.

    ``home`` (keyframe 0 of every arm model) is all-zero joints. Tool combinations whose tool
    would reach below the floor there add ``init``; -1 means the model's ``qpos0``.
    """
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init")
    if key >= 0:
        return key
    return 0 if model.nkey else -1


def select_arm_scene(robot: str, gripper: str | None, scene: str) -> tuple[Path | None, str | None]:
    """Validate a (robot, gripper, scene) selection; return the static path or composite name."""
    if robot not in ARMS:
        raise ConfigError(f"unknown robot {robot!r}")
    if scene not in robots.SCENES:
        raise ConfigError(f"unknown scene {scene!r}")
    if gripper is not None and gripper not in ARMS[robot].grippers:
        raise ConfigError(f"{robot} does not support gripper {gripper}")
    if scene == "obstacles":
        if robot != "xarm6" or gripper is not None:
            raise ConfigError(f"scene {scene!r} is only available for the bare xarm6")
        return _require_file(MODELS_ROOT / "xarm6" / "scene_obstacles.xml"), None
    if scene in robots.TASK_SCENES:
        if robots.TASK_GRIPPERS.get(robot) != gripper:
            raise ConfigError(f"scene {scene!r} has no task scene for {robot} + {gripper}")
        _require_parts(robot, gripper)
        return None, robots.combo_name(robot, gripper)
    if gripper is None:
        return _require_file(MODELS_ROOT / robot / "scene.xml"), None
    name = robots.combo_name(robot, gripper)
    if name not in RECIPES:
        raise ConfigError(f"{robot} + {gripper} MJCF is not built yet")
    _require_parts(robot, gripper)
    return None, name


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise ModelResourceError(f"scene file does not exist: {path}")
    return path


def _require_parts(robot: str, gripper: str | None) -> None:
    recipe = RECIPES[robots.combo_name(robot, gripper)] if gripper else None
    parts = [robot]
    if recipe:
        adapter = TOOLS[recipe.tool].adapter if recipe.tool else None
        for p in (adapter, recipe.tool, "ft_sensor" if recipe.ft_sensor else None):
            if p:
                parts.append(p)
    for part in parts:
        _require_file(MODELS_ROOT / part / "scene.xml")
        _require_file(MODELS_ROOT / part / f"{part}.xml")


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
) -> Path:
    """Write a standalone ``<name>.xml`` and ``scene.xml`` to ``out_dir``; return the scene path.

    Mesh paths are absolute unless ``copy_meshes`` copies them to ``out_dir/assets``.
    """
    components = dict(arm=arm, base_plate=base_plate, ft_sensor=ft_sensor, gripper=gripper,
                      camera=camera, camera_stand=camera_stand)
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = (Path(models_root) if models_root is not None else MODELS_ROOT).resolve()
    scene_path = out / "scene.xml"
    for scene in (False, True):
        spec = mujoco.MjSpec.from_string(_xml(name, scene, models_root, **components))
        if copy_meshes:
            for mesh in spec.meshes:
                source = Path(mesh.file)
                # <root>/<part>/assets/<part>/<sub>/<file> -> assets/<part>/<sub>/<file>
                relative = Path(*source.relative_to(root).parts[2:])
                target = out / "assets" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copyfile(source, target)
                mesh.file = f"assets/{relative.as_posix()}"
            spec.modelfiledir = f"{out}{os.sep}"  # to_xml compiles; resolve the copied meshes
        stem = spec.modelname.removesuffix(" scene")
        _atomic_write(scene_path if scene else out / f"{stem}.xml", _to_xml(spec))
    return scene_path
