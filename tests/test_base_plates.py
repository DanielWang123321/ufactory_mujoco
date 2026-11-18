"""Tests for the 8 public mounting base-plate models and 5 arm-on-plate assemblies (no gmsh)."""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.optimize import least_squares

from ufactory_mujoco.model_paths import MODELS_ROOT, resolve_model_path

STL_TRIANGLE = np.dtype([("normal", "<f4", (3,)), ("points", "<f4", (3, 3)), ("attribute", "<u2")])

PLATES: dict[str, dict] = {
    "xarm": {"thickness_mm": 8.0, "extent_mm": (185.0, 136.0), "collision_count": 17, "solid_xy_mm": (75.0, 0.0)},
    "uf850": {"thickness_mm": 10.0, "extent_mm": (260.0, 190.0), "collision_count": 1, "solid_xy_mm": (105.0, 0.0)},
    "lite6": {"thickness_mm": 5.0, "extent_mm": (190.0, 130.0), "collision_count": 1, "solid_xy_mm": (70.0, 0.0)},
}

ARM_ASSEMBLY: dict[str, tuple[str, tuple[float, float, float], float]] = {
    "xarm5": ("xarm", (-0.00012433, -0.00038722, -0.00001016), 0.000002356),
    "xarm6": ("xarm", (0.00007243, 0.00002863, -0.00004715), -0.00726035),
    "xarm7": ("xarm", (-0.00057017, 0.00006319, 0.00001328), -0.00725905),
    "uf850": ("uf850", (-0.00001021, 0.00000350, 0.00000420), 0.0),
    "lite6": ("lite6", (0.00298920, 0.00001614, 0.00171732), 0.000000298),
}


def _read_binary_stl(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    assert len(raw) >= 84, f"Binary STL too short: {path}"
    count = int.from_bytes(raw[80:84], "little")
    assert len(raw) == 84 + 50 * count and 0 < count <= 200_000, f"Invalid STL header/count in {path}"
    return np.frombuffer(raw, dtype=STL_TRIANGLE, offset=84, count=count)["points"].astype(np.float64)


def _expected_openings_xy_mm(plate: str) -> list[tuple[float, float]]:
    if plate == "xarm":
        pts = [(0.0, 0.0)]
        pts += [(55.0 * math.cos(math.radians(a)), 55.0 * math.sin(math.radians(a))) for a in range(0, 360, 30)]
        pts += [(x, y) for x in (-83.0, 83.0) for y in (-37.5, 37.5)]
        pts += [(-58.945, -53.601), (58.945, -53.601), (-58.863, 51.253), (58.743, 51.253)]
        return pts
    if plate == "uf850":
        pts = [(0.0, 0.0)]
        pts += [(x, y) for x in (-60.104076, 60.104076) for y in (-60.104076, 60.104076)]
        pts += [(x, y) for x in (-115.0, 115.0) for y in (-70.0, 70.0)]
        pts += [(x, y) for x in (-85.0, 85.0) for y in (-80.0, 80.0)]
        return pts
    return (
        [(0.0, 0.0)]
        + [(x, y) for x in (-57.0, 57.0) for y in (-57.0, 57.0)]
        + [(x, y) for x in (-80.0, 80.0) for y in (-50.0, 50.0)]
    )


def _point_on_top_surface(triangles_mm: np.ndarray, point: tuple[float, float]) -> bool:
    top = triangles_mm[np.all(np.abs(triangles_mm[:, :, 2]) < 0.0001, axis=1), :, :2]
    a, b, c = top[:, 0], top[:, 1], top[:, 2]
    v0, v1, v2 = c - a, b - a, np.asarray(point, dtype=float) - a

    def cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    denom = cross(v0, v1)
    valid = np.abs(denom) > 1e-12
    u = np.divide(cross(v2, v1), denom, out=np.full_like(denom, -1.0), where=valid)
    v = np.divide(cross(v0, v2), denom, out=np.full_like(denom, -1.0), where=valid)
    return bool(np.any(valid & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9)))


@pytest.mark.parametrize("plate", tuple(PLATES))
def test_stl_dimensions_surfaces_and_openings(plate):
    """Verify binary visual/collision STLs, envelope dimensions, z=0 top face, and open holes."""
    cfg = PLATES[plate]
    asset_root = MODELS_ROOT / f"{plate}_base_plate" / "assets" / f"{plate}_base_plate"
    vis_triangles_mm = _read_binary_stl(asset_root / "visual" / "visual.stl") * 1000.0
    col_paths = sorted((asset_root / "collision").glob("collision_*.stl"))
    assert len(col_paths) == cfg["collision_count"]

    extents_mm = np.ptp(vis_triangles_mm.reshape(-1, 3), axis=0)
    expected_extents = (*cfg["extent_mm"], cfg["thickness_mm"])
    np.testing.assert_allclose(extents_mm, expected_extents, atol=0.02, rtol=0)
    assert float(vis_triangles_mm[:, :, 2].max()) == pytest.approx(0.0, abs=1e-5)
    assert float(vis_triangles_mm[:, :, 2].min()) == pytest.approx(-cfg["thickness_mm"], abs=1e-5)

    for xy in _expected_openings_xy_mm(plate):
        assert not _point_on_top_surface(vis_triangles_mm, xy), (plate, xy)
    assert _point_on_top_surface(vis_triangles_mm, cfg["solid_xy_mm"])

    col_pts_mm = np.concatenate([_read_binary_stl(p).reshape(-1, 3) * 1000.0 for p in col_paths], axis=0)
    assert float(col_pts_mm[:, 2].max()) == pytest.approx(0.0, abs=1e-5)
    assert float(col_pts_mm[:, 2].min()) == pytest.approx(-cfg["thickness_mm"], abs=1e-5)
    np.testing.assert_allclose(np.ptp(col_pts_mm, axis=0), expected_extents, atol=0.05, rtol=0)


@pytest.mark.parametrize(
    "name",
    [f"{p}_base_plate" for p in PLATES] + [f"{a}_with_base_plate" for a in ARM_ASSEMBLY],
)
def test_mjcf_authoring_groups_and_floor_alignment(name):
    """Verify meshdir='assets', geom/site groups, and scene floor plane at z = -thickness."""
    xml_path = MODELS_ROOT / name / f"{name}.xml"
    root = ET.parse(xml_path).getroot()
    compiler = root.find("compiler")
    assert compiler is not None and compiler.get("meshdir") == "assets"
    for mesh_el in root.findall("./asset/mesh"):
        assert not mesh_el.get("file", "").startswith("..")

    model = mujoco.MjModel.from_xml_path(str(resolve_model_path(name)))
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    plate = name.removesuffix("_base_plate") if name.endswith("_base_plate") and "_with_" not in name else (
        ARM_ASSEMBLY[name.removesuffix("_with_base_plate")][0]
    )
    thickness_m = PLATES[plate]["thickness_mm"] / 1000.0

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert floor_id >= 0
    assert float(data.geom_xpos[floor_id, 2]) == pytest.approx(-thickness_m, abs=1e-6)

    mount_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "arm_mount")
    assert mount_site >= 0 and int(model.site_group[mount_site]) == 4
    vis_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "base_plate_visual")
    assert vis_geom >= 0 and int(model.geom_group[vis_geom]) == 2
    assert int(model.geom_contype[vis_geom]) == 0 and int(model.geom_conaffinity[vis_geom]) == 0
    col_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "base_plate_collision_00")
    assert col_geom >= 0 and int(model.geom_group[col_geom]) == 3


@pytest.mark.parametrize("plate", tuple(PLATES))
def test_standalone_plate_relocatable_across_cwd(plate):
    """Verify standalone base plate package loads after copying to a separate directory."""
    model_name = f"{plate}_base_plate"
    with tempfile.TemporaryDirectory(prefix="plate_reloc_") as tmp:
        tmp_path = Path(tmp)
        copied = tmp_path / "copied" / model_name
        shutil.copytree(MODELS_ROOT / model_name, copied)
        other_cwd = tmp_path / "other_cwd"
        other_cwd.mkdir()
        code = (
            "import mujoco, sys; m = mujoco.MjModel.from_xml_path(sys.argv[1]); "
            "assert m.nmesh > 0 and m.ngeom >= 2"
        )
        for xml in (MODELS_ROOT / model_name / f"{model_name}.xml", copied / f"{model_name}.xml"):
            subprocess.run([sys.executable, "-c", code, str(xml)], cwd=other_cwd, check=True)


@pytest.mark.parametrize(
    ("plate", "where"),
    [(p, w) for p in PLATES for w in ("solid", "central_hole", "mounting_hole")]
    + [("xarm", "tongue_center"), ("xarm", "tongue_edge"), ("xarm", "tongue_seam")],
)
def test_sphere_lands_on_solid_and_filled_holes(plate, where):
    """Drop-sphere collision probes verify solid surface, filled holes, and xArm tongue/seam."""
    mount = {"xarm": (0.055, 0.0), "uf850": (0.060104076, 0.060104076), "lite6": (0.057, 0.057)}[plate]
    locations = {
        "solid": (PLATES[plate]["solid_xy_mm"][0] / 1000.0, 0.0),
        "central_hole": (0.0, 0.0),
        "mounting_hole": mount,
        "tongue_center": (0.0, -0.067),
        "tongue_edge": (0.060, -0.070),
        "tongue_seam": (0.0, -0.063),
    }
    x, y = locations[where]
    radius = 0.004 if where in ("solid", "central_hole") else 0.001

    spec = mujoco.MjSpec.from_file(str(MODELS_ROOT / f"{plate}_base_plate" / f"{plate}_base_plate.xml"))
    if radius == 0.001:
        spec.option.timestep = 0.0005
    ball = spec.worldbody.add_body(name="ball", pos=[x, y, 0.01 if radius == 0.001 else 0.08])
    ball.add_freejoint()
    ball.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[radius, 0.0, 0.0], mass=0.02)
    model = spec.compile()
    data = mujoco.MjData(model)
    for _ in range(round(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert float(data.qpos[2]) == pytest.approx(radius, abs=0.0001)
    assert float(np.linalg.norm(data.qvel[:3])) < 0.001
    contacts = [
        (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2),
        )
        for c in data.contact[: data.ncon]
    ]
    assert any(
        "ball_geom" in pair and any(n and n.startswith("base_plate_collision") for n in pair)
        for pair in contacts
    )


def _upstream_base_vertices_mm(arm: str) -> np.ndarray:
    obj = "link_base_white.obj" if arm == "lite6" else "link_base.obj"
    path = MODELS_ROOT / arm / "assets" / arm / "visual" / obj
    return np.array(
        [
            [float(v) * 1000.0 for v in line.split()[1:4]]
            for line in path.open(encoding="utf-8")
            if line.startswith("v ")
        ]
    )


def _nominal_robot_holes_mm(arm: str) -> tuple[np.ndarray, np.ndarray, float]:
    if arm.startswith("xarm"):
        centers = np.array(
            [
                [55.0, 0.0],
                [27.5, 47.631397],
                [-27.5, 47.631397],
                [27.5, -47.631397],
                [-27.5, -47.631397],
            ]
        )
        return centers, np.zeros(2), 2.75
    if arm == "uf850":
        return (
            np.array([[x, y] for x in (-60.104076, 60.104076) for y in (-60.104076, 60.104076)]),
            np.zeros(2),
            4.25,
        )
    return np.array([[x, y] for x in (-57.0, 57.0) for y in (-57.0, 57.0)]), np.array([-3.0, 0.0]), 2.6


@pytest.mark.parametrize("arm", tuple(ARM_ASSEMBLY))
def test_fitted_robot_mounting_holes_and_z0_plane(arm):
    """Verify upstream link_base holes align within <0.01 mm and lowest visual plane sits at z=0."""
    vertices = _upstream_base_vertices_mm(arm)
    lowest_z = float(vertices[:, 2].min())
    bottom = vertices[np.abs(vertices[:, 2] - lowest_z) < 0.001, :2]
    centers, shift, radius = _nominal_robot_holes_mm(arm)
    _, position, yaw = ARM_ASSEMBLY[arm]
    rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    assert lowest_z + position[2] * 1000.0 == pytest.approx(0.0, abs=1e-5)

    for nominal in centers:
        guess = nominal + shift
        candidates = bottom[np.abs(np.linalg.norm(bottom - guess, axis=1) - radius) < 1.0]
        assert len(candidates) > 50
        fit = least_squares(
            lambda p, samples=candidates: np.linalg.norm(samples - p[:2], axis=1) - p[2],
            [*guess, radius],
            loss="soft_l1",
            f_scale=0.03,
        ).x
        transformed = rot @ fit[:2] + np.asarray(position[:2]) * 1000.0
        assert transformed == pytest.approx(nominal, abs=0.01)


@pytest.mark.parametrize("arm", tuple(ARM_ASSEMBLY))
def test_arm_on_base_plate_preserves_robot_and_steps_stably(arm):
    """Verify arm+base_plate keeps upstream joints/keyframes, mounting contact distance, and 500-step stability."""
    reference = mujoco.MjModel.from_xml_path(str(resolve_model_path(arm)))
    model = mujoco.MjModel.from_xml_path(str(resolve_model_path(f"{arm}_with_base_plate")))
    assert (model.nq, model.nv, model.njnt, model.nu, model.nkey) == (
        reference.nq,
        reference.nv,
        reference.njnt,
        reference.nu,
        reference.nkey,
    )
    np.testing.assert_allclose(model.qpos0, reference.qpos0)
    np.testing.assert_allclose(model.jnt_axis, reference.jnt_axis)
    np.testing.assert_allclose(model.jnt_range, reference.jnt_range)
    np.testing.assert_allclose(model.key_qpos, reference.key_qpos)
    np.testing.assert_allclose(model.key_ctrl, reference.key_ctrl)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0

    plate_geoms = [
        i
        for i in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("base_plate_collision")
    ]
    base_geoms = [
        i
        for i in range(model.ngeom)
        if int(model.geom_group[i]) == 3
        and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])) or "") == "link_base"
    ]
    assert plate_geoms and base_geoms
    nearest = min(mujoco.mj_geomDistance(model, data, a, b, 0.1, None) for a in plate_geoms for b in base_geoms)
    assert -0.0001 < nearest < (0.0018 if arm == "lite6" else 0.0001)

    for _ in range(500):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert float(np.max(np.abs(data.qvel))) < 0.1
    assert data.ncon == 0
