"""Tests for the 13 public ft_sensor models, AI1500Reader, SimRobot integration, and Example 11."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from ufactory_mujoco import (
    AI1500Reader,
    FTSensorReader,
    SafetyGateError,
    SimRobot,
    apply_ft_sensor_wrench,
    is_hardware_ft_sensor_enabled,
    read_hardware_ft_sensor,
    read_hardware_ft_sensor_config,
)
from ufactory_mujoco.compose import is_composed_model
from ufactory_mujoco.ft_sensor import EXT_NOISE_STD, RAW_NOISE_STD
from ufactory_mujoco import MODELS_ROOT, resolve_model_path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_SOURCES = REPO_ROOT / "dev" / "asserts" / "ai1500" / "sources"
ROBOTS = ("xarm5", "xarm6", "xarm7", "uf850")
TOOLS = ("none", "gripper", "vacuum_gripper")
BASE_SUFFIX = {"none": "", "gripper": "_with_gripper", "vacuum_gripper": "_with_vacuum_gripper"}
COMBO_SUFFIX = {
    "none": "_with_ft_sensor",
    "gripper": "_with_ft_sensor_and_gripper",
    "vacuum_gripper": "_with_ft_sensor_and_vacuum_gripper",
}
FLANGE_BODY = {"xarm5": "link5", "xarm6": "link6", "xarm7": "link7", "uf850": "link6"}
ALL_FT_MODELS = ("ft_sensor",) + tuple(
    f"{arm}{COMBO_SUFFIX[tool]}" for arm, tool in itertools.product(ROBOTS, TOOLS)
)


def _name_id(model: mujoco.MjModel, kind: int, name: str) -> int:
    idx = mujoco.mj_name2id(model, kind, name)
    assert idx >= 0, f"Missing {kind}: {name}"
    return idx


def _key_data(model: mujoco.MjModel, key_name: str = "home") -> mujoco.MjData:
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    return data


def _standalone() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(resolve_model_path("ft_sensor")))
    data = _key_data(model)
    return model, data


def test_source_hashes_and_visual_objs_when_dev_present():
    """Verify canonical ft_sensor visual OBJs match the frozen GLB in dev/ when present."""
    if not (DEV_SOURCES / "SHA256SUMS").is_file():
        pytest.skip("dev/asserts/ai1500/sources is not present")

    for line in (DEV_SOURCES / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((DEV_SOURCES / name).read_bytes()).hexdigest() == expected

    raw = (DEV_SOURCES / "model.glb").read_bytes()
    json_len, _ = struct.unpack_from("<I4s", raw, 12)
    gltf = json.loads(raw[20 : 20 + json_len])
    bin_len, _ = struct.unpack_from("<I4s", raw, 20 + json_len)
    binary = raw[28 + json_len : 28 + json_len + bin_len]

    def accessor(index: int) -> np.ndarray:
        item = gltf["accessors"][index]
        view = gltf["bufferViews"][item["bufferView"]]
        dtype = np.dtype({5123: "<u2", 5125: "<u4", 5126: "<f4"}[item["componentType"]])
        width = {"SCALAR": 1, "VEC3": 3}[item["type"]]
        return np.ndarray(
            (item["count"], width),
            dtype=dtype,
            buffer=binary,
            offset=view.get("byteOffset", 0) + item.get("byteOffset", 0),
            strides=(view.get("byteStride", width * dtype.itemsize), dtype.itemsize),
        ).copy()

    cad_flange = np.array([-0.00424, 0.00668, 0.0657])
    part_count = 0
    for node in gltf["nodes"]:
        if "mesh" not in node:
            continue
        part_count += 1
        prim = gltf["meshes"][node["mesh"]]["primitives"][0]
        src_verts = accessor(prim["attributes"]["POSITION"]).astype(float) - cad_flange
        src_norms = accessor(prim["attributes"]["NORMAL"]).astype(float)
        src_faces = accessor(prim["indices"]).reshape(-1, 3)

        obj_path = MODELS_ROOT / "ft_sensor" / "assets" / "ft_sensor" / "visual" / f"{node['name']}.obj"
        assert obj_path.is_file()
        verts, norms, faces = [], [], 0
        with obj_path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("v "):
                    verts.append([float(v) for v in line.split()[1:]])
                elif line.startswith("vn "):
                    norms.append([float(v) for v in line.split()[1:]])
                elif line.startswith("f "):
                    faces += 1
        np.testing.assert_allclose(verts, src_verts, atol=1e-9, rtol=0)
        np.testing.assert_allclose(norms[: len(src_norms)], src_norms, atol=1e-9, rtol=0)
        np.testing.assert_allclose(norms[len(src_norms) :], -src_norms, atol=1e-9, rtol=0)
        assert faces == 2 * len(src_faces)
    assert part_count == 15


@pytest.mark.parametrize("name", ALL_FT_MODELS)
def test_mjcf_authoring_and_scene_step_contracts(name):
    """Verify XML conventions, 1 ms timestep, zero warnings, and clean simulation."""
    xml_path = resolve_model_path(name, scene=False)
    root = ET.parse(xml_path).getroot()
    compiler = root.find("compiler")
    assert compiler is not None
    if not is_composed_model(name):
        assert compiler.get("meshdir") == "assets"
    for mesh_el in root.findall("./asset/mesh"):
        file_attr = mesh_el.get("file", "")
        assert not file_attr.startswith("..")
        if file_attr.endswith(".obj") and "ft_sensor/visual/" in file_attr:
            assert mesh_el.get("inertia") == "shell"

    model = mujoco.MjModel.from_xml_path(str(resolve_model_path(name)))
    assert model.opt.timestep == pytest.approx(0.001)
    sensor_base = _name_id(model, mujoco.mjtObj.mjOBJ_BODY, "ft_sensor_base")
    sensor_out = _name_id(model, mujoco.mjtObj.mjOBJ_BODY, "ft_sensor_output")
    m_base = float(model.body_mass[sensor_base])
    m_out = float(model.body_mass[sensor_out])
    assert m_base == pytest.approx(0.45)
    assert m_out == pytest.approx(0.15)
    assert m_base + m_out == pytest.approx(0.6)
    np.testing.assert_allclose(model.body_ipos[sensor_out], [0.0, 0.0, -0.02273], atol=1e-12)
    com_combined_z = (
        m_base * model.body_ipos[sensor_base, 2] + m_out * (0.0586 + model.body_ipos[sensor_out, 2])
    ) / 0.6
    assert com_combined_z == pytest.approx(0.022, abs=1e-12)
    site_id = _name_id(model, mujoco.mjtObj.mjOBJ_SITE, "ft_sensor_site")
    np.testing.assert_allclose(model.site_pos[site_id], [0.0, 0.0, -0.0266], atol=1e-12)
    assert int(model.site_group[site_id]) == 4

    data = _key_data(model, "home")
    assert data.ncon == 0
    reader = FTSensorReader(model, data)
    reader.update()
    for _ in range(50):
        mujoco.mj_step(model, data)
        reading = reader.update()
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(reading.raw).all()
    assert not any(w.number for w in data.warning)


@pytest.mark.parametrize("robot,tool", tuple(itertools.product(ROBOTS, TOOLS)))
def test_assembly_preserves_arm_tool_and_offsets_tcp(robot, tool):
    """Mounted sensor preserves upstream kinematics/actuators and shifts TCP +58.6 mm along flange +Z."""
    base_name = robot + BASE_SUFFIX[tool]
    combo_name = robot + COMBO_SUFFIX[tool]
    source = mujoco.MjModel.from_xml_path(str(resolve_model_path(base_name, scene=False)))
    target = mujoco.MjModel.from_xml_path(str(resolve_model_path(combo_name, scene=False)))

    assert (target.nq, target.nv, target.nu, target.njnt, target.neq) == (
        source.nq,
        source.nv,
        source.nu,
        source.njnt,
        source.neq,
    )
    for attr in (
        "jnt_type",
        "jnt_range",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "actuator_gainprm",
        "actuator_biasprm",
        "eq_data",
    ):
        np.testing.assert_allclose(getattr(target, attr), getattr(source, attr), atol=1e-12, rtol=0)
    for i in range(source.nbody):
        body_name = mujoco.mj_id2name(source, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name is None:
            continue
        new_id = _name_id(target, mujoco.mjtObj.mjOBJ_BODY, body_name)
        assert target.body_mass[new_id] == pytest.approx(source.body_mass[i], abs=1e-12)

    assert target.nsensor == source.nsensor + 2
    if tool == "none":
        # Bare arm+sensor assemblies use the vacuum-gripper elevated ready/home keyframes.
        vg_name = f"{robot}_with_vacuum_gripper"
        pose_ref = mujoco.MjModel.from_xml_path(str(resolve_model_path(vg_name, scene=False)))
    else:
        pose_ref = source

    for key in ("home", "ready"):
        ref_data = _key_data(pose_ref, key)
        new_data = _key_data(target, key)
        if tool == "none":
            flange = _name_id(pose_ref, mujoco.mjtObj.mjOBJ_BODY, FLANGE_BODY[robot])
            expected_pos = ref_data.xpos[flange] + ref_data.xmat[flange].reshape(3, 3) @ [0.0, 0.0, 0.0586]
            expected_mat = ref_data.xmat[flange].reshape(3, 3)
        else:
            old_tcp = _name_id(source, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
            flange = _name_id(source, mujoco.mjtObj.mjOBJ_BODY, FLANGE_BODY[robot])
            expected_pos = ref_data.site_xpos[old_tcp] + ref_data.xmat[flange].reshape(3, 3) @ [0.0, 0.0, 0.0586]
            expected_mat = ref_data.site_xmat[old_tcp].reshape(3, 3)
        new_tcp = _name_id(target, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        np.testing.assert_allclose(new_data.site_xpos[new_tcp], expected_pos, atol=1e-6, rtol=0)
        np.testing.assert_allclose(new_data.site_xmat[new_tcp].reshape(3, 3), expected_mat, atol=1e-6, rtol=0)


@pytest.mark.parametrize("robot", ROBOTS)
def test_mounted_g2_actuator_remains_movable(robot):
    name = f"{robot}_with_ft_sensor_and_gripper"
    model = mujoco.MjModel.from_xml_path(str(resolve_model_path(name)))
    data = _key_data(model, "home")
    actuator = _name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
    joint = _name_id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint")
    adr = int(model.jnt_qposadr[joint])
    initial = float(data.qpos[adr])
    data.ctrl[actuator] = float(model.actuator_ctrlrange[actuator, 1])
    for _ in range(400):
        mujoco.mj_step(model, data)
    assert abs(float(data.qpos[adr]) - initial) > 0.02
    assert np.isfinite(data.qpos).all()
    assert not any(w.number for w in data.warning)


@pytest.mark.parametrize("axis,sign", tuple(itertools.product(range(6), (-1.0, 1.0))))
def test_each_applied_axis_and_direction(axis, sign):
    model, data = _standalone()
    wrench = np.zeros(6)
    wrench[axis] = sign * (4.0 if axis < 3 else 2.0)
    apply_ft_sensor_wrench(model, data, wrench[:3], wrench[3:])
    reading = AI1500Reader(model, data).update()
    expected_grav = np.array([0.0, 0.0, -0.15 * 9.81, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(reading.ext, wrench, atol=1e-8, rtol=0)
    np.testing.assert_allclose(reading.raw, wrench + expected_grav, atol=1e-8, rtol=0)


def test_output_load_gravity_offset_and_rotated_frame():
    spec = mujoco.MjSpec.from_file(str(MODELS_ROOT / "ft_sensor" / "ft_sensor.xml"))
    payload_body = spec.body("ft_sensor_output").add_body(name="payload", pos=[0.1, 0.0, 0.0])
    payload_body.add_geom(
        name="payload",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.01, 0.0, 0.0],
        mass=1.0,
        contype=0,
        conaffinity=0,
    )
    model = spec.compile()
    data = _key_data(model)
    reading = AI1500Reader(model, data).update()
    np.testing.assert_allclose(reading.raw, [0.0, 0.0, -1.15 * 9.81, 0.0, 0.981, 0.0], atol=1e-8, rtol=0)
    np.testing.assert_allclose(reading.ext, np.zeros(6), atol=1e-8, rtol=0)

    # Rotate 90 deg about +X so sensor +Z points along -Y world and +Y points along +Z world (gravity along -Y_local)
    spec_rot = mujoco.MjSpec.from_file(str(MODELS_ROOT / "ft_sensor" / "ft_sensor.xml"))
    spec_rot.body("ft_sensor_base").quat = [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]
    rot_model = spec_rot.compile()
    rot_data = _key_data(rot_model)
    unloaded = AI1500Reader(rot_model, rot_data).update()
    np.testing.assert_allclose(unloaded.ext, np.zeros(6), atol=1e-8, rtol=0)
    np.testing.assert_allclose(unloaded.raw[:3], [0.0, -0.15 * 9.81, 0.0], atol=1e-8, rtol=0)

    local = np.array([2.0, 3.0, 4.0, 0.5, -0.7, 0.9])
    apply_ft_sensor_wrench(rot_model, rot_data, local[:3], local[3:])
    loaded = AI1500Reader(rot_model, rot_data).update()
    np.testing.assert_allclose(loaded.ext, local, atol=1e-8, rtol=0)

    # Force applied at +58.6 mm output face (+26.6 mm ahead of sensing site) induces r x F torque
    clear_model, clear_data = _standalone()
    apply_ft_sensor_wrench(clear_model, clear_data, [10.0, 0.0, 0.0], [0.0, 0.0, 0.0], at_output_face=True)
    face_reading = AI1500Reader(clear_model, clear_data).update()
    np.testing.assert_allclose(face_reading.ext, [10.0, 0.0, 0.0, 0.0, 0.266, 0.0], atol=1e-8, rtol=0)


def test_200hz_sample_hold_and_reset():
    model, data = _standalone()
    reader = AI1500Reader(model, data)
    assert reader.update().sample_time_s == 0.0
    apply_ft_sensor_wrench(model, data, [2.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    same_time = reader.update()
    assert same_time.raw[0] == pytest.approx(2.0)
    assert same_time.ext[0] == pytest.approx(2.0)
    assert same_time.sampled[0] == 0.0
    for _ in range(4):
        mujoco.mj_step(model, data)
        sample = reader.update()
        assert sample.sample_time_s == 0.0
        assert sample.sampled[0] == 0.0
    mujoco.mj_step(model, data)
    sample = reader.update()
    assert sample.sample_time_s == pytest.approx(0.005)
    assert sample.sampled[0] == pytest.approx(2.0)

    mujoco.mj_step(model, data)
    mujoco.mj_step(model, data)
    with pytest.raises(ValueError, match="each 1 ms step"):
        reader.update()
    reader.reset()
    assert reader.update().sample_time_s == pytest.approx(0.007)

    mujoco.mj_resetData(model, data)
    with pytest.raises(ValueError, match="reset"):
        reader.update()
    reader.reset()
    assert reader.update().sample_time_s == 0.0


@pytest.mark.parametrize("axis,limit,overload", [(0, 400.0, 600.0), (3, 20.0, 30.0)])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_range_overload_and_quantization(axis, limit, overload, sign):
    model, data = _standalone()
    reader = AI1500Reader(model, data)
    for value, expected_range, expected_overload in (
        (limit, False, False),
        (limit + 0.001, True, False),
        (overload - 0.001, True, False),
        (overload, True, True),
    ):
        wrench = np.zeros(6)
        wrench[axis] = sign * value
        apply_ft_sensor_wrench(model, data, wrench[:3], wrench[3:])
        reader.reset()
        reading = reader.update()
        assert bool(reading.overrange[axis]) is expected_range
        assert bool(reading.overload[axis]) is expected_overload
        assert reading.raw[axis] == pytest.approx(sign * value)
        assert reading.sampled[axis] == float(np.float32(reading.raw[axis]))


def test_invalid_timestep_rejected_by_reader_and_sim_robot():
    model, data = _standalone()
    model.opt.timestep = 0.002
    with pytest.raises(ValueError, match="0.001"):
        AI1500Reader(model, data)

    with pytest.raises(ValueError, match="0.001"):
        SimRobot(resolve_model_path("ft_sensor"), timestep=0.002)


def test_downstream_contact_reaches_sensor():
    spec = mujoco.MjSpec.from_file(str(MODELS_ROOT / "ft_sensor" / "ft_sensor.xml"))
    spec.body("ft_sensor_output").add_geom(
        name="plate",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, 0.02],
        size=[0.025, 0.025, 0.005],
        mass=0.0,
        contype=1,
        conaffinity=1,
    )
    ball = spec.worldbody.add_body(name="ball", pos=[0.0, 0.0, 0.21])
    ball.add_freejoint()
    ball.add_geom(
        name="ball_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.01, 0.0, 0.0],
        mass=0.1,
        friction=[0.8, 0.01, 0.001],
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    reader = AI1500Reader(model, data)
    reader.update()
    peak = 0.0
    for _ in range(400):
        mujoco.mj_step(model, data)
        peak = max(peak, abs(float(reader.update().raw[2])))
    assert peak > 1.0
    assert data.ncon > 0
    rest_reading = reader.update()
    assert rest_reading.raw[2] == pytest.approx(-(0.15 + 0.1) * 9.81, abs=0.02)
    assert rest_reading.ext[2] == pytest.approx(-0.981, abs=0.02)
    assert not any(w.number for w in data.warning)


def test_sim_robot_auto_attaches_reader_and_g2_works():
    with SimRobot(resolve_model_path("xarm6")) as bare:
        assert not bare.has_ft_sensor
        assert bare.ft_sensor is None
        assert bare.ft_sensor_reading is None
        with pytest.raises(NotImplementedError, match="no force/torque sensor"):
            bare.get_ft_sensor_reading()

    with SimRobot(resolve_model_path("xarm6_with_ft_sensor_and_gripper")) as robot:
        assert robot.has_ft_sensor
        assert isinstance(robot.ft_sensor, AI1500Reader)
        assert robot.ft_sensor_reading is not None
        assert robot.has_gripper
        robot.close_gripper(wait=True)
        closed_gap_mm = robot.get_gripper_gap(measured=True) * 1000.0
        robot.open_gripper(wait=True)
        open_gap_mm = robot.get_gripper_gap(measured=True) * 1000.0
        assert closed_gap_mm < 5.0
        assert abs(open_gap_mm - 84.0) < 2.0

        reading = robot.get_ft_sensor_reading()
        assert np.isfinite(reading.sampled).all()
        assert reading.sample_time_s > 0.0
        assert reading.source == "sim"
        # At ready (tool pointing vertically down), static raw equals G2 + floating head weight (+8.84 N along +Z).
        np.testing.assert_allclose(reading.raw[:3], [0.0, 0.0, 8.8429], atol=0.05, rtol=0)
        np.testing.assert_allclose(reading.ext, np.zeros(6), atol=0.02, rtol=0)
        apply_ft_sensor_wrench(robot.model, robot.data, [3.0, -2.0, 0.0], [0.0, 0.0, 0.0])
        tared = robot.zero_ft_sensor()
        np.testing.assert_allclose(tared.ext, np.zeros(6), atol=1e-6, rtol=0)
        robot.data.xfrc_applied[:] = 0.0
        reset_reading = robot.reset_ft_sensor()
        assert reset_reading.sample_time_s == pytest.approx(robot.data.time)


def test_example_11_headless_and_non_sensor_rejection():
    script = REPO_ROOT / "examples" / "simulation" / "11_force_torque_sensor.py"
    for model_name in ("xarm6_with_ft_sensor_and_gripper", "ft_sensor"):
        res = subprocess.run(
            [sys.executable, str(script), "--model", model_name, "--duration", "0.2", "--headless"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert res.returncode == 0, res.stderr + res.stdout
        assert "PASS" in res.stdout

    rejected = subprocess.run(
        [sys.executable, str(script), "--model", "xarm6", "--headless"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert rejected.returncode == 2
    assert "no force/torque sensor" in rejected.stderr


def test_clear_ft_sensor_wrench_zeros_applied_load():
    from ufactory_mujoco import apply_ft_sensor_wrench, clear_ft_sensor_wrench

    with SimRobot(MODELS_ROOT / "ft_sensor" / "scene.xml", viewer=False) as robot:
        base = robot.get_ft_sensor_reading().raw.copy()
        apply_ft_sensor_wrench(robot.model, robot.data, [2.0, -3.0, -5.0], [0.1, -0.1, 0.05])
        loaded = robot.get_ft_sensor_reading().raw.copy()
        assert np.linalg.norm(loaded - base) > 5.0
        clear_ft_sensor_wrench(robot.model, robot.data)
        cleared = robot.get_ft_sensor_reading().raw.copy()
        np.testing.assert_allclose(cleared, base, atol=1e-9)


def test_two_body_split_preserves_mass_matrix_and_gravity_bias():
    xml_path = resolve_model_path("xarm6_with_ft_sensor", scene=False)
    m_split = mujoco.MjModel.from_xml_path(str(xml_path))
    d_split = mujoco.MjData(m_split)

    # Merge the two sensor inertials back into ft_sensor_base to verify dynamics equivalence
    spec = mujoco.MjSpec.from_file(str(xml_path))
    base_b = spec.body("ft_sensor_base")
    out_b = spec.body("ft_sensor_output")
    radius, height = 0.0388, 0.0636
    base_b.mass = 0.6
    base_b.ipos = [0.0, 0.0, 0.022]
    base_b.inertia = [
        0.6 * (3 * radius**2 + height**2) / 12,
        0.6 * (3 * radius**2 + height**2) / 12,
        0.6 * radius**2 / 2,
    ]
    out_b.mass = 0.0
    out_b.inertia = [0.0, 0.0, 0.0]
    m_single = spec.compile()
    d_single = mujoco.MjData(m_single)

    rng = np.random.default_rng(7)
    for _ in range(5):
        q = rng.uniform(-1.0, 1.0, m_split.nq)
        qd = rng.uniform(-1.0, 1.0, m_split.nv)
        for m, d in ((m_split, d_split), (m_single, d_single)):
            d.qpos[:] = q
            d.qvel[:] = qd
            mujoco.mj_forward(m, d)
        m_a = np.zeros((m_split.nv, m_split.nv))
        m_b = np.zeros((m_single.nv, m_single.nv))
        mujoco.mj_fullM(m_split, d_split, m_a)
        mujoco.mj_fullM(m_single, d_single, m_b)
        np.testing.assert_allclose(m_a, m_b, atol=1e-14, rtol=0)
        np.testing.assert_allclose(d_split.qfrc_bias, d_single.qfrc_bias, atol=1e-14, rtol=0)


def test_calibrated_noise_and_hardware_ft_helpers():
    model, data = _standalone()
    reader = AI1500Reader(model, data, noise_std="calibrated", seed=0)
    samples_raw = []
    samples_ext = []
    for _ in range(500):
        mujoco.mj_step(model, data)
        rd = reader.update()
        samples_raw.append(rd.raw)
        samples_ext.append(rd.ext)
    np.testing.assert_allclose(np.std(samples_raw, axis=0), RAW_NOISE_STD, rtol=0.18, atol=1e-4)
    np.testing.assert_allclose(np.std(samples_ext, axis=0), EXT_NOISE_STD, rtol=0.18, atol=1e-4)

    class _FakeHW:
        def __init__(self) -> None:
            self.ft_sensor_is_enable = True
            self.err_calls = 0
            self.error_code = 0

        def get_err_warn_code(self):
            return 0, (self.error_code, 0)

        def get_ft_sensor_config(self):
            return 0, [0, 1, 1, 1, 200, 0.17, 0.0, [0.0, 0.0, 0.0042], [0.0] * 6]

        def get_ft_sensor_data(self, is_raw: bool = False):
            return 0, ([1.0, 2.0, 3.0, 0.1, 0.2, 0.3] if is_raw else [0.1, 0.0, -0.2, 0.0, 0.01, 0.0])

        def get_ft_sensor_error(self):
            self.err_calls += 1
            return 0, 0

    hw = _FakeHW()
    assert is_hardware_ft_sensor_enabled(hw) is True
    cfg = read_hardware_ft_sensor_config(hw)
    assert cfg["ft_freq_hz"] == 200 and cfg["ft_mass_kg"] == pytest.approx(0.17)
    rd = read_hardware_ft_sensor(hw)
    assert rd.source == "hardware"
    np.testing.assert_allclose(rd.raw, [1.0, 2.0, 3.0, 0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(rd.ext, [0.1, 0.0, -0.2, 0.0, 0.01, 0.0], atol=1e-6)
    assert hw.err_calls == 0

    hw.error_code = 11
    with pytest.raises(SafetyGateError, match="controller error"):
        read_hardware_ft_sensor(hw)


@pytest.mark.parametrize("name", ALL_FT_MODELS)
def test_adapter_ring_stainless_material_and_scene_specular(name):
    """Flange adapter ring (part_02) renders as bright cool stainless steel, distinct from matte part_01."""
    model = mujoco.MjModel.from_xml_path(str(resolve_model_path(name)))
    ring_mat = _name_id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "ft_sensor_part_02_mat")
    output_mat = _name_id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "ft_sensor_part_01_mat")

    ring_rgba = model.mat_rgba[ring_mat]
    assert ring_rgba[0] < ring_rgba[1] < ring_rgba[2] < 0.85
    assert float(model.mat_specular[ring_mat]) >= 0.95
    assert float(model.mat_metallic[ring_mat]) >= 0.9
    assert 0.0 <= float(model.mat_roughness[ring_mat]) <= 0.2
    assert float(model.mat_specular[output_mat]) <= 0.4
    assert np.all(model.vis.headlight.specular > 0.0)
