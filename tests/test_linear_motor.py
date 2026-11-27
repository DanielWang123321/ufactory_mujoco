"""Tests for AL0700 linear motor assets, dynamic composition, control, drag chain, and CLI."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from ufactory_mujoco import (
    MODELS_ROOT,
    DragChain,
    RailConfig,
    RailSimulation,
    SimRobot,
    compose_model,
    create_rail_viewer,
    has_linear_motor,
    move_to_rail_position,
    resolve_model_path,
)
from ufactory_mujoco.linear_motor import AL0700_MOUNT_ROTATION, AL0700_ROTATION, to_world
from ufactory_mujoco.robots import RAIL_ARMS, RECIPES, needs_combo_init

ROOT = Path(__file__).resolve().parents[1]
AL0700_DIR = MODELS_ROOT / "al0700"
SOURCE_DIR = ROOT / "dev" / "linear_motor" / "sources"


def _read_obj_vertices_and_faces(path: Path) -> tuple[np.ndarray, int]:
    verts: list[list[float]] = []
    faces = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("f "):
                faces += 1
    return np.asarray(verts, dtype=float), faces


def test_al0700_assets_materials_inertials_and_relocation() -> None:
    """Verify 51 watertight OBJs, 9 collision STLs, sRGB/PBR materials, 20 kg mass, and relocation."""
    entries = {p.name for p in AL0700_DIR.iterdir()}
    assert entries == {"al0700.xml", "scene.xml", "assets"}

    vis_dir = AL0700_DIR / "assets" / "al0700" / "visual"
    col_dir = AL0700_DIR / "assets" / "al0700" / "collision"
    objs = sorted(vis_dir.glob("part_*.obj"))
    stls = sorted(col_dir.glob("*.stl"))
    assert [p.name for p in objs] == [f"part_{i:02d}.obj" for i in range(51)]
    assert [p.name for p in stls] == [f"carriage_{i:02d}.stl" for i in range(3)] + [
        f"fixed_{i:02d}.stl" for i in range(6)
    ]

    all_verts: list[np.ndarray] = []
    total_faces = 0
    for obj in objs:
        verts, nfaces = _read_obj_vertices_and_faces(obj)
        all_verts.append(verts)
        total_faces += nfaces
    assert total_faces == 176659
    bounds = np.vstack(all_verts)
    np.testing.assert_allclose(np.ptp(bounds, axis=0), [1.0926, 0.213, 0.10702], atol=1e-7)

    # Check relative paths and single-sided mesh tags in static XMLs.
    for xml_path in (AL0700_DIR / "al0700.xml", AL0700_DIR / "scene.xml"):
        xml_root = ET.parse(xml_path).getroot()
        for el in xml_root.iter():
            for attr in ("file", "meshdir"):
                if attr in el.attrib:
                    assert not el.attrib[attr].startswith("/")
        for mesh_el in xml_root.findall("./asset/mesh"):
            if mesh_el.get("file", "").endswith(".obj"):
                assert "inertia" not in mesh_el.attrib

    # Check sRGB/PBR materials and 20 kg total mass + positive-definite inertia.
    xml_root = ET.parse(AL0700_DIR / "al0700.xml").getroot()
    materials = {m.get("name"): m.attrib for m in xml_root.findall("./asset/material")}
    assert set(materials) == {"al0700_black", "al0700_dark_gray", "al0700_silver"}
    for mat in materials.values():
        assert float(mat["metallic"]) > 0.0
        assert float(mat["roughness"]) > 0.0
        assert float(mat["rgba"].split()[0]) > 0.1

    m = mujoco.MjModel.from_xml_path(str(AL0700_DIR / "scene.xml"))
    assert has_linear_motor(m)
    assert sum(m.body_mass[m.body(name).id] for name in ("al0700_fixed", "al0700_carriage")) == pytest.approx(20.0)
    for body_name in ("al0700_fixed", "al0700_carriage"):
        bid = m.body(body_name).id
        assert np.all(m.body_inertia[bid] > 0.0)

    # Standalone copy loads cleanly from an unrelated CWD.
    with tempfile.TemporaryDirectory(prefix="al0700_reloc_") as tmp:
        tmp_path = Path(tmp)
        copied = tmp_path / "copied" / "al0700"
        shutil.copytree(AL0700_DIR, copied)
        other_cwd = tmp_path / "other_cwd"
        other_cwd.mkdir()
        code = (
            "import mujoco, sys; m = mujoco.MjModel.from_xml_path(sys.argv[1]); "
            "assert m.nq == 1 and m.nmesh == 60"
        )
        subprocess.run([sys.executable, "-c", code, str(copied / "scene.xml")], cwd=other_cwd, check=True)


def test_al0700_matches_dev_source_glb_when_present() -> None:
    """When maintainer dev/linear_motor/sources exists, verify OBJ geometry matches GLB."""
    glb_path = SOURCE_DIR / "model.glb"
    if not glb_path.is_file():
        pytest.skip("Maintainer dev/linear_motor/sources/model.glb is absent")
    sys.path.insert(0, str(ROOT / "dev" / "linear_motor"))
    try:
        from asset_io import read_parts  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    parts = read_parts()
    assert len(parts) == 51
    for name, part in parts.items():
        obj_path = AL0700_DIR / "assets" / "al0700" / "visual" / f"{name}.obj"
        verts, nfaces = _read_obj_vertices_and_faces(obj_path)
        assert nfaces == len(part["faces"])
        np.testing.assert_allclose(verts, to_world(part["vertices"]), atol=1e-9)
    assert AL0700_ROTATION.shape == (3, 3)


@pytest.mark.parametrize("model_name", ["al0700", *(f"{arm}_with_al0700" for arm in RAIL_ARMS)])
def test_named_rail_models_load_and_step_warning_free(model_name: str) -> None:
    warnings: list[str] = []
    mujoco.set_mju_user_warning(warnings.append)
    try:
        sim = RailSimulation(model_name)
        m, d = sim.model, sim.data
        arm = model_name.split("_with_")[0] if "_with_" in model_name else None
        assert m.nq == (int(arm[-1]) + 1 if arm else 1)
        assert m.nu == m.nq
        assert sum(m.mesh(i).name.startswith("al0700_") for i in range(m.nmesh)) == 60
        for key in range(m.nkey):
            mujoco.mj_resetDataKeyframe(m, d, key)
            for _ in range(50):
                sim.step()
            assert np.isfinite(d.qpos).all()
        assert not warnings
    finally:
        mujoco.set_mju_user_warning(None)


@pytest.mark.parametrize(
    ("arm", "gripper", "ft_sensor", "camera"),
    [
        ("xarm5", "gripper_g2", False, None),
        ("xarm5", "gripper_g2", False, "d435"),
        ("xarm6", "gripper_g2", False, None),
        ("xarm6", "bio_gripper_g2", False, None),
        ("xarm6", "robotiq_2f_85", False, None),
        ("xarm6", "robotiq_2f_140", False, None),
        ("xarm6", "vacuum_gripper", False, None),
        ("xarm6", None, True, None),
        ("xarm6", "gripper_g2", True, None),
        ("xarm6", "vacuum_gripper", True, None),
        ("xarm6", "gripper_g2", True, "d435"),
        ("xarm6", "robotiq_2f_140", False, "d435"),
        ("xarm7", "bio_gripper_g2", False, None),
        ("xarm7", "robotiq_2f_85", False, None),
    ],
)
def test_stacked_accessory_compositions_compile_and_start_contact_free(
    arm: str, gripper: str | None, ft_sensor: bool, camera: str | None
) -> None:
    """Stacked xArm 5/6/7 + AL0700 + end-effector/sensor/camera combinations start contact-free."""
    warnings: list[str] = []
    mujoco.set_mju_user_warning(warnings.append)
    try:
        sim = RailSimulation(arm=arm, gripper=gripper, ft_sensor=ft_sensor, camera=camera)
    finally:
        mujoco.set_mju_user_warning(None)
    assert not warnings
    m, d = sim.model, sim.data
    assert has_linear_motor(m)
    init_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "init")
    assert init_id >= 0
    for rail_m in (0.0, 0.35, 0.7):
        mujoco.mj_resetDataKeyframe(m, d, init_id)
        d.qpos[sim.rail.qpos_address] = rail_m
        mujoco.mj_forward(m, d)
        assert d.ncon == 0, f"Unexpected contact at {rail_m} m for {sim.name}"
    for _ in range(25):
        sim.step()
    assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()


def test_stacked_controls_and_incompatible_compositions() -> None:
    """Independent rail, arm, gripper, and F/T readings work together; bad combos are rejected."""
    sim = RailSimulation(arm="xarm6", gripper="gripper_g2", ft_sensor=True, camera="d435")
    assert sim.gripper_spec is not None and sim.gripper_spec.name == "gripper_g2"
    assert sim.ft_sensor is not None
    sim.set_arm_target_rad("joint1", 0.15)
    sim.set_gripper_ctrl(0.5)
    report = sim.move_to(250.0, speed_mm_s=500.0)
    assert report["settled"]
    assert report["error_mm"] <= 0.1
    reading = sim.get_ft_sensor_reading()
    assert np.isfinite(reading.raw).all() and np.isfinite(reading.ext).all()

    for bad_arm in ("lite6", "uf850"):
        with pytest.raises(ValueError, match="linear_motor"):
            compose_model(arm=bad_arm, linear_motor=True)
    with pytest.raises(ValueError, match="base_plate"):
        compose_model(arm="xarm6", linear_motor=True, base_plate=True)
    with pytest.raises(ValueError):
        RailSimulation("xarm6")
    with pytest.raises(ValueError):
        RailSimulation("al0700", arm="xarm6")


@pytest.mark.parametrize("arm", RAIL_ARMS)
def test_mount_fk_original_inertia_mass_matrix_and_gravity(arm: str) -> None:
    """Mounted xArm 5/6/7 preserve link inertials, arm M(q) sub-block, gravity bias, and TCP FK."""
    sim = RailSimulation(f"{arm}_with_al0700")
    m, d = sim.model, sim.data
    original = mujoco.MjModel.from_xml_path(str(MODELS_ROOT / arm / f"{arm}.xml"))
    reference = mujoco.MjData(original)
    base_id = m.body("link_base").id
    dofs = [int(m.jnt_dofadr[m.joint(f"joint{i + 1}").id]) for i in range(original.nv)]

    for body in range(1, original.nbody):
        b = m.body(original.body(body).name).id
        for prop in ("body_mass", "body_ipos", "body_inertia"):
            np.testing.assert_allclose(getattr(m, prop)[b], getattr(original, prop)[body], atol=1e-12)

    for pose in (np.zeros(original.nq), np.linspace(-0.1, 0.1, original.nq)):
        reference.qpos[:] = pose
        mujoco.mj_forward(original, reference)
        original_mass = np.zeros((original.nv, original.nv))
        mujoco.mj_fullM(original, reference, original_mass)
        for mm in (0.0, 0.1, 280.5, 350.0, 699.9, 700.0):
            d.qpos[sim.rail.qpos_address] = mm / 1000.0
            for i, angle in enumerate(pose):
                d.qpos[m.jnt_qposadr[m.joint(f"joint{i + 1}").id]] = angle
            mujoco.mj_forward(m, d)
            np.testing.assert_allclose(d.xpos[base_id], [mm / 1000.0, 0.0, 0.107], atol=1e-9)
            np.testing.assert_allclose(d.xmat[base_id].reshape(3, 3), AL0700_MOUNT_ROTATION, atol=1e-9)
            expected = reference.site("attachment_site").xpos @ AL0700_MOUNT_ROTATION.T + [mm / 1000.0, 0.0, 0.107]
            np.testing.assert_allclose(d.site("attachment_site").xpos, expected, atol=1e-6)
            np.testing.assert_allclose(
                d.site("attachment_site").xmat.reshape(3, 3),
                AL0700_MOUNT_ROTATION @ reference.site("attachment_site").xmat.reshape(3, 3),
                atol=1e-6,
            )
            mass = np.zeros((m.nv, m.nv))
            mujoco.mj_fullM(m, d, mass)
            np.testing.assert_allclose(mass[np.ix_(dofs, dofs)], original_mass, atol=1e-9)
            np.testing.assert_allclose(d.qfrc_bias[dofs], reference.qfrc_bias, atol=1e-9)


@pytest.mark.parametrize("model_name", ["al0700", *(f"{arm}_with_al0700" for arm in RAIL_ARMS)])
def test_repeated_physical_motion_and_independent_targets(model_name: str) -> None:
    sim = RailSimulation(model_name)
    if sim.arm_actuators:
        sim.set_arm_target_rad("joint1", 0.1)
    targets = sim.data.ctrl[list(sim.arm_actuators.values())].copy()
    for position in (350.0, 700.0, 0.0, 700.0, 0.0, 280.5):
        result = move_to_rail_position(sim, position, 500.0)
        assert result["error_mm"] <= 0.1
        assert abs(float(result["velocity_mm_s"])) <= 0.5
        assert float(result["duration_s"]) <= float(result["trajectory_s"]) + 3.002
        assert float(result["peak_force_n"]) <= 160.0
        np.testing.assert_array_equal(sim.data.ctrl[list(sim.arm_actuators.values())], targets)


def test_mid_motion_reverse_and_stationary_hold() -> None:
    sim = RailSimulation("xarm6_with_al0700")
    sim.rail.set_target_mm(700.0, 1000.0)
    for _ in range(1200):
        sim.step()
    assert sim.rail.get_velocity_mm_s() > 10.0
    result = move_to_rail_position(sim, 100.0, 1000.0)
    assert result["settled"]
    for _ in range(1500):
        sim.step()
    assert abs(sim.rail.get_position_mm() - 100.0) < 0.1


@pytest.mark.parametrize("arm", RAIL_ARMS)
def test_default_speed_joint_step_at_both_limits(arm: str) -> None:
    sim = RailSimulation(f"{arm}_with_al0700")
    for joint_degrees, destination in ((10.0, 700.0), (-10.0, 0.0)):
        sim.set_arm_target_rad("joint1", np.deg2rad(joint_degrees))
        measurement = move_to_rail_position(sim, destination)
        assert float(measurement["min_position_mm"]) >= -0.1
        assert float(measurement["max_position_mm"]) <= 700.1


def test_quintic_command_limits() -> None:
    sim = RailSimulation("al0700")
    sim.rail.set_target_mm(700.0, 100.0)
    time_grid = np.linspace(0.0, sim.rail.duration_s, 5001)
    commands = []
    for t in time_grid:
        sim.data.time = t
        sim.rail.update(1e-9)
        commands.append(sim.data.ctrl[sim.rail.actuator_id] * 1000.0)
    velocity = np.gradient(commands, time_grid)
    acceleration = np.gradient(velocity, time_grid)
    assert max(velocity) <= 100.001
    assert max(abs(acceleration)) <= 200.001
    assert commands[0] == pytest.approx(0.0, abs=1e-9)
    assert commands[-1] == pytest.approx(700.0)


@pytest.mark.parametrize(
    ("position", "speed"),
    [
        (-0.1, 100.0),
        (700.1, 100.0),
        (float("nan"), 100.0),
        (float("inf"), 100.0),
        (True, 100.0),
        (0.0, 0.0),
        (0.0, -1.0),
        (0.0, 1001.0),
        (0.0, float("nan")),
        (0.0, float("inf")),
    ],
)
def test_invalid_commands_and_configs_rejected(position, speed) -> None:
    sim = RailSimulation("xarm6_with_al0700")
    before = sim.data.ctrl.copy()
    with pytest.raises(ValueError):
        sim.rail.set_target_mm(position, speed)
    np.testing.assert_array_equal(sim.data.ctrl, before)
    assert sim.rail.target_mm == 0.0

    for kwargs in (
        {"total_mass_kg": 0.0},
        {"carriage_mass_kg": 20.0},
        {"default_speed_mm_s": 1200.0},
        {"limit_timeconstant_s": 0.001},
    ):
        with pytest.raises(ValueError):
            RailConfig(**kwargs)


def test_collision_hulls_hit_external_probes() -> None:
    for group, position in (("fixed", [0.3435, 0.00176, 0.012]), ("carriage", [-0.0065, 0.00176, 0.102])):
        spec = mujoco.MjSpec.from_file(str(AL0700_DIR / "scene.xml"))
        probe = spec.worldbody.add_body(name="external_probe", pos=position)
        probe.add_freejoint()
        probe.add_geom(name="probe", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.002, 0.0, 0.0], mass=0.01)
        spec.keys.clear()
        m = spec.compile()
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        hits = []
        for contact in d.contact[: d.ncon]:
            names = {m.geom(contact.geom1).name, m.geom(contact.geom2).name}
            if "probe" in names:
                hits.extend(name for name in names if name.startswith(f"al0700_{group}_"))
        assert hits, group


def test_drag_chain_endpoints_faces_and_repeatability() -> None:
    sim = RailSimulation("al0700")
    chain = sim.chain
    original = chain.original_cad.copy()
    fixed = (original[:, 1] < 0.0) & (original[:, 2] > 0.32)
    moving = (original[:, 1] > 0.03) & (original[:, 2] > 0.32)
    bend = original[:, 2] < 0.09
    assert fixed.any() and moving.any() and bend.any()
    start = int(sim.model.mesh_faceadr[chain.mesh_id])
    count = int(sim.model.mesh_facenum[chain.mesh_id])
    faces = sim.model.mesh_face[start : start + count]

    def normals(vertices: np.ndarray) -> np.ndarray:
        triangles = vertices[faces]
        return np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])

    initial_normals = normals(original)
    chain.update(350.0)
    baseline = sim.model.mesh_vert[chain.vertex_slice].copy()
    for mm in range(0, 701):
        chain.update(float(mm))
        actual = chain.cad_vertices()
        np.testing.assert_allclose(actual[fixed, 2], original[fixed, 2], atol=1e-7)
        np.testing.assert_allclose(actual[moving, 2] - original[moving, 2], -mm / 1000.0, atol=1e-7)
        np.testing.assert_allclose(actual[bend, 2] - original[bend, 2], -0.22 - mm / 2000.0, atol=1e-7)
        assert np.all(np.einsum("ij,ij->i", normals(actual), initial_normals) > 0.0)
    for mm in (700.0, 0.0, 700.0, 350.0):
        chain.update(mm)
    np.testing.assert_array_equal(sim.model.mesh_vert[chain.vertex_slice], baseline)
    for geom in chain.geom_ids:
        assert sim.model.geom_contype[geom] == 0 and sim.model.geom_conaffinity[geom] == 0
        assert sim.model.geom_rbound[geom] >= np.linalg.norm(baseline, axis=1).max() - 1e-6


def test_viewer_upload_is_outside_lock_and_close_joins(monkeypatch) -> None:
    from ufactory_mujoco import viewer as viewer_mod

    locked = False
    uploads: list[object] = []

    class Handle:
        def __init__(self) -> None:
            self.opt = type("Options", (), {"geomgroup": np.ones(6)})()

        @contextmanager
        def lock(self):
            nonlocal locked
            assert not locked
            locked = True
            try:
                yield
            finally:
                locked = False

        def update_mesh(self, mesh: int) -> None:
            assert not locked
            uploads.append(mesh)

        def sync(self) -> None:
            assert not locked

        def close(self) -> None:
            uploads.append("closed")

    def fake_init(self, model, data) -> None:
        self._handle = Handle()
        self._thread = type("Thread", (), {"join": lambda _self: uploads.append("joined")})()

    monkeypatch.setattr(viewer_mod.PassiveViewer, "__init__", fake_init)
    monkeypatch.setattr(viewer_mod, "configure_backend", lambda: None)

    sim = RailSimulation("al0700")
    viewer = create_rail_viewer(sim)
    sim.data.qpos[sim.rail.qpos_address] = 0.35
    viewer.update_chain()
    viewer.close()
    assert uploads == [sim.chain.mesh_id, "closed", "joined"]
    assert DragChain(sim.model).mesh_id == sim.chain.mesh_id


def test_sim_robot_rejects_linear_motor_and_examples_gate_cleanly() -> None:
    for model_name in ("al0700", "xarm6_with_al0700"):
        with pytest.raises(ValueError, match="AL0700"):
            SimRobot(model_name, viewer=False)
        with pytest.raises(ValueError, match="AL0700"):
            SimRobot(resolve_model_path(model_name), viewer=False)

    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    for script, extra in (
        ("01_load_and_view.py", ["--model", "al0700", "--headless"]),
        ("01_load_and_view.py", ["--model", "xarm6_with_al0700", "--headless"]),
    ):
        res = subprocess.run(
            [sys.executable, str(ROOT / "examples" / "simulation" / script), *extra],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert res.returncode == 0, res.stderr

    for script in ("02_joint_control.py", "04_gripper_control.py", "11_force_torque_sensor.py"):
        res = subprocess.run(
            [
                sys.executable,
                str(ROOT / "examples" / "simulation" / script),
                "--model",
                "xarm6_with_al0700",
                "--headless",
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert res.returncode == 2, f"{script} expected exit 2, got {res.returncode}: {res.stderr}"


def test_example_12_cli_and_export_model_roundtrip() -> None:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    script = str(ROOT / "examples" / "simulation" / "12_linear_motor.py")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report_path = tmp_path / "nested" / "rail_report.json"
        cases = (
            (["--model", "al0700", "--position-mm", "350", "--headless"], 0),
            (
                [
                    "--model",
                    "xarm6_with_al0700",
                    "--round-trips",
                    "1",
                    "--joint",
                    "joint1=10",
                    "--headless",
                    "--report",
                    str(report_path),
                ],
                0,
            ),
            (
                [
                    "--arm",
                    "xarm6",
                    "--gripper",
                    "gripper_g2",
                    "--ft-sensor",
                    "--camera",
                    "d435",
                    "--position-mm",
                    "250",
                    "--headless",
                ],
                0,
            ),
            (["--model", "xarm6", "--headless"], 2),
            (["--model", "al0700", "--arm", "xarm6", "--headless"], 2),
            (["--gripper", "gripper_g2", "--headless"], 2),
            (["--position-mm", "701", "--headless"], 2),
            (["--round-trips", "-1", "--headless"], 2),
        )
        for args, expected_rc in cases:
            res = subprocess.run(
                [sys.executable, script, *args],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=60,
            )
            assert res.returncode == expected_rc, f"{args}: got {res.returncode}\n{res.stderr}"

        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert data["status"] == "passed"
        assert data["model"] == "xarm6_with_al0700"
        assert len(data["moves"]) == 2
        assert all(move["settled"] and move["error_mm"] <= 0.1 for move in data["moves"])

        # Portable export round-trip with copy-meshes.
        export_dir = tmp_path / "exported_rail"
        res_exp = subprocess.run(
            [
                sys.executable,
                "-m",
                "ufactory_mujoco",
                "export-model",
                "--arm",
                "xarm6",
                "--linear-motor",
                "--gripper",
                "gripper_g2",
                "--out-dir",
                str(export_dir),
                "--copy-meshes",
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert res_exp.returncode == 0, res_exp.stderr
        exported_sim = RailSimulation(export_dir / "scene.xml")
        assert exported_sim.move_to(200.0, speed_mm_s=500.0)["settled"]
        assert needs_combo_init(RECIPES["xarm6_with_al0700"]) is False
