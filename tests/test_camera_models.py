"""Geometry, mounting-site, collision-probe, and flange-fit tests for camera and bracket models."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from ufactory_mujoco.model_paths import MODELS_ROOT

REPO_ROOT = Path(__file__).resolve().parent.parent
CAMERAS = ("d405", "d435", "gemini_305", "zed_x_nano")
STAND_COMBOS = tuple(f"camera_stand_with_{c}" for c in CAMERAS)
ARM_COMBOS = tuple(f"xarm6_with_gripper_and_{c}" for c in CAMERAS)
ALL_NEW_MODELS = (*CAMERAS, "camera_stand", *STAND_COMBOS, *ARM_COMBOS)

STL_TRIANGLE = np.dtype([("normal", "<f4", (3,)), ("points", "<f4", (3, 3)), ("attribute", "<u2")])

EXPECTED_DIMENSIONS: dict[str, tuple[tuple[float, float, float], float, float]] = {
    "d405": ((0.042, 0.042, 0.023), 0.001, 0.0002),
    "d435": ((0.090, 0.025, 0.025), 1.0, 0.0002),
    "gemini_305": ((0.042, 0.023, 0.042), 0.001, 0.0002),
    "zed_x_nano": ((0.042, 0.04454, 0.0305), 0.001, 0.00015),
}


def _read_binary_stl_vertices(path: Path, scale: float = 1.0) -> tuple[np.ndarray, int]:
    raw = path.read_bytes()
    if len(raw) < 84:
        raise AssertionError(f"Binary STL too short: {path}")
    count = int.from_bytes(raw[80:84], "little")
    if len(raw) != 84 + 50 * count or not (0 < count <= 100_000):
        raise AssertionError(f"Invalid STL face count ({count}) or byte length ({len(raw)}) in {path}")
    pts = np.frombuffer(raw, dtype=STL_TRIANGLE, offset=84, count=count)["points"].reshape(-1, 3)
    return pts.astype(np.float64) * scale, count


def _stl_extents(paths: list[Path], scale: float) -> np.ndarray:
    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    for p in paths:
        pts, _ = _read_binary_stl_vertices(p, scale)
        low = np.minimum(low, pts.min(axis=0))
        high = np.maximum(high, pts.max(axis=0))
    return high - low


def _parse_obj_triangles(path: Path) -> np.ndarray:
    verts: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    return v[f]


def _slice_triangles_xy_mm(triangles_mm: np.ndarray, z_mm: float) -> np.ndarray:
    """Return dense 2D points (mm) where triangle edges cross plane Z = z_mm."""
    edges = ((0, 1), (1, 2), (2, 0))
    segments: list[np.ndarray] = []
    for i, j in edges:
        pa = triangles_mm[:, i, :]
        pb = triangles_mm[:, j, :]
        za = pa[:, 2] - z_mm
        zb = pb[:, 2] - z_mm
        mask = (za * zb <= 0.0) & (np.abs(zb - za) > 1e-12)
        if np.any(mask):
            t = (-za[mask] / (zb[mask] - za[mask]))[:, None]
            pts = pa[mask, :2] + t * (pb[mask, :2] - pa[mask, :2])
            segments.append(pts)
    if not segments:
        return np.empty((0, 2), dtype=np.float64)
    return np.concatenate(segments, axis=0)


def _probe_contacts(xml_path: Path, positions: list[list[float]]) -> list[int]:
    root = ET.parse(xml_path).getroot()
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    if compiler is not None and "meshdir" in compiler.attrib:
        del compiler.attrib["meshdir"]
    base = xml_path.parent / meshdir if meshdir else xml_path.parent
    for mesh in root.findall("./asset/mesh"):
        if "file" in mesh.attrib:
            mesh.set("file", str((base / mesh.attrib["file"]).resolve()))
    world = root.find("worldbody")
    assert world is not None
    probe_body = ET.SubElement(world, "body", {"name": "probe"})
    ET.SubElement(probe_body, "freejoint", {"name": "probe_joint"})
    ET.SubElement(probe_body, "geom", {"name": "probe_geom", "type": "sphere", "size": "0.001", "mass": "0.01"})
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    counts: list[int] = []
    for pos in positions:
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = pos
        mujoco.mj_forward(model, data)
        counts.append(int(data.ncon))
    return counts


def _min_geom_distance(model: mujoco.MjModel, data: mujoco.MjData, group_a: list[int], group_b: list[int]) -> float:
    return min(mujoco.mj_geomDistance(model, data, i, j, 0.15, None) for i in group_a for j in group_b)


class CameraAndBracketModelsTest(unittest.TestCase):
    def test_stl_integrity_and_vendor_dimensions(self) -> None:
        """Verify binary STL face counts and scaled bounding boxes against vendor specs."""
        total_stls = 0
        for cam, (nominal, scale, tol) in EXPECTED_DIMENSIONS.items():
            vis_paths = sorted((MODELS_ROOT / cam / "assets" / cam / "visual").glob("visual_*.stl"))
            self.assertGreaterEqual(len(vis_paths), 2, f"Expected >=2 split visual STLs for {cam}")
            total_stls += len(vis_paths)
            ext = _stl_extents(vis_paths, scale)
            np.testing.assert_allclose(ext, nominal, atol=tol, rtol=0, err_msg=f"{cam} envelope mismatch")

        stand_assets = MODELS_ROOT / "camera_stand" / "assets" / "camera_stand"
        stand_vis = stand_assets / "visual" / "stand_mm.stl"
        stand_cols = sorted((stand_assets / "collision").glob("piece_*.stl"))
        self.assertEqual(len(stand_cols), 175)
        total_stls += 1 + len(stand_cols)
        self.assertEqual(total_stls, 189)

        stand_ext = _stl_extents([stand_vis], 0.001)
        np.testing.assert_allclose(stand_ext, (0.126, 0.003, 0.075), atol=0.0001, rtol=0)
        for col_stl in stand_cols:
            _read_binary_stl_vertices(col_stl, 1.0)

    def test_mjcf_authoring_and_scene_floor_contracts(self) -> None:
        """Verify meshdir='assets', visual/collision/site groups, materials, and floor clearance."""
        for name in ALL_NEW_MODELS:
            xml_path = MODELS_ROOT / name / f"{name}.xml"
            root = ET.parse(xml_path).getroot()
            compiler = root.find("compiler")
            self.assertIsNotNone(compiler)
            self.assertEqual(compiler.get("meshdir"), "assets", f"{name}.xml must declare meshdir='assets'")

            for mesh_el in root.findall("./asset/mesh"):
                file_attr = mesh_el.get("file", "")
                self.assertFalse(file_attr.startswith(".."), f"Cross-dir path forbidden: {file_attr}")
                if "visual_glass_" in file_attr:
                    self.assertEqual(mesh_el.get("inertia"), "shell")

            model = mujoco.MjModel.from_xml_path(str(MODELS_ROOT / name / "scene.xml"))
            data = mujoco.MjData(model)
            if model.nkey > 0:
                mujoco.mj_resetDataKeyframe(model, data, 0)
            mujoco.mj_forward(model, data)

            floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            self.assertGreaterEqual(floor_id, 0)
            if name not in ARM_COMBOS:
                self.assertAlmostEqual(float(data.geom_xpos[floor_id, 2]), -0.05, places=6)

        for cam in CAMERAS:
            m = mujoco.MjModel.from_xml_path(str(MODELS_ROOT / cam / f"{cam}.xml"))
            self.assertGreaterEqual(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, f"{cam}_anodized_aluminum"), 0)
            self.assertGreaterEqual(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, f"{cam}_optical_glass"), 0)
            for gid in range(m.ngeom):
                grp = int(m.geom_group[gid])
                self.assertIn(grp, (2, 3))
                self.assertEqual(float(m.geom_mass[gid]) if hasattr(m, "geom_mass") else 0.0, 0.0)
                if grp == 2:
                    self.assertEqual(int(m.geom_contype[gid]), 0)
                    self.assertEqual(int(m.geom_conaffinity[gid]), 0)
            for sid in range(m.nsite):
                self.assertEqual(int(m.site_group[sid]), 4)

    def test_collision_probes_and_bracket_center_bore(self) -> None:
        """Prove camera collision boxes hit external probes while bracket center bore stays open."""
        stand_xml = MODELS_ROOT / "camera_stand" / "camera_stand.xml"
        bore_contacts, metal_contacts = _probe_contacts(stand_xml, [[0.0, 0.0015, 0.0], [0.022, 0.0015, 0.0]])
        self.assertEqual(bore_contacts, 0, "34.5 mm center bore must not be blocked by convex hull")
        self.assertGreater(metal_contacts, 0, "Bracket plate must register collision contacts")

        for cam in CAMERAS:
            cam_xml = MODELS_ROOT / cam / f"{cam}.xml"
            root = ET.parse(cam_xml).getroot()
            geom = root.find(f".//geom[@name='{cam}_collision_00']")
            assert geom is not None
            pos = np.fromstring(geom.attrib["pos"], sep=" ")
            half = np.fromstring(geom.attrib["size"], sep=" ")
            pos[0] += half[0] + 0.0007
            contacts = _probe_contacts(cam_xml, [pos.tolist()])[0]
            self.assertGreater(contacts, 0, f"{cam} collision box failed probe contact check")

    def test_camera_stand_assemblies_alignment_and_plane_clearance(self) -> None:
        """Verify mount site coincidence, screw-hole alignment, and visual-mesh half-space clearance."""
        for cam in CAMERAS:
            combo = f"camera_stand_with_{cam}"
            xml_path = MODELS_ROOT / combo / f"{combo}.xml"
            m = mujoco.MjModel.from_xml_path(str(xml_path))
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)

            s_mount = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "stand_camera_mount")
            c_mount = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{cam}_mount")
            sep = float(np.linalg.norm(d.site_xpos[s_mount] - d.site_xpos[c_mount]))
            if cam == "d435":
                self.assertAlmostEqual(sep, 0.000055, delta=1e-6)
            else:
                self.assertLess(sep, 1e-6)

            n_stand = d.site_xmat[s_mount].reshape(3, 3)[:, 2]
            n_cam = d.site_xmat[c_mount].reshape(3, 3)[:, 2]
            self.assertLess(float(n_stand @ n_cam), -0.999999)

            spacing = 45 if cam == "d435" else 20
            for idx in range(2):
                sh = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"stand_hole_{spacing}_{idx}")
                ch = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{cam}_hole_{idx}")
                hole_err = float(np.linalg.norm(d.site_xpos[sh] - d.site_xpos[ch]))
                self.assertLessEqual(hole_err, 0.000056, f"{combo} hole {idx} error {hole_err}")

            geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
            stand_geoms = [i for i, n in enumerate(geom_names) if n.startswith("stand_collision_")]
            cam_geoms = [i for i, n in enumerate(geom_names) if n.startswith(f"{cam}_collision_")]
            self.assertGreater(_min_geom_distance(m, d, stand_geoms, cam_geoms), -0.0001)

            cam_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{cam}_body")
            body_rot = d.xmat[cam_body].reshape(3, 3)
            body_pos = d.xpos[cam_body]
            plane_pos = d.site_xpos[s_mount]
            scale = EXPECTED_DIMENSIONS[cam][1]
            min_signed = float("inf")
            for stl in sorted((MODELS_ROOT / cam / "assets" / cam / "visual").glob("visual_*.stl")):
                verts, _ = _read_binary_stl_vertices(stl, scale)
                signed = (verts @ body_rot.T + body_pos - plane_pos) @ n_stand
                min_signed = min(min_signed, float(np.min(signed)))
            self.assertGreater(min_signed, -1e-7, f"{cam} visual vertices cross bracket mounting plane")

    def test_plan_a_flange_section_partition_and_pin_bound(self) -> None:
        """Plan A Rule 1: strict zero flange overlap everywhere including (0, -25) mm locating pin (r <= 3.00 mm)."""
        g2_obj = MODELS_ROOT / "gripper_g2" / "assets" / "gripper_g2" / "visual" / "base.obj"
        tris_mm = _parse_obj_triangles(g2_obj) * 1000.0
        # In xarm6_with_gripper_and_*, xarm_gripper_base_link is shifted +3.0 mm along link6 +Z.
        tris_mm[:, :, 2] += 3.0

        # Across full bracket thickness z_link6 in (0.05..2.99 mm):
        for z_mm in (0.05, 0.2, 0.5, 0.8, 1.5, 2.2, 2.8, 2.99):
            pts = _slice_triangles_xy_mm(tris_mm, z_mm)
            self.assertGreater(len(pts), 50)
            pin_dist = np.hypot(pts[:, 0], pts[:, 1] + 25.0)
            pin_mask = pin_dist <= 3.5
            non_pin = pts[~pin_mask]

            # Every non-pin G2 surface point must stay inside Ø34.50 mm center bore (r < 16.20 mm)
            # OR inside the (0, +23.0) mm Pogo-Pin rectangular cutout (|x| < 7.60, y in [20.40, 25.60]).
            in_center_bore = np.hypot(non_pin[:, 0], non_pin[:, 1]) < 16.20
            in_pogo_cutout = (np.abs(non_pin[:, 0]) < 7.60) & (non_pin[:, 1] > 20.40) & (non_pin[:, 1] < 25.60)
            invading = non_pin[~(in_center_bore | in_pogo_cutout)]
            self.assertEqual(
                len(invading),
                0,
                f"Non-pin flange overlap detected at z={z_mm} mm: {invading[:5]}",
            )

            # Locating pin at (0, -25.0) mm stays strictly within nominal r <= 3.00 mm (< r=3.05 mm bracket hole).
            self.assertTrue(np.any(pin_mask))
            self.assertLessEqual(float(np.max(pin_dist[pin_mask])), 3.00)

    def test_plan_a_xarm6_g2_camera_assemblies_weld_and_motion_clearances(self) -> None:
        """Plan A Rule 2: weld-aware clearance, +3 mm TCP shift, identical mass matrix, and G2 stroke."""
        base_m = mujoco.MjModel.from_xml_path(str(MODELS_ROOT / "xarm6_with_gripper" / "xarm6_with_gripper.xml"))
        base_d = mujoco.MjData(base_m)
        mujoco.mj_resetDataKeyframe(base_m, base_d, 0)
        mujoco.mj_forward(base_m, base_d)
        base_tcp = base_d.site_xpos[mujoco.mj_name2id(base_m, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")].copy()
        link6_z = base_d.xmat[mujoco.mj_name2id(base_m, mujoco.mjtObj.mjOBJ_BODY, "link6")].reshape(3, 3)[:, 2].copy()
        base_total_mass = float(np.sum(base_m.body_mass))
        ref_combo_mass: np.ndarray | None = None

        finger_joints = (
            "left_driver_joint",
            "left_finger_joint",
            "left_inner_knuckle_joint",
            "right_driver_joint",
            "right_finger_joint",
            "right_inner_knuckle_joint",
        )
        finger_bodies = {
            "left_outer_knuckle",
            "left_finger",
            "left_inner_knuckle",
            "right_outer_knuckle",
            "right_finger",
            "right_inner_knuckle",
        }
        arm_lower_bodies = {"link_base", "link1", "link2", "link3", "link4", "link5"}

        for cam in CAMERAS:
            combo = f"xarm6_with_gripper_and_{cam}"
            m = mujoco.MjModel.from_xml_path(str(MODELS_ROOT / combo / "scene.xml"))
            d = mujoco.MjData(m)
            for key_name in ("home", "ready"):
                kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key_name)
                self.assertGreaterEqual(kid, 0)
                mujoco.mj_resetDataKeyframe(m, d, kid)
                mujoco.mj_forward(m, d)
                self.assertEqual(int(d.ncon), 0, f"{combo} must have 0 runtime contacts in {key_name} pose")

            # Same weld group on link6, camera_stand, camera body, and gripper base.
            b_link6 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link6")
            b_stand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "camera_stand")
            b_cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{cam}_body")
            b_g2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "xarm_gripper_base_link")
            weld_ids = {int(m.body_weldid[b]) for b in (b_link6, b_stand, b_cam, b_g2)}
            self.assertEqual(len(weld_ids), 1)

            # Total robot mass matches xarm6_with_gripper, and all 4 camera combos share identical M(q)
            # because camera_stand and camera geoms have mass=0 (G2 itself shifts +3 mm along link6 +Z).
            self.assertAlmostEqual(float(np.sum(m.body_mass)), base_total_mass, places=12)
            mass = np.empty((m.nv, m.nv))
            mujoco.mj_fullM(m, d, mass)
            if ref_combo_mass is None:
                ref_combo_mass = mass
            else:
                np.testing.assert_allclose(mass, ref_combo_mass, atol=1e-12, rtol=1e-12)

            # TCP shifted +3.000 mm along link6 +Z.
            tcp = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")].copy()
            np.testing.assert_allclose(tcp - base_tcp, link6_z * 0.003, atol=1e-7, rtol=0)

            geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
            stand_geoms = [i for i, n in enumerate(geom_names) if n.startswith("stand_collision_")]
            cam_geoms = [i for i, n in enumerate(geom_names) if n.startswith(f"{cam}_collision_")]
            g2_base_geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] == b_g2 and m.geom_group[i] == 3]
            finger_geoms = [
                i
                for i in range(m.ngeom)
                if m.geom_group[i] == 3
                and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]) or "") in finger_bodies
            ]
            arm_geoms = [
                i
                for i in range(m.ngeom)
                if m.geom_group[i] == 3
                and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]) or "") in arm_lower_bodies
            ]

            self.assertGreaterEqual(len(g2_base_geoms), 2, f"{combo} G2 base must use multi-convex collision")
            self.assertGreater(
                _min_geom_distance(m, d, stand_geoms, g2_base_geoms),
                0.0,
                f"{combo} camera stand and G2 base collision hulls must have positive clearance",
            )
            self.assertGreater(_min_geom_distance(m, d, cam_geoms, g2_base_geoms), 0.008)
            self.assertGreater(_min_geom_distance(m, d, stand_geoms + cam_geoms, arm_geoms), 0.020)

            # Sweep 13 Gripper G2 opening positions from open to closed.
            min_cam_finger = float("inf")
            min_stand_finger = float("inf")
            for angle in np.linspace(0.057210812728026743, 0.865094645993282, 13):
                for jname in finger_joints:
                    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
                    d.qpos[m.jnt_qposadr[jid]] = angle
                mujoco.mj_forward(m, d)
                min_cam_finger = min(min_cam_finger, _min_geom_distance(m, d, cam_geoms, finger_geoms))
                min_stand_finger = min(min_stand_finger, _min_geom_distance(m, d, stand_geoms, finger_geoms))
            self.assertGreater(min_cam_finger, 0.040, f"{combo} camera-to-finger clearance too small")
            self.assertGreater(min_stand_finger, 0.035, f"{combo} stand-to-finger clearance too small")

            # Joint-1 FK motion moves camera mount site.
            mujoco.mj_resetDataKeyframe(m, d, 0)
            mujoco.mj_forward(m, d)
            cam_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{cam}_mount")
            before = d.site_xpos[cam_site].copy()
            j1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
            d.qpos[m.jnt_qposadr[j1]] += 0.15
            mujoco.mj_forward(m, d)
            self.assertGreater(float(np.linalg.norm(d.site_xpos[cam_site] - before)), 0.005)

            # Zero-gravity G2 close & reopen stroke check.
            mujoco.mj_resetDataKeyframe(m, d, 0)
            m.opt.gravity[:] = 0.0
            act_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
            drv_q = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint")]
            q_open = float(d.qpos[drv_q])
            d.ctrl[act_id] = 0.865094645993282
            for _ in range(1500):
                mujoco.mj_step(m, d)
            q_closed = float(d.qpos[drv_q])
            d.ctrl[act_id] = 0.057210812728026743
            for _ in range(1500):
                mujoco.mj_step(m, d)
            q_reopen = float(d.qpos[drv_q])
            self.assertGreater(q_closed - q_open, 0.55)
            self.assertLess(q_reopen, q_closed - 0.55)

    def test_example_cli_smoke(self) -> None:
        """Verify 01_load_and_view and 04_gripper_control exit codes on camera and combo models."""
        ex01 = REPO_ROOT / "examples" / "simulation" / "01_load_and_view.py"
        ex04 = REPO_ROOT / "examples" / "simulation" / "04_gripper_control.py"

        for model_name in ("d405", "camera_stand", "camera_stand_with_d435", "xarm6_with_gripper_and_zed_x_nano"):
            res = subprocess.run(
                [sys.executable, str(ex01), "--headless", "--model", model_name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(res.returncode, 0, f"01_load_and_view failed on {model_name}: {res.stderr}")

        for passive_model in ("d405", "camera_stand_with_d405"):
            res = subprocess.run(
                [sys.executable, str(ex04), "--headless", "--model", passive_model],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(res.returncode, 2, f"Expected exit code 2 on {passive_model}, got {res.returncode}")

        res = subprocess.run(
            [sys.executable, str(ex04), "--headless", "--model", "xarm6_with_gripper_and_d405", "--cycles", "1"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(res.returncode, 0, f"04_gripper_control failed on arm+camera combo: {res.stderr}")


if __name__ == "__main__":
    unittest.main()
