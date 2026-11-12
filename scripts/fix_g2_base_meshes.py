#!/usr/bin/env python3
"""Fix Gripper G2 base visual locating-pin bulge and split base collision into multi-convex STLs.

Roadmap v0.1.3 (M2):
1. ``models/gripper_g2/assets/gripper_g2/visual/base.obj``:
   - Clamp the (0, -25) mm ISO Ø6 locating-pin decimation polygon bulge (up to r=3.34 mm)
     radially inward to r <= 2.98 mm (strictly inside the nominal r=3.00 mm pin radius and
     the Ø6.10 mm / r=3.05 mm camera-bracket locating-pin hole).
   - Snap Draco-quantized housing bottom flange vertices in [-0.02, 0.0) mm to z = 0.0 mm
     so the G2 housing sits flush on z_link6 = +3.0 mm without micro-grazing the bracket.
   - Leaves ``models/gripper_g2/source/`` completely untouched and preserves vertex/face counts
     and overall AABB.
2. ``models/gripper_g2/assets/gripper_g2/collision/base_link_col_00..NN.stl``:
   - Replace single-mesh ``base_link.stl`` (whose single MuJoCo convex hull bridged across
     z_g2 = -5..0 mm and swallowed the 3 mm camera mounting bracket) with deterministic
     watertight convex STL pieces:
       * ``base_link_col_00.stl``: z <= 0 center boss (r <= 16.03 mm < 17.25 mm bracket bore)
       * ``base_link_col_01.stl``: z <= 0 Ø6 locating pin (r <= 2.98 mm < 3.05 mm pin hole)
       * ``base_link_col_02..NN.stl``: seeded CoACD convex decomposition of the z > 0 housing
         (with z >= +0.02 mm bottom clearance plane and exact outer Y/Z envelope retention).
"""

from __future__ import annotations

from pathlib import Path
import struct

import coacd
import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
G2_ASSETS = REPO_ROOT / "models" / "gripper_g2" / "assets" / "gripper_g2"
BASE_OBJ = G2_ASSETS / "visual" / "base.obj"
COL_DIR = G2_ASSETS / "collision"

PIN_CENTER_XY = np.array([0.0, -0.025], dtype=np.float64)
PIN_MAX_RADIUS_M = 0.00298
HOUSING_CLEARANCE_Z_M = 0.00002


def _write_binary_stl(path: Path, mesh: trimesh.Trimesh) -> None:
    """Write a deterministic binary STL with float32 face normals and vertices."""
    hull = trimesh.convex.convex_hull(mesh.vertices)
    hull.fix_normals()
    tris = np.asarray(hull.triangles, dtype=np.float32)
    normals = np.asarray(hull.face_normals, dtype=np.float32)
    header = b"ufactory_mujoco g2 convex hull".ljust(80, b"\0")
    buf = bytearray(header)
    buf.extend(struct.pack("<I", len(tris)))
    for normal, tri in zip(normals, tris, strict=True):
        buf.extend(struct.pack("<3f", *normal))
        for pt in tri:
            buf.extend(struct.pack("<3f", *pt))
        buf.extend(struct.pack("<H", 0))
    path.write_bytes(bytes(buf))


def _classify_components(mesh: trimesh.Trimesh) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Classify B-rep face patches of base.obj into (boss_faces, pin_faces, housing_faces)."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)

    # Build connected components of faces sharing vertex indices.
    vert_to_faces: list[list[int]] = [[] for _ in range(len(verts))]
    for f_idx, tri in enumerate(faces):
        for v_idx in tri:
            vert_to_faces[int(v_idx)].append(f_idx)

    visited = np.zeros(len(faces), dtype=bool)
    boss_face_groups: list[np.ndarray] = []
    pin_face_groups: list[np.ndarray] = []
    housing_face_groups: list[np.ndarray] = []

    for start_f in range(len(faces)):
        if visited[start_f]:
            continue
        stack = [start_f]
        visited[start_f] = True
        comp_f: list[int] = []
        while stack:
            cur = stack.pop()
            comp_f.append(cur)
            for v_idx in faces[cur]:
                for nb in vert_to_faces[int(v_idx)]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)

        f_arr = np.asarray(comp_f, dtype=np.int64)
        comp_v = verts[np.unique(faces[f_arr])]
        z_min_mm = float(comp_v[:, 2].min() * 1000.0)
        z_max_mm = float(comp_v[:, 2].max() * 1000.0)
        r0_max_mm = float(np.hypot(comp_v[:, 0], comp_v[:, 1]).max() * 1000.0)
        r_pin_max_mm = float(
            np.hypot(comp_v[:, 0] - PIN_CENTER_XY[0], comp_v[:, 1] - PIN_CENTER_XY[1]).max() * 1000.0
        )

        if z_min_mm < -0.1 and r0_max_mm < 16.5:
            boss_face_groups.append(f_arr)
        elif (z_min_mm < -0.1 and r_pin_max_mm < 4.0) or (
            abs(z_min_mm - 2.0013) < 0.02 and abs(z_max_mm - 2.0013) < 0.02 and r_pin_max_mm < 3.4
        ):
            pin_face_groups.append(f_arr)
        else:
            housing_face_groups.append(f_arr)

    return boss_face_groups, pin_face_groups, housing_face_groups


def fix_visual_base_obj() -> tuple[trimesh.Trimesh, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Clamp locating-pin polygon bulge in base.obj while preserving vertex/face count & AABB."""
    raw_lines = BASE_OBJ.read_text(encoding="utf-8").splitlines()
    mesh = trimesh.load(BASE_OBJ, force="mesh", process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64)

    boss_fg, pin_fg, housing_fg = _classify_components(mesh)
    pin_vert_ids = np.unique(faces[np.concatenate(pin_fg)])
    housing_vert_ids = np.unique(faces[np.concatenate(housing_fg)])

    # 1. Radially clamp locating-pin vertices to r <= 2.98 mm.
    dx = verts[pin_vert_ids, 0] - PIN_CENTER_XY[0]
    dy = verts[pin_vert_ids, 1] - PIN_CENTER_XY[1]
    r = np.hypot(dx, dy)
    over = r > PIN_MAX_RADIUS_M
    scale = np.ones_like(r)
    scale[over] = PIN_MAX_RADIUS_M / r[over]
    verts[pin_vert_ids, 0] = PIN_CENTER_XY[0] + dx * scale
    verts[pin_vert_ids, 1] = PIN_CENTER_XY[1] + dy * scale

    # 2. Snap Draco-quantized housing bottom flange vertices in [-0.02, 0.0) mm to z = 0.0.
    hz = verts[housing_vert_ids, 2]
    snap_mask = (hz >= -0.00002) & (hz < 0.0)
    verts[housing_vert_ids[snap_mask], 2] = 0.0

    # Rewrite only the modified `v ` lines in base.obj so vertex/normal/face indices stay 1:1.
    modified_v = set(int(i) for i in pin_vert_ids[over]) | set(int(i) for i in housing_vert_ids[snap_mask])
    v_cursor = 0
    out_lines: list[str] = []
    for line in raw_lines:
        if line.startswith("v "):
            if v_cursor in modified_v:
                vx, vy, vz = verts[v_cursor]
                out_lines.append(f"v {vx:.8f} {vy:.8f} {vz:.8f}")
            else:
                out_lines.append(line)
            v_cursor += 1
        else:
            out_lines.append(line)

    assert v_cursor == len(verts), f"Vertex count mismatch: {v_cursor} != {len(verts)}"
    BASE_OBJ.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    updated_mesh = trimesh.Mesh if False else trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return updated_mesh, boss_fg, pin_fg, housing_fg


def split_collision_base_stl(
    mesh: trimesh.Trimesh,
    boss_fg: list[np.ndarray],
    pin_fg: list[np.ndarray],
    housing_fg: list[np.ndarray],
) -> list[Path]:
    """Decompose Gripper G2 base into watertight convex STLs with z=0 flange separation."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # Piece 00: z <= 0 center boss convex hull
    boss_v = verts[np.unique(faces[np.concatenate(boss_fg)])]
    boss_hull = trimesh.convex.convex_hull(boss_v)

    # Piece 01: z <= 0 locating pin convex hull (already clamped to r <= 2.98 mm)
    pin_v = verts[np.unique(faces[np.concatenate(pin_fg)])]
    pin_hull = trimesh.convex.convex_hull(pin_v)

    # Pieces 02..NN: CoACD decomposition of z > 0 housing
    housing_f = faces[np.concatenate(housing_fg)]
    used_v, inv = np.unique(housing_f, return_inverse=True)
    hv = verts[used_v].copy()
    orig_min = hv.min(axis=0).copy()
    orig_max = hv.max(axis=0).copy()
    orig_min[2] = HOUSING_CLEARANCE_Z_M
    hv[:, 2] = np.maximum(hv[:, 2], HOUSING_CLEARANCE_Z_M)

    housing_mesh = trimesh.Trimesh(vertices=hv, faces=inv.reshape(-1, 3), process=True)
    coacd.set_log_level("error")
    cmesh = coacd.Mesh(housing_mesh.vertices, housing_mesh.faces)
    parts = coacd.run_coacd(
        cmesh,
        threshold=0.08,
        max_convex_hull=8,
        preprocess_mode="auto",
        preprocess_resolution=60,
        mcts_iterations=80,
        mcts_max_depth=3,
        decimate=True,
        max_ch_vertex=64,
        seed=0,
    )

    housing_hulls: list[trimesh.Trimesh] = []
    part_centers: list[np.ndarray] = []
    part_pts: list[np.ndarray] = []
    for p_v, _ in parts:
        pv = np.clip(np.asarray(p_v, dtype=np.float64), orig_min, orig_max)
        part_pts.append(pv)
        part_centers.append(pv.mean(axis=0))

    # Preserve exact outer surface extremum points of the housing so AABB matches CAD within < 1 um.
    extremum_indices = [
        int(np.argmin(hv[:, 0])),
        int(np.argmax(hv[:, 0])),
        int(np.argmin(hv[:, 1])),
        int(np.argmax(hv[:, 1])),
        int(np.argmax(hv[:, 2])),
    ]
    centers_arr = np.asarray(part_centers)
    for ext_idx in extremum_indices:
        pt = hv[ext_idx]
        nearest = int(np.argmin(np.linalg.norm(centers_arr - pt, axis=1)))
        part_pts[nearest] = np.vstack([part_pts[nearest], pt[None, :]])

    for pv in part_pts:
        housing_hulls.append(trimesh.convex.convex_hull(pv))

    # Sort housing hulls deterministically by centroid (z, y, x).
    housing_hulls.sort(key=lambda h: tuple(np.round(h.vertices.mean(axis=0), 6)))
    all_hulls = [boss_hull, pin_hull, *housing_hulls]

    # Remove legacy single-file base_link.stl and stale base_link_col_*.stl.
    legacy = COL_DIR / "base_link.stl"
    if legacy.exists():
        legacy.unlink()
    for old in COL_DIR.glob("base_link_col_*.stl"):
        old.unlink()

    written: list[Path] = []
    for idx, hull in enumerate(all_hulls):
        out = COL_DIR / f"base_link_col_{idx:02d}.stl"
        _write_binary_stl(out, hull)
        written.append(out)
    return written


def main() -> None:
    updated_mesh, boss_fg, pin_fg, housing_fg = fix_visual_base_obj()
    written = split_collision_base_stl(updated_mesh, boss_fg, pin_fg, housing_fg)
    total_faces = sum(len(trimesh.load(p, force="mesh").faces) for p in written)
    total_bytes = sum(p.stat().st_size for p in written)
    print(
        f"Updated {BASE_OBJ.relative_to(REPO_ROOT)} and wrote {len(written)} convex STLs "
        f"({total_faces} total faces, {total_bytes:,} bytes)."
    )


if __name__ == "__main__":
    main()
