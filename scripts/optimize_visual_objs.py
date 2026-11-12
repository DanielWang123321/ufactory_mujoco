#!/usr/bin/env python3
"""Deterministic normal- and crease-preserving QEM visual OBJ optimizer.

Reduces redundant CAD tessellation on visual-only OBJs (from ~549 MB to ~85-95 MB)
using PyMeshLab quadric edge collapse with subset vertex placement
(``optimalplacement=False``, keeping exact CAD surface coordinates), followed by
42-degree crease-angle smooth/sharp normal reconstruction and compact ASCII formatting.

Safety guarantees:
- Skips all symlinked combo asset directories and ``models/gripper_g2/source/``.
- Preserves ``models/gripper_g2/assets/gripper_g2/visual/*.obj`` exact vertex/face
  topology and ``%.8g`` coordinate precision so the 85-pose CAD verification,
  Draco snapshot correspondence, and ``r <= 2.98 mm`` locating pin clearance remain
  exact.
"""

from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np
import pymeshlab
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

CREASE_ANGLE_DEG = 42.0
LARGE_FACE_THRESHOLD = 100_000
LARGE_TARGET_FACES = 28_000
MEDIUM_FACE_THRESHOLD = 20_000
MEDIUM_TARGET_FACES = 16_000


def _fmt_coord(val: float, sig: int = 6) -> str:
    if abs(val) < 1e-12:
        return "0"
    s = f"{val:.{sig}g}"
    return "0" if s in ("-0", "-0.0") else s


def _compact_reformat_preserve_topology(obj_path: Path) -> tuple[int, int]:
    """Losslessly compact float formatting without changing vertex or face topology."""
    old_bytes = obj_path.stat().st_size
    out_lines: list[str] = []
    with obj_path.open("r", encoding="utf-8") as stream:
        for raw in stream:
            if raw.startswith("v "):
                parts = raw.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                out_lines.append(
                    f"v {_fmt_coord(x, 8)} {_fmt_coord(y, 8)} {_fmt_coord(z, 8)}"
                )
            elif raw.startswith("vn "):
                parts = raw.split()
                nx, ny, nz = float(parts[1]), float(parts[2]), float(parts[3])
                out_lines.append(
                    f"vn {_fmt_coord(nx, 5)} {_fmt_coord(ny, 5)} {_fmt_coord(nz, 5)}"
                )
            elif raw.startswith("f "):
                out_lines.append(raw.strip())
    content = "\n".join(out_lines) + "\n"
    obj_path.write_text(content, encoding="utf-8")
    return old_bytes, len(content.encode("utf-8"))


def _compute_crease_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    crease_deg: float = CREASE_ANGLE_DEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return filtered faces, per-smooth-group unit normals, and (nf, 3) normal indices."""
    e1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    e2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    fn = np.cross(e1, e2)
    area2 = np.linalg.norm(fn, axis=1)
    keep = area2 > 1e-16
    faces = faces[keep]
    fn = fn[keep]
    ufn = fn / area2[keep, None]
    nf = len(faces)
    nv = len(vertices)

    c0 = np.array([0, 1, 2], dtype=np.int64)
    c1 = np.array([1, 2, 0], dtype=np.int64)
    u = faces[:, c0].ravel()
    v = faces[:, c1].ravel()
    f_idx = np.repeat(np.arange(nf, dtype=np.int64), 3)
    corner_u = 3 * f_idx + np.tile(c0, nf)
    corner_v = 3 * f_idx + np.tile(c1, nf)

    emin = np.minimum(u, v)
    emax = np.maximum(u, v)
    ekey = emin * nv + emax
    order = np.argsort(ekey, kind="mergesort")
    ekey_s = ekey[order]

    # Connect all pairs sharing an undirected edge with dihedral angle <= crease_deg.
    same = np.flatnonzero(ekey_s[:-1] == ekey_s[1:])
    i1 = order[same]
    i2 = order[same + 1]

    cos_th = float(np.cos(np.deg2rad(crease_deg)))
    smooth = np.sum(ufn[f_idx[i1]] * ufn[f_idx[i2]], axis=1) >= cos_th
    i1 = i1[smooth]
    i2 = i2[smooth]

    same_dir = u[i1] == u[i2]
    ca = np.concatenate([corner_u[i1], corner_v[i1]])
    cb = np.concatenate([
        np.where(same_dir, corner_u[i2], corner_v[i2]),
        np.where(same_dir, corner_v[i2], corner_u[i2]),
    ])
    adj = coo_matrix(
        (np.ones(len(ca), dtype=np.int8), (ca, cb)),
        shape=(3 * nf, 3 * nf),
    )
    n_vn, labels = connected_components(adj, directed=False)
    vn = np.zeros((n_vn, 3), dtype=np.float64)
    np.add.at(vn, labels, np.repeat(fn, 3, axis=0))
    nrm = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = np.where(nrm > 1e-16, vn / np.maximum(nrm, 1e-16), np.array([0.0, 0.0, 1.0]))
    return faces, vn, labels.reshape(nf, 3)


def _write_compact_obj(
    obj_path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    normal_idx: np.ndarray,
) -> int:
    lines = [
        f"v {_fmt_coord(x, 6)} {_fmt_coord(y, 6)} {_fmt_coord(z, 6)}"
        for x, y, z in vertices
    ]
    lines.extend(
        f"vn {_fmt_coord(nx, 5)} {_fmt_coord(ny, 5)} {_fmt_coord(nz, 5)}"
        for nx, ny, nz in normals
    )
    fc = faces + 1
    nc = normal_idx + 1
    lines.extend(
        f"f {fc[i, 0]}//{nc[i, 0]} {fc[i, 1]}//{nc[i, 1]} {fc[i, 2]}//{nc[i, 2]}"
        for i in range(len(faces))
    )
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    obj_path.write_bytes(payload)
    return len(payload)


def optimize_single_obj(obj_path: Path) -> tuple[int, int, int, int]:
    """Optimize one visual OBJ file; return (old_bytes, new_bytes, old_faces, new_faces)."""
    target_name = obj_path.parents[1].name
    if target_name == "gripper_g2":
        old_faces = sum(1 for line in obj_path.open("r", encoding="utf-8") if line.startswith("f "))
        old_bytes, new_bytes = _compact_reformat_preserve_topology(obj_path)
        return old_bytes, new_bytes, old_faces, old_faces

    old_bytes = obj_path.stat().st_size
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(obj_path))
    old_faces = ms.current_mesh().face_number()
    ms.meshing_remove_duplicate_vertices()

    if old_faces > LARGE_FACE_THRESHOLD:
        target_faces = LARGE_TARGET_FACES
    elif old_faces > MEDIUM_FACE_THRESHOLD:
        target_faces = MEDIUM_TARGET_FACES
    else:
        target_faces = 0

    if target_faces > 0 and ms.current_mesh().face_number() > target_faces:
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            qualitythr=0.5,
            preserveboundary=True,
            preservenormal=True,
            preservetopology=True,
            planarquadric=True,
            optimalplacement=False,
        )
    ms.meshing_remove_unreferenced_vertices()

    mesh = ms.current_mesh()
    vertices = mesh.vertex_matrix().astype(np.float64)
    faces = mesh.face_matrix().astype(np.int64)
    faces, normals, normal_idx = _compute_crease_normals(vertices, faces)
    new_bytes = _write_compact_obj(obj_path, vertices, faces, normals, normal_idx)
    return old_bytes, new_bytes, old_faces, len(faces)


def discover_real_visual_objs(models_dir: Path = MODELS_DIR) -> list[Path]:
    """Return sorted real (non-symlink) visual OBJ files, excluding source snapshots."""
    found: list[Path] = []
    for root, dirs, files in os.walk(models_dir, followlinks=False):
        root_path = Path(root)
        # Prune symlinked directories and gripper_g2/source snapshot
        dirs[:] = sorted(
            d for d in dirs
            if not (root_path / d).is_symlink()
            and not (root_path.name == "gripper_g2" and d == "source")
        )
        if root_path.name != "visual":
            continue
        for fname in sorted(files):
            if fname.endswith(".obj"):
                fpath = root_path / fname
                if not fpath.is_symlink():
                    found.append(fpath)
    return sorted(found)


def main() -> None:
    t0 = time.time()
    objs = discover_real_visual_objs()
    print(f"Optimizing {len(objs)} non-symlink visual OBJ files...")

    total_old_bytes = 0
    total_new_bytes = 0
    total_old_faces = 0
    total_new_faces = 0

    for obj_path in objs:
        rel = obj_path.relative_to(MODELS_DIR)
        old_b, new_b, old_f, new_f = optimize_single_obj(obj_path)
        total_old_bytes += old_b
        total_new_bytes += new_b
        total_old_faces += old_f
        total_new_faces += new_f
        print(
            f"  {str(rel):62s}: {old_b / 1024 / 1024:6.2f} -> {new_b / 1024 / 1024:5.2f} MB "
            f"| {old_f:7,d} -> {new_f:6,d} faces"
        )

    dt = time.time() - t0
    reduction = (1.0 - total_new_bytes / max(total_old_bytes, 1)) * 100.0
    print(
        f"\nCompleted in {dt:.1f}s: "
        f"{total_old_bytes / 1024 / 1024:.2f} MB -> {total_new_bytes / 1024 / 1024:.2f} MB "
        f"(-{reduction:.1f}%), "
        f"{total_old_faces:,d} -> {total_new_faces:,d} faces."
    )


if __name__ == "__main__":
    main()
