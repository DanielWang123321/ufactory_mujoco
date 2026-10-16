#!/usr/bin/env python3
"""Convert UFACTORY robotic arms and accessories GLB visual assets to multi-material OBJs for MuJoCo.

Decodes Draco-compressed GLB meshes from ufactory_genesis, aligns coordinate frames
(glTF Y-up to MuJoCo Z-up), groups sub-meshes by material category (white paint,
silver metal, black accents, seals, gray paint), and exports smooth-normal OBJ files.
Also copies collision STLs.

Repairs inconsistent/inward triangle winding for MuJoCo's back-face culling while
retaining original vertices and hard-edge normals.

Usage:
    python scripts/convert_glb_to_obj.py --ref-dir /path/to/ufactory_genesis/assets/urdf --target all
    UFACTORY_GENESIS_ROOT=/path/to/ufactory_genesis python scripts/convert_glb_to_obj.py --target xarm7
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import trimesh


def orient_mesh_outward(mesh: trimesh.Trimesh) -> int:
    """Repair winding in place without welding the exported mesh; return flip count.

    Position-only welding on a temporary mesh joins seams split for hard normals.
    Adjacent faces are oriented consistently, then each connected patch is made
    outward using signed volume about its own center. Near-zero-volume patches
    keep their original winding: an open/flat patch has no reliable inside/outside.
    """
    import numpy as np
    import trimesh

    if len(mesh.faces) == 0:
        return 0

    original_faces = mesh.faces.copy()
    vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
    welded_faces = inverse[original_faces]
    topology = trimesh.Trimesh(vertices=vertices, faces=welded_faces, process=False)
    trimesh.repair.fix_winding(topology)
    flipped = np.any(topology.faces != welded_faces, axis=1)

    components = trimesh.graph.connected_components(
        topology.face_adjacency, nodes=np.arange(len(original_faces)), min_len=1
    )
    for component in components:
        faces = topology.faces[component]
        points = vertices[np.unique(faces)]
        triangles = vertices[faces] - points.mean(axis=0)
        volume = np.einsum(
            "ij,ij->i", triangles[:, 0],
            np.cross(triangles[:, 1], triangles[:, 2]),
        ).sum() / 6.0
        # Relative tolerance works independently of mesh units and translation.
        # Tiny net volumes of open decorative patches do not determine a side.
        volume_tolerance = np.linalg.norm(np.ptp(points, axis=0)) ** 3 * 1e-6
        if abs(volume) <= volume_tolerance:
            flipped[component] = False
        elif volume < 0:
            flipped[component] = ~flipped[component]

    count = int(flipped.sum())
    if count == 0:
        return 0

    normals = mesh.vertex_normals.copy()
    flipped_vertices = np.zeros(len(mesh.vertices), dtype=bool)
    unchanged_vertices = np.zeros(len(mesh.vertices), dtype=bool)
    flipped_vertices[original_faces[flipped].ravel()] = True
    unchanged_vertices[original_faces[~flipped].ravel()] = True
    shared_vertices = flipped_vertices & unchanged_vertices
    normals[flipped_vertices & ~unchanged_vertices] *= -1

    repaired_faces = original_faces.copy()
    repaired_faces[flipped] = repaired_faces[flipped, ::-1]
    mesh.faces = repaired_faces
    if np.any(shared_vertices):
        # A shared normal cannot be both kept and negated. Recompute only those
        # normals on the original (unwelded) vertices to preserve hard edges.
        normals[shared_vertices] = mesh.vertex_normals[shared_vertices]
    mesh.vertex_normals = normals
    return count


def classify_material(color: list[float] | tuple[float, ...], metal: float) -> str:
    """Classify sub-mesh PBR material properties into unified categories."""
    if metal > 0.5:
        return "silver"
    rgb = color[:3]
    if max(rgb) < 0.6:
        return "black"
    if rgb[2] > rgb[0] + 0.02:
        return "seal"
    if 0.7 < min(rgb) < 0.85 and max(rgb) - min(rgb) < 0.05:
        return "gray"
    return "white"


# Upstream GLB primitives stored in an inconsistent local frame (known
# ufactory_genesis data bug; their own URDFs avoid these GLB parts and use the
# STLs instead). Skip them so they don't pollute the merged material OBJs.
_SKIP_PRIMITIVES: dict[str, set[int]] = {
    "finger1.glb": {4},  # white fingertip pad, wrong frame (spans y +/-24 mm)
}


def get_target_specs(ref_root: Path, out_root: Path) -> dict[str, dict[str, Path]]:
    """Return dictionary of source/destination paths for each robot arm and accessory."""
    return {
        "xarm6": {
            "src_glb": ref_root / "xarm6/meshes/xarm6_1305/visual_glb",
            "src_col": ref_root / "xarm6/meshes/xarm6_1305/collision",
            "out_vis": out_root / "xarm6/assets/xarm6/visual",
            "out_col": out_root / "xarm6/assets/xarm6/collision",
        },
        "xarm5": {
            "src_glb": ref_root / "xarm5/meshes/xarm5_1305/visual_glb",
            "src_col": ref_root / "xarm5/meshes/xarm5_1305/collision",
            "out_vis": out_root / "xarm5/assets/xarm5/visual",
            "out_col": out_root / "xarm5/assets/xarm5/collision",
        },
        "xarm7": {
            "src_glb": ref_root / "xarm7/meshes/xarm7_1305/visual_glb",
            "src_col": ref_root / "xarm7/meshes/xarm7_1305/collision",
            "out_vis": out_root / "xarm7/assets/xarm7/visual",
            "out_col": out_root / "xarm7/assets/xarm7/collision",
        },
        "uf850": {
            "src_glb": ref_root / "uf850/meshes/uf850/visual_glb",
            "src_col": ref_root / "uf850/meshes/uf850/collision",
            "out_vis": out_root / "uf850/assets/uf850/visual",
            "out_col": out_root / "uf850/assets/uf850/collision",
        },
        "lite6": {
            "src_glb": ref_root / "lite6/meshes/lite6/visual_glb",
            "src_col": ref_root / "lite6/meshes/lite6/collision",
            "out_vis": out_root / "lite6/assets/lite6/visual",
            "out_col": out_root / "lite6/assets/lite6/collision",
        },
        "gripper_g2": {
            "src_glb": ref_root / "gripper_g2/meshes/visual/visual_glb",
            "src_col": ref_root / "gripper_g2/meshes/collision",
            "out_vis": out_root / "gripper_g2/assets/gripper_g2/visual",
            "out_col": out_root / "gripper_g2/assets/gripper_g2/collision",
        },
        "bio_gripper_g2": {
            "src_glb": ref_root / "bio_gripper_g2/meshes/visual/visual_glb",
            "src_col": ref_root / "bio_gripper_g2/meshes/visual",
            "out_vis": out_root / "bio_gripper_g2/assets/bio_gripper_g2/visual",
            "out_col": out_root / "bio_gripper_g2/assets/bio_gripper_g2/collision",
        },
        "lite6_gripper": {
            "src_glb": ref_root / "lite6_gripper/meshes/visual/visual_glb",
            "src_col": ref_root / "lite6_gripper/meshes/collision",
            "out_vis": out_root / "lite6_gripper/assets/lite6_gripper/visual",
            "out_col": out_root / "lite6_gripper/assets/lite6_gripper/collision",
        },
        "lite6_vacuum_gripper": {
            "src_glb": ref_root / "lite6_vacuum_gripper/meshes/visual",
            "src_col": ref_root / "lite6_vacuum_gripper/meshes/visual",
            "out_vis": out_root / "lite6_vacuum_gripper/assets/lite6_vacuum_gripper/visual",
            "out_col": out_root / "lite6_vacuum_gripper/assets/lite6_vacuum_gripper/collision",
        },
    }


def convert_single_target(
    name: str,
    src_glb_dir: Path,
    src_col_dir: Path,
    out_vis_dir: Path,
    out_col_dir: Path,
) -> dict[str, int]:
    """Convert visual GLBs and copy collision STLs for a single target."""
    import genesis as gs
    import genesis.utils.gltf as gltf_utils
    import trimesh

    print(f"\n==================== Converting [{name}] ====================")
    out_vis_dir.mkdir(parents=True, exist_ok=True)
    out_col_dir.mkdir(parents=True, exist_ok=True)

    stats = {"stl_count": 0, "obj_count": 0, "flipped_faces": 0}

    # 1. Copy collision STLs
    stl_files = sorted(src_col_dir.glob("*.stl"))
    print(f"[1/2] Copying {len(stl_files)} collision STLs from {src_col_dir} -> {out_col_dir}...")
    for stl_path in stl_files:
        dst = out_col_dir / stl_path.name
        shutil.copy2(stl_path, dst)
        stats["stl_count"] += 1
        print(f"  Copied collision mesh: {dst.name}")

    # 2. Convert visual GLBs -> multi-geom OBJs
    glb_files = sorted(src_glb_dir.glob("*.glb"))
    # In lite6_vacuum_gripper, visual file name has visual_link6 suffix; filter out any static combo if needed
    # Only keep the standalone/per-link GLBs
    glb_files = [g for g in glb_files if not g.name.endswith("_static.glb")]
    if not glb_files:
        raise FileNotFoundError(f"No GLB files found in {src_glb_dir}")

    print(f"[2/2] Converting {len(glb_files)} visual GLBs from {src_glb_dir} -> {out_vis_dir}...")
    for glb_path in glb_files:
        link_name = glb_path.stem
        # Normalize vacuum gripper file name
        if link_name == "lite6_vacuum_gripper_visual_link6":
            link_name = "vacuum_gripper"

        with open(glb_path, "rb") as fp:
            magic, ver, length = struct.unpack("<4sII", fp.read(12))
            chunk_len, chunk_type = struct.unpack("<I4s", fp.read(8))
            gltf = json.loads(fp.read(chunk_len).decode("utf-8"))

        meshes_raw = gltf.get("meshes", [])
        materials = gltf.get("materials", [])
        parsed = gltf_utils.parse_mesh_glb(
            str(glb_path), True, 1.0, True, gs.surfaces.Default()
        )

        skip = _SKIP_PRIMITIVES.get(glb_path.name, set())
        categories: dict[str, list[trimesh.Trimesh]] = {}
        for i, m in enumerate(parsed):
            if i in skip:
                print(f"    skipping primitive {i} ({glb_path.name}): inconsistent frame")
                continue
            mat_idx = meshes_raw[i]["primitives"][0].get("material", 0)
            pbr = materials[mat_idx].get("pbrMetallicRoughness", {}) if mat_idx < len(materials) else {}
            color = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
            metal = pbr.get("metallicFactor", 0.0)

            cat = classify_material(color, metal)
            categories.setdefault(cat, []).append(m.trimesh)

        print(f"  {link_name}: {len(parsed)} primitives across {list(categories.keys())}")
        for cat, trimesh_list in categories.items():
            if len(trimesh_list) == 1:
                merged = trimesh_list[0]
            else:
                merged = trimesh.util.concatenate(trimesh_list)

            flipped_count = orient_mesh_outward(merged)
            stats["flipped_faces"] += flipped_count

            # Suffix with material name if link has multiple components
            obj_name = f"{link_name}_{cat}.obj" if len(categories) > 1 else f"{link_name}.obj"
            out_file = out_vis_dir / obj_name
            merged.export(str(out_file), include_normals=True)
            stats["obj_count"] += 1
            print(f"    -> {obj_name} ({len(merged.vertices)} verts, "
                  f"{len(merged.faces)} faces, flipped {flipped_count})")

    return stats


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_ref = (
        Path(os.environ["UFACTORY_GENESIS_ROOT"]) / "assets" / "urdf"
        if "UFACTORY_GENESIS_ROOT" in os.environ
        else (repo_root.parent / "ufactory_genesis" / "assets" / "urdf")
    )

    out_root = repo_root / "models"

    parser = argparse.ArgumentParser(description="Convert UFACTORY GLB assets to MuJoCo OBJs.")
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=default_ref,
        help="Path to ufactory_genesis assets/urdf (default: $UFACTORY_GENESIS_ROOT/assets/urdf or ../ufactory_genesis/assets/urdf)",
    )
    parser.add_argument(
        "--target",
        default="all",
        help="Target robot or accessory to convert (default: all)",
    )
    args = parser.parse_args()

    target_specs = get_target_specs(args.ref_dir, out_root)
    valid_targets = ["all"] + list(target_specs.keys())
    if args.target not in valid_targets:
        parser.error(f"argument --target: invalid choice: {args.target!r} (choose from {', '.join(valid_targets)})")

    try:
        import genesis as gs
    except ImportError as e:
        print(f"Error: Missing dependency: {e}", file=sys.stderr)
        print("Please run with an environment containing genesis (pip install genesis-world trimesh)", file=sys.stderr)
        sys.exit(1)

    gs.init(backend=gs.cpu)

    targets_to_run = list(target_specs.keys()) if args.target == "all" else [args.target]

    summary = {}
    for tgt in targets_to_run:
        spec = target_specs[tgt]
        stats = convert_single_target(
            name=tgt,
            src_glb_dir=spec["src_glb"],
            src_col_dir=spec["src_col"],
            out_vis_dir=spec["out_vis"],
            out_col_dir=spec["out_col"],
        )
        summary[tgt] = stats

    print("\n==================== SUMMARY ====================")
    total_objs = sum(s["obj_count"] for s in summary.values())
    total_stls = sum(s["stl_count"] for s in summary.values())
    total_flips = sum(s["flipped_faces"] for s in summary.values())
    for tgt, s in summary.items():
        print(f"  {tgt:20s}: {s['obj_count']} OBJs, {s['stl_count']} STLs (flipped {s['flipped_faces']} faces)")
    print(f"Total: {total_objs} OBJs, {total_stls} STLs, {total_flips} total flipped faces.")
    print("All conversions completed successfully!")


if __name__ == "__main__":
    main()
