#!/usr/bin/env python3
"""Make selected visual OBJs double-sided (equivalent of glTF ``doubleSided``).

MuJoCo culls back faces. Thin-walled or open CAD parts whose face orientation is
unreliable then show holes. For each OBJ in ``DOUBLE_SIDED_VISUALS`` this script
appends a negated copy of every ``vn`` line and a reversed twin of every face that
references the negated normals. ``v`` lines are unchanged. The script is
idempotent: files that already carry the twin faces are skipped.

MJCF files that reference these OBJs must declare ``inertia="shell"`` on the
``<mesh>`` (see ``add_shell_inertia``); the ``exact`` inertia mode rejects the
duplicated, oppositely oriented faces.

Pipeline order: convert_glb_to_obj.py -> fix_g2_base_meshes.py -> this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# Paths relative to models/. Chosen by a culling on/off depth comparison:
# all Gripper G2 visuals, plus every other visual mesh whose see-through
# pixel ratio exceeds 1 % (see worklog 0.1.5).
DOUBLE_SIDED_VISUALS: tuple[str, ...] = (
    "bio_gripper_g2/assets/bio_gripper_g2/visual/bio_gripper_g2_base_silver.obj",
    "bio_gripper_g2/assets/bio_gripper_g2/visual/bio_gripper_g2_base_white.obj",
    "bio_gripper_g2/assets/bio_gripper_g2/visual/bio_gripper_g2_left_finger.obj",
    "bio_gripper_g2/assets/bio_gripper_g2/visual/bio_gripper_g2_right_finger.obj",
    "gripper_g2/assets/gripper_g2/visual/base.obj",
    "gripper_g2/assets/gripper_g2/visual/left_finger.obj",
    "gripper_g2/assets/gripper_g2/visual/left_inner_knuckle.obj",
    "gripper_g2/assets/gripper_g2/visual/left_outer_knuckle.obj",
    "gripper_g2/assets/gripper_g2/visual/right_finger.obj",
    "gripper_g2/assets/gripper_g2/visual/right_inner_knuckle.obj",
    "gripper_g2/assets/gripper_g2/visual/right_outer_knuckle.obj",
    "lite6/assets/lite6/visual/link2_gray.obj",
    "lite6/assets/lite6/visual/link6_seal.obj",
    "lite6/assets/lite6/visual/link6_silver.obj",
    "lite6_gripper/assets/lite6_gripper/visual/finger1_white.obj",
    "lite6_gripper/assets/lite6_gripper/visual/finger2_white.obj",
    "uf850/assets/uf850/visual/link3_silver.obj",
    "uf850/assets/uf850/visual/link4_silver.obj",
    "uf850/assets/uf850/visual/link6.obj",
    "vacuum_gripper/assets/vacuum_gripper/visual/vacuum_gripper_white.obj",
    "xarm5/assets/xarm5/visual/link3_silver.obj",
    "xarm5/assets/xarm5/visual/link5_black.obj",
    "xarm5/assets/xarm5/visual/link5_silver.obj",
    "xarm5/assets/xarm5/visual/link5_white.obj",
    "xarm6/assets/xarm6/visual/link1_seal.obj",
    "xarm6/assets/xarm6/visual/link4_silver.obj",
    "xarm6/assets/xarm6/visual/link6_black.obj",
    "xarm6/assets/xarm6/visual/link6_silver.obj",
    "xarm6/assets/xarm6/visual/link6_white.obj",
    "xarm7/assets/xarm7/visual/link2_silver.obj",
    "xarm7/assets/xarm7/visual/link4_silver.obj",
    "xarm7/assets/xarm7/visual/link5_silver.obj",
    "xarm7/assets/xarm7/visual/link6_silver.obj",
    "xarm7/assets/xarm7/visual/link7_black.obj",
    "xarm7/assets/xarm7/visual/link7_silver.obj",
    "xarm7/assets/xarm7/visual/link7_white.obj",
)


def double_sided_paths() -> list[Path]:
    return [MODELS_DIR / rel for rel in DOUBLE_SIDED_VISUALS]


def _is_double_sided_target(file_attr: str) -> bool:
    """True if an MJCF ``<mesh file=...>`` value points at a listed OBJ.

    ``file_attr`` is relative to the model's meshdir (``<model>/assets``), e.g.
    ``gripper_g2/visual/base.obj``. Combo models reach the same files through
    symlinked asset directories, so matching on ``<part>/visual/<name>.obj`` is
    sufficient.
    """
    norm = file_attr.replace("\\", "/")
    for rel in DOUBLE_SIDED_VISUALS:
        tail = rel.split("/assets/", 1)[1]
        if norm == tail or norm.endswith("/" + tail):
            return True
    return False


def add_shell_inertia(xml_text: str) -> str:
    """Add ``inertia="shell"`` to every ``<mesh .../>`` that references a listed OBJ."""

    def repl(m: re.Match[str]) -> str:
        tag = m.group(0)
        if "inertia=" in tag:
            return tag
        fm = re.search(r'file="([^"]+)"', tag)
        if not fm or not _is_double_sided_target(fm.group(1)):
            return tag
        body = tag[:-2].rstrip()
        return f'{body} inertia="shell"/>'

    return re.sub(r"<mesh\b[^>]*/>", repl, xml_text)


def apply_shell_inertia(root: ET.Element) -> None:
    """ElementTree variant of ``add_shell_inertia`` (used by build_camera_mjcf.py)."""
    for mesh in root.iter("mesh"):
        file_attr = mesh.get("file")
        if file_attr and "inertia" not in mesh.attrib and _is_double_sided_target(file_attr):
            mesh.set("inertia", "shell")


def _parse(path: Path) -> tuple[list[str], list[str], list[list[str]]]:
    """Return (non-vn non-f lines, vn lines, face corner token lists)."""
    other: list[str] = []
    normals: list[str] = []
    faces: list[list[str]] = []
    with path.open("r", encoding="utf-8") as stream:
        for raw in stream:
            line = raw.rstrip("\n")
            if line.startswith("vn "):
                normals.append(line)
            elif line.startswith("f "):
                faces.append(line.split()[1:])
            else:
                other.append(line)
    return other, normals, faces


def _corner(tok: str) -> tuple[str, str, str]:
    parts = tok.split("/")
    v = parts[0]
    vt = parts[1] if len(parts) > 1 else ""
    vn = parts[2] if len(parts) > 2 else ""
    return v, vt, vn


def _neg_normal(line: str) -> tuple[float, float, float]:
    x, y, z = (float(t) for t in line.split()[1:4])
    return -x, -y, -z


def is_double_sided(path: Path) -> bool:
    """True if the second half of faces/normals is the reversed/negated first half."""
    _, normals, faces = _parse(path)
    nf, nn = len(faces), len(normals)
    if nf == 0 or nf % 2 or nn % 2:
        return False
    hf, hn = nf // 2, nn // 2
    for a, b in zip(faces[:hf], faces[hf:]):
        ca = [_corner(t) for t in a]
        cb = [_corner(t) for t in b]
        if len(ca) != len(cb):
            return False
        if [c[0] for c in cb] != [ca[0][0]] + [c[0] for c in reversed(ca[1:])]:
            return False
        for x, y in zip([ca[0]] + list(reversed(ca[1:])), cb):
            if x[2] and (not y[2] or int(y[2]) != int(x[2]) + hn):
                return False
    for a, b in zip(normals[:hn], normals[hn:]):
        na = _neg_normal(a)
        nb = tuple(float(t) for t in b.split()[1:4])
        if any(abs(p - q) > 1e-6 for p, q in zip(na, nb)):
            return False
    return True


def _fmt(val: float) -> str:
    s = f"{val:.8g}"
    return "0" if s in ("-0", "0", "-0.0") else s


def make_double_sided(path: Path) -> bool:
    """Append twin faces and negated normals. Returns False if already done."""
    if is_double_sided(path):
        return False
    other, normals, faces = _parse(path)
    hn = len(normals)
    out = list(other)
    out.extend(normals)
    for line in normals:
        x, y, z = _neg_normal(line)
        out.append(f"vn {_fmt(x)} {_fmt(y)} {_fmt(z)}")
    out.extend("f " + " ".join(f) for f in faces)
    for f in faces:
        cs = [_corner(t) for t in f]
        rev = [cs[0]] + list(reversed(cs[1:]))
        toks = []
        for v, vt, vn in rev:
            if vn:
                toks.append(f"{v}/{vt}/{int(vn) + hn}")
            elif vt:
                toks.append(f"{v}/{vt}")
            else:
                toks.append(v)
        out.append("f " + " ".join(toks))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--check", action="store_true", help="only verify, exit 1 if any file is single-sided")
    args = ap.parse_args()
    bad = 0
    for path in double_sided_paths():
        rel = path.relative_to(ROOT)
        if args.check:
            ok = is_double_sided(path)
            bad += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {rel}")
        else:
            print(f"{'doubled' if make_double_sided(path) else 'skip   '} {rel}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
