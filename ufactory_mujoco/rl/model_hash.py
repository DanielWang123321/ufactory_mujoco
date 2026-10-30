"""Content fingerprint of an MJCF scene and its file-backed assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


class ModelHashError(ValueError):
    """The scene or one of its model dependencies cannot be fingerprinted."""


def _sha256(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        raise ModelHashError(f"cannot read model dependency {path}: {exc}") from exc


def _canonical_element(element: ET.Element) -> dict:
    attributes = dict(element.attrib)
    # Packaging relocates the canonical meshes and rewrites only this setting.
    # The referenced mesh contents below still bind its effective value.
    if element.tag == "compiler":
        attributes.pop("meshdir", None)
    return {
        "tag": element.tag,
        "attributes": attributes,
        "text": (element.text or "").strip(),
        "children": [_canonical_element(child) for child in element],
    }


def runtime_model_sha256(scene_path: str | Path) -> str:
    """Hash the scene, recursively included MJCFs and their referenced assets.

    MuJoCo resolves included paths and compiler asset directories relative to
    the main MJCF directory. Logical references and file contents are hashed;
    absolute installation paths and the packaging-only meshdir rewrite are not.
    """
    scene = Path(scene_path).absolute()
    root_dir = scene.parent
    documents: list[tuple[str, ET.Element]] = []
    active: set[Path] = set()
    seen: set[Path] = set()

    def visit(path: Path, logical: str) -> None:
        identity = path.resolve()
        if identity in active:
            raise ModelHashError(f"cyclic MJCF include: {path}")
        if identity in seen:
            raise ModelHashError(f"duplicate MJCF include: {path}")
        active.add(identity)
        seen.add(identity)
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError) as exc:
            raise ModelHashError(f"cannot read model XML {path}: {exc}") from exc
        documents.append((logical, root))
        for include in root.iter("include"):
            name = include.get("file")
            if not name:
                raise ModelHashError(f"include without file in {path}")
            visit(root_dir / name, name)
        active.remove(identity)

    visit(scene, "<scene>")
    compiler: dict[str, str] = {}
    for _, root in documents:
        for item in root.iter("compiler"):
            compiler.update(item.attrib)
    assetdir = compiler.get("assetdir", "")
    meshdir = compiler.get("meshdir", assetdir)
    texturedir = compiler.get("texturedir", assetdir)
    strip = compiler.get("strippath", "false").lower() in ("true", "1")
    assets = []
    directories = {"mesh": meshdir, "skin": meshdir, "texture": texturedir, "hfield": assetdir}
    for logical, root in documents:
        for element in root.iter():
            name = element.get("file")
            if not name or element.tag == "include":
                continue
            if element.tag not in directories:
                raise ModelHashError(f"unsupported file-backed MJCF element {element.tag!r} in {logical}")
            actual_name = Path(name).name if strip else name
            path = root_dir / directories[element.tag] / actual_name
            assets.append({"document": logical, "type": element.tag, "file": name, "sha256": _sha256(path)})

    body = {
        "schema_version": 1,
        "documents": [{"name": name, "xml": _canonical_element(root)} for name, root in documents],
        "assets": assets,
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_runtime_model(scene_path: str | Path, expected_sha256: str) -> str:
    actual = runtime_model_sha256(scene_path)
    if actual != expected_sha256:
        raise ModelHashError(f"runtime model hash mismatch for {scene_path}: expected {expected_sha256}, got {actual}")
    return actual
