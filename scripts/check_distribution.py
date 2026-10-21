#!/usr/bin/env python3
"""Check both release archives against the source inventory without extraction."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import tarfile
import zipfile


def check(path, reference):
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            assert len({e.filename for e in entries}) == len(entries), "Duplicate ZIP members"
            assert not any(stat.S_ISLNK(e.external_attr >> 16) for e in entries), "Wheel contains symlink"
            files = {e.filename: archive.read(e) for e in entries if not e.is_dir()}
        assert not any(n.endswith("_build_support.py") for n in files)
        assert any(n.endswith(".dist-info/entry_points.txt") and b"ufactory-mujoco" in b
                   for n, b in files.items()), "Missing console entry point"
        for name in reference["licenses"]:
            assert any(".dist-info/licenses/" in n and n.endswith("/" + name) for n in files), name
    else:
        with tarfile.open(path, "r:gz") as archive:
            entries = archive.getmembers()
            assert all(e.isfile() or e.isdir() for e in entries), "sdist contains links or special files"
            files = {e.name.split("/", 1)[1]: archive.extractfile(e).read() for e in entries if e.isfile()}
        assert "_build_support.py" in files and "setup.py" in files, "Missing sdist build hooks"
    data = {n.removeprefix("ufactory_mujoco/data/"): b for n, b in files.items()
            if n.startswith("ufactory_mujoco/data/")}
    expected = {f"{group}/{name}" for group in ("models", "meshes", "configs", "licenses")
                for name in reference[group]}
    assert data.keys() == expected, f"Inventory mismatch: {set(data) ^ expected}"
    for group in ("meshes", "configs", "licenses"):
        for name, sha in reference[group].items():
            assert hashlib.sha256(data[f"{group}/{name}"]).hexdigest() == sha, f"Changed {group}/{name}"
    for name, baseline in reference["models"].items():
        xml = data[f"models/{name}"].decode()
        assert 'meshdir="assets"' not in xml, f"Unstaged XML: {name}"
        xml = re.sub(r'\smeshdir\s*=\s*"[^"]*"', "", xml)
        assert hashlib.sha256(xml.encode()).hexdigest() == baseline["sha256_without_meshdir"], name
    for name in files:
        parts = Path(name).parts
        assert not ({"dev", "build", "__pycache__", "logs", "weights"} & set(parts)), name
        assert not name.endswith((".pyc", ".zip", ".log", ".pt", ".pth")), name
    return {"file": path.name, "bytes": path.stat().st_size,
            "resource_bytes": sum(map(len, data.values())),
            "models": len(reference["models"]), "meshes": len(reference["meshes"]),
            "configs": len(reference["configs"]), "licenses": len(reference["licenses"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    archives = sorted(args.dist_dir.glob("*.whl")) + sorted(args.dist_dir.glob("*.tar.gz"))
    assert len(archives) == 2, "Expected one wheel and one sdist"
    result = {"success": False, "archives": []}
    try:
        for path in archives:
            result["archives"].append(check(path, reference))
        result["success"] = True
        print(json.dumps(result, indent=2))
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
