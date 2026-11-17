#!/usr/bin/env python3
"""Standalone distribution check and wheel acceptance; CI runs this without source checkout.

Record the source reference before building:
    python scripts/accept_installed.py --record-reference build/reference.json
Check built sdist and wheel archives without extraction:
    python scripts/accept_installed.py --check-dist dist --reference build/reference.json --report build/dist.json
Then run from an empty directory with an ordinary core-only wheel install:
    python -I accept_installed.py --reference reference.json --out-dir reports
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tarfile
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resource_reference():
    import mujoco
    import numpy as np
    from ufactory_mujoco.config import load_task_config
    from ufactory_mujoco.model_paths import CONFIGS_ROOT, MODELS_ROOT, RESOURCE_ROOT

    result = {"models": {}, "meshes": {}, "configs": {}, "config_hashes": {}, "licenses": {}}
    for path in sorted(MODELS_ROOT.glob("*/*.xml")):
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        mass = np.empty((model.nv, model.nv))
        mujoco.mj_fullM(model, data, mass)
        # meshdir is the only authored XML attribute that the build changes.
        xml = re.sub(r'\smeshdir\s*=\s*"[^"]*"', "", path.read_text(encoding="utf-8"))
        result["models"][path.relative_to(MODELS_ROOT).as_posix()] = {
            "nq": model.nq, "nv": model.nv, "nu": model.nu,
            "sha256_without_meshdir": hashlib.sha256(xml.encode()).hexdigest(),
            "mass_matrix": mass.tolist(),
        }
    mesh_root = RESOURCE_ROOT / "meshes"
    if mesh_root.is_dir():
        result["meshes"] = {p.relative_to(mesh_root).as_posix(): digest(p)
                            for p in sorted(mesh_root.rglob("*")) if p.is_file()}
    else:
        for asset in sorted(MODELS_ROOT.glob("*/assets/*")):
            if asset.is_symlink():
                continue
            for path in sorted(asset.rglob("*")):
                if path.is_file():
                    key = (Path(asset.name) / path.relative_to(asset)).as_posix()
                    assert key not in result["meshes"], f"Duplicate canonical mesh: {key}"
                    result["meshes"][key] = digest(path)
    for path in sorted(CONFIGS_ROOT.rglob("*.yaml")):
        key = path.relative_to(CONFIGS_ROOT).as_posix()
        result["configs"][key] = digest(path)
        if key == "default.yaml" or key.startswith(("pick_place/", "box_pack/")):
            result["config_hashes"][key] = load_task_config(None if key == "default.yaml" else path).sha256
    licenses = RESOURCE_ROOT / "licenses"
    for path in sorted(licenses.iterdir()):
        if path.is_file():
            result["licenses"][path.name] = digest(path)
    # The source keeps the project notices at its root; distributions also put
    # them next to the upstream texts inside package data.
    for name in ("LICENSE", "NOTICE"):
        path = RESOURCE_ROOT / name
        if path.is_file():
            result["licenses"][name] = digest(path)
    assert len(result["models"]) >= 99 and len(result["meshes"]) >= 444
    assert len(result["configs"]) == 12 and len(result["config_hashes"]) == 11 and len(result["licenses"]) == 6
    return result


def check_archive(path, reference):
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            assert len({e.filename for e in entries}) == len(entries), "Duplicate ZIP members"
            assert not any(stat.S_ISLNK(e.external_attr >> 16) for e in entries), "Wheel contains symlink"
            files = {e.filename: archive.read(e) for e in entries if not e.is_dir()}
        assert not any(n.endswith("_build_support.py") for n in files)
        entry_points = next((b for n, b in files.items() if n.endswith(".dist-info/entry_points.txt")), None)
        assert entry_points is not None, "Missing .dist-info/entry_points.txt"
        for script_name in (
            b"ufactory-mujoco",
            b"dynamics-sim-check",
            b"dynamics-sim-collision-check",
            b"dynamics-hardware-check",
            b"dynamics-report-compare",
        ):
            assert script_name in entry_points, f"Missing console entry point: {script_name.decode()}"
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
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "resource_bytes": sum(map(len, data.values())),
        "models": len(reference["models"]),
        "meshes": len(reference["meshes"]),
        "configs": len(reference["configs"]),
        "licenses": len(reference["licenses"]),
    }


def check_distributions(dist_dir, reference_path, report_path):
    reference = strict_json(reference_path)
    archives = sorted(dist_dir.glob("*.whl")) + sorted(dist_dir.glob("*.tar.gz"))
    assert len(archives) == 2, "Expected one wheel and one sdist"
    result = {"success": False, "archives": []}
    try:
        for path in archives:
            result["archives"].append(check_archive(path, reference))
        result["success"] = True
        print(json.dumps(result, indent=2))
    finally:
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")


def compare_resources(expected, actual):
    import numpy as np

    for key in ("meshes", "configs", "config_hashes", "licenses"):
        assert actual[key] == expected[key], f"Resource mismatch: {key}"
    assert actual["models"].keys() == expected["models"].keys()
    for name, baseline in expected["models"].items():
        model = actual["models"][name]
        for key in ("nq", "nv", "nu", "sha256_without_meshdir"):
            assert model[key] == baseline[key], f"{name}: {key} changed"
        np.testing.assert_allclose(model["mass_matrix"], baseline["mass_matrix"], atol=1e-12, rtol=1e-12,
                                   err_msg=f"{name}: mass matrix changed")


def strict_json(path):
    def invalid(value):
        raise AssertionError(f"Non-finite JSON: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def accept(args, summary):
    import ufactory_mujoco
    from ufactory_mujoco.model_paths import CONFIGS_ROOT, RESOURCE_ROOT

    package = Path(ufactory_mujoco.__file__).resolve()
    assert package.is_relative_to(Path(sys.prefix).resolve()), f"Imported outside install: {package}"
    assert RESOURCE_ROOT == package.parent / "data", "Wheel must use bundled resources"
    assert not any(p.is_symlink() for p in RESOURCE_ROOT.rglob("*")), "Symlink in installed resources"
    assert not os.environ.get("PYTHONPATH"), "Acceptance requires an unset PYTHONPATH"
    assert "UFACTORY_MUJOCO_MODELS" not in os.environ, "Acceptance must use bundled models"
    optional = ("trimesh", "coacd", "xarm")
    assert all(importlib.util.find_spec(name) is None for name in optional), "Optional dependency in core env"
    summary.update(python=sys.version, package=str(package), resource_root=str(RESOURCE_ROOT))
    compare_resources(strict_json(args.reference), resource_reference())
    summary["resources"] = "pass"
    from ufactory_mujoco import SimRobot
    with SimRobot() as robot:
        assert robot.n_arm == 6, "SimRobot() must default to the bundled xArm6 scene"
    summary["sim_robot_default"] = "pass"
    from ufactory_mujoco.robots import resolve_arm_scene
    with SimRobot(resolve_arm_scene("xarm6", "bio_gripper_g2", "scene")) as robot:
        assert robot.bio_g2_controller is not None
        robot.close_gripper()
        closed = robot.get_gripper_gap(measured=True)
        robot.open_gripper()
        opened = robot.get_gripper_gap(measured=True)
    assert abs(closed - 0.071) < 0.001 and abs(opened - 0.150) < 0.001, (closed, opened)
    with SimRobot(resolve_arm_scene("xarm6", "vacuum_gripper", "scene")) as robot:
        assert robot.gripper_kind == "vacuum" and not robot.has_gripper
    summary["end_effectors_0_1_3"] = "pass"
    # Exercise commands with a read-only installation. JSON belongs in the
    # working directory, and Python can import without writing bytecode here.
    for path in package.parent.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    package.parent.chmod(0o555)

    out = args.out_dir.resolve()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("UFACTORY_MUJOCO_MODELS", None)

    def run(label, arguments, expected=0, *, overrides=None, no_steps=False, executable="ufactory-mujoco"):
        if no_steps:
            code = (
                "import runpy; from unittest.mock import patch; "
                "guard=patch('mujoco.mj_step',side_effect=AssertionError('dry-run stepped physics')); "
                "guard.start(); runpy.run_module('ufactory_mujoco',run_name='__main__')"
            )
            cmd = [sys.executable, "-I", "-c", code, *arguments]
        else:
            cmd = [str(Path(sys.executable).with_name(executable)), *arguments]
        result = subprocess.run(cmd, cwd=out, env=env | (overrides or {}),
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
        (out / f"{label}.log").write_text(result.stdout, encoding="utf-8")
        summary["commands"].append({"name": label, "exit_code": result.returncode, "expected": expected})
        assert result.returncode == expected, f"{label}: exit {result.returncode}\n{result.stdout}"
        print(f"{label}: PASS", flush=True)
        return result

    check = subprocess.run([sys.executable, "-m", "pip", "check"], text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    (out / "pip-check.log").write_text(check.stdout, encoding="utf-8")
    assert check.returncode == 0, check.stdout
    run("help", ["--help"])
    for task in ("pick-place", "box-pack", "dynamics"):
        run(f"{task}-help", [task, "--help"])
    for task in ("pick-place", "box-pack"):
        for arm in ("lite6", "xarm5", "xarm6", "xarm7", "uf850"):
            for dry in (True, False):
                label = f"{task}-{arm}-{'dry' if dry else 'sim'}"
                arguments = [task, "--robot", arm, "--headless", "--report", f"{label}/report.json"]
                if dry:
                    arguments.append("--dry-run")
                run(label, arguments, no_steps=dry)
                report = strict_json(out / label / "report.json")
                assert report["ok"] and report["success"] and report["completed"]
                if not dry:
                    assert len(report["cycles"]) == 1
                    cycle = report["cycles"][0]
                    assert cycle["bilateral_contact"] and cycle["cube_lifted"]
                    assert cycle["home_drift_mm"] < 10
                    if task == "pick-place":
                        assert cycle["place_error_mm"] < 50
                    else:
                        assert cycle["cube_rested"] and cycle["in_box"] and cycle["bounds_margin_mm"] >= -1
    run("dynamics-all", ["dynamics", "--robot", "all", "--samples", "32", "--seed", "0",
                         "--headless", "--report", "dynamics/report.json"])
    dynamics = strict_json(out / "dynamics/report.json")
    assert dynamics["success"] and dynamics["completed"] and len(dynamics["robots"]) == 5
    assert all(r["success"] and r["motion"]["steps"] > 0 for r in dynamics["robots"])
    payload = next(r["payload"] for r in dynamics["robots"] if r["robot"] == "xarm6")
    assert payload["success"] and payload["max_baseline_error_nm"] <= 1e-6
    run("dynamics-sim-check-xarm6",
        ["--robot", "xarm6", "--random-count", "2", "--require-reference", "--report", "dyn-sim/xarm6.csv"],
        executable="dynamics-sim-check")
    assert (out / "dyn-sim/xarm6.csv").is_file() and (out / "dyn-sim/xarm6.jsonl").is_file()
    run("dynamics-hw-dry-lite6",
        ["--robot", "lite6", "--dry-run", "--require-reference", "--poses", "0,1,9"],
        executable="dynamics-hardware-check")
    assert list((out / "reports/dyn_ver_lite6").glob("*/*.csv")), "Missing default CWD dynamics report"
    run("dynamics-report-compare",
        ["dyn-sim/xarm6.jsonl", "dyn-sim/xarm6.jsonl"],
        executable="dynamics-report-compare")
    for task in ("pick-place", "box-pack", "dynamics"):
        run(f"{task}-invalid-robot", [task, "--robot", "invalid", "--headless"], 2)
        run(f"{task}-invalid-models", [task, "--headless", "--report", f"{task}-invalid-models.json"], 1,
            overrides={"UFACTORY_MUJOCO_MODELS": str(out / "missing-models")})
        failed = strict_json(out / f"{task}-invalid-models.json")
        assert not failed["success"]
        run(f"{task}-empty-models", [task, "--headless"], 1,
            overrides={"UFACTORY_MUJOCO_MODELS": str(out)})
        # Existing directory cannot be overwritten as a JSON report.
        run(f"{task}-report-error", [task, "--headless", "--report", str(out)] +
            (["--dry-run"] if task != "dynamics" else ["--robot", "xarm5", "--samples", "1"]), 1)
    for task in ("pick-place", "box-pack"):
        run(f"{task}-missing-config", [task, "--config", "missing.yaml", "--headless"], 2)
        run(f"{task}-invalid-cycles", [task, "--cycles", "0", "--headless"], 2)
    run("box-invalid-speed", ["box-pack", "--speed", "nan", "--headless"], 2)
    # Exercise the installed preflight failure and its strict JSON report.
    import yaml
    for task, section, field in (("pick-place", "pick_place", "target_xy"),
                                 ("box-pack", "box_pack", "drop_xy")):
        overlay = yaml.safe_load((CONFIGS_ROOT / section / "xarm6.yaml").read_text())
        overlay.setdefault(section, {})[field] = [9.0, 9.0] if section == "pick_place" else [[9.0, 9.0]] * 3
        path = out / f"{task}-unreachable.yaml"
        path.write_text(yaml.safe_dump(overlay), encoding="utf-8")
        run(f"{task}-preflight-fail", [task, "--config", str(path), "--dry-run", "--headless",
                                       "--report", f"{task}-preflight-fail.json"], 1, no_steps=True)
        report = strict_json(out / f"{task}-preflight-fail.json")
        assert not report["ok"] and not report["success"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-reference", type=Path)
    parser.add_argument("--check-dist", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    if args.record_reference:
        args.record_reference.parent.mkdir(parents=True, exist_ok=True)
        args.record_reference.write_text(json.dumps(resource_reference(), allow_nan=False, indent=2) + "\n")
        print(f"Source reference: {args.record_reference}")
        return
    if args.check_dist:
        if args.reference is None:
            parser.error("--reference is required with --check-dist")
        check_distributions(args.check_dist, args.reference, args.report)
        return
    if args.reference is None:
        parser.error("--reference is required for installed acceptance")
    args.reference = args.reference.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"success": False, "commands": []}
    try:
        accept(args, summary)
        summary["success"] = True
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.out_dir / "acceptance.json").write_text(json.dumps(summary, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
