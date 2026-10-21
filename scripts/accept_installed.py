#!/usr/bin/env python3
"""Standalone wheel acceptance; CI downloads this file without checking out source.

Record the source reference before building:
    python scripts/accept_installed.py --record-reference build/reference.json
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
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resource_reference():
    import mujoco
    import numpy as np
    from ufactory_mujoco.config import load_task_config
    from ufactory_mujoco.resource_paths import CONFIGS_ROOT, MODELS_ROOT, RESOURCE_ROOT

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
    assert len(result["models"]) >= 48 and len(result["meshes"]) >= 135
    assert len(result["configs"]) == 11 and len(result["licenses"]) == 5
    return result


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
    from ufactory_mujoco.resource_paths import RESOURCE_ROOT

    package = Path(ufactory_mujoco.__file__).resolve()
    assert package.is_relative_to(Path(sys.prefix).resolve()), f"Imported outside install: {package}"
    assert RESOURCE_ROOT == package.parent / "data", "Wheel must use bundled resources"
    assert not any(p.is_symlink() for p in RESOURCE_ROOT.rglob("*")), "Symlink in installed resources"
    assert not os.environ.get("PYTHONPATH"), "Acceptance requires an unset PYTHONPATH"
    assert "UFACTORY_MUJOCO_MODELS" not in os.environ, "Acceptance must use bundled models"
    optional = ("torch", "gymnasium", "stable_baselines3", "trimesh", "xacro", "xarm")
    assert all(importlib.util.find_spec(name) is None for name in optional), "Optional dependency in core env"
    summary.update(python=sys.version, package=str(package), resource_root=str(RESOURCE_ROOT))
    compare_resources(strict_json(args.reference), resource_reference())
    summary["resources"] = "pass"
    # Exercise commands with a read-only installation. JSON belongs in the
    # working directory, and Python can import without writing bytecode here.
    for path in package.parent.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    package.parent.chmod(0o555)

    out = args.out_dir.resolve()
    command = Path(sys.executable).with_name("ufactory-mujoco")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("UFACTORY_MUJOCO_MODELS", None)

    def run(label, arguments, expected=0, *, overrides=None, no_steps=False):
        if no_steps:
            code = (
                "import runpy; from unittest.mock import patch; "
                "guard=patch('mujoco.mj_step',side_effect=AssertionError('dry-run stepped physics')); "
                "guard.start(); runpy.run_module('ufactory_mujoco',run_name='__main__')"
            )
            cmd = [sys.executable, "-I", "-c", code, *arguments]
        else:
            cmd = [str(command), *arguments]
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
    from ufactory_mujoco.resource_paths import CONFIGS_ROOT
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
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    if args.record_reference:
        args.record_reference.parent.mkdir(parents=True, exist_ok=True)
        args.record_reference.write_text(json.dumps(resource_reference(), allow_nan=False, indent=2) + "\n")
        print(f"Source reference: {args.record_reference}")
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
