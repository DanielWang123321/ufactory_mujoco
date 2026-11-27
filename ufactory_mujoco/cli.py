"""Unified console entry point for simulation tasks, model export, and hardware checks."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

_TASK_COMMANDS = {
    "pick-place": ("ufactory_mujoco.tasks.pick_place", "main"),
    "box-pack": ("ufactory_mujoco.tasks.box_pack", "main"),
    "dynamics": ("ufactory_mujoco.tasks.dynamics", "main"),
    "export-calibration": ("ufactory_mujoco.dynamics.cli", "cli_export_calibration"),
    "kinematics-check": ("ufactory_mujoco.kinematics_validation", "main"),
}
_ALL_COMMANDS = (*_TASK_COMMANDS, "export-model")


def _export_model_cli(argv: list[str]) -> int:
    from .compose import COMPOSED_MODELS, available_models, export_model
    from .robots import ARMS

    parser = argparse.ArgumentParser(
        prog="ufactory-mujoco export-model",
        description="Export a static or MjSpec-composed model XML and scene.xml for MuJoCo simulate.",
    )
    parser.add_argument(
        "model", nargs="?", default=None,
        help="named model or combo (e.g. xarm6_with_ft_sensor_and_gripper)",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="destination directory (default: ./<model>)")
    parser.add_argument(
        "--copy-meshes", action="store_true",
        help="copy referenced meshes into <out-dir>/assets/ for portable export",
    )
    parser.add_argument("--list", action="store_true", help="list all static and dynamically composed model names")
    parser.add_argument("--arm", choices=tuple(ARMS), default=None, help="custom composition arm")
    parser.add_argument(
        "--linear-motor",
        action="store_true",
        help="custom composition: mount arm on AL0700 linear motor",
    )
    parser.add_argument("--base-plate", action="store_true", help="custom composition: mount arm on its base plate")
    parser.add_argument("--ft-sensor", action="store_true", help="custom composition: mount AI1500 F/T sensor")
    parser.add_argument(
        "--gripper", default=None,
        help="custom composition gripper (gripper_g2, bio_gripper_g2, vacuum_gripper, ...)",
    )
    parser.add_argument("--camera", default=None, help="custom composition camera (d405, d435, gemini_305, zed_x_nano)")
    parser.add_argument("--camera-stand", action="store_true", help="custom composition: include camera stand")
    args = parser.parse_args(argv)

    if args.list:
        composed = set(COMPOSED_MODELS)
        for name in available_models():
            tag = "composed" if name in composed else "static"
            print(f"{name:42s} [{tag}]")
        return 0

    if args.model is None and args.arm is None and not args.camera_stand and not args.linear_motor:
        parser.error(
            "specify a MODEL name, --list, or custom composition flags (--arm / --camera-stand / --linear-motor)"
        )

    default_dir = args.model or (
        f"{args.arm}_exported" if args.arm else ("al0700_exported" if args.linear_motor else "camera_stand_exported")
    )
    out_dir = args.out_dir or Path(default_dir)
    try:
        scene_path = export_model(
            args.model,
            out_dir,
            arm=args.arm,
            base_plate=args.base_plate,
            ft_sensor=args.ft_sensor,
            gripper=args.gripper,
            camera=args.camera,
            camera_stand=args.camera_stand,
            linear_motor=args.linear_motor,
            copy_meshes=args.copy_meshes,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Exported scene: {scene_path}")
    return 0


def main(argv=None):
    """Dispatch ``ufactory-mujoco <command>`` and return its exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="ufactory-mujoco", description="UFACTORY MuJoCo simulation tasks and utilities",
    )
    parser.add_argument("command", choices=_ALL_COMMANDS, help="command to run; use COMMAND --help")
    args = parser.parse_args(argv[:1])
    if args.command == "export-model":
        return _export_model_cli(argv[1:])
    mod_name, fn_name = _TASK_COMMANDS[args.command]
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)(argv[1:])
