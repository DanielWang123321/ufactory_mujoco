#!/usr/bin/env python3
"""Extract per-robot kinematic calibration YAML from an xArm control box.

Output uses schema_version 1 bound to the full robot serial number:
  joints.joint1..N: {x, y, z, roll, pitch, yaw} (meters, radians)

Usage:
    python scripts/gen_kinematics_params.py 192.168.1.60
    python scripts/gen_kinematics_params.py 192.168.1.60 L56A01 calibrations
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ufactory_mujoco.calibration import (  # noqa: E402
    export_kinematics_calibration,
    fetch_kinematics_bytes,
    robot_name_from_firmware,
    validate_ipv4_address,
)

DEFAULT_OUTPUT_DIR = ROOT / "calibrations"

__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "fetch_kinematics_bytes",
    "main",
    "robot_name_from_firmware",
]


def _parse_ipv4(value: str) -> str:
    try:
        return validate_ipv4_address(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot_ip", type=_parse_ipv4, help="xArm controller IPv4 address")
    parser.add_argument(
        "kinematics_suffix",
        nargs="?",
        default=None,
        help="filename suffix (default: last 6 chars of SN)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="output directory (default: calibrations/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="export even when SN model code indicates nominal kinematics",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.output_dir).resolve() if args.output_dir else DEFAULT_OUTPUT_DIR
    return export_kinematics_calibration(
        args.robot_ip,
        out_dir,
        suffix=args.kinematics_suffix,
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
