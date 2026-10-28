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
import ipaddress
from pathlib import Path
import socket
import struct
import sys

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ufactory_mujoco.calibration import (  # noqa: E402
    get_robot_sn,
    has_per_unit_kinematics_calibration,
    kinematics_suffix_from_sn,
    parse_sn_model_code,
)

DEFAULT_OUTPUT_DIR = ROOT / "calibrations"


def _parse_ipv4(value: str) -> str:
    try:
        addr = ipaddress.ip_address(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid IPv4 address {value!r}; use dotted-decimal form, e.g. 192.168.1.60"
        ) from None
    if addr.version != 4:
        raise argparse.ArgumentTypeError(f"IPv6 is not supported by xArm controller: {value!r}")
    return str(addr)


def robot_name_from_firmware(dof: int, robot_type: int) -> str:
    """Resolve canonical robot name from firmware kinematics header bytes."""
    if dof == 6 and robot_type == 9:
        return "lite6"
    if dof == 6 and robot_type == 12:
        return "uf850"
    if dof in (5, 6, 7):
        return f"xarm{dof}"
    raise ValueError(f"unsupported firmware robot descriptor: dof={dof}, type={robot_type}")


def fetch_kinematics_bytes(robot_ip: str, timeout: float = 10.0) -> bytes:
    """Read 179-byte factory kinematics register block from TCP port 502."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect((robot_ip, 502))
        sock.sendall(bytes([0x00, 0x01, 0x00, 0x02, 0x00, 0x01, 0x08]))
        return sock.recv(179)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot_ip", type=_parse_ipv4, help="xArm controller IPv4 address")
    parser.add_argument("kinematics_suffix", nargs="?", default=None,
                        help="filename suffix (default: last 6 chars of SN)")
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="output directory (default: calibrations/)")
    parser.add_argument("--force", action="store_true",
                        help="export even when SN model code indicates nominal kinematics")
    args = parser.parse_args(argv)

    from xarm.wrapper import XArmAPI

    arm = None
    try:
        arm = XArmAPI(args.robot_ip, is_radian=True)
        if not getattr(arm, "connected", False):
            print(f"[Failed] cannot connect to {args.robot_ip}", file=sys.stderr)
            return 1
        sn = get_robot_sn(arm)
    except Exception as exc:
        print(f"[Failed] cannot connect to {args.robot_ip}: {exc}", file=sys.stderr)
        return 1
    finally:
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass

    suffix = args.kinematics_suffix or kinematics_suffix_from_sn(sn)
    print(f"serial_number    : {sn}")
    print(f"kinematics_suffix: {suffix}")

    try:
        recv_data = fetch_kinematics_bytes(args.robot_ip)
    except OSError as exc:
        print(f"[Failed] cannot read kinematics from {args.robot_ip}:502: {exc}", file=sys.stderr)
        return 1
    if len(recv_data) != 179 or not recv_data[8]:
        valid = 0 if len(recv_data) < 9 else recv_data[8]
        print(f"[Failed] recv_len={len(recv_data)}, valid={valid}", file=sys.stderr)
        return 1

    robot_dof = int(recv_data[9])
    robot_type = int(recv_data[10])
    robot_name = robot_name_from_firmware(robot_dof, robot_type)
    print(f"robot_name       : {robot_name}")

    if not args.force and not has_per_unit_kinematics_calibration(sn, robot_name):
        code = parse_sn_model_code(sn)
        print(
            f"[Skipped] SN model code {code} indicates no per-unit compensation. "
            "Pass --force to export anyway.",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.output_dir).resolve() if args.output_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{robot_name}_kinematics_{suffix}.yaml"

    params = struct.unpack("<42f", recv_data[11:])
    joints: dict[str, dict[str, float]] = {}
    for i in range(robot_dof):
        joint_param: dict[str, float] = {}
        for axis_idx, axis in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
            value = float(params[i * 6 + axis_idx])
            if axis in ("x", "y", "z") and abs(value) > 1.0:
                value /= 1000.0
            joint_param[axis] = value
        joints[f"joint{i + 1}"] = joint_param

    payload = {
        "schema_version": 1,
        "robot_key": f"{robot_name}_1305" if robot_name.startswith("xarm") else robot_name,
        "serial_number": sn,
        "units": {"position": "m", "angle": "rad"},
        "joints": joints,
    }
    with out_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"[Success] saved to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

