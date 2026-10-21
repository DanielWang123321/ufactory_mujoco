"""Console entry point for the three packaged simulation tasks."""

import argparse
import importlib
import sys

_COMMANDS = {"pick-place": "pick_place", "box-pack": "box_pack", "dynamics": "dynamics"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="ufactory-mujoco", description="UFACTORY MuJoCo simulation tasks")
    parser.add_argument("command", choices=tuple(_COMMANDS), help="task to run; use COMMAND --help")
    args = parser.parse_args(argv[:1])
    task = importlib.import_module(f"ufactory_mujoco.tasks.{_COMMANDS[args.command]}")
    return task.main(argv[1:])
