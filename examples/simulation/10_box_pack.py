#!/usr/bin/env python3
"""Compatibility entry point for :mod:`ufactory_mujoco.tasks.box_pack`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ufactory_mujoco.tasks import box_pack as _task

if __name__ == "__main__":
    _task.main()
else:
    # Keep legacy imports (including private helpers) bound to the implementation.
    sys.modules[__name__] = _task
