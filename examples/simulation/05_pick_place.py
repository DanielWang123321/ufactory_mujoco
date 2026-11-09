#!/usr/bin/env python3
"""Example entry point for :mod:`ufactory_mujoco.tasks.pick_place`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ufactory_mujoco.tasks import pick_place as _task

if __name__ == "__main__":
    _task.main()
else:
    # Importing this example yields the packaged task module itself.
    sys.modules[__name__] = _task
