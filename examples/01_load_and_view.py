#!/usr/bin/env python3
"""Example 01: load the xArm 6 scene and open the interactive viewer.

Usage:
    python examples/01_load_and_view.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.viewer import launch  # noqa: E402

import mujoco  # noqa: E402

MODEL = Path(__file__).resolve().parent.parent / "models" / "xarm6" / "scene.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    print(f"model: {MODEL.name}  nq={model.nq} nu={model.nu} nbody={model.nbody}")
    print("joints:", [f"{model.joint(i).name} [{lo:+.3f},{hi:+.3f}]"
                      for i, (lo, hi) in zip(range(model.njnt), model.jnt_range)])
    print("keyframes:", [model.key(i).name for i in range(model.nkey)])
    print("TCP home:", data.site(model.site("attachment_site").id).xpos)
    print("viewer controls: mouse orbit/pan, Tab panel, Space pause. Close window to exit.")

    launch(model, data)


if __name__ == "__main__":
    main()
