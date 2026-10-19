#!/usr/bin/env python3
"""Example 01: load the xArm 6 scene and open the interactive viewer.

Usage:
    python examples/simulation/01_load_and_view.py
    python examples/simulation/01_load_and_view.py --pose ready
    python examples/simulation/01_load_and_view.py --model lite6 --headless
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.model_paths import resolve_model_path  # noqa: E402
from src.viewer import launch  # noqa: E402

import mujoco  # noqa: E402



def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a robot scene in MuJoCo.")
    parser.add_argument(
        "--model",
        default="xarm6",
        help="Robot name (e.g. 'xarm6') or path to a scene/model XML (default: xarm6)",
    )
    parser.add_argument(
        "--pose",
        choices=["home", "ready"],
        default="home",
        help="initial keyframe (default: home)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="load the model and print its summary without opening the viewer",
    )
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.pose)
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    print(f"model: {model_path.name} ({args.model})  nq={model.nq} nu={model.nu} nbody={model.nbody}")
    print("joints:", [f"{model.joint(i).name} [{lo:+.3f},{hi:+.3f}]"
                      for i, (lo, hi) in zip(range(model.njnt), model.jnt_range)])
    print("keyframes:", [model.key(i).name for i in range(model.nkey)])
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if site_id >= 0:
        print(f"TCP {args.pose}:", data.site(site_id).xpos)
    if args.headless:
        return
    print("viewer controls: mouse orbit/pan, Tab panel, Space pause. Close window to exit.")

    launch(model, data)


if __name__ == "__main__":
    main()
