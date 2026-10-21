#!/usr/bin/env python3
"""Example 02: speed-based joint control with quintic interpolation.

xArm 6 defaults to the 20 ufactory_genesis dynamics validation poses.
Other models retain their ready -> demo -> ready presets.

Usage:
    python examples/simulation/02_joint_control.py            # with interactive viewer
    python examples/simulation/02_joint_control.py --speed 120
    python examples/simulation/02_joint_control.py --headless # no rendering, prints metrics
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from ufactory_mujoco import SimRobot  # noqa: E402
from ufactory_mujoco.model_paths import resolve_model_path  # noqa: E402

# Presets for joint demo movements: [ready, demo]
PRESETS: dict[str, tuple[list[float], list[float]]] = {
    "xarm6": (
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # ready = all-zero joints
        [0.3, -1.2, -1.0, 0.6, 0.6, 0.5],
    ),
    # Gripper G2 combo: all-zero would push the fingers into the floor, so ready is the
    # IK pose with the gripper TCP at (300, 0, 300) mm, rpy (180, 0, 0).
    "xarm6_with_gripper": (
        [0.0, -0.631529917, -0.833150486, 0.0, 1.464680403, 0.0],
        [0.3, -1.2, -1.0, 0.6, 0.6, 0.5],
    ),
    "lite6": (
        np.deg2rad([0.0, 9.9, 31.8, 0.0, 21.9, 0.0]).tolist(),  # Studio factory ready pose
        [0.3, -0.8, 0.3, 0.2, 0.5, -0.3],
    ),
    "xarm5": (
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.3, -0.7, -0.3, 0.2, 0.6],
    ),
    "xarm5_with_gripper": (
        [0.0, -0.5, 0.0, 0.0, 0.5],
        [0.3, -0.7, -0.3, 0.2, 0.6],
    ),
    "xarm7": (
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.3, -0.7, 0.2, 0.5, 0.2, 0.2, -0.3],
    ),
    "xarm7_with_gripper": (
        [0.0, -0.5, 0.0, 0.0, 0.5, 0.0, 0.0],
        [0.3, -0.7, 0.2, 0.5, 0.2, 0.2, -0.3],
    ),
    "uf850": (
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.3, -0.7, -0.3, 0.2, 0.4, -0.3],
    ),
    "uf850_with_gripper": (
        [0.0, -0.5, 0.0, 0.0, 0.5, 0.0],
        [0.3, -0.7, -0.3, 0.2, 0.4, -0.3],
    ),
}

# Copied from ufactory_genesis at be495199ae604182231c921f7a2f27a5acec9418:
# https://github.com/xArm-Developer/ufactory_genesis/blob/be495199ae604182231c921f7a2f27a5acec9418/assets/configs/dynamics_validation_pose.yaml
# poses_config.py expands each home -> endpoint -> home into 5 + 5 targets.
XARM6_DYNAMICS_ENDPOINTS_DEG = (
    (90, -90, -60, 170, 160, -179),
    (-90, -90, -60, -170, -20, 179),
)


def get_dynamics_targets() -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    home = np.zeros(6)
    targets = []
    for endpoint_deg in XARM6_DYNAMICS_ENDPOINTS_DEG:
        endpoint = np.deg2rad(endpoint_deg)
        for k in (*range(1, 6), *range(4, -1, -1)):
            targets.append((f"dynamics {len(targets):02d}", endpoint * (k / 5)))
    return home, targets


def positive_float(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def move_duration(q_start: np.ndarray, q_goal: np.ndarray, speed_deg_s: float,
                  timestep: float) -> float:
    # max(d/ds (10s^3 - 15s^4 + 6s^5)) = 1.875, so speed bounds the peak
    # commanded joint velocity, rather than the mean velocity.
    return max(timestep, 1.875 * float(np.max(np.abs(q_goal - q_start)))
               / np.deg2rad(speed_deg_s))


def get_arm_targets(robot: SimRobot, model_name: str) -> tuple[np.ndarray, np.ndarray]:
    key = model_name if model_name in PRESETS else model_name.split("_with_")[0]
    if key in PRESETS:
        ready, demo = PRESETS[key]
        return np.array(ready, dtype=float), np.array(demo, dtype=float)

    # Fallback: extract from keyframe 'ready'
    ready_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    if ready_id >= 0:
        ready = robot.model.key(ready_id).qpos[:robot._n_arm_qpos].copy()
    else:
        ready = np.zeros(robot._n_arm_qpos)

    demo = ready.copy()
    demo[0] += 0.2
    return ready, demo


def main() -> None:
    ap = argparse.ArgumentParser(description="Joint position control demo for UFACTORY arms.")
    ap.add_argument(
        "--model",
        default="xarm6",
        help="Robot arm model name (e.g. 'xarm6', 'lite6', 'xarm7', 'xarm5', 'uf850', 'xarm6_with_gripper')",
    )
    ap.add_argument("--motion", choices=("auto", "dynamics", "demo"), default="auto",
                    help="auto: dynamics for bare xarm6, demo presets for other models")
    ap.add_argument("--speed", type=positive_float, default=90.0,
                    help="peak commanded joint speed in degrees/s (default: 90)")
    ap.add_argument("--duration", type=positive_float,
                    help="fixed seconds per move, overriding --speed")
    ap.add_argument("--sync-hz", type=positive_float, default=30.0,
                    help="viewer state sync rate, not drawing FPS (default: 30)")
    ap.add_argument("--render-quality", choices=("fast", "full"), default="fast",
                    help="fast disables shadows and reflections; full restores them")
    ap.add_argument("--headless", action="store_true", help="no viewer, print metrics")
    args = ap.parse_args()

    motion = args.motion
    if motion == "auto":
        motion = "dynamics" if args.model == "xarm6" else "demo"
    if motion == "dynamics" and args.model != "xarm6":
        ap.error("dynamics motion currently supports the bare xarm6 only; use --motion demo")

    model_path = resolve_model_path(args.model)

    with SimRobot(model_path=model_path, viewer=not args.headless, viewer_hz=args.sync_hz,
                  low_power_rendering=args.render_quality == "fast") as robot:
        print(f"Loaded: {args.model} ({robot._n_arm_actuators} arm actuators)")
        if motion == "dynamics":
            ready_q, targets = get_dynamics_targets()
        else:
            ready_q, demo_q = get_arm_targets(robot, args.model)
            targets = [("demo", demo_q), ("ready", ready_q)]
        print(f"Motion: {motion} ({len(targets)} targets), "
              + (f"{args.duration:g} s/move" if args.duration is not None
                 else f"peak target speed {args.speed:g} deg/s"))
        if not args.headless:
            print(f"Rendering: {args.render_quality}, state sync {args.sync_hz:g} Hz")

        # Set initial pose to ready
        robot.data.qpos[:robot._n_arm_qpos] = ready_q
        robot.data.ctrl[:robot._n_arm_actuators] = ready_q
        mujoco.mj_forward(robot.model, robot.data)
        robot.settle(0.5)

        print("Start joints:", np.round(robot.get_joint_positions(), 4))

        worst_error = 0.0
        for target_name, q in targets:
            if robot._viewer is not None and not robot._viewer.is_running():
                print("Viewer closed; motion stopped.")
                return
            duration = args.duration if args.duration is not None else move_duration(
                robot.data.ctrl[:robot._n_arm_actuators], q, args.speed,
                robot.model.opt.timestep,
            )
            print(f"Moving to {target_name} pose ({duration:.3f} s)...")
            robot.move_joints(q, duration=duration)
            if robot._viewer is not None and not robot._viewer.is_running():
                print("Viewer closed; motion stopped.")
                return
            worst_error = max(worst_error, float(np.max(np.abs(robot.get_joint_positions() - q))))

        err = np.abs(robot.get_joint_positions() - ready_q)
        print(f"Final joints: {np.round(robot.get_joint_positions(), 4)}")
        print(f"Final |error| per joint (rad): {np.round(err, 5)}")
        print(f"Worst waypoint error: {worst_error:.5f} rad")
        ok = worst_error < 0.01
        print("PASS" if ok else "FAIL", f"(max error {worst_error:.5f} rad < 0.01)")
        if args.headless and not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
