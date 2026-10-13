#!/usr/bin/env python3
"""Example 05: cube pick-and-place with xArm 6 + Gripper G2.

Replicates the ufactory_genesis pick_place example: same scene geometry,
task sequence, and success criteria on the MuJoCo backend.

Scene (models/xarm6_with_gripper/scene_pick_place.xml):
    30 mm / 17 g red cube at (0.30, 0), visual-only target marker at (0.30, 0.30).
    The arm base sits at the world origin, so base-frame task coordinates from
    the reference apply verbatim.

Task:
    gripper points straight down (quat wxyz (0,1,0,0)) for the whole sequence:
    home above cube -> pre-grasp (+0.10 m) -> descend to grasp height ->
    close to a 22 mm gap (over-close below the 30 mm cube, held by friction)
    with 2.0 s close + 0.5 s hold -> lift -> transit -> descend -> open ->
    retreat -> return home. Cartesian segments respect 0.15 m/s, 0.8 m/s^2.

Success criteria (ufactory_genesis pick_place):
    place_error_mm = ||cube final pos - target||  < 50 mm
    home_drift_mm  = ||TCP final pos - home||      < 10 mm

Usage:
    python examples/05_pick_place.py                # GUI viewer
    python examples/05_pick_place.py --headless     # metrics only
    python examples/05_pick_place.py --cycles 2     # repeat the task
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from src import SimRobot  # noqa: E402

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"

GRIP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # 180 deg about X, TCP z-axis points down

# Task geometry replicating ufactory_genesis pick_place (flange frame).
CUBE_HALF = 0.015
FLANGE_GRASP_Z = 0.1871      # 0.010 clearance + 0.015 push-down + 0.061 pad + 0.1011 flange->finger center
FLANGE_PRE_GRASP_Z = 0.2871  # grasp + 0.10 approach
FLANGE_LIFT_Z = 0.30         # lift / transit / home height
TARGET_XY = (0.30, 0.30)

GRASP_GAP_M = 0.022    # over-close target finger gap (< cube 0.03 m, friction hold)
GRIPPER_SPAN_M = 0.084  # G2 finger gap at normalized 0.0 (fully open)

CART_V = 0.15   # m/s, matches the reference Cartesian velocity limit
CART_A = 0.8    # m/s^2


def resolve_scene(model_arg: str) -> Path:
    """Resolve the pick-and-place scene for xarm6_with_gripper (or a custom XML path)."""
    p = Path(model_arg)
    if p.is_file():
        return p
    if model_arg == "xarm6_with_gripper":
        scene = MODELS_ROOT / model_arg / "scene_pick_place.xml"
        if scene.is_file():
            return scene
    available = [d.name for d in MODELS_ROOT.iterdir() if (d / "scene_pick_place.xml").is_file()]
    raise FileNotFoundError(
        f"Pick-and-place needs the 'xarm6_with_gripper' task scene "
        f"(models/xarm6_with_gripper/scene_pick_place.xml) or a direct XML path; "
        f"got '{model_arg}'. Models with a pick-place scene: {available}"
    )


def _cart_duration(dist: float) -> float:
    """Segment duration matching the reference trapezoidal profile (0.15 m/s, 0.8 m/s^2)."""
    return dist / CART_V + CART_V / CART_A + 0.1


def tcp_task_heights(robot: SimRobot) -> dict[str, float]:
    """Convert reference flange-frame heights to TCP-frame heights.

    Measures the flange->TCP z offset with the gripper pointing down, then
    shifts the reference heights by that offset. Guards the gripper-down
    assumption on the measured site orientation. The IK is seeded through a
    high waypoint because the crouched initial keyframe is a poor IK seed for
    low gripper-down targets.

    Returns (heights, q_home): q_home is the converged full-qpos IK solution
    at the lift pose, reused as the joint-space deployment target (the initial
    keyframe TCP orientation is far from gripper-down, so a Cartesian deploy
    would force a fast reorientation the PD tracker cannot follow).
    """
    q_seed, ok = robot.solve_tcp([0.30, 0.0, FLANGE_LIFT_Z + 0.15], GRIP_DOWN_QUAT)
    if not ok:
        raise RuntimeError("IK failed while reaching the seed pose")
    q, ok = robot.solve_tcp([0.30, 0.0, FLANGE_LIFT_Z], GRIP_DOWN_QUAT, q_init=q_seed)
    if not ok:
        raise RuntimeError("IK failed while measuring the flange->TCP offset")
    robot.eval_data.qpos[:] = q
    mujoco.mj_forward(robot.model, robot.eval_data)
    site_id = robot._site
    z_axis = robot.eval_data.site_xmat[site_id].reshape(3, 3)[:, 2]
    if float(z_axis @ np.array([0.0, 0.0, -1.0])) < 0.999:
        raise RuntimeError(f"TCP is not gripper-down (site z-axis {z_axis}); quat convention mismatch")
    tcp_z = float(robot.eval_data.site_xpos[site_id][2])
    link6_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "link6")
    flange_z = float(robot.eval_data.body(link6_id).xpos[2])
    off = tcp_z - flange_z  # ~ -0.172 with Gripper G2 mounted
    heights = {
        "grasp_z": FLANGE_GRASP_Z + off,
        "pre_grasp_z": FLANGE_PRE_GRASP_Z + off,
        "lift_z": FLANGE_LIFT_Z + off,
    }
    # Deployment config: the same branch re-solved at the actual TCP lift height
    # (q_meas holds the site at FLANGE_LIFT_Z, 0.17 m higher; deploying there
    # would force an unnecessary descent before the task even starts).
    q_home, ok = robot.solve_tcp([0.30, 0.0, heights["lift_z"]], GRIP_DOWN_QUAT, q_init=q)
    if not ok:
        raise RuntimeError("IK failed while solving the deployment pose")
    return heights, q_home


def run_pick_place(robot: SimRobot, cycles: int = 1) -> list[dict]:
    """Run the full pick-and-place sequence; returns per-cycle metric dicts."""
    model = robot.model
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    if cube_id < 0:
        raise ValueError("Scene has no 'cube' body; load scene_pick_place.xml")
    if not robot.has_gripper:
        raise ValueError("Robot has no gripper actuator")

    h, q_home = tcp_task_heights(robot)
    grasp_close = 1.0 - GRASP_GAP_M / GRIPPER_SPAN_M  # ~0.738 normalized
    target3 = np.array([TARGET_XY[0], TARGET_XY[1], CUBE_HALF])

    def viewer_closed() -> bool:
        return robot._viewer is not None and not robot._viewer.is_running()

    results = []
    for cycle in range(1, cycles + 1):
        if viewer_closed():
            break
        cube_xy = robot.data.body(cube_id).xpos[:2].copy()  # pick wherever the cube rests
        print(f"\n--- Cycle {cycle}/{cycles}: cube at ({cube_xy[0]:.3f}, {cube_xy[1]:.3f}) ---")

        home = np.array([cube_xy[0], cube_xy[1], h["lift_z"]])
        pre_grasp = np.array([cube_xy[0], cube_xy[1], h["pre_grasp_z"]])
        grasp = np.array([cube_xy[0], cube_xy[1], h["grasp_z"]])
        above_target = np.array([TARGET_XY[0], TARGET_XY[1], h["lift_z"]])
        place = np.array([TARGET_XY[0], TARGET_XY[1], h["grasp_z"]])

        print("  Open gripper and deploy above cube ...")
        robot.open_gripper(wait=True)
        # Joint-space deployment to the lift pose (like the reference MoveJ
        # home seed): the initial keyframe TCP orientation is far from
        # gripper-down, and a Cartesian deploy would reorient too fast to track.
        robot.move_joints(q_home[:robot._n_arm_qpos], duration=3.0)
        deploy_xy = np.array([0.30, 0.0, h["lift_z"]])
        robot.move_tcp(home, GRIP_DOWN_QUAT,
                       duration=_cart_duration(np.linalg.norm(home - deploy_xy)))
        if viewer_closed():
            break
        print("  Approach pre-grasp and descend ...")
        robot.move_tcp(pre_grasp, GRIP_DOWN_QUAT, duration=_cart_duration(h["lift_z"] - h["pre_grasp_z"]))
        robot.move_tcp(grasp, GRIP_DOWN_QUAT, duration=_cart_duration(h["pre_grasp_z"] - h["grasp_z"]))
        if viewer_closed():
            break

        print(f"  Close gripper to gap {GRASP_GAP_M * 1000:.0f} mm (over-close) ...")
        robot.set_gripper(grasp_close)
        robot.settle(2.0)  # close motion
        robot.settle(0.5)  # hold

        print("  Lift and transit to target ...")
        robot.move_tcp(np.array([cube_xy[0], cube_xy[1], h["lift_z"]]), GRIP_DOWN_QUAT,
                       duration=_cart_duration(h["lift_z"] - h["grasp_z"]))
        cube_lift_z = float(robot.data.body(cube_id).xpos[2])
        cube_lifted = cube_lift_z > CUBE_HALF + 0.05
        robot.move_tcp(above_target, GRIP_DOWN_QUAT,
                       duration=_cart_duration(np.linalg.norm(above_target - home)))
        if viewer_closed():
            break

        print("  Descend and release ...")
        robot.move_tcp(place, GRIP_DOWN_QUAT, duration=_cart_duration(h["lift_z"] - h["grasp_z"]))
        robot.open_gripper(wait=True)
        robot.settle(1.0)  # let the cube settle on the floor

        print("  Retreat and return home ...")
        robot.move_tcp(above_target, GRIP_DOWN_QUAT, duration=_cart_duration(h["lift_z"] - h["grasp_z"]))
        robot.move_tcp(home, GRIP_DOWN_QUAT,
                       duration=_cart_duration(np.linalg.norm(home - above_target)))
        robot.settle(0.5)

        cube_final = robot.data.body(cube_id).xpos.copy()
        tcp_final, _ = robot.get_tcp_pose()
        place_error_mm = float(np.linalg.norm(cube_final - target3) * 1000.0)
        home_drift_mm = float(np.linalg.norm(tcp_final - home) * 1000.0)
        result = {
            "cycle": cycle,
            "cube_lifted": bool(cube_lifted),
            "place_error_mm": place_error_mm,
            "home_drift_mm": home_drift_mm,
        }
        results.append(result)
        print(f"  cube_lifted={result['cube_lifted']} (z={cube_lift_z:.3f})  "
              f"place_error={place_error_mm:.1f} mm  home_drift={home_drift_mm:.1f} mm")

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Cube pick-and-place with xArm 6 + Gripper G2.")
    ap.add_argument("--model", default="xarm6_with_gripper",
                    help="model name with a pick-place scene, or path to a scene XML")
    ap.add_argument("--headless", action="store_true", help="no viewer, print metrics")
    ap.add_argument("--cycles", type=int, default=1, help="number of pick-and-place cycles (default: 1)")
    args = ap.parse_args()

    scene_path = resolve_scene(args.model)
    with SimRobot(scene_path, viewer=not args.headless) as robot:
        results = run_pick_place(robot, cycles=args.cycles)

    ok = bool(results) and all(
        r["cube_lifted"] and r["place_error_mm"] < 50.0 and r["home_drift_mm"] < 10.0
        for r in results
    )
    print("\nPick-and-place " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
