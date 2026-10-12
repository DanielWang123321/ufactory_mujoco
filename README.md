# ufactory_mujoco

[English](README.md) | [中文](README_zh.md)

MuJoCo simulation suite for UFACTORY robots — featuring high-fidelity visual assets, joint control, Cartesian inverse kinematics (IK), and interactive visualization.

## Features

- **High-Fidelity Visuals**: Smooth-normal, multi-material OBJ visual meshes (PBR white shell, silver metal joints, black accents) with dedicated collision STLs.
- **Full Asset Ecosystem**: High-precision assets for all UFACTORY robot arms (xArm 5, xArm 6, xArm 7, UF850, Lite6) and accessories (Gripper G2, Bio Gripper G2, Lite6 Gripper, Lite6 Vacuum Gripper) converted from [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis).
- **Control & Kinematics**: Quintic polynomial joint interpolation, gravity-compensated feedforward, and damped least-squares (DLS) Cartesian IK.
- **Robust Viewer**: Passive viewer with real-time sync, graceful exit handling, and Linux Wayland/X11 compatibility.

## Quick Start

### Installation

```bash
conda create -n py314 python=3.14 -y
conda activate py314
pip install -r requirements.txt
```

### Examples

> 📖 **Visual Command Guide**: For complete commands categorized by robot model (Lite 6, xArm 5/6/7, UF850) and components (static/movable), see **[docs/visual_command_guide.md](docs/visual_command_guide.md)**.

```bash
# Interactive viewer: load model and inspect (supports all arms, grippers, and combos)
python examples/01_load_and_view.py
python examples/01_load_and_view.py --model lite6 --pose ready
python examples/01_load_and_view.py --model xarm6_with_gripper --pose ready

# Joint-space control (supports all arms via --model)
python examples/02_joint_control.py --model lite6
python examples/02_joint_control.py --model xarm6
python examples/02_joint_control.py --model xarm6 --speed 120

# Cartesian IK: line + arc trajectory
python examples/03_cartesian_ik.py

# Gripper motion control (Gripper G2, Bio Gripper G2, Lite 6 Gripper)
python examples/04_gripper_control.py --model lite6_with_gripper
python examples/04_gripper_control.py --model xarm6_with_gripper
python examples/04_gripper_control.py --model bio_gripper_g2
```

Bare `xarm6` defaults to the 20 `ufactory_genesis` dynamics validation poses (two round trips), with a peak target speed of 90 degrees/s. Other models retain their demo presets. Adjust `--speed`, or use `--motion demo --duration 2.5` for the original example. Reusing poses does not establish MuJoCo torque validation.

The joint example disables shadows and reflections by default to reduce GPU drawing cost; `--render-quality full` restores them. `--sync-hz 30` controls viewer state updates, while its drawing thread runs independently. The physics timestep stays at 2 ms. `--headless` disables the viewer entirely; MuJoCo physics runs on the CPU.

### Running Tests

```bash
# Run unit tests and asset validation in py314
python -m unittest discover -s tests -v

# Run GLB conversion regression in py313 (requires genesis / trimesh)
/opt/conda_envs/py313/bin/python -m unittest discover -s tests -p 'test_convert_glb_to_obj.py' -v
```

## Project Structure

```
models/                 Robot and accessory models & meshes
  xarm6/                xArm 6 model (scene.xml, xarm6.xml, visual OBJs, collision STLs)
  xarm5/, xarm7/, ...   Assets for xArm 5/7, UF850, Lite6, and grippers (G2, Bio, Lite6)
src/                    Core library
  sim_robot.py          SimRobot: physics stepping, joint/TCP control, viewer sync
  robot_interface.py    Abstract base class (sim / real robot interface)
  ik.py                 Damped least-squares (DLS) Cartesian IK solver
  viewer.py             GLFW/X11 passive viewer lifecycle wrapper
examples/               Runnable examples (load_and_view, joint_control, cartesian_ik, gripper_control)
scripts/                Asset and model tooling
  convert_glb_to_obj.py Batch GLB -> OBJ converter with winding repair & Draco support
  build_all_mjcf.py     Generate standalone MJCF models and scene XMLs for all targets
  build_model.py        URDF to MJCF draft generator
tests/                  Unit tests (viewer lifecycle, asset validation, converter regression)
```

## Model Notes

- Kinematics and inertial tensors are preserved from official UFACTORY definitions ([xarm_ros](https://github.com/xArm-Developer/xarm_ros) / [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis)).
- Actuator parameters follow [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) conventions.
- Visual meshes (`group=2`, `contype=0 conaffinity=0`) and collision meshes (`group=3`) are strictly decoupled.

## License

MIT
