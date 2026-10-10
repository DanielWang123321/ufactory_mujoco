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

```bash
# Interactive viewer: load model and inspect (default: xarm6, supports all arms, grippers, and combos)
python examples/01_load_and_view.py
python examples/01_load_and_view.py --model xarm7 --pose ready
python examples/01_load_and_view.py --model uf850
python examples/01_load_and_view.py --model lite6
python examples/01_load_and_view.py --model gripper_g2
python examples/01_load_and_view.py --model xarm6_with_gripper --pose ready

# Joint-space control (quintic interpolation)
python examples/02_joint_control.py            # GUI viewer
python examples/02_joint_control.py --headless  # print tracking metrics

# Cartesian IK: line + arc trajectory
python examples/03_cartesian_ik.py             # GUI viewer
python examples/03_cartesian_ik.py --headless   # print tracking metrics
```

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
examples/               Runnable examples (load_and_view, joint_control, cartesian_ik)
scripts/                Asset and model tooling
  convert_glb_to_obj.py Batch GLB -> OBJ converter with winding repair & Draco support
  build_all_mjcf.py     Generate standalone MJCF models and scene XMLs for all targets
  build_model.py        URDF to MJCF draft generator
tests/                  Unit tests (viewer lifecycle, asset validation, converter regression)
```

## Asset Conversion Tooling

To re-convert or update assets from [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis):

```bash
# Convert all robot arms and accessories
/opt/conda_envs/py313/bin/python scripts/convert_glb_to_obj.py --target all

# Or convert a specific robot / gripper
/opt/conda_envs/py313/bin/python scripts/convert_glb_to_obj.py --target xarm7
/opt/conda_envs/py313/bin/python scripts/convert_glb_to_obj.py --target gripper_g2
```

The converter automatically decodes Draco-compressed GLB meshes, aligns coordinate frames (glTF Y-up to MuJoCo Z-up), classifies materials, repairs inconsistent face winding (`orient_mesh_outward`) to prevent back-face culling artifacts, and preserves vertex normals and hard edges.

## Model Notes

- Kinematics and inertial tensors are preserved from official UFACTORY definitions ([xarm_ros](https://github.com/xArm-Developer/xarm_ros) / [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis)).
- Actuator parameters follow [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) conventions.
- Visual meshes (`group=2`, `contype=0 conaffinity=0`) and collision meshes (`group=3`) are strictly decoupled.

## License

MIT
