# ufactory_mujoco

xArm 6 MuJoCo simulation examples — joint control, Cartesian IK, and passive viewer.

## Quick Start

```bash
conda create -n py314 python=3.14 -y
conda activate py314
pip install -r requirements.txt
```

### Examples

```bash
# Interactive viewer: load model and inspect
python examples/01_load_and_view.py

# Joint-space control (quintic interpolation)
python examples/02_joint_control.py            # GUI
python examples/02_joint_control.py --headless  # print metrics

# Cartesian IK: line + arc trajectory
python examples/03_cartesian_ik.py             # GUI
python examples/03_cartesian_ik.py --headless   # print metrics
```

## Project Structure

```
models/xarm6/       MJCF model (scene.xml, xarm6.xml, STL meshes)
src/                Core library
  sim_robot.py        SimRobot: physics stepping, joint/TCP control, viewer sync
  robot_interface.py  Abstract base class (sim / real robot)
  ik.py               Damped least-squares IK
  viewer.py           GLFW/X11 passive viewer wrapper
examples/           Usage examples
scripts/            Model build tooling
tests/              Unit tests
```

## Model

- Source: [UFACTORY xarm_ros](https://github.com/xArm-Developer/xarm_ros) `xarm_description` (URDF + STL)
- Converted to MJCF via `scripts/build_model.py`
- Actuator gains follow [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) conventions

## License

MIT
