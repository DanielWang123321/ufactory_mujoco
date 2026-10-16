# ufactory_mujoco

[English](README.md) | [中文](README_zh.md)

MuJoCo simulation and reinforcement learning suite for UFACTORY robots — featuring high-fidelity visual assets, kinematics/dynamics toolkits, motion planning, and Gymnasium RL environments.

## Features

- **High-Fidelity Visuals**: Smooth-normal, multi-material OBJ visual meshes (PBR white shell, silver metal joints, black accents) paired with dedicated collision STLs.
- **Full Asset Ecosystem**: All UFACTORY robot arms (`lite6`, `xarm5`, `xarm6`, `xarm7`, `uf850`) and grippers (`gripper_g2`, `bio_gripper_g2`, `lite6_gripper`, `lite6_vacuum_gripper`) converted from [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis).
- **Kinematics & Motion Planning**: Per-site FK, analytical Jacobian, damped least-squares (DLS) IK (`src/kinematics.py`, `src/ik.py`), PTP quintic/C2 spline trajectories (`src/trajectory.py`), and RRT-Connect obstacle avoidance (`src/collision.py`, `src/path_planning.py`).
- **Rigid-Body Dynamics**: Mass matrix $M(q)$, gravity/Coriolis terms, inverse/forward dynamics, and TCP payload injection (`src/dynamics.py`), cross-validated against real xArm 6 hardware.
- **Reinforcement Learning**: Gymnasium TCP-reaching environment (`src/rl/reach_env.py`) with calibrated workspace profiles and PPO policies achieving $\ge 96\%$ success at the $< 5\text{ mm}$ threshold across all five arms.
- **Robust Viewer**: Passive viewer with real-time clock sync, graceful exit handling, and Linux Wayland/X11 compatibility (`src/viewer.py`).

## Quick Start

### 1. Installation

```bash
conda create -n py314 python=3.14 -y
conda activate py314
pip install -r requirements.txt

# Optional: RL training & PPO rollout dependencies (gymnasium + stable-baselines3)
pip install -r requirements-rl.txt
```

### 2. Documentation & Examples

| Category | Documentation | Description |
| :--- | :--- | :--- |
| **Simulation (`01`–`09`)** | [examples/simulation/README.md](examples/simulation/README.md) ([中文](examples/simulation/README_zh.md)) | Model viewer, joint/Cartesian control, grippers, pick-and-place, trajectory/RRT planning, kinematics & dynamics |
| **RL Reach (`01_reach`)** | [examples/rl/reach/README.md](examples/rl/reach/README.md) ([中文](examples/rl/reach/README_zh.md)) | Multi-robot TCP position reaching environment, PPO policy evaluation & training |
| **Command Matrix** | [docs/visual_command_guide.md](docs/visual_command_guide.md) | Model-by-model launch commands for all arms, grippers, and combinations |

```bash
# Interactive model viewer
python examples/simulation/01_load_and_view.py --model xarm6_with_gripper --pose ready

# Joint control / Cube pick-and-place / RRT-Connect obstacle avoidance
python examples/simulation/02_joint_control.py --model xarm6
python examples/simulation/05_pick_place.py
python examples/simulation/09_obstacle_avoidance.py

# RL TCP reaching (PPO policy rollout; use --random if stable-baselines3 is not installed)
python examples/rl/reach/01_reach.py --model xarm6
```

> Pass `--headless` to any motion or RL example (`02`–`09`, `01_reach.py`) for offscreen execution and metric verification.

### 3. Running Tests

```bash
# Unit tests and full MJCF asset validation
python -m unittest discover -s tests -v
```

*(Tests requiring `trimesh` or `gymnasium`/`stable-baselines3` skip automatically when those optional packages are absent.)*

## Project Structure

```text
models/                 Robot and accessory MJCF models & visual/collision meshes
  xarm6/, lite6/, ...   xArm 5/6/7, UF850, Lite6, and grippers (standalone & combined scenes)
src/                    Core simulation, control, planning, and RL library
  sim_robot.py          SimRobot: physics stepping, joint/TCP control, viewer sync
  robot_interface.py    Abstract base class aligning sim and real robot interfaces
  ik.py                 Damped least-squares (DLS) Cartesian IK solver
  kinematics.py         FK, TCP Jacobian, velocity-level DLS IK, manipulability metrics
  trajectory.py         JointTrajectory, PTP quintic/LSPB, C2 spline, Cartesian SLERP line
  dynamics.py           Mass matrix, gravity/Coriolis, inverse/forward dynamics, payload injection
  collision.py          CollisionChecker: clearance and contact queries
  path_planning.py      RRT-Connect + path shortcutting for collision-free joint trajectories
  rl/reach_env.py       Gymnasium TCP-reaching environment (all 5 arms)
  model_paths.py        Model name -> MJCF XML path resolution
  viewer.py             GLFW/X11 passive viewer lifecycle wrapper
examples/
  simulation/           Simulation examples 01–09 (see examples/simulation/README.md)
  rl/reach/             RL TCP reaching example (see examples/rl/reach/README.md)
scripts/                Asset conversion, workspace calibration, dynamics replay & RL training
tests/                  Unit and regression test suite
```

## Model Notes

- Kinematics and inertial tensors are preserved from official UFACTORY definitions ([xarm_ros](https://github.com/xArm-Developer/xarm_ros) / [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis)).
- Actuator parameters follow [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) conventions.
- Visual meshes (`group=2`, `contype=0 conaffinity=0`) and collision meshes (`group=3`) are strictly decoupled.

## License

MIT
