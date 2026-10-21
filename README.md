# ufactory_mujoco

[English](README.md) | [中文](README_zh.md)

MuJoCo simulation and reinforcement learning suite for UFACTORY robots — featuring high-fidelity visual assets, kinematics/dynamics toolkits, motion planning, and Gymnasium RL environments.

## Features

- **High-Fidelity Visuals**: Smooth-normal, multi-material OBJ visual meshes (PBR white shell, silver metal joints, black accents) paired with dedicated collision STLs.
- **Full Asset Ecosystem**: All UFACTORY robot arms (`lite6`, `xarm5`, `xarm6`, `xarm7`, `uf850`) and grippers (`gripper_g2`, `bio_gripper_g2`, `lite6_gripper`, `lite6_vacuum_gripper`) converted from [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis).
- **Kinematics & Motion Planning**: Per-site FK, analytical Jacobian, damped least-squares (DLS) IK (`ufactory_mujoco/kinematics.py`, `ufactory_mujoco/ik.py`), PTP quintic/C2 spline trajectories (`ufactory_mujoco/trajectory.py`), and RRT-Connect obstacle avoidance (`ufactory_mujoco/collision.py`, `ufactory_mujoco/path_planning.py`).
- **Rigid-Body Dynamics**: Mass matrix $M(q)$, gravity/Coriolis terms, inverse/forward dynamics, and TCP payload injection (`ufactory_mujoco/dynamics.py`), cross-validated against real xArm 6 hardware.
- **Reinforcement Learning**: Gymnasium TCP-reaching environment (`ufactory_mujoco/rl/reach_env.py`) with calibrated workspace profiles and PPO policies achieving $\ge 96\%$ success at the $< 5\text{ mm}$ threshold across all five arms.
- **Robust Viewer**: Passive viewer with real-time clock sync, graceful exit handling, and Linux Wayland/X11 compatibility (`ufactory_mujoco/viewer.py`).

## Quick Start

### 1. Installation

Python **3.13** is recommended (3.12 also works; all core and RL dependencies ship wheels for both).

```bash
# The repo is large (~600 MB of meshes); a shallow clone is much faster
git clone --depth 1 https://github.com/DanielWang123321/ufactory_mujoco.git
cd ufactory_mujoco

python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # installs the ufactory_mujoco package + core deps

# Optional: RL training & PPO rollout (install CPU torch first, see requirements-rl.txt)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[rl]"
```

> Exact versions verified in CI: `pip install -r requirements-lock.txt`.

### macOS and Windows

- **macOS**: the interactive viewer (`launch_passive`) must run on the main thread, so start GUI examples with `mjpython` (shipped with the `mujoco` package): `mjpython examples/simulation/01_load_and_view.py`. `--headless` runs work with plain `python`.
- **Windows**: activate the venv with `.venv\Scripts\activate`; the viewer works with plain `python`. `MUJOCO_GL` and Xvfb are Linux-only concerns. Paths in examples are resolved relative to the repo, so run them from any directory.

### Headless / remote machines

- `--headless` (examples `01`–`10`, `01_reach.py`) runs physics only and needs no display or GL; do **not** set `MUJOCO_GL` for it.
- If you set `MUJOCO_GL=egl`, the system must provide `libEGL` (e.g. `apt install libegl1`), otherwise `import mujoco` fails. `MUJOCO_GL=osmesa` needs `libosmesa6`.
- To open the interactive viewer on a server without a display, use Xvfb: `xvfb-run -a python examples/simulation/01_load_and_view.py`.

### 2. Documentation & Examples

| Category | Documentation | Description |
| :--- | :--- | :--- |
| **Simulation (`01`–`10`)** | [examples/simulation/README.md](examples/simulation/README.md) ([中文](examples/simulation/README_zh.md)) | Model viewer, joint/Cartesian control, grippers, pick-and-place, box-packing, trajectory/RRT planning, kinematics & dynamics |
| **RL Reach (`01_reach`)** | [examples/rl/reach/README.md](examples/rl/reach/README.md) ([中文](examples/rl/reach/README_zh.md)) | Multi-robot TCP position reaching environment, PPO policy evaluation & training |
| **Command Matrix** | [docs/visual_command_guide.md](docs/visual_command_guide.md) | Model-by-model launch commands for all arms, grippers, and combinations |

```bash
# Interactive model viewer
python examples/simulation/01_load_and_view.py --model xarm6_with_gripper --pose ready

# Joint control / Cube pick-and-place / RRT-Connect obstacle avoidance / Box packing
python examples/simulation/02_joint_control.py --model xarm6
python examples/simulation/05_pick_place.py
python examples/simulation/09_obstacle_avoidance.py
python examples/simulation/10_box_pack.py

# RL TCP reaching (PPO policy rollout; use --random if stable-baselines3 is not installed)
python examples/rl/reach/01_reach.py --model xarm6
```

> Pass `--headless` to any motion or RL example (`01`–`10`, `01_reach.py`) for offscreen execution and metric verification.

### 3. Running Tests

```bash
# Unit tests and full MJCF asset validation
python -m unittest discover -s tests -v
```

*(Tests requiring `trimesh` or `gymnasium`/`stable-baselines3` skip automatically when those optional packages are absent.)*

## Project Structure

```text
configs/                Strict YAML task configurations (default, pick_place, box_pack)
models/                 Robot and accessory MJCF models & visual/collision meshes
  xarm6/, lite6/, ...   xArm 5/6/7, UF850, Lite6, and grippers (standalone & combined scenes)
ufactory_mujoco/        Core simulation, control, planning, and RL library
  sim_robot.py          SimRobot: physics stepping, joint/TCP control, viewer sync
  robot_interface.py    Abstract base class aligning sim and real robot interfaces
  robots.py             Typed registry for the 5 UFACTORY arms, combo paths & reach workspaces
  config.py             Strict YAML task config loader, validation & content hashing
  grippers.py           Physical gap <-> actuator mapping and bilateral finger-pad contact
  preflight.py          Static IK/collision preflight for pick-and-place and box-packing
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
  simulation/           Simulation examples 01–10 (see examples/simulation/README.md)
  rl/reach/             RL TCP reaching example (see examples/rl/reach/README.md)
scripts/                Asset conversion, workspace calibration, dynamics replay & RL training
tests/                  Unit and regression test suite
docs/internal/          Internal design/roadmap notes (not user documentation)
```

## Model Notes

- Kinematics and inertial tensors are preserved from official UFACTORY definitions ([xarm_ros](https://github.com/xArm-Developer/xarm_ros) / [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis)).
- Actuator parameters follow [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) conventions.
- Visual meshes (`group=2`, `contype=0 conaffinity=0`) and collision meshes (`group=3`) are strictly decoupled.

## License

Code: MIT (see `LICENSE`). Robot models are derived from UFACTORY assets (xarm_ros, BSD-3-Clause; ufactory_genesis, MIT) — see `NOTICE`.
