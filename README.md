# ufactory_mujoco

[English](README.md) | [中文](README_zh.md)

MuJoCo simulation suite for UFACTORY robots — featuring high-fidelity visual assets, joint control, Cartesian inverse kinematics (IK), and interactive visualization.

## Features

- **High-Fidelity Visuals**: Smooth-normal, multi-material OBJ visual meshes (PBR white shell, silver metal joints, black accents) with dedicated collision STLs.
- **Full Asset Ecosystem**: High-precision assets for all UFACTORY robot arms (xArm 5, xArm 6, xArm 7, UF850, Lite6) and accessories (Gripper G2, Bio Gripper G2, Lite6 Gripper, Lite6 Vacuum Gripper) converted from [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis).
- **Control & Kinematics**: Quintic polynomial joint interpolation, gravity-compensated feedforward, and damped least-squares (DLS) Cartesian IK; plus per-site FK, TCP Jacobian, velocity-level IK, and manipulability/singularity metrics (`src/kinematics.py`).
- **Rigid-Body Dynamics**: Mass matrix, gravity/Coriolis terms, inverse/forward dynamics, and trajectory feedforward torques (`src/dynamics.py`), with TCP payload injection and real-robot cross-validation scripts.
- **Reinforcement Learning**: Gymnasium TCP-reaching environment covering all five arms (lite6, xarm5, xarm6, xarm7, uf850; `src/rl/reach_env.py` with calibrated per-robot workspace profiles), PPO training pipeline (`scripts/train_reach_rl.py --model`) and trained-policy rollout example.
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
python examples/simulation/01_load_and_view.py
python examples/simulation/01_load_and_view.py --model lite6 --pose ready
python examples/simulation/01_load_and_view.py --model xarm6_with_gripper --pose ready

# Joint-space control (supports all arms via --model)
python examples/simulation/02_joint_control.py --model lite6
python examples/simulation/02_joint_control.py --model xarm6
python examples/simulation/02_joint_control.py --model xarm6 --speed 120

# Cartesian IK: line + arc trajectory
python examples/simulation/03_cartesian_ik.py

# Gripper motion control (Gripper G2, Bio Gripper G2, Lite 6 Gripper)
python examples/simulation/04_gripper_control.py --model lite6_with_gripper
python examples/simulation/04_gripper_control.py --model xarm6_with_gripper
python examples/simulation/04_gripper_control.py --model bio_gripper_g2

# Cube pick-and-place (xArm 6 + Gripper G2, replicates ufactory_genesis pick_place)
python examples/simulation/05_pick_place.py
python examples/simulation/05_pick_place.py --headless     # prints place_error_mm / home_drift_mm, exits non-zero on failure

# Trajectory planning: PTP quintic, multi-waypoint spline, Cartesian line, dynamics feedforward
python examples/simulation/06_trajectory_planning.py
python examples/simulation/06_trajectory_planning.py --headless

# Kinematics toolkit: FK/Jacobian cross-checks, velocity-level IK, singularity metrics
python examples/simulation/07_kinematics.py
python examples/simulation/07_kinematics.py --headless

# Dynamics toolkit: mass matrix, gravity/Coriolis, ID/FD round trip, payload, feedforward
python examples/simulation/08_dynamics.py
python examples/simulation/08_dynamics.py --headless

# Obstacle avoidance: RRT-Connect around pillar + board (xArm 6 task scene)
python examples/simulation/09_obstacle_avoidance.py
python examples/simulation/09_obstacle_avoidance.py --headless

# RL TCP reaching: roll out a trained PPO policy (train with scripts/train_reach_rl.py)
python examples/rl/01_reach.py                     # xArm 6 by default
python examples/rl/01_reach.py --model lite6       # lite6 / xarm5 / xarm7 / uf850
python examples/rl/01_reach.py --headless     # prints success_rate / final TCP error, exits non-zero on failure
python examples/rl/01_reach.py --random       # untrained baseline (works without stable-baselines3)
```

> **Layout since 0.0.6**: examples are grouped into `examples/simulation/` (01–06, shown above) and `examples/rl/` (RL examples, numbered per category). Migration: `examples/0X_*.py` → `examples/simulation/0X_*.py` (01–06); `examples/07_rl_reach.py` → `examples/rl/01_reach.py`.

Bare `xarm6` defaults to the 20 `ufactory_genesis` dynamics validation poses (two round trips), with a peak target speed of 90 degrees/s. Other models retain their demo presets. Adjust `--speed`, or use `--motion demo --duration 2.5` for the original example. Reusing poses does not establish MuJoCo torque validation.

The joint example disables shadows and reflections by default to reduce GPU drawing cost; `--render-quality full` restores them. `--sync-hz 30` controls viewer state updates, while its drawing thread runs independently. The physics timestep stays at 2 ms. `--headless` disables the viewer entirely; MuJoCo physics runs on the CPU.

### Running Tests

```bash
# Run unit tests and asset validation in py314
python -m unittest discover -s tests -v

# Run GLB conversion regression in py313 (requires genesis / trimesh)
/opt/conda_envs/py313/bin/python -m unittest discover -s tests -p 'test_convert_glb_to_obj.py' -v

# Run the full suite including RL training tests in py313 (gymnasium + stable-baselines3)
/opt/conda_envs/py313/bin/python -m unittest discover -s tests -v
```

RL tests skip automatically when gymnasium / stable-baselines3 are not installed (e.g. the py314 default).

## Dynamics

`src/dynamics.py` provides standard rigid-body dynamics on any arm MJCF:

```python
from src.dynamics import ArmDynamics

dyn = ArmDynamics("models/xarm6/xarm6.xml", payload=(3.0, [0, 0, 0.021]))  # 3 kg, CoM in site-frame meters
M = dyn.mass_matrix(q)                     # dense M(q)
g = dyn.gravity(q)                         # g(q)
c = dyn.coriolis(q, qd)                    # C(q,qd)qd
tau = dyn.inverse_dynamics(q, qd, qdd)     # actuator torque, exact inverse of forward_dynamics
qdd = dyn.forward_dynamics(q, qd, tau)
tau_ff = dyn.feedforward_traj(q_traj, dt)  # torques along a joint trajectory
```

Payload COM is relative to the specified site (`attachment_site` by default), including its position and orientation. Payloads are point masses with zero inertia about their own COM; zero mass adds no body. For combined models, arm-only velocity/acceleration inputs keep the gripper stationary; use full positions to select its opening. Arm-only torques solve the arm mass-matrix block with gripper acceleration fixed at zero; full torques solve all DOFs. `full=True` returns all DOFs and is required for inverse dynamics with nonzero gripper acceleration. Equality-constraint forces are excluded.

Cross-validate against a real xArm 6 (xarm-python-sdk, payload read from `tcp_load`):

```bash
python scripts/compare_dynamics_xarm6.py                     # record sweeps + replay ID comparison
python scripts/compare_dynamics_xarm6.py --replay build/dynamics/run_xxx.csv  # payload from .meta.json
python scripts/compare_dynamics_xarm6.py --replay build/dynamics/run_20261012_172940.csv --payload 3 0 0 21
python scripts/compare_torque_xarm6.py --no-move             # static holding-torque comparison
```

Recording preserves the CSV columns and saves a same-stem `.meta.json` (version 1, `payload.mass_kg`, `payload.com_m`, `payload.frame="attachment_site"`). Replay prefers explicit `--payload M X Y Z` (kg and flange COM mm), then metadata (COM meters). Legacy CSVs without metadata require `--payload`; use `--payload 0 0 0 0` for no load.

## Reinforcement Learning

`src/rl/reach_env.py` implements `ReachEnv`, a Gymnasium environment covering all
five arms (lite6, xarm5, xarm6, xarm7, uf850; `XArm6ReachEnv` stays as a
compatibility alias) where the TCP chases a randomly sampled target point. The
base-frame target box, shoulder pre-filter origin and radial band are calibrated
per robot (`ROBOT_PROFILES`, derived by `scripts/measure_reach_workspace.py` from
FK sampling plus IK reachability and guarded by regression tests). Actions are
joint-position increments (±0.05 rad per 20 ms control step, clamped to joint
limits); the `3+3n`-dim observation (n = joint count: 18 for xarm5, 21 for
lite6/xarm6/uf850, 24 for xarm7) contains the TCP-target delta, normalized joint
positions/velocities and the last action. Reward is potential shaping on the
target distance plus a success bonus (5 mm threshold, aligned with the
ufactory_genesis standard) and small step/action/joint-limit
costs; episodes are 200 control steps (4 s).

```bash
# Train PPO (py313: gymnasium + stable-baselines3, see requirements-rl.txt)
/opt/conda_envs/py313/bin/python scripts/train_reach_rl.py --model xarm6   # 300k steps, 8 envs
/opt/conda_envs/py313/bin/python scripts/train_reach_rl.py --model lite6   # any of the five arms
/opt/conda_envs/py313/bin/python scripts/train_reach_rl.py --model xarm5 --total-steps 1000000

# Evaluate / visualize the trained policy
/opt/conda_envs/py313/bin/python examples/rl/01_reach.py --model xarm6 --headless --episodes 20
```

Training saves `build/rl/<model>/ppo_reach.zip` plus `build/rl/<model>/metrics.json`
(success rate, final TCP error, package versions). The success criterion and the
acceptance bar follow the ufactory_genesis reach standard: success is a TCP
distance **strictly below 5 mm**, and each acceptance group requires a success
rate **>= 95%** with a mean termination distance **<= 10 mm** over 512 targets.
Reference policies (2M PPO steps, [256,256,128] MLP, ent_coef 0.001, ~11 min per
arm on 8 CPU workers; uf850 needed 4M steps), evaluated over two independent
512-target groups (`--seed 17000` / `--seed 17512`) via
`examples/rl/01_reach.py --headless`:

| model | success rate (g1 / g2) | mean final TCP error (g1 / g2) | mean steps |
| :--- | ---: | ---: | ---: |
| lite6 | 99.8% / 99.4% | 4.3 / 4.4 mm | 82 |
| xarm5 | 99.4% / 99.0% | 4.2 / 4.2 mm | 95 |
| xarm6 | 100% / 99.8% | 4.1 / 4.1 mm | 72 |
| xarm7 | 99.6% / 99.0% | 4.3 / 4.3 mm | 88 |
| uf850 | 98.1% / 96.3% | 4.4 / 4.9 mm | 110 |

(The xArm 6 300k-step run from 0.0.5 reached 96% over 50 episodes at the old
15 mm threshold.) The environment itself only needs mujoco + numpy: without
gymnasium it falls back to a minimal space stub so `examples/rl/01_reach.py
--random` still runs on the py314 default.

## Project Structure

```
models/                 Robot and accessory models & meshes
  xarm6/                xArm 6 model (scene.xml, scene_reach.xml, xarm6.xml, visual OBJs, collision STLs)
  xarm5/, xarm7/, ...   Assets for xArm 5/7, UF850, Lite6, and grippers (G2, Bio, Lite6)
src/                    Core library
  sim_robot.py          SimRobot: physics stepping, joint/TCP control, viewer sync
  robot_interface.py    Abstract base class (sim / real robot interface)
  ik.py                 Damped least-squares (DLS) Cartesian IK solver
  kinematics.py         FK, TCP Jacobian, velocity-level DLS IK, manipulability/singularity metrics
  trajectory.py         JointTrajectory data model, PTP quintic/LSPB, spline waypoints, Cartesian line
  dynamics.py           Mass matrix, gravity/Coriolis, inverse/forward dynamics, payload injection
  collision.py          CollisionChecker: clearance/collision queries for planning
  path_planning.py      RRT-Connect + shortcut for collision-free joint paths
  rl/reach_env.py       Gymnasium TCP-reaching environment (xArm 6, PPO training target)
  model_paths.py        Model name -> XML resolution (MODELS_ROOT, resolve_model_path)
  viewer.py             GLFW/X11 passive viewer lifecycle wrapper
examples/
  simulation/           Simulation examples (load_and_view, joint_control, cartesian_ik,
                        gripper_control, pick_place, trajectory_planning)
  rl/                   Reinforcement-learning examples (01_reach)
scripts/                Asset and model tooling
  convert_glb_to_obj.py Batch GLB -> OBJ converter with winding repair & Draco support
  build_all_mjcf.py     Generate standalone MJCF models and scene XMLs for all targets
  build_model.py        URDF to MJCF draft generator
  compare_dynamics_xarm6.py  Real-robot dynamic recording + inverse-dynamics replay
  compare_torque_xarm6.py    Static holding-torque comparison (real vs MuJoCo)
  train_reach_rl.py     PPO training for the RL TCP-reaching task (py313)
tests/                  Unit tests (viewer lifecycle, asset validation, converter regression)
```

## Model Notes

- Kinematics and inertial tensors are preserved from official UFACTORY definitions ([xarm_ros](https://github.com/xArm-Developer/xarm_ros) / [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis)).
- Actuator parameters follow [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) conventions.
- Visual meshes (`group=2`, `contype=0 conaffinity=0`) and collision meshes (`group=3`) are strictly decoupled.

## License

MIT
