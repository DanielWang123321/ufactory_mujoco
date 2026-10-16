# Simulation Examples (`examples/simulation`)

[English](README.md) | [中文](README_zh.md)

MuJoCo simulation examples (`01`–`09`) covering interactive model inspection, joint/Cartesian motion control, gripper actuation, pick-and-place manipulation, time-parameterized trajectory planning, RRT-Connect obstacle avoidance, and kinematics/dynamics toolkits.

> 📖 For a model-by-model command matrix (Lite 6, xArm 5/6/7, UF850; standalone and combined with grippers), see **[docs/visual_command_guide.md](../../docs/visual_command_guide.md)**.

## 1. Examples Overview

| Script | Core Module(s) | Supported Models | Description |
| :--- | :--- | :--- | :--- |
| `01_load_and_view.py` | `src.viewer`, `src.model_paths` | All arms, grippers & combos | Load MJCF scene at `home` or `ready` keyframe in the interactive viewer |
| `02_joint_control.py` | `src.sim_robot` | All arms & arm+gripper combos | Speed-based quintic joint interpolation (`xarm6` runs 20 dynamics poses by default) |
| `03_cartesian_ik.py` | `src.sim_robot`, `src.ik` | `xarm6` | Cartesian straight-line (`move_tcp`) and X-Z circle tracking via DLS IK |
| `04_gripper_control.py` | `src.sim_robot` | Standalone grippers & combos | Position-controlled open/close cycles (`0.0` = open, `1.0` = closed) |
| `05_pick_place.py` | `src.sim_robot`, `src.ik` | `xarm6_with_gripper` | Cube pick-and-place replicating `ufactory_genesis` (`place_error < 50 mm`, `home_drift < 10 mm`) |
| `06_trajectory_planning.py` | `src.trajectory`, `src.dynamics` | `xarm6` | Analytical PTP quintic, C2 multi-waypoint spline, Cartesian SLERP line & feedforward torques |
| `07_kinematics.py` | `src.kinematics` | `xarm6` | Per-site FK, analytical vs finite-difference Jacobian, velocity-level DLS IK, manipulability |
| `08_dynamics.py` | `src.dynamics` | `xarm6` | Mass matrix $M(q)$, gravity/Coriolis, ID/FD round-trip, payload injection & torque envelope |
| `09_obstacle_avoidance.py` | `src.collision`, `src.path_planning` | `xarm6` (`scene_obstacles.xml`) | RRT-Connect + shortcutting + spline/quintic time parameterization around pillar & board |

## 2. Running the Examples

All scripts run in the default `py314` environment (`mujoco` + `numpy`). Pass `--headless` to `02`–`09` to run offscreen and print quantitative verification metrics.

### 2.1 Model Viewing & Basic Control (`01`–`04`)

```bash
# 01: Interactive viewer (all arms, grippers, and combinations)
python examples/simulation/01_load_and_view.py
python examples/simulation/01_load_and_view.py --model lite6 --pose ready
python examples/simulation/01_load_and_view.py --model xarm6_with_gripper --pose ready

# 02: Joint-space quintic trajectory control
python examples/simulation/02_joint_control.py --model lite6
python examples/simulation/02_joint_control.py --model xarm6 --speed 120
python examples/simulation/02_joint_control.py --model xarm6 --motion demo --duration 2.5

# 03: Cartesian IK trajectory tracking (line + circle)
python examples/simulation/03_cartesian_ik.py
python examples/simulation/03_cartesian_ik.py --headless

# 04: Gripper open/close control (Gripper G2, Bio Gripper G2, Lite 6 Gripper)
python examples/simulation/04_gripper_control.py --model xarm6_with_gripper
python examples/simulation/04_gripper_control.py --model lite6_with_gripper
python examples/simulation/04_gripper_control.py --model bio_gripper_g2 --cycles 5
```

> **Notes on `02_joint_control.py`**:
> - Bare `xarm6` defaults to the 20 `ufactory_genesis` dynamics validation poses (two round trips) at a peak target speed of `90 deg/s`. Other models run `ready -> demo -> ready`.
> - Shadows and reflections are disabled by default (`--render-quality fast`) to reduce GPU cost on high-poly meshes; pass `--render-quality full` to restore them.
> - `--sync-hz 30` sets the viewer state-sync rate while physics steps remain at `2 ms`.

### 2.2 Manipulation & Motion Planning (`05`, `06`, `09`)

```bash
# 05: Cube pick-and-place (xArm 6 + Gripper G2)
python examples/simulation/05_pick_place.py
python examples/simulation/05_pick_place.py --headless

# 06: Time-parameterized trajectory planning (PTP quintic, C2 spline, Cartesian line)
python examples/simulation/06_trajectory_planning.py
python examples/simulation/06_trajectory_planning.py --headless --speed 120

# 09: Collision-free RRT-Connect planning around pillar and overhead board
python examples/simulation/09_obstacle_avoidance.py
python examples/simulation/09_obstacle_avoidance.py --headless
python examples/simulation/09_obstacle_avoidance.py --seed 3 --safety-margin 0.01
```

### 2.3 Kinematics & Rigid-Body Dynamics Toolkits (`07`, `08`)

```bash
# 07: Kinematics verification (FK, Jacobian, velocity DLS IK, singularity metrics)
python examples/simulation/07_kinematics.py
python examples/simulation/07_kinematics.py --headless

# 08: Rigid-body dynamics verification (M(q), g(q), C(q,qd)qd, ID/FD, 3 kg payload)
python examples/simulation/08_dynamics.py
python examples/simulation/08_dynamics.py --headless
```

#### Programmatic Dynamics API (`src/dynamics.py`)

```python
from src.dynamics import ArmDynamics

# Optional payload: (mass_kg, com_xyz_m) expressed in the attachment_site frame
dyn = ArmDynamics("models/xarm6/xarm6.xml", payload=(3.0, [0, 0, 0.021]))
M = dyn.mass_matrix(q)                     # dense mass matrix M(q)
g = dyn.gravity(q)                         # gravity vector g(q)
c = dyn.coriolis(q, qd)                    # velocity product term C(q,qd)qd
tau = dyn.inverse_dynamics(q, qd, qdd)     # actuator torque (exact inverse of forward_dynamics)
qdd = dyn.forward_dynamics(q, qd, tau)
tau_ff = dyn.feedforward_traj(q_traj, dt)  # feedforward torques along a trajectory
```

Cross-validation scripts against a real xArm 6 (`xarm-python-sdk`):

```bash
python scripts/compare_dynamics_xarm6.py                                      # record + ID replay
python scripts/compare_dynamics_xarm6.py --replay build/dynamics/run_xxx.csv  # replay from .meta.json
python scripts/compare_torque_xarm6.py --no-move                              # static holding torque
```

## 3. CLI Options Summary

| Option | Applicable Scripts | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model <name>` | `01`, `02`, `04`, `05` | `xarm6` (`05`: `xarm6_with_gripper`) | Model name or path to MJCF XML |
| `--pose <home\|ready>` | `01` | `home` | Initial keyframe pose |
| `--headless` | `02`–`09` | `False` | Run offscreen and print verification metrics |
| `--speed <deg/s>` | `02`, `06` | `90.0` | Peak joint speed in degrees/s |
| `--motion <mode>` | `02` | `auto` | Motion preset: `auto`, `dynamics` (20 poses), or `demo` |
| `--duration <s>` | `02` | `None` | Fixed segment duration (overrides `--speed`) |
| `--sync-hz <Hz>` | `02` | `30.0` | Viewer state-sync rate |
| `--render-quality` | `02` | `fast` | `fast` (no shadows/reflections) or `full` |
| `--cycles <N>` | `04`, `05` | `3` (`04`) / `1` (`05`) | Number of execution cycles |
| `--seed <N>` | `09` | `0` | RNG seed for RRT-Connect and path shortcutting |
| `--safety-margin <m>` | `09` | `0.002` | Minimum allowed obstacle clearance in meters |
| `--step <rad>` | `09` | `0.15` | RRT-Connect joint-space steer step size (rad) |
| `--max-iter <N>` | `09` | `2000` | Maximum RRT-Connect iterations |

