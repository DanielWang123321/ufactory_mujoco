# Simulation Examples (`examples/simulation`)

[English](README.md) | [中文](README_zh.md)

MuJoCo simulation examples (`01`–`10`) covering interactive model inspection, joint and Cartesian control, gripper servoing, cube pick-and-place, time-parameterized trajectory planning, RRT-Connect obstacle avoidance, box-packing showcase, kinematics/dynamics toolkits, and gated real-robot workflows.

Requires Python **3.12–3.14** and the core install (see the [installation guide](../../README.md#installation)). Examples `05`, `08` and `10` invoke the packaged CLI tasks `ufactory-mujoco pick-place`, `dynamics` and `box-pack`, which also work from an installed wheel outside the repository checkout. See [Section 3](#3-cli-options-summary) for flags, [Section 4](#4-commands-by-model) for all 44 models, and [Section 5](#5-viewer-controls) for viewer shortcuts.

## 1. Examples Overview

| Script | Core Module(s) | Supported Models | Description |
| :--- | :--- | :--- | :--- |
| `01_load_and_view.py` | `ufactory_mujoco.viewer`, `ufactory_mujoco.model_paths` | All 44 models (arms, grippers, cameras, brackets & combos) | Load MJCF scene at `home` or `ready` keyframe in the interactive viewer |
| `02_joint_control.py` | `ufactory_mujoco.sim_robot` | All 30 arm & combination models | Speed-based quintic joint interpolation (`xarm6` runs 20 dynamics poses by default) |
| `03_cartesian_ik.py` | `ufactory_mujoco.sim_robot`, `ufactory_mujoco.ik` | `xarm6` | Cartesian straight-line (`move_tcp`) and X-Z circle tracking via DLS IK |
| `04_gripper_control.py` | `ufactory_mujoco.sim_robot`, `ufactory_mujoco.g2_controller`, `ufactory_mujoco.bio_g2_controller` | 19 jaw-gripper models (standalone & combos) | Cycle `close (1.0) -> open (0.0) -> half (0.5) -> open (0.0)`; G2 and Bio G2 support `--speed-percent` & `--force-percent` |
| `05_pick_place.py` | `ufactory_mujoco.tasks.pick_place`, `ufactory_mujoco.sim_robot`, `ufactory_mujoco.preflight` | Five arm+gripper task combos | Cube pick-and-place with bilateral contact check & preflight (`place_error < 50 mm`, `home_drift < 10 mm`) |
| `06_trajectory_planning.py` | `ufactory_mujoco.trajectory`, `ufactory_mujoco.dynamics` | `xarm6` | Analytical PTP quintic, C2 multi-waypoint spline, Cartesian SLERP line & feedforward torques |
| `07_kinematics.py` | `ufactory_mujoco.kinematics` | `xarm6` | Per-site FK, analytical vs finite-difference Jacobian, velocity-level DLS IK, manipulability |
| `08_dynamics.py` | `ufactory_mujoco.tasks.dynamics`, `ufactory_mujoco.dynamics`, `ufactory_mujoco.collision` | All five bare arms | Dynamics, fixed collision fixtures, command preflight & monitored motion; xArm6 3 kg payload regression |
| `09_obstacle_avoidance.py` | `ufactory_mujoco.collision`, `ufactory_mujoco.path_planning` | `xarm6` (`scene_obstacles.xml`) | RRT-Connect + shortcutting + spline/quintic time parameterization around pillar & board |
| `10_box_pack.py` | `ufactory_mujoco.tasks.box_pack`, `ufactory_mujoco.sim_robot`, `ufactory_mujoco.preflight` | Five arm+gripper task combos | Box-packing showcase: cubes released 50 mm above the rim free-fall into the box (1 mm tolerance); multi-cycle & speed multiplier |

## 2. Running the Examples

All scripts need only the core install. `--headless` runs `02`–`10` without a viewer and prints verification metrics; `01 --headless` only loads the model and prints a summary.

### 2.1 Model Viewing & Basic Control (`01`–`04`)

```bash
# 01: Interactive viewer (all 44 models)
python examples/simulation/01_load_and_view.py
python examples/simulation/01_load_and_view.py --model lite6 --pose ready
python examples/simulation/01_load_and_view.py --model xarm6_with_gripper --pose ready
python examples/simulation/01_load_and_view.py --model camera_stand_with_d405
python examples/simulation/01_load_and_view.py --model xarm6_with_d435 --pose ready
python examples/simulation/01_load_and_view.py --model uf850_with_gripper_and_d435 --pose ready

# 02: Joint-space quintic trajectory control (all 30 arm & combination models)
python examples/simulation/02_joint_control.py --model lite6
python examples/simulation/02_joint_control.py --model xarm6 --speed 120
python examples/simulation/02_joint_control.py --model xarm6 --motion demo --duration 2.5
python examples/simulation/02_joint_control.py --model xarm7_with_d435 --headless

# 03: Cartesian IK trajectory tracking (line + circle)
python examples/simulation/03_cartesian_ik.py
python examples/simulation/03_cartesian_ik.py --headless

# 04: Gripper open/close control (Gripper G2, Bio Gripper G2, Lite 6 Gripper)
python examples/simulation/04_gripper_control.py --model xarm6_with_gripper
python examples/simulation/04_gripper_control.py --model xarm6_with_gripper --speed-percent 80 --force-percent 75
python examples/simulation/04_gripper_control.py --model lite6_with_gripper
python examples/simulation/04_gripper_control.py --model bio_gripper_g2 --cycles 5
```

> **Notes on `02_joint_control.py`**:
> - Bare `xarm6` defaults to the 20 `ufactory_genesis` dynamics validation poses (two round trips) at a peak target speed of `90 deg/s`. Other models run `ready -> demo -> ready` (combinations without an explicit entry in `PRESETS` start from their own `ready` keyframe and use their arm's `demo` pose).
> - Shadows and reflections are disabled by default (`--render-quality fast`) to reduce GPU cost on high-poly meshes; pass `--render-quality full` to restore them.
> - `--sync-hz 30` sets the viewer state-sync rate while physics steps remain at `2 ms`.

### 2.2 Manipulation & Motion Planning (`05`, `06`, `09`, `10`)

```bash
# 05: Cube pick-and-place (equivalent to `ufactory-mujoco pick-place`)
python examples/simulation/05_pick_place.py --headless
python examples/simulation/05_pick_place.py --robot lite6 --executor servo_j --headless
python examples/simulation/05_pick_place.py --robot xarm7 --dry-run
ufactory-mujoco pick-place --robot xarm6 --headless --report build/reports/pick_place.json

# 06: Time-parameterized trajectory planning (PTP quintic, C2 spline, Cartesian line)
python examples/simulation/06_trajectory_planning.py
python examples/simulation/06_trajectory_planning.py --headless --speed 120

# 09: Collision-free RRT-Connect planning around pillar and overhead board
python examples/simulation/09_obstacle_avoidance.py
python examples/simulation/09_obstacle_avoidance.py --headless
python examples/simulation/09_obstacle_avoidance.py --seed 3 --safety-margin 0.01

# 10: Box-packing showcase (equivalent to `ufactory-mujoco box-pack`)
python examples/simulation/10_box_pack.py --headless
python examples/simulation/10_box_pack.py --robot lite6 --cycles 3 --speed 2.0 --headless
python examples/simulation/10_box_pack.py --robot xarm7 --dry-run
ufactory-mujoco box-pack --robot uf850 --cycles 3 --headless --report build/reports/box_pack.json
```

### 2.3 Kinematics & Rigid-Body Dynamics Toolkits (`07`, `08`)

```bash
# 07: Kinematics verification (FK, Jacobian, velocity DLS IK, singularity metrics)
python examples/simulation/07_kinematics.py
python examples/simulation/07_kinematics.py --headless

# 08: Dynamics and monitored motion (default: xarm6; headless also steps physics)
python examples/simulation/08_dynamics.py
python examples/simulation/08_dynamics.py --headless
python examples/simulation/08_dynamics.py --robot lite6 --headless --samples 32 --seed 0
python examples/simulation/08_dynamics.py --robot all --headless --report build/dynamics/report.json

# Schema v4 dynamics validation (aligned with ufactory_genesis, 20-pose + random offline check)
dynamics-sim-check --robot xarm6 --random-count 20 --require-reference
dynamics-hardware-check --robot lite6 --dry-run --require-reference --strict-static --strict-static-layers l2b
```

`08_dynamics.py` (`ufactory-mujoco dynamics`) verifies all five bare arms without requiring an xArm SDK:

- **Static and random pose checks**: evaluates `home`, `ready`, `target` and 32 joint-limit-bounded random poses (`--seed 0` by default) in bare-arm scenes without added payloads—checking finite symmetric positive-definite mass matrices, finite gravity, zero Coriolis at rest, ID/FD reciprocity, feedforward torque inside actuator ranges, and fixed safe/colliding pose fixtures (random collisions are classified separately).
- **8.4 s monitored motion**: runs `home -> ready (1.5 s) -> target (3.0 s) -> ready (3.0 s)` with `0.3 s` settling after each move. Commands are preflighted at every physics step, and an independent `CollisionChecker` inspects integrated joint positions after every step (`penetration <= 1e-6 m`, `joint_limit_excess <= 1e-6 rad`, settled arrival error `< 0.01 rad`; NaN/Inf states, MuJoCo warnings, early exits, or closing the viewer fail validation). `--robot all` retains every arm's result and continues after a single-arm failure.
- **Reports and baselines**: `--report` writes strict JSON with versions, environment, paths, thresholds, metrics, and failure reasons (exit codes: `0` pass, `1` fail, `2` invalid arguments). `xarm6` additionally preserves the 3 kg payload feedforward check and six-axis gravity regression (`<= 1e-6 Nm`; J2/J3 retain a historical 5% tolerance without re-running hardware). Fixed poses, thresholds, and baselines live in [`ufactory_mujoco/tasks/dynamics.py`](../../ufactory_mujoco/tasks/dynamics.py).

For `ufactory_genesis`-aligned 20-pose static torque verification (PD hold vs analytical reference and hardware SDK, L1 + L2a/L2b/L3a/L3b layered checks, Schema v4 CSV/JSONL reports, and `_torque.png` plots), use `dynamics-sim-check`, `dynamics-hardware-check --dry-run`, and `dynamics-report-compare` (or `python -m ufactory_mujoco.dynamics <subcommand>`).

#### Programmatic Dynamics API (`ufactory_mujoco/dynamics`)

```python
from ufactory_mujoco.dynamics import ArmDynamics

# Optional payload: (mass_kg, com_xyz_m) expressed in the attachment_site frame
dyn = ArmDynamics("models/xarm6/xarm6.xml", payload=(3.0, [0, 0, 0.021]))
M = dyn.mass_matrix(q)                     # dense mass matrix M(q)
g = dyn.gravity(q)                         # gravity vector g(q)
c = dyn.coriolis(q, qd)                    # velocity product term C(q,qd)qd
tau = dyn.inverse_dynamics(q, qd, qdd)     # actuator torque (exact inverse of forward_dynamics)
qdd = dyn.forward_dynamics(q, qd, tau)
tau_ff = dyn.feedforward_traj(q_traj, dt)  # feedforward torques along a trajectory
```

### 2.4 Real Robot & Controller Simulation (`xarm-python-sdk`)

Hardware modes require `xarm-python-sdk` (`pip install xarm-python-sdk`), which is not needed for simulation. Replace the IP address with your controller's (or set `XARM_IP`). Every hardware mode rejects a calibration YAML whose serial number or SHA-256 does not match the connected controller.

- **Zero physical motion checks**: `kinematics_validation`, `08 --mode real-read-only`, and `--mode sdk-sim` (`05`, `08`, `10`) do not move the physical joints (`sdk-sim` replays trajectories inside the controller's built-in simulator and restores non-simulation mode on exit).
- **Physical motion gate**: `--mode real` (`05`, `08`, `10`) requires healthy controller telemetry, passing preflight, and `--confirm-real`, and enforces `cycles == 1`. In non-`sim` modes, `08_dynamics.py` requires a single arm key (not `--robot all`). All hardware dynamics validation commands (`dynamics-hardware-check`, `dynamics-sim-collision-check`, `08 --mode sdk-sim`/`real`) enforce a hard joint speed ceiling of `<= 40 °/s` (`0.6981 rad/s`).

```bash
# 1. Extract the per-unit calibration YAML (calibrations/<arm>_kinematics_<SN>.yaml)
python -m ufactory_mujoco.dynamics export-calibration 192.168.1.60 --output-dir calibrations

# 2. Read-only FK/IK comparison and zero-motion static dynamics checks
python -m ufactory_mujoco.kinematics_validation --robot xarm6 --ip 192.168.1.60 --calibration calibrations/xarm6_kinematics_<SN>.yaml
python examples/simulation/08_dynamics.py --robot xarm6 --mode real-read-only --ip 192.168.1.60 --calibration calibrations/xarm6_kinematics_<SN>.yaml
dynamics-hardware-check --robot lite6 --ip 192.168.1.60 --confirm-real --hold-current-only --require-reference

# 3. Controller built-in simulator replay (no physical joint motion)
python examples/simulation/05_pick_place.py --robot xarm6 --mode sdk-sim --ip 192.168.1.60 --calibration calibrations/xarm6_kinematics_<SN>.yaml
python examples/simulation/10_box_pack.py --robot xarm6 --mode sdk-sim --ip 192.168.1.60 --calibration calibrations/xarm6_kinematics_<SN>.yaml
dynamics-sim-collision-check --robot lite6 --ip 192.168.1.60

# 4. Physical motion (requires --confirm-real; ensure clear workspace & E-stop)
ufactory-mujoco pick-place --robot xarm6 --mode real --confirm-real --ip 192.168.1.60 --calibration calibrations/xarm6_kinematics_<SN>.yaml
dynamics-hardware-check --robot xarm6 --ip 192.168.1.60 --confirm-real --require-reference
dynamics-report-compare reports/dyn_ver_<SN>/<old>.jsonl reports/dyn_ver_<SN>/<new>.jsonl
```

## 3. CLI Options Summary

### 3.1 Task & Hardware Options (`05`, `08`, `10` / `ufactory-mujoco`)

| Option | Scripts | Default | Description |
| :--- | :--- | :--- | :--- |
| `--robot <arm>` | `05`, `08`, `10` | `xarm6` | Arm key (`lite6`, `xarm5`, `xarm6`, `xarm7`, `uf850`); `08` (`--mode sim`) also accepts `all` |
| `--gripper <name>` | `05`, `10` | `None` (from YAML) | Override the mounted gripper in task config |
| `--executor <mode>` | `05`, `10` | `servo_cartesian` | Trajectory executor: `servo_cartesian` or `servo_j` |
| `--config <path>` | `05`, `10` | `configs/<task>/<robot>.yaml` | Custom task overlay YAML path |
| `--mode <mode>` | `05`, `08`, `10` | `sim` | Execution mode: `sim`, `dry-run` (`05`/`10`), `real-read-only` (`08`), `sdk-sim`, or `real` |
| `--dry-run` | `05`, `10` | `False` | Run joint-limit/collision preflight only without stepping physics |
| `--cycles <N>` | `05`, `10` | `1` | Number of cycles (`10`: `1..3`; `sdk-sim`/`real` enforce `1`) |
| `--speed <x>` | `10` | YAML `speed_scale` (`1.0`) | Motion speed multiplier (`>1` is faster; free-fall physics is not scaled) |
| `--samples <N>` | `08` | `32` | Positive number of random dynamics poses per arm |
| `--seed <N>` | `08` | `0` | RNG seed for random dynamics pose sampling (integer) |
| `--report <path>` | `05`, `08`, `10` | `None` | Write preflight, dynamics validation or cycle execution report as JSON |
| `--ip <addr>` | `05`, `08`, `10` | `XARM_IP` | xArm controller IPv4 for `sdk-sim`, `real`, `real-read-only`; required if `XARM_IP` is unset |
| `--calibration <path>` | `05`, `08`, `10` | `None` | Required per-unit kinematic calibration YAML for hardware modes |
| `--confirm-real` | `05`, `08`, `10` | `False` | Explicit safety confirmation required before `--mode real` motion |
| `--headless` | `05`, `08`, `10` | `False` | Run without a viewer and print verification metrics |

### 3.2 Script-Specific Options (`01`–`04`, `06`, `07`, `09`)

Examples `03` and `07` accept only `--headless`.

| Option | Scripts | Default | Description |
| :--- | :--- | :--- | :--- |
| `--headless` | `01`–`04`, `06`, `07`, `09` | `False` | Run without a viewer and print metrics (`01`: load and print summary only) |
| `--model <name>` | `01`, `02`, `04` | `xarm6` (`04`: `xarm6_with_gripper`) | Model directory name or path to an MJCF XML file |
| `--pose <home\|ready>` | `01` | `home` | Initial keyframe pose |
| `--motion <mode>` | `02` | `auto` | Motion preset: `auto`, `dynamics` (20 poses, bare `xarm6` only), or `demo` |
| `--speed <deg/s>` | `02`, `06` | `90.0` | Peak target joint speed in degrees/s |
| `--duration <s>` | `02` | `None` | Fixed seconds per segment (overrides `--speed`) |
| `--sync-hz <Hz>` | `02` | `30.0` | Viewer state-sync rate (physics timestep remains `2 ms`) |
| `--render-quality` | `02` | `fast` | `fast` (disables shadows/reflections) or `full` |
| `--cycles <N>` | `04` | `3` | Number of open/close demonstration cycles |
| `--speed-percent <pct>` | `04` | `100.0` | Gripper G2 / Bio Gripper G2 speed percentage (`0..100`; ignored by Lite 6 gripper) |
| `--force-percent <pct>` | `04` | `100.0` | Gripper G2 / Bio Gripper G2 force percentage (`1..100`; ignored by Lite 6 gripper) |
| `--seed <N>` | `09` | `0` | RNG seed for RRT-Connect sampling and path shortcutting |
| `--safety-margin <m>` | `09` | `0.002` | Minimum allowed obstacle clearance in meters |
| `--step <rad>` | `09` | `0.15` | RRT-Connect joint-space steer step size (rad) |
| `--max-iter <N>` | `09` | `2000` | Maximum RRT-Connect iterations |

## 4. Commands by Model

`01`, `02` and `04` take a model name with `--model`; `05`, `08` and `10` take an arm key with `--robot`. Every command opens the viewer unless `--headless` is given.

### 4.1 Arms and Combinations

| Arm | Bare arm (`--model`) | Combinations (`--model`) | Arm key (`--robot`) |
| :--- | :--- | :--- | :--- |
| Lite 6 | `lite6` | `lite6_with_gripper`, `lite6_with_vacuum_gripper` | `lite6` |
| xArm 5 | `xarm5` | `xarm5_with_gripper`, `xarm5_with_bio_gripper`, `xarm5_with_vacuum_gripper`, `xarm5_with_d435`, `xarm5_with_gripper_and_d435` | `xarm5` |
| xArm 6 | `xarm6` | `xarm6_with_gripper`, `xarm6_with_bio_gripper`, `xarm6_with_vacuum_gripper`, `xarm6_with_d435`, `xarm6_with_gripper_and_<camera>` | `xarm6` |
| xArm 7 | `xarm7` | `xarm7_with_gripper`, `xarm7_with_bio_gripper`, `xarm7_with_vacuum_gripper`, `xarm7_with_d435`, `xarm7_with_gripper_and_d435` | `xarm7` |
| UF850 | `uf850` | `uf850_with_gripper`, `uf850_with_bio_gripper`, `uf850_with_vacuum_gripper`, `uf850_with_d435`, `uf850_with_gripper_and_d435` | `uf850` |

`<arm>_with_gripper` mounts Gripper G2 (the Lite 6 gripper on Lite 6). `<camera>` is `d405`, `d435`, `gemini_305` or `zed_x_nano`.

Model applicability across scripts (`44` model directories in total):

- `01_load_and_view.py`: all `44` models (bare arms, combinations, standalone tools/cameras/brackets).
- `02_joint_control.py`: all `30` models with an arm (`5` bare arms + `25` arm combinations).
- `04_gripper_control.py`: all `19` models with a jaw actuator (`3` standalone grippers + `16` arm/camera combinations with a jaw gripper).
- `05_pick_place.py`, `10_box_pack.py`: `5` task pairs selected by `--robot` (`lite6` with `lite6_gripper`; the other four with `gripper_g2`).
- `08_dynamics.py`: `5` bare arms via `--robot` (plus `all` in `sim` mode). Examples `03`, `06`, `07` and `09` run on the bare `xarm6` only.

```bash
# Any arm or combination: static view and joint motion
python examples/simulation/01_load_and_view.py --model xarm7_with_bio_gripper --pose ready
python examples/simulation/02_joint_control.py --model xarm5_with_vacuum_gripper

# Combinations with jaws (Gripper G2, Bio Gripper G2, Lite 6 gripper): open/close
python examples/simulation/04_gripper_control.py --model uf850_with_bio_gripper
python examples/simulation/04_gripper_control.py --model xarm6_with_gripper_and_d405

# Tasks: --robot selects the arm and its task gripper
python examples/simulation/05_pick_place.py --robot xarm5
python examples/simulation/10_box_pack.py --robot uf850 --cycles 3
python examples/simulation/08_dynamics.py --robot all --headless
```

### 4.2 Standalone End Effectors, Cameras and Brackets

| Model (`--model`) | `01_load_and_view.py` | `04_gripper_control.py` |
| :--- | :---: | :--- |
| `gripper_g2` | yes | open/close with `--speed-percent`, `--force-percent` |
| `bio_gripper_g2` | yes | open/close with `--speed-percent`, `--force-percent` |
| `lite6_gripper` | yes | open/close |
| `lite6_vacuum_gripper`, `vacuum_gripper` | yes | rejected (no jaw actuator, exit code `2`) |
| `d405`, `d435`, `gemini_305`, `zed_x_nano` | yes | rejected (exit code `2`) |
| `camera_stand`, `camera_stand_with_<camera>` | yes | rejected (exit code `2`) |

Camera and bracket models have `mass=0` and no optical sensor. Enable Site Group `4` in the viewer's right panel to show their mounting faces and screw holes.

### 4.3 Keyframes

- **Bare xArm 5/6/7 and UF850**: `home` and `ready` are all-zero joint angles.
- **Lite 6 (bare and combinations)**: `home` is all zeros; `ready` is the Studio factory pose `[0, 9.9, 31.8, 0, 21.9, 0]°` (flange near `[200, 0, 200]` mm, orientation `[180, 0, 0]°`).
- **xArm 5/6/7 and UF850 combinations**: `ready` is the zero-roll downward IK pose placing Gripper G2's TCP at `(300, 0, 300)` mm with `RPY = (180, 0, 0)°` (`xArm 6` uses the same pose for `home`; `xArm 5`, `xArm 7`, and `UF850` define their own `home` in each MJCF—note that `uf850_with_*` uses all-zero `home`, so `--pose ready` is recommended when inspecting combinations).
- **Tool offsets and D435 orientation**: combination TCPs include the mounted tool offset; `<arm>_with_d435` shifts `attachment_site` `+3 mm` along flange `+Z` for the bracket plate thickness. Mounted `d435` assemblies orient the camera for a human-eye upright view (image top toward base `+X`, USB-C port on flange `+Y` / base `Y-` in tool-down poses).
- **Standalone models**: movable grippers (`gripper_g2`, `bio_gripper_g2`, `lite6_gripper`) define `open` and `close` keyframes; passive vacuum grippers, cameras, and stands have no keyframes (`nkey == 0`) and load in their default pose.

## 5. Viewer Controls

From the MuJoCo viewer's built-in help (`F1`):

- **Left drag**: rotate. **Right drag**: pan. **Scroll or middle drag**: zoom.
- **Double-click**: select a body. **Right double-click**: center the view on the clicked point. **Ctrl + right double-click**: track the selected body.
- **Ctrl + left drag / Ctrl + right drag**: apply a torque / force to the selected body.
- **Space**: pause or resume. **Backspace**: reset. Both control physics only in `01_load_and_view.py`; the other examples step physics themselves.
- **C**: contact points. **F**: contact forces. **0–5**: toggle geom groups (`2` visual meshes, `3` collision meshes).
- **Ctrl + A**: align the view. **Tab / Shift + Tab**: show or hide the left / right panel. **F1**: help.
