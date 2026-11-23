# ufactory_mujoco

![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue) ![MuJoCo](https://img.shields.io/badge/mujoco-4.x-lightgrey) ![License](https://img.shields.io/badge/license-MIT%20%2F%20BSD--3--Clause%20%2F%20Apache--2.0-green) [![CI](https://github.com/DanielWang123321/ufactory_mujoco/actions/workflows/ci.yml/badge.svg)](https://github.com/DanielWang123321/ufactory_mujoco/actions/workflows/ci.yml)

[English](README.md) | [中文](README_zh.md) · [Changelog](CHANGELOG.md)

MuJoCo simulation suite for UFACTORY robots (Lite 6, xArm 5/6/7, UF850), grippers, depth cameras, force/torque sensor, and mounting base plates: detailed MJCF models, kinematics and dynamics tools, motion planning, scripted manipulation tasks, and a gated interface to a real xArm.

## Quick Start

Requires Python **3.12–3.14** (see [Installation](#installation)).

```bash
git clone --depth 1 https://github.com/DanielWang123321/ufactory_mujoco.git && cd ufactory_mujoco
python -m pip install -e .
python examples/simulation/01_load_and_view.py --model xarm6_with_gripper --pose ready
```

Without a display, run a task headless: `ufactory-mujoco pick-place --robot xarm6 --headless`.

## Features

- **Models**: all five arms, five end effectors, four depth cameras, camera bracket, AI1500 6-axis force/torque sensor (`ft_sensor` with deterministic `200 Hz` `AI1500Reader`) and mounting base plates (`*_base_plate`). Only the atomic parts are stored as MJCF (`ufactory_mujoco/models/`); every arm + accessory combination and task scene is assembled at runtime with MuJoCo's native `MjSpec.attach` (`compose.py`), including custom multi-accessory stacks and static export via `ufactory-mujoco export-model`. Visual meshes and multi-convex collision shapes stay separate; Gripper G2 and Bio Gripper G2 use calibrated dynamics and rate-limited position/force servos (`g2.py`, `bio_g2.py`).
- **Kinematics and planning**: FK, Jacobians, damped least-squares IK, quintic/LSPB/spline trajectories, Cartesian lines, collision checking, and RRT-Connect (`kinematics.py`, `trajectory.py`, `planning.py`).
- **Dynamics**: mass matrix, gravity/Coriolis, inverse/forward dynamics, payloads, five-arm motion validation (`tasks/dynamics.py`), and `ufactory_genesis`-aligned Schema v4 static torque validation (`dynamics-sim-check`, `dynamics-sim-collision-check`, `dynamics-hardware-check`, `dynamics-report-compare` in `dynamics/`).
- **Scripted tasks**: cube pick-and-place and box packing for five arm/gripper pairs, with preflight (joint limits, penetration, RRT-Connect detours) and JSON reports (`tasks/`, `preflight.py`).
- **Real robot**: `XArmRobot` is read-only by default; motion needs a per-unit calibration, controller checks, and `--confirm-real`. `sdk-sim` replays trajectories in the controller's built-in simulator (`xarm_robot.py`, `calibration.py`, `safety.py`).

## Supported Robots and Tasks

| Arm | Built gripper models | `pick-place` / `box-pack` | `dynamics` |
| :--- | :--- | :--- | :---: |
| `lite6` | `lite6_gripper`, `lite6_vacuum_gripper` | with `lite6_gripper` | yes |
| `xarm5` | `gripper_g2`, `bio_gripper_g2`, `vacuum_gripper`, `ft_sensor`, `ft_sensor_and_gripper`, `ft_sensor_and_vacuum_gripper` | with `gripper_g2` | yes |
| `xarm6` | `gripper_g2`, `bio_gripper_g2`, `vacuum_gripper`, `ft_sensor`, `ft_sensor_and_gripper`, `ft_sensor_and_vacuum_gripper` | with `gripper_g2` | yes |
| `xarm7` | `gripper_g2`, `bio_gripper_g2`, `vacuum_gripper`, `ft_sensor`, `ft_sensor_and_gripper`, `ft_sensor_and_vacuum_gripper` | with `gripper_g2` | yes |
| `uf850` | `gripper_g2`, `bio_gripper_g2`, `vacuum_gripper`, `ft_sensor`, `ft_sensor_and_gripper`, `ft_sensor_and_vacuum_gripper` | with `gripper_g2` | yes |

Named models cover 5 bare arms, 5 end effectors, 4 depth cameras (`d405`, `d435`, `gemini_305`, `zed_x_nano`), camera stands, the AI1500 F/T sensor, mounting base plates (`{xarm,uf850,lite6}_base_plate`) and arm + accessory combinations. `ufactory-mujoco export-model --list` prints every name. Load a model with `01_load_and_view.py --model <name>`, or export a standalone MJCF directory with `ufactory-mujoco export-model <name> --out-dir <dir>`; model names are explained in [examples/simulation/README.md](examples/simulation/README.md#models).

## Installation

| Component | Supported range | Verified versions | Notes |
| :--- | :--- | :--- | :--- |
| **Python** | **3.12, 3.13, 3.14** (`>=3.12`) | `3.12.13`, `3.13.13`, `3.14.6` | Covered by CI and release checks; **Python ≤ 3.11 is unsupported**; Python ≥ 3.15 is untested |
| **MuJoCo** | **≥ 4.0, < 5** | `4.0.0`, `4.0.1` | MuJoCo 3.x is no longer supported from `v0.1.5` |
| Core deps | `numpy>=2.0,<3`, `scipy>=1.14,<2`, `pyyaml>=6.0,<7` | see `requirements-lock.txt` | Installed automatically with `pip install -e .` |
| `.[test]` (optional) | `pytest>=8` | — | Needed only to run the unit test suite |
| `xarm-python-sdk` (optional) | matching controller SDK | — | Needed only for `--mode sdk-sim` / `real` / `real-read-only`, `dynamics-hardware-check` / `dynamics-sim-collision-check`, and calibration export |
| `matplotlib` (optional) | `>=3.8` | — | Optional; generates `_torque.png` comparison plots in `dynamics-hardware-check` (skipped when absent) |

The visual meshes are large, so clone with `--depth 1`. Create an isolated environment with `venv` or `conda`:

```bash
git clone --depth 1 https://github.com/DanielWang123321/ufactory_mujoco.git && cd ufactory_mujoco

# Option A: venv (Python 3.12 / 3.13 / 3.14)
python3.13 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e .

# Option B: conda (Python 3.12 / 3.13 / 3.14)
conda create -n ufactory python=3.14 -y && conda activate ufactory
python -m pip install -e .
```

`requirements-lock.txt` pins the exact versions verified in CI (MuJoCo 4.0.0).

**Upgrading an existing environment from MuJoCo 3 to MuJoCo 4**: MuJoCo 4 splits its Python package into `mujoco-core`, `mujoco-studio` and `mujoco`. Upgrading in place (`pip install -U mujoco`) leaves a broken installation, and `import mujoco` then fails with an error about missing `mujoco-core` bindings. Uninstall first:

```bash
pip uninstall -y mujoco && pip install "mujoco>=4,<5"
```

If the environment is already broken, repair it with `pip install --force-reinstall --no-deps mujoco mujoco-core mujoco-studio`. New environments are not affected.

**Wheel**: [GitHub Releases](https://github.com/DanielWang123321/ufactory_mujoco/releases) from v0.1.0 on include a self-contained wheel with all models, scenes, task configs and licenses bundled. Install it with `pip install ufactory_mujoco-<version>-py3-none-any.whl`; the `ufactory-mujoco` and `dynamics-*` commands then run from any directory. Build one yourself with `python -m build`.

## Command-Line Tasks

```bash
ufactory-mujoco pick-place --robot xarm6
ufactory-mujoco box-pack --robot lite6 --cycles 3
ufactory-mujoco dynamics --robot all --headless --report reports/dynamics.json
ufactory-mujoco export-model xarm6_with_ft_sensor_and_gripper --out-dir exported/xarm6_ft_g2
ufactory-mujoco export-calibration <ROBOT_IP> --output-dir calibrations
ufactory-mujoco kinematics-check --robot xarm6 --ip <ROBOT_IP> --calibration calibrations/xarm6_kinematics_<SN>.yaml
```

- `python -m ufactory_mujoco <command>` is equivalent. Examples `05`, `08` and `10` run the same code.
- `pick-place`, `box-pack` and `dynamics` open the viewer; `--headless` runs them without it.
- `COMMAND --help` lists the options.
- `--dry-run` only runs the preflight and never steps physics.
- Exit codes: `0` success, `1` runtime, preflight, safety-gate, missing model resource (`UFACTORY_MUJOCO_MODELS`), or report failure, `2` invalid arguments or configuration (including task/scene mismatch).
- `--report` paths are relative to the current directory; parent directories are created.
- Hardware modes (`--mode sdk-sim` / `real`, and `real-read-only` for `dynamics`) need `--ip` (or `XARM_IP`) and a matching `--calibration` YAML. `real` also needs `--confirm-real`. `sdk-sim` and `real` run one cycle only.

| Environment variable | Effect |
| :--- | :--- |
| `UFACTORY_MUJOCO_MODELS` | Use another model tree instead of the bundled or repository models |
| `XARM_IP` | Default controller IP for hardware modes |
| `MUJOCO_GL` | Linux OpenGL backend (`egl`, `osmesa`); not needed for `--headless` |

## Platform Notes

- **macOS**: start GUI examples with `mjpython` (shipped with `mujoco`), e.g. `mjpython examples/simulation/01_load_and_view.py`. `--headless` works with plain `python`.
- **Windows**: the viewer works with plain `python`.
- **Headless Linux**: `--headless` needs no display or OpenGL; do not set `MUJOCO_GL` for it. With `01_load_and_view.py` it only loads the model and prints a summary. `MUJOCO_GL=egl` needs `libegl1`, `MUJOCO_GL=osmesa` needs `libosmesa6`. To open the viewer on a server without a display, use `xvfb-run -a python examples/simulation/01_load_and_view.py`.

## Documentation

| Topic | Document |
| :--- | :--- |
| Simulation examples `01`–`11`, model names, start poses, hardware commands, viewer controls | [examples/simulation/README.md](examples/simulation/README.md) ([中文](examples/simulation/README_zh.md)) |
| Contributing, security policy, code of conduct | [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) |
| Release history | [CHANGELOG.md](CHANGELOG.md) |

## Running Tests

```bash
python -m pip install -e ".[test]"
python -m pytest tests
ruff check .                      # lint as in CI (pip install ruff)
```

Tests that need optional packages (`trimesh`, `pinocchio`, `DracoPy`) or maintainer-only source snapshots are skipped automatically when absent.

## Known Limitations

- Bare-arm 20-pose static torque validation (`dynamics-hardware-check`, aligned with `ufactory_genesis` Schema v4, hardware joint speed capped at `<= 40 °/s`) was verified across all five arm families under Python 3.14.6 and MuJoCo 4.0.1: one xArm 5 (firmware v2.8.2, `--nominal-model`; 20/20 PASS, L2 torque error 0.520–0.780 Nm — controller port-502 `0x08` returns degenerate parallel-axis kinematics, so `export-calibration` skips with exit code `2` and the nominal MJCF matches static torques while carrying `~2.6–4.2 mm` Cartesian FK difference vs controller POE), one xArm 6 (firmware v2.8.0; 20/20 PASS, L2 torque error 0.470–0.765 Nm; re-validated on `XI130505L56A01`, firmware v2.8.2, with updated `UJ_BR_2403` inertials: 20/20 PASS, L2 torque error 0.029–0.183 Nm, mean 0.097 Nm), one xArm 7 (firmware v2.8.2; 20/20 PASS, L2 torque error 0.816–3.303 Nm), one Lite 6 (firmware v2.8.2; 20/20 PASS, L2 torque error 0.028–0.156 Nm), and one UF850 (firmware v2.8.100; 20/20 PASS, L2 torque error 0.086–0.569 Nm); hardware task modes (`pick-place` / `box-pack`) were validated on one xArm 6 (v2.8.0), and the 3 kg payload static torque comparison (≤ 2.5 %) is a single historical measurement.
- The vacuum grippers (`lite6_vacuum_gripper`, `vacuum_gripper` and their arm combinations) are static models without a suction model. They, standalone cameras, `camera_stand*`, `<arm>_with_d435`, `ft_sensor`, `<arm>_with_ft_sensor`, `*_base_plate`, and `<arm>_with_base_plate` have no jaw actuator, so `04_gripper_control.py` rejects them (exit code `2`).
- Depth camera and `camera_stand` models provide visual geometry, mounting/screw-hole `<site>`s (`group=4`), and approximate collision geometry (`group=3`) with `mass=0` and no MuJoCo `<camera>` optical sensor; inject camera/bracket inertia via `attach_payload` for dynamics tasks.
- The `ft_sensor` models represent the UFACTORY AI1500 (`0.6 kg` total split into `0.45 kg` fixed base and `0.15 kg` floating output head at `CoM +3.87 mm` ahead of the `+32.0 mm` sensing site `ft_sensor_site`, `+58.6 mm` output face, `1 ms` physics / `200 Hz` IEEE-754 `fp32` sampling, `±400 N / ±20 Nm` overrange, `±600 N / ±30 Nm` overload) with real-hardware polarity (`+` = external wrench applied onto the output face), dual `raw` (floating head + downstream gravity) and `ext` (dynamically gravity- and zero-compensated) channels, optional calibrated noise (`noise_std="calibrated"`), and `XArmRobot` read-only hardware bridge + fail-closed `--gripper ft_sensor*` preflight gates (avoiding `SERVO_R16B` queries); hysteresis, crosstalk, thermal drift, and controller low-pass filter dynamics are not modeled. Lite 6 (`lite6`) and Bio Gripper G2 (`bio_gripper_g2`) do not have built `ft_sensor` combinations.
- Backend semantics (`RobotInterface`): `SimRobot.move_tcp` interpolates a straight Cartesian line with SLERP, whereas `XArmRobot.move_tcp` solves endpoint IK and moves in joint space (`XArmRobot.move_joints` is capped by `max_joint_speed_rad_s`, and `SimRobot(..., wait=False)` skips only the trailing `settle`); use `plan_cartesian_trajectory` + `execute_trajectory` when identical straight-line Cartesian paths are required across simulation and hardware. For Gripper G2 / Bio Gripper G2, `speed_percent=0` pauses jaw motion while holding contact preload, and subsequent calls omitting `speed_percent` automatically resume at the last positive speed.
- Mounting base plates (`*_base_plate`) use hole-filled convex collision prisms (`group=3`) for solid mounting-clearance simulation; visual meshes (`group=2`) retain all through-holes and slots, so collision probes are not suitable for simulating bolts passing through plate holes. In `<arm>_with_base_plate`, `link_base` is aligned to the plate's mounting holes/surface via least-squares fitting, introducing millimeter-scale flange/TCP offsets relative to bare-arm origin (`xarm6`: `Δy ≈ -1.47 mm`; `lite6`: `Δx ≈ +2.99 mm`, `Δz ≈ +1.72 mm`).
- Task reports are byte-identical on MuJoCo 4.0.0 and 4.0.1; other MuJoCo versions can change results numerically.

## Project Structure

```text
calibrations/              Per-unit kinematic calibration YAML (serial-bound, SHA-256 fingerprinted)
ufactory_mujoco/
  models/                  Atomic MJCF parts with their meshes, and the xarm6 obstacle scene
  configs/                 Task and validation configs: default.yaml (with per-task/arm presets),
                           dynamics_validation_pose.yaml
  sim_robot.py             SimRobot: physics stepping, joint/TCP/gripper/FT-sensor control, viewer
  xarm_robot.py            XArmRobot: real xArm, read-only unless motion is approved
  robot_interface.py       Common interface of SimRobot and XArmRobot
  robots.py                Model registry data: arms, composite recipes, mounting data
  compose.py               MjSpec.attach assembly, task scenes, model loading and export
  config.py                Strict YAML task configs, presets, and content hashing
  calibration.py           Per-unit calibration, model patching, TCP/flange transforms
  safety.py                Hardware safety gate and sdk-sim replay
  kinematics_validation.py Read-only FK/IK comparison with the controller
  ft_sensor.py             AI1500 6-axis F/T sensor 200 Hz reader and local wrench helper
  g2.py                    Gripper G2 linkage model, force table, and vectorized position/force servo
  bio_g2.py                Bio Gripper G2 calibrated model and rate-limited position/force servo
  grippers.py              Gripper specs, gap mapping, finger contact checks and forces
  preflight.py             Grasp geometry and pick-place/box-pack preflight with RRT-Connect detours
  kinematics.py            FK, Jacobians, manipulability, and DLS IK
  trajectory.py            JointTrajectory, quintic/LSPB, spline, Cartesian line
  planning.py              Collision/clearance queries, RRT-Connect, and path shortcutting
  dynamics/                Mass matrix, gravity/Coriolis, inverse/forward dynamics, payload, and Schema v4 torque validation CLI (`dynamics-*`)
  tasks/                   pick-place, box-pack and dynamics commands (sharing _common.py)
  cli.py, __main__.py      `ufactory-mujoco` / `python -m ufactory_mujoco` entry point
  viewer.py                Passive viewer with Linux GLFW/X11 handling
examples/simulation/       Examples 01–11
scripts/                   Distribution and installed-wheel verification (`accept_installed.py`)
tests/                     Test suite (pytest)
```

## Model Notes

- Kinematics and inertia come from the official UFACTORY definitions ([xarm_ros](https://github.com/xArm-Developer/xarm_ros), [xarm_ros2](https://github.com/xArm-Developer/xarm_ros2), [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis)); xArm 5/6/7 (`>= 1305`) link inertials use the official `UJ_BR_2403` parameter set.
- Actuator parameters follow [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) conventions; Lite 6 joint PD gains are tuned to a uniform `0.04 s` time constant for accurate Cartesian line tracking, while Gripper G2 and Bio Gripper G2 use calibrated servo models.
- Visual meshes (`group=2`, `contype=0 conaffinity=0`) and collision meshes (`group=3`) are kept separate. Camera and bracket mounting frames and screw holes are exposed as `<site>` elements in `group=4` (D435 includes a `0.055 mm` normal clearance so its ROS visual mesh stays outside the bracket plane; mounted `d435` assemblies orient the camera for a human-eye upright view with its USB-C port on flange `+Y` / base `Y-` in tool-down poses).

## License

Code: MIT (see `LICENSE`). Robot and accessory models include assets derived from UFACTORY (`xarm_ros`, BSD-3-Clause; `ufactory_genesis`, MIT; `ufactory_robot_viewer`, BSD-3-Clause), Intel RealSense ROS (`realsense-ros`, Apache-2.0), and published Orbbec / Stereolabs / UFACTORY mechanical CAD; see `NOTICE`.
