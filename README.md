# ufactory_mujoco

![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue) ![MuJoCo](https://img.shields.io/badge/mujoco-4.x-lightgrey) ![License](https://img.shields.io/badge/license-MIT-green) [![CI](https://github.com/DanielWang123321/ufactory_mujoco/actions/workflows/ci.yml/badge.svg)](https://github.com/DanielWang123321/ufactory_mujoco/actions/workflows/ci.yml)

[English](README.md) | [中文](README_zh.md) · [Changelog](CHANGELOG.md)

MuJoCo simulation suite for UFACTORY robots (Lite 6, xArm 5/6/7, UF850), grippers, and depth cameras: detailed MJCF models, kinematics and dynamics tools, motion planning, scripted manipulation tasks, and a gated interface to a real xArm.

## Quick Start

Requires Python **3.12–3.14** (see [Installation](#installation)).

```bash
git clone --depth 1 https://github.com/DanielWang123321/ufactory_mujoco.git && cd ufactory_mujoco
python -m pip install -e .
python examples/simulation/01_load_and_view.py --model xarm6_with_gripper --pose ready
```

Without a display, run a task headless: `ufactory-mujoco pick-place --robot xarm6 --headless`.

## Features

- **Models**: all five arms, five end effectors, four depth cameras, mounting bracket, and combined MJCF assemblies with separate visual meshes and multi-convex collision shapes. Gripper G2 and Bio Gripper G2 use calibrated dynamics and rate-limited position/force servos (`g2_model.py`, `g2_controller.py`, `bio_g2_model.py`, `bio_g2_controller.py`).
- **Kinematics and planning**: FK, Jacobians, damped least-squares IK, quintic/LSPB/spline trajectories, Cartesian lines, collision checking, and RRT-Connect (`kinematics.py`, `ik.py`, `trajectory.py`, `collision.py`, `path_planning.py`).
- **Dynamics**: mass matrix, gravity/Coriolis, inverse/forward dynamics, payloads, and a five-arm validation command (`dynamics.py`, `tasks/dynamics.py`).
- **Scripted tasks**: cube pick-and-place and box packing for five arm/gripper pairs, with preflight (joint limits, penetration, RRT-Connect detours) and JSON reports (`tasks/`, `preflight.py`).
- **Real robot**: `XArmRobot` is read-only by default; motion needs a per-unit calibration, controller checks, and `--confirm-real`. `sdk-sim` replays trajectories in the controller's built-in simulator (`xarm_robot.py`, `calibration.py`, `safety.py`).

## Supported Robots and Tasks

| Arm | Built gripper models | `pick-place` / `box-pack` | `dynamics` |
| :--- | :--- | :--- | :---: |
| `lite6` | `lite6_gripper`, `lite6_vacuum_gripper` | with `lite6_gripper` | yes |
| `xarm5` | `gripper_g2`, `bio_gripper_g2`, `vacuum_gripper` | with `gripper_g2` | yes |
| `xarm6` | `gripper_g2`, `bio_gripper_g2`, `vacuum_gripper` | with `gripper_g2` | yes |
| `xarm7` | `gripper_g2`, `bio_gripper_g2`, `vacuum_gripper` | with `gripper_g2` | yes |
| `uf850` | `gripper_g2`, `bio_gripper_g2`, `vacuum_gripper` | with `gripper_g2` | yes |

The repository ships **44 model directories** under `models/`: 5 bare arms, 5 standalone end effectors, 4 depth cameras (`d405`, `d435`, `gemini_305`, `zed_x_nano`), 5 `camera_stand*` models, 14 arm + end-effector combinations, and 11 arm + camera combinations (`xarm6_with_gripper_and_<camera>` plus `<arm>_with_d435` and `<arm>_with_gripper_and_d435` across `xarm5/6/7` and `uf850`). Load any of them with `01_load_and_view.py --model <name>`; model names and commands are listed in [examples/simulation/README.md](examples/simulation/README.md#4-commands-by-model).

## Installation

| Component | Supported range | Verified versions | Notes |
| :--- | :--- | :--- | :--- |
| **Python** | **3.12, 3.13, 3.14** (`>=3.12`) | `3.12.13`, `3.13.13`, `3.14.6` | Covered by CI and release checks; **Python ≤ 3.11 is unsupported**; Python ≥ 3.15 is untested |
| **MuJoCo** | **≥ 4.0, < 5** | `4.0.0`, `4.0.1` | MuJoCo 3.x is no longer supported from `v0.1.5` |
| Core deps | `numpy>=2.0,<3`, `scipy>=1.14,<2`, `pyyaml>=6.0,<7` | see `requirements-lock.txt` | Installed automatically with `pip install -e .` |
| `.[test]` (optional) | `pytest>=8` | — | Needed only to run the unit test suite |
| `xarm-python-sdk` (optional) | matching controller SDK | — | Needed only for `--mode sdk-sim` / `real` / `real-read-only` and `scripts/gen_kinematics_params.py` |

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

**Wheel**: [GitHub Releases](https://github.com/DanielWang123321/ufactory_mujoco/releases) from v0.1.0 on include a self-contained wheel with all models, scenes, task configs and licenses (about 112 MiB download, 630 MiB of resources once installed). Install it with `pip install ufactory_mujoco-0.1.5-py3-none-any.whl`; the `ufactory-mujoco` commands then run from any directory. Build one yourself with `python -m build`.

## Command-Line Tasks

```bash
ufactory-mujoco pick-place --robot xarm6 --headless
ufactory-mujoco box-pack --robot lite6 --cycles 3 --headless
ufactory-mujoco dynamics --robot all --headless --report reports/dynamics.json
```

- `python -m ufactory_mujoco <command>` is equivalent. Examples `05`, `08` and `10` run the same code.
- `COMMAND --help` lists the options. The full table is in [examples/simulation/README.md](examples/simulation/README.md#3-cli-options-summary).
- `--dry-run` only runs the preflight and never steps physics.
- Exit codes: `0` success, `1` runtime, preflight, safety-gate or report failure, `2` invalid arguments or configuration.
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
| Simulation examples `01`–`10`, CLI options, commands by model, viewer controls, hardware commands | [examples/simulation/README.md](examples/simulation/README.md) ([中文](examples/simulation/README_zh.md)) |
| Contributing, security policy, code of conduct | [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) |
| Release history | [CHANGELOG.md](CHANGELOG.md) |

## Running Tests

```bash
python -m pip install -e ".[test]"
python -m pytest tests
ruff check .                      # lint as in CI (pip install ruff)
```

Tests that need optional packages (`trimesh`) or maintainer-only source snapshots are skipped automatically when absent.

## Known Limitations

- Hardware modes were validated on one xArm 6 (firmware v2.8.0). The static torque comparison (≤ 2.5 % with a 3 kg payload) is a single historical measurement.
- The vacuum grippers (`lite6_vacuum_gripper`, `vacuum_gripper` and their arm combinations) are static models without a suction model. They, standalone cameras, `camera_stand*`, and `<arm>_with_d435` have no jaw actuator, so `04_gripper_control.py` rejects them (exit code `2`).
- Depth camera and `camera_stand` models provide visual geometry, mounting/screw-hole `<site>`s (`group=4`), and approximate collision geometry (`group=3`) with `mass=0` and no MuJoCo `<camera>` optical sensor; inject camera/bracket inertia via `attach_payload` for dynamics tasks.
- Task reports are byte-identical on MuJoCo 4.0.0 and 4.0.1; other MuJoCo versions can change results numerically.

## Project Structure

```text
calibrations/              Per-unit kinematic calibration YAML (SHA-256 checked)
configs/                   Task configs: default.yaml, pick_place/<arm>.yaml, box_pack/<arm>.yaml
models/                    MJCF models and meshes: arms, grippers, cameras, brackets and combinations;
                           scene variants scene, scene_pick_place, scene_box_pack, scene_obstacles
ufactory_mujoco/
  sim_robot.py             SimRobot: physics stepping, joint/TCP/gripper control, viewer
  xarm_robot.py            XArmRobot: real xArm, read-only unless motion is approved
  robot_interface.py       Common interface of SimRobot and XArmRobot
  robots.py                Registry of the five arms and compatible grippers
  config.py                Strict YAML task configs and their content hash
  calibration.py           Per-unit calibration, model patching, TCP/flange transforms
  safety.py                Hardware safety gate and sdk-sim replay
  kinematics_validation.py Read-only FK/IK comparison with the controller
  g2_model.py              Gripper G2 linkage geometry, force table, SDK unit conversions
  g2_controller.py         Gripper G2 vectorized position/force servo
  bio_g2_model.py          Bio Gripper G2 calibrated gap, speed and force model
  bio_g2_controller.py     Bio Gripper G2 rate-limited position/force servo
  grippers.py              Gripper specs, gap mapping, finger contact checks and forces
  preflight.py             Pick-place/box-pack preflight with RRT-Connect detours
  task_geometry.py         Grasp heights and orientation shared by tasks and preflight
  ik.py, kinematics.py     DLS IK; FK, Jacobians, manipulability
  trajectory.py            JointTrajectory, quintic/LSPB, spline, Cartesian line
  dynamics.py              Mass matrix, gravity/Coriolis, inverse/forward dynamics, payload
  collision.py             Collision and clearance queries
  path_planning.py         RRT-Connect and shortcutting
  tasks/                   pick-place, box-pack and dynamics commands
  cli.py, __main__.py      `ufactory-mujoco` / `python -m ufactory_mujoco` entry point
  viewer.py                Passive viewer with Linux GLFW/X11 handling
  model_paths.py           Resource roots and model name lookup
examples/simulation/       Examples 01–10
scripts/                   Calibration extraction, distribution and installed-wheel checks
tests/                     Test suite (pytest)
```

## Model Notes

- Kinematics and inertia come from the official UFACTORY definitions ([xarm_ros](https://github.com/xArm-Developer/xarm_ros), [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis)).
- Actuator parameters follow [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) conventions; Lite 6 joint PD gains are tuned to a uniform `0.04 s` time constant for accurate Cartesian line tracking, while Gripper G2 and Bio Gripper G2 use calibrated servo models.
- Visual meshes (`group=2`, `contype=0 conaffinity=0`) and collision meshes (`group=3`) are kept separate. Camera and bracket mounting frames and screw holes are exposed as `<site>` elements in `group=4` (D435 includes a `0.055 mm` normal clearance so its ROS visual mesh stays outside the bracket plane; mounted `d435` assemblies orient the camera for a human-eye upright view with its USB-C port on flange `+Y` / base `Y-` in tool-down poses).

## License

Code: MIT (see `LICENSE`). Robot and camera models include assets derived from UFACTORY (`xarm_ros`, BSD-3-Clause; `ufactory_genesis`, MIT), Intel RealSense ROS (`realsense-ros`, Apache-2.0), and published Orbbec / Stereolabs / UFACTORY mechanical CAD; see `NOTICE`.
