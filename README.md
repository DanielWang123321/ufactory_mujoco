# ufactory_mujoco

[English](README.md) | [中文](README_zh.md) · [Changelog](CHANGELOG.md) · [![CI](https://github.com/DanielWang123321/ufactory_mujoco/actions/workflows/ci.yml/badge.svg)](https://github.com/DanielWang123321/ufactory_mujoco/actions/workflows/ci.yml)

MuJoCo simulation suite for UFACTORY robots (Lite 6, xArm 5/6/7, UF850 and their grippers): detailed models, kinematics and dynamics tools, motion planning, scripted manipulation tasks, and a gated interface to a real xArm.

## Quick Start

```bash
git clone --depth 1 https://github.com/DanielWang123321/ufactory_mujoco.git && cd ufactory_mujoco
python -m pip install -e .
python examples/simulation/01_load_and_view.py --model xarm6_with_gripper --pose ready
```

Without a display, run a task headless: `ufactory-mujoco pick-place --robot xarm6 --headless`.

## Features

- **Models**: all five arms and five end effectors (Gripper G2, Bio Gripper G2, the Lite 6 gripper, and the xArm and Lite 6 vacuum grippers) converted from [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis), plus four depth cameras (`d405`, `d435`, `gemini_305`, `zed_x_nano`), the UFACTORY `camera_stand`, and their bracket / `xarm6` + Gripper G2 assemblies, with separate visual meshes and multi-convex collision shapes. Gripper G2 uses a calibrated 4-bar linkage, a measured force table, and a rate-limited servo (`g2_model.py`, `g2_controller.py`). Bio Gripper G2 uses calibrated 0.79 kg dynamics and a rate-limited position/force servo (`bio_g2_model.py`, `bio_g2_controller.py`).
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

Every arm, gripper, depth camera (`d405`, `d435`, `gemini_305`, `zed_x_nano`), and bracket assembly (`camera_stand`, `camera_stand_with_<camera>`, `xarm6_with_gripper_and_<camera>`) can also be loaded with `01_load_and_view.py --model <name>`. Model names and commands are listed in [examples/simulation/README.md](examples/simulation/README.md#4-commands-by-model).

## Installation

Python **3.12–3.14** and MuJoCo **≥ 4.0, < 5** (tested with 4.0.0 and 4.0.1; MuJoCo 3 is no longer supported from 0.1.5). The meshes are large, so clone with `--depth 1`.

```bash
python3.13 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e .       # package + core dependencies (mujoco, numpy, scipy, pyyaml)
```

| Extra | Install | Needed for |
| :--- | :--- | :--- |
| `test` | `pip install -e ".[test]"` | the test suite |
| `assets` | `pip install -e ".[assets]"` | asset regeneration only (see [models/gripper_g2/source/README.md](models/gripper_g2/source/README.md)) |

`requirements-lock.txt` pins the versions CI tests with (MuJoCo 4.0.0).

**Upgrading an existing environment from MuJoCo 3 to MuJoCo 4**: MuJoCo 4 splits its Python package into `mujoco-core`, `mujoco-studio` and `mujoco`. Upgrading in place (`pip install -U mujoco`) leaves a broken installation, and `import mujoco` then fails with an error about missing `mujoco-core` bindings. Uninstall first:

```bash
pip uninstall -y mujoco && pip install "mujoco>=4,<5"
```

If the environment is already broken, repair it with `pip install --force-reinstall --no-deps mujoco mujoco-core mujoco-studio`. New environments are not affected.

**Wheel**: [GitHub Releases](https://github.com/DanielWang123321/ufactory_mujoco/releases) from v0.1.0 on include a wheel with all models, scenes, task configs and licenses (about 112 MiB download, 630 MiB of resources once installed). Install it with `pip install ufactory_mujoco-0.1.4-py3-none-any.whl`; the `ufactory-mujoco` commands then run from any directory. Build one yourself with `python -m build`.

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
| `UFACTORY_GENESIS_ROOT` | `ufactory_genesis` checkout used by the asset conversion script |

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

Tests that need optional packages (`trimesh`, `DracoPy`/`pygltflib`) are skipped when those are missing.

## Known Limitations

- Hardware modes were validated on one xArm 6 (firmware v2.8.0). The static torque comparison (≤ 2.5 % with a 3 kg payload) is a single historical measurement.
- The vacuum grippers (`lite6_vacuum_gripper`, `vacuum_gripper` and their arm combinations) are static models without a suction model. They, standalone cameras, and `camera_stand*` have no jaw actuator, so `04_gripper_control.py` rejects them (exit code `2`).
- Depth camera and `camera_stand` models provide visual geometry, mounting/screw-hole `<site>`s (`group=4`), and approximate collision geometry (`group=3`) with `mass=0` and no MuJoCo `<camera>` optical sensor; inject camera/bracket inertia via `attach_payload` for dynamics tasks.
- Task reports are byte-identical on MuJoCo 4.0.0 and 4.0.1; other MuJoCo versions can change results numerically.

## Project Structure

```text
calibrations/              Per-unit kinematic calibration YAML (SHA-256 checked)
configs/                   Task configs: default.yaml, pick_place/<arm>.yaml, box_pack/<arm>.yaml
models/                    MJCF models and meshes: arms, grippers, cameras, brackets and combinations;
                           scene variants scene, scene_reach, scene_pick_place, scene_box_pack, scene_obstacles
ufactory_mujoco/
  sim_robot.py             SimRobot: physics stepping, joint/TCP/gripper control, viewer
  xarm_robot.py            XArmRobot: real xArm, read-only unless motion is approved
  robot_interface.py       Common interface of SimRobot and XArmRobot
  robots.py                Registry of the five arms, gripper combinations, reach workspaces
  config.py                Strict YAML task configs and their content hash
  calibration.py           Per-unit calibration, model patching, TCP/flange transforms
  safety.py                Hardware safety gate and sdk-sim replay
  kinematics_validation.py Read-only FK/IK comparison with the controller
  g2_model.py              Gripper G2 linkage geometry, force table, SDK unit conversions
  g2_controller.py         Gripper G2 servo (vectorized and scalar forms)
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
  resource_paths.py, model_paths.py  Resource roots and model name lookup
examples/simulation/       Examples 01–10
scripts/                   Asset conversion and MJCF generation, calibration and hardware
                           comparison, packaging checks
tests/                     Test suite (pytest)
```

## Model Notes

- Kinematics and inertia come from the official UFACTORY definitions ([xarm_ros](https://github.com/xArm-Developer/xarm_ros), [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis)).
- Actuator parameters follow [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) conventions.
- Visual meshes (`group=2`, `contype=0 conaffinity=0`) and collision meshes (`group=3`) are kept separate. Camera and bracket mounting frames and screw holes are exposed as `<site>` elements in `group=4` (D435 includes a `0.055 mm` normal clearance so its ROS visual mesh stays outside the bracket plane).

## License

Code: MIT (see `LICENSE`). Robot and camera models include assets derived from UFACTORY (`xarm_ros`, BSD-3-Clause; `ufactory_genesis`, MIT), Intel RealSense ROS (`realsense-ros`, Apache-2.0), and published Orbbec / Stereolabs / UFACTORY mechanical CAD; see `NOTICE`.
