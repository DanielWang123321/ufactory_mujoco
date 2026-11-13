# Changelog

## [0.1.4] - 2026-11-13

Cleanup and governance release. It removes the reinforcement-learning part and publishes the 0.1.3 asset changes, which were not released on their own. Models, meshes, task configs and task reports are byte-identical to 0.1.3.

### Added

- `CONTRIBUTING.md`, `SECURITY.md` and `CODE_OF_CONDUCT.md`.
- "Commands by Model" and "Viewer Controls" sections in the simulation READMEs, in English and Chinese. They replace the Chinese-only `docs/visual_command_guide.md`.

### Changed

- Dependencies are declared only in `pyproject.toml`. `requirements.txt`, `requirements-rl.txt` and `requirements-assets.txt` are removed; the two lock files remain. The `assets` extra is now `trimesh`, `coacd` and `pymeshlab` (`xacro` is no longer used).
- `scripts/accept_installed.py` and the CI smoke tests cover the Bio Gripper G2 and vacuum gripper models.

### Removed

- Reinforcement learning: the `ufactory_mujoco.rl` package (reach and pick-and-place environments, checkpoint contracts), `examples/rl/`, the scripts `train_reach_rl.py`, `train_pick_place_rl.py` and `measure_reach_workspace.py`, the `rl` extra, the RL tests and the `rl` CI job. The scripted pick-and-place and box-packing tasks are not affected.
- `scripts/build_model.py`, an unreferenced script that drafted the xArm 6 MJCF from `xarm_ros`.

### Fixed

- `02_joint_control.py` started combinations without their own preset (Bio Gripper G2, vacuum gripper and camera combinations) from the bare arm's all-zero pose, with the tool inside the floor, and its UF850 demo pose pushed Gripper G2 47 mm into the floor. Combinations now start from their own `ready` keyframe and the UF850 demo pose is raised.

## [0.1.3] - 2026-11-12

Asset synchronization with ufactory_genesis v0.3.6, released as part of 0.1.4. Collision geometry, Bio Gripper G2 and Lite 6 gripper dynamics and all visual meshes changed, so task reports differ from 0.1.2.

### Added

- xArm vacuum gripper (`vacuum_gripper`, 0.61 kg, 33 convex collision pieces) and its combinations `xarm5/xarm6/xarm7/uf850_with_vacuum_gripper`.
- Bio Gripper G2 combinations for xArm 5, xArm 7 and UF850 (`<arm>_with_bio_gripper`).
- Calibrated Bio Gripper G2 model and servo (`bio_g2_model.py`, `bio_g2_controller.py`): 71–150 mm gap with speed and force settings, used by `SimRobot`, `XArmRobot` and `04_gripper_control.py`.

### Changed

- Arm collision geometry: 111 convex STL pieces (17–27 per arm) replace one convex hull per link, and `link_base` has explicit inertial parameters.
- Gripper G2: the base collision shape is split into 10 convex pieces and the locating pin stays within its 3 mm radius. This removes the overlap with the camera bracket.
- Bio Gripper G2: calibrated 0.79 kg dynamics (was 0.4125 kg) and 5 convex collision pieces.
- Lite 6 gripper: jaw gap 20–38 mm, two-box finger collision shapes and a ±3 N drive (`kp 700`, `kv 20`). The Lite 6 pick-and-place and box-packing tasks use a 30 mm, 17 g cube, and box packing also passes with `servo_j`.
- Visual OBJ meshes were decimated with a crease-preserving quadric edge collapse, from 548.92 MB to 89.54 MB (5.64 M to 1.34 M faces). The wheel shrinks from 102 MiB to 36 MiB and its installed resources from 566 MiB to 128 MiB. A full test run now peaks at about 3 GB of memory instead of 7 GB.

## [0.1.2] - 2026-11-11

Depth camera and camera bracket models. Existing robot XMLs, meshes, task configs, task reports, and RL checkpoint hashes are byte-identical to 0.1.1.

### Added

- Four depth camera models (`d405`, `d435`, `gemini_305`, `zed_x_nano`), the UFACTORY mounting bracket (`camera_stand`), four bracket assemblies (`camera_stand_with_{d405,d435,gemini_305,zed_x_nano}`), and four xArm 6 + Gripper G2 + bracket + camera assemblies (`xarm6_with_gripper_and_{d405,d435,gemini_305,zed_x_nano}`), generated via `scripts/build_camera_mjcf.py` (13 new model directories, 26 XMLs, 189 canonical STL meshes).
- Split anodized-aluminum housing and glossy optical-glass visual meshes (`group=2`), box and 175-piece convex-prism collision geometry preserving the bracket's 34.5 mm center bore (`group=3`, `mass=0`), and mounting / screw-hole `<site>` frames (`group=4`).
- `licenses/realsense_ros.LICENSE` (Apache-2.0 for RealSense D405/D435 visual meshes) and updated `NOTICE` / `pyproject.toml` license metadata.
- `tests/test_camera_models.py` verifying binary STL bounds against vendor specs, collision probes, screw-hole alignment (`<= 0.055 mm`), visual-mesh mounting-plane clearance, flange clearance partitioning, and full-stroke Gripper G2 finger clearance.

### Fixed

- `SimRobot` now resets keyframeless passive models (`nkey == 0`) with `mj_resetData` instead of raising `FatalError: invalid keyframe index 0`, so `04_gripper_control.py` cleanly exits with code `2` on passive accessories.

## [0.1.1] - 2026-11-11

MuJoCo 4.0 support: no new features and no change to the simulation code. On MuJoCo 3.15 and 3.16, task reports are byte-identical to 0.1.0.

### Changed

- The MuJoCo requirement is `>=3.15,<5` (was `>=3.15,<4`). Tested with 3.15.0, 3.16.0, 3.17.0 and 4.0.0 on Python 3.12–3.14.
- `requirements-lock.txt` pins MuJoCo 4.0.0 (`mujoco`, `mujoco-core`, `mujoco-studio`). `requirements-lock-mujoco3.txt` holds the MuJoCo 3.16.0 set. CI runs the tests with both.
- RL benchmarks now come from policies retrained on MuJoCo 4.0.0 with the published recipes: reach succeeds on 97.1–100 % of two 512-target groups per arm, pick-and-place on 1000 of 1000 held-out episodes. Checkpoints are interchangeable between MuJoCo 3.16 and 4.0: those trained on MuJoCo 3 keep their success rates on 4.0.0 (pick-and-place 98.5 % against 98.1 %).
- Results on MuJoCo 4.0.0 differ numerically from 3.16.0 in 19 of the 30 reference task reports (placement error by at most 1.0 mm). Exit codes and task outcomes are the same, and the scripted expert still succeeds in 64 of 64 episodes.

### Fixed

- `provenance.json` records the MuJoCo version when only `mujoco-core` is installed.
- CI steps that pipe their output through `tee` now fail when the command fails.

### Documentation

- README: how to upgrade an existing environment from MuJoCo 3 to MuJoCo 4 (uninstall first), the two lock files, and a note that results differ between MuJoCo versions.

## [0.1.0] - 2026-11-10

Cleanup release: no new features. Simulation behaviour, task reports and checkpoint hashes are unchanged.

### Removed

- `scripts/migrate_pick_place_checkpoint.py` and the v1 checkpoint migration path. v1 pick-and-place checkpoints (46-dim observations) are rejected; retrain with the current recipe.
- `scripts/seed_sensitivity_experiment.py` (server-specific experiment driver).
- Deprecated aliases and unused helpers: `XArm6ReachEnv` and `XArm6PickPlaceEnv` (use `ReachEnv` and `PickPlaceEnv`), `rl.TARGET_LOW`/`TARGET_HIGH`, `ForceCurvePoint`, `gap_to_drive`, `calibration.sha256_file`, `gripper_g2_gap_m_to_sdk_pos_mm`, `GRIPPER_G2_JOINT_VELOCITY_RAD_S`, `rl.write_artifact_inventory`.
- Internal planning notes (`docs/internal/`) are no longer tracked.

### Changed

- RL evaluation helpers moved to `ufactory_mujoco.rl.evaluation` (new: `pick_place_env_kwargs`). `examples/rl/*/evaluate.py` are thin command-line wrappers and still re-export these names.
- Shared helpers: `tasks/_motion.py` (pick-place and box-pack motion), `_hashing.py` (SHA-256), `grippers.GRIPPER_ACTUATOR_NAMES`.
- Runtime error messages are in English. Preflight phase names, which are report keys, are unchanged.
- `ScalarG2Controller` is exported from `ufactory_mujoco`.
- Tests run with `pytest` (`pip install -e ".[test]"`); `tests/test_batch9_hardware_safety.py` is now `tests/test_hardware_safety.py`. Ruff settings live in `pyproject.toml`, and CI has a lint job.
- RL benchmarks were re-measured on MuJoCo 3.16.0 with the same checkpoints; success rates moved by at most 0.2 percentage points.

### Fixed

- `SimRobot()` without a model path loads `MODELS_ROOT/xarm6/scene.xml`, so it works from an installed wheel and honours `UFACTORY_MUJOCO_MODELS`.
- Documentation errors: Gripper G2 speed/force defaults (100 %), `--ip` default (`XARM_IP`), viewer shortcuts, `01 --headless` behaviour, reach observation sizes.

### Documentation

- READMEs rewritten with a quick start, supported robots and tasks, environment variables, how to obtain RL weights (they are not distributed), and known limitations. Option tables added for the RL scripts.

## [0.0.9] - 2026-10-30

- Calibrated Gripper G2 linkage, force table and servo; updated G2 assets for the standalone gripper and the four arm combinations.
- RL pick-and-place v2: model-hash contract, 70-dim observations, randomized spawns, pure PPO trained from scratch.

## [0.0.8] - 2026-10-28

- Five-arm dynamics validation (`ufactory-mujoco dynamics`), self-contained wheel and sdist with the `ufactory-mujoco` commands, CI on Python 3.12–3.14.
- RL recipes and checkpoint contracts, first RL pick-and-place, `XArmRobot` with per-unit calibration, `sdk-sim` and the `--confirm-real` safety gate.

## [0.0.7] - 2026-10-21

- RL reaching for all five arms with the 5 mm success criterion.
- Arm registry and strict YAML task configs, gripper gap commands with bilateral contact checks, five-arm pick-and-place with preflight and `--dry-run`, five-arm box packing.

## [0.0.6] - 2026-10-15

- Examples split into `simulation/` and `rl/`; kinematics, dynamics and RRT-Connect obstacle-avoidance examples.

## [0.0.5] - 2026-10-14

- RL TCP reaching for xArm 6 (Gymnasium environment and PPO training).

## [0.0.4] - 2026-10-13

- Trajectory planning, xArm 6 + Gripper G2 pick-and-place, dynamics validation.

## [0.0.3] - 2026-10-12

- Rigid-body dynamics module and a static cross-check against a real xArm 6 with a 3 kg payload.

## [0.0.2] - 2026-10-12

- Movable Gripper G2 and Bio Gripper G2 models.

## [0.0.1] - 2026-10-10

- MJCF models for Lite 6, xArm 5/6/7, UF850 and their grippers.
