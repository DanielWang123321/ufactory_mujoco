# Changelog

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
