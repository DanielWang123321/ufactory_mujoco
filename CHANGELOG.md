# Changelog

## [0.1.7] - 2026-11-19

Integrated the UFACTORY AI1500 6-axis force/torque sensor (`ft_sensor`) and three official mounting base plates (`*_base_plate`), expanding `models/` from 44 to 65 model directories (141 XMLs, 482 canonical meshes) with zero modifications to existing models, and hardened hardware safety gates, trajectory/kinematics planners, API input validation, and package governance.

### Added

- **Force/Torque Sensor (`ft_sensor`)**: 13 new MJCF model directories (`ft_sensor` plus `{xarm5,xarm6,xarm7,uf850}_with_ft_sensor`, `*_with_ft_sensor_and_gripper`, and `*_with_ft_sensor_and_vacuum_gripper`) using 15 double-sided visual OBJs (`inertia="shell"`), stainless-steel adapter ring (`part_02`) and fastener materials, `<headlight specular="0.55 0.57 0.6"/>` scene lighting, and a convex collision STL hull (total `mass=0.6 kg` split into `0.45 kg` base + `0.15 kg` floating output head at `CoM +3.87 mm` ahead of the `+32.0 mm` sensing site `ft_sensor_site`, `+58.6 mm` output face, `timestep="0.001"`).
- **Sensor Reader & `SimRobot` / `XArmRobot` Integration**: `ufactory_mujoco.ft_sensor` (`AI1500Reader`, `FTSensorReader`, `SensorReading`, `apply_ft_sensor_wrench`, `clear_ft_sensor_wrench`, `is_hardware_ft_sensor_enabled`, `read_hardware_ft_sensor_config`, `read_hardware_ft_sensor`), re-exported from `ufactory_mujoco`, providing `200 Hz` (`5`-step) IEEE-754 `fp32` sampling, real-hardware `+applied` polarity, dual `raw` / `ext` (dynamically gravity- and zero-compensated) channels, optional hardware-calibrated noise (`noise_std="calibrated"`, `RAW_NOISE_STD`, `EXT_NOISE_STD`), and `±400 N / ±20 Nm` overrange / `±600 N / ±30 Nm` overload tracking. `SimRobot` attaches `AI1500Reader` automatically (`has_ft_sensor`, `ft_sensor`, `ft_sensor_reading`, `get_ft_sensor_reading()`, `reset_ft_sensor()`, `zero_ft_sensor()`); `XArmRobot` adds read-only F/T config/reading APIs (avoiding `SERVO_R16B` state-reset queries) and fail-closed bare-arm preflight/motion gates across `XArmRobot`, `tasks.dynamics`, and `dynamics` CLI when a controller reports an enabled F/T sensor.
- **Simulation Example 11**: `examples/simulation/11_force_torque_sensor.py` demonstrating dual-channel (`raw`/`ext`) baseline and `+applied` wrench verification across all 13 `ft_sensor` models plus strictly read-only live hardware F/T sampling (`--ip <ROBOT_IP> --gripper ft_sensor`).
- **Mounting Base Plates (`base_plate`)**: 8 new MJCF model directories (`{xarm,uf850,lite6}_base_plate` and `{xarm5,xarm6,xarm7,uf850,lite6}_with_base_plate`) with CAD-derived visual STLs (`group=2`), hole-filled convex collision prism STLs (`group=3`), top mounting planes at `z=0`, scene floors at `-thickness` (`-8 / -10 / -5 mm`), long edges parallel to base `Y` (with the two-chamfered long side of `xarm_base_plate` facing base `+X`), and least-squares mounting-hole alignments (`< 0.01 mm` fit residual).
- **Public API, Hardware Telemetry & Safety Controls**: exported `__version__` (`"0.1.7"`), `quat_error`, and `clear_ft_sensor_wrench` from `ufactory_mujoco`; exposed `SimRobot` public helpers (`step`, `step_hook`, `n_arm_actuators`, `has_viewer`, `viewer_is_running`, `sync_viewer`, `gripper_actuator_id`, `gripper_ctrl_range`, `gripper_qpos`, `gripper_wait_duration`); added `ControllerConfigSnapshot` / `restore_controller_config()`, `--no-return-home`, and `--gripper` hardware preflight support across `dynamics-hardware-check` and `ufactory-mujoco dynamics`; and added real-gripper telemetry fields (`gripper_actuated`, `commanded_grasp_gap_m`, `measured_grasp_gap_m`, `object_grasped_inferred`) to `--mode real` task reports.
- **Licenses & Tests**: added `licenses/ufactory_robot_viewer.LICENSE` (BSD-3-Clause) and Sections 5–6 in `NOTICE`; added `tests/test_ft_sensor.py`, `tests/test_base_plates.py`, and regression tests across `test_hardware_safety.py`, `test_dynamics_suite.py`, `test_trajectory.py`, `test_kinematics.py`, `test_preflight.py`, and `test_installable_tasks.py`.

### Changed

- Updated `pyproject.toml` version to `0.1.7` with `[tool.pytest.ini_options] pythonpath = ["."]` and bumped `ufactory_mujoco/__init__.py:__version__` to `"0.1.7"`, updated `ufactory_mujoco/robots.py` (`{xarm5,xarm6,xarm7,uf850}` F/T gripper registries while keeping `_PICK_PLACE_DIRS` unchanged), and updated `tests/test_all_models.py` (`65` models and `ft_sensor*` downward `ready` checks) and `tests/test_converted_assets.py` (`DOUBLE_SIDED_COUNT = 51`).
- Modularized `ufactory_mujoco/dynamics/cli.py:cli_hardware_check` (reducing cyclomatic complexity and deduplicating `DynamicsSample` construction), added DOF-mismatch validation (`exit 2`) in `dynamics-report-compare`, added a `141`-XML zero-warning compilation step in `.github/workflows/ci.yml`, and updated `scripts/accept_installed.py` (`>= 141` XMLs, `>= 482` meshes, `7` licenses, read-only checks for `xarm6_with_ft_sensor_and_gripper` / `xarm6_with_base_plate`, and directory/file permission restoration in `finally`).
- Updated simulation examples (`02`, `03`, `04`, `06`, `07`, `09`, `11`) to use public `SimRobot` interfaces, CWD-independent `MODELS_ROOT` paths, explicit CLI argument validation, and non-zero exit self-checks; updated `README.md`, `README_zh.md`, `examples/simulation/README.md`, and `examples/simulation/README_zh.md`.
- Note on golden task reports: all 10 `pick-place` and `box-pack` `--dry-run` preflight reports are byte-identical to `0.1.6`, and all 30 task reports remain byte-identical across Python `3.12`/`3.13`/`3.14` and MuJoCo `4.0.0`/`4.0.1`; the 20 simulation task reports reflect minor numeric refinement (`<= 0.5 mm`, e.g. `xarm7` `pick-place` `place_error_mm` improved from `3.02 mm` to `2.65 mm`) from the body-frame SLERP and deterministic `qpos0` IK seeding fixes below.

### Fixed

- **Hardware safety gates**: `tasks.dynamics` hardware preflight and shadow streams now start from measured `q_now` instead of all-zero `home` and block unmodeled non-zero `tcp_load`/`tcp_offset` or enabled F/T sensors when `--gripper` is omitted; `dynamics-hardware-check --hold-current-only` is strictly read-only (mutually exclusive with `--dry-run`/`--clear-tcp-offset`); `Ctrl-C` halts immediately via `stop()` (`set_state(4)`) without unpreflighted `return_home`, and normal exit runs collision preflight before returning home; `dynamics-sim-collision-check` requires `--confirm-real`; `XArmRobot` validates joint limits, single-step servo deltas, and trajectory start/peak-velocity bounds, flushes `sdk-sim` queues before leaving simulation mode, and stops motion cleanly in `close()`; `kinematics_validation` avoids firmware `get_inverse_kinematics()` queries that could latch `warn_code=14`; and `preflight` checks non-base floor contacts instead of globally ignoring `"world"`.
- **Planning & kinematics**: `plan_cartesian_line` now uses body-frame right-multiplied `conj(quat0) * quat1` SLERP (eliminating world-frame orientation drift), copies and normalizes input quaternions without mutating caller arrays, and marks adjacent IK jumps (`> 0.5 rad`) with `ok=False`; `plan_ptp_quintic` and `plan_ptp_lspb` reject non-finite parameters and infeasible `duration < t_min`; `JointTrajectory` validates non-empty finite non-decreasing timestamps; `manipulability` restricts singular-value calculation to ancestor-chain DOFs so mounted grippers and free bodies no longer force zero manipulability; `SimRobot` with `q=None` syncs `data.qpos` and `plan_cartesian_trajectory` slices output to `n_arm` columns; `_flange_tcp_offset` seeds IK deterministically at `model.qpos0`; and `preflight` refines collision stepping (`_CHECK_RES = 0.02 rad`).
- **API & CLI validation**: Gripper G2 / Bio Gripper G2 automatically resume their last positive speed after `speed_percent=0` pause commands; `SimRobot` and `ArmDynamics` reject `NaN`/`Inf` and out-of-range inputs; `config.py` rejects `bool` values in numeric YAML fields and merges multi-level overlays; and task CLIs return exit code `2` on `--robot` vs `--config` conflicts, task/scene mismatches, or hardware `dynamics --robot all`.

### Hardware Validation

- **xArm 6 + Gripper G2 + 8 mm base plate** (firmware `v2.8.0`, Python `3.14.6`, MuJoCo `4.0.1`): read-only `kinematics_validation`, `--hold-current-only`, and `dynamics --gripper gripper_g2 --mode real-read-only` (`max static residual 0.658 Nm`, `err/warn = 0/0`); `sdk-sim` `dynamics`, `pick-place`, and `dynamics-sim-collision-check` (`<= 0.000004 rad` physical drift, full controller load/offset/collision config restoration); single-cycle `--mode real` `dynamics` (`arrival_error_rad = 0.0`) and empty-grasp `pick-place` (`21.94 mm` closed / `83.8 mm` open, `final_joint_error_rad = 4.94e-07`, `err/warn = 0/0`); and compound-rotation `plan_cartesian_line` round-trip (`+80 mm Y, +50 mm Z`, `+0.35 rad Z + 0.15 rad X`: `sdk-sim` `2e-06 rad` drift, `--mode real` `0.1882 mm / 0.0435°` outbound and `0.1909 mm / 0.044°` return arrival error, `0.0007 mm` max lateral deviation, `err/warn = 0/0`).
- **xArm 6 + AI1500 F/T Sensor** (firmware `v2.8.0`, Python `3.14.6`, MuJoCo `4.0.1`): 6-pose static gravity calibration (`0.150 kg` floating head mass, `<= 0.013 N / 0.00124 Nm` static residual, `200 Hz` IEEE-754 `fp32` polling, calibrated `RAW_NOISE_STD`/`EXT_NOISE_STD`); bare-arm `--mode sdk-sim` blocked (exit `1`); `--gripper ft_sensor` `real-read-only` (`1.758 Nm` max static delta) and `sdk-sim` (`272` samples, `7.43e-5 rad` max joint error, `0.043 mm` max EE error, `8e-6 rad` physical drift); and `11_force_torque_sensor.py --ip` read-only live sampling (`86` samples over `0.5 s`, `err/warn = 0/0`, `state=2` preserved).

## [0.1.6] - 2026-11-17

Integrated `ufactory_genesis`-aligned Schema v4 dynamics validation and real-robot static torque verification into the public `ufactory_mujoco.dynamics` subpackage and console scripts, enforced a `<= 40 °/s` hardware dynamics speed ceiling, and validated bare-arm hardware runs across all five arm families (xArm 5, xArm 6, xArm 7, Lite 6, and UF850).

### Added

- `ufactory_mujoco.dynamics` subpackage (preserving `ArmDynamics` and `attach_payload` exports at both `ufactory_mujoco` and `ufactory_mujoco.dynamics`), adding Schema v4 validation dataclasses, five-arm 20-pose expansion (`configs/dynamics_validation_pose.yaml`), closed-loop MuJoCo PD-hold probing, independent `MujocoRneaReference` / optional `PinocchioReference` oracles, L1/L2a/L2b/L3a/L3b static diagnostics, PD tracking-saturation gating, `RealRobotSession` safety checks, CSV/JSONL/`_torque.png` reporting (gracefully skipping plots when `matplotlib` is absent), and `python -m ufactory_mujoco.dynamics <subcommand>` CLI.
- Four `ufactory_genesis`-aligned console scripts in `pyproject.toml`: `dynamics-sim-check`, `dynamics-sim-collision-check`, `dynamics-hardware-check`, and `dynamics-report-compare`.
- Shared port-502 calibration extraction helpers in `ufactory_mujoco.calibration` (`robot_name_from_firmware`, `fetch_kinematics_bytes`, `decode_kinematics_payload`, `DegenerateCalibrationError`, `export_kinematics_calibration`), reused by both `scripts/gen_kinematics_params.py` and `python -m ufactory_mujoco.dynamics export-calibration`.
- `tests/test_dynamics_suite.py` covering five-arm 20-pose simulation regressions, Schema v4 cross-repo CSV/JSONL compatibility, mocked `RealRobotSession` safety gates, `<= 40 °/s` speed-ceiling enforcement, `xarm_ros2`-aligned port-502 decode and `xarm5` degeneracy skip tests (`exit 2`), optional `PinocchioReference` cross-engine checks, and `default_validation_poses` joint-limit bounds.

### Changed

- Enforced a hard `<= 40 °/s` (`0.6981 rad/s`, `MAX_DYNAMICS_MOVE_SPEED_RAD_S`) joint speed ceiling via `validate_dynamics_move_speed` across `dynamics-hardware-check`, `dynamics-sim-collision-check`, `RealRobotSession.move_to`, `XArmRobot(..., max_joint_speed_rad_s=...)`, and `ufactory-mujoco dynamics --mode sdk-sim`/`real`.
- Updated `scripts/accept_installed.py` to track the 12th packaged YAML (`configs/dynamics_validation_pose.yaml`) and smoke-test `dynamics-sim-check`, `dynamics-hardware-check --dry-run`, and `dynamics-report-compare` under read-only wheel installations.
- Updated `.gitignore` to ignore local `reports/` and per-unit `calibrations/*.yaml` exports (retaining the tracked `calibrations/xarm6_kinematics_L56A01.yaml` test fixture).

### Fixed

- `ufactory_mujoco.calibration`: aligned `decode_kinematics_payload` directly with `xarm_ros2` (removing the legacy `abs(value) > 1.0 -> /= 1000.0` heuristic) and raised `DegenerateCalibrationError` so `export_kinematics_calibration` skips non-physical `> ±2.0 m` port-502 parallel-axis packets (e.g. `xarm5`) cleanly with exit code `2` and directs users to `--nominal-model` without writing invalid YAMLs.
- `ufactory_mujoco.kinematics_validation`: adjusted `default_validation_poses("lite6")["pose_b"]` (`J3 = 0.6 rad`) to stay within Lite 6's `joint3` range `[-0.061087, 5.23599] rad`, aligned `default_validation_poses("uf850")["ready"]` (`[0.0, -0.5, 0.0, 0.0, 0.5, 0.0]`) with `ufactory_genesis` / runtime defaults, and aligned `ik_ok` with the declared `1.0 mm` / `0.5°` tolerance on both MuJoCo and controller FK reprojections so singular wrist poses (`home`) do not fail solely on the inner solver's `0.1 mm` polish flag.

### Hardware Validation

- **xArm 5** (bare flange, firmware `v2.8.2`, `--nominal-model`, Python `3.14.6`, MuJoCo `4.0.1`): controller internal simulation self-collision check `20/20 PASS` (`0.000002 rad` physical drift); full 20-pose hardware static torque check `20/20 PASS` (`torque_l2` error `0.520–0.780 Nm`, mean `0.622 Nm`, L2a/L2b/L3a/L3b all `20/20 pass`).
- **xArm 6** (bare flange, firmware `v2.8.0`, Python `3.14.6`, MuJoCo `4.0.1`): controller internal simulation self-collision check `20/20 PASS` (`0.000000 rad` physical drift); full 20-pose hardware static torque check `20/20 PASS` (`torque_l2` error `0.470–0.765 Nm`, mean `0.619 Nm`, L2a/L2b/L3a/L3b all `20/20 pass`).
- **xArm 7** (bare flange, firmware `v2.8.2`, Python `3.14.6`, MuJoCo `4.0.1`): standard FK/IK validation `PASS` (FK `<= 0.0003 mm`, IK `<= 0.095 mm`), controller internal simulation self-collision check `20/20 PASS` (`0.000004 rad` physical drift); full 20-pose hardware static torque check `20/20 PASS` (`torque_l2` error `0.816–3.303 Nm`, mean `1.750 Nm`, L2a/L2b/L3a/L3b all `20/20 pass`).
- **Lite 6** (bare flange, firmware `v2.8.2`, Python `3.14.6`, MuJoCo `4.0.1`): standard FK/IK validation `PASS` (FK `<= 0.649 mm`, IK `<= 0.660 mm`), controller internal simulation self-collision check `20/20 PASS` (`0.000080 rad` physical drift); full 20-pose hardware static torque check `20/20 PASS` (`torque_l2` error `0.028–0.156 Nm`, mean `0.094 Nm`, L2a/L2b/L3a/L3b all `20/20 pass`).
- **UF850** (bare flange, firmware `v2.8.100`, Python `3.14.6`, MuJoCo `4.0.1`): standard FK/IK validation `PASS` (FK `<= 0.854 mm`, IK `<= 0.871 mm`), controller internal simulation self-collision check `20/20 PASS` (`0.000004 rad` physical drift); full 20-pose hardware static torque check `20/20 PASS` (`torque_l2` error `0.086–0.569 Nm`, mean `0.240 Nm`, L2a/L2b/L3a/L3b all `20/20 pass`).

## [0.1.5] - 2026-11-16

Full-precision visual meshes, multi-arm RealSense D435 assemblies, Lite 6 joint PD tuning, end of MuJoCo 3 support, and a smaller public API. Task reports for xArm 5/6/7 and UF850 are byte-identical to 0.1.4; Lite 6 simulation task reports reflect the tighter Cartesian line tracking.

### Breaking

- `ufactory_mujoco` exports 39 names instead of 84. Lower-level helpers stay importable from their submodules, for example `from ufactory_mujoco.preflight import preflight_pick_place` or `from ufactory_mujoco.grippers import gap_to_position`.
- Gripper API on `RobotInterface`, `SimRobot` and `XArmRobot`:

  | 0.1.4 | 0.1.5 |
  | :--- | :--- |
  | `get_measured_gripper_gap()` | `get_gripper_gap(measured=True)` |
  | `set_gripper_command(mm, ...)` | `set_gripper_gap(mm / 1000, ...)` |
  | `SimRobot.get_gripper_control_output()` | `SimRobot.gripper_control_output` (property) |
  | `SimRobot.get_gripper_contact_forces(body)` | `grippers.gripper_object_contact_forces_n(model, data, spec, body)` |
  | `SimRobot.tcp_jacobian(q)` | `kinematics.pose_jacobian(model, data, site, q)` |

- Removed helpers: `kinematics.pose_error`, `kinematics.condition_number` (use `np.linalg.cond(pose_jacobian(...))`), `path_planning.path_length`, `bio_g2_model.bio_g2_physics_summary`, `bio_g2_model.MASS_KG`, `bio_g2_model.BIO_G2_PHYSICS_PROFILE`, `bio_g2_model.bounded` (use `g2_model.checked`), `preflight.PACK_PHASES`, and `XArmRobot.get_controller_tcp_pose`.
- The `assets` extra is removed.

### Changed

- MuJoCo 3 is no longer supported: the requirement is `mujoco>=4,<5` (was `>=3.15,<5`). Tested with MuJoCo 4.0.0 and 4.0.1 on Python 3.12–3.14.
- Consolidated `ufactory_mujoco/resource_paths.py` into `ufactory_mujoco/model_paths.py`, inlined task config JSON hashing into `ufactory_mujoco/config.py`, and merged `scripts/check_distribution.py` into `scripts/accept_installed.py --check-dist`.
- Asset conversion and MJCF generation tools (`build_all_mjcf.py`, `build_camera_mjcf.py`, `convert_glb_to_obj.py`, `fix_g2_base_meshes.py`, `make_double_sided_objs.py`) are now maintained outside the public tree. `models/` stays the published output; `scripts/` keeps `accept_installed.py` and `gen_kinematics_params.py`.
- Pick-and-place and box-packing share one `Mover` helper for Cartesian moves and preflight detours.
- Updated `xarm5`, `xarm7`, and `uf850` combination `ready` keyframes (and `02_joint_control.py` presets) to exact Gripper G2 `(300, 0, 300) mm` downward IK poses (`RPY = (180°, 0°, 0°)`), eliminating wrist roll tilt on `xarm5`/`xarm7` and raising `uf850_with_gripper` `ready` floor clearance from `4.9 mm` to `185.5 mm`.
- Tuned Lite 6 joint position actuator PD gains (`J1–3`: `kv 200 -> 80`; `J4–6`: `kp 40/80/40 -> 160/320/160`, `kv 8/16/8 -> 6.4/12.8/6.4`) to a uniform `0.04 s` time constant, reducing Cartesian line tracking deviation in `pick-place` and `box-pack` from `6–26 mm` to `<= 0.13 mm` with zero overshoot.

### Fixed

- Creases on xArm 5/6/7, UF850 and Lite 6: dark bands at joints and polygonal outlines on rounded edges. The 0.1.3 decimation reduced rounded surfaces to a few faces and smoothed normals across creases, and MuJoCo's classic renderer flattens any corner normal more than 36.9° from its face normal. The arm and accessory visual meshes are restored to full upstream tessellation; their vertices match the ufactory_genesis v0.3.6 GLBs within 1.4e-8 m.
- Holes in the Gripper G2 housing, and smaller see-through patches on the Bio Gripper G2, vacuum gripper and arm end parts. MuJoCo culls back faces, and the converter oriented the faces of open and thin-walled parts wrongly. 36 visual OBJs are now double-sided: each face has a reversed twin with a negated normal, the equivalent of glTF `doubleSided`. Their `<mesh>` elements declare `inertia="shell"`.

### Added

- Asset tests: visual OBJ size band, a limit on the area drawn with flattened normals, and double-sided OBJs with `inertia="shell"`.
- Seven RealSense D435 arm assemblies (`{xarm5,xarm6,xarm7,uf850}_with_d435` and `{xarm5,xarm7,uf850}_with_gripper_and_d435`, 14 XMLs reusing existing meshes via relative symlinks), plus regression tests locking D435's human-eye upright view and base `Y-` (flange `+Y`) USB-C port orientation across all eight xArm/UF850 D435 assemblies.

### Removed

- `scripts/optimize_visual_objs.py`.
- `requirements-lock-mujoco3.txt` and the MuJoCo 3 CI job.
- Remaining RL-only scenes and interfaces (`models/*/scene_reach.xml`, `"reach"` in `SCENES`, `ReachWorkspace`, `RobotProfile`, `robot_profile`, `ROBOT_PROFILES`, `ScalarG2Controller`, and `ufactory_mujoco._hashing`).
- Unreferenced Lite 6 gripper collision meshes (`finger1.stl`, `finger2.stl`), the `models/gripper_g2/source/` upstream GLB/STL/URDF snapshot (keeping the 85-pose reference CSV in `tests/fixtures/`), and historical xArm 6 comparison scripts (`scripts/compare_dynamics_xarm6.py`, `scripts/compare_torque_xarm6.py`).

### Documentation

- Root READMEs (`README.md`, `README_zh.md`): added Python/MuJoCo/License/CI badges and a supported Python (3.12–3.14) / MuJoCo (4.x) version matrix with `venv` and `conda` setup examples, fixed the broken table columns in `README_zh.md`, updated the wheel filename example to `0.1.5`, and removed stale `UFACTORY_GENESIS_ROOT` / `DracoPy` references.
- Simulation READMEs (`examples/simulation/README.md`, `README_zh.md`): corrected model coverage for `02_joint_control.py` (30 arm/combo models) and `04_gripper_control.py` (19 jaw-gripper models), added multi-arm D435 and `ufactory-mujoco` examples, restructured `08_dynamics.py` notes and the real-robot / `sdk-sim` section, and split the CLI option reference into task/hardware vs script-specific tables.

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
