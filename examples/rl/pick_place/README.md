# RL Example: Tabletop Pick and Place (`pick_place`)

[English](README.md) | [中文](README_zh.md)

Reinforcement learning example for tabletop cube pick-and-place with `xArm 6 + Gripper G2` in MuJoCo, using `PickPlaceEnv` (`ufactory_mujoco/rl/pick_place_env.py`), `PickPlaceExpert`, and Stable-Baselines3 pure-PPO policies trained from scratch under strict v2 checkpoint-manifest contracts.

## 1. Environment & MDP Specification

- **Scene, Calibrated G2 & Model Contract (`env_version: 2`)**: Uses `models/xarm6_with_gripper/scene_pick_place.xml`, `configs/pick_place/xarm6.yaml`, and calibrated Gripper G2 4-bar linkage kinematics & rate-limited PI+PD servo control (`ufactory_mujoco.g2_model.G2Model`, `ufactory_mujoco.g2_controller.G2Controller`, `50%` speed / `50%` force at `50 Hz`). Version 2 binds the scene, recursively included MJCFs, and referenced meshes to the checkpoint via `runtime_model_sha256`.
- **Randomized Spawn Domain**: Cube spawns uniformly in $x \in [0.30, 0.50]\text{ m}$, $y \in [-0.20, 0.20]\text{ m}$ ($z = 15\text{ mm}$) and must be placed at target $[0.30, 0.30]\text{ m}$. Deterministic evaluation uses `make_spawns(seed, episodes)` (`spawn_sha256` recorded in reports).
- **Action Space**: `Box(-1.0, 1.0, (4,), float32)` (`cartesian_and_g2_gap_increment`):
  - `action[0:3]`: Cartesian TCP setpoint increments $\Delta p \in [-5, 5]\text{ mm}$ per $20\text{ ms}$ control step ($10 \times 2\text{ ms}$ physics steps), bounded by workspace limits, a $25\text{ mm}$ TCP leash, near-table safety guardrails, and damped least-squares IK (`attachment_site`, downward grasp orientation).
  - `action[3]`: Gripper G2 jaw gap increment $\Delta g \in [-3, 3]\text{ mm}$ per control step, tracked by `G2Controller`. Zero action holds current TCP setpoint and jaw gap.
- **Observation Space**: `Box((70,), float32)`:
  - Normalized joint positions (`6`), scaled joint velocities (`6`), target-gated TCP position (`3`) & scaled TCP velocity (`3`), TCP-relative cube XY + Z (`3`), cube quaternion (`4`), scaled cube linear (`3`) & angular (`3`) velocities, post-lift target vector (`3`), left & right fingertip pad vectors relative to cube (`6`), commanded & measured G2 gap (`2`), normalized TCP setpoint error (`3`), bilateral finger contact forces & flags (`4`), cube-table & robot-table contact loads (`2`), drag/tilt/hold ratios (`3`), carry/lift flags (`2`), press ratio & touchdown state (`3`), release height & confirmation flags (`3`), previous action (`4`), milestone flags (`2`), and remaining time & lift progress (`2`).
- **Strict Acceptance & Safety Gates (`PickPlaceMetrics`)**:
  - **Pre-lift**: Table drag $\le 10\text{ mm}$, cube tilt $\le 15^\circ$.
  - **Lift**: Cube bottom clearance must reach $\ge 65\text{ mm}$ (cube center $z \ge 80\text{ mm}$) under stable bilateral finger contact ($> 0.2\text{ N}$ for $\ge 3$ steps).
  - **Touchdown & Release**: Table-supported touchdown ($\le 2\text{ mm}$ clearance, $> 0.05\text{ N}$ normal force), post-touchdown downward TCP press $\le 1.0\text{ mm}$, release height $\le 3.0\text{ mm}$ (both fingers $< 0.05\text{ N}$), and zero robot-table collision.
  - **Success**: Cube within $20\text{ mm}$ XY of target (`[0.30, 0.30]`), jaw gap $\ge 35\text{ mm}$, cube speed $< 0.03\text{ m/s}$, held continuously for **10 steps** ($0.20\text{ s}$).

## 2. Quick Start (Evaluation & Rollout)

> **Environment Note**: PPO checkpoints require `stable-baselines3` (`requirements-rl.txt`). The baselines (`--expert`, `--zero-policy`, `--random-policy`) run with core dependencies only.

```bash
# Strict 64-episode headless evaluation of trained pure-PPO v2 checkpoint across randomized spawns
python examples/rl/pick_place/evaluate.py --episodes 64 --seed 18001 --min-success-rate 0.95 --max-press-mm 1.0 --quiet-episodes

# Evaluate the independent closed-loop physics expert baseline (no stable-baselines3 required)
python examples/rl/pick_place/evaluate.py --expert --episodes 64 --seed 18001 --quiet-episodes

# Evaluate untrained zero-action or random baselines
python examples/rl/pick_place/evaluate.py --zero-policy --episodes 5 --quiet-episodes

# Interactive GUI rollout (or --headless)
python examples/rl/pick_place/01_pick_place.py
python examples/rl/pick_place/01_pick_place.py --headless --episodes 10 --min-success-rate 0.95 --max-press-mm 1.0
python examples/rl/pick_place/01_pick_place.py --expert --headless --episodes 5
```

## 3. Pure-PPO Training from Scratch & Artifact Contract

Declarative parameters live in `examples/rl/pick_place/recipe.yaml` (bound to `configs/pick_place/xarm6.yaml` by SHA-256). Policies are trained from random initialization with pure PPO (no expert demonstrations or behavior-cloning warmstart) via `scripts/train_pick_place_rl.py`:

```bash
# Full multi-worker pure-PPO training from scratch (early-stops after consecutive 64-episode strict passes)
python scripts/train_pick_place_rl.py --n-envs 128 --n-workers 24 --device cuda --seed 1

# Quick CPU smoke run
python scripts/train_pick_place_rl.py --total-steps 256 --n-envs 4 --n-workers 2 --n-steps 32 --batch-size 64 --eval-episodes 2 --device cpu
```

Training writes to `build/rl/pick_place/xarm6_g2_v2/`:
- `ppo_pick_place.zip`: Selected best PPO policy checkpoint
- `config.yaml`: Resolved environment, reward, and training configuration with canonical `config_sha256`
- `ppo_pick_place.checkpoint_manifest.json`: SHA-256 manifest binding `ppo_pick_place.zip`, `config.yaml`, task config, and `runtime_model_sha256`
- `provenance.json`, `metrics.json`, and `eval_history.json`: Source file hashes, runtime versions, periodic evaluation trace, and final evaluation metrics

Legacy v1 (`46`-dim) checkpoints lack `runtime_model_sha256` and do not match the v2 `70`-dim observation contract; both normal loading and `scripts/migrate_pick_place_checkpoint.py` reject incompatible v1 checkpoints with `ArtifactError`.

## 4. Benchmark Results

Evaluated over **64 randomized-spawn episodes** ($x \in [0.30, 0.50]\text{ m}$, $y \in [-0.20, 0.20]\text{ m} \to [0.30, 0.30]\text{ m}$, held-out `--seed 18001`, `spawn_sha256: 7a2ebbc7...ec608`, `examples/rl/pick_place/evaluate.py`):

| Policy | Success Rate (`>= 95%`) | Max Press (`<= 1.0 mm`) | Max Release Height (`<= 3.0 mm`) | Mean Target Error (`<= 20.0 mm`) | Max Clearance (`>= 65.0 mm`) | Mean Steps |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Pure-PPO v2 (from scratch)** (`ppo_pick_place.zip`) | **100.0% (64/64)** | **0.124 mm** | **0.105 mm** | **4.71 mm** | **70.98 mm** | 196.0 |
| `PickPlaceExpert` (`--expert`, v2) | **100.0% (64/64)** | **0.000 mm** | **1.797 mm** | **0.23 mm** | **97.75 mm** | 294.7 |

Trained locally from scratch with pure PPO on `i7-14700F` + `RTX 4060 Ti` (`128` envs across `24` workers, early-stopped at `9,000,064` steps in `19.5 min` at `~7,689 FPS`, zero collision/drag/tilt/high-release failures).
