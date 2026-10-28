# RL Example: Tabletop Pick and Place (`pick_place`)

[English](README.md) | [中文](README_zh.md)

Reinforcement learning example for tabletop cube pick-and-place with `xArm 6 + Gripper G2` in MuJoCo, using `PickPlaceEnv` (`ufactory_mujoco/rl/pick_place_env.py`), `PickPlaceExpert`, and Stable-Baselines3 PPO policies under strict checkpoint-manifest contracts.

## 1. Environment & MDP Specification

- **Scene & Runtime Config**: Reuses Batch 2–3 scene `models/xarm6_with_gripper/scene_pick_place.xml`, declarative config `configs/pick_place/xarm6.yaml` (bound via SHA-256), and `ufactory_mujoco.grippers` (`gap_to_position`, `position_to_gap`).
- **Action Space**: `Box(-1.0, 1.0, (4,), float32)` (`cartesian_and_g2_gap_increment`):
  - `action[0:3]`: Cartesian TCP setpoint increments $\Delta p \in [-5, 5]\text{ mm}$ per $20\text{ ms}$ control step ($10 \times 2\text{ ms}$ physics steps), bounded by workspace limits, a $25\text{ mm}$ TCP leash, and damped least-squares IK (`attachment_site`, downward grasp orientation).
  - `action[3]`: Gripper G2 jaw gap increment $\Delta g \in [-3, 3]\text{ mm}$ per control step, mapped via `gap_to_position`. Zero action holds current TCP setpoint and jaw gap.
- **Observation Space**: `Box((46,), float32)`:
  - Normalized joint positions (`6`), scaled joint velocities (`6`), TCP position (`3`), cube position (`3`), cube quaternion (`4`), scaled cube velocity (`3`), relative vector to target (`3`), finger-pad midpoint relative to cube (`3`), commanded & measured G2 gap (`2`), normalized TCP setpoint error (`3`), bilateral finger contact forces & flags (`4`), previous action (`4`), and task progress flags (`[ever_lifted_clear, release_confirmed]`, `2`).
- **Strict Acceptance & Safety Gates (`PickPlaceMetrics`)**:
  - **Pre-lift**: Table drag $\le 10\text{ mm}$, cube tilt $\le 15^\circ$.
  - **Lift**: Cube bottom clearance must reach $\ge 70\text{ mm}$ under stable bilateral finger contact ($\ge 0.5\text{ N}$ for $\ge 3$ steps).
  - **Touchdown & Release**: Table-supported touchdown ($\le 2\text{ mm}$ clearance, $> 0.15\text{ N}$ normal force), post-touchdown downward TCP press $\le 1.0\text{ mm}$, release height $\le 3.0\text{ mm}$, and zero robot-table collision.
  - **Success**: Cube within $20\text{ mm}$ XY of target (`[0.35, 0.10]`), jaw gap $\ge 35\text{ mm}$, cube speed $< 0.03\text{ m/s}$, held continuously for **10 steps** ($0.20\text{ s}$).

## 2. Quick Start (Evaluation & Rollout)

> **Environment Note**: Rolling out or evaluating trained PPO checkpoints requires `stable-baselines3` (`requirements-rl.txt`). The `--expert`, `--zero-policy`, and `--random` baselines run with core dependencies only.

```bash
# Strict headless evaluation of trained PPO checkpoint (validates config.yaml + checkpoint_manifest.json)
python examples/rl/pick_place/evaluate.py --episodes 20 --min-success-rate 0.90 --max-press-mm 1.0 --quiet-episodes

# Evaluate the independent physics expert baseline (no stable-baselines3 required)
python examples/rl/pick_place/evaluate.py --expert --episodes 10 --quiet-episodes

# Evaluate untrained zero-action or random baselines
python examples/rl/pick_place/evaluate.py --zero-policy --episodes 5 --quiet-episodes

# Interactive GUI rollout (or --headless)
python examples/rl/pick_place/01_pick_place.py
python examples/rl/pick_place/01_pick_place.py --headless --episodes 5
python examples/rl/pick_place/01_pick_place.py --expert --headless --episodes 3
```

## 3. Policy Training & Artifact Contract

Declarative parameters live in `examples/rl/pick_place/recipe.yaml` (bound to `configs/pick_place/xarm6.yaml` by SHA-256). Train with `scripts/train_pick_place_rl.py`:

```bash
# Train xArm6 + Gripper G2 pick-and-place PPO policy (expert warmstart + PPO + checkpoint ranking)
python scripts/train_pick_place_rl.py --total-steps 4096 --n-envs 4

# Pure PPO smoke run without warmstart
python scripts/train_pick_place_rl.py --warmstart-episodes 0 --total-steps 256 --n-envs 1
```

Training writes to `build/rl/pick_place/xarm6/`:
- `ppo_pick_place.zip`: Selected best PPO policy checkpoint
- `config.yaml`: Resolved environment, reward, and training configuration with canonical `config_sha256`
- `ppo_pick_place.checkpoint_manifest.json`: SHA-256 manifest binding `ppo_pick_place.zip`, `config.yaml`, and `configs/pick_place/xarm6.yaml`
- `provenance.json` and `metrics.json`: Source file hashes, runtime versions, and evaluation metrics

## 4. Benchmark Results

Evaluated over 20 episodes (`--seed 17000`, `examples/rl/pick_place/evaluate.py`):

| Policy | Success Rate | Max Press (`<= 1.0 mm`) | Release Height (`<= 3.0 mm`) | Mean Target Error (`<= 20.0 mm`) | Max Clearance (`>= 70.0 mm`) | Mean Steps |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ppo_pick_place.zip` | **100.0%** | **0.115 mm** | **0.00 mm** | **3.86 mm** | **92.41 mm** | 312 |
| `PickPlaceExpert` (`--expert`) | **100.0%** | **0.143 mm** | **0.00 mm** | **1.60 mm** | **90.15 mm** | 274 |
| `--zero-policy` | 0.0% | 0.000 mm | 0.00 mm | 300.00 mm | 0.00 mm | 380 |

