# RL Example: Tabletop Pick and Place (`pick_place`)

[English](README.md) | [中文](README_zh.md)

Reinforcement learning example for tabletop cube pick-and-place with `xArm 6 + Gripper G2` in MuJoCo, using the Gymnasium environment `PickPlaceEnv` (`ufactory_mujoco/rl/pick_place_env.py`). Policies are trained with **pure PPO from scratch** (no expert demonstrations or behavior-cloning warmstart), loaded under v2 model-hash contracts, evaluated on deterministic held-out spawns, and replayed visually.

## 1. Environment & MDP Specification

- **Scene, Calibrated G2 & Model Contract (`env_version: 2`)**: Uses `models/xarm6_with_gripper/scene_pick_place.xml`, `configs/pick_place/xarm6.yaml`, the calibrated Gripper G2 4-bar linkage geometry and force curve (`ufactory_mujoco.g2_model.G2Model`), and a rate-limited position/velocity servo with conditional PI anti-windup and force-percent torque limiting (`ufactory_mujoco.g2_controller.ScalarG2Controller`, the bit-identical single-environment form of `G2Controller`; `50 Hz`, `50%` speed / `50%` force). Version 2 binds the scene, recursively included MJCFs, and referenced meshes to the checkpoint via `runtime_model_sha256`.
- **Episode & Initial State**: At most `400` steps per episode ($8\text{ s}$, truncated on timeout). The arm starts at a fixed home pose (TCP $(0.30, 0.00, 0.1265)\text{ m}$, pointing down) with the gripper pre-opened to $50\text{ mm}$; the cube is $30\text{ mm}$ wide and weighs $17\text{ g}$.
- **Randomized Spawn Domain**: Cube spawns uniformly in $x \in [0.30, 0.50]\text{ m}$, $y \in [-0.20, 0.20]\text{ m}$ ($z = 15\text{ mm}$) and must be placed at target $[0.30, 0.30]\text{ m}$. Deterministic evaluation uses `make_spawns(episodes, seed=...)` (`evaluate.py`) and records `spawn_sha256`.
- **Action Space**: `Box(-1.0, 1.0, (4,), float32)` (`cartesian_and_g2_gap_increment`):
  - `action[0:3]`: Cartesian TCP setpoint increments $\Delta p \in [-5, 5]\text{ mm}$ per $20\text{ ms}$ control step ($10 \times 2\text{ ms}$ physics steps), clipped to the workspace box and a $25\text{ mm}$ TCP leash, then solved by damped least-squares IK (`attachment_site`, downward grasp orientation). The previous setpoint is kept if IK fails.
  - `action[3]`: Gripper G2 jaw gap increment $\Delta g \in [-3, 3]\text{ mm}$; the commanded gap is clamped to $[12, 84]\text{ mm}$ and tracked by this servo. Zero action holds the current TCP setpoint and jaw gap.
  - Safety guardrails: before lift, the TCP setpoint stays $\ge 5.5\text{ mm}$ and downward motion is blocked once the cube is gripped; after lift, with the cube $\le 20\text{ mm}$ above the table, descent is limited to $20\text{ mm/s}$ and the setpoint never goes more than $0.4\text{ mm}$ below the estimated touchdown height.
- **Observation Space**: `Box(-10, 10, (70,), float32)`, in order:
  - Normalized joint positions (`6`), scaled joint velocities (`6`), lift-gated TCP position (`3`; height only before lift, target-relative XY plus height after lift), scaled TCP velocity (`3`);
  - TCP-relative cube XY plus cube height (`3`), cube unit quaternion (`4`), scaled cube linear (`3`) and angular (`3`) velocities, post-lift target-relative cube vector (`3`), left and right pad centers relative to the cube (`6`);
  - Commanded and measured G2 gap (`2`), TCP setpoint error (`3`), finger contact forces (`2`) and flags (`2`), cube-table (after lift) and robot-table contact loads (`2`), previous action (`4`);
  - Drag/tilt/hold ratios (`3`), carry/lift flags (`2`), press ratio / touchdown flag / post-touchdown TCP sink (`3`), release height ratio / release confirmed / high-release flag (`3`), milestone flags (`2`), remaining-time ratio and lift progress (`2`).
- **Acceptance Gates (`PickPlaceMetrics`)**:
  - **Pre-lift**: Table drag $\le 10\text{ mm}$, cube tilt $\le 15^\circ$.
  - **Lift**: Cube bottom clearance $\ge 65\text{ mm}$ (cube center $z \ge 80\text{ mm}$) under stable bilateral contact ($> 0.2\text{ N}$ for $\ge 3$ steps; $< 0.05\text{ N}$ counts as released).
  - **Touchdown & Release**: After table-supported touchdown ($\le 2\text{ mm}$ clearance, $> 0.05\text{ N}$ normal force), TCP press $\le 1.0\text{ mm}$; release height $\le 3.0\text{ mm}$ (release is confirmed only after both fingers stay $< 0.05\text{ N}$ for 3 consecutive steps).
  - **Success**: Cube within $20\text{ mm}$ XY of the target, tilt $\le 15^\circ$, table-supported, jaw gap $\ge 35\text{ mm}$, cube speed $< 0.03\text{ m/s}$, held for **10 steps** ($0.20\text{ s}$).
  - **Failure (terminates the episode)**: Drag or tilt over limit, drop (cube carried, then on the table outside the target with no grip for 15 steps), press over limit, high release, or robot/gripper-table contact force $> 0.2\text{ N}$.

## 2. Quick Start (Evaluation & Rollout)

> **Environment Note**: PPO checkpoints require `stable-baselines3` (`requirements-rl.txt`). The local `py314` env does not have it; benchmark results were produced in `py313`. The scripted expert and baselines (`--expert`, `--zero-policy`, `--random`) need core dependencies only.

```bash
# Evaluate the pure-PPO v2 checkpoint on 64 held-out spawns (validates config.yaml, checkpoint_manifest.json, and model hash first)
python examples/rl/pick_place/evaluate.py --episodes 64 --seed 18001 --min-success-rate 0.95 --max-press-mm 1.0 --quiet-episodes

# Evaluate the scripted expert (no stable-baselines3 required)
python examples/rl/pick_place/evaluate.py --expert --episodes 64 --seed 18001 --quiet-episodes

# Evaluate zero-action or uniform-random baselines
python examples/rl/pick_place/evaluate.py --zero-policy --episodes 5 --quiet-episodes
python examples/rl/pick_place/evaluate.py --random --episodes 5 --quiet-episodes

# Save summary and per-episode JSON/CSV reports
python examples/rl/pick_place/evaluate.py --episodes 64 --seed 18001 --quiet-episodes --report build/rl/pick_place/eval_64.json

# Interactive GUI rollout (or --headless)
python examples/rl/pick_place/01_pick_place.py
python examples/rl/pick_place/01_pick_place.py --headless --episodes 10 --min-success-rate 0.95 --max-press-mm 1.0
python examples/rl/pick_place/01_pick_place.py --expert --headless --episodes 5
```

- `evaluate.py` PASS checks only three items: success rate $\ge$ `--min-success-rate` (default `0.90`), max press $\le$ `--max-press-mm`, and mean target error $\le$ `--max-target-error-mm` (default `20 mm`). Collision, drag, tilt, drop, and high-release counts are reported in the output JSON.
- `PickPlaceExpert` (`--expert`) is a scripted expert: a closed-loop state machine on privileged simulator state (ground-truth cube and pad poses, internal `PickPlaceMetrics` state). It is not used for training; it only shows the task is solvable under the same constraints and gates.

## 3. Pure-PPO Training from Scratch & Artifact Contract

Declarative parameters live in `examples/rl/pick_place/recipe.yaml`. `scripts/train_pick_place_rl.py` trains Stable-Baselines3 PPO from random initialization, with no expert demonstrations or behavior-cloning warmstart:

```bash
# Full multi-worker training (writes to build/rl/pick_place/xarm6_g2_v2/ by default, overwriting existing weights)
python scripts/train_pick_place_rl.py --n-envs 128 --n-workers 24 --device cuda --seed 1

# Quick CPU smoke run (separate output directory, keeps the published weights intact)
python scripts/train_pick_place_rl.py --total-steps 256 --n-envs 4 --n-workers 2 --n-steps 32 --batch-size 64 --eval-episodes 2 --device cpu --out-dir build/rl/pick_place/smoke
```

**Training Strategy**:
- **Reward**: Potential-based shaping $r = \Phi(s') - \Phi(s) - 0.015$ (`reward_gamma: 1.0`). $\Phi$ is a weighted sum of approach, pre-grasp, grasp, lift (largest weight, based on the highest held height so far), transit, place, release, and hold terms; on failure $\Phi(s')$ is set to 0, taking back the accumulated potential. One-time milestone bonuses (first contact, carry, lift, touchdown, place; success `+40`) and safety penalties (press, fast near-table descent, high release, robot-table contact, table overload, action magnitude; failure `-10`) are added.
- **PPO**: MLP `[256, 256]` (ELU), `n_steps 128`, `batch_size 2048`, `gamma 0.995`, etc. in `recipe.yaml`; the number of parallel envs is set by `--n-envs`.
- **Evaluation & Early Stop**: Every `1M` steps, a deterministic 64-episode evaluation runs on spawns from `seed = 17000 + training seed`. The gate requires success rate $\ge 95\%$, max press $\le 1.0\text{ mm}$, mean target error $\le 20\text{ mm}$, and zero collision, drag, tilt, and high-release failures. Training stops after 2 consecutive passes, otherwise runs to `total_steps` (`50M`).
- **Checkpoint Selection**: `checkpoint_score` (`evaluate.py`) compares success rate, low-release rate, target touchdown rate, max press, mean target error, etc. in order; the best checkpoint is copied to `ppo_pick_place.zip`.

**Artifacts** (default `build/rl/pick_place/xarm6_g2_v2/`):
- `ppo_pick_place.zip`: Selected best policy checkpoint
- `config.yaml`: Resolved environment, reward, and training configuration with `config_sha256`; `env` records the task-config hash `runtime_config_sha256` and scene-model hash `runtime_model_sha256`
- `ppo_pick_place.checkpoint_manifest.json`: SHA-256 of `ppo_pick_place.zip`, `config.yaml`, and the task config, plus observation/action dimensions
- `provenance.json`, `metrics.json`, `eval_history.json`: Source file hashes and package versions, final evaluation metrics and training time, periodic evaluation history
- `checkpoints/`: Snapshots every `1M` steps, `ppo_pick_place_best.zip`, gate-passing snapshots `ppo_pick_place_pass_<N>.zip`, and `ppo_pick_place_final.zip`

**Load Validation**: Before loading, `load_validated_ppo` checks the zip and `config.yaml` hashes and observation/action dimensions, re-resolves the environment config from the current `recipe.yaml` and `configs/pick_place/xarm6.yaml` and compares it field by field, and recomputes `runtime_model_sha256` from the current scene; any mismatch raises `ArtifactError`. Legacy v1 (`46`-dim observation) checkpoints lack `runtime_model_sha256` and do not match the v2 `70`-dim observation; loading rejects them; retrain with the current recipe.

## 4. Benchmark Results

Published checkpoint `ppo_pick_place.zip` (trained locally with seed `1`) and the scripted expert on held-out spawns (spawn domain $x \in [0.30, 0.50]\text{ m}$, $y \in [-0.20, 0.20]\text{ m} \to [0.30, 0.30]\text{ m}$, `examples/rl/pick_place/evaluate.py`):

| Policy | Evaluation Set | Success Rate | Max Press (`<= 1.0 mm`) | Max Release Height (`<= 3.0 mm`) | Mean Target Error (`<= 20.0 mm`) | Max Clearance (`>= 65.0 mm`) | Mean Steps |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Pure-PPO v2** (`ppo_pick_place.zip`) | 64 episodes, `--seed 18001` | **100.0% (64/64)** | 0.124 mm | 0.105 mm | 4.71 mm | 70.98 mm | 196.0 |
| **Pure-PPO v2** (`ppo_pick_place.zip`) | 1000 episodes, `--seed 30001–34001`, 200 each | **98.1% (981/1000)** | 0.181 mm | 0.258 mm | 12.12 mm\* | 72.54 mm | 193.4 |
| `PickPlaceExpert` scripted expert (`--expert`) | 64 episodes, `--seed 18001` | **100.0% (64/64)** | 0.000 mm | 1.797 mm | 0.23 mm | 97.75 mm | 294.7 |

- `spawn_sha256` for `--seed 18001` is `7a2ebbc7...ec608`. The 1000-episode run had 19 failures: 16 tilt, 2 drop, 1 drag; no collision, press, or high-release failures. \*Mean target error includes failed episodes and is not directly comparable with the 64-episode result.
- Reproduced in `py313` (Python 3.13.13, MuJoCo 3.10.0, SB3 2.9.0, torch 2.10.0+cu128). In `py314` (MuJoCo 3.15.0), the scripted expert differs slightly (max release height 1.799 mm, mean error 0.35 mm, max clearance 97.72 mm, 294.8 mean steps).
- The published checkpoint was trained locally (`i7-14700F` + `RTX 4060 Ti`, `128` envs / `24` workers) with seed `1` and early-stopped after two consecutive gate passes at `9,000,064` steps in `19.5 min` (`~7,689 FPS`).
- After the environment CPU-overhead optimizations (scalar G2 controller, cached kinematics, potential reuse, precomputed IK; bit-identical environment behavior), steady-state training throughput on the same machine and setup (excluding periodic evaluation) rose from about `9,700` to `14,700` steps/s (about `1.5x`).

## 5. Training Stability (Seed Sensitivity)

Training outcomes with the current recipe depend strongly on the random seed; the published checkpoint does not mean every seed reaches the same result.

On a server (`i9-14900K` + `RTX 5090`, same software versions), 20 seeds (`10001–29001`) were each trained for `50M` steps (periodic evaluation and early stop disabled) and evaluated at 10M / 20M / 50M steps on 1000 held-out episodes (evaluation seeds `50001–54001`):

| Training Steps | Seeds with `>= 95%` Success | Seeds with 0% Success | Mean Success Rate |
| :--- | ---: | ---: | ---: |
| 10M | 10/20 | 7/20 | 58.5% |
| 20M | 14/20 | 6/20 | 69.6% |
| 50M | 14/20 | 3/20 | 79.9% |

- **Bit-reproducible on one machine**: Each seed was trained 3 times; policy parameter hashes and per-episode evaluation records were identical.
- **Seed outcomes do not transfer across machines**: With the same recipe at 10M steps, seed `1` reached 100% locally but 0% on the server; seeds `3001` / `4001` reached 0% locally but about 97% on the server. The cause is not yet isolated (the machines differ in GPU and worker layout).
- **Longer training is not always better**: Seeds `12001`, `15001`, and `21001` regressed from 20M to 50M (86.4%, 86.7%, 43.2%).
- **The training gate is optimistic**: Over 1000 episodes, most models with $\ge 95\%$ success still show a few tilt, drag, or high-release failures; judged by the training gate (Section 3), only 3/20 (10M), 6/20 (20M), and 3/20 (50M) seeds pass.
- **Recommendation**: Train several seeds and select by evaluation on independent held-out spawns.
