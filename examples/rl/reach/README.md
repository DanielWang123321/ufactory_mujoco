# RL Example: TCP Position Reaching (`reach`)

[English](README.md) | [中文](README_zh.md)

Reinforcement learning example for 3D TCP position reaching across all five UFACTORY robotic arms (`lite6`, `xarm5`, `xarm6`, `xarm7`, `uf850`) in MuJoCo, using a PPO policy trained on `ReachEnv` (`src/rl/reach_env.py`).

## 1. Environment & MDP Specification

- **Scene**: `models/<robot>/scene_reach.xml` (arm + visual target sphere `target_mocap`; TCP tracked at `attachment_site`).
- **Action Space**: `Box(-1.0, 1.0, (n,), float32)` — joint position increments $\Delta q \in [-0.05, 0.05]\text{ rad}$ per $20\text{ ms}$ control step ($10 \times 2\text{ ms}$ physics steps), clamped to joint limits.
- **Observation Space**: `Box((3 + 3n,), float32)` where $n$ is the joint count (`18` for `xarm5`, `21` for `lite6`/`xarm6`/`uf850`, `24` for `xarm7`):
  - `obs[0:3]`: TCP-to-target position error $p_{\text{tcp}} - p_{\text{target}}$ (m)
  - `obs[3:3+n]`: midpoint-normalized joint positions $q_{\text{norm}} \in [-1.5, 1.5]$
  - `obs[3+n:3+2n]`: scaled joint velocities $\dot{q} / 3.0 \in [-5.0, 5.0]$
  - `obs[3+2n:3+3n]`: previous action $a_{t-1} \in [-1.0, 1.0]$
- **Reward & Termination**:
  - Distance potential shaping $200 \cdot (d_{t-1} - d_t)$ minus step cost ($0.02$), quadratic action penalty ($0.001 \|a\|^2$), and joint-limit margin penalty ($0.01$ per joint within $2\%$ of limits).
  - **Success**: TCP distance strictly below **5 mm** ($d_t < 0.005\text{ m}$, terminates episode immediately).
  - **Timeout**: Truncated at **200 control steps** ($4.0\text{ s}$ simulated time).
- **Workspace Calibration**: Target boxes and radial pre-filter bands (`ROBOT_PROFILES`) are calibrated per robot via FK point clouds and IK reachability ($\ge 90\%$) using `scripts/measure_reach_workspace.py`.

## 2. Quick Start (Evaluation & Rollout)

> **Weights are not shipped with the repo** (`build/` is git-ignored). Train a policy first (Section 3); it is saved to `build/rl/<model>/ppo_reach.zip`. Measured on an 8-core CPU / 15 GB RAM machine without GPU: the default recipe (300k steps, 8 envs) trains xArm 6 in about **100 s**, but each environment process uses about **1.3 GB RAM** (high-poly visual meshes), so 8 envs need ~11 GB. On machines with less memory pass `--n-envs 2`–`4`. The 300k-step quick recipe reaches only ~40% success; use the production recipe for the benchmark numbers.

> **Environment Note**: Rolling out trained PPO policies requires `stable-baselines3` (see `requirements-rl.txt`). The untrained `--random` baseline only requires the core dependencies.

```bash
# Interactive GUI rollout (10 episodes, xArm 6 by default)
python examples/rl/reach/01_reach.py

# Run on other robot models
python examples/rl/reach/01_reach.py --model lite6
python examples/rl/reach/01_reach.py --model xarm5
python examples/rl/reach/01_reach.py --model xarm7
python examples/rl/reach/01_reach.py --model uf850

# Headless evaluation (prints metrics, exits 0 on PASS / 1 on FAIL)
python examples/rl/reach/01_reach.py --model xarm6 --headless --episodes 20

# Roll out toward a fixed base-frame target (x, y, z in meters)
python examples/rl/reach/01_reach.py --target 0.35 0.15 0.25

# Untrained random policy baseline (no stable-baselines3 required)
python examples/rl/reach/01_reach.py --random --headless
```

### CLI Options (`01_reach.py`)

| Option | Default | Description |
| :--- | :--- | :--- |
| `--model` | `xarm6` | Robot model: `lite6`, `xarm5`, `xarm6`, `xarm7`, `uf850` |
| `--policy` | `build/rl/<model>/ppo_reach.zip` | Path to trained PPO policy `.zip` |
| `--episodes` | `10` | Number of evaluation episodes |
| `--seed` | `0` | Base random seed (`seed + ep` per episode) |
| `--device` | `cpu` | PyTorch device (`cpu` or `cuda`) |
| `--headless` | `False` | Disable MuJoCo viewer and run at maximum speed |
| `--random` | `False` | Use uniform random actions instead of a trained policy |
| `--target X Y Z` | `None` | Fixed target position in base frame (m) |
| `--min-success-rate` | `0.95` | Acceptance threshold for success rate |
| `--max-mean-dist-mm` | `10.0` | Acceptance bound for mean final TCP error (mm) |

## 3. Policy Training

Train or re-train PPO policies with `scripts/train_reach_rl.py`:

```bash
# Default quick training (300k steps, [64,64] MLP, 8 parallel envs)
python scripts/train_reach_rl.py --model xarm6

# Full production recipe aligned with 5 mm acceptance standard (2M steps, [256,256,128] MLP)
python scripts/train_reach_rl.py \
    --model xarm6 --total-steps 2000000 --net-arch 256,256,128 --ent-coef 0.001 --eval-episodes 200
```

Training outputs `ppo_reach.zip`, periodic checkpoints under `checkpoints/`, and an evaluation summary `metrics.json` in `build/rl/<model>/`.

## 4. Benchmark Results

Evaluated against the `ufactory_genesis` reach standard (success strictly `< 5 mm`, acceptance requires success rate **$\ge 95\%$** and mean final distance **$\le 10\text{ mm}$** over two independent 512-target groups with `--seed 17000` and `--seed 17512`):

| Model | Success Rate (Group 1 / Group 2) | Mean Final Distance (Group 1 / Group 2) | Mean Steps | Training Steps |
| :--- | ---: | ---: | ---: | ---: |
| `lite6` | 99.8% / 99.4% | 4.3 / 4.4 mm | 82 | 2M |
| `xarm5` | 99.4% / 99.0% | 4.2 / 4.2 mm | 95 | 2M |
| `xarm6` | 100.0% / 99.8% | 4.1 / 4.1 mm | 72 | 2M |
| `xarm7` | 99.6% / 99.0% | 4.3 / 4.3 mm | 88 | 2M |
| `uf850` | 98.1% / 96.3% | 4.4 / 4.9 mm | 110 | 4M |

