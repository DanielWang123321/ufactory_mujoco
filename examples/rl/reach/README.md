# RL Example: TCP Position Reaching (`reach`)

[English](README.md) | [中文](README_zh.md)

Reinforcement learning example for 3D TCP position reaching across all five UFACTORY robotic arms (`lite6`, `xarm5`, `xarm6`, `xarm7`, `uf850`) in MuJoCo, using a PPO policy trained on `ReachEnv` (`ufactory_mujoco/rl/reach_env.py`).

## 1. Environment & MDP Specification

- **Scene**: `models/<robot>/scene_reach.xml` (arm + visual target sphere `target_mocap`; TCP tracked at `attachment_site`).
- **Action Space**: `Box(-1.0, 1.0, (n,), float32)` — joint position increments $\Delta q \in [-0.05, 0.05]\text{ rad}$ per $20\text{ ms}$ control step ($10 \times 2\text{ ms}$ physics steps), clamped to joint limits.
- **Observation Space**: `Box((3 + 3n,), float32)` where $n$ is the joint count, i.e. `18` dimensions for `xarm5`, `21` for `lite6`/`xarm6`/`uf850` and `24` for `xarm7`:
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

> **Weights are not shipped with the repo or the releases** (`build/` is git-ignored). Train a policy first (Section 3); it is saved to `build/rl/<model>/ppo_reach.zip` together with `metrics.json`. Training envs load a *lite-visual* model (visual meshes dropped, collision meshes shown; physics identical), so each worker uses about **0.3 GB RAM** instead of ~1.5 GB, and `--n-envs` defaults to an automatic value from CPU cores and free memory. Measured on an 8-core CPU / 15 GB RAM machine without GPU (xArm 6, 8 envs, ~2.9 GB total): the quick recipe (300k steps) takes **~65 s** and reaches only ~54% success; the production recipe (2M steps, `256,256,128`) takes **~16 min** and reaches **100% success, 3.9 mm mean error** over 200 episodes.

Training and PPO rollout need the `rl` extra (PyTorch, Gymnasium, Stable-Baselines3); see the [installation guide](../../../README.md#installation). Check it with:

```bash
python -c "import torch, gymnasium, stable_baselines3; print('RL dependencies OK')"
```

If the default checkpoint `build/rl/xarm6/ppo_reach.zip` is absent, first run `python scripts/train_reach_rl.py --model xarm6` (see Section 3). The untrained `--random` baseline needs only the core dependencies but does not demonstrate trained reaching.

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

# Strict checkpoint-manifest evaluation CLI (single model or all 5 arms)
python examples/rl/reach/evaluate.py --model xarm6 --episodes 50 --quiet-episodes
python examples/rl/reach/evaluate.py --model all --episodes 20 --min-success-rate 0.95 --max-mean-dist-m 0.010 --out build/rl/reach_eval.json --quiet-episodes

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
| `--recipe` | `examples/rl/reach/recipe.yaml` | Recipe used to check the checkpoint contract |
| `--min-success-rate` | `0.95` | Acceptance threshold for success rate |
| `--max-mean-dist-mm` | `10.0` | Acceptance bound for mean final TCP error (mm) |
| `--report`, `--summary-json` | `None` | Write the summary and per-episode results to JSON |

### CLI Options (`evaluate.py`)

| Option | Default | Description |
| :--- | :--- | :--- |
| `--model` | `xarm6` | Robot model, or `all` for the five arms |
| `--checkpoint`, `--policy` | `build/rl/<model>/ppo_reach.zip` | Checkpoint to evaluate (not allowed with `--model all`) |
| `--recipe` | `examples/rl/reach/recipe.yaml` | Recipe used to check the checkpoint contract |
| `--episodes` | `32` | Episodes per robot |
| `--seed` | `17000` | Base seed (`seed + episode`) |
| `--device` | `cpu` | PyTorch device |
| `--min-success-rate` | `0.95` | Required success rate |
| `--max-mean-dist-mm` | `10.0` | Required mean final distance (mm); `--max-mean-dist-m` sets it in meters |
| `--summary-json`, `--report`, `--out` | `None` | Write the summary, episode records and artifact hashes to JSON |
| `--quiet-episodes` | `False` | Print only the aggregate summary |

`evaluate.py` exits with `1` if any robot misses a threshold.

## 3. Policy Training

Train or re-train PPO policies with `scripts/train_reach_rl.py`:

```bash
# Default quick training (300k steps, [64,64] MLP, auto parallel envs)
python scripts/train_reach_rl.py --model xarm6

# Full production recipe aligned with 5 mm acceptance standard (2M steps, [256,256,128] MLP)
python scripts/train_reach_rl.py \
    --model xarm6 --total-steps 2000000 --net-arch 256,256,128 --ent-coef 0.001 --eval-episodes 200

# Backfill config.yaml + ppo_reach.checkpoint_manifest.json for existing checkpoints
python scripts/train_reach_rl.py --model all --backfill-manifest
```

Declarative parameters live in `examples/rl/reach/recipe.yaml`. Training outputs `ppo_reach.zip`, `config.yaml`, `ppo_reach.checkpoint_manifest.json`, `provenance.json`, periodic checkpoints under `checkpoints/`, and `metrics.json` in `build/rl/<model>/`.

## 4. Benchmark Results

Evaluated against the `ufactory_genesis` reach standard (success strictly `< 5 mm`, acceptance requires success rate **$\ge 95\%$** and mean final distance **$\le 10\text{ mm}$** over two independent 512-target groups with `--seed 17000` and `--seed 17512`):

| Model | Success Rate (Group 1 / Group 2) | Mean Final Distance (Group 1 / Group 2) | Mean Steps | Training Steps |
| :--- | ---: | ---: | ---: | ---: |
| `lite6` | 98.2% / 97.9% | 4.4 / 4.4 mm | 84 | 2M |
| `xarm5` | 97.1% / 97.9% | 4.5 / 4.4 mm | 97 | 2M |
| `xarm6` | 100.0% / 100.0% | 4.0 / 4.1 mm | 68 | 2M |
| `xarm7` | 99.2% / 99.6% | 4.4 / 4.3 mm | 91 | 2M |
| `uf850` | 98.4% / 98.4% | 4.3 / 4.8 mm | 108 | 4M |

Trained once per arm with the production recipe above (recipe seed `0`) and measured with Python 3.13.13, MuJoCo 4.0.0, Stable-Baselines3 2.9.0 and PyTorch 2.10.0 (CPU inference).

Checkpoints work across MuJoCo versions. Evaluated on MuJoCo 3.16.0, these checkpoints give the same success counts except for one episode (`lite6` group 1: 98.4%). The checkpoints trained on MuJoCo 3 for 0.1.0 (`lite6` 99.8% / 99.4%, `xarm5` 99.2% / 99.0%, `xarm6` 100.0% / 99.8%, `xarm7` 99.6% / 99.0%, `uf850` 98.0% / 96.3%) give identical success counts on MuJoCo 3.16.0 and 4.0.0. The two sets differ because they come from different training runs.

