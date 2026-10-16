# 强化学习示例：TCP 位置到达 (`reach`)

[English](README.md) | [中文](README_zh.md)

基于 MuJoCo 与 Gymnasium 环境 `ReachEnv`（`ufactory_mujoco/rl/reach_env.py`）的机械臂末端（TCP）三维位置到达强化学习示例，支持 UFACTORY 全系 5 款机械臂（`lite6`、`xarm5`、`xarm6`、`xarm7`、`uf850`）的 PPO 策略回放、无头评测与训练。

## 1. 环境与 MDP 规格

- **场景资产**：`models/<robot>/scene_reach.xml`（机械臂本体 + 视觉目标球 `target_mocap`，末端跟踪点为 `attachment_site`）。
- **动作空间**：`Box(-1.0, 1.0, (n,), float32)` —— 关节位置增量控制，每控制步（$20\text{ ms} = 10 \times 2\text{ ms}$ 物理步）最大增量为 $\Delta q \in [-0.05, 0.05]\text{ rad}$，自动钳制于关节限位内。
- **观测空间**：`Box((3 + 3n,), float32)`，$n$ 为关节数（`xarm5` 为 `18` 维，`lite6`/`xarm6`/`uf850` 为 `21` 维，`xarm7` 为 `24` 维）：
  - `obs[0:3]`：TCP 相对目标位置偏差 $p_{\text{tcp}} - p_{\text{target}}$（m）
  - `obs[3:3+n]`：按限位中点归一化的关节角度 $q_{\text{norm}} \in [-1.5, 1.5]$
  - `obs[3+n:3+2n]`：缩放关节角速度 $\dot{q} / 3.0 \in [-5.0, 5.0]$
  - `obs[3+2n:3+3n]`：上一控制步动作 $a_{t-1} \in [-1.0, 1.0]$
- **奖励与终止条件**：
  - 距离势能塑形 $200 \cdot (d_{t-1} - d_t)$，扣除步数代价（$0.02$）、动作平方惩罚（$0.001 \|a\|^2$）及近限位惩罚（距限位 $2\%$ 以内每关节扣除 $0.01$）。
  - **成功判定**：TCP 到目标点距离**严格小于 5 mm**（$d_t < 0.005\text{ m}$，立即终止回合并记为成功）。
  - **超时截断**：单回合上限 **200 控制步**（仿真时长 $4.0\text{ s}$）。
- **工作空间标定**：各机型的目标采样盒与肩部径向预筛带固化于 `ROBOT_PROFILES`（由 `scripts/measure_reach_workspace.py` 通过 FK 点云采样 + IK 可达率 $\ge 90\%$ 标定）。

## 2. 快速运行（评估与回放）

> **仓库不附带预训练权重**（`build/` 已被 git 忽略），需先按第 3 节训练，结果（`ppo_reach.zip` 和 `metrics.json`）保存在 `build/rl/<model>/`。训练环境加载“轻量可视”模型（去掉可视网格、显示碰撞网格，物理完全一致），每个进程约 **0.3 GB 内存**（原约 1.5 GB），`--n-envs` 默认按 CPU 核数和可用内存自动设置。实测（8 核 CPU、15 GB 内存、无 GPU；xArm 6，8 个环境，共约 2.9 GB）：快速配方（30 万步）约 **65 秒**，成功率仅约 54%；完整配方（200 万步，`256,256,128`）约 **16 分钟**，200 回合**成功率 100%、平均误差 3.9 mm**。

> **环境说明**：加载 PPO 权重回放需安装 `stable-baselines3`（见 `requirements-rl.txt`）。未训练基线 `--random` 只需核心依赖。

```bash
# 交互式 GUI 演示（默认 xArm 6，运行 10 个回合）
python examples/rl/reach/01_reach.py

# 切换其他机械臂型号
python examples/rl/reach/01_reach.py --model lite6
python examples/rl/reach/01_reach.py --model xarm5
python examples/rl/reach/01_reach.py --model xarm7
python examples/rl/reach/01_reach.py --model uf850

# 无头模式评估（输出成功率与末端误差指标，达标退出码为 0，否则为 1）
python examples/rl/reach/01_reach.py --model xarm6 --headless --episodes 20

# 指定基座坐标系下的固定目标点 (x, y, z，单位：米)
python examples/rl/reach/01_reach.py --target 0.35 0.15 0.25

# 未训练随机策略基线（无需 stable-baselines3）
python examples/rl/reach/01_reach.py --random --headless
```

### 命令行参数说明 (`01_reach.py`)

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--model` | `xarm6` | 机械臂型号：`lite6`、`xarm5`、`xarm6`、`xarm7`、`uf850` |
| `--policy` | `build/rl/<model>/ppo_reach.zip` | 训练好的 PPO 策略权重文件路径 |
| `--episodes` | `10` | 评估/演示回合数 |
| `--seed` | `0` | 起始随机种子（第 `ep` 回合使用 `seed + ep`） |
| `--device` | `cpu` | PyTorch 推理设备（`cpu` 或 `cuda`） |
| `--headless` | `False` | 关闭交互视窗，纯终端输出评测指标 |
| `--random` | `False` | 使用均匀随机动作基线（不加载策略权重） |
| `--target X Y Z` | `None` | 固定目标点基座坐标（米），不指定则随机采样 |
| `--min-success-rate` | `0.95` | 验收通过的最低成功率阈值 |
| `--max-mean-dist-mm` | `10.0` | 验收通过的最大平均终止距离上限（mm） |

## 3. 策略训练

使用 `scripts/train_reach_rl.py` 训练或重训各机型策略：

```bash
# 默认快速训练（300k 步，[64,64] MLP，并行环境数自动）
python scripts/train_reach_rl.py --model xarm6

# 对齐 5 mm 验收标准的完整训练配方（2M 步，[256,256,128] MLP）
python scripts/train_reach_rl.py \
    --model xarm6 --total-steps 2000000 --net-arch 256,256,128 --ent-coef 0.001 --eval-episodes 200
```

训练产物保存在 `build/rl/<model>/` 下，包含最终策略 `ppo_reach.zip`、阶段检查点 `checkpoints/` 以及评估报告 `metrics.json`。

## 4. 基准验收结果

对齐 `ufactory_genesis` reach 评测标准（单回合成功阈值 `< 5 mm`；每组 512 个独立目标要求**成功率 $\ge 95\%$** 且**平均终止距离 $\le 10\text{ mm}$**，使用 `--seed 17000` 与 `--seed 17512` 两组独立种子验收）：

| 机型 | 成功率（组1 / 组2） | 平均终止距离（组1 / 组2） | 平均步数 | 训练步数 |
| :--- | ---: | ---: | ---: | ---: |
| `lite6` | 99.8% / 99.4% | 4.3 / 4.4 mm | 82 | 2M |
| `xarm5` | 99.4% / 99.0% | 4.2 / 4.2 mm | 95 | 2M |
| `xarm6` | 100.0% / 99.8% | 4.1 / 4.1 mm | 72 | 2M |
| `xarm7` | 99.6% / 99.0% | 4.3 / 4.3 mm | 88 | 2M |
| `uf850` | 98.1% / 96.3% | 4.4 / 4.9 mm | 110 | 4M |

