# 强化学习示例：桌面抓取与放置 (`pick_place`)

[English](README.md) | [中文](README_zh.md)

基于 MuJoCo 与 Gymnasium 环境 `PickPlaceEnv`（`ufactory_mujoco/rl/pick_place_env.py`）的 `xArm 6 + Gripper G2` 桌面物块抓取与放置强化学习示例，复用批次 2–3 场景资产、任务配置与平行夹爪接口，支持严格检查点契约校验、独立评测与可视化回放。

## 1. 环境与 MDP 规格

- **场景与任务配置**：直接复用 `models/xarm6_with_gripper/scene_pick_place.xml`、`configs/pick_place/xarm6.yaml`（通过 SHA-256 绑定物理几何配置）与 `ufactory_mujoco.grippers`（`gap_to_position` / `position_to_gap`）。
- **动作空间**：`Box(-1.0, 1.0, (4,), float32)`（`cartesian_and_g2_gap_increment`）：
  - `action[0:3]`：末端笛卡尔设定点增量 $\Delta p \in [-5, 5]\text{ mm}$（每控制步 $20\text{ ms} = 10 \times 2\text{ ms}$ 物理步），受工作空间盒、$25\text{ mm}$ 末端牵引半径及阻尼最小二乘 IK（`attachment_site`，垂直向下抓取姿态）约束。
  - `action[3]`：Gripper G2 爪间开口增量 $\Delta g \in [-3, 3]\text{ mm}$，经 `gap_to_position` 转换为驱动关节指令；零动作保持当前末端设定点与夹爪开口。
- **观测空间**：`Box((46,), float32)`：
  - 包含归一化关节角（`6`）、缩放关节角速度（`6`）、末端位置（`3`）、物块位置（`3`）、物块四元数（`4`）、物块速度（`3`）、目标相对位置（`3`）、指腹中点相对物块偏差（`3`）、指令与实测夹爪开口（`2`）、末端设定点偏差（`3`）、双指接触力与接触标志（`4`）、上一步动作（`4`）及阶段完成标志（`[ever_lifted_clear, release_confirmed]`，`2`）。
- **严格过程与结果验收门禁（`PickPlaceMetrics`）**：
  - **起升前**：桌面拖拽位移 $\le 10\text{ mm}$、物块倾角 $\le 15^\circ$。
  - **安全抬升**：双指稳定夹持（接触力 $> 0.5\text{ N}$ 连续 $\ge 3$ 步）下，物块底面净空达到 $\ge 70\text{ mm}$。
  - **轻触放置与低位释放**：桌面支撑接触（净空 $\le 2\text{ mm}$ 且桌面法向力 $> 0.15\text{ N}$）后，末端下压深度 $\le 1.0\text{ mm}$，确认释放高度 $\le 3.0\text{ mm}$，且机械臂/夹爪与桌面零碰撞。
  - **成功保持**：物块距目标点（`[0.35, 0.10]`）水平误差 $\le 20\text{ mm}$、夹爪张开 $\ge 35\text{ mm}$、物块速度 $< 0.03\text{ m/s}$ 且连续稳定保持 **10 步**（$0.20\text{ s}$）。

## 2. 快速运行（评估与回放）

> **环境说明**：评估或回放 PPO 策略权重需安装 `stable-baselines3`（见 `requirements-rl.txt`）。`--expert`、`--zero-policy` 与 `--random` 基线仅需核心依赖即可运行。

```bash
# 严格评测训练好的 PPO 检查点（自动校验 config.yaml 与 checkpoint_manifest.json 哈希契约）
python examples/rl/pick_place/evaluate.py --episodes 20 --min-success-rate 0.90 --max-press-mm 1.0 --quiet-episodes

# 评测独立闭环物理专家基线（无需 stable-baselines3）
python examples/rl/pick_place/evaluate.py --expert --episodes 10 --quiet-episodes

# 评测全零动作或随机策略基线
python examples/rl/pick_place/evaluate.py --zero-policy --episodes 5 --quiet-episodes

# 交互式 GUI 回放（或 --headless 无头回放）
python examples/rl/pick_place/01_pick_place.py
python examples/rl/pick_place/01_pick_place.py --headless --episodes 5
python examples/rl/pick_place/01_pick_place.py --expert --headless --episodes 3
```

## 3. 策略训练与检查点契约

声明式超参数位于 `examples/rl/pick_place/recipe.yaml`（通过 SHA-256 绑定 `configs/pick_place/xarm6.yaml`）。使用 `scripts/train_pick_place_rl.py` 训练：

```bash
# 训练 xArm6 + Gripper G2 抓取放置 PPO 策略（专家预热 + PPO 强化学习 + 检查点多维排序优选）
python scripts/train_pick_place_rl.py --total-steps 4096 --n-envs 4

# 纯 PPO 无预热冒烟训练
python scripts/train_pick_place_rl.py --warmstart-episodes 0 --total-steps 256 --n-envs 1
```

训练产物保存在 `build/rl/pick_place/xarm6/`：
- `ppo_pick_place.zip`：优选出的最佳 PPO 策略权重
- `config.yaml`：完整固化的环境、奖励与训练配置（含 `config_sha256`）
- `ppo_pick_place.checkpoint_manifest.json`：绑定 `ppo_pick_place.zip`、`config.yaml` 与 `configs/pick_place/xarm6.yaml` 的 SHA-256 清单
- `provenance.json` 与 `metrics.json`：源码文件哈希溯源、运行环境版本与评测报告

## 4. 基准验收结果

20 回合评测结果（`--seed 17000`，`examples/rl/pick_place/evaluate.py`）：

| 策略 | 成功率 | 最大下压深度 (`<= 1.0 mm`) | 释放高度 (`<= 3.0 mm`) | 平均落点误差 (`<= 20.0 mm`) | 最大抬升净空 (`>= 70.0 mm`) | 平均步数 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ppo_pick_place.zip` | **100.0%** | **0.115 mm** | **0.00 mm** | **3.86 mm** | **92.41 mm** | 312 |
| `PickPlaceExpert` (`--expert`) | **100.0%** | **0.143 mm** | **0.00 mm** | **1.60 mm** | **90.15 mm** | 274 |
| `--zero-policy` | 0.0% | 0.000 mm | 0.00 mm | 300.00 mm | 0.00 mm | 380 |

