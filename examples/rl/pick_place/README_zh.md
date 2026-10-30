# 强化学习示例：桌面抓取与放置 (`pick_place`)

[English](README.md) | [中文](README_zh.md)

基于 MuJoCo 与 Gymnasium 环境 `PickPlaceEnv`（`ufactory_mujoco/rl/pick_place_env.py`）的 `xArm 6 + Gripper G2` 桌面物块抓取与放置强化学习示例，采用**纯 PPO 从零训练**（无专家轨迹或行为克隆预热），支持 v2 模型哈希契约校验、独立随机出生点评测与可视化回放。

## 1. 环境与 MDP 规格

- **场景、标定 G2 与模型契约（`env_version: 2`）**：使用 `models/xarm6_with_gripper/scene_pick_place.xml`、`configs/pick_place/xarm6.yaml`，以及标定 Gripper G2 四连杆非线性几何与限速 PI+PD 位置-力矩混合伺服控制器（`ufactory_mujoco.g2_model.G2Model`、`ufactory_mujoco.g2_controller.G2Controller`，`50 Hz` 控制频率，`50%` 速度 / `50%` 力控）。版本 2 通过 `runtime_model_sha256` 将场景、递归包含的 MJCF 和引用网格绑定到检查点。
- **随机出生域**：物块在 $x \in [0.30, 0.50]\text{ m}$、$y \in [-0.20, 0.20]\text{ m}$（$z = 15\text{ mm}$）内均匀随机生成，目标放置点为 $[0.30, 0.30]\text{ m}$。确定性评测使用 `make_spawns(seed, episodes)` 生成可复现出生序列并记录 `spawn_sha256`。
- **动作空间**：`Box(-1.0, 1.0, (4,), float32)`（`cartesian_and_g2_gap_increment`）：
  - `action[0:3]`：末端笛卡尔设定点增量 $\Delta p \in [-5, 5]\text{ mm}$（每控制步 $20\text{ ms} = 10 \times 2\text{ ms}$ 物理步），受工作空间盒、$25\text{ mm}$ 末端牵引半径、近桌面安全护栏及阻尼最小二乘 IK（`attachment_site`，垂直向下抓取姿态）约束。
  - `action[3]`：Gripper G2 爪间开口增量 $\Delta g \in [-3, 3]\text{ mm}$，由 `G2Controller` 闭环跟踪；零动作保持当前末端设定点与夹爪开口。
- **观测空间**：`Box((70,), float32)`：
  - 包含归一化关节角（`6`）、缩放关节角速度（`6`）、阶段门控末端位置（`3`）与末端线速度（`3`）、末端相对物块平面偏差与物块高度（`3`）、物块单位四元数（`4`）、缩放物块线速度（`3`）与角速度（`3`）、起升后目标相对位置（`3`）、左右指腹中心相对物块偏差（`6`）、指令与实测夹爪开口（`2`）、末端设定点偏差（`3`）、双指接触力与接触标志（`4`）、物块-桌面与机械臂-桌面接触载荷（`2`）、拖拽/倾角/保持进度比（`3`）、夹起/安全抬升标志（`2`）、下压比例与触桌状态（`3`）、释放高度比例与确认标志（`3`）、上一步动作（`4`）、里程碑奖励标志（`2`）以及剩余步数比例与抬升进度（`2`）。
- **严格过程与结果验收门禁（`PickPlaceMetrics`）**：
  - **起升前**：桌面拖拽位移 $\le 10\text{ mm}$、物块倾角 $\le 15^\circ$。
  - **安全抬升**：双指稳定夹持（接触力 $> 0.2\text{ N}$ 连续 $\ge 3$ 步）下，物块底面净空达到 $\ge 65\text{ mm}$（质心高度 $z \ge 80\text{ mm}$）。
  - **轻触放置与低位释放**：桌面支撑接触（净空 $\le 2\text{ mm}$ 且桌面法向力 $> 0.05\text{ N}$）后，末端下压深度 $\le 1.0\text{ mm}$，确认释放高度 $\le 3.0\text{ mm}$（双指接触力均 $< 0.05\text{ N}$），且机械臂/夹爪与桌面零碰撞。
  - **成功保持**：物块距目标点（`[0.30, 0.30]`）水平误差 $\le 20\text{ mm}$、夹爪张开 $\ge 35\text{ mm}$、物块速度 $< 0.03\text{ m/s}$ 且连续稳定保持 **10 步**（$0.20\text{ s}$）。

## 2. 快速运行（评估与回放）

> **环境说明**：评估或回放 PPO 策略权重需安装 `stable-baselines3`（见 `requirements-rl.txt`）；基线策略（`--expert`、`--zero-policy`、`--random-policy`）仅需核心依赖。

```bash
# 在 64 个随机出生点上严格评测纯 PPO v2 检查点（自动校验 config.yaml、checkpoint_manifest.json 与模型哈希）
python examples/rl/pick_place/evaluate.py --episodes 64 --seed 18001 --min-success-rate 0.95 --max-press-mm 1.0 --quiet-episodes

# 评测独立闭环物理专家基线（无需 stable-baselines3）
python examples/rl/pick_place/evaluate.py --expert --episodes 64 --seed 18001 --quiet-episodes

# 评测全零动作或随机策略基线
python examples/rl/pick_place/evaluate.py --zero-policy --episodes 5 --quiet-episodes

# 交互式 GUI 回放（或 --headless 无头回放）
python examples/rl/pick_place/01_pick_place.py
python examples/rl/pick_place/01_pick_place.py --headless --episodes 10 --min-success-rate 0.95 --max-press-mm 1.0
python examples/rl/pick_place/01_pick_place.py --expert --headless --episodes 5
```

## 3. 纯 PPO 从零训练与检查点契约

声明式超参数位于 `examples/rl/pick_place/recipe.yaml`（通过 SHA-256 绑定 `configs/pick_place/xarm6.yaml`）。`scripts/train_pick_place_rl.py` 完全使用纯 PPO 从随机初始化训练，不依赖任何专家轨迹或行为克隆预热：

```bash
# 多进程纯 PPO 从零完整训练（连续通过 64 回合严格评测门禁后自动早停）
python scripts/train_pick_place_rl.py --n-envs 128 --n-workers 24 --device cuda --seed 1

# CPU 快速冒烟训练
python scripts/train_pick_place_rl.py --total-steps 256 --n-envs 4 --n-workers 2 --n-steps 32 --batch-size 64 --eval-episodes 2 --device cpu
```

训练产物保存在 `build/rl/pick_place/xarm6_g2_v2/`：
- `ppo_pick_place.zip`：优选出的最佳纯 PPO 策略权重
- `config.yaml`：完整固化的环境、奖励与训练配置（含 `config_sha256`）
- `ppo_pick_place.checkpoint_manifest.json`：绑定 `ppo_pick_place.zip`、`config.yaml`、任务配置与 `runtime_model_sha256` 的 SHA-256 清单
- `provenance.json`、`metrics.json` 与 `eval_history.json`：源码文件哈希溯源、运行环境版本、周期评测历史与最终评测报告

旧版 v1（`46` 维观测）检查点缺少 `runtime_model_sha256` 且与 v2 `70` 维空间契约不兼容，常规加载与 `scripts/migrate_pick_place_checkpoint.py` 均会抛出 `ArtifactError` 予以拒绝。

## 4. 基准验收结果

64 回合随机出生域独立评测结果（$x \in [0.30, 0.50]\text{ m}$，$y \in [-0.20, 0.20]\text{ m} \to [0.30, 0.30]\text{ m}$，留出种子 `--seed 18001`，`spawn_sha256: 7a2ebbc7...ec608`，`examples/rl/pick_place/evaluate.py`）：

| 策略 | 成功率 (`>= 95%`) | 最大下压深度 (`<= 1.0 mm`) | 最大释放高度 (`<= 3.0 mm`) | 平均落点误差 (`<= 20.0 mm`) | 最大抬升净空 (`>= 65.0 mm`) | 平均步数 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Pure-PPO v2（从零训练）**（`ppo_pick_place.zip`） | **100.0% (64/64)** | **0.124 mm** | **0.105 mm** | **4.71 mm** | **70.98 mm** | 196.0 |
| `PickPlaceExpert`（`--expert`，v2） | **100.0% (64/64)** | **0.000 mm** | **1.797 mm** | **0.23 mm** | **97.75 mm** | 294.7 |

本机（`i7-14700F` + `RTX 4060 Ti`，`128` 并行环境 / `24` 进程）从 0 纯 PPO 训练至 `9,000,064` 步连续两轮达标自动早停，耗时 `19.5 min`（`~7,689 FPS`），碰撞、拖拽、倾覆与高空释放失败率均为 `0%`。
