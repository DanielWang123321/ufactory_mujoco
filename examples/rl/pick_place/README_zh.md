# 强化学习示例：桌面抓取与放置 (`pick_place`)

[English](README.md) | [中文](README_zh.md)

基于 MuJoCo 与 Gymnasium 环境 `PickPlaceEnv`（`ufactory_mujoco/rl/pick_place_env.py`）的 `xArm 6 + Gripper G2` 桌面物块抓取与放置强化学习示例，采用**纯 PPO 从零训练**（无专家轨迹或行为克隆预热），支持 v2 模型哈希契约校验、确定性留出出生点评测与可视化回放。

## 1. 环境与 MDP 规格

- **场景、标定 G2 与模型契约（`env_version: 2`）**：使用 `models/xarm6_with_gripper/scene_pick_place.xml`、`configs/pick_place/xarm6.yaml`，以及标定 Gripper G2 四连杆几何与力曲线（`ufactory_mujoco.g2_model.G2Model`）和限速位置/速度伺服（条件积分抗饱和、按力控百分比限幅力矩；`ufactory_mujoco.g2_controller.ScalarG2Controller`，即 `G2Controller` 的单环境实现，结果逐位相同；`50 Hz`，`50%` 速度 / `50%` 力控）。版本 2 通过 `runtime_model_sha256` 将场景、递归包含的 MJCF 和引用网格绑定到检查点。
- **回合与初始状态**：每回合最多 `400` 步（$8\text{ s}$，超时截断）。机械臂从固定 home 位出发（TCP $(0.30, 0.00, 0.1265)\text{ m}$，垂直向下），夹爪预张 $50\text{ mm}$；物块边长 $30\text{ mm}$、质量 $17\text{ g}$。
- **随机出生域**：物块在 $x \in [0.30, 0.50]\text{ m}$、$y \in [-0.20, 0.20]\text{ m}$（$z = 15\text{ mm}$）内均匀随机生成，目标放置点为 $[0.30, 0.30]\text{ m}$。确定性评测使用 `make_spawns(episodes, seed=...)`（`ufactory_mujoco.rl.evaluation`）生成可复现出生序列并记录 `spawn_sha256`。
- **动作空间**：`Box(-1.0, 1.0, (4,), float32)`（`cartesian_and_g2_gap_increment`）：
  - `action[0:3]`：末端笛卡尔设定点增量 $\Delta p \in [-5, 5]\text{ mm}$（每控制步 $20\text{ ms} = 10 \times 2\text{ ms}$ 物理步），经工作空间盒裁剪与 $25\text{ mm}$ 末端牵引半径限制后，由阻尼最小二乘 IK（`attachment_site`，垂直向下抓取姿态）求解；IK 失败时保持原设定点。
  - `action[3]`：爪间开口增量 $\Delta g \in [-3, 3]\text{ mm}$，指令开口限幅在 $[12, 84]\text{ mm}$，由该伺服闭环跟踪。零动作保持当前末端设定点与夹爪开口。
  - 安全护栏：起升前 TCP 设定点不低于 $5.5\text{ mm}$，夹住物块后禁止下压；起升后物块离桌 $\le 20\text{ mm}$ 时下降速度限制在 $20\text{ mm/s}$，末端设定点不低于估计触桌高度下方 $0.4\text{ mm}$。
- **观测空间**：`Box(-10, 10, (70,), float32)`，按顺序为：
  - 归一化关节角（`6`）、缩放关节角速度（`6`）、门控末端位置（`3`，起升前仅高度，起升后为相对目标点的水平偏差与高度）、缩放末端线速度（`3`）；
  - 物块相对末端的水平偏差与物块高度（`3`）、物块单位四元数（`4`）、缩放物块线速度（`3`）与角速度（`3`）、起升后目标相对物块位置（`3`）、左右指腹中心相对物块偏差（`6`）；
  - 指令与实测夹爪开口（`2`）、末端设定点偏差（`3`）、双指接触力（`2`）与接触标志（`2`）、物块-桌面（起升后）与机械臂-桌面接触载荷（`2`）、上一步动作（`4`）；
  - 拖拽/倾角/保持进度比（`3`）、夹起/安全抬升标志（`2`）、下压比例/触桌标志/触桌后末端下沉（`3`）、释放高度比例/释放确认/高位释放标志（`3`）、里程碑奖励标志（`2`）、剩余步数比例与抬升进度（`2`）。
- **验收门禁（`PickPlaceMetrics`）**：
  - **起升前**：桌面拖拽位移 $\le 10\text{ mm}$、物块倾角 $\le 15^\circ$。
  - **安全抬升**：双指稳定夹持（接触力 $> 0.2\text{ N}$ 连续 $\ge 3$ 步，$< 0.05\text{ N}$ 视为松开）下，物块底面净空 $\ge 65\text{ mm}$（质心 $z \ge 80\text{ mm}$）。
  - **轻触放置与低位释放**：桌面支撑接触（净空 $\le 2\text{ mm}$ 且桌面法向力 $> 0.05\text{ N}$）后末端下压 $\le 1.0\text{ mm}$；释放高度 $\le 3.0\text{ mm}$（双指接触力均 $< 0.05\text{ N}$ 连续 3 步才确认释放）。
  - **成功**：物块距目标点水平误差 $\le 20\text{ mm}$、倾角 $\le 15^\circ$、桌面支撑、夹爪张开 $\ge 35\text{ mm}$、物块速度 $< 0.03\text{ m/s}$，连续保持 **10 步**（$0.20\text{ s}$）。
  - **失败即终止**：拖拽或倾角超限、掉落（夹起后物块落在目标区外且连续 15 步无夹持）、下压超限、高位释放、机械臂/夹爪与桌面接触力 $> 0.2\text{ N}$。

## 2. 快速运行（评估与回放）

> **依赖**：评估或回放 PPO 权重需安装 `rl` 可选依赖（Stable-Baselines3），见[安装说明](../../../README_zh.md#安装)。脚本专家与基线策略（`--expert`、`--zero-policy`、`--random`）仅需核心安装。不发布训练好的权重，需按第 3 节训练。

```bash
# 在 64 个留出出生点上评测纯 PPO v2 检查点（先校验 config.yaml、checkpoint_manifest.json 与模型哈希）
python examples/rl/pick_place/evaluate.py --episodes 64 --seed 18001 --min-success-rate 0.95 --max-press-mm 1.0 --quiet-episodes

# 评测脚本专家（无需 stable-baselines3）
python examples/rl/pick_place/evaluate.py --expert --episodes 64 --seed 18001 --quiet-episodes

# 评测全零动作或均匀随机基线
python examples/rl/pick_place/evaluate.py --zero-policy --episodes 5 --quiet-episodes
python examples/rl/pick_place/evaluate.py --random --episodes 5 --quiet-episodes

# 保存汇总与逐回合 JSON/CSV 报告
python examples/rl/pick_place/evaluate.py --episodes 64 --seed 18001 --quiet-episodes --report build/rl/pick_place/eval_64.json

# 交互式 GUI 回放（或 --headless 无头回放）
python examples/rl/pick_place/01_pick_place.py
python examples/rl/pick_place/01_pick_place.py --headless --episodes 10 --min-success-rate 0.95 --max-press-mm 1.0
python examples/rl/pick_place/01_pick_place.py --expert --headless --episodes 5
```

- `evaluate.py` 的 PASS 只判三项：成功率 $\ge$ `--min-success-rate`（默认 `0.90`）、最大下压 $\le$ `--max-press-mm`、平均落点误差 $\le$ `--max-target-error-mm`（默认 `20 mm`）。碰撞、拖拽、倾覆、掉落、高位释放等失败计数见输出 JSON。
- `PickPlaceExpert`（`--expert`）是脚本专家：基于仿真特权状态（物块与指腹真值、`PickPlaceMetrics` 内部状态）的闭环状态机，不参与训练，仅用于验证任务在相同约束与门禁下可完成。

### 命令行参数（`evaluate.py`、`01_pick_place.py`）

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--checkpoint`、`--policy` | `build/rl/pick_place/xarm6_g2_v2/ppo_pick_place.zip` | PPO 检查点（`evaluate.py` 另支持 `--ckpt`） |
| `--expert` / `--zero-policy` / `--random` | 关 | 用脚本专家、全零或均匀随机基线代替检查点 |
| `--episodes` | `10`（`01`：`5`） | 评测回合数 |
| `--seed` | `17000` | 出生点种子；第 `i` 回合使用 `seed + i` |
| `--recipe` | `examples/rl/pick_place/recipe.yaml` | 用于校验检查点契约的配方 |
| `--runtime-config` | `configs/pick_place/xarm6.yaml` | 任务配置覆盖（仅 `evaluate.py`） |
| `--device` | `cpu` | PyTorch 设备 |
| `--min-success-rate` | `0.90`（基线：`0`） | PASS 的成功率阈值 |
| `--max-press-mm` | `1.0` | PASS 的最大下压阈值 |
| `--max-target-error-mm` | `20.0` | PASS 的平均落点误差阈值 |
| `--report`、`--summary-json`、`--out` | `None` | 将汇总与逐回合记录写入 JSON（及 CSV） |
| `--quiet-episodes` | `False` | 只打印汇总（`evaluate.py`） |
| `--headless` | `False` | 不开视窗运行（`01_pick_place.py`） |

PASS 判定未通过时两个脚本均以退出码 `1` 结束；基线不做判定。

## 3. 纯 PPO 从零训练与检查点契约

声明式超参数位于 `examples/rl/pick_place/recipe.yaml`。`scripts/train_pick_place_rl.py` 使用 Stable-Baselines3 PPO 从随机初始化训练，不依赖专家轨迹或行为克隆预热：

```bash
# 多进程完整训练（默认输出到 build/rl/pick_place/xarm6_g2_v2/，会覆盖其中现有权重）
python scripts/train_pick_place_rl.py --n-envs 128 --n-workers 24 --device cuda --seed 1

# 第 5 节种子实验的配置：固定并行布局、单个 PyTorch 线程、关闭周期评测
python scripts/train_pick_place_rl.py --n-envs 128 --n-workers 3 --worker-env-counts 64,32,32 \
    --torch-threads 1 --device cuda --seed 1 --total-steps 10000000 --disable-periodic-eval \
    --out-dir build/rl/pick_place/seed_1

# CPU 快速冒烟训练（单独输出目录，避免覆盖默认权重）
python scripts/train_pick_place_rl.py --total-steps 256 --n-envs 4 --n-workers 2 --n-steps 32 --batch-size 64 --eval-episodes 2 --device cpu --out-dir build/rl/pick_place/smoke
```

**训练策略**：
- **奖励**：势函数塑形 $r = \Phi(s') - \Phi(s) - 0.015$（`reward_gamma: 1.0`）。$\Phi$ 由接近、预抓取、抓握、抬升（权重最大，按历史最大夹持高度计）、运送、放置、释放、保持各阶段项加权组成；失败时 $\Phi(s')$ 置 0，扣回已积累的势能。另加一次性里程碑奖励（首次接触、夹起、抬升、触桌、放置，成功 `+40`）与安全惩罚（下压、近桌快速下降、高位释放、碰桌、桌面过载、动作幅度，失败 `-10`）。
- **PPO**：MLP `[256, 256]`（ELU）、`n_steps 128`、`batch_size 2048`、`gamma 0.995` 等见 `recipe.yaml`；并行环境数由 `--n-envs` 指定。
- **评测与早停**：每 `1M` 步在 `seed = 17000 + 训练种子` 的 64 个出生点上确定性评测。门禁为成功率 $\ge 95\%$、最大下压 $\le 1.0\text{ mm}$、平均落点误差 $\le 20\text{ mm}$，且碰撞、拖拽、倾覆、高位释放均为 0；连续 2 次通过即早停，否则训练至 `total_steps`（`50M`）。
- **选模型**：按 `checkpoint_score`（`ufactory_mujoco.rl.evaluation`）依次比较成功率、低位释放率、目标区触桌率、最大下压、平均落点误差等，最优者复制为 `ppo_pick_place.zip`。

**训练参数**（未指定时取自 `recipe.yaml`；完整列表见 `--help`）：

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--total-steps`、`--seed` | `50000000`、`1` | 训练预算与种子 |
| `--n-envs`、`--n-workers` | 按 CPU 核数与内存自动设置；`min(n_envs, CPU 数)` | 并行环境数与工作进程数 |
| `--worker-env-counts`、`--worker-cpu-sets` | 平均分配、不绑核 | 各进程环境数（如 `64,32,32`）与各进程 CPU 编号（如 `0,16;2,18`） |
| `--device`、`--torch-threads` | `cpu`、PyTorch 默认 | PyTorch 设备与线程数 |
| `--deterministic` | 关 | 启用 PyTorch 确定性算法并关闭 cuDNN benchmark |
| `--eval-interval`、`--eval-episodes`、`--early-stop-evaluations`、`--min-success-rate` | `1000000`、`64`、`2`、`0.95` | 周期评测与早停 |
| `--disable-periodic-eval`、`--no-final-eval` | 关 | 跳过周期评测或最终评测（用于外部评测） |
| `--save-interval`、`--checkpoint-steps` | `1000000`、无 | 周期检查点；在指定步数额外保存 |
| `--progress-json` | 无 | 约每 5 秒写入当前步数与吞吐 |
| `--resume` | 无 | 从 SB3 检查点继续训练 |
| `--recipe`、`--runtime-config` | `recipe.yaml`、`configs/pick_place/xarm6.yaml` | 配方与任务配置 |
| `--spawn-xy-lower`、`--spawn-xy-upper` | 配方 | 覆盖出生域，格式 `x,y` |
| `--n-steps`、`--batch-size`、`--n-epochs`、`--learning-rate`、`--clip-range`、`--vf-coef`、`--ent-coef`、`--max-grad-norm`、`--net-arch`、`--activation`、`--init-std` | 配方 | PPO 超参数覆盖 |
| `--out-dir` | `build/rl/pick_place/xarm6_g2_v2` | 输出目录；默认值会覆盖示例加载的权重 |

**训练产物**（默认 `build/rl/pick_place/xarm6_g2_v2/`）：
- `ppo_pick_place.zip`：选出的最佳策略权重
- `config.yaml`：完整固化的环境、奖励与训练配置（含 `config_sha256`）；`env` 中记录任务配置哈希 `runtime_config_sha256` 与场景模型哈希 `runtime_model_sha256`
- `ppo_pick_place.checkpoint_manifest.json`：记录 `ppo_pick_place.zip`、`config.yaml` 与任务配置的 SHA-256 及观测/动作维度
- `provenance.json`、`metrics.json`、`eval_history.json`：源码文件哈希与依赖版本、最终评测指标与训练耗时、周期评测历史
- `checkpoints/`：每 `1M` 步快照、`ppo_pick_place_best.zip`、门禁通过快照 `ppo_pick_place_pass_<N>.zip` 与 `ppo_pick_place_final.zip`

**加载校验**：`load_validated_ppo` 在加载前校验 zip 与 `config.yaml` 哈希及观测/动作维度，用当前 `recipe.yaml` 与 `configs/pick_place/xarm6.yaml` 重新解析环境配置并逐项比对，同时按当前场景重算 `runtime_model_sha256`；任一不符即抛出 `ArtifactError`。旧版 v1（`46` 维观测）检查点缺少 `runtime_model_sha256` 且与 v2 的 `70` 维观测不兼容，加载时会被拒绝，需用当前配方重新训练。

## 4. 基准验收结果

参考权重 `ppo_pick_place.zip`（本机训练种子 `1`，不发布）与脚本专家在留出出生点上的结果（出生域 $x \in [0.30, 0.50]\text{ m}$，$y \in [-0.20, 0.20]\text{ m} \to [0.30, 0.30]\text{ m}$，`examples/rl/pick_place/evaluate.py`）：

| 策略 | 评测集 | 成功率 | 最大下压 (`<= 1.0 mm`) | 最大释放高度 (`<= 3.0 mm`) | 平均落点误差 (`<= 20.0 mm`) | 最大抬升净空 (`>= 65.0 mm`) | 平均步数 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Pure-PPO v2**（`ppo_pick_place.zip`） | 64 回合，`--seed 18001` | **100.0% (64/64)** | 0.132 mm | 0.112 mm | 5.17 mm | 71.60 mm | 195.4 |
| **Pure-PPO v2**（`ppo_pick_place.zip`） | 1000 回合，`--seed 30001–34001` 各 200 | **98.1% (981/1000)** | 0.186 mm | 0.462 mm | 12.93 mm\* | 71.92 mm | 191.9 |
| `PickPlaceExpert` 脚本专家（`--expert`） | 64 回合，`--seed 18001` | **100.0% (64/64)** | 0.000 mm | 1.799 mm | 0.35 mm | 97.72 mm | 294.8 |

- `--seed 18001` 的 `spawn_sha256` 为 `7a2ebbc7...ec608`。1000 回合中失败 19 次：倾覆 18、掉落 1，无碰撞、拖拽、下压超限或高位释放。\*平均落点误差含失败回合，不能与 64 回合结果直接比较。
- 测试环境：Python 3.13.13、MuJoCo 3.16.0、SB3 2.9.0、PyTorch 2.10.0（CPU 推理）；脚本专家在 MuJoCo 3.15.0 下结果相同。0.1.0 之前该权重在 MuJoCo 3.10.0 下测得的成功率相同（64/64 与 98.1%）。
- 参考权重在本机（`i7-14700F` + `RTX 4060 Ti`，`128` 并行环境 / `24` 进程）以种子 `1` 训练，`9,000,064` 步时连续两次通过门禁早停，耗时 `19.5 min`（约 `7,689 FPS`）。
- 环境 CPU 开销优化后（G2 标量控制器、几何缓存、势函数复用、IK 预计算；环境行为逐位不变），同机同配置的训练稳态吞吐（不含周期评测）由约 `9,700` 提升到约 `14,700` 步/s（约 `1.5` 倍）。

## 5. 训练稳定性（种子敏感性）

当前配方的训练结果强依赖随机种子，参考权重不代表任意种子都能达到同等结果。以下实验均在 0.1.0 之前、MuJoCo 3.10.0 下完成。

在服务器（`i9-14900K` + `RTX 5090`，软件版本同上，配置同第 3 节命令）上用 20 个种子（`10001–29001`）各训练 `50M` 步（关闭周期评测与早停），在 10M / 20M / 50M 步处各用 1000 个留出回合（评测种子 `50001–54001`）评测：

| 训练步数 | 成功率 `>= 95%` 的种子 | 成功率 0% 的种子 | 平均成功率 |
| :--- | ---: | ---: | ---: |
| 10M | 10/20 | 7/20 | 58.5% |
| 20M | 14/20 | 6/20 | 69.6% |
| 50M | 14/20 | 3/20 | 79.9% |

- **同机同种子可逐位复现**：在固定并行布局（`--n-envs 128 --n-workers 3 --worker-env-counts 64,32,32`，绑核）与 `--torch-threads 1` 下，每个种子重复训练 3 次，策略参数哈希与逐回合评测记录完全一致；加 `--deterministic` 结果相同。
- **种子结果不能跨机器移植**：同配方训练 10M 步，种子 `1` 在本机为 100%、在服务器为 0%；种子 `3001` / `4001` 在本机为 0%、在服务器约 97%。原因尚未定位（两机 GPU 与并行布局不同）。
- **训练更久不一定更好**：种子 `12001`、`15001`、`21001` 在 50M 时比 20M 退步（86.4%、86.7%、43.2%）。
- **训练门禁偏乐观**：1000 回合下，成功率 $\ge 95\%$ 的模型多数仍有少量倾覆、拖拽或高位释放失败；按训练门禁标准（第 3 节）判定，通过的种子仅 3/20（10M）、6/20（20M）、3/20（50M）。
- **建议**：用多个种子训练，并在独立留出出生点上按评测结果选模型。
