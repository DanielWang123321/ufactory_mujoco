# 0.0.5 强化学习示例开发计划：xArm 6 TCP 位置到达

版本：v1.0（2026-10-14）｜对应 [roadmap.md](../roadmap.md) 0.0.5

## 1. 目标与范围

**目标**：训练并部署一个强化学习策略，控制 xArm 6 在关节空间增量指令下将 TCP 移动到随机目标点，提供训练脚本、可交互回放示例与自动化测试。

- **做**：Gymnasium 标准接口任务环境、PPO 训练管线、策略评估与回放示例、单测与文档。
- **不做**：抓取/避障等扩展任务、真机部署（`RobotInterface` 抽象已预留，后续可将策略动作层接入 XArmRobot）、多算法对比。

## 2. 环境与依赖决策

| 项 | 决策 | 理由 |
|---|---|---|
| 训练环境 | **py313**（torch/gymnasium/sb3/trimesh 齐备） | py314 无 torch/gymnasium 轮子，且项目默认 py314 不变 |
| 核心环境代码 | 仅依赖 mujoco + numpy + gymnasium（可选） | `reach_env.py` 缺 gymnasium 时退化基类，单测按现有 trimesh 模式 skip |
| RL 算法 | **Stable-Baselines3 PPO**（`MlpPolicy` 2×64） | 样本效率与稳定性成熟，免维护自研训练器 |
| 依赖清单 | 新增 `requirements-rl.txt`：`gymnasium>=1.0`、`stable-baselines3>=2.4` | 不污染 `requirements.txt` 的 mujoco/numpy 核心依赖 |
| 备选（若 sb3 安装受阻） | 纯 numpy 交叉熵方法（CEM）训练脚本 | 牺牲收敛速度，保证 py314 可运行 |

## 3. 任务定义（MDP）

| 要素 | 定义 |
|---|---|
| 场景 | 新增 `models/xarm6/scene_reach.xml`：复用 `xarm6.xml` + mocap 目标球（仅视觉无碰撞，同 05 标记球做法），目标随 episode 移动 |
| 动作 | `Box(-1, 1, (6,))`，关节位置增量 `Δq = clip(a, -1, 1) * 0.05 rad`，钳制到关节限位 |
| 观测 | `float32` 21 维：TCP−目标相对位置(3) + 关节位置归一化(6) + 关节速度缩放(6) + 上一动作(6) |
| 控制频率 | 每动作 10 物理步 × 2 ms = 20 ms；episode 200 步 = 4 s |
| 奖励 | 势能塑形 `2.0*(d_prev − d)/0.01` + 成功 `+10`（d<15 mm 终止）− 步数惩罚 `0.02` − 动作代价 `0.001·‖a‖²` − 近限位惩罚（势能塑形保证最优策略不变性） |
| 终止 | 成功（d<15 mm）或 200 步截断 |
| 随机化 | 目标采样于基座系工作空间 `x∈[0.25,0.45], y∈[-0.25,0.25], z∈[0.05,0.35]`（启动时以 IK 校验可达）；初始 `q0 = ready + U(-0.2, 0.2) rad`，`reset(seed=...)` 全程可复现 |

## 4. 文件结构（新增）

```
models/xarm6/scene_reach.xml      # 任务场景（mocap 目标球）
src/rl/__init__.py
src/rl/reach_env.py               # XArm6ReachEnv：Gymnasium Env，独立 MjModel/MjData，不依赖 SB3
scripts/train_reach_rl.py         # PPO 训练/评估/保存（build/rl/ppo_reach.zip + metrics.json）
examples/07_rl_reach.py           # --policy 加载回放 / --train 现场训练；--episodes/--headless
tests/test_reach_env.py           # 环境合约与短训练冒烟测试
requirements-rl.txt
```

设计约束：
- `reach_env.py` 直接持有 MjModel/MjData（不走 `SimRobot`），避免 viewer/实时钟开销；但复用 `src/ik.py` 做目标可达性校验与成功判据旁证。
- 环境注册名 `XArm6Reach-v0`，暴露标准 `gymnasium.Env` API，便于换算法（SAC/TD3）或换后端。

## 5. 里程碑

| 里程碑 | 内容 | 验收标准 | 预估 |
|---|---|---|---|
| **M0 脚手架** | `requirements-rl.txt`；py313 安装并跑通 sb3 官方样例一步 | `import stable_baselines3` 成功，版本记录 worklog | 0.5 d |
| **M1 场景与环境** | `scene_reach.xml` + `reach_env.py` + `tests/test_reach_env.py` | 环境合约（shape/dtype/seed 复现/终止/限位钳制）单测全过；1000 步随机 rollout 无 NaN、无穿透接触 | 1 d |
| **M2 训练管线** | `train_reach_rl.py`：PPO、EvalCallback、checkpoint、metrics.json | 300k steps 训练完成，评估**成功率 ≥ 80%**、平均最终 TCP 误差 **< 20 mm**（50 episode） | 1 d |
| **M3 示例与评估** | `07_rl_reach.py`：策略回放（GUI/无头）、`--train` 捷径 | 无头 10 episode 指标输出且退出码正确；GUI 回放目标球随 episode 移动 | 1 d |
| **M4 回归与文档** | 全量测试、README/README_zh、worklog、roadmap 勾选 | py314 全套通过（RL 项 skip）、py313 全套通过（含 RL 项）；`git diff --check` 干净 | 0.5 d |

合计约 **4 人日**（含训练调参缓冲）。

## 6. 测试计划（`tests/test_reach_env.py`）

1. **合约**：obs/action shape 与 dtype、`reset(seed)` 可复现、`step` 五元组、truncation/termination 语义（gymnasium 存在时跑 `gymnasium.utils.env_checker.check_env`）。
2. **物理一致性**：动作钳制到关节限位；目标移动后 obs 跟随；奖励势能性质（靠近为正、远离为负）。
3. **可达性校验**：采样目标经 `solve_ik` 验证在工作空间内。
4. **训练冒烟**：sb3 可用时 2048 steps 训练 + `predict` 前向（限时 < 60 s），缺失时 `skip`。
5. **回归**：现有 89 项不回归；07 无头示例纳入手工验收。

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| py314 无 torch/gymnasium | 训练/评估在 py313；核心代码零 RL 依赖，测试 skip 模式；备选纯 numpy CEM |
| PPO 不收敛或样本效率低 | 奖励已用势能塑形；先课程化（目标距离由近及远）；再调 `n_steps/gae_lambda/ent_coef`；最后才换 SAC |
| 训练速度不足 | 无渲染 MjData-only + `SubprocVecEnv(4~8)`；控制降采样 10 步；预期 CPU ≥ 1000 env-steps/s/进程 |
| 目标采样不可达导致奖励稀疏 | 采样盒以 IK 预校验过滤；观测含相对位置，塑形稠密 |
| 策略过拟合 ready 附近 | q0 随机化 + 目标全域采样；评估集用固定 seed 独立于训练 seed |

## 8. 交付物清单（2026-10-14 全部完成）

- [x] `models/xarm6/scene_reach.xml`
- [x] `src/rl/reach_env.py` + `src/rl/__init__.py`
- [x] `scripts/train_reach_rl.py`
- [x] `examples/07_rl_reach.py`
- [x] `tests/test_reach_env.py`（15 项：合约/奖励/求解性/gym 校验/PPO 冒烟）
- [x] `requirements-rl.txt`
- [x] `build/rl/ppo_reach.zip` + `metrics.json`（训练产物，build/ 在 .gitignore 中）
- [x] README/README_zh 增补；worklog 记录；roadmap 勾选

## 9. 验收结果（2026-10-14）

- M2 训练验收（超额达成）：300k steps / 8 envs / 114 s（2708 fps），50 回合评估成功率 **96%**（验收 ≥80%）、平均最终 TCP 误差 **15.2 mm**（验收 <20 mm）、平均 98.5 步到达。
- 示例验收：`07 --headless` 10 回合 100% 成功、平均 14.7 mm；`--random` 基线 0%（对照鲜明）；`--target` 固定目标 5/5；GUI 回放正常；py314 无 RL 依赖时 `--random` 正常。
- 测试验收：py313 全套 119 项全过（0 skip），py314 全套 119 项 OK（13 skip：10 trimesh + 3 RL）。
- 计划外收益：修复 MuJoCo 3.10/3.15 keyframe 填充语义差异导致的自由关节道具清零（0.0.4 存量）与 BIO 惯量容差跨版本差异（详见 worklog）。
