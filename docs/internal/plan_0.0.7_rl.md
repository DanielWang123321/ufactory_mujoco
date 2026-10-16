# 0.0.7 开发计划：强化学习 reach 示例扩展到全系 5 款机械臂

> 目标：把 0.0.5 的 xArm 6 TCP reach 强化学习示例推广到 lite6、xarm5、xarm7、uf850，使全系 5 款机械臂都有"训练 + 评估 + 演示"完整闭环；环境、脚本、示例统一为按机型参数化的单一实现。
>
> 进度：M0、M1、M2、M3 完成（2026-10-14）；0.0.7 全部完成，待提交。

## 1. 目标

- 一个通用 `ReachEnv`（替代 `XArm6ReachEnv`），`robot` 参数选择机型，覆盖 lite6 / xarm5 / xarm6 / xarm7 / uf850。
- 每款机型一个 `models/<robot>/scene_reach.xml` 任务场景（xarm6 已有）。
- `scripts/train_reach_rl.py --model` 与 `examples/rl/01_reach.py --model` 支持全机型训练、评估、演示。
- 每款机型交付一个训练好的 PPO 检查点（`models/<robot>/rl/`）与量化指标（成功率 / 到位误差）。

## 2. 范围

**做**：reach 环境参数化、4 个新任务场景、per-机型工作空间标定、训练脚本/示例多机型化、4 轮 PPO 训练与指标记录、测试参数化扩至全系、文档与发布收尾。

**不做**（留给后续版本）：多机型策略互换/迁移学习、非 xArm6 机型的避障场景与轨迹规划示例（0.0.6 交付物只覆盖 xarm6）、domain randomization、其他 RL 任务（push/peg 等）。

## 3. 现状基线（2026-10-14）

- 环境 `src/rl/reach_env.py::XArm6ReachEnv`：21 维观测 = TCP-target 偏差(3) + 关节角(n) + 关节速度(n) + 上一动作(n)，动作 = 关节增量；奖励= 势能塑形 + 成功 +10，15 mm 阈值，200 步上限。**机型耦合点**（本次全部解耦）：
  1. `_DEFAULT_SCENE` 固定 xarm6；2. `ARM_JOINTS/ARM_ACTUATORS` 固定 6 轴名；3. 工作空间盒 `TARGET_LOW/HIGH`、肩部高度 `_SHOULDER`、径向带 `_REACH_MIN/MAX` 为 xArm 6 标定值；4. `_OBS_LOW/HIGH` 尺寸写死 21 维。
- 场景 `models/xarm6/scene_reach.xml`：arm + `target_mocap` 目标球（无关键帧、无障碍物，模板简单可复制）。
- 工具链：`scripts/train_reach_rl.py`（PPO 300k、8 并行、`models/xarm6/rl/` 检查点）、`examples/rl/01_reach.py`（随机/策略对比）均硬编码 xArm 6。
- 机型差异标定（已核实）：
  | 机型 | 关节 | 执行器名 | 备注 |
  |---|---|---|---|
  | lite6 | joint1–6 | **无名（''×6）** | 名字查找不到，需按**顺序**回退解析 |
  | xarm5 | joint1–5 | act1–5 | 5 轴，观测/动作维度 24/5 |
  | xarm6 | joint1–6 | act1–6 | 基线（96% 成功率、15.2 mm） |
  | xarm7 | joint1–7 | act1–7 | 7 轴，观测/动作维度 28/7 |
  | uf850 | joint1–6 | act1–6 | 几何/工作空间与 xArm 系差异大 |

## 4. 策略与关键决策

1. **单一通用环境 + 机型 profile 表**，不为每款机型复制环境类：`ReachEnv(robot="lite6")`，内部从 `ROBOT_PROFILES` 取轴数、工作空间盒、肩部位置、径向带；观测边界按 n 动态构造（dim = 3 + 3n）。跨机型奖励/成功阈值/200 步上限**保持一致**（可比性）。
2. **执行器按"名字优先、顺序回退"解析**：`act1..actN` 存在则用之（xarm5/6/7/uf850），否则取前 N 个执行器（lite6）；关节名 `joint1..jointN` 全系一致，直接按名解析。这不修改模型资产（不破坏 Menagerie 同步）。
3. **工作空间 profile 以 FK 采样 + IK 可达率标定**：均匀采样关节限位内构型 2 万点得 TCP 点云，为每机型选目标盒（盒内均匀采样点经环境自带可达性 IK 校验，通过率 ≥ 90%）；冻结为 `ROBOT_PROFILES` 常量表，并以**回归测试**（固定种子、可达率阈值）保证表不腐化。标定还暴露了 xArm 6 现役盒远下角点径向 0.5586 m 超出原 `reach_max=0.55` 的预筛误杀区，已上调至 0.56 使「盒 ⊆ 径向带」对全机型成立。
4. **检查点按机型落盘** `build/rl/<model>/ppo_reach.zip`（沿用 0.0.5 的 build/rl 布局，按机型分目录；build/ 不入库）；`examples/rl/01_reach.py` 加 `--model` 自动加载对应检查点（缺失时提示训练命令）。不新增 `rl/02_*`（0.0.6 的编号决策仅在有新示例文件时顺延；本版本是"一个示例多机型"）。
5. **`XArm6ReachEnv` 保留一行别名**（= `ReachEnv` 默认 xarm6），标注 0.1.0 移除——代码 API 与 CLI 迁移不同，给外部导入留缓冲。
6. 训练预算每机型 300k 步（PPO、8 并行、种子固定，与 0.0.5 同超参），评估 200 个确定性回合。

## 5. 技术方案（API 与资产）

```python
# src/rl/reach_env.py
@dataclass(frozen=True)
class RobotProfile:
    robot: str                  # "lite6" / ...
    n_joints: int               # 5 / 6 / 7
    target_low: np.ndarray      # 目标采样盒下界 (3,)
    target_high: np.ndarray     # 上界 (3,)
    shoulder: np.ndarray        # 径向可达预筛用肩部原点 (3,)
    reach_min: float            # 径向带下限 (m)
    reach_max: float            # 上限 (m)

ROBOT_PROFILES: dict[str, RobotProfile]   # 5 机型标定值（M0 产出）

class ReachEnv(gym.Env):
    def __init__(self, model_path=None, *, robot: str = "xarm6",
                 render_mode=None, max_steps=200, decimation=10, delta=0.05,
                 success_dist=0.015, target_low=None, target_high=None,
                 q0_noise=0.2, validate_targets=True): ...
    # model_path 缺省 = models/<robot>/scene_reach.xml；其余接口不变

XArm6ReachEnv = ReachEnv      # 兼容别名，0.1.0 移除
```

- **场景资产**（4 新增）：`models/{lite6,xarm5,xarm7,uf850}/scene_reach.xml`，按 xarm6 版模板派生（include 各机型 `<robot>.xml` + `target_mocap` 目标球 15 mm，statistic/相机参数按臂展缩放），与 xarm6 版结构保持一致。
- **训练脚本**：`scripts/train_reach_rl.py --model {lite6,xarm5,xarm6,xarm7,uf850}`（默认 xarm6）；`make_env()` 工厂闭包解决 `make_vec_env` 的类传参；评估摘要按机型输出。
- **示例**：`examples/rl/01_reach.py --model ...`，其余逻辑不变。

## 6. 测试计划

- **环境契约参数化**（`tests/test_reach_env.py`，无 gymnasium 自动 skip）：对 5 机型逐一验证——观测/动作维度 = 3+3n / n、reset/step 契约、限位裁剪、成功判定、可达目标产生正奖励；执行器解析（lite6 走顺序回退，xarm5/7 轴数正确）。
- **profile 防腐化测试**：固定种子在各机型盒内采样目标，可达性 IK 校验通过率 ≥ 90%；盒各角点径向距离落在 `[reach_min, reach_max]`。
- **场景资产测试**：5 个 `scene_reach.xml` 可加载、含 `target_mocap` 与 `attachment_site`，执行器/关节数与 profile 一致。
- **既有保留**：0.0.5 的奖励塑形/边界/端到端回合测试（xarm6 语义不变）；双环境全量回归（py314 无 RL 依赖照旧 skip）。

## 7. 文档与收尾

- README/README_zh：多机型训练/评估命令块 + 训练结果表（5 机型）。
- `docs/plan_0.0.7_rl.md`：进度行 + 验收结果回填。
- roadmap 0.0.7 勾选；worklog 按里程碑记录。

## 8. 验收标准

1. `python -m unittest discover -s tests` 在 py313、py314 **双环境全量通过**（skip 规则不变）；`git diff --check` 干净。
2. 每机型 `examples/rl/01_reach.py --model <m> --headless` 输出 `RL reach SUCCESS`，随机策略对照存在；GUI 模式 5 机型均可演示。
3. 训练指标（200 确定性回合）：**每机型成功率 ≥ 85%**、平均到位误差 < 20 mm（xarm6 基线 96% / 15.2 mm；实测值全部记入验收表）。
4. profile 测试与场景资产测试全绿；lite6 顺序回退执行器路径有专项测试。

## 8a. 验收结果（2026-10-14，全部通过）

- **训练指标**：首轮 300k 步四款新机型成功率仅 16.5–41.5%（诊断：教师 73 步即达、失败距离 34–40 mm，属欠训练而非时间预算）；加码至 **1M 步**后全系达标。经 `examples/rl/01_reach.py --headless --episodes 200`（独立种子、确定性策略）验收：

| 机型 | 成功率 | 平均到位误差 | 平均步数 | 训练时长 |
| :--- | ---: | ---: | ---: | ---: |
| lite6 | **99.5%** | 14.2 mm | 85 | ~5 min |
| xarm5 | **95.5%** | 14.6 mm | 98 | ~5 min |
| xarm6 | **99.5%** | 14.2 mm | 69 | ~5 min |
| xarm7 | **99.0%** | 14.3 mm | 87 | ~5 min |
| uf850 | **96.0%** | 14.3 mm | 112 | ~5 min |

- **验收命令退出码全 0**（--min-success-rate 0.85）；GUI 回放冒烟（lite6 2/2 成功）通过；随机基线对照保留。
- **测试**：`tests/test_reach_env.py` 24 项（多机型契约 ×5、IK 教师全机型、lite6 执行器回退、命名执行器、场景资产 ×5、profile 防腐化 ×2）；全套回归通过（见 worklog 最终数字）。

## 8b. 修订：成功阈值与验收标准对齐 ufactory_genesis（5 mm，2026-10-14）

原 15 mm 阈值改为 **5 mm**（`ReachEnv(success_dist=0.005)` 默认值），与 ufactory_genesis 评测标准一致（其 `examples/rl/reach/env.py` 强制 `success_distance_m == 0.005`，判定为严格小于）。**验收标准同步对齐官方**：

| 项目 | ufactory_genesis 官方 | 本次对齐后 |
| :--- | :--- | :--- |
| 成功判定 | `dist < 0.005`（严格小于） | 同 |
| 验收成功率 | **≥ 95%**（每组 512 目标） | 同 |
| 验收平均终止距离 | **≤ 10 mm** | 同 |
| 验收协议 | seed 17000 / 17001 各 512 目标，两组独立 | 同 |

影响与结论：

- **伺服下垂揭示**：5 mm 判据下开环 IK 教师停滞于 ~8–10 mm（位置伺服重力下垂约 10 mrad 稳态关节误差）；镜像过冲补偿（ILC 式，gain 2.0）后教师全机型 20/20 达标，证明 5 mm 物理可达（需闭环积分行为）。测试教师 `_greedy_action` 已改用该补偿。
- **全系重训**：官方式超参（PPO `[256,256,128]` MLP、`ent_coef=0.001`、2M 步；uf850 需 4M 步）替代原 1M 步 `[64,64]` 配方（lite6 曾在 72.5% 徘徊）。
- **官方协议验收通过**（两组独立 512 目标，`--seed 17000` / `--seed 17512`，门槛 ≥95% 且 ≤10 mm）：

| 机型 | 成功率（组1 / 组2） | 平均终止距离（组1 / 组2） | 判定 |
| :--- | ---: | ---: | :---: |
| lite6 | 99.8% / 99.4% | 4.3 / 4.4 mm | ✅ |
| xarm5 | 99.4% / 99.0% | 4.2 / 4.2 mm | ✅ |
| xarm6 | 100% / 99.8% | 4.1 / 4.1 mm | ✅ |
| xarm7 | 99.6% / 99.0% | 4.3 / 4.3 mm | ✅ |
| uf850 | 98.1% / 96.3% | 4.4 / 4.9 mm | ✅（4M 步重训后；2M 步时组2 曾 94.1% 未过线） |

- 示例验收门对齐官方双指标（`--min-success-rate` 默认 0.95、`--max-mean-dist-mm` 默认 10.0，双条件才 PASS）；训练脚本新增 `--net-arch/--ent-coef` 旋钮。

## 9. 风险与预案

| 风险 | 预案 |
|---|---|
| uf850 运动学差异大（立柱式、臂展/肩位不同），同一套奖励/超参可能欠训练 | 标定专属工作空间盒；仍不达标时仅对该机型微调 `delta`/步数（记录差异），不改奖励结构 |
| xarm7 冗余自由度 → IK 可达率/收敛变慢 | 可达性校验用环境现有 DLS IK + 更宽迭代预算；profile 盒收窄 |
| lite6 无名执行器将来被模型同步改名 | 顺序回退是兜底；名字命中时优先用名字，测试双向覆盖 |
| PPO 训练未达 85%（随机种子敏感） | 更换种子重训一轮（预算 ×2）；仍不达标则报告实测值并放宽验收阈值（留记录） |
| 5 机型 × 300k 训练耗时（约 70 min） | 串行后台跑，训练与文档收尾并行 |

## 10. 交付物清单（2026-10-14 全部完成）

- [x] `src/rl/reach_env.py`：`RobotProfile`/`ROBOT_PROFILES`/`ReachEnv`（+ 兼容别名）
- [x] `models/{lite6,xarm5,xarm7,uf850}/scene_reach.xml`（4 新增，xarm6 模板派生）与 `scripts/measure_reach_workspace.py` 标定脚本
- [x] `scripts/train_reach_rl.py --model`、`examples/rl/01_reach.py --model`
- [x] `build/rl/<model>/` 训练检查点 ×5（xarm6 重训 1M / 4 新训）与指标记录（§8a）
- [x] 测试：环境契约参数化、profile 防腐化、场景资产、lite6 执行器回退
- [x] README/README_zh 命令与结果表、roadmap 0.0.7 勾选、worklog
