# 0.0.6 开发计划：示例整理（仿真 / 强化学习）与带避障轨迹规划

版本：v1.0（2026-10-14）｜对应 [roadmap.md](../roadmap.md) 0.0.6

> 进度：M0（`40a234d` 落库）、M1（`286e80f`）、M2（2026-10-14 完成）；M3 待开始。

## 1. 目标与范围

**目标**：示例按「仿真 / 强化学习」两类重组；仿真类补齐运动学、动力学、**带避障轨迹规划**三个主题的独立示例与支撑模块。

- **做**：示例目录重组（含引用迁移）、碰撞检测接口、RRT 避障路径规划模块与场景、运动学/动力学示例、文档与测试同步。
- **不做**：动态障碍物/时变避障、最优性证明（RRT*）、真机执行避障、RL 类示例扩展（属 0.0.7）。

## 2. 现状与约束

| 项 | 现状 | 对本计划的影响 |
|---|---|---|
| 示例组织 | 7 个脚本平铺 `examples/`（01 查看 → 07 RL 到达） | 需迁移目录 + 全量引用更新 |
| 代码引用 | 测试经 `importlib` 引用 01/02/04/05；02/04 引用 01 的 `resolve_model_path` | 迁移时统一改引用；`resolve_model_path` 提取到 `src/` 消除示例互引 |
| 文档引用 | `docs/visual_command_guide.md` 32 处 + README/README_zh 各 20 处命令 | 批量更新 + README 迁移对照表 |
| 运动学模块 | `src/kinematics.py`（并行任务，**未提交**）：FK/雅可比/速度级逆解/可操作度/条件数 | **M0 前置**：须先落库，运动学示例与避障规划依赖它 |
| 可复用底座 | `trajectory.py`（PTP 五次/LSPB/样条/笛卡尔直线）、`dynamics.py`、`SimRobot.execute_trajectory` | 避障规划产出的航点直接接入时间参数化与执行 |

## 3. 目标结构

```
examples/
  simulation/                 # 仿真类
    01_load_and_view.py       # 模型查看/检查（迁移，内容不变）
    02_joint_control.py       # 关节控制（迁移）
    03_cartesian_ik.py        # 笛卡尔 IK 直线/圆弧跟踪（迁移）
    04_gripper_control.py     # 夹爪控制（迁移）
    05_pick_place.py          # 方块抓放（迁移）
    06_trajectory_planning.py # 轨迹规划与时间参数化（迁移）
    07_kinematics.py          # 新增：FK/雅可比/速度级逆解/可操作度与奇异性
    08_dynamics.py            # 新增：质量矩阵、重力/科氏、正逆动力学、负载、前馈
    09_obstacle_avoidance.py  # 新增：RRT 避障规划 + 执行
  rl/                         # 强化学习类
    01_reach.py               # 07_rl_reach.py 迁移改名（为 0.0.7 多机型留编号位）
src/
  model_paths.py              # 新增：MODELS_ROOT + resolve_model_path（自 01 提取）
  collision.py                # 新增：CollisionChecker（构型碰撞/间隙查询）
  path_planning.py            # 新增：RRT-Connect + shortcut + 时间参数化衔接
models/xarm6/
  scene_obstacles.xml         # 新增：xArm 6 + 立柱/横板障碍任务场景
```

命名规则：类别内独立编号；0.0.7 的多机型 RL 示例顺延 `rl/02_*.py`。旧平铺路径 0.x 阶段**直接切换不留 shim**，README 提供迁移对照表。

## 4. 新功能设计

### 4.1 碰撞检测 `src/collision.py`

```python
class CollisionChecker:
    def __init__(self, model, *, margin: float = 0.0, ignore_contacts=None)  # 独立 MjData
    def in_collision(self, q: np.ndarray) -> bool        # q 为机械臂关节（6,）
    def min_distance(self, q: np.ndarray) -> float       # 最小间隙（m，负值为穿透）
```

- 独立 `MjData`，查询不污染 `SimRobot` 状态；`q` 沿用 `SimRobot` 机械臂关节约定（前 `n_arm_qpos`）。
- 判定：`mj_forward` 后收集接触对，`dist < -margin` 视为碰撞；相邻连杆接触依赖各模型 XML 既有 `<contact><exclude>`。
- 与规划器共用同一 checker，保证采样与执行判定口径一致。

### 4.2 带避障路径规划 `src/path_planning.py`

```python
def plan_collision_free(model, q_start, q_goal, *, checker, limits=None,
                        max_iter=2000, step=0.1, seed=None) -> np.ndarray  # (K, 6) 航点
def shortcut_path(checker, waypoints, *, iterations=200, seed=None) -> np.ndarray
```

- **直线快速通道**：起终点间插值（采样间距 ≤ `step`）无碰撞时直接返回两点，跳过搜索。
- **RRT-Connect**：双向树、`goal_bias=0.1`、步长 `step`、关节限位钳制；新边一律按 ≤`step` 采样做碰撞检查；达迭代上限抛 `PlanningError`（明确失败，不静默）。
- **shortcut**：随机采两航点，段内无碰撞则删除中间航点；长度单调不增。
- **时间参数化衔接**：航点交 `trajectory.plan_joint_waypoints`（C2 样条，速度/加速度限幅）得 `JointTrajectory`，由 `SimRobot.execute_trajectory` 执行——规划与执行全链路复用既有模块。

### 4.3 场景与示例

- `models/xarm6/scene_obstacles.xml`：xArm 6 + 立柱 + 横板（视觉/碰撞一致），`start`/`goal` 关键帧定义起终位姿，直线必碰撞、绕行可行；含 front/side 相机。
- `09_obstacle_avoidance.py`：直线（预期碰撞）vs RRT 绕行对比；GUI 双路径可视化（直线红/规划绿），`--headless` 输出航点数、路径长度、最小间隙、执行到位误差；成功判据：全程无接触 + 到位误差 < 0.01 rad，失败退出码非零。
- `07_kinematics.py`：FK 与 `mj_forward` 对照、雅可比与中央差分对照、速度级逆解跟踪小 twist、可操作度/条件数扫描（含全零位形真奇异点演示）。
- `08_dynamics.py`：质量矩阵对称正定、重力/科氏分解、正逆动力学往返、3 kg 负载注入对比、轨迹前馈力矩包络。

## 5. 迁移清单（无行为变化）

| 对象 | 动作 |
|---|---|
| 示例脚本 | 移动至 `examples/simulation|rl/`；`sys.path.insert` 深度 +1 层 |
| `resolve_model_path` | 提取至 `src/model_paths.py`；02/04 与 `tests/test_all_models.py` 改引 `src` |
| `tests/` 引用 | `importlib` 路径改 `examples.simulation.0X_*`、`examples.rl.01_reach` |
| README/README_zh | 示例命令、Project Structure、迁移对照表 |
| `docs/visual_command_guide.md` | 32 处命令前缀批量更新为 `examples/simulation/` |

## 6. 里程碑

| 里程碑 | 内容 | 验收标准 | 预估 |
|---|---|---|---|
| **M0 前置** | kinematics 流水线（`src/kinematics.py` 等）落库提交 | 全量测试通过；本计划依赖的 API 稳定 | —（依赖并行任务） |
| **M1 目录重组** | §5 全部迁移项 | 双环境全量测试通过；全部示例命令无头冒烟通过；文档命令 grep 核对无遗留旧路径 | 0.5–1 d |
| **M2 规划模块** | `src/collision.py` + `src/path_planning.py` + 单测 | 碰撞判定与已知构型一致；种子可复现；直线快速通道生效；路径全段无碰撞；shortcut 后仍有效且不增长度 | 1.5 d |
| **M3 避障场景/示例** | `scene_obstacles.xml` + `09_obstacle_avoidance.py` + 集成测试 | 直线必碰撞、规划路径最小间隙 > 0、SimRobot 执行全程无穿透接触、到位误差 < 0.01 rad | 1 d |
| **M4 主题示例与文档** | `07_kinematics`/`08_dynamics` + README/指南 + 发布 | 双环境全量回归通过；`git diff --check` 干净；roadmap/worklog 更新 | 1 d |

合计约 **4–4.5 人日**（不含 M0）。

## 7. 测试计划

新增 `tests/test_collision.py`、`tests/test_path_planning.py`（含示例集成）：

1. **碰撞判定**：已知碰撞/自由构型判定正确；`margin` 生效；查询后模型状态不变（独立 data）。
2. **RRT**：固定种子路径可复现；首尾为起终点；相邻段 ≤`step` 采样无碰撞；迭代上限抛 `PlanningError`。
3. **快速通道**：直连可行时返回两点路径；被障碍阻断时进入搜索。
4. **shortcut**：结果仍全段无碰撞、路径长度不增、航点数不增。
5. **时间参数化衔接**：轨迹 C2、速度/加速度限幅内、终点精确到点。
6. **集成**：`09 --headless` 全程无穿透接触（≤ 1 μm 判据沿用抓放测试）、到位误差 < 0.01 rad、退出码语义。
7. **迁移回归**：既有 120 项测试（重组后引用更新）双环境全绿；01–09/rl 全部示例无头冒烟。

## 8. 风险与对策

| 风险 | 对策 |
|---|---|
| 碰撞网格凸包近似 → 假阳/假阴 | 采样与执行同一 checker；`margin` 以已知构型测试标定；障碍场景留足间隙 |
| RRT 不收敛或耗时 | 直线快速通道、goal bias、迭代上限显式报错；场景障碍尺寸可控；测试限时 |
| 目录重组破坏外部脚本/文档 | 迁移对照表 + 全库 grep 核对旧路径；测试先行更新 |
| 与并行任务（kinematics）冲突 | M0 前置先提交；重组避开其改动文件（`sim_robot.py` 已含双方内容，不再改动） |
| 示例互引（02/04→01）随迁移失效 | `resolve_model_path` 提取至 `src/model_paths.py`，示例间零依赖 |

## 9. 交付物清单

- [ ] `examples/simulation/`（01–09）与 `examples/rl/01_reach.py`（迁移 + 3 个新增示例）
- [ ] `src/model_paths.py`、`src/collision.py`、`src/path_planning.py`
- [ ] `models/xarm6/scene_obstacles.xml`
- [ ] `tests/test_collision.py`、`tests/test_path_planning.py`
- [ ] README/README_zh（命令、结构、迁移表）、`docs/visual_command_guide.md` 命令更新
- [ ] roadmap 0.0.6 勾选、worklog 记录
