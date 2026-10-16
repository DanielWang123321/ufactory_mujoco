# 仿真示例 (`examples/simulation`)

[English](README.md) | [中文](README_zh.md)

基于 MuJoCo 的机器人运动控制与仿真示例集（`01`–`09`），涵盖模型交互预览、关节/笛卡尔空间控制、夹爪开闭、方块抓放、时间参数化轨迹规划、RRT-Connect 避障规划以及运动学/刚体动力学工具链演示。

> 📖 按产品型号（Lite 6、xArm 5/6/7、UF850）与组件状态（独立/组合、静态/可动）分类的完整指令速查矩阵，请查阅 **[docs/visual_command_guide.md](../../docs/visual_command_guide.md)**。

## 1. 示例概览

| 脚本 | 核心模块 | 支持模型 | 功能说明 |
| :--- | :--- | :--- | :--- |
| `01_load_and_view.py` | `src.viewer`, `src.model_paths` | 全系机械臂、夹爪及组合 | 载入 MJCF 场景（支持 `home` / `ready` 关键帧）并启动交互视窗 |
| `02_joint_control.py` | `src.sim_robot` | 全系机械臂及带夹爪组合 | 基于速度的五次多项式关节插值控制（裸臂 `xarm6` 默认运行 20 姿态动力学轨迹） |
| `03_cartesian_ik.py` | `src.sim_robot`, `src.ik` | `xarm6` | 基于阻尼最小二乘法（DLS）IK 的笛卡尔直线往返与 X-Z 平面圆弧轨迹跟踪 |
| `04_gripper_control.py` | `src.sim_robot` | 独立夹爪及机械臂+夹爪组合 | 夹爪位置开闭控制循环（`0.0` = 全开，`1.0` = 闭合） |
| `05_pick_place.py` | `src.sim_robot`, `src.ik` | `xarm6_with_gripper` | 方块抓放任务，复刻 `ufactory_genesis` 流程与判据（放置误差 `< 50 mm`、回零漂移 `< 10 mm`） |
| `06_trajectory_planning.py` | `src.trajectory`, `src.dynamics` | `xarm6` | 点到点五次多项式、多路点 C2 样条平滑、笛卡尔 SLERP 直线规划与动力学前馈力矩 |
| `07_kinematics.py` | `src.kinematics` | `xarm6` | 任意 site 正运动学、解析雅可比与有限差分校验、速度级 DLS 逆解、可操作度/奇异性指标 |
| `08_dynamics.py` | `src.dynamics` | `xarm6` | 质量矩阵 $M(q)$、重力/科氏力分解、正逆动力学互逆校验、3 kg 负载注入与前馈力矩包络 |
| `09_obstacle_avoidance.py` | `src.collision`, `src.path_planning` | `xarm6` (`scene_obstacles.xml`) | 立柱+横板场景下的 RRT-Connect 避障规划、路径剪枝（shortcut）与时间参数化执行 |

## 2. 运行指南

所有仿真示例均在默认的 `py314` 环境（仅需 `mujoco` + `numpy`）下运行。`02`–`09` 均支持添加 `--headless` 参数以关闭视窗并在终端输出量化验收指标。

### 2.1 模型预览与基础运动控制 (`01`–`04`)

```bash
# 01: 交互式查看器（支持所有机械臂、夹爪及组合模型）
python examples/simulation/01_load_and_view.py
python examples/simulation/01_load_and_view.py --model lite6 --pose ready
python examples/simulation/01_load_and_view.py --model xarm6_with_gripper --pose ready

# 02: 关节空间五次多项式轨迹控制
python examples/simulation/02_joint_control.py --model lite6
python examples/simulation/02_joint_control.py --model xarm6 --speed 120
python examples/simulation/02_joint_control.py --model xarm6 --motion demo --duration 2.5

# 03: 笛卡尔空间 IK 轨迹跟踪（直线 + 圆弧）
python examples/simulation/03_cartesian_ik.py
python examples/simulation/03_cartesian_ik.py --headless

# 04: 夹爪开闭控制（支持 Gripper G2、Bio Gripper G2、Lite 6 Gripper）
python examples/simulation/04_gripper_control.py --model xarm6_with_gripper
python examples/simulation/04_gripper_control.py --model lite6_with_gripper
python examples/simulation/04_gripper_control.py --model bio_gripper_g2 --cycles 5
```

> **`02_joint_control.py` 说明**：
> - 裸臂 `xarm6` 默认运行 `ufactory_genesis` 动力学验证的 20 个姿态（两组往返），峰值目标速度为 `90°/s`；其他型号运行 `ready -> demo -> ready` 预设。
> - 默认关闭阴影与反射（`--render-quality fast`）以降低高精度网格的 GPU 绘制开销，可通过 `--render-quality full` 恢复。
> - `--sync-hz 30` 控制查看器状态同步频率，物理步长保持 `2 ms`。

### 2.2 抓放操作与轨迹/避障规划 (`05`、`06`、`09`)

```bash
# 05: 方块抓放任务（xArm 6 + Gripper G2）
python examples/simulation/05_pick_place.py
python examples/simulation/05_pick_place.py --headless

# 06: 时间参数化轨迹规划（点到点五次、多路点 C2 样条、笛卡尔直线）
python examples/simulation/06_trajectory_planning.py
python examples/simulation/06_trajectory_planning.py --headless --speed 120

# 09: 避障轨迹规划（RRT-Connect 绕越立柱与横板）
python examples/simulation/09_obstacle_avoidance.py
python examples/simulation/09_obstacle_avoidance.py --headless
python examples/simulation/09_obstacle_avoidance.py --seed 3 --safety-margin 0.01
```

### 2.3 运动学与刚体动力学工具链 (`07`、`08`)

```bash
# 07: 运动学工具链校验（FK、雅可比对照、速度级逆解、奇异性指标）
python examples/simulation/07_kinematics.py
python examples/simulation/07_kinematics.py --headless

# 08: 刚体动力学工具链校验（质量矩阵、重力/科氏力、正逆动力学往返、3 kg 负载对标）
python examples/simulation/08_dynamics.py
python examples/simulation/08_dynamics.py --headless
```

#### 动力学模块编程接口 (`src/dynamics.py`)

```python
from src.dynamics import ArmDynamics

# payload=(质量 kg, attachment_site 坐标系下的质心 [x, y, z] m)
dyn = ArmDynamics("models/xarm6/xarm6.xml", payload=(3.0, [0, 0, 0.021]))
M = dyn.mass_matrix(q)                     # 稠密质量矩阵 M(q)
g = dyn.gravity(q)                         # 重力项 g(q)
c = dyn.coriolis(q, qd)                    # 科氏/离心项 C(q,qd)qd
tau = dyn.inverse_dynamics(q, qd, qdd)     # 逆动力学执行器力矩（与 forward_dynamics 严格互逆）
qdd = dyn.forward_dynamics(q, qd, tau)
tau_ff = dyn.feedforward_traj(q_traj, dt)  # 沿关节轨迹的前馈力矩序列
```

与真机 xArm 6（`xarm-python-sdk`）交叉验证脚本：

```bash
python scripts/compare_dynamics_xarm6.py                                      # 真机运动记录 + 逆动力学回放比对
python scripts/compare_dynamics_xarm6.py --replay build/dynamics/run_xxx.csv  # 从 .meta.json 恢复负载回放
python scripts/compare_torque_xarm6.py --no-move                              # 静态保持力矩比对
```

## 3. 命令行参数汇总

| 参数 | 适用脚本 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--model <name>` | `01`, `02`, `04`, `05` | `xarm6`（`05` 为 `xarm6_with_gripper`） | 模型名称或自定义 MJCF XML 路径 |
| `--pose <home\|ready>` | `01` | `home` | 初始加载关键帧姿态 |
| `--headless` | `02`–`09` | `False` | 无头模式（关闭视窗，终端输出精度与验收指标） |
| `--speed <deg/s>` | `02`, `06` | `90.0` | 关节峰值目标速度（度/秒） |
| `--motion <mode>` | `02` | `auto` | 动作模式：`auto`、`dynamics`（20 姿态）或 `demo` |
| `--duration <s>` | `02` | `None` | 固定单段运动时长（秒，优先级高于 `--speed`） |
| `--sync-hz <Hz>` | `02` | `30.0` | 查看器状态同步频率（Hz） |
| `--render-quality` | `02` | `fast` | 渲染质量：`fast`（关闭阴影/反射）或 `full` |
| `--cycles <N>` | `04`, `05` | `3` (`04`) / `1` (`05`) | 动作执行循环次数 |
| `--seed <N>` | `09` | `0` | RRT-Connect 与路径剪枝随机种子 |
| `--safety-margin <m>` | `09` | `0.002` | 规划安全间隙阈值（米） |
| `--step <rad>` | `09` | `0.15` | RRT-Connect 关节空间扩展步长（rad） |
| `--max-iter <N>` | `09` | `2000` | RRT-Connect 最大采样迭代次数 |

