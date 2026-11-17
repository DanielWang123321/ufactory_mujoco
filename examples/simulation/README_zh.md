# 仿真示例 (`examples/simulation`)

[English](README.md) | [中文](README_zh.md)

基于 MuJoCo 的机器人运动控制与仿真示例集（`01`–`10`），涵盖模型交互预览、关节/笛卡尔控制、夹爪伺服、方块抓放、时间参数化轨迹规划、RRT-Connect 避障规划、装箱展示、运动学/刚体动力学工具链以及真机安全门控流程。

需要 Python **3.12–3.14** 与核心安装（详见[安装说明](../../README_zh.md#安装)）。示例 `05`、`08`、`10` 直接调用已安装命令 `ufactory-mujoco pick-place`、`dynamics` 与 `box-pack`，通过 wheel 安装后也可脱离源码目录运行。参数汇总见[第 3 节](#3-命令行参数汇总)，全部 44 套模型名与启动命令见[第 4 节](#4-按机型的命令)，视窗快捷键见[第 5 节](#5-视窗操作)。

## 1. 示例概览

| 脚本 | 核心模块 | 支持模型 | 功能说明 |
| :--- | :--- | :--- | :--- |
| `01_load_and_view.py` | `ufactory_mujoco.viewer`, `ufactory_mujoco.model_paths` | 全部 44 套模型（机械臂、夹爪、相机、支架及组合） | 载入 MJCF 场景（支持 `home` / `ready` 关键帧）并启动交互视窗 |
| `02_joint_control.py` | `ufactory_mujoco.sim_robot` | 全部 30 套含机械臂及组合模型 | 基于速度的五次多项式关节插值控制（裸臂 `xarm6` 默认运行 20 姿态动力学轨迹） |
| `03_cartesian_ik.py` | `ufactory_mujoco.sim_robot`, `ufactory_mujoco.ik` | `xarm6` | 基于阻尼最小二乘法（DLS）IK 的笛卡尔直线往返与 X-Z 平面圆弧轨迹跟踪 |
| `04_gripper_control.py` | `ufactory_mujoco.sim_robot`, `ufactory_mujoco.g2_controller`, `ufactory_mujoco.bio_g2_controller` | 19 套可动夹爪模型（独立夹爪及组合） | 执行 `闭合 (1.0) -> 全开 (0.0) -> 半开 (0.5) -> 全开 (0.0)` 循环；G2 与 Bio G2 支持 `--speed-percent` 与 `--force-percent` |
| `05_pick_place.py` | `ufactory_mujoco.tasks.pick_place`, `ufactory_mujoco.sim_robot`, `ufactory_mujoco.preflight` | 5 组机械臂+夹爪任务组合 | 方块抓放任务：支持双侧指垫接触判据与轨迹预检（放置误差 `< 50 mm`、回零漂移 `< 10 mm`） |
| `06_trajectory_planning.py` | `ufactory_mujoco.trajectory`, `ufactory_mujoco.dynamics` | `xarm6` | 点到点五次多项式、多路点 C2 样条平滑、笛卡尔 SLERP 直线规划与动力学前馈力矩 |
| `07_kinematics.py` | `ufactory_mujoco.kinematics` | `xarm6` | 任意 site 正运动学、解析雅可比与有限差分校验、速度级 DLS 逆解、可操作度/奇异性指标 |
| `08_dynamics.py` | `ufactory_mujoco.tasks.dynamics`, `ufactory_mujoco.dynamics`, `ufactory_mujoco.collision` | 五款裸臂 | 动力学、固定碰撞姿态、命令预检与逐步运动监测；xArm6 3 kg 负载回归 |
| `09_obstacle_avoidance.py` | `ufactory_mujoco.collision`, `ufactory_mujoco.path_planning` | `xarm6` (`scene_obstacles.xml`) | 立柱+横板场景下的 RRT-Connect 避障规划、路径剪枝（shortcut）与时间参数化执行 |
| `10_box_pack.py` | `ufactory_mujoco.tasks.box_pack`, `ufactory_mujoco.sim_robot`, `ufactory_mujoco.preflight` | 5 组机械臂+夹爪任务组合 | 装箱展示：方块在盒沿上方 50 mm 释放自由落体，最终碰撞边界须在盒内（1 mm 容差）；支持多周期与速度倍率 |

## 2. 运行指南

所有脚本只需核心安装。`--headless` 让 `02`–`10` 不开视窗运行并输出验收指标；`01 --headless` 只加载模型并打印摘要。

### 2.1 模型预览与基础运动控制 (`01`–`04`)

```bash
# 01: 交互式查看器（全部 44 套模型）
python examples/simulation/01_load_and_view.py
python examples/simulation/01_load_and_view.py --model lite6 --pose ready
python examples/simulation/01_load_and_view.py --model xarm6_with_gripper --pose ready
python examples/simulation/01_load_and_view.py --model camera_stand_with_d405
python examples/simulation/01_load_and_view.py --model xarm6_with_d435 --pose ready
python examples/simulation/01_load_and_view.py --model uf850_with_gripper_and_d435 --pose ready

# 02: 关节空间五次多项式轨迹控制（全部 30 套含机械臂模型）
python examples/simulation/02_joint_control.py --model lite6
python examples/simulation/02_joint_control.py --model xarm6 --speed 120
python examples/simulation/02_joint_control.py --model xarm6 --motion demo --duration 2.5
python examples/simulation/02_joint_control.py --model xarm7_with_d435 --headless

# 03: 笛卡尔空间 IK 轨迹跟踪（直线 + 圆弧）
python examples/simulation/03_cartesian_ik.py
python examples/simulation/03_cartesian_ik.py --headless

# 04: 夹爪开闭控制（Gripper G2、Bio Gripper G2、Lite 6 Gripper）
python examples/simulation/04_gripper_control.py --model xarm6_with_gripper
python examples/simulation/04_gripper_control.py --model xarm6_with_gripper --speed-percent 80 --force-percent 75
python examples/simulation/04_gripper_control.py --model lite6_with_gripper
python examples/simulation/04_gripper_control.py --model bio_gripper_g2 --cycles 5
```

> **`02_joint_control.py` 说明**：
> - 裸臂 `xarm6` 默认运行 `ufactory_genesis` 动力学验证的 20 个姿态（两组往返），峰值目标速度为 `90°/s`；其他型号运行 `ready -> demo -> ready` 预设（未在 `PRESETS` 中单独定义的组合模型会从自身的 `ready` 关键帧出发并复用对应机械臂的 `demo` 姿态）。
> - 默认关闭阴影与反射（`--render-quality fast`）以降低高精度网格的 GPU 绘制开销，可通过 `--render-quality full` 恢复。
> - `--sync-hz 30` 控制查看器状态同步频率，物理步长保持 `2 ms`。

### 2.2 抓放操作与轨迹/避障规划 (`05`、`06`、`09`、`10`)

```bash
# 05: 方块抓放任务（与 `ufactory-mujoco pick-place` 等价）
python examples/simulation/05_pick_place.py --headless
python examples/simulation/05_pick_place.py --robot lite6 --executor servo_j --headless
python examples/simulation/05_pick_place.py --robot xarm7 --dry-run
ufactory-mujoco pick-place --robot xarm6 --headless --report build/reports/pick_place.json

# 06: 时间参数化轨迹规划（点到点五次、多路点 C2 样条、笛卡尔直线）
python examples/simulation/06_trajectory_planning.py
python examples/simulation/06_trajectory_planning.py --headless --speed 120

# 09: 避障轨迹规划（RRT-Connect 绕越立柱与横板）
python examples/simulation/09_obstacle_avoidance.py
python examples/simulation/09_obstacle_avoidance.py --headless
python examples/simulation/09_obstacle_avoidance.py --seed 3 --safety-margin 0.01

# 10: 装箱展示（与 `ufactory-mujoco box-pack` 等价）
python examples/simulation/10_box_pack.py --headless
python examples/simulation/10_box_pack.py --robot lite6 --cycles 3 --speed 2.0 --headless
python examples/simulation/10_box_pack.py --robot xarm7 --dry-run
ufactory-mujoco box-pack --robot uf850 --cycles 3 --headless --report build/reports/box_pack.json
```

### 2.3 运动学与刚体动力学工具链 (`07`、`08`)

```bash
# 07: 运动学工具链校验（FK、雅可比对照、速度级逆解、奇异性指标）
python examples/simulation/07_kinematics.py
python examples/simulation/07_kinematics.py --headless

# 08: 动力学与运动验收（默认 xarm6；无头模式同样执行运动）
python examples/simulation/08_dynamics.py
python examples/simulation/08_dynamics.py --headless
python examples/simulation/08_dynamics.py --robot lite6 --headless --samples 32 --seed 0
python examples/simulation/08_dynamics.py --robot all --headless --report build/dynamics/report.json

# Schema v4 动力学验证（对齐 ufactory_genesis，20 姿态 + 随机姿态离线检查）
dynamics-sim-check --robot xarm6 --random-count 20 --require-reference
dynamics-hardware-check --robot lite6 --dry-run --require-reference --strict-static --strict-static-layers l2b
```

`08_dynamics.py`（`ufactory-mujoco dynamics`）在不连接 xArm SDK 的情况下校验全部五款裸臂：

- **静态与随机采样检查**：在无附加负载的普通裸臂场景下检查 `home`、`ready`、`target` 与限位内 32 个随机姿态（默认 `--seed 0`），验证质量矩阵有限且对称正定、重力有限、静止科氏项为零、正逆动力学互逆、前馈力矩不超过执行器上限，并校验固定安全/碰撞姿态夹具（随机姿态的碰撞单独分类统计）。
- **8.4 s 监测运动**：依次执行 `home -> ready (1.5 s) -> target (3.0 s) -> ready (3.0 s)`（每段后收敛 `0.3 s`）。运动前按物理步长预检全部命令，积分后每个物理步由独立 `CollisionChecker` 检查实际关节位置（穿透 `<= 1e-6 m`、超限 `<= 1e-6 rad`、收敛到位误差 `< 0.01 rad`；非有限状态、MuJoCo 警告、提前结束或完成前关闭视窗均判失败）。`--robot all` 保留各臂结果并在单臂失败后继续其余机型。
- **报告与基线**：`--report` 输出严格 JSON，包含版本、环境、路径、阈值、指标与失败原因（退出码：`0` 通过，`1` 失败/未完成，`2` 参数非法）。`xarm6` 另保留 `3 kg` 负载前馈演示与六轴重力基线回归（`<= 1e-6 Nm`；J2/J3 沿用历史 5% 参考容差，不重新连接真机）。固定姿态、阈值与基线定义在 [`ufactory_mujoco/tasks/dynamics.py`](../../ufactory_mujoco/tasks/dynamics.py)。

如需执行对齐 `ufactory_genesis` 的 20 姿态静态力矩验证（PD 持位力矩 vs 解析参考/真机 SDK、L1 主判定与 L2a/L2b/L3a/L3b 分层诊断、Schema v4 CSV/JSONL 报告及 `_torque.png` 绘图），请使用 `dynamics-sim-check`、`dynamics-hardware-check --dry-run` 与 `dynamics-report-compare`（或 `python -m ufactory_mujoco.dynamics <子命令>`）。

#### 动力学模块编程接口 (`ufactory_mujoco/dynamics`)

```python
from ufactory_mujoco.dynamics import ArmDynamics

# payload=(质量 kg, attachment_site 坐标系下的质心 [x, y, z] m)
dyn = ArmDynamics("models/xarm6/xarm6.xml", payload=(3.0, [0, 0, 0.021]))
M = dyn.mass_matrix(q)                     # 稠密质量矩阵 M(q)
g = dyn.gravity(q)                         # 重力项 g(q)
c = dyn.coriolis(q, qd)                    # 科氏/离心项 C(q,qd)qd
tau = dyn.inverse_dynamics(q, qd, qdd)     # 逆动力学执行器力矩（与 forward_dynamics 严格互逆）
qdd = dyn.forward_dynamics(q, qd, tau)
tau_ff = dyn.feedforward_traj(q_traj, dt)  # 沿关节轨迹的前馈力矩序列
```

### 2.4 真机与控制器仿真 (`xarm-python-sdk`)

真机及内部仿真模式需要安装 `xarm-python-sdk`（`pip install xarm-python-sdk`），纯仿真无需安装。请将命令中的 IP 替换为自己的控制器地址（或设置环境变量 `XARM_IP`）。所有真机相关模式都会拒绝序列号或 SHA-256 与当前机械臂不符的标定文件。

- **零物理运动核验**：`kinematics_validation`、`08 --mode real-read-only` 以及 `05`/`08`/`10` 的 `--mode sdk-sim` 不会让物理机械臂运动（`sdk-sim` 在控制器内置仿真中重放轨迹，并在退出时恢复非仿真模式）。
- **真机运动门禁**：`--mode real`（`05`、`08`、`10`）要求控制器状态正常、轨迹预检通过且显式传入 `--confirm-real`，并强制限制单周期运行（`cycles == 1`）。`08_dynamics.py` 在非 `sim` 模式下必须指定单台机型（不支持 `--robot all`）。所有动力学验证真机命令（`dynamics-hardware-check`、`dynamics-sim-collision-check`、`08 --mode sdk-sim`/`real`）对关节速度强制实施 `<= 40 °/s`（`0.6981 rad/s`）安全上限。

```bash
# 1. 提取单机运动学标定 YAML（输出至 calibrations/<arm>_kinematics_<SN>.yaml）
python -m ufactory_mujoco.dynamics export-calibration 192.168.1.60 --output-dir calibrations

# 2. 只读 FK/IK 对照与零物理运动静态动力学检查
python -m ufactory_mujoco.kinematics_validation --robot xarm6 --ip 192.168.1.60 --calibration calibrations/xarm6_kinematics_<SN>.yaml
python examples/simulation/08_dynamics.py --robot xarm6 --mode real-read-only --ip 192.168.1.60 --calibration calibrations/xarm6_kinematics_<SN>.yaml
dynamics-hardware-check --robot lite6 --ip 192.168.1.60 --confirm-real --hold-current-only --require-reference

# 3. 控制器内置仿真重放（不驱动物理关节）
python examples/simulation/05_pick_place.py --robot xarm6 --mode sdk-sim --ip 192.168.1.60 --calibration calibrations/xarm6_kinematics_<SN>.yaml
python examples/simulation/10_box_pack.py --robot xarm6 --mode sdk-sim --ip 192.168.1.60 --calibration calibrations/xarm6_kinematics_<SN>.yaml
dynamics-sim-collision-check --robot lite6 --ip 192.168.1.60

# 4. 真机物理运动（必须加 --confirm-real；请确保工作区安全且急停可用）
ufactory-mujoco pick-place --robot xarm6 --mode real --confirm-real --ip 192.168.1.60 --calibration calibrations/xarm6_kinematics_<SN>.yaml
dynamics-hardware-check --robot xarm6 --ip 192.168.1.60 --confirm-real --require-reference
dynamics-report-compare reports/dyn_ver_<SN>/<old>.jsonl reports/dyn_ver_<SN>/<new>.jsonl
```

## 3. 命令行参数汇总

### 3.1 任务与真机参数 (`05`、`08`、`10` / `ufactory-mujoco`)

| 参数 | 适用脚本 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--robot <arm>` | `05`, `08`, `10` | `xarm6` | 机械臂键名（`lite6`、`xarm5`、`xarm6`、`xarm7`、`uf850`）；`08`（`--mode sim`）另支持 `all` |
| `--gripper <name>` | `05`, `10` | `None`（取自 YAML） | 覆盖任务配置中的夹爪型号 |
| `--executor <mode>` | `05`, `10` | `servo_cartesian` | 轨迹执行器模式：`servo_cartesian` 或 `servo_j` |
| `--config <path>` | `05`, `10` | `configs/<task>/<robot>.yaml` | 自定义任务 overlay YAML 路径 |
| `--mode <mode>` | `05`, `08`, `10` | `sim` | 运行模式：`sim`、`dry-run`（`05`/`10`）、`real-read-only`（`08`）、`sdk-sim` 或 `real` |
| `--dry-run` | `05`, `10` | `False` | 仅执行关节限位与碰撞预检并输出阶段状态，不步进物理仿真 |
| `--cycles <N>` | `05`, `10` | `1` | 动作执行循环次数（`10` 为 `1..3`；`sdk-sim`/`real` 强制为 `1`） |
| `--speed <x>` | `10` | YAML 中的 `speed_scale`（`1.0`） | 运动速度倍率（`>1` 加快；自由落体物理不缩放） |
| `--samples <N>` | `08` | `32` | 每臂随机动力学姿态数，须为正整数 |
| `--seed <N>` | `08` | `0` | 随机动力学姿态采样种子（整数） |
| `--report <path>` | `05`, `08`, `10` | `None` | 将预检、动力学验收或周期执行报告导出为 JSON 文件 |
| `--ip <addr>` | `05`, `08`, `10` | `XARM_IP` | `sdk-sim`、`real`、`real-read-only` 使用的控制器 IPv4；未设置 `XARM_IP` 时必填 |
| `--calibration <path>` | `05`, `08`, `10` | `None` | 真机/内部仿真模式必需的单机运动学标定 YAML 路径 |
| `--confirm-real` | `05`, `08`, `10` | `False` | `--mode real` 真机运动必需的显式安全确认标志 |
| `--headless` | `05`, `08`, `10` | `False` | 不开视窗运行并输出验收指标 |

### 3.2 示例专属参数 (`01`–`04`、`06`、`07`、`09`)

示例 `03` 与 `07` 仅接受 `--headless` 参数。

| 参数 | 适用脚本 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--headless` | `01`–`04`, `06`, `07`, `09` | `False` | 不开视窗运行并输出指标（`01` 只加载模型并打印摘要） |
| `--model <name>` | `01`, `02`, `04` | `xarm6`（`04` 为 `xarm6_with_gripper`） | 模型目录名或自定义 MJCF XML 路径 |
| `--pose <home\|ready>` | `01` | `home` | 初始加载关键帧姿态 |
| `--motion <mode>` | `02` | `auto` | 动作模式：`auto`、`dynamics`（20 姿态，仅限裸臂 `xarm6`）或 `demo` |
| `--speed <deg/s>` | `02`, `06` | `90.0` | 关节峰值目标速度（度/秒） |
| `--duration <s>` | `02` | `None` | 固定单段运动时长（秒，优先级高于 `--speed`） |
| `--sync-hz <Hz>` | `02` | `30.0` | 查看器状态同步频率（物理步长保持 `2 ms`） |
| `--render-quality` | `02` | `fast` | 渲染质量：`fast`（关闭阴影/反射）或 `full` |
| `--cycles <N>` | `04` | `3` | 夹爪开闭演示循环次数 |
| `--speed-percent <pct>` | `04` | `100.0` | Gripper G2 / Bio Gripper G2 速度百分比（`0..100`；Lite 6 夹爪忽略） |
| `--force-percent <pct>` | `04` | `100.0` | Gripper G2 / Bio Gripper G2 力控百分比（`1..100`；Lite 6 夹爪忽略） |
| `--seed <N>` | `09` | `0` | RRT-Connect 与路径剪枝随机种子 |
| `--safety-margin <m>` | `09` | `0.002` | 规划最小安全间隙阈值（米） |
| `--step <rad>` | `09` | `0.15` | RRT-Connect 关节空间扩展步长（rad） |
| `--max-iter <N>` | `09` | `2000` | RRT-Connect 最大采样迭代次数 |

## 4. 按机型的命令

`01`、`02`、`04` 用 `--model` 指定模型名；`05`、`08`、`10` 用 `--robot` 指定机械臂键名。所有命令默认打开视窗，加 `--headless` 可无头运行。

### 4.1 机械臂与组合

| 机械臂 | 裸臂（`--model`） | 组合模型（`--model`） | 机械臂键名（`--robot`） |
| :--- | :--- | :--- | :--- |
| Lite 6 | `lite6` | `lite6_with_gripper`、`lite6_with_vacuum_gripper` | `lite6` |
| xArm 5 | `xarm5` | `xarm5_with_gripper`、`xarm5_with_bio_gripper`、`xarm5_with_vacuum_gripper`、`xarm5_with_d435`、`xarm5_with_gripper_and_d435` | `xarm5` |
| xArm 6 | `xarm6` | `xarm6_with_gripper`、`xarm6_with_bio_gripper`、`xarm6_with_vacuum_gripper`、`xarm6_with_d435`、`xarm6_with_gripper_and_<相机>` | `xarm6` |
| xArm 7 | `xarm7` | `xarm7_with_gripper`、`xarm7_with_bio_gripper`、`xarm7_with_vacuum_gripper`、`xarm7_with_d435`、`xarm7_with_gripper_and_d435` | `xarm7` |
| UF850 | `uf850` | `uf850_with_gripper`、`uf850_with_bio_gripper`、`uf850_with_vacuum_gripper`、`uf850_with_d435`、`uf850_with_gripper_and_d435` | `uf850` |

`<机械臂>_with_gripper` 安装的是 Gripper G2（Lite 6 为 Lite 6 夹爪）。`<相机>` 为 `d405`、`d435`、`gemini_305` 或 `zed_x_nano`。

各脚本的模型适用范围（共 `44` 套模型目录）：

- `01_load_and_view.py`：支持全部 `44` 套模型（裸臂、组合、独立末端/相机/支架）。
- `02_joint_control.py`：支持全部 `30` 套含机械臂的模型（`5` 款裸臂 + `25` 套机械臂组合）。
- `04_gripper_control.py`：支持全部 `19` 套带可动夹爪执行器的模型（`3` 款独立夹爪 + `16` 套带夹爪的机械臂/相机组合）。
- `05_pick_place.py`、`10_box_pack.py`：通过 `--robot` 选择 `5` 套任务组合（`lite6` 配 `lite6_gripper`，其余四臂配 `gripper_g2`）。
- `08_dynamics.py`：通过 `--robot` 选择 `5` 款裸臂（`sim` 模式另支持 `all`）；示例 `03`、`06`、`07`、`09` 只支持裸臂 `xarm6`。

```bash
# 任意机械臂或组合：静态查看与关节运动
python examples/simulation/01_load_and_view.py --model xarm7_with_bio_gripper --pose ready
python examples/simulation/02_joint_control.py --model xarm5_with_vacuum_gripper

# 带夹爪的组合（Gripper G2、Bio Gripper G2、Lite 6 夹爪）：开合
python examples/simulation/04_gripper_control.py --model uf850_with_bio_gripper
python examples/simulation/04_gripper_control.py --model xarm6_with_gripper_and_d405

# 任务：--robot 选择机械臂及其任务夹爪
python examples/simulation/05_pick_place.py --robot xarm5
python examples/simulation/10_box_pack.py --robot uf850 --cycles 3
python examples/simulation/08_dynamics.py --robot all --headless
```

### 4.2 独立末端工具、相机与支架

| 模型（`--model`） | `01_load_and_view.py` | `04_gripper_control.py` |
| :--- | :---: | :--- |
| `gripper_g2` | 支持 | 开合，支持 `--speed-percent`、`--force-percent` |
| `bio_gripper_g2` | 支持 | 开合，支持 `--speed-percent`、`--force-percent` |
| `lite6_gripper` | 支持 | 开合 |
| `lite6_vacuum_gripper`、`vacuum_gripper` | 支持 | 拒绝（没有夹爪执行器，退出码 `2`） |
| `d405`、`d435`、`gemini_305`、`zed_x_nano` | 支持 | 拒绝（退出码 `2`） |
| `camera_stand`、`camera_stand_with_<相机>` | 支持 | 拒绝（退出码 `2`） |

相机与支架模型 `mass=0`，不含光学成像传感器。在查看器右侧面板开启 Site Group `4` 可显示安装面与螺钉孔位。

### 4.3 关键帧

- **裸臂 xArm 5/6/7 与 UF850**：`home` 与 `ready` 均为全零关节角。
- **Lite 6（裸臂及组合）**：`home` 为全零；`ready` 为 Studio 出厂位姿 `[0, 9.9, 31.8, 0, 21.9, 0]°`（法兰约在 `[200, 0, 200]` mm，姿态 `[180, 0, 0]°`）。
- **xArm 5/6/7 与 UF850 组合**：`ready` 为安装 Gripper G2 时 TCP 位于 `(300, 0, 300)` mm、姿态 `(180, 0, 0)°` 的无滚转朝下 IK 位姿（`xArm 6` 的 `home` 与之相同；`xArm 5`、`xArm 7` 与 `UF850` 组合的 `home` 各自定义在 MJCF 中——注意 `uf850_with_*` 的 `home` 为全零竖直姿态，预览组合模型时建议使用 `--pose ready`）。
- **工具偏移与 D435 视野朝向**：组合模型的 TCP 包含夹爪或吸盘的工具偏移；`<机械臂>_with_d435` 将 `attachment_site` 沿法兰 `+Z` 偏移 `+3 mm`（相机支架板厚）。安装在机械臂上的 `d435` 组合采用与人眼一致的正向视野（工具朝下时画面上方朝基座 `+X`，USB-C 接口位于法兰 `+Y` / 基座 `Y-` 侧）。
- **独立模型**：独立可动夹爪（`gripper_g2`、`bio_gripper_g2`、`lite6_gripper`）定义 `open` 与 `close` 关键帧；真空吸盘、独立相机及支架无关键帧（`nkey == 0`），以默认姿态加载。

## 5. 视窗操作

以下按 MuJoCo 查看器的内置帮助（`F1`）整理：

- **左键拖拽**：旋转视角。**右键拖拽**：平移视角。**滚轮或中键拖拽**：缩放。
- **双击**：选中物体。**右键双击**：以点击处为视角中心。**Ctrl + 右键双击**：跟踪选中物体。
- **Ctrl + 左键拖拽 / Ctrl + 右键拖拽**：对选中物体施加扭矩 / 力。
- **空格**：暂停或继续。**Backspace**：重置。两者只在 `01_load_and_view.py` 中控制物理，其余示例由脚本自己步进物理。
- **C**：显示接触点。**F**：显示接触力。**0–5**：切换几何组（`2` 为视觉网格，`3` 为碰撞网格）。
- **Ctrl + A**：对齐视角。**Tab / Shift + Tab**：显示或隐藏左 / 右面板。**F1**：帮助。
