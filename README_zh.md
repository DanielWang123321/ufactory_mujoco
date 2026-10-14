# ufactory_mujoco

[English](README.md) | [中文](README_zh.md)

基于 MuJoCo 的 UFACTORY 机器人仿真套件 —— 提供高精细视觉资产、关节控制、笛卡尔逆运动学（IK）及交互式可视化。

## 功能特性

- **高精细视觉质感**：具备平滑法线与材质分色（烤漆白外壳、阳极氧化铝转轴、黑色部件及密封环）的高精度 OBJ 视觉网格，配合物理专用的 STL 碰撞网格。
- **全系列资产支持**：包含从 [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis) 官方解构转换的全部机械臂（xArm 5、xArm 6、xArm 7、UF850、Lite6）及末端配件（Gripper G2、Bio Gripper G2、Lite6 Gripper、Lite6 Vacuum Gripper）。
- **运动控制与运动学**：支持五次多项式关节插值平滑规划、重力补偿前馈控制、以及阻尼最小二乘法（DLS）笛卡尔逆运动学解算。
- **刚体动力学模块**：质量矩阵、重力/科氏力项、逆/正动力学及轨迹前馈力矩计算（`src/dynamics.py`），支持 TCP 负载注入与真机交叉比对脚本。
- **强化学习示例**：xArm 6 TCP 位置到达 Gymnasium 环境（`src/rl/reach_env.py`）、PPO 训练管线（`scripts/train_reach_rl.py`）与训练策略回放示例。
- **稳定可靠的查看器**：集成实时时钟同步、异常安全退出保护，针对 Linux Wayland / X11 进行了深度适配。

## 快速上手

### 环境安装

```bash
conda create -n py314 python=3.14 -y
conda activate py314
pip install -r requirements.txt
```

### 运行示例

> 📖 **完整可视化指令指南**：按产品型号（Lite 6、xArm 5/6/7、UF850）与组件状态（静态/可动）分类的启动控制指令矩阵，请查阅 **[docs/visual_command_guide.md](docs/visual_command_guide.md)**。

```bash
# 交互式查看器：载入模型并预览（支持所有机械臂、配件及组合）
python examples/01_load_and_view.py
python examples/01_load_and_view.py --model lite6 --pose ready
python examples/01_load_and_view.py --model xarm6_with_gripper --pose ready

# 关节空间轨迹控制（支持通过 --model 指定任意机械臂）
python examples/02_joint_control.py --model lite6
python examples/02_joint_control.py --model xarm6
python examples/02_joint_control.py --model xarm6 --speed 120

# 笛卡尔空间 IK 跟踪控制：直线 + 圆弧轨迹
python examples/03_cartesian_ik.py

# 夹爪运动控制（支持 Gripper G2, Bio Gripper G2, Lite 6 Gripper 独立及组合）
python examples/04_gripper_control.py --model lite6_with_gripper
python examples/04_gripper_control.py --model xarm6_with_gripper
python examples/04_gripper_control.py --model bio_gripper_g2

# 方块抓放（xArm 6 + Gripper G2，复刻 ufactory_genesis pick_place）
python examples/05_pick_place.py
python examples/05_pick_place.py --headless     # 输出 place_error_mm / home_drift_mm，失败时退出码非零

# 轨迹规划：点到点五次、多路点样条平滑、笛卡尔直线与动力学前馈
python examples/06_trajectory_planning.py
python examples/06_trajectory_planning.py --headless

# 强化学习 TCP 到达：加载训练好的 PPO 策略回放（训练用 scripts/train_reach_rl.py）
python examples/07_rl_reach.py
python examples/07_rl_reach.py --headless     # 输出 success_rate / TCP 误差，失败时退出码非零
python examples/07_rl_reach.py --random       # 未训练基线（无需 stable-baselines3）
```

关节示例中，裸臂 `xarm6` 默认运行 `ufactory_genesis` 动力学验证的 20 个姿态（两组往返），峰值目标速度为 90°/s；其他型号保留原动作预设。`--speed` 调节速度，`--motion demo --duration 2.5` 复现原示例。复用姿态不代表已经完成 MuJoCo 力矩验证。

示例默认关闭阴影与反射，以降低高精度网格的 GPU 绘制开销；`--render-quality full` 恢复效果。`--sync-hz 30` 控制查看器状态同步频率，绘制线程仍独立运行；物理步长保持 2 ms。`--headless` 完全关闭查看器，MuJoCo 物理在 CPU 上运行。

### 运行测试套件

```bash
# 在 py314 环境下运行单元测试与全量资产 MuJoCo 加载校验
python -m unittest discover -s tests -v

# 在 py313 环境下运行 GLB 转换与面绕序自愈回归测试（需 genesis / trimesh）
/opt/conda_envs/py313/bin/python -m unittest discover -s tests -p 'test_convert_glb_to_obj.py' -v

# 在 py313 环境下运行含强化学习训练测试的全套回归（需 gymnasium + stable-baselines3）
/opt/conda_envs/py313/bin/python -m unittest discover -s tests -v
```

缺少 gymnasium / stable-baselines3 时（如 py314 默认环境），强化学习相关测试自动跳过。

## 动力学模块

`src/dynamics.py` 在任意机械臂 MJCF 上提供标准刚体动力学计算：

```python
from src.dynamics import ArmDynamics

dyn = ArmDynamics("models/xarm6/xarm6.xml", payload=(3.0, [0, 0, 0.021]))  # 3 kg，COM 为 site 系米制坐标
M = dyn.mass_matrix(q)                     # 稠密质量矩阵 M(q)
g = dyn.gravity(q)                         # 重力项 g(q)
c = dyn.coriolis(q, qd)                    # 速度项 C(q,qd)qd
tau = dyn.inverse_dynamics(q, qd, qdd)     # 执行器力矩，与 forward_dynamics 严格互逆
qdd = dyn.forward_dynamics(q, qd, tau)
tau_ff = dyn.feedforward_traj(q_traj, dt)  # 关节轨迹前馈力矩
```

负载 COM 相对指定 site（默认 `attachment_site`）的原点与坐标轴表达，注入时计入 site 的平移和旋转。负载采用自身转动惯量为零的点质量；零质量不添加 body。组合模型的机械臂速度/加速度输入默认将夹爪项补零，完整位置向量可指定开度。仅输入机械臂力矩时求解机械臂质量矩阵块、夹爪加速度固定为零；完整力矩输入求解全部自由度。`full=True` 返回全部自由度，逆动力学要求非零夹爪加速度时必须启用。不计入闭环等式约束力。

基于 xarm-python-sdk 与真机 xArm 6 交叉验证（负载自动从 `tcp_load` 读取）：

```bash
python scripts/compare_dynamics_xarm6.py                     # 真机运动记录 + 逆动力学回放比对
python scripts/compare_dynamics_xarm6.py --replay build/dynamics/run_xxx.csv  # 从 .meta.json 恢复负载
python scripts/compare_dynamics_xarm6.py --replay build/dynamics/run_20261012_172940.csv --payload 3 0 0 21
python scripts/compare_torque_xarm6.py --no-move             # 静态保持力矩比对
```

记录保留 CSV 列格式，并保存同名 `.meta.json`（版本 1，字段 `payload.mass_kg`、`payload.com_m`、`payload.frame="attachment_site"`）。回放优先采用显式 `--payload M X Y Z`（kg + 法兰系 COM mm），其次采用元数据（COM 单位 m）。无元数据的旧 CSV 必须指定 `--payload`；零负载使用 `--payload 0 0 0 0`。

## 强化学习模块

`src/rl/reach_env.py` 实现 `XArm6ReachEnv`（Gymnasium 环境）：xArm 6 的 TCP 追踪随机采样的目标点（基座系工作空间 `x∈[0.25, 0.45], y∈[-0.25, 0.25], z∈[0.05, 0.35]`，启动时经 IK 可达性校验）。动作为关节位置增量（每控制步 ±0.05 rad，20 ms/步，限位钳制）；21 维观测包含 TCP−目标相对位置、归一化关节位置/速度与上一动作。奖励采用势能塑形（目标距离缩减）+ 成功奖励（15 mm）− 步数/动作/近限位惩罚；每回合 200 控制步（4 s）。

```bash
# 训练 PPO（py313：gymnasium + stable-baselines3，见 requirements-rl.txt）
/opt/conda_envs/py313/bin/python scripts/train_reach_rl.py            # 300k 步，8 并行环境
/opt/conda_envs/py313/bin/python scripts/train_reach_rl.py --total-steps 1000000

# 评估 / 可视化训练策略
/opt/conda_envs/py313/bin/python examples/07_rl_reach.py --headless --episodes 20
```

训练产物保存于 `build/rl/ppo_reach.zip` 与 `build/rl/metrics.json`（成功率、TCP 误差、包版本）。参考的 300k 步训练在 50 回合评估上达到 96% 成功率（平均 TCP 误差 15 mm、约 98 步到达，8 进程 CPU 约 2700 env-steps/s）。环境本身仅依赖 mujoco + numpy：无 gymnasium 时退化为空间存根，`examples/07_rl_reach.py --random` 在 py314 默认环境仍可运行。

## 项目结构

```
models/                 机器人及配件模型与几何资产
  xarm6/                xArm 6 模型（scene.xml、scene_reach.xml、xarm6.xml、视觉 OBJ、碰撞 STL）
  xarm5/, xarm7/, ...   xArm 5/7、UF850、Lite6 及各款夹爪（G2、Bio、Lite6）的资产目录
src/                    核心控制与仿真库
  sim_robot.py          SimRobot：物理步进、关节/TCP控制、渲染同步
  robot_interface.py    机械臂抽象基类（支持仿真与真机接口对齐）
  ik.py                 阻尼最小二乘法（DLS）笛卡尔 IK 解算器
  trajectory.py         JointTrajectory 数据模型、PTP 五次/LSPB、样条多路点、笛卡尔直线
  dynamics.py           质量矩阵、重力/科氏力、逆/正动力学、负载注入
  rl/reach_env.py       Gymnasium TCP 到达环境（xArm 6，PPO 训练目标）
  viewer.py             GLFW/X11 交互查看器生命周期封装
examples/               可执行示例（模型预览、关节控制、笛卡尔IK、夹爪控制、方块抓放、
                        轨迹规划、强化学习到达）
scripts/                模型与资产转换工具
  convert_glb_to_obj.py 批量 GLB -> OBJ 转换器（支持 Draco 解压与绕序自愈）
  build_all_mjcf.py     全系列 MJCF 模型与场景 XML 自动化生成工具
  build_model.py        官方 URDF -> MJCF 草稿生成工具
  compare_dynamics_xarm6.py  真机动态记录与逆动力学回放比对
  compare_torque_xarm6.py    静态保持力矩比对（真机 vs MuJoCo）
  train_reach_rl.py     强化学习 TCP 到达 PPO 训练脚本（py313）
tests/                  测试套件（查看器生命周期、全资产加载校验、绕序自愈回归）
```

## 模型规范

- 运动学参数与惯量张量严格继承自 UFACTORY 官方定义（[xarm_ros](https://github.com/xArm-Developer/xarm_ros) / [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis)）。
- 执行器增益与动力学配置参照 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 规范。
- 视觉几何（`group="2"`, `contype="0" conaffinity="0"`）与碰撞几何（`group="3"`）严格分层解耦。

## 开源协议

MIT

