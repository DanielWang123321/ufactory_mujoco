# ufactory_mujoco

[English](README.md) | [中文](README_zh.md)

基于 MuJoCo 的 UFACTORY 机器人仿真与强化学习套件 —— 提供高精细视觉资产、运动学与动力学工具链、避障轨迹规划及 Gymnasium 强化学习环境。

## 功能特性

- **高精细视觉质感**：具备平滑法线与材质分色（烤漆白外壳、阳极氧化铝转轴、黑色部件及密封环）的高精度 OBJ 视觉网格，配合物理专用的 STL 碰撞网格。
- **全系列资产支持**：包含从 [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis) 官方解构转换的全部机械臂（`lite6`、`xarm5`、`xarm6`、`xarm7`、`uf850`）及末端配件（`gripper_g2`、`bio_gripper_g2`、`lite6_gripper`、`lite6_vacuum_gripper`）。
- **运动学与轨迹规划**：提供任意 site 正运动学、TCP 解析雅可比、阻尼最小二乘法（DLS）逆运动学（`ufactory_mujoco/kinematics.py`、`ufactory_mujoco/ik.py`），点到点五次/多路点 C2 样条轨迹（`ufactory_mujoco/trajectory.py`），以及 RRT-Connect 避障路径规划（`ufactory_mujoco/collision.py`、`ufactory_mujoco/path_planning.py`）。
- **刚体动力学模块**：支持质量矩阵 $M(q)$、重力/科氏力项、正/逆动力学与末端负载注入（`ufactory_mujoco/dynamics.py`），并通过真机 xArm 6 交叉比对验证。
- **强化学习环境**：全系 5 款机械臂通用的 TCP 位置到达 Gymnasium 环境（`ufactory_mujoco/rl/reach_env.py`，含标定工作空间），预训练 PPO 策略在 `< 5 mm` 官方标准下成功率均达 $\ge 96\%$。
- **稳定可靠的查看器**：集成实时时钟同步与异常安全退出保护，深度适配 Linux Wayland / X11（`ufactory_mujoco/viewer.py`）。

## 快速上手

### 1. 环境安装

推荐 Python **3.13**（3.12 亦可；核心与 RL 依赖在这两个版本上都有预编译包）。

```bash
# 仓库较大（网格约 600 MB），建议浅克隆
git clone --depth 1 https://github.com/DanielWang123321/ufactory_mujoco.git
cd ufactory_mujoco

python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # 安装 ufactory_mujoco 包及核心依赖

# 可选：强化学习训练与 PPO 回放（先装 CPU 版 torch，详见 requirements-rl.txt）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[rl]"
```

> CI 验证过的精确版本：`pip install -r requirements-lock.txt`。

### macOS 与 Windows

- **macOS**：交互查看器（`launch_passive`）必须在主线程运行，GUI 示例请用 `mujoco` 自带的 `mjpython` 启动：`mjpython examples/simulation/01_load_and_view.py`；`--headless` 用普通 `python` 即可。
- **Windows**：用 `.venv\Scripts\activate` 激活虚拟环境，查看器用普通 `python` 即可；`MUJOCO_GL` 与 Xvfb 仅与 Linux 有关。示例中的路径均相对仓库解析，可在任意目录运行。

### 无显示器 / 远程机器

- `--headless`（示例 `01`–`10`、`01_reach.py`）只跑物理，不需要显示器和 OpenGL，**不要**为此设置 `MUJOCO_GL`。
- 若设置 `MUJOCO_GL=egl`，系统需有 `libEGL`（如 `apt install libegl1`），否则 `import mujoco` 会直接报错；`MUJOCO_GL=osmesa` 需要 `libosmesa6`。
- 在无显示器的服务器上打开交互查看器，可用 Xvfb：`xvfb-run -a python examples/simulation/01_load_and_view.py`。

### 2. 示例与文档导航

| 类别 | 说明文档 | 内容概览 |
| :--- | :--- | :--- |
| **仿真示例 (`01`–`10`)** | [examples/simulation/README_zh.md](examples/simulation/README_zh.md) ([English](examples/simulation/README.md)) | 模型预览、关节/笛卡尔控制、夹爪控制、方块抓放、多方块装箱、轨迹与避障规划、运动学与动力学工具链 |
| **强化学习 (`01_reach`)** | [examples/rl/reach/README_zh.md](examples/rl/reach/README_zh.md) ([English](examples/rl/reach/README.md)) | 全系 5 款机械臂 TCP 位置到达环境、PPO 策略回放评测与训练指南 |
| **全机型指令速查** | [docs/visual_command_guide.md](docs/visual_command_guide.md) | 按产品型号（Lite 6、xArm 5/6/7、UF850）与组件状态分类的启动指令矩阵 |

```bash
# 交互式模型预览
python examples/simulation/01_load_and_view.py --model xarm6_with_gripper --pose ready

# 关节轨迹控制 / 方块抓放 / RRT-Connect 避障规划 / 多方块装箱
python examples/simulation/02_joint_control.py --model xarm6
python examples/simulation/05_pick_place.py
python examples/simulation/09_obstacle_avoidance.py
python examples/simulation/10_box_pack.py

# 强化学习 TCP 到达（加载 PPO 预训练策略；无 stable-baselines3 时可用 --random）
python examples/rl/reach/01_reach.py --model xarm6
```

> 所有运动与强化学习示例（`01`–`10`、`01_reach.py`）均支持 `--headless` 参数，用于无头模式运行并输出量化验收指标。

### 3. 运行测试套件

```bash
# 运行单元测试与全量 MJCF 资产加载校验
python -m unittest discover -s tests -v
```

*（依赖 `trimesh` 或 `gymnasium`/`stable-baselines3` 的专项测试在未安装对应包时会自动跳过。）*

## 项目结构

```text
configs/                严格类型校验的 YAML 任务配置（default、pick_place、box_pack）
models/                 机器人及配件 MJCF 模型与视觉/碰撞网格资产
  xarm6/, lite6/, ...   xArm 5/6/7、UF850、Lite6 及各款夹爪（独立与组合场景）
ufactory_mujoco/        核心仿真、控制、规划与强化学习库
  sim_robot.py          SimRobot：物理步进、关节/TCP控制、渲染同步
  robot_interface.py    机械臂抽象基类（支持仿真与真机接口对齐）
  robots.py             5 款机械臂规格注册表、组合场景路径解析与冻结工作空间
  config.py             严格 YAML 任务配置加载、校验与稳定内容哈希
  grippers.py           物理开口间距 <-> 执行器映射及双侧指垫接触检测
  preflight.py          抓放与装箱任务静态 IK/碰撞预检及 RRT 备选路径规划
  ik.py                 阻尼最小二乘法（DLS）笛卡尔 IK 解算器
  kinematics.py         FK、TCP 雅可比、速度级 DLS 逆解、可操作度/奇异性指标
  trajectory.py         JointTrajectory、PTP 五次/LSPB、C2 样条、笛卡尔 SLERP 直线
  dynamics.py           质量矩阵、重力/科氏力、逆/正动力学、负载注入
  collision.py          CollisionChecker：碰撞与最小间隙查询
  path_planning.py      RRT-Connect + 路径剪枝（shortcut）无碰撞规划
  rl/reach_env.py       Gymnasium TCP 到达强化学习环境（支持全系 5 款机械臂）
  model_paths.py        模型名称 → MJCF XML 路径解析
  viewer.py             GLFW/X11 交互查看器生命周期封装
examples/
  simulation/           仿真示例 01–10（详见 examples/simulation/README_zh.md）
  rl/reach/             强化学习 reach 示例（详见 examples/rl/reach/README_zh.md）
scripts/                资产转换、工作空间标定、真机动力学回放比对与 RL 训练脚本
tests/                  单元测试与回归测试套件
docs/internal/          内部设计与路线图记录（非用户文档）
```

## 模型规范

- 运动学参数与惯量张量严格继承自 UFACTORY 官方定义（[xarm_ros](https://github.com/xArm-Developer/xarm_ros) / [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis)）。
- 执行器增益与动力学配置参照 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 规范。
- 视觉几何（`group="2"`, `contype="0" conaffinity="0"`）与碰撞几何（`group="3"`）严格分层解耦。

## 开源协议

代码采用 MIT（见 `LICENSE`）。机器人模型源自 UFACTORY 资产（xarm_ros，BSD-3-Clause；ufactory_genesis，MIT），详见 `NOTICE`。
