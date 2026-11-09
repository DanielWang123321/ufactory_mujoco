# ufactory_mujoco

[English](README.md) | [中文](README_zh.md)

基于 MuJoCo 的 UFACTORY 机器人仿真与强化学习套件 —— 提供高精细视觉资产、运动学与动力学工具链、避障轨迹规划及 Gymnasium 强化学习环境。

## 功能特性

- **高精细视觉质感**：具备平滑法线与材质分色（烤漆白外壳、阳极氧化铝转轴、黑色部件及密封环）的高精度 OBJ 视觉网格，配合物理专用的 STL 碰撞网格。
- **全系列资产支持**：包含从 [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis) 官方解构转换的全部机械臂（`lite6`、`xarm5`、`xarm6`、`xarm7`、`uf850`）及末端配件（`gripper_g2`、`bio_gripper_g2`、`lite6_gripper`、`lite6_vacuum_gripper`），其中 Gripper G2 支持标定四连杆非线性几何映射、实测速度-电流夹持力查表及限速 PI+PD 位置-力矩混合闭环伺服控制（`ufactory_mujoco/g2_model.py`、`ufactory_mujoco/g2_controller.py`）。
- **运动学与轨迹规划**：提供任意 site 正运动学、TCP 解析雅可比、阻尼最小二乘法（DLS）逆运动学（`ufactory_mujoco/kinematics.py`、`ufactory_mujoco/ik.py`），点到点五次/多路点 C2 样条轨迹（`ufactory_mujoco/trajectory.py`），以及 RRT-Connect 避障路径规划（`ufactory_mujoco/collision.py`、`ufactory_mujoco/path_planning.py`）。
- **刚体动力学模块**：支持质量矩阵 $M(q)$、重力/科氏力项、正/逆动力学与末端负载注入（`ufactory_mujoco/dynamics.py`），包含 xArm 6 真机交叉验证；`08_dynamics.py` 覆盖五款裸臂的碰撞预检、逐步运动验收及 JSON 报告。
- **强化学习环境与契约**：全系 5 款机械臂通用的 TCP 位置到达环境（`ufactory_mujoco/rl/reach_env.py`，`< 5 mm` 标准下成功率 $\ge 96\%$）以及 `xArm 6 + Gripper G2` 桌面物块抓放环境（`ufactory_mujoco/rl/pick_place_env.py`，从零训练的纯 PPO v2 策略与闭环专家基线在 64 回合随机出生域评测中成功率均为 $100\%$、触地下压 $\le 1\text{ mm}$）。抓放检查点通过 SHA-256 绑定训练配置、场景和网格（`ufactory_mujoco/rl/artifacts.py`）。
- **真机接口与安全确认门**：`XArmRobot`（`ufactory_mujoco/xarm_robot.py`）实现 `RobotInterface`，默认只读观测，支持单机运动学标定（`ufactory_mujoco/calibration.py`）、控制器 FK/IK 比对（`ufactory_mujoco/kinematics_validation.py`）、`sdk-sim` 控制器内部仿真及 `--confirm-real` 单周期安全门禁（`ufactory_mujoco/safety.py`）。
- **稳定可靠的查看器**：集成实时时钟同步与异常安全退出保护，深度适配 Linux Wayland / X11（`ufactory_mujoco/viewer.py`）。

## 快速上手

### 1. 环境安装

核心仿真支持 Python **3.12、3.13、3.14**；RL CI 使用 **Python 3.13 + CPU PyTorch**。

使用已构建的 wheel，无需克隆源码：

```bash
python -m pip install /path/to/ufactory_mujoco-0.1.0.dev0-py3-none-any.whl
ufactory-mujoco pick-place --robot xarm6 --headless
ufactory-mujoco box-pack --robot lite6 --cycles 1 --headless
ufactory-mujoco dynamics --robot all --samples 32 --seed 0 --headless --report reports/dynamics.json
```

wheel 自带全部模型、场景、默认/机型 YAML 和许可证（压缩约 89 MiB、资源展开约 533 MiB）。可从任意目录运行，无需下载资源或写入安装目录。核心安装不需要 RL、资产转换工具或 xArm SDK。

源码开发安装：

```bash
# 网格较大，建议浅克隆
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

使用 setuptools 构建两种产物：

```bash
python -m pip install build twine
python -m build                 # 先 sdist，再从 sdist 重建 wheel
python -m twine check dist/*
# 可选，仅用于资产再生成：
python -m pip install -e ".[assets]"
```

`ufactory-mujoco pick-place` 和 `box-pack` 支持 `--robot`、`--gripper`、`--executor`、`--config`、`--cycles`、`--headless`、`--dry-run`、`--mode {sim,dry-run,sdk-sim,real}`、`--ip`、`--calibration`、`--confirm-real`、`--report`；装箱另支持 `--speed`。`ufactory-mujoco dynamics` 支持 `--mode {sim,real-read-only,sdk-sim,real}`、`--ip`、`--calibration` 与 `--confirm-real`。各子命令可用 `--help` 查询。Lite6 装箱沿用现有 `servo_cartesian` 限制以保证运送稳定。原三个仿真脚本继续作为 `ufactory_mujoco.tasks` 的兼容包装。

退出码：**0** 成功，**1** 运行/预检/安全门/报告失败，**2** 参数或配置非法。报告路径相对当前目录，自动创建父目录；`--dry-run` 不步进物理；`--mode sdk-sim` 与 `--mode real` 强制 `cycles == 1`，要求匹配的 `--calibration` 标定文件，且物理运动必须显式传入 `--confirm-real`。`UFACTORY_MUJOCO_MODELS` 优先覆盖模型目录；否则普通安装读取包内资源，源码/editable 读取仓库资源。配置哈希仅取决于内容，不受安装路径影响。

CI 包含 Python 3.12/3.13/3.14 的 editable 核心回归及无源码 wheel 验收，以及 Python 3.13 CPU Reach 与 Pick-and-Place RL 测试及训练/评测烟测；失败时仍保存日志与 JSON 报告。

### macOS 与 Windows

- **macOS**：交互查看器（`launch_passive`）必须在主线程运行，GUI 示例请用 `mujoco` 自带的 `mjpython` 启动：`mjpython examples/simulation/01_load_and_view.py`；`--headless` 用普通 `python` 即可。
- **Windows**：用 `.venv\Scripts\activate` 激活虚拟环境，查看器用普通 `python` 即可；`MUJOCO_GL` 与 Xvfb 仅与 Linux 有关。示例中的路径均相对仓库解析，可在任意目录运行。

### 无显示器 / 远程机器

- `--headless`（示例 `01`–`10`、`01_reach.py`、`01_pick_place.py`）只跑物理，不需要显示器和 OpenGL，**不要**为此设置 `MUJOCO_GL`。
- 若设置 `MUJOCO_GL=egl`，系统需有 `libEGL`（如 `apt install libegl1`），否则 `import mujoco` 会直接报错；`MUJOCO_GL=osmesa` 需要 `libosmesa6`。
- 在无显示器的服务器上打开交互查看器，可用 Xvfb：`xvfb-run -a python examples/simulation/01_load_and_view.py`。

### 2. 示例与文档导航

| 类别 | 说明文档 | 内容概览 |
| :--- | :--- | :--- |
| **仿真示例 (`01`–`10`)** | [examples/simulation/README_zh.md](examples/simulation/README_zh.md) ([English](examples/simulation/README.md)) | 模型预览、关节/笛卡尔控制、夹爪控制、方块抓放、多方块装箱、轨迹与避障规划、运动学与动力学工具链 |
| **强化学习到达 (`01_reach`)** | [examples/rl/reach/README_zh.md](examples/rl/reach/README_zh.md) ([English](examples/rl/reach/README.md)) | 全系 5 款机械臂 TCP 位置到达环境、PPO 策略回放评测与训练指南 |
| **强化学习抓放 (`01_pick_place`)** | [examples/rl/pick_place/README_zh.md](examples/rl/pick_place/README_zh.md) ([English](examples/rl/pick_place/README.md)) | `xArm 6 + Gripper G2` 桌面物块抓放环境、闭环专家基线与 PPO 契约评测 |
| **全机型指令速查** | [docs/visual_command_guide.md](docs/visual_command_guide.md) | 按产品型号（Lite 6、xArm 5/6/7、UF850）与组件状态分类的启动指令矩阵 |

```bash
# 交互式模型预览
python examples/simulation/01_load_and_view.py --model xarm6_with_gripper --pose ready

# 关节轨迹控制 / 方块抓放 / RRT-Connect 避障规划 / 多方块装箱
python examples/simulation/02_joint_control.py --model xarm6
python examples/simulation/05_pick_place.py
python examples/simulation/09_obstacle_avoidance.py
python examples/simulation/10_box_pack.py

# 强化学习 TCP 到达与桌面抓放（无 stable-baselines3 时可用 --random / --expert）
python examples/rl/reach/01_reach.py --model xarm6
python examples/rl/pick_place/01_pick_place.py --expert --headless
```

> 所有运动与强化学习示例（`01`–`10`、`01_reach.py`、`01_pick_place.py`）均支持 `--headless` 参数，用于无头模式运行并输出量化验收指标。

### 3. 运行测试套件

```bash
# 运行单元测试与全量 MJCF 资产加载校验
python -m unittest discover -s tests -v
```

*（依赖 `trimesh` 或 `gymnasium`/`stable-baselines3` 的专项测试在未安装对应包时会自动跳过。）*

## 项目结构

```text
calibrations/           单机控制器运动学标定 YAML 文件（含 SHA-256 校验）
configs/                严格类型校验的 YAML 任务配置（default、pick_place、box_pack）
models/                 机器人及配件 MJCF 模型与视觉/碰撞网格资产
  xarm6/, lite6/, ...   xArm 5/6/7、UF850、Lite6 及各款夹爪（独立与组合场景）
ufactory_mujoco/        核心仿真、控制、规划、真机接口与强化学习库
  sim_robot.py          SimRobot：物理步进、关节/TCP控制、渲染同步
  xarm_robot.py         XArmRobot：默认只读会话与安全门授权后的真机控制接口
  robot_interface.py    机械臂抽象基类（支持仿真与真机接口对齐）
  robots.py             5 款机械臂规格注册表、组合场景路径解析与冻结工作空间
  config.py             严格 YAML 任务配置加载、校验与稳定内容哈希
  calibration.py        单机运动学标定加载、MuJoCo 模型修补与 TCP/法兰变换
  safety.py             真机安全确认门、单周期硬上限与 sdk-sim 内部仿真重放
  kinematics_validation.py 与实机控制器固件的只读 FK/IK 交叉校验
  g2_model.py           Gripper G2 四连杆非线性几何、实测速度-电流夹持力查表与 SDK 脉冲/转速换算
  g2_controller.py      Gripper G2 限速 PI+PD 位置-力矩混合闭环伺服控制器
  grippers.py           物理开口间距 <-> 执行器映射、双侧指垫接触检测与单指净接触力求和
  preflight.py          抓放与装箱任务静态 IK/碰撞预检及 RRT 备选路径规划
  ik.py                 阻尼最小二乘法（DLS）笛卡尔 IK 解算器
  kinematics.py         FK、TCP 雅可比、速度级 DLS 逆解、可操作度/奇异性指标
  trajectory.py         JointTrajectory、PTP 五次/LSPB、C2 样条、笛卡尔 SLERP 直线
  dynamics.py           质量矩阵、重力/科氏力、逆/正动力学、负载注入
  collision.py          CollisionChecker：碰撞与最小间隙查询
  path_planning.py      RRT-Connect + 路径剪枝（shortcut）无碰撞规划
  rl/reach_env.py       Gymnasium TCP 到达强化学习环境（支持全系 5 款机械臂）
  rl/pick_place_env.py  Gymnasium 桌面物块抓放强化学习环境与闭环专家基线
  rl/artifacts.py       声明式 RL 配方、配置哈希与检查点契约校验
  rl/model_hash.py      递归 MJCF 场景与网格资产 SHA-256 内容指纹校验
  tasks/                可安装 pick-place、box-pack 与 dynamics 任务命令
  resource_paths.py     源码/editable 与打包资源路径解析
  model_paths.py        模型名称 → MJCF XML 路径解析
  viewer.py             GLFW/X11 交互查看器生命周期封装
examples/
  simulation/           仿真示例 01–10（详见 examples/simulation/README_zh.md）
  rl/reach/             强化学习 reach 示例与评测（详见 examples/rl/reach/README_zh.md）
  rl/pick_place/        强化学习 pick-and-place 示例与评测（详见 examples/rl/pick_place/README_zh.md）
scripts/                资产转换、单机标定提取、真机动力学回放比对与 RL 训练脚本
tests/                  单元测试与回归测试套件
docs/                   按机械臂分类的命令速查
```

## 模型规范

- 运动学参数与惯量张量严格继承自 UFACTORY 官方定义（[xarm_ros](https://github.com/xArm-Developer/xarm_ros) / [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis)）。
- 执行器增益与动力学配置参照 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 规范。
- 视觉几何（`group="2"`, `contype="0" conaffinity="0"`）与碰撞几何（`group="3"`）严格分层解耦。

## 开源协议

代码采用 MIT（见 `LICENSE`）。机器人模型源自 UFACTORY 资产（xarm_ros，BSD-3-Clause；ufactory_genesis，MIT），详见 `NOTICE`。
