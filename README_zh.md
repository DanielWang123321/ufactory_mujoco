# ufactory_mujoco

![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue) ![MuJoCo](https://img.shields.io/badge/mujoco-4.x-lightgrey) ![License](https://img.shields.io/badge/license-MIT-green) [![CI](https://github.com/DanielWang123321/ufactory_mujoco/actions/workflows/ci.yml/badge.svg)](https://github.com/DanielWang123321/ufactory_mujoco/actions/workflows/ci.yml)

[English](README.md) | [中文](README_zh.md) · [更新记录](CHANGELOG.md)

基于 MuJoCo 的 UFACTORY 机器人（Lite 6、xArm 5/6/7、UF850）、夹爪与深度相机仿真套件：精细 MJCF 模型、运动学与动力学工具、轨迹规划、脚本化操作任务，以及带安全门的 xArm 真机接口。

## 快速上手

需要 Python **3.12–3.14**（详见[安装](#安装)）。

```bash
git clone --depth 1 https://github.com/DanielWang123321/ufactory_mujoco.git && cd ufactory_mujoco
python -m pip install -e .
python examples/simulation/01_load_and_view.py --model xarm6_with_gripper --pose ready
```

无显示器时可无头运行任务：`ufactory-mujoco pick-place --robot xarm6 --headless`。

## 功能

- **模型**：全部 5 款机械臂、5 款末端工具、4 款深度相机、相机支架及组合 MJCF 模型，视觉网格与多凸包碰撞体分离。Gripper G2 与 Bio Gripper G2 采用标定动力学与限速位置/力伺服（`g2_model.py`、`g2_controller.py`、`bio_g2_model.py`、`bio_g2_controller.py`）。
- **运动学与规划**：正运动学、雅可比、阻尼最小二乘逆解、五次/LSPB/样条轨迹、笛卡尔直线、碰撞检测与 RRT-Connect（`kinematics.py`、`ik.py`、`trajectory.py`、`collision.py`、`path_planning.py`）。
- **动力学**：质量矩阵、重力/科氏力、正逆动力学、负载、五臂运动验收命令（`tasks/dynamics.py`），以及对齐 `ufactory_genesis` 的 Schema v4 真机静态力矩验证工具链（`dynamics/`，提供 `dynamics-sim-check`、`dynamics-sim-collision-check`、`dynamics-hardware-check`、`dynamics-report-compare`）。
- **脚本化任务**：5 组机械臂/夹爪的方块抓放与装箱，含预检（关节限位、穿透、RRT-Connect 绕行）和 JSON 报告（`tasks/`、`preflight.py`）。
- **真机**：`XArmRobot` 默认只读；运动需单机标定、控制器状态检查和 `--confirm-real`。`sdk-sim` 在控制器内置仿真中重放轨迹（`xarm_robot.py`、`calibration.py`、`safety.py`）。

## 支持的机型与任务

| 机械臂 | 已构建的夹爪模型 | `pick-place` / `box-pack` | `dynamics` |
| :--- | :--- | :--- | :---: |
| `lite6` | `lite6_gripper`、`lite6_vacuum_gripper` | 配 `lite6_gripper` | 支持 |
| `xarm5` | `gripper_g2`、`bio_gripper_g2`、`vacuum_gripper` | 配 `gripper_g2` | 支持 |
| `xarm6` | `gripper_g2`、`bio_gripper_g2`、`vacuum_gripper` | 配 `gripper_g2` | 支持 |
| `xarm7` | `gripper_g2`、`bio_gripper_g2`、`vacuum_gripper` | 配 `gripper_g2` | 支持 |
| `uf850` | `gripper_g2`、`bio_gripper_g2`、`vacuum_gripper` | 配 `gripper_g2` | 支持 |

仓库在 `models/` 下共内置 **44 套模型目录**：5 款裸臂、5 款独立末端工具、4 款深度相机（`d405`、`d435`、`gemini_305`、`zed_x_nano`）、5 套 `camera_stand*` 支架模型、14 套机械臂 + 末端组合，以及 11 套机械臂 + 相机组合（`xarm6_with_gripper_and_<相机>` 以及 `xarm5/6/7`、`uf850` 的 `<机械臂>_with_d435` 与 `<机械臂>_with_gripper_and_d435`）。均可通过 `01_load_and_view.py --model <名称>` 加载，完整模型名与启动命令见 [examples/simulation/README_zh.md](examples/simulation/README_zh.md#4-按机型的命令)。

## 安装

| 组件 | 支持范围 | 已验证版本 | 说明 |
| :--- | :--- | :--- | :--- |
| **Python** | **3.12、3.13、3.14** (`>=3.12`) | `3.12.13`、`3.13.13`、`3.14.6` | CI 与发版验收全量覆盖三版本；**不支持 Python ≤ 3.11**；Python ≥ 3.15 未测试 |
| **MuJoCo** | **≥ 4.0、< 5** | `4.0.0`、`4.0.1` | 自 `v0.1.5` 起不再支持 MuJoCo 3.x |
| 核心依赖 | `numpy>=2.0,<3`、`scipy>=1.14,<2`、`pyyaml>=6.0,<7` | 见 `requirements-lock.txt` | 执行 `pip install -e .` 时自动安装 |
| `.[test]`（可选） | `pytest>=8` | — | 仅运行单元测试套件时需要 |
| `xarm-python-sdk`（可选） | 与控制器匹配的版本 | — | 仅 `--mode sdk-sim` / `real` / `real-read-only`、`dynamics-hardware-check` / `dynamics-sim-collision-check` 与标定导出需要 |
| `matplotlib`（可选） | `>=3.8` | — | 可选；用于在 `dynamics-hardware-check` 生成 `_torque.png` 力矩对比图（未安装时自动跳过） |

网格体积较大，请使用 `--depth 1` 克隆。可通过 `venv` 或 `conda` 创建隔离环境：

```bash
git clone --depth 1 https://github.com/DanielWang123321/ufactory_mujoco.git && cd ufactory_mujoco

# 方式 A：venv（Python 3.12 / 3.13 / 3.14）
python3.13 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e .

# 方式 B：conda（Python 3.12 / 3.13 / 3.14）
conda create -n ufactory python=3.14 -y && conda activate ufactory
python -m pip install -e .
```

`requirements-lock.txt` 固定了 CI 测试使用的精确版本（MuJoCo 4.0.0）。

**把已有环境从 MuJoCo 3 升级到 MuJoCo 4**：MuJoCo 4 把 Python 包拆成了 `mujoco-core`、`mujoco-studio` 和 `mujoco`。原地升级（`pip install -U mujoco`）会得到损坏的安装，之后 `import mujoco` 报错，提示缺少 `mujoco-core`。请先卸载再安装：

```bash
pip uninstall -y mujoco && pip install "mujoco>=4,<5"
```

环境已经损坏时，用 `pip install --force-reinstall --no-deps mujoco mujoco-core mujoco-studio` 修复。新建的环境不受影响。

**wheel**：自 v0.1.0 起，[GitHub Release](https://github.com/DanielWang123321/ufactory_mujoco/releases) 附带自包含 wheel，内含全部模型、场景、任务配置和许可证（下载约 112 MiB，安装后资源约 630 MiB）。用 `pip install ufactory_mujoco-0.1.6-py3-none-any.whl` 安装后，`ufactory-mujoco` 与 `dynamics-*` 命令可在任意目录运行。也可用 `python -m build` 自行构建。

## 命令行任务

```bash
ufactory-mujoco pick-place --robot xarm6 --headless
ufactory-mujoco box-pack --robot lite6 --cycles 3 --headless
ufactory-mujoco dynamics --robot all --headless --report reports/dynamics.json
```

- `python -m ufactory_mujoco <命令>` 与之等价；示例 `05`、`08`、`10` 运行的是同一份代码。
- `命令 --help` 列出参数；完整参数表见 [examples/simulation/README_zh.md](examples/simulation/README_zh.md#3-命令行参数汇总)。
- `--dry-run` 只做预检，不步进物理。
- 退出码：`0` 成功；`1` 运行、预检、安全门或报告失败；`2` 参数或配置非法。
- `--report` 路径相对当前目录，自动创建父目录。
- 真机模式（`--mode sdk-sim` / `real`，`dynamics` 另有 `real-read-only`）需要 `--ip`（或 `XARM_IP`）和匹配的 `--calibration` 标定文件；`real` 还需 `--confirm-real`；`sdk-sim` 与 `real` 只运行一个周期。

| 环境变量 | 作用 |
| :--- | :--- |
| `UFACTORY_MUJOCO_MODELS` | 使用指定的模型目录，替代包内或仓库中的模型 |
| `XARM_IP` | 真机模式的默认控制器 IP |
| `MUJOCO_GL` | Linux OpenGL 后端（`egl`、`osmesa`）；`--headless` 不需要 |

## 平台说明

- **macOS**：GUI 示例用 `mujoco` 自带的 `mjpython` 启动，如 `mjpython examples/simulation/01_load_and_view.py`；`--headless` 用普通 `python` 即可。
- **Windows**：查看器用普通 `python` 即可。
- **无显示器的 Linux**：`--headless` 不需要显示器和 OpenGL，不要为此设置 `MUJOCO_GL`。`01_load_and_view.py` 的 `--headless` 只加载模型并打印摘要。`MUJOCO_GL=egl` 需要 `libegl1`，`MUJOCO_GL=osmesa` 需要 `libosmesa6`。在无显示器的服务器上打开查看器可用 `xvfb-run -a python examples/simulation/01_load_and_view.py`。

## 文档

| 内容 | 文档 |
| :--- | :--- |
| 仿真示例 `01`–`10`、命令行参数、按机型的命令、视窗操作、真机命令 | [examples/simulation/README_zh.md](examples/simulation/README_zh.md)（[English](examples/simulation/README.md)） |
| 贡献指南、安全策略、行为准则（英文） | [CONTRIBUTING.md](CONTRIBUTING.md)、[SECURITY.md](SECURITY.md)、[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) |
| 版本历史 | [CHANGELOG.md](CHANGELOG.md) |

## 运行测试

```bash
python -m pip install -e ".[test]"
python -m pytest tests
ruff check .                      # 与 CI 相同的静态检查（pip install ruff）
```

缺少可选包（`trimesh`）或维护者本地源快照时，相关测试自动跳过。

## 已知限制

- 对齐 `ufactory_genesis` Schema v4 的裸臂 20 姿态真机静态力矩验证（`dynamics-hardware-check`）已在 Python 3.14.6、MuJoCo 4.0.1 下完成一台 xArm 6（固件 v2.8.0，20/20 PASS，L2 力矩误差 0.470–0.765 Nm）与一台 Lite 6（固件 v2.8.2，20/20 PASS，L2 力矩误差 0.028–0.156 Nm）实测；抓放/装箱真机模式已在 xArm 6（v2.8.0）上验证，3 kg 负载静态力矩比对（≤ 2.5 %）为单次历史实测。
- 真空吸盘（`lite6_vacuum_gripper`、`vacuum_gripper` 及其机械臂组合）为静态模型，不含吸附模型。它们与独立相机、`camera_stand*`、`<机械臂>_with_d435` 都没有夹爪执行器，`04_gripper_control.py` 会拒绝这些模型（退出码 `2`）。
- 深度相机与 `camera_stand` 模型提供外观几何、安装面/螺钉孔 `<site>`（`group=4`）与近似碰撞体（`group=3`），`mass=0` 且不含 MuJoCo `<camera>` 光学成像传感器；动力学场景请通过 `attach_payload` 注入相机与支架负载。
- 任务报告在 MuJoCo 4.0.0 与 4.0.1 下逐字节一致；其他 MuJoCo 版本可能使结果在数值上不同。

## 项目结构

```text
calibrations/              单机运动学标定 YAML（SHA-256 校验）
configs/                   任务与验证配置：default.yaml、dynamics_validation_pose.yaml、
                           pick_place/<机械臂>.yaml、box_pack/<机械臂>.yaml
models/                    MJCF 模型与网格：机械臂、夹爪、相机、支架及组合；
                           场景变体 scene、scene_pick_place、scene_box_pack、scene_obstacles
ufactory_mujoco/
  sim_robot.py             SimRobot：物理步进、关节/TCP/夹爪控制、查看器
  xarm_robot.py            XArmRobot：xArm 真机，未获运动授权时只读
  robot_interface.py       SimRobot 与 XArmRobot 的共同接口
  robots.py                5 款机械臂注册表与兼容夹爪
  config.py                严格 YAML 任务配置及其内容哈希
  calibration.py           单机标定、模型修补、TCP/法兰变换
  safety.py                真机安全门与 sdk-sim 重放
  kinematics_validation.py 与控制器的只读 FK/IK 比对
  g2_model.py              Gripper G2 连杆几何、夹持力表、SDK 单位换算
  g2_controller.py         Gripper G2 向量化位置/力伺服
  bio_g2_model.py          Bio Gripper G2 标定的开口、速度与夹持力模型
  bio_g2_controller.py     Bio Gripper G2 限速位置/力伺服
  grippers.py              夹爪规格、开口映射、指垫接触检测与接触力
  preflight.py             抓放/装箱预检，含 RRT-Connect 绕行
  task_geometry.py         任务与预检共用的抓取高度和姿态
  ik.py, kinematics.py     DLS 逆解；正运动学、雅可比、可操作度
  trajectory.py            JointTrajectory、五次/LSPB、样条、笛卡尔直线
  dynamics/                质量矩阵、重力/科氏力、正逆动力学、负载与 Schema v4 真机力矩验证工具链（`dynamics-*`）
  collision.py             碰撞与间隙查询
  path_planning.py         RRT-Connect 与路径剪枝
  tasks/                   pick-place、box-pack、dynamics 命令
  cli.py, __main__.py      `ufactory-mujoco` / `python -m ufactory_mujoco` 入口
  viewer.py                被动查看器，处理 Linux GLFW/X11
  model_paths.py           资源根目录与模型名解析
examples/simulation/       示例 01–10
scripts/                   标定提取、发布包与安装检查
tests/                     测试套件（pytest）
```

## 模型说明

- 运动学与惯量来自 UFACTORY 官方定义（[xarm_ros](https://github.com/xArm-Developer/xarm_ros)、[ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis)）。
- 执行器参数遵循 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 规范；其中 Lite 6 关节位置执行器 PD 增益按统一时间常数 `0.04 s` 调优以提升笛卡尔直线跟踪精度，Gripper G2 与 Bio Gripper G2 采用实测标定伺服模型。
- 视觉网格（`group=2`，`contype=0 conaffinity=0`）与碰撞网格（`group=3`）分离；相机与支架安装基准及螺钉孔以 `group=4` 的 `<site>` 提供（D435 含 `0.055 mm` 法向间隙以避免 ROS 视觉网格越过支架安装面；安装在机械臂上的 `d435` 采用与人眼一致的正向视野朝向，工具朝下姿态下 USB-C 口位于法兰 `+Y` / 基座 `Y-` 侧）。

## 许可

代码采用 MIT（见 `LICENSE`）。机器人与相机模型源自 UFACTORY 资产（`xarm_ros`，BSD-3-Clause；`ufactory_genesis`，MIT）、Intel RealSense ROS（`realsense-ros`，Apache-2.0）以及奥比中光 / Stereolabs / UFACTORY 公开机械 CAD，详见 `NOTICE`。
