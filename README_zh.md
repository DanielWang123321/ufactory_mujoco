# ufactory_mujoco

[English](README.md) | [中文](README_zh.md) · [更新记录](CHANGELOG.md) · [![CI](https://github.com/DanielWang123321/ufactory_mujoco/actions/workflows/ci.yml/badge.svg)](https://github.com/DanielWang123321/ufactory_mujoco/actions/workflows/ci.yml)

基于 MuJoCo 的 UFACTORY 机器人（Lite 6、xArm 5/6/7、UF850 及其夹爪）仿真与强化学习套件：精细模型、运动学与动力学工具、轨迹规划、脚本化操作任务、Gymnasium 强化学习环境，以及带安全门的 xArm 真机接口。

## 快速上手

```bash
git clone --depth 1 https://github.com/DanielWang123321/ufactory_mujoco.git && cd ufactory_mujoco
python -m pip install -e .
python examples/simulation/01_load_and_view.py --model xarm6_with_gripper --pose ready
```

无显示器时可无头运行任务：`ufactory-mujoco pick-place --robot xarm6 --headless`。

## 功能

- **模型**：由 [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis) 转换的全部 5 款机械臂与 4 款夹爪，以及 4 款深度相机（`d405`、`d435`、`gemini_305`、`zed_x_nano`）、UFACTORY 相机支架（`camera_stand`）及其支架组合与 `xarm6` + Gripper G2 组合，视觉网格与碰撞体分离。Gripper G2 采用标定的四连杆几何、实测夹持力表和限速伺服（`g2_model.py`、`g2_controller.py`）。
- **运动学与规划**：正运动学、雅可比、阻尼最小二乘逆解、五次/LSPB/样条轨迹、笛卡尔直线、碰撞检测与 RRT-Connect（`kinematics.py`、`ik.py`、`trajectory.py`、`collision.py`、`path_planning.py`）。
- **动力学**：质量矩阵、重力/科氏力、正逆动力学、负载，以及五臂验收命令（`dynamics.py`、`tasks/dynamics.py`）。
- **脚本化任务**：5 组机械臂/夹爪的方块抓放与装箱，含预检（关节限位、穿透、RRT-Connect 绕行）和 JSON 报告（`tasks/`、`preflight.py`）。
- **强化学习**：全部 5 款机械臂的 TCP 到达，以及 xArm 6 + Gripper G2 方块抓放；检查点通过 SHA-256 绑定配置与场景（`rl/`）。结果见 [到达](examples/rl/reach/README_zh.md#4-基准验收结果)、[抓放](examples/rl/pick_place/README_zh.md#4-基准验收结果)。
- **真机**：`XArmRobot` 默认只读；运动需单机标定、控制器状态检查和 `--confirm-real`。`sdk-sim` 在控制器内置仿真中重放轨迹（`xarm_robot.py`、`calibration.py`、`safety.py`）。

## 支持的机型与任务

| 机械臂 | 已构建的夹爪模型 | `pick-place` / `box-pack` | `dynamics` | RL 到达 | RL 抓放 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `lite6` | `lite6_gripper`、`lite6_vacuum_gripper` | 配 `lite6_gripper` | 支持 | 支持 | — |
| `xarm5` | `gripper_g2` | 配 `gripper_g2` | 支持 | 支持 | — |
| `xarm6` | `gripper_g2`、`bio_gripper_g2` | 配 `gripper_g2` | 支持 | 支持 | 配 `gripper_g2` |
| `xarm7` | `gripper_g2` | 配 `gripper_g2` | 支持 | 支持 | — |
| `uf850` | `gripper_g2` | 配 `gripper_g2` | 支持 | 支持 | — |

每款机械臂、夹爪、深度相机（`d405`、`d435`、`gemini_305`、`zed_x_nano`）及相机支架组合（`camera_stand`、`camera_stand_with_<相机>`、`xarm6_with_gripper_and_<相机>`）均可通过 `01_load_and_view.py --model <名称>` 加载。各模型的启动命令见 [docs/visual_command_guide.md](docs/visual_command_guide.md)。

## 安装

需要 Python **3.12–3.14** 与 MuJoCo **≥ 3.15、< 5**（已测试 3.15.0、3.16.0、3.17.0 与 4.0.0）。网格较大，请用 `--depth 1` 克隆。

```bash
python3.13 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e .       # 安装包及核心依赖（mujoco、numpy、scipy、pyyaml）
```

| 可选依赖 | 安装 | 用途 |
| :--- | :--- | :--- |
| `rl` | `pip install torch --index-url https://download.pytorch.org/whl/cpu`（或 CUDA 版），再 `pip install -e ".[rl]"` | PPO 训练与已训练策略回放 |
| `test` | `pip install -e ".[test]"` | 测试套件 |
| `assets` | `pip install -e ".[assets]"` | 仅用于资产再生成（见 [models/gripper_g2/source/README.md](models/gripper_g2/source/README.md)） |

`requirements-lock.txt` 固定了 CI 测试使用的版本（MuJoCo 4.0.0）；`requirements-lock-mujoco3.txt` 是对应的 MuJoCo 3.16.0 版本集合。

**把已有环境从 MuJoCo 3 升级到 MuJoCo 4**：MuJoCo 4 把 Python 包拆成了 `mujoco-core`、`mujoco-studio` 和 `mujoco`。原地升级（`pip install -U mujoco`）会得到损坏的安装，之后 `import mujoco` 报错，提示缺少 `mujoco-core`。请先卸载再安装：

```bash
pip uninstall -y mujoco && pip install "mujoco>=4,<5"
```

环境已经损坏时，用 `pip install --force-reinstall --no-deps mujoco mujoco-core mujoco-studio` 修复。新建的环境不受影响。

**wheel**：自 v0.1.0 起，[GitHub Release](https://github.com/DanielWang123321/ufactory_mujoco/releases) 附带 wheel，内含全部模型、场景、任务配置和许可证（下载约 102 MiB，安装后资源约 566 MiB）。用 `pip install ufactory_mujoco-0.1.2-py3-none-any.whl` 安装后，`ufactory-mujoco` 命令可在任意目录运行。也可用 `python -m build` 自行构建。

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
| `UFACTORY_GENESIS_ROOT` | 资产转换脚本使用的 `ufactory_genesis` 目录 |

## 强化学习权重

仓库和 Release 均**不包含**训练好的权重，需在本地训练（需安装 `rl` 依赖）：

```bash
python scripts/train_reach_rl.py --model xarm6 --total-steps 2000000 --net-arch 256,256,128 --ent-coef 0.001 --eval-episodes 200
python scripts/train_pick_place_rl.py --n-envs 128 --n-workers 24 --device cuda --seed 1
```

示例默认加载 `build/rl/<机械臂>/ppo_reach.zip` 和 `build/rl/pick_place/xarm6_g2_v2/ppo_pick_place.zip`。没有权重时，可用未训练的 `--random` 基线（到达）或脚本专家 `--expert`（抓放），两者都不需要 `rl` 依赖。训练耗时、种子敏感性与结果见 [到达](examples/rl/reach/README_zh.md) 和 [抓放](examples/rl/pick_place/README_zh.md) 文档。

## 平台说明

- **macOS**：GUI 示例用 `mujoco` 自带的 `mjpython` 启动，如 `mjpython examples/simulation/01_load_and_view.py`；`--headless` 用普通 `python` 即可。
- **Windows**：查看器用普通 `python` 即可。
- **无显示器的 Linux**：`--headless` 不需要显示器和 OpenGL，不要为此设置 `MUJOCO_GL`。`01_load_and_view.py` 的 `--headless` 只加载模型并打印摘要。`MUJOCO_GL=egl` 需要 `libegl1`，`MUJOCO_GL=osmesa` 需要 `libosmesa6`。在无显示器的服务器上打开查看器可用 `xvfb-run -a python examples/simulation/01_load_and_view.py`。

## 文档

| 内容 | 文档 |
| :--- | :--- |
| 仿真示例 `01`–`10`、命令行参数、真机命令 | [examples/simulation/README_zh.md](examples/simulation/README_zh.md)（[English](examples/simulation/README.md)） |
| 强化学习 TCP 到达 | [examples/rl/reach/README_zh.md](examples/rl/reach/README_zh.md)（[English](examples/rl/reach/README.md)） |
| 强化学习方块抓放 | [examples/rl/pick_place/README_zh.md](examples/rl/pick_place/README_zh.md)（[English](examples/rl/pick_place/README.md)） |
| 各模型启动命令 | [docs/visual_command_guide.md](docs/visual_command_guide.md) |
| 版本历史 | [CHANGELOG.md](CHANGELOG.md) |

## 运行测试

```bash
python -m pip install -e ".[test]"
python -m pytest tests
ruff check .                      # 与 CI 相同的静态检查（pip install ruff）
```

缺少可选包（`gymnasium`/`stable-baselines3`、`trimesh`、`DracoPy`/`pygltflib`）或本地训练权重时，相关测试自动跳过。

## 已知限制

- 不发布训练好的强化学习权重，见[强化学习权重](#强化学习权重)。
- 强化学习抓放仅支持 xArm 6 + Gripper G2。当前配方的训练结果强烈依赖随机种子，且不能跨机器复现（[详情](examples/rl/pick_place/README_zh.md#5-训练稳定性种子敏感性)）。
- 真机模式仅在一台 xArm 6（固件 v2.8.0）上验证。静态力矩比对（3 kg 负载，≤ 2.5 %）为单次历史实测。
- Bio Gripper G2 仅有 xArm 6 组合。`lite6_vacuum_gripper`、独立相机及 `camera_stand*` 没有夹爪执行器，`04_gripper_control.py` 会拒绝该模型（退出码 `2`）。
- 深度相机与 `camera_stand` 模型提供外观几何、安装面/螺钉孔 `<site>`（`group=4`）与近似碰撞体（`group=3`），`mass=0` 且不含 MuJoCo `<camera>` 光学成像传感器；动力学场景请通过 `attach_payload` 注入相机与支架负载。
- Lite 6 装箱仅在默认的 `servo_cartesian` 执行器下可靠。
- 不同 MuJoCo 版本的仿真结果不完全相同：在 3.16、3.17 与 4.0 之间，脚本化抓放的放置误差最多相差 4 mm，任务成败不变。基准结果均注明测量时的版本。

## 项目结构

```text
calibrations/              单机运动学标定 YAML（SHA-256 校验）
configs/                   任务配置：default.yaml、pick_place/<机械臂>.yaml、box_pack/<机械臂>.yaml
models/                    MJCF 模型与网格：机械臂、夹爪、机械臂+夹爪组合；
                           场景变体 scene、scene_reach、scene_pick_place、scene_box_pack、scene_obstacles
ufactory_mujoco/
  sim_robot.py             SimRobot：物理步进、关节/TCP/夹爪控制、查看器
  xarm_robot.py            XArmRobot：xArm 真机，未获运动授权时只读
  robot_interface.py       SimRobot 与 XArmRobot 的共同接口
  robots.py                5 款机械臂注册表、夹爪组合、到达工作空间
  config.py                严格 YAML 任务配置及其内容哈希
  calibration.py           单机标定、模型修补、TCP/法兰变换
  safety.py                真机安全门与 sdk-sim 重放
  kinematics_validation.py 与控制器的只读 FK/IK 比对
  g2_model.py              Gripper G2 连杆几何、夹持力表、SDK 单位换算
  g2_controller.py         Gripper G2 伺服（向量化与标量两种实现）
  grippers.py              夹爪规格、开口映射、指垫接触检测与接触力
  preflight.py             抓放/装箱预检，含 RRT-Connect 绕行
  task_geometry.py         任务、预检与强化学习共用的抓取高度和姿态
  ik.py, kinematics.py     DLS 逆解；正运动学、雅可比、可操作度
  trajectory.py            JointTrajectory、五次/LSPB、样条、笛卡尔直线
  dynamics.py              质量矩阵、重力/科氏力、正逆动力学、负载
  collision.py             碰撞与间隙查询
  path_planning.py         RRT-Connect 与路径剪枝
  rl/                      到达与抓放环境、评测、检查点契约、模型哈希
  tasks/                   pick-place、box-pack、dynamics 命令
  cli.py, __main__.py      `ufactory-mujoco` / `python -m ufactory_mujoco` 入口
  viewer.py                被动查看器，处理 Linux GLFW/X11
  resource_paths.py, model_paths.py  资源根目录与模型名解析
examples/simulation/       示例 01–10
examples/rl/               reach/ 与 pick_place/：回放、评测、recipe.yaml
scripts/                   强化学习训练、资产转换与 MJCF 生成、标定与真机比对、打包检查
tests/                     测试套件（pytest）
docs/                      各模型启动命令
```

## 模型说明

- 运动学与惯量来自 UFACTORY 官方定义（[xarm_ros](https://github.com/xArm-Developer/xarm_ros)、[ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis)）。
- 执行器参数遵循 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 规范。
- 视觉网格（`group=2`，`contype=0 conaffinity=0`）与碰撞网格（`group=3`）分离；相机与支架安装基准及螺钉孔以 `group=4` 的 `<site>` 提供（D435 含 `0.055 mm` 法向间隙以避免 ROS 视觉网格越过支架安装面）。

## 许可

代码采用 MIT（见 `LICENSE`）。机器人与相机模型源自 UFACTORY 资产（`xarm_ros`，BSD-3-Clause；`ufactory_genesis`，MIT）、Intel RealSense ROS（`realsense-ros`，Apache-2.0）以及奥比中光 / Stereolabs / UFACTORY 公开机械 CAD，详见 `NOTICE`。
