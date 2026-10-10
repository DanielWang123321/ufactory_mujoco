# ufactory_mujoco

[English](README.md) | [中文](README_zh.md)

基于 MuJoCo 的 UFACTORY 机器人仿真套件 —— 提供高精细视觉资产、关节控制、笛卡尔逆运动学（IK）及交互式可视化。

## 功能特性

- **高精细视觉质感**：具备平滑法线与材质分色（烤漆白外壳、阳极氧化铝转轴、黑色部件及密封环）的高精度 OBJ 视觉网格，配合物理专用的 STL 碰撞网格。
- **全系列资产支持**：包含从 [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis) 官方解构转换的全部机械臂（xArm 5、xArm 6、xArm 7、UF850、Lite6）及末端配件（Gripper G2、Bio Gripper G2、Lite6 Gripper、Lite6 Vacuum Gripper）。
- **运动控制与动力学**：支持五次多项式关节插值平滑规划、重力补偿前馈控制、以及阻尼最小二乘法（DLS）笛卡尔逆运动学解算。
- **稳定可靠的查看器**：集成实时时钟同步、异常安全退出保护，针对 Linux Wayland / X11 进行了深度适配。

## 快速上手

### 环境安装

```bash
conda create -n py314 python=3.14 -y
conda activate py314
pip install -r requirements.txt
```

### 运行示例

```bash
# 交互式查看器：载入模型并预览（默认 xarm6，支持任意机械臂、夹爪或组合模型）
python examples/01_load_and_view.py
python examples/01_load_and_view.py --model xarm7 --pose ready
python examples/01_load_and_view.py --model uf850
python examples/01_load_and_view.py --model lite6
python examples/01_load_and_view.py --model gripper_g2
python examples/01_load_and_view.py --model xarm6_with_gripper --pose ready

# 关节空间轨迹控制（五次多项式插值）
python examples/02_joint_control.py            # GUI 窗口模式
python examples/02_joint_control.py --headless  # 无窗口模式（输出到位精度指标）

# 笛卡尔空间 IK 跟踪控制：直线 + 圆弧轨迹
python examples/03_cartesian_ik.py             # GUI 窗口模式
python examples/03_cartesian_ik.py --headless   # 无窗口模式（输出轨迹跟踪精度）
```

### 运行测试套件

```bash
# 在 py314 环境下运行单元测试与全量资产 MuJoCo 加载校验
python -m unittest discover -s tests -v

# 在 py313 环境下运行 GLB 转换与面绕序自愈回归测试（需 genesis / trimesh）
/opt/conda_envs/py313/bin/python -m unittest discover -s tests -p 'test_convert_glb_to_obj.py' -v
```

## 项目结构

```
models/                 机器人及配件模型与几何资产
  xarm6/                xArm 6 模型（scene.xml、xarm6.xml、视觉 OBJ、碰撞 STL）
  xarm5/, xarm7/, ...   xArm 5/7、UF850、Lite6 及各款夹爪（G2、Bio、Lite6）的资产目录
src/                    核心控制与仿真库
  sim_robot.py          SimRobot：物理步进、关节/TCP控制、渲染同步
  robot_interface.py    机械臂抽象基类（支持仿真与真机接口对齐）
  ik.py                 阻尼最小二乘法（DLS）笛卡尔 IK 解算器
  viewer.py             GLFW/X11 交互查看器生命周期封装
examples/               可执行示例（模型预览、关节控制、笛卡尔IK）
scripts/                模型与资产转换工具
  convert_glb_to_obj.py 批量 GLB -> OBJ 转换器（支持 Draco 解压与绕序自愈）
  build_all_mjcf.py     全系列 MJCF 模型与场景 XML 自动化生成工具
  build_model.py        官方 URDF -> MJCF 草稿生成工具
tests/                  测试套件（查看器生命周期、全资产加载校验、绕序自愈回归）
```

## 资产转换工具说明

若需重新转换或更新来自 [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis) 的资产：

```bash
# 一键转换全部机械臂与配件
/opt/conda_envs/py313/bin/python scripts/convert_glb_to_obj.py --target all

# 或转换指定机械臂/夹爪
/opt/conda_envs/py313/bin/python scripts/convert_glb_to_obj.py --target xarm7
/opt/conda_envs/py313/bin/python scripts/convert_glb_to_obj.py --target gripper_g2
```

转换器全自动处理：
- Google Draco 压缩网格解码；
- glTF Y-up 到 MuJoCo Z-up 坐标系对齐；
- 基于材质属性的自动化子网格分类与合并；
- 三角面绕序自愈与朝外判定（`orient_mesh_outward`），彻底避免 MuJoCo 默认背面剔除下的破面现象；
- 完整保留平滑法线与硬边特征。

## 模型规范

- 运动学参数与惯量张量严格继承自 UFACTORY 官方定义（[xarm_ros](https://github.com/xArm-Developer/xarm_ros) / [ufactory_genesis](https://github.com/xArm-Developer/ufactory_genesis)）。
- 执行器增益与动力学配置参照 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 规范。
- 视觉几何（`group="2"`, `contype="0" conaffinity="0"`）与碰撞几何（`group="3"`）严格分层解耦。

## 开源协议

MIT

