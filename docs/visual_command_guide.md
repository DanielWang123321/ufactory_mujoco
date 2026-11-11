# UFACTORY MuJoCo 各模型启动命令（中文）

按机型列出 `examples/simulation/` 与 `examples/rl/` 的启动命令（速查表省略 `examples/simulation/` 前缀）。安装、`ufactory-mujoco` 命令行任务与退出码见 [README_zh.md](../README_zh.md)；完整参数表见 [examples/simulation/README_zh.md](../examples/simulation/README_zh.md#3-命令行参数汇总)。

所有命令默认打开交互视窗；加 `--headless` 可无头运行，`05_pick_place.py` 与 `10_box_pack.py` 加 `--dry-run` 只做预检。

## 1. 速查矩阵

| 机型 | 机械臂本体（静态） | 机械臂本体（关节运动） | 配件（静态） | 配件（可动） | 机械臂 + 配件（静态） | 机械臂 + 配件（可动/任务） | 强化学习 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Lite 6** | `01_load_and_view.py --model lite6` | `02_joint_control.py --model lite6` | `01_load_and_view.py --model lite6_gripper` | `04_gripper_control.py --model lite6_gripper` | `01_load_and_view.py --model lite6_with_gripper` | `04_gripper_control.py --model lite6_with_gripper`<br>`05_pick_place.py --robot lite6`<br>`10_box_pack.py --robot lite6` | `examples/rl/reach/01_reach.py --model lite6` |
| **xArm 6** | `01_load_and_view.py --model xarm6` | `02_joint_control.py --model xarm6` | `01_load_and_view.py --model gripper_g2` | `04_gripper_control.py --model gripper_g2` | `01_load_and_view.py --model xarm6_with_gripper` | `04_gripper_control.py --model xarm6_with_gripper`<br>`05_pick_place.py --robot xarm6`<br>`10_box_pack.py --robot xarm6` | `examples/rl/reach/01_reach.py --model xarm6`<br>`examples/rl/pick_place/01_pick_place.py` |
| **xArm 7** | `01_load_and_view.py --model xarm7` | `02_joint_control.py --model xarm7` | *(Gripper G2，同上)* | *(Gripper G2，同上)* | `01_load_and_view.py --model xarm7_with_gripper` | `04_gripper_control.py --model xarm7_with_gripper`<br>`05_pick_place.py --robot xarm7`<br>`10_box_pack.py --robot xarm7` | `examples/rl/reach/01_reach.py --model xarm7` |
| **xArm 5** | `01_load_and_view.py --model xarm5` | `02_joint_control.py --model xarm5` | *(Gripper G2，同上)* | *(Gripper G2，同上)* | `01_load_and_view.py --model xarm5_with_gripper` | `04_gripper_control.py --model xarm5_with_gripper`<br>`05_pick_place.py --robot xarm5`<br>`10_box_pack.py --robot xarm5` | `examples/rl/reach/01_reach.py --model xarm5` |
| **UF 850** | `01_load_and_view.py --model uf850` | `02_joint_control.py --model uf850` | *(Gripper G2，同上)* | *(Gripper G2，同上)* | `01_load_and_view.py --model uf850_with_gripper` | `04_gripper_control.py --model uf850_with_gripper`<br>`05_pick_place.py --robot uf850`<br>`10_box_pack.py --robot uf850` | `examples/rl/reach/01_reach.py --model uf850` |

强化学习示例在没有训练权重时可用 `--random`（到达）或 `--expert`（抓放）运行。

## 2. 按机型的命令

### 2.1 Lite 6

`ready` 为 Studio 出厂关节角 `[0, 9.9, 31.8, 0, 21.9, 0]`°（法兰约在 `[200, 0, 200]` mm，姿态 `[180, 0, 0]`°），`home` 为全零关节角。组合模型的 TCP 包含夹爪或吸盘的工具偏移。

```bash
# 机械臂本体：静态查看与关节运动
python examples/simulation/01_load_and_view.py --model lite6 --pose ready
python examples/simulation/02_joint_control.py --model lite6

# 独立配件：Lite 6 夹爪（静态、开合）与真空吸盘（静态）
python examples/simulation/01_load_and_view.py --model lite6_gripper
python examples/simulation/04_gripper_control.py --model lite6_gripper
python examples/simulation/01_load_and_view.py --model lite6_vacuum_gripper

# 机械臂 + 夹爪：静态、开合、关节运动、抓放、装箱
python examples/simulation/01_load_and_view.py --model lite6_with_gripper --pose ready
python examples/simulation/04_gripper_control.py --model lite6_with_gripper
python examples/simulation/02_joint_control.py --model lite6_with_gripper
python examples/simulation/05_pick_place.py --robot lite6
python examples/simulation/10_box_pack.py --robot lite6 --cycles 3

# 机械臂 + 真空吸盘：静态模型，没有开合
python examples/simulation/01_load_and_view.py --model lite6_with_vacuum_gripper --pose ready
python examples/simulation/02_joint_control.py --model lite6_with_vacuum_gripper
```

### 2.2 xArm 6

裸臂与 Bio Gripper G2 组合的 `home` / `ready` 为全零关节角。Gripper G2 组合抬高机械臂以免夹爪插入地面：`xarm6_with_gripper` 的 `home` / `ready` 为 TCP 位于 `(300, 0, 300)` mm、姿态 `(180, 0, 0)`° 时的关节角 `[0, -36.184, -47.736, 0, 83.920, 0]`°；xArm 5/7 与 UF 850 组合的关键帧见各自的 MJCF。

裸臂 `02_joint_control.py` 默认运行 20 个动力学验证姿态（两组往返，峰值 90°/s）；`--speed 120` 加快，`--motion demo --duration 2.5` 改用演示动作，`--render-quality full` 恢复阴影和反射。

```bash
# 机械臂本体：静态查看、关节运动、笛卡尔逆解（直线 + 圆）
python examples/simulation/01_load_and_view.py --model xarm6 --pose ready
python examples/simulation/02_joint_control.py --model xarm6
python examples/simulation/03_cartesian_ik.py

# 运动学（FK、雅可比、速度级逆解、奇异性指标）与轨迹规划
python examples/simulation/07_kinematics.py
python examples/simulation/06_trajectory_planning.py

# 动力学验收（--robot 可选 lite6/xarm5/xarm6/xarm7/uf850/all，无需 xArm SDK）
python examples/simulation/08_dynamics.py
python examples/simulation/08_dynamics.py --robot all --headless --report build/dynamics/report.json

# 避障规划（立柱 + 横板场景）
python examples/simulation/09_obstacle_avoidance.py

# 独立配件：Gripper G2（连杆开合与速度/力控制）、Bio Gripper G2（双滑块开合）
python examples/simulation/01_load_and_view.py --model gripper_g2
python examples/simulation/04_gripper_control.py --model gripper_g2 --speed-percent 80 --force-percent 75
python examples/simulation/01_load_and_view.py --model bio_gripper_g2
python examples/simulation/04_gripper_control.py --model bio_gripper_g2

# 机械臂 + Gripper G2：静态、开合、关节运动、抓放、装箱
python examples/simulation/01_load_and_view.py --model xarm6_with_gripper --pose ready
python examples/simulation/04_gripper_control.py --model xarm6_with_gripper
python examples/simulation/02_joint_control.py --model xarm6_with_gripper
python examples/simulation/05_pick_place.py --robot xarm6
python examples/simulation/10_box_pack.py --robot xarm6 --cycles 3

# 机械臂 + Bio Gripper G2：静态、开合、关节运动
python examples/simulation/01_load_and_view.py --model xarm6_with_bio_gripper --pose ready
python examples/simulation/04_gripper_control.py --model xarm6_with_bio_gripper
python examples/simulation/02_joint_control.py --model xarm6_with_bio_gripper

# 强化学习抓放（训练好的 PPO 策略，或 --expert 脚本专家）
python examples/rl/pick_place/01_pick_place.py --expert

# 控制器内置仿真 sdk-sim：物理机械臂不动，需本机标定（见仿真示例文档）
python examples/simulation/05_pick_place.py --robot xarm6 --mode sdk-sim --ip 192.168.1.60 --calibration calibrations/xarm6_kinematics_<SN>.yaml
```

### 2.3 xArm 7、xArm 5、UF 850

三款机型命令相同，将 `xarm7` 换成 `xarm5` 或 `uf850` 即可：

```bash
# 机械臂本体与 Gripper G2 组合：静态查看
python examples/simulation/01_load_and_view.py --model xarm7 --pose ready
python examples/simulation/01_load_and_view.py --model xarm7_with_gripper --pose ready

# 关节运动
python examples/simulation/02_joint_control.py --model xarm7
python examples/simulation/02_joint_control.py --model xarm7_with_gripper

# 夹爪开合、抓放、装箱
python examples/simulation/04_gripper_control.py --model xarm7_with_gripper
python examples/simulation/05_pick_place.py --robot xarm7
python examples/simulation/10_box_pack.py --robot xarm7 --cycles 3
```

### 2.4 深度相机与相机支架

提供 RealSense D405 / D435、奥比中光 Gemini 305、Stereolabs ZED X Nano 4 款深度相机，以及 UFACTORY 相机支架（`camera_stand`）、支架组合与 `xarm6` + Gripper G2 + 支架 + 相机组合（`mass=0`，不含光学成像传感器；在查看器右侧面板开启 Site Group `4` 可显示安装面与螺钉孔位）：

```bash
# 独立深度相机与独立相机支架（静态预览）
python examples/simulation/01_load_and_view.py --model d405
python examples/simulation/01_load_and_view.py --model d435
python examples/simulation/01_load_and_view.py --model gemini_305
python examples/simulation/01_load_and_view.py --model zed_x_nano
python examples/simulation/01_load_and_view.py --model camera_stand

# 相机支架 + 深度相机组合（静态预览）
python examples/simulation/01_load_and_view.py --model camera_stand_with_d405
python examples/simulation/01_load_and_view.py --model camera_stand_with_d435
python examples/simulation/01_load_and_view.py --model camera_stand_with_gemini_305
python examples/simulation/01_load_and_view.py --model camera_stand_with_zed_x_nano

# xArm 6 + Gripper G2 + 相机支架 + 深度相机（静态预览、关节运动、夹爪开合）
python examples/simulation/01_load_and_view.py --model xarm6_with_gripper_and_d405 --pose ready
python examples/simulation/01_load_and_view.py --model xarm6_with_gripper_and_d435 --pose ready
python examples/simulation/01_load_and_view.py --model xarm6_with_gripper_and_gemini_305 --pose ready
python examples/simulation/01_load_and_view.py --model xarm6_with_gripper_and_zed_x_nano --pose ready
python examples/simulation/02_joint_control.py --model xarm6_with_gripper_and_d405
python examples/simulation/04_gripper_control.py --model xarm6_with_gripper_and_d405
```

## 3. 视窗操作

以下按 MuJoCo 查看器的内置帮助（`F1`）整理：

- **左键拖拽**：旋转视角；**右键拖拽**：平移视角；**滚轮或中键拖拽**：缩放。
- **双击**：选中物体；**右键双击**：以点击处为视角中心；**Ctrl + 右键双击**：跟踪选中物体。
- **Ctrl + 左键拖拽 / Ctrl + 右键拖拽**：对选中物体施加扭矩 / 力。
- **空格**：暂停/继续；**Backspace**：重置。两者只在 `01_load_and_view.py` 中控制物理，其余示例由脚本自己步进物理。
- **C**：显示接触点；**F**：显示接触力；**0–5**：切换几何组（`2` 为视觉网格，`3` 为碰撞网格）。
- **Ctrl + A**：对齐视角；**Tab / Shift + Tab**：显示/隐藏左右面板；**F1**：帮助。
