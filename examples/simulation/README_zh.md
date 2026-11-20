# 仿真示例 (`examples/simulation`)

[English](README.md) | [中文](README_zh.md)

示例只需核心安装（见[安装说明](../../README_zh.md#安装)）。下面的命令都会打开 3D 视窗；macOS 上请用 `mjpython` 代替 `python` 启动。加 `--headless` 可不开视窗运行（例如在 CI 或服务器上），脚本改为打印检查结果，`01` 则只打印模型摘要。每个脚本的参数都可以用 `--help` 查看。

## 示例

| 脚本 | 演示内容 | 模型 |
| :--- | :--- | :--- |
| `01_load_and_view.py` | 在视窗中查看任意模型 | 全部 |
| `02_joint_control.py` | 五次多项式关节运动；裸臂 `xarm6` 运行动力学校验的 20 个姿态 | 含机械臂的模型 |
| `03_cartesian_ik.py` | 用 DLS 逆解跟踪直线和圆 | `xarm6` |
| `04_gripper_control.py` | 闭合、张开、半开；Gripper G2 与 Bio Gripper G2 支持 `--speed-percent` 和 `--force-percent` | 带可动夹爪的模型 |
| `05_pick_place.py` | 带预检的方块抓放 | `--robot`，五款机械臂 |
| `06_trajectory_planning.py` | 点到点五次多项式、C2 样条、笛卡尔直线、前馈力矩 | `xarm6` |
| `07_kinematics.py` | 正运动学、雅可比与有限差分对照、速度级逆解、可操作度 | `xarm6` |
| `08_dynamics.py` | 动力学检查，然后监测 home → ready → target → ready 运动 | `--robot`，五款裸臂 |
| `09_obstacle_avoidance.py` | RRT-Connect 绕过立柱和上方横板 | `xarm6` |
| `10_box_pack.py` | 把方块投进盒子 | `--robot`，五款机械臂 |
| `11_force_torque_sensor.py` | AI1500 六维力/力矩读数（200 Hz） | 含 `ft_sensor` 的模型 |

## 运行

```bash
python examples/simulation/01_load_and_view.py --model xarm6_with_gripper
python examples/simulation/02_joint_control.py --model lite6
python examples/simulation/03_cartesian_ik.py
python examples/simulation/04_gripper_control.py --model xarm6_with_bio_gripper --speed-percent 50
python examples/simulation/05_pick_place.py --robot lite6
python examples/simulation/06_trajectory_planning.py
python examples/simulation/07_kinematics.py
python examples/simulation/08_dynamics.py --robot uf850
python examples/simulation/09_obstacle_avoidance.py
python examples/simulation/10_box_pack.py --robot xarm7 --cycles 3
python examples/simulation/11_force_torque_sensor.py --model xarm6_with_ft_sensor_and_vacuum_gripper
```

- `05`、`08`、`10` 与 `ufactory-mujoco pick-place`、`dynamics`、`box-pack` 运行同一套代码。`--robot` 选择机械臂（`lite6`、`xarm5`、`xarm6`、`xarm7`、`uf850`）；`05` 与 `10` 给 `lite6` 装 Lite 6 夹爪，给其余机械臂装 Gripper G2。`--dry-run` 只做预检，`--report <文件>` 输出 JSON 报告。
- 对齐 `ufactory_genesis` 的静态力矩验证有单独的命令：仿真用 `dynamics-sim-check`，真机用 `dynamics-hardware-check`，比较两份报告用 `dynamics-report-compare`。

## 模型

`--model` 接受模型名或 MJCF 路径，`ufactory-mujoco export-model --list` 列出全部模型名。组合模型命名为 `<机械臂>_with_<部件>`：

| 机械臂 | `<部件>` |
| :--- | :--- |
| `xarm5`、`xarm6`、`xarm7`、`uf850` | `gripper`（Gripper G2）、`bio_gripper`、`vacuum_gripper`、`d435`、`gripper_and_d435`、`ft_sensor`、`ft_sensor_and_gripper`、`ft_sensor_and_vacuum_gripper`、`base_plate` |
| 仅 `xarm6` | `gripper_and_d405`、`gripper_and_gemini_305`、`gripper_and_zed_x_nano` |
| `lite6` | `gripper`（Lite 6 夹爪）、`vacuum_gripper`、`base_plate` |

独立部件可以在 `01` 中查看：夹爪 `gripper_g2`、`bio_gripper_g2`、`lite6_gripper`、`vacuum_gripper`、`lite6_vacuum_gripper`；相机 `d405`、`d435`、`gemini_305`、`zed_x_nano`；`camera_stand` 与 `camera_stand_with_<相机>`；`ft_sensor`；安装底板 `xarm_base_plate`、`uf850_base_plate`、`lite6_base_plate`。`04` 还支持三款独立的可动夹爪，`11` 还支持 `ft_sensor`。脚本遇到不支持的模型时以退出码 `2` 退出。

## 起始姿态

- `home`：所有模型都是关节全零。
- `ready`：工具竖直朝下。装有工具、相机或力矩传感器的 xArm 与 UF850 模型，`ready` 让 Gripper G2 的 TCP 位于 `(300, 0, 300)` mm，其他工具沿用同一组关节角。Lite 6 模型用出厂位姿 `[0, 9.9, 31.8, 0, 21.9, 0]°`。xArm 与 UF850 裸臂（包括装了底板的）的 `ready` 仍是全零。
- 装有工具、相机或力矩传感器的 xArm 5/6/7 模型另有 `init`，与 `ready` 相同，因为在全零姿态下其中很多工具会伸到地面以下。
- 示例在模型有 `init` 时从 `init` 起步，否则从 `home` 起步；`01 --pose` 可以选其他关键帧。`05` 与 `10` 让每款机械臂的 TCP 从 `(300, 0, 300)` mm 竖直朝下起步。在 `05` 中这个位置正好在方块上方，五款机械臂都把方块从 `(300, 0)` 搬到 `(300, 300)` mm。
- 独立的可动夹爪有 `open` 与 `close` 关键帧，其他独立部件没有关键帧。

## 真机

真机模式需要 `pip install xarm-python-sdk` 和从机械臂导出的标定文件；标定文件的序列号或机型与所连控制器不一致时拒绝运行。请把 IP 换成自己控制器的地址。

```bash
# 1. 导出本机标定，写入 calibrations/<arm>_kinematics_<SN>.yaml
ufactory-mujoco export-calibration 192.168.1.60 --output-dir calibrations

# 2. 只读检查：机械臂不动
ufactory-mujoco kinematics-check --robot xarm6 --ip 192.168.1.60 --calibration calibrations/xarm6_kinematics_<SN>.yaml
python examples/simulation/08_dynamics.py --robot xarm6 --mode real-read-only --ip 192.168.1.60 --calibration calibrations/xarm6_kinematics_<SN>.yaml
python examples/simulation/11_force_torque_sensor.py --ip 192.168.1.60 --robot xarm6 --gripper ft_sensor

# 3. 在控制器内置仿真中重放：机械臂不动
python examples/simulation/05_pick_place.py --robot xarm6 --mode sdk-sim --ip 192.168.1.60 --calibration calibrations/xarm6_kinematics_<SN>.yaml

# 4. 真机运动：清空工作区，确保急停可用
python examples/simulation/05_pick_place.py --robot xarm6 --mode real --confirm-real --ip 192.168.1.60 --calibration calibrations/xarm6_kinematics_<SN>.yaml
dynamics-hardware-check --robot xarm6 --ip 192.168.1.60 --confirm-real --require-reference
```

- `--mode real` 只运行一个周期，并且要同时满足三个条件才会启动：传入 `--confirm-real`、控制器没有报错、从实测关节角出发的预检通过。`05` 与 `10` 还会驱动夹爪开合。
- 真机动力学检查把关节速度限制在 40 °/s 以内。导出标定时被跳过的机械臂（老序列号、xArm 5）用 `--nominal-model` 运行这些检查。
- 装有工具时，给 `08` 传 `--gripper <名称>`，让预检把工具考虑进去。

## 视窗操作

在视窗中按 `F1` 查看完整列表。

- **左键拖拽**：旋转。**右键拖拽**：平移。**滚轮**：缩放。
- **双击**：选中物体。**Ctrl + 左键 / 右键拖拽**：对选中物体施加扭矩 / 力。
- **空格**：暂停。**Backspace**：重置。两者只在 `01` 中有效，其余示例由脚本自己步进物理。
- **C** / **F**：显示接触点 / 接触力。**0–5**：切换几何组（`2` 为视觉网格，`3` 为碰撞网格）。在右侧面板打开 Site 组 `4` 可以显示安装基准和螺钉孔。
- **Tab** / **Shift + Tab**：显示或隐藏左 / 右面板。
