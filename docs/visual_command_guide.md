# UFACTORY MuJoCo 可视化指令指南 (Visual Command Guide)

本文档按产品型号（Lite 6、xArm 系列、UF850）与工作模式（静态查看 / 可动仿真）分类整理所有启动与控制指令。所有仿真示例脚本均位于 `examples/simulation/` 目录下（下文命令已含完整路径，速查表中省略路径前缀）；强化学习示例位于 `examples/rl/`。

---

## 1. 全景速查矩阵

| 产品型号 | 机械臂本体 (静态) | 机械臂本体 (关节运动) | 配件 (静态) | 配件 (可动) | 机械臂 + 配件 (静态) | 机械臂 + 配件 (可动) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Lite 6** | `01_load_and_view.py --model lite6` | `02_joint_control.py --model lite6` | `01_load_and_view.py --model lite6_gripper` | `04_gripper_control.py --model lite6_gripper` | `01_load_and_view.py --model lite6_with_gripper` | `04_gripper_control.py --model lite6_with_gripper` |
| **xArm 6** | `01_load_and_view.py --model xarm6` | `02_joint_control.py --model xarm6` | `01_load_and_view.py --model gripper_g2` | `04_gripper_control.py --model gripper_g2` | `01_load_and_view.py --model xarm6_with_gripper` | `04_gripper_control.py --model xarm6_with_gripper` |
| **xArm 7** | `01_load_and_view.py --model xarm7` | `02_joint_control.py --model xarm7` | *(通用配件)* | *(通用配件)* | — | — |
| **xArm 5** | `01_load_and_view.py --model xarm5` | `02_joint_control.py --model xarm5` | *(通用配件)* | *(通用配件)* | — | — |
| **UF 850** | `01_load_and_view.py --model uf850` | `02_joint_control.py --model uf850` | *(通用配件)* | *(通用配件)* | — | — |

> **提示**：所有命令默认打开 MuJoCo 交互视窗；添加 `--headless` 参数可用于无头服务器执行与自动化测试。

---

## 2. 按产品型号启动指令

### 2.1 Lite 6 系列

`ready` 使用 Studio 出厂关节角度 `[0, 9.9, 31.8, 0, 21.9, 0]`（度），`home` 为全零关节角度。
该准备姿态的本体法兰位置约为 `[200, 0, 200]` mm，姿态约为 `[180, 0, 0]` 度；组合模型的 TCP 位置包含夹爪或吸盘的工具偏移。

#### (1) 机械臂本体
- **静态查看**：
  ```bash
  # 默认就绪姿态 (Ready)
  python examples/simulation/01_load_and_view.py --model lite6 --pose ready

  # 零位姿态 (Home)
  python examples/simulation/01_load_and_view.py --model lite6 --pose home
  ```
- **关节运动仿真**（Ready ↔ Demo 轨迹插值）：
  ```bash
  python examples/simulation/02_joint_control.py --model lite6
  ```

#### (2) 独立配件
- **Lite 6 夹爪 (静态)**：
  ```bash
  python examples/simulation/01_load_and_view.py --model lite6_gripper
  ```
- **Lite 6 夹爪 (可动开闭演示)**：
  ```bash
  python examples/simulation/04_gripper_control.py --model lite6_gripper
  ```
- **Lite 6 真空吸盘 (静态)**：
  ```bash
  python examples/simulation/01_load_and_view.py --model lite6_vacuum_gripper
  ```

#### (3) 机械臂 + 配件组合
- **Lite 6 + 夹爪 (静态)**：
  ```bash
  python examples/simulation/01_load_and_view.py --model lite6_with_gripper --pose ready
  ```
- **Lite 6 + 夹爪 (可动抓取循环)**：
  ```bash
  python examples/simulation/04_gripper_control.py --model lite6_with_gripper
  ```
- **Lite 6 + 夹爪 (关节轨迹运动)**：
  ```bash
  python examples/simulation/02_joint_control.py --model lite6_with_gripper
  ```
- **Lite 6 + 真空吸盘 (静态)**：
  ```bash
  python examples/simulation/01_load_and_view.py --model lite6_with_vacuum_gripper --pose ready
  ```
- **Lite 6 + 真空吸盘 (关节轨迹运动)**：
  ```bash
  python examples/simulation/02_joint_control.py --model lite6_with_vacuum_gripper
  ```

---

### 2.2 xArm 6 系列

裸臂关节示例默认运行 Genesis 动力学验证的 20 个姿态（两组往返），峰值目标速度 90°/s。`--speed 120` 加快运动；`--motion demo --duration 2.5` 使用原动作。默认关闭阴影和反射，`--render-quality full` 恢复；状态同步默认 30 Hz（`--sync-hz`），不限制原生查看器绘制帧率。`--headless` 完全关闭绘制。

#### (1) 机械臂本体
- **静态查看**：
  ```bash
  python examples/simulation/01_load_and_view.py --model xarm6 --pose ready
  ```
- **关节空间插值控制**：
  ```bash
  python examples/simulation/02_joint_control.py --model xarm6
  ```
- **笛卡尔空间逆运动学 (IK) 轨迹跟踪**：
  ```bash
  # 矩形末端轨迹跟踪
  python examples/simulation/03_cartesian_ik.py
  ```
- **运动学工具链（FK / 雅可比 / 速度级逆解 / 奇异性指标）**：
  ```bash
  python examples/simulation/07_kinematics.py
  python examples/simulation/07_kinematics.py --headless
  ```
- **动力学工具链（质量矩阵 / 正逆动力学 / 负载 / 前馈包络）**：
  ```bash
  python examples/simulation/08_dynamics.py
  python examples/simulation/08_dynamics.py --headless
  ```
- **避障轨迹规划任务（立柱 + 横板场景）**：
  ```bash
  python examples/simulation/09_obstacle_avoidance.py
  python examples/simulation/09_obstacle_avoidance.py --headless
  ```

#### (2) 独立配件
- **Gripper G2 (静态)**：
  ```bash
  python examples/simulation/01_load_and_view.py --model gripper_g2
  ```
- **Gripper G2 (可动连杆开闭控制)**：
  ```bash
  python examples/simulation/04_gripper_control.py --model gripper_g2
  ```
- **Bio Gripper G2 (静态)**：
  ```bash
  python examples/simulation/01_load_and_view.py --model bio_gripper_g2
  ```
- **Bio Gripper G2 (可动双滑块开闭控制)**：
  ```bash
  python examples/simulation/04_gripper_control.py --model bio_gripper_g2
  ```

#### (3) 机械臂 + 配件组合
- **xArm 6 + Gripper G2 (静态)**：
  ```bash
  python examples/simulation/01_load_and_view.py --model xarm6_with_gripper --pose ready
  ```
- **xArm 6 + Gripper G2 (可动控制)**：
  ```bash
  python examples/simulation/04_gripper_control.py --model xarm6_with_gripper
  ```
- **xArm 6 + Gripper G2 (方块抓放任务)**：
  ```bash
  python examples/simulation/05_pick_place.py
  ```
- **xArm 6 + Gripper G2 (关节轨迹运动)**：
  ```bash
  python examples/simulation/02_joint_control.py --model xarm6_with_gripper
  ```
- **xArm 6 + Bio Gripper G2 (静态)**：
  ```bash
  python examples/simulation/01_load_and_view.py --model xarm6_with_bio_gripper --pose ready
  ```
- **xArm 6 + Bio Gripper G2 (可动控制)**：
  ```bash
  python examples/simulation/04_gripper_control.py --model xarm6_with_bio_gripper
  ```
- **xArm 6 + Bio Gripper G2 (关节轨迹运动)**：
  ```bash
  python examples/simulation/02_joint_control.py --model xarm6_with_bio_gripper
  ```

---

### 2.3 xArm 7 系列

- **静态查看**：
  ```bash
  python examples/simulation/01_load_and_view.py --model xarm7 --pose ready
  ```
- **关节运动仿真**（7 自由度联动）：
  ```bash
  python examples/simulation/02_joint_control.py --model xarm7
  ```

---

### 2.4 xArm 5 系列

- **静态查看**：
  ```bash
  python examples/simulation/01_load_and_view.py --model xarm5 --pose ready
  ```
- **关节运动仿真**（5 自由度联动）：
  ```bash
  python examples/simulation/02_joint_control.py --model xarm5
  ```

---

### 2.5 UF 850 系列

- **静态查看**：
  ```bash
  python examples/simulation/01_load_and_view.py --model uf850 --pose ready
  ```
- **关节运动仿真**（6 自由度联动）：
  ```bash
  python examples/simulation/02_joint_control.py --model uf850
  ```

---

## 3. 通用参数说明

| 参数 | 适用脚本 | 说明 | 示例 |
| :--- | :--- | :--- | :--- |
| `--model <name>` | 所有示例 | 指定模型名称或自定义 xml 路径 | `--model xarm6_with_gripper` |
| `--pose <home\|ready>` | `01_load_and_view.py` | 指定初始加载关键帧姿态 | `--pose ready` |
| `--headless` | `02`–`06`, `09` | 无头模式（不启动视窗，纯终端输出精度指标） | `--headless` |
| `--seed <N>` | `09_obstacle_avoidance.py` | RRT/shortcut 随机种子（结果可复现） | `--seed 3` |
| `--safety-margin <m>` | `09_obstacle_avoidance.py` | 规划安全间隙（米，默认 0.002） | `--safety-margin 0.01` |
| `--cycles <N>` | `04_gripper_control.py` | 夹爪开闭测试循环次数（默认 3） | `--cycles 5` |
| `--cycles <N>` | `05_pick_place.py` | 抓放循环次数（默认 1，第二次起从当前方块位置抓取） | `--cycles 2` |

---

## 4. MuJoCo 视窗交互说明

在交互视窗中，你可以使用以下按键与鼠标交互：

- **鼠标左键拖拽**：旋转视角
- **鼠标右键拖拽**：缩放视角
- **双击鼠标中键**：重置视线焦点至点击点
- **Ctrl + 鼠标右键**：对连杆施加外力拖拽
- **空格键 (Space)**：暂停 / 继续物理仿真
- **Backspace**：重置仿真状态到初始姿态
- **Ctrl + A**：显示 / 隐藏物理接触点与法向量
- **Ctrl + C**：显示 / 隐藏碰撞几何体

