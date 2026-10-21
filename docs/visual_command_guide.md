# UFACTORY MuJoCo 可视化指令指南 (Visual Command Guide)

本文档按产品型号（Lite 6、xArm 系列、UF850）与工作模式（静态查看 / 可动仿真）分类整理所有启动与控制指令。所有仿真示例脚本均位于 `examples/simulation/` 目录下（下文命令已含完整路径，速查表中省略路径前缀）；强化学习示例位于 `examples/rl/`。


安装 wheel 后，抓放、装箱、动力学也可在任意目录使用统一入口（旧脚本继续兼容）：

```bash
ufactory-mujoco pick-place --robot xarm6 --headless
ufactory-mujoco box-pack --robot lite6 --cycles 1 --headless
ufactory-mujoco dynamics --robot all --samples 32 --seed 0 --headless --report reports/dynamics.json
```

抓放/装箱加 `--dry-run` 只预检；报告相对当前目录并自动建父目录。退出码 0/1/2 分别表示通过、运行失败、参数非法。安装与构建见 [README_zh.md](../README_zh.md#1-环境安装)。

---

## 1. 全景速查矩阵

| 产品型号 | 机械臂本体 (静态) | 机械臂本体 (关节运动) | 配件 (静态) | 配件 (可动) | 机械臂 + 配件 (静态) | 机械臂 + 配件 (可动/任务) | 强化学习 (Reach) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Lite 6** | `01_load_and_view.py --model lite6` | `02_joint_control.py --model lite6` | `01_load_and_view.py --model lite6_gripper` | `04_gripper_control.py --model lite6_gripper` | `01_load_and_view.py --model lite6_with_gripper` | `04_gripper_control.py --model lite6_with_gripper`<br>`05_pick_place.py --robot lite6`<br>`10_box_pack.py --robot lite6` | `examples/rl/reach/01_reach.py --model lite6` |
| **xArm 6** | `01_load_and_view.py --model xarm6` | `02_joint_control.py --model xarm6` | `01_load_and_view.py --model gripper_g2` | `04_gripper_control.py --model gripper_g2` | `01_load_and_view.py --model xarm6_with_gripper` | `04_gripper_control.py --model xarm6_with_gripper`<br>`05_pick_place.py --robot xarm6`<br>`10_box_pack.py --robot xarm6` | `examples/rl/reach/01_reach.py --model xarm6` |
| **xArm 7** | `01_load_and_view.py --model xarm7` | `02_joint_control.py --model xarm7` | *(通用 Gripper G2 / Bio G2)* | *(通用 Gripper G2 / Bio G2)* | `01_load_and_view.py --model xarm7_with_gripper` | `04_gripper_control.py --model xarm7_with_gripper`<br>`05_pick_place.py --robot xarm7`<br>`10_box_pack.py --robot xarm7` | `examples/rl/reach/01_reach.py --model xarm7` |
| **xArm 5** | `01_load_and_view.py --model xarm5` | `02_joint_control.py --model xarm5` | *(通用 Gripper G2 / Bio G2)* | *(通用 Gripper G2 / Bio G2)* | `01_load_and_view.py --model xarm5_with_gripper` | `04_gripper_control.py --model xarm5_with_gripper`<br>`05_pick_place.py --robot xarm5`<br>`10_box_pack.py --robot xarm5` | `examples/rl/reach/01_reach.py --model xarm5` |
| **UF 850** | `01_load_and_view.py --model uf850` | `02_joint_control.py --model uf850` | *(通用 Gripper G2 / Bio G2)* | *(通用 Gripper G2 / Bio G2)* | `01_load_and_view.py --model uf850_with_gripper` | `04_gripper_control.py --model uf850_with_gripper`<br>`05_pick_place.py --robot uf850`<br>`10_box_pack.py --robot uf850` | `examples/rl/reach/01_reach.py --model uf850` |

> **提示**：所有命令默认打开 MuJoCo 交互视窗；添加 `--headless` 参数可用于无头服务器执行与自动化测试，`05_pick_place.py` 与 `10_box_pack.py` 支持 `--dry-run` 仅运行轨迹预检。

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
- **Lite 6 + 夹爪 (方块抓放与装箱展示)**：
  ```bash
  python examples/simulation/05_pick_place.py --robot lite6
  python examples/simulation/10_box_pack.py --robot lite6 --cycles 3
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

> 关键帧约定：xArm 5/6/7、UF 850 本体及 Bio 夹爪组合的 `home` / `ready` 均为机械臂全零关节角度；Gripper G2 组合（`xarm6_with_gripper`）的 `home` / `ready` 均为 TCP 位于 `(300, 0, 300)` mm、姿态 `(180, 0, 0)` 度的关节角 `[0, -36.184, -47.736, 0, 83.920, 0]`（度），避免全零时夹爪插入地面；Lite 6 系列保持不变。夹爪关节保持原值。

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
- **动力学工具链（五臂验收 / 运动监测 / xArm6 负载回归）**：
  ```bash
  python examples/simulation/08_dynamics.py
  python examples/simulation/08_dynamics.py --headless
  python examples/simulation/08_dynamics.py --robot all --headless --report build/dynamics/batch6.json
  ```
  `--robot` 可选 lite6/xarm5/xarm6/xarm7/uf850/all，默认 xarm6；`--samples 32 --seed 0` 控制随机采样。无头模式同样执行约 8.4 秒运动，报告涵盖积分后的实际状态检查。无需 xArm SDK。
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
- **xArm 6 + Gripper G2 (方块抓放与装箱展示)**：
  ```bash
  python examples/simulation/05_pick_place.py --robot xarm6
  python examples/simulation/10_box_pack.py --robot xarm6 --cycles 3
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
  python examples/simulation/01_load_and_view.py --model xarm7_with_gripper --pose ready
  ```
- **关节运动仿真**（7 自由度联动）：
  ```bash
  python examples/simulation/02_joint_control.py --model xarm7
  python examples/simulation/02_joint_control.py --model xarm7_with_gripper
  ```
- **夹爪开闭、方块抓放与装箱展示**：
  ```bash
  python examples/simulation/04_gripper_control.py --model xarm7_with_gripper
  python examples/simulation/05_pick_place.py --robot xarm7
  python examples/simulation/10_box_pack.py --robot xarm7 --cycles 3
  ```

---

### 2.4 xArm 5 系列

- **静态查看**：
  ```bash
  python examples/simulation/01_load_and_view.py --model xarm5 --pose ready
  python examples/simulation/01_load_and_view.py --model xarm5_with_gripper --pose ready
  ```
- **关节运动仿真**（5 自由度联动）：
  ```bash
  python examples/simulation/02_joint_control.py --model xarm5
  python examples/simulation/02_joint_control.py --model xarm5_with_gripper
  ```
- **夹爪开闭、方块抓放与装箱展示**：
  ```bash
  python examples/simulation/04_gripper_control.py --model xarm5_with_gripper
  python examples/simulation/05_pick_place.py --robot xarm5
  python examples/simulation/10_box_pack.py --robot xarm5 --cycles 3
  ```

---

### 2.5 UF 850 系列

- **静态查看**：
  ```bash
  python examples/simulation/01_load_and_view.py --model uf850 --pose ready
  python examples/simulation/01_load_and_view.py --model uf850_with_gripper --pose ready
  ```
- **关节运动仿真**（6 自由度联动）：
  ```bash
  python examples/simulation/02_joint_control.py --model uf850
  python examples/simulation/02_joint_control.py --model uf850_with_gripper
  ```
- **夹爪开闭、方块抓放与装箱展示**：
  ```bash
  python examples/simulation/04_gripper_control.py --model uf850_with_gripper
  python examples/simulation/05_pick_place.py --robot uf850
  python examples/simulation/10_box_pack.py --robot uf850 --cycles 3
  ```

---

## 3. 通用参数说明

| 参数 | 适用脚本 | 说明 | 示例 |
| :--- | :--- | :--- | :--- |
| `--model <name>` | `01`, `02`, `04`, `01_reach` | 指定模型名称或自定义 xml 路径 | `--model xarm6_with_gripper` |
| `--robot <arm>` | `05_pick_place.py`, `10_box_pack.py` | 机械臂键名（自动关联对应任务 YAML） | `--robot lite6` |
| `--executor <mode>` | `05_pick_place.py`, `10_box_pack.py` | 轨迹执行器：`servo_cartesian` 或 `servo_j` | `--executor servo_j` |
| `--dry-run` | `05_pick_place.py`, `10_box_pack.py` | 仅做限位与碰撞预检并输出阶段状态，不步进物理 | `--dry-run` |
| `--pose <home\|ready>` | `01_load_and_view.py` | 指定初始加载关键帧姿态 | `--pose ready` |
| `--headless` | `02`–`10`, `01_reach` | 无头模式（不启动视窗，纯终端输出精度指标） | `--headless` |
| `--seed <N>` | `09_obstacle_avoidance.py`, `01_reach` | 随机种子（规划/评估结果可复现） | `--seed 3` |
| `--safety-margin <m>` | `09_obstacle_avoidance.py` | 规划安全间隙（米，默认 0.002） | `--safety-margin 0.01` |
| `--cycles <N>` | `04`, `05`, `10` | 循环执行次数（`04` 默认 3；`05`/`10` 默认 1） | `--cycles 3` |
| `--speed <x>` | `10_box_pack.py` | 运动速度倍率（`>1` 加快；自由落体不缩放） | `--speed 2.0` |

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

---

## 5. 批次 5 修复前问题复现（dev 工具）

```bash
# xArm5 长距离 deploy：修复前 move_tcp 路径 IK 不收敛（报错后视窗停留，关闭退出）
python dev/visual/box_pack/xarm5_deploy_prefix.py
# 对比：修复后 move_joints 到预解算构型，deploy 成功
python dev/visual/box_pack/xarm5_deploy_prefix.py --fixed

# Lite 6 掉块：修复前几何（宽供料台 + 浅抓取 + 远盒），运送中方块滑脱落在盒外
python dev/visual/box_pack/lite6_drop_prefix.py
# Lite 6 掉块：当前几何 + servo_j 关节空间运送（等价 --executor servo_j）
python dev/visual/box_pack/lite6_drop_prefix.py --servo-j
```

