# 批次 9：真机 `XArmRobot` 与安全确认门

## 目标与边界

对照 `ufactory_genesis` 0.3.2 与 [todolist.md](../../todolist.md) 批次 9：
1. **`XArmRobot` 实现 `RobotInterface`**：管理 xArm SDK 会话生命周期（`connect`/`disconnect`/`__enter__`/`__exit__`），默认只读（`read_only=True`），仅在持有通过全部安全检查签发的 `MotionApproval` 令牌后解锁运动与夹爪指令。
2. **单机运动学标定与 FK/IK 固件比对**：从控制器提取关节相对位姿标定（`scripts/gen_kinematics_params.py` → `calibrations/<robot>_kinematics_<suffix>.yaml`，含 `schema_version: 1`、序列号与规范化 `sha256`），修补本地 MuJoCo 模型并与控制器固件 `get_forward_kinematics` / `get_inverse_kinematics` 只读比对（位置误差 `< 1.0 mm`、姿态误差 `< 0.5°`）。
3. **安全确认门与单周期上限**：连接真机必须提供标定文件且序列号、`sha256`、控制器错误/警告码、自由度全部通过；`sdk-sim` 与 `real` 强制单周期硬上限（`cycles == 1`）；未显式传入 `--confirm-real` 时拒绝一切真机运动。
4. **真机动力学检查沿用确认门**：`ufactory-mujoco dynamics`（`--mode {sim,real-read-only,sdk-sim,real}`）、`scripts/compare_torque_xarm6.py` 与 `scripts/compare_dynamics_xarm6.py` 接入同一安全门；`--no-move` / `real-read-only` / `--replay` 严格零运动调用。
5. **`sdk-sim` 控制器内部仿真**：启用 `set_simulation_robot(True)` 在控制器内部虚拟机械臂重放轨迹，校核关节跟踪误差、末端位置误差与碰撞裕度，`finally` 恢复非仿真模式并验证物理编码器关节漂移 `<= 1e-3 rad`。

## 实施方案

1. **标定模块（`ufactory_mujoco/calibration.py`、`scripts/gen_kinematics_params.py`）**：
   - `KinematicsCalibration` 校验 `schema_version == 1`、`robot_key`、`serial_number`、`joints`（`joint1..jointN` 的 `x, y, z, roll, pitch, yaw`）及规范化 JSON `sha256`。
   - `apply_calibration_to_model` 将单机标定写入 MuJoCo `body_pos` / `body_quat` 并调用 `mj_forward`。
   - `pose_flange_to_tcp` / `pose_tcp_to_flange` 完成 MuJoCo `attachment_site`（法兰，单位 `m` + 四元数 `wxyz`）与控制器 TCP（含 `tcp_offset`，单位 `mm` + 外旋 `roll/pitch/yaw rad`）的双向变换。
2. **安全门与 `sdk-sim`（`ufactory_mujoco/safety.py`）**：
   - `verify_hardware_identity_and_calibration` 校验控制器连接、错误码/警告码（均为 `0`）、自由度、控制器 `get_robot_sn()` 与标定 YAML 的序列号及 `sha256`。
   - `enforce_single_cycle` 在 `sdk-sim` 与 `real` 模式强制 `cycles == 1`。
   - `authorize_real_motion` 在标定、预检与 `--confirm-real` 全部通过后签发不可伪造的 `MotionApproval` 令牌。
   - `run_sdk_simulation` 记录切换前物理关节角 `q_physical_before`，开启 `set_simulation_robot(True)` 重放速度受限轨迹并采集固件虚拟反馈，`finally` 强制恢复 `set_simulation_robot(False)` 与 `mode=0, state=0`，核验 `physical_joint_drift_rad <= 1e-3 rad`。
3. **真机接口与任务集成（`ufactory_mujoco/xarm_robot.py`、`kinematics_validation.py`、`tasks/`）**：
   - `XArmRobot` 实现 `RobotInterface` 全部方法；只读观测（`get_joint_positions`、`get_joint_states`、`get_tcp_pose`、`get_gripper_position`、`solve_tcp`）零运动调用；未调用 `unlock_motion(approval)` 时所有运动/夹爪方法抛出 `SafetyGateError`。
   - `pick-place`、`box-pack` 支持 `--mode {sim,dry-run,sdk-sim,real}`、`--ip`、`--calibration`、`--confirm-real`；`dynamics` 支持 `--mode {sim,real-read-only,sdk-sim,real}`。

## 验收结果（2026-10-28）

实机环境：`xArm 6`（IP `192.168.1.60`，序列号 `XI130505L56A01`，固件 `v2.8.0`，末端挂载 Gripper G2 + D405 相机，`tcp_load = 0.82 kg @ [0, 0, 48] mm`，`tcp_offset = [0, 0, 172, 0, 0, 0]`）。全流程在物理机械臂保持静止下完成验收：

1. **单机标定提取与 FK/IK 控制器比对**：
   - `python scripts/gen_kinematics_params.py 192.168.1.60` → 生成 `calibrations/xarm6_kinematics_L56A01.yaml`（`sha256: 374295277732b8c9dd13c24c7937518f6c659a9aeffd7cf8a2ca2a51071580b9`）。
   - `python -m ufactory_mujoco.kinematics_validation --robot xarm6 --ip 192.168.1.60 --calibration calibrations/xarm6_kinematics_L56A01.yaml --report build/hardware/fk_ik_xarm6.json`：
     - 5 组姿态（含实机当前姿态）FK 最大位置误差 `0.0013 mm`（阈值 `< 1.0 mm`），最大姿态误差 `0.0002°`（阈值 `< 0.5°`）。
     - 控制器 IK 往返最大末端位置误差 `0.1501 mm`（阈值 `< 1.0 mm`），最大姿态误差 `0.0001°`。
2. **只读静力学与动力学观测（零运动指令）**：
   - `python scripts/compare_torque_xarm6.py --no-move --calibration calibrations/xarm6_kinematics_L56A01.yaml --report build/hardware/torque_readonly.json`：当前姿态 TCP 位置误差 `0.001 mm`，六轴重力力矩最大绝对差 `0.637 Nm`（`0.0%–4.0%`）。
   - `python -m ufactory_mujoco dynamics --robot xarm6 --mode real-read-only --ip 192.168.1.60 --calibration calibrations/xarm6_kinematics_L56A01.yaml --report build/hardware/dynamics_readonly.json`：通过（4201 步预检零穿透/零超限，静态力矩差 `0.637 Nm`）。
3. **`sdk-sim` 控制器内部仿真验收（物理关节静止）**：
   - `pick-place`（`build/hardware/sdk_sim_pick_place.json`）：410 步重放通过，最大关节跟踪误差 `0.092 mrad`，最大末端误差 `0.042 mm`，最小碰撞距离 `+12.97 mm`，物理关节漂移 `0.002 mrad`（`<= 1.0 mrad`）。
   - `box-pack`（`build/hardware/sdk_sim_box_pack.json`）：626 步重放通过，最大关节跟踪误差 `0.096 mrad`，最大末端误差 `0.037 mm`，最小碰撞距离 `+12.97 mm`，物理关节漂移 `0.002 mrad`。
   - `dynamics`（`build/hardware/sdk_sim_dynamics.json`）：260 步重放通过，最大关节跟踪误差 `0.078 mrad`，最大末端误差 `0.060 mm`，最小碰撞距离 `+21.02 mm`，物理关节漂移 `0.002 mrad`。
4. **安全门阻断与单元测试**：
   - 缺失 `--calibration`、缺失 `--confirm-real`、`--cycles 2` 均以退出码 `1` 拒绝执行且零物理运动。
   - 新增 `tests/test_batch9_hardware_safety.py`（10 项离线单测），全量测试套件通过。

