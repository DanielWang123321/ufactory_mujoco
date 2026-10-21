# 批次 1 计划：配置与机器人注册表

> 目标：给后续批次一个只读的机型注册表和严格的任务配置。五款机械臂的身份、末端、夹爪兼容和现有 MJCF 从这里解析；Reach 的工作空间标定并入同一张表；抓放和轨迹执行按实际轴数走 `RobotInterface`。
>
> 进度：完成（2026-10-20）。全量单元测试 176 通过、13 跳过。

## 1. 目标

- 一个代码注册表覆盖 lite6 / xarm5 / xarm6 / xarm7 / uf850：自由度、TCP site、夹爪兼容、已有场景路径。
- Reach 的 `RobotProfile` 数值只保存在该注册表，`src.rl` 继续导出同名字段。
- YAML 任务配置：仓库默认值、严格 overlay、解析结果的 SHA-256。
- `RobotInterface` 公开 `n_arm`。关节指令长度必须等于 `n_arm`。`05_pick_place.py` 不再读取 `_n_arm_qpos`。
- 验收：五款裸臂场景可解析；未知字段、不兼容夹爪、缺失组合模型都报错；`python -m unittest discover -s tests` 通过。

## 2. 范围

**做**

- `src/robots.py`：机型与夹爪兼容、场景解析。
- `src/config.py` 与 `configs/default.yaml`：加载、严格合并、哈希。
- 把 Reach 标定表移入注册表，并保持 `src.rl.ROBOT_PROFILES` 的字段名。
- `RobotInterface.n_arm`，以及 `move_joints` / `execute_trajectory` 的长度检查。
- `examples/simulation/05_pick_place.py` 改用 `n_arm`。
- `requirements.txt` 增加 `pyyaml`。
- 注册表、配置、轴数的单元测试。

**不做**

- 不生成 xarm5 / xarm7 / uf850 的带夹爪 MJCF。这些组合在注册表里标为兼容但资产缺失，构建留给批次 3。
- 不把抓放几何、间隙、速度从 `05_pick_place.py` 搬进 YAML。那是批次 3。
- 不改夹爪 0–1 开合，不改 Reach 奖励、阈值和工作空间数字。
- 不把 `02` / `04` / `07` / `08` / `09` 改成纯接口调用。它们要直接读 `MjModel` / `MjData`。
- 不引入 Genesis 的安全策略、PD 增益、真机标定和打包场景字段。

## 3. 现状

注册表是两套互不相通的表：

- [src/model_paths.py](../src/model_paths.py) 只把名字解析到 `scene.xml` 或同名 XML，没有自由度、末端和夹爪。
- [src/rl/reach_env.py](../src/rl/reach_env.py) 的 `RobotProfile` 只有 Reach 工作空间：`n_joints`、目标盒、肩部、径向带。`scripts/train_reach_rl.py`、`examples/rl/reach/01_reach.py`、`tests/test_reach_env.py` 都读 `ROBOT_PROFILES[robot].n_joints` 和 `target_low`。

接口注释写着 6 轴，实现已经按模型切分。[src/sim_robot.py](../src/sim_robot.py) 用夹爪执行器名前缀计算 `_n_arm_qpos`，裸臂等于全部执行器。`get_joint_positions` 返回的就是这段。轨迹模块的 `n_joints` 来自数组长度，不写死 6。

仍绑在 6 轴或私有字段上的调用：

- [src/robot_interface.py](../src/robot_interface.py) 类文档写 “6-DoF”。没有公开轴数。
- [examples/simulation/05_pick_place.py](../examples/simulation/05_pick_place.py) 用 `q_home[:robot._n_arm_qpos]`。方块位姿仍读 `robot.data.body`，这是场景状态，本批次保留。
- [examples/simulation/06_trajectory_planning.py](../examples/simulation/06_trajectory_planning.py) 的关节点是 xArm6 的 6 维示例数据。执行已经走 `move_joints`、`execute_trajectory`、`get_tcp_pose`。`plan_joint_trajectory` 只是 `plan_ptp_quintic` 的封装，留在 `SimRobot`，不升到抽象接口。

已有组合模型只有四个目录：

| 机械臂 | 自由度 | 声明兼容的夹爪 | 已有组合场景 |
| --- | --- | --- | --- |
| lite6 | 6 | lite6_gripper、lite6_vacuum_gripper | `lite6_with_gripper`、`lite6_with_vacuum_gripper` |
| xarm5 | 5 | gripper_g2、bio_gripper_g2 | 无 |
| xarm6 | 6 | gripper_g2、bio_gripper_g2 | `xarm6_with_gripper`、`xarm6_with_bio_gripper` |
| xarm7 | 7 | gripper_g2、bio_gripper_g2 | 无 |
| uf850 | 6 | gripper_g2、bio_gripper_g2 | 无 |

五款裸臂都有 `scene.xml`、`scene_reach.xml`，TCP site 都是 `attachment_site`。`scene_obstacles.xml` 只有 xarm6。`scene_pick_place.xml` 只有 `xarm6_with_gripper`。独立夹爪场景继续走 `resolve_model_path`，不进机械臂注册表。

`requirements.txt` 没有 PyYAML。

## 4. 决策

1. **身份在代码里，任务选择在 YAML 里。** 自由度、site、夹爪兼容、Reach 标定不能被 overlay 改写。YAML 只选择 `robot`、`gripper` 和 `scene`。
2. **兼容与资产分开。** 夹爪兼容矩阵与 Genesis 公开示例一致。解析函数在文件不存在时抛出明确错误，不返回空路径，也不在本批次生成模型。
3. **Reach 字段名保持不变。** 数字搬到 `src/robots.py` 后，`RobotProfile` 仍提供 `robot`、`n_joints`、`target_low`、`target_high`、`shoulder`、`reach_min`、`reach_max`。`src.rl` 再导出，现有训练脚本和测试的 import 不变。
4. **轴数是接口的一部分。** `n_arm` 为抽象属性。`SimRobot.n_arm` 返回现有的 `_n_arm_qpos`。`set_joint_positions`、`move_joints`、`execute_trajectory` 在长度不为 `n_arm` 时抛出 `ValueError`。
5. **`move_tcp` 使用已解析的 site。** 现在 `move_tcp` 和 `ik_trajectory` 把 site 写死为 `attachment_site`，而构造函数已经允许回退到 `link_tcp`。改为使用 `self.model.site(self._site).name`。五款机械臂的注册 site 仍是 `attachment_site`。
6. **保留 `resolve_model_path`。** 查看器、独立夹爪和带任务后缀的目录名继续用它。注册表负责五款机械臂及其已知场景种类。

## 5. 技术方案

```text
src/robots.py          ArmSpec、Reach 标定、夹爪兼容、resolve_arm_scene
src/config.py          load_task_config、ConfigError、config_sha256
configs/default.yaml   schema_version、robot、gripper、scene
src/rl/reach_env.py    改为从注册表取 RobotProfile，不再自带数字
src/robot_interface.py n_arm，以及关节向量长度检查的约定
src/sim_robot.py       实现 n_arm；move_tcp 使用解析后的 site
```

```python
@dataclass(frozen=True)
class ReachWorkspace:
    target_low: tuple[float, float, float]
    target_high: tuple[float, float, float]
    shoulder: tuple[float, float, float]
    reach_min: float
    reach_max: float


@dataclass(frozen=True)
class ArmSpec:
    key: str
    dof: int
    tcp_site: str
    grippers: tuple[str, ...]   # 声明兼容，不表示 MJCF 已存在
    reach: ReachWorkspace


def robot_profile(key: str) -> RobotProfile:
    """旧字段名视图：n_joints 等于 ArmSpec.dof，标定值原样返回。"""


def resolve_arm_scene(robot: str, gripper: str | None, scene: str) -> Path:
    """返回已存在的 XML。不兼容或文件缺失时抛出 ConfigError。"""
```

`RobotProfile` 留在 `src/robots.py`。`reach_env.ROBOT_PROFILES` 由五款 `robot_profile()` 组成，键和数值与现在的表一致。

场景种类：

| `scene` | 解析规则 |
| --- | --- |
| `scene` | 无夹爪：`models/<robot>/scene.xml`。有夹爪：`models/<组合名>/scene.xml` |
| `reach` | 仅裸臂，`models/<robot>/scene_reach.xml`。同时指定夹爪则报错 |
| `obstacles` | 仅 `xarm6` 裸臂 |
| `pick_place` | 仅 `xarm6` + `gripper_g2` |

组合目录名沿用现有资产：`xarm6_with_gripper`、`xarm6_with_bio_gripper`、`lite6_with_gripper`、`lite6_with_vacuum_gripper`。xarm5 / xarm7 / uf850 加 G2 或 Bio G2 时，错误信息写明“兼容但 MJCF 尚未构建”。lite6 配 `gripper_g2`、xarm6 配 `lite6_gripper` 属于不兼容，错误信息与“文件缺失”区分开。

默认配置：

```yaml
schema_version: 1
robot: xarm6
gripper: null
scene: scene
```

加载规则：

- 根节点必须是 mapping，`schema_version` 必须为 1。
- overlay 只能覆盖已有键。未知键、错误类型（例如 `robot` 写成列表）抛出 `ConfigError`。
- `gripper: null` 表示裸臂。其他值必须是该机型 `grippers` 中的名字。
- 哈希对解析后的 `schema_version`、`robot`、`gripper`、`scene` 做 canonical JSON（键排序、无空白），不把来源路径和哈希本身算进去。同一解析结果哈希稳定；改任一字段则哈希改变。

`05_pick_place.py` 只把 `robot._n_arm_qpos` 换成 `robot.n_arm`。方块 body、间隙和路点仍留在该文件。

## 6. 测试

新增 `tests/test_robots.py`：

- 五款裸臂的 `dof` 为 5 / 6 / 6 / 7 / 6（xarm5、lite6、xarm6、xarm7、uf850）。
- `resolve_arm_scene(robot, None, "scene")` 与 `scene_reach` 指向现有文件，模型里存在 `attachment_site`，裸臂执行器数等于 `dof`。
- xarm6 + gripper_g2 的 `scene` 指向 `models/xarm6_with_gripper/scene.xml`；`pick_place` 指向 `scene_pick_place.xml`。
- lite6 + gripper_g2 报不兼容。xarm5 + gripper_g2 报资产缺失。
- `reach` 与非空夹爪同时出现时报错。

新增 `tests/test_config.py`：

- 只加载默认文件得到 xarm6、无夹爪、`scene`。
- 未知字段、`schema_version: 2`、非 mapping 根节点抛出 `ConfigError`。
- 同一 overlay 两次哈希相同；改 `robot` 后哈希不同。

扩展现有测试：

- `SimRobot` 加载 xarm5 / xarm6 / xarm7 时 `n_arm` 为 5 / 6 / 7。
- `move_joints` 传入错误长度时抛出 `ValueError`，不写 ctrl。
- `tests/test_reach_env.py` 不改断言。它继续从 `src.rl` 读取 `ROBOT_PROFILES`。

收尾运行 `python -m unittest discover -s tests`。

## 7. 实施顺序

1. 注册表与 `resolve_arm_scene`，补 `tests/test_robots.py`。
2. 把 Reach 数字移入注册表，`reach_env` 改为引用；跑 Reach 测试确认标定未变。
3. YAML 加载、严格 overlay、哈希，补 `tests/test_config.py`，依赖写入 `requirements.txt`。
4. `n_arm`、关节长度检查、`move_tcp` 的 site，以及 `05_pick_place.py` 的一处替换。
5. 全量单元测试。通过后把 [todolist.md](../todolist.md) 批次 1 的对应项勾上。
