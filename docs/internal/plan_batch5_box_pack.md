# 批次 5 计划：装箱展示

> 目标：在批次 3 的抓放执行和批次 4 的阶段/预检之上做装箱展示。方块从供料位抓取，运送到盒子上方，在盒沿上方 50 mm 释放后自由落体，允许反弹；多周期逐块装箱；速度倍率缩放运动时长；五臂仿真与 dry-run。
>
> 进度：完成（2026-10-20）。五臂 sim 3 周期与 dry-run 全过；入盒边界裕度 ≥ −0.1 mm（容差 1 mm），回位漂移 ≤ 0.8 mm。全量单元测试 202 通过、13 跳过。

## 1. 目标

- 盒子场景：静态盒体（底板 + 四壁）+ 3 个供料方块，五款带夹爪组合各一份 MJCF。
- 轨迹控制的装箱流程：接近 → 抓取（双侧指垫接触）→ 抬起 → 运送 → 释放（盒沿上方 50 mm 自由落体）→ 回位。
- 多周期：`--cycles N`，每周期装一块，周期数不超过供料方块数。
- 速度倍率：`speed_scale` 与 `--speed`，缩放段时长与笛卡尔速度/加速度，不缩放自由落体物理。
- 验收：所有物块静止后最终碰撞边界落在盒内（1 mm 接触容差）；抓取、运送、释放、回位阶段通过；五臂仿真与 dry-run 均通过。

对照 ufactory_genesis 0.3.2 的 Packaging Showcase：盒沿上方 50 mm 释放、自然下落、允许反弹；回归判据是最终碰撞包围盒在盒内（1 mm 接触容差），而不是到目标中心的距离；抓取、运送、释放、回位检查保留。

## 2. 范围

**做**

- `models/<combo>/scene_box_pack.xml` × 5（沿用批次 3 的 `_PICK_PLACE_DIRS` 组合目录）：`pack_box` 静态盒体（底板 + 四壁 box geom），`cube_0..2` 三个带 freejoint 的供料方块。
- `src/robots.py`：`SCENES` 增加 `"box_pack"`，解析到上述文件。
- `configs/default.yaml` 新增 `box_pack` 节；`configs/box_pack/<robot>.yaml` × 5；`src/config.py` 新增 `BoxPackConfig` 与校验，`_HASH_FIELDS` 加入 `box_pack`。
- `src/preflight.py`：`preflight_box_pack`，逐周期输出相位报告。
- `examples/simulation/10_box_pack.py`：执行入口与报告。
- `tests/test_box_pack.py`，以及受影响的现有测试同步。

**不做**

- 不引入 MjSpec 程序化建模；沿用静态 MJCF + YAML 镜像值 + 一致性测试的现有模式。
- 不做强化学习装箱（Genesis 侧同为计划项，todolist 已排除）。
- 不涉及真机与 sdk-sim（批次 9）。
- 不生成包装箱纹理（参考实现的 `.[showcase]` 纹理与本仓库验收无关）。
- 不改批次 3/4 已验收的抓放行为与阈值。

## 3. 现状

- 场景：`models/<combo>/scene_pick_place.xml` 内联地板、方块（freejoint，xy 运行时由 YAML 写入、z 保留 authored 高度）、目标标记。无任何带壁容器资产。
- 配置：`src/config.py` 只有 `PickPlaceConfig`，严格 overlay，未知字段抛 `ConfigError`；无 `speed_scale`。
- 预检：`src/preflight.py` 的 `PHASES` 绑定抓放六阶段；`make_checker` 的 `_IGNORED_BODIES` 写死 `("world", "cube", "pick_stand", "place_stand")`，单方块名；盒壁需要参与碰撞检查但当前没有盒体。
- 执行：`05_pick_place.py` 的放置是下降到抓取高度轻放，无自由落体；`--cycles` 已存在但每周期从方块实际位置续抓，没有多方块管理。
- 参考语义：释放高度为盒沿上方 50 mm；验收用碰撞包围盒 + 1 mm 容差；阶段为抓取、运送、释放、回位。

## 4. 决策

1. **沿用静态场景模式。** 盒体与供料方块写进每组合的 `scene_box_pack.xml`；尺寸、质量在 YAML 里保存镜像值，由测试断言一致，不在运行时改模型。方块数固定 3 个，`cycles` 不得超过 3。
2. **盒体参与碰撞检查。** 预检 checker 按前缀忽略 `cube*`（供料方块），忽略 `world` 与 lite6 手指对；`pack_box` 不忽略——运送与释放段必须越过盒沿，穿透盒壁即预检失败。
3. **释放判定分两层。** 预检层：释放点 xy 必须落在盒内腔，失败原因 `out_of_box`；执行层：开爪后 settle 至方块线/角速度低于阈值（超时 `settle_timeout_s`），再判定碰撞边界。
4. **验收用碰撞边界，不用中心距。** 静止后方块中心 ± `cube_half_m` 的包围盒：水平方向 `|x−cx|+half ≤ inner_half_x+1e-3`（y 同理）；竖直方向底面不低于盒底板顶面 −1 mm、顶面不高于盒沿 +1 mm。
5. **速度倍率只缩放指令时长。** 段时长与命令式 settle 除以 `speed_scale`，`cart_v×s`、`cart_a×s²`；自由落体的 settle 按速度阈值收敛，超时时长不缩放。默认 1.0，`--speed` 经严格 overlay 覆盖。
6. **阶段名对齐验收口径。** `PACK_PHASES = ("接近", "抓取", "抬起", "运送", "释放", "回位")`；`抓取` 相位校验 `grasp_gap_m` 在夹爪行程内（同抓放的 `闭合` 校验）。

## 5. 技术方案

```text
models/<combo>/scene_box_pack.xml   pack_box + cube_0..2，沿用 _PICK_PLACE_DIRS
src/robots.py                       SCENES += "box_pack"
configs/default.yaml                新增 box_pack 节（xarm6 默认值）
configs/box_pack/<robot>.yaml       五臂 overlay（lite6 几何单独缩小）
src/config.py                       BoxPackConfig、校验、哈希字段
src/preflight.py                    PACK_PHASES、preflight_box_pack、cube* 前缀忽略
examples/simulation/10_box_pack.py  CLI、周期循环、自由落体、报告
```

配置 schema（`box_pack` 节，全部必填，校验规则同 `PickPlaceConfig`）：

```yaml
box_pack:
  feed_xy: [[0.30, -0.18], [0.36, -0.18], [0.42, -0.18]]  # 每块供料位，非空 vec2 列表，长度即方块数
  cube_half_m: 0.015
  cube_mass_kg: 0.017
  grasp_gap_m: 0.022
  box_center_xy: [0.30, 0.30]
  box_inner_half_xy: [0.075, 0.075]   # 内腔半宽；三块 30 mm 方块可并排落底
  box_floor_m: 0.005                  # 底板厚度
  box_wall_m: 0.005                   # 壁厚
  box_rim_z_m: 0.06                   # 盒沿 z（世界系）
  release_above_rim_m: 0.05           # 释放点 = 盒沿上方 50 mm
  approach_m: 0.10
  lift_z_m: 0.30
  flange_grasp_z_m: 0.1871
  cart_v_m_s: 0.15
  cart_a_m_s2: 0.8
  speed_scale: 1.0                    # 正数，>1 加快
  settle_timeout_s: 4.0               # 自由落体 settle 上限
```

预检：`preflight_box_pack(robot, config, cycles)` 复用 `plan_segment` / `_connect`。每周期段序：deploy→pre→grasp（接近）、gap 校验（抓取）、grasp→lift（抬起）、lift→盒心上方 lift_z（运送）、上方→释放点（释放，附加 `out_of_box` 判定）、释放点→上方→home（回位）。高度由 `10_box_pack.py` 的高度助手测出（同 05 的 `tcp_task_heights` 模式，dry-run 与执行同位姿）。报告按周期分组写入 `as_dict`。

执行（每周期 i）：

1. `_set_cube` 同款把 `cube_i` 写到 `feed_xy[i]`；
2. 接近 → `_close_until_contact(robot, "cube_i", grasp_gap_m)` 双垫接触，否则抛 `GraspError`；
3. 抬起到 lift_z → 运送到盒心上方 → 下降到 `box_rim_z_m + release_above_rim_m`（TCP 高度含 flange→TCP 偏移）；
4. 开爪释放，settle 至 `cube_i` 速度收敛或超时；记录释放高度与落停位姿；
5. 回上方 → 回位；记录 `cube_lifted`、`released_z`、`in_box`、`home_drift_mm`。

CLI：`--robot --gripper --executor --config --headless --dry-run --report --cycles N --speed S`。退出码：全部周期 `cube_lifted and in_box and home_drift_mm < 10` 为 0；dry-run 以预检 ok 为 0。

## 6. 测试

新增 `tests/test_box_pack.py`（headless，沿用 test_pick_place.py 模式）：

- 配置：默认 `box_pack` 解析；未知键、`speed_scale: 0`、空 `feed_xy`、非 vec2 供料位抛 `ConfigError`；五臂 overlay 均可加载；`test_config.py` 若钉住哈希则同步更新。
- 场景：五组合加载后 `pack_box`、`cube_0..2` 存在；盒体内腔、壁厚、方块尺寸/质量与 YAML 镜像一致。
- dry-run：xarm6 全周期全相位 ok、无障碍无绕行；盒心改到不可达处 → fail 且 `data.time == 0`（不步进物理）；释放点改到盒外 → `out_of_box`。
- 仿真：xarm6 + G2 单周期入盒（1 mm 容差）且 `home_drift < 10 mm`；五臂各 1 周期 headless 入盒；xarm5/xarm6/xarm7/uf850 另跑 `servo_j`；xarm6 `--cycles 3` 三块全部入盒；`--speed 2.0` 的仿真时长低于 1.0 基线且仍入盒；释放后方块 z 曾低于释放高度（确认自由落体发生）。
- 已知限制：lite6 装箱固定 `servo_cartesian`。2 g 方块在 lite6 小夹爪中握持临界，`servo_j` 关节空间执行运送时蠕变滑脱；抓放能幸存是因为运送仅 5 cm 且放置下降会复位方块。
- 回归：`python -m unittest discover -s tests` 全量通过（批次 4 基线 188 通过、13 跳过 + 新增用例）。

## 7. 实施顺序

1. 场景 MJCF × 5 + `SCENES` 注册 + `box_pack` 配置节与五臂 overlay，补场景/配置测试。
2. `preflight_box_pack` 与 checker 前缀忽略，补 dry-run 测试。
3. `10_box_pack.py` 执行与报告，先 xarm6 单周期跑通入盒。
4. 多周期与速度倍率，补对应测试。
5. 五臂仿真 + dry-run 全量验收；更新 `examples/simulation/README.md` 清单与 `worklog.md`；勾选 [todolist.md](../todolist.md) 批次 5。
