# 批次 8：强化学习（Reach 契约化与 Pick and Place 抓放）

## 目标与边界

对照 `ufactory_genesis` 0.3.2 与 [todolist.md](../../todolist.md) 批次 8：
1. **Reach**：补齐声明式 `examples/rl/reach/recipe.yaml`、独立评估入口 `examples/rl/reach/evaluate.py`、训练配置与 checkpoint 契约校验（`config.yaml` + `.checkpoint_manifest.json`）；不改变已达标的五臂策略阈值（`< 5 mm`、成功率 `≥ 95%`、平均终止距离 `≤ 10 mm`），并为现有五臂达标权重回填契约清单。
2. **Pick and Place（xArm6 + Gripper G2，桌面方块）**：复用批次 2–3 的场景（`models/xarm6_with_gripper/scene_pick_place.xml`）、任务配置（`configs/pick_place/xarm6.yaml`）与夹爪接口（`ufactory_mujoco.grippers`），实现 `PickPlaceEnv`、独立评估指标 `PickPlaceMetrics`、闭环脚本专家基线 `PickPlaceExpert`、PPO 训练脚本 `scripts/train_pick_place_rl.py` 与评估回放入口 `examples/rl/pick_place/01_pick_place.py` / `evaluate.py`。
3. **兼容性边界**：继续使用 `stable-baselines3`；核心环境仅依赖 `mujoco` + `numpy`（`py314` 无 `gymnasium`/`stable-baselines3` 时可导入并运行 `--random`/`--expert`）；不向 `configs/` 添加 RL 配方文件，保持批次 7 的 11 个任务配置与发行包清单不变。

## 实施方案

1. **契约与配方模块（`ufactory_mujoco/rl/artifacts.py`）**：
   - 严格校验 `recipe.yaml` 顶层节（`schema_version`, `task`, `environment`, `reward`, `train`）、已知字段与有限数值；Reach 强制 `success_dist == 0.005`；Pick-and-Place 仅允许 `xarm6` 并绑定 `configs/pick_place/xarm6.yaml` 的 `sha256`。
   - 训练输出 `config.yaml`（含 canonical JSON `config_sha256`）、`<stem>.checkpoint_manifest.json`（含 `checkpoint_sha256`, `config_sha256`, `observation_dim`, `action_dim`, `action_contract`, `runtime_config_sha256`）与 `provenance.json`。
   - `load_validated_ppo` 在调用 `PPO.load` 前完成文件哈希、配置哈希、维度与契约校验。
2. **Reach 评估与权重契约化**：
   - `scripts/train_reach_rl.py` 接入 `--recipe` 与 `--backfill-manifest`，自动生成并校验五臂 checkpoint manifest。
   - `examples/rl/reach/evaluate.py` 与 `01_reach.py` 支持契约校验与 `--report` / `--summary-json` 严格 JSON 输出。
3. **Pick and Place 环境、训练与评估（`ufactory_mujoco/rl/pick_place_env.py`）**：
   - 4 维连续增量动作 `[Δx, Δy, Δz, Δgap]`（`cartesian_and_g2_gap_increment`），末端保持 `GRIP_DOWN_QUAT`，经带步长/限位保护的阻尼最小二乘 IK 与 `qfrc_bias` 重力补偿驱动关节，夹爪经 `gap_to_position` 驱动 Gripper G2。
   - `PickPlaceMetrics` 独立追踪双侧夹持、安全抬升（净空 `≥ 65 mm`）、抬起前拖拽（`≤ 10 mm`）与倾角（`≤ 15°`）、目标区触地过压（`press_mm ≤ 1.0 mm`）、低空释放确认（`release_height ≤ 3.0 mm` 且间隙 `≥ 35 mm`）、目标平面误差（`target_error_mm ≤ 20.0 mm`）、碰桌力与 `0.2 s` 稳定保持。
   - 提供 `PickPlaceExpert`、`scripts/train_pick_place_rl.py`、`examples/rl/pick_place/01_pick_place.py` 与 `examples/rl/pick_place/evaluate.py`。

## 验收结果（2026-10-23）

1. **Reach 五臂契约校验与评测**：
   - `python scripts/train_reach_rl.py --model all --backfill-manifest` 为五臂达标权重生成并校验 `config.yaml`、`ppo_reach.checkpoint_manifest.json` 与 `provenance.json`。
   - `python examples/rl/reach/evaluate.py --model all --episodes 20 --min-success-rate 0.95 --max-mean-dist-m 0.010 --out build/rl/reach_eval.json --quiet-episodes`：
     - `lite6`：成功率 `100.0%`，平均误差 `4.3 mm`
     - `xarm5`：成功率 `100.0%`，平均误差 `4.4 mm`
     - `xarm6`：成功率 `100.0%`，平均误差 `4.2 mm`
     - `xarm7`：成功率 `100.0%`，平均误差 `4.3 mm`
     - `uf850`：成功率 `100.0%`，平均误差 `4.1 mm`
2. **Pick and Place（xArm6 + Gripper G2）训练、契约与评测**：
   - `python examples/rl/pick_place/evaluate.py --expert --episodes 10 --quiet-episodes`：`py314`（MuJoCo 3.15.0）与 `py313`（MuJoCo 3.10.0）双环境 100% 通过。
   - `python scripts/train_pick_place_rl.py --total-steps 4096 --n-envs 4`：生成 `build/rl/pick_place/xarm6/{ppo_pick_place.zip,config.yaml,ppo_pick_place.checkpoint_manifest.json,provenance.json,metrics.json}`。
   - `python examples/rl/pick_place/evaluate.py --episodes 20 --min-success-rate 0.90 --max-press-mm 1.0 --out build/rl/pick_place/xarm6/eval.json --quiet-episodes`：
     - 成功率：`100.0%`（`20 / 20`）
     - 最大下压深度：`0.115 mm`（阈值 `<= 1.0 mm`）
     - 释放高度：`0.00 mm`（阈值 `<= 3.0 mm`）
     - 最大抬升净空：`92.4 mm`（阈值 `>= 65 mm`）
     - 平均目标误差：`3.86 mm`（阈值 `<= 20.0 mm`）
     - 拖拽/倾覆/掉落/高空抛投/碰桌失败率：`0.0%`
3. **测试与回归**：
   - 新增/扩展 `tests/test_reach_env.py`（30 项）与 `tests/test_pick_place_rl.py`（11 项），在 `py314` 与 `py313` 全量通过。
