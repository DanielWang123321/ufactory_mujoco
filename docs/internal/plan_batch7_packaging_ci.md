# 批次 7：可安装包与 CI

## 目标与边界

交付离开源码目录仍可运行的自包含 wheel/sdist，统一提供抓放、装箱及动力学命令。保持开发版本 `0.1.0.dev0`，保留批次 6 工作区修改。不发布 PyPI，不扩展 RL，不连接真机，不调整模型参数。

## 实施计划

1. 沿用 setuptools，以 `pyproject.toml` 为核心、`[rl]`、`[assets]` 依赖来源，同步 requirements。核心不安装 PyTorch、Gymnasium、转换工具或 xArm SDK。
2. 构建时把全部模型/场景、默认及五臂 YAML、项目和资产许可证及 NOTICE 收入包内。网格按原始资产路径集中存放，每份只复制一次；仅修改构建目录 XML 的 `meshdir`，移除对 14 个目录软链接的依赖。运行无需下载或写安装目录。
3. 模型环境变量覆盖优先；普通安装使用包内资源；源码/editable 使用仓库资源。保留路径接口和配置内容哈希。
4. 实现迁入 `ufactory_mujoco.tasks`，几何/接触辅助逻辑独立；包内不导入 examples。三个旧脚本为兼容包装，功能测试转向包内实现。
5. 注册 `ufactory-mujoco pick-place`、`box-pack`、`dynamics`，沿用机型、执行器、周期、速度、预检及报告选项。退出码：成功 0，运行失败 1，参数非法 2。报告相对当前目录，自动建父目录；写入失败非零。

## CI 与验收

- 核心：Ubuntu + Python 3.12/3.13/3.14，editable，全量 unittest 和既有 headless 示例。
- 构建：先 sdist，再从 sdist 构建 wheel；`twine check`；检查清单、网格去重、许可证和体积，上传产物。预估 wheel 89 MiB、展开资源 533 MiB（不含依赖）。排除权重、日志、诊断、构建残留。
- 安装：3.12/3.13/3.14 独立 job 不检出源码，只下载 wheel、验收脚本和基准。清除 PYTHONPATH，确认导入安装位置；`pip check`，检查无可选依赖、无软链接，全部 XML/YAML 可加载；对比自由度、执行器、质量矩阵和配置哈希。
- 抓放：五臂各一次 headless 和 dry-run，双侧接触，放置误差 <50 mm，回位漂移 <10 mm。
- 装箱：五臂各一次 headless 和 dry-run，物块静止且碰撞边界在盒内（1 mm 接触容差），回位 <10 mm；保留多周期、速度、双执行器和 Lite6 执行器限制。
- 动力学：`--robot all --samples 32 --seed 0 --headless`，保持批次 6 判据及 xArm6 六轴 3 kg 基线。
- 命令：主/子命令帮助、旧脚本，非法机型/参数、缺配置、无效模型覆盖、预检失败、报告写入失败。dry-run 不步进，JSON 结果与退出码一致。
- RL：Python 3.13 CPU PyTorch + `[rl]`，保留 Reach 测试和 4096 步训练及指标检查。
- 无需 GPU/查看器；失败时也上传日志/JSON。同步中英文安装/命令文档、worklog；验收后更新任务清单及 roadmap。

## 参考

- [setuptools package data](https://setuptools.pypa.io/en/stable/userguide/datafiles.html)
- [PyPA project.scripts](https://packaging.python.org/en/latest/specifications/pyproject-toml/#entry-points)

## 验收记录

2026-10-21 本地验收：

- 开发版本保持 `0.1.0.dev0`；`python -m build` 的 sdist → wheel、直接源码构建 wheel 均通过清单检查，`twine check` 通过。
- 最终 wheel **93,649,598 B（89.31 MiB）**，sdist **91,617,906 B（87.37 MiB）**；包内资源 **558,328,958 B（532.46 MiB）**。共 48 XML、135 网格、11 YAML、5 许可文本，无软链接、重复网格路径或本地构建/权重/诊断文件。
- Python 3.12.13 / 3.13.13 / 3.14.6 普通安装各通过 **44 项命令检查**及 `pip check`；无 RL/资产转换/SDK，清除 PYTHONPATH，从仓库外启动，并将安装包目录改为只读。CI 安装 job 不 checkout 源码；本地验证禁止源码导入路径，未隐藏磁盘上的源码文件。
- 所有模型与场景的 XML 内容（仅 meshdir 不同）、自由度、执行器及质量矩阵一致；全部 YAML 字节及配置内容哈希一致。源码模型树无修改；原批次 6 的 21 个函数/类 AST 保持一致，入口和报告写入迁至包内。
- 五臂各完成抓放/装箱 headless 与 dry-run；最大放置误差 **44.9292 mm**，抓放回位最大 **0.4289 mm**，装箱回位最大 **0.5754 mm**，装箱边界最小余量 **−0.1078 mm**（通过 1 mm 容差）。dry-run 以物理步进拦截及仿真时间验证。
- 五臂动力学各 4201 步；固定种子 32 样本、逐步监测及 xArm6 六轴 3 kg 基线均通过。
- Python 3.13 CPU PyTorch 2.14.1 + RL extras：Reach **27 项通过**，4096 步 / 2 环境训练完成，2 回合评估及 metrics 契约通过。烟测不要求训练后成功率。
- Python 3.12 / 3.13 / 3.14 最终全量各 **252 项（238 通过、14 可选依赖跳过）**，各 **22 条既有 headless 示例及 1 条预期失败通过**。3.12/3.14 并发时以 137 终止的进程已串行复核通过；最终记录为 `build/batch7/verification.json`。批次 7 验收完成。
- 产物及逐项 JSON/日志位于 `build/batch7/`；GitHub Actions 尚未远端运行，已校验 YAML、shell 语法、三版本矩阵及无 checkout 安装结构。
