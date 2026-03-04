# 实验配置/算法解耦重构落地方案（ExperimentSpec + Registry）
更新时间：2026-02-21

## 目标与范围
目标：把当前“`ABLATION_SETTINGS` 大字典 + `main.py` 内大量 `if/elif` 分支读配置键”的模式，重构为“ExperimentSpec（实验规格）+ 组件注册表（Registry）+ 可插拔策略组件”的架构，使不同实验（尤其 full_model 系列 vs baselines）在代码层面隔离，避免修改一个策略点影响其他算法。

范围（本轮重构覆盖）：
- 实验选择与配置装配：`src/experiments/ablation_config.py`、`src/run_parallel_strict.py`
- 主流程装配点：`src/main.py`（ActiveLearningPipeline）
- agent/λ 策略：`src/agent/toolbox.py`（目前承担 λ_policy 逻辑）
- core 采样器的依赖方向：`src/core/sampler.py`（当前反向依赖 `agent.config.AgentThresholds`）

非目标（先不动或最后再动）：
- 训练器实现与模型结构：`src/core/trainer.py`、`src/core/model.py`
- 数据池生成与数据集：`src/core/data_preprocessing.py`、`src/core/dataset.py`
- analysis/monitor 逻辑：先通过 trace schema 兼容保持不破坏（必要时后续再抽象）

---

## 现状问题（代码级）
- `ABLATION_SETTINGS` 作为弱类型 dict 聚合了实验定义，同时承载了策略参数、阈值覆盖、后处理等“实现细节”，导致新增/改动点需要同时修改多个分支位置（[ablation_config.py](file:///Users/anykong/AAL_SD/src/experiments/ablation_config.py)、[main.py](file:///Users/anykong/AAL_SD/src/main.py)）。
- `ActiveLearningPipeline` 既做编排又做策略实现：sampler 初始化、λ 控制（rule/random/policy）、rollback 阈值、候选后处理、日志策略等全部混在一起（[main.py](file:///Users/anykong/AAL_SD/src/main.py)）。
- `core/sampler.py` 反向依赖 `agent/config.py`（DIP 违背），使 agent 阈值改动可能影响 core 算法（[sampler.py](file:///Users/anykong/AAL_SD/src/core/sampler.py#L8-L216)）。
- full_model 变体通过复制粘贴 dict 条目扩展，容易出现“版本漂移/漏改”。

---

## 目标架构（代码级蓝图）

### 1) ExperimentSpec：实验规格（组装单元）
每个 experiment_name 对应一个 ExperimentSpec（或 builder 函数），Spec 负责装配本实验的组件（sampler、λ provider、后处理、agent controller、trace 配置等）。

建议位置：
- `src/experiments/specs/`：每个实验或实验族一个文件（例如 `full_model.py`、`full_model_v4.py`、`baseline_entropy.py`）。
- `src/experiments/registry.py`：全局注册表。

ExperimentSpec 需要提供的最小信息（建议字段/方法）：
- `name: str`
- `description: str`
- `tags: list[str]`（如 `["full_model", "agent", "paper"]`，供筛选/报告使用）
- `build(config) -> ExperimentRuntime`（返回本实验运行时需要的组件集合）

其中 `ExperimentRuntime`（或等价结构）包含：
- `sampler: Sampler`
- `lambda_provider: LambdaProvider | None`
- `postprocessor: SelectionPostprocessor | None`
- `training_state_policy: TrainingStatePolicy`
- `rollback_policy: RollbackPolicy`
- `agent_controller: AgentController | None`
- `control_permissions: dict[str,bool]`（如果启用 agent）
- `trace_options: TraceOptions`（确保分析脚本字段稳定）

重要约束：
- Pipeline 不再读取“魔法 key”（如 `lambda_policy`、`selection_postprocess` 等）；这些变为组件内部参数。
- `ExperimentSpec` 是“装配层”，允许依赖 core + agent，但 core 不能反向依赖 agent。

---

### 2) Registry：注册表（开闭原则）
把“根据字符串选择实现”的 `if/elif` 结构变为注册表：
- `SamplerRegistry: dict[str, Callable[[Config], Sampler]]`
- `LambdaProviderRegistry: dict[str, Callable[[Config, Params], LambdaProvider]]`
- `PostprocessorRegistry: dict[str, Callable[[Config, Params], SelectionPostprocessor]]`
- `ExperimentRegistry: dict[str, Callable[[], ExperimentSpec]]`

新增实验/算法的变更应该收敛为：
- 新增一个实现文件（或类）
- 在 registry 中注册
- 不修改 `main.py` 主流程分支

---

### 3) 依赖方向（强制）
- `src/core/*`：只依赖 numpy/torch/utils，不依赖 `src/agent/*`、不依赖 experiments/spec。
- `src/agent/*`：可依赖 core 的接口/上下文结构，但不应直接读 `controller.exp_config` 这种弱类型 dict。
- `src/experiments/*`：作为装配层，可以同时引用 core 与 agent 的实现。

---

## 代码变更落地步骤（按阶段推进、可逐步合并）

### 阶段 A：引入 Spec 外壳 + 兼容适配（行为不变，风险最低）
目标：不改变现有实验行为，只改变“配置如何进入 Pipeline”。

变更点：
1. 新增模块
   - `src/experiments/specs/types.py`：定义 ExperimentSpec/ExperimentRuntime 的协议（可以用 typing.Protocol 或简单 dataclass）
   - `src/experiments/registry.py`：提供 `get_experiment_spec(name)` 或 `build_experiment_runtime(name, config)`
2. 在 `src/experiments/ablation_config.py` 上做“适配器层”
   - 保留 `ABLATION_SETTINGS` 不动（短期保证可回退）
   - 新增一个 `build_spec_from_legacy_dict(name, legacy_cfg) -> ExperimentSpec` 的工厂
3. 修改 `src/main.py`
   - `ActiveLearningPipeline.__init__` 不再直接使用 `ABLATION_SETTINGS.get(experiment_name)` 作为唯一来源
   - 改为：通过 registry 获取 `ExperimentRuntime`（但内部先仍然调用 legacy cfg 的行为实现）
4. 修改 `src/run_parallel_strict.py`
   - 保持 include/exclude/manifest 逻辑不变
   - manifest 中同时记录：legacy dict 与 spec 的 “schema_version/组件名” 等（便于后续比对）

验收标准：
- `python -m pytest -q` 全通过
- 选取一个包含 full_model 和 baseline 的最小 batch（n_rounds=2）跑通
- 生成的 trace 事件类型/关键字段不变

---

### 阶段 B：把 Pipeline 内策略分支替换为组件调用（OCP 收敛）
目标：删减 `main.py` 中与实验相关的分支，改为调用 `runtime.*` 组件。

拆分建议（先在同文件内抽函数，稳定后再物理拆文件）：
- `main.py` 中抽出：
  - `build_sampler(runtime, config)` 或直接由 runtime 提供实例
  - `compute_lambda_override(runtime, round_ctx)` 由 `lambda_provider` 决定
  - `postprocess_ranked(runtime, ranked, unlabeled_info, round_ctx)` 由 `postprocessor` 决定
  - `compute_training_state(runtime, trainer_outputs)` 由 `training_state_policy` 决定
  - `should_rollback(runtime, training_state)` 由 `rollback_policy` 决定（或作为 training_state 的一部分）

具体替换点（main.py 关键位置）：
- sampler 初始化：现有 `if self.sampler_type == ...` 改为 `runtime.sampler`
- λ 决策：现有 `lambda_override = ...` 多分支改为 `runtime.lambda_provider.lambda_for_round(ctx)`
- selection_postprocess / candidate_constraints：由 postprocessor 实现（pipeline 不再读 exp_config keys）
- enable_l3_logging/topk/max_selected：归入 `TraceOptions`

验收标准：
- full_model / baseline 的 rank/selection 行为与阶段 A 对齐（允许 trace meta 更丰富，但字段必须兼容）
- 新增一个“空实验 spec”不会触碰 main.py 分支（仅 registry 注册）

---

### 阶段 C：切断 core -> agent 的依赖（DIP 修复，防止阈值外溢）
目标：core 采样器不再 import `agent.config.AgentThresholds`。

现状点位：
- `src/core/sampler.py` 使用 `AgentThresholds.calculate_lambda_t`（[sampler.py](file:///Users/anykong/AAL_SD/src/core/sampler.py#L208-L216)）

改造原则：
- 把 `calculate_lambda_t(progress, alpha)` 移到 core 或 utils 的纯函数（例如 `src/core/lambda_utils.py` 或 `src/utils/math.py`）
- 或由 `LambdaProvider` 计算并直接把 override 传给 sampler，sampler 内不负责 sigmoid 逻辑

验收标准：
- `src/core/*` 不再引用 `src/agent/*`
- full_model 与 baseline 结果在短跑（2 rounds）上行为一致

---

### 阶段 D：full_model 系列“组合式变体”（去复制、避免漂移）
目标：full_model 变体不再是大量复制 dict 条目，而是“base spec + delta 覆盖”。

实现策略：
- `FullModelBaseSpec`：定义默认 rollback/training_state_policy/trace_options/lambda_provider 等
- 变体 spec 仅覆盖差异参数：
  - v3_optimized：阈值/步长/上限/EMA/cooling
  - v4_robust_severe_last：severe 判定 key/logic + postprocessor + constraints

落地结构建议：
- `src/experiments/specs/full_model_base.py`
- `src/experiments/specs/full_model_variants.py`（或拆多个文件）

验收标准：
- full_model 系列新增一个变体只需：新增一个 spec builder，不修改 main.py
- 删除/弃用某变体不会影响其他 spec

---

## 关键接口草案（便于落地时统一口径）
以下仅为代码级“契约草案”，具体名字可调整，但建议保持职责边界一致。

### Sampler
- 输入：`unlabeled_info: dict[sample_id, info]`、`labeled_features`、`context`
- 输出：`list[RankedItem]`（至少包含 `sample_id`、`final_score`；可包含 `uncertainty/knowledge_gain/lambda_t` 等）

### LambdaProvider
- `lambda_for_round(context) -> float | None`
- context 至少包含：`round_num`、`labeled_count`、`total_budget`、`training_state`、`score_stats`（如需要）

### SelectionPostprocessor
- `postprocess(ranked_items, unlabeled_info, context) -> ranked_items`

### TrainingStatePolicy
- `build_training_state(perf_history, epoch_mious, grad_tvc_values, ...) -> dict`

### RollbackPolicy
- `should_rollback(training_state) -> bool`
- 或者训练状态直接生成 `rollback_flag`，policy 只负责阈值/方式

### AgentController（可选）
- 仅在 agent 实验启用，提供：`decide_controls(context) -> ControlDecision`
- 受 `control_permissions` 限制；当禁止某 action 时必须自动降级到安全策略

---

## trace / analysis / monitor 的兼容策略（必须提前规划）
当前依赖点：
- `monitor_and_recover.py` 与 `analysis/plot_strategy_trajectory.py` 会读取 trace 中的字段（如 sampler_type、lambda、postprocess meta 等）

兼容原则：
- 保持事件类型不变：`initialized`、`round_summary`、`overfit_signal`、`lambda_policy_apply`、`l3_selection` 等（如需新增，新增不破坏旧字段）
- 增加 `trace_schema_version` 字段（建议在 `initialized` 事件里写一次）
- 对 analysis 脚本：提供字段 fallback（例如没有 `postprocess` 时按未应用处理）

---

## 风险点与应对
- 风险：重构过程容易引入“行为微变”导致实验可复现性下降  
  - 应对：阶段 A 先引入适配器，阶段 B 再替换实现；每阶段用短跑固定 seed 做回归对齐
- 风险：agent 与 λ policy 逻辑目前散落在 toolbox 与 pipeline  
  - 应对：把 λ policy 收敛到 LambdaProvider；toolbox 只做工具暴露与权限校验，不承担策略本身
- 风险：core 与 agent 的边界不清晰导致依赖循环  
  - 应对：强制 core 不 import agent；策略装配在 experiments 层完成

---

## 执行顺序建议（最小化冲突）
1) 阶段 A：spec 外壳 + legacy 适配（最重要，先打通骨架）
2) 阶段 B：sampler / λ / postprocess 组件化（把 main.py 分支收敛）
3) 阶段 C：切断 core->agent（提升可维护性与隔离性）
4) 阶段 D：full_model 变体组合化（减少复制、提升迭代效率）

---

## 编码进展跟踪（后续重构执行时同步更新）
状态说明：`todo / doing / done / blocked`

### Milestone A：Spec 外壳 + legacy 适配
- [x] (done) 新增 `experiments/specs/types.py` 定义 Spec/Runtime 契约
- [x] (done) 新增 `experiments/registry.py` 与最小注册入口
- [x] (done) `main.py` 接入 registry，但保持行为不变
- [x] (done) `run_parallel_strict.py` manifest 记录 spec 元信息

### Milestone B：组件化替换 main.py 分支
- [x] (done) sampler 初始化由 registry/runtime 提供
- [x] (done) λ 决策由 LambdaProvider 统一提供
- [x] (done) postprocess/constraints 迁移到 SelectionPostprocessor
- [x] (done) trace_options 抽出并保持字段兼容

补充落实（针对“配置项选择实现不允许 if/elif”这一约束）：
- [x] (done) `sampler_type` 通过 registry 映射到 builder（不再 if/elif 选 sampler 实现）
- [x] (done) `selection_postprocess.mode` / `candidate_constraints` 通过 registry 映射到实现（不再 if/elif 选实现）
- [x] (done) `training_state_policy.miou_signal` 通过 registry 映射到实现（不再 if/elif 选信号实现）

### Milestone C：切断 core->agent
- [ ] (todo) 替换 `core/sampler.py` 对 `AgentThresholds` 的依赖
- [ ] (todo) 增加对应回归测试（确保 λ 逻辑不漂移）

### Milestone D：full_model 组合式变体
- [ ] (todo) 引入 full_model base spec
- [ ] (todo) 迁移 v3/v4 变体为 delta 覆盖
- [ ] (todo) 删除/降级 `ablation_config.py` 中复制配置（保留向后兼容入口或 alias）

### 回归验证（每个 Milestone 完成必须满足）
- [ ] (todo) `python -m pytest -q` 通过
- [ ] (todo) 固定 seed 的 2-round smoke run 行为对齐

---

## 附录 A：Legacy 配置键 → 新组件映射表（必须执行的迁移口径）
本表用于把当前 `ABLATION_SETTINGS[exp_name]` 中的“魔法 key”逐步迁移到组件参数，避免在迁移期间出现“同一语义多处实现”。

> 约定：阶段 A 保留 legacy key，但通过适配器将其装配进 runtime；阶段 B 起，`main.py` 不再直接读取这些 key。

| Legacy key（当前 dict） | 当前读取位置 | 新归属（组件/层） | 迁移策略（阶段） | 备注 |
|---|---|---|---|---|
| `description` | `main.py` 初始化日志 | `ExperimentSpec.description` | A | 必须保留用于报告与 manifest |
| `use_agent` | `main.py` agent setup | `ExperimentRuntime.agent_controller is not None` | A/B | 语义从布尔改为“是否装配 controller” |
| `sampler_type` | `main.py` sampler `if/elif` | `runtime.sampler`（由 registry/spec 装配） | B | sampler 的字符串 type 仍可保留在 trace/manifest |
| `k_definition` | `main.py` 设置 `self.k_definition` | `runtime.sampler` 参数或 `SamplerContext` | B | 建议归入 Sampler 构造参数 |
| `score_normalization` | `main.py`（ADKUCSSampler 参数） | `runtime.sampler` 参数 | B | 仅 ADKUCSSampler 有意义 |
| `lambda_override` | `main.py` rank 流程 | `runtime.lambda_provider`（FixedLambdaProvider） | B | 禁止出现“Pipeline+Toolbox 双重 override” |
| `lambda_controller`（random/rule_based） | `main.py::_apply_lambda_controller` | `runtime.lambda_provider`（Random/RuleBased） | B | 迁移后删除 `_apply_lambda_controller` |
| `lambda_policy`（warmup_risk_closed_loop） | `toolbox.py::apply_round_lambda_policy` + `main.py` 调用时机 | `runtime.lambda_provider`（WarmupRiskClosedLoop） | B | Toolbox 不再承载策略实现，仅承载工具与权限 |
| `agent_threshold_overrides` | `toolbox.py::_agent_threshold` | `LambdaProvider/Controller` 的参数对象 | B/D | full_model 变体的 delta 应收敛到 spec 层 |
| `rollback_config` | `main.py` 训练后构造 rollback | `runtime.rollback_policy` | B | 训练状态与 rollback 逻辑建议拆分以便测试 |
| `training_state_policy` | `main.py` 训练后生成 miou_signal | `runtime.training_state_policy` | B | 统一产生 `training_state` 的 schema |
| `selection_postprocess` | `main.py::_postprocess_ranked_ids` | `runtime.postprocessor` | B | 将 mode/参数从 dict 迁移为实现参数 |
| `candidate_constraints` | `_postprocess_ranked_ids` | `runtime.postprocessor` 内部参数 | B | 约束属于后处理逻辑的一部分 |
| `enable_l3_logging` / `l3_topk` / `l3_max_selected` | `main.py::_append_l3_selection` | `runtime.trace_options` | B | trace 策略不应该散落在 pipeline 中 |
| `control_permissions` | `main.py` 设置 toolbox 权限 | `runtime.control_permissions` | A/B | 在 agent 实验必须显式给出（研究模式要求） |

迁移完成后的目标：`main.py` 不再出现对 `exp_config.get("<key>")` 的业务分支读取（允许 trace 中记录原始配置作为 meta，但不参与决策）。

---

## 附录 B：文件级变更清单（按阶段）
本清单用于拆任务与评审范围控制，避免“改动扩散到非目标模块”。

### 阶段 A（Spec 外壳 + legacy 适配）
- 修改：`src/main.py`（将 experiment_name -> runtime 的装配入口前置；保留旧行为）
- 修改：`src/experiments/ablation_config.py`（新增 legacy→spec 适配器函数；`ABLATION_SETTINGS` 原样保留）
- 新增：`src/experiments/registry.py`（ExperimentRegistry + build_runtime）
- 新增：`src/experiments/specs/types.py`（Spec/Runtime/Context 的 typing 契约）
- 修改：`src/run_parallel_strict.py`（manifest 写入 spec 元信息；不改变 include/exclude 行为）
- 可能修改：`src/experiments/run_all_experiments.py`、`src/experiments/run_control_ablation.py`、`src/experiments/run_multi_seed.py`（如果它们直接读取 `ABLATION_SETTINGS`，阶段 A 只做兼容适配，不改行为）

### 阶段 B（组件化替换 main.py 分支）
- 修改：`src/main.py`（删除/收敛 sampler 初始化分支；λ 决策与后处理改为 runtime 组件调用）
- 修改：`src/agent/toolbox.py`（移除策略实现入口：不再读取 `exp_config` 做 lambda_policy；仅保留工具与权限）
- 新增：`src/experiments/components/`（建议目录，容纳 LambdaProvider/Postprocessor/TrainingStatePolicy/RollbackPolicy 的实现）
  - 例如：`lambda_providers.py`、`postprocessors.py`、`training_state.py`、`rollback.py`

### 阶段 C（切断 core->agent）
- 修改：`src/core/sampler.py`（移除 `from agent.config import AgentThresholds`）
- 新增或修改：`src/core/lambda_utils.py` 或 `src/utils/math.py`（承载纯函数 `calculate_lambda_t`）

### 阶段 D（full_model 组合式变体）
- 新增：`src/experiments/specs/full_model_base.py`
- 新增：`src/experiments/specs/full_model_variants.py`（或拆分多个文件）
- 修改：`src/experiments/ablation_config.py`（逐步降级为 alias/清单；或仅保留向后兼容入口）

---

## 附录 C：Runtime Context 契约（字段、来源、生命周期）
目标：让组件之间通过稳定、可测试的 context 交互，避免重新形成“隐式共享状态”。

### 1) RoundContext（每轮构造一次）
建议字段（最小集合）：
- `run_id: str`
- `experiment_name: str`
- `seed: int`
- `round_num: int`（与当前实现一致：1-based 或 0-based 必须统一；建议对齐当前 trace 的 round 写法）
- `labeled_count: int`
- `unlabeled_count: int`
- `total_budget: int`
- `query_size: int`
- `k_definition: str`
- `training_state: dict | None`（由 TrainingStatePolicy 产出）
- `ranking_metadata: dict | None`（由 Sampler/后处理产出，用于 trace）
- `score_stats: dict | None`（如需要：U/K 均值、方差等）

来源映射：
- run_id/experiment/seed：pipeline 初始化时确定
- labeled/unlabeled/test 计数：pool 加载后可得
- query_size：来自 config + 控制器决策（若允许）
- training_state：训练结束后由 policy 产出并写入 pipeline 状态

生命周期规则：
- `training_state`：在每轮训练结束后更新；选样时读取的是“上一轮训练结果”
- `ranking_metadata`：每轮 rank 后更新，仅用于当轮 trace/监控

### 2) SamplerContext（可选，偏 sampler 内部）
当某些 sampler 需要额外信息（例如 BALD 的 n_mc_samples）时，通过 spec 装配进 sampler，而不是在 pipeline 中以 `if sampler_type == ...` 方式分支。

---

## 附录 D：Trace Schema 与兼容规则（analysis/monitor 稳定性保障）
目标：重构期间不破坏现有分析脚本与监控脚本的读取逻辑。

### 1) 强制要求（重构期间不得破坏）
- 保持事件类型集合兼容：至少包含当前已有的 `initialized`、`round_summary`、`overfit_signal`、`lambda_policy_apply`、`l3_selection`（新增允许，但旧事件不得改名）。
- 在 `initialized` 事件写入一次：`trace_schema_version`（整数）与 `experiment_runtime`（组件名/版本 meta）。
- `round_summary` 中保持：
  - `sampler` 字段存在，且包含 `sampler_type`（字符串）与必要的审计字段
  - `lambda_controller` 字段允许为 None，但若存在需包含 `mode/lambda/round`

### 2) 版本化策略（建议）
- `trace_schema_version = 1`：现有字段（当前实现）
- `trace_schema_version = 2`：引入 runtime 组件 meta（不删除旧字段，仅补充）
- analysis/monitor 读取规则：
  - 优先读新版字段（若存在）
  - 否则 fallback 到旧字段（保持当前脚本可用）

### 3) manifest 与 trace 的对应
- manifest 记录 `legacy_ablation_dict`（可选）与 `runtime_components`（必选）
- trace 记录 `runtime_components` 的精简版（避免过大）

---

## 附录 E：Toolbox/Agent 边界与迁移规则（避免策略再度“回流”）
目标：Toolbox 只负责“工具调用 + 权限校验 + 运行态数据访问”，不负责策略实现。

### 1) 迁移前（现状）
- Toolbox 同时承担：阈值读取（含 overrides）+ lambda_policy 实现 + 控制状态缓存（`lambda_override_round`）

### 2) 迁移后（目标）
- LambdaProvider/AgentController 承担策略与阈值（由 spec 装配参数）
- Toolbox 承担：
  - `get_top_k_samples` / `get_sample_details` / `get_score_distribution` / `finalize_selection` 等工具
  - `control_permissions` 校验（禁止的 action 必须可解释且可降级）
  - 可选：保存 agent 决策产物（仅作为 trace meta，不参与决策）

### 3) 禁止事项（避免耦合反弹）
- 禁止 Toolbox 从 `controller.exp_config` 读取策略 key 决策
- 禁止在 pipeline 与 toolbox 同时维护“lambda override 的 state”（必须单点：LambdaProvider）

---

## 附录 F：弃用（deprecation）与回滚策略（确保可控演进）
目标：既能快速迁移到新架构，又保留研究复现与回退路径。

### 1) 弃用策略
- 阶段 A/B：`ABLATION_SETTINGS` 保留为唯一来源，但通过适配器装配 runtime（允许双写 meta：legacy 与 runtime）
- 阶段 C 起：新增实验必须以 spec 方式实现；`ABLATION_SETTINGS` 仅允许：
  - alias（指向 spec 名）
  - 或兼容条目（必须标注为 legacy-only，且不再扩展功能）
- 阶段 D 完成后：`ABLATION_SETTINGS` 降级为“实验清单/别名表”，不再承载策略参数（参数进入 spec 文件）

### 2) 回滚策略
- 保留一个运行期开关（建议在 Config 或 env 中）：
  - `AAL_SD_EXPERIMENT_RUNTIME=legacy|spec`
- 当出现行为漂移无法快速定位时，可切回 legacy 路径验证对齐，再逐步迁移定位差异点。
