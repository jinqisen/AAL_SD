# AAL-SD（合并版）实验/复现/实现对齐说明

本文档以当前代码实现为准，目标是把 `.trae/documents/` 中分散且重复的计划、报告与修复说明收敛为单一可执行版本。

## 1. 快速开始

### 1.1 自检

```bash
python -m pytest -q
```

### 1.2 跑一组严格并发实验

```bash
python src/run_parallel_strict.py
```

### 1.3 续跑已有 run

```bash
python src/run_parallel_strict.py --resume <run_id>
```

## 2. 目录与产物约定（以代码为准）

### 2.1 关键入口

- 主流程：[`src/main.py`](file:///Users/anykong/AAL_SD/src/main.py)（`ActiveLearningPipeline`）
- 并发跑批：[`src/run_parallel_strict.py`](file:///Users/anykong/AAL_SD/src/run_parallel_strict.py)
- 消融矩阵：[`src/experiments/ablation_config.py`](file:///Users/anykong/AAL_SD/src/experiments/ablation_config.py)
- 监控与恢复：[`src/monitor_and_recover.py`](file:///Users/anykong/AAL_SD/src/monitor_and_recover.py)
- 一致性检查：[`inspect_integrity.py`](file:///Users/anykong/AAL_SD/inspect_integrity.py)

### 2.2 run 目录布局

- 运行状态与追踪：`results/runs/<run_id>/`
  - `<exp>_status.json`：进度、关键指标、运行参数快照
  - `<exp>_trace.jsonl`：事件流（初始化、每轮训练/选样、回退与异常等）
  - `manifest.json`：run 级配置与指纹（用于复现与审计）
- Checkpoint：`results/checkpoints/<run_id>/<exp>_state.json`
- Pools（以代码为准）：`results/pools/<run_id>/<exp>/`
  - `labeled_pool.csv`
  - `unlabeled_pool.csv`
  - `test_pool.csv`

## 3. 训练与预算口径（以配置为准）

配置来源：[`src/config.py`](file:///Users/anykong/AAL_SD/src/config.py)

- 训练协议：`FIX_EPOCHS_PER_ROUND=True`，默认 `EPOCHS_PER_ROUND=10`
- 初始标注比例：`INITIAL_LABELED_SIZE=0.05`
- 总预算：`TOTAL_BUDGET=int(ESTIMATED_TOTAL_SAMPLES * BUDGET_RATIO)`，默认 `BUDGET_RATIO=0.4`
- 轮数：`N_ROUNDS=15`
- Query size：默认按 `(TOTAL_BUDGET - initial_estimate) / N_ROUNDS` 自动估算（见 `Config.QUERY_SIZE`）
- 评估切分：`AAL_SD_EVAL_SPLIT`（默认 `internal`）

## 4. 实验矩阵（以 ablation_config 为准）

实验名称是稳定接口：跑批脚本与恢复逻辑都以 `experiment_name` 为键。完整配置见 [`ABLATION_SETTINGS`](file:///Users/anykong/AAL_SD/src/experiments/ablation_config.py#L4-L574)。

### 4.1 主方法与变体（Agent + AD-KUCS）

- `full_model`：主方法默认配置（warmup + 风险闭环 λ + rollback，自带 L3 logging）
- `full_model_policy_lambda` / `full_model_default_thresholds`：阈值/策略对照
- `full_model_v3_*` / `full_model_v4_*` / `full_model_v5_*`：鲁棒性与风险策略强化变体

### 4.2 关键消融（对照“为什么需要 LLM/策略闭环”）

- `no_agent`：移除 Agent，仅 AD-KUCS 数值路径（λ 非 Agent 决策）
- `fixed_lambda`：固定 `λ=0.5`（对应“无动态策略”的核心对照）
- `random_lambda`：随机控制 λ（无 LLM）
- `rule_based_controller_r1` / `rule_based_controller_r2` / `rule_based_controller_r3`：规则控制器对照（对齐论文“LLM vs 规则”）
- `no_cold_start`：移除不确定性-only 与 warmup，直接进入闭环
- `fixed_k`：固定 K 阶段定义（`k_definition=coreset_to_labeled_fixed`）
- `no_normalization`：关闭 U/K 归一化（`score_normalization=False`）
- `uncertainty_only`：固定 `λ=0.0`
- `knowledge_only`：固定 `λ=1.0`

### 4.3 基线（非 Agent）

- `baseline_random`
- `baseline_entropy`
- `baseline_coreset`
- `baseline_bald`
- `baseline_dial_style`
- `baseline_wang_style`
- `baseline_llm_us` / `baseline_llm_rs`（LLM 驱动但不走本方法闭环，用于“LLM 引入本身”对照）

### 4.4 Agent 控制权限消融

- `agent_control_lambda`：只允许 `set_lambda`
- `agent_control_budget`：只允许 `set_query_size`
- `full_model_fixed_epochs_lambda_budget`：固定 epochs，仅允许控制 λ 与 query_size

多随机种子默认实验清单见 [`MULTI_SEED_DEFAULTS["paper_experiments"]`](file:///Users/anykong/AAL_SD/src/experiments/ablation_config.py#L576-L597)。

## 5. AD-KUCS 定义（以实现为准）

实现入口：[`src/core/sampler.py`](file:///Users/anykong/AAL_SD/src/core/sampler.py)

### 5.1 不确定性 U(x)

- 定义：像素级熵的均值（实现为 `log2` 熵）  
- 位置：`ADKUCSSampler._calculate_uncertainty`

### 5.2 知识增益 K(x)：coreset-to-labeled

- 定义：`K(x)=min_{l∈L} ||f_x - f_l|| / max_pairwise_dist(L)`（归一化后越大越“新颖/覆盖不足”）
- 约束：已标注集为空视为实验/数据错误，直接报错终止（避免“无锚点 K”导致结论失效）
- 位置：`ADKUCSSampler._coreset_to_labeled_scores`

### 5.3 融合与调度

- 融合：`Score(x) = (1-λ)·U(x) + λ·K(x)`
- 默认 λ：随进度的 sigmoid（`AgentThresholds.calculate_lambda_t(progress, alpha)`）
- U/K 默认 min-max 归一化（可由实验关闭）

## 6. Agent 集成与工具边界（以实现为准）

- Prompt 模板：[`src/agent/prompt_template.py`](file:///Users/anykong/AAL_SD/src/agent/prompt_template.py)
  - System Prompt 会显示 `control_permissions` 并收敛可用动作空间
- 工具箱：`Toolbox`（权限默认拒绝，显式授权；最终由 `finalize_selection` 提交）

在科研严格模式下，相关开关位于 [`src/config.py`](file:///Users/anykong/AAL_SD/src/config.py#L219-L222)，用于控制回退与降级是否允许进入结果。

## 7. 恢复与一致性（以实现为准）

核心目标是让恢复具备幂等性，且避免 “CSV 超前于 Checkpoint” 的未来数据污染。

- Pools 路径由 `ActiveLearningPipeline._resolve_pools_dir` 统一解析：[`main.py`](file:///Users/anykong/AAL_SD/src/main.py#L398-L413)
- 状态与 trace 写入均为 `.tmp -> replace` / append 语义，恢复时以 checkpoint 与状态为准进行回滚与截断
- 推荐排错流程：先看 `<exp>_status.json` 与 `<exp>_trace.jsonl`，再运行 `inspect_integrity.py <run_id>`
