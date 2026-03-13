# AAL-SD Policy & Prompt 改进空间分析（基于 4 个 run）

生成时间：2026-03-13  
范围：
- `baseline_20260311_201728_seed42`
- `baseline_20260311_201728_seed43`
- `baseline_20260309_211601_seed42`
- `baseline_20260309_211601_seed43`

参考框架：[`auto_tuning_framework_design.md`](file:///Users/anykong/AD-KUCS/AAL_SD/AAL-SD-Doc/auto_tuning_framework_design.md)（指标体系 + diagnose 规则 + 调优逻辑）

## 0. 指标口径与数据源说明

本仓库结果目录中存在两类“最终指标”来源：
- `performance_history` / `detailed_results_report.md` 的每轮 mIoU：来自 `best_val`（val 选模）轨迹
- `experiment_results.json` 的 `final_miou`：汇总口径的最终指标（在多处报告表格中被展示为“最终 mIoU”）

因此会出现 `round 16 mIoU`（val）高于 `final_miou` 的情况，这是 val→final 口径差异导致的现象，不应直接判为数据错误。若要严格对齐，应在分析中明确使用字段名（`best_val_mIoU` vs `final_miou`）。

## 1. 结果总览：Full model vs Baseline（跨 seed）

### 1.1 seed42（20260309，含 baseline 与对照/变体）

证据：
- [`summary_report.md`](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed42/summary_report.md)
- [`baseline_comparison_report.md`](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed42/baseline_comparison_report.md)

结论（按 `final_miou`）：
- best baseline：`baseline_dial_style`（0.7160）优于 `full_model_A_lambda_policy`（0.6994）
- `full_model_A_lambda_policy` 的 ALC 领先（0.6101），但最终 `final_miou` 未能稳定压过强 baseline

解读（按调优逻辑）：
- 学习效率（ALC）高但最终未领先，符合“后期增长不足/late-stage plateau”的典型特征：早期 U 拉升快，但后期 K 引入强度与风险联动不足，导致 test/汇总口径无法追平某些 baseline 的长尾收益。

### 1.2 seed43（20260309，含 baseline 与对照/变体）

证据：[`summary_report.md`](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed43/summary_report.md)

结论（按 `final_miou`）：
- best baseline：`baseline_random`（0.7390）显著优于 `full_model_A_lambda_policy`（0.7212）
- `full_model_A_lambda_policy` 的 ALC 仍领先（0.6112）

解读：
- 若 baseline_random 在该 seed 下表现异常强，说明“选择策略”更依赖数据分布/seed 偶然性；Full 的优势体现为曲线更稳（ALC/早期增益），但仍需要强化后期策略以减少跨 seed 被“偶然强 baseline”反超的概率。

## 2. Full model vs 变体：有效方向与不鲁棒点

### 2.1 20260309：ab_tune 与 lambda_agent 的总体收益不明显

证据：[`summary_report.md`](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed42/summary_report.md)、[`summary_report.md`](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed43/summary_report.md)

观察：
- `full_model_B_lambda_agent`（显式 set_lambda）在两个 seed 上均未体现优势
- `ab_tune_hi_ep10 / ab_tune_lo_ep10` 对 `final_miou` 与 ALC 的提升不稳定

推断：
- “放权给 agent 显式调 λ”在信号不足（如 K/U 轨迹、风险统计不全）时更容易出现抖动或保守，最终收益不如 policy 闭环稳定。

### 2.2 20260311：ramp+guardrail 在 seed42 有收益，但跨 seed 不鲁棒

证据：
- seed42：[`summary_report.md`](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed42/summary_report.md)、[`detailed_results_report.md`](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed42/detailed_results_report.md)
- seed43：[`summary_report.md`](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed43/summary_report.md)、[`detailed_results_report.md`](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed43/detailed_results_report.md)
- ramp/guardrail 的实际参数在 trace 初始化段可见：[`full_model_A_lambda_policy_ramp_guardrail_trace.jsonl`](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed42/full_model_A_lambda_policy_ramp_guardrail_trace.jsonl)

观察（按 `final_miou`）：
- seed42：`ramp_guardrail`（0.7098）> `full_model`（0.6991）
- seed43：`ramp_guardrail`（0.6644）<< `full_model`（0.7007）
- `train_probe` 两 seed 均偏弱（~0.68）

推断：
- `late_stage_ramp` 的强推策略能解决后期 plateau，但如果 ramp 与 overfit 风险/val→test gap 的联动不足，就会出现 seed 敏感（某些 seed 下过推 K、导致后期退化）。
- `train_probe` 改变 TVC/风险信号统计性质，容易导致风险门控误判，进而影响 λ 决策稳定性。

## 3. Policy 改进空间（按优先级）

### P0：把 late-stage ramp 从“固定日程”改为“条件触发 + 风险联动”

依据：设计文档的 `diagnose()` 规则中已明确了 `late_stage_plateau`（late_stage_gain）、`over_conservative`（lambda_max）等判据。  
建议：
- 触发：`late_stage_gain` 低且 `overfit_risk` 低、TVC 稳定时才 ramp
- ramp 终点/斜率与风险耦合：风险偏高时降低 `end_lambda` 或延后 ramp 起点

### P1：把 guardrail 从“硬阈值”改为“自适应阈值 + 配额联动”

当前 guardrail 的 `u_median_min/u_low_frac_max` 作为常数，可能随 seed/轮次漂移。  
建议：
- `u_median_min` 使用最近 k 轮或 topk 分布的分位数动态估计
- 将 “fallback_quota_u_frac” 与 “lambda_step_down” 解耦，避免一触发就过度降 λ

### P2：把轮内振荡（epoch 级）纳入风险与稳定性闭环

建议把 `epoch_miou_volatility`、`tvc_sign_flip_rate`（设计文档已定义）接入 diagnose→policy 映射：当轮内振荡显著时，下轮限制 λ 上升或减少 epochs（若 epochs 可调）。

## 4. Prompt 改进空间（前提：LLM 链路可用）

在多个报告中出现 “LLM Client not available”，且 trace 中 `context.llm_mode` 多为空，说明实验中存在 LLM 未参与决策的情况。prompt 优化只有在 LLM 链路稳定可用后才具备收益。

建议（prompt 结构层面）：
- 将输入拆为三块：`diagnostics`、`issues`、`recent_history`，并要求输出“可裁剪的参数建议”（按白名单与范围）
- 强制输出：主要瓶颈一句话 + 3 条按优先级排序的建议（每条含预期收益与风险）
- 将“禁止越权/禁止 set_lambda”等权限明确写入 system prompt，并要求给出“在权限内的替代动作”（例如只能通过 finalize_selection 间接影响 λ）

## 5. 可观测性缺口：l3_selection 与 lambda_effective 的结论与整改

### 5.1 l3_selection “看不到”多为检索/可视化问题

原因：
- `l3_selection` 单行 JSON 体积巨大（含 top_items/selected_items 数组），IDE/grep 输出经常截断，导致误以为未采集。

整改（降低检索成本，不改策略）：
- 在写入 `l3_selection` 的同时额外写入一条轻量事件 `l3_selection_stats`，包含：
  - `u_median_selected/k_median_selected`
  - `u_median_top/k_median_top`
  - `topk/selected_limit/source`

落点：[`main.py`](file:///Users/anykong/AD-KUCS/AAL_SD/src/main.py) 的 `_append_l3_selection()`。

### 5.2 selection 事件中 lambda_effective=null：记录口径缺失（instrumentation gap）

原因：
- `selection` 事件的 `lambda_effective` 来自 `_last_ranking_metadata`；在 agent_finalize_selection 路径下，可能只补了 `_last_ranked_items`，未补 `_last_ranking_metadata`，导致 `_sampler_audit()` 写入 null。

整改（不改策略，只补 trace）：
- 在 `_sampler_audit()` 中，当 ranking_meta 缺失时，按优先级从同轮控制事件回填：
  1) `lambda_guard.lambda_after`
  2) `lambda_override.applied`
  3) `lambda_policy_apply.applied`
  4) 最后兜底从 `toolbox.control_state.lambda_override_round` 取值

落点：[`main.py`](file:///Users/anykong/AD-KUCS/AAL_SD/src/main.py) 的 `_sampler_audit()`。

## 6. 本次实现落地与验证建议

已落地改动（不改变策略/训练，仅增强 trace 观测）：
- `selection` 事件补齐 `lambda_effective/lambda_source` 的回填逻辑
- 新增 `l3_selection_stats` 轻量事件（便于形成 u/k trajectory）

验证建议：
- 运行单测：`python -m pytest -q`
- 对任意 run 的 trace，grep `l3_selection_stats` 应每轮可见；grep `selection` 的 `lambda_effective` 不应在 agent 路径下为 null

## 7. 进展

- 2026-03-13：完成 trace 观测缺口整改（lambda_effective 回填、l3_selection_stats 事件）
- 2026-03-13：完成本报告初版输出

