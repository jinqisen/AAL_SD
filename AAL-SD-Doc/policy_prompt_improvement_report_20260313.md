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

### 2.3 离线回填的 l3 统计：U/K 中位数轨迹（新增证据）

离线工具已在 `results/runs/<run_id>/reports/` 生成 `backfill_l3_selection_stats_*.csv`（每个实验一份），并在每个 run 下生成汇总：
- seed42（20260311）汇总：[`backfill_l3_selection_stats_summary.csv`](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed42/reports/backfill_l3_selection_stats_summary.csv)
- seed43（20260311）汇总：[`backfill_l3_selection_stats_summary.csv`](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed43/reports/backfill_l3_selection_stats_summary.csv)
- seed42（20260309）汇总：[`backfill_l3_selection_stats_summary.csv`](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed42/reports/backfill_l3_selection_stats_summary.csv)
- seed43（20260309）汇总：[`backfill_l3_selection_stats_summary.csv`](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed43/reports/backfill_l3_selection_stats_summary.csv)

关键发现 1（对之前推断的反证）：在 `ramp_guardrail` 的后期轮次，`selected_ids` 命中 `top_items` 的覆盖率显著下降，导致 `u_median_top/k_median_top` 以及基于 `selected_items` 的统计都不再能代表“全选中集合”
- 新增一致性指标：`coverage_selected_in_top = |selected_ids ∩ top_items| / |selected_ids|`，以及 `selected_scored_frac_u/k`（选中集合里能拿到 U/K 的比例）
- seed43 `ramp_guardrail`：round10 覆盖率≈0.1818，round13 覆盖率=0.0，round15 覆盖率≈0.0455（见 [`backfill_l3_selection_stats_full_model_A_lambda_policy_ramp_guardrail.csv`](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed43/reports/backfill_l3_selection_stats_full_model_A_lambda_policy_ramp_guardrail.csv) 的 `coverage_selected_in_top/selected_scored_frac_*` 列）
- seed42 `ramp_guardrail`：round10 覆盖率≈0.2386，round11 覆盖率=0.0，round15 覆盖率≈0.0341（见 [`backfill_l3_selection_stats_full_model_A_lambda_policy_ramp_guardrail.csv`](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed42/reports/backfill_l3_selection_stats_full_model_A_lambda_policy_ramp_guardrail.csv)）

结论（更严格）：当覆盖率显著低于 1.0 时，`u_median_top/k_median_top` 只能说明“记录到的那部分 top_items 统计极端”，不能推导“全未标注池 topk 前沿退化”；同理，当 `selected_scored_frac_u/k` 很低时，`u_median_selected/k_median_selected` 只基于少量命中样本计算，不能代表全选中集合。

关键发现 2（对之前结论的校正）：不同实验的 `topk` 与 `selected` 统计一致性差异较大，说明“l3 日志的 top_items/selected_items 可能不是同一个排序空间”，因此它更像“调试视角的候选快照”而不是严格意义上的“全池 topk”
- 在部分 full_model 实验中，`u_median_top == u_median_selected` 且 `k_median_top == k_median_selected`（例如 seed43 full_model 的多个轮次），更像“候选 cache 被当作 topk”而非全量 pool 排序结果（见 [`backfill_l3_selection_stats_full_model_A_lambda_policy.csv`](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed43/reports/backfill_l3_selection_stats_full_model_A_lambda_policy.csv)）。

解读（更严格）：在 agent 模式下，这组 l3 轨迹目前只能作为“近似信号”，适合做“异常侦测/触发器候选”，不适合作为定量阈值直接驱动 λ 策略闭环。若要让它可用于闭环，应首先保证两件事：
- 覆盖率：`coverage_selected_in_top` 接近 1（否则 top/selected 统计都会偏置）
- 选中集合可度量：能从真实 `selected_ids` 上直接计算 U/K 分布（而非通过 top_items 命中推断），例如强制记录 `selected_score_stats`（U/K 的 mean/p50/p75 等）

关键发现 3：20260309 的 baseline 覆盖不完整并非离线工具问题，而是原始 trace 未提供所需信号
- `baseline_random/bald/entropy/coreset` 这类策略无法回填 u/k 中位数（在汇总中常见 `rounds_with_stats=0`），因为它们原始 trace 中没有 l3 级别分解或缺少 K 指标。
- `baseline_dial_style/baseline_wang_style` 与 full_model 系列可回填（常见 `rounds_with_stats=15`），因为它们在实现里显式计算并记录了 U/K 相关结构。

## 3. Policy 改进空间（按优先级）

### P0：把 late-stage ramp 从“固定日程”改为“条件触发 + 风险联动”

依据：设计文档的 `diagnose()` 规则中已明确了 `late_stage_plateau`（late_stage_gain）、`over_conservative`（lambda_max）等判据。  
建议：
- 触发：`late_stage_gain` 低且 `overfit_risk` 低、TVC 稳定时才 ramp
- ramp 终点/斜率与风险耦合：风险偏高时降低 `end_lambda` 或延后 ramp 起点
- 修订：不要在现阶段用 `u_median_top/k_median_top` 做闭环触发（因为 ramp 的后期覆盖率显著下降，见 2.3），最多作为离线告警；闭环触发应基于可靠信号（late_stage_gain/overfit_risk/TVC 等）

### P1：把 guardrail 从“硬阈值”改为“自适应阈值 + 配额联动”

当前 guardrail 的 `u_median_min/u_low_frac_max` 作为常数，可能随 seed/轮次漂移。  
建议：
- `u_median_min` 使用最近 k 轮或 topk 分布的分位数动态估计
- 将 “fallback_quota_u_frac” 与 “lambda_step_down” 解耦，避免一触发就过度降 λ
- 修订：在修复覆盖率与“选中集合可度量”之前，不建议引入基于 topk 的硬约束；优先把 guardrail 做成“直接在 selected_ids 上计算 U 分布”的约束（以 `selection_guardrail.stats_before/stats_after` 为准），并将 topk 统计仅用于提示“是否需要检查排序空间一致性”

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
- 2026-03-13：基于离线回填 CSV（u/k 中位数轨迹）更新策略结论与风险点

## 8. 证据对照表（关键论断 → 原始数据锚点）

本节把报告中的关键论断逐条映射到可复核的数据文件与字段/检索锚点，便于审稿式核查。

### 8.1 口径与指标

- 论断：`performance_history[*].mIoU`（每轮“选模 checkpoint 的 val mIoU”，也是 ALC 的计算口径）与 `final_miou`（最终输出口径）可能不一致，不能混称“最终 mIoU”  
  - 原始数据：四个 run 的 `experiment_results.json`  
    - ALC/曲线口径：`performance_history[*].mIoU`（以及 `labeled_size/model_selection/selected_epoch`）  
    - 选模口径补充：`performance_history[*].best_val_mIoU`、`best_val_epoch`（同轮训练过程内的 best-val）  
    - 最终输出口径：`final_miou`（优先来自最终报告 `final_report.mIoU@test_split`；若最终报告不可用则回退为最后一轮选模的 `performance_history[-1].mIoU@val`）  
    - seed42-0309：[experiment_results.json](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed42/experiment_results.json)  
    - seed43-0309：file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed43/experiment_results.json  
    - seed42-0311：file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed42/experiment_results.json  
    - seed43-0311：file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed43/experiment_results.json  
  - 交叉印证：`detailed_results_report.md` 的“性能历史”表（每轮 best_val）与 `summary_report.md` 的“最终 mIoU”列  
  - 代码口径来源：[`main.py`](file:///Users/anykong/AD-KUCS/AAL_SD/src/main.py) 的结果聚合逻辑（ALC 使用 `performance_history[*].mIoU`；最终报告使用 `test_split` 的 `final_report`，并在不可用时回退）  

### 8.2 Full model vs Baseline（跨 seed）结论

- 论断：seed42（20260309）中 `baseline_dial_style(final_miou)` 高于 `full_model_A_lambda_policy(final_miou)`  
  - 原始数据：seed42-0309 [summary_report.md](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed42/summary_report.md)（“最终 mIoU”表）  
  - 可复核字段：seed42-0309 [experiment_results.json](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed42/experiment_results.json)（各实验 `final_miou`）  

- 论断：seed43（20260309）中 `baseline_random(final_miou)` 显著高于 `full_model_A_lambda_policy(final_miou)`  
  - 原始数据：seed43-0309 [summary_report.md](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed43/summary_report.md)  
  - 可复核字段：seed43-0309 `experiment_results.json`（各实验 `final_miou`）  

- 论断：Full model 的 ALC 在对应 run 内排名更靠前（但 final_miou 未必领先）  
  - 原始数据：对应 run 的 `summary_report.md`（“ALC 排名”）与 `experiment_results.json`（字段：`alc`/`final_miou`）  

### 8.3 Full model vs 变体结论

- 论断：20260309 中 `full_model_B_lambda_agent` 与 `ab_tune_*` 的收益不稳定/整体不显著  
  - 原始数据：seed42-0309 与 seed43-0309 的 [summary_report.md](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed42/summary_report.md)、[summary_report.md](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed43/summary_report.md)（ALC 与 final_miou 对比）  
  - 可复核字段：对应 `experiment_results.json` 中 `full_model_B_lambda_agent`、`full_model_A_lambda_policy_ab_tune_*` 的 `final_miou/alc`  

- 论断：20260311 中 ramp_guardrail 在 seed42 有收益但 seed43 明显失败（跨 seed 不鲁棒）  
  - 原始数据：seed42-0311 [summary_report.md](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed42/summary_report.md)、seed43-0311 [summary_report.md](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed43/summary_report.md)  
  - 可复核字段：对应 `experiment_results.json` 中 `full_model_A_lambda_policy` 与 `full_model_A_lambda_policy_ramp_guardrail` 的 `final_miou`  

### 8.4 离线回填 l3 统计（新增证据）与“需要审视”的依据

- 论断：离线回填会为每个实验生成 `backfill_l3_selection_stats_<exp>.csv`，并产出 `summary.csv`  
  - 原始数据（汇总入口）：  
    - seed42-0311：[backfill_l3_selection_stats_summary.csv](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed42/reports/backfill_l3_selection_stats_summary.csv)  
    - seed43-0311：file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed43/reports/backfill_l3_selection_stats_summary.csv  
    - seed42-0309：file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed42/reports/backfill_l3_selection_stats_summary.csv  
    - seed43-0309：file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed43/reports/backfill_l3_selection_stats_summary.csv  

- 论断：ramp_guardrail 的 `u_median_top/k_median_top` 在部分后期轮次出现极端值（但不能直接等同“全池前沿退化”）  
  - 原始数据：  
    - seed42-0311 ramp_guardrail：[`backfill_l3_selection_stats_full_model_A_lambda_policy_ramp_guardrail.csv`](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed42/reports/backfill_l3_selection_stats_full_model_A_lambda_policy_ramp_guardrail.csv)（列：`u_median_top/k_median_top/u_median_selected/k_median_selected`）  
    - seed43-0311 ramp_guardrail：[`backfill_l3_selection_stats_full_model_A_lambda_policy_ramp_guardrail.csv`](file:///Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed43/reports/backfill_l3_selection_stats_full_model_A_lambda_policy_ramp_guardrail.csv)  
  - 需要审视的证据点（报告中引用）：在 ramp_guardrail 的后期轮次，新增的一致性列 `coverage_selected_in_top` 显著下降（甚至为 0），因此 topk/selected 的中位数不能代表“全选中集合”或“全池前沿”  

- 论断：20260309 的部分 baseline（random/bald/entropy/coreset）回填为 0 并非离线工具失败，而是原始 trace 不含必要信号  
  - 原始数据：对应 run 的 `summary.csv` 里 `rounds_with_stats=0` 的实验行 + 该实验 trace 中缺少 `l3_selection`/K 指标事件  

### 8.5 Policy 改进项（每条建议对应的可验证证据）

- P0：late-stage ramp 从固定日程改为“条件触发 + 风险联动”  
  - 可验证信号（设计框架定义）：`late_stage_gain`（final - round8）、`overfit_risk`、`tvc_*`、`lambda_trajectory/lambda_volatility`（参见 [auto_tuning_framework_design.md](file:///Users/anykong/AD-KUCS/AAL_SD/AAL-SD-Doc/auto_tuning_framework_design.md) 的 diagnostics/diagnose 表）  
  - 原始事件锚点：各实验 `*_trace.jsonl` 中的 `round_summary`、`controller_step.state`（字段含 `overfit_risk/grad_train_val_cos_*` 等）与 `lambda_policy_apply`  
  - 辅助证据：离线回填 CSV（仅作告警级别，不作硬约束）中的 `u_median_top/k_median_top` 极端波动轮次  

- P1：guardrail 改为自适应阈值 + 配额联动，并补“前沿退化监控”（先告警后闭环）  
  - 原始事件锚点：`*_trace.jsonl` 中 `selection_guardrail` 事件（字段：`stats_before/stats_after/thresholds`）  
  - 评估方式：对比 guardrail 触发轮次前后 `selected` 集合的 `u_median/frac_u_lt` 变化，以及对应轮次 `mIoU` 变化  
  - 离线回填支撑：`backfill_l3_selection_stats_*.csv` 新增列 `coverage_selected_in_top`、`selected_scored_frac_u/k`，用于判定 topk/selected 统计是否可用；当这些一致性指标较低时，topk 仅用于提示而不用于约束  

- P2：轮内振荡（epoch 级）纳入风险与稳定性闭环  
  - 可验证信号（设计框架定义）：`epoch_miou_volatility`、`tvc_sign_flip_rate`（参见 [auto_tuning_framework_design.md](file:///Users/anykong/AD-KUCS/AAL_SD/AAL-SD-Doc/auto_tuning_framework_design.md)）  
  - 原始事件锚点：`*_trace.jsonl` 中 `epoch_end`（轮内序列）与 `controller_step.state.grad_*`（若启用）  

### 8.6 Prompt 改进项的前置条件（可证伪的可用性判据）

- 论断：若 LLM 未参与决策，则 prompt 改动无法解释性能差异  
  - 原始证据：`reports/*anomaly_report*.md` 中 “LLM Client not available” 记录、以及 trace 中 `selection.context.llm_mode` / `llm_degraded` 事件（若触发）  
  - 可证伪条件：提供 LLM on/off 的严格消融（同 seed/同预算/同配置，仅开关 LLM 或 prompt）并在 trace 中出现明确的 LLM 调用/模式标记  
