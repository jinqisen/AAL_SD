# λ 调整策略与数据分析链路（基于 results 可复现）

本文回答两个问题：
1) 现有数据分析工具是否足够支撑 λ policy 调优？  
2) 是否需要额外分析 `results/pools` 与 `results/logs_md`，以及如何做“数据分布”类分析？

面向的实现与数据位置：
- 策略实现：`/Users/anykong/AD-KUCS/AAL_SD/src/agent/toolbox.py`
- 实验配置：`/Users/anykong/AD-KUCS/AAL_SD/src/experiments/ablation_config.py`
- 运行数据：`/Users/anykong/AD-KUCS/AAL_SD/results`
- Trace：`results/runs/<run_id>/*_trace.jsonl`

---

## 1. 现有工具是否足够？

结论分两层：

### 1.1 对“λ 控制闭环本身”的调优：基本足够（trace 为主）

只要你要回答的是：
- 哪一轮 λ 为什么上调/下调（触发了哪个 policy rule / cooling / guardrail）？
- λ 的轨迹与 `mIoU gain`、`rollback_flag`、`overfit_risk`、TVC 的关系如何？
- 选样时 U/K 的相对强弱是否导致策略误判？

那么 `*_trace.jsonl` 已经足够，原因是 trace 里包含：
- `lambda_policy_apply`（policy 的 applied/base/rule）
- `overfit_signal`（risk 与 TVC 汇总）
- `controller_step`（miou_delta、rollback_flag、最终 action.lambda）
- `selection_guardrail` / `lambda_guard`（安全阀与 clamp 的实际作用）
- `l3_selection`（top-k/selected 的 U/K/score 统计，能做“选中集合的分布”）

对应的现成分析脚本：
- `src/analysis/plot_strategy_trajectory.py`  
  - 导出 per-round CSV 与 `*_lambda_diagnostics.csv`
  - 输出 `overfit_risk vs mIoU gain`、`lambda vs tvc`、轨迹图等

### 1.2 对“数据分布偏置/数据切片”的调优：trace 不够，pools 才有用

如果你要回答的是：
- 选到的样本在 **mask 正样本面积比例**（或 has_positive）上是否偏置？
- 某些 round 选样是否过度倾向“全正/全负”或某一类难例？
- 不同实验之间选样重叠度、覆盖率、长尾覆盖如何？

这些属于“数据分布/数据切片”分析，trace 只给了 sample_id（整数）与 U/K（来自模型），并不包含 mask 统计。此时需要：
- `results/pools/<run_id>/_base/*.csv` 提供 sample_id → mask_path 映射
- 读取 `.h5` mask 才能计算分布指标（正像素比例、是否含正样本等）

`results/logs_md` 的作用更偏“人类可读汇总”，一般不建议作为主数据源（可用于快速浏览、交叉验证）。

---

## 2. 参数表（你关心的 λ policy 超参数）

| 参数 | 影响路径 | 效果 |
|---|---|---|
| LAMBDA_DELTA_UP | risk_control 阶段低风险时 λ 上调步长 | 越大→λ 上升越快 |
| LAMBDA_DELTA_DOWN | rollback/severe 时 λ 下调步长 | 越大→回撤时 λ 下降越猛 |
| LAMBDA_CLAMP_MIN/MAX | 硬边界 | 限制 λ 活动范围 |
| lambda_smoothing_alpha | EMA 平滑系数 | =1.0 无平滑，<1.0 惰性越大 |
| lambda_max_step | 单轮 λ 最大变化 | 限制突变幅度 |
| OVERFIT_RISK_HI | severe 判定门槛 | 越高→越难触发降 λ |
| OVERFIT_RISK_LO | 允许 λ 上调的风险门槛 | 越高→越难上调 λ |
| LAMBDA_DOWN_COOLING_ROUNDS | 连续下调冷却期 | >0 抑制连续下调 |
| late_stage_ramp | 后期 λ 地板线性抬升 | 强制后期 K 权重 |
| selection_guardrail | U 底线安全阀 | 选中样本 U 偏低时降 λ 或 quota_u |

建议把该表与每轮 trace 中的字段对齐（下节提供导出方式），避免“只看配置、不看实际 applied”。

---

## 3. 推荐的可复现分析路径（按问题类型选数据源）

### 3.1 问题：λ 为什么这样走？策略是否过敏/过钝？

数据源：`results/runs/<run_id>/<exp>_trace.jsonl`

导出方式：
```bash
python src/analysis/plot_strategy_trajectory.py \
  results/runs/<run_id> \
  --experiment_name <exp> \
  --export_csv
```

重点查看：
- `<exp>_lambda_diagnostics.csv` 中的  
  - `lambda_policy_rule`、`policy_ci_*`、`policy_in_cooling`  
  - `overfit_risk`、`grad_tvc_last`、`miou_delta/miou_gain`
- 图：`lambda_vs_tvc`、`overfit_vs_miou_gain`、trajectory

### 3.2 问题：选样集合的 U/K 分布是否支持当前决策？

数据源：trace 中的 `l3_selection.top_items/selected_items`

优点：
- 不需要重新跑模型，也不需要 pools
- 能直接对齐“被选中集合”和“前 topK 候选”的 U/K/score 统计

限制：
- 只覆盖 top-k（默认 256）及 selected 的分布，不是全池分布

### 3.3 问题：是否出现“数据分布偏置”（比如 mask 正样本面积分布漂移）？

数据源：
- `results/runs/<run_id>/<exp>_trace.jsonl`（提供每轮 selected_ids）
- `results/pools/<run_id>/_base/*.csv`（提供 sample_id → mask_path）
- `.h5` mask 本体（提供像素级统计）

已补齐的离线分析脚本：
- `src/analysis/analyze_selection_mask_distribution.py`

用法示例：
```bash
python src/analysis/analyze_selection_mask_distribution.py \
  results/runs/<run_id>/<exp>_trace.jsonl \
  --output results/runs/<run_id>/_analysis
```

输出：
- `<run_id>_<exp>_mask_distribution_by_round.csv`  
  - `selected_has_positive_rate`：本轮新增样本含正比例  
  - `selected_positive_frac_*`：本轮新增样本正像素比例分位数/均值  
  - `labeled_*`：累计 labeled 的相同统计  
  - `train_masks_sha256`：用于确认分析对应的数据版本

### 3.4 是否需要 `logs_md`？

一般不需要作为“分析主数据源”，原因：
- `logs_md` 是对运行结果的汇总文本，丢失了细粒度字段
- 你真正要做相关性、触发频次统计、分位数比较时，还是得回到 trace / csv

它适合用于：
- 快速浏览某个 run 的整体结果
- 与 trace 导出的结论做 sanity check

---

## 4. 两条最常用的“调参决策链路”（建议你固定成 SOP）

### 4.1 λ 偏大导致 U starvation（topK/selected 的 U 偏低）

证据优先级：
1) trace `selection_guardrail` 触发频次（是否频繁 step_down）  
2) `l3_selection` 的 `selected_items.uncertainty` 分布（p50/p75 是否偏低）  
3) mIoU gain 与 rollback 是否同步变差

对应调参顺序：
1) 先调 `selection_guardrail`（阈值与 quota_u）  
2) 再调 `late_stage_ramp`（降低 end_lambda 或延后 start_round）  
3) 最后才动 `LAMBDA_DELTA_*`（避免把动态响应问题误当成目标函数问题）

### 4.2 过拟合/不稳定导致 λ 被频繁压低（K 利用不足）

证据优先级：
1) trace `severe_overfit_lambda_down` 触发频次、`policy_in_cooling` 时长  
2) `overfit_risk` 与 `grad_tvc_last` 的分布（是否大量进入负对齐）  
3) mask 分布是否被选到“极端样本”（正像素极少/极多导致训练不稳）

对应调参顺序：
1) 若 probe 改成 `train_holdout`，优先确认 risk 尺度是否变化（必要时用 CI 或重标定 HI/LO）  
2) 调 `OVERFIT_RISK_HI/LO` 与 `risk_ci_window/quantile`（控制触发率）  
3) 调 `LAMBDA_DOWN_COOLING_ROUNDS` / `lambda_smoothing_alpha`（控制振荡）

