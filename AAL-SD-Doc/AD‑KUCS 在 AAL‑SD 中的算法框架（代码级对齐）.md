# AD‑KUCS 在 AAL‑SD 中的算法框架（代码级对齐）

本文基于仓库 `/Users/anykong/AD-KUCS/AAL_SD/src` 的实现，给出 AD‑KUCS 在 AAL‑SD 框架里的**整体流程**与**control frame（闭环控制）**细节，重点说明：
- 控制指标（观测信号）有哪些、如何计算；
- control policy（尤其是 λ policy）是什么；
- control policy 中有哪些关键超参数，以及它们如何影响最终用于选样的 λ。

---

## 1. 总览：Round‑based 主动学习闭环

AAL‑SD 采用标准的 Round‑based Active Learning（AL）闭环：

1. 在当前 labeled pool 上训练若干 epoch；
2. 在 val 上评估得到 mIoU / F1；
3. 从训练过程与评估过程提取“过拟合/不稳定”诊断信号（TVC、overfit_risk、rollback 等），形成 `training_state`；
4. 用 AD‑KUCS 对 unlabeled pool 计算 `U(x), K(x)`，并用 `Score(x) = (1-λ)·U(x) + λ·K(x)` 排序；
5. 在预算约束下选择 query_size 个样本加入 labeled pool，进入下一轮。

主循环实现在：
- `src/main.py`：[main.py](file:///Users/anykong/AD-KUCS/AAL_SD/src/main.py)

---

## 2. 核心模块与职责划分

### 2.1 Sampler（AD‑KUCS 的 U/K/Score 与排序）

- 实现类：`ADKUCSSampler`
- 位置：[`src/core/sampler.py`](file:///Users/anykong/AD-KUCS/AAL_SD/src/core/sampler.py)

关键职责：
- 从模型预测概率图/特征中计算不确定性 `U(x)`；
- 从 unlabeled 特征分布中计算代表性/知识增益 `K(x)`；
- 计算融合分数 `Score(x)` 并排序。

### 2.2 Trainer（训练、评估与梯度诊断信号）

- 位置：[`src/core/trainer.py`](file:///Users/anykong/AD-KUCS/AAL_SD/src/core/trainer.py)

关键职责：
- `train_one_epoch()` 训练并可选生成梯度诊断 payload；
- `evaluate()` 在 val/test 上评估 mIoU / F1；
- 梯度诊断的核心是 train‑val gradient cosine（TVC），作为过拟合风险的代理指标。

### 2.3 Agent/Toolbox（control frame、λ policy 与 guardrails）

当实验开启 `use_agent=True` 时：

- `Toolbox` 负责提供观测工具与控制动作接口，并实现 `lambda_policy` 与选样 guardrail。
- 位置：[`src/agent/toolbox.py`](file:///Users/anykong/AD-KUCS/AAL_SD/src/agent/toolbox.py)
- prompt 模板：[`src/agent/prompt_template.py`](file:///Users/anykong/AD-KUCS/AAL_SD/src/agent/prompt_template.py)
- 管理器：[`src/agent/agent_manager.py`](file:///Users/anykong/AD-KUCS/AAL_SD/src/agent/agent_manager.py)

---

## 3. 每轮（Round）详细数据流（代码路径）

### 3.1 训练与评估（生成 epoch‑level 轨迹）

在每个 round（非最终 test‑only round）：
- 循环 `EPOCHS_PER_ROUND`：
  - `trainer.train_one_epoch(labeled_loader, grad_probe_loader=...)`
  - `trainer.evaluate(val_loader)`
- 训练过程中会记录每个 epoch 的 `mIoU`，并记录每个 epoch 的 `grad.train_val_cos`（若启用 probe）。

代码参考：
- 训练与评估循环：[main.py:L1535-L1615](file:///Users/anykong/AD-KUCS/AAL_SD/src/main.py#L1535-L1615)

### 3.2 过拟合代理信号（TVC）与 overfit_risk

#### (1) TVC（train‑val gradient cosine）
`Trainer.train_one_epoch()` 会：
- 在若干 train batch 上抽取梯度 probe 向量（可取均值向量 mean_vec）；
- 再对 1 个 probe batch（val 或 train_holdout）计算梯度向量 v_vec；
- 输出 `train_val_cos = cosine(mean_vec, v_vec)`。

代码参考：
- TVC 计算：[trainer.py:L266-L316](file:///Users/anykong/AD-KUCS/AAL_SD/src/core/trainer.py#L266-L316)

#### (2) overfit_risk（标量风险）
main 将若干 epoch 的 TVC 序列压缩为统计量：
- `tvc_mean, tvc_min, tvc_max, tvc_last, tvc_neg_rate`
并定义：
- `overfit_risk = tvc_neg_rate + 0.5*max(0, -tvc_min) + 0.5*max(0, -tvc_last)`

代码参考：
- overfit_risk 计算：[main.py:L1751-L1775](file:///Users/anykong/AD-KUCS/AAL_SD/src/main.py#L1751-L1775)

### 3.3 回撤（rollback_flag）与自适应回撤阈值

main 计算 `miou_delta = miou_signal - prev_miou`，并将回撤阈值设为与 epoch 波动相关的自适应量：
- `epoch_std = std(epoch_mious)`
- `tau = max(tau_min, std_factor * epoch_std)`
- `rollback_threshold_used = -tau`
- `rollback_flag = (miou_delta < rollback_threshold_used)`

代码参考：
- rollback 逻辑：[main.py:L1887-L1916](file:///Users/anykong/AD-KUCS/AAL_SD/src/main.py#L1887-L1916)

### 3.4 training_state 的形成与注入 toolbox

main 将关键控制指标打包进 `training_state`，并在 `use_agent=True` 时注入 toolbox：
- `self.toolbox.set_training_state(training_state)`

代码参考：
- training_state 字段：[main.py:L1925-L1958](file:///Users/anykong/AD-KUCS/AAL_SD/src/main.py#L1925-L1958)
- 注入点：[main.py:L1959-L1962](file:///Users/anykong/AD-KUCS/AAL_SD/src/main.py#L1959-L1962)

---

## 4. AD‑KUCS 打分：U(x)、K(x)、Score(x)

### 4.1 不确定性 U(x)：像素熵的聚合

默认实现：
- 对像素概率 `prob_map` 计算熵 `H(p) = -∑ p log2(p+eps)`；
- 将熵图按策略聚合为标量：默认 `mean`；也支持“仅对高熵像素取均值”。

代码参考：
- 熵聚合与阈值聚合：[sampler.py:L191-L209](file:///Users/anykong/AD-KUCS/AAL_SD/src/core/sampler.py#L191-L209)

### 4.2 K(x)：未标注池聚类代表性（当前实际实现）

当前 `rank_samples()` 的 K 实现为**未标注池特征 KMeans**：
1. 对 unlabeled features 执行 KMeans，`n_clusters=min(88, |U|)`；
2. 对每个样本计算到其簇中心的距离 `d(x)`；
3. 在 unlabeled 池内用 `d_max` 归一化；
4. 代表性分数：
   - `K(x) = 1 - d(x)/max(d_max, 1e-12)`（越靠近簇中心，K 越大）

代码参考：
- KMeans 与 K 的实现：[sampler.py:L734-L763](file:///Users/anykong/AD-KUCS/AAL_SD/src/core/sampler.py#L734-L763)

> 重要：虽然 sampler 接口接收 `labeled_features`，但当前 `rank_samples()` 版本并未使用它来计算 K。

### 4.3 融合 Score(x)

- `Score(x) = (1-λ)·U_norm(x) + λ·K_norm(x)`
- 排序后取 Top‑k。

代码参考：
- 融合与排序：[sampler.py:L760-L778](file:///Users/anykong/AD-KUCS/AAL_SD/src/core/sampler.py#L760-L778)

---

## 5. Control Frame（闭环控制）：观测、动作、约束、评价

AAL‑SD 在工程上把 control frame 拆成四块：

### 5.1 观测（Observations / 控制指标）

#### (A) 性能与趋势指标（mIoU）
- `last_miou`：本轮选定模型（last_epoch 或 best_val）的 mIoU；
- `prev_miou`：上一轮信号；
- `miou_delta = last_miou - prev_miou`；
- `miou_low_gain_streak`：Toolbox 内部维护的“低增益且低风险”的连续轮数（用于允许小幅上调 λ）。

代码参考：
- streak 更新逻辑：[toolbox.py:L1643-L1676](file:///Users/anykong/AD-KUCS/AAL_SD/src/agent/toolbox.py#L1643-L1676)

#### (B) 回撤指标（rollback）
- `rollback_threshold`：自适应阈值（负值）；
- `rollback_flag`：`miou_delta` 是否低于回撤阈值。

代码参考：
- rollback 计算：[main.py:L1887-L1916](file:///Users/anykong/AD-KUCS/AAL_SD/src/main.py#L1887-L1916)

#### (C) 过拟合/泛化风险指标（TVC 与 overfit_risk）
- `grad_train_val_cos_*`：TVC 的统计（mean/min/max/last/neg_rate）；
- `overfit_risk`：由 TVC 统计压缩得到的风险标量。

代码参考：
- TVC 计算：[trainer.py:L266-L316](file:///Users/anykong/AD-KUCS/AAL_SD/src/core/trainer.py#L266-L316)
- overfit_risk 计算：[main.py:L1751-L1775](file:///Users/anykong/AD-KUCS/AAL_SD/src/main.py#L1751-L1775)

#### (D) 预算/资源约束指标
- `current_labeled_count`、`total_budget`、`remaining_budget`
- `epochs_cap`（Toolbox/AgentConstraints）

代码参考：
- training_state 字段包含预算：[main.py:L1925-L1958](file:///Users/anykong/AD-KUCS/AAL_SD/src/main.py#L1925-L1958)
- toolbox 的 system_status 输出：[toolbox.py:L852-L949](file:///Users/anykong/AD-KUCS/AAL_SD/src/agent/toolbox.py#L852-L949)

#### (E) U/K 分布统计（用于策略判断与解释）
agent prompt 要求每轮通过工具获取分布（直方图/分位数等），作为“提高/降低 λ”的证据来源。

代码参考：
- prompt 约束与策略建议：[prompt_template.py:L110-L136](file:///Users/anykong/AD-KUCS/AAL_SD/src/agent/prompt_template.py#L110-L136)

### 5.2 动作（Actions / 控制量）

在 agent 模式下，可被允许的动作包括：
- `set_lambda`
- `set_query_size`
- `set_epochs_per_round`
- `set_hyperparameter(alpha)`
- `finalize_selection`（必须执行）

动作权限由 experiment config 中的 `control_permissions` 控制。

代码参考：
- prompt 动作空间拼装：[prompt_template.py:L15-L75](file:///Users/anykong/AD-KUCS/AAL_SD/src/agent/prompt_template.py#L15-L75)

### 5.3 约束（Constraints）

工程层面硬约束包括：
- `remaining_budget`：query_size 不能超过剩余预算；
- `epochs <= EPOCHS_CAP`；
- `λ ∈ [LAMBDA_CLAMP_MIN, LAMBDA_CLAMP_MAX]`（policy/guardrail 级夹紧），以及全局 `[0,1]`；
- 工具调用步数/每阶段调用次数限制（prompt 里显式声明）。

### 5.4 评价（Objective）

prompt 明确将目标写为：
- 最大化 ALC 与最终 mIoU；
- 降低不稳定性：回撤、过拟合、训练过载、预算越界、无效动作。

代码参考：
- 目标定义：[prompt_template.py:L95-L99](file:///Users/anykong/AD-KUCS/AAL_SD/src/agent/prompt_template.py#L95-L99)

---

## 6. Control Policy：λ policy 的具体定义与执行

### 6.1 最终 λ 的优先级（谁最终决定 λ？）

当 sampler 为 ADKUCSSampler 时，main 在进入 sampler 前会解析 `lambda_override`：
- `exp_config.lambda_override`（最高优先级，固定覆盖）
- `toolbox.control_state.lambda_override_round`（agent 可能写入）
- `lambda_policy`（toolbox 生成并写入 override_round）
- 否则 sampler 自己的 sigmoid(progress, alpha)

代码参考：
- override 解析：[main.py:L734-L762](file:///Users/anykong/AD-KUCS/AAL_SD/src/main.py#L734-L762)
- 使用 override 计算最终 λ 并传入 sampler：[main.py:L2543-L2570](file:///Users/anykong/AD-KUCS/AAL_SD/src/main.py#L2543-L2570)

### 6.2 warmup_risk_closed_loop：分相位的闭环 λ 策略

Toolbox 的策略实现是 `_compute_policy_lambda_for_round()`，mode 为 `warmup_risk_closed_loop`。

代码参考：
- policy 主体：[toolbox.py:L366-L703](file:///Users/anykong/AD-KUCS/AAL_SD/src/agent/toolbox.py#L366-L703)

#### Phase 1：uncertainty_only（前若干轮强制 λ=0 或 r1_lambda）
- 条件：`round_num <= uncertainty_only_rounds`
- 目的：先用 U 驱动探索与稳定学习启动。

参考：
- [toolbox.py:L410-L426](file:///Users/anykong/AD-KUCS/AAL_SD/src/agent/toolbox.py#L410-L426)

#### Phase 2：warmup（固定或 seeded uniform 的 λ）
- 条件：`warmup_start_round <= round_num < warmup_start_round + warmup_rounds`
- 目的：从纯 U 过渡到 U/K 混合。

参考：
- [toolbox.py:L428-L466](file:///Users/anykong/AD-KUCS/AAL_SD/src/agent/toolbox.py#L428-L466)

#### Phase 3：risk_control（闭环控制）
从 `risk_control_start_round` 起，依据风险信号调 λ：

- rollback：若 `rollback_flag==True`，则 `λ ← max(λ - delta_down, clamp_min)`；
- severe overfit：由 `overfit_risk` 与 TVC 信号组成（and/or），触发后降低 λ，并支持 cooling；
- 非 severe：在“低风险、低增益”等条件满足时允许小幅上调 λ，否则保持。

参考：
- severe 判定与 CI 触发：[toolbox.py:L493-L533](file:///Users/anykong/AD-KUCS/AAL_SD/src/agent/toolbox.py#L493-L533)
- risk_control 主逻辑：[toolbox.py:L562-L638](file:///Users/anykong/AD-KUCS/AAL_SD/src/agent/toolbox.py#L562-L638)

#### Phase 4：late_stage_ramp（后期 λ 下限抬升）
- 在不处于 rollback/severe 规则的前提下，将 λ 拉到一个随 round 增长的 floor。

参考：
- [toolbox.py:L641-L669](file:///Users/anykong/AD-KUCS/AAL_SD/src/agent/toolbox.py#L641-L669)

#### Phase 5：平滑（EMA）与单步最大变化（max_step）
- `lambda_smoothing="ema"`：对 λ 进行 EMA 平滑；
- `lambda_max_step`：限制单轮 λ 变化幅度。

参考：
- [toolbox.py:L671-L686](file:///Users/anykong/AD-KUCS/AAL_SD/src/agent/toolbox.py#L671-L686)

---

## 7. λ policy 的关键超参数与“对最终 λ 的影响路径”

这里的“最终 λ”指：进入 `ADKUCSSampler.rank_samples()` 的 `lambda_override` 或 sampler 自适应 λ。

### 7.1 直接定义策略结构的参数（lambda_policy 字段）

- `uncertainty_only_rounds` / `r1_lambda`：
  - 控制前期是否完全由 U 驱动（λ≈0）。
- `warmup_start_round` / `warmup_rounds` / `warmup_lambda` / `warmup_lambda_range`：
  - 控制从纯 U 向混合的过渡速度与初始 λ 水平。
- `risk_control_start_round`：
  - 控制闭环介入时间：越早越容易受到 rollback/overfit 信号压制。
- `severe_logic`：
  - `and` 更严格（需要 risk 与 tvc 同时命中），`or` 更敏感。
- `severe_tvc_key`：
  - 选择 `grad_train_val_cos_last`（更关注近期）或 `grad_train_val_cos_min`（更保守）进入 severe 判定。
- `risk_trigger="ci"` + `risk_ci_window/risk_ci_quantile/risk_ci_min_samples`：
  - 用历史分位数形成自适应阈值（相对异常检测），影响 severe 命中率，从而影响“降 λ”频率。
- `late_stage_ramp`：
  - 形成后期 λ 的“地板”，提高 K 的权重下限。
- `lambda_smoothing/lambda_smoothing_alpha` 与 `lambda_max_step`：
  - 控制 λ 的时间平滑与稳定性（抑制抖动与过快变化）。

### 7.2 通过阈值与步长影响策略动作的参数（AgentThresholds / overrides）

Toolbox 通过 `_agent_threshold()` 支持实验级覆盖（`agent_threshold_overrides`），典型项：
- `LAMBDA_CLAMP_MIN / LAMBDA_CLAMP_MAX`：
  - 决定最终输出 λ 的硬范围；
- `LAMBDA_DELTA_DOWN / LAMBDA_DELTA_UP`：
  - 决定 rollback / 允许上调时每轮改变幅度；
- `OVERFIT_RISK_HI` / `OVERFIT_TVC_MIN_HI`：
  - severe 门槛（触发降 λ 的敏感度）；
- `OVERFIT_RISK_LO` / `MIOU_LOW_GAIN_THRESH` / `MIOU_LOW_GAIN_STREAK`：
  - “低风险+低增益”允许上调 λ 的触发条件；
- `OVERFIT_RISK_LAMBDA_UP_MAX` / `LAMBDA_UP_K_U_GAP_MIN`：
  - “K 占优且低风险低增益”上调 λ 的另一条触发路径；
- `OVERFIT_RISK_EMA_ALPHA`：
  - 对 risk 做 EMA 平滑，影响 severe 触发的稳定性；
- `LAMBDA_DOWN_COOLING_ROUNDS`：
  - cooling 期限制连续调整，降低策略振荡。

参考：
- 阈值读取与覆盖：[toolbox.py:L358-L365](file:///Users/anykong/AD-KUCS/AAL_SD/src/agent/toolbox.py#L358-L365)
- 默认阈值定义：[agent/config.py](file:///Users/anykong/AD-KUCS/AAL_SD/src/agent/config.py)

### 7.3 Guardrails：对“最终选择”的隐式影响（也会改 λ）

#### (1) selection_guardrail：U 底线安全阀
当选中的 topK 样本整体 U 偏低时，guardrail 会：
- 逐步下调 λ（更偏 U）并重新按融合分数排序；
- 或启用“quota_u”混合方案保证一定比例的高 U 样本。

参考：
- guardrail 主体：[toolbox.py:L146-L317](file:///Users/anykong/AD-KUCS/AAL_SD/src/agent/toolbox.py#L146-L317)

#### (2) set_lambda 的二次裁剪（当允许 agent 显式设 λ 时）
`set_lambda()` 会在权限开启时执行多重裁剪：
- clamp 到 policy min/max；
- clamp 到“建议 sigmoid λ ± adjust_range”；
- rollback/severe 时禁止上调或强制下调（overfit_guard）。

参考：
- set_lambda 实现与 overfit_guard：[toolbox.py:L1129-L1399](file:///Users/anykong/AD-KUCS/AAL_SD/src/agent/toolbox.py#L1129-L1399)

---

## 8. prompt 与“实际算法”一致性的注意点（已在 prompt 中修正）

- 当前 sampler 的 K 实际为“未标注池 KMeans 代表性”（到簇中心距离反转）。
- `k_definition` 在当前版本主要作为标识字段在系统状态/日志中传递，不会切换 rank_samples 的 K 公式。
- 同时，实现层面仍存在一个前置约束：score 预计算阶段要求 labeled pool 非空，否则会报错（实现行为在 prompt 中应保持明确）。

参考：
- prompt 的 K 描述生成：[prompt_template.py](file:///Users/anykong/AD-KUCS/AAL_SD/src/agent/prompt_template.py)
- sampler 的 K 实现：[sampler.py:L734-L763](file:///Users/anykong/AD-KUCS/AAL_SD/src/core/sampler.py#L734-L763)

---

## 9. 读代码的推荐入口（按“理解控制闭环”的最短路径）

1. 训练状态与风险信号如何生成：  
   - [main.py:L1751-L1962](file:///Users/anykong/AD-KUCS/AAL_SD/src/main.py#L1751-L1962)

2. λ policy（warmup+risk closed loop）如何执行：  
   - [toolbox.py:L366-L737](file:///Users/anykong/AD-KUCS/AAL_SD/src/agent/toolbox.py#L366-L737)

3. 选样 guardrail 如何改变最终选择/λ：  
   - [toolbox.py:L146-L317](file:///Users/anykong/AD-KUCS/AAL_SD/src/agent/toolbox.py#L146-L317)

4. AD‑KUCS 的 U/K/Score 如何计算：  
   - [sampler.py:L191-L209](file:///Users/anykong/AD-KUCS/AAL_SD/src/core/sampler.py#L191-L209)（U）
   - [sampler.py:L734-L778](file:///Users/anykong/AD-KUCS/AAL_SD/src/core/sampler.py#L734-L778)（K 与 Score）
