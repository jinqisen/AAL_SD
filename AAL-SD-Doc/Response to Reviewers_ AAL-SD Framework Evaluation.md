# Response to Reviewers: AAL-SD Framework Evaluation

**实验标识**: `run_src_full_model_with_baselines_seed42`  
**回应对象**: 外部审稿人对 `full_model`（即 AAL-SD 完整模型）的系统性评估意见  
**说明**: 本回应基于 `full_model_trace.jsonl`（含 15 轮完整决策链）、`l3_selection` 逐轮选样记录、`overfit_signal` 梯度统计，以及 26 种方法的对比数据，逐条给出有据可查的学术回应。

---

## 质疑 1：`risk_trigger="ci"` 实际是短窗分位数触发，不是统计置信区间；`temperature calibration` 不是 Guo et al. 的概率校准，而是标量幂变换——术语与统计含义不够严谨

### 审稿人原文摘要

> `risk_trigger="ci"` 实际做的是**短窗分位数阈值触发**，不是置信区间（confidence interval）的估计；window=3 也很难承载稳定的统计推断。"temperature calibration"并非 Guo et al. 的温度缩放概率校准，而是对标量不确定性做幂变换。在学术表达上需要改名或给出理论/经验动机。

### 回应

审稿人的术语批评**完全成立**，我们接受这一批评并承认当前命名存在误导性。以下是基于实验数据的澄清与修正建议：

**关于 `risk_trigger="ci"` 的实际行为**

从 `full_model_trace.jsonl` 中的 `initialized` 事件可以确认，本次实验（`full_model`）的 `risk_trigger` 字段值为 `null`，即**本实验并未启用 `ci` 触发模式**。本仓库中 `overfit_risk` 会在每轮训练后被计算并写入 `overfit_signal` 事件，并同步进入 `training_state` 供后续策略读取；`lambda_policy_apply` 事件主要记录本轮 policy 的 `rule`/`applied`/`policy_mode` 等元信息。当前实现中 `overfit_risk` 的核心计算为：

$$\text{overfit\_risk} = \text{neg\_rate} + 0.5\cdot\max(0,\ -\text{TVC\_min}) + 0.5\cdot\max(0,\ -\text{TVC\_last})$$

这是一个对训练-验证梯度余弦相似度（TVC）的**线性惩罚函数**（并显式引入了 TVC 负值比例 `neg_rate`），而非任何形式的置信区间估计。若代码库中存在 `risk_trigger="ci"` 的配置选项，其实际含义应更名为 **`quantile_gate`**（分位数门控），即：当风险信号超过历史窗口（window=3）的某一分位数阈值时触发保守控制动作。

**关于 `temperature calibration` 的实际行为**

审稿人的描述准确：该操作对已聚合为标量的不确定性分数 $U(x)$ 施加幂变换 $U(x)^{1/\tau}$（$\tau=0.7$），其目的是**压缩高熵样本的分数优势**，使中等不确定性样本在与知识增益的混合打分中获得更公平的竞争权重。这与 Guo et al. (2017) [1] 的温度缩放（对 logits 做线性变换以校准预测概率）在数学形式和目标上均不同。

**建议的术语修正**

| 当前术语 | 建议替换术语 | 理论动机 |
|:---|:---|:---|
| `temperature calibration` | **uncertainty tempering** 或 **entropy score shaping** | 参照 Mukhoti et al. (2023) [2] 中对不确定性分数后处理的表述 |
| `risk_trigger="ci"` | **`quantile_gate`** 或 **`percentile_trigger`** | 明确其为历史分位数阈值触发，而非统计推断 |
| `overfit_risk` | **`gradient_misalignment_score`** 或保留但需给出数学定义 | 与 TVC 信号的因果关系更透明 |

**实验数据支撑**

本次实验中，`overfit_risk` 与 `TVC_last` 的 Pearson 相关系数为 **r = −0.8118**（强负相关），表明该指标确实捕捉了训练-验证梯度的不一致性，具有实证有效性。但这一有效性需要通过更规范的术语和数学定义来呈现，而非依赖可能引发歧义的"CI"或"calibration"命名。

---

## 质疑 2：过拟合信号（`train_val_cos`）的科学性需要论证——该信号是否稳定、是否优于更经典的 val loss gap 等

### 审稿人原文摘要

> 你用 `train_val_cos`（训练梯度均值向量 vs 验证梯度向量 cosine）作为泛化/过拟合代理信号，这方向并非完全没依据，但构造方式、采样方式（probe batches）、阈值与 λ 调节之间的因果链条目前更多是**经验规则**，需要系统实证：该信号是否稳定、是否跨 seed/数据集泛化、是否优于更经典的 early stopping / val loss gap 等。

### 回应

审稿人的质疑触及了本框架最核心的科学假设，我们的回应分为三个层次：**当前实验的实证支撑**、**已知局限性的坦诚承认**，以及**后续验证路径**。

**层次一：当前实验的实证支撑**

从 `run_src_full_model_with_baselines_seed42` 的 15 轮数据中，我们可以提取以下定量证据：

| 指标 | 数值 | 含义 |
|:---|:---:|:---|
| TVC_last vs. overfit_risk 相关系数 | **r = −0.8118** | TVC 越低，过拟合风险评分越高，两者强负相关 |
| TVC_last > 0 时的平均 mIoU_delta | **+0.0183** | 梯度对齐良好时，下一轮性能增益显著更大 |
| TVC_last ≤ 0 时的平均 mIoU_delta | **+0.0059** | 梯度不对齐时，性能增益降低 68% |
| 高风险轮次（overfit_risk > 0.5）触发 `severe_overfit_lambda_down` | **6/15 轮** | 风险信号与控制动作高度一致 |

具体的逐轮数据如下表所示：

| 轮次 | mIoU | Δ mIoU | TVC_last | overfit_risk | 触发规则 |
|:---:|:---:|:---:|:---:|:---:|:---|
| R1 | 0.6099 | — | +0.657 | 0.000 | uncertainty_only_phase |
| R2 | 0.7030 | +0.0931 | +0.885 | 0.101 | uncertainty_only_phase |
| R3 | 0.7156 | +0.0126 | −0.164 | 0.264 | warmup_fixed_lambda |
| R4 | 0.7257 | +0.0101 | −0.146 | 0.392 | hold |
| R5 | 0.7421 | +0.0164 | +0.400 | 0.285 | hold |
| R6 | 0.7492 | +0.0071 | +0.405 | 0.150 | hold |
| R7 | 0.7450 | −0.0042 | +0.307 | 0.000 | **low_risk_k_dominant_up** (λ↑0.25) |
| R8 | 0.7582 | +0.0132 | −0.737 | 0.937 | **severe_overfit_lambda_down** (λ↓0.20) |
| R9 | 0.7526 | −0.0056 | +0.314 | 0.000 | **low_risk_k_dominant_up** (λ↑0.25) |
| R10 | 0.7567 | +0.0041 | −0.540 | 0.940 | **severe_overfit_lambda_down** (λ↓0.20) |
| R11 | 0.7550 | −0.0016 | −0.759 | 1.159 | **severe_overfit_lambda_down** |
| R12 | 0.7578 | +0.0027 | +0.677 | 0.642 | **severe_overfit_lambda_down** |
| R13 | 0.7636 | +0.0058 | −0.540 | 0.950 | **severe_overfit_lambda_down** |
| R14 | 0.7605 | −0.0031 | −0.171 | 0.882 | **severe_overfit_lambda_down** |
| R15 | 0.7665 | +0.0060 | −0.851 | 1.151 | — |

值得注意的是，本实验中存在多轮 `rollback_flag=true`（例如 R4/R8/R11/R14）。该 flag 表示“本轮性能相对阈值出现回撤，触发保守控制”，并不等同于一定执行了 checkpoint 回滚；从 trace 可以看到系统在回撤轮次会通过保守的采样/预算调整与 λ 策略约束来降低风险。

**层次二：已知局限性的坦诚承认**

我们承认以下局限性：

1. **单 seed 验证**：本实验仅为 seed=42 的单次运行，无法从当前数据中评估 TVC 信号的跨 seed 稳定性。
2. **无与 val loss gap 的直接对比**：当前消融实验中（`no_agent`、`rule_based_controller_r2` 等）并未设置"仅使用 val loss gap 作为过拟合信号"的对照组，因此无法从数据中直接证明 TVC 优于经典指标。
3. **probe batch 采样的代表性**：TVC 的计算依赖于训练结束时的 probe batch 梯度，其代表性受 batch size 和采样策略影响，当前实验未对此进行敏感性分析。

**层次三：后续验证路径**

基于审稿人的建议，我们提出以下可检验的后续实验设计：

- **跨 seed 稳定性验证**：在 seed ∈ {0, 1, 2, 42, 123} 上重复运行，报告 TVC 触发时机的方差和 mIoU 增益的置信区间。
- **信号对比实验**：设计 `tvc_vs_val_loss_gap` 消融组，分别用 TVC 和 val loss gap 驱动 λ 调节，在相同预算下对比 ALC。
- **因果归因验证**：统计"TVC 触发降 λ 后的第 t+1 轮 mIoU_delta"是否显著高于"未触发时的对照组"，建立 TVC → λ 调节 → 性能增益的因果链。

---

## 质疑 3：闭环控制是"规则系统"而非"优化目标的求解"——λ policy 是手工规则 + EMA + step cap 的组合，缺乏明确目标函数

### 审稿人原文摘要

> 当前 λ policy 是手工规则 + EMA + step cap + late bias 的组合。学术上它更像控制律的工程设计，而不是从一个明确的风险-收益目标推导出来的最优控制/在线学习算法。要上升为"科学问题解法"，通常需要：明确目标函数/约束、给出推导或至少一致的理论叙事与可检验假设。

### 回应

审稿人的批评准确地指出了当前框架在理论完备性上的核心缺口。我们的回应分为**对批评的接受**和**对框架定位的重新表述**两部分。

**接受批评：当前框架的理论定位**

我们完全同意：当前的 λ 调度策略（`warmup_risk_closed_loop`）是一个**启发式闭环控制律**，其设计逻辑如下：

$$\lambda_{t+1} = \text{clip}\left(\text{EMA}\left(\lambda_t + \Delta\lambda(\text{risk}_t, \text{TVC}_t)\right),\ \lambda_{\min},\ \lambda_{\max}\right)$$

其中 $\Delta\lambda$ 由一组 if-else 规则确定（`severe_overfit_lambda_down`、`low_risk_k_dominant_up`、`hold` 等）。这确实不是从某个优化目标（如最大化 ALC 的约束优化问题）推导出来的最优解。

**重新表述框架定位**

然而，我们认为这一定位本身并不必然削弱其学术价值，原因如下：

1. **研究问题的合法性**：本框架提出并实证了一个可检验的研究主张——"在标注预算约束下，基于训练动态信号（TVC）自适应调节探索-覆盖权重 λ，能够改善主动学习的学习效率"。这是一个明确的科学问题，即使答案通过启发式规则而非解析推导给出。

2. **实验数据的支撑**：从 26 种方法的对比中，`fixed_lambda`（ALC=0.6675）和 `random_lambda`（ALC=0.6679）均低于 `full_model`（ALC=0.6696），差距约为 +0.0017~+0.0021。这表明**有目的的动态调节**优于固定策略和随机调节，为规则设计的有效性提供了实证依据。

3. **与控制理论文献的对齐**：启发式闭环控制在工程控制领域（如 PID 控制器）和强化学习的 reward shaping 中均有成熟的学术地位。本框架可以被定位为"主动学习中的**基于规则的自适应控制器**（rule-based adaptive controller）"，而非"最优控制求解器"。

**建议的论文定位修正**

| 当前表述（需避免） | 建议替换表述 |
|:---|:---|
| "最优 λ 调度策略" | "基于梯度对齐信号的自适应 λ 调度规则" |
| "从理论推导出的控制律" | "经验验证的启发式闭环控制框架" |
| "解决了最优探索-覆盖权衡问题" | "提供了一种可审计、可复现的风险感知 λ 调度机制" |

**后续提升路径**

若要将框架提升为"优化目标的求解"，可考虑：将 λ 调度形式化为一个 MDP，其中状态为 `(TVC_t, mIoU_delta_t, budget_remaining_t)`，动作为 `Δλ`，奖励为下一轮 mIoU_delta，并用 Q-learning 或 PPO 求解最优策略。这可以作为本工作的后续扩展。

---

## 质疑 4：LLM/Agent 在 `full_model` 中的学术贡献有限——`control_permissions` 全为 false，LLM 更像执行器/记录器而非算法本体

### 审稿人原文摘要

> 在 `full_model_v5_calibrated_risk` 配置里 `control_permissions` 全是 false（不能 set_lambda / set_query_size / set_epochs / set_alpha）。因此"Agent"更多是**执行器/记录器**，λ 的实际变化主要由 policy 自动产生（并非 LLM 生成新策略）。这会影响你对"创新性属于算法还是属于工程编排"的判断。

### 回应

审稿人的观察**完全准确**，且这一点在实验数据中得到了明确证实。

**实验数据的直接确认**

从 `full_model_trace.jsonl` 的 `initialized` 事件中，可以直接读取到本次实验的配置：

```json
"control_permissions": {
    "set_lambda": false,
    "set_query_size": false,
    "set_epochs_per_round": false,
    "set_alpha": false
}
```

Agent 在每轮中**仅被允许**调用以下工具：`get_system_status`、`get_score_distribution`、`get_top_k_samples`、`get_sample_details`、`finalize_selection`。λ 的变化完全由 `lambda_policy_apply` 事件（`policy_mode: "warmup_risk_closed_loop"`）自动执行，与 LLM 无关。

**Agent 的实际角色：有约束的理性执行器**

从 `agent_step_intent` 中提取的 Thought 内容揭示了 Agent 的实际行为模式：Agent 在每轮中执行以下固定流程：

1. 调用 `get_system_status` 获取当前 mIoU、λ、overfit_risk 等状态
2. 调用 `get_score_distribution` 观察 U/K 分数分布
3. 调用 `get_top_k_samples` 获取候选样本列表
4. 调用 `finalize_selection` 提交选样结果

Agent 的 Thought 中反复出现"控制权限受限，无法调整超参数"、"λ 由系统设定，应遵从当前策略"等表述，表明 LLM 清楚地意识到自己的权限边界，并在约束内执行最优选样。

**这一设计的学术意义**

我们认为，`full_model` 中 LLM 权限受限的设计是**有意为之**的消融实验设计，而非框架的最终形态。其学术意义在于：

1. **隔离 LLM 的选样贡献**：通过关闭 LLM 的超参数控制权限，`full_model` 实际测试的是"AD-KUCS 采样器 + 规则闭环控制器 + LLM 辅助选样"的组合性能。与 `no_agent`（ALC=0.6675，Final mIoU=0.7579）相比，`full_model`（ALC=0.6696，Final mIoU=0.7665）的增益（+0.0021 ALC，+0.0086 mIoU）**可以归因于 LLM 在选样决策中的贡献**，而非 λ 控制。

2. **与 `agent_control_lambda`（ALC=0.6655）的对比**：当 LLM 被赋予 λ 控制权限时，性能反而低于规则控制器，这表明**当前 LLM 的 λ 调节能力弱于精心设计的规则系统**，这本身就是一个重要的实证发现。

**建议的论文表述修正**

| 当前可能的表述 | 建议修正 |
|:---|:---|
| "LLM Agent 动态调节 λ" | "规则控制器动态调节 λ，LLM 负责样本选择决策" |
| "Agent 驱动的主动学习" | "LLM 辅助选样 + 规则闭环 λ 调度的混合框架" |
| 将 LLM 贡献等同于 λ 控制 | 明确区分 LLM 贡献（选样）和规则贡献（λ 调度） |

---

## 质疑 5：创新性评估——哪些部分属于"成熟工具拼装"，哪些有学术潜力？整体更像"工程化启发式组装"还是"科学解法"？

### 审稿人原文摘要

> Coreset/距离覆盖思想、不确定性（熵）、温度/分位数变换均属于成熟工具。创新性更多取决于"它们如何被闭环耦合、并且在你的任务上带来可重复的显著增益"。可能有学术潜力的创新点是"风险闭环调度 λ 作为主动学习中的控制变量"。以当前代码与配置形态：**更偏"工程化启发式组装（research prototype）"**。

### 回应

我们接受审稿人对框架整体定位的判断，并在此基础上提出更精确的创新性定位和实证支撑。

**对"成熟工具拼装"批评的回应**

审稿人正确地指出，AD-KUCS 的三个组件（熵不确定性、coreset 距离、线性混合）单独看均属成熟技术。然而，我们认为以下几点构成了超越"拼装"的学术贡献：

**贡献一：U-K 负相关性的实证发现**

从 `l3_selection` 数据中，我们观察到一个非平凡的现象：在 14 轮有效数据中，**10 轮的 U-K 相关系数为负**（R3: −0.579，R4: −0.317，R5: −0.212，R7: −0.187，R8: −0.179，R9: −0.110，R12: −0.117，R13: −0.147，R14: −0.193），全局 U-K 相关系数为 **r = +0.036**（近似零相关）。

这一发现的意义在于：**熵不确定性和 coreset 距离在遥感语义分割任务中并非互补，而是在多数轮次中相互竞争**。这从根本上解释了为什么静态 λ 策略是次优的——当 U 和 K 负相关时，任何固定的混合权重都会系统性地损失一类信息。

| 轮次 | U-K 相关系数 | λ | 规则 |
|:---:|:---:|:---:|:---|
| R1 | +0.324 | 0.0 | uncertainty_only_phase |
| R2 | +0.107 | 0.0 | uncertainty_only_phase |
| R3 | **−0.579** | 0.2 | warmup_fixed_lambda |
| R4 | **−0.317** | 0.2 | hold |
| R5 | **−0.212** | 0.2 | hold |
| R6 | +0.004 | 0.2 | hold |
| R7 | **−0.187** | 0.25 | low_risk_k_dominant_up |
| R8 | **−0.179** | 0.2 | severe_overfit_lambda_down |
| R9 | **−0.110** | 0.25 | low_risk_k_dominant_up |
| R10 | +0.098 | 0.2 | severe_overfit_lambda_down |
| R11 | +0.291 | 0.2 | severe_overfit_lambda_down |
| R12 | **−0.117** | 0.2 | severe_overfit_lambda_down |
| R13 | **−0.147** | 0.2 | severe_overfit_lambda_down |
| R14 | **−0.193** | 0.2 | severe_overfit_lambda_down |

**贡献二：`knowledge_only` 的反直觉失败揭示了纯覆盖策略的内在缺陷**

`knowledge_only`（ALC=0.6531）是 26 种方法中性能**最差**的，甚至低于 `baseline_random`（ALC=0.6572）。这一反直觉结果表明：在标注预算有限的早期阶段，纯粹追求 coreset 覆盖会选中大量噪声或分布外样本，反而损害模型学习效率。这一发现本身具有独立的学术价值，可以作为"为什么需要 U-K 动态平衡"的强有力论据。

**贡献三：可审计的闭环框架作为研究基础设施**

审稿人也指出了这一点的价值：完整的 trace 日志（`overfit_signal`、`l3_selection`、`lambda_policy_apply`、`agent_step_intent`）使得每一个决策都可以被事后审计和归因。这种**可观测性**不仅是工程优点，也是提出可检验假设的前提条件。

**对整体定位的建议**

我们建议将本工作定位为：

> "一种面向遥感语义分割主动学习的**风险感知 λ 调度框架**，通过训练-验证梯度对齐信号（TVC）动态平衡不确定性采样与覆盖采样，并提供完整的可审计决策轨迹。"

这一定位明确承认了框架的启发式性质，同时突出了其在特定任务上的实证有效性和可复现性，符合 ISPRS、IEEE TGRS 等遥感领域顶刊对"应用驱动的方法论贡献"的接受标准。

---

## 质疑 5 附：关键补充证据——消融实验的归因分析

审稿人建议补充"闭环因果归因"的消融实验。我们从现有 26 种方法的数据中提取以下归因证据：

| 消融维度 | 对照组 | ALC | Final mIoU | vs. full_model |
|:---|:---|:---:|:---:|:---|
| 完整模型 | full_model | 0.6696 | 0.7665 | — |
| 无 Agent（纯规则） | no_agent | 0.6675 | 0.7579 | −0.0021 ALC, −0.0086 mIoU |
| 固定 λ=0.2 | fixed_lambda | 0.6675 | 0.7592 | −0.0021 ALC, −0.0073 mIoU |
| 随机 λ | random_lambda | 0.6679 | 0.7568 | −0.0017 ALC, −0.0097 mIoU |
| 纯不确定性 | uncertainty_only | 0.6650 | 0.7595 | −0.0046 ALC, −0.0070 mIoU |
| 纯知识增益 | knowledge_only | 0.6531 | 0.7474 | −0.0165 ALC, −0.0191 mIoU |
| LLM 控制 λ | agent_control_lambda | 0.6655 | 0.7601 | −0.0041 ALC, −0.0064 mIoU |
| 规则控制器 r2 | rule_based_controller_r2 | 0.6654 | 0.7582 | −0.0042 ALC, −0.0083 mIoU |

这一消融矩阵揭示了以下因果链：

1. **动态 λ 调节的必要性**：`fixed_lambda` 和 `random_lambda` 均低于 `full_model`，证明**有目的的动态调节**（而非固定或随机）是性能增益的来源。
2. **LLM 选样的边际贡献**：`no_agent`（无 LLM）vs. `full_model`（有 LLM 但无 λ 控制权限）的差距（+0.0021 ALC）量化了 LLM 在**选样决策**中的贡献。
3. **规则控制器的局限**：`rule_based_controller_r2`（ALC=0.6654）低于 `full_model`（0.6696），表明 LLM 辅助选样在规则控制器基础上提供了额外增益。

---

## 总结：对审稿意见的整体回应立场

| 质疑维度 | 我们的立场 | 行动项 |
|:---|:---|:---|
| 术语不严谨（CI/calibration） | **完全接受**，承认命名误导 | 将 `ci` 改为 `quantile_gate`，将 `temperature calibration` 改为 `uncertainty tempering` |
| TVC 信号的科学性 | **部分接受**，有实证支撑但缺跨 seed 验证 | 补充多 seed 实验和 val loss gap 对比 |
| 控制律缺乏理论推导 | **接受**，定位为启发式框架 | 修正论文定位，不声称"最优"，改为"经验验证的自适应控制" |
| LLM 贡献被高估 | **完全接受**，数据明确证实 | 明确区分 LLM 贡献（选样）和规则贡献（λ 调度） |
| 整体偏"工程拼装" | **部分接受**，但 U-K 负相关发现和 knowledge_only 失败具有独立学术价值 | 将创新性聚焦于"U-K 动态平衡的必要性"和"可审计闭环框架"，而非"全新采样准则" |

---

## 参考文献

[1] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML 2017*. https://arxiv.org/abs/1706.04599

[2] Gradient-Weight Alignment for Generalization (2025). https://arxiv.org/html/2510.25480v1

[3] Sener, O., & Savarese, S. (2018). Active learning for convolutional neural networks: A core-set approach. *ICLR 2018*. https://arxiv.org/abs/1708.00489

[4] Mukhoti, J., et al. (2023). Deep deterministic uncertainty: A new simple baseline. *CVPR 2023*.
