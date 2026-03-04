# 技术方案文档：构建梯度感知型主动学习 Agent (AAL-SD-G)

**版本**: 1.0
**作者**: Manus AI
**日期**: 2026年02月11日

## 1. 背景与问题陈述

### 1.1. 背景

本项目旨在通过主动学习（Active Learning, AL）技术，在有限的标注预算下，最大化滑坡灾害语义分割模型的性能。AAL-SD 框架支持在不同消融设置中引入 LLM Agent 参与决策，但在当前仓库的默认 full_model 配置下，关键控制策略（如 `lambda` 的计算与约束）主要由控制器侧执行，并通过 trace 记录信号与动作以保证可复现与可审计。

### 1.2. 问题陈述

经过对基线实验数据的深入分析，我们定位了当前 AAL-SD 框架的核心问题：

1.  **性能未达预期**：在单种子的实验中，AAL-SD 的最终学习曲线下面积（Area Under Learning Curve, ALC）指标（0.7282）并未显著超越，甚至略低于更简单的基线策略，如 LLM-US（0.7336）和 BALD（0.7333）。

2.  **控制逻辑可能与泛化风险耦合**：在部分历史实现路径或特定消融设置中（尤其是当允许 Agent 直接影响 `lambda`，或采用单一日程/启发式驱动 `lambda` 上升时），容易出现“mIoU 上升→继续提高 `lambda`→更偏向多样性”的短视行为模式。

3.  **过拟合加剧**：在这些实验设置里，梯度对齐信号（如 `train_val_cos`）提示 `lambda` 的升高可能与过拟合风险上升相关。若缺乏“风险信号→动作约束→回撤/纠偏”的闭环，训练后期持续上调 `lambda` 的倾向会更容易把模型推向过拟合边缘，从而抵消主动学习带来的潜在收益。

**结论**：当 Agent（或简化控制逻辑）被赋予直接调参职责但缺乏结构化的风险信号与约束时，很容易变成“短视控制器”。AAL-SD-G 的目标，是把梯度诊断信号与过拟合风险显式纳入闭环：让系统能够“感知”内部学习状态，并让动作受到约束与审计，而不是依赖不可控的启发式上调。

---

## 2. 提出的解决方案：AAL-SD-G (梯度感知型)

我们提出 **AAL-SD-G (Gradient-Aware)** 方案，对现有 Agent 进行系统性升级，核心是将**梯度诊断信号**作为一等公民纳入 Agent 的决策循环中。

### 2.1. 核心思想

借鉴学术前沿的研究成果 [1, 7, 11]，我们认为模型在训练过程中产生的梯度，不仅是优化参数的工具，更是反映其内部学习状态和泛化潜力的“心电图”。通过监测训练集和验证集上的梯度方向一致性（`train_val_cos`），我们可以实时评估模型的泛化健康度。一个“聪明”的 Agent 应该能够：

-   **感知 (Perceive)**：读取 `train_val_cos` 等梯度信号，判断模型是处于“健康学习”、“学习停滞”还是“过拟合”状态。
-   **决策 (Decide)**：基于对当前状态的判断，输出对控制策略的建议（例如 `lambda` 的调整方向与幅度），并在允许的消融配置下触发受约束的策略更新。
-   **行动 (Act)**：当感知到过拟合风险时，优先采取“降温”措施（如把 `lambda` 拉回到策略下界附近）；`epochs/query_size` 等动作在当前默认实现中按配置固定，可作为后续扩展动作空间。

### 2.2. 方案优势

-   **理论坚实**：将梯度作为泛化代理指标有充分的学术文献支持 [7, 8, 9]。
-   **对症下药**：直接解决了当前 Agent 决策逻辑缺陷导致过拟合的核心问题。
-   **创新性强**：将 LLM Agent 作为视觉 AL 流程的“梯度感知”元控制器，在学术上具有明确的创新性 [6, 10]。
-   **可分阶段实施**：可以从简单的基于规则的系统快速迭代到一个更复杂的基于强化学习的智能体，风险可控。

---

## 3. 详细设计方案

### 3.1. 梯度诊断指标体系（已实现）

在当前 AAL-SD 框架中，以下梯度诊断指标已通过 `training_state` 机制实现并可供 Agent 使用：

**核心指标（已实现）：**
- **`train_val_cos`**: 训练集与验证集梯度方向余弦相似度，计算公式：
  $$\text{train\_val\_cos} = \frac{\mathbf{g}_{\text{train}} \cdot \mathbf{g}_{\text{val}}}{\|\mathbf{g}_{\text{train}}\| \cdot \|\mathbf{g}_{\text{val}}\|}$$
  其中 $\mathbf{g}_{\text{train}}$ 和 $\mathbf{g}_{\text{val}}$ 分别表示在训练集和验证集上计算的梯度向量。

- **`tvc_neg_rate`**: TVC 负值比例，反映梯度对齐恶化程度
- **`tvc_min`**: 当前轮次 TVC 最小值
- **`tvc_last`**: 当前轮次最后一个 epoch 的 TVC 值

**过拟合风险指标（已实现）：**
- **`overfit_risk`**: 综合风险评分，计算公式：
  $$\text{overfit\_risk} = \text{tvc\_neg\_rate} + 0.5 \cdot \max(0, -\text{tvc\_min}) + 0.5 \cdot \max(0, -\text{tvc\_last})$$

这些指标通过 `get_system_status` 工具向 Agent 提供，并记录在 trace 日志中确保可审计性。

### 3.2. 扩展指标体系（未来研究方向）

除了已实现的指标外，我们还设计了分层的量化指标体系作为未来研究方向，将原始的梯度数据转化为 Agent 可理解的洞察。

| 指标层级 | 指标名称 | 计算方式 | 含义与作用 | 实现状态 |
| :--- | :--- | :--- | :--- | :--- |
| **核心健康度** | `train_val_cos` (TVC) | 训练/验证集梯度向量余弦相似度 | **泛化方向盘**：衡量学习方向是否正确。 | ✅ 已实现 |
| | `tvc_trend` | `TVC_t - TVC_{t-1}` | **健康度趋势**：判断泛化能力是在改善还是恶化。 | 🔄 部分实现 |
| | `tvc_intra_round_slope` | 轮内各 Epoch TVC 的线性回归斜率 | **过拟合速度计**：衡量在单轮训练中过拟合的速度。 | 🔄 部分实现 |
| **梯度结构** | `backbone_head_ratio` | Backbone/Head 梯度范数比 | **学习重心**：判断模型是在学特征还是在调分类器。 | ⏳ 未来扩展 |
| | `grad_total_norm` | 所有可训练参数的梯度总范数 | **学习强度**：判断模型是否接近收敛。 | ⏳ 未来扩展 |
| **综合诊断** | `OverfittingRiskScore` (ORS) | TVC, tvc_trend, tvc_intra_slope 的加权组合 | **过拟合警报器**：综合评估当前的过拟合风险。 | ⏳ 未来扩展 |
| | `GeneralizationHealthIndex` (GHI) | TVC 和 ORS 的非线性组合 | **总健康仪表盘**：[0, 1] 区间的综合健康度评分。 | ⏳ 未来扩展 |

### 3.3. 实施阶段一：基于规则的梯度感知 Agent (AAL-SD-G v1.0)

此阶段的目标是快速实现一个基于专家规则的 Agent，以验证核心思想。

#### 3.2.1. 代码修改

1.  **`src/core/trainer.py`**: `train_one_epoch` 函数已经实现了 `train_val_cos` 的计算和返回。确保其返回的 `grad` 字典包含了计算上述所有指标所需的原始数据（如 `total_norm`, `backbone_norm`, `head_norm`, `cos_consecutive` 等）。


3.  **`src/agent/toolbox.py`**: 直接扩展现有的 `get_system_status` 工具。该工具从 `self.training_state` 中读取状态信息（包括梯度诊断指标），并以结构化 JSON 形式返回给 Agent。

4.  **`src/agent/prompt_template.py`**: 修改 `PromptBuilder` 中的系统提示（System Prompt），加入新的“梯度感知决策框架 (GDF)”。

#### 3.2.2. 新的 Agent 决策逻辑 (GDF)

在 System Prompt 中明确定义三个区域和对应的行动策略：

> -   **健康区 (GHI > 0.6)**: 模型泛化良好。**策略：大胆探索**。动作：在安全边界内适度提高 `lambda`（并确保可回撤、可审计）。
> -   **警告区 (0.3 < GHI <= 0.6)**: 出现过拟合迹象。**策略：保守利用**。动作：降低 `lambda`，把系统拉回更稳健的利用侧。
> -   **危险区 (GHI <= 0.3)**: 严重过拟合。**策略：紧急制动**。动作：将 `lambda` 压到策略下界附近（例如 0.2），并触发回撤/保护策略；`epochs` 的动态调节可作为后续扩展项。

### 3.4. 实施阶段二：基于强化学习的自适应 Agent (AAL-SD-G v2.0) - 研究提案

**状态说明：此为研究提案，非当前实现**

此阶段将 Agent 的决策过程形式化为一个强化学习（RL）问题，使其能够自主学习最优策略，而不是依赖于硬编码的规则。

#### 3.4.1. MDP 形式化（研究提案）

-   **状态 (State)**: `get_system_status` 返回的完整 JSON，包含性能、预算与梯度诊断等状态字段（具体取决于启用的监控信号）。
-   **动作 (Action)**: 一个离散的动作空间，优先从 `lambda` 调整开始；在扩展版本中再引入 `(lambda, max_epochs, query_size)` 等联合动作（未来扩展）。
-   **奖励 (Reward)**: 一个复合奖励函数，`R_t = w_perf * (mIoU_t - mIoU_{t-1}) + w_gen * (GHI_t - GHI_{t-1})`，同时奖励性能提升和泛化改善。

#### 3.4.2. 学习算法选型（研究提案）

-   **短期方案：Thompson Sampling**：将每个离散动作视为一个多臂老虎机（Multi-Armed Bandit）的臂。此方法简单、高效，且与 TAILOR 论文 [3] 的思想一致，非常适合作为 RL 的初步尝试。
-   **长期方案：深度强化学习 (PPO)**：使用 PPO (Proximal Policy Optimization) 算法训练一个策略网络，学习从复杂的状态到最优动作的映射。这需要将整个主动学习流程封装成一个标准的 `gym` 环境。

---

## 4. 实施与验证计划

### 4.1. 实施路线图

1.  **Sprint 1 (1-2周)**: **实现 AAL-SD-G v1.0 (规则 Agent)**
    -   [ ] 完成 `main.py` 和 `toolbox.py` 的代码修改，实现梯度诊断指标的计算与传递。
    -   [ ] 更新 `prompt_template.py`，集成新的 GDF 决策规则。
    -   [ ] 运行一次完整的15轮实验，验证新 Agent 的行为是否符合预期。

2.  **Sprint 2 (2-3周)**: **对比实验与分析**
    -   [ ] 运行多组（建议至少3个不同种子）对比实验：`AAL-SD (原版)` vs `AAL-SD-G v1.0` vs `Random` vs `BALD`。
    -   [ ] 分析实验结果，重点对比 ALC、最终 mIoU、以及 `train_val_cos` 的演化曲线，用数据证明新方案的有效性。

3.  **Sprint 3 (长期)**: **探索 AAL-SD-G v2.0 (RL Agent)**
    -   [ ] 技术选型：确定使用 Thompson Sampling 还是 PPO。
    -   [ ] 将主动学习流程封装成 `gym` 环境。
    -   [ ] 实现并训练 RL Agent。

### 4.2. 风险与缓解

-   **风险**: 梯度计算可能增加训练开销。
    -   **缓解**: `train_val_cos` 的计算仅在每个 epoch 结束后进行一次，且只使用一个 batch 的验证数据，对总体时间影响可控。在 `trainer.py` 中已有实现，开销已被评估。
-   **风险**: 新的规则或奖励函数可能引入新的偏见。
    -   **缓解**: 保持迭代和实验驱动。v1.0 的规则是基于现有实验数据设计的，v2.0 的 RL 方法旨在让数据自己说话，减少人为设计的偏见。

---

## 5. 参考文献

[1] Ash, J. T., et al. (2019). Deep Batch Active Learning by Diverse Gradient Embeddings (BADGE). *ICLR*.

[2] Liu, S., et al. (2024). Large Language Model Agent for Hyper-Parameter Optimization (AgentHPO). *arXiv:2402.01881*.

[3] Zhang, J., et al. (2023). Algorithm Selection for Deep Active Learning with Imbalanced Datasets (TAILOR). *NeurIPS*.

[4] Xia, D., et al. (2025). Dual-View Gradient Probes: Disentangling Uncertainty for Deep Active Learning. *ACM MM*.

[5] Zhang, C. (2025). Pushing the Limits of Active Data Selection with Gradient Matching (GIST). *MIT Thesis*.

[6] Xia, Y., Mukherjee, S., et al. (2025). From Selection to Generation: A Survey of LLM-based Active Learning. *ACL*.

[7] Shi, Y., et al. (2021). Gradient Matching for Domain Generalization (Fish). *ICLR*.

[8] Rame, A., et al. (2022). Fishr: Invariant Gradient Variances for Out-of-Distribution Generalization. *ICML*.

[9] Hölzl, F., & von Rueden, L. (2025). Gradient-Weight Alignment as a Train-Time Proxy for Generalization. *OpenReview*.

[10] Konyushkova, K., et al. (2017). Learning Active Learning from Data. *NeurIPS*.

[11] Liu, Z., et al. (2021). Influence Selection for Active Learning. *ICCV*.
