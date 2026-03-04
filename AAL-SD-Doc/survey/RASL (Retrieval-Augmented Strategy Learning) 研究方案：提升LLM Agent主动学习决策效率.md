# RASL (Retrieval-Augmented Strategy Learning) 研究方案：提升LLM Agent主动学习决策效率（学术版）

## 摘要

RASL（Retrieval-Augmented Strategy Learning）旨在解决 AAL-SD 中 LLM Agent 的“冷启动低效”问题：在主动学习早期，Agent 缺乏对特定数据集与任务（滑坡语义分割）的先验，容易产生保守或随机策略，从而降低 ALC（Area Under the Learning Curve）。RASL 将“轮级（round-level）主动学习决策”形式化为上下文 bandit / 轮级 MDP：在每轮基于当前状态 \(s_t\) 选择动作 \(a_t\)（基础动作通常为 \(\lambda_t\)，即不确定性与知识增益的权重），并通过回报 \(r_t\) 评估该轮采样对后续性能与稳定性的综合贡献。RASL 的实现形态是“检索增强的 in-context 策略学习”：把历史实验轨迹抽取为（状态、动作、结果/回报）经验三元组存入策略记忆库，通过相似度检索得到可比的成功/失败经验作为上下文，引导 LLM 在当前轮输出更稳健、更可解释的策略。

## 主要贡献与结论（本任务讨论的要点汇总）

1. 任务对齐：在滑坡语义分割场景中，“采样是否好”必须用端到端收益（如 \(\Delta mIoU\)、滑坡召回、边界质量、ALC）衡量，不能用单一训练过程信号替代。
2. 指标定位：train_val_cos 适合作为训练健康度/风险信号（惩罚项或约束项），但不是采样优劣的充分指标；仅优化 train_val_cos 会驱动策略过度保守。
3. 奖励设计：推荐 reward 采用“端到端收益 + 过拟合/冲突风险惩罚 +（可选）预算效率”的组合，使策略学习能够识别“收益-风险拐点”，尤其是在后期平台期避免低收益高风险行为。
4. 实证支撑：除 `/results/logs_md` 的多 seed（42/43/44）轮级曲线外，进一步基于 seed42–46、多个策略（full_model 与经典 baselines）的 trace 做统计可观察到：`overfit_risk` 更像“下一轮边际收益”的领先负向信号（以每条“run×strategy”轨迹内相关汇总，`corr(overfit_risk(t), ΔmIoU(t→t+1))` 的均值约为 `pearson≈-0.388`、`spearman≈-0.334`），而其与最终 mIoU 的相关会受到策略类别/学习阶段混杂影响（例如 pooled 的 `corr(overfit_mean, last/best mIoU)` 为正，但控制 `seed+strategy` 固定效应后显著减弱）。因此 reward 与约束应优先对齐“下一轮收益 + 尖峰风险治理”，而不是直接把均值风险当作最终性能的单调解释变量。

## 1. 研究背景与动机

在AAL-SD（Agent-Assisted Active Learning for Semantic Segmentation）框架的第一阶段研究中，我们发现大型语言模型（LLM）Agent在主动学习的初期阶段，由于缺乏对特定数据集（如Landslide4Sense）的先验知识，往往倾向于采取保守或随机的探索策略。尽管Agent具备强大的推理能力，但这种“冷启动”问题限制了其在有限标注预算下实现最优ALC（Area Under the Learning Curve）表现。为了克服这一挑战，本研究旨在引入检索增强策略学习（Retrieval-Augmented Strategy Learning, RASL）框架，通过赋予Agent“经验”来显著提升其决策效率和主动学习性能。

## 2. 核心创新点：RASL 框架

RASL框架的核心思想是通过“检索增强”机制，使LLM Agent能够有效地利用和复用历史实验中积累的成功策略。该框架主要由三个相互协作的组件构成：策略记忆库（Policy Memory Bank）、策略检索器（Policy Retriever）和策略生成器（Policy Generator）。策略记忆库负责存储主动学习过程中积累的“状态-动作-结果”三元组经验。策略检索器则根据当前的学习状态，从记忆库中高效地检索出最相似且表现优异的历史成功案例。最后，策略生成器将检索到的经验作为 In-context Learning 的示例，引导生成更优的$\lambda_t$（不确定性与知识增益的权重）。在扩展版本中，也可以把$query\_size$与$epochs\_per\_round$纳入动作空间；但在当前仓库的默认 full\_model 配置下，这两者通常由配置固定而非由 Agent 直接控制。**（此为研究提案，非当前实现）**

### 2.1 形式化问题定义：轮级上下文 bandit / 轮级 MDP

主动学习的轮级过程（round-level active learning）可抽象为“带上下文的序列决策”。

令第 \(t\) 轮开始时的系统状态为 \(s_t\)，动作 \(a_t\) 为本轮的采样/策略选择（基础版本主要为 \(\lambda_t\)），环境转移由一次“采样-标注-训练-评估”闭环定义。定义折扣因子 \(\gamma\in(0,1]\)，则目标是最大化期望回报：

\[
\max_{\pi}\ \mathbb{E}_{\pi}\Big[\sum_{t=1}^{T}\gamma^{t-1} r_t\Big]
\]

在工程实现层面，由于轮级动作对后续多轮表现均有影响，严格建模可视作有限步长 MDP；在经验复用/检索增强的设定下，也可以采用更稳定的“上下文 bandit 近似”，将 \(r_t\) 主要定义为“本轮动作对下一轮及其附近窗口”的可观测收益，从而提高经验的可复用性与检索匹配度。

### 2.2 状态、动作、回报：与滑坡端到端识别对齐

为保证“采样决策 \(\rightarrow\) 模型识别质量”的因果对齐，建议将状态拆为四类子状态并拼接为向量或结构化对象：

\[
s_t=\big(s^{perf}_t,\ s^{data}_t,\ s^{risk}_t,\ s^{ls}_t\big)
\]

- \(s^{perf}_t\)：端到端性能与趋势（例如当前 mIoU、\(\Delta mIoU\) 及其滑动统计、平台期指示）
- \(s^{data}_t\)：未标注池的可采样性分布（例如 \(U(x)\)、\(K(x)\) 的分位数/相关系数、空间聚集度）
- \(s^{risk}_t\)：训练健康度/风险（例如轮内后半程的 train_val_cos 汇总、验证曲线波动）
- \(s^{ls}_t\)：滑坡任务对齐特征（例如滑坡类召回、滑坡边界质量代理、滑坡像素占比的估计与不确定性）

基础动作空间可定义为：

\[
a_t=\lambda_t \in [0,1]
\]

若采用扩展动作空间，可写为 \(a_t=(\lambda_t, q_t, e_t)\)，其中 \(q_t\) 为 query_size，\(e_t\) 为每轮训练 epoch；但在本仓库默认 full_model 预算口径下，\(q_t,e_t\) 更适合作为常量记录项而非 Agent 直接决策变量。

### 2.3 理论视角：in-context 策略学习可被视作隐式贝叶斯推断

将“检索到的历史轮级经验”视作上下文 \(\mathcal{C}_t=\{(s_i,a_i,r_i)\}_{i=1}^{k}\)，则 LLM 在当前轮输出动作 \(a_t\) 的过程可解释为：在一个潜在“策略概念” \(\phi\)（例如某一阶段的最优决策规则、风险偏好、或针对数据分布的启发式）上进行后验推断，并据此进行条件决策。该视角与“in-context learning 作为隐式贝叶斯推断”的解释一致：[24]
\[
p(\phi\mid \mathcal{C}_t)\propto p(\mathcal{C}_t\mid \phi)\,p(\phi),\quad
a_t\sim \pi(\cdot\mid s_t,\phi)
\]

进一步地，在“从交互轨迹中进行 in-context 决策/强化学习”的设定中，监督式预训练的序列模型可在上下文中实现近似的后验采样式决策，从而具备“经验 \(\rightarrow\) 快速适配”的性质：[25][26] 这为 RASL 的“非参数化策略学习（检索 + ICL）”提供了更直接的理论支撑：我们并不要求在 15 轮内对策略做稳定的梯度更新，而是利用外部经验库把“过去多 run/多 seed 的试错”压缩为可复用的上下文证据链。

## 3. 关键技术挑战

RASL框架的实现面临多项关键技术挑战。首先是**状态表征（State Representation）**问题，即如何将高维的分割模型状态和数据分布（如mIoU趋势、不确定性分布等）有效地转化为可供检索的向量表示，以确保检索的准确性和效率。其次是**经验评估（Experience Valuation）**，需要明确定义何为“成功”的经验，例如，是带来最大mIoU增益的决策，还是在特定阶段表现出最佳稳定性的策略。最后是**跨域迁移（Cross-domain Transfer）**的潜力与局限性，即从其他遥感任务（如建筑物提取）中检索到的策略是否能够有效地迁移并应用于滑坡检测任务，这涉及到经验的泛化能力和领域适应性。

## 4. 预期目标

本研究预期通过RASL框架的引入，实现以下目标：显著缩短主动学习的“冷启动”期，使Agent在早期阶段就能做出高质量决策。在相同的标注预算下，RASL的ALC表现将显著优于纯AAL-SD框架。此外，RASL还将增强Agent决策的可解释性，通过引用历史案例，Agent能够清晰地阐述其决策依据，从而提升整个主动学习过程的透明度和可信度。

## 5. RASL 框架核心架构设计

RASL 框架将围绕策略记忆库、策略检索器和策略生成器这三个核心组件进行构建，以实现Agent的经验学习和复用。

### 5.1 策略记忆库 (Policy Memory Bank)

策略记忆库是RASL框架的核心组成部分，用于系统化地存储Agent在主动学习过程中积累的经验。每条经验记录均以一个结构化的三元组`(State, Action, Outcome)`的形式存在。

*   **State (状态)**：详细描述决策时的环境信息，包括当前模型在验证集上的 mIoU、F1-score 等性能指标，mIoU 的历史变化趋势（如 miou\_delta 及其滑动平均），未标注池中 U(x) 和 K(x) 分数的统计特征及其联合分布，当前已标注样本数量和剩余预算比例，以及当前策略参数（如$\lambda_t$）。若扩展动作空间，则可在 State 中附带$query\_size/epochs\_per\_round$等字段用于学习与归因；在默认 full\_model 配置下，这些字段更适合作为“配置常量/记录项”而非可决策变量。

*   **Action (动作)**：指在特定状态下输出的决策。基础动作可以仅包含$\lambda_t$；在扩展版本中可进一步包含$query\_size$与$epochs\_per\_round$等联合动作。

*   **Outcome (结果)**：评估该动作执行后对模型性能产生的影响，包括短期收益（下一轮的mIoU增益，即`miou_delta`）、长期收益（通过加权或回溯机制评估对最终ALC的贡献），以及训练过程中的稳定性指标（如mIoU波动，即`sigma_epoch_mIoU`）。

这些经验将以结构化数据（如JSON格式或数据库记录）的形式存储，并通过提取关键特征（例如，使用Sentence-BERT或其他嵌入模型对State进行编码）作为检索索引，以便后续高效检索。

### 5.2 策略检索器 (Policy Retriever)

策略检索器负责根据当前学习状态，从策略记忆库中高效地检索出最相关的历史经验。检索过程首先将当前的学习状态转化为向量表示作为**检索查询（Retrieval Query）**。随后，通过**相似度度量（Similarity Metric）**（如余弦相似度）计算当前状态向量与记忆库中所有存储的State向量之间的相似度。检索算法可以采用基于向量相似度的检索技术（如Faiss、Annoy），或探索更复杂的基于图的检索方法。检索到的经验将根据相似度进行排序，并可根据其**Outcome**进行进一步过滤，例如优先检索那些被评估为“成功”的经验。

### 5.3 策略生成器 (Policy Generator) 与提示词工程 (Prompt Engineering)

策略生成器通过精心的**提示词工程（Prompt Engineering）**，将检索到的历史经验作为上下文信息，引导 LLM Agent 做出更明智的决策。提示词结构将包含：明确 Agent 角色和目标的**系统指令**（如优化 ALC、平衡 U/K），详细描述当前模型性能、数据分布和预算信息的**当前状态描述**，将检索到的`(State, Action, Outcome)`三元组作为 Few-shot 示例或 In-context Learning 的一部分提供给 Agent 的**检索到的经验**，以及要求基于当前状态和历史经验输出新的决策（基础版本输出$\lambda_t$；扩展版本可输出$query\_size$、$epochs\_per\_round$等）并提供决策理由的**决策指令**。通过分析检索到的成功案例，系统能够避免重复历史错误，并借鉴最优实践；决策执行后的结果再回写到策略记忆库中，形成持续演化的闭环。

### 5.4 检索增强策略学习的数学化描述（Policy as RAG over experience）

将记忆库记为 \(\mathcal{M}=\{m_i\}_{i=1}^{N}\)，每条经验
\(m_i=(s_i,a_i,o_i,r_i)\)，其中 \(o_i\) 为可选的训练过程摘要（epoch 曲线、回滚事件等），\(r_i\) 为标准化回报或多指标 outcome 的压缩表示。

给定当前状态 \(s_t\)，检索器定义一个相似度函数 \(\mathrm{sim}(\cdot,\cdot)\)，返回 Top-\(k\) 经验集合：

\[
\mathcal{R}(s_t)=\mathrm{TopK}_{m_i\in\mathcal{M}}\, \mathrm{sim}(f(s_t), f(s_i))
\]

其中 \(f(\cdot)\) 为状态编码器（可为手工特征归一化后的向量，也可为学习到的嵌入）。策略生成器可视作条件分布：

\[
\pi(a_t\mid s_t,\mathcal{R}(s_t))=\mathrm{LLM}\Big(\mathrm{Prompt}(s_t,\mathcal{R}(s_t))\Big)
\]

这一定义强调：RASL 并不依赖对 \(\pi\) 的参数化梯度更新，而是通过“经验检索 \(\rightarrow\) in-context 泛化”实现快速策略适配；其可解释性来自于 \(\mathcal{R}(s_t)\) 中可引用的证据链。

### 5.5 端到端滑坡识别流程中的 sampling 作用机理

滑坡语义分割中的主动学习采样，其作用机理可被拆为“数据分布重加权 + 难例/边界补齐 + 类别稀缺校正”三条通路：

\[
\mathcal{D}^{lab}_{t+1} = \mathcal{D}^{lab}_t \cup \mathcal{Q}_t,\quad
\mathcal{Q}_t \subset \mathcal{D}^{unlab}_t,\ |\mathcal{Q}_t|=q_t
\]

其中 \(\mathcal{Q}_t\) 由 acquisition function 驱动（例如结合不确定性 \(U(x)\) 与知识增益 \(K(x)\) 的混合评分 \(A_{\lambda}(x)=\lambda U(x)+(1-\lambda)K(x)\)）。这一轮样本被标注并加入训练后，更新后的模型 \(\theta_{t+1}\) 的变化在滑坡任务上主要体现在：

- 对滑坡类稀缺样本的覆盖率提升（提高 \(R^{ls}\) 或降低漏检）
- 对边界/破碎区域的拟合能力提升（减少边界错分与碎片化）
- 对域内多样性（地貌、植被、阴影、水体混淆）覆盖更均衡，从而提升泛化

因此，RASL 的 state 与 reward 必须显式包含“滑坡类别与边界质量”的端到端指标或其可计算代理，否则策略会与任务目标脱钩。

### 5.6 轮级算法流程（概念性）

对第 \(t\) 轮，RASL 的闭环可概括为：

1. 用当前模型在未标注池上推理，得到 \(U(x),K(x)\) 等统计并构造 \(s_t\)。
2. 检索 \(\mathcal{R}(s_t)\) 并由 LLM 生成动作 \(a_t\)（通常为 \(\lambda_t\)）。
3. 由 \(a_t\) 定义 acquisition，对未标注池选择 \(\mathcal{Q}_t\)，完成标注并训练得到 \(\theta_{t+1}\)。
4. 在验证/测试集上评估并计算回报 \(r_t\)，把 \((s_t,a_t,o_t,r_t)\) 写回 \(\mathcal{M}\)。

## 6. 实验方案与评估指标设计

### 6.1 实验目标

本研究的实验目标主要包括三个方面：首先，**验证RASL的有效性**，旨在证明RASL框架能够显著提升LLM Agent在主动学习中的决策质量，尤其是在冷启动阶段。其次，**量化性能提升**，通过与AAL-SD（无RAG）及其他基线方法进行对比，量化RASL在ALC和最终模型性能上的具体提升。最后，**分析Agent行为**，深入分析Agent在引入RAG后的决策逻辑和策略演变，以验证其“经验复用”的能力和效果。

### 6.2 实验设置

*   **数据集**：为确保与AAL-SD阶段实验结果的可比性，本研究将继续使用Landslide4Sense数据集。
*   **模型**：沿用DeepLabV3+ with ResNet50 backbone，保持模型架构的一致性。
*   **主动学习协议**：初始标注集、总标注预算和每轮查询大小（$query\_size$）将保持与AAL-SD阶段相同，以实现ALC的直接比较。实验轮次也将保持15轮主动学习。
*   **基线方法**：本研究将RASL与以下方法进行对比：
    *   **AAL-SD (Optimized)**：使用上一阶段优化后的K定义和自适应Rollback阈值的AAL-SD框架作为主要对比基线。
    *   **传统基线**：包括Random、Entropy、Core-Set和BALD等经典主动学习策略。
    *   **消融实验**：
        *   **RASL-No-RAG**：等同于AAL-SD (Optimized)，用于验证RAG模块对性能提升的贡献。
        *   **RASL-Random-Retrieval**：RAG模块随机检索经验而非基于相似度，旨在验证检索机制的有效性。
        *   **RASL-Fixed-Policy**：Agent仅在冷启动阶段使用RAG检索到的固定策略，之后恢复AAL-SD模式，用于分析RAG对冷启动阶段的影响。

### 6.3 策略记忆库构建

策略记忆库的构建是RASL框架成功的关键。我们将利用AAL-SD阶段所有实验（包括基线和消融实验）的trace日志，从中提取结构化的`(State, Action, Outcome)`三元组。在数据筛选上，将特别关注`full_model`和`fixed_lambda`等表现良好的实验轨迹，以确保记忆库中经验的质量。在**状态编码**方面，我们将对State中的数值特征进行归一化处理，并可能采用主成分分析（PCA）或自编码器（Autoencoder）等技术进行降维，以便于向量化表示和高效的相似度计算。**经验筛选**将初步筛选出ALC表现优异的实验轨迹中的决策点作为高质量经验，以构建一个精炼且有效的策略记忆库。

### 6.4 评估指标

本研究将采用以下指标对RASL框架进行全面评估：

1.  **主要指标**：
    *   **Area Under the Learning Curve (ALC)**：作为衡量主动学习效率的综合指标，其计算方式将与AAL-SD阶段保持一致。
    *   **最终mIoU**：在达到总标注预算时，模型在测试集上的mIoU性能。
2.  **辅助指标**：
    *   **Agent决策质量**：通过分析决策（基础版本为$\lambda_t$；如启用扩展动作则包含$query\_size$等）与后续 mIoU 增益的相关性，以及评估不同学习阶段策略选择的多样性来衡量。
    *   **检索效率**：评估检索器从记忆库中找到“最佳”经验的准确率和召回率。
    *   **计算开销**：量化RAG模块引入的额外计算时间开销，以评估其实用性。

### 6.5 预期结果

我们预期RASL框架将显著提升主动学习的性能。具体而言，RASL有望在ALC和最终mIoU上显著超越AAL-SD (Optimized) 和所有基线方法。尤其是在主动学习的早期阶段（冷启动），RASL将表现出更快的性能增长。此外，我们期望Agent的决策将更具“经验”和“一致性”，减少不必要的探索或保守行为，从而使整个主动学习过程更加稳定和高效。

### 6.6 `/results` 多 seed 轮级证据（full_model, seeds 42/43/44）

为支撑 reward 设计与“平台期/边际递减”假设，本任务对如下日志文件中每轮“当前轮次最佳结果”的 mIoU 进行了解析汇总：

- `results/logs_md/full_model_20260207_paper_ms_seed42.md`
- `results/logs_md/full_model_20260207_paper_ms_seed43.md`
- `results/logs_md/full_model_20260207_paper_ms_seed44.md`

R1-R15 轮级最佳 mIoU（mean/std 为跨 seed 统计）：

| Round | seed42 | seed43 | seed44 | mean | std |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.6046 | 0.6087 | 0.6646 | 0.6260 | 0.0274 |
| 2 | 0.6906 | 0.7100 | 0.6796 | 0.6934 | 0.0126 |
| 3 | 0.7042 | 0.7108 | 0.7210 | 0.7120 | 0.0069 |
| 4 | 0.7293 | 0.7369 | 0.7269 | 0.7310 | 0.0043 |
| 5 | 0.7300 | 0.7392 | 0.7383 | 0.7358 | 0.0041 |
| 6 | 0.7449 | 0.7416 | 0.7413 | 0.7426 | 0.0016 |
| 7 | 0.7506 | 0.7465 | 0.7423 | 0.7465 | 0.0034 |
| 8 | 0.7526 | 0.7362 | 0.7528 | 0.7472 | 0.0078 |
| 9 | 0.7487 | 0.7524 | 0.7523 | 0.7511 | 0.0017 |
| 10 | 0.7595 | 0.7569 | 0.7514 | 0.7559 | 0.0034 |
| 11 | 0.7606 | 0.7435 | 0.7508 | 0.7516 | 0.0070 |
| 12 | 0.7572 | 0.7426 | 0.7413 | 0.7470 | 0.0072 |
| 13 | 0.7568 | 0.7582 | 0.7578 | 0.7576 | 0.0006 |
| 14 | 0.7485 | 0.7577 | 0.7545 | 0.7536 | 0.0038 |
| 15 | 0.7601 | 0.7583 | 0.7500 | 0.7561 | 0.0044 |

从多 seed 证据可得到两条稳定结论：

1. 学习曲线形态高度一致：早期（R1-R6）提升显著，后期（约 R6 之后）进入平台区间，边际收益明显降低。
2. RASL 的 reward 设计需要能“识别并适应平台期”：在低收益区间若继续采取高风险采样/训练配置，将更可能引发不稳定或过拟合而无实质收益。

### 6.7 train_val_cos 的理论依据、适用性与边界（学术审视）

定义训练与验证损失的梯度分别为 \(g_{train}=\nabla_\theta L_{train}(\theta)\)、\(g_{val}=\nabla_\theta L_{val}(\theta)\)，则

\[
train\_val\_cos=\frac{\langle g_{train},g_{val}\rangle}{\|g_{train}\|\|g_{val}\|}
\]

理论合理性要点：

- 在一阶近似下，\(train\_val\_cos<0\) 表示训练更新方向与验证目标方向冲突。
- 该“负余弦=冲突”的判据与多任务优化中的梯度冲突/对齐框架一致（例如 PCGrad 使用余弦相似度判断冲突并进行投影修正）。类似地，GradNorm 通过动态调节梯度尺度来塑造训练动力学，强调“梯度是训练行为的可控变量”。[5][27]

适用边界要点：

- train_val_cos 是“风险/健康度”信号，而非“收益”信号；因此不应独立作为采样优劣的主指标。
- 在 RASL 中推荐将其用作惩罚项或约束项，并与端到端收益指标共同构成 reward。

进一步地，若希望把“训练健康度”作为硬约束（而非软惩罚），可以采用约束优化的表述：在最大化端到端收益的同时，要求训练-验证梯度对齐不低于阈值 \(\tau\)（或要求负对齐风险不超过阈值）：

\[
\max_{\pi}\ \mathbb{E}_{\pi}\Big[\sum_{t=1}^{T}\gamma^{t-1} \hat{r}^{gain}_t\Big]
\quad \text{s.t.}\quad \mathbb{E}_{\pi}\big[c^-_t\big]\le \delta
\]

其中 \(\hat{r}^{gain}_t\) 表示由 \(\Delta mIoU\)、\(\Delta R^{ls}\)、\(\Delta B\) 等构成的收益项，\(c^-_t\) 由 train_val_cos 派生的风险项。该形式的优点是：在平台期收益天然变小的阶段，策略不会为了追求微小收益而持续“吃掉”风险预算。

### 6.8 Reward 设计建议（端到端收益 + 风险惩罚 + 可选预算效率）

为避免策略投机（只追求过程指标或只追求短期 mIoU），推荐默认 reward 采用可分解结构：

#### 6.8.1 Outcome \(\rightarrow\) 标量回报：窗口化定义与可复现归一化

记第 \(t\) 轮训练与评估完成后的轮级端到端指标为：\(mIoU_t\)、滑坡类召回 \(R^{ls}_t\)、边界质量指标 \(B_t\)（可取 boundary IoU 或 boundary-F1 等）。定义收益项采用窗口 \(W\) 的差分：

\[
\Delta_W mIoU_t = mIoU_{t+W}-mIoU_t,\quad
\Delta_W R^{ls}_t = R^{ls}_{t+W}-R^{ls}_t,\quad
\Delta_W B_t = B_{t+W}-B_t
\]

其中 \(W=1\) 对应“本轮决策对下一轮结果”的一阶近邻收益，经验最易复用；若希望降低噪声，可取 \(W\in\{2,3\}\)。

为使不同 run、不同阶段的经验可比，推荐对每个指标使用分位数归一化（在线可计算）。给定某个指标序列 \(\{z\}\) 在参考集合 \(\mathcal{H}_t\) 上的经验分布（例如：同一数据集同一模型的历史经验库；或同 run、同阶段的历史轮集合），记 \(Q_p(z;\mathcal{H}_t)\) 为分位数，则：

\[
\tilde{z}_t =
\mathrm{clip}\Bigg(
\frac{z_t - Q_{0.5}(z;\mathcal{H}_t)}{Q_{0.9}(z;\mathcal{H}_t)-Q_{0.1}(z;\mathcal{H}_t)+\epsilon},
-c,\ c
\Bigg)
\]

其中 \(\epsilon>0\) 防止分母为零，\(\mathrm{clip}(\cdot)\) 防止极端离群点主导回报（\(c\) 可取 3）。该定义可以直接被复现：只需固定 \(\mathcal{H}_t\) 的构造规则与分位数参数。

风险项建议由轮内后半程 train_val_cos 的统计构造。记第 \(t\) 轮后半程（例如后 \(K\) 个 epoch）的平均为 \(\bar{c}_t^{late}\)，则：

\[
c^-_t=\max(0,-\bar{c}^{late}_t),\quad \tilde{c}^-_t=\mathrm{Norm}(c^-_t)
\]

其中 \(\mathrm{Norm}(\cdot)\) 可沿用上述分位数归一化。

#### 6.8.2 平台期检测：可核验判据与分段回报

定义轮级增益 \(g_t=mIoU_t-mIoU_{t-1}\)，用窗口 \(H\) 的滑动平均估计边际收益：

\[
\bar{g}_t=\frac{1}{H}\sum_{h=0}^{H-1} g_{t-h}
\]

给定阈值 \(\varepsilon_{gain}>0\) 与最短持续长度 \(L\)，定义平台期指示变量：

\[
p_t=\mathbb{I}\Big(\bar{g}_t<\varepsilon_{gain}\ \land\ \forall j\in\{0,\dots,L-1\},\ \bar{g}_{t-j}<\varepsilon_{gain}\Big)
\]

在平台期（\(p_t=1\)）应提高“风险抑制/单位标注收益”的权重，使策略在低收益区间避免高风险行为。一个可复现的分段回报写法为：

\[
r_t =
\hat{r}^{gain}_t
- \big(w_4 + w_4^{plat}\,p_t\big)\tilde{c}^{-}_{t}
+ w_5\,p_t\cdot\frac{\tilde{\Delta_W mIoU}_t}{\Delta|\mathcal{D}^{lab}_t|+\epsilon}
\]

其中 \(\hat{r}^{gain}_t = w_1\tilde{\Delta_W mIoU}_t + w_2\tilde{\Delta_W R^{ls}}_t + w_3\tilde{\Delta_W B}_t\)。该机制与多 seed 证据一致：当进入平台区间时，收益项自然变小，分段回报会自动把策略偏好从“强探索”转向“稳健/效率”。

取 \(W=1\) 且不显式引入平台期分段（等价于令 \(p_t=0\)）时，上述回报可退化为：

\[
r_t =
w_1\tilde{\Delta_1 mIoU}_t
+ w_2\tilde{\Delta_1 R^{ls}}_t
+ w_3\tilde{\Delta_1 B}_t
- w_4\tilde{c}^{-}_{t}
\]

该退化形式更接近“单步回报”的上下文 bandit 设定，便于在经验库较小的情况下稳定检索与复用。

#### 6.8.3 梯度动态解耦视角：把“收益-风险-效率”拆成可观测的四维机制

仅用 \(\lambda_t\) 去解释“训练是否稳定/是否高效”往往会混淆多个机制。为让经验可复用、也让策略更可解释，可在不改变主目标（端到端指标）前提下，把训练动力学拆成四类可观测统计，并写入 \(s^{risk}_t\)（或单独扩展为 \(s^{grad}_t\)）：

- **分布冲击（distribution shock）**：梯度方向在轮间的变化幅度（例如相邻 epoch/轮的梯度余弦差或表示漂移），用于刻画“采样导致的数据分布更新有多剧烈”。
- **收敛速率（convergence rate）**：梯度强度或有效步长代理（例如损失下降斜率、梯度范数的稳态区间），用于刻画“是否在快速学到新信息”。
- **训练稳定性（training stability）**：梯度方向一致性/波动（例如 train_val_cos 的分位数、负对齐事件率），用于刻画“是否出现训练-验证目标冲突与不稳定”。
- **学习效率（learning efficiency）**：单位预算收益（例如 \(\Delta_W mIoU\) / 每轮标注量，或 \(\Delta_W mIoU\) / 每轮训练开销），用于刻画“是否值得继续高风险/高代价探索”。

该拆分直接对应本项目已有的经验归纳：\(\lambda\) 的调整会同时改变“分布冲击—收敛速率—稳定性—效率”的权衡，因此把这些维度显式写入状态/回报，有助于检索到“机制相似”的历史经验，而不是只做表面指标匹配（例如只看当前 mIoU）。这一点也与前期“梯度动态解耦”材料中对 \(\lambda\) 影响模式的总结一致（见附录 6.11 的对齐清单）。

### 6.9 学术严谨性审视与验证设计（面向滑坡语义分割）

#### 6.9.1 是否充分考虑滑坡领域 sampling 的数据特性

滑坡遥感分割的采样与识别存在一组稳定的“领域结构”，若不显式建模，AL 策略容易失真：

- 空间自相关与簇状分布：滑坡往往成片出现，同一地貌/地物背景下相邻 patch 高相关，容易导致“批次内重复信息”。
- 类别极不平衡与漏检代价高：滑坡像素占比低、且误检/漏检对灾害任务代价不对称，单纯用整体 mIoU 可能掩盖滑坡类的改善/退化。
- 边界与破碎结构占主导：滑坡边界模糊、碎片化预测常见，采样对“边界/细节”的补齐比对“内部区域”的补齐更关键。
- 难负样本丰富：阴影、水体、裸土、崩塌痕迹等与滑坡外观相近，采样需要覆盖难负样本，避免模型在背景类上过拟合而牺牲滑坡召回。

因此，本方案在 state 与 reward 中显式纳入 \(s_t^{ls}\)（滑坡对齐特征）与 \(\Delta_W R^{ls},\Delta_W B\)（滑坡召回与边界质量增益），并建议在 \(s_t^{data}\) 中增加“空间聚集度/多样性”特征（例如基于 patch 坐标/聚类的覆盖率统计），以对冲空间自相关带来的采样偏置。

#### 6.9.2 Reward 指标设置的合理性（收益-风险可分解）

reward 的合理性取决于它是否满足两点：端到端目标一致性与可稳定估计性。

- 一致性：以 \(\Delta_W mIoU\) 作为总体性能收益，以 \(\Delta_W R^{ls}\) 与 \(\Delta_W B\) 强化滑坡类与边界质量，保证采样决策与灾害识别目标对齐。
- 可估计性：用窗口差分 \(W\) 把长期信用分配问题压缩为轮级可观测增量，并通过分位数归一化减少不同 run/阶段的尺度漂移。
- 风险项定位清晰：train_val_cos 派生的 \(c^-_t\) 仅作为“训练健康度/冲突风险”惩罚或约束，不取代端到端收益；这避免策略被过程信号劫持而过度保守。

若担心“多指标加权”引入主观性，可在实验中做权重敏感性分析（固定 \(w_1\) 为 1，对 \(w_2,w_3,w_4\) 做网格/贝叶斯搜索或使用分段约束形式，把风险作为硬约束）。

#### 6.9.3 是否补充了可观测性（避免不可实现的状态/回报）

轮级决策时刻可获得的信息应限定为“在选择 \(\mathcal{Q}_t\) 之前可观测”的量：

- 预测侧：未标注池上的 logits/不确定性图、\(U(x),K(x)\) 统计、空间聚集度、多样性覆盖率。
- 训练侧：上一轮训练曲线摘要（loss、mIoU 波动）、train_val_cos 的轮内统计、回滚事件等。
- 评估侧：验证/测试集的 \(mIoU_t,R^{ls}_t,B_t\)（用于构造下一轮的 \(s_{t+1}\) 与回报 \(r_t\)，而非作为本轮动作的先验）。

严格说，该过程更接近 POMDP：真实“数据分布与可学性”不可完全观测。RASL 的检索增强记忆 \(\mathcal{R}(s_t)\) 可以被视作一种“外部记忆/信念补全”，在不引入不可观测变量的前提下提升策略稳健性与可解释性。

#### 6.9.4 实验设计如何验证（因果归因与统计稳健）

建议把验证拆为四类，分别回答“有效吗、为什么、何时有效、代价如何”：

1. 有效性（主结论）：与 AAL-SD（无检索增强）、Entropy、Core-set、BALD、Random 比较 ALC 与最终指标（含滑坡类召回/边界指标），多 seed 报告均值与方差。
2. 机制归因（必要性）：RASL-No-RAG、RASL-Random-Retrieval、只用收益项 vs 收益+风险、是否启用平台期分段 \(p_t\) 等消融。
3. 阶段性（何时有效）：按轮分段报告（早期/中期/平台期）的平均增益与风险事件率；验证 RASL 是否主要缩短冷启动，并在平台期降低“低收益高风险”行为。
4. 成本与鲁棒性：报告检索开销（时间/内存）、经验库规模敏感性（用部分历史轨迹构库）、以及训练预算敏感性（例如每轮 5/10/20 epoch 的对照，仅用于验证结论鲁棒，不改变默认口径）。

#### 6.9.5 学术创新性是否足够（相对现有 AL/RAG/RL 的差异点）

本方案的创新性主要来自“把检索增强用于轮级主动学习策略学习”，并针对滑坡分割补齐了可观测 state 与可复现 reward：

- 方法层：以“经验检索 + in-context 策略生成”替代参数化策略梯度更新，在小数据/小回合（15 轮）下更符合可行性与稳定性。
- 目标层：将滑坡任务的类不平衡与边界主导误差显式写入 state/reward，而非仅做通用 mIoU/entropy 驱动。
- 风险层：把 train_val_cos 定位为可解释的风险信号，引入软惩罚或硬约束，使策略在平台期具备“风险预算”概念。
- 证据层：给出多 seed 轮级曲线证据支撑平台期假设，并据此设计分段回报与平台期检测机制。

#### 6.9.6 RASL 是否能在 15 轮、每轮 10 epoch（总 150 epoch）下站住脚

关键点在于：RASL 解决的是“样本选择与策略冷启动”的问题，而不是“把训练算力堆满”。即使总训练 epoch 不大，只要每轮的模型更新能够对采样差异产生可测的端到端响应，策略差异就会累积到 ALC 上。

同时，需要诚实地指出边界：若 10 epoch/轮导致单轮增益噪声过大，信用分配会变难。为此，本方案通过 (i) 窗口化收益 \(\Delta_W\)（可取 \(W=2/3\) 降噪）、(ii) 分位数归一化、(iii) 平台期分段与风险预算，来提高在“短训练、少回合”条件下的可学习性与可复用性。实验上应通过“训练预算敏感性”对照验证：结论在 5/10/20 epoch/轮下是否保持一致或呈现合理退化。

### 6.10 相关工作对齐：动态策略、策略学习与 LLM 经验学习

从“为什么需要 RASL”出发，现有工作给出了三类关键启发，但仍留有空缺：

1. **语义分割 AL 的阶段性与策略非万能性**：在语义分割场景中，数据冗余、预算大小、是否采用（半/自）监督预训练会显著改变“哪种 acquisition 更好”，因此不存在跨场景通吃的固定策略；实践上常见“早期更需要多样性覆盖，后期更需要不确定性挖掘”的阶段性切换规律。[10][11][12] 在“目标域标注昂贵”的设定下，也有把主动学习与域适应耦合起来的框架，可作为语义分割场景下的补充参照。[29]
2. **从数据/经验中“学习策略”而非手工指定**：包括把策略选择建模为 bandit（从一组策略中在线挑选）[13]、或直接从历史 AL 过程学习一个“预期误差下降/收益”的回归器或策略模型。[14][15][16] 也存在通过 RL 学习“可迁移的主动学习策略网络”的工作，强调跨图/跨域的策略泛化，与 RASL 的“跨 run/跨数据集经验复用”形成互补参照。[30] 在“动态调参/动态阈值”方向，亦存在用元学习在非平稳流式数据下动态调整主动学习行为的工作，可作为 RASL 的补充参照（但任务形态不同）。[28]
3. **LLM Agent 的经验记忆与语言化强化**：在不更新模型参数的前提下，通过“经验池/记忆 + 自然语言归纳/反思 + 检索回放”提升决策质量，证明了“经验 \(\rightarrow\) 知识 \(\rightarrow\) 决策”在 Agent 场景的可行性。[17][18] 进一步地，LLM 也被观察到具备从奖励信号进行 in-context 学习的能力，为“非参数化策略学习”提供了更直接的理论与经验依据。[19][20] 与此同时，检索增强也已被用于强化学习以提升样本效率与抗遗忘，提供了“经验库 + 检索器”在决策系统中的另一类实现范式参照。[31]

RASL 的定位与差异点在于：把上述三条线索统一到**轮级主动学习决策**上，并且强制对齐滑坡语义分割的端到端目标与可观测性约束——不直接“让 LLM 选样本”，而是让 LLM 在检索到的可比经验上下文中输出轮级动作（基础形态为 \(\lambda_t\)），并用可复现的窗口化收益、分位数归一化与风险约束来评估经验与回报，从而提升冷启动阶段的决策效率并降低平台期的无效高风险探索。[21][22]

### 6.11 与前期材料的覆盖度对齐（10 份）

下表用于核验：本研究方案的关键论点/设计均可在前期材料中找到对应依据或批判性约束（内部文档为本仓库材料，便于追溯）。

1. 方法论创新点与差异化定位：见 [RASL 方法论创新性查新记录](../RASL%20方法论创新性查新记录.md)、[RASL 方法论学术创新性评价报告](../RASL%20方法论学术创新性评价报告.md)；对应本方案第 2/5/6.10 节。
2. 核心学术问题推导（非平稳、少回合、轮级决策）：见 [RASL 核心学术问题的深度推导](../RASL%20核心学术问题的深度推导.md)、[RASL_ A Theoretical Framework for In-Context Meta-Strategy Learning in Non-Stationary Environments](../RASL_%20A%20Theoretical%20Framework%20for%20In-Context%20Meta-Strategy%20Learning%20in%20Non-Stationary%20Environments.md)；对应本方案第 2.1/2.3 节。
3. 梯度动力学与“解耦视角”的机制化表达：见 [RASL深度研究方案 v2.0：基于梯度动态解耦的视角](../RASL%E6%B7%B1%E5%BA%A6%E7%A0%94%E7%A9%B6%E6%96%B9%E6%A1%88%20v2.0%EF%BC%9A%E5%9F%BA%E4%BA%8E%E6%A2%AF%E5%BA%A6%E5%8A%A8%E6%80%81%E8%A7%A3%E8%80%A6%E7%9A%84%E8%A7%86%E8%A7%92.md)、[RASL_Deep_Research_Proposal_Final_v4](../RASL_Deep_Research_Proposal_Final_v4.md)；对应本方案第 6.7/6.8.3 节。
4. 文献支撑版需求总结与检索记录（相关工作范围界定）：见 [RASL 深度研究方案需求总结（文献支撑版） (1)](../RASL%20%E6%B7%B1%E5%BA%A6%E7%A0%94%E7%A9%B6%E6%96%B9%E6%A1%88%E9%9C%80%E6%B1%82%E6%80%BB%E7%BB%93%EF%BC%88%E6%96%87%E7%8C%AE%E6%94%AF%E6%92%91%E7%89%88%EF%BC%89%20(1).md)、[RASL 需求总结文献检索记录 (1)](../RASL%20%E9%9C%80%E6%B1%82%E6%80%BB%E7%BB%93%E6%96%87%E7%8C%AE%E6%A3%80%E7%B4%A2%E8%AE%B0%E5%BD%95%20(1).md)；对应本方案第 6.10/参考文献节。
5. 审稿人视角的批判性约束（可实现性、可观测性、归因与消融）：见 [RASL 研究方案：审稿人视角下的批判性评估](../RASL%20%E7%A0%94%E7%A9%B6%E6%96%B9%E6%A1%88%EF%BC%9A%E5%AE%A1%E7%A8%BF%E4%BA%BA%E8%A7%86%E8%A7%92%E4%B8%8B%E7%9A%84%E6%89%B9%E5%88%A4%E6%80%A7%E8%AF%84%E4%BC%B0.md)；对应本方案第 6.9 节。
6. 面向滑坡任务的策略迁移与 state/reward 对齐：见 [RASL_ Retrieval-Augmented Meta-Strategy Learning for Active Landslide Mapping](../RASL_%20Retrieval-Augmented%20Meta-Strategy%20Learning%20for%20Active%20Landslide%20Mapping.md)；对应本方案第 2.2/5.5/6.9.1 节。
## 7. 参考文献

[1] Settles, B. (2009). *Active Learning Literature Survey*. University of Wisconsin-Madison Technical Report 1648. https://burrsettles.com/pub/settles.activelearning.pdf
[2] Houlsby, N., Huszár, F., Ghahramani, Z., & Lengyel, M. (2011). *Bayesian Active Learning for Classification and Preference Learning*. arXiv:1112.5745. https://arxiv.org/abs/1112.5745
[3] Sener, O., & Savarese, S. (2018). *Active Learning for Convolutional Neural Networks: A Core-Set Approach*. ICLR 2018. arXiv:1708.00489. https://arxiv.org/abs/1708.00489
[4] Xie, B., Yuan, L., Li, S., Liu, C. H., & Cheng, X. (2022). *Towards Fewer Annotations: Active Learning via Region Impurity and Prediction Uncertainty for Domain Adaptive Semantic Segmentation*. CVPR 2022. arXiv:2111.12940. https://arxiv.org/abs/2111.12940
[5] Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020). *Gradient Surgery for Multi-Task Learning*. NeurIPS 2020. arXiv:2001.06782. https://arxiv.org/abs/2001.06782
[6] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. arXiv:2005.11401. https://arxiv.org/abs/2005.11401
[7] Park, J. S., O’Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). *Generative Agents: Interactive Simulacra of Human Behavior*. UIST 2023. arXiv:2304.03442. https://arxiv.org/abs/2304.03442
[8] Ghorbanzadeh, O., Xu, Y., Ghamisi, P., Kopp, M., & Kreil, D. (2022). *Landslide4Sense: Reference Benchmark Data and Deep Learning Models for Landslide Detection*. arXiv:2206.00515. https://arxiv.org/abs/2206.00515
[9] Landslide4Sense Project Page. https://www.iarai.ac.at/landslide4sense/
[10] Mittal, S., Niemeijer, J., Schäfer, J. P., & Brox, T. (2023). *Best Practices in Active Learning for Semantic Segmentation*. arXiv:2302.04075. https://arxiv.org/abs/2302.04075
[11] Doucet, P., Estermann, B., Aczel, T., & Wattenhofer, R. (2024). *Bridging Diversity and Uncertainty in Active Learning with Self-Supervised Pre-Training*. arXiv:2403.03728. https://arxiv.org/abs/2403.03728
[12] Hacohen, G., & Weinshall, D. (2023). *How to Select Which Active Learning Strategy is Best Suited for Your Specific Problem and Budget*. NeurIPS 2023. arXiv:2306.03543. https://arxiv.org/abs/2306.03543
[13] Hsu, W.-N., & Lin, H.-T. (2015). *Active Learning by Learning*. AAAI 2015. https://www.semanticscholar.org/paper/Active-Learning-by-Learning-Hsu-Lin/e7e4790b4622d339dcf681ab84c5804551c1b5a3
[14] Konyushkova, K., Sznitman, R., & Fua, P. (2017). *Learning Active Learning from Data*. NeurIPS 2017. arXiv:1703.03365. https://arxiv.org/abs/1703.03365
[15] Fang, M., Li, Y., & Cohn, T. (2017). *Learning how to Active Learn: A Deep Reinforcement Learning Approach*. EMNLP 2017. https://aclanthology.org/D17-1063/
[16] Pang, K., Dong, M., Wu, Y., & Hospedales, T. (2018). *Meta-Learning Transferable Active Learning Policies by Deep Reinforcement Learning*. arXiv:1806.04798. https://arxiv.org/abs/1806.04798
[17] Zhao, A., Huang, D., Xu, Q., Lin, M., Liu, Y.-J., & Huang, G. (2024). *ExpeL: LLM Agents Are Experiential Learners*. AAAI 2024. arXiv:2308.10144. https://arxiv.org/abs/2308.10144
[18] Shinn, N., Labash, B., Gopinath, A., Reddy, A., Elkins, K., Yao, S., & Singh, S. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023. arXiv:2303.11366. https://arxiv.org/abs/2303.11366
[19] Monea, G., Bosselut, A., Brantley, K., & Artzi, Y. (2025). *LLMs Are In-Context Bandit Reinforcement Learners*. COLM 2025. arXiv:2410.05362. https://arxiv.org/abs/2410.05362
[20] Song, K., Moeini, A., Wang, P., Gong, L., Chandra, R., Qi, Y., & Zhang, S. (2025). *Reward Is Enough: LLMs Are In-Context Reinforcement Learners*. arXiv:2506.06303. https://arxiv.org/abs/2506.06303
[21] Arjona-Medina, J. A., Gillhofer, M., Widrich, M., Unterthiner, T., Brandstetter, J., & Hochreiter, S. (2019). *RUDDER: Return Decomposition for Delayed Rewards*. arXiv:1806.07857. https://arxiv.org/abs/1806.07857
[22] Harutyunyan, A., Dabney, W., Mesnard, T., Azar, M. G., Piot, B., Heess, N., van Hasselt, H., Wayne, G., Singh, S., Precup, D., & Munos, R. (2019). *Hindsight Credit Assignment*. NeurIPS 2019. arXiv:1912.02503. https://arxiv.org/abs/1912.02503
[23] Xia, Y., Mukherjee, S., Xie, Z., Wu, J., Li, X., Aponte, R., Lyu, H., Barrow, J., Chen, H., Dernoncourt, F., Kveton, B., Yu, T., Zhang, R., Gu, J., Ahmed, N. K., Wang, Y., Chen, X., Deilamsalehy, H., Kim, S., Hu, Z., Zhao, Y., Lipka, N., Yoon, S., Huang, T.-H. K., Wang, Z., Mathur, P., Pal, S., Mukherjee, K., Zhang, Z., Park, N., Nguyen, T. H., Luo, J., Rossi, R. A., & McAuley, J. (2025). *From Selection to Generation: A Survey of LLM-based Active Learning*. arXiv:2502.11767. https://arxiv.org/abs/2502.11767
[24] Xie, S. M., Raghunathan, A., Liang, P., & Ma, T. (2022). *An Explanation of In-context Learning as Implicit Bayesian Inference*. ICLR 2022. arXiv:2111.02080. https://arxiv.org/abs/2111.02080
[25] Lee, J. N., Xie, A., Pacchiano, A., Chandak, Y., Finn, C., Nachum, O., & Brunskill, E. (2023). *Supervised Pretraining Can Learn In-Context Reinforcement Learning*. NeurIPS 2023. arXiv:2306.14892. https://arxiv.org/abs/2306.14892
[26] Lin, L., Bai, Y., & Mei, S. (2024). *Transformers as Decision Makers: Provable In-Context Reinforcement Learning via Supervised Pretraining*. arXiv:2310.08566. https://arxiv.org/abs/2310.08566
[27] Chen, Z., Badrinarayanan, V., & Lee, C.-Y. (2018). *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks*. ICML 2018. arXiv:1711.02257. https://arxiv.org/abs/1711.02257
[28] Martins, V. E., Cano, A., & Barbon Junior, S. (2023). *Meta-learning for dynamic tuning of active learning on stream classification*. Pattern Recognition, 138, 109359. https://doi.org/10.1016/j.patcog.2023.109359
[29] Wu, T.-H., Liou, Y.-S., Yuan, S.-J., Lee, H.-Y., Chen, T.-I., Huang, K.-C., & Hsu, W. H. (2022). *D2ADA: Dynamic Density-aware Active Domain Adaptation for Semantic Segmentation*. ECCV 2022. arXiv:2202.06484. https://arxiv.org/abs/2202.06484
[30] Hu, S., Xiong, Z., Qu, M., Yuan, X., Côté, M.-A., Liu, Z., & Tang, J. (2020). *Graph Policy Network for Transferable Active Learning on Graphs*. NeurIPS 2020. arXiv:2006.13463. https://arxiv.org/abs/2006.13463
[31] Goyal, A., Friesen, A. L., Banino, A., Weber, T., Ke, N. R., Puigdomenech Badia, A., Guez, A., Mirza, M., Humphreys, P. C., Konyushkova, K., Sifre, L., Valko, M., Osindero, S., Lillicrap, T., Heess, N., & Blundell, C. (2022). *Retrieval-Augmented Reinforcement Learning*. ICML 2022. arXiv:2202.08417. https://arxiv.org/abs/2202.08417
