# AAL-SD 基线实验与消融实验方案（基于创新性评估）

**作者**: Manus AI  
**日期**: 2026年2月14日

---

## 目录

1. [实验设计总体原则](#1-实验设计总体原则)
2. [基线实验方案](#2-基线实验方案)
3. [消融实验方案](#3-消融实验方案)
4. [实验优先级与资源规划](#4-实验优先级与资源规划)
5. [实施路径与代码修改指南](#5-实施路径与代码修改指南)

---

## 1. 实验设计总体原则

本实验方案严格对齐AAL-SD的**三大核心创新点**，确保每个实验都能直接回应审稿人可能的质疑，并最大化地支撑论文的创新声明。

### 1.1 三大核心创新点与对应实验目标

| 核心创新点 | 创新性评级 | 对应实验目标 |
| :--- | :--- | :--- |
| **创新点1：LLM作为元策略控制器** | **强** | 证明LLM动态调整λ_t优于固定策略、随机策略、基于规则的启发式策略 |
| **创新点2：应用于遥感滑坡检测的完整框架** | **强** | 证明AAL-SD优于领域内关键工作（DIAL、Wang & Brenning）以及传统AL策略 |
| **创新点3：AD-KUCS的自适应设计** | **中等** | 证明"冷启动+标准"两阶段知识增益计算、不确定性与知识增益的动态权衡的有效性 |

### 1.2 实验设计的三个层次

**第一层：基线实验（Baseline Experiments）**  
目标是将AAL-SD与领域内最相关的工作以及传统AL策略进行横向对比，证明其在应用领域的优越性。

**第二层：消融实验（Ablation Studies）**  
目标是逐层剥离AAL-SD的核心组件，精确量化每个创新点的独立贡献。

**第三层：深度分析实验（In-depth Analysis）**  
目标是通过梯度证据、可解释性分析等手段，构建从"LLM决策"到"性能提升"的完整因果链。

---

## 2. 基线实验方案

基线实验旨在回答核心问题：**"AAL-SD相比于现有方法有多大的性能提升？"**

### 2.1 基线组设置

我们设计以下基线组，涵盖传统AL策略、领域内关键工作、以及随机基线：

| 基线组编号 | 基线组名称 | 方法描述 | 对应文献 | 实验目的 |
| :---: | :--- | :--- | :--- | :--- |
| **B1** | **Random Sampling** | 随机选择样本进行标注 | 标准基线 | 证明主动学习的必要性 |
| **B2** | **Entropy Sampling** | 基于像素级平均熵的不确定性采样（纯不确定性） | Settles (2009) | 证明单一不确定性策略的局限性 |
| **B3** | **CoreSet** | 基于特征空间聚类的多样性采样（纯多样性） | Sener & Savarese (2018) | 证明单一多样性策略的局限性 |
| **B4** | **BALD** | 基于MC Dropout的贝叶斯不确定性采样 | Gal et al. (2017) | 证明更精细的不确定性度量的效果 |
| **B5** | **Fixed λ=0.5** | 固定不确定性与知识增益的权重为0.5（等权重） | 本文消融 | 证明动态调整λ_t的必要性 |
| **B6** | **DIAL-style** | 复现DIAL论文的核心策略（纯不确定性采样+交互式标注） | Lenczner et al. (2022) | 证明相比领域内关键工作的优越性 |
| **B7** | **Wang-style** | 复现Wang & Brenning的核心策略（不确定性采样+委员会查询） | Wang & Brenning (2021) | 证明相比滑坡检测领域工作的优越性 |
| **B8** | **AAL-SD (Full)** | 完整的AAL-SD框架（LLM动态调整λ_t） | 本文 | 主方法 |

### 2.2 实验协议

**数据集与划分**：与原始AAL-SD实验保持一致，使用Landslide4Sense数据集，训练池3,039张，测试池760张，初始标注151张（5%）。

**训练协议**：所有基线组使用相同的训练协议（10 epochs/round，15轮，每轮查询88张）。

**随机种子**：每个基线组运行3个不同的随机种子（42, 123, 456），报告均值与95%置信区间。

**评估指标**：
- **主要指标**：ALC（Area Under Learning Curve）
- **辅助指标**：最终mIoU、学习曲线形状（早期增长速率、后期饱和点）

### 2.3 预期结果与论证逻辑

**假设H1**：AAL-SD的ALC显著高于所有基线组（B1-B7），证明其整体优越性。

**假设H2**：纯不确定性（B2）和纯多样性（B3）的性能均低于AAL-SD，证明动态权衡的必要性。

**假设H3**：固定λ=0.5（B5）的性能低于AAL-SD，证明LLM动态调整的价值。

**假设H4**：AAL-SD优于DIAL-style（B6）和Wang-style（B7），证明相比领域内关键工作的创新性。

**论证逻辑**：通过与B1-B7的全面对比，构建一个"从弱到强"的基线梯度，证明AAL-SD在每个维度上的优越性。

---

## 3. 消融实验方案

消融实验旨在回答核心问题：**"AAL-SD的哪些组件是其性能提升的关键？"**

### 3.1 消融组设置

我们设计以下消融组，逐层剥离AAL-SD的核心组件：

| 消融组编号 | 消融组名称 | 配置描述 | 控制变量 | 实验目的 |
| :---: | :--- | :--- | :--- | :--- |
| **A0** | **AAL-SD (Full)** | 完整的AAL-SD框架 | 基线 | 主方法 |
| **A1** | **No LLM Agent** | 使用固定λ=0.5，移除LLM Agent | LLM控制器 | 量化LLM的贡献 |
| **A2** | **Random λ** | 每轮随机选择λ_t ∈ [0, 1] | LLM决策质量 | 证明LLM决策的合理性 |
| **A3** | **Rule-based Controller** | 使用简单的启发式规则调整λ_t（例如：`if mIoU_increase < 0.01: λ_t += 0.1`） | LLM vs 规则 | **关键消融**：证明LLM的必要性 |
| **A4** | **No Cold-Start** | 全程使用"标准阶段"的知识增益计算（新颖性策略） | 冷启动策略 | 量化冷启动策略的贡献 |
| **A5** | **Fixed K (Representativeness)** | 全程使用"冷启动阶段"的知识增益计算（代表性策略） | 知识增益策略 | 量化两阶段设计的贡献 |
| **A6** | **Pure Uncertainty (λ=0)** | 固定λ_t=0（纯不确定性采样） | 知识增益的作用 | 证明知识增益的必要性 |
| **A7** | **Pure Diversity (λ=1)** | 固定λ_t=1（纯知识增益采样） | 不确定性的作用 | 证明不确定性的必要性 |
| **A8** | **No Normalization** | 不对U(x)和K(x)进行min-max归一化 | 得分归一化 | 量化归一化的作用 |

### 3.2 关键消融实验：Rule-based Controller (A3)

这是**最关键的消融实验**，直接回应审稿人的核心质疑："为什么需要LLM？简单的规则是否足够？"

**规则设计**：我们设计以下三种启发式规则作为对照：

| 规则编号 | 规则名称 | 规则逻辑 | 设计理由 |
| :---: | :--- | :--- | :--- |
| **Rule-1** | **Performance-based** | `if mIoU_increase < threshold: λ_t += 0.1 else: λ_t -= 0.1` | 基于性能增长停滞调整策略 |
| **Rule-2** | **Round-based** | `λ_t = 0.2 if round < 5 else 0.8` | 早期偏向不确定性，后期偏向多样性 |
| **Rule-3** | **Hybrid** | 结合Rule-1和Rule-2的逻辑 | 更复杂的启发式规则 |

**实验目的**：证明即使是精心设计的启发式规则，也无法达到LLM的动态决策质量。

### 3.3 实验协议

**数据集与划分**：与基线实验保持一致。

**训练协议**：所有消融组使用相同的训练协议（10 epochs/round，15轮，每轮查询88张）。

**随机种子**：每个消融组运行3个不同的随机种子（42, 123, 456），报告均值与95%置信区间。

**评估指标**：
- **主要指标**：ALC
- **辅助指标**：最终mIoU、学习曲线形状

### 3.4 预期结果与论证逻辑

**假设H5**：A1（无LLM）的ALC显著低于A0（完整AAL-SD），量化LLM的贡献。

**假设H6**：A3（基于规则）的ALC低于A0，证明LLM的复杂推理能力优于简单规则。

**假设H7**：A4（无冷启动）和A5（固定代表性）的ALC均低于A0，证明两阶段知识增益设计的有效性。

**假设H8**：A6（纯不确定性）和A7（纯多样性）的ALC均显著低于A0，证明动态权衡的必要性。

**论证逻辑**：通过逐层剥离，构建一个"组件贡献金字塔"，清晰展示每个设计选择的独立价值。

---

## 4. 实验优先级与资源规划

### 4.1 实验优先级排序

根据对论文说服力的影响程度，我们将实验分为三个优先级：

| 优先级 | 实验组 | 实验目的 | 预计GPU时间 | 关键性 |
| :---: | :--- | :--- | :--- | :--- |
| **P0（最高）** | B1, B2, B3, B5, A1, A3 | 证明LLM的必要性、动态调整的价值 | 约36小时 | **必须完成** |
| **P1（高）** | B6, B7, A4, A5, A6, A7 | 证明相比领域内工作的优越性、组件贡献量化 | 约36小时 | **强烈建议完成** |
| **P2（中）** | B4, A2, A8 | 补充对比、细节优化 | 约18小时 | **可选** |

**总计GPU时间**：约90小时（P0+P1+P2），如果资源有限，优先完成P0和P1。

### 4.2 时间规划

| 阶段 | 任务 | 预计时间 |
| :--- | :--- | :--- |
| **第1周** | P0实验（B1, B2, B3, B5, A1, A3）：代码实现+实验运行 | 5个工作日 |
| **第2周** | P1实验（B6, B7, A4, A5, A6, A7）：代码实现+实验运行 | 5个工作日 |
| **第3周** | P2实验（B4, A2, A8）+结果整理 | 3个工作日 |
| **第3-4周** | 论文撰写、图表制作 | 2个工作日 |

**总计**：约3周（15个工作日）。

---

## 5. 实施路径与代码修改指南

### 5.1 基线实验的代码实现

**B1: Random Sampling**  
- 修改`src/core/sampler.py`，增加`RandomSampler`类，随机选择样本。

**B2: Entropy Sampling**  
- 修改`src/core/sampler.py`，增加`EntropySampler`类，仅使用不确定性得分。

**B3: CoreSet**  
- 修改`src/core/sampler.py`，增加`CoreSetSampler`类，使用K-Means聚类选择最具代表性的样本。

**B4: BALD**  
- 修改`src/core/model.py`，实现MC Dropout的多次前向传播。
- 修改`src/core/sampler.py`，增加`BALDSampler`类，计算BALD不确定性。

**B5: Fixed λ=0.5**  
- 修改`src/agent/agent_manager.py`，增加配置参数`lambda_mode='fixed'`，固定λ_t=0.5。

**B6: DIAL-style**  
- 修改`src/core/sampler.py`，增加`DIALSampler`类，使用纯不确定性采样。
- 注：DIAL的交互式标注部分可以简化，仅保留其核心采样策略。

**B7: Wang-style**  
- 修改`src/core/sampler.py`，增加`WangSampler`类，结合不确定性采样和委员会查询（使用多个模型的不一致性）。

### 5.2 消融实验的代码实现

**A1: No LLM Agent**  
- 修改`src/agent/agent_manager.py`，增加配置参数`use_agent=False`，固定λ_t=0.5。

**A2: Random λ**  
- 修改`src/agent/agent_manager.py`，增加配置参数`lambda_mode='random'`，每轮随机选择λ_t。

**A3: Rule-based Controller**  
- 修改`src/agent/agent_manager.py`，增加`RuleBasedController`类，实现Rule-1、Rule-2、Rule-3三种规则。

**A4: No Cold-Start**  
- 修改`src/core/sampler.py`，增加配置参数`knowledge_strategy='always_novelty'`，全程使用新颖性策略。

**A5: Fixed K (Representativeness)**  
- 修改`src/core/sampler.py`，增加配置参数`knowledge_strategy='always_representativeness'`，全程使用代表性策略。

**A6: Pure Uncertainty (λ=0)**  
- 修改`src/agent/agent_manager.py`，增加配置参数`lambda_mode='fixed'`，固定λ_t=0。

**A7: Pure Diversity (λ=1)**  
- 修改`src/agent/agent_manager.py`，增加配置参数`lambda_mode='fixed'`，固定λ_t=1。

**A8: No Normalization**  
- 修改`src/core/sampler.py`，增加配置参数`normalize_scores=False`，不进行min-max归一化。

### 5.3 实验脚本自动化

创建`experiments/run_all_baselines_ablations.py`，自动化运行所有基线和消融实验：

```python
# 伪代码示例
experiments = {
    'B1': {'sampler': 'random'},
    'B2': {'sampler': 'entropy'},
    'B3': {'sampler': 'coreset'},
    'B5': {'sampler': 'adkucs', 'lambda_mode': 'fixed', 'lambda_value': 0.5},
    'A1': {'sampler': 'adkucs', 'use_agent': False},
    'A3': {'sampler': 'adkucs', 'controller': 'rule-based', 'rule_type': 'rule-1'},
    # ... 其他实验组
}

for exp_name, config in experiments.items():
    for seed in [42, 123, 456]:
        run_experiment(exp_name, config, seed)
```

### 5.4 结果可视化

创建`experiments/visualize_results.py`，自动化生成以下图表：

1. **学习曲线对比图**：所有基线组和消融组在同一张图上，展示mIoU随标注样本数的变化。
2. **ALC柱状图**：所有基线组和消融组的ALC对比，带误差棒（95%置信区间）。
3. **组件贡献分析图**：量化每个组件的ALC贡献百分比。
4. **λ_t轨迹对比图**：对比AAL-SD、Rule-based Controller、Random λ的λ_t轨迹。

---

## 6. 总结

本实验方案严格对齐AAL-SD的三大核心创新点，通过**8个基线组**和**8个消融组**的全面对比，构建了一个从"横向对比"到"纵向剖析"的完整证据链。

**关键亮点**：
1. **Rule-based Controller消融实验**：直接回应"LLM的必要性"质疑，是最关键的实验。
2. **DIAL-style和Wang-style基线**：直接对比领域内关键工作，证明AAL-SD的优越性。
3. **优先级排序**：确保在资源有限的情况下，优先完成最关键的实验（P0和P1）。

遵循本方案，您的论文将具备坚实的实验支撑，能够有效回应审稿人的质疑，最大化地展示AAL-SD的学术价值。
