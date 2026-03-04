# AAL-SD 实验数据采集与分析方案

**作者**: Manus AI  
**日期**: 2026年2月14日

---

## 目录

1. [数据采集方案总体设计](#1-数据采集方案总体设计)
2. [实验数据采集清单](#2-实验数据采集清单)
3. [数据分析方案](#3-数据分析方案)
4. [可视化与呈现方案](#4-可视化与呈现方案)
5. [论文论证逻辑与证据链](#5-论文论证逻辑与证据链)
6. [实施指南与代码框架](#6-实施指南与代码框架)

---

## 1. 数据采集方案总体设计

### 1.1 设计原则

实验数据采集遵循以下三个核心原则，确保能够构建完整的证据链：

**原则1：可追溯性（Traceability）**  
每个实验结果都必须能够追溯到具体的配置、随机种子、训练轮次，确保实验的可复现性。

**原则2：多粒度性（Multi-granularity）**  
采集从"全局指标"（如最终mIoU）到"细粒度过程数据"（如每轮选择的样本ID、梯度统计）的多层次数据，支持不同深度的分析。

**原则3：因果可解释性（Causal Interpretability）**  
采集的数据必须能够支撑"LLM决策 → 样本分布变化 → 模型训练动态 → 性能提升"的完整因果链分析。

### 1.2 数据采集的三个层次

| 层次 | 数据类型 | 采集频率 | 用途 |
| :--- | :--- | :--- | :--- |
| **L1：全局指标** | 最终性能、ALC、训练时间 | 每个实验结束时 | 横向对比、基线评估 |
| **L2：轮次级数据** | 每轮mIoU、λ_t、样本选择统计 | 每轮主动学习结束时 | 学习曲线、策略演化分析 |
| **L3：样本级数据** | 每个样本的U(x)、K(x)、选择概率 | 每轮采样时 | 样本分布分析、可解释性分析 |

---

## 2. 实验数据采集清单

### 2.1 L1：全局指标（实验级别）

每个实验运行结束时，采集以下全局指标：

| 指标名称 | 数据类型 | 计算方式 | 用途 |
| :--- | :--- | :--- | :--- |
| **final_miou** | float | 最后一轮在测试集上的mIoU | 最终性能对比 |
| **alc** | float | 学习曲线下面积（Area Under Learning Curve） | 标注效率对比 |
| **total_training_time** | float (hours) | 所有轮次的累计训练时间 | 效率分析 |
| **total_sampling_time** | float (seconds) | 所有轮次的累计采样时间 | 采样策略的计算开销 |
| **convergence_round** | int | mIoU达到90%最终性能的轮次 | 收敛速度分析 |
| **experiment_config** | dict | 实验配置（sampler类型、λ模式等） | 可追溯性 |
| **random_seed** | int | 随机种子 | 可复现性 |

**存储格式**：JSON文件  
**文件路径**：`results/runs/<run_id>/<exp>_status.json`（`result` 字段包含 `alc/final_mIoU/final_f1/budget_history` 等）  
**多 seed 汇总（如生成）**：`results/runs/<run_id>/multi_seed_summary.json`

### 2.2 L2：轮次级数据（主动学习循环级别）

每轮主动学习结束时，采集以下数据：

#### 2.2.1 性能指标

| 指标名称 | 数据类型 | 计算方式 | 用途 |
| :--- | :--- | :--- | :--- |
| **round** | int | 当前轮次编号（0-14） | 时间轴 |
| **train_miou** | float | 当前轮次在训练集上的mIoU | 过拟合监控 |
| **val_miou** | float | 当前轮次在验证集上的mIoU | 学习曲线主指标 |
| **test_miou** | float | 当前轮次在测试集上的mIoU | 泛化能力评估 |
| **miou_increase** | float | 相比上一轮的mIoU增长 | 性能增长速率 |
| **labeled_samples_count** | int | 当前已标注样本总数 | 标注预算 |

#### 2.2.2 控制策略 / Agent 决策数据（视实验配置启用）

| 指标名称 | 数据类型 | 计算方式 | 用途 |
| :--- | :--- | :--- | :--- |
| **lambda_t** | float | 控制器侧策略/Agent（若启用）给出的 λ_t | 策略演化分析 |
| **agent_justification** | str | LLM生成的决策理由（如果实现可解释性） | 可解释性分析 |
| **agent_input_context** | dict | 输入给LLM的上下文（mIoU、梯度统计等） | 决策依据分析 |
| **agent_response_time** | float (seconds) | LLM推理时间 | 效率分析 |

#### 2.2.3 梯度证据数据

| 指标名称 | 数据类型 | 计算方式 | 用途 |
| :--- | :--- | :--- | :--- |
| **train_val_cos** | float | 训练集与验证集梯度的余弦相似度 | 梯度一致性分析 |
| **grad_norm_train** | float | 训练集梯度的L2范数 | 梯度幅度分析 |
| **grad_norm_val** | float | 验证集梯度的L2范数 | 梯度幅度分析 |
| **grad_variance** | float | 梯度的方差 | 训练稳定性分析 |

#### 2.2.4 样本选择统计

| 指标名称 | 数据类型 | 计算方式 | 用途 |
| :--- | :--- | :--- | :--- |
| **selected_sample_ids** | list[int] | 本轮选择的样本ID列表 | 样本分布分析 |
| **avg_uncertainty** | float | 选择样本的平均不确定性 | 采样策略分析 |
| **avg_knowledge_gain** | float | 选择样本的平均知识增益 | 采样策略分析 |
| **sample_diversity_score** | float | 选择样本的多样性得分（如平均簇间距离） | 采样策略分析 |
| **class_distribution** | dict | 选择样本的类别分布（滑坡/非滑坡） | 类别平衡分析 |

**存储格式**：trace（JSONL）与 status（JSON）为主  
**文件路径**：`results/runs/<run_id>/<exp>_trace.jsonl`（`epoch_end/round_summary/controller_step/lambda_policy_apply/overfit_signal` 等）与 `results/runs/<run_id>/<exp>_status.json`

### 2.3 L3：样本级数据（采样级别）

每轮采样时，对**未标注池中的所有样本**采集以下数据：

| 指标名称 | 数据类型 | 计算方式 | 用途 |
| :--- | :--- | :--- | :--- |
| **sample_id** | int | 样本ID | 唯一标识 |
| **uncertainty_score** | float | 不确定性得分U(x) | 采样策略分析 |
| **knowledge_gain_score** | float | 知识增益得分K(x) | 采样策略分析 |
| **combined_score** | float | 综合得分S(x) = (1-λ)U(x) + λK(x) | 采样策略分析 |
| **is_selected** | bool | 是否被选中 | 样本分布分析 |
| **cluster_id** | int | 所属簇ID（用于知识增益计算） | 聚类分析 |
| **feature_vector** | array | 特征向量（用于t-SNE可视化） | 特征空间可视化 |

**存储格式**：默认以 trace（JSONL）记录，可复现/可回放  
**文件路径**：`results/runs/<run_id>/<exp>_trace.jsonl`（事件包括 `epoch_end`、`controller_step`、`lambda_policy_apply`、`l3_selection` 等）

**注意**：由于样本级数据量巨大（每轮~3000样本 × 15轮 × 3种子），建议仅在需要深度分析时采集（如AAL-SD、关键消融组）。

---

## 3. 数据分析方案

### 3.1 分析维度与对应的核心问题

| 分析维度 | 核心问题 | 使用的数据 | 分析方法 |
| :--- | :--- | :--- | :--- |
| **A1：整体性能对比** | AAL-SD相比基线组有多大提升？ | `*_status.json` 的 `result.alc/final_mIoU/final_f1` 与多 seed 汇总 | 统计显著性检验（t-test）、ALC对比 |
| **A2：学习曲线分析** | AAL-SD的学习曲线形状有何特点？ | `*_trace.jsonl` 的 `round_summary/epoch_end` 与 `result.budget_history` | 曲线拟合、早期增长速率、后期饱和点 |
| **A3：策略演化分析** | λ_t 如何逐轮变化？ | `*_trace.jsonl` 的 `lambda_policy_apply.applied` 或 `controller_step.action.lambda` | λ_t轨迹可视化、与性能关联分析 |
| **A4：梯度证据分析** | λ_t 与梯度一致性的关系？ | `epoch_end.grad.train_val_cos`、`overfit_signal.grad_train_val_cos_*`/`overfit_risk` 与 λ_t | 相关性/分段统计、典型轮次案例分析 |
| **A5：样本分布分析** | AAL-SD选择的样本有何特点？ | `l3_selection`（top-k/selected 的 U/K/λ_t）与 pools CSV | U-K 平面分布、top-k 近似尾部分析 |
| **A6：可解释性分析** | 决策理由是否与状态/动作一致？ | `controller_step.reasoning`（以及 LLM 请求事件如启用） | 案例分析、理由-动作-结果一致性 |
| **A7：组件贡献量化** | 每个组件贡献了多少ALC？ | 各消融组 `*_status.json` 的 `result.alc` | ALC差值计算、贡献百分比 |

### 3.2 详细分析方案

#### 3.2.1 A1：整体性能对比分析

**目标**：证明AAL-SD在ALC和最终mIoU上显著优于所有基线组。

**数据来源**：所有实验组的 `results/runs/<run_id>/<exp>_status.json`（`result.alc/final_mIoU/final_f1/budget_history`）；多 seed 场景可使用 `src/analysis/plot_paper_figures.py` 读取 `multi_seed_summary.json`（若存在）进行汇总统计。

**分析步骤**：
1. 计算每个实验组的ALC均值与95%置信区间（3个种子）
2. 进行配对t检验（paired t-test），检验AAL-SD与每个基线组的ALC差异是否显著（p < 0.05）
3. 计算效应量（Cohen's d），量化提升幅度
4. 生成ALC柱状图，标注显著性（*表示p < 0.05，**表示p < 0.01）

**预期产出**：
- 表格：所有实验组的ALC、最终mIoU、统计显著性
- 图表：ALC柱状图（带误差棒和显著性标注）

#### 3.2.2 A2：学习曲线分析

**目标**：分析AAL-SD的学习曲线形状，证明其在早期和后期都具有优势。

**数据来源**：`results/runs/<run_id>/<exp>_trace.jsonl`（`epoch_end`/`round_summary`）与 `*_status.json` 的 `result.budget_history`。

**分析步骤**：
1. 绘制所有实验组的学习曲线（val_miou vs labeled_samples_count）
2. 计算早期增长速率（前5轮的平均mIoU增长）
3. 计算后期饱和点（mIoU达到90%最终性能的轮次）
4. 拟合学习曲线（如对数函数、幂律函数），分析曲线参数

**预期产出**：
- 图表：学习曲线对比图（所有实验组在同一张图上）
- 表格：早期增长速率、后期饱和点对比

#### 3.2.3 A3：策略演化分析

**目标**：可视化LLM如何动态调整λ_t，证明其决策的合理性。

**数据来源**：`*_trace.jsonl` 的 `lambda_policy_apply.applied`（policy λ）与/或 `controller_step.action.lambda`（闭环控制动作 λ）。

**分析步骤**：
1. 绘制λ_t随轮次的变化轨迹（AAL-SD、Rule-based Controller、Random λ对比）
2. 分析λ_t的变化模式（如早期偏向不确定性、后期偏向多样性）
3. 标注关键转折点（如λ_t发生显著变化的轮次），分析其与性能变化的关联

**预期产出**：
- 图表：λ_t轨迹对比图（AAL-SD vs Rule-based vs Random）
- 文本：关键转折点的案例分析

#### 3.2.4 A4：梯度证据分析

**目标**：构建"LLM调整λ_t → 样本分布改变 → 梯度方向变化 → 性能提升"的完整因果链。

**数据来源**：`*_trace.jsonl` 的 `epoch_end.grad.train_val_cos`（轮内序列）、`overfit_signal.grad_train_val_cos_*`/`overfit_risk`（轮聚合信号）以及 `lambda_policy_apply.applied`/`controller_step.action.lambda`（λ_t）。跨轮增益可用 `round_summary.mIoU` 或 `training_state.miou_delta` 推导。

**分析步骤**：
1. 绘制λ_t与train_val_cos的双轴时间序列图，观察两者的关联
2. 计算λ_t与train_val_cos的Pearson相关系数，检验相关性显著性
3. 绘制train_val_cos与miou_increase的散点图，分析梯度一致性与性能增长的关系
4. 选择典型的"决策转折点"（如λ_t发生显著变化的轮次），深入分析该轮的样本选择、梯度变化、性能变化

**预期产出**：
- 图表1：λ_t与train_val_cos的双轴时间序列图
- 图表2：train_val_cos与miou_increase的散点图
- 表格：Pearson相关系数与p值
- 文本：典型决策转折点的案例分析

#### 3.2.5 A5：样本分布分析

**目标**：分析AAL-SD选择的样本在不确定性-知识增益空间中的分布特点。

**数据来源**：默认使用 `*_trace.jsonl` 的 `l3_selection` 事件（top-k 与 selected 的 `uncertainty/knowledge_gain/lambda_t`）；若需要“全候选池分布”，需在运行时额外持久化分位数/直方图或离线重算。

**分析步骤**：
1. 选择典型轮次（如第1、5、10、15轮），绘制不确定性-知识增益散点图
2. 用不同颜色标注"已选择"和"未选择"的样本
3. 分析选择样本的分布模式（如是否覆盖高不确定性+高知识增益的区域）
4. 使用t-SNE对特征向量降维，可视化选择样本在特征空间中的分布

**预期产出**：
- 图表1：不确定性-知识增益散点图（4个典型轮次）
- 图表2：t-SNE特征空间可视化（选择样本 vs 未选择样本）

#### 3.2.6 A6：可解释性分析

**目标**：分析LLM生成的决策理由是否合理，增强论文的可信度。

**数据来源**：`*_trace.jsonl` 的 `controller_step.reasoning`（以及启用后可用的 LLM 请求/响应事件）。

**分析步骤**：
1. 选择典型的决策转折点（如λ_t发生显著变化的轮次）
2. 提取LLM生成的决策理由，分析其是否与性能变化、梯度证据一致
3. 构建"决策理由 → λ_t调整 → 样本分布变化 → 性能变化"的完整案例

**预期产出**：
- 文本：3-5个典型决策转折点的案例分析（包含LLM决策理由、λ_t变化、性能变化）

#### 3.2.7 A7：组件贡献量化

**目标**：量化AAL-SD的每个核心组件（LLM控制器、冷启动策略、动态权衡）对ALC的贡献。

**数据来源**：所有消融组 `*_status.json` 的 `result.alc`（必要时结合 `manifest.json`/multi-seed 汇总）。

**分析步骤**：
1. 计算每个消融组相对于完整AAL-SD的ALC差值（ΔALC）
2. 将ΔALC转化为贡献百分比（ΔALC / ALC_full × 100%）
3. 生成组件贡献金字塔图，展示每个组件的独立贡献

**预期产出**：
- 表格：消融组ALC对比与贡献百分比
- 图表：组件贡献金字塔图

---

## 4. 可视化与呈现方案

### 4.1 论文核心图表清单

根据论文的论证逻辑，我们需要生成以下核心图表：

| 图表编号 | 图表类型 | 标题 | 对应分析 | 论文位置 |
| :---: | :--- | :--- | :--- | :--- |
| **Fig. 1** | 学习曲线图 | Learning Curves of AAL-SD and Baselines | A2 | 实验结果主图 |
| **Fig. 2** | ALC柱状图 | ALC Comparison Across Methods | A1 | 实验结果主图 |
| **Fig. 3** | λ_t轨迹图 | Evolution of λ_t in AAL-SD vs Rule-based | A3 | 策略演化分析 |
| **Fig. 4** | 梯度证据图 | Correlation between λ_t and Gradient Consistency | A4 | 因果链论证 |
| **Fig. 5** | 样本分布图 | Sample Distribution in Uncertainty-Knowledge Space | A5 | 采样策略分析 |
| **Fig. 6** | 组件贡献图 | Ablation Study: Component Contributions | A7 | 消融研究 |
| **Table 1** | 性能对比表 | Performance Comparison of All Methods | A1 | 实验结果主表 |
| **Table 2** | 消融研究表 | Ablation Study Results | A7 | 消融研究 |

### 4.2 图表设计规范

**风格统一**：所有图表使用统一的配色方案（如AAL-SD用红色、基线组用灰色系）、字体（Times New Roman）、线型。

**信息密度**：每张图表应传达明确的核心信息，避免信息过载。

**学术规范**：
- 所有图表必须有清晰的标题、轴标签、图例
- 误差棒使用95%置信区间
- 显著性标注使用标准符号（*表示p < 0.05，**表示p < 0.01）

### 4.3 关键图表的详细设计

#### Fig. 1: Learning Curves of AAL-SD and Baselines

**设计要点**：
- X轴：已标注样本数（Labeled Samples）
- Y轴：验证集mIoU（Validation mIoU）
- 曲线：AAL-SD（红色粗线）、关键基线组（灰色细线）
- 阴影区域：95%置信区间
- 标注：在AAL-SD曲线的关键点标注轮次编号

**呈现效果**：一眼看出AAL-SD的学习曲线始终在基线组之上，证明其标注效率优势。

#### Fig. 4: Correlation between λ_t and Gradient Consistency

**设计要点**：
- 双轴时间序列图：
  - 左Y轴：λ_t（蓝色折线）
  - 右Y轴：train_val_cos（橙色折线）
  - X轴：轮次（Round）
- 标注：在λ_t发生显著变化的轮次标注"Decision Turning Point"
- 插图：Pearson相关系数与p值

**呈现效果**：直观展示LLM调整λ_t与梯度一致性的关联，构建因果链。

---

## 5. 论文论证逻辑与证据链

### 5.1 核心论点与证据链映射

| 核心论点 | 证据链 | 使用的图表/表格 |
| :--- | :--- | :--- |
| **论点1：AAL-SD整体优于现有方法** | 基线实验 → ALC显著提升 → 学习曲线优势 | Fig. 1, Fig. 2, Table 1 |
| **论点2：LLM控制器优于规则** | 消融实验（A3） → λ_t轨迹对比 → 性能差异 | Fig. 3, Table 2 |
| **论点3：动态权衡优于固定策略** | 消融实验（A1, A6, A7） → ALC对比 → 组件贡献 | Fig. 6, Table 2 |
| **论点4：LLM决策有梯度证据支撑** | 梯度证据分析 → λ_t与train_val_cos关联 → 因果链 | Fig. 4 |
| **论点5：AAL-SD选择高质量样本** | 样本分布分析 → 不确定性-知识增益空间覆盖 | Fig. 5 |

### 5.2 论文结构与证据呈现顺序

**第4节：实验结果（Experimental Results）**

**4.1 整体性能对比（Overall Performance Comparison）**  
- 呈现：Table 1, Fig. 1, Fig. 2
- 论证：AAL-SD在ALC和最终mIoU上显著优于所有基线组

**4.2 消融研究（Ablation Study）**  
- 呈现：Table 2, Fig. 6
- 论证：量化每个组件的贡献，证明LLM控制器、动态权衡的必要性

**4.3 策略演化分析（Strategy Evolution Analysis）**  
- 呈现：Fig. 3
- 论证：LLM动态调整λ_t的合理性，优于规则和随机策略

**4.4 梯度证据分析（Gradient Evidence Analysis）**  
- 呈现：Fig. 4
- 论证：构建"LLM决策 → 梯度一致性 → 性能提升"的完整因果链

**4.5 样本分布分析（Sample Distribution Analysis）**  
- 呈现：Fig. 5
- 论证：AAL-SD选择的样本在不确定性-知识增益空间中的分布特点

---

## 6. 实施指南与代码框架

### 6.1 数据采集的代码实现

**修改`src/core/trainer.py`**：在训练循环中记录L2轮次级数据

```python
# 伪代码示例
class Trainer:
    def train_one_round(self, round_idx):
        # ... 训练逻辑 ...
        
        # 采集L2数据
        round_metrics = {
            'round': round_idx,
            'train_miou': self.evaluate(self.train_loader),
            'val_miou': self.evaluate(self.val_loader),
            'test_miou': self.evaluate(self.test_loader),
            'miou_increase': current_miou - prev_miou,
            'labeled_samples_count': len(self.labeled_pool),
            'train_val_cos': self.compute_gradient_cosine(),
            'grad_norm_train': self.compute_gradient_norm('train'),
            'grad_norm_val': self.compute_gradient_norm('val'),
        }
        
        # 保存到CSV
        self.save_round_metrics(round_metrics)
```

**修改`src/core/sampler.py`**：在采样时记录L3样本级数据

```python
# 伪代码示例
class ADKUCSSampler:
    def select_samples(self, unlabeled_pool, k):
        # 计算所有样本的得分
        sample_scores = []
        for sample in unlabeled_pool:
            u_score = self.compute_uncertainty(sample)
            k_score = self.compute_knowledge_gain(sample)
            combined_score = (1 - self.lambda_t) * u_score + self.lambda_t * k_score
            
            sample_scores.append({
                'sample_id': sample.id,
                'uncertainty_score': u_score,
                'knowledge_gain_score': k_score,
                'combined_score': combined_score,
                'cluster_id': sample.cluster_id,
            })
        
        # 选择Top-K
        selected_samples = sorted(sample_scores, key=lambda x: x['combined_score'], reverse=True)[:k]
        
        # 标注is_selected
        for score in sample_scores:
            score['is_selected'] = score['sample_id'] in [s['sample_id'] for s in selected_samples]
        
        # 保存到HDF5
        self.save_sample_scores(sample_scores, round_idx)
        
        return selected_samples
```

**修改`src/agent/agent_manager.py`**：记录LLM决策数据

```python
# 伪代码示例
class AgentManager:
    def decide_lambda(self, context):
        # LLM推理
        start_time = time.time()
        response = self.llm.generate(context)
        response_time = time.time() - start_time
        
        # 解析λ_t和决策理由
        lambda_t = self.parse_lambda(response)
        justification = self.parse_justification(response)
        
        # 记录决策数据
        decision_data = {
            'lambda_t': lambda_t,
            'agent_justification': justification,
            'agent_input_context': context,
            'agent_response_time': response_time,
        }
        
        self.save_decision_data(decision_data, round_idx)
        
        return lambda_t
```

### 6.2 数据分析的代码框架

创建`experiments/analyze_results.py`，自动化执行所有分析：

```python
# 伪代码示例
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, pearsonr

class ResultAnalyzer:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        
    def analyze_overall_performance(self):
        """A1：整体性能对比分析"""
        # 加载所有实验组的 <exp>_status.json（或 multi_seed_summary.json）
        # 计算ALC均值、置信区间
        # 进行t检验
        # 生成ALC柱状图
        pass
    
    def analyze_learning_curves(self):
        """A2：学习曲线分析"""
        # 解析所有实验组的 <exp>_trace.jsonl（epoch_end/round_summary）与 <exp>_status.json（budget_history）
        # 绘制学习曲线
        # 计算早期增长速率、后期饱和点
        pass
    
    def analyze_strategy_evolution(self):
        """A3：策略演化分析"""
        # 加载AAL-SD、Rule-based、Random的lambda_t数据
        # 绘制λ_t轨迹对比图
        pass
    
    def analyze_gradient_evidence(self):
        """A4：梯度证据分析"""
        # 加载AAL-SD的lambda_t、train_val_cos数据
        # 绘制双轴时间序列图
        # 计算Pearson相关系数
        pass
    
    def analyze_sample_distribution(self):
        """A5：样本分布分析"""
        # 解析 trace 中的 l3_selection（如启用）以及 data/pools 下的 labeled/unlabeled/test pool CSV
        # 绘制不确定性-知识增益散点图
        # 绘制t-SNE特征空间可视化
        pass
    
    def analyze_component_contribution(self):
        """A7：组件贡献量化"""
        # 加载所有消融组的 <exp>_status.json（或 multi_seed_summary.json）
        # 计算ALC差值与贡献百分比
        # 生成组件贡献金字塔图
        pass
    
    def generate_all_figures(self):
        """生成论文所需的所有图表"""
        self.analyze_overall_performance()
        self.analyze_learning_curves()
        self.analyze_strategy_evolution()
        self.analyze_gradient_evidence()
        self.analyze_sample_distribution()
        self.analyze_component_contribution()

# 使用示例
analyzer = ResultAnalyzer('results/')
analyzer.generate_all_figures()
```

### 6.3 实施检查清单

在开始实验之前，确保以下准备工作已完成：

- [ ] 修改`trainer.py`，增加L2轮次级数据采集
- [ ] 修改`sampler.py`，增加L3样本级数据采集（仅关键实验组）
- [ ] 修改`agent_manager.py`，增加LLM决策数据采集
- [ ] 使用现有分析脚本解析 trace/status（如 `src/analysis/plot_paper_figures.py`、`src/analysis/plot_strategy_trajectory.py`）
- [ ] 测试数据采集流程，确保数据格式正确
- [ ] 测试数据分析流程，确保图表生成正确

---

## 7. 总结

本文档提供了完整的实验数据采集与分析方案，确保实验的每个环节都能为论文的核心论点提供有力支撑。

**关键要点**：
1. **三层数据采集**：从全局指标到样本级数据，支持多粒度分析
2. **七维数据分析**：覆盖性能对比、策略演化、梯度证据、样本分布等所有关键维度
3. **完整证据链**：从数据采集到图表生成，构建"LLM决策 → 性能提升"的完整因果链
4. **工程可落地**：提供详细的代码实现指南，确保方案可执行

遵循本方案，您的实验将具备坚实的数据支撑，能够有效回应审稿人的质疑，最大化地展示AAL-SD的学术价值。
