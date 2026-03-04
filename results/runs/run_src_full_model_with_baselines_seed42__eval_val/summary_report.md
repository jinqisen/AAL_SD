# AAL-SD 实验结果摘要报告

**生成时间**: 2026-02-27 23:27:36

---

## 1. 实验概览

本报告总结了AAL-SD框架在Landslide4Sense数据集上的基线对比和消融实验结果。

### 1.1 实验配置

| 配置项 | 值 |
|-------|-----|
| 数据集 | Landslide4Sense |
| 初始标注比例 | 5.0% |
| 测试集比例 | 20.0% |
| 主动学习轮数 | 15 |
| 每轮查询样本数 | 88 |
| 总标注预算 | 1519 |

### 1.2 实验列表

| 实验名称 | 描述 | 状态 |
|---------|------|------|
| full_model | 完整模型（最佳λ日程）：R1-2 λ=0.0；R3 warmup=0.2；R4+ 风险闭环λ（不调整query_size） | ❌ |
| full_model_policy_lambda | 完整模型（规则闭环λ）：固定Warmup(0.2) + 自适应阈值 rollback + 风险驱动 λ 闭环控制（不调整 query_size） | ❌ |
| full_model_fixed_epochs_lambda_budget | 消融（固定epochs=10）：仅允许控制λ与query_size | ❌ |
| full_model_default_thresholds | 完整模型（旧阈值）：固定Warmup(0.2)+默认过拟合阈值(0.8/0.5) + 自适应阈值 rollback + 风险驱动 λ 闭环控制（不调整 query_size） | ❌ |
| full_model_v5_calibrated_risk | 完整模型V5：U校准 + 风险CI触发 + λ平滑与后期偏U约束 | ✅ |
| fixed_k | 消融：固定K阶段（保持coreset-to-labeled），仅保留λ闭环 | ❌ |
| fixed_lambda | 消融：固定λ=0.5（对应方案A1：No LLM Agent） | ✅ |
| knowledge_only | 固定λ=1，仅使用知识增益 | ✅ |
| no_cold_start | 消融：去除冷启动与warmup，直接进入风险闭环 | ❌ |
| no_agent | 消融：移除Agent；λ由sigmoid自适应(随标注进度变化)，仅使用AD-KUCS数值策略选样 | ✅ |
| no_normalization | 消融：不进行U/K归一化 | ✅ |
| random_lambda | 消融：随机λ控制（无LLM） | ✅ |
| baseline_random | 随机采样基线 | ✅ |
| baseline_entropy | 熵采样基线 | ✅ |
| baseline_coreset | Core-Set采样基线 | ✅ |
| baseline_bald | BALD采样基线 | ✅ |
| baseline_wang_style | Wang-style基线：两阶段U→K重排 | ✅ |
| baseline_llm_us | LLM仅基于不确定性分数 | ✅ |
| baseline_llm_rs | LLM仅基于随机分数 | ✅ |
| baseline_dial_style | DIAL-style基线：分簇多样性约束+不确定性 | ✅ |

---
## 2. 核心结果对比

### 2.1 性能指标汇总

| 实验名称 | ALC | 最终 mIoU | 最终 F1-Score |
|---------|-----|----------|--------------|
| full_model_v5_calibrated_risk | 0.2297 | 0.7069 | 0.7958 |
| fixed_lambda | 0.0648 | 0.7107 | 0.7995 |
| knowledge_only | 0.0636 | 0.7192 | 0.8076 |
| no_agent | 0.0626 | 0.7181 | 0.8064 |
| no_normalization | 0.0635 | 0.7192 | 0.8076 |
| random_lambda | 0.0629 | 0.7260 | 0.8141 |
| baseline_random | 0.0600 | 0.7018 | 0.7904 |
| baseline_entropy | 0.0636 | 0.7141 | 0.8027 |
| baseline_coreset | 0.0631 | 0.7179 | 0.8066 |
| baseline_bald | 0.0641 | 0.7312 | 0.8188 |
| baseline_wang_style | 0.0621 | 0.7186 | 0.8070 |
| baseline_llm_us | 0.0627 | 0.7281 | 0.8158 |
| baseline_llm_rs | 0.0621 | 0.7221 | 0.8102 |
| baseline_dial_style | 0.0640 | 0.7109 | 0.7997 |

### 2.2 性能排名

**按ALC排名**:

1. full_model_v5_calibrated_risk: 0.2297
2. fixed_lambda: 0.0648
3. baseline_bald: 0.0641
4. baseline_dial_style: 0.0640
5. knowledge_only: 0.0636
6. baseline_entropy: 0.0636
7. no_normalization: 0.0635
8. baseline_coreset: 0.0631
9. random_lambda: 0.0629
10. baseline_llm_us: 0.0627
11. no_agent: 0.0626
12. baseline_wang_style: 0.0621
13. baseline_llm_rs: 0.0621
14. baseline_random: 0.0600

**按最终mIoU排名**:

1. baseline_bald: 0.7312
2. baseline_llm_us: 0.7281
3. random_lambda: 0.7260
4. baseline_llm_rs: 0.7221
5. knowledge_only: 0.7192
6. no_normalization: 0.7192
7. baseline_wang_style: 0.7186
8. no_agent: 0.7181
9. baseline_coreset: 0.7179
10. baseline_entropy: 0.7141
11. baseline_dial_style: 0.7109
12. fixed_lambda: 0.7107
13. full_model_v5_calibrated_risk: 0.7069
14. baseline_random: 0.7018

### 2.3 topk回退统计

| 实验名称 | 回退轮次数 | 回退补齐总数 | 回退比例 |
|---------|-----------|------------|---------|
| full_model_v5_calibrated_risk | 0 | 0 | 0.00% |
| fixed_lambda | 0 | 0 | 0.00% |
| knowledge_only | 0 | 0 | 0.00% |
| no_agent | 0 | 0 | 0.00% |
| no_normalization | 0 | 0 | 0.00% |
| random_lambda | 0 | 0 | 0.00% |
| baseline_random | 0 | 0 | 0.00% |
| baseline_entropy | 0 | 0 | 0.00% |
| baseline_coreset | 0 | 0 | 0.00% |
| baseline_bald | 0 | 0 | 0.00% |
| baseline_wang_style | 0 | 0 | 0.00% |
| baseline_llm_us | 0 | 0 | 0.00% |
| baseline_llm_rs | 0 | 0 | 0.00% |
| baseline_dial_style | 0 | 0 | 0.00% |

---

*报告生成于 2026-02-27 23:27:36*
