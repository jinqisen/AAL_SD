# AAL-SD 实验结果摘要报告

**生成时间**: 2026-03-25 21:52:09

---

## 1. 实验概览

本报告总结了AAL-SD框架在Landslide4Sense数据集上的基线对比和消融实验结果。

### 1.1 实验配置

| 配置项 | 值 |
|-------|-----|
| 数据集 | Landslide4Sense |
| 初始标注比例 | 5.0% |
| train split | train |
| val split | val |
| test split | test |
| 主动学习轮数 | 16 |
| 每轮查询样本数 | 88 |
| 总标注预算 | 1519 |

### 1.2 实验列表

| 实验名称 | 描述 | 状态 |
|---------|------|------|
| full_model_A_lambda_policy | 对照A / AAL-SD (Full)：基于 auto_opt_iter15_cand2_02（warmup+risk闭环 + 后期ramp + U-guardrail；train_holdout grad_probe） | ✅ |
| full_model_B_lambda_agent | 对照B / AAL-SD (Agent-λ)：与 full_model_A_lambda_policy 完全一致，唯一区别：λ 由 LLM Agent 通过 set_lambda 显式决定，而非 policy 自动填充 | ✅ |
| no_agent | 消融：w/o Agent（argmax）；保留三阶段λ policy，与full_model一致 | ✅ |
| fixed_lambda | 消融：Fixed λ=0.5；保留Agent，隔离三阶段λ policy贡献 | ✅ |
| uncertainty_only | 固定λ=0，仅使用不确定性 | ✅ |
| knowledge_only | 固定λ=1，仅使用知识增益 | ✅ |

---
## 2. 核心结果对比

### 2.1 性能指标汇总

| 实验名称 | ALC(val选模) | 最后选模 mIoU(val) | 最后选模 F1(val) | 最终报告 mIoU(test) | 最终报告 F1(test) |
|---------|------------|------------------|---------------|----------------------|---------------------|
| full_model_A_lambda_policy | 0.5868 | 0.7108908094827846 | 0.7997725454601183 | 0.7098 | 0.7993 |
| full_model_B_lambda_agent | 0.5955 | 0.6936672076604814 | 0.782633058486983 | 0.6788 | 0.7678 |
| no_agent | 0.6012 | 0.7044854455527128 | 0.793364274288018 | 0.6934 | 0.7828 |
| fixed_lambda | 0.5953 | 0.6937431338544734 | 0.7833028592694246 | 0.6705 | 0.7596 |
| uncertainty_only | 0.6031 | 0.7161898486159108 | 0.8044862548182743 | 0.6973 | 0.7865 |
| knowledge_only | 0.5799 | 0.6580234031278527 | 0.7450123453574998 | 0.6545 | 0.7415 |

### 2.2 性能排名

**按ALC排名**:

1. uncertainty_only: 0.6031
2. no_agent: 0.6012
3. full_model_B_lambda_agent: 0.5955
4. fixed_lambda: 0.5953
5. full_model_A_lambda_policy: 0.5868
6. knowledge_only: 0.5799

**按最终报告mIoU(test)排名**:

1. full_model_A_lambda_policy: 0.7098
2. uncertainty_only: 0.6973
3. no_agent: 0.6934
4. full_model_B_lambda_agent: 0.6788
5. fixed_lambda: 0.6705
6. knowledge_only: 0.6545

### 2.3 topk回退统计

| 实验名称 | 回退轮次数 | 回退补齐总数 | 回退比例 |
|---------|-----------|------------|---------|
| full_model_A_lambda_policy | 0 | 0 | 0.00% |
| full_model_B_lambda_agent | 0 | 0 | 0.00% |
| no_agent | 0 | 0 | 0.00% |
| fixed_lambda | 0 | 0 | 0.00% |
| uncertainty_only | 0 | 0 | 0.00% |
| knowledge_only | 0 | 0 | 0.00% |

---

*报告生成于 2026-03-25 21:52:09*
