# AAL-SD 实验结果摘要报告

**生成时间**: 2026-03-25 01:18:24

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
| full_model_A_lambda_policy | 0.5925 | 0.6982068326424853 | 0.7868719352924511 | 0.6835 | 0.7724 |
| full_model_B_lambda_agent | 0.5969 | 0.736648887417037 | 0.8235866047267313 | 0.7397 | 0.8267 |
| no_agent | 0.5975 | 0.7168672472877536 | 0.8055343321649362 | 0.6900 | 0.7798 |
| fixed_lambda | 0.5998 | 0.7280792302513821 | 0.8159401350998905 | 0.7266 | 0.8151 |
| uncertainty_only | 0.6025 | 0.7397653355419184 | 0.8262935004277383 | 0.6947 | 0.7840 |
| knowledge_only | 0.5949 | 0.7506990259090165 | 0.8361061490780249 | 0.7260 | 0.8145 |

### 2.2 性能排名

**按ALC排名**:

1. uncertainty_only: 0.6025
2. fixed_lambda: 0.5998
3. no_agent: 0.5975
4. full_model_B_lambda_agent: 0.5969
5. knowledge_only: 0.5949
6. full_model_A_lambda_policy: 0.5925

**按最终报告mIoU(test)排名**:

1. full_model_B_lambda_agent: 0.7397
2. fixed_lambda: 0.7266
3. knowledge_only: 0.7260
4. uncertainty_only: 0.6947
5. no_agent: 0.6900
6. full_model_A_lambda_policy: 0.6835

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

*报告生成于 2026-03-25 01:18:24*
