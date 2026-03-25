# AAL-SD 实验结果摘要报告

**生成时间**: 2026-03-24 14:29:51

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
| full_model_A_lambda_policy | 0.6084 | 0.7469824968211745 | 0.8326480291380034 | 0.7470 | 0.8326 |
| full_model_B_lambda_agent | 0.5922 | 0.6919264522962254 | 0.7811746558560048 | 0.6919 | 0.7812 |
| no_agent | 0.6002 | 0.683587134660879 | 0.7726084860157473 | 0.6834 | 0.7730 |
| fixed_lambda | 0.5996 | 0.7029571224586069 | 0.7920276925550707 | 0.7046 | 0.7941 |
| uncertainty_only | 0.6029 | 0.7230105420882483 | 0.8111717466954856 | 0.7189 | 0.8077 |
| knowledge_only | 0.5788 | 0.7051632478330295 | 0.7941965075969353 | 0.6883 | 0.7781 |

### 2.2 性能排名

**按ALC排名**:

1. full_model_A_lambda_policy: 0.6084
2. uncertainty_only: 0.6029
3. no_agent: 0.6002
4. fixed_lambda: 0.5996
5. full_model_B_lambda_agent: 0.5922
6. knowledge_only: 0.5788

**按最终报告mIoU(test)排名**:

1. full_model_A_lambda_policy: 0.7470
2. uncertainty_only: 0.7189
3. fixed_lambda: 0.7046
4. full_model_B_lambda_agent: 0.6919
5. knowledge_only: 0.6883
6. no_agent: 0.6834

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

*报告生成于 2026-03-24 14:29:51*
