# AAL-SD 实验结果摘要报告

**生成时间**: 2026-03-07 21:05:51

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
| full_model_A_lambda_policy | 对照A / AAL-SD (Full)：LLM 参与决策与解释；λ由 warmup+风险闭环策略生成并在工具层 clamp（启用CI阈值+AND严重判定+EMA/冷却/限步长；默认禁止 set_lambda；不调整query_size/epochs） | ❌ |
| baseline_random | 随机采样基线 | ❌ |
| baseline_entropy | 熵采样基线 | ❌ |
| baseline_dial_style | DIAL-style基线：分簇多样性约束+不确定性 | ❌ |
| baseline_coreset | Core-Set采样基线 | ❌ |
| baseline_bald | BALD采样基线 | ❌ |
| baseline_wang_style | Wang-style基线：两阶段U→K重排 | ❌ |

---

---

*报告生成于 2026-03-07 21:05:51*
