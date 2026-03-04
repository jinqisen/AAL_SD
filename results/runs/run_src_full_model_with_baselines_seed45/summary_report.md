# AAL-SD 实验结果摘要报告

**生成时间**: 2026-02-26 19:03:46

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
| baseline_random | 随机采样基线 | ✅ |
| full_model | 完整模型（最佳λ日程）：R1-2 λ=0.0；R3 warmup=0.2；R4+ 风险闭环λ（不调整query_size） | ✅ |
| baseline_entropy | 熵采样基线 | ✅ |
| baseline_coreset | Core-Set采样基线 | ✅ |
| baseline_bald | BALD采样基线 | ✅ |
| baseline_dial_style | DIAL-style基线：分簇多样性约束+不确定性 | ✅ |
| baseline_wang_style | Wang-style基线：两阶段U→K重排 | ✅ |
| baseline_llm_us | LLM仅基于不确定性分数 | ✅ |
| baseline_llm_rs | LLM仅基于随机分数 | ✅ |

---
## 2. 核心结果对比

### 2.1 性能指标汇总

| 实验名称 | ALC | 最终 mIoU | 最终 F1-Score |
|---------|-----|----------|--------------|
| baseline_random | 0.6552 | 0.7563 | 0.8416 |
| full_model | 0.6689 | 0.7632 | 0.8474 |
| baseline_entropy | 0.6701 | 0.7577 | 0.8429 |
| baseline_coreset | 0.6585 | 0.7522 | 0.8382 |
| baseline_bald | 0.6646 | 0.7567 | 0.8420 |
| baseline_dial_style | 0.6657 | 0.7573 | 0.8425 |
| baseline_wang_style | 0.6706 | 0.7594 | 0.8442 |
| baseline_llm_us | 0.6688 | 0.7683 | 0.8516 |
| baseline_llm_rs | 0.6558 | 0.7523 | 0.8382 |

### 2.2 性能排名

**按ALC排名**:

1. baseline_wang_style: 0.6706
2. baseline_entropy: 0.6701
3. full_model: 0.6689
4. baseline_llm_us: 0.6688
5. baseline_dial_style: 0.6657
6. baseline_bald: 0.6646
7. baseline_coreset: 0.6585
8. baseline_llm_rs: 0.6558
9. baseline_random: 0.6552

**按最终mIoU排名**:

1. baseline_llm_us: 0.7683
2. full_model: 0.7632
3. baseline_wang_style: 0.7594
4. baseline_entropy: 0.7577
5. baseline_dial_style: 0.7573
6. baseline_bald: 0.7567
7. baseline_random: 0.7563
8. baseline_llm_rs: 0.7523
9. baseline_coreset: 0.7522

### 2.3 topk回退统计

| 实验名称 | 回退轮次数 | 回退补齐总数 | 回退比例 |
|---------|-----------|------------|---------|
| baseline_random | 0 | 0 | 0.00% |
| full_model | 0 | 0 | 0.00% |
| baseline_entropy | 0 | 0 | 0.00% |
| baseline_coreset | 0 | 0 | 0.00% |
| baseline_bald | 0 | 0 | 0.00% |
| baseline_dial_style | 0 | 0 | 0.00% |
| baseline_wang_style | 0 | 0 | 0.00% |
| baseline_llm_us | 0 | 0 | 0.00% |
| baseline_llm_rs | 0 | 0 | 0.00% |

---

*报告生成于 2026-02-26 19:03:46*
