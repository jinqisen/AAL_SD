# AAL-SD 实验结果摘要报告

**生成时间**: 2026-02-08 17:36:40

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
| full_model | 完整模型（我们的方法）：Agent驱动的自适应梯度选样（epochs固定=10） | ✅ |
| baseline_entropy | 熵采样基线 | ✅ |
| baseline_random | 随机采样基线 | ✅ |
| no_agent | 消融：移除Agent，仅使用固定规则的AD-KUCS选样 | ✅ |
| fixed_lambda | 固定λ=0.5 | ✅ |

---
## 2. 核心结果对比

### 2.1 性能指标汇总

| 实验名称 | ALC | 最终 mIoU | 最终 F1-Score |
|---------|-----|----------|--------------|
| full_model | 0.6657 | 0.7601 | 0.8446 |
| baseline_entropy | 0.6668 | 0.7614 | 0.8458 |
| baseline_random | 0.6593 | 0.7626 | 0.8470 |
| no_agent | 0.6665 | 0.7636 | 0.8479 |
| fixed_lambda | 0.6671 | 0.7620 | 0.8463 |

### 2.2 性能排名

**按ALC排名**:

1. fixed_lambda: 0.6671
2. baseline_entropy: 0.6668
3. no_agent: 0.6665
4. full_model: 0.6657
5. baseline_random: 0.6593

**按最终mIoU排名**:

1. no_agent: 0.7636
2. baseline_random: 0.7626
3. fixed_lambda: 0.7620
4. baseline_entropy: 0.7614
5. full_model: 0.7601

### 2.3 topk回退统计

| 实验名称 | 回退轮次数 | 回退补齐总数 | 回退比例 |
|---------|-----------|------------|---------|
| full_model | 0 | 0 | 0.00% |
| baseline_entropy | 0 | 0 | 0.00% |
| baseline_random | 0 | 0 | 0.00% |
| no_agent | 0 | 0 | 0.00% |
| fixed_lambda | 0 | 0 | 0.00% |

---

*报告生成于 2026-02-08 17:36:40*
