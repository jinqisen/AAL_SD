# AAL-SD 实验结果摘要报告

**生成时间**: 2026-02-08 19:40:46

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
| full_model | 0.6649 | 0.7500 | 0.8362 |
| baseline_entropy | 0.6649 | 0.7631 | 0.8474 |
| baseline_random | 0.6478 | 0.7436 | 0.8309 |
| no_agent | 0.6619 | 0.7546 | 0.8402 |
| fixed_lambda | 0.6633 | 0.7569 | 0.8424 |

### 2.2 性能排名

**按ALC排名**:

1. baseline_entropy: 0.6649
2. full_model: 0.6649
3. fixed_lambda: 0.6633
4. no_agent: 0.6619
5. baseline_random: 0.6478

**按最终mIoU排名**:

1. baseline_entropy: 0.7631
2. fixed_lambda: 0.7569
3. no_agent: 0.7546
4. full_model: 0.7500
5. baseline_random: 0.7436

### 2.3 topk回退统计

| 实验名称 | 回退轮次数 | 回退补齐总数 | 回退比例 |
|---------|-----------|------------|---------|
| full_model | 0 | 0 | 0.00% |
| baseline_entropy | 0 | 0 | 0.00% |
| baseline_random | 0 | 0 | 0.00% |
| no_agent | 0 | 0 | 0.00% |
| fixed_lambda | 0 | 0 | 0.00% |

---

*报告生成于 2026-02-08 19:40:46*
