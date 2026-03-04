# AAL-SD 实验结果摘要报告

**生成时间**: 2026-02-08 19:35:08

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
| full_model | 0.6666 | 0.7583 | 0.8433 |
| baseline_entropy | 0.6654 | 0.7582 | 0.8430 |
| baseline_random | 0.6504 | 0.7476 | 0.8340 |
| no_agent | 0.6646 | 0.7607 | 0.8453 |
| fixed_lambda | 0.6674 | 0.7596 | 0.8443 |

### 2.2 性能排名

**按ALC排名**:

1. fixed_lambda: 0.6674
2. full_model: 0.6666
3. baseline_entropy: 0.6654
4. no_agent: 0.6646
5. baseline_random: 0.6504

**按最终mIoU排名**:

1. no_agent: 0.7607
2. fixed_lambda: 0.7596
3. full_model: 0.7583
4. baseline_entropy: 0.7582
5. baseline_random: 0.7476

### 2.3 topk回退统计

| 实验名称 | 回退轮次数 | 回退补齐总数 | 回退比例 |
|---------|-----------|------------|---------|
| full_model | 0 | 0 | 0.00% |
| baseline_entropy | 0 | 0 | 0.00% |
| baseline_random | 0 | 0 | 0.00% |
| no_agent | 0 | 0 | 0.00% |
| fixed_lambda | 0 | 0 | 0.00% |

---

*报告生成于 2026-02-08 19:35:08*
