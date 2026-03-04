# AAL-SD 实验结果摘要报告

**生成时间**: 2026-03-01 23:34:32

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
| baseline_entropy | 熵采样基线 | ✅ |
| baseline_random | 随机采样基线 | ✅ |
| baseline_coreset | Core-Set采样基线 | ✅ |
| baseline_bald | BALD采样基线 | ✅ |
| baseline_dial_style | DIAL-style基线：分簇多样性约束+不确定性 | ✅ |
| baseline_wang_style | Wang-style基线：两阶段U→K重排 | ✅ |

---
## 2. 核心结果对比

### 2.1 性能指标汇总

| 实验名称 | ALC | 最终 mIoU | 最终 F1-Score |
|---------|-----|----------|--------------|
| baseline_entropy | 0.6666 | 0.7551 | 0.8408 |
| baseline_random | 0.6595 | 0.7583 | 0.8433 |
| baseline_coreset | 0.6538 | 0.7558 | 0.8412 |
| baseline_bald | 0.6680 | 0.7558 | 0.8412 |
| baseline_dial_style | 0.6664 | 0.7622 | 0.8467 |
| baseline_wang_style | 0.6715 | 0.7604 | 0.8449 |

### 2.2 性能排名

**按ALC排名**:

1. baseline_wang_style: 0.6715
2. baseline_bald: 0.6680
3. baseline_entropy: 0.6666
4. baseline_dial_style: 0.6664
5. baseline_random: 0.6595
6. baseline_coreset: 0.6538

**按最终mIoU排名**:

1. baseline_dial_style: 0.7622
2. baseline_wang_style: 0.7604
3. baseline_random: 0.7583
4. baseline_coreset: 0.7558
5. baseline_bald: 0.7558
6. baseline_entropy: 0.7551

### 2.3 topk回退统计

| 实验名称 | 回退轮次数 | 回退补齐总数 | 回退比例 |
|---------|-----------|------------|---------|
| baseline_entropy | 0 | 0 | 0.00% |
| baseline_random | 0 | 0 | 0.00% |
| baseline_coreset | 0 | 0 | 0.00% |
| baseline_bald | 0 | 0 | 0.00% |
| baseline_dial_style | 0 | 0 | 0.00% |
| baseline_wang_style | 0 | 0 | 0.00% |

---

*报告生成于 2026-03-01 23:34:32*
