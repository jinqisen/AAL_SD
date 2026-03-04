# AAL-SD 实验结果摘要报告

**生成时间**: 2026-03-01 02:17:24

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
| baseline_entropy | 0.6674 | 0.7555 | 0.8409 |
| baseline_random | 0.6563 | 0.7391 | 0.8268 |
| baseline_coreset | 0.6578 | 0.7507 | 0.8370 |
| baseline_bald | 0.6673 | 0.7644 | 0.8483 |
| baseline_dial_style | 0.6681 | 0.7626 | 0.8469 |
| baseline_wang_style | 0.6714 | 0.7560 | 0.8412 |

### 2.2 性能排名

**按ALC排名**:

1. baseline_wang_style: 0.6714
2. baseline_dial_style: 0.6681
3. baseline_entropy: 0.6674
4. baseline_bald: 0.6673
5. baseline_coreset: 0.6578
6. baseline_random: 0.6563

**按最终mIoU排名**:

1. baseline_bald: 0.7644
2. baseline_dial_style: 0.7626
3. baseline_wang_style: 0.7560
4. baseline_entropy: 0.7555
5. baseline_coreset: 0.7507
6. baseline_random: 0.7391

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

*报告生成于 2026-03-01 02:17:24*
