# AAL-SD 实验结果摘要报告

**生成时间**: 2026-03-27 09:01:00

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
| old_controller_ab | A/B 对照A：旧 warmup_risk_closed_loop，保留训练侧信号主导逻辑，但补全 geometry logging | ✅ |
| geometry_controller_v0 | A/B 对照B：geometry controller v0（sens_up + asymmetry_ratio 主导，training signal 仅作 safety fallback） | ✅ |

---
## 2. 核心结果对比

### 2.1 性能指标汇总

| 实验名称 | ALC(val选模) | 最后选模 mIoU(val) | 最后选模 F1(val) | 最终报告 mIoU(test) | 最终报告 F1(test) |
|---------|------------|------------------|---------------|----------------------|---------------------|
| old_controller_ab | 0.6044 | 0.7161532915460483 | 0.8045366600641782 | 0.7057 | 0.7948 |
| geometry_controller_v0 | 0.5942 | 0.6941903714620599 | 0.7833134514115213 | 0.6759 | 0.7650 |

### 2.2 性能排名

**按ALC排名**:

1. old_controller_ab: 0.6044
2. geometry_controller_v0: 0.5942

**按最终报告mIoU(test)排名**:

1. old_controller_ab: 0.7057
2. geometry_controller_v0: 0.6759

### 2.3 topk回退统计

| 实验名称 | 回退轮次数 | 回退补齐总数 | 回退比例 |
|---------|-----------|------------|---------|
| old_controller_ab | 0 | 0 | 0.00% |
| geometry_controller_v0 | 0 | 0 | 0.00% |

---

*报告生成于 2026-03-27 09:01:00*
