# 详细实验结果报告

**生成时间**: 2026-03-07 21:05:40

---

## 实验详细结果

### baseline_random

**描述**: 随机采样基线

**状态**: failed

**错误**: Missing masks for split=train (example_ids=['image_1', 'image_10', 'image_100']) dir=/Users/anykong/AAL_SD/data/Landslide4Sense/TrainData/mask

---

### baseline_entropy

**描述**: 熵采样基线

**状态**: failed

**错误**: Missing masks for split=train (example_ids=['image_1', 'image_10', 'image_100']) dir=/Users/anykong/AAL_SD/data/Landslide4Sense/TrainData/mask

---

### full_model_A_lambda_policy

**描述**: 对照A / AAL-SD (Full)：LLM 参与决策与解释；λ由 warmup+风险闭环策略生成并在工具层 clamp（启用CI阈值+AND严重判定+EMA/冷却/限步长；默认禁止 set_lambda；不调整query_size/epochs）

**状态**: failed

**错误**: Missing masks for split=train (example_ids=['image_1', 'image_10', 'image_100']) dir=/Users/anykong/AAL_SD/data/Landslide4Sense/TrainData/mask

---

### baseline_dial_style

**描述**: DIAL-style基线：分簇多样性约束+不确定性

**状态**: failed

**错误**: Missing masks for split=train (example_ids=['image_1', 'image_10', 'image_100']) dir=/Users/anykong/AAL_SD/data/Landslide4Sense/TrainData/mask

---

### baseline_bald

**描述**: BALD采样基线

**状态**: failed

**错误**: Missing masks for split=train (example_ids=['image_1', 'image_10', 'image_100']) dir=/Users/anykong/AAL_SD/data/Landslide4Sense/TrainData/mask

---

### baseline_coreset

**描述**: Core-Set采样基线

**状态**: failed

**错误**: Missing masks for split=train (example_ids=['image_1', 'image_10', 'image_100']) dir=/Users/anykong/AAL_SD/data/Landslide4Sense/TrainData/mask

---

### baseline_wang_style

**描述**: Wang-style基线：两阶段U→K重排

**状态**: failed

**错误**: Missing masks for split=train (example_ids=['image_1', 'image_10', 'image_100']) dir=/Users/anykong/AAL_SD/data/Landslide4Sense/TrainData/mask

---


*报告生成于 2026-03-07 21:05:40*
