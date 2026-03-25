# 详细实验结果报告

**生成时间**: 2026-03-26 03:36:35

---

## 实验详细结果

### full_model_A_lambda_policy

**描述**: 对照A / AAL-SD (Full)：基于 auto_opt_iter15_cand2_02（warmup+risk闭环 + 后期ramp + U-guardrail；train_holdout grad_probe）

**状态**: failed

**错误**: LLM Agent failed at Round 1: LLMAPIError: Error calling API: HTTPSConnectionPool(host='api.siliconflow.cn', port=443): Max retries exceeded with url: /v1/chat/completions (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1032)')))

---

### full_model_B_lambda_agent

**描述**: 对照B / AAL-SD (Agent-λ)：与 full_model_A_lambda_policy 完全一致，唯一区别：λ 由 LLM Agent 通过 set_lambda 显式决定，而非 policy 自动填充

**状态**: failed

**错误**: LLM Agent failed at Round 1: LLMAPIError: Error calling API: HTTPSConnectionPool(host='api.siliconflow.cn', port=443): Max retries exceeded with url: /v1/chat/completions (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1032)')))

---

### fixed_lambda

**描述**: 消融：Fixed λ=0.5；保留Agent，隔离三阶段λ policy贡献

**状态**: failed

**错误**: LLM Agent failed at Round 1: LLMAPIError: Error calling API: HTTPSConnectionPool(host='api.siliconflow.cn', port=443): Max retries exceeded with url: /v1/chat/completions (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1032)')))

---

### no_agent

**描述**: 消融：w/o Agent（argmax）；保留三阶段λ policy，与full_model一致

**状态**: success

**ALC**: 0.5939

**最终 mIoU**: 0.6685

**最终 F1-Score**: 0.7571

#### 性能历史

| 轮次 | mIoU | F1-Score | 标注样本数 |
|------|------|----------|-----------|
| 1 | 0.5532 | 0.6050 | 189 |
| 2 | 0.5784 | 0.6429 | 277 |
| 3 | 0.6446 | 0.7301 | 365 |
| 4 | 0.6857 | 0.7743 | 453 |
| 5 | 0.6464 | 0.7313 | 541 |
| 6 | 0.7104 | 0.7993 | 629 |
| 7 | 0.7125 | 0.8011 | 717 |
| 8 | 0.6685 | 0.7560 | 805 |
| 9 | 0.6769 | 0.7656 | 893 |
| 10 | 0.6965 | 0.7858 | 981 |
| 11 | 0.7082 | 0.7970 | 1069 |
| 12 | 0.6952 | 0.7842 | 1157 |
| 13 | 0.7195 | 0.8078 | 1245 |
| 14 | 0.7028 | 0.7918 | 1333 |
| 15 | 0.6999 | 0.7889 | 1421 |
| 16 | 0.6999 | 0.7889 | 1509 |

---

### uncertainty_only

**描述**: 固定λ=0，仅使用不确定性

**状态**: success

**ALC**: 0.6012

**最终 mIoU**: 0.7165

**最终 F1-Score**: 0.8054

#### 性能历史

| 轮次 | mIoU | F1-Score | 标注样本数 |
|------|------|----------|-----------|
| 1 | 0.5497 | 0.5995 | 189 |
| 2 | 0.6676 | 0.7548 | 277 |
| 3 | 0.6421 | 0.7261 | 365 |
| 4 | 0.6341 | 0.7174 | 453 |
| 5 | 0.6865 | 0.7752 | 541 |
| 6 | 0.6894 | 0.7784 | 629 |
| 7 | 0.6875 | 0.7766 | 717 |
| 8 | 0.7176 | 0.8060 | 805 |
| 9 | 0.7147 | 0.8033 | 893 |
| 10 | 0.6829 | 0.7714 | 981 |
| 11 | 0.6958 | 0.7851 | 1069 |
| 12 | 0.7110 | 0.7995 | 1157 |
| 13 | 0.7025 | 0.7911 | 1245 |
| 14 | 0.7054 | 0.7944 | 1333 |
| 15 | 0.7218 | 0.8100 | 1421 |
| 16 | 0.7218 | 0.8100 | 1509 |

---

### knowledge_only

**描述**: 固定λ=1，仅使用知识增益

**状态**: success

**ALC**: 0.5981

**最终 mIoU**: 0.7063

**最终 F1-Score**: 0.7956

#### 性能历史

| 轮次 | mIoU | F1-Score | 标注样本数 |
|------|------|----------|-----------|
| 1 | 0.5316 | 0.5714 | 189 |
| 2 | 0.5868 | 0.6548 | 277 |
| 3 | 0.6802 | 0.7686 | 365 |
| 4 | 0.6694 | 0.7567 | 453 |
| 5 | 0.6638 | 0.7512 | 541 |
| 6 | 0.6990 | 0.7880 | 629 |
| 7 | 0.7014 | 0.7902 | 717 |
| 8 | 0.6524 | 0.7380 | 805 |
| 9 | 0.6993 | 0.7879 | 893 |
| 10 | 0.7258 | 0.8138 | 981 |
| 11 | 0.6863 | 0.7748 | 1069 |
| 12 | 0.7181 | 0.8066 | 1157 |
| 13 | 0.7322 | 0.8193 | 1245 |
| 14 | 0.7046 | 0.7933 | 1333 |
| 15 | 0.7054 | 0.7944 | 1421 |
| 16 | 0.7054 | 0.7944 | 1509 |

---


*报告生成于 2026-03-26 03:36:35*
