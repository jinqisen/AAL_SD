# 实验日志

实验名称: full_model_B_lambda_agent
描述: 对照B / AAL-SD (Agent-λ)：与 full_model_A_lambda_policy 完全一致，唯一区别：λ 由 LLM Agent 通过 set_lambda 显式决定，而非 policy 自动填充
开始时间: 2026-03-25T21:48:52.612852

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.4695, mIoU=0.4929, F1=0.5173
- Epoch 2: Loss=0.2509, mIoU=0.4989, F1=0.5112
- Epoch 3: Loss=0.1590, mIoU=0.5188, F1=0.5495
- Epoch 4: Loss=0.1139, mIoU=0.5027, F1=0.5182
- Epoch 5: Loss=0.0908, mIoU=0.5197, F1=0.5506
- Epoch 6: Loss=0.0772, mIoU=0.5013, F1=0.5150
- Epoch 7: Loss=0.0680, mIoU=0.5065, F1=0.5249
- Epoch 8: Loss=0.0599, mIoU=0.5329, F1=0.5719
- Epoch 9: Loss=0.0547, mIoU=0.6086, F1=0.6845
- Epoch 10: Loss=0.0531, mIoU=0.5621, F1=0.6188

本轮结果: Round=1, Labeled=189, Selection=best_val (epoch=9), mIoU=0.6086, F1=0.6845, peak_mIoU=0.6086

Round=1, Labeled=189, mIoU=0.6086, F1=0.6845


**[ERROR] Round 1 失败: LLM Agent failed at Round 1: LLMAPIError: Error calling API: HTTPSConnectionPool(host='api.siliconflow.cn', port=443): Max retries exceeded with url: /v1/chat/completions (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1032)')))**

