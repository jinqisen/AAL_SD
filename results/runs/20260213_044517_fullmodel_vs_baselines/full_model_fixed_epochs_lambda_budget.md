# 实验日志

实验名称: full_model_fixed_epochs_lambda_budget
描述: 消融（固定epochs=10）：仅允许控制λ与query_size
开始时间: 2026-02-14T04:33:38.898521

## Round 1

Labeled Pool Size: 151

- Epoch 1: Loss=0.6107, mIoU=0.4692, F1=0.5092
- Epoch 2: Loss=0.3438, mIoU=0.5017, F1=0.5230
- Epoch 3: Loss=0.2137, mIoU=0.4992, F1=0.5164
- Epoch 4: Loss=0.1497, mIoU=0.5116, F1=0.5396
- Epoch 5: Loss=0.1196, mIoU=0.5078, F1=0.5324
- Epoch 6: Loss=0.0951, mIoU=0.5833, F1=0.6539
- Epoch 7: Loss=0.0817, mIoU=0.5983, F1=0.6750
- Epoch 8: Loss=0.0748, mIoU=0.5522, F1=0.6068
- Epoch 9: Loss=0.0643, mIoU=0.5875, F1=0.6587
- Epoch 10: Loss=0.0604, mIoU=0.6090, F1=0.6872

当前轮次最佳结果: Round=1, Labeled=151, mIoU=0.6090, F1=0.6872


**[ERROR] Round 1 失败: LLM Agent failed at Round 1: LLMAPIError: Error calling API: HTTPSConnectionPool(host='api.siliconflow.cn', port=443): Max retries exceeded with url: /v1/chat/completions (Caused by ProxyError('Unable to connect to proxy', NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x301386fd0>: Failed to establish a new connection: [Errno 61] Connection refused')))**

