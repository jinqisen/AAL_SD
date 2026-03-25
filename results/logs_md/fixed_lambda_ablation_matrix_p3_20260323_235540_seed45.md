# 实验日志

实验名称: fixed_lambda
描述: 消融：Fixed λ=0.5；保留Agent，隔离三阶段λ policy贡献
开始时间: 2026-03-25T22:47:41.223891

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.4692, mIoU=0.4951, F1=0.5273
- Epoch 2: Loss=0.2521, mIoU=0.5024, F1=0.5171
- Epoch 3: Loss=0.1600, mIoU=0.5226, F1=0.5543
- Epoch 4: Loss=0.1160, mIoU=0.5176, F1=0.5451
- Epoch 5: Loss=0.0916, mIoU=0.5216, F1=0.5522
- Epoch 6: Loss=0.0760, mIoU=0.5009, F1=0.5142
- Epoch 7: Loss=0.0662, mIoU=0.5014, F1=0.5150
- Epoch 8: Loss=0.0589, mIoU=0.4975, F1=0.5075
- Epoch 9: Loss=0.0550, mIoU=0.5348, F1=0.5751
- Epoch 10: Loss=0.0523, mIoU=0.5281, F1=0.5637

本轮结果: Round=1, Labeled=189, Selection=best_val (epoch=9), mIoU=0.5348, F1=0.5751, peak_mIoU=0.5348

Round=1, Labeled=189, mIoU=0.5348, F1=0.5751


**[ERROR] Round 1 失败: LLM Agent failed at Round 1: LLMAPIError: Error calling API: HTTPSConnectionPool(host='api.siliconflow.cn', port=443): Max retries exceeded with url: /v1/chat/completions (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1032)')))**

