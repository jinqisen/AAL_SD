# 实验日志

实验名称: fixed_k
描述: 消融：固定K阶段（保持coreset-to-labeled），仅保留λ闭环
开始时间: 2026-02-24T09:20:53.632334

## Round 1

Labeled Pool Size: 1383

- Epoch 1: Loss=0.2250, mIoU=0.5104, F1=0.5319
- Epoch 2: Loss=0.0996, mIoU=0.6614, F1=0.7489
- Epoch 3: Loss=0.0879, mIoU=0.6116, F1=0.6880
- Epoch 4: Loss=0.0826, mIoU=0.6843, F1=0.7727
- Epoch 5: Loss=0.0765, mIoU=0.6801, F1=0.7685
- Epoch 6: Loss=0.0728, mIoU=0.5909, F1=0.6608
- Epoch 7: Loss=0.0723, mIoU=0.7166, F1=0.8051
- Epoch 8: Loss=0.0687, mIoU=0.7009, F1=0.7896
- Epoch 9: Loss=0.0672, mIoU=0.6622, F1=0.7490
- Epoch 10: Loss=0.0660, mIoU=0.6446, F1=0.7294

当前轮次最佳结果: Round=1, Labeled=1383, mIoU=0.7166, F1=0.8051

## Round 2

Labeled Pool Size: 1471

- Epoch 1: Loss=0.2055, mIoU=0.5920, F1=0.6619
- Epoch 2: Loss=0.0937, mIoU=0.5945, F1=0.6652
- Epoch 3: Loss=0.0826, mIoU=0.6958, F1=0.7849
- Epoch 4: Loss=0.0773, mIoU=0.6735, F1=0.7612
- Epoch 5: Loss=0.0744, mIoU=0.5729, F1=0.6347
- Epoch 6: Loss=0.0707, mIoU=0.7087, F1=0.7975
- Epoch 7: Loss=0.0675, mIoU=0.7087, F1=0.7977
- Epoch 8: Loss=0.0646, mIoU=0.7056, F1=0.7944
- Epoch 9: Loss=0.0633, mIoU=0.6759, F1=0.7640
- Epoch 10: Loss=0.0618, mIoU=0.6751, F1=0.7631

当前轮次最佳结果: Round=2, Labeled=1471, mIoU=0.7087, F1=0.7977


**[ERROR] Round 2 失败: LLM Agent failed at Round 2: Only 48 valid selections, but 88 required**


--- [Checkpoint] 续跑开始时间: 2026-02-27T00:44:01.468254 ---

## Round 2

Labeled Pool Size: 1383

- Epoch 1: Loss=0.2145, mIoU=0.5594, F1=0.6145
- Epoch 2: Loss=0.0983, mIoU=0.5734, F1=0.6355
- Epoch 3: Loss=0.0872, mIoU=0.6440, F1=0.7281
- Epoch 4: Loss=0.0799, mIoU=0.6648, F1=0.7516
- Epoch 5: Loss=0.0756, mIoU=0.6904, F1=0.7792
- Epoch 6: Loss=0.0740, mIoU=0.6902, F1=0.7809
- Epoch 7: Loss=0.0702, mIoU=0.6356, F1=0.7182
- Epoch 8: Loss=0.0664, mIoU=0.6394, F1=0.7228
- Epoch 9: Loss=0.0647, mIoU=0.7014, F1=0.7902
- Epoch 10: Loss=0.0628, mIoU=0.6434, F1=0.7279

当前轮次最佳结果: Round=2, Labeled=1383, mIoU=0.7014, F1=0.7902

## Round 3

Labeled Pool Size: 1471

- Epoch 1: Loss=0.1616, mIoU=0.5592, F1=0.6143
- Epoch 2: Loss=0.0920, mIoU=0.5379, F1=0.5801
- Epoch 3: Loss=0.0809, mIoU=0.6934, F1=0.7825
- Epoch 4: Loss=0.0754, mIoU=0.6955, F1=0.7843
- Epoch 5: Loss=0.0702, mIoU=0.6095, F1=0.6860
- Epoch 6: Loss=0.0672, mIoU=0.6943, F1=0.7833
- Epoch 7: Loss=0.0633, mIoU=0.6151, F1=0.6930
- Epoch 8: Loss=0.0614, mIoU=0.6469, F1=0.7325
- Epoch 9: Loss=0.0590, mIoU=0.6940, F1=0.7830
- Epoch 10: Loss=0.0605, mIoU=0.6515, F1=0.7387

当前轮次最佳结果: Round=3, Labeled=1471, mIoU=0.6955, F1=0.7843


**[ERROR] Round 3 失败: LLM Agent failed at Round 3: Only 48 valid selections, but 88 required**

