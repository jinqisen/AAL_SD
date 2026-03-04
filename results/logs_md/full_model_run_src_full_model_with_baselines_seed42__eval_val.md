# 实验日志

实验名称: full_model
描述: 完整模型（最佳λ日程）：R1-2 λ=0.0；R3 warmup=0.2；R4+ 风险闭环λ（不调整query_size）
开始时间: 2026-02-24T03:10:58.055322

## Round 1

Labeled Pool Size: 1383

- Epoch 1: Loss=0.2297, mIoU=0.5967, F1=0.6685
- Epoch 2: Loss=0.0993, mIoU=0.6099, F1=0.6858
- Epoch 3: Loss=0.0870, mIoU=0.5736, F1=0.6359
- Epoch 4: Loss=0.0811, mIoU=0.6773, F1=0.7656
- Epoch 5: Loss=0.0778, mIoU=0.6507, F1=0.7360
- Epoch 6: Loss=0.0729, mIoU=0.7085, F1=0.7976
- Epoch 7: Loss=0.0690, mIoU=0.6619, F1=0.7496
- Epoch 8: Loss=0.0685, mIoU=0.6619, F1=0.7490
- Epoch 9: Loss=0.0665, mIoU=0.5936, F1=0.6707
- Epoch 10: Loss=0.0663, mIoU=0.6950, F1=0.7847

当前轮次最佳结果: Round=1, Labeled=1383, mIoU=0.7085, F1=0.7976

## Round 2

Labeled Pool Size: 1471

- Epoch 1: Loss=0.2114, mIoU=0.6257, F1=0.7063
- Epoch 2: Loss=0.0946, mIoU=0.6778, F1=0.7667
- Epoch 3: Loss=0.0825, mIoU=0.6723, F1=0.7605
- Epoch 4: Loss=0.0767, mIoU=0.6901, F1=0.7787
- Epoch 5: Loss=0.0704, mIoU=0.7074, F1=0.7959
- Epoch 6: Loss=0.0673, mIoU=0.6506, F1=0.7358
- Epoch 7: Loss=0.0641, mIoU=0.7246, F1=0.8128
- Epoch 8: Loss=0.0622, mIoU=0.6999, F1=0.7894
- Epoch 9: Loss=0.0590, mIoU=0.6584, F1=0.7448
- Epoch 10: Loss=0.0604, mIoU=0.7085, F1=0.7972

当前轮次最佳结果: Round=2, Labeled=1471, mIoU=0.7246, F1=0.8128


**[ERROR] Round 2 失败: LLM Agent failed at Round 2: Only 48 valid selections, but 88 required**


--- [Checkpoint] 续跑开始时间: 2026-02-26T22:19:09.753964 ---

## Round 2

Labeled Pool Size: 1383

- Epoch 1: Loss=0.2111, mIoU=0.5424, F1=0.5874
- Epoch 2: Loss=0.0969, mIoU=0.5537, F1=0.6221
- Epoch 3: Loss=0.0859, mIoU=0.5612, F1=0.6172
- Epoch 4: Loss=0.0805, mIoU=0.7146, F1=0.8030
- Epoch 5: Loss=0.0757, mIoU=0.6976, F1=0.7864
- Epoch 6: Loss=0.0722, mIoU=0.6662, F1=0.7538
- Epoch 7: Loss=0.0710, mIoU=0.6184, F1=0.6973
- Epoch 8: Loss=0.0670, mIoU=0.7001, F1=0.7888
- Epoch 9: Loss=0.0663, mIoU=0.6571, F1=0.7462
- Epoch 10: Loss=0.0634, mIoU=0.7208, F1=0.8094

当前轮次最佳结果: Round=2, Labeled=1383, mIoU=0.7208, F1=0.8094

## Round 3

Labeled Pool Size: 1471

- Epoch 1: Loss=0.1609, mIoU=0.5036, F1=0.5192
- Epoch 2: Loss=0.0934, mIoU=0.6401, F1=0.7242
- Epoch 3: Loss=0.0819, mIoU=0.5817, F1=0.6475
- Epoch 4: Loss=0.0748, mIoU=0.6764, F1=0.7646
- Epoch 5: Loss=0.0701, mIoU=0.6376, F1=0.7207
- Epoch 6: Loss=0.0670, mIoU=0.7029, F1=0.7919
- Epoch 7: Loss=0.0637, mIoU=0.6209, F1=0.7014
- Epoch 8: Loss=0.0608, mIoU=0.6352, F1=0.7245
- Epoch 9: Loss=0.0592, mIoU=0.6564, F1=0.7435
- Epoch 10: Loss=0.0590, mIoU=0.7097, F1=0.7985

当前轮次最佳结果: Round=3, Labeled=1471, mIoU=0.7097, F1=0.7985


**[ERROR] Round 3 失败: LLM Agent failed at Round 3: Only 48 valid selections, but 88 required**

