# 实验日志

实验名称: no_cold_start
描述: 消融：去除冷启动与warmup，直接进入风险闭环
开始时间: 2026-02-27T02:42:42.621034

## Round 1

Labeled Pool Size: 1383

- Epoch 1: Loss=0.2186, mIoU=0.5778, F1=0.6419
- Epoch 2: Loss=0.1000, mIoU=0.6070, F1=0.6819
- Epoch 3: Loss=0.0872, mIoU=0.6235, F1=0.7032
- Epoch 4: Loss=0.0798, mIoU=0.6657, F1=0.7539
- Epoch 5: Loss=0.0762, mIoU=0.6734, F1=0.7613
- Epoch 6: Loss=0.0738, mIoU=0.6106, F1=0.6951
- Epoch 7: Loss=0.0727, mIoU=0.6638, F1=0.7509
- Epoch 8: Loss=0.0684, mIoU=0.6998, F1=0.7887
- Epoch 9: Loss=0.0665, mIoU=0.6788, F1=0.7689
- Epoch 10: Loss=0.0654, mIoU=0.6956, F1=0.7852

当前轮次最佳结果: Round=1, Labeled=1383, mIoU=0.6998, F1=0.7887

## Round 2

Labeled Pool Size: 1471

- Epoch 1: Loss=0.2022, mIoU=0.5167, F1=0.5434
- Epoch 2: Loss=0.0928, mIoU=0.6653, F1=0.7533
- Epoch 3: Loss=0.0813, mIoU=0.6003, F1=0.6732
- Epoch 4: Loss=0.0763, mIoU=0.6748, F1=0.7625
- Epoch 5: Loss=0.0715, mIoU=0.7024, F1=0.7913
- Epoch 6: Loss=0.0678, mIoU=0.6785, F1=0.7665
- Epoch 7: Loss=0.0672, mIoU=0.6821, F1=0.7725
- Epoch 8: Loss=0.0635, mIoU=0.6762, F1=0.7650
- Epoch 9: Loss=0.0603, mIoU=0.6865, F1=0.7753
- Epoch 10: Loss=0.0584, mIoU=0.6600, F1=0.7469

当前轮次最佳结果: Round=2, Labeled=1471, mIoU=0.7024, F1=0.7913


**[ERROR] Round 2 失败: LLM Agent failed at Round 2: Only 48 valid selections, but 88 required**

