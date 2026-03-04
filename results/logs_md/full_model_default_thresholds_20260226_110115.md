# 实验日志

实验名称: full_model_default_thresholds
描述: 完整模型（旧阈值）：固定Warmup(0.2)+默认过拟合阈值(0.8/0.5) + 自适应阈值 rollback + 风险驱动 λ 闭环控制（不调整 query_size）
开始时间: 2026-02-26T20:45:30.608681

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.5726, mIoU=0.4983, F1=0.5419
- Epoch 2: Loss=0.2957, mIoU=0.5556, F1=0.6142
- Epoch 3: Loss=0.1794, mIoU=0.5244, F1=0.5572
- Epoch 4: Loss=0.1285, mIoU=0.5431, F1=0.5886
- Epoch 5: Loss=0.1045, mIoU=0.5455, F1=0.5926
- Epoch 6: Loss=0.0886, mIoU=0.5331, F1=0.5722
- Epoch 7: Loss=0.0794, mIoU=0.5337, F1=0.5730
- Epoch 8: Loss=0.0732, mIoU=0.5242, F1=0.5566
- Epoch 9: Loss=0.0661, mIoU=0.5133, F1=0.5372
- Epoch 10: Loss=0.0632, mIoU=0.5541, F1=0.6062

当前轮次最佳结果: Round=1, Labeled=189, mIoU=0.5556, F1=0.6142

## Round 2

Labeled Pool Size: 277

- Epoch 1: Loss=0.4652, mIoU=0.5008, F1=0.5383
- Epoch 2: Loss=0.2248, mIoU=0.5100, F1=0.5424
- Epoch 3: Loss=0.1523, mIoU=0.5471, F1=0.5950
- Epoch 4: Loss=0.1281, mIoU=0.5210, F1=0.5511
- Epoch 5: Loss=0.1083, mIoU=0.5813, F1=0.6470
- Epoch 6: Loss=0.0975, mIoU=0.5142, F1=0.5389
- Epoch 7: Loss=0.0950, mIoU=0.5792, F1=0.6442
- Epoch 8: Loss=0.0830, mIoU=0.5749, F1=0.6375
- Epoch 9: Loss=0.0810, mIoU=0.5673, F1=0.6265
- Epoch 10: Loss=0.0803, mIoU=0.5977, F1=0.6701

当前轮次最佳结果: Round=2, Labeled=277, mIoU=0.5977, F1=0.6701

## Round 3

Labeled Pool Size: 365

- Epoch 1: Loss=0.3063, mIoU=0.5222, F1=0.5530
- Epoch 2: Loss=0.1719, mIoU=0.5005, F1=0.5133
- Epoch 3: Loss=0.1384, mIoU=0.5016, F1=0.5154
- Epoch 4: Loss=0.1281, mIoU=0.5008, F1=0.5139
- Epoch 5: Loss=0.1197, mIoU=0.5135, F1=0.5376
- Epoch 6: Loss=0.1117, mIoU=0.5675, F1=0.6273
- Epoch 7: Loss=0.1121, mIoU=0.5357, F1=0.5763
- Epoch 8: Loss=0.1027, mIoU=0.5350, F1=0.5755
- Epoch 9: Loss=0.0982, mIoU=0.5892, F1=0.6580
- Epoch 10: Loss=0.0937, mIoU=0.6471, F1=0.7324

当前轮次最佳结果: Round=3, Labeled=365, mIoU=0.6471, F1=0.7324

## Round 4

Labeled Pool Size: 453

- Epoch 1: Loss=0.3707, mIoU=0.5426, F1=0.5941
- Epoch 2: Loss=0.1891, mIoU=0.5070, F1=0.5270
- Epoch 3: Loss=0.1542, mIoU=0.5082, F1=0.5278
- Epoch 4: Loss=0.1355, mIoU=0.6092, F1=0.6854
- Epoch 5: Loss=0.1289, mIoU=0.5520, F1=0.6030
- Epoch 6: Loss=0.1201, mIoU=0.6332, F1=0.7153
- Epoch 7: Loss=0.1131, mIoU=0.6074, F1=0.6827
- Epoch 8: Loss=0.1099, mIoU=0.6621, F1=0.7499
- Epoch 9: Loss=0.1073, mIoU=0.5787, F1=0.6431
- Epoch 10: Loss=0.1062, mIoU=0.6004, F1=0.6735

当前轮次最佳结果: Round=4, Labeled=453, mIoU=0.6621, F1=0.7499

## Round 5

Labeled Pool Size: 541

- Epoch 1: Loss=0.3317, mIoU=0.5716, F1=0.6327
- Epoch 2: Loss=0.1731, mIoU=0.5747, F1=0.6373
- Epoch 3: Loss=0.1444, mIoU=0.5442, F1=0.5904
- Epoch 4: Loss=0.1326, mIoU=0.6295, F1=0.7116
- Epoch 5: Loss=0.1321, mIoU=0.6744, F1=0.7625
- Epoch 6: Loss=0.1209, mIoU=0.5989, F1=0.6715
- Epoch 7: Loss=0.1150, mIoU=0.6700, F1=0.7581
- Epoch 8: Loss=0.1087, mIoU=0.6780, F1=0.7661
- Epoch 9: Loss=0.1065, mIoU=0.5906, F1=0.6605
- Epoch 10: Loss=0.1029, mIoU=0.6014, F1=0.6750

当前轮次最佳结果: Round=5, Labeled=541, mIoU=0.6780, F1=0.7661

## Round 6

Labeled Pool Size: 629

- Epoch 1: Loss=0.2992, mIoU=0.5582, F1=0.6130
- Epoch 2: Loss=0.1611, mIoU=0.5418, F1=0.5864
- Epoch 3: Loss=0.1368, mIoU=0.6868, F1=0.7755
- Epoch 4: Loss=0.1255, mIoU=0.6741, F1=0.7620
- Epoch 5: Loss=0.1205, mIoU=0.7128, F1=0.8011
- Epoch 6: Loss=0.1133, mIoU=0.6380, F1=0.7213
- Epoch 7: Loss=0.1115, mIoU=0.6685, F1=0.7558
- Epoch 8: Loss=0.1043, mIoU=0.6900, F1=0.7785
- Epoch 9: Loss=0.1006, mIoU=0.6582, F1=0.7446
- Epoch 10: Loss=0.0976, mIoU=0.6954, F1=0.7842

当前轮次最佳结果: Round=6, Labeled=629, mIoU=0.7128, F1=0.8011

## Round 7

Labeled Pool Size: 717

- Epoch 1: Loss=0.3457, mIoU=0.5551, F1=0.6077
- Epoch 2: Loss=0.1592, mIoU=0.6026, F1=0.6763
- Epoch 3: Loss=0.1356, mIoU=0.6187, F1=0.6974
- Epoch 4: Loss=0.1228, mIoU=0.5476, F1=0.5959
- Epoch 5: Loss=0.1180, mIoU=0.6780, F1=0.7686
- Epoch 6: Loss=0.1123, mIoU=0.6159, F1=0.6938
- Epoch 7: Loss=0.1095, mIoU=0.6502, F1=0.7355
- Epoch 8: Loss=0.1047, mIoU=0.6339, F1=0.7165
- Epoch 9: Loss=0.1016, mIoU=0.6293, F1=0.7108
- Epoch 10: Loss=0.0977, mIoU=0.6353, F1=0.7180

当前轮次最佳结果: Round=7, Labeled=717, mIoU=0.6780, F1=0.7686

## Round 8

Labeled Pool Size: 805

- Epoch 1: Loss=0.2926, mIoU=0.5372, F1=0.5791
- Epoch 2: Loss=0.1476, mIoU=0.6033, F1=0.6775
- Epoch 3: Loss=0.1298, mIoU=0.6527, F1=0.7390
- Epoch 4: Loss=0.1180, mIoU=0.6170, F1=0.6954
- Epoch 5: Loss=0.1105, mIoU=0.5338, F1=0.5733
- Epoch 6: Loss=0.1074, mIoU=0.6524, F1=0.7379
- Epoch 7: Loss=0.1009, mIoU=0.6418, F1=0.7260
- Epoch 8: Loss=0.0974, mIoU=0.6394, F1=0.7232
- Epoch 9: Loss=0.0934, mIoU=0.6266, F1=0.7076
- Epoch 10: Loss=0.0922, mIoU=0.6134, F1=0.6909

当前轮次最佳结果: Round=8, Labeled=805, mIoU=0.6527, F1=0.7390

## Round 9

Labeled Pool Size: 893

- Epoch 1: Loss=0.3369, mIoU=0.5277, F1=0.5627
- Epoch 2: Loss=0.1459, mIoU=0.5667, F1=0.6255
- Epoch 3: Loss=0.1262, mIoU=0.5148, F1=0.5399
- Epoch 4: Loss=0.1143, mIoU=0.5854, F1=0.6528
- Epoch 5: Loss=0.1111, mIoU=0.7035, F1=0.7923
- Epoch 6: Loss=0.1070, mIoU=0.6217, F1=0.7017
- Epoch 7: Loss=0.1024, mIoU=0.6625, F1=0.7495
- Epoch 8: Loss=0.0990, mIoU=0.7037, F1=0.7925
- Epoch 9: Loss=0.0952, mIoU=0.6719, F1=0.7605
