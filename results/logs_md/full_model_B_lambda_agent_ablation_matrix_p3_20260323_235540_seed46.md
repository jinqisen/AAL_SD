# 实验日志

实验名称: full_model_B_lambda_agent
描述: 对照B / AAL-SD (Agent-λ)：与 full_model_A_lambda_policy 完全一致，唯一区别：λ 由 LLM Agent 通过 set_lambda 显式决定，而非 policy 自动填充
开始时间: 2026-03-26T03:36:39.948571

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.4583, mIoU=0.4964, F1=0.5077
- Epoch 2: Loss=0.2551, mIoU=0.4921, F1=0.4971
- Epoch 3: Loss=0.1697, mIoU=0.5018, F1=0.5160
- Epoch 4: Loss=0.1252, mIoU=0.5110, F1=0.5332
- Epoch 5: Loss=0.1024, mIoU=0.5195, F1=0.5485
- Epoch 6: Loss=0.0863, mIoU=0.5177, F1=0.5451
- Epoch 7: Loss=0.0746, mIoU=0.5134, F1=0.5374
- Epoch 8: Loss=0.0664, mIoU=0.5081, F1=0.5277
- Epoch 9: Loss=0.0650, mIoU=0.5251, F1=0.5582
- Epoch 10: Loss=0.0598, mIoU=0.5212, F1=0.5514

本轮结果: Round=1, Labeled=189, Selection=best_val (epoch=9), mIoU=0.5251, F1=0.5582, peak_mIoU=0.5251

Round=1, Labeled=189, mIoU=0.5251, F1=0.5582

## Round 2

Labeled Pool Size: 277

- Epoch 1: Loss=0.5412, mIoU=0.5140, F1=0.5614
- Epoch 2: Loss=0.2674, mIoU=0.5218, F1=0.5526
- Epoch 3: Loss=0.1853, mIoU=0.5151, F1=0.5405
- Epoch 4: Loss=0.1498, mIoU=0.5089, F1=0.5292
- Epoch 5: Loss=0.1297, mIoU=0.5410, F1=0.5852
- Epoch 6: Loss=0.1214, mIoU=0.5465, F1=0.5943
- Epoch 7: Loss=0.1096, mIoU=0.5022, F1=0.5166
- Epoch 8: Loss=0.1038, mIoU=0.5403, F1=0.5840
- Epoch 9: Loss=0.1031, mIoU=0.5837, F1=0.6505
- Epoch 10: Loss=0.0954, mIoU=0.5038, F1=0.5196

本轮结果: Round=2, Labeled=277, Selection=best_val (epoch=9), mIoU=0.5837, F1=0.6505, peak_mIoU=0.5837

Round=2, Labeled=277, mIoU=0.5837, F1=0.6505

## Round 3

Labeled Pool Size: 365

- Epoch 1: Loss=0.4442, mIoU=0.5031, F1=0.5209
- Epoch 2: Loss=0.2048, mIoU=0.5230, F1=0.5550
- Epoch 3: Loss=0.1507, mIoU=0.5006, F1=0.5135
- Epoch 4: Loss=0.1288, mIoU=0.4987, F1=0.5101
- Epoch 5: Loss=0.1114, mIoU=0.5105, F1=0.5322
- Epoch 6: Loss=0.1056, mIoU=0.5632, F1=0.6201
- Epoch 7: Loss=0.0988, mIoU=0.6264, F1=0.7087
- Epoch 8: Loss=0.0935, mIoU=0.5370, F1=0.5786
- Epoch 9: Loss=0.0871, mIoU=0.5753, F1=0.6382
- Epoch 10: Loss=0.0833, mIoU=0.6742, F1=0.7625

本轮结果: Round=3, Labeled=365, Selection=best_val (epoch=10), mIoU=0.6742, F1=0.7625, peak_mIoU=0.6742

Round=3, Labeled=365, mIoU=0.6742, F1=0.7625

## Round 4

Labeled Pool Size: 453

- Epoch 1: Loss=0.4884, mIoU=0.5122, F1=0.5393
- Epoch 2: Loss=0.1954, mIoU=0.4956, F1=0.5040
- Epoch 3: Loss=0.1561, mIoU=0.5126, F1=0.5359
- Epoch 4: Loss=0.1374, mIoU=0.5113, F1=0.5336
- Epoch 5: Loss=0.1236, mIoU=0.5500, F1=0.5997
- Epoch 6: Loss=0.1233, mIoU=0.5197, F1=0.5487
- Epoch 7: Loss=0.1151, mIoU=0.5972, F1=0.6694
- Epoch 8: Loss=0.1116, mIoU=0.5705, F1=0.6311
- Epoch 9: Loss=0.1173, mIoU=0.5416, F1=0.5861
- Epoch 10: Loss=0.1085, mIoU=0.6665, F1=0.7537

本轮结果: Round=4, Labeled=453, Selection=best_val (epoch=10), mIoU=0.6665, F1=0.7537, peak_mIoU=0.6665

Round=4, Labeled=453, mIoU=0.6665, F1=0.7537

## Round 5

Labeled Pool Size: 541

- Epoch 1: Loss=0.4311, mIoU=0.5521, F1=0.6029
- Epoch 2: Loss=0.1819, mIoU=0.5290, F1=0.5650
- Epoch 3: Loss=0.1517, mIoU=0.6096, F1=0.6861
- Epoch 4: Loss=0.1356, mIoU=0.6339, F1=0.7161
- Epoch 5: Loss=0.1270, mIoU=0.6575, F1=0.7450
- Epoch 6: Loss=0.1211, mIoU=0.5780, F1=0.6422
- Epoch 7: Loss=0.1167, mIoU=0.5506, F1=0.6005
- Epoch 8: Loss=0.1115, mIoU=0.6381, F1=0.7222
- Epoch 9: Loss=0.1084, mIoU=0.6792, F1=0.7676
- Epoch 10: Loss=0.1006, mIoU=0.6715, F1=0.7593

本轮结果: Round=5, Labeled=541, Selection=best_val (epoch=9), mIoU=0.6792, F1=0.7676, peak_mIoU=0.6792

Round=5, Labeled=541, mIoU=0.6792, F1=0.7676

## Round 6

Labeled Pool Size: 629

- Epoch 1: Loss=0.3025, mIoU=0.5102, F1=0.5320
- Epoch 2: Loss=0.1630, mIoU=0.6166, F1=0.6945
- Epoch 3: Loss=0.1402, mIoU=0.5497, F1=0.5991
- Epoch 4: Loss=0.1257, mIoU=0.6229, F1=0.7026
- Epoch 5: Loss=0.1189, mIoU=0.6191, F1=0.6983
- Epoch 6: Loss=0.1133, mIoU=0.5824, F1=0.6487
- Epoch 7: Loss=0.1083, mIoU=0.5893, F1=0.6586
- Epoch 8: Loss=0.1025, mIoU=0.6970, F1=0.7857
- Epoch 9: Loss=0.1029, mIoU=0.6402, F1=0.7238
- Epoch 10: Loss=0.0989, mIoU=0.6587, F1=0.7452

本轮结果: Round=6, Labeled=629, Selection=best_val (epoch=8), mIoU=0.6970, F1=0.7857, peak_mIoU=0.6970

Round=6, Labeled=629, mIoU=0.6970, F1=0.7857

## Round 7

Labeled Pool Size: 717

- Epoch 1: Loss=0.2665, mIoU=0.5527, F1=0.6042
- Epoch 2: Loss=0.1493, mIoU=0.5747, F1=0.6373
- Epoch 3: Loss=0.1297, mIoU=0.6641, F1=0.7510
- Epoch 4: Loss=0.1233, mIoU=0.6482, F1=0.7330
- Epoch 5: Loss=0.1125, mIoU=0.6449, F1=0.7292
- Epoch 6: Loss=0.1063, mIoU=0.6576, F1=0.7438
- Epoch 7: Loss=0.1052, mIoU=0.7109, F1=0.7996
- Epoch 8: Loss=0.0989, mIoU=0.6576, F1=0.7442
- Epoch 9: Loss=0.0974, mIoU=0.6416, F1=0.7256
- Epoch 10: Loss=0.0940, mIoU=0.6659, F1=0.7532

本轮结果: Round=7, Labeled=717, Selection=best_val (epoch=7), mIoU=0.7109, F1=0.7996, peak_mIoU=0.7109

Round=7, Labeled=717, mIoU=0.7109, F1=0.7996

## Round 8

Labeled Pool Size: 805

- Epoch 1: Loss=0.2604, mIoU=0.5514, F1=0.6021
- Epoch 2: Loss=0.1461, mIoU=0.5414, F1=0.5859
- Epoch 3: Loss=0.1267, mIoU=0.5998, F1=0.6725
- Epoch 4: Loss=0.1150, mIoU=0.6678, F1=0.7552
- Epoch 5: Loss=0.1104, mIoU=0.6679, F1=0.7551
- Epoch 6: Loss=0.1048, mIoU=0.6767, F1=0.7650
- Epoch 7: Loss=0.1020, mIoU=0.7124, F1=0.8010
- Epoch 8: Loss=0.0956, mIoU=0.7212, F1=0.8095
- Epoch 9: Loss=0.0924, mIoU=0.7135, F1=0.8021
- Epoch 10: Loss=0.0900, mIoU=0.7053, F1=0.7943

本轮结果: Round=8, Labeled=805, Selection=best_val (epoch=8), mIoU=0.7212, F1=0.8095, peak_mIoU=0.7212

Round=8, Labeled=805, mIoU=0.7212, F1=0.8095

