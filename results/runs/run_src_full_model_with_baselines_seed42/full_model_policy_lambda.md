# 实验日志

实验名称: full_model_policy_lambda
描述: 完整模型（规则闭环λ）：固定Warmup(0.2) + 自适应阈值 rollback + 风险驱动 λ 闭环控制（不调整 query_size）
开始时间: 2026-02-19T09:47:15.805255

## Round 1

Labeled Pool Size: 151

- Epoch 1: Loss=0.6117, mIoU=0.4580, F1=0.5067
- Epoch 2: Loss=0.3465, mIoU=0.5009, F1=0.5215
- Epoch 3: Loss=0.2162, mIoU=0.4959, F1=0.5099
- Epoch 4: Loss=0.1464, mIoU=0.5025, F1=0.5223
- Epoch 5: Loss=0.1142, mIoU=0.5084, F1=0.5331
- Epoch 6: Loss=0.0932, mIoU=0.6111, F1=0.6912
- Epoch 7: Loss=0.0821, mIoU=0.5935, F1=0.6676
- Epoch 8: Loss=0.0716, mIoU=0.5513, F1=0.6053
- Epoch 9: Loss=0.0659, mIoU=0.5480, F1=0.6000
- Epoch 10: Loss=0.0629, mIoU=0.6097, F1=0.6882

当前轮次最佳结果: Round=1, Labeled=151, mIoU=0.6111, F1=0.6912

## Round 2

Labeled Pool Size: 239

- Epoch 1: Loss=0.4912, mIoU=0.5948, F1=0.6706
- Epoch 2: Loss=0.2528, mIoU=0.6218, F1=0.7039
- Epoch 3: Loss=0.1810, mIoU=0.6485, F1=0.7354
- Epoch 4: Loss=0.1472, mIoU=0.6544, F1=0.7420
- Epoch 5: Loss=0.1303, mIoU=0.6827, F1=0.7727
- Epoch 6: Loss=0.1165, mIoU=0.6724, F1=0.7618
- Epoch 7: Loss=0.1079, mIoU=0.6705, F1=0.7596
- Epoch 8: Loss=0.1027, mIoU=0.6378, F1=0.7227
- Epoch 9: Loss=0.0974, mIoU=0.6562, F1=0.7440
- Epoch 10: Loss=0.0924, mIoU=0.6973, F1=0.7875

当前轮次最佳结果: Round=2, Labeled=239, mIoU=0.6973, F1=0.7875

## Round 3

Labeled Pool Size: 327

- Epoch 1: Loss=0.3216, mIoU=0.5959, F1=0.6709
- Epoch 2: Loss=0.1853, mIoU=0.6445, F1=0.7310
- Epoch 3: Loss=0.1543, mIoU=0.6640, F1=0.7527
- Epoch 4: Loss=0.1396, mIoU=0.6288, F1=0.7119
- Epoch 5: Loss=0.1298, mIoU=0.6912, F1=0.7812
- Epoch 6: Loss=0.1263, mIoU=0.6909, F1=0.7808
- Epoch 7: Loss=0.1188, mIoU=0.6917, F1=0.7816
- Epoch 8: Loss=0.1132, mIoU=0.7106, F1=0.8003
- Epoch 9: Loss=0.1108, mIoU=0.6970, F1=0.7870
- Epoch 10: Loss=0.1094, mIoU=0.7043, F1=0.7942

当前轮次最佳结果: Round=3, Labeled=327, mIoU=0.7106, F1=0.8003

## Round 4

Labeled Pool Size: 415

- Epoch 1: Loss=0.3894, mIoU=0.6140, F1=0.6944
- Epoch 2: Loss=0.1987, mIoU=0.6558, F1=0.7435
- Epoch 3: Loss=0.1609, mIoU=0.6706, F1=0.7597
- Epoch 4: Loss=0.1373, mIoU=0.6717, F1=0.7607
- Epoch 5: Loss=0.1316, mIoU=0.6387, F1=0.7236
- Epoch 6: Loss=0.1220, mIoU=0.7089, F1=0.7986
- Epoch 7: Loss=0.1178, mIoU=0.6871, F1=0.7770
- Epoch 8: Loss=0.1128, mIoU=0.7355, F1=0.8238
- Epoch 9: Loss=0.1097, mIoU=0.7315, F1=0.8198
- Epoch 10: Loss=0.1059, mIoU=0.7306, F1=0.8190

当前轮次最佳结果: Round=4, Labeled=415, mIoU=0.7355, F1=0.8238

## Round 5

Labeled Pool Size: 503

- Epoch 1: Loss=0.3451, mIoU=0.5893, F1=0.6610
- Epoch 2: Loss=0.1757, mIoU=0.6518, F1=0.7389
- Epoch 3: Loss=0.1467, mIoU=0.6974, F1=0.7876
- Epoch 4: Loss=0.1345, mIoU=0.7144, F1=0.8041
- Epoch 5: Loss=0.1257, mIoU=0.7064, F1=0.7963
- Epoch 6: Loss=0.1175, mIoU=0.7196, F1=0.8089
- Epoch 7: Loss=0.1130, mIoU=0.7187, F1=0.8078
- Epoch 8: Loss=0.1075, mIoU=0.7237, F1=0.8128
- Epoch 9: Loss=0.1034, mIoU=0.7375, F1=0.8253
- Epoch 10: Loss=0.1026, mIoU=0.7231, F1=0.8119

当前轮次最佳结果: Round=5, Labeled=503, mIoU=0.7375, F1=0.8253

## Round 6

Labeled Pool Size: 591

- Epoch 1: Loss=0.3123, mIoU=0.6367, F1=0.7217
- Epoch 2: Loss=0.1565, mIoU=0.6403, F1=0.7255
- Epoch 3: Loss=0.1399, mIoU=0.6893, F1=0.7792
- Epoch 4: Loss=0.1247, mIoU=0.6920, F1=0.7818
- Epoch 5: Loss=0.1182, mIoU=0.7004, F1=0.7904
- Epoch 6: Loss=0.1109, mIoU=0.7126, F1=0.8021
- Epoch 7: Loss=0.1105, mIoU=0.6965, F1=0.7865
- Epoch 8: Loss=0.1046, mIoU=0.7445, F1=0.8314
- Epoch 9: Loss=0.0989, mIoU=0.7452, F1=0.8322
- Epoch 10: Loss=0.0961, mIoU=0.7213, F1=0.8102

当前轮次最佳结果: Round=6, Labeled=591, mIoU=0.7452, F1=0.8322

## Round 7

Labeled Pool Size: 679

- Epoch 1: Loss=0.3540, mIoU=0.6666, F1=0.7560
- Epoch 2: Loss=0.1609, mIoU=0.6988, F1=0.7889
- Epoch 3: Loss=0.1369, mIoU=0.6855, F1=0.7752
- Epoch 4: Loss=0.1209, mIoU=0.7237, F1=0.8125
- Epoch 5: Loss=0.1142, mIoU=0.7434, F1=0.8305
- Epoch 6: Loss=0.1056, mIoU=0.7332, F1=0.8212
- Epoch 7: Loss=0.1017, mIoU=0.7031, F1=0.7928
- Epoch 8: Loss=0.0985, mIoU=0.7311, F1=0.8193
- Epoch 9: Loss=0.0967, mIoU=0.7314, F1=0.8194
- Epoch 10: Loss=0.0917, mIoU=0.7489, F1=0.8351

当前轮次最佳结果: Round=7, Labeled=679, mIoU=0.7489, F1=0.8351

## Round 8

Labeled Pool Size: 767

- Epoch 1: Loss=0.2939, mIoU=0.6334, F1=0.7174
- Epoch 2: Loss=0.1459, mIoU=0.6747, F1=0.7640
- Epoch 3: Loss=0.1240, mIoU=0.6788, F1=0.7683
- Epoch 4: Loss=0.1104, mIoU=0.7408, F1=0.8284
- Epoch 5: Loss=0.1069, mIoU=0.6995, F1=0.7893
- Epoch 6: Loss=0.1000, mIoU=0.7354, F1=0.8232
- Epoch 7: Loss=0.0948, mIoU=0.7419, F1=0.8291
- Epoch 8: Loss=0.0914, mIoU=0.7399, F1=0.8272
- Epoch 9: Loss=0.0870, mIoU=0.7359, F1=0.8236
- Epoch 10: Loss=0.0852, mIoU=0.7268, F1=0.8153

当前轮次最佳结果: Round=8, Labeled=767, mIoU=0.7419, F1=0.8291

## Round 9

Labeled Pool Size: 855

- Epoch 1: Loss=0.3483, mIoU=0.6390, F1=0.7248
- Epoch 2: Loss=0.1449, mIoU=0.6791, F1=0.7687
- Epoch 3: Loss=0.1206, mIoU=0.7151, F1=0.8046
- Epoch 4: Loss=0.1095, mIoU=0.7078, F1=0.7974
- Epoch 5: Loss=0.1032, mIoU=0.7233, F1=0.8121
- Epoch 6: Loss=0.0999, mIoU=0.7067, F1=0.7962
- Epoch 7: Loss=0.0943, mIoU=0.7278, F1=0.8162
- Epoch 8: Loss=0.0907, mIoU=0.7543, F1=0.8399
- Epoch 9: Loss=0.0879, mIoU=0.7389, F1=0.8263
- Epoch 10: Loss=0.0849, mIoU=0.7168, F1=0.8060

当前轮次最佳结果: Round=9, Labeled=855, mIoU=0.7543, F1=0.8399

## Round 10

Labeled Pool Size: 943

- Epoch 1: Loss=0.3137, mIoU=0.6351, F1=0.7195
- Epoch 2: Loss=0.1321, mIoU=0.6410, F1=0.7264
- Epoch 3: Loss=0.1107, mIoU=0.6827, F1=0.7722
- Epoch 4: Loss=0.1017, mIoU=0.7090, F1=0.7985
- Epoch 5: Loss=0.0971, mIoU=0.7325, F1=0.8206
- Epoch 6: Loss=0.0920, mIoU=0.7344, F1=0.8222
- Epoch 7: Loss=0.0867, mIoU=0.7378, F1=0.8256
- Epoch 8: Loss=0.0858, mIoU=0.7442, F1=0.8309
- Epoch 9: Loss=0.0804, mIoU=0.7395, F1=0.8268
- Epoch 10: Loss=0.0785, mIoU=0.7592, F1=0.8439

当前轮次最佳结果: Round=10, Labeled=943, mIoU=0.7592, F1=0.8439

## Round 11

Labeled Pool Size: 1031

- Epoch 1: Loss=0.2155, mIoU=0.6790, F1=0.7689
- Epoch 2: Loss=0.1173, mIoU=0.6878, F1=0.7776
- Epoch 3: Loss=0.1004, mIoU=0.7169, F1=0.8062
- Epoch 4: Loss=0.0927, mIoU=0.6982, F1=0.7880
- Epoch 5: Loss=0.0871, mIoU=0.7133, F1=0.8026
- Epoch 6: Loss=0.0824, mIoU=0.7308, F1=0.8190
- Epoch 7: Loss=0.0799, mIoU=0.7611, F1=0.8456
- Epoch 8: Loss=0.0755, mIoU=0.7506, F1=0.8367
- Epoch 9: Loss=0.0747, mIoU=0.7481, F1=0.8344
- Epoch 10: Loss=0.0723, mIoU=0.7591, F1=0.8439

当前轮次最佳结果: Round=11, Labeled=1031, mIoU=0.7611, F1=0.8456

## Round 12

Labeled Pool Size: 1119

- Epoch 1: Loss=0.1985, mIoU=0.6548, F1=0.7423
- Epoch 2: Loss=0.1094, mIoU=0.7166, F1=0.8063
- Epoch 3: Loss=0.0961, mIoU=0.6595, F1=0.7473
- Epoch 4: Loss=0.0880, mIoU=0.6832, F1=0.7726
- Epoch 5: Loss=0.0819, mIoU=0.7000, F1=0.7897
- Epoch 6: Loss=0.0792, mIoU=0.7574, F1=0.8426
- Epoch 7: Loss=0.0756, mIoU=0.7608, F1=0.8454
- Epoch 8: Loss=0.0715, mIoU=0.7629, F1=0.8471
- Epoch 9: Loss=0.0682, mIoU=0.7571, F1=0.8421
- Epoch 10: Loss=0.0675, mIoU=0.7523, F1=0.8382

当前轮次最佳结果: Round=12, Labeled=1119, mIoU=0.7629, F1=0.8471

## Round 13

Labeled Pool Size: 1207

- Epoch 1: Loss=0.1854, mIoU=0.6679, F1=0.7571
- Epoch 2: Loss=0.1028, mIoU=0.6826, F1=0.7722
- Epoch 3: Loss=0.0914, mIoU=0.7304, F1=0.8188
- Epoch 4: Loss=0.0836, mIoU=0.7190, F1=0.8081
- Epoch 5: Loss=0.0795, mIoU=0.7255, F1=0.8142
- Epoch 6: Loss=0.0759, mIoU=0.7324, F1=0.8204
- Epoch 7: Loss=0.0727, mIoU=0.7535, F1=0.8391
- Epoch 8: Loss=0.0688, mIoU=0.7364, F1=0.8239
- Epoch 9: Loss=0.0672, mIoU=0.7573, F1=0.8423
- Epoch 10: Loss=0.0654, mIoU=0.7673, F1=0.8508

当前轮次最佳结果: Round=13, Labeled=1207, mIoU=0.7673, F1=0.8508

## Round 14

Labeled Pool Size: 1295

- Epoch 1: Loss=0.2392, mIoU=0.6706, F1=0.7599
- Epoch 2: Loss=0.1062, mIoU=0.7176, F1=0.8072
- Epoch 3: Loss=0.0927, mIoU=0.6984, F1=0.7882
- Epoch 4: Loss=0.0845, mIoU=0.6741, F1=0.7632
- Epoch 5: Loss=0.0788, mIoU=0.7142, F1=0.8035
- Epoch 6: Loss=0.0764, mIoU=0.7435, F1=0.8304
- Epoch 7: Loss=0.0720, mIoU=0.7515, F1=0.8373
- Epoch 8: Loss=0.0692, mIoU=0.7363, F1=0.8239
- Epoch 9: Loss=0.0681, mIoU=0.7409, F1=0.8281
- Epoch 10: Loss=0.0652, mIoU=0.7601, F1=0.8446

当前轮次最佳结果: Round=14, Labeled=1295, mIoU=0.7601, F1=0.8446

## Round 15

Labeled Pool Size: 1383

- Epoch 1: Loss=0.1688, mIoU=0.6464, F1=0.7328
- Epoch 2: Loss=0.0955, mIoU=0.6449, F1=0.7308
- Epoch 3: Loss=0.0846, mIoU=0.7224, F1=0.8113
- Epoch 4: Loss=0.0787, mIoU=0.7485, F1=0.8351
- Epoch 5: Loss=0.0743, mIoU=0.7375, F1=0.8250
- Epoch 6: Loss=0.0696, mIoU=0.7535, F1=0.8391
- Epoch 7: Loss=0.0684, mIoU=0.7580, F1=0.8429
- Epoch 8: Loss=0.0664, mIoU=0.7590, F1=0.8437
- Epoch 9: Loss=0.0625, mIoU=0.7608, F1=0.8455
- Epoch 10: Loss=0.0601, mIoU=0.7613, F1=0.8456

当前轮次最佳结果: Round=15, Labeled=1383, mIoU=0.7613, F1=0.8456


## 实验汇总

预算历史: [151, 239, 327, 415, 503, 591, 679, 767, 855, 943, 1031, 1119, 1207, 1295, 1383]
ALC: 0.6688
最终 mIoU: 0.7613
最终 F1: 0.8456
