# 实验日志

实验名称: full_model
描述: 完整模型（正式方案）：coreset-to-labeled K + 自适应阈值 rollback + 风险驱动 λ 闭环控制（不调整 query_size）
开始时间: 2026-02-13T04:45:41.743843

## Round 1

Labeled Pool Size: 151

- Epoch 1: Loss=0.6127, mIoU=0.4708, F1=0.5107
- Epoch 2: Loss=0.3470, mIoU=0.5142, F1=0.5462
- Epoch 3: Loss=0.2127, mIoU=0.5119, F1=0.5399
- Epoch 4: Loss=0.1403, mIoU=0.5322, F1=0.5751
- Epoch 5: Loss=0.1055, mIoU=0.5509, F1=0.6052
- Epoch 6: Loss=0.0843, mIoU=0.6176, F1=0.6996
- Epoch 7: Loss=0.0757, mIoU=0.5810, F1=0.6501
- Epoch 8: Loss=0.0661, mIoU=0.5409, F1=0.5888
- Epoch 9: Loss=0.0638, mIoU=0.5755, F1=0.6417
- Epoch 10: Loss=0.0589, mIoU=0.6238, F1=0.7062

当前轮次最佳结果: Round=1, Labeled=151, mIoU=0.6238, F1=0.7062

## Round 2

Labeled Pool Size: 239

- Epoch 1: Loss=0.4916, mIoU=0.5382, F1=0.5856
- Epoch 2: Loss=0.2526, mIoU=0.5999, F1=0.6760
- Epoch 3: Loss=0.1702, mIoU=0.6355, F1=0.7206
- Epoch 4: Loss=0.1406, mIoU=0.5950, F1=0.6687
- Epoch 5: Loss=0.1250, mIoU=0.6376, F1=0.7226
- Epoch 6: Loss=0.1096, mIoU=0.6490, F1=0.7359
- Epoch 7: Loss=0.1022, mIoU=0.6842, F1=0.7742
- Epoch 8: Loss=0.0982, mIoU=0.6757, F1=0.7653
- Epoch 9: Loss=0.0939, mIoU=0.6670, F1=0.7559
- Epoch 10: Loss=0.0882, mIoU=0.6982, F1=0.7884

当前轮次最佳结果: Round=2, Labeled=239, mIoU=0.6982, F1=0.7884

## Round 3

Labeled Pool Size: 327

- Epoch 1: Loss=0.3211, mIoU=0.5685, F1=0.6318
- Epoch 2: Loss=0.1860, mIoU=0.5871, F1=0.6579
- Epoch 3: Loss=0.1510, mIoU=0.5735, F1=0.6386
- Epoch 4: Loss=0.1380, mIoU=0.6414, F1=0.7270
- Epoch 5: Loss=0.1249, mIoU=0.6378, F1=0.7227
- Epoch 6: Loss=0.1200, mIoU=0.6956, F1=0.7857
- Epoch 7: Loss=0.1119, mIoU=0.7074, F1=0.7977
- Epoch 8: Loss=0.1094, mIoU=0.6977, F1=0.7875
- Epoch 9: Loss=0.1046, mIoU=0.7003, F1=0.7902
- Epoch 10: Loss=0.1021, mIoU=0.7234, F1=0.8125

当前轮次最佳结果: Round=3, Labeled=327, mIoU=0.7234, F1=0.8125

## Round 4

Labeled Pool Size: 415

- Epoch 1: Loss=0.3909, mIoU=0.5963, F1=0.6723
- Epoch 2: Loss=0.2000, mIoU=0.5835, F1=0.6529
- Epoch 3: Loss=0.1585, mIoU=0.6452, F1=0.7314
- Epoch 4: Loss=0.1383, mIoU=0.6941, F1=0.7842
- Epoch 5: Loss=0.1297, mIoU=0.6831, F1=0.7730
- Epoch 6: Loss=0.1215, mIoU=0.6949, F1=0.7850
- Epoch 7: Loss=0.1195, mIoU=0.6851, F1=0.7748
- Epoch 8: Loss=0.1117, mIoU=0.7284, F1=0.8170
- Epoch 9: Loss=0.1085, mIoU=0.7165, F1=0.8061
- Epoch 10: Loss=0.1016, mIoU=0.7296, F1=0.8182

当前轮次最佳结果: Round=4, Labeled=415, mIoU=0.7296, F1=0.8182

## Round 5

Labeled Pool Size: 503

- Epoch 1: Loss=0.3474, mIoU=0.6106, F1=0.6896
- Epoch 2: Loss=0.1788, mIoU=0.6744, F1=0.7638
- Epoch 3: Loss=0.1451, mIoU=0.7004, F1=0.7903
- Epoch 4: Loss=0.1330, mIoU=0.7270, F1=0.8159
- Epoch 5: Loss=0.1242, mIoU=0.7195, F1=0.8086
- Epoch 6: Loss=0.1180, mIoU=0.7208, F1=0.8098
- Epoch 7: Loss=0.1150, mIoU=0.7310, F1=0.8192
- Epoch 8: Loss=0.1074, mIoU=0.7292, F1=0.8175
- Epoch 9: Loss=0.1069, mIoU=0.7391, F1=0.8271
- Epoch 10: Loss=0.1023, mIoU=0.7423, F1=0.8294

当前轮次最佳结果: Round=5, Labeled=503, mIoU=0.7423, F1=0.8294

## Round 6

Labeled Pool Size: 591

- Epoch 1: Loss=0.3138, mIoU=0.6336, F1=0.7182
- Epoch 2: Loss=0.1645, mIoU=0.6672, F1=0.7560
- Epoch 3: Loss=0.1406, mIoU=0.7029, F1=0.7928
- Epoch 4: Loss=0.1260, mIoU=0.7210, F1=0.8101
- Epoch 5: Loss=0.1173, mIoU=0.7317, F1=0.8200
- Epoch 6: Loss=0.1109, mIoU=0.6992, F1=0.7890
- Epoch 7: Loss=0.1087, mIoU=0.7062, F1=0.7958
- Epoch 8: Loss=0.1035, mIoU=0.7448, F1=0.8317
- Epoch 9: Loss=0.0997, mIoU=0.7385, F1=0.8260
- Epoch 10: Loss=0.0984, mIoU=0.7491, F1=0.8357

当前轮次最佳结果: Round=6, Labeled=591, mIoU=0.7491, F1=0.8357

## Round 7

Labeled Pool Size: 679

- Epoch 1: Loss=0.3591, mIoU=0.6281, F1=0.7113
- Epoch 2: Loss=0.1646, mIoU=0.6943, F1=0.7846
- Epoch 3: Loss=0.1386, mIoU=0.6943, F1=0.7842
- Epoch 4: Loss=0.1211, mIoU=0.7249, F1=0.8137
- Epoch 5: Loss=0.1145, mIoU=0.7345, F1=0.8227
- Epoch 6: Loss=0.1075, mIoU=0.7167, F1=0.8060
- Epoch 7: Loss=0.1022, mIoU=0.7298, F1=0.8180
- Epoch 8: Loss=0.1010, mIoU=0.7203, F1=0.8093
- Epoch 9: Loss=0.0969, mIoU=0.7554, F1=0.8408
- Epoch 10: Loss=0.0932, mIoU=0.7508, F1=0.8368

当前轮次最佳结果: Round=7, Labeled=679, mIoU=0.7554, F1=0.8408

## Round 8

Labeled Pool Size: 767

- Epoch 1: Loss=0.2944, mIoU=0.6725, F1=0.7626
- Epoch 2: Loss=0.1465, mIoU=0.6659, F1=0.7545
- Epoch 3: Loss=0.1222, mIoU=0.6844, F1=0.7741
- Epoch 4: Loss=0.1104, mIoU=0.7344, F1=0.8225
- Epoch 5: Loss=0.1050, mIoU=0.7270, F1=0.8156
- Epoch 6: Loss=0.0997, mIoU=0.7078, F1=0.7974
- Epoch 7: Loss=0.0960, mIoU=0.7324, F1=0.8205
- Epoch 8: Loss=0.0932, mIoU=0.7420, F1=0.8291
- Epoch 9: Loss=0.0880, mIoU=0.7483, F1=0.8350
- Epoch 10: Loss=0.0877, mIoU=0.7260, F1=0.8146

当前轮次最佳结果: Round=8, Labeled=767, mIoU=0.7483, F1=0.8350

## Round 9

Labeled Pool Size: 855

- Epoch 1: Loss=0.3232, mIoU=0.6342, F1=0.7188
- Epoch 2: Loss=0.1439, mIoU=0.6895, F1=0.7795
- Epoch 3: Loss=0.1235, mIoU=0.7134, F1=0.8029
- Epoch 4: Loss=0.1161, mIoU=0.7375, F1=0.8253
- Epoch 5: Loss=0.1050, mIoU=0.7156, F1=0.8049
- Epoch 6: Loss=0.1007, mIoU=0.6695, F1=0.7582
- Epoch 7: Loss=0.0999, mIoU=0.7301, F1=0.8183
- Epoch 8: Loss=0.0961, mIoU=0.7228, F1=0.8118
- Epoch 9: Loss=0.0905, mIoU=0.7287, F1=0.8170
- Epoch 10: Loss=0.0909, mIoU=0.7183, F1=0.8073

当前轮次最佳结果: Round=9, Labeled=855, mIoU=0.7375, F1=0.8253

## Round 10

Labeled Pool Size: 943

- Epoch 1: Loss=0.3058, mIoU=0.6643, F1=0.7533
- Epoch 2: Loss=0.1323, mIoU=0.6444, F1=0.7303
- Epoch 3: Loss=0.1119, mIoU=0.7055, F1=0.7952
- Epoch 4: Loss=0.1047, mIoU=0.7294, F1=0.8179
- Epoch 5: Loss=0.1012, mIoU=0.7348, F1=0.8228
- Epoch 6: Loss=0.0960, mIoU=0.7357, F1=0.8235
- Epoch 7: Loss=0.0913, mIoU=0.7451, F1=0.8319
- Epoch 8: Loss=0.0897, mIoU=0.7105, F1=0.7999
- Epoch 9: Loss=0.0864, mIoU=0.7507, F1=0.8368
- Epoch 10: Loss=0.0829, mIoU=0.7541, F1=0.8395

当前轮次最佳结果: Round=10, Labeled=943, mIoU=0.7541, F1=0.8395

## Round 11

Labeled Pool Size: 1031

- Epoch 1: Loss=0.2166, mIoU=0.6745, F1=0.7640
- Epoch 2: Loss=0.1171, mIoU=0.6618, F1=0.7499
- Epoch 3: Loss=0.1028, mIoU=0.7208, F1=0.8099
- Epoch 4: Loss=0.0943, mIoU=0.7086, F1=0.7981
- Epoch 5: Loss=0.0885, mIoU=0.7343, F1=0.8221
- Epoch 6: Loss=0.0848, mIoU=0.7417, F1=0.8288
- Epoch 7: Loss=0.0825, mIoU=0.7438, F1=0.8307
- Epoch 8: Loss=0.0794, mIoU=0.7524, F1=0.8381
- Epoch 9: Loss=0.0762, mIoU=0.7493, F1=0.8354
- Epoch 10: Loss=0.0739, mIoU=0.7660, F1=0.8497

当前轮次最佳结果: Round=11, Labeled=1031, mIoU=0.7660, F1=0.8497

## Round 12

Labeled Pool Size: 1119

- Epoch 1: Loss=0.2001, mIoU=0.6598, F1=0.7479
- Epoch 2: Loss=0.1112, mIoU=0.7023, F1=0.7921
- Epoch 3: Loss=0.0958, mIoU=0.7396, F1=0.8270
- Epoch 4: Loss=0.0897, mIoU=0.7363, F1=0.8242
- Epoch 5: Loss=0.0840, mIoU=0.7421, F1=0.8293
- Epoch 6: Loss=0.0780, mIoU=0.7440, F1=0.8308
- Epoch 7: Loss=0.0748, mIoU=0.7426, F1=0.8296
- Epoch 8: Loss=0.0737, mIoU=0.7532, F1=0.8387
- Epoch 9: Loss=0.0711, mIoU=0.7401, F1=0.8273
- Epoch 10: Loss=0.0691, mIoU=0.7432, F1=0.8301

当前轮次最佳结果: Round=12, Labeled=1119, mIoU=0.7532, F1=0.8387

## Round 13

Labeled Pool Size: 1207

- Epoch 1: Loss=0.1866, mIoU=0.6572, F1=0.7451
- Epoch 2: Loss=0.1044, mIoU=0.6616, F1=0.7497
- Epoch 3: Loss=0.0918, mIoU=0.7199, F1=0.8090
- Epoch 4: Loss=0.0848, mIoU=0.7366, F1=0.8243
- Epoch 5: Loss=0.0799, mIoU=0.7157, F1=0.8049
- Epoch 6: Loss=0.0761, mIoU=0.7188, F1=0.8081
- Epoch 7: Loss=0.0764, mIoU=0.7610, F1=0.8456
- Epoch 8: Loss=0.0693, mIoU=0.7497, F1=0.8357
- Epoch 9: Loss=0.0667, mIoU=0.7660, F1=0.8498
- Epoch 10: Loss=0.0670, mIoU=0.7473, F1=0.8345

当前轮次最佳结果: Round=13, Labeled=1207, mIoU=0.7660, F1=0.8498

## Round 14

Labeled Pool Size: 1295

- Epoch 1: Loss=0.2483, mIoU=0.6642, F1=0.7527
- Epoch 2: Loss=0.1070, mIoU=0.6870, F1=0.7768
- Epoch 3: Loss=0.0923, mIoU=0.7102, F1=0.7997
- Epoch 4: Loss=0.0844, mIoU=0.7435, F1=0.8307
- Epoch 5: Loss=0.0802, mIoU=0.7181, F1=0.8073
- Epoch 6: Loss=0.0758, mIoU=0.7133, F1=0.8026
- Epoch 7: Loss=0.0707, mIoU=0.7425, F1=0.8294
- Epoch 8: Loss=0.0676, mIoU=0.7251, F1=0.8137
- Epoch 9: Loss=0.0667, mIoU=0.7523, F1=0.8380
- Epoch 10: Loss=0.0639, mIoU=0.7603, F1=0.8449

当前轮次最佳结果: Round=14, Labeled=1295, mIoU=0.7603, F1=0.8449

## Round 15

Labeled Pool Size: 1383

- Epoch 1: Loss=0.1703, mIoU=0.5908, F1=0.6628
- Epoch 2: Loss=0.0944, mIoU=0.6675, F1=0.7561
- Epoch 3: Loss=0.0849, mIoU=0.7275, F1=0.8162
- Epoch 4: Loss=0.0782, mIoU=0.7332, F1=0.8211
- Epoch 5: Loss=0.0727, mIoU=0.7522, F1=0.8380
- Epoch 6: Loss=0.0686, mIoU=0.7558, F1=0.8411
- Epoch 7: Loss=0.0657, mIoU=0.7091, F1=0.7985
- Epoch 8: Loss=0.0638, mIoU=0.7565, F1=0.8417
- Epoch 9: Loss=0.0616, mIoU=0.7551, F1=0.8404
- Epoch 10: Loss=0.0589, mIoU=0.7420, F1=0.8290

当前轮次最佳结果: Round=15, Labeled=1383, mIoU=0.7565, F1=0.8417


## 实验汇总

预算历史: [151, 239, 327, 415, 503, 591, 679, 767, 855, 943, 1031, 1119, 1207, 1295, 1383]
ALC: 0.6687
最终 mIoU: 0.7565
最终 F1: 0.8417
