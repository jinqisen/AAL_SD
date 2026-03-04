# 实验日志

实验名称: full_model
描述: 完整模型（正式方案）：coreset-to-labeled K + 自适应阈值 rollback + 风险驱动 λ 闭环控制（不调整 query_size）
开始时间: 2026-02-18T20:56:05.765348

## Round 1

Labeled Pool Size: 151

- Epoch 1: Loss=0.6121, mIoU=0.4727, F1=0.5106
- Epoch 2: Loss=0.3461, mIoU=0.5064, F1=0.5328
- Epoch 3: Loss=0.2152, mIoU=0.4982, F1=0.5145
- Epoch 4: Loss=0.1491, mIoU=0.5258, F1=0.5649
- Epoch 5: Loss=0.1209, mIoU=0.5339, F1=0.5780
- Epoch 6: Loss=0.0964, mIoU=0.5828, F1=0.6527
- Epoch 7: Loss=0.0840, mIoU=0.5988, F1=0.6749
- Epoch 8: Loss=0.0732, mIoU=0.5604, F1=0.6193
- Epoch 9: Loss=0.0662, mIoU=0.5580, F1=0.6156
- Epoch 10: Loss=0.0605, mIoU=0.6181, F1=0.6995

当前轮次最佳结果: Round=1, Labeled=151, mIoU=0.6181, F1=0.6995

## Round 2

Labeled Pool Size: 239

- Epoch 1: Loss=0.4958, mIoU=0.5695, F1=0.6421
- Epoch 2: Loss=0.2571, mIoU=0.5768, F1=0.6436
- Epoch 3: Loss=0.1766, mIoU=0.6272, F1=0.7103
- Epoch 4: Loss=0.1452, mIoU=0.6204, F1=0.7018
- Epoch 5: Loss=0.1272, mIoU=0.6487, F1=0.7356
- Epoch 6: Loss=0.1171, mIoU=0.6300, F1=0.7134
- Epoch 7: Loss=0.1046, mIoU=0.6448, F1=0.7310
- Epoch 8: Loss=0.0989, mIoU=0.6852, F1=0.7755
- Epoch 9: Loss=0.0922, mIoU=0.6763, F1=0.7658
- Epoch 10: Loss=0.0937, mIoU=0.6970, F1=0.7872

当前轮次最佳结果: Round=2, Labeled=239, mIoU=0.6970, F1=0.7872

## Round 3

Labeled Pool Size: 327

- Epoch 1: Loss=0.3208, mIoU=0.6354, F1=0.7209
- Epoch 2: Loss=0.1855, mIoU=0.6325, F1=0.7165
- Epoch 3: Loss=0.1500, mIoU=0.6471, F1=0.7335
- Epoch 4: Loss=0.1413, mIoU=0.6141, F1=0.6936
- Epoch 5: Loss=0.1275, mIoU=0.7063, F1=0.7962
- Epoch 6: Loss=0.1173, mIoU=0.6923, F1=0.7824
- Epoch 7: Loss=0.1108, mIoU=0.7176, F1=0.8070
- Epoch 8: Loss=0.1113, mIoU=0.7083, F1=0.7980
- Epoch 9: Loss=0.1060, mIoU=0.7117, F1=0.8013
- Epoch 10: Loss=0.0999, mIoU=0.7162, F1=0.8056

当前轮次最佳结果: Round=3, Labeled=327, mIoU=0.7176, F1=0.8070

## Round 4

Labeled Pool Size: 415

- Epoch 1: Loss=0.3893, mIoU=0.6343, F1=0.7192
- Epoch 2: Loss=0.1959, mIoU=0.6469, F1=0.7334
- Epoch 3: Loss=0.1550, mIoU=0.6798, F1=0.7694
- Epoch 4: Loss=0.1373, mIoU=0.6787, F1=0.7683
- Epoch 5: Loss=0.1343, mIoU=0.7219, F1=0.8112
- Epoch 6: Loss=0.1246, mIoU=0.6930, F1=0.7829
- Epoch 7: Loss=0.1183, mIoU=0.7124, F1=0.8019
- Epoch 8: Loss=0.1133, mIoU=0.7291, F1=0.8176
- Epoch 9: Loss=0.1077, mIoU=0.7130, F1=0.8024
- Epoch 10: Loss=0.1040, mIoU=0.7295, F1=0.8179

当前轮次最佳结果: Round=4, Labeled=415, mIoU=0.7295, F1=0.8179

## Round 5

Labeled Pool Size: 503

- Epoch 1: Loss=0.3482, mIoU=0.6321, F1=0.7162
- Epoch 2: Loss=0.1762, mIoU=0.6673, F1=0.7561
- Epoch 3: Loss=0.1481, mIoU=0.7007, F1=0.7907
- Epoch 4: Loss=0.1304, mIoU=0.6918, F1=0.7816
- Epoch 5: Loss=0.1248, mIoU=0.6871, F1=0.7769
- Epoch 6: Loss=0.1145, mIoU=0.7177, F1=0.8069
- Epoch 7: Loss=0.1089, mIoU=0.7112, F1=0.8007
- Epoch 8: Loss=0.1071, mIoU=0.7263, F1=0.8150
- Epoch 9: Loss=0.1029, mIoU=0.7355, F1=0.8233
- Epoch 10: Loss=0.1022, mIoU=0.7331, F1=0.8211

当前轮次最佳结果: Round=5, Labeled=503, mIoU=0.7355, F1=0.8233

## Round 6

Labeled Pool Size: 591

- Epoch 1: Loss=0.3048, mIoU=0.6456, F1=0.7321
- Epoch 2: Loss=0.1595, mIoU=0.6823, F1=0.7721
- Epoch 3: Loss=0.1346, mIoU=0.7210, F1=0.8106
- Epoch 4: Loss=0.1231, mIoU=0.6882, F1=0.7779
- Epoch 5: Loss=0.1206, mIoU=0.6876, F1=0.7772
- Epoch 6: Loss=0.1111, mIoU=0.7369, F1=0.8246
- Epoch 7: Loss=0.1074, mIoU=0.7288, F1=0.8173
- Epoch 8: Loss=0.1049, mIoU=0.7317, F1=0.8199
- Epoch 9: Loss=0.1003, mIoU=0.7432, F1=0.8302
- Epoch 10: Loss=0.0962, mIoU=0.7186, F1=0.8077

当前轮次最佳结果: Round=6, Labeled=591, mIoU=0.7432, F1=0.8302

## Round 7

Labeled Pool Size: 679

- Epoch 1: Loss=0.3533, mIoU=0.6647, F1=0.7537
- Epoch 2: Loss=0.1589, mIoU=0.7053, F1=0.7954
- Epoch 3: Loss=0.1362, mIoU=0.7142, F1=0.8038
- Epoch 4: Loss=0.1211, mIoU=0.7155, F1=0.8049
- Epoch 5: Loss=0.1138, mIoU=0.7398, F1=0.8274
- Epoch 6: Loss=0.1081, mIoU=0.7240, F1=0.8127
- Epoch 7: Loss=0.1036, mIoU=0.7386, F1=0.8261
- Epoch 8: Loss=0.0994, mIoU=0.7291, F1=0.8176
- Epoch 9: Loss=0.0945, mIoU=0.7500, F1=0.8361
- Epoch 10: Loss=0.0932, mIoU=0.7378, F1=0.8253

当前轮次最佳结果: Round=7, Labeled=679, mIoU=0.7500, F1=0.8361

## Round 8

Labeled Pool Size: 767

- Epoch 1: Loss=0.2954, mIoU=0.6469, F1=0.7335
- Epoch 2: Loss=0.1450, mIoU=0.6534, F1=0.7406
- Epoch 3: Loss=0.1239, mIoU=0.6953, F1=0.7851
- Epoch 4: Loss=0.1132, mIoU=0.7318, F1=0.8201
- Epoch 5: Loss=0.1065, mIoU=0.7405, F1=0.8279
- Epoch 6: Loss=0.1022, mIoU=0.7039, F1=0.7935
- Epoch 7: Loss=0.0965, mIoU=0.7428, F1=0.8300
- Epoch 8: Loss=0.0960, mIoU=0.7190, F1=0.8081
- Epoch 9: Loss=0.0892, mIoU=0.7457, F1=0.8324
- Epoch 10: Loss=0.0877, mIoU=0.7496, F1=0.8358

当前轮次最佳结果: Round=8, Labeled=767, mIoU=0.7496, F1=0.8358

## Round 9

Labeled Pool Size: 855

- Epoch 1: Loss=0.3230, mIoU=0.6564, F1=0.7447
- Epoch 2: Loss=0.1403, mIoU=0.6490, F1=0.7356
- Epoch 3: Loss=0.1207, mIoU=0.7132, F1=0.8026
- Epoch 4: Loss=0.1104, mIoU=0.7356, F1=0.8235
- Epoch 5: Loss=0.1057, mIoU=0.6627, F1=0.7509
- Epoch 6: Loss=0.1019, mIoU=0.7224, F1=0.8113
- Epoch 7: Loss=0.0988, mIoU=0.7391, F1=0.8267
- Epoch 8: Loss=0.0951, mIoU=0.7474, F1=0.8338
- Epoch 9: Loss=0.0907, mIoU=0.7608, F1=0.8455
- Epoch 10: Loss=0.0880, mIoU=0.7317, F1=0.8198

当前轮次最佳结果: Round=9, Labeled=855, mIoU=0.7608, F1=0.8455

## Round 10

Labeled Pool Size: 943

- Epoch 1: Loss=0.3108, mIoU=0.6779, F1=0.7682
- Epoch 2: Loss=0.1340, mIoU=0.7096, F1=0.7994
- Epoch 3: Loss=0.1124, mIoU=0.7240, F1=0.8128
- Epoch 4: Loss=0.1040, mIoU=0.7484, F1=0.8349
- Epoch 5: Loss=0.0975, mIoU=0.7491, F1=0.8355
- Epoch 6: Loss=0.0911, mIoU=0.7227, F1=0.8115
- Epoch 7: Loss=0.0893, mIoU=0.7436, F1=0.8305
- Epoch 8: Loss=0.0861, mIoU=0.7456, F1=0.8322
- Epoch 9: Loss=0.0819, mIoU=0.7440, F1=0.8308
- Epoch 10: Loss=0.0798, mIoU=0.7571, F1=0.8424

当前轮次最佳结果: Round=10, Labeled=943, mIoU=0.7571, F1=0.8424

## Round 11

Labeled Pool Size: 1031

- Epoch 1: Loss=0.2181, mIoU=0.6546, F1=0.7421
- Epoch 2: Loss=0.1159, mIoU=0.7160, F1=0.8055
- Epoch 3: Loss=0.0996, mIoU=0.7273, F1=0.8159
- Epoch 4: Loss=0.0920, mIoU=0.7047, F1=0.7943
- Epoch 5: Loss=0.0867, mIoU=0.7529, F1=0.8387
- Epoch 6: Loss=0.0830, mIoU=0.7523, F1=0.8381
- Epoch 7: Loss=0.0793, mIoU=0.7615, F1=0.8462
- Epoch 8: Loss=0.0756, mIoU=0.7389, F1=0.8270
- Epoch 9: Loss=0.0722, mIoU=0.7595, F1=0.8446
- Epoch 10: Loss=0.0702, mIoU=0.7613, F1=0.8457

当前轮次最佳结果: Round=11, Labeled=1031, mIoU=0.7615, F1=0.8462

## Round 12

Labeled Pool Size: 1119

- Epoch 1: Loss=0.2008, mIoU=0.6637, F1=0.7522
- Epoch 2: Loss=0.1116, mIoU=0.7033, F1=0.7932
- Epoch 3: Loss=0.0978, mIoU=0.7235, F1=0.8124
- Epoch 4: Loss=0.0907, mIoU=0.7205, F1=0.8095
- Epoch 5: Loss=0.0846, mIoU=0.7440, F1=0.8309
- Epoch 6: Loss=0.0809, mIoU=0.7501, F1=0.8363
- Epoch 7: Loss=0.0767, mIoU=0.7616, F1=0.8460
- Epoch 8: Loss=0.0741, mIoU=0.7592, F1=0.8441
- Epoch 9: Loss=0.0721, mIoU=0.7336, F1=0.8214
- Epoch 10: Loss=0.0700, mIoU=0.7327, F1=0.8206

当前轮次最佳结果: Round=12, Labeled=1119, mIoU=0.7616, F1=0.8460

## Round 13

Labeled Pool Size: 1207

- Epoch 1: Loss=0.1802, mIoU=0.6778, F1=0.7674
- Epoch 2: Loss=0.1022, mIoU=0.7220, F1=0.8113
- Epoch 3: Loss=0.0899, mIoU=0.6842, F1=0.7737
- Epoch 4: Loss=0.0840, mIoU=0.7014, F1=0.7911
- Epoch 5: Loss=0.0793, mIoU=0.7533, F1=0.8390
- Epoch 6: Loss=0.0762, mIoU=0.7475, F1=0.8339
- Epoch 7: Loss=0.0726, mIoU=0.7540, F1=0.8396
- Epoch 8: Loss=0.0689, mIoU=0.7368, F1=0.8243
- Epoch 9: Loss=0.0674, mIoU=0.7502, F1=0.8362
- Epoch 10: Loss=0.0636, mIoU=0.7632, F1=0.8475

当前轮次最佳结果: Round=13, Labeled=1207, mIoU=0.7632, F1=0.8475

## Round 14

Labeled Pool Size: 1295

- Epoch 1: Loss=0.2435, mIoU=0.6013, F1=0.6770
- Epoch 2: Loss=0.1041, mIoU=0.7133, F1=0.8037
- Epoch 3: Loss=0.0904, mIoU=0.6873, F1=0.7769
- Epoch 4: Loss=0.0841, mIoU=0.7382, F1=0.8258
- Epoch 5: Loss=0.0801, mIoU=0.7317, F1=0.8200
- Epoch 6: Loss=0.0762, mIoU=0.7064, F1=0.7960
- Epoch 7: Loss=0.0741, mIoU=0.7510, F1=0.8369
- Epoch 8: Loss=0.0701, mIoU=0.7476, F1=0.8340
- Epoch 9: Loss=0.0678, mIoU=0.7372, F1=0.8247
- Epoch 10: Loss=0.0670, mIoU=0.7471, F1=0.8335

当前轮次最佳结果: Round=14, Labeled=1295, mIoU=0.7510, F1=0.8369

## Round 15

Labeled Pool Size: 1383

- Epoch 1: Loss=0.1708, mIoU=0.6554, F1=0.7429
- Epoch 2: Loss=0.0962, mIoU=0.6241, F1=0.7059
- Epoch 3: Loss=0.0829, mIoU=0.7362, F1=0.8243
- Epoch 4: Loss=0.0777, mIoU=0.7016, F1=0.7913
- Epoch 5: Loss=0.0716, mIoU=0.7358, F1=0.8234
- Epoch 6: Loss=0.0675, mIoU=0.7532, F1=0.8390
- Epoch 7: Loss=0.0650, mIoU=0.7532, F1=0.8388
- Epoch 8: Loss=0.0633, mIoU=0.7198, F1=0.8087
- Epoch 9: Loss=0.0618, mIoU=0.7663, F1=0.8501
- Epoch 10: Loss=0.0593, mIoU=0.7525, F1=0.8382

当前轮次最佳结果: Round=15, Labeled=1383, mIoU=0.7663, F1=0.8501


## 实验汇总

预算历史: [151, 239, 327, 415, 503, 591, 679, 767, 855, 943, 1031, 1119, 1207, 1295, 1383]
ALC: 0.6694
最终 mIoU: 0.7663
最终 F1: 0.8501
