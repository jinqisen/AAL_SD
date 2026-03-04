# 实验日志

实验名称: no_agent
描述: 消融：移除Agent；λ由sigmoid自适应(随标注进度变化)，仅使用AD-KUCS数值策略选样
开始时间: 2026-03-02T10:59:41.594222

## Round 1

Labeled Pool Size: 151

- Epoch 1: Loss=0.5375, mIoU=0.4909, F1=0.5045
- Epoch 2: Loss=0.3055, mIoU=0.5029, F1=0.5237
- Epoch 3: Loss=0.1925, mIoU=0.5190, F1=0.5527
- Epoch 4: Loss=0.1376, mIoU=0.5221, F1=0.5576
- Epoch 5: Loss=0.1059, mIoU=0.5573, F1=0.6148
- Epoch 6: Loss=0.0866, mIoU=0.6216, F1=0.7049
- Epoch 7: Loss=0.0731, mIoU=0.6033, F1=0.6803
- Epoch 8: Loss=0.0682, mIoU=0.5771, F1=0.6439
- Epoch 9: Loss=0.0596, mIoU=0.6157, F1=0.6960
- Epoch 10: Loss=0.0556, mIoU=0.6159, F1=0.6964

当前轮次最佳结果: Round=1, Labeled=151, mIoU=0.6216, F1=0.7049

## Round 2

Labeled Pool Size: 239

- Epoch 1: Loss=0.3342, mIoU=0.4879, F1=0.4942
- Epoch 2: Loss=0.1919, mIoU=0.5098, F1=0.5355
- Epoch 3: Loss=0.1409, mIoU=0.5744, F1=0.6400
- Epoch 4: Loss=0.1219, mIoU=0.6044, F1=0.6814
- Epoch 5: Loss=0.1080, mIoU=0.5751, F1=0.6413
- Epoch 6: Loss=0.1044, mIoU=0.6451, F1=0.7313
- Epoch 7: Loss=0.0911, mIoU=0.6197, F1=0.7007
- Epoch 8: Loss=0.0845, mIoU=0.6548, F1=0.7425
- Epoch 9: Loss=0.0860, mIoU=0.6583, F1=0.7467
- Epoch 10: Loss=0.0815, mIoU=0.6679, F1=0.7570

当前轮次最佳结果: Round=2, Labeled=239, mIoU=0.6679, F1=0.7570

## Round 3

Labeled Pool Size: 327

- Epoch 1: Loss=0.3922, mIoU=0.5613, F1=0.6208
- Epoch 2: Loss=0.2015, mIoU=0.6104, F1=0.6891
- Epoch 3: Loss=0.1516, mIoU=0.6730, F1=0.7625
- Epoch 4: Loss=0.1330, mIoU=0.6342, F1=0.7184
- Epoch 5: Loss=0.1251, mIoU=0.6553, F1=0.7430
- Epoch 6: Loss=0.1125, mIoU=0.6933, F1=0.7833
- Epoch 7: Loss=0.1042, mIoU=0.6514, F1=0.7384
- Epoch 8: Loss=0.1031, mIoU=0.6882, F1=0.7781
- Epoch 9: Loss=0.0985, mIoU=0.6698, F1=0.7587
- Epoch 10: Loss=0.0958, mIoU=0.7224, F1=0.8116

当前轮次最佳结果: Round=3, Labeled=327, mIoU=0.7224, F1=0.8116

## Round 4

Labeled Pool Size: 415

- Epoch 1: Loss=0.3723, mIoU=0.6190, F1=0.7011
- Epoch 2: Loss=0.1906, mIoU=0.5892, F1=0.6608
- Epoch 3: Loss=0.1530, mIoU=0.6868, F1=0.7769
- Epoch 4: Loss=0.1364, mIoU=0.6859, F1=0.7757
- Epoch 5: Loss=0.1240, mIoU=0.6902, F1=0.7801
- Epoch 6: Loss=0.1170, mIoU=0.6790, F1=0.7685
- Epoch 7: Loss=0.1111, mIoU=0.6910, F1=0.7809
- Epoch 8: Loss=0.1058, mIoU=0.6732, F1=0.7624
- Epoch 9: Loss=0.1019, mIoU=0.7148, F1=0.8044
- Epoch 10: Loss=0.0968, mIoU=0.7266, F1=0.8153

当前轮次最佳结果: Round=4, Labeled=415, mIoU=0.7266, F1=0.8153

## Round 5

Labeled Pool Size: 503

- Epoch 1: Loss=0.3403, mIoU=0.5876, F1=0.6587
- Epoch 2: Loss=0.1728, mIoU=0.6615, F1=0.7499
- Epoch 3: Loss=0.1431, mIoU=0.6870, F1=0.7770
- Epoch 4: Loss=0.1263, mIoU=0.6999, F1=0.7898
- Epoch 5: Loss=0.1185, mIoU=0.7158, F1=0.8052
- Epoch 6: Loss=0.1105, mIoU=0.7135, F1=0.8030
- Epoch 7: Loss=0.1079, mIoU=0.7111, F1=0.8010
- Epoch 8: Loss=0.1061, mIoU=0.7143, F1=0.8037
- Epoch 9: Loss=0.0996, mIoU=0.7227, F1=0.8116
- Epoch 10: Loss=0.0971, mIoU=0.7171, F1=0.8063

当前轮次最佳结果: Round=5, Labeled=503, mIoU=0.7227, F1=0.8116

## Round 6

Labeled Pool Size: 591

- Epoch 1: Loss=0.3838, mIoU=0.5950, F1=0.6687
- Epoch 2: Loss=0.1716, mIoU=0.6382, F1=0.7232
- Epoch 3: Loss=0.1378, mIoU=0.6596, F1=0.7475
- Epoch 4: Loss=0.1245, mIoU=0.7044, F1=0.7942
- Epoch 5: Loss=0.1163, mIoU=0.7057, F1=0.7956
- Epoch 6: Loss=0.1110, mIoU=0.7235, F1=0.8124
- Epoch 7: Loss=0.1066, mIoU=0.7245, F1=0.8132
- Epoch 8: Loss=0.0997, mIoU=0.7451, F1=0.8319
- Epoch 9: Loss=0.0956, mIoU=0.7193, F1=0.8083
- Epoch 10: Loss=0.0940, mIoU=0.7375, F1=0.8251

当前轮次最佳结果: Round=6, Labeled=591, mIoU=0.7451, F1=0.8319

## Round 7

Labeled Pool Size: 679

- Epoch 1: Loss=0.3327, mIoU=0.6354, F1=0.7200
- Epoch 2: Loss=0.1541, mIoU=0.6779, F1=0.7674
- Epoch 3: Loss=0.1280, mIoU=0.6982, F1=0.7881
- Epoch 4: Loss=0.1146, mIoU=0.7323, F1=0.8209
- Epoch 5: Loss=0.1103, mIoU=0.7359, F1=0.8238
- Epoch 6: Loss=0.1050, mIoU=0.7318, F1=0.8200
- Epoch 7: Loss=0.0966, mIoU=0.7264, F1=0.8150
- Epoch 8: Loss=0.0943, mIoU=0.7360, F1=0.8237
- Epoch 9: Loss=0.0899, mIoU=0.7480, F1=0.8344
- Epoch 10: Loss=0.0876, mIoU=0.7339, F1=0.8218

当前轮次最佳结果: Round=7, Labeled=679, mIoU=0.7480, F1=0.8344

## Round 8

Labeled Pool Size: 767

- Epoch 1: Loss=0.3307, mIoU=0.5829, F1=0.6521
- Epoch 2: Loss=0.1523, mIoU=0.6694, F1=0.7584
- Epoch 3: Loss=0.1277, mIoU=0.6774, F1=0.7670
- Epoch 4: Loss=0.1141, mIoU=0.7075, F1=0.7973
- Epoch 5: Loss=0.1070, mIoU=0.7154, F1=0.8047
- Epoch 6: Loss=0.1022, mIoU=0.7289, F1=0.8173
- Epoch 7: Loss=0.0966, mIoU=0.7430, F1=0.8300
- Epoch 8: Loss=0.0921, mIoU=0.7407, F1=0.8280
- Epoch 9: Loss=0.0908, mIoU=0.7238, F1=0.8125
- Epoch 10: Loss=0.0867, mIoU=0.7336, F1=0.8215

当前轮次最佳结果: Round=8, Labeled=767, mIoU=0.7430, F1=0.8300

## Round 9

Labeled Pool Size: 855

- Epoch 1: Loss=0.2999, mIoU=0.6250, F1=0.7081
- Epoch 2: Loss=0.1393, mIoU=0.7078, F1=0.7976
- Epoch 3: Loss=0.1197, mIoU=0.7251, F1=0.8141
- Epoch 4: Loss=0.1101, mIoU=0.7033, F1=0.7930
- Epoch 5: Loss=0.1043, mIoU=0.7294, F1=0.8178
- Epoch 6: Loss=0.0990, mIoU=0.7149, F1=0.8042
- Epoch 7: Loss=0.0980, mIoU=0.7206, F1=0.8097
- Epoch 8: Loss=0.0931, mIoU=0.7199, F1=0.8090
- Epoch 9: Loss=0.0906, mIoU=0.7435, F1=0.8303
- Epoch 10: Loss=0.0869, mIoU=0.7487, F1=0.8351

当前轮次最佳结果: Round=9, Labeled=855, mIoU=0.7487, F1=0.8351

## Round 10

Labeled Pool Size: 943

- Epoch 1: Loss=0.2337, mIoU=0.6430, F1=0.7288
- Epoch 2: Loss=0.1229, mIoU=0.6751, F1=0.7644
- Epoch 3: Loss=0.1064, mIoU=0.6687, F1=0.7574
- Epoch 4: Loss=0.1004, mIoU=0.6645, F1=0.7529
- Epoch 5: Loss=0.0920, mIoU=0.7333, F1=0.8214
- Epoch 6: Loss=0.0876, mIoU=0.7340, F1=0.8219
- Epoch 7: Loss=0.0878, mIoU=0.7419, F1=0.8290
- Epoch 8: Loss=0.0834, mIoU=0.7480, F1=0.8343
- Epoch 9: Loss=0.0813, mIoU=0.7477, F1=0.8343
- Epoch 10: Loss=0.0770, mIoU=0.7516, F1=0.8375

当前轮次最佳结果: Round=10, Labeled=943, mIoU=0.7516, F1=0.8375

## Round 11

Labeled Pool Size: 1031

- Epoch 1: Loss=0.2041, mIoU=0.6823, F1=0.7723
- Epoch 2: Loss=0.1119, mIoU=0.6942, F1=0.7842
- Epoch 3: Loss=0.0999, mIoU=0.7217, F1=0.8107
- Epoch 4: Loss=0.0923, mIoU=0.7228, F1=0.8117
- Epoch 5: Loss=0.0859, mIoU=0.7513, F1=0.8375
- Epoch 6: Loss=0.0822, mIoU=0.7453, F1=0.8321
- Epoch 7: Loss=0.0803, mIoU=0.7606, F1=0.8455
- Epoch 8: Loss=0.0756, mIoU=0.7162, F1=0.8054
- Epoch 9: Loss=0.0751, mIoU=0.7588, F1=0.8436
- Epoch 10: Loss=0.0716, mIoU=0.7260, F1=0.8147

当前轮次最佳结果: Round=11, Labeled=1031, mIoU=0.7606, F1=0.8455

## Round 12

Labeled Pool Size: 1119

- Epoch 1: Loss=0.1978, mIoU=0.6004, F1=0.6758
- Epoch 2: Loss=0.1074, mIoU=0.6613, F1=0.7495
- Epoch 3: Loss=0.0935, mIoU=0.7285, F1=0.8170
- Epoch 4: Loss=0.0864, mIoU=0.7307, F1=0.8190
- Epoch 5: Loss=0.0806, mIoU=0.6993, F1=0.7890
- Epoch 6: Loss=0.0774, mIoU=0.7433, F1=0.8303
- Epoch 7: Loss=0.0732, mIoU=0.7437, F1=0.8307
- Epoch 8: Loss=0.0710, mIoU=0.7329, F1=0.8209
- Epoch 9: Loss=0.0681, mIoU=0.7403, F1=0.8275
- Epoch 10: Loss=0.0648, mIoU=0.7502, F1=0.8362

当前轮次最佳结果: Round=12, Labeled=1119, mIoU=0.7502, F1=0.8362

## Round 13

Labeled Pool Size: 1207

- Epoch 1: Loss=0.2363, mIoU=0.6154, F1=0.6953
- Epoch 2: Loss=0.1062, mIoU=0.6669, F1=0.7554
- Epoch 3: Loss=0.0893, mIoU=0.7199, F1=0.8090
- Epoch 4: Loss=0.0809, mIoU=0.7231, F1=0.8120
- Epoch 5: Loss=0.0750, mIoU=0.7448, F1=0.8316
- Epoch 6: Loss=0.0737, mIoU=0.6922, F1=0.7819
- Epoch 7: Loss=0.0726, mIoU=0.7466, F1=0.8339
- Epoch 8: Loss=0.0674, mIoU=0.7435, F1=0.8303
- Epoch 9: Loss=0.0640, mIoU=0.7503, F1=0.8362
- Epoch 10: Loss=0.0635, mIoU=0.7503, F1=0.8362

当前轮次最佳结果: Round=13, Labeled=1207, mIoU=0.7503, F1=0.8362

## Round 14

Labeled Pool Size: 1295

- Epoch 1: Loss=0.1678, mIoU=0.6553, F1=0.7430
- Epoch 2: Loss=0.0949, mIoU=0.7272, F1=0.8165
- Epoch 3: Loss=0.0825, mIoU=0.7012, F1=0.7910
- Epoch 4: Loss=0.0763, mIoU=0.7141, F1=0.8036
- Epoch 5: Loss=0.0727, mIoU=0.7318, F1=0.8199
- Epoch 6: Loss=0.0693, mIoU=0.7572, F1=0.8425
- Epoch 7: Loss=0.0652, mIoU=0.7524, F1=0.8382
- Epoch 8: Loss=0.0624, mIoU=0.7528, F1=0.8385
- Epoch 9: Loss=0.0615, mIoU=0.7475, F1=0.8346
- Epoch 10: Loss=0.0600, mIoU=0.7610, F1=0.8454

当前轮次最佳结果: Round=14, Labeled=1295, mIoU=0.7610, F1=0.8454

## Round 15

Labeled Pool Size: 1383

- Epoch 1: Loss=0.2064, mIoU=0.6423, F1=0.7281
- Epoch 2: Loss=0.0945, mIoU=0.7245, F1=0.8135
- Epoch 3: Loss=0.0808, mIoU=0.7053, F1=0.7950
- Epoch 4: Loss=0.0757, mIoU=0.7245, F1=0.8132
- Epoch 5: Loss=0.0712, mIoU=0.7517, F1=0.8377
- Epoch 6: Loss=0.0684, mIoU=0.7527, F1=0.8384
- Epoch 7: Loss=0.0653, mIoU=0.7573, F1=0.8424
- Epoch 8: Loss=0.0619, mIoU=0.7492, F1=0.8353
- Epoch 9: Loss=0.0600, mIoU=0.7368, F1=0.8243
- Epoch 10: Loss=0.0600, mIoU=0.7179, F1=0.8069

当前轮次最佳结果: Round=15, Labeled=1383, mIoU=0.7573, F1=0.8424


## 实验汇总

预算历史: [151, 239, 327, 415, 503, 591, 679, 767, 855, 943, 1031, 1119, 1207, 1295, 1383]
ALC: 0.6638
最终 mIoU: 0.7573
最终 F1: 0.8424
