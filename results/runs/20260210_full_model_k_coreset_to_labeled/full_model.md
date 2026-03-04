# 实验日志

实验名称: full_model
描述: 完整模型（正式方案）：coreset-to-labeled K + 自适应阈值 rollback + 风险驱动 λ 闭环控制（不调整 query_size）
开始时间: 2026-02-13T01:55:25.077518

## Round 1

Labeled Pool Size: 151

- Epoch 1: Loss=0.6116, mIoU=0.4668, F1=0.5083
- Epoch 2: Loss=0.3456, mIoU=0.5006, F1=0.5214
- Epoch 3: Loss=0.2158, mIoU=0.5141, F1=0.5440
- Epoch 4: Loss=0.1491, mIoU=0.5270, F1=0.5665
- Epoch 5: Loss=0.1155, mIoU=0.5290, F1=0.5695
- Epoch 6: Loss=0.0906, mIoU=0.6144, F1=0.6960
- Epoch 7: Loss=0.0799, mIoU=0.6031, F1=0.6813
- Epoch 8: Loss=0.0702, mIoU=0.5650, F1=0.6261
- Epoch 9: Loss=0.0646, mIoU=0.5865, F1=0.6573
- Epoch 10: Loss=0.0627, mIoU=0.5976, F1=0.6723

当前轮次最佳结果: Round=1, Labeled=151, mIoU=0.6144, F1=0.6960

## Round 2

Labeled Pool Size: 239

- Epoch 1: Loss=0.4952, mIoU=0.5377, F1=0.5850
- Epoch 2: Loss=0.2633, mIoU=0.5920, F1=0.6654
- Epoch 3: Loss=0.1892, mIoU=0.5990, F1=0.6742
- Epoch 4: Loss=0.1562, mIoU=0.6419, F1=0.7277
- Epoch 5: Loss=0.1446, mIoU=0.6559, F1=0.7438
- Epoch 6: Loss=0.1316, mIoU=0.6764, F1=0.7661
- Epoch 7: Loss=0.1156, mIoU=0.6884, F1=0.7786
- Epoch 8: Loss=0.1119, mIoU=0.6157, F1=0.6956
- Epoch 9: Loss=0.1098, mIoU=0.6860, F1=0.7759
- Epoch 10: Loss=0.1003, mIoU=0.7102, F1=0.8002

当前轮次最佳结果: Round=2, Labeled=239, mIoU=0.7102, F1=0.8002

## Round 3

Labeled Pool Size: 327

- Epoch 1: Loss=0.3174, mIoU=0.6092, F1=0.6882
- Epoch 2: Loss=0.1862, mIoU=0.6312, F1=0.7149
- Epoch 3: Loss=0.1517, mIoU=0.6411, F1=0.7267
- Epoch 4: Loss=0.1381, mIoU=0.6883, F1=0.7783
- Epoch 5: Loss=0.1287, mIoU=0.6899, F1=0.7802
- Epoch 6: Loss=0.1171, mIoU=0.7107, F1=0.8009
- Epoch 7: Loss=0.1142, mIoU=0.6945, F1=0.7845
- Epoch 8: Loss=0.1078, mIoU=0.7003, F1=0.7902
- Epoch 9: Loss=0.1044, mIoU=0.6194, F1=0.7002
- Epoch 10: Loss=0.1029, mIoU=0.6895, F1=0.7793

当前轮次最佳结果: Round=3, Labeled=327, mIoU=0.7107, F1=0.8009

## Round 4

Labeled Pool Size: 415

- Epoch 1: Loss=0.3875, mIoU=0.5987, F1=0.6745
- Epoch 2: Loss=0.1980, mIoU=0.6630, F1=0.7515
- Epoch 3: Loss=0.1555, mIoU=0.6532, F1=0.7405
- Epoch 4: Loss=0.1443, mIoU=0.6916, F1=0.7817
- Epoch 5: Loss=0.1289, mIoU=0.6595, F1=0.7474
- Epoch 6: Loss=0.1234, mIoU=0.6978, F1=0.7878
- Epoch 7: Loss=0.1158, mIoU=0.6993, F1=0.7891
- Epoch 8: Loss=0.1134, mIoU=0.7305, F1=0.8188
- Epoch 9: Loss=0.1098, mIoU=0.7430, F1=0.8302
- Epoch 10: Loss=0.1055, mIoU=0.7398, F1=0.8272

当前轮次最佳结果: Round=4, Labeled=415, mIoU=0.7430, F1=0.8302

## Round 5

Labeled Pool Size: 503

- Epoch 1: Loss=0.3463, mIoU=0.6569, F1=0.7457
- Epoch 2: Loss=0.1780, mIoU=0.6397, F1=0.7253
- Epoch 3: Loss=0.1485, mIoU=0.6904, F1=0.7803
- Epoch 4: Loss=0.1345, mIoU=0.7320, F1=0.8206
- Epoch 5: Loss=0.1274, mIoU=0.7146, F1=0.8040
- Epoch 6: Loss=0.1189, mIoU=0.7350, F1=0.8235
- Epoch 7: Loss=0.1151, mIoU=0.6823, F1=0.7718
- Epoch 8: Loss=0.1093, mIoU=0.7016, F1=0.7914
- Epoch 9: Loss=0.1038, mIoU=0.7329, F1=0.8212
- Epoch 10: Loss=0.1036, mIoU=0.7244, F1=0.8131

当前轮次最佳结果: Round=5, Labeled=503, mIoU=0.7350, F1=0.8235

## Round 6

Labeled Pool Size: 591

- Epoch 1: Loss=0.3121, mIoU=0.6135, F1=0.6931
- Epoch 2: Loss=0.1640, mIoU=0.6819, F1=0.7718
- Epoch 3: Loss=0.1412, mIoU=0.6985, F1=0.7885
- Epoch 4: Loss=0.1295, mIoU=0.6938, F1=0.7836
- Epoch 5: Loss=0.1195, mIoU=0.6848, F1=0.7745
- Epoch 6: Loss=0.1127, mIoU=0.7140, F1=0.8034
- Epoch 7: Loss=0.1086, mIoU=0.6833, F1=0.7729
- Epoch 8: Loss=0.1057, mIoU=0.7342, F1=0.8221
- Epoch 9: Loss=0.0997, mIoU=0.7482, F1=0.8348
- Epoch 10: Loss=0.0973, mIoU=0.7550, F1=0.8406

当前轮次最佳结果: Round=6, Labeled=591, mIoU=0.7550, F1=0.8406

## Round 7

Labeled Pool Size: 679

- Epoch 1: Loss=0.3526, mIoU=0.6747, F1=0.7647
- Epoch 2: Loss=0.1591, mIoU=0.6633, F1=0.7519
- Epoch 3: Loss=0.1338, mIoU=0.6963, F1=0.7863
- Epoch 4: Loss=0.1212, mIoU=0.7149, F1=0.8043
- Epoch 5: Loss=0.1134, mIoU=0.7240, F1=0.8127
- Epoch 6: Loss=0.1075, mIoU=0.7382, F1=0.8258
- Epoch 7: Loss=0.1035, mIoU=0.7484, F1=0.8349
- Epoch 8: Loss=0.0987, mIoU=0.7188, F1=0.8078
- Epoch 9: Loss=0.0954, mIoU=0.7397, F1=0.8270
- Epoch 10: Loss=0.0904, mIoU=0.7206, F1=0.8096

当前轮次最佳结果: Round=7, Labeled=679, mIoU=0.7484, F1=0.8349

## Round 8

Labeled Pool Size: 767

- Epoch 1: Loss=0.2913, mIoU=0.6488, F1=0.7360
- Epoch 2: Loss=0.1432, mIoU=0.6606, F1=0.7491
- Epoch 3: Loss=0.1221, mIoU=0.7150, F1=0.8046
- Epoch 4: Loss=0.1129, mIoU=0.7365, F1=0.8246
- Epoch 5: Loss=0.1032, mIoU=0.7397, F1=0.8272
- Epoch 6: Loss=0.0979, mIoU=0.7074, F1=0.7971
- Epoch 7: Loss=0.0937, mIoU=0.6932, F1=0.7829
- Epoch 8: Loss=0.0884, mIoU=0.7548, F1=0.8404
- Epoch 9: Loss=0.0866, mIoU=0.7239, F1=0.8126
- Epoch 10: Loss=0.0846, mIoU=0.7530, F1=0.8386

当前轮次最佳结果: Round=8, Labeled=767, mIoU=0.7548, F1=0.8404

## Round 9

Labeled Pool Size: 855

- Epoch 1: Loss=0.3446, mIoU=0.6522, F1=0.7400
- Epoch 2: Loss=0.1416, mIoU=0.6849, F1=0.7747
- Epoch 3: Loss=0.1193, mIoU=0.6936, F1=0.7835
- Epoch 4: Loss=0.1105, mIoU=0.7011, F1=0.7910
- Epoch 5: Loss=0.1034, mIoU=0.7191, F1=0.8083
- Epoch 6: Loss=0.1012, mIoU=0.7372, F1=0.8248
- Epoch 7: Loss=0.0958, mIoU=0.6988, F1=0.7885
- Epoch 8: Loss=0.0908, mIoU=0.7450, F1=0.8317
- Epoch 9: Loss=0.0871, mIoU=0.7072, F1=0.7968
- Epoch 10: Loss=0.0868, mIoU=0.7354, F1=0.8231

当前轮次最佳结果: Round=9, Labeled=855, mIoU=0.7450, F1=0.8317

## Round 10

Labeled Pool Size: 943

- Epoch 1: Loss=0.3095, mIoU=0.6692, F1=0.7585
- Epoch 2: Loss=0.1264, mIoU=0.7188, F1=0.8085
- Epoch 3: Loss=0.1127, mIoU=0.7036, F1=0.7934
- Epoch 4: Loss=0.1001, mIoU=0.7241, F1=0.8129
- Epoch 5: Loss=0.0948, mIoU=0.7046, F1=0.7944
- Epoch 6: Loss=0.0917, mIoU=0.7510, F1=0.8372
- Epoch 7: Loss=0.0855, mIoU=0.7527, F1=0.8387
- Epoch 8: Loss=0.0831, mIoU=0.7555, F1=0.8409
- Epoch 9: Loss=0.0797, mIoU=0.7543, F1=0.8398
- Epoch 10: Loss=0.0765, mIoU=0.7464, F1=0.8332

当前轮次最佳结果: Round=10, Labeled=943, mIoU=0.7555, F1=0.8409

## Round 11

Labeled Pool Size: 1031

- Epoch 1: Loss=0.2124, mIoU=0.6587, F1=0.7467
- Epoch 2: Loss=0.1146, mIoU=0.7224, F1=0.8118
- Epoch 3: Loss=0.0990, mIoU=0.7227, F1=0.8117
- Epoch 4: Loss=0.0929, mIoU=0.7348, F1=0.8227
- Epoch 5: Loss=0.0861, mIoU=0.7465, F1=0.8335
- Epoch 6: Loss=0.0850, mIoU=0.7230, F1=0.8118
- Epoch 7: Loss=0.0820, mIoU=0.7426, F1=0.8295
- Epoch 8: Loss=0.0770, mIoU=0.7412, F1=0.8283
- Epoch 9: Loss=0.0749, mIoU=0.7593, F1=0.8440
- Epoch 10: Loss=0.0710, mIoU=0.7584, F1=0.8433

当前轮次最佳结果: Round=11, Labeled=1031, mIoU=0.7593, F1=0.8440

## Round 12

Labeled Pool Size: 1119

- Epoch 1: Loss=0.1994, mIoU=0.6693, F1=0.7589
- Epoch 2: Loss=0.1096, mIoU=0.7251, F1=0.8144
- Epoch 3: Loss=0.0965, mIoU=0.7313, F1=0.8196
- Epoch 4: Loss=0.0893, mIoU=0.7437, F1=0.8306
- Epoch 5: Loss=0.0833, mIoU=0.7211, F1=0.8099
- Epoch 6: Loss=0.0797, mIoU=0.7471, F1=0.8336
- Epoch 7: Loss=0.0754, mIoU=0.7469, F1=0.8333
- Epoch 8: Loss=0.0733, mIoU=0.7569, F1=0.8420
- Epoch 9: Loss=0.0692, mIoU=0.7555, F1=0.8407
- Epoch 10: Loss=0.0666, mIoU=0.7677, F1=0.8511

当前轮次最佳结果: Round=12, Labeled=1119, mIoU=0.7677, F1=0.8511

## Round 13

Labeled Pool Size: 1207

- Epoch 1: Loss=0.1845, mIoU=0.6857, F1=0.7758
- Epoch 2: Loss=0.1044, mIoU=0.7001, F1=0.7900
- Epoch 3: Loss=0.0931, mIoU=0.7174, F1=0.8070
- Epoch 4: Loss=0.0846, mIoU=0.7011, F1=0.7908
- Epoch 5: Loss=0.0797, mIoU=0.7305, F1=0.8188
- Epoch 6: Loss=0.0763, mIoU=0.7322, F1=0.8202
- Epoch 7: Loss=0.0729, mIoU=0.7456, F1=0.8322
- Epoch 8: Loss=0.0700, mIoU=0.7405, F1=0.8278
- Epoch 9: Loss=0.0687, mIoU=0.7630, F1=0.8471
- Epoch 10: Loss=0.0654, mIoU=0.7668, F1=0.8504

当前轮次最佳结果: Round=13, Labeled=1207, mIoU=0.7668, F1=0.8504

## Round 14

Labeled Pool Size: 1295

- Epoch 1: Loss=0.2422, mIoU=0.6829, F1=0.7731
- Epoch 2: Loss=0.1061, mIoU=0.7237, F1=0.8127
- Epoch 3: Loss=0.0906, mIoU=0.6762, F1=0.7654
- Epoch 4: Loss=0.0839, mIoU=0.6759, F1=0.7650
- Epoch 5: Loss=0.0806, mIoU=0.7414, F1=0.8293
- Epoch 6: Loss=0.0776, mIoU=0.7350, F1=0.8227
- Epoch 7: Loss=0.0734, mIoU=0.7564, F1=0.8416
- Epoch 8: Loss=0.0692, mIoU=0.7382, F1=0.8256
- Epoch 9: Loss=0.0670, mIoU=0.7516, F1=0.8373
- Epoch 10: Loss=0.0649, mIoU=0.7646, F1=0.8484

当前轮次最佳结果: Round=14, Labeled=1295, mIoU=0.7646, F1=0.8484

## Round 15

Labeled Pool Size: 1383

- Epoch 1: Loss=0.1689, mIoU=0.6446, F1=0.7307
- Epoch 2: Loss=0.0938, mIoU=0.7089, F1=0.7986
- Epoch 3: Loss=0.0829, mIoU=0.7150, F1=0.8043
- Epoch 4: Loss=0.0772, mIoU=0.7018, F1=0.7916
- Epoch 5: Loss=0.0722, mIoU=0.7058, F1=0.7953
- Epoch 6: Loss=0.0683, mIoU=0.7510, F1=0.8371
- Epoch 7: Loss=0.0690, mIoU=0.7598, F1=0.8446
- Epoch 8: Loss=0.0635, mIoU=0.7499, F1=0.8360
- Epoch 9: Loss=0.0622, mIoU=0.7615, F1=0.8458
- Epoch 10: Loss=0.0619, mIoU=0.7531, F1=0.8386

当前轮次最佳结果: Round=15, Labeled=1383, mIoU=0.7615, F1=0.8458


## 实验汇总

预算历史: [151, 239, 327, 415, 503, 591, 679, 767, 855, 943, 1031, 1119, 1207, 1295, 1383]
ALC: 0.6709
最终 mIoU: 0.7615
最终 F1: 0.8458
