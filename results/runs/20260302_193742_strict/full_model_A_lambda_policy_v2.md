# 实验日志

实验名称: full_model_A_lambda_policy_v2
描述: 对照A（V2）：保持 warmup+风险闭环 λ policy，但放宽上调条件并启用EMA/冷却，避免长期卡在λ下限；不调整query_size/epochs
开始时间: 2026-03-02T19:37:45.771042

## Round 1

Labeled Pool Size: 151

- Epoch 1: Loss=0.5375, mIoU=0.4900, F1=0.5017
- Epoch 2: Loss=0.3043, mIoU=0.5027, F1=0.5231
- Epoch 3: Loss=0.1942, mIoU=0.5489, F1=0.6028
- Epoch 4: Loss=0.1394, mIoU=0.5439, F1=0.5942
- Epoch 5: Loss=0.1077, mIoU=0.5693, F1=0.6331
- Epoch 6: Loss=0.0874, mIoU=0.6308, F1=0.7159
- Epoch 7: Loss=0.0742, mIoU=0.6128, F1=0.6925
- Epoch 8: Loss=0.0665, mIoU=0.6239, F1=0.7064
- Epoch 9: Loss=0.0597, mIoU=0.6419, F1=0.7280
- Epoch 10: Loss=0.0560, mIoU=0.6450, F1=0.7317

当前轮次最佳结果: Round=1, Labeled=151, mIoU=0.6450, F1=0.7317

## Round 2

Labeled Pool Size: 239

- Epoch 1: Loss=0.3382, mIoU=0.4980, F1=0.5138
- Epoch 2: Loss=0.1933, mIoU=0.5858, F1=0.6563
- Epoch 3: Loss=0.1468, mIoU=0.6303, F1=0.7141
- Epoch 4: Loss=0.1296, mIoU=0.5906, F1=0.6630
- Epoch 5: Loss=0.1216, mIoU=0.6520, F1=0.7395
- Epoch 6: Loss=0.1079, mIoU=0.6645, F1=0.7532
- Epoch 7: Loss=0.1047, mIoU=0.6644, F1=0.7529
- Epoch 8: Loss=0.0944, mIoU=0.6849, F1=0.7751
- Epoch 9: Loss=0.0925, mIoU=0.6641, F1=0.7527
- Epoch 10: Loss=0.0855, mIoU=0.6931, F1=0.7836

当前轮次最佳结果: Round=2, Labeled=239, mIoU=0.6931, F1=0.7836

## Round 3

Labeled Pool Size: 327

- Epoch 1: Loss=0.3901, mIoU=0.5362, F1=0.5812
- Epoch 2: Loss=0.1899, mIoU=0.6249, F1=0.7076
- Epoch 3: Loss=0.1468, mIoU=0.6352, F1=0.7199
- Epoch 4: Loss=0.1302, mIoU=0.6488, F1=0.7357
- Epoch 5: Loss=0.1166, mIoU=0.6341, F1=0.7187
- Epoch 6: Loss=0.1081, mIoU=0.6712, F1=0.7603
- Epoch 7: Loss=0.1040, mIoU=0.6969, F1=0.7871
- Epoch 8: Loss=0.1043, mIoU=0.6951, F1=0.7851
- Epoch 9: Loss=0.0928, mIoU=0.6805, F1=0.7701
- Epoch 10: Loss=0.1009, mIoU=0.7076, F1=0.7975

当前轮次最佳结果: Round=3, Labeled=327, mIoU=0.7076, F1=0.7975

## Round 4

Labeled Pool Size: 415

- Epoch 1: Loss=0.3755, mIoU=0.6114, F1=0.6909
- Epoch 2: Loss=0.1870, mIoU=0.6239, F1=0.7059
- Epoch 3: Loss=0.1495, mIoU=0.6727, F1=0.7620
- Epoch 4: Loss=0.1305, mIoU=0.6903, F1=0.7803
- Epoch 5: Loss=0.1192, mIoU=0.7000, F1=0.7899
- Epoch 6: Loss=0.1099, mIoU=0.6934, F1=0.7833
- Epoch 7: Loss=0.1049, mIoU=0.7024, F1=0.7923
- Epoch 8: Loss=0.1041, mIoU=0.7023, F1=0.7921
- Epoch 9: Loss=0.0972, mIoU=0.7019, F1=0.7917
- Epoch 10: Loss=0.0980, mIoU=0.7299, F1=0.8184

当前轮次最佳结果: Round=4, Labeled=415, mIoU=0.7299, F1=0.8184

## Round 5

Labeled Pool Size: 503

- Epoch 1: Loss=0.3424, mIoU=0.6000, F1=0.6755
- Epoch 2: Loss=0.1718, mIoU=0.6105, F1=0.6891
- Epoch 3: Loss=0.1393, mIoU=0.6916, F1=0.7816
- Epoch 4: Loss=0.1290, mIoU=0.6464, F1=0.7328
- Epoch 5: Loss=0.1201, mIoU=0.6600, F1=0.7480
- Epoch 6: Loss=0.1146, mIoU=0.6938, F1=0.7837
- Epoch 7: Loss=0.1074, mIoU=0.7141, F1=0.8035
- Epoch 8: Loss=0.1040, mIoU=0.7166, F1=0.8060
- Epoch 9: Loss=0.0980, mIoU=0.7338, F1=0.8219
- Epoch 10: Loss=0.0947, mIoU=0.7158, F1=0.8051

当前轮次最佳结果: Round=5, Labeled=503, mIoU=0.7338, F1=0.8219

## Round 6

Labeled Pool Size: 591

- Epoch 1: Loss=0.3796, mIoU=0.6403, F1=0.7261
- Epoch 2: Loss=0.1709, mIoU=0.6909, F1=0.7810
- Epoch 3: Loss=0.1420, mIoU=0.6960, F1=0.7860
- Epoch 4: Loss=0.1259, mIoU=0.7077, F1=0.7974
- Epoch 5: Loss=0.1161, mIoU=0.7183, F1=0.8074
- Epoch 6: Loss=0.1115, mIoU=0.7012, F1=0.7909
- Epoch 7: Loss=0.1068, mIoU=0.7018, F1=0.7915
- Epoch 8: Loss=0.1002, mIoU=0.7304, F1=0.8186
- Epoch 9: Loss=0.0998, mIoU=0.7386, F1=0.8262
- Epoch 10: Loss=0.0948, mIoU=0.7466, F1=0.8332

当前轮次最佳结果: Round=6, Labeled=591, mIoU=0.7466, F1=0.8332

## Round 7

Labeled Pool Size: 679

- Epoch 1: Loss=0.3289, mIoU=0.6480, F1=0.7349
- Epoch 2: Loss=0.1576, mIoU=0.6760, F1=0.7655
- Epoch 3: Loss=0.1308, mIoU=0.7182, F1=0.8078
- Epoch 4: Loss=0.1172, mIoU=0.7116, F1=0.8012
- Epoch 5: Loss=0.1102, mIoU=0.7434, F1=0.8306
- Epoch 6: Loss=0.1039, mIoU=0.7171, F1=0.8063
- Epoch 7: Loss=0.0990, mIoU=0.7154, F1=0.8047
- Epoch 8: Loss=0.0991, mIoU=0.7346, F1=0.8225
- Epoch 9: Loss=0.0944, mIoU=0.7346, F1=0.8225
- Epoch 10: Loss=0.0910, mIoU=0.7368, F1=0.8246

当前轮次最佳结果: Round=7, Labeled=679, mIoU=0.7434, F1=0.8306

## Round 8

Labeled Pool Size: 767

- Epoch 1: Loss=0.3368, mIoU=0.6232, F1=0.7052
- Epoch 2: Loss=0.1488, mIoU=0.6577, F1=0.7455
- Epoch 3: Loss=0.1248, mIoU=0.6960, F1=0.7860
- Epoch 4: Loss=0.1153, mIoU=0.7192, F1=0.8087
- Epoch 5: Loss=0.1145, mIoU=0.7122, F1=0.8018
- Epoch 6: Loss=0.1060, mIoU=0.7467, F1=0.8334
- Epoch 7: Loss=0.0980, mIoU=0.7446, F1=0.8315
- Epoch 8: Loss=0.0963, mIoU=0.7464, F1=0.8331
- Epoch 9: Loss=0.0933, mIoU=0.7313, F1=0.8195
- Epoch 10: Loss=0.0892, mIoU=0.7433, F1=0.8302

当前轮次最佳结果: Round=8, Labeled=767, mIoU=0.7467, F1=0.8334

## Round 9

Labeled Pool Size: 855

- Epoch 1: Loss=0.3019, mIoU=0.6240, F1=0.7063
- Epoch 2: Loss=0.1403, mIoU=0.6845, F1=0.7743
- Epoch 3: Loss=0.1183, mIoU=0.6904, F1=0.7802
- Epoch 4: Loss=0.1100, mIoU=0.6767, F1=0.7660
- Epoch 5: Loss=0.1029, mIoU=0.7241, F1=0.8129
- Epoch 6: Loss=0.0972, mIoU=0.7297, F1=0.8180
- Epoch 7: Loss=0.0955, mIoU=0.7338, F1=0.8221
- Epoch 8: Loss=0.0939, mIoU=0.7467, F1=0.8333
- Epoch 9: Loss=0.0886, mIoU=0.7508, F1=0.8368
- Epoch 10: Loss=0.0865, mIoU=0.7467, F1=0.8332

当前轮次最佳结果: Round=9, Labeled=855, mIoU=0.7508, F1=0.8368

## Round 10

Labeled Pool Size: 943

- Epoch 1: Loss=0.2340, mIoU=0.6592, F1=0.7475
- Epoch 2: Loss=0.1249, mIoU=0.6923, F1=0.7822
- Epoch 3: Loss=0.1093, mIoU=0.7370, F1=0.8249
- Epoch 4: Loss=0.1005, mIoU=0.7108, F1=0.8004
- Epoch 5: Loss=0.0958, mIoU=0.7341, F1=0.8220
- Epoch 6: Loss=0.0906, mIoU=0.7297, F1=0.8180
- Epoch 7: Loss=0.0849, mIoU=0.7520, F1=0.8380
- Epoch 8: Loss=0.0825, mIoU=0.7529, F1=0.8387
- Epoch 9: Loss=0.0800, mIoU=0.7351, F1=0.8229
- Epoch 10: Loss=0.0773, mIoU=0.7421, F1=0.8291

当前轮次最佳结果: Round=10, Labeled=943, mIoU=0.7529, F1=0.8387

## Round 11

Labeled Pool Size: 1031

- Epoch 1: Loss=0.2089, mIoU=0.6902, F1=0.7804
- Epoch 2: Loss=0.1157, mIoU=0.7008, F1=0.7907
- Epoch 3: Loss=0.1014, mIoU=0.7044, F1=0.7942
- Epoch 4: Loss=0.0928, mIoU=0.7192, F1=0.8083
- Epoch 5: Loss=0.0891, mIoU=0.7395, F1=0.8269
- Epoch 6: Loss=0.0886, mIoU=0.7450, F1=0.8317
- Epoch 7: Loss=0.0818, mIoU=0.7375, F1=0.8250
- Epoch 8: Loss=0.0778, mIoU=0.7272, F1=0.8156
- Epoch 9: Loss=0.0752, mIoU=0.7566, F1=0.8418
- Epoch 10: Loss=0.0750, mIoU=0.7571, F1=0.8423

当前轮次最佳结果: Round=11, Labeled=1031, mIoU=0.7571, F1=0.8423

## Round 12

Labeled Pool Size: 1119

- Epoch 1: Loss=0.2009, mIoU=0.6611, F1=0.7494
- Epoch 2: Loss=0.1098, mIoU=0.7088, F1=0.7985
- Epoch 3: Loss=0.0941, mIoU=0.6891, F1=0.7811
- Epoch 4: Loss=0.0894, mIoU=0.7367, F1=0.8247
- Epoch 5: Loss=0.0829, mIoU=0.7341, F1=0.8220
- Epoch 6: Loss=0.0787, mIoU=0.7463, F1=0.8332
- Epoch 7: Loss=0.0759, mIoU=0.7576, F1=0.8426
- Epoch 8: Loss=0.0750, mIoU=0.7470, F1=0.8335
- Epoch 9: Loss=0.0702, mIoU=0.7367, F1=0.8243
- Epoch 10: Loss=0.0682, mIoU=0.7642, F1=0.8484

当前轮次最佳结果: Round=12, Labeled=1119, mIoU=0.7642, F1=0.8484

## Round 13

Labeled Pool Size: 1207

- Epoch 1: Loss=0.2365, mIoU=0.6487, F1=0.7356
- Epoch 2: Loss=0.1089, mIoU=0.6923, F1=0.7822
- Epoch 3: Loss=0.0929, mIoU=0.6861, F1=0.7758
- Epoch 4: Loss=0.0881, mIoU=0.6826, F1=0.7721
- Epoch 5: Loss=0.0823, mIoU=0.7144, F1=0.8037
- Epoch 6: Loss=0.0779, mIoU=0.7282, F1=0.8166
- Epoch 7: Loss=0.0724, mIoU=0.7595, F1=0.8443
- Epoch 8: Loss=0.0709, mIoU=0.7559, F1=0.8413
- Epoch 9: Loss=0.0689, mIoU=0.7582, F1=0.8430
- Epoch 10: Loss=0.0650, mIoU=0.7571, F1=0.8421

当前轮次最佳结果: Round=13, Labeled=1207, mIoU=0.7595, F1=0.8443

## Round 14

Labeled Pool Size: 1295

- Epoch 1: Loss=0.1699, mIoU=0.6666, F1=0.7554
- Epoch 2: Loss=0.0973, mIoU=0.6974, F1=0.7874
- Epoch 3: Loss=0.0849, mIoU=0.7302, F1=0.8186
- Epoch 4: Loss=0.0771, mIoU=0.7087, F1=0.7983
- Epoch 5: Loss=0.0749, mIoU=0.7313, F1=0.8203
- Epoch 6: Loss=0.0721, mIoU=0.7572, F1=0.8423
- Epoch 7: Loss=0.0666, mIoU=0.7485, F1=0.8347
- Epoch 8: Loss=0.0668, mIoU=0.7536, F1=0.8391
- Epoch 9: Loss=0.0612, mIoU=0.7611, F1=0.8455
- Epoch 10: Loss=0.0604, mIoU=0.7409, F1=0.8279

当前轮次最佳结果: Round=14, Labeled=1295, mIoU=0.7611, F1=0.8455

## Round 15

Labeled Pool Size: 1383

- Epoch 1: Loss=0.2033, mIoU=0.6755, F1=0.7650
- Epoch 2: Loss=0.0952, mIoU=0.6909, F1=0.7807
- Epoch 3: Loss=0.0838, mIoU=0.7250, F1=0.8138
- Epoch 4: Loss=0.0785, mIoU=0.7319, F1=0.8201
- Epoch 5: Loss=0.0721, mIoU=0.7349, F1=0.8227
- Epoch 6: Loss=0.0696, mIoU=0.7352, F1=0.8230
- Epoch 7: Loss=0.0670, mIoU=0.7411, F1=0.8282
- Epoch 8: Loss=0.0629, mIoU=0.7499, F1=0.8360
- Epoch 9: Loss=0.0605, mIoU=0.7315, F1=0.8195
- Epoch 10: Loss=0.0594, mIoU=0.7613, F1=0.8457

当前轮次最佳结果: Round=15, Labeled=1383, mIoU=0.7613, F1=0.8457


## 实验汇总

预算历史: [151, 239, 327, 415, 503, 591, 679, 767, 855, 943, 1031, 1119, 1207, 1295, 1383]
ALC: 0.6678
最终 mIoU: 0.7613
最终 F1: 0.8457
