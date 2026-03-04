# 实验日志

实验名称: full_model_A_lambda_policy
描述: 对照A / AAL-SD (Full)：LLM 参与决策与解释；λ由 warmup+风险闭环策略生成并在工具层 clamp（启用CI阈值+AND严重判定+EMA/冷却/限步长；默认禁止 set_lambda；不调整query_size/epochs）
开始时间: 2026-03-03T02:20:16.197604

## Round 1

Labeled Pool Size: 151

- Epoch 1: Loss=0.5380, mIoU=0.4900, F1=0.5017
- Epoch 2: Loss=0.3066, mIoU=0.5053, F1=0.5281
- Epoch 3: Loss=0.1952, mIoU=0.5347, F1=0.5799
- Epoch 4: Loss=0.1413, mIoU=0.5394, F1=0.5871
- Epoch 5: Loss=0.1086, mIoU=0.5462, F1=0.5973
- Epoch 6: Loss=0.0878, mIoU=0.5973, F1=0.6732
- Epoch 7: Loss=0.0755, mIoU=0.5743, F1=0.6402
- Epoch 8: Loss=0.0671, mIoU=0.6037, F1=0.6811
- Epoch 9: Loss=0.0608, mIoU=0.6158, F1=0.6961
- Epoch 10: Loss=0.0552, mIoU=0.6305, F1=0.7142

当前轮次最佳结果: Round=1, Labeled=151, mIoU=0.6305, F1=0.7142

## Round 2

Labeled Pool Size: 239

- Epoch 1: Loss=0.3391, mIoU=0.4978, F1=0.5138
- Epoch 2: Loss=0.1982, mIoU=0.5554, F1=0.6120
- Epoch 3: Loss=0.1429, mIoU=0.6228, F1=0.7049
- Epoch 4: Loss=0.1239, mIoU=0.6241, F1=0.7064
- Epoch 5: Loss=0.1152, mIoU=0.6418, F1=0.7278
- Epoch 6: Loss=0.1039, mIoU=0.6817, F1=0.7723
- Epoch 7: Loss=0.0973, mIoU=0.6781, F1=0.7679
- Epoch 8: Loss=0.0933, mIoU=0.6794, F1=0.7692
- Epoch 9: Loss=0.0913, mIoU=0.6816, F1=0.7714
- Epoch 10: Loss=0.0826, mIoU=0.6988, F1=0.7891

当前轮次最佳结果: Round=2, Labeled=239, mIoU=0.6988, F1=0.7891

## Round 3

Labeled Pool Size: 327

- Epoch 1: Loss=0.3993, mIoU=0.5475, F1=0.5996
- Epoch 2: Loss=0.2025, mIoU=0.6187, F1=0.6997
- Epoch 3: Loss=0.1587, mIoU=0.6396, F1=0.7248
- Epoch 4: Loss=0.1365, mIoU=0.6671, F1=0.7559
- Epoch 5: Loss=0.1286, mIoU=0.6584, F1=0.7463
- Epoch 6: Loss=0.1177, mIoU=0.6813, F1=0.7710
- Epoch 7: Loss=0.1159, mIoU=0.6779, F1=0.7674
- Epoch 8: Loss=0.1201, mIoU=0.7083, F1=0.7983
- Epoch 9: Loss=0.1098, mIoU=0.7143, F1=0.8039
- Epoch 10: Loss=0.1034, mIoU=0.6777, F1=0.7671

当前轮次最佳结果: Round=3, Labeled=327, mIoU=0.7143, F1=0.8039

## Round 4

Labeled Pool Size: 415

- Epoch 1: Loss=0.3769, mIoU=0.5675, F1=0.6302
- Epoch 2: Loss=0.1951, mIoU=0.6107, F1=0.6894
- Epoch 3: Loss=0.1561, mIoU=0.6569, F1=0.7446
- Epoch 4: Loss=0.1402, mIoU=0.6785, F1=0.7680
- Epoch 5: Loss=0.1262, mIoU=0.7126, F1=0.8023
- Epoch 6: Loss=0.1207, mIoU=0.6982, F1=0.7881
- Epoch 7: Loss=0.1138, mIoU=0.7159, F1=0.8052
- Epoch 8: Loss=0.1115, mIoU=0.7176, F1=0.8069
- Epoch 9: Loss=0.1085, mIoU=0.7280, F1=0.8167
- Epoch 10: Loss=0.1028, mIoU=0.7219, F1=0.8109

当前轮次最佳结果: Round=4, Labeled=415, mIoU=0.7280, F1=0.8167


--- [Checkpoint] 续跑开始时间: 2026-03-03T09:50:49.628866 ---

## Round 5

Labeled Pool Size: 503

- Epoch 1: Loss=0.3364, mIoU=0.5769, F1=0.6436
- Epoch 2: Loss=0.1790, mIoU=0.6149, F1=0.6947
- Epoch 3: Loss=0.1505, mIoU=0.6874, F1=0.7772
- Epoch 4: Loss=0.1368, mIoU=0.6837, F1=0.7734
- Epoch 5: Loss=0.1273, mIoU=0.6821, F1=0.7716
- Epoch 6: Loss=0.1203, mIoU=0.7096, F1=0.7992
- Epoch 7: Loss=0.1163, mIoU=0.7198, F1=0.8090
- Epoch 8: Loss=0.1113, mIoU=0.7318, F1=0.8202
- Epoch 9: Loss=0.1070, mIoU=0.7370, F1=0.8248
- Epoch 10: Loss=0.1039, mIoU=0.7432, F1=0.8302

当前轮次最佳结果: Round=5, Labeled=503, mIoU=0.7432, F1=0.8302


--- [Checkpoint] 续跑开始时间: 2026-03-03T10:01:17.685142 ---

## Round 5

Labeled Pool Size: 503

- Epoch 1: Loss=0.3371, mIoU=0.6137, F1=0.6934
- Epoch 2: Loss=0.1771, mIoU=0.6073, F1=0.6849
- Epoch 3: Loss=0.1497, mIoU=0.6892, F1=0.7791
- Epoch 4: Loss=0.1343, mIoU=0.6832, F1=0.7728
- Epoch 5: Loss=0.1307, mIoU=0.6799, F1=0.7695

--- [Checkpoint] 续跑开始时间: 2026-03-03T10:06:46.707015 ---

## Round 5

Labeled Pool Size: 503

- Epoch 1: Loss=0.3370, mIoU=0.6306, F1=0.7145

--- [Checkpoint] 续跑开始时间: 2026-03-03T10:07:38.707128 ---

## Round 5

Labeled Pool Size: 503


--- [Checkpoint] 续跑开始时间: 2026-03-03T10:08:31.673258 ---

## Round 5

Labeled Pool Size: 503

- Epoch 1: Loss=0.3366, mIoU=0.6010, F1=0.6768
- Epoch 2: Loss=0.1771, mIoU=0.6454, F1=0.7315
- Epoch 3: Loss=0.1484, mIoU=0.6824, F1=0.7721
- Epoch 4: Loss=0.1348, mIoU=0.7066, F1=0.7964
- Epoch 5: Loss=0.1257, mIoU=0.6720, F1=0.7610
- Epoch 6: Loss=0.1195, mIoU=0.7082, F1=0.7979
- Epoch 7: Loss=0.1153, mIoU=0.7293, F1=0.8178
- Epoch 8: Loss=0.1099, mIoU=0.7252, F1=0.8141
- Epoch 9: Loss=0.1047, mIoU=0.7299, F1=0.8185
- Epoch 10: Loss=0.1038, mIoU=0.7203, F1=0.8093

当前轮次最佳结果: Round=5, Labeled=503, mIoU=0.7299, F1=0.8185


--- [Checkpoint] 续跑开始时间: 2026-03-03T11:27:26.165357 ---

## Round 5

Labeled Pool Size: 503

- Epoch 1: Loss=0.3369, mIoU=0.6119, F1=0.6911
- Epoch 2: Loss=0.1790, mIoU=0.6191, F1=0.6999
- Epoch 3: Loss=0.1489, mIoU=0.6803, F1=0.7699
- Epoch 4: Loss=0.1346, mIoU=0.6850, F1=0.7747
- Epoch 5: Loss=0.1293, mIoU=0.6665, F1=0.7551
- Epoch 6: Loss=0.1191, mIoU=0.7281, F1=0.8169
- Epoch 7: Loss=0.1142, mIoU=0.7259, F1=0.8147
- Epoch 8: Loss=0.1101, mIoU=0.7280, F1=0.8168
- Epoch 9: Loss=0.1062, mIoU=0.7267, F1=0.8155
- Epoch 10: Loss=0.1035, mIoU=0.7373, F1=0.8250

当前轮次最佳结果: Round=5, Labeled=503, mIoU=0.7373, F1=0.8250

## Round 6

Labeled Pool Size: 591

- Epoch 1: Loss=0.3810, mIoU=0.6269, F1=0.7099
- Epoch 2: Loss=0.1754, mIoU=0.6850, F1=0.7749
- Epoch 3: Loss=0.1420, mIoU=0.6676, F1=0.7563
- Epoch 4: Loss=0.1303, mIoU=0.7170, F1=0.8064
- Epoch 5: Loss=0.1178, mIoU=0.7331, F1=0.8212
- Epoch 6: Loss=0.1123, mIoU=0.7272, F1=0.8158
- Epoch 7: Loss=0.1047, mIoU=0.7297, F1=0.8181
- Epoch 8: Loss=0.1043, mIoU=0.7367, F1=0.8253
- Epoch 9: Loss=0.0997, mIoU=0.7435, F1=0.8306
- Epoch 10: Loss=0.0950, mIoU=0.7057, F1=0.7953

当前轮次最佳结果: Round=6, Labeled=591, mIoU=0.7435, F1=0.8306

## Round 7

Labeled Pool Size: 679

- Epoch 1: Loss=0.3339, mIoU=0.6425, F1=0.7287
- Epoch 2: Loss=0.1599, mIoU=0.6482, F1=0.7348
- Epoch 3: Loss=0.1314, mIoU=0.7033, F1=0.7933
- Epoch 4: Loss=0.1180, mIoU=0.7218, F1=0.8108
- Epoch 5: Loss=0.1118, mIoU=0.7042, F1=0.7939
- Epoch 6: Loss=0.1057, mIoU=0.7010, F1=0.7908
- Epoch 7: Loss=0.1000, mIoU=0.7438, F1=0.8307
- Epoch 8: Loss=0.0972, mIoU=0.7205, F1=0.8095
- Epoch 9: Loss=0.0930, mIoU=0.7350, F1=0.8228
- Epoch 10: Loss=0.0887, mIoU=0.7155, F1=0.8047

当前轮次最佳结果: Round=7, Labeled=679, mIoU=0.7438, F1=0.8307

## Round 8

Labeled Pool Size: 767

- Epoch 1: Loss=0.3528, mIoU=0.6503, F1=0.7385
- Epoch 2: Loss=0.1504, mIoU=0.6705, F1=0.7594
- Epoch 3: Loss=0.1287, mIoU=0.7036, F1=0.7935
- Epoch 4: Loss=0.1163, mIoU=0.6868, F1=0.7765
- Epoch 5: Loss=0.1105, mIoU=0.7348, F1=0.8229
- Epoch 6: Loss=0.1058, mIoU=0.6735, F1=0.7628
- Epoch 7: Loss=0.1024, mIoU=0.7381, F1=0.8256
- Epoch 8: Loss=0.0980, mIoU=0.7386, F1=0.8261
- Epoch 9: Loss=0.0944, mIoU=0.7126, F1=0.8020
- Epoch 10: Loss=0.0903, mIoU=0.6866, F1=0.7762

当前轮次最佳结果: Round=8, Labeled=767, mIoU=0.7386, F1=0.8261

## Round 9

Labeled Pool Size: 855

- Epoch 1: Loss=0.3321, mIoU=0.6494, F1=0.7364
- Epoch 2: Loss=0.1419, mIoU=0.6653, F1=0.7538
- Epoch 3: Loss=0.1182, mIoU=0.7091, F1=0.7987
- Epoch 4: Loss=0.1076, mIoU=0.7238, F1=0.8126
- Epoch 5: Loss=0.1017, mIoU=0.6957, F1=0.7854
- Epoch 6: Loss=0.0972, mIoU=0.7012, F1=0.7909
- Epoch 7: Loss=0.0957, mIoU=0.7489, F1=0.8353
- Epoch 8: Loss=0.0898, mIoU=0.7466, F1=0.8330
- Epoch 9: Loss=0.0874, mIoU=0.7488, F1=0.8349
- Epoch 10: Loss=0.0838, mIoU=0.7324, F1=0.8204

当前轮次最佳结果: Round=9, Labeled=855, mIoU=0.7489, F1=0.8353

## Round 10

Labeled Pool Size: 943

- Epoch 1: Loss=0.2385, mIoU=0.6730, F1=0.7630
- Epoch 2: Loss=0.1277, mIoU=0.6844, F1=0.7741
- Epoch 3: Loss=0.1077, mIoU=0.7155, F1=0.8048
- Epoch 4: Loss=0.1000, mIoU=0.7281, F1=0.8168
- Epoch 5: Loss=0.0948, mIoU=0.7348, F1=0.8227
- Epoch 6: Loss=0.0917, mIoU=0.7109, F1=0.8003
- Epoch 7: Loss=0.0869, mIoU=0.7506, F1=0.8366
- Epoch 8: Loss=0.0832, mIoU=0.7517, F1=0.8375
- Epoch 9: Loss=0.0817, mIoU=0.7535, F1=0.8391
- Epoch 10: Loss=0.0777, mIoU=0.7604, F1=0.8450

当前轮次最佳结果: Round=10, Labeled=943, mIoU=0.7604, F1=0.8450

## Round 11

Labeled Pool Size: 1031

- Epoch 1: Loss=0.2068, mIoU=0.6909, F1=0.7812
- Epoch 2: Loss=0.1168, mIoU=0.7055, F1=0.7952
- Epoch 3: Loss=0.1025, mIoU=0.7346, F1=0.8226
- Epoch 4: Loss=0.0936, mIoU=0.7485, F1=0.8349
- Epoch 5: Loss=0.0894, mIoU=0.7440, F1=0.8310
- Epoch 6: Loss=0.0837, mIoU=0.7379, F1=0.8253
- Epoch 7: Loss=0.0810, mIoU=0.7292, F1=0.8191
- Epoch 8: Loss=0.0799, mIoU=0.7584, F1=0.8433
- Epoch 9: Loss=0.0745, mIoU=0.7643, F1=0.8483
- Epoch 10: Loss=0.0732, mIoU=0.7504, F1=0.8363

当前轮次最佳结果: Round=11, Labeled=1031, mIoU=0.7643, F1=0.8483

## Round 12

Labeled Pool Size: 1119

- Epoch 1: Loss=0.2027, mIoU=0.6471, F1=0.7337
- Epoch 2: Loss=0.1106, mIoU=0.7121, F1=0.8018
- Epoch 3: Loss=0.0972, mIoU=0.7313, F1=0.8196
- Epoch 4: Loss=0.0902, mIoU=0.7206, F1=0.8096
- Epoch 5: Loss=0.0865, mIoU=0.7243, F1=0.8129
- Epoch 6: Loss=0.0826, mIoU=0.7494, F1=0.8356
- Epoch 7: Loss=0.0792, mIoU=0.7399, F1=0.8273
- Epoch 8: Loss=0.0772, mIoU=0.7427, F1=0.8296
- Epoch 9: Loss=0.0723, mIoU=0.7468, F1=0.8333
- Epoch 10: Loss=0.0699, mIoU=0.7636, F1=0.8477

当前轮次最佳结果: Round=12, Labeled=1119, mIoU=0.7636, F1=0.8477

## Round 13

Labeled Pool Size: 1207

- Epoch 1: Loss=0.2350, mIoU=0.6784, F1=0.7681
- Epoch 2: Loss=0.1100, mIoU=0.7193, F1=0.8085
- Epoch 3: Loss=0.0945, mIoU=0.7340, F1=0.8224
- Epoch 4: Loss=0.0899, mIoU=0.7204, F1=0.8095
- Epoch 5: Loss=0.0821, mIoU=0.6550, F1=0.7423
- Epoch 6: Loss=0.0795, mIoU=0.7034, F1=0.7930
- Epoch 7: Loss=0.0752, mIoU=0.7477, F1=0.8340
- Epoch 8: Loss=0.0719, mIoU=0.7616, F1=0.8460
- Epoch 9: Loss=0.0695, mIoU=0.7558, F1=0.8410
- Epoch 10: Loss=0.0670, mIoU=0.7645, F1=0.8485

当前轮次最佳结果: Round=13, Labeled=1207, mIoU=0.7645, F1=0.8485

## Round 14

Labeled Pool Size: 1295

- Epoch 1: Loss=0.1730, mIoU=0.7021, F1=0.7923
- Epoch 2: Loss=0.0984, mIoU=0.7105, F1=0.8001
- Epoch 3: Loss=0.0865, mIoU=0.7124, F1=0.8035
- Epoch 4: Loss=0.0815, mIoU=0.7314, F1=0.8198
- Epoch 5: Loss=0.0751, mIoU=0.7302, F1=0.8185
- Epoch 6: Loss=0.0725, mIoU=0.7279, F1=0.8163
- Epoch 7: Loss=0.0702, mIoU=0.7327, F1=0.8206
- Epoch 8: Loss=0.0662, mIoU=0.7624, F1=0.8467
- Epoch 9: Loss=0.0645, mIoU=0.7671, F1=0.8507
- Epoch 10: Loss=0.0620, mIoU=0.7407, F1=0.8278

当前轮次最佳结果: Round=14, Labeled=1295, mIoU=0.7671, F1=0.8507

## Round 15

Labeled Pool Size: 1383

- Epoch 1: Loss=0.2074, mIoU=0.6965, F1=0.7870
- Epoch 2: Loss=0.0987, mIoU=0.7304, F1=0.8191
- Epoch 3: Loss=0.0879, mIoU=0.7366, F1=0.8244
- Epoch 4: Loss=0.0800, mIoU=0.7412, F1=0.8285
- Epoch 5: Loss=0.0758, mIoU=0.7327, F1=0.8207
- Epoch 6: Loss=0.0730, mIoU=0.7265, F1=0.8150
- Epoch 7: Loss=0.0715, mIoU=0.7108, F1=0.8002
- Epoch 8: Loss=0.0680, mIoU=0.7337, F1=0.8215
- Epoch 9: Loss=0.0654, mIoU=0.7457, F1=0.8322
- Epoch 10: Loss=0.0629, mIoU=0.7651, F1=0.8490

当前轮次最佳结果: Round=15, Labeled=1383, mIoU=0.7651, F1=0.8490


## 实验汇总

预算历史: [151, 239, 327, 415, 503, 591, 679, 767, 855, 943, 1031, 1119, 1207, 1295, 1383]
ALC: 0.6693
最终 mIoU: 0.7651
最终 F1: 0.8490
