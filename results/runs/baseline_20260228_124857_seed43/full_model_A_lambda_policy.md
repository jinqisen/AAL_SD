# 实验日志

实验名称: full_model_A_lambda_policy
描述: 对照A / AAL-SD (Full)：LLM 参与决策与解释；λ由 warmup+风险闭环策略生成并在工具层 clamp（启用CI阈值+AND严重判定+EMA/冷却/限步长；默认禁止 set_lambda；不调整query_size/epochs）
开始时间: 2026-03-03T19:01:13.667883

## Round 1

Labeled Pool Size: 151

- Epoch 1: Loss=0.3548, mIoU=0.4987, F1=0.5156
- Epoch 2: Loss=0.2048, mIoU=0.5276, F1=0.5668
- Epoch 3: Loss=0.1383, mIoU=0.5623, F1=0.6235
- Epoch 4: Loss=0.1036, mIoU=0.5850, F1=0.6559
- Epoch 5: Loss=0.0839, mIoU=0.5796, F1=0.6476
- Epoch 6: Loss=0.0729, mIoU=0.6295, F1=0.7141
- Epoch 7: Loss=0.0661, mIoU=0.6373, F1=0.7234
- Epoch 8: Loss=0.0596, mIoU=0.6395, F1=0.7255
- Epoch 9: Loss=0.0564, mIoU=0.6199, F1=0.7016
- Epoch 10: Loss=0.0533, mIoU=0.6191, F1=0.7003

当前轮次最佳结果: Round=1, Labeled=151, mIoU=0.6395, F1=0.7255

## Round 2

Labeled Pool Size: 239

- Epoch 1: Loss=0.4401, mIoU=0.6064, F1=0.6855
- Epoch 2: Loss=0.2325, mIoU=0.6238, F1=0.7061
- Epoch 3: Loss=0.1715, mIoU=0.5919, F1=0.6642
- Epoch 4: Loss=0.1398, mIoU=0.6411, F1=0.7267
- Epoch 5: Loss=0.1325, mIoU=0.6852, F1=0.7754
- Epoch 6: Loss=0.1166, mIoU=0.6581, F1=0.7460
- Epoch 7: Loss=0.1088, mIoU=0.6611, F1=0.7494
- Epoch 8: Loss=0.1023, mIoU=0.6787, F1=0.7683
- Epoch 9: Loss=0.1004, mIoU=0.6847, F1=0.7746
- Epoch 10: Loss=0.0948, mIoU=0.6848, F1=0.7746

当前轮次最佳结果: Round=2, Labeled=239, mIoU=0.6852, F1=0.7754

## Round 3

Labeled Pool Size: 327

- Epoch 1: Loss=0.4055, mIoU=0.6040, F1=0.6810
- Epoch 2: Loss=0.2052, mIoU=0.6334, F1=0.7175
- Epoch 3: Loss=0.1549, mIoU=0.6216, F1=0.7029
- Epoch 4: Loss=0.1331, mIoU=0.5941, F1=0.6671
- Epoch 5: Loss=0.1201, mIoU=0.6901, F1=0.7801
- Epoch 6: Loss=0.1200, mIoU=0.6465, F1=0.7328
- Epoch 7: Loss=0.1112, mIoU=0.6471, F1=0.7332
- Epoch 8: Loss=0.1033, mIoU=0.7035, F1=0.7933
- Epoch 9: Loss=0.0972, mIoU=0.6954, F1=0.7852
- Epoch 10: Loss=0.0932, mIoU=0.6910, F1=0.7808

当前轮次最佳结果: Round=3, Labeled=327, mIoU=0.7035, F1=0.7933

## Round 4

Labeled Pool Size: 415

- Epoch 1: Loss=0.3633, mIoU=0.5991, F1=0.6745
- Epoch 2: Loss=0.1846, mIoU=0.6513, F1=0.7383
- Epoch 3: Loss=0.1436, mIoU=0.6640, F1=0.7524
- Epoch 4: Loss=0.1322, mIoU=0.6836, F1=0.7733
- Epoch 5: Loss=0.1205, mIoU=0.6966, F1=0.7865
- Epoch 6: Loss=0.1126, mIoU=0.7038, F1=0.7936
- Epoch 7: Loss=0.1062, mIoU=0.7159, F1=0.8055
- Epoch 8: Loss=0.1043, mIoU=0.6825, F1=0.7719
- Epoch 9: Loss=0.1012, mIoU=0.7094, F1=0.7989
- Epoch 10: Loss=0.0960, mIoU=0.7061, F1=0.7958

当前轮次最佳结果: Round=4, Labeled=415, mIoU=0.7159, F1=0.8055

## Round 5

Labeled Pool Size: 503

- Epoch 1: Loss=0.4067, mIoU=0.6329, F1=0.7171
- Epoch 2: Loss=0.1827, mIoU=0.6620, F1=0.7503
- Epoch 3: Loss=0.1452, mIoU=0.6447, F1=0.7306
- Epoch 4: Loss=0.1294, mIoU=0.7110, F1=0.8010
- Epoch 5: Loss=0.1261, mIoU=0.7220, F1=0.8111
- Epoch 6: Loss=0.1155, mIoU=0.7222, F1=0.8112
- Epoch 7: Loss=0.1117, mIoU=0.7051, F1=0.7949
- Epoch 8: Loss=0.1097, mIoU=0.7243, F1=0.8130
- Epoch 9: Loss=0.1033, mIoU=0.7172, F1=0.8063
- Epoch 10: Loss=0.0985, mIoU=0.7247, F1=0.8133

当前轮次最佳结果: Round=5, Labeled=503, mIoU=0.7247, F1=0.8133

## Round 6

Labeled Pool Size: 591

- Epoch 1: Loss=0.3479, mIoU=0.6566, F1=0.7447
- Epoch 2: Loss=0.1677, mIoU=0.6732, F1=0.7624
- Epoch 3: Loss=0.1382, mIoU=0.6742, F1=0.7634
- Epoch 4: Loss=0.1249, mIoU=0.7104, F1=0.8000
- Epoch 5: Loss=0.1185, mIoU=0.7097, F1=0.7994
- Epoch 6: Loss=0.1126, mIoU=0.7236, F1=0.8128
- Epoch 7: Loss=0.1054, mIoU=0.7275, F1=0.8160
- Epoch 8: Loss=0.1026, mIoU=0.7318, F1=0.8199
- Epoch 9: Loss=0.0967, mIoU=0.7391, F1=0.8266
- Epoch 10: Loss=0.0964, mIoU=0.7410, F1=0.8282

当前轮次最佳结果: Round=6, Labeled=591, mIoU=0.7410, F1=0.8282

## Round 7

Labeled Pool Size: 679

- Epoch 1: Loss=0.3930, mIoU=0.6254, F1=0.7080
- Epoch 2: Loss=0.1638, mIoU=0.7002, F1=0.7904
- Epoch 3: Loss=0.1353, mIoU=0.6810, F1=0.7706
- Epoch 4: Loss=0.1207, mIoU=0.6860, F1=0.7758
- Epoch 5: Loss=0.1124, mIoU=0.7187, F1=0.8078
- Epoch 6: Loss=0.1062, mIoU=0.7210, F1=0.8099
- Epoch 7: Loss=0.1006, mIoU=0.7269, F1=0.8153
- Epoch 8: Loss=0.0978, mIoU=0.7283, F1=0.8166
- Epoch 9: Loss=0.0944, mIoU=0.7230, F1=0.8117
- Epoch 10: Loss=0.0917, mIoU=0.7440, F1=0.8307

当前轮次最佳结果: Round=7, Labeled=679, mIoU=0.7440, F1=0.8307

## Round 8

Labeled Pool Size: 767

- Epoch 1: Loss=0.3442, mIoU=0.6465, F1=0.7330
- Epoch 2: Loss=0.1474, mIoU=0.6863, F1=0.7762
- Epoch 3: Loss=0.1244, mIoU=0.7160, F1=0.8057
- Epoch 4: Loss=0.1160, mIoU=0.7208, F1=0.8098
- Epoch 5: Loss=0.1100, mIoU=0.7068, F1=0.7963
- Epoch 6: Loss=0.1057, mIoU=0.7368, F1=0.8244
- Epoch 7: Loss=0.1023, mIoU=0.7159, F1=0.8051
- Epoch 8: Loss=0.0973, mIoU=0.7372, F1=0.8249
- Epoch 9: Loss=0.0952, mIoU=0.7193, F1=0.8083
- Epoch 10: Loss=0.0923, mIoU=0.7264, F1=0.8149

当前轮次最佳结果: Round=8, Labeled=767, mIoU=0.7372, F1=0.8249

## Round 9

Labeled Pool Size: 855

- Epoch 1: Loss=0.2442, mIoU=0.6365, F1=0.7214
- Epoch 2: Loss=0.1337, mIoU=0.6824, F1=0.7720
- Epoch 3: Loss=0.1143, mIoU=0.7086, F1=0.7982
- Epoch 4: Loss=0.1070, mIoU=0.7186, F1=0.8077
- Epoch 5: Loss=0.1009, mIoU=0.7111, F1=0.8004
- Epoch 6: Loss=0.0954, mIoU=0.7454, F1=0.8320
- Epoch 7: Loss=0.0914, mIoU=0.7413, F1=0.8284
- Epoch 8: Loss=0.0879, mIoU=0.7389, F1=0.8263
- Epoch 9: Loss=0.0846, mIoU=0.7476, F1=0.8339
- Epoch 10: Loss=0.0815, mIoU=0.7534, F1=0.8389

当前轮次最佳结果: Round=9, Labeled=855, mIoU=0.7534, F1=0.8389

## Round 10

Labeled Pool Size: 943

- Epoch 1: Loss=0.2162, mIoU=0.6818, F1=0.7720
- Epoch 2: Loss=0.1226, mIoU=0.6812, F1=0.7707
- Epoch 3: Loss=0.1063, mIoU=0.6871, F1=0.7766
- Epoch 4: Loss=0.1002, mIoU=0.7123, F1=0.8017
- Epoch 5: Loss=0.0936, mIoU=0.7221, F1=0.8109
- Epoch 6: Loss=0.0890, mIoU=0.7143, F1=0.8035
- Epoch 7: Loss=0.0850, mIoU=0.7428, F1=0.8304
- Epoch 8: Loss=0.0826, mIoU=0.7416, F1=0.8285
- Epoch 9: Loss=0.0802, mIoU=0.7520, F1=0.8379
- Epoch 10: Loss=0.0777, mIoU=0.7437, F1=0.8308

当前轮次最佳结果: Round=10, Labeled=943, mIoU=0.7520, F1=0.8379

## Round 11

Labeled Pool Size: 1031

- Epoch 1: Loss=0.2091, mIoU=0.6470, F1=0.7334
- Epoch 2: Loss=0.1146, mIoU=0.6725, F1=0.7614
- Epoch 3: Loss=0.1025, mIoU=0.7119, F1=0.8014
- Epoch 4: Loss=0.0944, mIoU=0.7352, F1=0.8230
- Epoch 5: Loss=0.0884, mIoU=0.7419, F1=0.8293
- Epoch 6: Loss=0.0853, mIoU=0.7499, F1=0.8360
- Epoch 7: Loss=0.0816, mIoU=0.7439, F1=0.8307
- Epoch 8: Loss=0.0795, mIoU=0.7309, F1=0.8189
- Epoch 9: Loss=0.0758, mIoU=0.7578, F1=0.8427
- Epoch 10: Loss=0.0724, mIoU=0.7511, F1=0.8369

当前轮次最佳结果: Round=11, Labeled=1031, mIoU=0.7578, F1=0.8427

## Round 12

Labeled Pool Size: 1119

- Epoch 1: Loss=0.2371, mIoU=0.6338, F1=0.7182
- Epoch 2: Loss=0.1159, mIoU=0.6337, F1=0.7175
- Epoch 3: Loss=0.1010, mIoU=0.7200, F1=0.8093
- Epoch 4: Loss=0.0943, mIoU=0.7110, F1=0.8004
- Epoch 5: Loss=0.0873, mIoU=0.7429, F1=0.8299
- Epoch 6: Loss=0.0836, mIoU=0.7124, F1=0.8016
- Epoch 7: Loss=0.0811, mIoU=0.7502, F1=0.8362
- Epoch 8: Loss=0.0771, mIoU=0.7419, F1=0.8289
- Epoch 9: Loss=0.0742, mIoU=0.7544, F1=0.8397
- Epoch 10: Loss=0.0710, mIoU=0.7474, F1=0.8337

当前轮次最佳结果: Round=12, Labeled=1119, mIoU=0.7544, F1=0.8397

## Round 13

Labeled Pool Size: 1207

- Epoch 1: Loss=0.1787, mIoU=0.6152, F1=0.6948
- Epoch 2: Loss=0.1039, mIoU=0.6991, F1=0.7891
- Epoch 3: Loss=0.0920, mIoU=0.7277, F1=0.8161
- Epoch 4: Loss=0.0850, mIoU=0.7051, F1=0.7947
- Epoch 5: Loss=0.0805, mIoU=0.7268, F1=0.8152
- Epoch 6: Loss=0.0762, mIoU=0.7375, F1=0.8250
- Epoch 7: Loss=0.0738, mIoU=0.7277, F1=0.8160
- Epoch 8: Loss=0.0705, mIoU=0.7628, F1=0.8470
- Epoch 9: Loss=0.0682, mIoU=0.7611, F1=0.8455
- Epoch 10: Loss=0.0679, mIoU=0.7559, F1=0.8410

当前轮次最佳结果: Round=13, Labeled=1207, mIoU=0.7628, F1=0.8470

## Round 14

Labeled Pool Size: 1295

- Epoch 1: Loss=0.2233, mIoU=0.6362, F1=0.7207
- Epoch 2: Loss=0.1040, mIoU=0.7137, F1=0.8033
- Epoch 3: Loss=0.0924, mIoU=0.6500, F1=0.7366
- Epoch 4: Loss=0.0832, mIoU=0.7107, F1=0.8000
- Epoch 5: Loss=0.0796, mIoU=0.7196, F1=0.8086
- Epoch 6: Loss=0.0765, mIoU=0.7418, F1=0.8290
- Epoch 7: Loss=0.0749, mIoU=0.7215, F1=0.8103
- Epoch 8: Loss=0.0709, mIoU=0.7376, F1=0.8250
- Epoch 9: Loss=0.0679, mIoU=0.7480, F1=0.8342
- Epoch 10: Loss=0.0665, mIoU=0.7452, F1=0.8317

当前轮次最佳结果: Round=14, Labeled=1295, mIoU=0.7480, F1=0.8342

## Round 15

Labeled Pool Size: 1383

- Epoch 1: Loss=0.2091, mIoU=0.6322, F1=0.7162
- Epoch 2: Loss=0.0992, mIoU=0.6750, F1=0.7642
- Epoch 3: Loss=0.0865, mIoU=0.7009, F1=0.7905
- Epoch 4: Loss=0.0780, mIoU=0.7405, F1=0.8280
- Epoch 5: Loss=0.0741, mIoU=0.7380, F1=0.8255
- Epoch 6: Loss=0.0725, mIoU=0.7473, F1=0.8339
- Epoch 7: Loss=0.0692, mIoU=0.7505, F1=0.8367
- Epoch 8: Loss=0.0655, mIoU=0.7452, F1=0.8318
- Epoch 9: Loss=0.0649, mIoU=0.7575, F1=0.8425
- Epoch 10: Loss=0.0626, mIoU=0.7316, F1=0.8195

当前轮次最佳结果: Round=15, Labeled=1383, mIoU=0.7575, F1=0.8425


## 实验汇总

预算历史: [151, 239, 327, 415, 503, 591, 679, 767, 855, 943, 1031, 1119, 1207, 1295, 1383]
ALC: 0.6633
最终 mIoU: 0.7575
最终 F1: 0.8425
