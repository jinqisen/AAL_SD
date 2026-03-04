# 实验日志

实验名称: full_model_B_lambda_agent
描述: 对照B：授权LLM/Agent 在显式约束下逐轮 set_lambda（并记录到trace）；启用CI阈值+AND严重判定+回撤禁升+EMA/冷却/限步长；禁止使用policy自动填充本轮λ（必须显式set_lambda）；不调整query_size/epochs
开始时间: 2026-03-03T10:08:31.660814

## Round 1

Labeled Pool Size: 151

- Epoch 1: Loss=0.5368, mIoU=0.4893, F1=0.5004
- Epoch 2: Loss=0.3070, mIoU=0.4956, F1=0.5098
- Epoch 3: Loss=0.1961, mIoU=0.5076, F1=0.5324
- Epoch 4: Loss=0.1397, mIoU=0.5422, F1=0.5915
- Epoch 5: Loss=0.1084, mIoU=0.5368, F1=0.5823
- Epoch 6: Loss=0.0878, mIoU=0.6153, F1=0.6963
- Epoch 7: Loss=0.0736, mIoU=0.5844, F1=0.6543
- Epoch 8: Loss=0.0671, mIoU=0.5771, F1=0.6439
- Epoch 9: Loss=0.0583, mIoU=0.6353, F1=0.7202
- Epoch 10: Loss=0.0552, mIoU=0.6553, F1=0.7433

当前轮次最佳结果: Round=1, Labeled=151, mIoU=0.6553, F1=0.7433

## Round 2

Labeled Pool Size: 239

- Epoch 1: Loss=0.3400, mIoU=0.5249, F1=0.5623
- Epoch 2: Loss=0.1967, mIoU=0.6146, F1=0.6948
- Epoch 3: Loss=0.1490, mIoU=0.6070, F1=0.6846
- Epoch 4: Loss=0.1272, mIoU=0.6390, F1=0.7243
- Epoch 5: Loss=0.1170, mIoU=0.5844, F1=0.6542
- Epoch 6: Loss=0.1094, mIoU=0.6830, F1=0.7729

--- [Checkpoint] 续跑开始时间: 2026-03-03T11:27:26.164980 ---

## Round 2

Labeled Pool Size: 239

- Epoch 1: Loss=0.3388, mIoU=0.5142, F1=0.5438
- Epoch 2: Loss=0.1984, mIoU=0.5924, F1=0.6654
- Epoch 3: Loss=0.1493, mIoU=0.5930, F1=0.6662
- Epoch 4: Loss=0.1286, mIoU=0.6126, F1=0.6920
- Epoch 5: Loss=0.1193, mIoU=0.5957, F1=0.6697
- Epoch 6: Loss=0.1101, mIoU=0.6270, F1=0.7101
- Epoch 7: Loss=0.1026, mIoU=0.6223, F1=0.7039
- Epoch 8: Loss=0.1017, mIoU=0.6622, F1=0.7506
- Epoch 9: Loss=0.0970, mIoU=0.6456, F1=0.7318
- Epoch 10: Loss=0.0911, mIoU=0.6878, F1=0.7778

当前轮次最佳结果: Round=2, Labeled=239, mIoU=0.6878, F1=0.7778

## Round 3

Labeled Pool Size: 327

- Epoch 1: Loss=0.4028, mIoU=0.6017, F1=0.6788
- Epoch 2: Loss=0.2108, mIoU=0.6085, F1=0.6868
- Epoch 3: Loss=0.1627, mIoU=0.5935, F1=0.6666
- Epoch 4: Loss=0.1414, mIoU=0.6902, F1=0.7813
- Epoch 5: Loss=0.1323, mIoU=0.7042, F1=0.7945
- Epoch 6: Loss=0.1217, mIoU=0.6747, F1=0.7640
- Epoch 7: Loss=0.1171, mIoU=0.7223, F1=0.8115
- Epoch 8: Loss=0.1117, mIoU=0.7124, F1=0.8020
- Epoch 9: Loss=0.1080, mIoU=0.6862, F1=0.7760
- Epoch 10: Loss=0.1068, mIoU=0.7095, F1=0.7993

当前轮次最佳结果: Round=3, Labeled=327, mIoU=0.7223, F1=0.8115

## Round 4

Labeled Pool Size: 415

- Epoch 1: Loss=0.3730, mIoU=0.6567, F1=0.7449
- Epoch 2: Loss=0.1931, mIoU=0.6488, F1=0.7357
- Epoch 3: Loss=0.1550, mIoU=0.6655, F1=0.7541
- Epoch 4: Loss=0.1350, mIoU=0.6895, F1=0.7793
- Epoch 5: Loss=0.1272, mIoU=0.6947, F1=0.7846
- Epoch 6: Loss=0.1195, mIoU=0.7142, F1=0.8037
- Epoch 7: Loss=0.1178, mIoU=0.7128, F1=0.8023
- Epoch 8: Loss=0.1116, mIoU=0.6472, F1=0.7335
- Epoch 9: Loss=0.1118, mIoU=0.7289, F1=0.8174
- Epoch 10: Loss=0.1049, mIoU=0.7290, F1=0.8177

当前轮次最佳结果: Round=4, Labeled=415, mIoU=0.7290, F1=0.8177

## Round 5

Labeled Pool Size: 503

- Epoch 1: Loss=0.3425, mIoU=0.5744, F1=0.6401
- Epoch 2: Loss=0.1783, mIoU=0.6712, F1=0.7606
- Epoch 3: Loss=0.1495, mIoU=0.7063, F1=0.7962
- Epoch 4: Loss=0.1336, mIoU=0.7048, F1=0.7946
- Epoch 5: Loss=0.1227, mIoU=0.6992, F1=0.7890
- Epoch 6: Loss=0.1220, mIoU=0.7196, F1=0.8090
- Epoch 7: Loss=0.1138, mIoU=0.7165, F1=0.8058
- Epoch 8: Loss=0.1096, mIoU=0.7319, F1=0.8200
- Epoch 9: Loss=0.1060, mIoU=0.7198, F1=0.8089
- Epoch 10: Loss=0.1018, mIoU=0.7290, F1=0.8174

当前轮次最佳结果: Round=5, Labeled=503, mIoU=0.7319, F1=0.8200

## Round 6

Labeled Pool Size: 591

- Epoch 1: Loss=0.3790, mIoU=0.6399, F1=0.7256
- Epoch 2: Loss=0.1697, mIoU=0.6657, F1=0.7545
- Epoch 3: Loss=0.1392, mIoU=0.6921, F1=0.7820
- Epoch 4: Loss=0.1275, mIoU=0.7216, F1=0.8108
- Epoch 5: Loss=0.1181, mIoU=0.7267, F1=0.8154
- Epoch 6: Loss=0.1091, mIoU=0.7232, F1=0.8120
- Epoch 7: Loss=0.1073, mIoU=0.7271, F1=0.8157
- Epoch 8: Loss=0.1043, mIoU=0.7435, F1=0.8309
- Epoch 9: Loss=0.0989, mIoU=0.7479, F1=0.8342
- Epoch 10: Loss=0.0970, mIoU=0.7311, F1=0.8192

当前轮次最佳结果: Round=6, Labeled=591, mIoU=0.7479, F1=0.8342

## Round 7

Labeled Pool Size: 679

- Epoch 1: Loss=0.3273, mIoU=0.6667, F1=0.7559
- Epoch 2: Loss=0.1559, mIoU=0.6891, F1=0.7791
- Epoch 3: Loss=0.1334, mIoU=0.6805, F1=0.7700
- Epoch 4: Loss=0.1187, mIoU=0.7027, F1=0.7925
- Epoch 5: Loss=0.1117, mIoU=0.7153, F1=0.8046
- Epoch 6: Loss=0.1070, mIoU=0.6968, F1=0.7866
- Epoch 7: Loss=0.1013, mIoU=0.6957, F1=0.7854
- Epoch 8: Loss=0.0995, mIoU=0.7455, F1=0.8322
- Epoch 9: Loss=0.0958, mIoU=0.7265, F1=0.8150
- Epoch 10: Loss=0.0930, mIoU=0.7539, F1=0.8395

当前轮次最佳结果: Round=7, Labeled=679, mIoU=0.7539, F1=0.8395

## Round 8

Labeled Pool Size: 767

- Epoch 1: Loss=0.3596, mIoU=0.6287, F1=0.7126
- Epoch 2: Loss=0.1492, mIoU=0.7096, F1=0.7994
- Epoch 3: Loss=0.1261, mIoU=0.6918, F1=0.7817
- Epoch 4: Loss=0.1135, mIoU=0.7112, F1=0.8009
- Epoch 5: Loss=0.1092, mIoU=0.7176, F1=0.8068
- Epoch 6: Loss=0.1029, mIoU=0.7480, F1=0.8348
- Epoch 7: Loss=0.0967, mIoU=0.7404, F1=0.8277
- Epoch 8: Loss=0.0956, mIoU=0.7424, F1=0.8295
- Epoch 9: Loss=0.0927, mIoU=0.7198, F1=0.8088
- Epoch 10: Loss=0.0896, mIoU=0.7433, F1=0.8302

当前轮次最佳结果: Round=8, Labeled=767, mIoU=0.7480, F1=0.8348

## Round 9

Labeled Pool Size: 855

- Epoch 1: Loss=0.3251, mIoU=0.6385, F1=0.7239
- Epoch 2: Loss=0.1371, mIoU=0.6870, F1=0.7768
- Epoch 3: Loss=0.1209, mIoU=0.7020, F1=0.7917
- Epoch 4: Loss=0.1086, mIoU=0.6987, F1=0.7884
- Epoch 5: Loss=0.1005, mIoU=0.6988, F1=0.7885
- Epoch 6: Loss=0.0958, mIoU=0.7377, F1=0.8254
- Epoch 7: Loss=0.0933, mIoU=0.7394, F1=0.8267
- Epoch 8: Loss=0.0906, mIoU=0.7407, F1=0.8279
- Epoch 9: Loss=0.0879, mIoU=0.7298, F1=0.8182
- Epoch 10: Loss=0.0859, mIoU=0.7594, F1=0.8441

当前轮次最佳结果: Round=9, Labeled=855, mIoU=0.7594, F1=0.8441

## Round 10

Labeled Pool Size: 943

- Epoch 1: Loss=0.2350, mIoU=0.6724, F1=0.7620
- Epoch 2: Loss=0.1269, mIoU=0.6853, F1=0.7751
- Epoch 3: Loss=0.1103, mIoU=0.6727, F1=0.7617
- Epoch 4: Loss=0.1008, mIoU=0.6880, F1=0.7776
- Epoch 5: Loss=0.0957, mIoU=0.7234, F1=0.8122
- Epoch 6: Loss=0.0910, mIoU=0.7492, F1=0.8355
- Epoch 7: Loss=0.0861, mIoU=0.7215, F1=0.8103
- Epoch 8: Loss=0.0826, mIoU=0.7507, F1=0.8368
- Epoch 9: Loss=0.0786, mIoU=0.7477, F1=0.8340
- Epoch 10: Loss=0.0755, mIoU=0.7569, F1=0.8420

当前轮次最佳结果: Round=10, Labeled=943, mIoU=0.7569, F1=0.8420

## Round 11

Labeled Pool Size: 1031

- Epoch 1: Loss=0.2057, mIoU=0.6810, F1=0.7711
- Epoch 2: Loss=0.1148, mIoU=0.7087, F1=0.7984
- Epoch 3: Loss=0.0995, mIoU=0.7357, F1=0.8238
- Epoch 4: Loss=0.0926, mIoU=0.7397, F1=0.8270
- Epoch 5: Loss=0.0876, mIoU=0.7251, F1=0.8138
- Epoch 6: Loss=0.0831, mIoU=0.7246, F1=0.8132
- Epoch 7: Loss=0.0807, mIoU=0.7555, F1=0.8409
- Epoch 8: Loss=0.0771, mIoU=0.7473, F1=0.8336
- Epoch 9: Loss=0.0745, mIoU=0.7560, F1=0.8412
- Epoch 10: Loss=0.0742, mIoU=0.7583, F1=0.8434

当前轮次最佳结果: Round=11, Labeled=1031, mIoU=0.7583, F1=0.8434

## Round 12

Labeled Pool Size: 1119

- Epoch 1: Loss=0.2005, mIoU=0.6147, F1=0.6943
- Epoch 2: Loss=0.1089, mIoU=0.7035, F1=0.7933
- Epoch 3: Loss=0.0956, mIoU=0.7401, F1=0.8276
- Epoch 4: Loss=0.0893, mIoU=0.7037, F1=0.7933
- Epoch 5: Loss=0.0839, mIoU=0.7409, F1=0.8281
- Epoch 6: Loss=0.0799, mIoU=0.7420, F1=0.8290
- Epoch 7: Loss=0.0773, mIoU=0.7552, F1=0.8407
- Epoch 8: Loss=0.0741, mIoU=0.7427, F1=0.8296
- Epoch 9: Loss=0.0721, mIoU=0.7409, F1=0.8280
- Epoch 10: Loss=0.0687, mIoU=0.7606, F1=0.8451

当前轮次最佳结果: Round=12, Labeled=1119, mIoU=0.7606, F1=0.8451

## Round 13

Labeled Pool Size: 1207

- Epoch 1: Loss=0.2361, mIoU=0.6890, F1=0.7791
- Epoch 2: Loss=0.1111, mIoU=0.6882, F1=0.7780
- Epoch 3: Loss=0.0966, mIoU=0.7048, F1=0.7945
- Epoch 4: Loss=0.0900, mIoU=0.7193, F1=0.8083
- Epoch 5: Loss=0.0835, mIoU=0.7042, F1=0.7939
- Epoch 6: Loss=0.0778, mIoU=0.7330, F1=0.8209
- Epoch 7: Loss=0.0754, mIoU=0.7352, F1=0.8233
- Epoch 8: Loss=0.0723, mIoU=0.7579, F1=0.8430
- Epoch 9: Loss=0.0696, mIoU=0.7534, F1=0.8390
- Epoch 10: Loss=0.0680, mIoU=0.7576, F1=0.8425

当前轮次最佳结果: Round=13, Labeled=1207, mIoU=0.7579, F1=0.8430

## Round 14

Labeled Pool Size: 1295

- Epoch 1: Loss=0.1738, mIoU=0.6408, F1=0.7261
- Epoch 2: Loss=0.1023, mIoU=0.6996, F1=0.7896
- Epoch 3: Loss=0.0875, mIoU=0.7242, F1=0.8129
- Epoch 4: Loss=0.0795, mIoU=0.7303, F1=0.8185
- Epoch 5: Loss=0.0759, mIoU=0.7295, F1=0.8178
- Epoch 6: Loss=0.0727, mIoU=0.7521, F1=0.8379
- Epoch 7: Loss=0.0701, mIoU=0.7308, F1=0.8189
- Epoch 8: Loss=0.0682, mIoU=0.7387, F1=0.8261
- Epoch 9: Loss=0.0672, mIoU=0.7634, F1=0.8478
- Epoch 10: Loss=0.0659, mIoU=0.7591, F1=0.8438

当前轮次最佳结果: Round=14, Labeled=1295, mIoU=0.7634, F1=0.8478

## Round 15

Labeled Pool Size: 1383

- Epoch 1: Loss=0.2050, mIoU=0.6747, F1=0.7642
- Epoch 2: Loss=0.0997, mIoU=0.7212, F1=0.8105
- Epoch 3: Loss=0.0877, mIoU=0.7008, F1=0.7905
- Epoch 4: Loss=0.0819, mIoU=0.7487, F1=0.8352
- Epoch 5: Loss=0.0778, mIoU=0.7402, F1=0.8274
- Epoch 6: Loss=0.0740, mIoU=0.7368, F1=0.8247
- Epoch 7: Loss=0.0702, mIoU=0.7226, F1=0.8113
- Epoch 8: Loss=0.0686, mIoU=0.7423, F1=0.8295
- Epoch 9: Loss=0.0682, mIoU=0.7178, F1=0.8068
- Epoch 10: Loss=0.0630, mIoU=0.7634, F1=0.8476

当前轮次最佳结果: Round=15, Labeled=1383, mIoU=0.7634, F1=0.8476


## 实验汇总

预算历史: [151, 239, 327, 415, 503, 591, 679, 767, 855, 943, 1031, 1119, 1207, 1295, 1383]
ALC: 0.6701
最终 mIoU: 0.7634
最终 F1: 0.8476
