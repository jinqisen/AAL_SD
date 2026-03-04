# 实验日志

实验名称: full_model_B_lambda_agent
描述: 对照B：授权LLM/Agent 在显式约束下逐轮 set_lambda（并记录到trace）；启用CI阈值+AND严重判定+回撤禁升+EMA/冷却/限步长；禁止使用policy自动填充本轮λ（必须显式set_lambda）；不调整query_size/epochs
开始时间: 2026-03-03T22:38:18.837221

## Round 1

Labeled Pool Size: 151

- Epoch 1: Loss=0.4861, mIoU=0.5031, F1=0.5457
- Epoch 2: Loss=0.2774, mIoU=0.5304, F1=0.5747
- Epoch 3: Loss=0.1828, mIoU=0.5402, F1=0.5894
- Epoch 4: Loss=0.1258, mIoU=0.5707, F1=0.6362
- Epoch 5: Loss=0.1002, mIoU=0.5825, F1=0.6523
- Epoch 6: Loss=0.0855, mIoU=0.6094, F1=0.6882
- Epoch 7: Loss=0.0731, mIoU=0.5938, F1=0.6673
- Epoch 8: Loss=0.0653, mIoU=0.6218, F1=0.7036
- Epoch 9: Loss=0.0599, mIoU=0.6452, F1=0.7317
- Epoch 10: Loss=0.0537, mIoU=0.6392, F1=0.7247

当前轮次最佳结果: Round=1, Labeled=151, mIoU=0.6452, F1=0.7317

## Round 2

Labeled Pool Size: 239

- Epoch 1: Loss=0.4384, mIoU=0.5234, F1=0.5627
- Epoch 2: Loss=0.2469, mIoU=0.5677, F1=0.6307
- Epoch 3: Loss=0.1838, mIoU=0.5758, F1=0.6420
- Epoch 4: Loss=0.1571, mIoU=0.6363, F1=0.7211
- Epoch 5: Loss=0.1460, mIoU=0.6490, F1=0.7362
- Epoch 6: Loss=0.1358, mIoU=0.6617, F1=0.7501
- Epoch 7: Loss=0.1245, mIoU=0.6992, F1=0.7900
- Epoch 8: Loss=0.1204, mIoU=0.6706, F1=0.7597
- Epoch 9: Loss=0.1136, mIoU=0.7075, F1=0.7977
- Epoch 10: Loss=0.1091, mIoU=0.6999, F1=0.7900

当前轮次最佳结果: Round=2, Labeled=239, mIoU=0.7075, F1=0.7977

## Round 3

Labeled Pool Size: 327

- Epoch 1: Loss=0.4899, mIoU=0.6111, F1=0.6936
- Epoch 2: Loss=0.2402, mIoU=0.6267, F1=0.7098
- Epoch 3: Loss=0.1813, mIoU=0.6180, F1=0.6988
- Epoch 4: Loss=0.1582, mIoU=0.6784, F1=0.7682
- Epoch 5: Loss=0.1442, mIoU=0.6803, F1=0.7702
- Epoch 6: Loss=0.1350, mIoU=0.6388, F1=0.7238
- Epoch 7: Loss=0.1296, mIoU=0.6975, F1=0.7882
- Epoch 8: Loss=0.1233, mIoU=0.7093, F1=0.7991
- Epoch 9: Loss=0.1192, mIoU=0.6940, F1=0.7838
- Epoch 10: Loss=0.1178, mIoU=0.7059, F1=0.7956

当前轮次最佳结果: Round=3, Labeled=327, mIoU=0.7093, F1=0.7991

## Round 4

Labeled Pool Size: 415

- Epoch 1: Loss=0.4159, mIoU=0.5731, F1=0.6385
- Epoch 2: Loss=0.2052, mIoU=0.6294, F1=0.7129
- Epoch 3: Loss=0.1615, mIoU=0.6929, F1=0.7829
- Epoch 4: Loss=0.1440, mIoU=0.6311, F1=0.7146
- Epoch 5: Loss=0.1320, mIoU=0.7122, F1=0.8019
- Epoch 6: Loss=0.1267, mIoU=0.7025, F1=0.7924
- Epoch 7: Loss=0.1194, mIoU=0.7175, F1=0.8073
- Epoch 8: Loss=0.1184, mIoU=0.7020, F1=0.7918
- Epoch 9: Loss=0.1118, mIoU=0.7063, F1=0.7960
- Epoch 10: Loss=0.1101, mIoU=0.7316, F1=0.8199

当前轮次最佳结果: Round=4, Labeled=415, mIoU=0.7316, F1=0.8199

## Round 5

Labeled Pool Size: 503

- Epoch 1: Loss=0.4465, mIoU=0.6005, F1=0.6767
- Epoch 2: Loss=0.1887, mIoU=0.6042, F1=0.6809
- Epoch 3: Loss=0.1566, mIoU=0.6705, F1=0.7596
- Epoch 4: Loss=0.1427, mIoU=0.6929, F1=0.7828
- Epoch 5: Loss=0.1315, mIoU=0.7103, F1=0.7999
- Epoch 6: Loss=0.1242, mIoU=0.7053, F1=0.7951
- Epoch 7: Loss=0.1218, mIoU=0.7111, F1=0.8007
- Epoch 8: Loss=0.1160, mIoU=0.7229, F1=0.8120
- Epoch 9: Loss=0.1133, mIoU=0.7310, F1=0.8192
- Epoch 10: Loss=0.1059, mIoU=0.7306, F1=0.8190

当前轮次最佳结果: Round=5, Labeled=503, mIoU=0.7310, F1=0.8192

## Round 6

Labeled Pool Size: 591

- Epoch 1: Loss=0.3807, mIoU=0.6262, F1=0.7099
- Epoch 2: Loss=0.1712, mIoU=0.6601, F1=0.7485
- Epoch 3: Loss=0.1433, mIoU=0.7201, F1=0.8099
- Epoch 4: Loss=0.1326, mIoU=0.7184, F1=0.8078
- Epoch 5: Loss=0.1241, mIoU=0.7183, F1=0.8077
- Epoch 6: Loss=0.1162, mIoU=0.7036, F1=0.7934
- Epoch 7: Loss=0.1135, mIoU=0.7357, F1=0.8238
- Epoch 8: Loss=0.1121, mIoU=0.7191, F1=0.8081
- Epoch 9: Loss=0.1085, mIoU=0.7259, F1=0.8147
- Epoch 10: Loss=0.1024, mIoU=0.7328, F1=0.8210

当前轮次最佳结果: Round=6, Labeled=591, mIoU=0.7357, F1=0.8238

## Round 7

Labeled Pool Size: 679

- Epoch 1: Loss=0.2774, mIoU=0.6440, F1=0.7304
- Epoch 2: Loss=0.1505, mIoU=0.6714, F1=0.7605
- Epoch 3: Loss=0.1285, mIoU=0.6854, F1=0.7752
- Epoch 4: Loss=0.1204, mIoU=0.7292, F1=0.8182
- Epoch 5: Loss=0.1097, mIoU=0.6867, F1=0.7764
- Epoch 6: Loss=0.1043, mIoU=0.7180, F1=0.8072
- Epoch 7: Loss=0.1012, mIoU=0.7476, F1=0.8342
- Epoch 8: Loss=0.0964, mIoU=0.7226, F1=0.8114
- Epoch 9: Loss=0.0950, mIoU=0.7427, F1=0.8297
- Epoch 10: Loss=0.0910, mIoU=0.7448, F1=0.8315

当前轮次最佳结果: Round=7, Labeled=679, mIoU=0.7476, F1=0.8342

## Round 8

Labeled Pool Size: 767

- Epoch 1: Loss=0.2431, mIoU=0.6637, F1=0.7532
- Epoch 2: Loss=0.1380, mIoU=0.6582, F1=0.7460
- Epoch 3: Loss=0.1198, mIoU=0.6813, F1=0.7708
- Epoch 4: Loss=0.1126, mIoU=0.7052, F1=0.7950
- Epoch 5: Loss=0.1032, mIoU=0.7132, F1=0.8027
- Epoch 6: Loss=0.0978, mIoU=0.7323, F1=0.8205
- Epoch 7: Loss=0.0952, mIoU=0.7528, F1=0.8387
- Epoch 8: Loss=0.0920, mIoU=0.7455, F1=0.8322
- Epoch 9: Loss=0.0891, mIoU=0.7301, F1=0.8183
- Epoch 10: Loss=0.0859, mIoU=0.7505, F1=0.8366

当前轮次最佳结果: Round=8, Labeled=767, mIoU=0.7528, F1=0.8387

## Round 9

Labeled Pool Size: 855

- Epoch 1: Loss=0.2320, mIoU=0.6684, F1=0.7576
- Epoch 2: Loss=0.1302, mIoU=0.6963, F1=0.7864
- Epoch 3: Loss=0.1121, mIoU=0.6975, F1=0.7874
- Epoch 4: Loss=0.1069, mIoU=0.6670, F1=0.7556
- Epoch 5: Loss=0.1004, mIoU=0.7258, F1=0.8145
- Epoch 6: Loss=0.0936, mIoU=0.7194, F1=0.8084
- Epoch 7: Loss=0.0913, mIoU=0.7521, F1=0.8381
- Epoch 8: Loss=0.0865, mIoU=0.7420, F1=0.8290
- Epoch 9: Loss=0.0839, mIoU=0.7412, F1=0.8284
- Epoch 10: Loss=0.0816, mIoU=0.7434, F1=0.8302

当前轮次最佳结果: Round=9, Labeled=855, mIoU=0.7521, F1=0.8381

## Round 10

Labeled Pool Size: 943

- Epoch 1: Loss=0.2703, mIoU=0.6659, F1=0.7551
- Epoch 2: Loss=0.1295, mIoU=0.6982, F1=0.7883
- Epoch 3: Loss=0.1101, mIoU=0.6918, F1=0.7817
- Epoch 4: Loss=0.1022, mIoU=0.6819, F1=0.7715
- Epoch 5: Loss=0.0955, mIoU=0.7415, F1=0.8286
- Epoch 6: Loss=0.0900, mIoU=0.7309, F1=0.8191
- Epoch 7: Loss=0.0854, mIoU=0.7501, F1=0.8362
- Epoch 8: Loss=0.0829, mIoU=0.7424, F1=0.8294
- Epoch 9: Loss=0.0803, mIoU=0.7397, F1=0.8270
- Epoch 10: Loss=0.0772, mIoU=0.7510, F1=0.8369

当前轮次最佳结果: Round=10, Labeled=943, mIoU=0.7510, F1=0.8369

## Round 11

Labeled Pool Size: 1031

- Epoch 1: Loss=0.1931, mIoU=0.6682, F1=0.7577
- Epoch 2: Loss=0.1142, mIoU=0.6886, F1=0.7786
- Epoch 3: Loss=0.1009, mIoU=0.7237, F1=0.8127
- Epoch 4: Loss=0.0938, mIoU=0.7425, F1=0.8297
- Epoch 5: Loss=0.0872, mIoU=0.7231, F1=0.8120
- Epoch 6: Loss=0.0828, mIoU=0.7390, F1=0.8264
- Epoch 7: Loss=0.0788, mIoU=0.7170, F1=0.8062
- Epoch 8: Loss=0.0785, mIoU=0.7579, F1=0.8429
- Epoch 9: Loss=0.0743, mIoU=0.7589, F1=0.8437
- Epoch 10: Loss=0.0729, mIoU=0.7483, F1=0.8346

当前轮次最佳结果: Round=11, Labeled=1031, mIoU=0.7589, F1=0.8437

## Round 12

Labeled Pool Size: 1119

- Epoch 1: Loss=0.2454, mIoU=0.5973, F1=0.6719
- Epoch 2: Loss=0.1138, mIoU=0.6896, F1=0.7795
- Epoch 3: Loss=0.0990, mIoU=0.7010, F1=0.7908
- Epoch 4: Loss=0.0924, mIoU=0.7204, F1=0.8095
- Epoch 5: Loss=0.0846, mIoU=0.7386, F1=0.8263
- Epoch 6: Loss=0.0810, mIoU=0.7378, F1=0.8253
- Epoch 7: Loss=0.0773, mIoU=0.7477, F1=0.8340
- Epoch 8: Loss=0.0747, mIoU=0.7519, F1=0.8380
- Epoch 9: Loss=0.0725, mIoU=0.7531, F1=0.8388
- Epoch 10: Loss=0.0709, mIoU=0.7477, F1=0.8340

当前轮次最佳结果: Round=12, Labeled=1119, mIoU=0.7531, F1=0.8388

## Round 13

Labeled Pool Size: 1207

- Epoch 1: Loss=0.2120, mIoU=0.6209, F1=0.7024
- Epoch 2: Loss=0.1050, mIoU=0.6996, F1=0.7895
- Epoch 3: Loss=0.0951, mIoU=0.6751, F1=0.7643
- Epoch 4: Loss=0.0867, mIoU=0.7290, F1=0.8174
- Epoch 5: Loss=0.0835, mIoU=0.7416, F1=0.8288
- Epoch 6: Loss=0.0779, mIoU=0.7248, F1=0.8135
- Epoch 7: Loss=0.0769, mIoU=0.7351, F1=0.8230
- Epoch 8: Loss=0.0743, mIoU=0.7391, F1=0.8265
- Epoch 9: Loss=0.0710, mIoU=0.7651, F1=0.8492
- Epoch 10: Loss=0.0683, mIoU=0.7574, F1=0.8425

当前轮次最佳结果: Round=13, Labeled=1207, mIoU=0.7651, F1=0.8492

## Round 14

Labeled Pool Size: 1295

- Epoch 1: Loss=0.2143, mIoU=0.6378, F1=0.7227
- Epoch 2: Loss=0.1032, mIoU=0.7091, F1=0.7987
- Epoch 3: Loss=0.0902, mIoU=0.7388, F1=0.8265
- Epoch 4: Loss=0.0829, mIoU=0.7312, F1=0.8194
- Epoch 5: Loss=0.0787, mIoU=0.7311, F1=0.8192
- Epoch 6: Loss=0.0746, mIoU=0.7538, F1=0.8395
- Epoch 7: Loss=0.0735, mIoU=0.7549, F1=0.8403
- Epoch 8: Loss=0.0693, mIoU=0.7456, F1=0.8322
- Epoch 9: Loss=0.0679, mIoU=0.7663, F1=0.8501
- Epoch 10: Loss=0.0647, mIoU=0.7572, F1=0.8422

当前轮次最佳结果: Round=14, Labeled=1295, mIoU=0.7663, F1=0.8501

## Round 15

Labeled Pool Size: 1383

- Epoch 1: Loss=0.1975, mIoU=0.6083, F1=0.6865
- Epoch 2: Loss=0.0982, mIoU=0.6859, F1=0.7757
- Epoch 3: Loss=0.0835, mIoU=0.7293, F1=0.8177
- Epoch 4: Loss=0.0761, mIoU=0.7480, F1=0.8347
- Epoch 5: Loss=0.0733, mIoU=0.7298, F1=0.8181
- Epoch 6: Loss=0.0688, mIoU=0.7358, F1=0.8234
- Epoch 7: Loss=0.0664, mIoU=0.7613, F1=0.8458
- Epoch 8: Loss=0.0648, mIoU=0.7434, F1=0.8304
- Epoch 9: Loss=0.0621, mIoU=0.7488, F1=0.8349
- Epoch 10: Loss=0.0594, mIoU=0.7493, F1=0.8354

当前轮次最佳结果: Round=15, Labeled=1383, mIoU=0.7613, F1=0.8458


## 实验汇总

预算历史: [151, 239, 327, 415, 503, 591, 679, 767, 855, 943, 1031, 1119, 1207, 1295, 1383]
ALC: 0.6687
最终 mIoU: 0.7613
最终 F1: 0.8458
