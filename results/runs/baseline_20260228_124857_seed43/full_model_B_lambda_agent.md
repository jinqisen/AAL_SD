# 实验日志

实验名称: full_model_B_lambda_agent
描述: 对照B：授权LLM/Agent 在显式约束下逐轮 set_lambda（并记录到trace）；启用CI阈值+AND严重判定+回撤禁升+EMA/冷却/限步长；禁止使用policy自动填充本轮λ（必须显式set_lambda）；不调整query_size/epochs
开始时间: 2026-03-03T19:01:13.682830

## Round 1

Labeled Pool Size: 151

- Epoch 1: Loss=0.3547, mIoU=0.5033, F1=0.5252
- Epoch 2: Loss=0.2047, mIoU=0.5164, F1=0.5485
- Epoch 3: Loss=0.1378, mIoU=0.5389, F1=0.5870
- Epoch 4: Loss=0.1039, mIoU=0.5559, F1=0.6133
- Epoch 5: Loss=0.0849, mIoU=0.5647, F1=0.6261
- Epoch 6: Loss=0.0734, mIoU=0.6238, F1=0.7071
- Epoch 7: Loss=0.0638, mIoU=0.6209, F1=0.7036
- Epoch 8: Loss=0.0592, mIoU=0.6314, F1=0.7159
- Epoch 9: Loss=0.0573, mIoU=0.6103, F1=0.6892
- Epoch 10: Loss=0.0523, mIoU=0.6151, F1=0.6953

当前轮次最佳结果: Round=1, Labeled=151, mIoU=0.6314, F1=0.7159

## Round 2

Labeled Pool Size: 239

- Epoch 1: Loss=0.4404, mIoU=0.5918, F1=0.6671
- Epoch 2: Loss=0.2407, mIoU=0.5994, F1=0.6745
- Epoch 3: Loss=0.1711, mIoU=0.6316, F1=0.7153
- Epoch 4: Loss=0.1385, mIoU=0.6431, F1=0.7288
- Epoch 5: Loss=0.1299, mIoU=0.6290, F1=0.7120
- Epoch 6: Loss=0.1161, mIoU=0.6593, F1=0.7473
- Epoch 7: Loss=0.1086, mIoU=0.6490, F1=0.7355
- Epoch 8: Loss=0.1019, mIoU=0.6908, F1=0.7808
- Epoch 9: Loss=0.0996, mIoU=0.6782, F1=0.7681
- Epoch 10: Loss=0.0999, mIoU=0.6440, F1=0.7299

当前轮次最佳结果: Round=2, Labeled=239, mIoU=0.6908, F1=0.7808

## Round 3

Labeled Pool Size: 327

- Epoch 1: Loss=0.4074, mIoU=0.5843, F1=0.6550
- Epoch 2: Loss=0.2072, mIoU=0.6280, F1=0.7111
- Epoch 3: Loss=0.1586, mIoU=0.6503, F1=0.7373
- Epoch 4: Loss=0.1403, mIoU=0.6593, F1=0.7472
- Epoch 5: Loss=0.1286, mIoU=0.6978, F1=0.7878
- Epoch 6: Loss=0.1221, mIoU=0.6921, F1=0.7819
- Epoch 7: Loss=0.1135, mIoU=0.6784, F1=0.7677
- Epoch 8: Loss=0.1033, mIoU=0.6980, F1=0.7877
- Epoch 9: Loss=0.1048, mIoU=0.7050, F1=0.7952
- Epoch 10: Loss=0.1027, mIoU=0.6858, F1=0.7753

当前轮次最佳结果: Round=3, Labeled=327, mIoU=0.7050, F1=0.7952

## Round 4

Labeled Pool Size: 415

- Epoch 1: Loss=0.3649, mIoU=0.5989, F1=0.6746
- Epoch 2: Loss=0.1916, mIoU=0.6308, F1=0.7144
- Epoch 3: Loss=0.1528, mIoU=0.6887, F1=0.7786
- Epoch 4: Loss=0.1383, mIoU=0.6954, F1=0.7853
- Epoch 5: Loss=0.1281, mIoU=0.7137, F1=0.8033
- Epoch 6: Loss=0.1230, mIoU=0.7185, F1=0.8078
- Epoch 7: Loss=0.1178, mIoU=0.7073, F1=0.7972
- Epoch 8: Loss=0.1122, mIoU=0.7178, F1=0.8070
- Epoch 9: Loss=0.1086, mIoU=0.7098, F1=0.7994
- Epoch 10: Loss=0.1039, mIoU=0.7260, F1=0.8147

当前轮次最佳结果: Round=4, Labeled=415, mIoU=0.7260, F1=0.8147

## Round 5

Labeled Pool Size: 503

- Epoch 1: Loss=0.4106, mIoU=0.6430, F1=0.7297
- Epoch 2: Loss=0.1863, mIoU=0.6707, F1=0.7596
- Epoch 3: Loss=0.1475, mIoU=0.6849, F1=0.7746
- Epoch 4: Loss=0.1303, mIoU=0.6970, F1=0.7868
- Epoch 5: Loss=0.1255, mIoU=0.7014, F1=0.7912
- Epoch 6: Loss=0.1168, mIoU=0.7246, F1=0.8136
- Epoch 7: Loss=0.1103, mIoU=0.7022, F1=0.7919
- Epoch 8: Loss=0.1079, mIoU=0.7261, F1=0.8147
- Epoch 9: Loss=0.1014, mIoU=0.7085, F1=0.7980
- Epoch 10: Loss=0.0982, mIoU=0.7305, F1=0.8187

当前轮次最佳结果: Round=5, Labeled=503, mIoU=0.7305, F1=0.8187

## Round 6

Labeled Pool Size: 591

- Epoch 1: Loss=0.3461, mIoU=0.6391, F1=0.7244
- Epoch 2: Loss=0.1607, mIoU=0.6862, F1=0.7761
- Epoch 3: Loss=0.1377, mIoU=0.6237, F1=0.7055
- Epoch 4: Loss=0.1275, mIoU=0.7254, F1=0.8146
- Epoch 5: Loss=0.1190, mIoU=0.7137, F1=0.8031
- Epoch 6: Loss=0.1185, mIoU=0.7285, F1=0.8171
- Epoch 7: Loss=0.1092, mIoU=0.7243, F1=0.8130
- Epoch 8: Loss=0.1062, mIoU=0.7337, F1=0.8217
- Epoch 9: Loss=0.1028, mIoU=0.7469, F1=0.8334
- Epoch 10: Loss=0.1000, mIoU=0.7374, F1=0.8249

当前轮次最佳结果: Round=6, Labeled=591, mIoU=0.7469, F1=0.8334

## Round 7

Labeled Pool Size: 679

- Epoch 1: Loss=0.3947, mIoU=0.6241, F1=0.7061
- Epoch 2: Loss=0.1663, mIoU=0.6857, F1=0.7755
- Epoch 3: Loss=0.1362, mIoU=0.6856, F1=0.7752
- Epoch 4: Loss=0.1224, mIoU=0.7226, F1=0.8114
- Epoch 5: Loss=0.1161, mIoU=0.7084, F1=0.7979
- Epoch 6: Loss=0.1096, mIoU=0.6978, F1=0.7874
- Epoch 7: Loss=0.1044, mIoU=0.7301, F1=0.8183
- Epoch 8: Loss=0.1000, mIoU=0.7440, F1=0.8308
- Epoch 9: Loss=0.0987, mIoU=0.7312, F1=0.8193
- Epoch 10: Loss=0.0940, mIoU=0.7457, F1=0.8323

当前轮次最佳结果: Round=7, Labeled=679, mIoU=0.7457, F1=0.8323

## Round 8

Labeled Pool Size: 767

- Epoch 1: Loss=0.3214, mIoU=0.6315, F1=0.7155
- Epoch 2: Loss=0.1498, mIoU=0.6601, F1=0.7482
- Epoch 3: Loss=0.1278, mIoU=0.6626, F1=0.7506
- Epoch 4: Loss=0.1148, mIoU=0.6939, F1=0.7836
- Epoch 5: Loss=0.1118, mIoU=0.7067, F1=0.7963
- Epoch 6: Loss=0.1044, mIoU=0.7296, F1=0.8179
- Epoch 7: Loss=0.0998, mIoU=0.7237, F1=0.8123
- Epoch 8: Loss=0.0946, mIoU=0.7392, F1=0.8268
- Epoch 9: Loss=0.0919, mIoU=0.7138, F1=0.8030
- Epoch 10: Loss=0.0916, mIoU=0.7416, F1=0.8286

当前轮次最佳结果: Round=8, Labeled=767, mIoU=0.7416, F1=0.8286

## Round 9

Labeled Pool Size: 855

- Epoch 1: Loss=0.2475, mIoU=0.6239, F1=0.7062
- Epoch 2: Loss=0.1313, mIoU=0.6794, F1=0.7690
- Epoch 3: Loss=0.1125, mIoU=0.7315, F1=0.8198
- Epoch 4: Loss=0.1074, mIoU=0.6829, F1=0.7723
- Epoch 5: Loss=0.0980, mIoU=0.7244, F1=0.8130
- Epoch 6: Loss=0.0933, mIoU=0.7337, F1=0.8215
- Epoch 7: Loss=0.0901, mIoU=0.7219, F1=0.8106
- Epoch 8: Loss=0.0878, mIoU=0.7498, F1=0.8362
- Epoch 9: Loss=0.0847, mIoU=0.7403, F1=0.8275
- Epoch 10: Loss=0.0818, mIoU=0.7492, F1=0.8356

当前轮次最佳结果: Round=9, Labeled=855, mIoU=0.7498, F1=0.8362

## Round 10

Labeled Pool Size: 943

- Epoch 1: Loss=0.2172, mIoU=0.5959, F1=0.6698
- Epoch 2: Loss=0.1237, mIoU=0.6693, F1=0.7580
- Epoch 3: Loss=0.1069, mIoU=0.6919, F1=0.7815
- Epoch 4: Loss=0.0990, mIoU=0.7313, F1=0.8195
- Epoch 5: Loss=0.0934, mIoU=0.7248, F1=0.8134
- Epoch 6: Loss=0.0890, mIoU=0.7095, F1=0.7989
- Epoch 7: Loss=0.0840, mIoU=0.7223, F1=0.8111
- Epoch 8: Loss=0.0815, mIoU=0.7263, F1=0.8147
- Epoch 9: Loss=0.0789, mIoU=0.7469, F1=0.8332
- Epoch 10: Loss=0.0771, mIoU=0.7485, F1=0.8347

当前轮次最佳结果: Round=10, Labeled=943, mIoU=0.7485, F1=0.8347

## Round 11

Labeled Pool Size: 1031

- Epoch 1: Loss=0.2094, mIoU=0.6565, F1=0.7444
- Epoch 2: Loss=0.1159, mIoU=0.6405, F1=0.7255
- Epoch 3: Loss=0.1003, mIoU=0.6956, F1=0.7853
- Epoch 4: Loss=0.0929, mIoU=0.6881, F1=0.7777
- Epoch 5: Loss=0.0891, mIoU=0.7391, F1=0.8265
- Epoch 6: Loss=0.0842, mIoU=0.7116, F1=0.8008
- Epoch 7: Loss=0.0809, mIoU=0.7522, F1=0.8381
- Epoch 8: Loss=0.0781, mIoU=0.7285, F1=0.8167
- Epoch 9: Loss=0.0733, mIoU=0.7261, F1=0.8146
- Epoch 10: Loss=0.0711, mIoU=0.7486, F1=0.8347

当前轮次最佳结果: Round=11, Labeled=1031, mIoU=0.7522, F1=0.8381

## Round 12

Labeled Pool Size: 1119

- Epoch 1: Loss=0.2462, mIoU=0.6328, F1=0.7168
- Epoch 2: Loss=0.1133, mIoU=0.7177, F1=0.8074
- Epoch 3: Loss=0.0988, mIoU=0.7335, F1=0.8218
- Epoch 4: Loss=0.0916, mIoU=0.7412, F1=0.8285
- Epoch 5: Loss=0.0847, mIoU=0.7352, F1=0.8229
- Epoch 6: Loss=0.0802, mIoU=0.7361, F1=0.8237
- Epoch 7: Loss=0.0772, mIoU=0.7104, F1=0.7998
- Epoch 8: Loss=0.0747, mIoU=0.6877, F1=0.7771
- Epoch 9: Loss=0.0733, mIoU=0.7479, F1=0.8341
- Epoch 10: Loss=0.0697, mIoU=0.7389, F1=0.8261

当前轮次最佳结果: Round=12, Labeled=1119, mIoU=0.7479, F1=0.8341

## Round 13

Labeled Pool Size: 1207

- Epoch 1: Loss=0.1807, mIoU=0.6487, F1=0.7353
- Epoch 2: Loss=0.1032, mIoU=0.7005, F1=0.7904
- Epoch 3: Loss=0.0903, mIoU=0.7052, F1=0.7948
- Epoch 4: Loss=0.0832, mIoU=0.7408, F1=0.8282
- Epoch 5: Loss=0.0786, mIoU=0.7050, F1=0.7944
- Epoch 6: Loss=0.0755, mIoU=0.7330, F1=0.8208
- Epoch 7: Loss=0.0734, mIoU=0.7311, F1=0.8191
- Epoch 8: Loss=0.0696, mIoU=0.7535, F1=0.8390
- Epoch 9: Loss=0.0671, mIoU=0.7569, F1=0.8422
- Epoch 10: Loss=0.0646, mIoU=0.7541, F1=0.8395

当前轮次最佳结果: Round=13, Labeled=1207, mIoU=0.7569, F1=0.8422

## Round 14

Labeled Pool Size: 1295

- Epoch 1: Loss=0.2260, mIoU=0.6588, F1=0.7468
- Epoch 2: Loss=0.1036, mIoU=0.7119, F1=0.8022
- Epoch 3: Loss=0.0887, mIoU=0.7287, F1=0.8172
- Epoch 4: Loss=0.0836, mIoU=0.6919, F1=0.7815
- Epoch 5: Loss=0.0788, mIoU=0.7367, F1=0.8244
- Epoch 6: Loss=0.0740, mIoU=0.7129, F1=0.8022
- Epoch 7: Loss=0.0701, mIoU=0.7528, F1=0.8384
- Epoch 8: Loss=0.0688, mIoU=0.7555, F1=0.8408
- Epoch 9: Loss=0.0672, mIoU=0.7556, F1=0.8409
- Epoch 10: Loss=0.0641, mIoU=0.7572, F1=0.8421

当前轮次最佳结果: Round=14, Labeled=1295, mIoU=0.7572, F1=0.8421

## Round 15

Labeled Pool Size: 1383

- Epoch 1: Loss=0.2061, mIoU=0.6722, F1=0.7612
- Epoch 2: Loss=0.0965, mIoU=0.6862, F1=0.7757
- Epoch 3: Loss=0.0853, mIoU=0.6793, F1=0.7686
- Epoch 4: Loss=0.0789, mIoU=0.7091, F1=0.7985
- Epoch 5: Loss=0.0750, mIoU=0.7466, F1=0.8332
- Epoch 6: Loss=0.0698, mIoU=0.7468, F1=0.8332
- Epoch 7: Loss=0.0670, mIoU=0.7610, F1=0.8455
- Epoch 8: Loss=0.0634, mIoU=0.7618, F1=0.8461
- Epoch 9: Loss=0.0614, mIoU=0.7578, F1=0.8427
- Epoch 10: Loss=0.0608, mIoU=0.7473, F1=0.8336

当前轮次最佳结果: Round=15, Labeled=1383, mIoU=0.7618, F1=0.8461


## 实验汇总

预算历史: [151, 239, 327, 415, 503, 591, 679, 767, 855, 943, 1031, 1119, 1207, 1295, 1383]
ALC: 0.6647
最终 mIoU: 0.7618
最终 F1: 0.8461
