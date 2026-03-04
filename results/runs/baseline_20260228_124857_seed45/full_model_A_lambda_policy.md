# 实验日志

实验名称: full_model_A_lambda_policy
描述: 对照A / AAL-SD (Full)：LLM 参与决策与解释；λ由 warmup+风险闭环策略生成并在工具层 clamp（启用CI阈值+AND严重判定+EMA/冷却/限步长；默认禁止 set_lambda；不调整query_size/epochs）
开始时间: 2026-03-03T22:38:18.837360

## Round 1

Labeled Pool Size: 151

- Epoch 1: Loss=0.4860, mIoU=0.4929, F1=0.5253
- Epoch 2: Loss=0.2783, mIoU=0.5455, F1=0.5995
- Epoch 3: Loss=0.1822, mIoU=0.5477, F1=0.6022
- Epoch 4: Loss=0.1260, mIoU=0.5825, F1=0.6538
- Epoch 5: Loss=0.0998, mIoU=0.5853, F1=0.6565
- Epoch 6: Loss=0.0842, mIoU=0.5956, F1=0.6700
- Epoch 7: Loss=0.0721, mIoU=0.5887, F1=0.6607
- Epoch 8: Loss=0.0645, mIoU=0.6106, F1=0.6896
- Epoch 9: Loss=0.0608, mIoU=0.6118, F1=0.6910
- Epoch 10: Loss=0.0550, mIoU=0.6870, F1=0.7777

当前轮次最佳结果: Round=1, Labeled=151, mIoU=0.6870, F1=0.7777

## Round 2

Labeled Pool Size: 239

- Epoch 1: Loss=0.4329, mIoU=0.5823, F1=0.6544
- Epoch 2: Loss=0.2446, mIoU=0.5760, F1=0.6425
- Epoch 3: Loss=0.1801, mIoU=0.6129, F1=0.6925
- Epoch 4: Loss=0.1512, mIoU=0.5909, F1=0.6632
- Epoch 5: Loss=0.1360, mIoU=0.5986, F1=0.6735
- Epoch 6: Loss=0.1258, mIoU=0.6869, F1=0.7771
- Epoch 7: Loss=0.1239, mIoU=0.6743, F1=0.7638
- Epoch 8: Loss=0.1138, mIoU=0.6503, F1=0.7372
- Epoch 9: Loss=0.1110, mIoU=0.6760, F1=0.7658
- Epoch 10: Loss=0.1055, mIoU=0.7041, F1=0.7943

当前轮次最佳结果: Round=2, Labeled=239, mIoU=0.7041, F1=0.7943

## Round 3

Labeled Pool Size: 327

- Epoch 1: Loss=0.4881, mIoU=0.6115, F1=0.6951
- Epoch 2: Loss=0.2235, mIoU=0.6269, F1=0.7101
- Epoch 3: Loss=0.1723, mIoU=0.6264, F1=0.7093
- Epoch 4: Loss=0.1495, mIoU=0.6784, F1=0.7682
- Epoch 5: Loss=0.1469, mIoU=0.6823, F1=0.7723
- Epoch 6: Loss=0.1369, mIoU=0.6552, F1=0.7428
- Epoch 7: Loss=0.1282, mIoU=0.7192, F1=0.8087
- Epoch 8: Loss=0.1276, mIoU=0.6359, F1=0.7204
- Epoch 9: Loss=0.1189, mIoU=0.7007, F1=0.7906
- Epoch 10: Loss=0.1151, mIoU=0.7082, F1=0.7979

当前轮次最佳结果: Round=3, Labeled=327, mIoU=0.7192, F1=0.8087

## Round 4

Labeled Pool Size: 415

- Epoch 1: Loss=0.4117, mIoU=0.5793, F1=0.6472
- Epoch 2: Loss=0.1993, mIoU=0.6194, F1=0.7006
- Epoch 3: Loss=0.1575, mIoU=0.6649, F1=0.7536
- Epoch 4: Loss=0.1399, mIoU=0.6713, F1=0.7604
- Epoch 5: Loss=0.1276, mIoU=0.6802, F1=0.7699
- Epoch 6: Loss=0.1186, mIoU=0.7053, F1=0.7952
- Epoch 7: Loss=0.1159, mIoU=0.7071, F1=0.7969
- Epoch 8: Loss=0.1154, mIoU=0.6930, F1=0.7829
- Epoch 9: Loss=0.1074, mIoU=0.7000, F1=0.7898
- Epoch 10: Loss=0.1042, mIoU=0.7040, F1=0.7937

当前轮次最佳结果: Round=4, Labeled=415, mIoU=0.7071, F1=0.7969

## Round 5

Labeled Pool Size: 503

- Epoch 1: Loss=0.4506, mIoU=0.6057, F1=0.6835
- Epoch 2: Loss=0.1962, mIoU=0.6330, F1=0.7171
- Epoch 3: Loss=0.1583, mIoU=0.6504, F1=0.7373
- Epoch 4: Loss=0.1382, mIoU=0.6895, F1=0.7794
- Epoch 5: Loss=0.1275, mIoU=0.6776, F1=0.7670
- Epoch 6: Loss=0.1213, mIoU=0.7119, F1=0.8014
- Epoch 7: Loss=0.1163, mIoU=0.7379, F1=0.8261
- Epoch 8: Loss=0.1125, mIoU=0.7137, F1=0.8033
- Epoch 9: Loss=0.1086, mIoU=0.7359, F1=0.8238
- Epoch 10: Loss=0.1057, mIoU=0.7356, F1=0.8236

当前轮次最佳结果: Round=5, Labeled=503, mIoU=0.7379, F1=0.8261

## Round 6

Labeled Pool Size: 591

- Epoch 1: Loss=0.3998, mIoU=0.5899, F1=0.6620
- Epoch 2: Loss=0.1721, mIoU=0.6615, F1=0.7499
- Epoch 3: Loss=0.1384, mIoU=0.6572, F1=0.7449
- Epoch 4: Loss=0.1279, mIoU=0.6747, F1=0.7638
- Epoch 5: Loss=0.1211, mIoU=0.7328, F1=0.8210
- Epoch 6: Loss=0.1138, mIoU=0.6988, F1=0.7886
- Epoch 7: Loss=0.1085, mIoU=0.7441, F1=0.8312
- Epoch 8: Loss=0.1055, mIoU=0.7365, F1=0.8243
- Epoch 9: Loss=0.0997, mIoU=0.7239, F1=0.8126
- Epoch 10: Loss=0.0956, mIoU=0.7130, F1=0.8024

当前轮次最佳结果: Round=6, Labeled=591, mIoU=0.7441, F1=0.8312

## Round 7

Labeled Pool Size: 679

- Epoch 1: Loss=0.2744, mIoU=0.5952, F1=0.6692
- Epoch 2: Loss=0.1471, mIoU=0.6120, F1=0.6909
- Epoch 3: Loss=0.1252, mIoU=0.7117, F1=0.8014
- Epoch 4: Loss=0.1181, mIoU=0.6915, F1=0.7814
- Epoch 5: Loss=0.1107, mIoU=0.7119, F1=0.8015
- Epoch 6: Loss=0.1057, mIoU=0.6455, F1=0.7316
- Epoch 7: Loss=0.1019, mIoU=0.7477, F1=0.8342
- Epoch 8: Loss=0.0952, mIoU=0.6972, F1=0.7869
- Epoch 9: Loss=0.0949, mIoU=0.7264, F1=0.8150
- Epoch 10: Loss=0.0894, mIoU=0.7499, F1=0.8360

当前轮次最佳结果: Round=7, Labeled=679, mIoU=0.7499, F1=0.8360

## Round 8

Labeled Pool Size: 767

- Epoch 1: Loss=0.2407, mIoU=0.6435, F1=0.7295
- Epoch 2: Loss=0.1363, mIoU=0.6741, F1=0.7634
- Epoch 3: Loss=0.1196, mIoU=0.6934, F1=0.7833
- Epoch 4: Loss=0.1096, mIoU=0.7131, F1=0.8025
- Epoch 5: Loss=0.1038, mIoU=0.7083, F1=0.7979
- Epoch 6: Loss=0.0982, mIoU=0.7386, F1=0.8261
- Epoch 7: Loss=0.0965, mIoU=0.7274, F1=0.8158
- Epoch 8: Loss=0.0898, mIoU=0.7404, F1=0.8277
- Epoch 9: Loss=0.0868, mIoU=0.7476, F1=0.8341
- Epoch 10: Loss=0.0837, mIoU=0.7467, F1=0.8333

当前轮次最佳结果: Round=8, Labeled=767, mIoU=0.7476, F1=0.8341

## Round 9

Labeled Pool Size: 855

- Epoch 1: Loss=0.2309, mIoU=0.6572, F1=0.7453
- Epoch 2: Loss=0.1305, mIoU=0.6782, F1=0.7678
- Epoch 3: Loss=0.1133, mIoU=0.6937, F1=0.7835
- Epoch 4: Loss=0.1013, mIoU=0.7244, F1=0.8133
- Epoch 5: Loss=0.0967, mIoU=0.7117, F1=0.8012
- Epoch 6: Loss=0.0933, mIoU=0.7411, F1=0.8284
- Epoch 7: Loss=0.0903, mIoU=0.7514, F1=0.8373
- Epoch 8: Loss=0.0862, mIoU=0.7347, F1=0.8225
- Epoch 9: Loss=0.0827, mIoU=0.7586, F1=0.8435
- Epoch 10: Loss=0.0817, mIoU=0.7343, F1=0.8222

当前轮次最佳结果: Round=9, Labeled=855, mIoU=0.7586, F1=0.8435

## Round 10

Labeled Pool Size: 943

- Epoch 1: Loss=0.2732, mIoU=0.6155, F1=0.6958
- Epoch 2: Loss=0.1305, mIoU=0.6315, F1=0.7151
- Epoch 3: Loss=0.1121, mIoU=0.7091, F1=0.7988
- Epoch 4: Loss=0.1031, mIoU=0.7183, F1=0.8076
- Epoch 5: Loss=0.0956, mIoU=0.7420, F1=0.8294
- Epoch 6: Loss=0.0914, mIoU=0.7161, F1=0.8053
- Epoch 7: Loss=0.0860, mIoU=0.7498, F1=0.8360
- Epoch 8: Loss=0.0839, mIoU=0.7215, F1=0.8104
- Epoch 9: Loss=0.0804, mIoU=0.7414, F1=0.8285
- Epoch 10: Loss=0.0779, mIoU=0.7383, F1=0.8257

当前轮次最佳结果: Round=10, Labeled=943, mIoU=0.7498, F1=0.8360

## Round 11

Labeled Pool Size: 1031

- Epoch 1: Loss=0.1968, mIoU=0.6544, F1=0.7439
- Epoch 2: Loss=0.1153, mIoU=0.6819, F1=0.7718
- Epoch 3: Loss=0.1031, mIoU=0.7042, F1=0.7940
- Epoch 4: Loss=0.0940, mIoU=0.7467, F1=0.8336
- Epoch 5: Loss=0.0886, mIoU=0.7317, F1=0.8200
- Epoch 6: Loss=0.0844, mIoU=0.7110, F1=0.8005
- Epoch 7: Loss=0.0802, mIoU=0.7368, F1=0.8243
- Epoch 8: Loss=0.0783, mIoU=0.7488, F1=0.8350
- Epoch 9: Loss=0.0761, mIoU=0.7545, F1=0.8399
- Epoch 10: Loss=0.0734, mIoU=0.7599, F1=0.8446

当前轮次最佳结果: Round=11, Labeled=1031, mIoU=0.7599, F1=0.8446

## Round 12

Labeled Pool Size: 1119

- Epoch 1: Loss=0.2346, mIoU=0.6278, F1=0.7109
- Epoch 2: Loss=0.1148, mIoU=0.6987, F1=0.7887
- Epoch 3: Loss=0.0999, mIoU=0.7162, F1=0.8055
- Epoch 4: Loss=0.0930, mIoU=0.7412, F1=0.8286
- Epoch 5: Loss=0.0871, mIoU=0.7434, F1=0.8304
- Epoch 6: Loss=0.0854, mIoU=0.7214, F1=0.8102
- Epoch 7: Loss=0.0807, mIoU=0.7358, F1=0.8235
- Epoch 8: Loss=0.0773, mIoU=0.7527, F1=0.8385
- Epoch 9: Loss=0.0741, mIoU=0.7307, F1=0.8190
- Epoch 10: Loss=0.0725, mIoU=0.7332, F1=0.8211

当前轮次最佳结果: Round=12, Labeled=1119, mIoU=0.7527, F1=0.8385

## Round 13

Labeled Pool Size: 1207

- Epoch 1: Loss=0.2216, mIoU=0.6139, F1=0.6936
- Epoch 2: Loss=0.1090, mIoU=0.6899, F1=0.7797
- Epoch 3: Loss=0.0942, mIoU=0.7140, F1=0.8034
- Epoch 4: Loss=0.0869, mIoU=0.7423, F1=0.8300
- Epoch 5: Loss=0.0834, mIoU=0.7527, F1=0.8386
- Epoch 6: Loss=0.0780, mIoU=0.7529, F1=0.8388
- Epoch 7: Loss=0.0750, mIoU=0.7475, F1=0.8338
- Epoch 8: Loss=0.0721, mIoU=0.7370, F1=0.8245
- Epoch 9: Loss=0.0710, mIoU=0.7545, F1=0.8399
- Epoch 10: Loss=0.0686, mIoU=0.7366, F1=0.8243

当前轮次最佳结果: Round=13, Labeled=1207, mIoU=0.7545, F1=0.8399

## Round 14

Labeled Pool Size: 1295

- Epoch 1: Loss=0.2192, mIoU=0.6552, F1=0.7431
- Epoch 2: Loss=0.1031, mIoU=0.6966, F1=0.7865
- Epoch 3: Loss=0.0893, mIoU=0.7213, F1=0.8103
- Epoch 4: Loss=0.0836, mIoU=0.7551, F1=0.8406
- Epoch 5: Loss=0.0784, mIoU=0.7238, F1=0.8125
- Epoch 6: Loss=0.0742, mIoU=0.6963, F1=0.7861
- Epoch 7: Loss=0.0725, mIoU=0.7201, F1=0.8092
- Epoch 8: Loss=0.0720, mIoU=0.7428, F1=0.8298
- Epoch 9: Loss=0.0674, mIoU=0.7524, F1=0.8381
- Epoch 10: Loss=0.0646, mIoU=0.7603, F1=0.8449

当前轮次最佳结果: Round=14, Labeled=1295, mIoU=0.7603, F1=0.8449

## Round 15

Labeled Pool Size: 1383

- Epoch 1: Loss=0.1906, mIoU=0.6433, F1=0.7292
- Epoch 2: Loss=0.0967, mIoU=0.7096, F1=0.7994
- Epoch 3: Loss=0.0850, mIoU=0.6619, F1=0.7500
- Epoch 4: Loss=0.0792, mIoU=0.7486, F1=0.8351
- Epoch 5: Loss=0.0765, mIoU=0.7495, F1=0.8358
- Epoch 6: Loss=0.0719, mIoU=0.7497, F1=0.8357
- Epoch 7: Loss=0.0690, mIoU=0.7640, F1=0.8481
- Epoch 8: Loss=0.0666, mIoU=0.7643, F1=0.8482
- Epoch 9: Loss=0.0641, mIoU=0.7571, F1=0.8420
- Epoch 10: Loss=0.0630, mIoU=0.7590, F1=0.8438

当前轮次最佳结果: Round=15, Labeled=1383, mIoU=0.7643, F1=0.8482


## 实验汇总

预算历史: [151, 239, 327, 415, 503, 591, 679, 767, 855, 943, 1031, 1119, 1207, 1295, 1383]
ALC: 0.6693
最终 mIoU: 0.7643
最终 F1: 0.8482
