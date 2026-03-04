# 实验日志

实验名称: full_model_B_lambda_agent
描述: 对照B：授权LLM/Agent 在显式约束下逐轮 set_lambda（并记录到trace）；启用CI阈值+AND严重判定+回撤禁升+EMA/冷却/限步长；禁止使用policy自动填充本轮λ（必须显式set_lambda）；不调整query_size/epochs
开始时间: 2026-03-03T19:01:13.682928

## Round 1

Labeled Pool Size: 151

- Epoch 1: Loss=0.4821, mIoU=0.5015, F1=0.5427
- Epoch 2: Loss=0.2763, mIoU=0.5702, F1=0.6374
- Epoch 3: Loss=0.1775, mIoU=0.5932, F1=0.6695
- Epoch 4: Loss=0.1302, mIoU=0.5968, F1=0.6733
- Epoch 5: Loss=0.1009, mIoU=0.6168, F1=0.6987
- Epoch 6: Loss=0.0863, mIoU=0.6368, F1=0.7241
- Epoch 7: Loss=0.0720, mIoU=0.6346, F1=0.7202
- Epoch 8: Loss=0.0666, mIoU=0.6251, F1=0.7085
- Epoch 9: Loss=0.0647, mIoU=0.6476, F1=0.7361
- Epoch 10: Loss=0.0563, mIoU=0.6330, F1=0.7176

当前轮次最佳结果: Round=1, Labeled=151, mIoU=0.6476, F1=0.7361

## Round 2

Labeled Pool Size: 239

- Epoch 1: Loss=0.4474, mIoU=0.5286, F1=0.5735
- Epoch 2: Loss=0.2454, mIoU=0.5937, F1=0.6685
- Epoch 3: Loss=0.1697, mIoU=0.6221, F1=0.7046
- Epoch 4: Loss=0.1479, mIoU=0.6699, F1=0.7604
- Epoch 5: Loss=0.1244, mIoU=0.6371, F1=0.7222
- Epoch 6: Loss=0.1172, mIoU=0.6149, F1=0.6952
- Epoch 7: Loss=0.1127, mIoU=0.6538, F1=0.7417
- Epoch 8: Loss=0.1061, mIoU=0.6681, F1=0.7572
- Epoch 9: Loss=0.1037, mIoU=0.6744, F1=0.7642
- Epoch 10: Loss=0.0938, mIoU=0.6830, F1=0.7732

当前轮次最佳结果: Round=2, Labeled=239, mIoU=0.6830, F1=0.7732

## Round 3

Labeled Pool Size: 327

- Epoch 1: Loss=0.3960, mIoU=0.5383, F1=0.5855
- Epoch 2: Loss=0.2095, mIoU=0.5487, F1=0.6016
- Epoch 3: Loss=0.1617, mIoU=0.5976, F1=0.6724
- Epoch 4: Loss=0.1390, mIoU=0.6230, F1=0.7052
- Epoch 5: Loss=0.1285, mIoU=0.6454, F1=0.7323
- Epoch 6: Loss=0.1210, mIoU=0.6894, F1=0.7798
- Epoch 7: Loss=0.1103, mIoU=0.6758, F1=0.7654
- Epoch 8: Loss=0.1082, mIoU=0.7047, F1=0.7951
- Epoch 9: Loss=0.1038, mIoU=0.6873, F1=0.7773
- Epoch 10: Loss=0.0965, mIoU=0.6761, F1=0.7656

当前轮次最佳结果: Round=3, Labeled=327, mIoU=0.7047, F1=0.7951

## Round 4

Labeled Pool Size: 415

- Epoch 1: Loss=0.4454, mIoU=0.6303, F1=0.7146
- Epoch 2: Loss=0.2066, mIoU=0.6472, F1=0.7340
- Epoch 3: Loss=0.1596, mIoU=0.6763, F1=0.7660
- Epoch 4: Loss=0.1444, mIoU=0.6794, F1=0.7691
- Epoch 5: Loss=0.1309, mIoU=0.6349, F1=0.7193
- Epoch 6: Loss=0.1234, mIoU=0.7093, F1=0.7993
- Epoch 7: Loss=0.1199, mIoU=0.7037, F1=0.7939
- Epoch 8: Loss=0.1135, mIoU=0.7244, F1=0.8137
- Epoch 9: Loss=0.1074, mIoU=0.7137, F1=0.8034
- Epoch 10: Loss=0.1093, mIoU=0.7129, F1=0.8026

当前轮次最佳结果: Round=4, Labeled=415, mIoU=0.7244, F1=0.8137

## Round 5

Labeled Pool Size: 503

- Epoch 1: Loss=0.3753, mIoU=0.6072, F1=0.6853
- Epoch 2: Loss=0.1783, mIoU=0.6607, F1=0.7492
- Epoch 3: Loss=0.1465, mIoU=0.6376, F1=0.7226
- Epoch 4: Loss=0.1311, mIoU=0.6373, F1=0.7222
- Epoch 5: Loss=0.1205, mIoU=0.7103, F1=0.8002
- Epoch 6: Loss=0.1144, mIoU=0.7252, F1=0.8144
- Epoch 7: Loss=0.1082, mIoU=0.7126, F1=0.8022
- Epoch 8: Loss=0.1064, mIoU=0.7129, F1=0.8025
- Epoch 9: Loss=0.1009, mIoU=0.7098, F1=0.7995
- Epoch 10: Loss=0.0983, mIoU=0.7128, F1=0.8023

当前轮次最佳结果: Round=5, Labeled=503, mIoU=0.7252, F1=0.8144

## Round 6

Labeled Pool Size: 591

- Epoch 1: Loss=0.4187, mIoU=0.6370, F1=0.7227
- Epoch 2: Loss=0.1727, mIoU=0.6543, F1=0.7421
- Epoch 3: Loss=0.1439, mIoU=0.6630, F1=0.7516
- Epoch 4: Loss=0.1295, mIoU=0.7088, F1=0.7989
- Epoch 5: Loss=0.1218, mIoU=0.6987, F1=0.7886
- Epoch 6: Loss=0.1158, mIoU=0.6905, F1=0.7804
- Epoch 7: Loss=0.1122, mIoU=0.6412, F1=0.7266
- Epoch 8: Loss=0.1107, mIoU=0.6805, F1=0.7700
- Epoch 9: Loss=0.1027, mIoU=0.7333, F1=0.8217
- Epoch 10: Loss=0.0988, mIoU=0.7443, F1=0.8313

当前轮次最佳结果: Round=6, Labeled=591, mIoU=0.7443, F1=0.8313

## Round 7

Labeled Pool Size: 679

- Epoch 1: Loss=0.3706, mIoU=0.6222, F1=0.7044
- Epoch 2: Loss=0.1569, mIoU=0.6746, F1=0.7641
- Epoch 3: Loss=0.1334, mIoU=0.6809, F1=0.7707
- Epoch 4: Loss=0.1231, mIoU=0.7049, F1=0.7949
- Epoch 5: Loss=0.1143, mIoU=0.6967, F1=0.7868
- Epoch 6: Loss=0.1104, mIoU=0.7297, F1=0.8185
- Epoch 7: Loss=0.1072, mIoU=0.7007, F1=0.7906
- Epoch 8: Loss=0.1034, mIoU=0.7240, F1=0.8132
- Epoch 9: Loss=0.0998, mIoU=0.7402, F1=0.8277
- Epoch 10: Loss=0.0974, mIoU=0.7280, F1=0.8166

当前轮次最佳结果: Round=7, Labeled=679, mIoU=0.7402, F1=0.8277

## Round 8

Labeled Pool Size: 767

- Epoch 1: Loss=0.2606, mIoU=0.6559, F1=0.7440
- Epoch 2: Loss=0.1393, mIoU=0.6846, F1=0.7749
- Epoch 3: Loss=0.1214, mIoU=0.7178, F1=0.8073
- Epoch 4: Loss=0.1119, mIoU=0.6922, F1=0.7821
- Epoch 5: Loss=0.1028, mIoU=0.7151, F1=0.8045
- Epoch 6: Loss=0.0972, mIoU=0.7275, F1=0.8161
- Epoch 7: Loss=0.0949, mIoU=0.7222, F1=0.8112
- Epoch 8: Loss=0.0923, mIoU=0.7329, F1=0.8210
- Epoch 9: Loss=0.0903, mIoU=0.7115, F1=0.8010
- Epoch 10: Loss=0.0866, mIoU=0.7304, F1=0.8186

当前轮次最佳结果: Round=8, Labeled=767, mIoU=0.7329, F1=0.8210

## Round 9

Labeled Pool Size: 855

- Epoch 1: Loss=0.2276, mIoU=0.5955, F1=0.6697
- Epoch 2: Loss=0.1284, mIoU=0.6571, F1=0.7450
- Epoch 3: Loss=0.1120, mIoU=0.6813, F1=0.7710
- Epoch 4: Loss=0.1041, mIoU=0.6994, F1=0.7894
- Epoch 5: Loss=0.0986, mIoU=0.6967, F1=0.7868
- Epoch 6: Loss=0.0944, mIoU=0.7361, F1=0.8241
- Epoch 7: Loss=0.0931, mIoU=0.7305, F1=0.8189
- Epoch 8: Loss=0.0869, mIoU=0.7324, F1=0.8206
- Epoch 9: Loss=0.0836, mIoU=0.7434, F1=0.8304
- Epoch 10: Loss=0.0799, mIoU=0.7523, F1=0.8384

当前轮次最佳结果: Round=9, Labeled=855, mIoU=0.7523, F1=0.8384

## Round 10

Labeled Pool Size: 943

- Epoch 1: Loss=0.2191, mIoU=0.6514, F1=0.7394
- Epoch 2: Loss=0.1223, mIoU=0.7057, F1=0.7960
- Epoch 3: Loss=0.1057, mIoU=0.7089, F1=0.7987
- Epoch 4: Loss=0.0982, mIoU=0.6956, F1=0.7859
- Epoch 5: Loss=0.0935, mIoU=0.7337, F1=0.8222
- Epoch 6: Loss=0.0914, mIoU=0.7301, F1=0.8185
- Epoch 7: Loss=0.0860, mIoU=0.7421, F1=0.8293
- Epoch 8: Loss=0.0836, mIoU=0.7491, F1=0.8355
- Epoch 9: Loss=0.0806, mIoU=0.7419, F1=0.8290
- Epoch 10: Loss=0.0771, mIoU=0.7407, F1=0.8280

当前轮次最佳结果: Round=10, Labeled=943, mIoU=0.7491, F1=0.8355

## Round 11

Labeled Pool Size: 1031

- Epoch 1: Loss=0.2565, mIoU=0.6262, F1=0.7092
- Epoch 2: Loss=0.1223, mIoU=0.6875, F1=0.7776
- Epoch 3: Loss=0.1055, mIoU=0.6715, F1=0.7606
- Epoch 4: Loss=0.0949, mIoU=0.7339, F1=0.8221
- Epoch 5: Loss=0.0893, mIoU=0.7354, F1=0.8234
- Epoch 6: Loss=0.0871, mIoU=0.7342, F1=0.8224
- Epoch 7: Loss=0.0808, mIoU=0.7129, F1=0.8025
- Epoch 8: Loss=0.0799, mIoU=0.7252, F1=0.8139
- Epoch 9: Loss=0.0768, mIoU=0.7115, F1=0.8010
- Epoch 10: Loss=0.0729, mIoU=0.7262, F1=0.8148

当前轮次最佳结果: Round=11, Labeled=1031, mIoU=0.7354, F1=0.8234

## Round 12

Labeled Pool Size: 1119

- Epoch 1: Loss=0.1875, mIoU=0.6764, F1=0.7661
- Epoch 2: Loss=0.1087, mIoU=0.6813, F1=0.7711
- Epoch 3: Loss=0.0974, mIoU=0.7133, F1=0.8030
- Epoch 4: Loss=0.0897, mIoU=0.6900, F1=0.7798
- Epoch 5: Loss=0.0851, mIoU=0.7255, F1=0.8142
- Epoch 6: Loss=0.0785, mIoU=0.7413, F1=0.8287
- Epoch 7: Loss=0.0765, mIoU=0.7507, F1=0.8369
- Epoch 8: Loss=0.0732, mIoU=0.7393, F1=0.8268
- Epoch 9: Loss=0.0738, mIoU=0.7324, F1=0.8209
- Epoch 10: Loss=0.0686, mIoU=0.7548, F1=0.8404

当前轮次最佳结果: Round=12, Labeled=1119, mIoU=0.7548, F1=0.8404

## Round 13

Labeled Pool Size: 1207

- Epoch 1: Loss=0.2321, mIoU=0.6055, F1=0.6828
- Epoch 2: Loss=0.1088, mIoU=0.7137, F1=0.8034
- Epoch 3: Loss=0.0935, mIoU=0.7376, F1=0.8256
- Epoch 4: Loss=0.0874, mIoU=0.7320, F1=0.8203
- Epoch 5: Loss=0.0839, mIoU=0.7373, F1=0.8254
- Epoch 6: Loss=0.0781, mIoU=0.7099, F1=0.7994
- Epoch 7: Loss=0.0753, mIoU=0.7426, F1=0.8296
- Epoch 8: Loss=0.0712, mIoU=0.7470, F1=0.8336
- Epoch 9: Loss=0.0684, mIoU=0.7357, F1=0.8235
- Epoch 10: Loss=0.0692, mIoU=0.7527, F1=0.8385

当前轮次最佳结果: Round=13, Labeled=1207, mIoU=0.7527, F1=0.8385

## Round 14

Labeled Pool Size: 1295

- Epoch 1: Loss=0.2160, mIoU=0.6759, F1=0.7658
- Epoch 2: Loss=0.1019, mIoU=0.7142, F1=0.8042
- Epoch 3: Loss=0.0864, mIoU=0.7130, F1=0.8028
- Epoch 4: Loss=0.0801, mIoU=0.7299, F1=0.8186
- Epoch 5: Loss=0.0752, mIoU=0.7085, F1=0.7981
- Epoch 6: Loss=0.0717, mIoU=0.7502, F1=0.8365
- Epoch 7: Loss=0.0685, mIoU=0.7349, F1=0.8228
- Epoch 8: Loss=0.0677, mIoU=0.7392, F1=0.8267
- Epoch 9: Loss=0.0641, mIoU=0.7326, F1=0.8206
- Epoch 10: Loss=0.0617, mIoU=0.7533, F1=0.8390

当前轮次最佳结果: Round=14, Labeled=1295, mIoU=0.7533, F1=0.8390

## Round 15

Labeled Pool Size: 1383

- Epoch 1: Loss=0.2084, mIoU=0.6741, F1=0.7643
- Epoch 2: Loss=0.0972, mIoU=0.6790, F1=0.7686
- Epoch 3: Loss=0.0841, mIoU=0.7120, F1=0.8016
- Epoch 4: Loss=0.0775, mIoU=0.7260, F1=0.8146
- Epoch 5: Loss=0.0730, mIoU=0.7479, F1=0.8344
- Epoch 6: Loss=0.0693, mIoU=0.7240, F1=0.8127
- Epoch 7: Loss=0.0660, mIoU=0.7386, F1=0.8260
- Epoch 8: Loss=0.0630, mIoU=0.7600, F1=0.8447
- Epoch 9: Loss=0.0608, mIoU=0.7588, F1=0.8437
- Epoch 10: Loss=0.0585, mIoU=0.7369, F1=0.8245

当前轮次最佳结果: Round=15, Labeled=1383, mIoU=0.7600, F1=0.8447


## 实验汇总

预算历史: [151, 239, 327, 415, 503, 591, 679, 767, 855, 943, 1031, 1119, 1207, 1295, 1383]
ALC: 0.6622
最终 mIoU: 0.7600
最终 F1: 0.8447
