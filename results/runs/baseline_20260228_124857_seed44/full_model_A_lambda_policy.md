# 实验日志

实验名称: full_model_A_lambda_policy
描述: 对照A / AAL-SD (Full)：LLM 参与决策与解释；λ由 warmup+风险闭环策略生成并在工具层 clamp（启用CI阈值+AND严重判定+EMA/冷却/限步长；默认禁止 set_lambda；不调整query_size/epochs）
开始时间: 2026-03-03T19:01:13.670308

## Round 1

Labeled Pool Size: 151

- Epoch 1: Loss=0.4832, mIoU=0.4979, F1=0.5360
- Epoch 2: Loss=0.2755, mIoU=0.5456, F1=0.6005
- Epoch 3: Loss=0.1709, mIoU=0.5527, F1=0.6095
- Epoch 4: Loss=0.1210, mIoU=0.5932, F1=0.6685
- Epoch 5: Loss=0.0933, mIoU=0.5935, F1=0.6678
- Epoch 6: Loss=0.0808, mIoU=0.6250, F1=0.7089
- Epoch 7: Loss=0.0702, mIoU=0.6202, F1=0.7027
- Epoch 8: Loss=0.0654, mIoU=0.6310, F1=0.7161
- Epoch 9: Loss=0.0625, mIoU=0.6522, F1=0.7424
- Epoch 10: Loss=0.0559, mIoU=0.6334, F1=0.7182

当前轮次最佳结果: Round=1, Labeled=151, mIoU=0.6522, F1=0.7424

## Round 2

Labeled Pool Size: 239

- Epoch 1: Loss=0.4457, mIoU=0.5510, F1=0.6071
- Epoch 2: Loss=0.2450, mIoU=0.6221, F1=0.7053
- Epoch 3: Loss=0.1717, mIoU=0.6476, F1=0.7358
- Epoch 4: Loss=0.1461, mIoU=0.6753, F1=0.7657
- Epoch 5: Loss=0.1296, mIoU=0.6441, F1=0.7302
- Epoch 6: Loss=0.1242, mIoU=0.6482, F1=0.7351
- Epoch 7: Loss=0.1099, mIoU=0.6863, F1=0.7767
- Epoch 8: Loss=0.1057, mIoU=0.6788, F1=0.7690
- Epoch 9: Loss=0.1019, mIoU=0.6935, F1=0.7845
- Epoch 10: Loss=0.0950, mIoU=0.7119, F1=0.8019

当前轮次最佳结果: Round=2, Labeled=239, mIoU=0.7119, F1=0.8019

## Round 3

Labeled Pool Size: 327

- Epoch 1: Loss=0.4007, mIoU=0.6009, F1=0.6783
- Epoch 2: Loss=0.2093, mIoU=0.6329, F1=0.7175
- Epoch 3: Loss=0.1620, mIoU=0.6626, F1=0.7520
- Epoch 4: Loss=0.1434, mIoU=0.6697, F1=0.7591
- Epoch 5: Loss=0.1329, mIoU=0.6955, F1=0.7861
- Epoch 6: Loss=0.1241, mIoU=0.7026, F1=0.7929
- Epoch 7: Loss=0.1199, mIoU=0.7066, F1=0.7966
- Epoch 8: Loss=0.1122, mIoU=0.7203, F1=0.8103
- Epoch 9: Loss=0.1086, mIoU=0.7093, F1=0.7991
- Epoch 10: Loss=0.1025, mIoU=0.7266, F1=0.8155

当前轮次最佳结果: Round=3, Labeled=327, mIoU=0.7266, F1=0.8155

## Round 4

Labeled Pool Size: 415

- Epoch 1: Loss=0.4485, mIoU=0.6124, F1=0.6924
- Epoch 2: Loss=0.2029, mIoU=0.6310, F1=0.7149
- Epoch 3: Loss=0.1603, mIoU=0.6559, F1=0.7438
- Epoch 4: Loss=0.1471, mIoU=0.6902, F1=0.7803
- Epoch 5: Loss=0.1304, mIoU=0.6925, F1=0.7826
- Epoch 6: Loss=0.1240, mIoU=0.7253, F1=0.8147
- Epoch 7: Loss=0.1161, mIoU=0.7221, F1=0.8116
- Epoch 8: Loss=0.1134, mIoU=0.7156, F1=0.8052
- Epoch 9: Loss=0.1115, mIoU=0.6926, F1=0.7826
- Epoch 10: Loss=0.1047, mIoU=0.7035, F1=0.7935

当前轮次最佳结果: Round=4, Labeled=415, mIoU=0.7253, F1=0.8147

## Round 5

Labeled Pool Size: 503

- Epoch 1: Loss=0.3765, mIoU=0.6298, F1=0.7138
- Epoch 2: Loss=0.1807, mIoU=0.6567, F1=0.7448
- Epoch 3: Loss=0.1493, mIoU=0.6809, F1=0.7707
- Epoch 4: Loss=0.1337, mIoU=0.6894, F1=0.7795
- Epoch 5: Loss=0.1251, mIoU=0.7060, F1=0.7966
- Epoch 6: Loss=0.1196, mIoU=0.7207, F1=0.8106
- Epoch 7: Loss=0.1110, mIoU=0.7260, F1=0.8149
- Epoch 8: Loss=0.1076, mIoU=0.7242, F1=0.8137
- Epoch 9: Loss=0.1075, mIoU=0.6878, F1=0.7776
- Epoch 10: Loss=0.1002, mIoU=0.7309, F1=0.8194

当前轮次最佳结果: Round=5, Labeled=503, mIoU=0.7309, F1=0.8194

## Round 6

Labeled Pool Size: 591

- Epoch 1: Loss=0.4172, mIoU=0.6106, F1=0.6898
- Epoch 2: Loss=0.1780, mIoU=0.6591, F1=0.7474
- Epoch 3: Loss=0.1400, mIoU=0.6781, F1=0.7677
- Epoch 4: Loss=0.1312, mIoU=0.7063, F1=0.7965
- Epoch 5: Loss=0.1190, mIoU=0.6769, F1=0.7664
- Epoch 6: Loss=0.1124, mIoU=0.7087, F1=0.7984
- Epoch 7: Loss=0.1092, mIoU=0.7287, F1=0.8174
- Epoch 8: Loss=0.1053, mIoU=0.7266, F1=0.8162
- Epoch 9: Loss=0.1025, mIoU=0.7366, F1=0.8245
- Epoch 10: Loss=0.0987, mIoU=0.7294, F1=0.8180

当前轮次最佳结果: Round=6, Labeled=591, mIoU=0.7366, F1=0.8245

## Round 7

Labeled Pool Size: 679

- Epoch 1: Loss=0.3649, mIoU=0.6355, F1=0.7208
- Epoch 2: Loss=0.1540, mIoU=0.6480, F1=0.7348
- Epoch 3: Loss=0.1320, mIoU=0.7019, F1=0.7922
- Epoch 4: Loss=0.1226, mIoU=0.7050, F1=0.7949
- Epoch 5: Loss=0.1142, mIoU=0.6640, F1=0.7526
- Epoch 6: Loss=0.1085, mIoU=0.7247, F1=0.8138
- Epoch 7: Loss=0.1053, mIoU=0.6455, F1=0.7315
- Epoch 8: Loss=0.0996, mIoU=0.7388, F1=0.8269
- Epoch 9: Loss=0.0988, mIoU=0.7226, F1=0.8116
- Epoch 10: Loss=0.0966, mIoU=0.7427, F1=0.8301

当前轮次最佳结果: Round=7, Labeled=679, mIoU=0.7427, F1=0.8301

## Round 8

Labeled Pool Size: 767

- Epoch 1: Loss=0.2582, mIoU=0.6399, F1=0.7259
- Epoch 2: Loss=0.1382, mIoU=0.6825, F1=0.7725
- Epoch 3: Loss=0.1189, mIoU=0.7007, F1=0.7909
- Epoch 4: Loss=0.1087, mIoU=0.7163, F1=0.8058
- Epoch 5: Loss=0.1017, mIoU=0.7228, F1=0.8119
- Epoch 6: Loss=0.0990, mIoU=0.7357, F1=0.8238
- Epoch 7: Loss=0.0926, mIoU=0.7231, F1=0.8121
- Epoch 8: Loss=0.0905, mIoU=0.7223, F1=0.8114
- Epoch 9: Loss=0.0862, mIoU=0.7496, F1=0.8359
- Epoch 10: Loss=0.0839, mIoU=0.7453, F1=0.8321

当前轮次最佳结果: Round=8, Labeled=767, mIoU=0.7496, F1=0.8359

## Round 9

Labeled Pool Size: 855

- Epoch 1: Loss=0.2265, mIoU=0.6539, F1=0.7418
- Epoch 2: Loss=0.1250, mIoU=0.6963, F1=0.7864
- Epoch 3: Loss=0.1101, mIoU=0.6949, F1=0.7849
- Epoch 4: Loss=0.1016, mIoU=0.7156, F1=0.8050
- Epoch 5: Loss=0.0944, mIoU=0.7410, F1=0.8285
- Epoch 6: Loss=0.0883, mIoU=0.7395, F1=0.8273
- Epoch 7: Loss=0.0848, mIoU=0.6947, F1=0.7846
- Epoch 8: Loss=0.0841, mIoU=0.7359, F1=0.8238
- Epoch 9: Loss=0.0818, mIoU=0.7543, F1=0.8400
- Epoch 10: Loss=0.0783, mIoU=0.7557, F1=0.8413

当前轮次最佳结果: Round=9, Labeled=855, mIoU=0.7557, F1=0.8413

## Round 10

Labeled Pool Size: 943

- Epoch 1: Loss=0.2186, mIoU=0.6185, F1=0.6998
- Epoch 2: Loss=0.1184, mIoU=0.6762, F1=0.7659
- Epoch 3: Loss=0.1048, mIoU=0.6201, F1=0.7012
- Epoch 4: Loss=0.0965, mIoU=0.7244, F1=0.8138
- Epoch 5: Loss=0.0916, mIoU=0.7231, F1=0.8121
- Epoch 6: Loss=0.0885, mIoU=0.6893, F1=0.7791
- Epoch 7: Loss=0.0846, mIoU=0.7104, F1=0.8000
- Epoch 8: Loss=0.0798, mIoU=0.7399, F1=0.8274
- Epoch 9: Loss=0.0786, mIoU=0.7465, F1=0.8332
- Epoch 10: Loss=0.0746, mIoU=0.7239, F1=0.8127

当前轮次最佳结果: Round=10, Labeled=943, mIoU=0.7465, F1=0.8332

## Round 11

Labeled Pool Size: 1031

- Epoch 1: Loss=0.2463, mIoU=0.6264, F1=0.7097
- Epoch 2: Loss=0.1187, mIoU=0.6964, F1=0.7867
- Epoch 3: Loss=0.1028, mIoU=0.6810, F1=0.7707
- Epoch 4: Loss=0.0950, mIoU=0.7329, F1=0.8213
- Epoch 5: Loss=0.0889, mIoU=0.6755, F1=0.7647
- Epoch 6: Loss=0.0852, mIoU=0.7246, F1=0.8139
- Epoch 7: Loss=0.0801, mIoU=0.7145, F1=0.8039
- Epoch 8: Loss=0.0765, mIoU=0.7251, F1=0.8140
- Epoch 9: Loss=0.0752, mIoU=0.7321, F1=0.8201
- Epoch 10: Loss=0.0712, mIoU=0.7538, F1=0.8395

当前轮次最佳结果: Round=11, Labeled=1031, mIoU=0.7538, F1=0.8395

## Round 12

Labeled Pool Size: 1119

- Epoch 1: Loss=0.1863, mIoU=0.6663, F1=0.7555
- Epoch 2: Loss=0.1079, mIoU=0.6117, F1=0.6907
- Epoch 3: Loss=0.0963, mIoU=0.6543, F1=0.7416
- Epoch 4: Loss=0.0887, mIoU=0.7208, F1=0.8100
- Epoch 5: Loss=0.0814, mIoU=0.7302, F1=0.8190
- Epoch 6: Loss=0.0797, mIoU=0.7244, F1=0.8132
- Epoch 7: Loss=0.0746, mIoU=0.7489, F1=0.8355
- Epoch 8: Loss=0.0713, mIoU=0.7499, F1=0.8361
- Epoch 9: Loss=0.0711, mIoU=0.7504, F1=0.8367
- Epoch 10: Loss=0.0679, mIoU=0.7580, F1=0.8431

当前轮次最佳结果: Round=12, Labeled=1119, mIoU=0.7580, F1=0.8431

## Round 13

Labeled Pool Size: 1207

- Epoch 1: Loss=0.2283, mIoU=0.6739, F1=0.7646
- Epoch 2: Loss=0.1075, mIoU=0.6898, F1=0.7797
- Epoch 3: Loss=0.0958, mIoU=0.6419, F1=0.7274
- Epoch 4: Loss=0.0875, mIoU=0.7388, F1=0.8265
- Epoch 5: Loss=0.0823, mIoU=0.7391, F1=0.8266
- Epoch 6: Loss=0.0792, mIoU=0.7431, F1=0.8302
- Epoch 7: Loss=0.0766, mIoU=0.7476, F1=0.8343
- Epoch 8: Loss=0.0741, mIoU=0.7490, F1=0.8354
- Epoch 9: Loss=0.0707, mIoU=0.7464, F1=0.8330
- Epoch 10: Loss=0.0682, mIoU=0.7589, F1=0.8439

当前轮次最佳结果: Round=13, Labeled=1207, mIoU=0.7589, F1=0.8439

## Round 14

Labeled Pool Size: 1295

- Epoch 1: Loss=0.2164, mIoU=0.6674, F1=0.7573
- Epoch 2: Loss=0.1042, mIoU=0.7099, F1=0.8007
- Epoch 3: Loss=0.0910, mIoU=0.6988, F1=0.7888
- Epoch 4: Loss=0.0832, mIoU=0.7339, F1=0.8220
- Epoch 5: Loss=0.0778, mIoU=0.7144, F1=0.8037
- Epoch 6: Loss=0.0735, mIoU=0.7436, F1=0.8305
- Epoch 7: Loss=0.0702, mIoU=0.7363, F1=0.8239
- Epoch 8: Loss=0.0699, mIoU=0.7447, F1=0.8315
- Epoch 9: Loss=0.0673, mIoU=0.7427, F1=0.8297
- Epoch 10: Loss=0.0627, mIoU=0.7603, F1=0.8450

当前轮次最佳结果: Round=14, Labeled=1295, mIoU=0.7603, F1=0.8450

## Round 15

Labeled Pool Size: 1383

- Epoch 1: Loss=0.2090, mIoU=0.6694, F1=0.7586
- Epoch 2: Loss=0.0986, mIoU=0.7177, F1=0.8072
- Epoch 3: Loss=0.0841, mIoU=0.7312, F1=0.8196
- Epoch 4: Loss=0.0780, mIoU=0.6993, F1=0.7891
- Epoch 5: Loss=0.0732, mIoU=0.7283, F1=0.8169
- Epoch 6: Loss=0.0711, mIoU=0.7560, F1=0.8414
- Epoch 7: Loss=0.0670, mIoU=0.7385, F1=0.8259
- Epoch 8: Loss=0.0639, mIoU=0.7044, F1=0.7941
- Epoch 9: Loss=0.0627, mIoU=0.7524, F1=0.8384
- Epoch 10: Loss=0.0604, mIoU=0.7516, F1=0.8374

当前轮次最佳结果: Round=15, Labeled=1383, mIoU=0.7560, F1=0.8414


## 实验汇总

预算历史: [151, 239, 327, 415, 503, 591, 679, 767, 855, 943, 1031, 1119, 1207, 1295, 1383]
ALC: 0.6679
最终 mIoU: 0.7560
最终 F1: 0.8414
