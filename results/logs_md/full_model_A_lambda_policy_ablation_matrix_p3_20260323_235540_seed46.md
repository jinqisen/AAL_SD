# 实验日志

实验名称: full_model_A_lambda_policy
描述: 对照A / AAL-SD (Full)：基于 auto_opt_iter15_cand2_02（warmup+risk闭环 + 后期ramp + U-guardrail；train_holdout grad_probe）
开始时间: 2026-03-26T03:36:39.948990

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.4568, mIoU=0.4971, F1=0.5084
- Epoch 2: Loss=0.2560, mIoU=0.4966, F1=0.5061
- Epoch 3: Loss=0.1711, mIoU=0.5005, F1=0.5135
- Epoch 4: Loss=0.1267, mIoU=0.5096, F1=0.5306
- Epoch 5: Loss=0.1022, mIoU=0.5063, F1=0.5245
- Epoch 6: Loss=0.0867, mIoU=0.5282, F1=0.5637
- Epoch 7: Loss=0.0756, mIoU=0.5084, F1=0.5283
- Epoch 8: Loss=0.0665, mIoU=0.5145, F1=0.5397
- Epoch 9: Loss=0.0649, mIoU=0.5506, F1=0.6007
- Epoch 10: Loss=0.0611, mIoU=0.5192, F1=0.5478

本轮结果: Round=1, Labeled=189, Selection=best_val (epoch=9), mIoU=0.5506, F1=0.6007, peak_mIoU=0.5506

Round=1, Labeled=189, mIoU=0.5506, F1=0.6007

## Round 5

Labeled Pool Size: 365

- Epoch 1: Loss=0.5188, mIoU=0.5318, F1=0.5703
- Epoch 2: Loss=0.2366, mIoU=0.5003, F1=0.5130
- Epoch 3: Loss=0.1744, mIoU=0.5430, F1=0.5885
- Epoch 4: Loss=0.1514, mIoU=0.5320, F1=0.5700
- Epoch 5: Loss=0.1403, mIoU=0.5317, F1=0.5696
- Epoch 6: Loss=0.1321, mIoU=0.6505, F1=0.7356
- Epoch 7: Loss=0.1263, mIoU=0.5626, F1=0.6193
- Epoch 8: Loss=0.1187, mIoU=0.6597, F1=0.7464
- Epoch 9: Loss=0.1140, mIoU=0.6413, F1=0.7250
- Epoch 10: Loss=0.1128, mIoU=0.5826, F1=0.6487

本轮结果: Round=5, Labeled=365, Selection=best_val (epoch=8), mIoU=0.6597, F1=0.7464, peak_mIoU=0.6597

Round=5, Labeled=365, mIoU=0.6597, F1=0.7464

## Round 2

Labeled Pool Size: 277

- Epoch 1: Loss=0.5409, mIoU=0.5387, F1=0.5998
- Epoch 2: Loss=0.2441, mIoU=0.4972, F1=0.5071
- Epoch 3: Loss=0.1651, mIoU=0.5834, F1=0.6498
- Epoch 4: Loss=0.1299, mIoU=0.5003, F1=0.5133
- Epoch 5: Loss=0.1172, mIoU=0.5188, F1=0.5471
- Epoch 6: Loss=0.1112, mIoU=0.5223, F1=0.5535
- Epoch 7: Loss=0.1068, mIoU=0.6137, F1=0.6908
- Epoch 8: Loss=0.1022, mIoU=0.5086, F1=0.5287
- Epoch 9: Loss=0.0909, mIoU=0.5537, F1=0.6058
- Epoch 10: Loss=0.0936, mIoU=0.5847, F1=0.6523

本轮结果: Round=2, Labeled=277, Selection=best_val (epoch=7), mIoU=0.6137, F1=0.6908, peak_mIoU=0.6137

Round=2, Labeled=277, mIoU=0.6137, F1=0.6908

## Round 6

Labeled Pool Size: 453

- Epoch 1: Loss=0.3449, mIoU=0.5318, F1=0.5786
- Epoch 2: Loss=0.1833, mIoU=0.5460, F1=0.5935
- Epoch 3: Loss=0.1511, mIoU=0.5423, F1=0.5876
- Epoch 4: Loss=0.1431, mIoU=0.5349, F1=0.5756
- Epoch 5: Loss=0.1317, mIoU=0.6229, F1=0.7029
- Epoch 6: Loss=0.1204, mIoU=0.5232, F1=0.5553
- Epoch 7: Loss=0.1174, mIoU=0.6464, F1=0.7310
- Epoch 8: Loss=0.1118, mIoU=0.5522, F1=0.6040
- Epoch 9: Loss=0.1094, mIoU=0.6044, F1=0.6794
- Epoch 10: Loss=0.1054, mIoU=0.6467, F1=0.7316

本轮结果: Round=6, Labeled=453, Selection=best_val (epoch=10), mIoU=0.6467, F1=0.7316, peak_mIoU=0.6467

Round=6, Labeled=453, mIoU=0.6467, F1=0.7316

## Round 7

Labeled Pool Size: 541

- Epoch 1: Loss=0.2955, mIoU=0.5105, F1=0.5324
- Epoch 2: Loss=0.1675, mIoU=0.5037, F1=0.5195
- Epoch 3: Loss=0.1413, mIoU=0.5319, F1=0.5699
- Epoch 4: Loss=0.1294, mIoU=0.6092, F1=0.6852
- Epoch 5: Loss=0.1261, mIoU=0.5754, F1=0.6384
- Epoch 6: Loss=0.1210, mIoU=0.5884, F1=0.6571
- Epoch 7: Loss=0.1130, mIoU=0.6803, F1=0.7684
- Epoch 8: Loss=0.1089, mIoU=0.6735, F1=0.7619
- Epoch 9: Loss=0.1051, mIoU=0.6154, F1=0.6933
- Epoch 10: Loss=0.0997, mIoU=0.6312, F1=0.7129

本轮结果: Round=7, Labeled=541, Selection=best_val (epoch=7), mIoU=0.6803, F1=0.7684, peak_mIoU=0.6803

Round=7, Labeled=541, mIoU=0.6803, F1=0.7684

## Round 8

Labeled Pool Size: 629

- Epoch 1: Loss=0.2875, mIoU=0.5011, F1=0.5144
- Epoch 2: Loss=0.1639, mIoU=0.5233, F1=0.5550
- Epoch 3: Loss=0.1396, mIoU=0.5782, F1=0.6426
- Epoch 4: Loss=0.1271, mIoU=0.6209, F1=0.7008
- Epoch 5: Loss=0.1196, mIoU=0.5459, F1=0.5933
## Round 3

Labeled Pool Size: 365

- Epoch 6: Loss=0.1138, mIoU=0.5955, F1=0.6675
- Epoch 1: Loss=0.4531, mIoU=0.5334, F1=0.5728
- Epoch 2: Loss=0.2267, mIoU=0.5186, F1=0.5467
- Epoch 7: Loss=0.1086, mIoU=0.6017, F1=0.6754
- Epoch 3: Loss=0.1739, mIoU=0.5045, F1=0.5210
- Epoch 4: Loss=0.1485, mIoU=0.5539, F1=0.6059
- Epoch 8: Loss=0.1039, mIoU=0.6550, F1=0.7421
- Epoch 5: Loss=0.1353, mIoU=0.5472, F1=0.5952
- Epoch 9: Loss=0.1003, mIoU=0.6505, F1=0.7359
- Epoch 6: Loss=0.1245, mIoU=0.6207, F1=0.7002
- Epoch 7: Loss=0.1247, mIoU=0.5539, F1=0.6061
- Epoch 10: Loss=0.0949, mIoU=0.6467, F1=0.7316

本轮结果: Round=8, Labeled=629, Selection=best_val (epoch=8), mIoU=0.6550, F1=0.7421, peak_mIoU=0.6550

Round=8, Labeled=629, mIoU=0.6550, F1=0.7421

- Epoch 8: Loss=0.1203, mIoU=0.5561, F1=0.6096
- Epoch 9: Loss=0.1162, mIoU=0.6706, F1=0.7586
- Epoch 10: Loss=0.1115, mIoU=0.5841, F1=0.6510

本轮结果: Round=3, Labeled=365, Selection=best_val (epoch=9), mIoU=0.6706, F1=0.7586, peak_mIoU=0.6706

Round=3, Labeled=365, mIoU=0.6706, F1=0.7586

## Round 4

Labeled Pool Size: 453

- Epoch 1: Loss=0.4936, mIoU=0.5166, F1=0.5436
- Epoch 2: Loss=0.2235, mIoU=0.5096, F1=0.5305
- Epoch 3: Loss=0.1741, mIoU=0.5251, F1=0.5583
- Epoch 4: Loss=0.1557, mIoU=0.5266, F1=0.5611
## Round 9

Labeled Pool Size: 717

- Epoch 5: Loss=0.1391, mIoU=0.6246, F1=0.7049
- Epoch 6: Loss=0.1322, mIoU=0.5582, F1=0.6138
- Epoch 1: Loss=0.3387, mIoU=0.5043, F1=0.5207
- Epoch 7: Loss=0.1254, mIoU=0.6219, F1=0.7017
- Epoch 2: Loss=0.1660, mIoU=0.5274, F1=0.5623
- Epoch 8: Loss=0.1204, mIoU=0.6377, F1=0.7210
- Epoch 9: Loss=0.1143, mIoU=0.6835, F1=0.7723
- Epoch 3: Loss=0.1376, mIoU=0.6585, F1=0.7451
- Epoch 10: Loss=0.1107, mIoU=0.6297, F1=0.7118

本轮结果: Round=4, Labeled=453, Selection=best_val (epoch=9), mIoU=0.6835, F1=0.7723, peak_mIoU=0.6835

Round=4, Labeled=453, mIoU=0.6835, F1=0.7723

- Epoch 4: Loss=0.1260, mIoU=0.6817, F1=0.7701
- Epoch 5: Loss=0.1166, mIoU=0.6913, F1=0.7801
- Epoch 6: Loss=0.1148, mIoU=0.6632, F1=0.7498
- Epoch 7: Loss=0.1095, mIoU=0.6437, F1=0.7314
- Epoch 8: Loss=0.1031, mIoU=0.7003, F1=0.7893
- Epoch 9: Loss=0.0997, mIoU=0.6648, F1=0.7519
## Round 5

Labeled Pool Size: 541

- Epoch 10: Loss=0.0963, mIoU=0.6926, F1=0.7814

本轮结果: Round=9, Labeled=717, Selection=best_val (epoch=8), mIoU=0.7003, F1=0.7893, peak_mIoU=0.7003

Round=9, Labeled=717, mIoU=0.7003, F1=0.7893

- Epoch 1: Loss=0.4467, mIoU=0.5845, F1=0.6525
- Epoch 2: Loss=0.2007, mIoU=0.5320, F1=0.5700
- Epoch 3: Loss=0.1575, mIoU=0.6661, F1=0.7534
- Epoch 4: Loss=0.1406, mIoU=0.6675, F1=0.7553
- Epoch 5: Loss=0.1350, mIoU=0.6536, F1=0.7393
- Epoch 6: Loss=0.1241, mIoU=0.6854, F1=0.7741
- Epoch 7: Loss=0.1178, mIoU=0.6427, F1=0.7303
- Epoch 8: Loss=0.1111, mIoU=0.6547, F1=0.7405
## Round 10

Labeled Pool Size: 805

- Epoch 9: Loss=0.1125, mIoU=0.6360, F1=0.7187
- Epoch 10: Loss=0.1057, mIoU=0.6787, F1=0.7669

本轮结果: Round=5, Labeled=541, Selection=best_val (epoch=6), mIoU=0.6854, F1=0.7741, peak_mIoU=0.6854

Round=5, Labeled=541, mIoU=0.6854, F1=0.7741

- Epoch 1: Loss=0.2419, mIoU=0.5638, F1=0.6211
- Epoch 2: Loss=0.1448, mIoU=0.5949, F1=0.6658
- Epoch 3: Loss=0.1283, mIoU=0.6505, F1=0.7358
- Epoch 4: Loss=0.1151, mIoU=0.6965, F1=0.7854
## Round 6

Labeled Pool Size: 629

- Epoch 5: Loss=0.1109, mIoU=0.6858, F1=0.7742
- Epoch 1: Loss=0.3022, mIoU=0.5091, F1=0.5302
- Epoch 6: Loss=0.1057, mIoU=0.6269, F1=0.7077
- Epoch 2: Loss=0.1677, mIoU=0.5567, F1=0.6104
- Epoch 7: Loss=0.1003, mIoU=0.5244, F1=0.5571
- Epoch 3: Loss=0.1431, mIoU=0.5909, F1=0.6610
- Epoch 8: Loss=0.0970, mIoU=0.6348, F1=0.7172
- Epoch 4: Loss=0.1291, mIoU=0.5885, F1=0.6573
- Epoch 9: Loss=0.0956, mIoU=0.6571, F1=0.7432
- Epoch 5: Loss=0.1235, mIoU=0.5834, F1=0.6501
- Epoch 6: Loss=0.1195, mIoU=0.6672, F1=0.7544
- Epoch 10: Loss=0.0918, mIoU=0.6864, F1=0.7754

本轮结果: Round=10, Labeled=805, Selection=best_val (epoch=4), mIoU=0.6965, F1=0.7854, peak_mIoU=0.6965

Round=10, Labeled=805, mIoU=0.6965, F1=0.7854

- Epoch 7: Loss=0.1091, mIoU=0.6348, F1=0.7177
- Epoch 8: Loss=0.1055, mIoU=0.6782, F1=0.7665
- Epoch 9: Loss=0.1021, mIoU=0.6398, F1=0.7266
- Epoch 10: Loss=0.0985, mIoU=0.6546, F1=0.7407

本轮结果: Round=6, Labeled=629, Selection=best_val (epoch=8), mIoU=0.6782, F1=0.7665, peak_mIoU=0.6782

Round=6, Labeled=629, mIoU=0.6782, F1=0.7665

## Round 11

Labeled Pool Size: 893

- Epoch 1: Loss=0.2998, mIoU=0.5001, F1=0.5127
- Epoch 2: Loss=0.1448, mIoU=0.5768, F1=0.6406
- Epoch 3: Loss=0.1235, mIoU=0.6297, F1=0.7112
- Epoch 4: Loss=0.1143, mIoU=0.6824, F1=0.7711
- Epoch 5: Loss=0.1083, mIoU=0.6442, F1=0.7285
- Epoch 6: Loss=0.1033, mIoU=0.6553, F1=0.7439
- Epoch 7: Loss=0.0964, mIoU=0.6711, F1=0.7588
- Epoch 8: Loss=0.0927, mIoU=0.6610, F1=0.7481
## Round 7

Labeled Pool Size: 717

- Epoch 9: Loss=0.0891, mIoU=0.6657, F1=0.7530
- Epoch 1: Loss=0.2640, mIoU=0.5153, F1=0.5409
- Epoch 2: Loss=0.1545, mIoU=0.5957, F1=0.6672
- Epoch 10: Loss=0.0868, mIoU=0.6874, F1=0.7766

本轮结果: Round=11, Labeled=893, Selection=best_val (epoch=10), mIoU=0.6874, F1=0.7766, peak_mIoU=0.6874

Round=11, Labeled=893, mIoU=0.6874, F1=0.7766

- Epoch 3: Loss=0.1357, mIoU=0.5329, F1=0.5716
- Epoch 4: Loss=0.1214, mIoU=0.5168, F1=0.5437
- Epoch 5: Loss=0.1180, mIoU=0.6249, F1=0.7054
- Epoch 6: Loss=0.1102, mIoU=0.6505, F1=0.7357
## Round 12

Labeled Pool Size: 981

- Epoch 7: Loss=0.1060, mIoU=0.6375, F1=0.7214
- Epoch 1: Loss=0.2733, mIoU=0.5191, F1=0.5477
- Epoch 8: Loss=0.1010, mIoU=0.6884, F1=0.7770
- Epoch 9: Loss=0.1000, mIoU=0.6865, F1=0.7754
- Epoch 2: Loss=0.1349, mIoU=0.5880, F1=0.6565
- Epoch 10: Loss=0.0969, mIoU=0.6792, F1=0.7678

本轮结果: Round=7, Labeled=717, Selection=best_val (epoch=8), mIoU=0.6884, F1=0.7770, peak_mIoU=0.6884

Round=7, Labeled=717, mIoU=0.6884, F1=0.7770

- Epoch 3: Loss=0.1174, mIoU=0.5846, F1=0.6517
- Epoch 4: Loss=0.1062, mIoU=0.6786, F1=0.7667
- Epoch 5: Loss=0.1006, mIoU=0.6980, F1=0.7869
- Epoch 6: Loss=0.0952, mIoU=0.7001, F1=0.7891
## Round 8

Labeled Pool Size: 805

- Epoch 7: Loss=0.0911, mIoU=0.6667, F1=0.7548
- Epoch 1: Loss=0.2550, mIoU=0.5374, F1=0.5794
- Epoch 8: Loss=0.0868, mIoU=0.7214, F1=0.8098
- Epoch 2: Loss=0.1473, mIoU=0.5834, F1=0.6502
- Epoch 9: Loss=0.0857, mIoU=0.7241, F1=0.8121
- Epoch 3: Loss=0.1271, mIoU=0.6632, F1=0.7502
- Epoch 4: Loss=0.1188, mIoU=0.6742, F1=0.7622
- Epoch 10: Loss=0.0838, mIoU=0.7040, F1=0.7935

本轮结果: Round=12, Labeled=981, Selection=best_val (epoch=9), mIoU=0.7241, F1=0.8121, peak_mIoU=0.7241

Round=12, Labeled=981, mIoU=0.7241, F1=0.8121

- Epoch 5: Loss=0.1150, mIoU=0.6374, F1=0.7212
- Epoch 6: Loss=0.1077, mIoU=0.6635, F1=0.7507
- Epoch 7: Loss=0.1051, mIoU=0.7008, F1=0.7902
## Round 13

Labeled Pool Size: 1069

- Epoch 8: Loss=0.0981, mIoU=0.6796, F1=0.7679
- Epoch 9: Loss=0.0959, mIoU=0.6208, F1=0.7003
- Epoch 1: Loss=0.2651, mIoU=0.5782, F1=0.6427
- Epoch 10: Loss=0.0914, mIoU=0.6724, F1=0.7603

本轮结果: Round=8, Labeled=805, Selection=best_val (epoch=7), mIoU=0.7008, F1=0.7902, peak_mIoU=0.7008

Round=8, Labeled=805, mIoU=0.7008, F1=0.7902

- Epoch 2: Loss=0.1312, mIoU=0.5990, F1=0.6714
- Epoch 3: Loss=0.1133, mIoU=0.5556, F1=0.6084
- Epoch 4: Loss=0.1038, mIoU=0.6468, F1=0.7318
- Epoch 5: Loss=0.0977, mIoU=0.6870, F1=0.7761
- Epoch 6: Loss=0.0921, mIoU=0.6693, F1=0.7577
- Epoch 7: Loss=0.0877, mIoU=0.6708, F1=0.7588
- Epoch 8: Loss=0.0848, mIoU=0.6682, F1=0.7559
- Epoch 9: Loss=0.0833, mIoU=0.6767, F1=0.7657
## Round 9

Labeled Pool Size: 893

- Epoch 1: Loss=0.3016, mIoU=0.5391, F1=0.5824
- Epoch 10: Loss=0.0814, mIoU=0.6354, F1=0.7183

本轮结果: Round=13, Labeled=1069, Selection=best_val (epoch=5), mIoU=0.6870, F1=0.7761, peak_mIoU=0.6870

Round=13, Labeled=1069, mIoU=0.6870, F1=0.7761

- Epoch 2: Loss=0.1509, mIoU=0.5967, F1=0.6686
- Epoch 3: Loss=0.1294, mIoU=0.6123, F1=0.6894
- Epoch 4: Loss=0.1169, mIoU=0.7040, F1=0.7928
- Epoch 5: Loss=0.1109, mIoU=0.7121, F1=0.8008
## Round 14

Labeled Pool Size: 1157

- Epoch 6: Loss=0.1065, mIoU=0.7276, F1=0.8154
- Epoch 7: Loss=0.1026, mIoU=0.6645, F1=0.7515
- Epoch 1: Loss=0.2375, mIoU=0.6064, F1=0.6813
- Epoch 8: Loss=0.0960, mIoU=0.6894, F1=0.7791
- Epoch 2: Loss=0.1236, mIoU=0.6338, F1=0.7161
- Epoch 9: Loss=0.0917, mIoU=0.6996, F1=0.7891
- Epoch 3: Loss=0.1069, mIoU=0.6591, F1=0.7453
- Epoch 10: Loss=0.0912, mIoU=0.7042, F1=0.7929

本轮结果: Round=9, Labeled=893, Selection=best_val (epoch=6), mIoU=0.7276, F1=0.8154, peak_mIoU=0.7276

Round=9, Labeled=893, mIoU=0.7276, F1=0.8154

- Epoch 4: Loss=0.0999, mIoU=0.7057, F1=0.7949
- Epoch 5: Loss=0.0933, mIoU=0.7038, F1=0.7925
- Epoch 6: Loss=0.0888, mIoU=0.6785, F1=0.7665
## Round 10

Labeled Pool Size: 981

- Epoch 7: Loss=0.0847, mIoU=0.6878, F1=0.7766
- Epoch 1: Loss=0.2219, mIoU=0.5008, F1=0.5142
- Epoch 8: Loss=0.0824, mIoU=0.6326, F1=0.7145
- Epoch 2: Loss=0.1316, mIoU=0.5777, F1=0.6417
- Epoch 9: Loss=0.0802, mIoU=0.7086, F1=0.7977
- Epoch 3: Loss=0.1184, mIoU=0.6451, F1=0.7293
- Epoch 10: Loss=0.0764, mIoU=0.6520, F1=0.7386

本轮结果: Round=14, Labeled=1157, Selection=best_val (epoch=9), mIoU=0.7086, F1=0.7977, peak_mIoU=0.7086

Round=14, Labeled=1157, mIoU=0.7086, F1=0.7977

- Epoch 4: Loss=0.1098, mIoU=0.6397, F1=0.7230
- Epoch 5: Loss=0.1025, mIoU=0.6436, F1=0.7279
- Epoch 6: Loss=0.0967, mIoU=0.6955, F1=0.7842
## Round 15

Labeled Pool Size: 1245

- Epoch 7: Loss=0.0945, mIoU=0.6476, F1=0.7325
- Epoch 1: Loss=0.2533, mIoU=0.5368, F1=0.5783
- Epoch 8: Loss=0.0907, mIoU=0.6651, F1=0.7525
- Epoch 2: Loss=0.1206, mIoU=0.6693, F1=0.7572
- Epoch 9: Loss=0.0877, mIoU=0.6427, F1=0.7269
- Epoch 10: Loss=0.0857, mIoU=0.6562, F1=0.7430

本轮结果: Round=10, Labeled=981, Selection=best_val (epoch=6), mIoU=0.6955, F1=0.7842, peak_mIoU=0.6955

Round=10, Labeled=981, mIoU=0.6955, F1=0.7842

- Epoch 3: Loss=0.1021, mIoU=0.6836, F1=0.7726
- Epoch 4: Loss=0.0947, mIoU=0.6667, F1=0.7551
- Epoch 5: Loss=0.0885, mIoU=0.6618, F1=0.7491
## Round 11

Labeled Pool Size: 1069

- Epoch 1: Loss=0.2580, mIoU=0.6279, F1=0.7088
- Epoch 6: Loss=0.0875, mIoU=0.7054, F1=0.7942
- Epoch 2: Loss=0.1325, mIoU=0.6307, F1=0.7124
- Epoch 7: Loss=0.0830, mIoU=0.6768, F1=0.7648
- Epoch 3: Loss=0.1177, mIoU=0.6513, F1=0.7365
- Epoch 8: Loss=0.0786, mIoU=0.7066, F1=0.7957
- Epoch 4: Loss=0.1079, mIoU=0.6741, F1=0.7626
- Epoch 9: Loss=0.0760, mIoU=0.6913, F1=0.7804
- Epoch 5: Loss=0.1041, mIoU=0.6643, F1=0.7530
- Epoch 10: Loss=0.0736, mIoU=0.6541, F1=0.7408

本轮结果: Round=15, Labeled=1245, Selection=best_val (epoch=8), mIoU=0.7066, F1=0.7957, peak_mIoU=0.7066

Round=15, Labeled=1245, mIoU=0.7066, F1=0.7957

- Epoch 6: Loss=0.0994, mIoU=0.6757, F1=0.7637
- Epoch 7: Loss=0.0940, mIoU=0.6576, F1=0.7453
- Epoch 8: Loss=0.0920, mIoU=0.7059, F1=0.7951
- Epoch 9: Loss=0.0905, mIoU=0.6721, F1=0.7603
- Epoch 10: Loss=0.0877, mIoU=0.6903, F1=0.7800

本轮结果: Round=11, Labeled=1069, Selection=best_val (epoch=8), mIoU=0.7059, F1=0.7951, peak_mIoU=0.7059

Round=11, Labeled=1069, mIoU=0.7059, F1=0.7951

## Round 16

Labeled Pool Size: 1333


**[ERROR] Round 16 失败: Missing previous round best checkpoint for final test-only round: results/runs/ablation_matrix_p3_20260323_235540_seed46/full_model_A_lambda_policy_round_models/round_15_best_val.pt**

## Round 12

Labeled Pool Size: 1157

- Epoch 1: Loss=0.2541, mIoU=0.5341, F1=0.5735
- Epoch 2: Loss=0.1254, mIoU=0.5244, F1=0.5571
- Epoch 3: Loss=0.1090, mIoU=0.6083, F1=0.6839
- Epoch 4: Loss=0.1001, mIoU=0.6755, F1=0.7634
- Epoch 5: Loss=0.0947, mIoU=0.6944, F1=0.7843
- Epoch 6: Loss=0.0895, mIoU=0.6636, F1=0.7505
- Epoch 7: Loss=0.0855, mIoU=0.6459, F1=0.7308
- Epoch 8: Loss=0.0825, mIoU=0.6757, F1=0.7640
- Epoch 9: Loss=0.0787, mIoU=0.6580, F1=0.7442
- Epoch 10: Loss=0.0770, mIoU=0.6408, F1=0.7245

本轮结果: Round=12, Labeled=1157, Selection=best_val (epoch=5), mIoU=0.6944, F1=0.7843, peak_mIoU=0.6944

Round=12, Labeled=1157, mIoU=0.6944, F1=0.7843

## Round 13

Labeled Pool Size: 1245

- Epoch 1: Loss=0.2440, mIoU=0.5126, F1=0.5359
- Epoch 2: Loss=0.1195, mIoU=0.6655, F1=0.7525
- Epoch 3: Loss=0.1047, mIoU=0.6928, F1=0.7814
- Epoch 4: Loss=0.0971, mIoU=0.6308, F1=0.7123
- Epoch 5: Loss=0.0899, mIoU=0.6565, F1=0.7426
- Epoch 6: Loss=0.0865, mIoU=0.6628, F1=0.7505
- Epoch 7: Loss=0.0845, mIoU=0.6513, F1=0.7365
- Epoch 8: Loss=0.0821, mIoU=0.6398, F1=0.7234
- Epoch 9: Loss=0.0766, mIoU=0.7083, F1=0.7972
- Epoch 10: Loss=0.0750, mIoU=0.6881, F1=0.7774

本轮结果: Round=13, Labeled=1245, Selection=best_val (epoch=9), mIoU=0.7083, F1=0.7972, peak_mIoU=0.7083

Round=13, Labeled=1245, mIoU=0.7083, F1=0.7972

## Round 14

Labeled Pool Size: 1333

- Epoch 1: Loss=0.2148, mIoU=0.5576, F1=0.6115
- Epoch 2: Loss=0.1129, mIoU=0.5520, F1=0.6028
- Epoch 3: Loss=0.0998, mIoU=0.6874, F1=0.7762
- Epoch 4: Loss=0.0925, mIoU=0.6388, F1=0.7220
- Epoch 5: Loss=0.0883, mIoU=0.7090, F1=0.7982
- Epoch 6: Loss=0.0834, mIoU=0.6745, F1=0.7629
- Epoch 7: Loss=0.0809, mIoU=0.6330, F1=0.7157
- Epoch 8: Loss=0.0779, mIoU=0.6827, F1=0.7711
- Epoch 9: Loss=0.0741, mIoU=0.6275, F1=0.7088
- Epoch 10: Loss=0.0718, mIoU=0.6492, F1=0.7343

本轮结果: Round=14, Labeled=1333, Selection=best_val (epoch=5), mIoU=0.7090, F1=0.7982, peak_mIoU=0.7090

Round=14, Labeled=1333, mIoU=0.7090, F1=0.7982

## Round 15

Labeled Pool Size: 1421

- Epoch 1: Loss=0.2351, mIoU=0.6415, F1=0.7253
- Epoch 2: Loss=0.1122, mIoU=0.6089, F1=0.6850
- Epoch 3: Loss=0.0969, mIoU=0.5765, F1=0.6401
- Epoch 4: Loss=0.0890, mIoU=0.7332, F1=0.8205
- Epoch 5: Loss=0.0833, mIoU=0.6124, F1=0.6894
- Epoch 6: Loss=0.0801, mIoU=0.7110, F1=0.7996
- Epoch 7: Loss=0.0751, mIoU=0.6937, F1=0.7832
- Epoch 8: Loss=0.0728, mIoU=0.6650, F1=0.7523
- Epoch 9: Loss=0.0705, mIoU=0.6821, F1=0.7717
- Epoch 10: Loss=0.0692, mIoU=0.6586, F1=0.7470

本轮结果: Round=15, Labeled=1421, Selection=best_val (epoch=4), mIoU=0.7332, F1=0.8205, peak_mIoU=0.7332

Round=15, Labeled=1421, mIoU=0.7332, F1=0.8205

## Round 16

Labeled Pool Size: 1509

- Test-only mode: Round 16 loads Round 15 best_val checkpoint (epoch=4, val_mIoU=0.7332, val_F1=0.8205)

本轮结果: Round=16, Labeled=1509, Selection=prev_round_best_val (source_round=15, epoch=4), mIoU=0.7332, F1=0.8205, peak_mIoU=0.7332

Round=16, Labeled=1509, mIoU=0.7332, F1=0.8205


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421, 1509]
ALC(基于每轮选模val mIoU): 0.6036
最后一轮选模 mIoU(val): 0.7332019297185448
最后一轮选模 F1(val): 0.8205063384199253
最终报告 mIoU(test): 0.706011707982644
最终报告 F1(test): 0.7953387635276729
最终输出 mIoU: 0.7060 (source=final_report)
最终输出 F1: 0.7953 (source=final_report)
最终 Test Split: test
最终 Report: {'loss': 0.04978070317301899, 'mIoU': 0.706011707982644, 'f1_score': 0.7953387635276729}
