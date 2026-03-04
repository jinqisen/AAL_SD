# 实验日志

实验名称: full_model_default_thresholds
描述: 完整模型（旧阈值）：固定Warmup(0.2)+默认过拟合阈值(0.8/0.5) + 自适应阈值 rollback + 风险驱动 λ 闭环控制（不调整 query_size）
开始时间: 2026-02-19T04:31:23.329113

## Round 1

Labeled Pool Size: 151

- Epoch 1: Loss=0.6124, mIoU=0.4588, F1=0.5003
- Epoch 2: Loss=0.3457, mIoU=0.5052, F1=0.5304
- Epoch 3: Loss=0.2169, mIoU=0.5109, F1=0.5381
- Epoch 4: Loss=0.1521, mIoU=0.5291, F1=0.5697
- Epoch 5: Loss=0.1178, mIoU=0.5400, F1=0.5880
- Epoch 6: Loss=0.0943, mIoU=0.5909, F1=0.6638
- Epoch 7: Loss=0.0814, mIoU=0.6099, F1=0.6889
- Epoch 8: Loss=0.0697, mIoU=0.5426, F1=0.5914
- Epoch 9: Loss=0.0645, mIoU=0.5890, F1=0.6606
- Epoch 10: Loss=0.0576, mIoU=0.6107, F1=0.6895

当前轮次最佳结果: Round=1, Labeled=151, mIoU=0.6107, F1=0.6895

## Round 2

Labeled Pool Size: 239

- Epoch 1: Loss=0.4916, mIoU=0.6134, F1=0.6969
- Epoch 2: Loss=0.2545, mIoU=0.6182, F1=0.6990
- Epoch 3: Loss=0.1762, mIoU=0.6439, F1=0.7301
- Epoch 4: Loss=0.1424, mIoU=0.6453, F1=0.7314
- Epoch 5: Loss=0.1296, mIoU=0.6842, F1=0.7744
- Epoch 6: Loss=0.1180, mIoU=0.6554, F1=0.7429
- Epoch 7: Loss=0.1069, mIoU=0.6946, F1=0.7848
- Epoch 8: Loss=0.1063, mIoU=0.6615, F1=0.7499
- Epoch 9: Loss=0.0995, mIoU=0.5904, F1=0.6624
- Epoch 10: Loss=0.0909, mIoU=0.7069, F1=0.7968

当前轮次最佳结果: Round=2, Labeled=239, mIoU=0.7069, F1=0.7968

## Round 3

Labeled Pool Size: 327

- Epoch 1: Loss=0.3218, mIoU=0.6197, F1=0.7012
- Epoch 2: Loss=0.1867, mIoU=0.6301, F1=0.7138
- Epoch 3: Loss=0.1536, mIoU=0.6145, F1=0.6942
- Epoch 4: Loss=0.1337, mIoU=0.6662, F1=0.7549
- Epoch 5: Loss=0.1314, mIoU=0.6913, F1=0.7812
- Epoch 6: Loss=0.1198, mIoU=0.6953, F1=0.7853
- Epoch 7: Loss=0.1152, mIoU=0.6863, F1=0.7765
- Epoch 8: Loss=0.1112, mIoU=0.7044, F1=0.7943
- Epoch 9: Loss=0.1074, mIoU=0.6674, F1=0.7561
- Epoch 10: Loss=0.1020, mIoU=0.7256, F1=0.8145

当前轮次最佳结果: Round=3, Labeled=327, mIoU=0.7256, F1=0.8145

## Round 4

Labeled Pool Size: 415

- Epoch 1: Loss=0.3890, mIoU=0.6184, F1=0.7014
- Epoch 2: Loss=0.1976, mIoU=0.6131, F1=0.6925
- Epoch 3: Loss=0.1558, mIoU=0.6660, F1=0.7547
- Epoch 4: Loss=0.1389, mIoU=0.6909, F1=0.7809
- Epoch 5: Loss=0.1312, mIoU=0.6398, F1=0.7250
- Epoch 6: Loss=0.1228, mIoU=0.6799, F1=0.7694
- Epoch 7: Loss=0.1205, mIoU=0.7072, F1=0.7969
- Epoch 8: Loss=0.1149, mIoU=0.6890, F1=0.7789
- Epoch 9: Loss=0.1128, mIoU=0.7130, F1=0.8025
- Epoch 10: Loss=0.1048, mIoU=0.7351, F1=0.8231

当前轮次最佳结果: Round=4, Labeled=415, mIoU=0.7351, F1=0.8231

## Round 5

Labeled Pool Size: 503

- Epoch 1: Loss=0.3415, mIoU=0.6142, F1=0.6940
- Epoch 2: Loss=0.1737, mIoU=0.6500, F1=0.7369
- Epoch 3: Loss=0.1435, mIoU=0.6774, F1=0.7669
- Epoch 4: Loss=0.1288, mIoU=0.7219, F1=0.8110
- Epoch 5: Loss=0.1206, mIoU=0.6853, F1=0.7750
- Epoch 6: Loss=0.1155, mIoU=0.7299, F1=0.8190
- Epoch 7: Loss=0.1096, mIoU=0.7189, F1=0.8080
- Epoch 8: Loss=0.1069, mIoU=0.7141, F1=0.8036
- Epoch 9: Loss=0.1015, mIoU=0.7394, F1=0.8270
- Epoch 10: Loss=0.1002, mIoU=0.7234, F1=0.8122

当前轮次最佳结果: Round=5, Labeled=503, mIoU=0.7394, F1=0.8270

## Round 6

Labeled Pool Size: 591

- Epoch 1: Loss=0.3027, mIoU=0.6196, F1=0.7007
- Epoch 2: Loss=0.1559, mIoU=0.6411, F1=0.7265
- Epoch 3: Loss=0.1329, mIoU=0.6948, F1=0.7848
- Epoch 4: Loss=0.1231, mIoU=0.7113, F1=0.8009
- Epoch 5: Loss=0.1102, mIoU=0.7096, F1=0.7993
- Epoch 6: Loss=0.1076, mIoU=0.7375, F1=0.8251
- Epoch 7: Loss=0.1016, mIoU=0.7345, F1=0.8225
- Epoch 8: Loss=0.0991, mIoU=0.7373, F1=0.8249
- Epoch 9: Loss=0.0953, mIoU=0.7522, F1=0.8384
- Epoch 10: Loss=0.0932, mIoU=0.7266, F1=0.8152

当前轮次最佳结果: Round=6, Labeled=591, mIoU=0.7522, F1=0.8384

## Round 7

Labeled Pool Size: 679

- Epoch 1: Loss=0.3527, mIoU=0.6464, F1=0.7330
- Epoch 2: Loss=0.1572, mIoU=0.6567, F1=0.7443
- Epoch 3: Loss=0.1331, mIoU=0.7125, F1=0.8023
- Epoch 4: Loss=0.1200, mIoU=0.7172, F1=0.8065
- Epoch 5: Loss=0.1109, mIoU=0.7215, F1=0.8105
- Epoch 6: Loss=0.1039, mIoU=0.7327, F1=0.8210
- Epoch 7: Loss=0.1022, mIoU=0.7295, F1=0.8178
- Epoch 8: Loss=0.0956, mIoU=0.7301, F1=0.8183
- Epoch 9: Loss=0.0926, mIoU=0.7379, F1=0.8254
- Epoch 10: Loss=0.0896, mIoU=0.7463, F1=0.8328

当前轮次最佳结果: Round=7, Labeled=679, mIoU=0.7463, F1=0.8328

## Round 8

Labeled Pool Size: 767

- Epoch 1: Loss=0.2908, mIoU=0.6234, F1=0.7055
- Epoch 2: Loss=0.1422, mIoU=0.7036, F1=0.7936
- Epoch 3: Loss=0.1220, mIoU=0.6682, F1=0.7569
- Epoch 4: Loss=0.1113, mIoU=0.7239, F1=0.8128
- Epoch 5: Loss=0.1044, mIoU=0.7267, F1=0.8153
- Epoch 6: Loss=0.0995, mIoU=0.7009, F1=0.7907
- Epoch 7: Loss=0.0958, mIoU=0.7408, F1=0.8280
- Epoch 8: Loss=0.0926, mIoU=0.7158, F1=0.8050
- Epoch 9: Loss=0.0904, mIoU=0.7203, F1=0.8093
- Epoch 10: Loss=0.0859, mIoU=0.7373, F1=0.8248

当前轮次最佳结果: Round=8, Labeled=767, mIoU=0.7408, F1=0.8280

## Round 9

Labeled Pool Size: 855

- Epoch 1: Loss=0.3214, mIoU=0.6276, F1=0.7106
- Epoch 2: Loss=0.1433, mIoU=0.7084, F1=0.7985
- Epoch 3: Loss=0.1202, mIoU=0.6873, F1=0.7770
- Epoch 4: Loss=0.1098, mIoU=0.7383, F1=0.8259
- Epoch 5: Loss=0.1075, mIoU=0.7378, F1=0.8256
- Epoch 6: Loss=0.0998, mIoU=0.7423, F1=0.8293
- Epoch 7: Loss=0.0954, mIoU=0.7152, F1=0.8045
- Epoch 8: Loss=0.0938, mIoU=0.7402, F1=0.8276
- Epoch 9: Loss=0.0895, mIoU=0.7479, F1=0.8342
- Epoch 10: Loss=0.0872, mIoU=0.7606, F1=0.8453

当前轮次最佳结果: Round=9, Labeled=855, mIoU=0.7606, F1=0.8453

## Round 10

Labeled Pool Size: 943

- Epoch 1: Loss=0.2760, mIoU=0.6345, F1=0.7189
- Epoch 2: Loss=0.1338, mIoU=0.6945, F1=0.7848
- Epoch 3: Loss=0.1162, mIoU=0.7032, F1=0.7931
- Epoch 4: Loss=0.1055, mIoU=0.7354, F1=0.8236
- Epoch 5: Loss=0.1010, mIoU=0.7212, F1=0.8102
- Epoch 6: Loss=0.0973, mIoU=0.7443, F1=0.8314
- Epoch 7: Loss=0.0904, mIoU=0.7260, F1=0.8146
- Epoch 8: Loss=0.0892, mIoU=0.7061, F1=0.7957
- Epoch 9: Loss=0.0860, mIoU=0.7302, F1=0.8184
- Epoch 10: Loss=0.0830, mIoU=0.7569, F1=0.8421

当前轮次最佳结果: Round=10, Labeled=943, mIoU=0.7569, F1=0.8421

## Round 11

Labeled Pool Size: 1031

- Epoch 1: Loss=0.2179, mIoU=0.6043, F1=0.6812
- Epoch 2: Loss=0.1181, mIoU=0.6976, F1=0.7875
- Epoch 3: Loss=0.1029, mIoU=0.7317, F1=0.8199
- Epoch 4: Loss=0.0954, mIoU=0.7188, F1=0.8079
- Epoch 5: Loss=0.0905, mIoU=0.7516, F1=0.8376
- Epoch 6: Loss=0.0854, mIoU=0.7030, F1=0.7927
- Epoch 7: Loss=0.0833, mIoU=0.7519, F1=0.8377
- Epoch 8: Loss=0.0784, mIoU=0.7508, F1=0.8367
- Epoch 9: Loss=0.0741, mIoU=0.7447, F1=0.8314
- Epoch 10: Loss=0.0719, mIoU=0.7646, F1=0.8485

当前轮次最佳结果: Round=11, Labeled=1031, mIoU=0.7646, F1=0.8485

## Round 12

Labeled Pool Size: 1119

- Epoch 1: Loss=0.2012, mIoU=0.6463, F1=0.7326
- Epoch 2: Loss=0.1123, mIoU=0.6698, F1=0.7588
- Epoch 3: Loss=0.1008, mIoU=0.6716, F1=0.7605
- Epoch 4: Loss=0.0913, mIoU=0.7481, F1=0.8345
- Epoch 5: Loss=0.0843, mIoU=0.7178, F1=0.8070
- Epoch 6: Loss=0.0795, mIoU=0.7529, F1=0.8388
- Epoch 7: Loss=0.0769, mIoU=0.7531, F1=0.8388
- Epoch 8: Loss=0.0761, mIoU=0.7561, F1=0.8416
- Epoch 9: Loss=0.0718, mIoU=0.7337, F1=0.8215
- Epoch 10: Loss=0.0704, mIoU=0.7625, F1=0.8468

当前轮次最佳结果: Round=12, Labeled=1119, mIoU=0.7625, F1=0.8468

## Round 13

Labeled Pool Size: 1207

- Epoch 1: Loss=0.1874, mIoU=0.6874, F1=0.7775
- Epoch 2: Loss=0.1057, mIoU=0.6978, F1=0.7877
- Epoch 3: Loss=0.0920, mIoU=0.7104, F1=0.8000
- Epoch 4: Loss=0.0867, mIoU=0.7276, F1=0.8163
- Epoch 5: Loss=0.0808, mIoU=0.7517, F1=0.8377
- Epoch 6: Loss=0.0773, mIoU=0.7516, F1=0.8374
- Epoch 7: Loss=0.0734, mIoU=0.7542, F1=0.8398
- Epoch 8: Loss=0.0705, mIoU=0.7495, F1=0.8356
- Epoch 9: Loss=0.0684, mIoU=0.7469, F1=0.8333
- Epoch 10: Loss=0.0665, mIoU=0.7485, F1=0.8347

当前轮次最佳结果: Round=13, Labeled=1207, mIoU=0.7542, F1=0.8398

## Round 14

Labeled Pool Size: 1295

- Epoch 1: Loss=0.2353, mIoU=0.6147, F1=0.6944
- Epoch 2: Loss=0.1063, mIoU=0.7134, F1=0.8030
- Epoch 3: Loss=0.0930, mIoU=0.7064, F1=0.7960
- Epoch 4: Loss=0.0869, mIoU=0.7336, F1=0.8216
- Epoch 5: Loss=0.0810, mIoU=0.7265, F1=0.8164
- Epoch 6: Loss=0.0775, mIoU=0.7288, F1=0.8172
- Epoch 7: Loss=0.0754, mIoU=0.7431, F1=0.8300
- Epoch 8: Loss=0.0730, mIoU=0.7584, F1=0.8433
- Epoch 9: Loss=0.0693, mIoU=0.7579, F1=0.8428
- Epoch 10: Loss=0.0698, mIoU=0.7444, F1=0.8312

当前轮次最佳结果: Round=14, Labeled=1295, mIoU=0.7584, F1=0.8433

## Round 15

Labeled Pool Size: 1383

- Epoch 1: Loss=0.1689, mIoU=0.6349, F1=0.7192
- Epoch 2: Loss=0.0949, mIoU=0.6659, F1=0.7545
- Epoch 3: Loss=0.0844, mIoU=0.6845, F1=0.7741
- Epoch 4: Loss=0.0784, mIoU=0.7431, F1=0.8304
- Epoch 5: Loss=0.0742, mIoU=0.7085, F1=0.7981
- Epoch 6: Loss=0.0701, mIoU=0.7490, F1=0.8351
- Epoch 7: Loss=0.0689, mIoU=0.7608, F1=0.8454
- Epoch 8: Loss=0.0644, mIoU=0.7568, F1=0.8420
- Epoch 9: Loss=0.0632, mIoU=0.7492, F1=0.8353
- Epoch 10: Loss=0.0606, mIoU=0.7598, F1=0.8444

当前轮次最佳结果: Round=15, Labeled=1383, mIoU=0.7608, F1=0.8454


## 实验汇总

预算历史: [151, 239, 327, 415, 503, 591, 679, 767, 855, 943, 1031, 1119, 1207, 1295, 1383]
ALC: 0.6700
最终 mIoU: 0.7608
最终 F1: 0.8454
