# 实验日志

实验名称: full_model_k_coreset_to_labeled
描述: 完整模型：K 使用 coreset-to-labeled 距离/覆盖（min_dist 到已标注特征）+ 自适应阈值 rollback
开始时间: 2026-02-10T18:11:21.320125

## Round 1

Labeled Pool Size: 151

- Epoch 1: Loss=0.6114, mIoU=0.4759, F1=0.5166
- Epoch 2: Loss=0.3409, mIoU=0.5047, F1=0.5299
- Epoch 3: Loss=0.2099, mIoU=0.5042, F1=0.5258
- Epoch 4: Loss=0.1445, mIoU=0.5066, F1=0.5302
- Epoch 5: Loss=0.1138, mIoU=0.5228, F1=0.5594
- Epoch 6: Loss=0.0905, mIoU=0.5987, F1=0.6754
- Epoch 7: Loss=0.0789, mIoU=0.6055, F1=0.6844
- Epoch 8: Loss=0.0704, mIoU=0.5506, F1=0.6044
- Epoch 9: Loss=0.0659, mIoU=0.5875, F1=0.6586
- Epoch 10: Loss=0.0591, mIoU=0.6101, F1=0.6888

当前轮次最佳结果: Round=1, Labeled=151, mIoU=0.6101, F1=0.6888

- 回退topk: 否，补齐数量: 0
## Round 2

Labeled Pool Size: 239

- Epoch 1: Loss=0.4966, mIoU=0.5684, F1=0.6338
- Epoch 2: Loss=0.2571, mIoU=0.6207, F1=0.7029
- Epoch 3: Loss=0.1769, mIoU=0.5968, F1=0.6713
- Epoch 4: Loss=0.1487, mIoU=0.6305, F1=0.7139
- Epoch 5: Loss=0.1286, mIoU=0.6798, F1=0.7696
- Epoch 6: Loss=0.1118, mIoU=0.6665, F1=0.7553
- Epoch 7: Loss=0.1071, mIoU=0.6389, F1=0.7241
- Epoch 8: Loss=0.1011, mIoU=0.6345, F1=0.7187
- Epoch 9: Loss=0.0967, mIoU=0.6841, F1=0.7740
- Epoch 10: Loss=0.0896, mIoU=0.6534, F1=0.7409

当前轮次最佳结果: Round=2, Labeled=239, mIoU=0.6841, F1=0.7740

- 回退topk: 否，补齐数量: 0
## Round 3

Labeled Pool Size: 327

- Epoch 1: Loss=0.3260, mIoU=0.5549, F1=0.6112
- Epoch 2: Loss=0.1876, mIoU=0.6456, F1=0.7321
- Epoch 3: Loss=0.1553, mIoU=0.6566, F1=0.7444
- Epoch 4: Loss=0.1354, mIoU=0.6417, F1=0.7273
- Epoch 5: Loss=0.1245, mIoU=0.6926, F1=0.7826
- Epoch 6: Loss=0.1222, mIoU=0.6923, F1=0.7825
- Epoch 7: Loss=0.1204, mIoU=0.6953, F1=0.7854
- Epoch 8: Loss=0.1111, mIoU=0.7115, F1=0.8012
- Epoch 9: Loss=0.1070, mIoU=0.7168, F1=0.8062
- Epoch 10: Loss=0.1046, mIoU=0.7131, F1=0.8027

当前轮次最佳结果: Round=3, Labeled=327, mIoU=0.7168, F1=0.8062

- 回退topk: 否，补齐数量: 0
## Round 4

Labeled Pool Size: 415

- Epoch 1: Loss=0.3872, mIoU=0.5998, F1=0.6757
- Epoch 2: Loss=0.1930, mIoU=0.6936, F1=0.7841
- Epoch 3: Loss=0.1548, mIoU=0.6266, F1=0.7094
- Epoch 4: Loss=0.1356, mIoU=0.6630, F1=0.7514
- Epoch 5: Loss=0.1256, mIoU=0.6758, F1=0.7652
- Epoch 6: Loss=0.1181, mIoU=0.7066, F1=0.7965
- Epoch 7: Loss=0.1134, mIoU=0.6808, F1=0.7703
- Epoch 8: Loss=0.1109, mIoU=0.7073, F1=0.7973
- Epoch 9: Loss=0.1029, mIoU=0.7293, F1=0.8179
- Epoch 10: Loss=0.1026, mIoU=0.7251, F1=0.8139

当前轮次最佳结果: Round=4, Labeled=415, mIoU=0.7293, F1=0.8179

- 回退topk: 否，补齐数量: 0
## Round 5

Labeled Pool Size: 503

- Epoch 1: Loss=0.3432, mIoU=0.6308, F1=0.7144
- Epoch 2: Loss=0.1724, mIoU=0.6355, F1=0.7201
- Epoch 3: Loss=0.1444, mIoU=0.6771, F1=0.7666
- Epoch 4: Loss=0.1319, mIoU=0.6916, F1=0.7818
- Epoch 5: Loss=0.1237, mIoU=0.7290, F1=0.8175
- Epoch 6: Loss=0.1159, mIoU=0.7312, F1=0.8196
- Epoch 7: Loss=0.1101, mIoU=0.7375, F1=0.8251
- Epoch 8: Loss=0.1074, mIoU=0.7214, F1=0.8104
- Epoch 9: Loss=0.1023, mIoU=0.7276, F1=0.8161
- Epoch 10: Loss=0.0998, mIoU=0.7241, F1=0.8128

当前轮次最佳结果: Round=5, Labeled=503, mIoU=0.7375, F1=0.8251

- 回退topk: 否，补齐数量: 0
## Round 6

Labeled Pool Size: 591

- Epoch 1: Loss=0.3130, mIoU=0.6096, F1=0.6884
- Epoch 2: Loss=0.1672, mIoU=0.6451, F1=0.7314
- Epoch 3: Loss=0.1413, mIoU=0.6812, F1=0.7708
- Epoch 4: Loss=0.1259, mIoU=0.6675, F1=0.7562
- Epoch 5: Loss=0.1185, mIoU=0.7212, F1=0.8104
- Epoch 6: Loss=0.1117, mIoU=0.7095, F1=0.7991
- Epoch 7: Loss=0.1104, mIoU=0.6971, F1=0.7869
- Epoch 8: Loss=0.1033, mIoU=0.7307, F1=0.8189
- Epoch 9: Loss=0.1014, mIoU=0.7525, F1=0.8383
- Epoch 10: Loss=0.0951, mIoU=0.7273, F1=0.8158

当前轮次最佳结果: Round=6, Labeled=591, mIoU=0.7525, F1=0.8383

- 回退topk: 否，补齐数量: 0
## Round 7

Labeled Pool Size: 679

- Epoch 1: Loss=0.3539, mIoU=0.6450, F1=0.7314
- Epoch 2: Loss=0.1615, mIoU=0.6582, F1=0.7461
- Epoch 3: Loss=0.1326, mIoU=0.7179, F1=0.8074
- Epoch 4: Loss=0.1197, mIoU=0.7185, F1=0.8080
- Epoch 5: Loss=0.1164, mIoU=0.7291, F1=0.8175
- Epoch 6: Loss=0.1091, mIoU=0.7357, F1=0.8235
- Epoch 7: Loss=0.1028, mIoU=0.7201, F1=0.8091
- Epoch 8: Loss=0.0983, mIoU=0.7384, F1=0.8259
- Epoch 9: Loss=0.0936, mIoU=0.7466, F1=0.8332
- Epoch 10: Loss=0.0920, mIoU=0.7562, F1=0.8416

当前轮次最佳结果: Round=7, Labeled=679, mIoU=0.7562, F1=0.8416

- 回退topk: 否，补齐数量: 0
## Round 8

Labeled Pool Size: 767

- Epoch 1: Loss=0.2920, mIoU=0.6552, F1=0.7430
- Epoch 2: Loss=0.1448, mIoU=0.6827, F1=0.7726
- Epoch 3: Loss=0.1206, mIoU=0.7044, F1=0.7943
- Epoch 4: Loss=0.1115, mIoU=0.7359, F1=0.8238
- Epoch 5: Loss=0.1026, mIoU=0.7183, F1=0.8074
- Epoch 6: Loss=0.0979, mIoU=0.7318, F1=0.8200
- Epoch 7: Loss=0.0940, mIoU=0.7435, F1=0.8304
- Epoch 8: Loss=0.0903, mIoU=0.7545, F1=0.8400
- Epoch 9: Loss=0.0869, mIoU=0.7453, F1=0.8321
- Epoch 10: Loss=0.0823, mIoU=0.7273, F1=0.8158

当前轮次最佳结果: Round=8, Labeled=767, mIoU=0.7545, F1=0.8400

- 回退topk: 否，补齐数量: 0
## Round 9

Labeled Pool Size: 855

- Epoch 1: Loss=0.3266, mIoU=0.6348, F1=0.7199
- Epoch 2: Loss=0.1431, mIoU=0.6524, F1=0.7394
- Epoch 3: Loss=0.1196, mIoU=0.7100, F1=0.7996
- Epoch 4: Loss=0.1095, mIoU=0.7274, F1=0.8165
- Epoch 5: Loss=0.1038, mIoU=0.7106, F1=0.8001
- Epoch 6: Loss=0.0987, mIoU=0.7061, F1=0.7957
- Epoch 7: Loss=0.0953, mIoU=0.7395, F1=0.8272
- Epoch 8: Loss=0.0923, mIoU=0.7521, F1=0.8383
- Epoch 9: Loss=0.0871, mIoU=0.7316, F1=0.8197
- Epoch 10: Loss=0.0855, mIoU=0.7465, F1=0.8330

当前轮次最佳结果: Round=9, Labeled=855, mIoU=0.7521, F1=0.8383

- 回退topk: 否，补齐数量: 0
## Round 10

Labeled Pool Size: 943

- Epoch 1: Loss=0.2974, mIoU=0.5903, F1=0.6624
- Epoch 2: Loss=0.1288, mIoU=0.6875, F1=0.7772
- Epoch 3: Loss=0.1100, mIoU=0.6498, F1=0.7370
- Epoch 4: Loss=0.1054, mIoU=0.6788, F1=0.7681
- Epoch 5: Loss=0.0972, mIoU=0.7425, F1=0.8295
- Epoch 6: Loss=0.0919, mIoU=0.7510, F1=0.8370
- Epoch 7: Loss=0.0881, mIoU=0.7513, F1=0.8372
- Epoch 8: Loss=0.0874, mIoU=0.7529, F1=0.8386
- Epoch 9: Loss=0.0809, mIoU=0.7423, F1=0.8293
- Epoch 10: Loss=0.0788, mIoU=0.7519, F1=0.8377

当前轮次最佳结果: Round=10, Labeled=943, mIoU=0.7529, F1=0.8386

- 回退topk: 否，补齐数量: 0
## Round 11

Labeled Pool Size: 1031

- Epoch 1: Loss=0.2137, mIoU=0.6335, F1=0.7176
- Epoch 2: Loss=0.1128, mIoU=0.7045, F1=0.7946
- Epoch 3: Loss=0.1005, mIoU=0.7320, F1=0.8205
- Epoch 4: Loss=0.0953, mIoU=0.7100, F1=0.7998
- Epoch 5: Loss=0.0856, mIoU=0.7461, F1=0.8327
- Epoch 6: Loss=0.0823, mIoU=0.6800, F1=0.7694
- Epoch 7: Loss=0.0808, mIoU=0.7498, F1=0.8361
- Epoch 8: Loss=0.0771, mIoU=0.7324, F1=0.8203
- Epoch 9: Loss=0.0736, mIoU=0.7435, F1=0.8304
- Epoch 10: Loss=0.0711, mIoU=0.7438, F1=0.8306

当前轮次最佳结果: Round=11, Labeled=1031, mIoU=0.7498, F1=0.8361

- 回退topk: 否，补齐数量: 0
## Round 12

Labeled Pool Size: 1119

- Epoch 1: Loss=0.1986, mIoU=0.6764, F1=0.7661
- Epoch 2: Loss=0.1088, mIoU=0.6692, F1=0.7581
- Epoch 3: Loss=0.0926, mIoU=0.7239, F1=0.8128
- Epoch 4: Loss=0.0869, mIoU=0.7233, F1=0.8121
- Epoch 5: Loss=0.0809, mIoU=0.7355, F1=0.8234
- Epoch 6: Loss=0.0782, mIoU=0.7303, F1=0.8197
- Epoch 7: Loss=0.0739, mIoU=0.7374, F1=0.8250
- Epoch 8: Loss=0.0696, mIoU=0.7588, F1=0.8437
- Epoch 9: Loss=0.0677, mIoU=0.7401, F1=0.8274
- Epoch 10: Loss=0.0671, mIoU=0.7484, F1=0.8347

当前轮次最佳结果: Round=12, Labeled=1119, mIoU=0.7588, F1=0.8437

- 回退topk: 否，补齐数量: 0
## Round 13

Labeled Pool Size: 1207

- Epoch 1: Loss=0.1785, mIoU=0.6707, F1=0.7599
- Epoch 2: Loss=0.1010, mIoU=0.7240, F1=0.8131
- Epoch 3: Loss=0.0892, mIoU=0.7076, F1=0.7973
- Epoch 4: Loss=0.0829, mIoU=0.6929, F1=0.7826
- Epoch 5: Loss=0.0761, mIoU=0.7426, F1=0.8297
- Epoch 6: Loss=0.0733, mIoU=0.6552, F1=0.7425
- Epoch 7: Loss=0.0687, mIoU=0.7458, F1=0.8324
- Epoch 8: Loss=0.0653, mIoU=0.7505, F1=0.8364
- Epoch 9: Loss=0.0641, mIoU=0.7469, F1=0.8334
- Epoch 10: Loss=0.0619, mIoU=0.7640, F1=0.8481

当前轮次最佳结果: Round=13, Labeled=1207, mIoU=0.7640, F1=0.8481

- 回退topk: 否，补齐数量: 0
## Round 14

Labeled Pool Size: 1295

- Epoch 1: Loss=0.2384, mIoU=0.6597, F1=0.7478
- Epoch 2: Loss=0.0996, mIoU=0.7227, F1=0.8122
- Epoch 3: Loss=0.0888, mIoU=0.7063, F1=0.7960
- Epoch 4: Loss=0.0791, mIoU=0.7195, F1=0.8086
- Epoch 5: Loss=0.0756, mIoU=0.7493, F1=0.8357
- Epoch 6: Loss=0.0720, mIoU=0.7102, F1=0.7996
- Epoch 7: Loss=0.0691, mIoU=0.7471, F1=0.8336
- Epoch 8: Loss=0.0681, mIoU=0.7444, F1=0.8312
- Epoch 9: Loss=0.0646, mIoU=0.7182, F1=0.8073
- Epoch 10: Loss=0.0632, mIoU=0.7266, F1=0.8152

当前轮次最佳结果: Round=14, Labeled=1295, mIoU=0.7493, F1=0.8357

- 回退topk: 否，补齐数量: 0
## Round 15

Labeled Pool Size: 1383

- Epoch 1: Loss=0.1651, mIoU=0.6733, F1=0.7628
- Epoch 2: Loss=0.0918, mIoU=0.6901, F1=0.7799
- Epoch 3: Loss=0.0786, mIoU=0.7409, F1=0.8284
- Epoch 4: Loss=0.0721, mIoU=0.7110, F1=0.8004
- Epoch 5: Loss=0.0674, mIoU=0.7495, F1=0.8360
- Epoch 6: Loss=0.0654, mIoU=0.7490, F1=0.8356
- Epoch 7: Loss=0.0613, mIoU=0.7314, F1=0.8194
- Epoch 8: Loss=0.0583, mIoU=0.7565, F1=0.8417
- Epoch 9: Loss=0.0578, mIoU=0.7569, F1=0.8420
- Epoch 10: Loss=0.0555, mIoU=0.7631, F1=0.8471

当前轮次最佳结果: Round=15, Labeled=1383, mIoU=0.7631, F1=0.8471


## 实验汇总

预算历史: [151, 239, 327, 415, 503, 591, 679, 767, 855, 943, 1031, 1119, 1207, 1295, 1383]
ALC: 0.6676
最终 mIoU: 0.7631
最终 F1: 0.8471
