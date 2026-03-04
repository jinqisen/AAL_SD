# 实验日志

实验名称: full_model_B_lambda_agent_v2
描述: 对照B（V2）：仍要求逐轮显式 set_lambda，但放宽上调约束、允许早期λ=0，并提高λ上限以跟随 suggested_lambda；不调整query_size/epochs
开始时间: 2026-03-02T22:31:33.012305

## Round 1

Labeled Pool Size: 151

- Epoch 1: Loss=0.5367, mIoU=0.4910, F1=0.5044
- Epoch 2: Loss=0.3043, mIoU=0.5023, F1=0.5230
- Epoch 3: Loss=0.1891, mIoU=0.5112, F1=0.5397
- Epoch 4: Loss=0.1311, mIoU=0.5238, F1=0.5605
- Epoch 5: Loss=0.0985, mIoU=0.5518, F1=0.6065
- Epoch 6: Loss=0.0816, mIoU=0.6159, F1=0.6971
- Epoch 7: Loss=0.0698, mIoU=0.6048, F1=0.6819
- Epoch 8: Loss=0.0653, mIoU=0.6237, F1=0.7060
- Epoch 9: Loss=0.0586, mIoU=0.6189, F1=0.6999
- Epoch 10: Loss=0.0551, mIoU=0.6587, F1=0.7472

当前轮次最佳结果: Round=1, Labeled=151, mIoU=0.6587, F1=0.7472

## Round 2

Labeled Pool Size: 239

- Epoch 1: Loss=0.3444, mIoU=0.5000, F1=0.5177
- Epoch 2: Loss=0.2044, mIoU=0.5863, F1=0.6572
- Epoch 3: Loss=0.1588, mIoU=0.6210, F1=0.7027
- Epoch 4: Loss=0.1374, mIoU=0.6288, F1=0.7119
- Epoch 5: Loss=0.1231, mIoU=0.6515, F1=0.7386
- Epoch 6: Loss=0.1138, mIoU=0.6773, F1=0.7669
- Epoch 7: Loss=0.1137, mIoU=0.6608, F1=0.7490
- Epoch 8: Loss=0.1043, mIoU=0.6808, F1=0.7707
- Epoch 9: Loss=0.1012, mIoU=0.6990, F1=0.7896
- Epoch 10: Loss=0.0929, mIoU=0.6937, F1=0.7839

当前轮次最佳结果: Round=2, Labeled=239, mIoU=0.6990, F1=0.7896

## Round 3

Labeled Pool Size: 327

- Epoch 1: Loss=0.4043, mIoU=0.5778, F1=0.6456
- Epoch 2: Loss=0.2107, mIoU=0.6392, F1=0.7248
- Epoch 3: Loss=0.1638, mIoU=0.6727, F1=0.7620
- Epoch 4: Loss=0.1459, mIoU=0.7035, F1=0.7936
- Epoch 5: Loss=0.1337, mIoU=0.6954, F1=0.7853
- Epoch 6: Loss=0.1240, mIoU=0.7006, F1=0.7908
- Epoch 7: Loss=0.1217, mIoU=0.6954, F1=0.7854
- Epoch 8: Loss=0.1145, mIoU=0.7126, F1=0.8023
- Epoch 9: Loss=0.1099, mIoU=0.7248, F1=0.8138
- Epoch 10: Loss=0.1102, mIoU=0.7329, F1=0.8213

当前轮次最佳结果: Round=3, Labeled=327, mIoU=0.7329, F1=0.8213

## Round 4

Labeled Pool Size: 415

- Epoch 1: Loss=0.3769, mIoU=0.5867, F1=0.6581
- Epoch 2: Loss=0.1991, mIoU=0.6843, F1=0.7745
- Epoch 3: Loss=0.1625, mIoU=0.6806, F1=0.7702
- Epoch 4: Loss=0.1414, mIoU=0.7022, F1=0.7922
- Epoch 5: Loss=0.1318, mIoU=0.6919, F1=0.7818
- Epoch 6: Loss=0.1268, mIoU=0.7143, F1=0.8042
- Epoch 7: Loss=0.1205, mIoU=0.6964, F1=0.7862
- Epoch 8: Loss=0.1169, mIoU=0.6807, F1=0.7703
- Epoch 9: Loss=0.1152, mIoU=0.6873, F1=0.7770
- Epoch 10: Loss=0.1078, mIoU=0.7334, F1=0.8217

当前轮次最佳结果: Round=4, Labeled=415, mIoU=0.7334, F1=0.8217

## Round 5

Labeled Pool Size: 503

- Epoch 1: Loss=0.3416, mIoU=0.5945, F1=0.6682
- Epoch 2: Loss=0.1776, mIoU=0.6229, F1=0.7047
- Epoch 3: Loss=0.1454, mIoU=0.6865, F1=0.7764
- Epoch 4: Loss=0.1307, mIoU=0.6836, F1=0.7733
- Epoch 5: Loss=0.1245, mIoU=0.6940, F1=0.7839
- Epoch 6: Loss=0.1151, mIoU=0.6994, F1=0.7892
- Epoch 7: Loss=0.1118, mIoU=0.7214, F1=0.8105
- Epoch 8: Loss=0.1111, mIoU=0.7334, F1=0.8216
- Epoch 9: Loss=0.1034, mIoU=0.7132, F1=0.8028
- Epoch 10: Loss=0.1009, mIoU=0.7324, F1=0.8206

当前轮次最佳结果: Round=5, Labeled=503, mIoU=0.7334, F1=0.8216

## Round 6

Labeled Pool Size: 591

- Epoch 1: Loss=0.3759, mIoU=0.6446, F1=0.7309
- Epoch 2: Loss=0.1696, mIoU=0.6668, F1=0.7556
- Epoch 3: Loss=0.1436, mIoU=0.6892, F1=0.7790
- Epoch 4: Loss=0.1287, mIoU=0.7264, F1=0.8153
- Epoch 5: Loss=0.1192, mIoU=0.7223, F1=0.8113
- Epoch 6: Loss=0.1149, mIoU=0.6611, F1=0.7491
- Epoch 7: Loss=0.1100, mIoU=0.7248, F1=0.8136
- Epoch 8: Loss=0.1052, mIoU=0.7284, F1=0.8178
- Epoch 9: Loss=0.1004, mIoU=0.7297, F1=0.8180
- Epoch 10: Loss=0.0977, mIoU=0.7382, F1=0.8258

当前轮次最佳结果: Round=6, Labeled=591, mIoU=0.7382, F1=0.8258

## Round 7

Labeled Pool Size: 679

- Epoch 1: Loss=0.3260, mIoU=0.6707, F1=0.7606
- Epoch 2: Loss=0.1554, mIoU=0.6982, F1=0.7884
- Epoch 3: Loss=0.1336, mIoU=0.7013, F1=0.7912
- Epoch 4: Loss=0.1175, mIoU=0.7302, F1=0.8187
- Epoch 5: Loss=0.1089, mIoU=0.7159, F1=0.8052
- Epoch 6: Loss=0.1065, mIoU=0.7068, F1=0.7966
- Epoch 7: Loss=0.0997, mIoU=0.7188, F1=0.8079
- Epoch 8: Loss=0.0976, mIoU=0.7458, F1=0.8325
- Epoch 9: Loss=0.0936, mIoU=0.7459, F1=0.8325
- Epoch 10: Loss=0.0893, mIoU=0.7537, F1=0.8393

当前轮次最佳结果: Round=7, Labeled=679, mIoU=0.7537, F1=0.8393

## Round 8

Labeled Pool Size: 767

- Epoch 1: Loss=0.3392, mIoU=0.6258, F1=0.7083
- Epoch 2: Loss=0.1514, mIoU=0.6982, F1=0.7883
- Epoch 3: Loss=0.1259, mIoU=0.7152, F1=0.8047
- Epoch 4: Loss=0.1165, mIoU=0.7045, F1=0.7945
- Epoch 5: Loss=0.1073, mIoU=0.7328, F1=0.8211
- Epoch 6: Loss=0.1008, mIoU=0.7343, F1=0.8223
- Epoch 7: Loss=0.0987, mIoU=0.7477, F1=0.8341
- Epoch 8: Loss=0.0937, mIoU=0.7401, F1=0.8275
- Epoch 9: Loss=0.0900, mIoU=0.7336, F1=0.8215
- Epoch 10: Loss=0.0881, mIoU=0.7553, F1=0.8407

当前轮次最佳结果: Round=8, Labeled=767, mIoU=0.7553, F1=0.8407

## Round 9

Labeled Pool Size: 855

- Epoch 1: Loss=0.3126, mIoU=0.6527, F1=0.7404
- Epoch 2: Loss=0.1385, mIoU=0.7032, F1=0.7933
- Epoch 3: Loss=0.1213, mIoU=0.7043, F1=0.7941
- Epoch 4: Loss=0.1087, mIoU=0.6853, F1=0.7749
- Epoch 5: Loss=0.1023, mIoU=0.7384, F1=0.8259
- Epoch 6: Loss=0.0985, mIoU=0.7213, F1=0.8103
- Epoch 7: Loss=0.0960, mIoU=0.7463, F1=0.8329
- Epoch 8: Loss=0.0929, mIoU=0.7421, F1=0.8291
- Epoch 9: Loss=0.0905, mIoU=0.7422, F1=0.8292
- Epoch 10: Loss=0.0878, mIoU=0.7484, F1=0.8348

当前轮次最佳结果: Round=9, Labeled=855, mIoU=0.7484, F1=0.8348

## Round 10

Labeled Pool Size: 943

- Epoch 1: Loss=0.2330, mIoU=0.6442, F1=0.7302
- Epoch 2: Loss=0.1234, mIoU=0.6739, F1=0.7631
- Epoch 3: Loss=0.1047, mIoU=0.7037, F1=0.7935
- Epoch 4: Loss=0.0974, mIoU=0.7086, F1=0.7982
- Epoch 5: Loss=0.0925, mIoU=0.7498, F1=0.8362
- Epoch 6: Loss=0.0886, mIoU=0.7113, F1=0.8007
- Epoch 7: Loss=0.0842, mIoU=0.7240, F1=0.8127
- Epoch 8: Loss=0.0795, mIoU=0.7474, F1=0.8338
- Epoch 9: Loss=0.0761, mIoU=0.7574, F1=0.8424
- Epoch 10: Loss=0.0751, mIoU=0.7432, F1=0.8301

当前轮次最佳结果: Round=10, Labeled=943, mIoU=0.7574, F1=0.8424

## Round 11

Labeled Pool Size: 1031

- Epoch 1: Loss=0.2055, mIoU=0.6470, F1=0.7336
- Epoch 2: Loss=0.1147, mIoU=0.6939, F1=0.7840
- Epoch 3: Loss=0.1002, mIoU=0.7223, F1=0.8113
- Epoch 4: Loss=0.0909, mIoU=0.7189, F1=0.8081
- Epoch 5: Loss=0.0852, mIoU=0.7156, F1=0.8048
- Epoch 6: Loss=0.0826, mIoU=0.7454, F1=0.8321
- Epoch 7: Loss=0.0803, mIoU=0.7556, F1=0.8411
- Epoch 8: Loss=0.0792, mIoU=0.7495, F1=0.8359
- Epoch 9: Loss=0.0733, mIoU=0.7559, F1=0.8411
- Epoch 10: Loss=0.0708, mIoU=0.7611, F1=0.8457

当前轮次最佳结果: Round=11, Labeled=1031, mIoU=0.7611, F1=0.8457

## Round 12

Labeled Pool Size: 1119

- Epoch 1: Loss=0.1971, mIoU=0.6575, F1=0.7454
- Epoch 2: Loss=0.1071, mIoU=0.6554, F1=0.7428
- Epoch 3: Loss=0.0956, mIoU=0.7250, F1=0.8138
- Epoch 4: Loss=0.0898, mIoU=0.7273, F1=0.8158
- Epoch 5: Loss=0.0814, mIoU=0.7448, F1=0.8316
- Epoch 6: Loss=0.0779, mIoU=0.7267, F1=0.8152
- Epoch 7: Loss=0.0768, mIoU=0.7391, F1=0.8265
- Epoch 8: Loss=0.0719, mIoU=0.7539, F1=0.8394
- Epoch 9: Loss=0.0703, mIoU=0.7462, F1=0.8328
- Epoch 10: Loss=0.0674, mIoU=0.7580, F1=0.8429

当前轮次最佳结果: Round=12, Labeled=1119, mIoU=0.7580, F1=0.8429

## Round 13

Labeled Pool Size: 1207

- Epoch 1: Loss=0.2288, mIoU=0.6609, F1=0.7493
- Epoch 2: Loss=0.1075, mIoU=0.7115, F1=0.8012
- Epoch 3: Loss=0.0937, mIoU=0.6897, F1=0.7795
- Epoch 4: Loss=0.0872, mIoU=0.7146, F1=0.8040
- Epoch 5: Loss=0.0807, mIoU=0.7359, F1=0.8236
- Epoch 6: Loss=0.0767, mIoU=0.7467, F1=0.8332
- Epoch 7: Loss=0.0734, mIoU=0.7362, F1=0.8238
- Epoch 8: Loss=0.0707, mIoU=0.7644, F1=0.8483
- Epoch 9: Loss=0.0689, mIoU=0.7643, F1=0.8482
- Epoch 10: Loss=0.0661, mIoU=0.7582, F1=0.8430

当前轮次最佳结果: Round=13, Labeled=1207, mIoU=0.7644, F1=0.8483

## Round 14

Labeled Pool Size: 1295

- Epoch 1: Loss=0.1713, mIoU=0.6631, F1=0.7515
- Epoch 2: Loss=0.0995, mIoU=0.6865, F1=0.7762
- Epoch 3: Loss=0.0880, mIoU=0.7294, F1=0.8179
- Epoch 4: Loss=0.0796, mIoU=0.7386, F1=0.8260
- Epoch 5: Loss=0.0749, mIoU=0.7566, F1=0.8418
- Epoch 6: Loss=0.0722, mIoU=0.7298, F1=0.8181
- Epoch 7: Loss=0.0707, mIoU=0.7218, F1=0.8106
- Epoch 8: Loss=0.0671, mIoU=0.7657, F1=0.8494
- Epoch 9: Loss=0.0644, mIoU=0.7446, F1=0.8314
- Epoch 10: Loss=0.0621, mIoU=0.7629, F1=0.8470

当前轮次最佳结果: Round=14, Labeled=1295, mIoU=0.7657, F1=0.8494

## Round 15

Labeled Pool Size: 1383

- Epoch 1: Loss=0.2137, mIoU=0.7018, F1=0.7921
- Epoch 2: Loss=0.0997, mIoU=0.7223, F1=0.8115
- Epoch 3: Loss=0.0872, mIoU=0.7202, F1=0.8092
- Epoch 4: Loss=0.0807, mIoU=0.7340, F1=0.8220
- Epoch 5: Loss=0.0756, mIoU=0.7146, F1=0.8055
- Epoch 6: Loss=0.0732, mIoU=0.7068, F1=0.7964
- Epoch 7: Loss=0.0705, mIoU=0.7376, F1=0.8252
- Epoch 8: Loss=0.0675, mIoU=0.7073, F1=0.7968
- Epoch 9: Loss=0.0657, mIoU=0.7465, F1=0.8329
- Epoch 10: Loss=0.0619, mIoU=0.7367, F1=0.8242

当前轮次最佳结果: Round=15, Labeled=1383, mIoU=0.7465, F1=0.8329


## 实验汇总

预算历史: [151, 239, 327, 415, 503, 591, 679, 767, 855, 943, 1031, 1119, 1207, 1295, 1383]
ALC: 0.6695
最终 mIoU: 0.7465
最终 F1: 0.8329
