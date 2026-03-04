# 实验日志

实验名称: no_agent
描述: 消融：移除Agent；λ由sigmoid自适应(随标注进度变化)，仅使用AD-KUCS数值策略选样
开始时间: 2026-02-20T06:20:30.821832

## Round 1

Labeled Pool Size: 151

- Epoch 1: Loss=0.6121, mIoU=0.4594, F1=0.5104
- Epoch 2: Loss=0.3445, mIoU=0.5060, F1=0.5314
- Epoch 3: Loss=0.2131, mIoU=0.5235, F1=0.5614
- Epoch 4: Loss=0.1462, mIoU=0.5273, F1=0.5674
- Epoch 5: Loss=0.1124, mIoU=0.5383, F1=0.5852
- Epoch 6: Loss=0.0915, mIoU=0.6037, F1=0.6815
- Epoch 7: Loss=0.0813, mIoU=0.6193, F1=0.7023
- Epoch 8: Loss=0.0722, mIoU=0.6010, F1=0.6771
- Epoch 9: Loss=0.0643, mIoU=0.6293, F1=0.7135
- Epoch 10: Loss=0.0619, mIoU=0.6222, F1=0.7042

当前轮次最佳结果: Round=1, Labeled=151, mIoU=0.6293, F1=0.7135

## Round 2

Labeled Pool Size: 239

- Epoch 1: Loss=0.4977, mIoU=0.5140, F1=0.5449
- Epoch 2: Loss=0.2607, mIoU=0.5528, F1=0.6080
- Epoch 3: Loss=0.1777, mIoU=0.6365, F1=0.7218
- Epoch 4: Loss=0.1422, mIoU=0.6136, F1=0.6930
- Epoch 5: Loss=0.1259, mIoU=0.6631, F1=0.7517
- Epoch 6: Loss=0.1176, mIoU=0.6467, F1=0.7331
- Epoch 7: Loss=0.1126, mIoU=0.6406, F1=0.7259
- Epoch 8: Loss=0.1013, mIoU=0.6633, F1=0.7517
- Epoch 9: Loss=0.0995, mIoU=0.6925, F1=0.7825
- Epoch 10: Loss=0.0942, mIoU=0.6981, F1=0.7885

当前轮次最佳结果: Round=2, Labeled=239, mIoU=0.6981, F1=0.7885

## Round 3

Labeled Pool Size: 327

- Epoch 1: Loss=0.3171, mIoU=0.5669, F1=0.6294
- Epoch 2: Loss=0.1826, mIoU=0.6123, F1=0.6916
- Epoch 3: Loss=0.1500, mIoU=0.6105, F1=0.6892
- Epoch 4: Loss=0.1374, mIoU=0.6724, F1=0.7618
- Epoch 5: Loss=0.1250, mIoU=0.6558, F1=0.7435
- Epoch 6: Loss=0.1197, mIoU=0.6724, F1=0.7616
- Epoch 7: Loss=0.1174, mIoU=0.6839, F1=0.7738
- Epoch 8: Loss=0.1089, mIoU=0.6381, F1=0.7230
- Epoch 9: Loss=0.1038, mIoU=0.7023, F1=0.7928
- Epoch 10: Loss=0.0996, mIoU=0.7038, F1=0.7938

当前轮次最佳结果: Round=3, Labeled=327, mIoU=0.7038, F1=0.7938

## Round 4

Labeled Pool Size: 415

- Epoch 1: Loss=0.3880, mIoU=0.6395, F1=0.7261
- Epoch 2: Loss=0.1954, mIoU=0.6297, F1=0.7130
- Epoch 3: Loss=0.1534, mIoU=0.6871, F1=0.7771
- Epoch 4: Loss=0.1383, mIoU=0.6658, F1=0.7548
- Epoch 5: Loss=0.1290, mIoU=0.7182, F1=0.8078
- Epoch 6: Loss=0.1191, mIoU=0.7025, F1=0.7925
- Epoch 7: Loss=0.1127, mIoU=0.7031, F1=0.7930
- Epoch 8: Loss=0.1106, mIoU=0.7207, F1=0.8100
- Epoch 9: Loss=0.1042, mIoU=0.7266, F1=0.8153
- Epoch 10: Loss=0.1005, mIoU=0.7396, F1=0.8272

当前轮次最佳结果: Round=4, Labeled=415, mIoU=0.7396, F1=0.8272

## Round 5

Labeled Pool Size: 503

- Epoch 1: Loss=0.3412, mIoU=0.6235, F1=0.7057
- Epoch 2: Loss=0.1698, mIoU=0.6339, F1=0.7180
- Epoch 3: Loss=0.1402, mIoU=0.7015, F1=0.7914
- Epoch 4: Loss=0.1254, mIoU=0.7123, F1=0.8020
- Epoch 5: Loss=0.1180, mIoU=0.6966, F1=0.7864
- Epoch 6: Loss=0.1095, mIoU=0.7318, F1=0.8205
- Epoch 7: Loss=0.1033, mIoU=0.7331, F1=0.8213
- Epoch 8: Loss=0.1009, mIoU=0.7389, F1=0.8265
- Epoch 9: Loss=0.0978, mIoU=0.7246, F1=0.8134
- Epoch 10: Loss=0.0948, mIoU=0.7183, F1=0.8074

当前轮次最佳结果: Round=5, Labeled=503, mIoU=0.7389, F1=0.8265

## Round 6

Labeled Pool Size: 591

- Epoch 1: Loss=0.3095, mIoU=0.6289, F1=0.7122
- Epoch 2: Loss=0.1582, mIoU=0.6651, F1=0.7536
- Epoch 3: Loss=0.1329, mIoU=0.6947, F1=0.7847
- Epoch 4: Loss=0.1220, mIoU=0.7189, F1=0.8088
- Epoch 5: Loss=0.1145, mIoU=0.7254, F1=0.8141
- Epoch 6: Loss=0.1068, mIoU=0.7292, F1=0.8178
- Epoch 7: Loss=0.1036, mIoU=0.7199, F1=0.8091
- Epoch 8: Loss=0.1022, mIoU=0.7425, F1=0.8298
- Epoch 9: Loss=0.0949, mIoU=0.7149, F1=0.8042
- Epoch 10: Loss=0.0907, mIoU=0.7390, F1=0.8264

当前轮次最佳结果: Round=6, Labeled=591, mIoU=0.7425, F1=0.8298

## Round 7

Labeled Pool Size: 679

- Epoch 1: Loss=0.3522, mIoU=0.6241, F1=0.7065
- Epoch 2: Loss=0.1588, mIoU=0.6828, F1=0.7727
- Epoch 3: Loss=0.1347, mIoU=0.7081, F1=0.7978
- Epoch 4: Loss=0.1198, mIoU=0.6871, F1=0.7768
- Epoch 5: Loss=0.1139, mIoU=0.7133, F1=0.8029
- Epoch 6: Loss=0.1087, mIoU=0.7159, F1=0.8054
- Epoch 7: Loss=0.1026, mIoU=0.7239, F1=0.8127
- Epoch 8: Loss=0.0985, mIoU=0.6911, F1=0.7809
- Epoch 9: Loss=0.0950, mIoU=0.7116, F1=0.8010
- Epoch 10: Loss=0.0914, mIoU=0.7448, F1=0.8322

当前轮次最佳结果: Round=7, Labeled=679, mIoU=0.7448, F1=0.8322

## Round 8

Labeled Pool Size: 767

- Epoch 1: Loss=0.2952, mIoU=0.6253, F1=0.7077
- Epoch 2: Loss=0.1434, mIoU=0.6743, F1=0.7635
- Epoch 3: Loss=0.1232, mIoU=0.7101, F1=0.8000
- Epoch 4: Loss=0.1132, mIoU=0.7177, F1=0.8070
- Epoch 5: Loss=0.1020, mIoU=0.7439, F1=0.8310
- Epoch 6: Loss=0.0952, mIoU=0.7412, F1=0.8285
- Epoch 7: Loss=0.0898, mIoU=0.7293, F1=0.8177
- Epoch 8: Loss=0.0893, mIoU=0.7470, F1=0.8336
- Epoch 9: Loss=0.0836, mIoU=0.7444, F1=0.8318
- Epoch 10: Loss=0.0866, mIoU=0.7479, F1=0.8343

当前轮次最佳结果: Round=8, Labeled=767, mIoU=0.7479, F1=0.8343

## Round 9

Labeled Pool Size: 855

- Epoch 1: Loss=0.3361, mIoU=0.6587, F1=0.7474
- Epoch 2: Loss=0.1401, mIoU=0.6509, F1=0.7379
- Epoch 3: Loss=0.1195, mIoU=0.7126, F1=0.8022
- Epoch 4: Loss=0.1068, mIoU=0.6992, F1=0.7890
- Epoch 5: Loss=0.1035, mIoU=0.6959, F1=0.7857
- Epoch 6: Loss=0.0977, mIoU=0.7238, F1=0.8126
- Epoch 7: Loss=0.0960, mIoU=0.7315, F1=0.8196
- Epoch 8: Loss=0.0916, mIoU=0.7279, F1=0.8163
- Epoch 9: Loss=0.0890, mIoU=0.7415, F1=0.8286
- Epoch 10: Loss=0.0836, mIoU=0.7097, F1=0.7992

当前轮次最佳结果: Round=9, Labeled=855, mIoU=0.7415, F1=0.8286

## Round 10

Labeled Pool Size: 943

- Epoch 1: Loss=0.3118, mIoU=0.6593, F1=0.7482
- Epoch 2: Loss=0.1299, mIoU=0.6772, F1=0.7665
- Epoch 3: Loss=0.1083, mIoU=0.7011, F1=0.7909
- Epoch 4: Loss=0.1003, mIoU=0.6595, F1=0.7473
- Epoch 5: Loss=0.0937, mIoU=0.6926, F1=0.7823
- Epoch 6: Loss=0.0884, mIoU=0.7503, F1=0.8365
- Epoch 7: Loss=0.0872, mIoU=0.7386, F1=0.8260
- Epoch 8: Loss=0.0811, mIoU=0.7422, F1=0.8293
- Epoch 9: Loss=0.0786, mIoU=0.7094, F1=0.7988
- Epoch 10: Loss=0.0763, mIoU=0.7540, F1=0.8395

当前轮次最佳结果: Round=10, Labeled=943, mIoU=0.7540, F1=0.8395

## Round 11

Labeled Pool Size: 1031

- Epoch 1: Loss=0.2146, mIoU=0.6455, F1=0.7318
- Epoch 2: Loss=0.1164, mIoU=0.7133, F1=0.8029
- Epoch 3: Loss=0.1010, mIoU=0.7260, F1=0.8148
- Epoch 4: Loss=0.0908, mIoU=0.7338, F1=0.8218
- Epoch 5: Loss=0.0854, mIoU=0.7462, F1=0.8329
- Epoch 6: Loss=0.0824, mIoU=0.7361, F1=0.8238
- Epoch 7: Loss=0.0781, mIoU=0.7503, F1=0.8363
- Epoch 8: Loss=0.0750, mIoU=0.7506, F1=0.8365
- Epoch 9: Loss=0.0712, mIoU=0.7255, F1=0.8140
- Epoch 10: Loss=0.0693, mIoU=0.7627, F1=0.8469

当前轮次最佳结果: Round=11, Labeled=1031, mIoU=0.7627, F1=0.8469

## Round 12

Labeled Pool Size: 1119

- Epoch 1: Loss=0.1987, mIoU=0.6429, F1=0.7288
- Epoch 2: Loss=0.1090, mIoU=0.7018, F1=0.7917
- Epoch 3: Loss=0.0943, mIoU=0.6965, F1=0.7863
- Epoch 4: Loss=0.0864, mIoU=0.7445, F1=0.8317
- Epoch 5: Loss=0.0823, mIoU=0.7381, F1=0.8255
- Epoch 6: Loss=0.0753, mIoU=0.7529, F1=0.8390
- Epoch 7: Loss=0.0734, mIoU=0.7563, F1=0.8417
- Epoch 8: Loss=0.0720, mIoU=0.7565, F1=0.8417
- Epoch 9: Loss=0.0682, mIoU=0.7517, F1=0.8374
- Epoch 10: Loss=0.0668, mIoU=0.7458, F1=0.8323

当前轮次最佳结果: Round=12, Labeled=1119, mIoU=0.7565, F1=0.8417

## Round 13

Labeled Pool Size: 1207

- Epoch 1: Loss=0.1810, mIoU=0.6620, F1=0.7503
- Epoch 2: Loss=0.1010, mIoU=0.7071, F1=0.7968
- Epoch 3: Loss=0.0877, mIoU=0.7101, F1=0.7997
- Epoch 4: Loss=0.0819, mIoU=0.7123, F1=0.8017
- Epoch 5: Loss=0.0777, mIoU=0.7451, F1=0.8321
- Epoch 6: Loss=0.0727, mIoU=0.7222, F1=0.8110
- Epoch 7: Loss=0.0694, mIoU=0.7564, F1=0.8417
- Epoch 8: Loss=0.0678, mIoU=0.7306, F1=0.8187
- Epoch 9: Loss=0.0639, mIoU=0.7303, F1=0.8185
- Epoch 10: Loss=0.0638, mIoU=0.7565, F1=0.8417

当前轮次最佳结果: Round=13, Labeled=1207, mIoU=0.7565, F1=0.8417

## Round 14

Labeled Pool Size: 1295

- Epoch 1: Loss=0.2287, mIoU=0.6698, F1=0.7591
- Epoch 2: Loss=0.0981, mIoU=0.6492, F1=0.7357
- Epoch 3: Loss=0.0851, mIoU=0.7181, F1=0.8073
- Epoch 4: Loss=0.0797, mIoU=0.7239, F1=0.8134
- Epoch 5: Loss=0.0762, mIoU=0.7463, F1=0.8332
- Epoch 6: Loss=0.0724, mIoU=0.7373, F1=0.8250
- Epoch 7: Loss=0.0687, mIoU=0.7526, F1=0.8384
- Epoch 8: Loss=0.0657, mIoU=0.7503, F1=0.8363
- Epoch 9: Loss=0.0632, mIoU=0.7391, F1=0.8264
- Epoch 10: Loss=0.0622, mIoU=0.7334, F1=0.8213

当前轮次最佳结果: Round=14, Labeled=1295, mIoU=0.7526, F1=0.8384

## Round 15

Labeled Pool Size: 1383

- Epoch 1: Loss=0.1647, mIoU=0.6972, F1=0.7873
- Epoch 2: Loss=0.0903, mIoU=0.6064, F1=0.6836
- Epoch 3: Loss=0.0779, mIoU=0.7234, F1=0.8123
- Epoch 4: Loss=0.0730, mIoU=0.7360, F1=0.8237
- Epoch 5: Loss=0.0673, mIoU=0.7196, F1=0.8085
- Epoch 6: Loss=0.0660, mIoU=0.7286, F1=0.8169
- Epoch 7: Loss=0.0633, mIoU=0.7549, F1=0.8405
- Epoch 8: Loss=0.0605, mIoU=0.7549, F1=0.8402
- Epoch 9: Loss=0.0578, mIoU=0.7579, F1=0.8428
- Epoch 10: Loss=0.0556, mIoU=0.7444, F1=0.8311

当前轮次最佳结果: Round=15, Labeled=1383, mIoU=0.7579, F1=0.8428


## 实验汇总

预算历史: [151, 239, 327, 415, 503, 591, 679, 767, 855, 943, 1031, 1119, 1207, 1295, 1383]
ALC: 0.6665
最终 mIoU: 0.7579
最终 F1: 0.8428
