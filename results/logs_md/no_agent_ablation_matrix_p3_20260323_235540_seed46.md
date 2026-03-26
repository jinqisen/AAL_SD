# 实验日志

实验名称: no_agent
描述: 消融：w/o Agent（argmax）；保留三阶段λ policy，与full_model一致
开始时间: 2026-03-26T07:05:40.787875

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.4589, mIoU=0.4946, F1=0.5031
- Epoch 2: Loss=0.2577, mIoU=0.4921, F1=0.4972
- Epoch 3: Loss=0.1726, mIoU=0.5073, F1=0.5261
- Epoch 4: Loss=0.1285, mIoU=0.5036, F1=0.5193
- Epoch 5: Loss=0.1020, mIoU=0.5080, F1=0.5275
- Epoch 6: Loss=0.0870, mIoU=0.5088, F1=0.5291
- Epoch 7: Loss=0.0763, mIoU=0.4975, F1=0.5076
- Epoch 8: Loss=0.0673, mIoU=0.5123, F1=0.5354
- Epoch 9: Loss=0.0643, mIoU=0.5110, F1=0.5331
- Epoch 10: Loss=0.0591, mIoU=0.4956, F1=0.5038

本轮结果: Round=1, Labeled=189, Selection=best_val (epoch=8), mIoU=0.5123, F1=0.5354, peak_mIoU=0.5123

Round=1, Labeled=189, mIoU=0.5123, F1=0.5354

## Round 2

Labeled Pool Size: 277

- Epoch 1: Loss=0.5450, mIoU=0.5174, F1=0.5518
- Epoch 2: Loss=0.2780, mIoU=0.5150, F1=0.5403
- Epoch 3: Loss=0.1948, mIoU=0.4991, F1=0.5106
- Epoch 4: Loss=0.1571, mIoU=0.5094, F1=0.5301
- Epoch 5: Loss=0.1424, mIoU=0.5120, F1=0.5349
- Epoch 6: Loss=0.1275, mIoU=0.5492, F1=0.5983
- Epoch 7: Loss=0.1207, mIoU=0.5493, F1=0.5984
- Epoch 8: Loss=0.1121, mIoU=0.5360, F1=0.5768
- Epoch 9: Loss=0.1102, mIoU=0.5481, F1=0.5966
- Epoch 10: Loss=0.1035, mIoU=0.5228, F1=0.5542

本轮结果: Round=2, Labeled=277, Selection=best_val (epoch=7), mIoU=0.5493, F1=0.5984, peak_mIoU=0.5493

Round=2, Labeled=277, mIoU=0.5493, F1=0.5984

## Round 3

Labeled Pool Size: 365

- Epoch 1: Loss=0.4531, mIoU=0.5519, F1=0.6047
- Epoch 2: Loss=0.2325, mIoU=0.5486, F1=0.5975
- Epoch 3: Loss=0.1739, mIoU=0.5962, F1=0.6676
- Epoch 4: Loss=0.1526, mIoU=0.6017, F1=0.6751
- Epoch 5: Loss=0.1376, mIoU=0.5600, F1=0.6151
- Epoch 6: Loss=0.1294, mIoU=0.5655, F1=0.6240
- Epoch 7: Loss=0.1261, mIoU=0.6380, F1=0.7220
- Epoch 8: Loss=0.1142, mIoU=0.5401, F1=0.5835
- Epoch 9: Loss=0.1119, mIoU=0.5334, F1=0.5725
- Epoch 10: Loss=0.1056, mIoU=0.6342, F1=0.7167

本轮结果: Round=3, Labeled=365, Selection=best_val (epoch=7), mIoU=0.6380, F1=0.7220, peak_mIoU=0.6380

Round=3, Labeled=365, mIoU=0.6380, F1=0.7220

## Round 4

Labeled Pool Size: 453

- Epoch 1: Loss=0.5031, mIoU=0.5383, F1=0.5876
- Epoch 2: Loss=0.2268, mIoU=0.4990, F1=0.5106
- Epoch 3: Loss=0.1724, mIoU=0.5028, F1=0.5177
- Epoch 4: Loss=0.1458, mIoU=0.5619, F1=0.6183
- Epoch 5: Loss=0.1328, mIoU=0.5455, F1=0.5924
- Epoch 6: Loss=0.1266, mIoU=0.6611, F1=0.7481
- Epoch 7: Loss=0.1219, mIoU=0.6358, F1=0.7184
- Epoch 8: Loss=0.1163, mIoU=0.6388, F1=0.7221
- Epoch 9: Loss=0.1127, mIoU=0.5849, F1=0.6520
- Epoch 10: Loss=0.1053, mIoU=0.6576, F1=0.7442

本轮结果: Round=4, Labeled=453, Selection=best_val (epoch=6), mIoU=0.6611, F1=0.7481, peak_mIoU=0.6611

Round=4, Labeled=453, mIoU=0.6611, F1=0.7481

## Round 5

Labeled Pool Size: 541

- Epoch 1: Loss=0.4412, mIoU=0.5981, F1=0.6709
- Epoch 2: Loss=0.1924, mIoU=0.5254, F1=0.5587
- Epoch 3: Loss=0.1533, mIoU=0.6147, F1=0.6927
- Epoch 4: Loss=0.1368, mIoU=0.5777, F1=0.6416
- Epoch 5: Loss=0.1317, mIoU=0.6371, F1=0.7199
- Epoch 6: Loss=0.1223, mIoU=0.6326, F1=0.7145
- Epoch 7: Loss=0.1181, mIoU=0.6594, F1=0.7458
- Epoch 8: Loss=0.1145, mIoU=0.6657, F1=0.7538
- Epoch 9: Loss=0.1121, mIoU=0.5823, F1=0.6483
- Epoch 10: Loss=0.1075, mIoU=0.6421, F1=0.7260

本轮结果: Round=5, Labeled=541, Selection=best_val (epoch=8), mIoU=0.6657, F1=0.7538, peak_mIoU=0.6657

Round=5, Labeled=541, mIoU=0.6657, F1=0.7538

## Round 6

Labeled Pool Size: 629

- Epoch 1: Loss=0.3011, mIoU=0.5043, F1=0.5205
- Epoch 2: Loss=0.1688, mIoU=0.6183, F1=0.6967
- Epoch 3: Loss=0.1424, mIoU=0.6308, F1=0.7123
- Epoch 4: Loss=0.1267, mIoU=0.7065, F1=0.7952
- Epoch 5: Loss=0.1201, mIoU=0.6389, F1=0.7223
- Epoch 6: Loss=0.1173, mIoU=0.6144, F1=0.6920
- Epoch 7: Loss=0.1099, mIoU=0.6556, F1=0.7429
- Epoch 8: Loss=0.1057, mIoU=0.6776, F1=0.7661
- Epoch 9: Loss=0.1020, mIoU=0.6708, F1=0.7591
- Epoch 10: Loss=0.0992, mIoU=0.6653, F1=0.7524

本轮结果: Round=6, Labeled=629, Selection=best_val (epoch=4), mIoU=0.7065, F1=0.7952, peak_mIoU=0.7065

Round=6, Labeled=629, mIoU=0.7065, F1=0.7952

## Round 7

Labeled Pool Size: 717

- Epoch 1: Loss=0.2653, mIoU=0.5693, F1=0.6298
- Epoch 2: Loss=0.1499, mIoU=0.5188, F1=0.5472
- Epoch 3: Loss=0.1300, mIoU=0.5619, F1=0.6184
- Epoch 4: Loss=0.1192, mIoU=0.5797, F1=0.6448
- Epoch 5: Loss=0.1137, mIoU=0.6507, F1=0.7380
- Epoch 6: Loss=0.1061, mIoU=0.6341, F1=0.7165
- Epoch 7: Loss=0.1027, mIoU=0.6221, F1=0.7022
- Epoch 8: Loss=0.1001, mIoU=0.6360, F1=0.7187
- Epoch 9: Loss=0.0959, mIoU=0.6423, F1=0.7286
- Epoch 10: Loss=0.0922, mIoU=0.6433, F1=0.7276

本轮结果: Round=7, Labeled=717, Selection=best_val (epoch=5), mIoU=0.6507, F1=0.7380, peak_mIoU=0.6507

Round=7, Labeled=717, mIoU=0.6507, F1=0.7380


--- [Checkpoint] 续跑开始时间: 2026-03-26T08:00:45.079535 ---

## Round 7

Labeled Pool Size: 717

## Round 8

Labeled Pool Size: 805

- Epoch 1: Loss=0.2640, mIoU=0.5345, F1=0.5745
- Epoch 1: Loss=0.2546, mIoU=0.5173, F1=0.5534
- Epoch 2: Loss=0.1506, mIoU=0.5060, F1=0.5238
- Epoch 2: Loss=0.1417, mIoU=0.5102, F1=0.5381
- Epoch 3: Loss=0.1327, mIoU=0.6601, F1=0.7466
- Epoch 3: Loss=0.1228, mIoU=0.5495, F1=0.5994
- Epoch 4: Loss=0.1211, mIoU=0.5664, F1=0.6250
- Epoch 5: Loss=0.1138, mIoU=0.6589, F1=0.7457
- Epoch 4: Loss=0.1137, mIoU=0.6253, F1=0.7061
- Epoch 6: Loss=0.1076, mIoU=0.6486, F1=0.7335
- Epoch 5: Loss=0.1064, mIoU=0.6064, F1=0.6817
- Epoch 7: Loss=0.1053, mIoU=0.6959, F1=0.7846
- Epoch 6: Loss=0.1028, mIoU=0.5641, F1=0.6220
- Epoch 8: Loss=0.1033, mIoU=0.5832, F1=0.6497
- Epoch 7: Loss=0.0992, mIoU=0.6756, F1=0.7639
- Epoch 9: Loss=0.1006, mIoU=0.6802, F1=0.7696
- Epoch 8: Loss=0.0934, mIoU=0.6790, F1=0.7676
- Epoch 10: Loss=0.0961, mIoU=0.6403, F1=0.7243

本轮结果: Round=7, Labeled=717, Selection=best_val (epoch=7), mIoU=0.6959, F1=0.7846, peak_mIoU=0.6959

Round=7, Labeled=717, mIoU=0.6959, F1=0.7846

## Round 8

Labeled Pool Size: 805

- Epoch 9: Loss=0.0928, mIoU=0.6763, F1=0.7648
- Epoch 1: Loss=0.2552, mIoU=0.5603, F1=0.6157
- Epoch 10: Loss=0.0879, mIoU=0.6718, F1=0.7601

本轮结果: Round=8, Labeled=805, Selection=best_val (epoch=8), mIoU=0.6790, F1=0.7676, peak_mIoU=0.6790

Round=8, Labeled=805, mIoU=0.6790, F1=0.7676

## Round 9

Labeled Pool Size: 893

- Epoch 2: Loss=0.1471, mIoU=0.5565, F1=0.6100
- Epoch 1: Loss=0.3009, mIoU=0.4990, F1=0.5106
- Epoch 3: Loss=0.1267, mIoU=0.6229, F1=0.7026
- Epoch 2: Loss=0.1460, mIoU=0.5775, F1=0.6417
- Epoch 4: Loss=0.1182, mIoU=0.6631, F1=0.7501
- Epoch 3: Loss=0.1244, mIoU=0.6524, F1=0.7378
- Epoch 5: Loss=0.1107, mIoU=0.6578, F1=0.7451
- Epoch 6: Loss=0.1048, mIoU=0.6686, F1=0.7561
- Epoch 4: Loss=0.1139, mIoU=0.6117, F1=0.6885
- Epoch 7: Loss=0.1007, mIoU=0.6302, F1=0.7129
- Epoch 5: Loss=0.1056, mIoU=0.6361, F1=0.7189
- Epoch 8: Loss=0.0974, mIoU=0.6930, F1=0.7818
- Epoch 6: Loss=0.1007, mIoU=0.6947, F1=0.7837
- Epoch 9: Loss=0.0945, mIoU=0.6686, F1=0.7561
- Epoch 7: Loss=0.0954, mIoU=0.6509, F1=0.7373
- Epoch 10: Loss=0.0917, mIoU=0.6932, F1=0.7822

本轮结果: Round=8, Labeled=805, Selection=best_val (epoch=10), mIoU=0.6932, F1=0.7822, peak_mIoU=0.6932

Round=8, Labeled=805, mIoU=0.6932, F1=0.7822

- Epoch 8: Loss=0.0917, mIoU=0.6312, F1=0.7139
## Round 9

Labeled Pool Size: 893

- Epoch 9: Loss=0.0876, mIoU=0.5428, F1=0.5883
- Epoch 1: Loss=0.2991, mIoU=0.5505, F1=0.6005
- Epoch 10: Loss=0.0863, mIoU=0.6755, F1=0.7637

本轮结果: Round=9, Labeled=893, Selection=best_val (epoch=6), mIoU=0.6947, F1=0.7837, peak_mIoU=0.6947

Round=9, Labeled=893, mIoU=0.6947, F1=0.7837

- Epoch 2: Loss=0.1469, mIoU=0.6803, F1=0.7688
## Round 10

Labeled Pool Size: 981

- Epoch 3: Loss=0.1233, mIoU=0.6594, F1=0.7457
- Epoch 1: Loss=0.2148, mIoU=0.5011, F1=0.5145
- Epoch 4: Loss=0.1142, mIoU=0.6658, F1=0.7531
- Epoch 2: Loss=0.1259, mIoU=0.5261, F1=0.5600
- Epoch 5: Loss=0.1069, mIoU=0.6977, F1=0.7870
- Epoch 3: Loss=0.1099, mIoU=0.6427, F1=0.7265
- Epoch 6: Loss=0.1015, mIoU=0.7016, F1=0.7904
- Epoch 4: Loss=0.1025, mIoU=0.6440, F1=0.7280
- Epoch 7: Loss=0.0962, mIoU=0.6982, F1=0.7870
- Epoch 5: Loss=0.0982, mIoU=0.6596, F1=0.7462
- Epoch 8: Loss=0.0944, mIoU=0.7045, F1=0.7938
- Epoch 6: Loss=0.0910, mIoU=0.6250, F1=0.7055
- Epoch 9: Loss=0.0893, mIoU=0.6769, F1=0.7668
- Epoch 7: Loss=0.0891, mIoU=0.6008, F1=0.6740
- Epoch 10: Loss=0.0873, mIoU=0.6926, F1=0.7822

本轮结果: Round=9, Labeled=893, Selection=best_val (epoch=8), mIoU=0.7045, F1=0.7938, peak_mIoU=0.7045

Round=9, Labeled=893, mIoU=0.7045, F1=0.7938

## Round 10

Labeled Pool Size: 981

- Epoch 8: Loss=0.0865, mIoU=0.6816, F1=0.7700
- Epoch 1: Loss=0.2137, mIoU=0.5506, F1=0.6006
- Epoch 9: Loss=0.0822, mIoU=0.7012, F1=0.7899
- Epoch 2: Loss=0.1265, mIoU=0.6195, F1=0.6981
- Epoch 10: Loss=0.0803, mIoU=0.6813, F1=0.7700

本轮结果: Round=10, Labeled=981, Selection=best_val (epoch=9), mIoU=0.7012, F1=0.7899, peak_mIoU=0.7012

Round=10, Labeled=981, mIoU=0.7012, F1=0.7899

## Round 11

Labeled Pool Size: 1069

- Epoch 3: Loss=0.1118, mIoU=0.6810, F1=0.7693
- Epoch 1: Loss=0.2669, mIoU=0.5262, F1=0.5603
- Epoch 4: Loss=0.1040, mIoU=0.6940, F1=0.7832
- Epoch 5: Loss=0.0985, mIoU=0.6568, F1=0.7449
- Epoch 2: Loss=0.1279, mIoU=0.5816, F1=0.6477
- Epoch 6: Loss=0.0919, mIoU=0.6845, F1=0.7733
- Epoch 3: Loss=0.1115, mIoU=0.6309, F1=0.7124
- Epoch 7: Loss=0.0899, mIoU=0.5944, F1=0.6655
- Epoch 4: Loss=0.0983, mIoU=0.6573, F1=0.7443
- Epoch 8: Loss=0.0859, mIoU=0.6663, F1=0.7558
- Epoch 5: Loss=0.0979, mIoU=0.6632, F1=0.7501
- Epoch 9: Loss=0.0831, mIoU=0.6094, F1=0.6852
- Epoch 6: Loss=0.0902, mIoU=0.7121, F1=0.8008
- Epoch 10: Loss=0.0812, mIoU=0.6944, F1=0.7838

本轮结果: Round=10, Labeled=981, Selection=best_val (epoch=10), mIoU=0.6944, F1=0.7838, peak_mIoU=0.6944

Round=10, Labeled=981, mIoU=0.6944, F1=0.7838

- Epoch 7: Loss=0.0864, mIoU=0.6720, F1=0.7596
## Round 11

Labeled Pool Size: 1069

- Epoch 8: Loss=0.0835, mIoU=0.7025, F1=0.7918
- Epoch 1: Loss=0.2641, mIoU=0.5164, F1=0.5428
- Epoch 9: Loss=0.0780, mIoU=0.6770, F1=0.7650
- Epoch 2: Loss=0.1272, mIoU=0.5577, F1=0.6118
- Epoch 10: Loss=0.0771, mIoU=0.6945, F1=0.7831

本轮结果: Round=11, Labeled=1069, Selection=best_val (epoch=6), mIoU=0.7121, F1=0.8008, peak_mIoU=0.7121

Round=11, Labeled=1069, mIoU=0.7121, F1=0.8008

- Epoch 3: Loss=0.1088, mIoU=0.5938, F1=0.6644
## Round 12

Labeled Pool Size: 1157

- Epoch 4: Loss=0.0978, mIoU=0.6770, F1=0.7653
- Epoch 1: Loss=0.2448, mIoU=0.5234, F1=0.5552
- Epoch 5: Loss=0.0947, mIoU=0.6472, F1=0.7322
- Epoch 2: Loss=0.1174, mIoU=0.5221, F1=0.5529
- Epoch 6: Loss=0.0883, mIoU=0.6148, F1=0.6924
- Epoch 3: Loss=0.0995, mIoU=0.5811, F1=0.6473
- Epoch 7: Loss=0.0827, mIoU=0.6995, F1=0.7888
- Epoch 4: Loss=0.0927, mIoU=0.6639, F1=0.7509
- Epoch 8: Loss=0.0810, mIoU=0.6730, F1=0.7612
- Epoch 5: Loss=0.0871, mIoU=0.6631, F1=0.7501
- Epoch 9: Loss=0.0775, mIoU=0.6234, F1=0.7040
- Epoch 6: Loss=0.0828, mIoU=0.6305, F1=0.7128
- Epoch 10: Loss=0.0762, mIoU=0.6596, F1=0.7467

本轮结果: Round=11, Labeled=1069, Selection=best_val (epoch=7), mIoU=0.6995, F1=0.7888, peak_mIoU=0.6995

Round=11, Labeled=1069, mIoU=0.6995, F1=0.7888

## Round 12

Labeled Pool Size: 1157

- Epoch 7: Loss=0.0790, mIoU=0.7103, F1=0.7995
- Epoch 1: Loss=0.2330, mIoU=0.5130, F1=0.5369
- Epoch 8: Loss=0.0740, mIoU=0.6442, F1=0.7295
- Epoch 2: Loss=0.1159, mIoU=0.5808, F1=0.6465
- Epoch 9: Loss=0.0724, mIoU=0.6225, F1=0.7025
- Epoch 3: Loss=0.0993, mIoU=0.6388, F1=0.7243
- Epoch 10: Loss=0.0692, mIoU=0.6651, F1=0.7528

本轮结果: Round=12, Labeled=1157, Selection=best_val (epoch=7), mIoU=0.7103, F1=0.7995, peak_mIoU=0.7103

Round=12, Labeled=1157, mIoU=0.7103, F1=0.7995

## Round 13

Labeled Pool Size: 1245

- Epoch 4: Loss=0.0919, mIoU=0.6071, F1=0.6824
- Epoch 1: Loss=0.2324, mIoU=0.5266, F1=0.5607
- Epoch 5: Loss=0.0869, mIoU=0.6780, F1=0.7659
- Epoch 2: Loss=0.1131, mIoU=0.6294, F1=0.7106
- Epoch 6: Loss=0.0819, mIoU=0.6799, F1=0.7684
- Epoch 3: Loss=0.0988, mIoU=0.6006, F1=0.6736
- Epoch 7: Loss=0.0772, mIoU=0.6819, F1=0.7706
- Epoch 4: Loss=0.0890, mIoU=0.7027, F1=0.7912
- Epoch 8: Loss=0.0751, mIoU=0.6707, F1=0.7592
- Epoch 9: Loss=0.0727, mIoU=0.6651, F1=0.7522
- Epoch 5: Loss=0.0846, mIoU=0.6664, F1=0.7540
- Epoch 10: Loss=0.0705, mIoU=0.7050, F1=0.7943

本轮结果: Round=12, Labeled=1157, Selection=best_val (epoch=10), mIoU=0.7050, F1=0.7943, peak_mIoU=0.7050

Round=12, Labeled=1157, mIoU=0.7050, F1=0.7943

- Epoch 6: Loss=0.0808, mIoU=0.6949, F1=0.7835
## Round 13

Labeled Pool Size: 1245

- Epoch 7: Loss=0.0770, mIoU=0.5879, F1=0.6564
- Epoch 1: Loss=0.2320, mIoU=0.5448, F1=0.5913
- Epoch 8: Loss=0.0729, mIoU=0.6958, F1=0.7850
- Epoch 2: Loss=0.1112, mIoU=0.5913, F1=0.6609
- Epoch 9: Loss=0.0723, mIoU=0.6425, F1=0.7265
- Epoch 3: Loss=0.0943, mIoU=0.6733, F1=0.7618
- Epoch 10: Loss=0.0690, mIoU=0.6667, F1=0.7550

本轮结果: Round=13, Labeled=1245, Selection=best_val (epoch=4), mIoU=0.7027, F1=0.7912, peak_mIoU=0.7027

Round=13, Labeled=1245, mIoU=0.7027, F1=0.7912

## Round 14

Labeled Pool Size: 1333

- Epoch 4: Loss=0.0874, mIoU=0.6396, F1=0.7232
- Epoch 5: Loss=0.0801, mIoU=0.6411, F1=0.7247
- Epoch 1: Loss=0.2130, mIoU=0.4992, F1=0.5110
- Epoch 6: Loss=0.0777, mIoU=0.6826, F1=0.7709
- Epoch 2: Loss=0.1032, mIoU=0.5126, F1=0.5360
- Epoch 7: Loss=0.0746, mIoU=0.6597, F1=0.7463
- Epoch 3: Loss=0.0893, mIoU=0.6680, F1=0.7556
- Epoch 8: Loss=0.0740, mIoU=0.6817, F1=0.7710
- Epoch 4: Loss=0.0806, mIoU=0.7004, F1=0.7892
- Epoch 9: Loss=0.0694, mIoU=0.6485, F1=0.7347
- Epoch 5: Loss=0.0783, mIoU=0.7113, F1=0.7999
- Epoch 10: Loss=0.0661, mIoU=0.6760, F1=0.7658

本轮结果: Round=13, Labeled=1245, Selection=best_val (epoch=6), mIoU=0.6826, F1=0.7709, peak_mIoU=0.6826

Round=13, Labeled=1245, mIoU=0.6826, F1=0.7709

- Epoch 6: Loss=0.0729, mIoU=0.6944, F1=0.7836
## Round 14

Labeled Pool Size: 1333

- Epoch 7: Loss=0.0690, mIoU=0.6997, F1=0.7890
- Epoch 1: Loss=0.2088, mIoU=0.5715, F1=0.6325
- Epoch 8: Loss=0.0669, mIoU=0.6919, F1=0.7811
- Epoch 2: Loss=0.1024, mIoU=0.5019, F1=0.5160
- Epoch 9: Loss=0.0638, mIoU=0.6686, F1=0.7563
- Epoch 3: Loss=0.0880, mIoU=0.5622, F1=0.6344
- Epoch 10: Loss=0.0616, mIoU=0.6323, F1=0.7141

本轮结果: Round=14, Labeled=1333, Selection=best_val (epoch=5), mIoU=0.7113, F1=0.7999, peak_mIoU=0.7113

Round=14, Labeled=1333, mIoU=0.7113, F1=0.7999

- Epoch 4: Loss=0.0816, mIoU=0.6046, F1=0.6799
## Round 15

Labeled Pool Size: 1421

- Epoch 5: Loss=0.0767, mIoU=0.7150, F1=0.8039
- Epoch 1: Loss=0.2258, mIoU=0.5853, F1=0.6528
- Epoch 6: Loss=0.0722, mIoU=0.6652, F1=0.7523
- Epoch 2: Loss=0.0987, mIoU=0.6236, F1=0.7041
- Epoch 7: Loss=0.0711, mIoU=0.6526, F1=0.7408
- Epoch 3: Loss=0.0880, mIoU=0.6070, F1=0.6827
- Epoch 8: Loss=0.0683, mIoU=0.6989, F1=0.7879
- Epoch 4: Loss=0.0797, mIoU=0.6999, F1=0.7890
- Epoch 9: Loss=0.0651, mIoU=0.6568, F1=0.7431
- Epoch 5: Loss=0.0758, mIoU=0.6973, F1=0.7864
- Epoch 10: Loss=0.0612, mIoU=0.6410, F1=0.7253

本轮结果: Round=14, Labeled=1333, Selection=best_val (epoch=5), mIoU=0.7150, F1=0.8039, peak_mIoU=0.7150

Round=14, Labeled=1333, mIoU=0.7150, F1=0.8039

## Round 15

Labeled Pool Size: 1421

- Epoch 6: Loss=0.0719, mIoU=0.6605, F1=0.7470
- Epoch 1: Loss=0.2261, mIoU=0.5998, F1=0.6729
- Epoch 7: Loss=0.0710, mIoU=0.6611, F1=0.7510
- Epoch 2: Loss=0.0978, mIoU=0.5815, F1=0.6474
- Epoch 8: Loss=0.0680, mIoU=0.7064, F1=0.7957
- Epoch 3: Loss=0.0843, mIoU=0.6184, F1=0.7009
- Epoch 9: Loss=0.0683, mIoU=0.7175, F1=0.8063
- Epoch 4: Loss=0.0767, mIoU=0.5823, F1=0.6606
- Epoch 10: Loss=0.0634, mIoU=0.7145, F1=0.8035

本轮结果: Round=15, Labeled=1421, Selection=best_val (epoch=9), mIoU=0.7175, F1=0.8063, peak_mIoU=0.7175

Round=15, Labeled=1421, mIoU=0.7175, F1=0.8063

## Round 16

Labeled Pool Size: 1509

- Test-only mode: Round 16 loads Round 15 best_val checkpoint (epoch=9, val_mIoU=0.7175, val_F1=0.8063)

本轮结果: Round=16, Labeled=1509, Selection=prev_round_best_val (source_round=15, epoch=9), mIoU=0.7175, F1=0.8063, peak_mIoU=0.7175

Round=16, Labeled=1509, mIoU=0.7175, F1=0.8063


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421, 1509]
ALC(基于每轮选模val mIoU): 0.5907
最后一轮选模 mIoU(val): 0.717522133653496
最后一轮选模 F1(val): 0.8062630530351675
最终报告 mIoU(test): 0.7122989237419229
最终报告 F1(test): 0.8018399482334886
最终输出 mIoU: 0.7123 (source=final_report)
最终输出 F1: 0.8018 (source=final_report)
最终 Test Split: test
最终 Report: {'loss': 0.04989525459939614, 'mIoU': 0.7122989237419229, 'f1_score': 0.8018399482334886}
- Epoch 5: Loss=0.0727, mIoU=0.6606, F1=0.7472
- Epoch 6: Loss=0.0675, mIoU=0.6216, F1=0.7012
- Epoch 7: Loss=0.0643, mIoU=0.6834, F1=0.7728
- Epoch 8: Loss=0.0629, mIoU=0.6679, F1=0.7553
- Epoch 9: Loss=0.0605, mIoU=0.6136, F1=0.6912
- Epoch 10: Loss=0.0582, mIoU=0.6860, F1=0.7760

本轮结果: Round=15, Labeled=1421, Selection=best_val (epoch=10), mIoU=0.6860, F1=0.7760, peak_mIoU=0.6860

Round=15, Labeled=1421, mIoU=0.6860, F1=0.7760

## Round 16

Labeled Pool Size: 1509

- Test-only mode: Round 16 loads Round 15 best_val checkpoint (epoch=10, val_mIoU=0.6860, val_F1=0.7760)

本轮结果: Round=16, Labeled=1509, Selection=prev_round_best_val (source_round=15, epoch=10), mIoU=0.6860, F1=0.7760, peak_mIoU=0.6860

Round=16, Labeled=1509, mIoU=0.6860, F1=0.7760


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421, 1509]
ALC(基于每轮选模val mIoU): 0.5894
最后一轮选模 mIoU(val): 0.685981631266767
最后一轮选模 F1(val): 0.7759766951859364
最终报告 mIoU(test): 0.6858070686628841
最终报告 F1(test): 0.7764692434224472
最终输出 mIoU: 0.6858 (source=final_report)
最终输出 F1: 0.7765 (source=final_report)
最终 Test Split: test
最终 Report: {'loss': 0.0576682513486594, 'mIoU': 0.6858070686628841, 'f1_score': 0.7764692434224472}
