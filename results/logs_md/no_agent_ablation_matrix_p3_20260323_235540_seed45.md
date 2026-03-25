# 实验日志

实验名称: no_agent
描述: 消融：w/o Agent（argmax）；保留三阶段λ policy，与full_model一致
开始时间: 2026-03-25T22:47:41.224108

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.4700, mIoU=0.4893, F1=0.5164
- Epoch 2: Loss=0.2531, mIoU=0.4964, F1=0.5057
- Epoch 3: Loss=0.1608, mIoU=0.5117, F1=0.5366
- Epoch 4: Loss=0.1140, mIoU=0.5225, F1=0.5546
- Epoch 5: Loss=0.0899, mIoU=0.5188, F1=0.5478
- Epoch 6: Loss=0.0759, mIoU=0.5078, F1=0.5271
- Epoch 7: Loss=0.0662, mIoU=0.5124, F1=0.5357
- Epoch 8: Loss=0.0582, mIoU=0.5066, F1=0.5251
- Epoch 9: Loss=0.0534, mIoU=0.5532, F1=0.6050
- Epoch 10: Loss=0.0547, mIoU=0.5259, F1=0.5599

本轮结果: Round=1, Labeled=189, Selection=best_val (epoch=9), mIoU=0.5532, F1=0.6050, peak_mIoU=0.5532

Round=1, Labeled=189, mIoU=0.5532, F1=0.6050

## Round 2

Labeled Pool Size: 277

- Epoch 1: Loss=0.4230, mIoU=0.5000, F1=0.5126
- Epoch 2: Loss=0.2206, mIoU=0.5407, F1=0.5848
- Epoch 3: Loss=0.1588, mIoU=0.5615, F1=0.6176
- Epoch 4: Loss=0.1389, mIoU=0.5646, F1=0.6225
- Epoch 5: Loss=0.1198, mIoU=0.5470, F1=0.5951
- Epoch 6: Loss=0.1132, mIoU=0.5557, F1=0.6086
- Epoch 7: Loss=0.1032, mIoU=0.5346, F1=0.5745
- Epoch 8: Loss=0.0992, mIoU=0.5614, F1=0.6178
- Epoch 9: Loss=0.0930, mIoU=0.5156, F1=0.5414
- Epoch 10: Loss=0.0893, mIoU=0.5784, F1=0.6429

本轮结果: Round=2, Labeled=277, Selection=best_val (epoch=10), mIoU=0.5784, F1=0.6429, peak_mIoU=0.5784

Round=2, Labeled=277, mIoU=0.5784, F1=0.6429

## Round 3

Labeled Pool Size: 365

- Epoch 1: Loss=0.4879, mIoU=0.5416, F1=0.5992
- Epoch 2: Loss=0.2324, mIoU=0.5263, F1=0.5612
- Epoch 3: Loss=0.1774, mIoU=0.5048, F1=0.5214
- Epoch 4: Loss=0.1571, mIoU=0.5103, F1=0.5317
- Epoch 5: Loss=0.1380, mIoU=0.5068, F1=0.5252
- Epoch 6: Loss=0.1283, mIoU=0.5496, F1=0.5989
- Epoch 7: Loss=0.1195, mIoU=0.6446, F1=0.7301
- Epoch 8: Loss=0.1136, mIoU=0.5972, F1=0.6699
- Epoch 9: Loss=0.1104, mIoU=0.5317, F1=0.5697
- Epoch 10: Loss=0.1063, mIoU=0.5566, F1=0.6100

本轮结果: Round=3, Labeled=365, Selection=best_val (epoch=7), mIoU=0.6446, F1=0.7301, peak_mIoU=0.6446

Round=3, Labeled=365, mIoU=0.6446, F1=0.7301

## Round 4

Labeled Pool Size: 453

- Epoch 1: Loss=0.4103, mIoU=0.5192, F1=0.5482
- Epoch 2: Loss=0.2053, mIoU=0.5269, F1=0.5615
- Epoch 3: Loss=0.1636, mIoU=0.5645, F1=0.6222
- Epoch 4: Loss=0.1463, mIoU=0.6540, F1=0.7397
- Epoch 5: Loss=0.1309, mIoU=0.6341, F1=0.7166
- Epoch 6: Loss=0.1214, mIoU=0.6193, F1=0.6980
- Epoch 7: Loss=0.1153, mIoU=0.6359, F1=0.7184
- Epoch 8: Loss=0.1138, mIoU=0.6857, F1=0.7743
- Epoch 9: Loss=0.1098, mIoU=0.6518, F1=0.7376
- Epoch 10: Loss=0.1036, mIoU=0.6626, F1=0.7494

本轮结果: Round=4, Labeled=453, Selection=best_val (epoch=8), mIoU=0.6857, F1=0.7743, peak_mIoU=0.6857

Round=4, Labeled=453, mIoU=0.6857, F1=0.7743

## Round 5

Labeled Pool Size: 541

- Epoch 1: Loss=0.4670, mIoU=0.5582, F1=0.6183
- Epoch 2: Loss=0.2031, mIoU=0.5860, F1=0.6538
- Epoch 3: Loss=0.1573, mIoU=0.5234, F1=0.5552
- Epoch 4: Loss=0.1431, mIoU=0.5235, F1=0.5553
- Epoch 5: Loss=0.1302, mIoU=0.5369, F1=0.5784
- Epoch 6: Loss=0.1233, mIoU=0.6036, F1=0.6780
- Epoch 7: Loss=0.1189, mIoU=0.6352, F1=0.7183
- Epoch 8: Loss=0.1134, mIoU=0.6464, F1=0.7313
- Epoch 9: Loss=0.1076, mIoU=0.6454, F1=0.7304
- Epoch 10: Loss=0.1052, mIoU=0.5954, F1=0.6667

本轮结果: Round=5, Labeled=541, Selection=best_val (epoch=8), mIoU=0.6464, F1=0.7313, peak_mIoU=0.6464

Round=5, Labeled=541, mIoU=0.6464, F1=0.7313

## Round 6

Labeled Pool Size: 629

- Epoch 1: Loss=0.3883, mIoU=0.5398, F1=0.5834
- Epoch 2: Loss=0.1796, mIoU=0.5345, F1=0.5746
- Epoch 3: Loss=0.1500, mIoU=0.5889, F1=0.6577
- Epoch 4: Loss=0.1340, mIoU=0.6083, F1=0.6838
- Epoch 5: Loss=0.1259, mIoU=0.5385, F1=0.5809
- Epoch 6: Loss=0.1205, mIoU=0.6460, F1=0.7306
- Epoch 7: Loss=0.1135, mIoU=0.7104, F1=0.7993
- Epoch 8: Loss=0.1118, mIoU=0.6417, F1=0.7257
- Epoch 9: Loss=0.1071, mIoU=0.6378, F1=0.7211
- Epoch 10: Loss=0.1041, mIoU=0.6537, F1=0.7403

本轮结果: Round=6, Labeled=629, Selection=best_val (epoch=7), mIoU=0.7104, F1=0.7993, peak_mIoU=0.7104

Round=6, Labeled=629, mIoU=0.7104, F1=0.7993

## Round 7

Labeled Pool Size: 717

- Epoch 1: Loss=0.2871, mIoU=0.5349, F1=0.5751
- Epoch 2: Loss=0.1577, mIoU=0.5371, F1=0.5787
- Epoch 3: Loss=0.1310, mIoU=0.6181, F1=0.6964
- Epoch 4: Loss=0.1252, mIoU=0.6262, F1=0.7070
- Epoch 5: Loss=0.1163, mIoU=0.6743, F1=0.7621
- Epoch 6: Loss=0.1095, mIoU=0.6727, F1=0.7603
- Epoch 7: Loss=0.1027, mIoU=0.6561, F1=0.7420
- Epoch 8: Loss=0.1028, mIoU=0.7125, F1=0.8011
- Epoch 9: Loss=0.0985, mIoU=0.6172, F1=0.6956
- Epoch 10: Loss=0.0939, mIoU=0.7110, F1=0.8000

本轮结果: Round=7, Labeled=717, Selection=best_val (epoch=8), mIoU=0.7125, F1=0.8011, peak_mIoU=0.7125

Round=7, Labeled=717, mIoU=0.7125, F1=0.8011

## Round 8

Labeled Pool Size: 805

- Epoch 1: Loss=0.2492, mIoU=0.5604, F1=0.6159
- Epoch 2: Loss=0.1403, mIoU=0.5988, F1=0.6711
- Epoch 3: Loss=0.1244, mIoU=0.5638, F1=0.6211
- Epoch 4: Loss=0.1133, mIoU=0.6661, F1=0.7543
- Epoch 5: Loss=0.1074, mIoU=0.6578, F1=0.7439
- Epoch 6: Loss=0.1043, mIoU=0.6685, F1=0.7560
- Epoch 7: Loss=0.0969, mIoU=0.6601, F1=0.7471
- Epoch 8: Loss=0.0936, mIoU=0.6527, F1=0.7384
- Epoch 9: Loss=0.0907, mIoU=0.6182, F1=0.6970
- Epoch 10: Loss=0.0896, mIoU=0.6431, F1=0.7273

本轮结果: Round=8, Labeled=805, Selection=best_val (epoch=6), mIoU=0.6685, F1=0.7560, peak_mIoU=0.6685

Round=8, Labeled=805, mIoU=0.6685, F1=0.7560

## Round 9

Labeled Pool Size: 893

- Epoch 1: Loss=0.2369, mIoU=0.5161, F1=0.5431
- Epoch 2: Loss=0.1339, mIoU=0.5046, F1=0.5212
- Epoch 3: Loss=0.1143, mIoU=0.5392, F1=0.5823
- Epoch 4: Loss=0.1063, mIoU=0.5744, F1=0.6370
- Epoch 5: Loss=0.0980, mIoU=0.5692, F1=0.6299
- Epoch 6: Loss=0.0939, mIoU=0.6273, F1=0.7083
- Epoch 7: Loss=0.0910, mIoU=0.6534, F1=0.7392
- Epoch 8: Loss=0.0860, mIoU=0.6769, F1=0.7656
- Epoch 9: Loss=0.0837, mIoU=0.5938, F1=0.6646
- Epoch 10: Loss=0.0809, mIoU=0.6112, F1=0.6880

本轮结果: Round=9, Labeled=893, Selection=best_val (epoch=8), mIoU=0.6769, F1=0.7656, peak_mIoU=0.6769

Round=9, Labeled=893, mIoU=0.6769, F1=0.7656

## Round 10

Labeled Pool Size: 981

- Epoch 1: Loss=0.2777, mIoU=0.5075, F1=0.5266
- Epoch 2: Loss=0.1295, mIoU=0.5303, F1=0.5673
- Epoch 3: Loss=0.1120, mIoU=0.6019, F1=0.6752
- Epoch 4: Loss=0.1042, mIoU=0.6807, F1=0.7692
- Epoch 5: Loss=0.0974, mIoU=0.5464, F1=0.5939
- Epoch 6: Loss=0.0924, mIoU=0.6552, F1=0.7418
- Epoch 7: Loss=0.0873, mIoU=0.6763, F1=0.7643
- Epoch 8: Loss=0.0897, mIoU=0.6304, F1=0.7118
- Epoch 9: Loss=0.0827, mIoU=0.6778, F1=0.7666
- Epoch 10: Loss=0.0799, mIoU=0.6965, F1=0.7858

本轮结果: Round=10, Labeled=981, Selection=best_val (epoch=10), mIoU=0.6965, F1=0.7858, peak_mIoU=0.6965

Round=10, Labeled=981, mIoU=0.6965, F1=0.7858

## Round 11

Labeled Pool Size: 1069

- Epoch 1: Loss=0.2001, mIoU=0.5762, F1=0.6396
- Epoch 2: Loss=0.1140, mIoU=0.5616, F1=0.6179
- Epoch 3: Loss=0.1009, mIoU=0.5655, F1=0.6236
- Epoch 4: Loss=0.0949, mIoU=0.7082, F1=0.7970
- Epoch 5: Loss=0.0861, mIoU=0.6819, F1=0.7704
- Epoch 6: Loss=0.0836, mIoU=0.6641, F1=0.7512
- Epoch 7: Loss=0.0818, mIoU=0.6861, F1=0.7745
- Epoch 8: Loss=0.0804, mIoU=0.6653, F1=0.7535
- Epoch 9: Loss=0.0757, mIoU=0.6445, F1=0.7289
- Epoch 10: Loss=0.0719, mIoU=0.6326, F1=0.7147

本轮结果: Round=11, Labeled=1069, Selection=best_val (epoch=4), mIoU=0.7082, F1=0.7970, peak_mIoU=0.7082

Round=11, Labeled=1069, mIoU=0.7082, F1=0.7970

## Round 12

Labeled Pool Size: 1157

- Epoch 1: Loss=0.2533, mIoU=0.5490, F1=0.5982
- Epoch 2: Loss=0.1155, mIoU=0.6105, F1=0.6869
- Epoch 3: Loss=0.1001, mIoU=0.5754, F1=0.6390
- Epoch 4: Loss=0.0909, mIoU=0.6823, F1=0.7710
- Epoch 5: Loss=0.0843, mIoU=0.6673, F1=0.7548
- Epoch 6: Loss=0.0809, mIoU=0.6952, F1=0.7842
- Epoch 7: Loss=0.0762, mIoU=0.6735, F1=0.7617
- Epoch 8: Loss=0.0743, mIoU=0.5983, F1=0.6709
- Epoch 9: Loss=0.0712, mIoU=0.6326, F1=0.7147
- Epoch 10: Loss=0.0713, mIoU=0.6864, F1=0.7751

本轮结果: Round=12, Labeled=1157, Selection=best_val (epoch=6), mIoU=0.6952, F1=0.7842, peak_mIoU=0.6952

Round=12, Labeled=1157, mIoU=0.6952, F1=0.7842

## Round 13

Labeled Pool Size: 1245

- Epoch 1: Loss=0.2303, mIoU=0.5617, F1=0.6181
- Epoch 2: Loss=0.1073, mIoU=0.6018, F1=0.6755
- Epoch 3: Loss=0.0927, mIoU=0.6107, F1=0.6925
- Epoch 4: Loss=0.0851, mIoU=0.6319, F1=0.7137
- Epoch 5: Loss=0.0785, mIoU=0.6657, F1=0.7533
- Epoch 6: Loss=0.0777, mIoU=0.6235, F1=0.7033
- Epoch 7: Loss=0.0709, mIoU=0.7019, F1=0.7907
- Epoch 8: Loss=0.0682, mIoU=0.7156, F1=0.8040
- Epoch 9: Loss=0.0659, mIoU=0.7195, F1=0.8078
- Epoch 10: Loss=0.0651, mIoU=0.7064, F1=0.7954

本轮结果: Round=13, Labeled=1245, Selection=best_val (epoch=9), mIoU=0.7195, F1=0.8078, peak_mIoU=0.7195

Round=13, Labeled=1245, mIoU=0.7195, F1=0.8078

## Round 14

Labeled Pool Size: 1333

- Epoch 1: Loss=0.2209, mIoU=0.5003, F1=0.5131
- Epoch 2: Loss=0.1012, mIoU=0.5604, F1=0.6158
- Epoch 3: Loss=0.0857, mIoU=0.5954, F1=0.6666
- Epoch 4: Loss=0.0788, mIoU=0.6578, F1=0.7441
- Epoch 5: Loss=0.0736, mIoU=0.6194, F1=0.6988
- Epoch 6: Loss=0.0701, mIoU=0.6280, F1=0.7094
- Epoch 7: Loss=0.0679, mIoU=0.7028, F1=0.7918
- Epoch 8: Loss=0.0673, mIoU=0.6825, F1=0.7713
- Epoch 9: Loss=0.0627, mIoU=0.5999, F1=0.6730
- Epoch 10: Loss=0.0603, mIoU=0.6888, F1=0.7777

本轮结果: Round=14, Labeled=1333, Selection=best_val (epoch=7), mIoU=0.7028, F1=0.7918, peak_mIoU=0.7028

Round=14, Labeled=1333, mIoU=0.7028, F1=0.7918

## Round 15

Labeled Pool Size: 1421

- Epoch 1: Loss=0.1971, mIoU=0.5439, F1=0.5898
- Epoch 2: Loss=0.0952, mIoU=0.5358, F1=0.5767
- Epoch 3: Loss=0.0823, mIoU=0.5263, F1=0.5604
- Epoch 4: Loss=0.0759, mIoU=0.6602, F1=0.7472
- Epoch 5: Loss=0.0716, mIoU=0.6999, F1=0.7889
- Epoch 6: Loss=0.0689, mIoU=0.6727, F1=0.7614
- Epoch 7: Loss=0.0642, mIoU=0.6696, F1=0.7584
- Epoch 8: Loss=0.0640, mIoU=0.6549, F1=0.7409
- Epoch 9: Loss=0.0609, mIoU=0.6315, F1=0.7134
- Epoch 10: Loss=0.0592, mIoU=0.6792, F1=0.7676

本轮结果: Round=15, Labeled=1421, Selection=best_val (epoch=5), mIoU=0.6999, F1=0.7889, peak_mIoU=0.6999

Round=15, Labeled=1421, mIoU=0.6999, F1=0.7889

## Round 16

Labeled Pool Size: 1509

- Test-only mode: Round 16 loads Round 15 best_val checkpoint (epoch=5, val_mIoU=0.6999, val_F1=0.7889)

本轮结果: Round=16, Labeled=1509, Selection=prev_round_best_val (source_round=15, epoch=5), mIoU=0.6999, F1=0.7889, peak_mIoU=0.6999

Round=16, Labeled=1509, mIoU=0.6999, F1=0.7889


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421, 1509]
ALC(基于每轮选模val mIoU): 0.5939
最后一轮选模 mIoU(val): 0.6999133437452996
最后一轮选模 F1(val): 0.7889312068102505
最终报告 mIoU(test): 0.6684985627973054
最终报告 F1(test): 0.7571166897577511
最终输出 mIoU: 0.6685 (source=final_report)
最终输出 F1: 0.7571 (source=final_report)
最终 Test Split: test
最终 Report: {'loss': 0.054986631446518004, 'mIoU': 0.6684985627973054, 'f1_score': 0.7571166897577511}
