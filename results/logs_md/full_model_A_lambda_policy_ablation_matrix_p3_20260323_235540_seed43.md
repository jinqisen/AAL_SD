# 实验日志

实验名称: full_model_A_lambda_policy
描述: 对照A / AAL-SD (Full)：基于 auto_opt_iter15_cand2_02（warmup+risk闭环 + 后期ramp + U-guardrail；train_holdout grad_probe）
开始时间: 2026-03-24T14:27:04.298883

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.3469, mIoU=0.5030, F1=0.5188
- Epoch 2: Loss=0.1956, mIoU=0.5180, F1=0.5555
- Epoch 3: Loss=0.1348, mIoU=0.5498, F1=0.6034
- Epoch 4: Loss=0.1053, mIoU=0.6056, F1=0.6811
- Epoch 5: Loss=0.0904, mIoU=0.5909, F1=0.6625
- Epoch 6: Loss=0.0767, mIoU=0.6064, F1=0.6829
- Epoch 7: Loss=0.0706, mIoU=0.5820, F1=0.6489
- Epoch 8: Loss=0.0663, mIoU=0.5672, F1=0.6267
- Epoch 9: Loss=0.0571, mIoU=0.6170, F1=0.6960
- Epoch 10: Loss=0.0548, mIoU=0.5409, F1=0.5855

本轮结果: Round=1, Labeled=189, Selection=best_val (epoch=9), mIoU=0.6170, F1=0.6960, peak_mIoU=0.6170

Round=1, Labeled=189, mIoU=0.6170, F1=0.6960

## Round 2

Labeled Pool Size: 277

- Epoch 1: Loss=0.4273, mIoU=0.5041, F1=0.5243
- Epoch 2: Loss=0.2136, mIoU=0.5061, F1=0.5240
- Epoch 3: Loss=0.1483, mIoU=0.5404, F1=0.5846
- Epoch 4: Loss=0.1181, mIoU=0.5512, F1=0.6017
- Epoch 5: Loss=0.1057, mIoU=0.5287, F1=0.5645
- Epoch 6: Loss=0.0953, mIoU=0.5675, F1=0.6269
- Epoch 7: Loss=0.0943, mIoU=0.5258, F1=0.5595
- Epoch 8: Loss=0.0886, mIoU=0.5075, F1=0.5265
- Epoch 9: Loss=0.0827, mIoU=0.5586, F1=0.6131
- Epoch 10: Loss=0.0787, mIoU=0.6288, F1=0.7098

本轮结果: Round=2, Labeled=277, Selection=best_val (epoch=10), mIoU=0.6288, F1=0.7098, peak_mIoU=0.6288

Round=2, Labeled=277, mIoU=0.6288, F1=0.7098

## Round 3

Labeled Pool Size: 365

- Epoch 1: Loss=0.4010, mIoU=0.5288, F1=0.5651
- Epoch 2: Loss=0.2066, mIoU=0.5700, F1=0.6303
- Epoch 3: Loss=0.1592, mIoU=0.5070, F1=0.5257
- Epoch 4: Loss=0.1382, mIoU=0.5101, F1=0.5312
- Epoch 5: Loss=0.1231, mIoU=0.5424, F1=0.5874
- Epoch 6: Loss=0.1200, mIoU=0.5053, F1=0.5224
- Epoch 7: Loss=0.1127, mIoU=0.5413, F1=0.5858
- Epoch 8: Loss=0.1052, mIoU=0.5762, F1=0.6399
- Epoch 9: Loss=0.1049, mIoU=0.5884, F1=0.6571
- Epoch 10: Loss=0.0928, mIoU=0.6345, F1=0.7172

本轮结果: Round=3, Labeled=365, Selection=best_val (epoch=10), mIoU=0.6345, F1=0.7172, peak_mIoU=0.6345

Round=3, Labeled=365, mIoU=0.6345, F1=0.7172

## Round 4

Labeled Pool Size: 453

- Epoch 1: Loss=0.3684, mIoU=0.5032, F1=0.5186
- Epoch 2: Loss=0.1899, mIoU=0.6045, F1=0.6788
- Epoch 3: Loss=0.1535, mIoU=0.5185, F1=0.5466
- Epoch 4: Loss=0.1377, mIoU=0.5282, F1=0.5635
- Epoch 5: Loss=0.1277, mIoU=0.5346, F1=0.5744
- Epoch 6: Loss=0.1187, mIoU=0.6169, F1=0.6953
- Epoch 7: Loss=0.1183, mIoU=0.6058, F1=0.6813
- Epoch 8: Loss=0.1084, mIoU=0.6592, F1=0.7456
- Epoch 9: Loss=0.1065, mIoU=0.6648, F1=0.7525
- Epoch 10: Loss=0.1050, mIoU=0.5687, F1=0.6287

本轮结果: Round=4, Labeled=453, Selection=best_val (epoch=9), mIoU=0.6648, F1=0.7525, peak_mIoU=0.6648

Round=4, Labeled=453, mIoU=0.6648, F1=0.7525

## Round 5

Labeled Pool Size: 541

- Epoch 1: Loss=0.4211, mIoU=0.6047, F1=0.6802
- Epoch 2: Loss=0.1956, mIoU=0.5228, F1=0.5543
- Epoch 3: Loss=0.1574, mIoU=0.5266, F1=0.5610
- Epoch 4: Loss=0.1375, mIoU=0.5628, F1=0.6198
- Epoch 5: Loss=0.1307, mIoU=0.5622, F1=0.6187
- Epoch 6: Loss=0.1225, mIoU=0.5726, F1=0.6344
- Epoch 7: Loss=0.1178, mIoU=0.6398, F1=0.7231
- Epoch 8: Loss=0.1120, mIoU=0.6638, F1=0.7508
- Epoch 9: Loss=0.1081, mIoU=0.6813, F1=0.7705
- Epoch 10: Loss=0.1043, mIoU=0.6696, F1=0.7573

本轮结果: Round=5, Labeled=541, Selection=best_val (epoch=9), mIoU=0.6813, F1=0.7705, peak_mIoU=0.6813

Round=5, Labeled=541, mIoU=0.6813, F1=0.7705

## Round 6

Labeled Pool Size: 629

- Epoch 1: Loss=0.3609, mIoU=0.5152, F1=0.5410
- Epoch 2: Loss=0.1731, mIoU=0.5609, F1=0.6169
- Epoch 3: Loss=0.1437, mIoU=0.5038, F1=0.5197
- Epoch 4: Loss=0.1290, mIoU=0.5833, F1=0.6498
- Epoch 5: Loss=0.1246, mIoU=0.6226, F1=0.7028
- Epoch 6: Loss=0.1156, mIoU=0.5556, F1=0.6086
- Epoch 7: Loss=0.1129, mIoU=0.6270, F1=0.7077
- Epoch 8: Loss=0.1065, mIoU=0.6615, F1=0.7484
- Epoch 9: Loss=0.1011, mIoU=0.6527, F1=0.7381
- Epoch 10: Loss=0.0983, mIoU=0.6655, F1=0.7529

本轮结果: Round=6, Labeled=629, Selection=best_val (epoch=10), mIoU=0.6655, F1=0.7529, peak_mIoU=0.6655

Round=6, Labeled=629, mIoU=0.6655, F1=0.7529

## Round 7

Labeled Pool Size: 717

- Epoch 1: Loss=0.3995, mIoU=0.5077, F1=0.5269
- Epoch 2: Loss=0.1700, mIoU=0.5375, F1=0.5794
- Epoch 3: Loss=0.1384, mIoU=0.5701, F1=0.6305
- Epoch 4: Loss=0.1256, mIoU=0.5341, F1=0.5736
- Epoch 5: Loss=0.1168, mIoU=0.6454, F1=0.7300
- Epoch 6: Loss=0.1092, mIoU=0.6023, F1=0.6760
- Epoch 7: Loss=0.1041, mIoU=0.5918, F1=0.6616
- Epoch 8: Loss=0.1026, mIoU=0.6465, F1=0.7313
- Epoch 9: Loss=0.0989, mIoU=0.6967, F1=0.7856
- Epoch 10: Loss=0.0945, mIoU=0.6095, F1=0.6855

本轮结果: Round=7, Labeled=717, Selection=best_val (epoch=9), mIoU=0.6967, F1=0.7856, peak_mIoU=0.6967

Round=7, Labeled=717, mIoU=0.6967, F1=0.7856

## Round 8

Labeled Pool Size: 805

- Epoch 1: Loss=0.3625, mIoU=0.5509, F1=0.6012
- Epoch 2: Loss=0.1560, mIoU=0.5721, F1=0.6336
- Epoch 3: Loss=0.1315, mIoU=0.6656, F1=0.7529
- Epoch 4: Loss=0.1207, mIoU=0.6378, F1=0.7208
- Epoch 5: Loss=0.1136, mIoU=0.6638, F1=0.7508
- Epoch 6: Loss=0.1089, mIoU=0.6545, F1=0.7407
- Epoch 7: Loss=0.1057, mIoU=0.6798, F1=0.7681
- Epoch 8: Loss=0.0995, mIoU=0.6893, F1=0.7782
- Epoch 9: Loss=0.1000, mIoU=0.6741, F1=0.7621
- Epoch 10: Loss=0.0976, mIoU=0.6790, F1=0.7680

本轮结果: Round=8, Labeled=805, Selection=best_val (epoch=8), mIoU=0.6893, F1=0.7782, peak_mIoU=0.6893

Round=8, Labeled=805, mIoU=0.6893, F1=0.7782

## Round 9

Labeled Pool Size: 893

- Epoch 1: Loss=0.2574, mIoU=0.5624, F1=0.6192
- Epoch 2: Loss=0.1409, mIoU=0.5764, F1=0.6400
- Epoch 3: Loss=0.1214, mIoU=0.6889, F1=0.7776
- Epoch 4: Loss=0.1139, mIoU=0.6822, F1=0.7712
- Epoch 5: Loss=0.1064, mIoU=0.6836, F1=0.7721
- Epoch 6: Loss=0.1009, mIoU=0.6858, F1=0.7758
- Epoch 7: Loss=0.0969, mIoU=0.6456, F1=0.7311
- Epoch 8: Loss=0.0934, mIoU=0.6093, F1=0.6855
- Epoch 9: Loss=0.0892, mIoU=0.6508, F1=0.7370
- Epoch 10: Loss=0.0892, mIoU=0.6731, F1=0.7613

本轮结果: Round=9, Labeled=893, Selection=best_val (epoch=3), mIoU=0.6889, F1=0.7776, peak_mIoU=0.6889

Round=9, Labeled=893, mIoU=0.6889, F1=0.7776

## Round 10

Labeled Pool Size: 981

- Epoch 1: Loss=0.2316, mIoU=0.5246, F1=0.5574
- Epoch 2: Loss=0.1306, mIoU=0.5588, F1=0.6134
- Epoch 3: Loss=0.1124, mIoU=0.6595, F1=0.7471
- Epoch 4: Loss=0.1059, mIoU=0.5915, F1=0.6612
- Epoch 5: Loss=0.0994, mIoU=0.6891, F1=0.7781
- Epoch 6: Loss=0.0942, mIoU=0.6924, F1=0.7810
- Epoch 7: Loss=0.0896, mIoU=0.6804, F1=0.7690
- Epoch 8: Loss=0.0865, mIoU=0.6469, F1=0.7319
- Epoch 9: Loss=0.0840, mIoU=0.6495, F1=0.7361
- Epoch 10: Loss=0.0817, mIoU=0.6429, F1=0.7286

本轮结果: Round=10, Labeled=981, Selection=best_val (epoch=6), mIoU=0.6924, F1=0.7810, peak_mIoU=0.6924

Round=10, Labeled=981, mIoU=0.6924, F1=0.7810

## Round 11

Labeled Pool Size: 1069

- Epoch 1: Loss=0.2228, mIoU=0.5281, F1=0.5639
- Epoch 2: Loss=0.1260, mIoU=0.5591, F1=0.6139
- Epoch 3: Loss=0.1116, mIoU=0.6575, F1=0.7436
- Epoch 4: Loss=0.1016, mIoU=0.6414, F1=0.7251
- Epoch 5: Loss=0.0993, mIoU=0.6485, F1=0.7334
- Epoch 6: Loss=0.0927, mIoU=0.7035, F1=0.7924
- Epoch 7: Loss=0.0878, mIoU=0.6865, F1=0.7758
- Epoch 8: Loss=0.0856, mIoU=0.7123, F1=0.8011
- Epoch 9: Loss=0.0834, mIoU=0.6998, F1=0.7889
- Epoch 10: Loss=0.0810, mIoU=0.5979, F1=0.6701

本轮结果: Round=11, Labeled=1069, Selection=best_val (epoch=8), mIoU=0.7123, F1=0.8011, peak_mIoU=0.7123

Round=11, Labeled=1069, mIoU=0.7123, F1=0.8011

## Round 12

Labeled Pool Size: 1157

- Epoch 1: Loss=0.2643, mIoU=0.5623, F1=0.6188
- Epoch 2: Loss=0.1282, mIoU=0.5139, F1=0.5383
- Epoch 3: Loss=0.1101, mIoU=0.6838, F1=0.7726
- Epoch 4: Loss=0.1015, mIoU=0.5890, F1=0.6579
- Epoch 5: Loss=0.0946, mIoU=0.5824, F1=0.6484
- Epoch 6: Loss=0.0894, mIoU=0.6601, F1=0.7466
- Epoch 7: Loss=0.0868, mIoU=0.6946, F1=0.7836
- Epoch 8: Loss=0.0834, mIoU=0.6578, F1=0.7449
- Epoch 9: Loss=0.0792, mIoU=0.6791, F1=0.7675
- Epoch 10: Loss=0.0770, mIoU=0.6801, F1=0.7692

本轮结果: Round=12, Labeled=1157, Selection=best_val (epoch=7), mIoU=0.6946, F1=0.7836, peak_mIoU=0.6946

Round=12, Labeled=1157, mIoU=0.6946, F1=0.7836

## Round 13

Labeled Pool Size: 1245

- Epoch 1: Loss=0.1924, mIoU=0.5401, F1=0.5836
- Epoch 2: Loss=0.1147, mIoU=0.6875, F1=0.7761
- Epoch 3: Loss=0.1019, mIoU=0.6731, F1=0.7609
- Epoch 4: Loss=0.0933, mIoU=0.6800, F1=0.7686
- Epoch 5: Loss=0.0888, mIoU=0.6264, F1=0.7077
- Epoch 6: Loss=0.0860, mIoU=0.6561, F1=0.7423
- Epoch 7: Loss=0.0804, mIoU=0.6705, F1=0.7580
- Epoch 8: Loss=0.0770, mIoU=0.6840, F1=0.7736
- Epoch 9: Loss=0.0751, mIoU=0.6435, F1=0.7278
- Epoch 10: Loss=0.0732, mIoU=0.6994, F1=0.7885

本轮结果: Round=13, Labeled=1245, Selection=best_val (epoch=10), mIoU=0.6994, F1=0.7885, peak_mIoU=0.6994

Round=13, Labeled=1245, mIoU=0.6994, F1=0.7885

## Round 14

Labeled Pool Size: 1333

- Epoch 1: Loss=0.2380, mIoU=0.5340, F1=0.5736
- Epoch 2: Loss=0.1162, mIoU=0.5425, F1=0.5876
- Epoch 3: Loss=0.1009, mIoU=0.6453, F1=0.7298
- Epoch 4: Loss=0.0927, mIoU=0.5498, F1=0.5994
- Epoch 5: Loss=0.0861, mIoU=0.5677, F1=0.6273
- Epoch 6: Loss=0.0835, mIoU=0.5773, F1=0.6417
- Epoch 7: Loss=0.0798, mIoU=0.6312, F1=0.7134
- Epoch 8: Loss=0.0763, mIoU=0.6240, F1=0.7042
- Epoch 9: Loss=0.0737, mIoU=0.6190, F1=0.6980
- Epoch 10: Loss=0.0722, mIoU=0.6206, F1=0.7027

本轮结果: Round=14, Labeled=1333, Selection=best_val (epoch=3), mIoU=0.6453, F1=0.7298, peak_mIoU=0.6453

Round=14, Labeled=1333, mIoU=0.6453, F1=0.7298

## Round 15

Labeled Pool Size: 1421

- Epoch 1: Loss=0.2227, mIoU=0.5051, F1=0.5220
- Epoch 2: Loss=0.1098, mIoU=0.6998, F1=0.7892
- Epoch 3: Loss=0.0949, mIoU=0.6703, F1=0.7576
- Epoch 4: Loss=0.0916, mIoU=0.5997, F1=0.6726
- Epoch 5: Loss=0.0834, mIoU=0.7143, F1=0.8027
- Epoch 6: Loss=0.0794, mIoU=0.6382, F1=0.7215
- Epoch 7: Loss=0.0754, mIoU=0.6612, F1=0.7482
- Epoch 8: Loss=0.0731, mIoU=0.5786, F1=0.6430
- Epoch 9: Loss=0.0703, mIoU=0.6342, F1=0.7168
- Epoch 10: Loss=0.0671, mIoU=0.6711, F1=0.7589

本轮结果: Round=15, Labeled=1421, Selection=best_val (epoch=5), mIoU=0.7143, F1=0.8027, peak_mIoU=0.7143

Round=15, Labeled=1421, mIoU=0.7143, F1=0.8027

## Round 16

Labeled Pool Size: 1509

- Test-only mode: Round 16 loads Round 15 best_val checkpoint (epoch=5, val_mIoU=0.7143, val_F1=0.8027)

本轮结果: Round=16, Labeled=1509, Selection=prev_round_best_val (source_round=15, epoch=5), mIoU=0.7143, F1=0.8027, peak_mIoU=0.7143

Round=16, Labeled=1509, mIoU=0.7143, F1=0.8027


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421, 1509]
ALC(基于每轮选模val mIoU): 0.5941
最后一轮选模 mIoU(val): 0.7143229184368194
最后一轮选模 F1(val): 0.8027359210762105
最终报告 mIoU(test): 0.678809175475257
最终报告 F1(test): 0.7677118480324707
最终输出 mIoU: 0.6788 (source=final_report)
最终输出 F1: 0.7677 (source=final_report)
最终 Test Split: test
最终 Report: {'loss': 0.046809047844726594, 'mIoU': 0.678809175475257, 'f1_score': 0.7677118480324707}
