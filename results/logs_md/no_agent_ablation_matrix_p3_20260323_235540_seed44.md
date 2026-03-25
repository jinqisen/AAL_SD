# 实验日志

实验名称: no_agent
描述: 消融：w/o Agent（argmax）；保留三阶段λ policy，与full_model一致
开始时间: 2026-03-25T04:28:33.868795

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.4667, mIoU=0.4974, F1=0.5242
- Epoch 2: Loss=0.2553, mIoU=0.5287, F1=0.5700
- Epoch 3: Loss=0.1635, mIoU=0.5466, F1=0.6082
- Epoch 4: Loss=0.1189, mIoU=0.5757, F1=0.6428
- Epoch 5: Loss=0.1005, mIoU=0.5526, F1=0.6040
- Epoch 6: Loss=0.0810, mIoU=0.5530, F1=0.6055
- Epoch 7: Loss=0.0707, mIoU=0.5668, F1=0.6258
- Epoch 8: Loss=0.0641, mIoU=0.5174, F1=0.5446
- Epoch 9: Loss=0.0585, mIoU=0.5256, F1=0.5594
- Epoch 10: Loss=0.0554, mIoU=0.5077, F1=0.5270

本轮结果: Round=1, Labeled=189, Selection=best_val (epoch=4), mIoU=0.5757, F1=0.6428, peak_mIoU=0.5757

Round=1, Labeled=189, mIoU=0.5757, F1=0.6428

## Round 2

Labeled Pool Size: 277

- Epoch 1: Loss=0.4285, mIoU=0.5024, F1=0.5194
- Epoch 2: Loss=0.2171, mIoU=0.5186, F1=0.5518
- Epoch 3: Loss=0.1495, mIoU=0.5042, F1=0.5244
- Epoch 4: Loss=0.1182, mIoU=0.5024, F1=0.5200
- Epoch 5: Loss=0.1029, mIoU=0.5006, F1=0.5158
- Epoch 6: Loss=0.0901, mIoU=0.5023, F1=0.5171
- Epoch 7: Loss=0.0820, mIoU=0.5550, F1=0.6113
- Epoch 8: Loss=0.0768, mIoU=0.5201, F1=0.5499
- Epoch 9: Loss=0.0722, mIoU=0.5404, F1=0.5846
- Epoch 10: Loss=0.0671, mIoU=0.5528, F1=0.6042

本轮结果: Round=2, Labeled=277, Selection=best_val (epoch=7), mIoU=0.5550, F1=0.6113, peak_mIoU=0.5550

Round=2, Labeled=277, mIoU=0.5550, F1=0.6113

## Round 3

Labeled Pool Size: 365

- Epoch 1: Loss=0.3886, mIoU=0.4946, F1=0.5026
- Epoch 2: Loss=0.1973, mIoU=0.5304, F1=0.5673
- Epoch 3: Loss=0.1498, mIoU=0.5259, F1=0.5595
- Epoch 4: Loss=0.1294, mIoU=0.5048, F1=0.5215
- Epoch 5: Loss=0.1132, mIoU=0.6155, F1=0.6933
- Epoch 6: Loss=0.1080, mIoU=0.6117, F1=0.6884
- Epoch 7: Loss=0.1083, mIoU=0.5886, F1=0.6576
- Epoch 8: Loss=0.1001, mIoU=0.5957, F1=0.6672
- Epoch 9: Loss=0.0946, mIoU=0.6139, F1=0.6911
- Epoch 10: Loss=0.0887, mIoU=0.5826, F1=0.6489

本轮结果: Round=3, Labeled=365, Selection=best_val (epoch=5), mIoU=0.6155, F1=0.6933, peak_mIoU=0.6155

Round=3, Labeled=365, mIoU=0.6155, F1=0.6933

## Round 4

Labeled Pool Size: 453

- Epoch 1: Loss=0.4473, mIoU=0.5135, F1=0.5567
- Epoch 2: Loss=0.1915, mIoU=0.5033, F1=0.5192
- Epoch 3: Loss=0.1473, mIoU=0.5279, F1=0.5644
- Epoch 4: Loss=0.1338, mIoU=0.5684, F1=0.6283
- Epoch 5: Loss=0.1229, mIoU=0.5128, F1=0.5363
- Epoch 6: Loss=0.1146, mIoU=0.5672, F1=0.6263
- Epoch 7: Loss=0.1129, mIoU=0.5791, F1=0.6437
- Epoch 8: Loss=0.1045, mIoU=0.5508, F1=0.6010
- Epoch 9: Loss=0.1017, mIoU=0.5938, F1=0.6647
- Epoch 10: Loss=0.0996, mIoU=0.6551, F1=0.7412

本轮结果: Round=4, Labeled=453, Selection=best_val (epoch=10), mIoU=0.6551, F1=0.7412, peak_mIoU=0.6551

Round=4, Labeled=453, mIoU=0.6551, F1=0.7412

## Round 5

Labeled Pool Size: 541

- Epoch 1: Loss=0.3825, mIoU=0.5007, F1=0.5137
- Epoch 2: Loss=0.1818, mIoU=0.6415, F1=0.7252
- Epoch 3: Loss=0.1465, mIoU=0.5262, F1=0.5602
- Epoch 4: Loss=0.1340, mIoU=0.6560, F1=0.7418
- Epoch 5: Loss=0.1235, mIoU=0.5773, F1=0.6411
- Epoch 6: Loss=0.1191, mIoU=0.6745, F1=0.7627
- Epoch 7: Loss=0.1176, mIoU=0.5921, F1=0.6623
- Epoch 8: Loss=0.1094, mIoU=0.6733, F1=0.7620
- Epoch 9: Loss=0.1015, mIoU=0.6658, F1=0.7530
- Epoch 10: Loss=0.1003, mIoU=0.6578, F1=0.7447

本轮结果: Round=5, Labeled=541, Selection=best_val (epoch=6), mIoU=0.6745, F1=0.7627, peak_mIoU=0.6745

Round=5, Labeled=541, mIoU=0.6745, F1=0.7627

## Round 6

Labeled Pool Size: 629

- Epoch 1: Loss=0.4254, mIoU=0.5190, F1=0.5475
- Epoch 2: Loss=0.1797, mIoU=0.5903, F1=0.6595
- Epoch 3: Loss=0.1468, mIoU=0.6229, F1=0.7025
- Epoch 4: Loss=0.1346, mIoU=0.6920, F1=0.7810
- Epoch 5: Loss=0.1216, mIoU=0.6917, F1=0.7805
- Epoch 6: Loss=0.1161, mIoU=0.6831, F1=0.7716
- Epoch 7: Loss=0.1133, mIoU=0.6732, F1=0.7615
- Epoch 8: Loss=0.1089, mIoU=0.6979, F1=0.7867
- Epoch 9: Loss=0.1061, mIoU=0.5488, F1=0.5979
- Epoch 10: Loss=0.1020, mIoU=0.6882, F1=0.7780

本轮结果: Round=6, Labeled=629, Selection=best_val (epoch=8), mIoU=0.6979, F1=0.7867, peak_mIoU=0.6979

Round=6, Labeled=629, mIoU=0.6979, F1=0.7867

## Round 7

Labeled Pool Size: 717

- Epoch 1: Loss=0.3671, mIoU=0.5219, F1=0.5528
- Epoch 2: Loss=0.1627, mIoU=0.6533, F1=0.7390
- Epoch 3: Loss=0.1400, mIoU=0.6747, F1=0.7629
- Epoch 4: Loss=0.1267, mIoU=0.6850, F1=0.7737
- Epoch 5: Loss=0.1189, mIoU=0.6826, F1=0.7709
- Epoch 6: Loss=0.1146, mIoU=0.6350, F1=0.7176
- Epoch 7: Loss=0.1120, mIoU=0.6670, F1=0.7545
- Epoch 8: Loss=0.1044, mIoU=0.6683, F1=0.7563
- Epoch 9: Loss=0.1001, mIoU=0.6803, F1=0.7690
- Epoch 10: Loss=0.0960, mIoU=0.6444, F1=0.7287

本轮结果: Round=7, Labeled=717, Selection=best_val (epoch=4), mIoU=0.6850, F1=0.7737, peak_mIoU=0.6850

Round=7, Labeled=717, mIoU=0.6850, F1=0.7737

## Round 8

Labeled Pool Size: 805

- Epoch 1: Loss=0.2650, mIoU=0.5596, F1=0.6152
- Epoch 2: Loss=0.1423, mIoU=0.6502, F1=0.7354
- Epoch 3: Loss=0.1253, mIoU=0.6647, F1=0.7517
- Epoch 4: Loss=0.1106, mIoU=0.6697, F1=0.7574
- Epoch 5: Loss=0.1068, mIoU=0.6209, F1=0.7025
- Epoch 6: Loss=0.1009, mIoU=0.6592, F1=0.7470
- Epoch 7: Loss=0.0969, mIoU=0.6818, F1=0.7700
- Epoch 8: Loss=0.0942, mIoU=0.6625, F1=0.7507
- Epoch 9: Loss=0.0896, mIoU=0.6955, F1=0.7845
- Epoch 10: Loss=0.0867, mIoU=0.6264, F1=0.7080

本轮结果: Round=8, Labeled=805, Selection=best_val (epoch=9), mIoU=0.6955, F1=0.7845, peak_mIoU=0.6955

Round=8, Labeled=805, mIoU=0.6955, F1=0.7845

## Round 9

Labeled Pool Size: 893

- Epoch 1: Loss=0.2352, mIoU=0.5241, F1=0.5586
- Epoch 2: Loss=0.1310, mIoU=0.6034, F1=0.6775
- Epoch 3: Loss=0.1165, mIoU=0.5864, F1=0.6548
- Epoch 4: Loss=0.1104, mIoU=0.6697, F1=0.7576
- Epoch 5: Loss=0.0995, mIoU=0.6463, F1=0.7310
- Epoch 6: Loss=0.0959, mIoU=0.6386, F1=0.7217
- Epoch 7: Loss=0.0933, mIoU=0.6892, F1=0.7781
- Epoch 8: Loss=0.0887, mIoU=0.6557, F1=0.7421
- Epoch 9: Loss=0.0867, mIoU=0.6395, F1=0.7243
- Epoch 10: Loss=0.0832, mIoU=0.6825, F1=0.7722

本轮结果: Round=9, Labeled=893, Selection=best_val (epoch=7), mIoU=0.6892, F1=0.7781, peak_mIoU=0.6892

Round=9, Labeled=893, mIoU=0.6892, F1=0.7781

## Round 10

Labeled Pool Size: 981

- Epoch 1: Loss=0.2278, mIoU=0.5421, F1=0.5869
- Epoch 2: Loss=0.1257, mIoU=0.5746, F1=0.6372
- Epoch 3: Loss=0.1125, mIoU=0.5879, F1=0.6663
- Epoch 4: Loss=0.1021, mIoU=0.6860, F1=0.7753
- Epoch 5: Loss=0.0930, mIoU=0.6234, F1=0.7034
- Epoch 6: Loss=0.0915, mIoU=0.6589, F1=0.7456
- Epoch 7: Loss=0.0862, mIoU=0.6189, F1=0.6977
- Epoch 8: Loss=0.0839, mIoU=0.6580, F1=0.7444
- Epoch 9: Loss=0.0817, mIoU=0.6242, F1=0.7044
- Epoch 10: Loss=0.0795, mIoU=0.6587, F1=0.7457

本轮结果: Round=10, Labeled=981, Selection=best_val (epoch=4), mIoU=0.6860, F1=0.7753, peak_mIoU=0.6860

Round=10, Labeled=981, mIoU=0.6860, F1=0.7753

## Round 11

Labeled Pool Size: 1069

- Epoch 1: Loss=0.2636, mIoU=0.5020, F1=0.5162
- Epoch 2: Loss=0.1222, mIoU=0.6219, F1=0.7013
- Epoch 3: Loss=0.1040, mIoU=0.6763, F1=0.7642
- Epoch 4: Loss=0.0960, mIoU=0.6157, F1=0.6952
- Epoch 5: Loss=0.0886, mIoU=0.6360, F1=0.7200
- Epoch 6: Loss=0.0854, mIoU=0.6686, F1=0.7573
- Epoch 7: Loss=0.0814, mIoU=0.6884, F1=0.7775
- Epoch 8: Loss=0.0778, mIoU=0.6419, F1=0.7259
- Epoch 9: Loss=0.0751, mIoU=0.6727, F1=0.7612
- Epoch 10: Loss=0.0742, mIoU=0.6440, F1=0.7287

本轮结果: Round=11, Labeled=1069, Selection=best_val (epoch=7), mIoU=0.6884, F1=0.7775, peak_mIoU=0.6884

Round=11, Labeled=1069, mIoU=0.6884, F1=0.7775

## Round 12

Labeled Pool Size: 1157

- Epoch 1: Loss=0.1910, mIoU=0.6350, F1=0.7180
- Epoch 2: Loss=0.1067, mIoU=0.5324, F1=0.5707
- Epoch 3: Loss=0.0924, mIoU=0.5602, F1=0.6156
- Epoch 4: Loss=0.0870, mIoU=0.6167, F1=0.6953
- Epoch 5: Loss=0.0823, mIoU=0.6716, F1=0.7611
- Epoch 6: Loss=0.0775, mIoU=0.6294, F1=0.7106
- Epoch 7: Loss=0.0745, mIoU=0.6354, F1=0.7189
- Epoch 8: Loss=0.0696, mIoU=0.7044, F1=0.7935
- Epoch 9: Loss=0.0692, mIoU=0.6401, F1=0.7236
- Epoch 10: Loss=0.0685, mIoU=0.6767, F1=0.7667

本轮结果: Round=12, Labeled=1157, Selection=best_val (epoch=8), mIoU=0.7044, F1=0.7935, peak_mIoU=0.7044

Round=12, Labeled=1157, mIoU=0.7044, F1=0.7935

## Round 13

Labeled Pool Size: 1245

- Epoch 1: Loss=0.2380, mIoU=0.5474, F1=0.5959
- Epoch 2: Loss=0.1074, mIoU=0.5325, F1=0.5710
- Epoch 3: Loss=0.0928, mIoU=0.5905, F1=0.6604
- Epoch 4: Loss=0.0868, mIoU=0.6519, F1=0.7381
- Epoch 5: Loss=0.0818, mIoU=0.6820, F1=0.7705
- Epoch 6: Loss=0.0749, mIoU=0.6898, F1=0.7797
- Epoch 7: Loss=0.0720, mIoU=0.6860, F1=0.7743
- Epoch 8: Loss=0.0698, mIoU=0.7089, F1=0.7979
- Epoch 9: Loss=0.0666, mIoU=0.6871, F1=0.7763
- Epoch 10: Loss=0.0640, mIoU=0.6410, F1=0.7284

本轮结果: Round=13, Labeled=1245, Selection=best_val (epoch=8), mIoU=0.7089, F1=0.7979, peak_mIoU=0.7089

Round=13, Labeled=1245, mIoU=0.7089, F1=0.7979

## Round 14

Labeled Pool Size: 1333

- Epoch 1: Loss=0.2186, mIoU=0.5174, F1=0.5447
- Epoch 2: Loss=0.1019, mIoU=0.6133, F1=0.6906
- Epoch 3: Loss=0.0865, mIoU=0.5773, F1=0.6413
- Epoch 4: Loss=0.0773, mIoU=0.6211, F1=0.7004
- Epoch 5: Loss=0.0729, mIoU=0.6360, F1=0.7188
- Epoch 6: Loss=0.0714, mIoU=0.6873, F1=0.7763
- Epoch 7: Loss=0.0672, mIoU=0.6562, F1=0.7430
- Epoch 8: Loss=0.0636, mIoU=0.6806, F1=0.7692
- Epoch 9: Loss=0.0623, mIoU=0.7127, F1=0.8013
- Epoch 10: Loss=0.0590, mIoU=0.6724, F1=0.7601

本轮结果: Round=14, Labeled=1333, Selection=best_val (epoch=9), mIoU=0.7127, F1=0.8013, peak_mIoU=0.7127

Round=14, Labeled=1333, mIoU=0.7127, F1=0.8013

## Round 15

Labeled Pool Size: 1421

- Epoch 1: Loss=0.2101, mIoU=0.6132, F1=0.6931
- Epoch 2: Loss=0.0942, mIoU=0.7067, F1=0.7955
- Epoch 3: Loss=0.0823, mIoU=0.6162, F1=0.6942
- Epoch 4: Loss=0.0748, mIoU=0.5804, F1=0.6463
- Epoch 5: Loss=0.0704, mIoU=0.7208, F1=0.8096
- Epoch 6: Loss=0.0673, mIoU=0.5975, F1=0.6695
- Epoch 7: Loss=0.0634, mIoU=0.6628, F1=0.7497
- Epoch 8: Loss=0.0606, mIoU=0.6111, F1=0.6876
- Epoch 9: Loss=0.0584, mIoU=0.6836, F1=0.7722
- Epoch 10: Loss=0.0569, mIoU=0.7208, F1=0.8092

本轮结果: Round=15, Labeled=1421, Selection=best_val (epoch=5), mIoU=0.7208, F1=0.8096, peak_mIoU=0.7208

Round=15, Labeled=1421, mIoU=0.7208, F1=0.8096

## Round 16

Labeled Pool Size: 1509

- Test-only mode: Round 16 loads Round 15 best_val checkpoint (epoch=5, val_mIoU=0.7208, val_F1=0.8096)

本轮结果: Round=16, Labeled=1509, Selection=prev_round_best_val (source_round=15, epoch=5), mIoU=0.7208, F1=0.8096, peak_mIoU=0.7208

Round=16, Labeled=1509, mIoU=0.7208, F1=0.8096


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421, 1509]
ALC(基于每轮选模val mIoU): 0.5920
最后一轮选模 mIoU(val): 0.7208441877839645
最后一轮选模 F1(val): 0.8096424032861129
最终报告 mIoU(test): 0.6971205375238316
最终报告 F1(test): 0.7875518572121556
最终输出 mIoU: 0.6971 (source=final_report)
最终输出 F1: 0.7876 (source=final_report)
最终 Test Split: test
最终 Report: {'loss': 0.057150302454829216, 'mIoU': 0.6971205375238316, 'f1_score': 0.7875518572121556}
