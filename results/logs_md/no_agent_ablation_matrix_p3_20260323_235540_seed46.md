# 实验日志

实验名称: no_agent
描述: 消融：w/o Agent（argmax）；保留三阶段λ policy，与full_model一致
开始时间: 2026-03-25T23:49:26.480551

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.4573, mIoU=0.5019, F1=0.5187
- Epoch 2: Loss=0.2552, mIoU=0.4938, F1=0.5006
- Epoch 3: Loss=0.1678, mIoU=0.5082, F1=0.5286
- Epoch 4: Loss=0.1243, mIoU=0.5125, F1=0.5363
- Epoch 5: Loss=0.0994, mIoU=0.5125, F1=0.5358
- Epoch 6: Loss=0.0849, mIoU=0.5349, F1=0.5752
- Epoch 7: Loss=0.0751, mIoU=0.5196, F1=0.5487
- Epoch 8: Loss=0.0659, mIoU=0.5273, F1=0.5620
- Epoch 9: Loss=0.0630, mIoU=0.5256, F1=0.5591
- Epoch 10: Loss=0.0601, mIoU=0.5150, F1=0.5403

本轮结果: Round=1, Labeled=189, Selection=best_val (epoch=6), mIoU=0.5349, F1=0.5752, peak_mIoU=0.5349

Round=1, Labeled=189, mIoU=0.5349, F1=0.5752

## Round 2

Labeled Pool Size: 277

- Epoch 1: Loss=0.5422, mIoU=0.5387, F1=0.5995
- Epoch 2: Loss=0.2672, mIoU=0.5039, F1=0.5210
- Epoch 3: Loss=0.1861, mIoU=0.5361, F1=0.5773
- Epoch 4: Loss=0.1483, mIoU=0.5041, F1=0.5206
- Epoch 5: Loss=0.1329, mIoU=0.5083, F1=0.5281
- Epoch 6: Loss=0.1259, mIoU=0.5409, F1=0.5851
- Epoch 7: Loss=0.1201, mIoU=0.6057, F1=0.6808
- Epoch 8: Loss=0.1118, mIoU=0.5208, F1=0.5508
- Epoch 9: Loss=0.1103, mIoU=0.5303, F1=0.5674
- Epoch 10: Loss=0.1025, mIoU=0.5373, F1=0.5797

本轮结果: Round=2, Labeled=277, Selection=best_val (epoch=7), mIoU=0.6057, F1=0.6808, peak_mIoU=0.6057

Round=2, Labeled=277, mIoU=0.6057, F1=0.6808

## Round 3

Labeled Pool Size: 365

- Epoch 1: Loss=0.4536, mIoU=0.5654, F1=0.6238
- Epoch 2: Loss=0.2270, mIoU=0.5453, F1=0.5923
- Epoch 3: Loss=0.1722, mIoU=0.5019, F1=0.5160
- Epoch 4: Loss=0.1509, mIoU=0.5254, F1=0.5587
- Epoch 5: Loss=0.1335, mIoU=0.5021, F1=0.5163
- Epoch 6: Loss=0.1240, mIoU=0.5452, F1=0.5920
- Epoch 7: Loss=0.1160, mIoU=0.5521, F1=0.6032
- Epoch 8: Loss=0.1148, mIoU=0.5627, F1=0.6202
- Epoch 9: Loss=0.1122, mIoU=0.5968, F1=0.6686
- Epoch 10: Loss=0.1038, mIoU=0.6845, F1=0.7729

本轮结果: Round=3, Labeled=365, Selection=best_val (epoch=10), mIoU=0.6845, F1=0.7729, peak_mIoU=0.6845

Round=3, Labeled=365, mIoU=0.6845, F1=0.7729

## Round 4

Labeled Pool Size: 453

- Epoch 1: Loss=0.4849, mIoU=0.5698, F1=0.6382
- Epoch 2: Loss=0.2089, mIoU=0.5259, F1=0.5596
- Epoch 3: Loss=0.1655, mIoU=0.5209, F1=0.5509
- Epoch 4: Loss=0.1497, mIoU=0.5463, F1=0.5939
- Epoch 5: Loss=0.1402, mIoU=0.5457, F1=0.5974
- Epoch 6: Loss=0.1333, mIoU=0.5493, F1=0.5986
- Epoch 7: Loss=0.1233, mIoU=0.5928, F1=0.6648
- Epoch 8: Loss=0.1201, mIoU=0.5796, F1=0.6447
- Epoch 9: Loss=0.1159, mIoU=0.5926, F1=0.6632
- Epoch 10: Loss=0.1109, mIoU=0.6054, F1=0.6808

本轮结果: Round=4, Labeled=453, Selection=best_val (epoch=10), mIoU=0.6054, F1=0.6808, peak_mIoU=0.6054

Round=4, Labeled=453, mIoU=0.6054, F1=0.6808

## Round 5

Labeled Pool Size: 541

- Epoch 1: Loss=0.4463, mIoU=0.5560, F1=0.6096
- Epoch 2: Loss=0.2022, mIoU=0.5236, F1=0.5560
- Epoch 3: Loss=0.1601, mIoU=0.6282, F1=0.7094
- Epoch 4: Loss=0.1405, mIoU=0.5277, F1=0.5630
- Epoch 5: Loss=0.1315, mIoU=0.5177, F1=0.5454
- Epoch 6: Loss=0.1227, mIoU=0.6397, F1=0.7230
- Epoch 7: Loss=0.1248, mIoU=0.5893, F1=0.6586
- Epoch 8: Loss=0.1183, mIoU=0.5245, F1=0.5572
- Epoch 9: Loss=0.1123, mIoU=0.6188, F1=0.6975
- Epoch 10: Loss=0.1069, mIoU=0.6409, F1=0.7246

本轮结果: Round=5, Labeled=541, Selection=best_val (epoch=10), mIoU=0.6409, F1=0.7246, peak_mIoU=0.6409

Round=5, Labeled=541, mIoU=0.6409, F1=0.7246

## Round 6

Labeled Pool Size: 629

- Epoch 1: Loss=0.2998, mIoU=0.5630, F1=0.6212
- Epoch 2: Loss=0.1676, mIoU=0.5741, F1=0.6368
- Epoch 3: Loss=0.1420, mIoU=0.6297, F1=0.7118
- Epoch 4: Loss=0.1284, mIoU=0.6700, F1=0.7587
- Epoch 5: Loss=0.1204, mIoU=0.6715, F1=0.7595
- Epoch 6: Loss=0.1139, mIoU=0.6040, F1=0.6785
- Epoch 7: Loss=0.1092, mIoU=0.7040, F1=0.7929
- Epoch 8: Loss=0.1104, mIoU=0.6118, F1=0.6891
- Epoch 9: Loss=0.1037, mIoU=0.6736, F1=0.7617
- Epoch 10: Loss=0.1011, mIoU=0.6462, F1=0.7308

本轮结果: Round=6, Labeled=629, Selection=best_val (epoch=7), mIoU=0.7040, F1=0.7929, peak_mIoU=0.7040

Round=6, Labeled=629, mIoU=0.7040, F1=0.7929

## Round 7

Labeled Pool Size: 717

- Epoch 1: Loss=0.2669, mIoU=0.5141, F1=0.5387
- Epoch 2: Loss=0.1522, mIoU=0.6280, F1=0.7090
- Epoch 3: Loss=0.1341, mIoU=0.5528, F1=0.6040
- Epoch 4: Loss=0.1218, mIoU=0.6632, F1=0.7500
- Epoch 5: Loss=0.1152, mIoU=0.7049, F1=0.7939
- Epoch 6: Loss=0.1110, mIoU=0.6770, F1=0.7650
- Epoch 7: Loss=0.1059, mIoU=0.6801, F1=0.7683
- Epoch 8: Loss=0.1017, mIoU=0.6761, F1=0.7639
- Epoch 9: Loss=0.0995, mIoU=0.6993, F1=0.7882
- Epoch 10: Loss=0.0957, mIoU=0.6924, F1=0.7817

本轮结果: Round=7, Labeled=717, Selection=best_val (epoch=5), mIoU=0.7049, F1=0.7939, peak_mIoU=0.7049

Round=7, Labeled=717, mIoU=0.7049, F1=0.7939

## Round 8

Labeled Pool Size: 805

- Epoch 1: Loss=0.2550, mIoU=0.5902, F1=0.6614
- Epoch 2: Loss=0.1428, mIoU=0.5089, F1=0.5292
- Epoch 3: Loss=0.1224, mIoU=0.5562, F1=0.6096
- Epoch 4: Loss=0.1130, mIoU=0.5990, F1=0.6778
- Epoch 5: Loss=0.1088, mIoU=0.5695, F1=0.6299
- Epoch 6: Loss=0.1018, mIoU=0.6316, F1=0.7180
- Epoch 7: Loss=0.0981, mIoU=0.5947, F1=0.6661
- Epoch 8: Loss=0.0950, mIoU=0.6808, F1=0.7702
- Epoch 9: Loss=0.0915, mIoU=0.6880, F1=0.7770
- Epoch 10: Loss=0.0887, mIoU=0.6701, F1=0.7587

本轮结果: Round=8, Labeled=805, Selection=best_val (epoch=9), mIoU=0.6880, F1=0.7770, peak_mIoU=0.6880

Round=8, Labeled=805, mIoU=0.6880, F1=0.7770

## Round 9

Labeled Pool Size: 893

- Epoch 1: Loss=0.3014, mIoU=0.5581, F1=0.6129
- Epoch 2: Loss=0.1453, mIoU=0.5843, F1=0.6512
- Epoch 3: Loss=0.1243, mIoU=0.6659, F1=0.7531
- Epoch 4: Loss=0.1128, mIoU=0.6109, F1=0.6874
- Epoch 5: Loss=0.1042, mIoU=0.6903, F1=0.7790
- Epoch 6: Loss=0.1030, mIoU=0.6642, F1=0.7509
- Epoch 7: Loss=0.0972, mIoU=0.6826, F1=0.7711
- Epoch 8: Loss=0.0914, mIoU=0.6987, F1=0.7877
- Epoch 9: Loss=0.0884, mIoU=0.6909, F1=0.7799
- Epoch 10: Loss=0.0855, mIoU=0.6938, F1=0.7836

本轮结果: Round=9, Labeled=893, Selection=best_val (epoch=8), mIoU=0.6987, F1=0.7877, peak_mIoU=0.6987

Round=9, Labeled=893, mIoU=0.6987, F1=0.7877

## Round 10

Labeled Pool Size: 981

- Epoch 1: Loss=0.2146, mIoU=0.5251, F1=0.5581
- Epoch 2: Loss=0.1244, mIoU=0.5890, F1=0.6578
- Epoch 3: Loss=0.1124, mIoU=0.6574, F1=0.7436
- Epoch 4: Loss=0.1002, mIoU=0.6889, F1=0.7777
- Epoch 5: Loss=0.0972, mIoU=0.7045, F1=0.7932
- Epoch 6: Loss=0.0904, mIoU=0.6965, F1=0.7863
- Epoch 7: Loss=0.0885, mIoU=0.7085, F1=0.7971
- Epoch 8: Loss=0.0852, mIoU=0.6922, F1=0.7813
- Epoch 9: Loss=0.0825, mIoU=0.7088, F1=0.7976
- Epoch 10: Loss=0.0775, mIoU=0.6735, F1=0.7633

本轮结果: Round=10, Labeled=981, Selection=best_val (epoch=9), mIoU=0.7088, F1=0.7976, peak_mIoU=0.7088

Round=10, Labeled=981, mIoU=0.7088, F1=0.7976

## Round 11

Labeled Pool Size: 1069

- Epoch 1: Loss=0.2532, mIoU=0.5516, F1=0.6024
- Epoch 2: Loss=0.1232, mIoU=0.5716, F1=0.6327
- Epoch 3: Loss=0.1065, mIoU=0.6692, F1=0.7568
- Epoch 4: Loss=0.0978, mIoU=0.6885, F1=0.7777
- Epoch 5: Loss=0.0926, mIoU=0.6505, F1=0.7391
- Epoch 6: Loss=0.0897, mIoU=0.6251, F1=0.7060
- Epoch 7: Loss=0.0881, mIoU=0.6497, F1=0.7361
- Epoch 8: Loss=0.0814, mIoU=0.6795, F1=0.7691
- Epoch 9: Loss=0.0786, mIoU=0.6231, F1=0.7047
- Epoch 10: Loss=0.0775, mIoU=0.6988, F1=0.7876

本轮结果: Round=11, Labeled=1069, Selection=best_val (epoch=10), mIoU=0.6988, F1=0.7876, peak_mIoU=0.6988

Round=11, Labeled=1069, mIoU=0.6988, F1=0.7876

## Round 12

Labeled Pool Size: 1157

- Epoch 1: Loss=0.2423, mIoU=0.5235, F1=0.5556
- Epoch 2: Loss=0.1176, mIoU=0.5677, F1=0.6270
- Epoch 3: Loss=0.1007, mIoU=0.6276, F1=0.7085
- Epoch 4: Loss=0.0922, mIoU=0.6805, F1=0.7702
- Epoch 5: Loss=0.0875, mIoU=0.6487, F1=0.7372
- Epoch 6: Loss=0.0849, mIoU=0.6909, F1=0.7797
- Epoch 7: Loss=0.0793, mIoU=0.6360, F1=0.7215
- Epoch 8: Loss=0.0765, mIoU=0.6096, F1=0.6859
- Epoch 9: Loss=0.0777, mIoU=0.6783, F1=0.7665
- Epoch 10: Loss=0.0730, mIoU=0.6804, F1=0.7692

本轮结果: Round=12, Labeled=1157, Selection=best_val (epoch=6), mIoU=0.6909, F1=0.7797, peak_mIoU=0.6909

Round=12, Labeled=1157, mIoU=0.6909, F1=0.7797

## Round 13

Labeled Pool Size: 1245

- Epoch 1: Loss=0.2330, mIoU=0.6817, F1=0.7703
- Epoch 2: Loss=0.1096, mIoU=0.5914, F1=0.6611
- Epoch 3: Loss=0.0959, mIoU=0.6655, F1=0.7529
- Epoch 4: Loss=0.0881, mIoU=0.6353, F1=0.7225
- Epoch 5: Loss=0.0818, mIoU=0.6479, F1=0.7331
- Epoch 6: Loss=0.0766, mIoU=0.6487, F1=0.7337
- Epoch 7: Loss=0.0756, mIoU=0.5538, F1=0.6056
- Epoch 8: Loss=0.0740, mIoU=0.6715, F1=0.7595
- Epoch 9: Loss=0.0700, mIoU=0.6478, F1=0.7327
- Epoch 10: Loss=0.0675, mIoU=0.6336, F1=0.7164

本轮结果: Round=13, Labeled=1245, Selection=best_val (epoch=1), mIoU=0.6817, F1=0.7703, peak_mIoU=0.6817

Round=13, Labeled=1245, mIoU=0.6817, F1=0.7703

## Round 14

Labeled Pool Size: 1333

- Epoch 1: Loss=0.2070, mIoU=0.5399, F1=0.5833
- Epoch 2: Loss=0.0993, mIoU=0.5592, F1=0.6140
- Epoch 3: Loss=0.0885, mIoU=0.6590, F1=0.7459
- Epoch 4: Loss=0.0795, mIoU=0.7015, F1=0.7901
- Epoch 5: Loss=0.0768, mIoU=0.6942, F1=0.7833
- Epoch 6: Loss=0.0724, mIoU=0.6495, F1=0.7349
- Epoch 7: Loss=0.0698, mIoU=0.6409, F1=0.7247
- Epoch 8: Loss=0.0658, mIoU=0.6310, F1=0.7126
- Epoch 9: Loss=0.0641, mIoU=0.7070, F1=0.7961
- Epoch 10: Loss=0.0625, mIoU=0.6630, F1=0.7508

本轮结果: Round=14, Labeled=1333, Selection=best_val (epoch=9), mIoU=0.7070, F1=0.7961, peak_mIoU=0.7070

Round=14, Labeled=1333, mIoU=0.7070, F1=0.7961

## Round 15

Labeled Pool Size: 1421

- Epoch 1: Loss=0.2245, mIoU=0.5485, F1=0.5974
- Epoch 2: Loss=0.0987, mIoU=0.6133, F1=0.6907
- Epoch 3: Loss=0.0848, mIoU=0.5963, F1=0.6678
- Epoch 4: Loss=0.0771, mIoU=0.6455, F1=0.7329
- Epoch 5: Loss=0.0744, mIoU=0.7023, F1=0.7918
- Epoch 6: Loss=0.0697, mIoU=0.6820, F1=0.7705
- Epoch 7: Loss=0.0660, mIoU=0.6866, F1=0.7752
- Epoch 8: Loss=0.0652, mIoU=0.6972, F1=0.7869
- Epoch 9: Loss=0.0644, mIoU=0.6970, F1=0.7863
- Epoch 10: Loss=0.0612, mIoU=0.6815, F1=0.7705

本轮结果: Round=15, Labeled=1421, Selection=best_val (epoch=5), mIoU=0.7023, F1=0.7918, peak_mIoU=0.7023

Round=15, Labeled=1421, mIoU=0.7023, F1=0.7918

## Round 16

Labeled Pool Size: 1509

- Test-only mode: Round 16 loads Round 15 best_val checkpoint (epoch=5, val_mIoU=0.7023, val_F1=0.7918)

本轮结果: Round=16, Labeled=1509, Selection=prev_round_best_val (source_round=15, epoch=5), mIoU=0.7023, F1=0.7918, peak_mIoU=0.7023

Round=16, Labeled=1509, mIoU=0.7023, F1=0.7918


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421, 1509]
ALC(基于每轮选模val mIoU): 0.5921
最后一轮选模 mIoU(val): 0.7023300559827783
最后一轮选模 F1(val): 0.7918022000323353
最终报告 mIoU(test): 0.7087695327641192
最终报告 F1(test): 0.7986432407015067
最终输出 mIoU: 0.7088 (source=final_report)
最终输出 F1: 0.7986 (source=final_report)
最终 Test Split: test
最终 Report: {'loss': 0.06162293896079064, 'mIoU': 0.7087695327641192, 'f1_score': 0.7986432407015067}
