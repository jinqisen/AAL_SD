# 实验日志

实验名称: fixed_lambda
描述: 消融：Fixed λ=0.5；保留Agent，隔离三阶段λ policy贡献
开始时间: 2026-03-26T07:39:43.157505

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.4569, mIoU=0.5010, F1=0.5158
- Epoch 2: Loss=0.2541, mIoU=0.4947, F1=0.5022
- Epoch 3: Loss=0.1699, mIoU=0.4979, F1=0.5085
- Epoch 4: Loss=0.1246, mIoU=0.5246, F1=0.5578
- Epoch 5: Loss=0.1010, mIoU=0.5067, F1=0.5255
- Epoch 6: Loss=0.0851, mIoU=0.5232, F1=0.5551
- Epoch 7: Loss=0.0751, mIoU=0.5054, F1=0.5229
- Epoch 8: Loss=0.0658, mIoU=0.5342, F1=0.5741
- Epoch 9: Loss=0.0653, mIoU=0.5321, F1=0.5704
- Epoch 10: Loss=0.0600, mIoU=0.4988, F1=0.5102

本轮结果: Round=1, Labeled=189, Selection=best_val (epoch=8), mIoU=0.5342, F1=0.5741, peak_mIoU=0.5342

Round=1, Labeled=189, mIoU=0.5342, F1=0.5741

## Round 2

Labeled Pool Size: 277

- Epoch 1: Loss=0.5445, mIoU=0.5063, F1=0.5418
- Epoch 2: Loss=0.2702, mIoU=0.5306, F1=0.5693
- Epoch 3: Loss=0.1880, mIoU=0.5038, F1=0.5208
- Epoch 4: Loss=0.1524, mIoU=0.5313, F1=0.5689
- Epoch 5: Loss=0.1410, mIoU=0.5844, F1=0.6524
- Epoch 6: Loss=0.1312, mIoU=0.5987, F1=0.6712
- Epoch 7: Loss=0.1174, mIoU=0.5800, F1=0.6456
- Epoch 8: Loss=0.1122, mIoU=0.6191, F1=0.6983
- Epoch 9: Loss=0.1156, mIoU=0.5537, F1=0.6055
- Epoch 10: Loss=0.1056, mIoU=0.5886, F1=0.6574

本轮结果: Round=2, Labeled=277, Selection=best_val (epoch=8), mIoU=0.6191, F1=0.6983, peak_mIoU=0.6191

Round=2, Labeled=277, mIoU=0.6191, F1=0.6983

## Round 3

Labeled Pool Size: 365

- Epoch 1: Loss=0.4471, mIoU=0.5309, F1=0.5719
- Epoch 2: Loss=0.2175, mIoU=0.4997, F1=0.5129
- Epoch 3: Loss=0.1623, mIoU=0.4968, F1=0.5065
- Epoch 4: Loss=0.1402, mIoU=0.5413, F1=0.5856
- Epoch 5: Loss=0.1237, mIoU=0.5090, F1=0.5293
- Epoch 6: Loss=0.1103, mIoU=0.5955, F1=0.6673
- Epoch 7: Loss=0.1089, mIoU=0.6044, F1=0.6790
- Epoch 8: Loss=0.1001, mIoU=0.5618, F1=0.6181
- Epoch 9: Loss=0.0975, mIoU=0.6204, F1=0.7006
- Epoch 10: Loss=0.0886, mIoU=0.5740, F1=0.6363

本轮结果: Round=3, Labeled=365, Selection=best_val (epoch=9), mIoU=0.6204, F1=0.7006, peak_mIoU=0.6204

Round=3, Labeled=365, mIoU=0.6204, F1=0.7006

## Round 4

Labeled Pool Size: 453

- Epoch 1: Loss=0.4924, mIoU=0.5309, F1=0.5766
- Epoch 2: Loss=0.2095, mIoU=0.4994, F1=0.5144
- Epoch 3: Loss=0.1592, mIoU=0.5003, F1=0.5135
- Epoch 4: Loss=0.1367, mIoU=0.5401, F1=0.5838
- Epoch 5: Loss=0.1240, mIoU=0.5137, F1=0.5379
- Epoch 6: Loss=0.1191, mIoU=0.6576, F1=0.7438
- Epoch 7: Loss=0.1134, mIoU=0.5242, F1=0.5566
- Epoch 8: Loss=0.1087, mIoU=0.5834, F1=0.6501
- Epoch 9: Loss=0.1031, mIoU=0.6552, F1=0.7409
- Epoch 10: Loss=0.0974, mIoU=0.6813, F1=0.7698

本轮结果: Round=4, Labeled=453, Selection=best_val (epoch=10), mIoU=0.6813, F1=0.7698, peak_mIoU=0.6813

Round=4, Labeled=453, mIoU=0.6813, F1=0.7698

## Round 5

Labeled Pool Size: 541

- Epoch 1: Loss=0.4378, mIoU=0.4973, F1=0.5103
- Epoch 2: Loss=0.1836, mIoU=0.6003, F1=0.6731
- Epoch 3: Loss=0.1498, mIoU=0.5021, F1=0.5164
- Epoch 4: Loss=0.1344, mIoU=0.5547, F1=0.6072
- Epoch 5: Loss=0.1295, mIoU=0.6707, F1=0.7583
- Epoch 6: Loss=0.1206, mIoU=0.6977, F1=0.7869
- Epoch 7: Loss=0.1119, mIoU=0.6982, F1=0.7875
- Epoch 8: Loss=0.1094, mIoU=0.6401, F1=0.7249
- Epoch 9: Loss=0.1065, mIoU=0.7002, F1=0.7889
- Epoch 10: Loss=0.1009, mIoU=0.6855, F1=0.7740

本轮结果: Round=5, Labeled=541, Selection=best_val (epoch=9), mIoU=0.7002, F1=0.7889, peak_mIoU=0.7002

Round=5, Labeled=541, mIoU=0.7002, F1=0.7889

## Round 6

Labeled Pool Size: 629

- Epoch 1: Loss=0.2993, mIoU=0.5390, F1=0.5819
- Epoch 2: Loss=0.1649, mIoU=0.5189, F1=0.5474
- Epoch 3: Loss=0.1385, mIoU=0.5451, F1=0.5921
- Epoch 4: Loss=0.1227, mIoU=0.5167, F1=0.5433
- Epoch 5: Loss=0.1163, mIoU=0.6620, F1=0.7490
- Epoch 6: Loss=0.1101, mIoU=0.6225, F1=0.7026
- Epoch 7: Loss=0.1083, mIoU=0.6597, F1=0.7469
- Epoch 8: Loss=0.1038, mIoU=0.6008, F1=0.6748
- Epoch 9: Loss=0.1000, mIoU=0.6496, F1=0.7356
- Epoch 10: Loss=0.0960, mIoU=0.6061, F1=0.6810

本轮结果: Round=6, Labeled=629, Selection=best_val (epoch=5), mIoU=0.6620, F1=0.7490, peak_mIoU=0.6620

Round=6, Labeled=629, mIoU=0.6620, F1=0.7490

## Round 7

Labeled Pool Size: 717

- Epoch 1: Loss=0.2664, mIoU=0.5232, F1=0.5553
- Epoch 2: Loss=0.1509, mIoU=0.5440, F1=0.5906
- Epoch 3: Loss=0.1285, mIoU=0.5898, F1=0.6592
- Epoch 4: Loss=0.1164, mIoU=0.6178, F1=0.6963
- Epoch 5: Loss=0.1147, mIoU=0.6872, F1=0.7758
- Epoch 6: Loss=0.1080, mIoU=0.6862, F1=0.7747
- Epoch 7: Loss=0.1035, mIoU=0.6708, F1=0.7585
- Epoch 8: Loss=0.1003, mIoU=0.6978, F1=0.7869
- Epoch 9: Loss=0.0958, mIoU=0.6856, F1=0.7750
- Epoch 10: Loss=0.0916, mIoU=0.6658, F1=0.7529

本轮结果: Round=7, Labeled=717, Selection=best_val (epoch=8), mIoU=0.6978, F1=0.7869, peak_mIoU=0.6978

Round=7, Labeled=717, mIoU=0.6978, F1=0.7869

## Round 8

Labeled Pool Size: 805

- Epoch 1: Loss=0.2543, mIoU=0.5280, F1=0.5634
- Epoch 2: Loss=0.1438, mIoU=0.5277, F1=0.5629
- Epoch 3: Loss=0.1246, mIoU=0.5803, F1=0.6453
- Epoch 4: Loss=0.1157, mIoU=0.6687, F1=0.7564
- Epoch 5: Loss=0.1086, mIoU=0.6913, F1=0.7803
- Epoch 6: Loss=0.1045, mIoU=0.6776, F1=0.7657
- Epoch 7: Loss=0.0993, mIoU=0.6407, F1=0.7280
- Epoch 8: Loss=0.0972, mIoU=0.7111, F1=0.7999
- Epoch 9: Loss=0.0931, mIoU=0.6539, F1=0.7396
- Epoch 10: Loss=0.0906, mIoU=0.6834, F1=0.7729

本轮结果: Round=8, Labeled=805, Selection=best_val (epoch=8), mIoU=0.7111, F1=0.7999, peak_mIoU=0.7111

Round=8, Labeled=805, mIoU=0.7111, F1=0.7999

## Round 9

Labeled Pool Size: 893

- Epoch 1: Loss=0.2969, mIoU=0.5279, F1=0.5631
- Epoch 2: Loss=0.1451, mIoU=0.5539, F1=0.6062
- Epoch 3: Loss=0.1273, mIoU=0.5519, F1=0.6028
- Epoch 4: Loss=0.1145, mIoU=0.6900, F1=0.7790
- Epoch 5: Loss=0.1095, mIoU=0.6715, F1=0.7597
- Epoch 6: Loss=0.1046, mIoU=0.6359, F1=0.7232
- Epoch 7: Loss=0.0994, mIoU=0.6090, F1=0.6916
- Epoch 8: Loss=0.0956, mIoU=0.6811, F1=0.7701
- Epoch 9: Loss=0.0916, mIoU=0.6743, F1=0.7635
- Epoch 10: Loss=0.0898, mIoU=0.6589, F1=0.7474

本轮结果: Round=9, Labeled=893, Selection=best_val (epoch=4), mIoU=0.6900, F1=0.7790, peak_mIoU=0.6900

Round=9, Labeled=893, mIoU=0.6900, F1=0.7790

## Round 10

Labeled Pool Size: 981

- Epoch 1: Loss=0.2175, mIoU=0.5084, F1=0.5283
- Epoch 2: Loss=0.1308, mIoU=0.5660, F1=0.6245
- Epoch 3: Loss=0.1136, mIoU=0.6927, F1=0.7821
- Epoch 4: Loss=0.1068, mIoU=0.6753, F1=0.7630
- Epoch 5: Loss=0.0987, mIoU=0.6925, F1=0.7817
- Epoch 6: Loss=0.0946, mIoU=0.6973, F1=0.7860
- Epoch 7: Loss=0.0921, mIoU=0.6048, F1=0.6794
- Epoch 8: Loss=0.0887, mIoU=0.6754, F1=0.7635
- Epoch 9: Loss=0.0846, mIoU=0.6782, F1=0.7664
- Epoch 10: Loss=0.0814, mIoU=0.7453, F1=0.8312

本轮结果: Round=10, Labeled=981, Selection=best_val (epoch=10), mIoU=0.7453, F1=0.8312, peak_mIoU=0.7453

Round=10, Labeled=981, mIoU=0.7453, F1=0.8312

## Round 11

Labeled Pool Size: 1069

- Epoch 1: Loss=0.2601, mIoU=0.5213, F1=0.5517
- Epoch 2: Loss=0.1267, mIoU=0.5191, F1=0.5477
- Epoch 3: Loss=0.1129, mIoU=0.5524, F1=0.6039
- Epoch 4: Loss=0.1018, mIoU=0.6113, F1=0.6887
- Epoch 5: Loss=0.0951, mIoU=0.6807, F1=0.7689
- Epoch 6: Loss=0.0939, mIoU=0.6617, F1=0.7485
- Epoch 7: Loss=0.0898, mIoU=0.6959, F1=0.7847
- Epoch 8: Loss=0.0833, mIoU=0.7003, F1=0.7899
- Epoch 9: Loss=0.0835, mIoU=0.6609, F1=0.7480
- Epoch 10: Loss=0.0797, mIoU=0.6609, F1=0.7477

本轮结果: Round=11, Labeled=1069, Selection=best_val (epoch=8), mIoU=0.7003, F1=0.7899, peak_mIoU=0.7003

Round=11, Labeled=1069, mIoU=0.7003, F1=0.7899

## Round 12

Labeled Pool Size: 1157

- Epoch 1: Loss=0.2453, mIoU=0.5155, F1=0.5412
- Epoch 2: Loss=0.1199, mIoU=0.6603, F1=0.7474
- Epoch 3: Loss=0.1017, mIoU=0.5775, F1=0.6414
- Epoch 4: Loss=0.0942, mIoU=0.5506, F1=0.6009
- Epoch 5: Loss=0.0898, mIoU=0.6767, F1=0.7657
- Epoch 6: Loss=0.0857, mIoU=0.6854, F1=0.7753
- Epoch 7: Loss=0.0841, mIoU=0.6739, F1=0.7620
- Epoch 8: Loss=0.0808, mIoU=0.6669, F1=0.7542
- Epoch 9: Loss=0.0754, mIoU=0.6783, F1=0.7665
- Epoch 10: Loss=0.0731, mIoU=0.6887, F1=0.7774

本轮结果: Round=12, Labeled=1157, Selection=best_val (epoch=10), mIoU=0.6887, F1=0.7774, peak_mIoU=0.6887

Round=12, Labeled=1157, mIoU=0.6887, F1=0.7774


--- [Checkpoint] 续跑开始时间: 2026-03-26T10:39:33.155840 ---

## Round 12

Labeled Pool Size: 1157

- Epoch 1: Loss=0.2456, mIoU=0.5113, F1=0.5337
- Epoch 2: Loss=0.1216, mIoU=0.5796, F1=0.6445
- Epoch 3: Loss=0.1039, mIoU=0.5792, F1=0.6438
- Epoch 4: Loss=0.0955, mIoU=0.5552, F1=0.6079
- Epoch 5: Loss=0.0875, mIoU=0.6891, F1=0.7778
- Epoch 6: Loss=0.0842, mIoU=0.6659, F1=0.7538
- Epoch 7: Loss=0.0832, mIoU=0.7063, F1=0.7952
- Epoch 8: Loss=0.0833, mIoU=0.6680, F1=0.7554
## Round 13

Labeled Pool Size: 1245

- Epoch 9: Loss=0.0769, mIoU=0.7199, F1=0.8081
- Epoch 1: Loss=0.2355, mIoU=0.5371, F1=0.5789
- Epoch 10: Loss=0.0735, mIoU=0.7117, F1=0.8004

本轮结果: Round=12, Labeled=1157, Selection=best_val (epoch=9), mIoU=0.7199, F1=0.8081, peak_mIoU=0.7199

Round=12, Labeled=1157, mIoU=0.7199, F1=0.8081

- Epoch 2: Loss=0.1117, mIoU=0.6418, F1=0.7291
- Epoch 3: Loss=0.0974, mIoU=0.6032, F1=0.6850
- Epoch 4: Loss=0.0909, mIoU=0.6933, F1=0.7819
- Epoch 5: Loss=0.0859, mIoU=0.6393, F1=0.7230
- Epoch 6: Loss=0.0816, mIoU=0.6608, F1=0.7502
## Round 13

Labeled Pool Size: 1245

- Epoch 7: Loss=0.0767, mIoU=0.7171, F1=0.8061
- Epoch 1: Loss=0.2338, mIoU=0.5177, F1=0.5452
- Epoch 8: Loss=0.0741, mIoU=0.6566, F1=0.7429
- Epoch 2: Loss=0.1095, mIoU=0.6519, F1=0.7396
- Epoch 9: Loss=0.0707, mIoU=0.6971, F1=0.7860
- Epoch 3: Loss=0.0950, mIoU=0.6886, F1=0.7772
- Epoch 10: Loss=0.0701, mIoU=0.6785, F1=0.7670

本轮结果: Round=13, Labeled=1245, Selection=best_val (epoch=7), mIoU=0.7171, F1=0.8061, peak_mIoU=0.7171

Round=13, Labeled=1245, mIoU=0.7171, F1=0.8061

- Epoch 4: Loss=0.0884, mIoU=0.6195, F1=0.6983
- Epoch 5: Loss=0.0849, mIoU=0.6721, F1=0.7596
- Epoch 6: Loss=0.0800, mIoU=0.6760, F1=0.7659
## Round 14

Labeled Pool Size: 1333

- Epoch 7: Loss=0.0770, mIoU=0.7459, F1=0.8317
- Epoch 1: Loss=0.2106, mIoU=0.6451, F1=0.7295
- Epoch 8: Loss=0.0749, mIoU=0.7162, F1=0.8045
- Epoch 2: Loss=0.1070, mIoU=0.6239, F1=0.7045
- Epoch 9: Loss=0.0714, mIoU=0.7068, F1=0.7960
- Epoch 3: Loss=0.0938, mIoU=0.6705, F1=0.7586
- Epoch 10: Loss=0.0691, mIoU=0.6754, F1=0.7634

本轮结果: Round=13, Labeled=1245, Selection=best_val (epoch=7), mIoU=0.7459, F1=0.8317, peak_mIoU=0.7459

Round=13, Labeled=1245, mIoU=0.7459, F1=0.8317

- Epoch 4: Loss=0.0864, mIoU=0.6603, F1=0.7468
- Epoch 5: Loss=0.0821, mIoU=0.7119, F1=0.8005
- Epoch 6: Loss=0.0769, mIoU=0.6846, F1=0.7732
## Round 14

Labeled Pool Size: 1333

- Epoch 7: Loss=0.0770, mIoU=0.7184, F1=0.8066
- Epoch 1: Loss=0.2091, mIoU=0.5521, F1=0.6031
- Epoch 8: Loss=0.0710, mIoU=0.7198, F1=0.8083
- Epoch 2: Loss=0.1037, mIoU=0.6673, F1=0.7547
- Epoch 9: Loss=0.0690, mIoU=0.7369, F1=0.8243
- Epoch 3: Loss=0.0902, mIoU=0.6135, F1=0.6906
- Epoch 10: Loss=0.0670, mIoU=0.6678, F1=0.7554

本轮结果: Round=14, Labeled=1333, Selection=best_val (epoch=9), mIoU=0.7369, F1=0.8243, peak_mIoU=0.7369

Round=14, Labeled=1333, mIoU=0.7369, F1=0.8243

- Epoch 4: Loss=0.0851, mIoU=0.6478, F1=0.7330
- Epoch 5: Loss=0.0789, mIoU=0.6658, F1=0.7528
## Round 15

Labeled Pool Size: 1421

- Epoch 6: Loss=0.0738, mIoU=0.5831, F1=0.6498
- Epoch 1: Loss=0.2286, mIoU=0.5868, F1=0.6549
- Epoch 7: Loss=0.0706, mIoU=0.6769, F1=0.7649
- Epoch 2: Loss=0.1039, mIoU=0.6087, F1=0.6843
- Epoch 8: Loss=0.0674, mIoU=0.6578, F1=0.7440
- Epoch 3: Loss=0.0895, mIoU=0.6610, F1=0.7478
- Epoch 9: Loss=0.0669, mIoU=0.6615, F1=0.7483
- Epoch 4: Loss=0.0825, mIoU=0.7130, F1=0.8015
- Epoch 10: Loss=0.0647, mIoU=0.6716, F1=0.7593

本轮结果: Round=14, Labeled=1333, Selection=best_val (epoch=7), mIoU=0.6769, F1=0.7649, peak_mIoU=0.6769

Round=14, Labeled=1333, mIoU=0.6769, F1=0.7649

- Epoch 5: Loss=0.0791, mIoU=0.7014, F1=0.7916
- Epoch 6: Loss=0.0763, mIoU=0.6848, F1=0.7736
- Epoch 7: Loss=0.0728, mIoU=0.6812, F1=0.7698
## Round 15

Labeled Pool Size: 1421

- Epoch 8: Loss=0.0701, mIoU=0.6896, F1=0.7785
- Epoch 1: Loss=0.2224, mIoU=0.5502, F1=0.6000
- Epoch 9: Loss=0.0664, mIoU=0.6791, F1=0.7676
- Epoch 2: Loss=0.0990, mIoU=0.6677, F1=0.7550
- Epoch 10: Loss=0.0651, mIoU=0.7129, F1=0.8017

本轮结果: Round=15, Labeled=1421, Selection=best_val (epoch=4), mIoU=0.7130, F1=0.8015, peak_mIoU=0.7130

Round=15, Labeled=1421, mIoU=0.7130, F1=0.8015

- Epoch 3: Loss=0.0857, mIoU=0.6810, F1=0.7694
- Epoch 4: Loss=0.0793, mIoU=0.6324, F1=0.7206
- Epoch 5: Loss=0.0751, mIoU=0.7180, F1=0.8065
- Epoch 6: Loss=0.0705, mIoU=0.6856, F1=0.7754
## Round 16

Labeled Pool Size: 1509

- Test-only mode: Round 16 loads Round 15 best_val checkpoint (epoch=4, val_mIoU=0.7130, val_F1=0.8015)

本轮结果: Round=16, Labeled=1509, Selection=prev_round_best_val (source_round=15, epoch=4), mIoU=0.7130, F1=0.8015, peak_mIoU=0.7130

Round=16, Labeled=1509, mIoU=0.7130, F1=0.8015


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421, 1509]
ALC(基于每轮选模val mIoU): 0.6018
最后一轮选模 mIoU(val): 0.712954808096699
最后一轮选模 F1(val): 0.801503475789046
最终报告 mIoU(test): 0.6947483148595675
最终报告 F1(test): 0.7842859721365001
最终输出 mIoU: 0.6947 (source=final_report)
最终输出 F1: 0.7843 (source=final_report)
最终 Test Split: test
最终 Report: {'loss': 0.046395530169829724, 'mIoU': 0.6947483148595675, 'f1_score': 0.7842859721365001}
- Epoch 7: Loss=0.0661, mIoU=0.7190, F1=0.8072
- Epoch 8: Loss=0.0668, mIoU=0.7085, F1=0.7972
- Epoch 9: Loss=0.0620, mIoU=0.7093, F1=0.7983
- Epoch 10: Loss=0.0613, mIoU=0.7075, F1=0.7968

本轮结果: Round=15, Labeled=1421, Selection=best_val (epoch=7), mIoU=0.7190, F1=0.8072, peak_mIoU=0.7190

Round=15, Labeled=1421, mIoU=0.7190, F1=0.8072

## Round 16

Labeled Pool Size: 1509

- Test-only mode: Round 16 loads Round 15 best_val checkpoint (epoch=7, val_mIoU=0.7190, val_F1=0.8072)

本轮结果: Round=16, Labeled=1509, Selection=prev_round_best_val (source_round=15, epoch=7), mIoU=0.7190, F1=0.8072, peak_mIoU=0.7190

Round=16, Labeled=1509, mIoU=0.7190, F1=0.8072


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421, 1509]
ALC(基于每轮选模val mIoU): 0.6024
最后一轮选模 mIoU(val): 0.7189897589540314
最后一轮选模 F1(val): 0.8071587289900901
最终报告 mIoU(test): 0.7145673585868514
最终报告 F1(test): 0.8033906477505006
最终输出 mIoU: 0.7146 (source=final_report)
最终输出 F1: 0.8034 (source=final_report)
最终 Test Split: test
最终 Report: {'loss': 0.04074724114499986, 'mIoU': 0.7145673585868514, 'f1_score': 0.8033906477505006}
