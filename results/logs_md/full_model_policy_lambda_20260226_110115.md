# 实验日志

实验名称: full_model_policy_lambda
描述: 完整模型（规则闭环λ）：固定Warmup(0.2) + 自适应阈值 rollback + 风险驱动 λ 闭环控制（不调整 query_size）
开始时间: 2026-02-26T17:33:37.990366

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.5717, mIoU=0.5019, F1=0.5527
- Epoch 2: Loss=0.2947, mIoU=0.5849, F1=0.6558
- Epoch 3: Loss=0.1778, mIoU=0.5418, F1=0.5864
- Epoch 4: Loss=0.1233, mIoU=0.5991, F1=0.6718
- Epoch 5: Loss=0.0999, mIoU=0.6138, F1=0.6912
- Epoch 6: Loss=0.0833, mIoU=0.5540, F1=0.6061
- Epoch 7: Loss=0.0778, mIoU=0.5132, F1=0.5371
- Epoch 8: Loss=0.0709, mIoU=0.5144, F1=0.5394
- Epoch 9: Loss=0.0656, mIoU=0.5172, F1=0.5444
- Epoch 10: Loss=0.0608, mIoU=0.5381, F1=0.5805

当前轮次最佳结果: Round=1, Labeled=189, mIoU=0.6138, F1=0.6912

## Round 2

Labeled Pool Size: 277

- Epoch 1: Loss=0.4716, mIoU=0.5275, F1=0.5649
- Epoch 2: Loss=0.2400, mIoU=0.5121, F1=0.5374
- Epoch 3: Loss=0.1696, mIoU=0.5828, F1=0.6495
- Epoch 4: Loss=0.1424, mIoU=0.5301, F1=0.5669
- Epoch 5: Loss=0.1254, mIoU=0.5321, F1=0.5702
- Epoch 6: Loss=0.1178, mIoU=0.6291, F1=0.7104
- Epoch 7: Loss=0.1108, mIoU=0.6044, F1=0.6788
- Epoch 8: Loss=0.1066, mIoU=0.5627, F1=0.6198
- Epoch 9: Loss=0.1104, mIoU=0.6218, F1=0.7016
- Epoch 10: Loss=0.0998, mIoU=0.5980, F1=0.6705

当前轮次最佳结果: Round=2, Labeled=277, mIoU=0.6291, F1=0.7104

## Round 3

Labeled Pool Size: 365

- Epoch 1: Loss=0.3144, mIoU=0.5150, F1=0.5405
- Epoch 2: Loss=0.1823, mIoU=0.5057, F1=0.5231
- Epoch 3: Loss=0.1524, mIoU=0.5632, F1=0.6203
- Epoch 4: Loss=0.1338, mIoU=0.5638, F1=0.6213
- Epoch 5: Loss=0.1289, mIoU=0.5487, F1=0.5977
- Epoch 6: Loss=0.1231, mIoU=0.5905, F1=0.6601
- Epoch 7: Loss=0.1145, mIoU=0.5574, F1=0.6115
- Epoch 8: Loss=0.1146, mIoU=0.6275, F1=0.7083
- Epoch 9: Loss=0.1086, mIoU=0.5799, F1=0.6451
- Epoch 10: Loss=0.1038, mIoU=0.5828, F1=0.6491

当前轮次最佳结果: Round=3, Labeled=365, mIoU=0.6275, F1=0.7083

## Round 4

Labeled Pool Size: 453

- Epoch 1: Loss=0.3792, mIoU=0.5277, F1=0.5635
- Epoch 2: Loss=0.1937, mIoU=0.4992, F1=0.5109
- Epoch 3: Loss=0.1571, mIoU=0.5652, F1=0.6235
- Epoch 4: Loss=0.1388, mIoU=0.5567, F1=0.6102
- Epoch 5: Loss=0.1297, mIoU=0.5949, F1=0.6658
- Epoch 6: Loss=0.1285, mIoU=0.6349, F1=0.7182
- Epoch 7: Loss=0.1173, mIoU=0.6495, F1=0.7345
- Epoch 8: Loss=0.1137, mIoU=0.6317, F1=0.7137
- Epoch 9: Loss=0.1101, mIoU=0.6088, F1=0.6845
- Epoch 10: Loss=0.1070, mIoU=0.6269, F1=0.7078

当前轮次最佳结果: Round=4, Labeled=453, mIoU=0.6495, F1=0.7345

## Round 5

Labeled Pool Size: 541

- Epoch 1: Loss=0.3368, mIoU=0.5320, F1=0.5702
- Epoch 2: Loss=0.1730, mIoU=0.5532, F1=0.6047
- Epoch 3: Loss=0.1483, mIoU=0.5678, F1=0.6272
- Epoch 4: Loss=0.1362, mIoU=0.6202, F1=0.7001
- Epoch 5: Loss=0.1275, mIoU=0.6441, F1=0.7289
- Epoch 6: Loss=0.1196, mIoU=0.6169, F1=0.6950
- Epoch 7: Loss=0.1148, mIoU=0.6884, F1=0.7777
- Epoch 8: Loss=0.1118, mIoU=0.6460, F1=0.7322
- Epoch 9: Loss=0.1068, mIoU=0.6604, F1=0.7484
- Epoch 10: Loss=0.1037, mIoU=0.6749, F1=0.7641

当前轮次最佳结果: Round=5, Labeled=541, mIoU=0.6884, F1=0.7777

## Round 6

Labeled Pool Size: 629

- Epoch 1: Loss=0.3089, mIoU=0.5266, F1=0.5613
- Epoch 2: Loss=0.1672, mIoU=0.5199, F1=0.5491
- Epoch 3: Loss=0.1404, mIoU=0.5333, F1=0.5724
- Epoch 4: Loss=0.1309, mIoU=0.6943, F1=0.7835
- Epoch 5: Loss=0.1240, mIoU=0.6882, F1=0.7768
- Epoch 6: Loss=0.1163, mIoU=0.6774, F1=0.7655
- Epoch 7: Loss=0.1134, mIoU=0.6523, F1=0.7381
- Epoch 8: Loss=0.1091, mIoU=0.6733, F1=0.7613
- Epoch 9: Loss=0.1054, mIoU=0.6664, F1=0.7539
- Epoch 10: Loss=0.1014, mIoU=0.6968, F1=0.7856

当前轮次最佳结果: Round=6, Labeled=629, mIoU=0.6968, F1=0.7856

## Round 7

Labeled Pool Size: 717

- Epoch 1: Loss=0.3469, mIoU=0.5166, F1=0.5434
- Epoch 2: Loss=0.1613, mIoU=0.5584, F1=0.6128
- Epoch 3: Loss=0.1392, mIoU=0.5358, F1=0.5764
- Epoch 4: Loss=0.1267, mIoU=0.6637, F1=0.7505
- Epoch 5: Loss=0.1212, mIoU=0.6054, F1=0.6806
- Epoch 6: Loss=0.1145, mIoU=0.6681, F1=0.7553
- Epoch 7: Loss=0.1106, mIoU=0.6907, F1=0.7796
- Epoch 8: Loss=0.1062, mIoU=0.6903, F1=0.7791
- Epoch 9: Loss=0.1025, mIoU=0.6798, F1=0.7682
- Epoch 10: Loss=0.0976, mIoU=0.7082, F1=0.7974

当前轮次最佳结果: Round=7, Labeled=717, mIoU=0.7082, F1=0.7974

## Round 8

Labeled Pool Size: 805

- Epoch 1: Loss=0.2983, mIoU=0.5018, F1=0.5159
- Epoch 2: Loss=0.1543, mIoU=0.5254, F1=0.5588
- Epoch 3: Loss=0.1305, mIoU=0.6184, F1=0.6971
- Epoch 4: Loss=0.1177, mIoU=0.6686, F1=0.7564
- Epoch 5: Loss=0.1114, mIoU=0.6761, F1=0.7646
- Epoch 6: Loss=0.1069, mIoU=0.6784, F1=0.7670
- Epoch 7: Loss=0.1014, mIoU=0.6799, F1=0.7684
- Epoch 8: Loss=0.0997, mIoU=0.6955, F1=0.7845
- Epoch 9: Loss=0.0937, mIoU=0.6527, F1=0.7411
- Epoch 10: Loss=0.0919, mIoU=0.6533, F1=0.7393

当前轮次最佳结果: Round=8, Labeled=805, mIoU=0.6955, F1=0.7845

## Round 9

Labeled Pool Size: 893

- Epoch 1: Loss=0.3393, mIoU=0.5361, F1=0.5770
- Epoch 2: Loss=0.1508, mIoU=0.6518, F1=0.7373
- Epoch 3: Loss=0.1275, mIoU=0.6244, F1=0.7043
- Epoch 4: Loss=0.1161, mIoU=0.6624, F1=0.7492
- Epoch 5: Loss=0.1108, mIoU=0.6560, F1=0.7426
- Epoch 6: Loss=0.1060, mIoU=0.7045, F1=0.7934
- Epoch 7: Loss=0.1016, mIoU=0.6198, F1=0.6986
- Epoch 8: Loss=0.0983, mIoU=0.7063, F1=0.7950
- Epoch 9: Loss=0.0932, mIoU=0.6984, F1=0.7877
- Epoch 10: Loss=0.0927, mIoU=0.6989, F1=0.7880

当前轮次最佳结果: Round=9, Labeled=893, mIoU=0.7063, F1=0.7950

## Round 10

Labeled Pool Size: 981

- Epoch 1: Loss=0.3130, mIoU=0.5903, F1=0.6597
- Epoch 2: Loss=0.1414, mIoU=0.6474, F1=0.7330
- Epoch 3: Loss=0.1211, mIoU=0.6707, F1=0.7594
- Epoch 4: Loss=0.1098, mIoU=0.6885, F1=0.7770
- Epoch 5: Loss=0.1051, mIoU=0.5967, F1=0.6687
- Epoch 6: Loss=0.0990, mIoU=0.7171, F1=0.8055
- Epoch 7: Loss=0.0957, mIoU=0.7211, F1=0.8092
- Epoch 8: Loss=0.0918, mIoU=0.6814, F1=0.7703
- Epoch 9: Loss=0.0891, mIoU=0.6926, F1=0.7813
- Epoch 10: Loss=0.0861, mIoU=0.7095, F1=0.7982

当前轮次最佳结果: Round=10, Labeled=981, mIoU=0.7211, F1=0.8092

## Round 11

Labeled Pool Size: 1069

- Epoch 1: Loss=0.2225, mIoU=0.5893, F1=0.6582
- Epoch 2: Loss=0.1265, mIoU=0.5570, F1=0.6107
- Epoch 3: Loss=0.1108, mIoU=0.6428, F1=0.7269
- Epoch 4: Loss=0.1016, mIoU=0.6769, F1=0.7654
- Epoch 5: Loss=0.0968, mIoU=0.6955, F1=0.7842
- Epoch 6: Loss=0.0931, mIoU=0.7105, F1=0.7996
- Epoch 7: Loss=0.0881, mIoU=0.5798, F1=0.6452
- Epoch 8: Loss=0.0873, mIoU=0.6977, F1=0.7868
- Epoch 9: Loss=0.0822, mIoU=0.6847, F1=0.7732
- Epoch 10: Loss=0.0806, mIoU=0.6778, F1=0.7661

当前轮次最佳结果: Round=11, Labeled=1069, mIoU=0.7105, F1=0.7996

## Round 12

Labeled Pool Size: 1157

- Epoch 1: Loss=0.2113, mIoU=0.5206, F1=0.5506
- Epoch 2: Loss=0.1225, mIoU=0.5932, F1=0.6636
- Epoch 3: Loss=0.1061, mIoU=0.6689, F1=0.7561
- Epoch 4: Loss=0.0986, mIoU=0.7019, F1=0.7907
- Epoch 5: Loss=0.0948, mIoU=0.7007, F1=0.7900
- Epoch 6: Loss=0.0902, mIoU=0.7255, F1=0.8131
- Epoch 7: Loss=0.0849, mIoU=0.7091, F1=0.7976
- Epoch 8: Loss=0.0820, mIoU=0.7052, F1=0.7941
- Epoch 9: Loss=0.0797, mIoU=0.7109, F1=0.7997
- Epoch 10: Loss=0.0785, mIoU=0.7081, F1=0.7968

当前轮次最佳结果: Round=12, Labeled=1157, mIoU=0.7255, F1=0.8131

## Round 13

Labeled Pool Size: 1245

- Epoch 1: Loss=0.1895, mIoU=0.5367, F1=0.5779
- Epoch 2: Loss=0.1124, mIoU=0.6307, F1=0.7126
- Epoch 3: Loss=0.1029, mIoU=0.6956, F1=0.7842
- Epoch 4: Loss=0.0950, mIoU=0.6494, F1=0.7346
- Epoch 5: Loss=0.0899, mIoU=0.6118, F1=0.6884
- Epoch 6: Loss=0.0867, mIoU=0.6907, F1=0.7794
- Epoch 7: Loss=0.0843, mIoU=0.6868, F1=0.7760
- Epoch 8: Loss=0.0797, mIoU=0.6685, F1=0.7558
- Epoch 9: Loss=0.0768, mIoU=0.6654, F1=0.7530
- Epoch 10: Loss=0.0763, mIoU=0.6685, F1=0.7558

当前轮次最佳结果: Round=13, Labeled=1245, mIoU=0.6956, F1=0.7842

## Round 14

Labeled Pool Size: 1333

- Epoch 1: Loss=0.2472, mIoU=0.4994, F1=0.5116
- Epoch 2: Loss=0.1147, mIoU=0.5071, F1=0.5258
- Epoch 3: Loss=0.1013, mIoU=0.6553, F1=0.7412
- Epoch 4: Loss=0.0967, mIoU=0.6884, F1=0.7768
- Epoch 5: Loss=0.0907, mIoU=0.6793, F1=0.7675
- Epoch 6: Loss=0.0862, mIoU=0.6945, F1=0.7832
- Epoch 7: Loss=0.0837, mIoU=0.6772, F1=0.7670
- Epoch 8: Loss=0.0805, mIoU=0.6976, F1=0.7867
- Epoch 9: Loss=0.0790, mIoU=0.7082, F1=0.7973
- Epoch 10: Loss=0.0756, mIoU=0.7010, F1=0.7908

当前轮次最佳结果: Round=14, Labeled=1333, mIoU=0.7082, F1=0.7973

## Round 15

Labeled Pool Size: 1421

- Epoch 1: Loss=0.1767, mIoU=0.5439, F1=0.5899
- Epoch 2: Loss=0.1033, mIoU=0.6692, F1=0.7565
- Epoch 3: Loss=0.0925, mIoU=0.6699, F1=0.7578
- Epoch 4: Loss=0.0863, mIoU=0.6824, F1=0.7720
- Epoch 5: Loss=0.0814, mIoU=0.6854, F1=0.7739
- Epoch 6: Loss=0.0780, mIoU=0.6869, F1=0.7755
- Epoch 7: Loss=0.0742, mIoU=0.6938, F1=0.7830
- Epoch 8: Loss=0.0724, mIoU=0.7075, F1=0.7964
- Epoch 9: Loss=0.0703, mIoU=0.7143, F1=0.8031
- Epoch 10: Loss=0.0678, mIoU=0.7126, F1=0.8012

当前轮次最佳结果: Round=15, Labeled=1421, mIoU=0.7143, F1=0.8031


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421]
ALC: 0.6038
最终 mIoU: 0.7143
最终 F1: 0.8031
