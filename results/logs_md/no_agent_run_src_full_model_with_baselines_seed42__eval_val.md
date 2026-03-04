# 实验日志

实验名称: no_agent
描述: 消融：移除Agent；λ由sigmoid自适应(随标注进度变化)，仅使用AD-KUCS数值策略选样
开始时间: 2026-02-27T01:20:38.894440

## Round 1

Labeled Pool Size: 1383

- Epoch 1: Loss=0.2155, mIoU=0.5163, F1=0.5425
- Epoch 2: Loss=0.0959, mIoU=0.5281, F1=0.5638
- Epoch 3: Loss=0.0829, mIoU=0.5620, F1=0.6184
- Epoch 4: Loss=0.0751, mIoU=0.6158, F1=0.6938
- Epoch 5: Loss=0.0705, mIoU=0.6674, F1=0.7558
- Epoch 6: Loss=0.0672, mIoU=0.6188, F1=0.6984
- Epoch 7: Loss=0.0655, mIoU=0.6749, F1=0.7638
- Epoch 8: Loss=0.0630, mIoU=0.6468, F1=0.7320
- Epoch 9: Loss=0.0618, mIoU=0.6539, F1=0.7413
- Epoch 10: Loss=0.0580, mIoU=0.6320, F1=0.7142

当前轮次最佳结果: Round=1, Labeled=1383, mIoU=0.6749, F1=0.7638

## Round 2

Labeled Pool Size: 1471

- Epoch 1: Loss=0.2043, mIoU=0.6131, F1=0.6899
- Epoch 2: Loss=0.0900, mIoU=0.6912, F1=0.7803
- Epoch 3: Loss=0.0756, mIoU=0.6891, F1=0.7780
- Epoch 4: Loss=0.0702, mIoU=0.6123, F1=0.6964
- Epoch 5: Loss=0.0675, mIoU=0.5371, F1=0.5788
- Epoch 6: Loss=0.0644, mIoU=0.6779, F1=0.7674
- Epoch 7: Loss=0.0624, mIoU=0.6504, F1=0.7390
- Epoch 8: Loss=0.0595, mIoU=0.6502, F1=0.7355
- Epoch 9: Loss=0.0573, mIoU=0.6902, F1=0.7792
- Epoch 10: Loss=0.0562, mIoU=0.7129, F1=0.8020

当前轮次最佳结果: Round=2, Labeled=1471, mIoU=0.7129, F1=0.8020

## Round 3

Labeled Pool Size: 1559

- Epoch 1: Loss=0.1487, mIoU=0.5115, F1=0.5340
- Epoch 2: Loss=0.0810, mIoU=0.6451, F1=0.7312
- Epoch 3: Loss=0.0697, mIoU=0.5938, F1=0.6644
- Epoch 4: Loss=0.0639, mIoU=0.6950, F1=0.7839
- Epoch 5: Loss=0.0627, mIoU=0.6855, F1=0.7745
- Epoch 6: Loss=0.0592, mIoU=0.7051, F1=0.7939
- Epoch 7: Loss=0.0566, mIoU=0.6276, F1=0.7084
- Epoch 8: Loss=0.0539, mIoU=0.7054, F1=0.7947
- Epoch 9: Loss=0.0518, mIoU=0.6926, F1=0.7828
- Epoch 10: Loss=0.0511, mIoU=0.7048, F1=0.7942

当前轮次最佳结果: Round=3, Labeled=1559, mIoU=0.7054, F1=0.7947

## Round 4

Labeled Pool Size: 1647

- Epoch 1: Loss=0.1695, mIoU=0.6095, F1=0.6854
- Epoch 2: Loss=0.0803, mIoU=0.6253, F1=0.7054
- Epoch 3: Loss=0.0685, mIoU=0.6710, F1=0.7604
- Epoch 4: Loss=0.0631, mIoU=0.6952, F1=0.7840
- Epoch 5: Loss=0.0594, mIoU=0.7128, F1=0.8016
- Epoch 6: Loss=0.0571, mIoU=0.6795, F1=0.7679
- Epoch 7: Loss=0.0561, mIoU=0.6404, F1=0.7243
- Epoch 8: Loss=0.0539, mIoU=0.6903, F1=0.7789
- Epoch 9: Loss=0.0527, mIoU=0.7081, F1=0.7968
- Epoch 10: Loss=0.0489, mIoU=0.7004, F1=0.7902

当前轮次最佳结果: Round=4, Labeled=1647, mIoU=0.7128, F1=0.8016

## Round 5

Labeled Pool Size: 1735

- Epoch 1: Loss=0.1668, mIoU=0.5295, F1=0.5658
- Epoch 2: Loss=0.0757, mIoU=0.6541, F1=0.7402
- Epoch 3: Loss=0.0663, mIoU=0.6914, F1=0.7800
- Epoch 4: Loss=0.0617, mIoU=0.5797, F1=0.6446
- Epoch 5: Loss=0.0593, mIoU=0.7092, F1=0.7980
- Epoch 6: Loss=0.0561, mIoU=0.6945, F1=0.7831
- Epoch 7: Loss=0.0539, mIoU=0.6779, F1=0.7677
- Epoch 8: Loss=0.0508, mIoU=0.7075, F1=0.7967
- Epoch 9: Loss=0.0495, mIoU=0.6512, F1=0.7375
- Epoch 10: Loss=0.0487, mIoU=0.6992, F1=0.7886

当前轮次最佳结果: Round=5, Labeled=1735, mIoU=0.7092, F1=0.7980

## Round 6

Labeled Pool Size: 1823

- Epoch 1: Loss=0.1568, mIoU=0.5346, F1=0.5744
- Epoch 2: Loss=0.0737, mIoU=0.5660, F1=0.6244
- Epoch 3: Loss=0.0632, mIoU=0.6901, F1=0.7789
- Epoch 4: Loss=0.0569, mIoU=0.6383, F1=0.7248
- Epoch 5: Loss=0.0528, mIoU=0.6769, F1=0.7651
- Epoch 6: Loss=0.0528, mIoU=0.5753, F1=0.6385
- Epoch 7: Loss=0.0509, mIoU=0.6878, F1=0.7770
- Epoch 8: Loss=0.0477, mIoU=0.6548, F1=0.7407
- Epoch 9: Loss=0.0462, mIoU=0.6792, F1=0.7684
- Epoch 10: Loss=0.0441, mIoU=0.7089, F1=0.7981

当前轮次最佳结果: Round=6, Labeled=1823, mIoU=0.7089, F1=0.7981

## Round 7

Labeled Pool Size: 1911

- Epoch 1: Loss=0.1818, mIoU=0.4960, F1=0.5045
- Epoch 2: Loss=0.0712, mIoU=0.6442, F1=0.7317
- Epoch 3: Loss=0.0621, mIoU=0.6777, F1=0.7672
- Epoch 4: Loss=0.0568, mIoU=0.6852, F1=0.7736
- Epoch 5: Loss=0.0550, mIoU=0.6796, F1=0.7676
- Epoch 6: Loss=0.0529, mIoU=0.7079, F1=0.7968
- Epoch 7: Loss=0.0509, mIoU=0.7065, F1=0.7952
- Epoch 8: Loss=0.0487, mIoU=0.6925, F1=0.7811
- Epoch 9: Loss=0.0469, mIoU=0.7264, F1=0.8142
- Epoch 10: Loss=0.0449, mIoU=0.7204, F1=0.8085

当前轮次最佳结果: Round=7, Labeled=1911, mIoU=0.7264, F1=0.8142

## Round 8

Labeled Pool Size: 1999

- Epoch 1: Loss=0.1570, mIoU=0.5393, F1=0.5822
- Epoch 2: Loss=0.0678, mIoU=0.6302, F1=0.7122
- Epoch 3: Loss=0.0580, mIoU=0.6854, F1=0.7737
- Epoch 4: Loss=0.0548, mIoU=0.6937, F1=0.7823
- Epoch 5: Loss=0.0512, mIoU=0.6940, F1=0.7834
- Epoch 6: Loss=0.0477, mIoU=0.6753, F1=0.7631
- Epoch 7: Loss=0.0468, mIoU=0.6756, F1=0.7640
- Epoch 8: Loss=0.0469, mIoU=0.6919, F1=0.7811
- Epoch 9: Loss=0.0429, mIoU=0.6891, F1=0.7777
- Epoch 10: Loss=0.0416, mIoU=0.7165, F1=0.8050

当前轮次最佳结果: Round=8, Labeled=1999, mIoU=0.7165, F1=0.8050

## Round 9

Labeled Pool Size: 2087

- Epoch 1: Loss=0.1880, mIoU=0.5168, F1=0.5436
- Epoch 2: Loss=0.0684, mIoU=0.6290, F1=0.7120
- Epoch 3: Loss=0.0586, mIoU=0.5437, F1=0.6167
- Epoch 4: Loss=0.0540, mIoU=0.6722, F1=0.7601
- Epoch 5: Loss=0.0508, mIoU=0.6688, F1=0.7569
- Epoch 6: Loss=0.0489, mIoU=0.5975, F1=0.6812
- Epoch 7: Loss=0.0463, mIoU=0.7107, F1=0.7998
- Epoch 8: Loss=0.0440, mIoU=0.6705, F1=0.7582
- Epoch 9: Loss=0.0421, mIoU=0.6795, F1=0.7678
- Epoch 10: Loss=0.0407, mIoU=0.5574, F1=0.6114

当前轮次最佳结果: Round=9, Labeled=2087, mIoU=0.7107, F1=0.7998

## Round 10

Labeled Pool Size: 2175

- Epoch 1: Loss=0.1610, mIoU=0.6308, F1=0.7123
- Epoch 2: Loss=0.0672, mIoU=0.5901, F1=0.6594
- Epoch 3: Loss=0.0585, mIoU=0.6540, F1=0.7414
- Epoch 4: Loss=0.0557, mIoU=0.6534, F1=0.7388
- Epoch 5: Loss=0.0519, mIoU=0.6497, F1=0.7346
- Epoch 6: Loss=0.0501, mIoU=0.5001, F1=0.5126
- Epoch 7: Loss=0.0481, mIoU=0.6933, F1=0.7825
- Epoch 8: Loss=0.0454, mIoU=0.6781, F1=0.7674
- Epoch 9: Loss=0.0441, mIoU=0.6576, F1=0.7439
- Epoch 10: Loss=0.0449, mIoU=0.6762, F1=0.7650

当前轮次最佳结果: Round=10, Labeled=2175, mIoU=0.6933, F1=0.7825

## Round 11

Labeled Pool Size: 2263

- Epoch 1: Loss=0.1223, mIoU=0.5048, F1=0.5215
- Epoch 2: Loss=0.0598, mIoU=0.6740, F1=0.7627
- Epoch 3: Loss=0.0520, mIoU=0.6065, F1=0.6892
- Epoch 4: Loss=0.0502, mIoU=0.6271, F1=0.7078
- Epoch 5: Loss=0.0469, mIoU=0.6981, F1=0.7879
- Epoch 6: Loss=0.0441, mIoU=0.6926, F1=0.7811
- Epoch 7: Loss=0.0426, mIoU=0.7205, F1=0.8086
- Epoch 8: Loss=0.0400, mIoU=0.5777, F1=0.6418
- Epoch 9: Loss=0.0383, mIoU=0.6227, F1=0.7025
- Epoch 10: Loss=0.0398, mIoU=0.7132, F1=0.8017

当前轮次最佳结果: Round=11, Labeled=2263, mIoU=0.7205, F1=0.8086

## Round 12

Labeled Pool Size: 2351

- Epoch 1: Loss=0.1154, mIoU=0.5813, F1=0.6470
- Epoch 2: Loss=0.0582, mIoU=0.5322, F1=0.5705
- Epoch 3: Loss=0.0503, mIoU=0.5635, F1=0.6416
- Epoch 4: Loss=0.0474, mIoU=0.6778, F1=0.7674
- Epoch 5: Loss=0.0444, mIoU=0.6594, F1=0.7463
- Epoch 6: Loss=0.0428, mIoU=0.7093, F1=0.7980
- Epoch 7: Loss=0.0411, mIoU=0.7002, F1=0.7889
- Epoch 8: Loss=0.0407, mIoU=0.6956, F1=0.7844
- Epoch 9: Loss=0.0388, mIoU=0.6715, F1=0.7608
- Epoch 10: Loss=0.0364, mIoU=0.7167, F1=0.8050

当前轮次最佳结果: Round=12, Labeled=2351, mIoU=0.7167, F1=0.8050

## Round 13

Labeled Pool Size: 2439

- Epoch 1: Loss=0.1090, mIoU=0.6394, F1=0.7230
- Epoch 2: Loss=0.0554, mIoU=0.6656, F1=0.7529
- Epoch 3: Loss=0.0491, mIoU=0.5722, F1=0.6340
- Epoch 4: Loss=0.0461, mIoU=0.6911, F1=0.7796
- Epoch 5: Loss=0.0429, mIoU=0.6957, F1=0.7847
- Epoch 6: Loss=0.0408, mIoU=0.5579, F1=0.6119
- Epoch 7: Loss=0.0389, mIoU=0.6820, F1=0.7715
- Epoch 8: Loss=0.0371, mIoU=0.6861, F1=0.7745
- Epoch 9: Loss=0.0363, mIoU=0.6126, F1=0.6896
- Epoch 10: Loss=0.0357, mIoU=0.6547, F1=0.7413

当前轮次最佳结果: Round=13, Labeled=2439, mIoU=0.6957, F1=0.7847

## Round 14

Labeled Pool Size: 2527

- Epoch 1: Loss=0.1481, mIoU=0.5333, F1=0.5726
- Epoch 2: Loss=0.0566, mIoU=0.5397, F1=0.5831
- Epoch 3: Loss=0.0498, mIoU=0.6340, F1=0.7163
- Epoch 4: Loss=0.0464, mIoU=0.6160, F1=0.6939
- Epoch 5: Loss=0.0432, mIoU=0.6733, F1=0.7610
- Epoch 6: Loss=0.0421, mIoU=0.6892, F1=0.7782
- Epoch 7: Loss=0.0400, mIoU=0.5569, F1=0.6113
- Epoch 8: Loss=0.0394, mIoU=0.5849, F1=0.6523
- Epoch 9: Loss=0.0370, mIoU=0.7125, F1=0.8015
- Epoch 10: Loss=0.0392, mIoU=0.7024, F1=0.7913

当前轮次最佳结果: Round=14, Labeled=2527, mIoU=0.7125, F1=0.8015

## Round 15

Labeled Pool Size: 2615

- Epoch 1: Loss=0.1052, mIoU=0.6533, F1=0.7390
- Epoch 2: Loss=0.0514, mIoU=0.6806, F1=0.7689
- Epoch 3: Loss=0.0465, mIoU=0.5462, F1=0.5936
- Epoch 4: Loss=0.0434, mIoU=0.6341, F1=0.7214
- Epoch 5: Loss=0.0406, mIoU=0.6845, F1=0.7734
- Epoch 6: Loss=0.0381, mIoU=0.6989, F1=0.7878
- Epoch 7: Loss=0.0374, mIoU=0.6155, F1=0.6934
- Epoch 8: Loss=0.0354, mIoU=0.7181, F1=0.8064
- Epoch 9: Loss=0.0344, mIoU=0.6860, F1=0.7747
- Epoch 10: Loss=0.0338, mIoU=0.6439, F1=0.7282

当前轮次最佳结果: Round=15, Labeled=2615, mIoU=0.7181, F1=0.8064


## 实验汇总

预算历史: [1383, 1471, 1559, 1647, 1735, 1823, 1911, 1999, 2087, 2175, 2263, 2351, 2439, 2527, 2615]
ALC: 0.0626
最终 mIoU: 0.7181
最终 F1: 0.8064
