# 实验日志

实验名称: fixed_lambda
描述: 消融：Fixed λ=0.5；保留Agent，隔离三阶段λ policy贡献
开始时间: 2026-03-24T05:56:56.009928

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.5166, mIoU=0.5027, F1=0.5219
- Epoch 2: Loss=0.2797, mIoU=0.5320, F1=0.5728
- Epoch 3: Loss=0.1736, mIoU=0.5443, F1=0.5943
- Epoch 4: Loss=0.1287, mIoU=0.5669, F1=0.6310
- Epoch 5: Loss=0.1047, mIoU=0.5465, F1=0.5959
- Epoch 6: Loss=0.0935, mIoU=0.5316, F1=0.5711
- Epoch 7: Loss=0.0794, mIoU=0.5248, F1=0.5578
- Epoch 8: Loss=0.0744, mIoU=0.5617, F1=0.6189
- Epoch 9: Loss=0.0718, mIoU=0.5408, F1=0.5857
- Epoch 10: Loss=0.0636, mIoU=0.5459, F1=0.5932

本轮结果: Round=1, Labeled=189, Selection=best_val (epoch=4), mIoU=0.5669, F1=0.6310, peak_mIoU=0.5669

Round=1, Labeled=189, mIoU=0.5669, F1=0.6310

## Round 2

Labeled Pool Size: 277

- Epoch 1: Loss=0.3326, mIoU=0.5066, F1=0.5254
- Epoch 2: Loss=0.1875, mIoU=0.5628, F1=0.6205
- Epoch 3: Loss=0.1441, mIoU=0.5153, F1=0.5416
- Epoch 4: Loss=0.1205, mIoU=0.5371, F1=0.5797
- Epoch 5: Loss=0.1133, mIoU=0.5458, F1=0.5931
- Epoch 6: Loss=0.1040, mIoU=0.5859, F1=0.6536
- Epoch 7: Loss=0.0987, mIoU=0.5517, F1=0.6032
- Epoch 8: Loss=0.0952, mIoU=0.5368, F1=0.5784
- Epoch 9: Loss=0.0869, mIoU=0.6114, F1=0.6886
- Epoch 10: Loss=0.0847, mIoU=0.5333, F1=0.5724

本轮结果: Round=2, Labeled=277, Selection=best_val (epoch=9), mIoU=0.6114, F1=0.6886, peak_mIoU=0.6114

Round=2, Labeled=277, mIoU=0.6114, F1=0.6886

## Round 3

Labeled Pool Size: 365

- Epoch 1: Loss=0.4022, mIoU=0.5300, F1=0.5675
- Epoch 2: Loss=0.2096, mIoU=0.5227, F1=0.5541
- Epoch 3: Loss=0.1644, mIoU=0.5058, F1=0.5233
- Epoch 4: Loss=0.1472, mIoU=0.5798, F1=0.6447
- Epoch 5: Loss=0.1302, mIoU=0.5594, F1=0.6144
- Epoch 6: Loss=0.1192, mIoU=0.6384, F1=0.7215
- Epoch 7: Loss=0.1161, mIoU=0.5846, F1=0.6517
- Epoch 8: Loss=0.1109, mIoU=0.5934, F1=0.6642
- Epoch 9: Loss=0.1094, mIoU=0.6052, F1=0.6799
- Epoch 10: Loss=0.1063, mIoU=0.6252, F1=0.7055

本轮结果: Round=3, Labeled=365, Selection=best_val (epoch=6), mIoU=0.6384, F1=0.7215, peak_mIoU=0.6384

Round=3, Labeled=365, mIoU=0.6384, F1=0.7215

## Round 4

Labeled Pool Size: 453

- Epoch 1: Loss=0.3622, mIoU=0.5110, F1=0.5372
- Epoch 2: Loss=0.1733, mIoU=0.5598, F1=0.6151
- Epoch 3: Loss=0.1341, mIoU=0.5775, F1=0.6413
- Epoch 4: Loss=0.1265, mIoU=0.5899, F1=0.6588
- Epoch 5: Loss=0.1139, mIoU=0.5877, F1=0.6564
- Epoch 6: Loss=0.1064, mIoU=0.5579, F1=0.6121
- Epoch 7: Loss=0.1023, mIoU=0.6807, F1=0.7700
- Epoch 8: Loss=0.1002, mIoU=0.6364, F1=0.7196
- Epoch 9: Loss=0.0949, mIoU=0.6557, F1=0.7416
- Epoch 10: Loss=0.0906, mIoU=0.6441, F1=0.7308

本轮结果: Round=4, Labeled=453, Selection=best_val (epoch=7), mIoU=0.6807, F1=0.7700, peak_mIoU=0.6807

Round=4, Labeled=453, mIoU=0.6807, F1=0.7700

## Round 5

Labeled Pool Size: 541

- Epoch 1: Loss=0.3383, mIoU=0.5629, F1=0.6197
- Epoch 2: Loss=0.1667, mIoU=0.5510, F1=0.6012
- Epoch 3: Loss=0.1326, mIoU=0.6111, F1=0.6874
- Epoch 4: Loss=0.1218, mIoU=0.5661, F1=0.6246
- Epoch 5: Loss=0.1121, mIoU=0.6652, F1=0.7523
- Epoch 6: Loss=0.1045, mIoU=0.6549, F1=0.7411
- Epoch 7: Loss=0.1024, mIoU=0.6822, F1=0.7713
- Epoch 8: Loss=0.0978, mIoU=0.6763, F1=0.7652
- Epoch 9: Loss=0.0930, mIoU=0.6834, F1=0.7723
- Epoch 10: Loss=0.0896, mIoU=0.6909, F1=0.7799

本轮结果: Round=5, Labeled=541, Selection=best_val (epoch=10), mIoU=0.6909, F1=0.7799, peak_mIoU=0.6909

Round=5, Labeled=541, mIoU=0.6909, F1=0.7799

## Round 6

Labeled Pool Size: 629

- Epoch 1: Loss=0.3872, mIoU=0.5771, F1=0.6410
- Epoch 2: Loss=0.1738, mIoU=0.5782, F1=0.6425
- Epoch 3: Loss=0.1411, mIoU=0.6112, F1=0.6877
- Epoch 4: Loss=0.1256, mIoU=0.6439, F1=0.7279
- Epoch 5: Loss=0.1184, mIoU=0.6002, F1=0.6733
- Epoch 6: Loss=0.1129, mIoU=0.5761, F1=0.6398
- Epoch 7: Loss=0.1056, mIoU=0.6608, F1=0.7474
- Epoch 8: Loss=0.0990, mIoU=0.6640, F1=0.7511
- Epoch 9: Loss=0.0974, mIoU=0.6911, F1=0.7797
- Epoch 10: Loss=0.0950, mIoU=0.6884, F1=0.7776

本轮结果: Round=6, Labeled=629, Selection=best_val (epoch=9), mIoU=0.6911, F1=0.7797, peak_mIoU=0.6911

Round=6, Labeled=629, mIoU=0.6911, F1=0.7797

## Round 7

Labeled Pool Size: 717

- Epoch 1: Loss=0.3345, mIoU=0.5188, F1=0.5477
- Epoch 2: Loss=0.1563, mIoU=0.5325, F1=0.5710
- Epoch 3: Loss=0.1317, mIoU=0.6030, F1=0.6769
- Epoch 4: Loss=0.1195, mIoU=0.6022, F1=0.6759
- Epoch 5: Loss=0.1140, mIoU=0.6103, F1=0.6862
- Epoch 6: Loss=0.1066, mIoU=0.6676, F1=0.7553
- Epoch 7: Loss=0.1046, mIoU=0.6548, F1=0.7406
- Epoch 8: Loss=0.1002, mIoU=0.6740, F1=0.7624
- Epoch 9: Loss=0.0922, mIoU=0.7139, F1=0.8023
- Epoch 10: Loss=0.0913, mIoU=0.7126, F1=0.8013

本轮结果: Round=7, Labeled=717, Selection=best_val (epoch=9), mIoU=0.7139, F1=0.8023, peak_mIoU=0.7139

Round=7, Labeled=717, mIoU=0.7139, F1=0.8023

## Round 8

Labeled Pool Size: 805

- Epoch 1: Loss=0.3402, mIoU=0.5319, F1=0.5700
- Epoch 2: Loss=0.1581, mIoU=0.6245, F1=0.7045
- Epoch 3: Loss=0.1331, mIoU=0.5995, F1=0.6724
- Epoch 4: Loss=0.1227, mIoU=0.6395, F1=0.7227
- Epoch 5: Loss=0.1139, mIoU=0.6744, F1=0.7623
- Epoch 6: Loss=0.1090, mIoU=0.6477, F1=0.7329
- Epoch 7: Loss=0.1025, mIoU=0.6815, F1=0.7707
- Epoch 8: Loss=0.0992, mIoU=0.6534, F1=0.7394
- Epoch 9: Loss=0.0974, mIoU=0.6327, F1=0.7151
- Epoch 10: Loss=0.0938, mIoU=0.6644, F1=0.7516

本轮结果: Round=8, Labeled=805, Selection=best_val (epoch=7), mIoU=0.6815, F1=0.7707, peak_mIoU=0.6815

Round=8, Labeled=805, mIoU=0.6815, F1=0.7707

## Round 9

Labeled Pool Size: 893

- Epoch 1: Loss=0.3465, mIoU=0.5498, F1=0.5995
- Epoch 2: Loss=0.1494, mIoU=0.5746, F1=0.6374
- Epoch 3: Loss=0.1251, mIoU=0.6096, F1=0.6932
- Epoch 4: Loss=0.1141, mIoU=0.6880, F1=0.7765
- Epoch 5: Loss=0.1089, mIoU=0.6682, F1=0.7556
- Epoch 6: Loss=0.1026, mIoU=0.6797, F1=0.7679
- Epoch 7: Loss=0.0952, mIoU=0.6771, F1=0.7653
- Epoch 8: Loss=0.0925, mIoU=0.6608, F1=0.7477
- Epoch 9: Loss=0.0896, mIoU=0.6921, F1=0.7808
- Epoch 10: Loss=0.0859, mIoU=0.6602, F1=0.7466

本轮结果: Round=9, Labeled=893, Selection=best_val (epoch=9), mIoU=0.6921, F1=0.7808, peak_mIoU=0.6921

Round=9, Labeled=893, mIoU=0.6921, F1=0.7808

## Round 10

Labeled Pool Size: 981

- Epoch 1: Loss=0.2453, mIoU=0.5654, F1=0.6239
- Epoch 2: Loss=0.1330, mIoU=0.5300, F1=0.5668
- Epoch 3: Loss=0.1148, mIoU=0.6007, F1=0.6740
- Epoch 4: Loss=0.1051, mIoU=0.5892, F1=0.6583
- Epoch 5: Loss=0.0986, mIoU=0.5764, F1=0.6399
- Epoch 6: Loss=0.0966, mIoU=0.6868, F1=0.7756
- Epoch 7: Loss=0.0939, mIoU=0.6684, F1=0.7559
- Epoch 8: Loss=0.0871, mIoU=0.6877, F1=0.7765
- Epoch 9: Loss=0.0857, mIoU=0.7016, F1=0.7909
- Epoch 10: Loss=0.0843, mIoU=0.6828, F1=0.7715

本轮结果: Round=10, Labeled=981, Selection=best_val (epoch=9), mIoU=0.7016, F1=0.7909, peak_mIoU=0.7016

Round=10, Labeled=981, mIoU=0.7016, F1=0.7909

## Round 11

Labeled Pool Size: 1069

- Epoch 1: Loss=0.2187, mIoU=0.5237, F1=0.5561
- Epoch 2: Loss=0.1234, mIoU=0.5653, F1=0.6235
- Epoch 3: Loss=0.1073, mIoU=0.5517, F1=0.6025
- Epoch 4: Loss=0.0987, mIoU=0.6698, F1=0.7575
- Epoch 5: Loss=0.0928, mIoU=0.6546, F1=0.7404
- Epoch 6: Loss=0.0895, mIoU=0.6944, F1=0.7833
- Epoch 7: Loss=0.0864, mIoU=0.6521, F1=0.7387
- Epoch 8: Loss=0.0808, mIoU=0.6787, F1=0.7677
- Epoch 9: Loss=0.0795, mIoU=0.6750, F1=0.7634

--- [Checkpoint] 续跑开始时间: 2026-03-24T09:39:45.702638 ---

## Round 11

Labeled Pool Size: 1069

- Epoch 10: Loss=0.0765, mIoU=0.7035, F1=0.7924

本轮结果: Round=11, Labeled=1069, Selection=best_val (epoch=10), mIoU=0.7035, F1=0.7924, peak_mIoU=0.7035

Round=11, Labeled=1069, mIoU=0.7035, F1=0.7924

- Epoch 1: Loss=0.2189, mIoU=0.5542, F1=0.6065
- Epoch 2: Loss=0.1227, mIoU=0.6267, F1=0.7072
- Epoch 3: Loss=0.1087, mIoU=0.6313, F1=0.7132
- Epoch 4: Loss=0.0990, mIoU=0.6287, F1=0.7098
## Round 12

Labeled Pool Size: 1157

- Epoch 5: Loss=0.0930, mIoU=0.5890, F1=0.6577
- Epoch 1: Loss=0.2065, mIoU=0.5326, F1=0.5712
- Epoch 6: Loss=0.0888, mIoU=0.6965, F1=0.7861
- Epoch 2: Loss=0.1175, mIoU=0.5171, F1=0.5441
- Epoch 7: Loss=0.0859, mIoU=0.6632, F1=0.7513
- Epoch 3: Loss=0.1011, mIoU=0.6538, F1=0.7406
- Epoch 8: Loss=0.0813, mIoU=0.6860, F1=0.7755
- Epoch 4: Loss=0.0933, mIoU=0.6655, F1=0.7528
- Epoch 9: Loss=0.0803, mIoU=0.6719, F1=0.7602
- Epoch 5: Loss=0.0882, mIoU=0.7085, F1=0.7974
- Epoch 10: Loss=0.0771, mIoU=0.6756, F1=0.7637

本轮结果: Round=11, Labeled=1069, Selection=best_val (epoch=6), mIoU=0.6965, F1=0.7861, peak_mIoU=0.6965

Round=11, Labeled=1069, mIoU=0.6965, F1=0.7861

- Epoch 6: Loss=0.0838, mIoU=0.7022, F1=0.7915
- Epoch 7: Loss=0.0787, mIoU=0.6535, F1=0.7395
- Epoch 8: Loss=0.0770, mIoU=0.6868, F1=0.7757
- Epoch 9: Loss=0.0737, mIoU=0.6446, F1=0.7294
- Epoch 10: Loss=0.0739, mIoU=0.6917, F1=0.7806

本轮结果: Round=12, Labeled=1157, Selection=best_val (epoch=5), mIoU=0.7085, F1=0.7974, peak_mIoU=0.7085

Round=12, Labeled=1157, mIoU=0.7085, F1=0.7974

## Round 13

Labeled Pool Size: 1245

- Epoch 1: Loss=0.2427, mIoU=0.5347, F1=0.5747
## Round 12

Labeled Pool Size: 1157

- Epoch 2: Loss=0.1130, mIoU=0.6414, F1=0.7252
- Epoch 1: Loss=0.2093, mIoU=0.6028, F1=0.6773
- Epoch 2: Loss=0.1152, mIoU=0.6136, F1=0.6910
- Epoch 3: Loss=0.0999, mIoU=0.5711, F1=0.6319
- Epoch 3: Loss=0.0994, mIoU=0.6089, F1=0.6850
- Epoch 4: Loss=0.0919, mIoU=0.5852, F1=0.6525
- Epoch 4: Loss=0.0962, mIoU=0.6530, F1=0.7407
- Epoch 5: Loss=0.0858, mIoU=0.5967, F1=0.6683
- Epoch 5: Loss=0.0903, mIoU=0.6816, F1=0.7703
- Epoch 6: Loss=0.0811, mIoU=0.5709, F1=0.6318
- Epoch 6: Loss=0.0836, mIoU=0.6774, F1=0.7659
- Epoch 7: Loss=0.0744, mIoU=0.7116, F1=0.8004
- Epoch 7: Loss=0.0803, mIoU=0.6858, F1=0.7742
- Epoch 8: Loss=0.0741, mIoU=0.6800, F1=0.7699
- Epoch 8: Loss=0.0782, mIoU=0.6900, F1=0.7792
- Epoch 9: Loss=0.0703, mIoU=0.6932, F1=0.7824
- Epoch 9: Loss=0.0738, mIoU=0.6842, F1=0.7738
- Epoch 10: Loss=0.0689, mIoU=0.6614, F1=0.7513

本轮结果: Round=13, Labeled=1245, Selection=best_val (epoch=7), mIoU=0.7116, F1=0.8004, peak_mIoU=0.7116

Round=13, Labeled=1245, mIoU=0.7116, F1=0.8004

- Epoch 10: Loss=0.0721, mIoU=0.7023, F1=0.7915

本轮结果: Round=12, Labeled=1157, Selection=best_val (epoch=10), mIoU=0.7023, F1=0.7915, peak_mIoU=0.7023

Round=12, Labeled=1157, mIoU=0.7023, F1=0.7915

## Round 14

Labeled Pool Size: 1333

## Round 13

Labeled Pool Size: 1245

- Epoch 1: Loss=0.1749, mIoU=0.5529, F1=0.6045
- Epoch 1: Loss=0.2427, mIoU=0.5601, F1=0.6156
- Epoch 2: Loss=0.0992, mIoU=0.5357, F1=0.5764
- Epoch 2: Loss=0.1153, mIoU=0.6099, F1=0.6860
- Epoch 3: Loss=0.0893, mIoU=0.5834, F1=0.6500
- Epoch 3: Loss=0.0997, mIoU=0.5766, F1=0.6400
- Epoch 4: Loss=0.0811, mIoU=0.5719, F1=0.6333
- Epoch 4: Loss=0.0931, mIoU=0.6919, F1=0.7811
- Epoch 5: Loss=0.0768, mIoU=0.6856, F1=0.7742
- Epoch 5: Loss=0.0868, mIoU=0.6703, F1=0.7580
- Epoch 6: Loss=0.0748, mIoU=0.6219, F1=0.7013
- Epoch 6: Loss=0.0832, mIoU=0.6056, F1=0.6808
- Epoch 7: Loss=0.0693, mIoU=0.7130, F1=0.8017
- Epoch 7: Loss=0.0797, mIoU=0.6784, F1=0.7668
- Epoch 8: Loss=0.0668, mIoU=0.6994, F1=0.7885
- Epoch 8: Loss=0.0794, mIoU=0.7218, F1=0.8101
- Epoch 9: Loss=0.0751, mIoU=0.6818, F1=0.7707
- Epoch 9: Loss=0.0657, mIoU=0.6894, F1=0.7780
- Epoch 10: Loss=0.0704, mIoU=0.6875, F1=0.7765

本轮结果: Round=13, Labeled=1245, Selection=best_val (epoch=8), mIoU=0.7218, F1=0.8101, peak_mIoU=0.7218

Round=13, Labeled=1245, mIoU=0.7218, F1=0.8101

- Epoch 10: Loss=0.0626, mIoU=0.7081, F1=0.7967

本轮结果: Round=14, Labeled=1333, Selection=best_val (epoch=7), mIoU=0.7130, F1=0.8017, peak_mIoU=0.7130

Round=14, Labeled=1333, mIoU=0.7130, F1=0.8017

## Round 15

Labeled Pool Size: 1421

## Round 14

Labeled Pool Size: 1333

- Epoch 1: Loss=0.2209, mIoU=0.5446, F1=0.5917
- Epoch 1: Loss=0.1768, mIoU=0.5422, F1=0.5872
- Epoch 2: Loss=0.1016, mIoU=0.5551, F1=0.6079
- Epoch 2: Loss=0.1029, mIoU=0.5109, F1=0.5329
- Epoch 3: Loss=0.0876, mIoU=0.6348, F1=0.7170
- Epoch 3: Loss=0.0890, mIoU=0.6763, F1=0.7641
- Epoch 4: Loss=0.0816, mIoU=0.6409, F1=0.7250
- Epoch 4: Loss=0.0827, mIoU=0.6475, F1=0.7322
- Epoch 5: Loss=0.0757, mIoU=0.5896, F1=0.6590
- Epoch 5: Loss=0.0785, mIoU=0.6280, F1=0.7088
- Epoch 6: Loss=0.0713, mIoU=0.7041, F1=0.7932
- Epoch 6: Loss=0.0757, mIoU=0.7075, F1=0.7962
- Epoch 7: Loss=0.0691, mIoU=0.6651, F1=0.7544
- Epoch 7: Loss=0.0702, mIoU=0.7095, F1=0.7982
- Epoch 8: Loss=0.0690, mIoU=0.6671, F1=0.7560
- Epoch 8: Loss=0.0678, mIoU=0.7058, F1=0.7947
- Epoch 9: Loss=0.0652, mIoU=0.7019, F1=0.7911
- Epoch 9: Loss=0.0650, mIoU=0.6753, F1=0.7638
- Epoch 10: Loss=0.0630, mIoU=0.6875, F1=0.7772

本轮结果: Round=15, Labeled=1421, Selection=best_val (epoch=6), mIoU=0.7041, F1=0.7932, peak_mIoU=0.7041

Round=15, Labeled=1421, mIoU=0.7041, F1=0.7932

- Epoch 10: Loss=0.0629, mIoU=0.6409, F1=0.7246

本轮结果: Round=14, Labeled=1333, Selection=best_val (epoch=7), mIoU=0.7095, F1=0.7982, peak_mIoU=0.7095

Round=14, Labeled=1333, mIoU=0.7095, F1=0.7982

## Round 16

Labeled Pool Size: 1509


**[ERROR] Round 16 失败: Missing previous round best checkpoint for final test-only round: results/runs/ablation_matrix_p3_20260323_235540_seed42/fixed_lambda_round_models/round_15_best_val.pt**

## Round 15

Labeled Pool Size: 1421

- Epoch 1: Loss=0.2164, mIoU=0.5428, F1=0.5881
- Epoch 2: Loss=0.1010, mIoU=0.5784, F1=0.6429
- Epoch 3: Loss=0.0888, mIoU=0.5790, F1=0.6436
- Epoch 4: Loss=0.0829, mIoU=0.6665, F1=0.7557
- Epoch 5: Loss=0.0765, mIoU=0.6516, F1=0.7374
- Epoch 6: Loss=0.0724, mIoU=0.6447, F1=0.7304
- Epoch 7: Loss=0.0703, mIoU=0.7030, F1=0.7920
- Epoch 8: Loss=0.0678, mIoU=0.6812, F1=0.7695
- Epoch 9: Loss=0.0646, mIoU=0.6369, F1=0.7234
- Epoch 10: Loss=0.0620, mIoU=0.6225, F1=0.7024

本轮结果: Round=15, Labeled=1421, Selection=best_val (epoch=7), mIoU=0.7030, F1=0.7920, peak_mIoU=0.7030

Round=15, Labeled=1421, mIoU=0.7030, F1=0.7920

## Round 16

Labeled Pool Size: 1509

- Test-only mode: Round 16 loads Round 15 best_val checkpoint (epoch=7, val_mIoU=0.7030, val_F1=0.7920)

本轮结果: Round=16, Labeled=1509, Selection=prev_round_best_val (source_round=15, epoch=7), mIoU=0.7030, F1=0.7920, peak_mIoU=0.7030

Round=16, Labeled=1509, mIoU=0.7030, F1=0.7920


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421, 1509]
ALC(基于每轮选模val mIoU): 0.5996
最后一轮选模 mIoU(val): 0.7029571224586069
最后一轮选模 F1(val): 0.7920276925550707
最终报告 mIoU(test): 0.7046264871955433
最终报告 F1(test): 0.7940974748508756
最终输出 mIoU: 0.7046 (source=final_report)
最终输出 F1: 0.7941 (source=final_report)
最终 Test Split: test
最终 Report: {'loss': 0.047807240639813245, 'mIoU': 0.7046264871955433, 'f1_score': 0.7940974748508756}
