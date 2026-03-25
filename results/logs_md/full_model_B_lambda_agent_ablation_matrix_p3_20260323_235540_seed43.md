# 实验日志

实验名称: full_model_B_lambda_agent
描述: 对照B / AAL-SD (Agent-λ)：与 full_model_A_lambda_policy 完全一致，唯一区别：λ 由 LLM Agent 通过 set_lambda 显式决定，而非 policy 自动填充
开始时间: 2026-03-24T14:27:04.299465

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.3464, mIoU=0.5058, F1=0.5249
- Epoch 2: Loss=0.1949, mIoU=0.5366, F1=0.5865
- Epoch 3: Loss=0.1349, mIoU=0.5495, F1=0.6051
- Epoch 4: Loss=0.1043, mIoU=0.5481, F1=0.6042
- Epoch 5: Loss=0.0881, mIoU=0.5810, F1=0.6515
- Epoch 6: Loss=0.0777, mIoU=0.6030, F1=0.6785
## Round 10

Labeled Pool Size: 981

- Epoch 7: Loss=0.0712, mIoU=0.5893, F1=0.6616
- Epoch 8: Loss=0.0672, mIoU=0.5859, F1=0.6537
- Epoch 9: Loss=0.0590, mIoU=0.6093, F1=0.6862
- Epoch 10: Loss=0.0555, mIoU=0.5596, F1=0.6151

本轮结果: Round=1, Labeled=189, Selection=best_val (epoch=9), mIoU=0.6093, F1=0.6862, peak_mIoU=0.6093

Round=1, Labeled=189, mIoU=0.6093, F1=0.6862

- Epoch 1: Loss=0.2275, mIoU=0.5404, F1=0.5844
- Epoch 2: Loss=0.1307, mIoU=0.5514, F1=0.6018
- Epoch 3: Loss=0.1142, mIoU=0.6515, F1=0.7367
- Epoch 4: Loss=0.1041, mIoU=0.6517, F1=0.7389
- Epoch 5: Loss=0.1008, mIoU=0.6665, F1=0.7556
- Epoch 6: Loss=0.0943, mIoU=0.6054, F1=0.6805
- Epoch 7: Loss=0.0926, mIoU=0.6088, F1=0.6851
- Epoch 8: Loss=0.0885, mIoU=0.5608, F1=0.6167
- Epoch 9: Loss=0.0867, mIoU=0.6766, F1=0.7649
- Epoch 10: Loss=0.0829, mIoU=0.6728, F1=0.7621

本轮结果: Round=10, Labeled=981, Selection=best_val (epoch=9), mIoU=0.6766, F1=0.7649, peak_mIoU=0.6766

Round=10, Labeled=981, mIoU=0.6766, F1=0.7649

## Round 2

Labeled Pool Size: 277

- Epoch 1: Loss=0.4266, mIoU=0.4969, F1=0.5070
- Epoch 2: Loss=0.2220, mIoU=0.4962, F1=0.5051
- Epoch 3: Loss=0.1552, mIoU=0.5046, F1=0.5213
- Epoch 4: Loss=0.1261, mIoU=0.5273, F1=0.5622
- Epoch 5: Loss=0.1122, mIoU=0.5388, F1=0.5815
- Epoch 6: Loss=0.1005, mIoU=0.5545, F1=0.6071
- Epoch 7: Loss=0.0881, mIoU=0.5082, F1=0.5280
- Epoch 8: Loss=0.0866, mIoU=0.5569, F1=0.6107
- Epoch 9: Loss=0.0890, mIoU=0.5030, F1=0.5182
- Epoch 10: Loss=0.0823, mIoU=0.5544, F1=0.6067

本轮结果: Round=2, Labeled=277, Selection=best_val (epoch=8), mIoU=0.5569, F1=0.6107, peak_mIoU=0.5569

Round=2, Labeled=277, mIoU=0.5569, F1=0.6107

## Round 3

Labeled Pool Size: 365

- Epoch 1: Loss=0.4025, mIoU=0.5346, F1=0.5769
- Epoch 2: Loss=0.2047, mIoU=0.5417, F1=0.5864
- Epoch 3: Loss=0.1564, mIoU=0.5584, F1=0.6130
- Epoch 4: Loss=0.1329, mIoU=0.5425, F1=0.5877
- Epoch 5: Loss=0.1227, mIoU=0.5457, F1=0.5928
- Epoch 6: Loss=0.1230, mIoU=0.6005, F1=0.6739
- Epoch 7: Loss=0.1139, mIoU=0.6259, F1=0.7064
- Epoch 8: Loss=0.1060, mIoU=0.5944, F1=0.6654
- Epoch 9: Loss=0.1002, mIoU=0.6331, F1=0.7156
- Epoch 10: Loss=0.0941, mIoU=0.6578, F1=0.7442

本轮结果: Round=3, Labeled=365, Selection=best_val (epoch=10), mIoU=0.6578, F1=0.7442, peak_mIoU=0.6578

Round=3, Labeled=365, mIoU=0.6578, F1=0.7442

## Round 11

Labeled Pool Size: 1069

## Round 4

Labeled Pool Size: 453

- Epoch 1: Loss=0.3709, mIoU=0.5247, F1=0.5582
- Epoch 1: Loss=0.2206, mIoU=0.5218, F1=0.5526
- Epoch 2: Loss=0.1936, mIoU=0.5796, F1=0.6451
- Epoch 3: Loss=0.1559, mIoU=0.5374, F1=0.5794
- Epoch 2: Loss=0.1231, mIoU=0.6266, F1=0.7075
- Epoch 4: Loss=0.1401, mIoU=0.5841, F1=0.6516
- Epoch 5: Loss=0.1280, mIoU=0.5608, F1=0.6167
- Epoch 3: Loss=0.1103, mIoU=0.6333, F1=0.7155
- Epoch 6: Loss=0.1251, mIoU=0.6278, F1=0.7085
- Epoch 7: Loss=0.1156, mIoU=0.6589, F1=0.7452
- Epoch 4: Loss=0.1009, mIoU=0.6703, F1=0.7584
- Epoch 8: Loss=0.1164, mIoU=0.5838, F1=0.6504
- Epoch 9: Loss=0.1092, mIoU=0.6626, F1=0.7496
- Epoch 10: Loss=0.1042, mIoU=0.5776, F1=0.6419

本轮结果: Round=4, Labeled=453, Selection=best_val (epoch=9), mIoU=0.6626, F1=0.7496, peak_mIoU=0.6626

Round=4, Labeled=453, mIoU=0.6626, F1=0.7496

- Epoch 5: Loss=0.0969, mIoU=0.6672, F1=0.7543
- Epoch 6: Loss=0.0905, mIoU=0.7018, F1=0.7905
- Epoch 7: Loss=0.0886, mIoU=0.6462, F1=0.7309
- Epoch 8: Loss=0.0865, mIoU=0.6948, F1=0.7834
- Epoch 9: Loss=0.0810, mIoU=0.6731, F1=0.7612
- Epoch 10: Loss=0.0791, mIoU=0.6422, F1=0.7262

本轮结果: Round=11, Labeled=1069, Selection=best_val (epoch=6), mIoU=0.7018, F1=0.7905, peak_mIoU=0.7018

Round=11, Labeled=1069, mIoU=0.7018, F1=0.7905

## Round 5

Labeled Pool Size: 541

- Epoch 1: Loss=0.3979, mIoU=0.5286, F1=0.5647
- Epoch 2: Loss=0.1780, mIoU=0.5328, F1=0.5716
- Epoch 3: Loss=0.1471, mIoU=0.5161, F1=0.5423
- Epoch 4: Loss=0.1374, mIoU=0.6242, F1=0.7061
- Epoch 5: Loss=0.1257, mIoU=0.6243, F1=0.7045
- Epoch 6: Loss=0.1203, mIoU=0.5652, F1=0.6234
- Epoch 7: Loss=0.1167, mIoU=0.5716, F1=0.6329
- Epoch 8: Loss=0.1123, mIoU=0.6644, F1=0.7536
- Epoch 9: Loss=0.1054, mIoU=0.6634, F1=0.7514
- Epoch 10: Loss=0.1033, mIoU=0.6778, F1=0.7671

本轮结果: Round=5, Labeled=541, Selection=best_val (epoch=10), mIoU=0.6778, F1=0.7671, peak_mIoU=0.6778

Round=5, Labeled=541, mIoU=0.6778, F1=0.7671

## Round 12

Labeled Pool Size: 1157

- Epoch 1: Loss=0.2604, mIoU=0.5371, F1=0.5785
- Epoch 2: Loss=0.1256, mIoU=0.5021, F1=0.5163
- Epoch 3: Loss=0.1098, mIoU=0.6636, F1=0.7505
- Epoch 4: Loss=0.0995, mIoU=0.6688, F1=0.7563
- Epoch 5: Loss=0.0932, mIoU=0.6316, F1=0.7133
- Epoch 6: Loss=0.0886, mIoU=0.6976, F1=0.7863
- Epoch 7: Loss=0.0839, mIoU=0.6990, F1=0.7883
## Round 6

Labeled Pool Size: 629

- Epoch 1: Loss=0.3628, mIoU=0.5299, F1=0.5666
- Epoch 8: Loss=0.0820, mIoU=0.6780, F1=0.7662
- Epoch 2: Loss=0.1768, mIoU=0.5669, F1=0.6259
- Epoch 3: Loss=0.1416, mIoU=0.5981, F1=0.6704
- Epoch 9: Loss=0.0783, mIoU=0.6844, F1=0.7730
- Epoch 4: Loss=0.1282, mIoU=0.6568, F1=0.7430
- Epoch 5: Loss=0.1199, mIoU=0.6724, F1=0.7604
- Epoch 10: Loss=0.0751, mIoU=0.7028, F1=0.7918

本轮结果: Round=12, Labeled=1157, Selection=best_val (epoch=10), mIoU=0.7028, F1=0.7918, peak_mIoU=0.7028

Round=12, Labeled=1157, mIoU=0.7028, F1=0.7918

- Epoch 6: Loss=0.1140, mIoU=0.6103, F1=0.6869
- Epoch 7: Loss=0.1121, mIoU=0.6220, F1=0.7016
- Epoch 8: Loss=0.1055, mIoU=0.6739, F1=0.7621
- Epoch 9: Loss=0.1029, mIoU=0.6783, F1=0.7668
- Epoch 10: Loss=0.0989, mIoU=0.6553, F1=0.7414

本轮结果: Round=6, Labeled=629, Selection=best_val (epoch=9), mIoU=0.6783, F1=0.7668, peak_mIoU=0.6783

Round=6, Labeled=629, mIoU=0.6783, F1=0.7668

## Round 13

Labeled Pool Size: 1245

## Round 7

Labeled Pool Size: 717

- Epoch 1: Loss=0.1960, mIoU=0.5746, F1=0.6371
- Epoch 1: Loss=0.4047, mIoU=0.5071, F1=0.5268
- Epoch 2: Loss=0.1717, mIoU=0.6554, F1=0.7415
- Epoch 2: Loss=0.1151, mIoU=0.6723, F1=0.7614
- Epoch 3: Loss=0.1391, mIoU=0.5743, F1=0.6368
- Epoch 4: Loss=0.1261, mIoU=0.6799, F1=0.7684
- Epoch 3: Loss=0.1041, mIoU=0.6998, F1=0.7889
- Epoch 5: Loss=0.1198, mIoU=0.6523, F1=0.7385
- Epoch 6: Loss=0.1134, mIoU=0.6931, F1=0.7823
- Epoch 4: Loss=0.0933, mIoU=0.6938, F1=0.7825
- Epoch 7: Loss=0.1098, mIoU=0.6537, F1=0.7393
- Epoch 5: Loss=0.0897, mIoU=0.6438, F1=0.7280
- Epoch 8: Loss=0.1035, mIoU=0.6886, F1=0.7778
- Epoch 9: Loss=0.1016, mIoU=0.6768, F1=0.7649
- Epoch 6: Loss=0.0877, mIoU=0.6069, F1=0.6818
- Epoch 10: Loss=0.0966, mIoU=0.6910, F1=0.7803

本轮结果: Round=7, Labeled=717, Selection=best_val (epoch=6), mIoU=0.6931, F1=0.7823, peak_mIoU=0.6931

Round=7, Labeled=717, mIoU=0.6931, F1=0.7823

- Epoch 7: Loss=0.0828, mIoU=0.7168, F1=0.8053
- Epoch 8: Loss=0.0801, mIoU=0.6959, F1=0.7856
- Epoch 9: Loss=0.0768, mIoU=0.7022, F1=0.7911
- Epoch 10: Loss=0.0731, mIoU=0.6731, F1=0.7610

本轮结果: Round=13, Labeled=1245, Selection=best_val (epoch=7), mIoU=0.7168, F1=0.8053, peak_mIoU=0.7168

Round=13, Labeled=1245, mIoU=0.7168, F1=0.8053

## Round 8

Labeled Pool Size: 805

- Epoch 1: Loss=0.3675, mIoU=0.5489, F1=0.5983
- Epoch 2: Loss=0.1642, mIoU=0.5533, F1=0.6051
- Epoch 3: Loss=0.1368, mIoU=0.5471, F1=0.5952
- Epoch 4: Loss=0.1241, mIoU=0.6422, F1=0.7260
## Round 14

Labeled Pool Size: 1333

- Epoch 5: Loss=0.1155, mIoU=0.6114, F1=0.6880
- Epoch 1: Loss=0.2334, mIoU=0.5596, F1=0.6146
- Epoch 6: Loss=0.1097, mIoU=0.5827, F1=0.6492
- Epoch 7: Loss=0.1055, mIoU=0.6288, F1=0.7104
- Epoch 2: Loss=0.1161, mIoU=0.6604, F1=0.7471
- Epoch 8: Loss=0.1024, mIoU=0.6390, F1=0.7231
- Epoch 9: Loss=0.0969, mIoU=0.6241, F1=0.7043
- Epoch 3: Loss=0.1002, mIoU=0.6637, F1=0.7507
- Epoch 10: Loss=0.0944, mIoU=0.6280, F1=0.7096

本轮结果: Round=8, Labeled=805, Selection=best_val (epoch=4), mIoU=0.6422, F1=0.7260, peak_mIoU=0.6422

Round=8, Labeled=805, mIoU=0.6422, F1=0.7260

- Epoch 4: Loss=0.0933, mIoU=0.6118, F1=0.6882
- Epoch 5: Loss=0.0874, mIoU=0.6088, F1=0.6845
- Epoch 6: Loss=0.0864, mIoU=0.6660, F1=0.7533
- Epoch 7: Loss=0.0815, mIoU=0.6672, F1=0.7551
- Epoch 8: Loss=0.0793, mIoU=0.6600, F1=0.7465
- Epoch 9: Loss=0.0758, mIoU=0.6763, F1=0.7647
- Epoch 10: Loss=0.0739, mIoU=0.6486, F1=0.7338

本轮结果: Round=14, Labeled=1333, Selection=best_val (epoch=9), mIoU=0.6763, F1=0.7647, peak_mIoU=0.6763

Round=14, Labeled=1333, mIoU=0.6763, F1=0.7647

## Round 9

Labeled Pool Size: 893

- Epoch 1: Loss=0.2612, mIoU=0.5227, F1=0.5542
- Epoch 2: Loss=0.1414, mIoU=0.5810, F1=0.6465
- Epoch 3: Loss=0.1244, mIoU=0.5752, F1=0.6382
- Epoch 4: Loss=0.1163, mIoU=0.6548, F1=0.7410
- Epoch 5: Loss=0.1072, mIoU=0.6693, F1=0.7587
- Epoch 6: Loss=0.1017, mIoU=0.6743, F1=0.7622
## Round 15

Labeled Pool Size: 1421

- Epoch 7: Loss=0.0985, mIoU=0.6415, F1=0.7255
- Epoch 1: Loss=0.2188, mIoU=0.5232, F1=0.5550
- Epoch 8: Loss=0.0938, mIoU=0.6932, F1=0.7822
- Epoch 9: Loss=0.0894, mIoU=0.6619, F1=0.7486
- Epoch 2: Loss=0.1109, mIoU=0.5278, F1=0.5628
- Epoch 10: Loss=0.0872, mIoU=0.6874, F1=0.7761

本轮结果: Round=9, Labeled=893, Selection=best_val (epoch=8), mIoU=0.6932, F1=0.7822, peak_mIoU=0.6932

Round=9, Labeled=893, mIoU=0.6932, F1=0.7822

- Epoch 3: Loss=0.0970, mIoU=0.6761, F1=0.7639
- Epoch 4: Loss=0.0900, mIoU=0.6792, F1=0.7675
- Epoch 5: Loss=0.0855, mIoU=0.6240, F1=0.7038
- Epoch 6: Loss=0.0818, mIoU=0.6926, F1=0.7823
- Epoch 7: Loss=0.0777, mIoU=0.6497, F1=0.7382
## Round 10

Labeled Pool Size: 981

- Epoch 8: Loss=0.0760, mIoU=0.6784, F1=0.7685
- Epoch 1: Loss=0.2310, mIoU=0.5139, F1=0.5384
- Epoch 9: Loss=0.0729, mIoU=0.6710, F1=0.7591
- Epoch 2: Loss=0.1317, mIoU=0.5931, F1=0.6635
- Epoch 3: Loss=0.1165, mIoU=0.5929, F1=0.6632
- Epoch 10: Loss=0.0720, mIoU=0.7366, F1=0.8236

本轮结果: Round=15, Labeled=1421, Selection=best_val (epoch=10), mIoU=0.7366, F1=0.8236, peak_mIoU=0.7366

Round=15, Labeled=1421, mIoU=0.7366, F1=0.8236

- Epoch 4: Loss=0.1078, mIoU=0.5994, F1=0.6721
- Epoch 5: Loss=0.1026, mIoU=0.6598, F1=0.7473
- Epoch 6: Loss=0.0998, mIoU=0.6466, F1=0.7324
- Epoch 7: Loss=0.0934, mIoU=0.6645, F1=0.7518
- Epoch 8: Loss=0.0905, mIoU=0.6609, F1=0.7485
- Epoch 9: Loss=0.0878, mIoU=0.7136, F1=0.8023
## Round 16

Labeled Pool Size: 1509

- Test-only mode: Round 16 loads Round 15 best_val checkpoint (epoch=10, val_mIoU=0.7366, val_F1=0.8236)

本轮结果: Round=16, Labeled=1509, Selection=prev_round_best_val (source_round=15, epoch=10), mIoU=0.7366, F1=0.8236, peak_mIoU=0.7366

Round=16, Labeled=1509, mIoU=0.7366, F1=0.8236


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421, 1509]
ALC(基于每轮选模val mIoU): 0.5969
最后一轮选模 mIoU(val): 0.736648887417037
最后一轮选模 F1(val): 0.8235866047267313
最终报告 mIoU(test): 0.7397126353304015
最终报告 F1(test): 0.8266786918701017
最终输出 mIoU: 0.7397 (source=final_report)
最终输出 F1: 0.8267 (source=final_report)
最终 Test Split: test
最终 Report: {'loss': 0.038137436860706656, 'mIoU': 0.7397126353304015, 'f1_score': 0.8266786918701017}
- Epoch 10: Loss=0.0852, mIoU=0.6931, F1=0.7827

本轮结果: Round=10, Labeled=981, Selection=best_val (epoch=9), mIoU=0.7136, F1=0.8023, peak_mIoU=0.7136

Round=10, Labeled=981, mIoU=0.7136, F1=0.8023

## Round 11

Labeled Pool Size: 1069

- Epoch 1: Loss=0.2240, mIoU=0.4993, F1=0.5110
- Epoch 2: Loss=0.1269, mIoU=0.6038, F1=0.6779
- Epoch 3: Loss=0.1112, mIoU=0.5436, F1=0.5895
- Epoch 4: Loss=0.1021, mIoU=0.6279, F1=0.7088
- Epoch 5: Loss=0.0972, mIoU=0.6141, F1=0.6916
- Epoch 6: Loss=0.0924, mIoU=0.5708, F1=0.6318
- Epoch 7: Loss=0.0883, mIoU=0.6348, F1=0.7178
- Epoch 8: Loss=0.0857, mIoU=0.6860, F1=0.7748
- Epoch 9: Loss=0.0818, mIoU=0.6699, F1=0.7576
- Epoch 10: Loss=0.0795, mIoU=0.6392, F1=0.7228

本轮结果: Round=11, Labeled=1069, Selection=best_val (epoch=8), mIoU=0.6860, F1=0.7748, peak_mIoU=0.6860

Round=11, Labeled=1069, mIoU=0.6860, F1=0.7748

## Round 12

Labeled Pool Size: 1157

- Epoch 1: Loss=0.2606, mIoU=0.6187, F1=0.6975
- Epoch 2: Loss=0.1233, mIoU=0.5301, F1=0.5671
- Epoch 3: Loss=0.1110, mIoU=0.6409, F1=0.7246
- Epoch 4: Loss=0.1021, mIoU=0.6357, F1=0.7237
- Epoch 5: Loss=0.0975, mIoU=0.6160, F1=0.6941
- Epoch 6: Loss=0.0932, mIoU=0.6857, F1=0.7743
- Epoch 7: Loss=0.0889, mIoU=0.7062, F1=0.7951
- Epoch 8: Loss=0.0838, mIoU=0.6579, F1=0.7449
- Epoch 9: Loss=0.0813, mIoU=0.7009, F1=0.7900
- Epoch 10: Loss=0.0781, mIoU=0.6516, F1=0.7403

本轮结果: Round=12, Labeled=1157, Selection=best_val (epoch=7), mIoU=0.7062, F1=0.7951, peak_mIoU=0.7062

Round=12, Labeled=1157, mIoU=0.7062, F1=0.7951

## Round 13

Labeled Pool Size: 1245

- Epoch 1: Loss=0.1934, mIoU=0.5167, F1=0.5434
- Epoch 2: Loss=0.1143, mIoU=0.5751, F1=0.6382
- Epoch 3: Loss=0.1007, mIoU=0.6249, F1=0.7050
- Epoch 4: Loss=0.0958, mIoU=0.6837, F1=0.7723
- Epoch 5: Loss=0.0891, mIoU=0.6624, F1=0.7497
- Epoch 6: Loss=0.0871, mIoU=0.5567, F1=0.6103
- Epoch 7: Loss=0.0825, mIoU=0.6586, F1=0.7452
- Epoch 8: Loss=0.0779, mIoU=0.6082, F1=0.6841
- Epoch 9: Loss=0.0758, mIoU=0.5949, F1=0.6662
- Epoch 10: Loss=0.0725, mIoU=0.6458, F1=0.7309

本轮结果: Round=13, Labeled=1245, Selection=best_val (epoch=4), mIoU=0.6837, F1=0.7723, peak_mIoU=0.6837

Round=13, Labeled=1245, mIoU=0.6837, F1=0.7723

## Round 14

Labeled Pool Size: 1333

- Epoch 1: Loss=0.2398, mIoU=0.6164, F1=0.6945
- Epoch 2: Loss=0.1129, mIoU=0.5612, F1=0.6170
- Epoch 3: Loss=0.1001, mIoU=0.6968, F1=0.7859
- Epoch 4: Loss=0.0929, mIoU=0.6865, F1=0.7752
- Epoch 5: Loss=0.0861, mIoU=0.6975, F1=0.7862
- Epoch 6: Loss=0.0841, mIoU=0.6989, F1=0.7878
- Epoch 7: Loss=0.0808, mIoU=0.7118, F1=0.8003
- Epoch 8: Loss=0.0769, mIoU=0.6763, F1=0.7644
- Epoch 9: Loss=0.0739, mIoU=0.6886, F1=0.7779
- Epoch 10: Loss=0.0725, mIoU=0.7124, F1=0.8012

本轮结果: Round=14, Labeled=1333, Selection=best_val (epoch=10), mIoU=0.7124, F1=0.8012, peak_mIoU=0.7124

Round=14, Labeled=1333, mIoU=0.7124, F1=0.8012

## Round 15

Labeled Pool Size: 1421

- Epoch 1: Loss=0.2214, mIoU=0.5254, F1=0.5589
- Epoch 2: Loss=0.1115, mIoU=0.5429, F1=0.5883
- Epoch 3: Loss=0.0957, mIoU=0.6410, F1=0.7250
- Epoch 4: Loss=0.0881, mIoU=0.6418, F1=0.7259
- Epoch 5: Loss=0.0832, mIoU=0.6611, F1=0.7479
- Epoch 6: Loss=0.0807, mIoU=0.6543, F1=0.7404
- Epoch 7: Loss=0.0776, mIoU=0.6706, F1=0.7580
- Epoch 8: Loss=0.0736, mIoU=0.5763, F1=0.6399
- Epoch 9: Loss=0.0722, mIoU=0.6784, F1=0.7675
- Epoch 10: Loss=0.0697, mIoU=0.6560, F1=0.7421

本轮结果: Round=15, Labeled=1421, Selection=best_val (epoch=9), mIoU=0.6784, F1=0.7675, peak_mIoU=0.6784

Round=15, Labeled=1421, mIoU=0.6784, F1=0.7675

## Round 16

Labeled Pool Size: 1509

- Test-only mode: Round 16 loads Round 15 best_val checkpoint (epoch=9, val_mIoU=0.6784, val_F1=0.7675)

本轮结果: Round=16, Labeled=1509, Selection=prev_round_best_val (source_round=15, epoch=9), mIoU=0.6784, F1=0.7675, peak_mIoU=0.6784

Round=16, Labeled=1509, mIoU=0.6784, F1=0.7675


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421, 1509]
ALC(基于每轮选模val mIoU): 0.5888
最后一轮选模 mIoU(val): 0.678356226720878
最后一轮选模 F1(val): 0.7674802828926675
最终报告 mIoU(test): 0.6815766903097926
最终报告 F1(test): 0.7712904011804701
最终输出 mIoU: 0.6816 (source=final_report)
最终输出 F1: 0.7713 (source=final_report)
最终 Test Split: test
最终 Report: {'loss': 0.05652288850862533, 'mIoU': 0.6815766903097926, 'f1_score': 0.7712904011804701}
