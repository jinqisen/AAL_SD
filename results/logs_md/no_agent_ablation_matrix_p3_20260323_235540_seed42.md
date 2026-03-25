# 实验日志

实验名称: no_agent
描述: 消融：w/o Agent（argmax）；保留三阶段λ policy，与full_model一致
开始时间: 2026-03-24T04:53:46.066240

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.5158, mIoU=0.5016, F1=0.5230
- Epoch 2: Loss=0.2792, mIoU=0.5188, F1=0.5488
- Epoch 3: Loss=0.1740, mIoU=0.5442, F1=0.5922
- Epoch 4: Loss=0.1276, mIoU=0.5799, F1=0.6466
- Epoch 5: Loss=0.1031, mIoU=0.5781, F1=0.6431
- Epoch 6: Loss=0.0954, mIoU=0.5322, F1=0.5704
- Epoch 7: Loss=0.0795, mIoU=0.5064, F1=0.5245
- Epoch 8: Loss=0.0745, mIoU=0.5876, F1=0.6560
- Epoch 9: Loss=0.0698, mIoU=0.5781, F1=0.6424
- Epoch 10: Loss=0.0659, mIoU=0.5373, F1=0.5792

本轮结果: Round=1, Labeled=189, Selection=best_val (epoch=8), mIoU=0.5876, F1=0.6560, peak_mIoU=0.5876

Round=1, Labeled=189, mIoU=0.5876, F1=0.6560

## Round 2

Labeled Pool Size: 277

- Epoch 1: Loss=0.3378, mIoU=0.5010, F1=0.5146
- Epoch 2: Loss=0.1921, mIoU=0.5099, F1=0.5314
- Epoch 3: Loss=0.1514, mIoU=0.5054, F1=0.5227
- Epoch 4: Loss=0.1255, mIoU=0.5317, F1=0.5697
- Epoch 5: Loss=0.1144, mIoU=0.5423, F1=0.5877
- Epoch 6: Loss=0.1075, mIoU=0.5178, F1=0.5455
- Epoch 7: Loss=0.1014, mIoU=0.5410, F1=0.5865
- Epoch 8: Loss=0.0980, mIoU=0.5300, F1=0.5669
- Epoch 9: Loss=0.0893, mIoU=0.5940, F1=0.6647
- Epoch 10: Loss=0.0870, mIoU=0.6190, F1=0.6981

本轮结果: Round=2, Labeled=277, Selection=best_val (epoch=10), mIoU=0.6190, F1=0.6981, peak_mIoU=0.6190

Round=2, Labeled=277, mIoU=0.6190, F1=0.6981

## Round 3

Labeled Pool Size: 365

- Epoch 1: Loss=0.4022, mIoU=0.5913, F1=0.6621
- Epoch 2: Loss=0.2099, mIoU=0.5841, F1=0.6508
- Epoch 3: Loss=0.1680, mIoU=0.5485, F1=0.5973
- Epoch 4: Loss=0.1449, mIoU=0.6344, F1=0.7168
- Epoch 5: Loss=0.1331, mIoU=0.5947, F1=0.6656
- Epoch 6: Loss=0.1258, mIoU=0.5074, F1=0.5263
- Epoch 7: Loss=0.1194, mIoU=0.6374, F1=0.7205
- Epoch 8: Loss=0.1154, mIoU=0.6572, F1=0.7435
- Epoch 9: Loss=0.1115, mIoU=0.5827, F1=0.6491
- Epoch 10: Loss=0.1056, mIoU=0.6400, F1=0.7234

本轮结果: Round=3, Labeled=365, Selection=best_val (epoch=8), mIoU=0.6572, F1=0.7435, peak_mIoU=0.6572

Round=3, Labeled=365, mIoU=0.6572, F1=0.7435

## Round 4

Labeled Pool Size: 453

- Epoch 1: Loss=0.3746, mIoU=0.5292, F1=0.5654
- Epoch 2: Loss=0.1966, mIoU=0.5141, F1=0.5388
- Epoch 3: Loss=0.1570, mIoU=0.5427, F1=0.5881
- Epoch 4: Loss=0.1422, mIoU=0.5351, F1=0.5752
- Epoch 5: Loss=0.1346, mIoU=0.6249, F1=0.7052
- Epoch 6: Loss=0.1232, mIoU=0.6800, F1=0.7681
- Epoch 7: Loss=0.1177, mIoU=0.6815, F1=0.7699
- Epoch 8: Loss=0.1142, mIoU=0.6829, F1=0.7713
- Epoch 9: Loss=0.1086, mIoU=0.6936, F1=0.7824
- Epoch 10: Loss=0.1064, mIoU=0.6672, F1=0.7548

本轮结果: Round=4, Labeled=453, Selection=best_val (epoch=9), mIoU=0.6936, F1=0.7824, peak_mIoU=0.6936

Round=4, Labeled=453, mIoU=0.6936, F1=0.7824

## Round 5

Labeled Pool Size: 541

- Epoch 1: Loss=0.3451, mIoU=0.5322, F1=0.5706
- Epoch 2: Loss=0.1810, mIoU=0.5274, F1=0.5625
- Epoch 3: Loss=0.1461, mIoU=0.5506, F1=0.6008
- Epoch 4: Loss=0.1363, mIoU=0.5183, F1=0.5465
- Epoch 5: Loss=0.1281, mIoU=0.5344, F1=0.5743
- Epoch 6: Loss=0.1200, mIoU=0.6051, F1=0.6801
- Epoch 7: Loss=0.1158, mIoU=0.6368, F1=0.7198
- Epoch 8: Loss=0.1133, mIoU=0.5543, F1=0.6068
- Epoch 9: Loss=0.1091, mIoU=0.6697, F1=0.7574
- Epoch 10: Loss=0.1034, mIoU=0.6384, F1=0.7218

本轮结果: Round=5, Labeled=541, Selection=best_val (epoch=9), mIoU=0.6697, F1=0.7574, peak_mIoU=0.6697

Round=5, Labeled=541, mIoU=0.6697, F1=0.7574

## Round 6

Labeled Pool Size: 629

- Epoch 1: Loss=0.3907, mIoU=0.6098, F1=0.6868
- Epoch 2: Loss=0.1819, mIoU=0.6253, F1=0.7057
- Epoch 3: Loss=0.1468, mIoU=0.5884, F1=0.6571
- Epoch 4: Loss=0.1329, mIoU=0.5958, F1=0.6672
- Epoch 5: Loss=0.1258, mIoU=0.5966, F1=0.6684
- Epoch 6: Loss=0.1198, mIoU=0.5886, F1=0.6573
- Epoch 7: Loss=0.1145, mIoU=0.6600, F1=0.7465
- Epoch 8: Loss=0.1100, mIoU=0.6886, F1=0.7775
- Epoch 9: Loss=0.1072, mIoU=0.6252, F1=0.7055
- Epoch 10: Loss=0.1009, mIoU=0.6742, F1=0.7620

本轮结果: Round=6, Labeled=629, Selection=best_val (epoch=8), mIoU=0.6886, F1=0.7775, peak_mIoU=0.6886

Round=6, Labeled=629, mIoU=0.6886, F1=0.7775

## Round 7

Labeled Pool Size: 717

- Epoch 1: Loss=0.3384, mIoU=0.4975, F1=0.5076
- Epoch 2: Loss=0.1629, mIoU=0.5556, F1=0.6088
- Epoch 3: Loss=0.1373, mIoU=0.5610, F1=0.6168
- Epoch 4: Loss=0.1230, mIoU=0.5514, F1=0.6020
- Epoch 5: Loss=0.1173, mIoU=0.6924, F1=0.7813
- Epoch 6: Loss=0.1121, mIoU=0.7053, F1=0.7951
- Epoch 7: Loss=0.1109, mIoU=0.6806, F1=0.7695
- Epoch 8: Loss=0.1038, mIoU=0.6689, F1=0.7567
- Epoch 9: Loss=0.1015, mIoU=0.6453, F1=0.7294
- Epoch 10: Loss=0.0960, mIoU=0.6957, F1=0.7843

本轮结果: Round=7, Labeled=717, Selection=best_val (epoch=6), mIoU=0.7053, F1=0.7951, peak_mIoU=0.7053

Round=7, Labeled=717, mIoU=0.7053, F1=0.7951

## Round 8

Labeled Pool Size: 805

- Epoch 1: Loss=0.3805, mIoU=0.5287, F1=0.5647
- Epoch 2: Loss=0.1580, mIoU=0.5706, F1=0.6317
- Epoch 3: Loss=0.1289, mIoU=0.6037, F1=0.6779
- Epoch 4: Loss=0.1178, mIoU=0.5586, F1=0.6132
- Epoch 5: Loss=0.1115, mIoU=0.6133, F1=0.6905
- Epoch 6: Loss=0.1065, mIoU=0.6780, F1=0.7658
- Epoch 7: Loss=0.1023, mIoU=0.6354, F1=0.7186
- Epoch 8: Loss=0.0982, mIoU=0.6251, F1=0.7057
- Epoch 9: Loss=0.0932, mIoU=0.6682, F1=0.7575
- Epoch 10: Loss=0.0943, mIoU=0.5899, F1=0.6591

本轮结果: Round=8, Labeled=805, Selection=best_val (epoch=6), mIoU=0.6780, F1=0.7658, peak_mIoU=0.6780

Round=8, Labeled=805, mIoU=0.6780, F1=0.7658

## Round 9

Labeled Pool Size: 893

- Epoch 1: Loss=0.3346, mIoU=0.5422, F1=0.5872
- Epoch 2: Loss=0.1432, mIoU=0.5135, F1=0.5377
- Epoch 3: Loss=0.1195, mIoU=0.5802, F1=0.6458
- Epoch 4: Loss=0.1102, mIoU=0.6896, F1=0.7787
- Epoch 5: Loss=0.1064, mIoU=0.5592, F1=0.6141
- Epoch 6: Loss=0.0986, mIoU=0.5890, F1=0.6579
- Epoch 7: Loss=0.0948, mIoU=0.5343, F1=0.5740
- Epoch 8: Loss=0.0916, mIoU=0.6004, F1=0.6736
- Epoch 9: Loss=0.0880, mIoU=0.6254, F1=0.7061
- Epoch 10: Loss=0.0847, mIoU=0.7470, F1=0.8328

本轮结果: Round=9, Labeled=893, Selection=best_val (epoch=10), mIoU=0.7470, F1=0.8328, peak_mIoU=0.7470

Round=9, Labeled=893, mIoU=0.7470, F1=0.8328

## Round 10

Labeled Pool Size: 981

- Epoch 1: Loss=0.2401, mIoU=0.5445, F1=0.5911
- Epoch 2: Loss=0.1247, mIoU=0.6497, F1=0.7355
- Epoch 3: Loss=0.1097, mIoU=0.6380, F1=0.7241
- Epoch 4: Loss=0.0997, mIoU=0.6371, F1=0.7202
- Epoch 5: Loss=0.0935, mIoU=0.6929, F1=0.7818
- Epoch 6: Loss=0.0893, mIoU=0.7074, F1=0.7963
- Epoch 7: Loss=0.0848, mIoU=0.6547, F1=0.7407
- Epoch 8: Loss=0.0838, mIoU=0.7031, F1=0.7924
- Epoch 9: Loss=0.0805, mIoU=0.7157, F1=0.8044
- Epoch 10: Loss=0.0767, mIoU=0.6675, F1=0.7556

本轮结果: Round=10, Labeled=981, Selection=best_val (epoch=9), mIoU=0.7157, F1=0.8044, peak_mIoU=0.7157

Round=10, Labeled=981, mIoU=0.7157, F1=0.8044

## Round 11

Labeled Pool Size: 1069

- Epoch 1: Loss=0.2083, mIoU=0.5564, F1=0.6098

--- [Checkpoint] 续跑开始时间: 2026-03-24T08:07:20.612725 ---

## Round 11

Labeled Pool Size: 1069

- Epoch 2: Loss=0.1151, mIoU=0.6092, F1=0.6856
- Epoch 1: Loss=0.2078, mIoU=0.5318, F1=0.5701
- Epoch 3: Loss=0.1012, mIoU=0.6670, F1=0.7548
- Epoch 2: Loss=0.1143, mIoU=0.5941, F1=0.6651
- Epoch 4: Loss=0.0953, mIoU=0.7095, F1=0.7983
- Epoch 3: Loss=0.1006, mIoU=0.6748, F1=0.7633
- Epoch 5: Loss=0.0876, mIoU=0.6814, F1=0.7700
- Epoch 4: Loss=0.0955, mIoU=0.6095, F1=0.6855
- Epoch 6: Loss=0.0832, mIoU=0.6702, F1=0.7577
- Epoch 5: Loss=0.0890, mIoU=0.6228, F1=0.7025
- Epoch 7: Loss=0.0802, mIoU=0.5901, F1=0.6594
- Epoch 6: Loss=0.0853, mIoU=0.6167, F1=0.6949
- Epoch 8: Loss=0.0827, mIoU=0.6701, F1=0.7598
- Epoch 7: Loss=0.0809, mIoU=0.6027, F1=0.6766
- Epoch 8: Loss=0.0780, mIoU=0.6401, F1=0.7275
- Epoch 9: Loss=0.0772, mIoU=0.6998, F1=0.7893

**[ERROR] Round 11 失败: [Errno 2] No such file or directory: 'results/runs/ablation_matrix_p3_20260323_235540_seed42/no_agent_status.json.tmp' -> 'results/runs/ablation_matrix_p3_20260323_235540_seed42/no_agent_status.json'**

- Epoch 9: Loss=0.0756, mIoU=0.6925, F1=0.7817
- Epoch 10: Loss=0.0729, mIoU=0.6140, F1=0.6913

本轮结果: Round=11, Labeled=1069, Selection=best_val (epoch=9), mIoU=0.6925, F1=0.7817, peak_mIoU=0.6925

Round=11, Labeled=1069, mIoU=0.6925, F1=0.7817

## Round 12

Labeled Pool Size: 1157

- Epoch 1: Loss=0.2032, mIoU=0.6143, F1=0.6918
- Epoch 2: Loss=0.1106, mIoU=0.5466, F1=0.5943
- Epoch 3: Loss=0.0956, mIoU=0.5431, F1=0.5889
- Epoch 4: Loss=0.0870, mIoU=0.5753, F1=0.6390
- Epoch 5: Loss=0.0823, mIoU=0.6775, F1=0.7680
- Epoch 6: Loss=0.0790, mIoU=0.7151, F1=0.8038
- Epoch 7: Loss=0.0759, mIoU=0.6219, F1=0.7016
- Epoch 8: Loss=0.0726, mIoU=0.6704, F1=0.7584
- Epoch 9: Loss=0.0701, mIoU=0.6932, F1=0.7828
- Epoch 10: Loss=0.0677, mIoU=0.7062, F1=0.7949

本轮结果: Round=12, Labeled=1157, Selection=best_val (epoch=6), mIoU=0.7151, F1=0.8038, peak_mIoU=0.7151

Round=12, Labeled=1157, mIoU=0.7151, F1=0.8038

## Round 13

Labeled Pool Size: 1245

- Epoch 1: Loss=0.2335, mIoU=0.5977, F1=0.6703
- Epoch 2: Loss=0.1035, mIoU=0.6392, F1=0.7254
- Epoch 3: Loss=0.0911, mIoU=0.6496, F1=0.7347
- Epoch 4: Loss=0.0819, mIoU=0.6773, F1=0.7655
- Epoch 5: Loss=0.0772, mIoU=0.6208, F1=0.7024
- Epoch 6: Loss=0.0742, mIoU=0.6171, F1=0.6952
- Epoch 7: Loss=0.0695, mIoU=0.6418, F1=0.7290
- Epoch 8: Loss=0.0665, mIoU=0.6654, F1=0.7548
- Epoch 9: Loss=0.0649, mIoU=0.6448, F1=0.7295
- Epoch 10: Loss=0.0622, mIoU=0.6526, F1=0.7387

本轮结果: Round=13, Labeled=1245, Selection=best_val (epoch=4), mIoU=0.6773, F1=0.7655, peak_mIoU=0.6773

Round=13, Labeled=1245, mIoU=0.6773, F1=0.7655

## Round 14

Labeled Pool Size: 1333

- Epoch 1: Loss=0.1717, mIoU=0.5052, F1=0.5223
- Epoch 2: Loss=0.0930, mIoU=0.6596, F1=0.7463
- Epoch 3: Loss=0.0819, mIoU=0.6811, F1=0.7702
- Epoch 4: Loss=0.0756, mIoU=0.6775, F1=0.7660
- Epoch 5: Loss=0.0713, mIoU=0.6986, F1=0.7881
- Epoch 6: Loss=0.0683, mIoU=0.6702, F1=0.7594
- Epoch 7: Loss=0.0649, mIoU=0.7042, F1=0.7934
- Epoch 8: Loss=0.0613, mIoU=0.6734, F1=0.7614
- Epoch 9: Loss=0.0615, mIoU=0.6870, F1=0.7755
- Epoch 10: Loss=0.0571, mIoU=0.6572, F1=0.7434

本轮结果: Round=14, Labeled=1333, Selection=best_val (epoch=7), mIoU=0.7042, F1=0.7934, peak_mIoU=0.7042

Round=14, Labeled=1333, mIoU=0.7042, F1=0.7934

## Round 15

Labeled Pool Size: 1421

- Epoch 1: Loss=0.2061, mIoU=0.5248, F1=0.5578
- Epoch 2: Loss=0.0958, mIoU=0.6517, F1=0.7377
- Epoch 3: Loss=0.0855, mIoU=0.6439, F1=0.7282
- Epoch 4: Loss=0.0761, mIoU=0.5696, F1=0.6331
- Epoch 5: Loss=0.0719, mIoU=0.6792, F1=0.7676
- Epoch 6: Loss=0.0682, mIoU=0.6704, F1=0.7579
- Epoch 7: Loss=0.0663, mIoU=0.6736, F1=0.7615
- Epoch 8: Loss=0.0631, mIoU=0.6750, F1=0.7639
- Epoch 9: Loss=0.0596, mIoU=0.6829, F1=0.7719
- Epoch 10: Loss=0.0590, mIoU=0.6836, F1=0.7726

本轮结果: Round=15, Labeled=1421, Selection=best_val (epoch=10), mIoU=0.6836, F1=0.7726, peak_mIoU=0.6836

Round=15, Labeled=1421, mIoU=0.6836, F1=0.7726

## Round 16

Labeled Pool Size: 1509

- Test-only mode: Round 16 loads Round 15 best_val checkpoint (epoch=10, val_mIoU=0.6836, val_F1=0.7726)

本轮结果: Round=16, Labeled=1509, Selection=prev_round_best_val (source_round=15, epoch=10), mIoU=0.6836, F1=0.7726, peak_mIoU=0.6836

Round=16, Labeled=1509, mIoU=0.6836, F1=0.7726


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421, 1509]
ALC(基于每轮选模val mIoU): 0.6002
最后一轮选模 mIoU(val): 0.683587134660879
最后一轮选模 F1(val): 0.7726084860157473
最终报告 mIoU(test): 0.6833718704386803
最终报告 F1(test): 0.7730270350379242
最终输出 mIoU: 0.6834 (source=final_report)
最终输出 F1: 0.7730 (source=final_report)
最终 Test Split: test
最终 Report: {'loss': 0.04802005197387189, 'mIoU': 0.6833718704386803, 'f1_score': 0.7730270350379242}
