# 实验日志

实验名称: fixed_lambda
描述: 消融：Fixed λ=0.5；保留Agent，隔离三阶段λ policy贡献
开始时间: 2026-03-24T20:57:39.241278

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.3463, mIoU=0.5022, F1=0.5172
- Epoch 2: Loss=0.1956, mIoU=0.5399, F1=0.5854
- Epoch 3: Loss=0.1354, mIoU=0.5611, F1=0.6192
- Epoch 4: Loss=0.1056, mIoU=0.5789, F1=0.6493
- Epoch 5: Loss=0.0889, mIoU=0.5799, F1=0.6476
- Epoch 6: Loss=0.0765, mIoU=0.6170, F1=0.6963
- Epoch 7: Loss=0.0704, mIoU=0.5998, F1=0.6754
- Epoch 8: Loss=0.0671, mIoU=0.5482, F1=0.5971
- Epoch 9: Loss=0.0599, mIoU=0.5868, F1=0.6604
- Epoch 10: Loss=0.0568, mIoU=0.5554, F1=0.6096

本轮结果: Round=1, Labeled=189, Selection=best_val (epoch=6), mIoU=0.6170, F1=0.6963, peak_mIoU=0.6170

Round=1, Labeled=189, mIoU=0.6170, F1=0.6963

## Round 2

Labeled Pool Size: 277

- Epoch 1: Loss=0.4288, mIoU=0.5000, F1=0.5155
- Epoch 2: Loss=0.2259, mIoU=0.5066, F1=0.5269
- Epoch 3: Loss=0.1616, mIoU=0.5200, F1=0.5512
- Epoch 4: Loss=0.1331, mIoU=0.5067, F1=0.5271
- Epoch 5: Loss=0.1165, mIoU=0.4997, F1=0.5119
- Epoch 6: Loss=0.1052, mIoU=0.5207, F1=0.5509
- Epoch 7: Loss=0.0962, mIoU=0.5488, F1=0.5976
- Epoch 8: Loss=0.0924, mIoU=0.5423, F1=0.5872
- Epoch 9: Loss=0.0898, mIoU=0.5495, F1=0.5989
- Epoch 10: Loss=0.0828, mIoU=0.5590, F1=0.6139

本轮结果: Round=2, Labeled=277, Selection=best_val (epoch=10), mIoU=0.5590, F1=0.6139, peak_mIoU=0.5590

Round=2, Labeled=277, mIoU=0.5590, F1=0.6139

## Round 3

Labeled Pool Size: 365

- Epoch 1: Loss=0.4024, mIoU=0.5671, F1=0.6266
- Epoch 2: Loss=0.2037, mIoU=0.5107, F1=0.5324
- Epoch 3: Loss=0.1530, mIoU=0.5515, F1=0.6020
- Epoch 4: Loss=0.1358, mIoU=0.5046, F1=0.5211
- Epoch 5: Loss=0.1272, mIoU=0.5452, F1=0.5919
- Epoch 6: Loss=0.1224, mIoU=0.6629, F1=0.7514
- Epoch 7: Loss=0.1135, mIoU=0.6239, F1=0.7038
- Epoch 8: Loss=0.1090, mIoU=0.6831, F1=0.7720
- Epoch 9: Loss=0.1037, mIoU=0.6467, F1=0.7314
- Epoch 10: Loss=0.1007, mIoU=0.6677, F1=0.7555

本轮结果: Round=3, Labeled=365, Selection=best_val (epoch=8), mIoU=0.6831, F1=0.7720, peak_mIoU=0.6831

Round=3, Labeled=365, mIoU=0.6831, F1=0.7720

## Round 4

Labeled Pool Size: 453

- Epoch 1: Loss=0.3684, mIoU=0.5387, F1=0.5815
- Epoch 2: Loss=0.1899, mIoU=0.5074, F1=0.5264
- Epoch 3: Loss=0.1521, mIoU=0.5227, F1=0.5541
- Epoch 4: Loss=0.1384, mIoU=0.5717, F1=0.6329
- Epoch 5: Loss=0.1282, mIoU=0.5766, F1=0.6404
- Epoch 6: Loss=0.1231, mIoU=0.6484, F1=0.7331
- Epoch 7: Loss=0.1160, mIoU=0.6578, F1=0.7440
- Epoch 8: Loss=0.1113, mIoU=0.5996, F1=0.6724
- Epoch 9: Loss=0.1038, mIoU=0.6333, F1=0.7155
- Epoch 10: Loss=0.1049, mIoU=0.6428, F1=0.7268

本轮结果: Round=4, Labeled=453, Selection=best_val (epoch=7), mIoU=0.6578, F1=0.7440, peak_mIoU=0.6578

Round=4, Labeled=453, mIoU=0.6578, F1=0.7440

## Round 5

Labeled Pool Size: 541

- Epoch 1: Loss=0.4223, mIoU=0.5104, F1=0.5376
- Epoch 2: Loss=0.1903, mIoU=0.5054, F1=0.5226
- Epoch 3: Loss=0.1494, mIoU=0.5683, F1=0.6280
- Epoch 4: Loss=0.1324, mIoU=0.5566, F1=0.6100
- Epoch 5: Loss=0.1222, mIoU=0.5287, F1=0.5644
- Epoch 6: Loss=0.1181, mIoU=0.6389, F1=0.7220

--- [Checkpoint] 续跑开始时间: 2026-03-24T21:46:18.773614 ---

## Round 5

Labeled Pool Size: 541

- Epoch 7: Loss=0.1121, mIoU=0.6670, F1=0.7546
- Epoch 1: Loss=0.4206, mIoU=0.5154, F1=0.5414
- Epoch 8: Loss=0.1078, mIoU=0.5860, F1=0.6536
- Epoch 2: Loss=0.1895, mIoU=0.5402, F1=0.5839
- Epoch 9: Loss=0.1047, mIoU=0.6714, F1=0.7590
- Epoch 3: Loss=0.1478, mIoU=0.6066, F1=0.6817
- Epoch 10: Loss=0.1012, mIoU=0.6608, F1=0.7473

本轮结果: Round=5, Labeled=541, Selection=best_val (epoch=9), mIoU=0.6714, F1=0.7590, peak_mIoU=0.6714

Round=5, Labeled=541, mIoU=0.6714, F1=0.7590

- Epoch 4: Loss=0.1315, mIoU=0.5828, F1=0.6490
- Epoch 5: Loss=0.1202, mIoU=0.5684, F1=0.6280
- Epoch 6: Loss=0.1159, mIoU=0.6630, F1=0.7498
- Epoch 7: Loss=0.1107, mIoU=0.6552, F1=0.7431
- Epoch 8: Loss=0.1057, mIoU=0.6121, F1=0.6889
- Epoch 9: Loss=0.1026, mIoU=0.6415, F1=0.7252
- Epoch 10: Loss=0.1005, mIoU=0.6257, F1=0.7063

本轮结果: Round=5, Labeled=541, Selection=best_val (epoch=6), mIoU=0.6630, F1=0.7498, peak_mIoU=0.6630

Round=5, Labeled=541, mIoU=0.6630, F1=0.7498

## Round 6

Labeled Pool Size: 629

- Epoch 1: Loss=0.3540, mIoU=0.5323, F1=0.5711
- Epoch 2: Loss=0.1708, mIoU=0.5174, F1=0.5446
- Epoch 3: Loss=0.1428, mIoU=0.5768, F1=0.6405
- Epoch 4: Loss=0.1279, mIoU=0.5403, F1=0.5839
## Round 6

Labeled Pool Size: 629

- Epoch 5: Loss=0.1217, mIoU=0.5849, F1=0.6526
- Epoch 1: Loss=0.3579, mIoU=0.5278, F1=0.5632
- Epoch 6: Loss=0.1190, mIoU=0.5395, F1=0.5829
- Epoch 2: Loss=0.1718, mIoU=0.5254, F1=0.5587
- Epoch 7: Loss=0.1060, mIoU=0.6189, F1=0.6974
- Epoch 3: Loss=0.1419, mIoU=0.6193, F1=0.6980
- Epoch 8: Loss=0.1039, mIoU=0.6799, F1=0.7682
- Epoch 4: Loss=0.1261, mIoU=0.6705, F1=0.7580
- Epoch 9: Loss=0.1017, mIoU=0.6632, F1=0.7503
- Epoch 5: Loss=0.1224, mIoU=0.5844, F1=0.6513
- Epoch 10: Loss=0.0978, mIoU=0.6287, F1=0.7099

本轮结果: Round=6, Labeled=629, Selection=best_val (epoch=8), mIoU=0.6799, F1=0.7682, peak_mIoU=0.6799

Round=6, Labeled=629, mIoU=0.6799, F1=0.7682

- Epoch 6: Loss=0.1186, mIoU=0.6679, F1=0.7555
- Epoch 7: Loss=0.1064, mIoU=0.6322, F1=0.7139
- Epoch 8: Loss=0.1026, mIoU=0.7110, F1=0.7998
- Epoch 9: Loss=0.1008, mIoU=0.6964, F1=0.7858
- Epoch 10: Loss=0.0989, mIoU=0.6775, F1=0.7659

本轮结果: Round=6, Labeled=629, Selection=best_val (epoch=8), mIoU=0.7110, F1=0.7998, peak_mIoU=0.7110

Round=6, Labeled=629, mIoU=0.7110, F1=0.7998

## Round 7

Labeled Pool Size: 717

- Epoch 1: Loss=0.3664, mIoU=0.5052, F1=0.5225
- Epoch 2: Loss=0.1677, mIoU=0.6318, F1=0.7134
## Round 7

Labeled Pool Size: 717

- Epoch 3: Loss=0.1398, mIoU=0.5839, F1=0.6507
- Epoch 1: Loss=0.3652, mIoU=0.5615, F1=0.6181
- Epoch 4: Loss=0.1296, mIoU=0.6731, F1=0.7611
- Epoch 2: Loss=0.1669, mIoU=0.5355, F1=0.5761
- Epoch 5: Loss=0.1190, mIoU=0.6763, F1=0.7642
- Epoch 3: Loss=0.1405, mIoU=0.5758, F1=0.6390
- Epoch 6: Loss=0.1163, mIoU=0.6762, F1=0.7643
- Epoch 4: Loss=0.1288, mIoU=0.6623, F1=0.7493
- Epoch 7: Loss=0.1141, mIoU=0.6861, F1=0.7746
- Epoch 5: Loss=0.1204, mIoU=0.6184, F1=0.6974
- Epoch 8: Loss=0.1085, mIoU=0.5704, F1=0.6311
- Epoch 6: Loss=0.1125, mIoU=0.5774, F1=0.6415
- Epoch 9: Loss=0.1040, mIoU=0.7115, F1=0.8002
- Epoch 7: Loss=0.1096, mIoU=0.6354, F1=0.7184
- Epoch 10: Loss=0.0997, mIoU=0.7041, F1=0.7928

本轮结果: Round=7, Labeled=717, Selection=best_val (epoch=9), mIoU=0.7115, F1=0.8002, peak_mIoU=0.7115

Round=7, Labeled=717, mIoU=0.7115, F1=0.8002

- Epoch 8: Loss=0.1069, mIoU=0.5605, F1=0.6167
- Epoch 9: Loss=0.1016, mIoU=0.6307, F1=0.7129
- Epoch 10: Loss=0.0989, mIoU=0.6930, F1=0.7823

本轮结果: Round=7, Labeled=717, Selection=best_val (epoch=10), mIoU=0.6930, F1=0.7823, peak_mIoU=0.6930

Round=7, Labeled=717, mIoU=0.6930, F1=0.7823

## Round 8

Labeled Pool Size: 805

- Epoch 1: Loss=0.3435, mIoU=0.5251, F1=0.5584
- Epoch 2: Loss=0.1548, mIoU=0.5365, F1=0.5780
## Round 8

Labeled Pool Size: 805

- Epoch 3: Loss=0.1320, mIoU=0.6211, F1=0.7003
- Epoch 1: Loss=0.3571, mIoU=0.5759, F1=0.6392
- Epoch 4: Loss=0.1220, mIoU=0.6436, F1=0.7274
- Epoch 2: Loss=0.1532, mIoU=0.5731, F1=0.6350
- Epoch 5: Loss=0.1197, mIoU=0.6516, F1=0.7371
- Epoch 3: Loss=0.1340, mIoU=0.5550, F1=0.6077
- Epoch 6: Loss=0.1102, mIoU=0.6349, F1=0.7175
- Epoch 4: Loss=0.1231, mIoU=0.6940, F1=0.7829
- Epoch 7: Loss=0.1054, mIoU=0.7078, F1=0.7965
- Epoch 5: Loss=0.1142, mIoU=0.6537, F1=0.7404
- Epoch 8: Loss=0.0992, mIoU=0.6791, F1=0.7672
- Epoch 6: Loss=0.1128, mIoU=0.6639, F1=0.7513
- Epoch 7: Loss=0.1036, mIoU=0.6913, F1=0.7801
- Epoch 9: Loss=0.0956, mIoU=0.6966, F1=0.7858

**[ERROR] Round 8 失败: [Errno 2] No such file or directory: 'results/runs/ablation_matrix_p3_20260323_235540_seed43/fixed_lambda_status.json.tmp' -> 'results/runs/ablation_matrix_p3_20260323_235540_seed43/fixed_lambda_status.json'**

- Epoch 10: Loss=0.0947, mIoU=0.6869, F1=0.7756

本轮结果: Round=8, Labeled=805, Selection=best_val (epoch=7), mIoU=0.7078, F1=0.7965, peak_mIoU=0.7078

Round=8, Labeled=805, mIoU=0.7078, F1=0.7965

## Round 9

Labeled Pool Size: 893

- Epoch 1: Loss=0.2580, mIoU=0.5228, F1=0.5544
- Epoch 2: Loss=0.1403, mIoU=0.5304, F1=0.5674
- Epoch 3: Loss=0.1232, mIoU=0.6548, F1=0.7414
- Epoch 4: Loss=0.1115, mIoU=0.6180, F1=0.6967
- Epoch 5: Loss=0.1077, mIoU=0.6799, F1=0.7685
- Epoch 6: Loss=0.1029, mIoU=0.6146, F1=0.6925
- Epoch 7: Loss=0.0961, mIoU=0.6284, F1=0.7103
- Epoch 8: Loss=0.0918, mIoU=0.6158, F1=0.6937
- Epoch 9: Loss=0.0884, mIoU=0.6964, F1=0.7857
- Epoch 10: Loss=0.0864, mIoU=0.6841, F1=0.7730

本轮结果: Round=9, Labeled=893, Selection=best_val (epoch=9), mIoU=0.6964, F1=0.7857, peak_mIoU=0.6964

Round=9, Labeled=893, mIoU=0.6964, F1=0.7857

## Round 10

Labeled Pool Size: 981

- Epoch 1: Loss=0.2275, mIoU=0.5324, F1=0.5710
- Epoch 2: Loss=0.1308, mIoU=0.5626, F1=0.6194
- Epoch 3: Loss=0.1129, mIoU=0.6558, F1=0.7422
- Epoch 4: Loss=0.1085, mIoU=0.6418, F1=0.7259
- Epoch 5: Loss=0.0997, mIoU=0.6720, F1=0.7616
- Epoch 6: Loss=0.0958, mIoU=0.6635, F1=0.7510
- Epoch 7: Loss=0.0906, mIoU=0.6370, F1=0.7210
- Epoch 8: Loss=0.0883, mIoU=0.6371, F1=0.7206
- Epoch 9: Loss=0.0858, mIoU=0.6707, F1=0.7587
- Epoch 10: Loss=0.0827, mIoU=0.6685, F1=0.7573

本轮结果: Round=10, Labeled=981, Selection=best_val (epoch=5), mIoU=0.6720, F1=0.7616, peak_mIoU=0.6720

Round=10, Labeled=981, mIoU=0.6720, F1=0.7616

## Round 11

Labeled Pool Size: 1069

- Epoch 1: Loss=0.2201, mIoU=0.5327, F1=0.5715
- Epoch 2: Loss=0.1214, mIoU=0.5542, F1=0.6062
- Epoch 3: Loss=0.1072, mIoU=0.6079, F1=0.6837
- Epoch 4: Loss=0.0991, mIoU=0.6408, F1=0.7248
- Epoch 5: Loss=0.0947, mIoU=0.6019, F1=0.6757
- Epoch 6: Loss=0.0900, mIoU=0.5961, F1=0.6675
- Epoch 7: Loss=0.0849, mIoU=0.6659, F1=0.7534
- Epoch 8: Loss=0.0823, mIoU=0.7059, F1=0.7950
- Epoch 9: Loss=0.0799, mIoU=0.7098, F1=0.7985
- Epoch 10: Loss=0.0761, mIoU=0.7024, F1=0.7916

本轮结果: Round=11, Labeled=1069, Selection=best_val (epoch=9), mIoU=0.7098, F1=0.7985, peak_mIoU=0.7098

Round=11, Labeled=1069, mIoU=0.7098, F1=0.7985

## Round 12

Labeled Pool Size: 1157

- Epoch 1: Loss=0.2504, mIoU=0.5127, F1=0.5362
- Epoch 2: Loss=0.1197, mIoU=0.5093, F1=0.5299
- Epoch 3: Loss=0.1051, mIoU=0.6380, F1=0.7213
- Epoch 4: Loss=0.0965, mIoU=0.6226, F1=0.7022
- Epoch 5: Loss=0.0920, mIoU=0.6464, F1=0.7309
- Epoch 6: Loss=0.0879, mIoU=0.6711, F1=0.7599
- Epoch 7: Loss=0.0818, mIoU=0.6395, F1=0.7233
- Epoch 8: Loss=0.0808, mIoU=0.6138, F1=0.6916
- Epoch 9: Loss=0.0768, mIoU=0.7081, F1=0.7969
- Epoch 10: Loss=0.0763, mIoU=0.5926, F1=0.6628

本轮结果: Round=12, Labeled=1157, Selection=best_val (epoch=9), mIoU=0.7081, F1=0.7969, peak_mIoU=0.7081

Round=12, Labeled=1157, mIoU=0.7081, F1=0.7969

## Round 13

Labeled Pool Size: 1245

- Epoch 1: Loss=0.1856, mIoU=0.5036, F1=0.5193
- Epoch 2: Loss=0.1060, mIoU=0.5727, F1=0.6344
- Epoch 3: Loss=0.0934, mIoU=0.5466, F1=0.5942
- Epoch 4: Loss=0.0864, mIoU=0.6424, F1=0.7264
- Epoch 5: Loss=0.0809, mIoU=0.6532, F1=0.7390
- Epoch 6: Loss=0.0770, mIoU=0.6986, F1=0.7874
- Epoch 7: Loss=0.0738, mIoU=0.6813, F1=0.7703
- Epoch 8: Loss=0.0721, mIoU=0.6326, F1=0.7147
- Epoch 9: Loss=0.0706, mIoU=0.6275, F1=0.7088
- Epoch 10: Loss=0.0667, mIoU=0.6364, F1=0.7198

本轮结果: Round=13, Labeled=1245, Selection=best_val (epoch=6), mIoU=0.6986, F1=0.7874, peak_mIoU=0.6986

Round=13, Labeled=1245, mIoU=0.6986, F1=0.7874

## Round 14

Labeled Pool Size: 1333

- Epoch 1: Loss=0.2209, mIoU=0.5144, F1=0.5394
- Epoch 2: Loss=0.1072, mIoU=0.5897, F1=0.6592
- Epoch 3: Loss=0.0954, mIoU=0.6795, F1=0.7678
- Epoch 4: Loss=0.0887, mIoU=0.7106, F1=0.7992
- Epoch 5: Loss=0.0829, mIoU=0.6897, F1=0.7785
- Epoch 6: Loss=0.0791, mIoU=0.6619, F1=0.7486
- Epoch 7: Loss=0.0754, mIoU=0.5858, F1=0.6534
- Epoch 8: Loss=0.0731, mIoU=0.6884, F1=0.7770
- Epoch 9: Loss=0.0700, mIoU=0.6651, F1=0.7522
- Epoch 10: Loss=0.0673, mIoU=0.6611, F1=0.7480

本轮结果: Round=14, Labeled=1333, Selection=best_val (epoch=4), mIoU=0.7106, F1=0.7992, peak_mIoU=0.7106

Round=14, Labeled=1333, mIoU=0.7106, F1=0.7992

## Round 15

Labeled Pool Size: 1421

- Epoch 1: Loss=0.2030, mIoU=0.5628, F1=0.6197
- Epoch 2: Loss=0.1027, mIoU=0.5940, F1=0.6649
- Epoch 3: Loss=0.0869, mIoU=0.6636, F1=0.7517
- Epoch 4: Loss=0.0828, mIoU=0.6796, F1=0.7685
- Epoch 5: Loss=0.0764, mIoU=0.6829, F1=0.7716
- Epoch 6: Loss=0.0727, mIoU=0.6843, F1=0.7728
- Epoch 7: Loss=0.0695, mIoU=0.7281, F1=0.8159
- Epoch 8: Loss=0.0679, mIoU=0.6823, F1=0.7713
- Epoch 9: Loss=0.0636, mIoU=0.6246, F1=0.7051
- Epoch 10: Loss=0.0602, mIoU=0.6793, F1=0.7688

本轮结果: Round=15, Labeled=1421, Selection=best_val (epoch=7), mIoU=0.7281, F1=0.8159, peak_mIoU=0.7281

Round=15, Labeled=1421, mIoU=0.7281, F1=0.8159

## Round 16

Labeled Pool Size: 1509

- Test-only mode: Round 16 loads Round 15 best_val checkpoint (epoch=7, val_mIoU=0.7281, val_F1=0.8159)

本轮结果: Round=16, Labeled=1509, Selection=prev_round_best_val (source_round=15, epoch=7), mIoU=0.7281, F1=0.8159, peak_mIoU=0.7281

Round=16, Labeled=1509, mIoU=0.7281, F1=0.8159


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421, 1509]
ALC(基于每轮选模val mIoU): 0.5998
最后一轮选模 mIoU(val): 0.7280792302513821
最后一轮选模 F1(val): 0.8159401350998905
最终报告 mIoU(test): 0.7265888926358128
最终报告 F1(test): 0.8150895813471577
最终输出 mIoU: 0.7266 (source=final_report)
最终输出 F1: 0.8151 (source=final_report)
最终 Test Split: test
最终 Report: {'loss': 0.043306168022099883, 'mIoU': 0.7265888926358128, 'f1_score': 0.8150895813471577}
