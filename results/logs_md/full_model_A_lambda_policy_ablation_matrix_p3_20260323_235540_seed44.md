# 实验日志

实验名称: full_model_A_lambda_policy
描述: 对照A / AAL-SD (Full)：基于 auto_opt_iter15_cand2_02（warmup+risk闭环 + 后期ramp + U-guardrail；train_holdout grad_probe）
开始时间: 2026-03-25T01:14:04.943876

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.4674, mIoU=0.4936, F1=0.5225
- Epoch 2: Loss=0.2575, mIoU=0.5234, F1=0.5556
- Epoch 3: Loss=0.1653, mIoU=0.5905, F1=0.6621
- Epoch 4: Loss=0.1215, mIoU=0.5815, F1=0.6481
- Epoch 5: Loss=0.1016, mIoU=0.5310, F1=0.5685
- Epoch 6: Loss=0.0809, mIoU=0.5494, F1=0.5992
- Epoch 7: Loss=0.0711, mIoU=0.5616, F1=0.6181
- Epoch 8: Loss=0.0641, mIoU=0.5082, F1=0.5280
- Epoch 9: Loss=0.0607, mIoU=0.5472, F1=0.5957
- Epoch 10: Loss=0.0561, mIoU=0.5840, F1=0.6509

本轮结果: Round=1, Labeled=189, Selection=best_val (epoch=3), mIoU=0.5905, F1=0.6621, peak_mIoU=0.5905

Round=1, Labeled=189, mIoU=0.5905, F1=0.6621

## Round 2

Labeled Pool Size: 277

- Epoch 1: Loss=0.4324, mIoU=0.5162, F1=0.5448
- Epoch 2: Loss=0.2266, mIoU=0.4957, F1=0.5041
- Epoch 3: Loss=0.1599, mIoU=0.5249, F1=0.5582
- Epoch 4: Loss=0.1331, mIoU=0.5121, F1=0.5350
- Epoch 5: Loss=0.1155, mIoU=0.5062, F1=0.5242
- Epoch 6: Loss=0.1027, mIoU=0.5303, F1=0.5673
- Epoch 7: Loss=0.0979, mIoU=0.5865, F1=0.6548
- Epoch 8: Loss=0.0917, mIoU=0.5381, F1=0.5816
- Epoch 9: Loss=0.0849, mIoU=0.5517, F1=0.6031
- Epoch 10: Loss=0.0829, mIoU=0.5857, F1=0.6548

本轮结果: Round=2, Labeled=277, Selection=best_val (epoch=7), mIoU=0.5865, F1=0.6548, peak_mIoU=0.5865

Round=2, Labeled=277, mIoU=0.5865, F1=0.6548

## Round 3

Labeled Pool Size: 365

- Epoch 1: Loss=0.3915, mIoU=0.5251, F1=0.5588
- Epoch 2: Loss=0.2101, mIoU=0.5028, F1=0.5177
- Epoch 3: Loss=0.1637, mIoU=0.5732, F1=0.6352
- Epoch 4: Loss=0.1402, mIoU=0.5805, F1=0.6457
- Epoch 5: Loss=0.1306, mIoU=0.5412, F1=0.5854
- Epoch 6: Loss=0.1224, mIoU=0.5715, F1=0.6328
- Epoch 7: Loss=0.1196, mIoU=0.5809, F1=0.6463
- Epoch 8: Loss=0.1131, mIoU=0.5221, F1=0.5531
- Epoch 9: Loss=0.1095, mIoU=0.6173, F1=0.6955
- Epoch 10: Loss=0.1081, mIoU=0.6051, F1=0.6802

本轮结果: Round=3, Labeled=365, Selection=best_val (epoch=9), mIoU=0.6173, F1=0.6955, peak_mIoU=0.6173

Round=3, Labeled=365, mIoU=0.6173, F1=0.6955

## Round 4

Labeled Pool Size: 453

- Epoch 1: Loss=0.4504, mIoU=0.5107, F1=0.5426
- Epoch 2: Loss=0.2129, mIoU=0.5073, F1=0.5274
- Epoch 3: Loss=0.1664, mIoU=0.5123, F1=0.5389
- Epoch 4: Loss=0.1448, mIoU=0.5487, F1=0.5986
- Epoch 5: Loss=0.1396, mIoU=0.5836, F1=0.6503
- Epoch 6: Loss=0.1309, mIoU=0.5990, F1=0.6719
- Epoch 7: Loss=0.1258, mIoU=0.6222, F1=0.7028
- Epoch 8: Loss=0.1161, mIoU=0.6376, F1=0.7211
- Epoch 9: Loss=0.1123, mIoU=0.5695, F1=0.6306
- Epoch 10: Loss=0.1100, mIoU=0.6131, F1=0.6902

本轮结果: Round=4, Labeled=453, Selection=best_val (epoch=8), mIoU=0.6376, F1=0.7211, peak_mIoU=0.6376

Round=4, Labeled=453, mIoU=0.6376, F1=0.7211

## Round 5

Labeled Pool Size: 541

- Epoch 1: Loss=0.3870, mIoU=0.6093, F1=0.6854
- Epoch 2: Loss=0.1899, mIoU=0.6024, F1=0.6767
- Epoch 3: Loss=0.1572, mIoU=0.5223, F1=0.5532
- Epoch 4: Loss=0.1397, mIoU=0.6494, F1=0.7344
- Epoch 5: Loss=0.1297, mIoU=0.5547, F1=0.6071
- Epoch 6: Loss=0.1221, mIoU=0.6080, F1=0.6835
- Epoch 7: Loss=0.1192, mIoU=0.6568, F1=0.7430
- Epoch 8: Loss=0.1149, mIoU=0.6675, F1=0.7556
- Epoch 9: Loss=0.1077, mIoU=0.6862, F1=0.7753
- Epoch 10: Loss=0.1060, mIoU=0.6836, F1=0.7734

本轮结果: Round=5, Labeled=541, Selection=best_val (epoch=9), mIoU=0.6862, F1=0.7753, peak_mIoU=0.6862

Round=5, Labeled=541, mIoU=0.6862, F1=0.7753

## Round 6

Labeled Pool Size: 629

- Epoch 1: Loss=0.3932, mIoU=0.5059, F1=0.5297
- Epoch 2: Loss=0.1843, mIoU=0.5132, F1=0.5371
- Epoch 3: Loss=0.1521, mIoU=0.5268, F1=0.5615
- Epoch 4: Loss=0.1396, mIoU=0.5739, F1=0.6364
- Epoch 5: Loss=0.1301, mIoU=0.6610, F1=0.7475
- Epoch 6: Loss=0.1224, mIoU=0.6372, F1=0.7204
- Epoch 7: Loss=0.1151, mIoU=0.5926, F1=0.6632
- Epoch 8: Loss=0.1121, mIoU=0.6374, F1=0.7228
- Epoch 9: Loss=0.1083, mIoU=0.6215, F1=0.7013
- Epoch 10: Loss=0.1029, mIoU=0.5850, F1=0.6524

本轮结果: Round=6, Labeled=629, Selection=best_val (epoch=5), mIoU=0.6610, F1=0.7475, peak_mIoU=0.6610

Round=6, Labeled=629, mIoU=0.6610, F1=0.7475

## Round 7

Labeled Pool Size: 717

- Epoch 1: Loss=0.3846, mIoU=0.5752, F1=0.6386
- Epoch 2: Loss=0.1723, mIoU=0.6579, F1=0.7441
- Epoch 3: Loss=0.1455, mIoU=0.6858, F1=0.7747
- Epoch 4: Loss=0.1319, mIoU=0.6254, F1=0.7057
- Epoch 5: Loss=0.1218, mIoU=0.6623, F1=0.7493
- Epoch 6: Loss=0.1160, mIoU=0.6297, F1=0.7109
- Epoch 7: Loss=0.1137, mIoU=0.6788, F1=0.7668
- Epoch 8: Loss=0.1085, mIoU=0.6829, F1=0.7725
- Epoch 9: Loss=0.1034, mIoU=0.6586, F1=0.7482
- Epoch 10: Loss=0.1006, mIoU=0.6666, F1=0.7555

本轮结果: Round=7, Labeled=717, Selection=best_val (epoch=3), mIoU=0.6858, F1=0.7747, peak_mIoU=0.6858

Round=7, Labeled=717, mIoU=0.6858, F1=0.7747

## Round 8

Labeled Pool Size: 805

- Epoch 1: Loss=0.2711, mIoU=0.5368, F1=0.5781
- Epoch 2: Loss=0.1483, mIoU=0.5596, F1=0.6148
- Epoch 3: Loss=0.1274, mIoU=0.6540, F1=0.7416
- Epoch 4: Loss=0.1159, mIoU=0.5705, F1=0.6315
- Epoch 5: Loss=0.1099, mIoU=0.6769, F1=0.7655
- Epoch 6: Loss=0.1046, mIoU=0.6690, F1=0.7567
- Epoch 7: Loss=0.1004, mIoU=0.6770, F1=0.7651
- Epoch 8: Loss=0.0971, mIoU=0.6978, F1=0.7869
- Epoch 9: Loss=0.0944, mIoU=0.6803, F1=0.7694
- Epoch 10: Loss=0.0918, mIoU=0.6963, F1=0.7852

本轮结果: Round=8, Labeled=805, Selection=best_val (epoch=8), mIoU=0.6978, F1=0.7869, peak_mIoU=0.6978

Round=8, Labeled=805, mIoU=0.6978, F1=0.7869

## Round 9

Labeled Pool Size: 893

- Epoch 1: Loss=0.2450, mIoU=0.5001, F1=0.5127
- Epoch 2: Loss=0.1392, mIoU=0.5206, F1=0.5502
- Epoch 3: Loss=0.1247, mIoU=0.5671, F1=0.6265
- Epoch 4: Loss=0.1136, mIoU=0.6835, F1=0.7727
- Epoch 5: Loss=0.1052, mIoU=0.5813, F1=0.6474
- Epoch 6: Loss=0.1028, mIoU=0.5725, F1=0.6343
- Epoch 7: Loss=0.1002, mIoU=0.6602, F1=0.7474
- Epoch 8: Loss=0.0947, mIoU=0.6107, F1=0.6873
- Epoch 9: Loss=0.0924, mIoU=0.6512, F1=0.7378
- Epoch 10: Loss=0.0870, mIoU=0.6392, F1=0.7235

本轮结果: Round=9, Labeled=893, Selection=best_val (epoch=4), mIoU=0.6835, F1=0.7727, peak_mIoU=0.6835

Round=9, Labeled=893, mIoU=0.6835, F1=0.7727

## Round 10

Labeled Pool Size: 981

- Epoch 1: Loss=0.2326, mIoU=0.5305, F1=0.5676
- Epoch 2: Loss=0.1319, mIoU=0.5278, F1=0.5629
- Epoch 3: Loss=0.1154, mIoU=0.6133, F1=0.6916
- Epoch 4: Loss=0.1061, mIoU=0.6625, F1=0.7516
- Epoch 5: Loss=0.1033, mIoU=0.6091, F1=0.6849
- Epoch 6: Loss=0.0954, mIoU=0.6990, F1=0.7878
- Epoch 7: Loss=0.0942, mIoU=0.7077, F1=0.7965
- Epoch 8: Loss=0.0889, mIoU=0.6406, F1=0.7244
- Epoch 9: Loss=0.0884, mIoU=0.6860, F1=0.7746
- Epoch 10: Loss=0.0866, mIoU=0.6736, F1=0.7619

本轮结果: Round=10, Labeled=981, Selection=best_val (epoch=7), mIoU=0.7077, F1=0.7965, peak_mIoU=0.7077

Round=10, Labeled=981, mIoU=0.7077, F1=0.7965

## Round 11

Labeled Pool Size: 1069

- Epoch 1: Loss=0.2751, mIoU=0.5146, F1=0.5397
- Epoch 2: Loss=0.1333, mIoU=0.5390, F1=0.5818
- Epoch 3: Loss=0.1174, mIoU=0.5740, F1=0.6364
- Epoch 4: Loss=0.1044, mIoU=0.6258, F1=0.7062
- Epoch 5: Loss=0.0988, mIoU=0.6417, F1=0.7255
- Epoch 6: Loss=0.0944, mIoU=0.6546, F1=0.7405
- Epoch 7: Loss=0.0944, mIoU=0.6862, F1=0.7751
- Epoch 8: Loss=0.0906, mIoU=0.5725, F1=0.6341
- Epoch 9: Loss=0.0868, mIoU=0.6710, F1=0.7591
- Epoch 10: Loss=0.0815, mIoU=0.6928, F1=0.7822

本轮结果: Round=11, Labeled=1069, Selection=best_val (epoch=10), mIoU=0.6928, F1=0.7822, peak_mIoU=0.6928

Round=11, Labeled=1069, mIoU=0.6928, F1=0.7822

## Round 12

Labeled Pool Size: 1157

- Epoch 1: Loss=0.2024, mIoU=0.6495, F1=0.7347
- Epoch 2: Loss=0.1184, mIoU=0.6177, F1=0.6963
- Epoch 3: Loss=0.1077, mIoU=0.6839, F1=0.7724
- Epoch 4: Loss=0.0990, mIoU=0.6330, F1=0.7150
- Epoch 5: Loss=0.0934, mIoU=0.6848, F1=0.7735
- Epoch 6: Loss=0.0876, mIoU=0.6959, F1=0.7854
- Epoch 7: Loss=0.0843, mIoU=0.6675, F1=0.7549
- Epoch 8: Loss=0.0818, mIoU=0.6611, F1=0.7479
- Epoch 9: Loss=0.0787, mIoU=0.6165, F1=0.6948
- Epoch 10: Loss=0.0753, mIoU=0.6855, F1=0.7748

本轮结果: Round=12, Labeled=1157, Selection=best_val (epoch=6), mIoU=0.6959, F1=0.7854, peak_mIoU=0.6959

Round=12, Labeled=1157, mIoU=0.6959, F1=0.7854

## Round 13

Labeled Pool Size: 1245

- Epoch 1: Loss=0.2462, mIoU=0.5049, F1=0.5216
- Epoch 2: Loss=0.1202, mIoU=0.5728, F1=0.6346
- Epoch 3: Loss=0.1071, mIoU=0.6502, F1=0.7352
- Epoch 4: Loss=0.0990, mIoU=0.7003, F1=0.7896
- Epoch 5: Loss=0.0922, mIoU=0.6593, F1=0.7465
- Epoch 6: Loss=0.0892, mIoU=0.6773, F1=0.7669
- Epoch 7: Loss=0.0841, mIoU=0.6439, F1=0.7281
- Epoch 8: Loss=0.0804, mIoU=0.6509, F1=0.7362
- Epoch 9: Loss=0.0787, mIoU=0.6891, F1=0.7783
- Epoch 10: Loss=0.0758, mIoU=0.6698, F1=0.7587

本轮结果: Round=13, Labeled=1245, Selection=best_val (epoch=4), mIoU=0.7003, F1=0.7896, peak_mIoU=0.7003

Round=13, Labeled=1245, mIoU=0.7003, F1=0.7896

## Round 14

Labeled Pool Size: 1333

- Epoch 1: Loss=0.2283, mIoU=0.5057, F1=0.5232
- Epoch 2: Loss=0.1133, mIoU=0.6688, F1=0.7561
- Epoch 3: Loss=0.1023, mIoU=0.6424, F1=0.7266
- Epoch 4: Loss=0.0918, mIoU=0.6691, F1=0.7568
- Epoch 5: Loss=0.0873, mIoU=0.6857, F1=0.7746
- Epoch 6: Loss=0.0849, mIoU=0.6329, F1=0.7151
- Epoch 7: Loss=0.0810, mIoU=0.6879, F1=0.7769
- Epoch 8: Loss=0.0771, mIoU=0.6882, F1=0.7773
- Epoch 9: Loss=0.0747, mIoU=0.6793, F1=0.7677
- Epoch 10: Loss=0.0732, mIoU=0.7192, F1=0.8074

本轮结果: Round=14, Labeled=1333, Selection=best_val (epoch=10), mIoU=0.7192, F1=0.8074, peak_mIoU=0.7192

Round=14, Labeled=1333, mIoU=0.7192, F1=0.8074

## Round 15

Labeled Pool Size: 1421

- Epoch 1: Loss=0.2212, mIoU=0.5144, F1=0.5393
- Epoch 2: Loss=0.1109, mIoU=0.5604, F1=0.6158
- Epoch 3: Loss=0.0954, mIoU=0.6390, F1=0.7227
- Epoch 4: Loss=0.0908, mIoU=0.6564, F1=0.7445
- Epoch 5: Loss=0.0856, mIoU=0.6977, F1=0.7870
- Epoch 6: Loss=0.0808, mIoU=0.6423, F1=0.7270
- Epoch 7: Loss=0.0780, mIoU=0.6371, F1=0.7209
- Epoch 8: Loss=0.0736, mIoU=0.7565, F1=0.8409
- Epoch 9: Loss=0.0722, mIoU=0.6894, F1=0.7785
- Epoch 10: Loss=0.0703, mIoU=0.6689, F1=0.7586

本轮结果: Round=15, Labeled=1421, Selection=best_val (epoch=8), mIoU=0.7565, F1=0.8409, peak_mIoU=0.7565

Round=15, Labeled=1421, mIoU=0.7565, F1=0.8409

## Round 16

Labeled Pool Size: 1509

- Test-only mode: Round 16 loads Round 15 best_val checkpoint (epoch=8, val_mIoU=0.7565, val_F1=0.8409)

本轮结果: Round=16, Labeled=1509, Selection=prev_round_best_val (source_round=15, epoch=8), mIoU=0.7565, F1=0.8409, peak_mIoU=0.7565

Round=16, Labeled=1509, mIoU=0.7565, F1=0.8409


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421, 1509]
ALC(基于每轮选模val mIoU): 0.5960
最后一轮选模 mIoU(val): 0.7565354017069874
最后一轮选模 F1(val): 0.8408967000049128
最终报告 mIoU(test): 0.7297256462962876
最终报告 F1(test): 0.8175667718546975
最终输出 mIoU: 0.7297 (source=final_report)
最终输出 F1: 0.8176 (source=final_report)
最终 Test Split: test
最终 Report: {'loss': 0.03671089896117337, 'mIoU': 0.7297256462962876, 'f1_score': 0.8175667718546975}
