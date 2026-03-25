# 实验日志

实验名称: full_model_B_lambda_agent
描述: 对照B / AAL-SD (Agent-λ)：与 full_model_A_lambda_policy 完全一致，唯一区别：λ 由 LLM Agent 通过 set_lambda 显式决定，而非 policy 自动填充
开始时间: 2026-03-25T01:14:04.943877

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.4656, mIoU=0.5038, F1=0.5289
- Epoch 2: Loss=0.2546, mIoU=0.5181, F1=0.5463
- Epoch 3: Loss=0.1628, mIoU=0.5958, F1=0.6696
- Epoch 4: Loss=0.1177, mIoU=0.5622, F1=0.6188
- Epoch 5: Loss=0.0996, mIoU=0.5326, F1=0.5711
- Epoch 4: Loss=0.0907, mIoU=0.6064, F1=0.6815
- Epoch 6: Loss=0.0794, mIoU=0.5736, F1=0.6358
- Epoch 7: Loss=0.0703, mIoU=0.6054, F1=0.6801
- Epoch 8: Loss=0.0632, mIoU=0.5221, F1=0.5529
- Epoch 9: Loss=0.0602, mIoU=0.5361, F1=0.5770
- Epoch 10: Loss=0.0565, mIoU=0.5281, F1=0.5639

本轮结果: Round=1, Labeled=189, Selection=best_val (epoch=7), mIoU=0.6054, F1=0.6801, peak_mIoU=0.6054

Round=1, Labeled=189, mIoU=0.6054, F1=0.6801

- Epoch 5: Loss=0.0850, mIoU=0.7106, F1=0.7998
- Epoch 6: Loss=0.0803, mIoU=0.6368, F1=0.7198
- Epoch 7: Loss=0.0778, mIoU=0.6320, F1=0.7157
- Epoch 8: Loss=0.0751, mIoU=0.7097, F1=0.7990
- Epoch 9: Loss=0.0721, mIoU=0.6845, F1=0.7743
- Epoch 10: Loss=0.0693, mIoU=0.6357, F1=0.7189

本轮结果: Round=14, Labeled=1333, Selection=best_val (epoch=5), mIoU=0.7106, F1=0.7998, peak_mIoU=0.7106

Round=14, Labeled=1333, mIoU=0.7106, F1=0.7998

## Round 2

Labeled Pool Size: 277

- Epoch 1: Loss=0.4324, mIoU=0.5071, F1=0.5274
- Epoch 2: Loss=0.2162, mIoU=0.4940, F1=0.5008
- Epoch 3: Loss=0.1469, mIoU=0.5015, F1=0.5152
- Epoch 4: Loss=0.1161, mIoU=0.5067, F1=0.5251
- Epoch 5: Loss=0.1046, mIoU=0.5497, F1=0.5991
- Epoch 6: Loss=0.0937, mIoU=0.4980, F1=0.5084
- Epoch 7: Loss=0.0892, mIoU=0.5283, F1=0.5636
- Epoch 8: Loss=0.0905, mIoU=0.5150, F1=0.5404
- Epoch 9: Loss=0.0908, mIoU=0.5313, F1=0.5690
- Epoch 10: Loss=0.0889, mIoU=0.6061, F1=0.6818

本轮结果: Round=2, Labeled=277, Selection=best_val (epoch=10), mIoU=0.6061, F1=0.6818, peak_mIoU=0.6061

Round=2, Labeled=277, mIoU=0.6061, F1=0.6818

## Round 15

Labeled Pool Size: 1421

- Epoch 1: Loss=0.2249, mIoU=0.5388, F1=0.5814
- Epoch 2: Loss=0.1079, mIoU=0.6283, F1=0.7101
- Epoch 3: Loss=0.0952, mIoU=0.6907, F1=0.7797
- Epoch 4: Loss=0.0891, mIoU=0.6718, F1=0.7609
- Epoch 5: Loss=0.0844, mIoU=0.6430, F1=0.7271
## Round 3

Labeled Pool Size: 365

- Epoch 1: Loss=0.3965, mIoU=0.4969, F1=0.5065
- Epoch 2: Loss=0.2083, mIoU=0.5047, F1=0.5212
- Epoch 6: Loss=0.0792, mIoU=0.6937, F1=0.7826
- Epoch 3: Loss=0.1601, mIoU=0.5069, F1=0.5256
- Epoch 4: Loss=0.1409, mIoU=0.5109, F1=0.5329
- Epoch 5: Loss=0.1281, mIoU=0.5068, F1=0.5252
- Epoch 6: Loss=0.1169, mIoU=0.5212, F1=0.5512
- Epoch 7: Loss=0.0758, mIoU=0.6786, F1=0.7674
- Epoch 7: Loss=0.1081, mIoU=0.5727, F1=0.6344
- Epoch 8: Loss=0.1043, mIoU=0.5548, F1=0.6072
- Epoch 9: Loss=0.1026, mIoU=0.5523, F1=0.6033
- Epoch 10: Loss=0.0993, mIoU=0.5384, F1=0.5808

本轮结果: Round=3, Labeled=365, Selection=best_val (epoch=7), mIoU=0.5727, F1=0.6344, peak_mIoU=0.5727

Round=3, Labeled=365, mIoU=0.5727, F1=0.6344

- Epoch 8: Loss=0.0728, mIoU=0.6482, F1=0.7334
- Epoch 9: Loss=0.0703, mIoU=0.6578, F1=0.7448
- Epoch 10: Loss=0.0686, mIoU=0.6742, F1=0.7629

本轮结果: Round=15, Labeled=1421, Selection=best_val (epoch=6), mIoU=0.6937, F1=0.7826, peak_mIoU=0.6937

Round=15, Labeled=1421, mIoU=0.6937, F1=0.7826

## Round 4

Labeled Pool Size: 453

- Epoch 1: Loss=0.4512, mIoU=0.5069, F1=0.5352
- Epoch 2: Loss=0.2070, mIoU=0.5206, F1=0.5572
- Epoch 3: Loss=0.1591, mIoU=0.5723, F1=0.6338
## Round 16

Labeled Pool Size: 1509

- Test-only mode: Round 16 loads Round 15 best_val checkpoint (epoch=6, val_mIoU=0.6937, val_F1=0.7826)

本轮结果: Round=16, Labeled=1509, Selection=prev_round_best_val (source_round=15, epoch=6), mIoU=0.6937, F1=0.7826, peak_mIoU=0.6937

Round=16, Labeled=1509, mIoU=0.6937, F1=0.7826


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421, 1509]
ALC(基于每轮选模val mIoU): 0.5955
最后一轮选模 mIoU(val): 0.6936672076604814
最后一轮选模 F1(val): 0.782633058486983
最终报告 mIoU(test): 0.6787576373640236
最终报告 F1(test): 0.7678479857548357
最终输出 mIoU: 0.6788 (source=final_report)
最终输出 F1: 0.7678 (source=final_report)
最终 Test Split: test
最终 Report: {'loss': 0.04816849654831458, 'mIoU': 0.6787576373640236, 'f1_score': 0.7678479857548357}
- Epoch 4: Loss=0.1372, mIoU=0.6392, F1=0.7224
- Epoch 5: Loss=0.1316, mIoU=0.5402, F1=0.5838
- Epoch 6: Loss=0.1240, mIoU=0.6590, F1=0.7454
- Epoch 7: Loss=0.1155, mIoU=0.6171, F1=0.6952
- Epoch 8: Loss=0.1153, mIoU=0.5682, F1=0.6279
- Epoch 9: Loss=0.1099, mIoU=0.5911, F1=0.6609
- Epoch 10: Loss=0.1068, mIoU=0.5888, F1=0.6574

本轮结果: Round=4, Labeled=453, Selection=best_val (epoch=6), mIoU=0.6590, F1=0.7454, peak_mIoU=0.6590

Round=4, Labeled=453, mIoU=0.6590, F1=0.7454

## Round 5

Labeled Pool Size: 541

- Epoch 1: Loss=0.3864, mIoU=0.5172, F1=0.5444
- Epoch 2: Loss=0.1887, mIoU=0.5615, F1=0.6179
- Epoch 3: Loss=0.1530, mIoU=0.5676, F1=0.6269
- Epoch 4: Loss=0.1373, mIoU=0.6145, F1=0.6921
- Epoch 5: Loss=0.1265, mIoU=0.6374, F1=0.7205
- Epoch 6: Loss=0.1177, mIoU=0.6479, F1=0.7333
- Epoch 7: Loss=0.1144, mIoU=0.6192, F1=0.6993
- Epoch 8: Loss=0.1122, mIoU=0.6576, F1=0.7450
- Epoch 9: Loss=0.1040, mIoU=0.6407, F1=0.7243
- Epoch 10: Loss=0.1027, mIoU=0.6755, F1=0.7640

本轮结果: Round=5, Labeled=541, Selection=best_val (epoch=10), mIoU=0.6755, F1=0.7640, peak_mIoU=0.6755

Round=5, Labeled=541, mIoU=0.6755, F1=0.7640

## Round 6

Labeled Pool Size: 629

- Epoch 1: Loss=0.4003, mIoU=0.5119, F1=0.5378
- Epoch 2: Loss=0.1811, mIoU=0.5939, F1=0.6647
- Epoch 3: Loss=0.1466, mIoU=0.5844, F1=0.6514
- Epoch 4: Loss=0.1375, mIoU=0.5724, F1=0.6346
- Epoch 5: Loss=0.1295, mIoU=0.6716, F1=0.7591
- Epoch 6: Loss=0.1179, mIoU=0.5955, F1=0.6667
- Epoch 7: Loss=0.1137, mIoU=0.6641, F1=0.7508
- Epoch 8: Loss=0.1093, mIoU=0.6926, F1=0.7811
- Epoch 9: Loss=0.1044, mIoU=0.6756, F1=0.7645
- Epoch 10: Loss=0.1007, mIoU=0.6684, F1=0.7557

本轮结果: Round=6, Labeled=629, Selection=best_val (epoch=8), mIoU=0.6926, F1=0.7811, peak_mIoU=0.6926

Round=6, Labeled=629, mIoU=0.6926, F1=0.7811

## Round 7

Labeled Pool Size: 717

- Epoch 1: Loss=0.3663, mIoU=0.5144, F1=0.5398
- Epoch 2: Loss=0.1652, mIoU=0.5341, F1=0.5738
- Epoch 3: Loss=0.1452, mIoU=0.6560, F1=0.7424
- Epoch 4: Loss=0.1319, mIoU=0.6544, F1=0.7399
- Epoch 5: Loss=0.1256, mIoU=0.6074, F1=0.6830
- Epoch 6: Loss=0.1164, mIoU=0.6253, F1=0.7057
- Epoch 7: Loss=0.1107, mIoU=0.6647, F1=0.7533
- Epoch 8: Loss=0.1099, mIoU=0.6863, F1=0.7752
- Epoch 9: Loss=0.1032, mIoU=0.5965, F1=0.6684
- Epoch 10: Loss=0.0980, mIoU=0.6643, F1=0.7535

本轮结果: Round=7, Labeled=717, Selection=best_val (epoch=8), mIoU=0.6863, F1=0.7752, peak_mIoU=0.6863

Round=7, Labeled=717, mIoU=0.6863, F1=0.7752

## Round 8

Labeled Pool Size: 805

- Epoch 1: Loss=0.2732, mIoU=0.5583, F1=0.6150
- Epoch 2: Loss=0.1496, mIoU=0.5858, F1=0.6534
- Epoch 3: Loss=0.1283, mIoU=0.6533, F1=0.7389
- Epoch 4: Loss=0.1203, mIoU=0.6567, F1=0.7429
- Epoch 5: Loss=0.1143, mIoU=0.6704, F1=0.7593
- Epoch 6: Loss=0.1067, mIoU=0.6921, F1=0.7810
- Epoch 7: Loss=0.1046, mIoU=0.6673, F1=0.7548
- Epoch 8: Loss=0.1018, mIoU=0.6340, F1=0.7212
- Epoch 9: Loss=0.0979, mIoU=0.6681, F1=0.7573
- Epoch 10: Loss=0.0940, mIoU=0.6355, F1=0.7184

本轮结果: Round=8, Labeled=805, Selection=best_val (epoch=6), mIoU=0.6921, F1=0.7810, peak_mIoU=0.6921

Round=8, Labeled=805, mIoU=0.6921, F1=0.7810

## Round 9

Labeled Pool Size: 893

- Epoch 1: Loss=0.2393, mIoU=0.5102, F1=0.5319
- Epoch 2: Loss=0.1387, mIoU=0.6817, F1=0.7702
- Epoch 3: Loss=0.1250, mIoU=0.5718, F1=0.6329
- Epoch 4: Loss=0.1138, mIoU=0.7019, F1=0.7911
- Epoch 5: Loss=0.1069, mIoU=0.6933, F1=0.7829
- Epoch 6: Loss=0.1048, mIoU=0.6983, F1=0.7872
- Epoch 7: Loss=0.1029, mIoU=0.6966, F1=0.7854
- Epoch 8: Loss=0.0966, mIoU=0.6824, F1=0.7719
- Epoch 9: Loss=0.0932, mIoU=0.7014, F1=0.7908
- Epoch 10: Loss=0.0905, mIoU=0.7054, F1=0.7946

本轮结果: Round=9, Labeled=893, Selection=best_val (epoch=10), mIoU=0.7054, F1=0.7946, peak_mIoU=0.7054

Round=9, Labeled=893, mIoU=0.7054, F1=0.7946

## Round 10

Labeled Pool Size: 981

- Epoch 1: Loss=0.2330, mIoU=0.5611, F1=0.6176
- Epoch 2: Loss=0.1364, mIoU=0.6406, F1=0.7242
- Epoch 3: Loss=0.1177, mIoU=0.6722, F1=0.7608
- Epoch 4: Loss=0.1096, mIoU=0.6854, F1=0.7741
- Epoch 5: Loss=0.1034, mIoU=0.6821, F1=0.7704
- Epoch 6: Loss=0.0993, mIoU=0.7136, F1=0.8023
- Epoch 7: Loss=0.0944, mIoU=0.6634, F1=0.7518
- Epoch 8: Loss=0.0902, mIoU=0.6566, F1=0.7432
- Epoch 9: Loss=0.0854, mIoU=0.7036, F1=0.7929
- Epoch 10: Loss=0.0846, mIoU=0.7039, F1=0.7934

本轮结果: Round=10, Labeled=981, Selection=best_val (epoch=6), mIoU=0.7136, F1=0.8023, peak_mIoU=0.7136

Round=10, Labeled=981, mIoU=0.7136, F1=0.8023

## Round 11

Labeled Pool Size: 1069

- Epoch 1: Loss=0.2680, mIoU=0.5513, F1=0.6017
- Epoch 2: Loss=0.1333, mIoU=0.5825, F1=0.6486
- Epoch 3: Loss=0.1147, mIoU=0.6048, F1=0.6793
- Epoch 4: Loss=0.1077, mIoU=0.5932, F1=0.6637
- Epoch 5: Loss=0.1022, mIoU=0.6801, F1=0.7696
- Epoch 6: Loss=0.0957, mIoU=0.6857, F1=0.7744
- Epoch 7: Loss=0.0929, mIoU=0.6922, F1=0.7809
- Epoch 8: Loss=0.0898, mIoU=0.6582, F1=0.7444
- Epoch 9: Loss=0.0859, mIoU=0.6833, F1=0.7722
- Epoch 10: Loss=0.0824, mIoU=0.6695, F1=0.7580

本轮结果: Round=11, Labeled=1069, Selection=best_val (epoch=7), mIoU=0.6922, F1=0.7809, peak_mIoU=0.6922

Round=11, Labeled=1069, mIoU=0.6922, F1=0.7809

## Round 12

Labeled Pool Size: 1157

- Epoch 1: Loss=0.2026, mIoU=0.5046, F1=0.5212
- Epoch 2: Loss=0.1199, mIoU=0.6250, F1=0.7051
- Epoch 3: Loss=0.1088, mIoU=0.6835, F1=0.7719
- Epoch 4: Loss=0.1003, mIoU=0.6029, F1=0.6767
- Epoch 5: Loss=0.0933, mIoU=0.6984, F1=0.7876
- Epoch 6: Loss=0.0904, mIoU=0.7013, F1=0.7900
- Epoch 7: Loss=0.0873, mIoU=0.6683, F1=0.7556
- Epoch 8: Loss=0.0829, mIoU=0.6775, F1=0.7673
- Epoch 9: Loss=0.0808, mIoU=0.6600, F1=0.7467
- Epoch 10: Loss=0.0779, mIoU=0.7206, F1=0.8091

本轮结果: Round=12, Labeled=1157, Selection=best_val (epoch=10), mIoU=0.7206, F1=0.8091, peak_mIoU=0.7206

Round=12, Labeled=1157, mIoU=0.7206, F1=0.8091

## Round 13

Labeled Pool Size: 1245

- Epoch 1: Loss=0.2388, mIoU=0.5422, F1=0.5875
- Epoch 2: Loss=0.1216, mIoU=0.6186, F1=0.6973
- Epoch 3: Loss=0.1069, mIoU=0.6635, F1=0.7503
- Epoch 4: Loss=0.1006, mIoU=0.6709, F1=0.7584
- Epoch 5: Loss=0.0943, mIoU=0.5734, F1=0.6356
- Epoch 6: Loss=0.0940, mIoU=0.6907, F1=0.7794
- Epoch 7: Loss=0.0893, mIoU=0.6635, F1=0.7506
- Epoch 8: Loss=0.0860, mIoU=0.7041, F1=0.7928
- Epoch 9: Loss=0.0831, mIoU=0.7125, F1=0.8012
- Epoch 10: Loss=0.0796, mIoU=0.7023, F1=0.7915

本轮结果: Round=13, Labeled=1245, Selection=best_val (epoch=9), mIoU=0.7125, F1=0.8012, peak_mIoU=0.7125

Round=13, Labeled=1245, mIoU=0.7125, F1=0.8012

## Round 14

Labeled Pool Size: 1333

- Epoch 1: Loss=0.2325, mIoU=0.5539, F1=0.6059
- Epoch 2: Loss=0.1153, mIoU=0.5400, F1=0.5836
- Epoch 3: Loss=0.0991, mIoU=0.6224, F1=0.7029
- Epoch 4: Loss=0.0923, mIoU=0.6006, F1=0.6737
- Epoch 5: Loss=0.0869, mIoU=0.5655, F1=0.6237
- Epoch 6: Loss=0.0825, mIoU=0.7304, F1=0.8182
- Epoch 7: Loss=0.0804, mIoU=0.6912, F1=0.7809
- Epoch 8: Loss=0.0763, mIoU=0.7143, F1=0.8029
- Epoch 9: Loss=0.0740, mIoU=0.6850, F1=0.7735
- Epoch 10: Loss=0.0716, mIoU=0.6586, F1=0.7453

本轮结果: Round=14, Labeled=1333, Selection=best_val (epoch=6), mIoU=0.7304, F1=0.8182, peak_mIoU=0.7304

Round=14, Labeled=1333, mIoU=0.7304, F1=0.8182

## Round 15

Labeled Pool Size: 1421

- Epoch 1: Loss=0.2231, mIoU=0.5406, F1=0.5846
- Epoch 2: Loss=0.1088, mIoU=0.5871, F1=0.6556
- Epoch 3: Loss=0.0965, mIoU=0.5916, F1=0.6615
- Epoch 4: Loss=0.0917, mIoU=0.6754, F1=0.7632
- Epoch 5: Loss=0.0853, mIoU=0.6536, F1=0.7391
- Epoch 6: Loss=0.0815, mIoU=0.6827, F1=0.7712
- Epoch 7: Loss=0.0774, mIoU=0.6719, F1=0.7607
- Epoch 8: Loss=0.0750, mIoU=0.6709, F1=0.7586
- Epoch 9: Loss=0.0714, mIoU=0.7252, F1=0.8131
- Epoch 10: Loss=0.0729, mIoU=0.6691, F1=0.7566

本轮结果: Round=15, Labeled=1421, Selection=best_val (epoch=9), mIoU=0.7252, F1=0.8131, peak_mIoU=0.7252

Round=15, Labeled=1421, mIoU=0.7252, F1=0.8131

## Round 16

Labeled Pool Size: 1509

- Test-only mode: Round 16 loads Round 15 best_val checkpoint (epoch=9, val_mIoU=0.7252, val_F1=0.8131)

本轮结果: Round=16, Labeled=1509, Selection=prev_round_best_val (source_round=15, epoch=9), mIoU=0.7252, F1=0.8131, peak_mIoU=0.7252

Round=16, Labeled=1509, mIoU=0.7252, F1=0.8131


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421, 1509]
ALC(基于每轮选模val mIoU): 0.5985
最后一轮选模 mIoU(val): 0.7251673156360965
最后一轮选模 F1(val): 0.8130981186314028
最终报告 mIoU(test): 0.7192500915997737
最终报告 F1(test): 0.8080035730205696
最终输出 mIoU: 0.7193 (source=final_report)
最终输出 F1: 0.8080 (source=final_report)
最终 Test Split: test
最终 Report: {'loss': 0.03836753529729322, 'mIoU': 0.7192500915997737, 'f1_score': 0.8080035730205696}
