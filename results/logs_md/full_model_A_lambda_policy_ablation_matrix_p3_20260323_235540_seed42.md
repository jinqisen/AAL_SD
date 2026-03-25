# 实验日志

实验名称: full_model_A_lambda_policy
描述: 对照A / AAL-SD (Full)：基于 auto_opt_iter15_cand2_02（warmup+risk闭环 + 后期ramp + U-guardrail；train_holdout grad_probe）
开始时间: 2026-03-23T23:55:47.871109

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.5156, mIoU=0.5161, F1=0.5497
- Epoch 2: Loss=0.2782, mIoU=0.5509, F1=0.6022
- Epoch 3: Loss=0.1738, mIoU=0.5672, F1=0.6289
- Epoch 4: Loss=0.1287, mIoU=0.5756, F1=0.6421
- Epoch 5: Loss=0.1054, mIoU=0.5508, F1=0.6012
- Epoch 6: Loss=0.0953, mIoU=0.5188, F1=0.5474
- Epoch 7: Loss=0.0823, mIoU=0.5256, F1=0.5592
- Epoch 8: Loss=0.0734, mIoU=0.5722, F1=0.6342
- Epoch 9: Loss=0.0687, mIoU=0.5340, F1=0.5738
- Epoch 10: Loss=0.0634, mIoU=0.5443, F1=0.5910

本轮结果: Round=1, Labeled=189, Selection=best_val (epoch=4), mIoU=0.5756, F1=0.6421, peak_mIoU=0.5756

Round=1, Labeled=189, mIoU=0.5756, F1=0.6421

## Round 2

Labeled Pool Size: 277

- Epoch 1: Loss=0.3313, mIoU=0.4939, F1=0.5005
- Epoch 2: Loss=0.1897, mIoU=0.5012, F1=0.5147
- Epoch 3: Loss=0.1450, mIoU=0.5489, F1=0.5979
- Epoch 4: Loss=0.1214, mIoU=0.5245, F1=0.5572
- Epoch 5: Loss=0.1156, mIoU=0.5307, F1=0.5680
- Epoch 6: Loss=0.1025, mIoU=0.5176, F1=0.5451
- Epoch 7: Loss=0.0960, mIoU=0.5286, F1=0.5645
- Epoch 8: Loss=0.0924, mIoU=0.6396, F1=0.7234
- Epoch 9: Loss=0.0903, mIoU=0.5356, F1=0.5762
- Epoch 10: Loss=0.0859, mIoU=0.5695, F1=0.6298

本轮结果: Round=2, Labeled=277, Selection=best_val (epoch=8), mIoU=0.6396, F1=0.7234, peak_mIoU=0.6396

Round=2, Labeled=277, mIoU=0.6396, F1=0.7234

## Round 3

Labeled Pool Size: 365

- Epoch 1: Loss=0.4013, mIoU=0.5145, F1=0.5408
- Epoch 2: Loss=0.2030, mIoU=0.5057, F1=0.5232
- Epoch 3: Loss=0.1522, mIoU=0.5690, F1=0.6289
- Epoch 4: Loss=0.1346, mIoU=0.5390, F1=0.5821
- Epoch 5: Loss=0.1224, mIoU=0.6233, F1=0.7031
- Epoch 6: Loss=0.1124, mIoU=0.6450, F1=0.7294
- Epoch 7: Loss=0.1089, mIoU=0.6722, F1=0.7599
- Epoch 8: Loss=0.1001, mIoU=0.6455, F1=0.7299
- Epoch 9: Loss=0.0970, mIoU=0.5763, F1=0.6399
- Epoch 10: Loss=0.0958, mIoU=0.6094, F1=0.6853

本轮结果: Round=3, Labeled=365, Selection=best_val (epoch=7), mIoU=0.6722, F1=0.7599, peak_mIoU=0.6722

Round=3, Labeled=365, mIoU=0.6722, F1=0.7599

## Round 4

Labeled Pool Size: 453

- Epoch 1: Loss=0.3789, mIoU=0.5064, F1=0.5244
- Epoch 2: Loss=0.1960, mIoU=0.6058, F1=0.6805
- Epoch 3: Loss=0.1585, mIoU=0.5936, F1=0.6642
- Epoch 4: Loss=0.1453, mIoU=0.6357, F1=0.7182
- Epoch 5: Loss=0.1297, mIoU=0.6109, F1=0.6871
- Epoch 6: Loss=0.1253, mIoU=0.6815, F1=0.7699
- Epoch 7: Loss=0.1198, mIoU=0.6198, F1=0.6986
- Epoch 8: Loss=0.1141, mIoU=0.6903, F1=0.7795
- Epoch 9: Loss=0.1093, mIoU=0.6954, F1=0.7841
- Epoch 10: Loss=0.1038, mIoU=0.7023, F1=0.7914

本轮结果: Round=4, Labeled=453, Selection=best_val (epoch=10), mIoU=0.7023, F1=0.7914, peak_mIoU=0.7023

Round=4, Labeled=453, mIoU=0.7023, F1=0.7914

## Round 5

Labeled Pool Size: 541

- Epoch 1: Loss=0.3453, mIoU=0.5010, F1=0.5144
- Epoch 2: Loss=0.1831, mIoU=0.5419, F1=0.5865
- Epoch 3: Loss=0.1489, mIoU=0.5611, F1=0.6170
- Epoch 4: Loss=0.1357, mIoU=0.5772, F1=0.6410
- Epoch 5: Loss=0.1263, mIoU=0.6592, F1=0.7456
- Epoch 6: Loss=0.1202, mIoU=0.6192, F1=0.6982
- Epoch 7: Loss=0.1151, mIoU=0.6908, F1=0.7798
- Epoch 8: Loss=0.1094, mIoU=0.6866, F1=0.7753
- Epoch 9: Loss=0.1084, mIoU=0.6656, F1=0.7534
- Epoch 10: Loss=0.1030, mIoU=0.6461, F1=0.7308

本轮结果: Round=5, Labeled=541, Selection=best_val (epoch=7), mIoU=0.6908, F1=0.7798, peak_mIoU=0.6908

Round=5, Labeled=541, mIoU=0.6908, F1=0.7798

## Round 6

Labeled Pool Size: 629

- Epoch 1: Loss=0.3861, mIoU=0.5069, F1=0.5298
- Epoch 2: Loss=0.1762, mIoU=0.6160, F1=0.6938
- Epoch 3: Loss=0.1481, mIoU=0.6045, F1=0.6791
- Epoch 4: Loss=0.1333, mIoU=0.6496, F1=0.7348
- Epoch 5: Loss=0.1264, mIoU=0.7002, F1=0.7891
- Epoch 6: Loss=0.1219, mIoU=0.6869, F1=0.7754
- Epoch 7: Loss=0.1155, mIoU=0.6992, F1=0.7883
- Epoch 8: Loss=0.1097, mIoU=0.6530, F1=0.7386
- Epoch 9: Loss=0.1068, mIoU=0.6839, F1=0.7724
- Epoch 10: Loss=0.1030, mIoU=0.6534, F1=0.7391

本轮结果: Round=6, Labeled=629, Selection=best_val (epoch=5), mIoU=0.7002, F1=0.7891, peak_mIoU=0.7002

Round=6, Labeled=629, mIoU=0.7002, F1=0.7891

## Round 7

Labeled Pool Size: 717

- Epoch 1: Loss=0.3366, mIoU=0.5031, F1=0.5184
- Epoch 2: Loss=0.1666, mIoU=0.5133, F1=0.5372
- Epoch 3: Loss=0.1382, mIoU=0.5493, F1=0.5985
- Epoch 4: Loss=0.1256, mIoU=0.6394, F1=0.7229
- Epoch 5: Loss=0.1165, mIoU=0.6401, F1=0.7242
- Epoch 6: Loss=0.1083, mIoU=0.6929, F1=0.7822
- Epoch 7: Loss=0.1058, mIoU=0.6746, F1=0.7625
- Epoch 8: Loss=0.1003, mIoU=0.6778, F1=0.7660
- Epoch 9: Loss=0.0966, mIoU=0.6856, F1=0.7746
- Epoch 10: Loss=0.0931, mIoU=0.6702, F1=0.7581

本轮结果: Round=7, Labeled=717, Selection=best_val (epoch=6), mIoU=0.6929, F1=0.7822, peak_mIoU=0.6929

Round=7, Labeled=717, mIoU=0.6929, F1=0.7822

## Round 8

Labeled Pool Size: 805

- Epoch 1: Loss=0.3774, mIoU=0.5146, F1=0.5398
- Epoch 2: Loss=0.1624, mIoU=0.5367, F1=0.5779
- Epoch 3: Loss=0.1317, mIoU=0.5643, F1=0.6220
- Epoch 4: Loss=0.1192, mIoU=0.5952, F1=0.6664
- Epoch 5: Loss=0.1115, mIoU=0.6243, F1=0.7049
- Epoch 6: Loss=0.1064, mIoU=0.6064, F1=0.6815
- Epoch 7: Loss=0.1039, mIoU=0.7028, F1=0.7917
- Epoch 8: Loss=0.0990, mIoU=0.6972, F1=0.7863
- Epoch 9: Loss=0.0966, mIoU=0.6351, F1=0.7187
- Epoch 10: Loss=0.0950, mIoU=0.6364, F1=0.7199

本轮结果: Round=8, Labeled=805, Selection=best_val (epoch=7), mIoU=0.7028, F1=0.7917, peak_mIoU=0.7028

Round=8, Labeled=805, mIoU=0.7028, F1=0.7917

## Round 9

Labeled Pool Size: 893

- Epoch 1: Loss=0.3355, mIoU=0.5074, F1=0.5264
- Epoch 2: Loss=0.1476, mIoU=0.6551, F1=0.7420
- Epoch 3: Loss=0.1261, mIoU=0.6176, F1=0.6961
- Epoch 4: Loss=0.1152, mIoU=0.5636, F1=0.6208
- Epoch 5: Loss=0.1111, mIoU=0.6950, F1=0.7840
- Epoch 6: Loss=0.1030, mIoU=0.6861, F1=0.7751
- Epoch 7: Loss=0.0999, mIoU=0.6584, F1=0.7449
- Epoch 8: Loss=0.0964, mIoU=0.6915, F1=0.7805
- Epoch 9: Loss=0.0941, mIoU=0.6954, F1=0.7842
- Epoch 10: Loss=0.0920, mIoU=0.6524, F1=0.7382

本轮结果: Round=9, Labeled=893, Selection=best_val (epoch=9), mIoU=0.6954, F1=0.7842, peak_mIoU=0.6954

Round=9, Labeled=893, mIoU=0.6954, F1=0.7842

## Round 10

Labeled Pool Size: 981

- Epoch 1: Loss=0.2486, mIoU=0.5628, F1=0.6197
- Epoch 2: Loss=0.1359, mIoU=0.5500, F1=0.5997
- Epoch 3: Loss=0.1177, mIoU=0.5866, F1=0.6545
- Epoch 4: Loss=0.1053, mIoU=0.6816, F1=0.7700
- Epoch 5: Loss=0.1020, mIoU=0.6617, F1=0.7508
- Epoch 6: Loss=0.0949, mIoU=0.6923, F1=0.7811
- Epoch 7: Loss=0.0916, mIoU=0.6904, F1=0.7796
- Epoch 8: Loss=0.0879, mIoU=0.6831, F1=0.7727
- Epoch 9: Loss=0.0850, mIoU=0.6797, F1=0.7685
- Epoch 10: Loss=0.0834, mIoU=0.6701, F1=0.7596

本轮结果: Round=10, Labeled=981, Selection=best_val (epoch=6), mIoU=0.6923, F1=0.7811, peak_mIoU=0.6923

Round=10, Labeled=981, mIoU=0.6923, F1=0.7811

## Round 11

Labeled Pool Size: 1069

- Epoch 1: Loss=0.2206, mIoU=0.5783, F1=0.6426
- Epoch 2: Loss=0.1257, mIoU=0.6393, F1=0.7227
- Epoch 3: Loss=0.1114, mIoU=0.6797, F1=0.7679
- Epoch 4: Loss=0.1015, mIoU=0.6608, F1=0.7477
- Epoch 5: Loss=0.0961, mIoU=0.6353, F1=0.7176
- Epoch 6: Loss=0.0923, mIoU=0.6620, F1=0.7487
- Epoch 7: Loss=0.0878, mIoU=0.7012, F1=0.7898
- Epoch 8: Loss=0.0857, mIoU=0.7109, F1=0.7997
- Epoch 9: Loss=0.0823, mIoU=0.6733, F1=0.7613
- Epoch 10: Loss=0.0806, mIoU=0.7057, F1=0.7945

本轮结果: Round=11, Labeled=1069, Selection=best_val (epoch=8), mIoU=0.7109, F1=0.7997, peak_mIoU=0.7109

Round=11, Labeled=1069, mIoU=0.7109, F1=0.7997

## Round 12

Labeled Pool Size: 1157

- Epoch 1: Loss=0.2153, mIoU=0.5614, F1=0.6176
- Epoch 2: Loss=0.1206, mIoU=0.6807, F1=0.7690
- Epoch 3: Loss=0.1063, mIoU=0.6279, F1=0.7088
- Epoch 4: Loss=0.0990, mIoU=0.6963, F1=0.7856
- Epoch 5: Loss=0.0939, mIoU=0.5725, F1=0.6502
- Epoch 6: Loss=0.0892, mIoU=0.5787, F1=0.6597
- Epoch 7: Loss=0.0878, mIoU=0.6509, F1=0.7362
- Epoch 8: Loss=0.0838, mIoU=0.6508, F1=0.7376
- Epoch 9: Loss=0.0804, mIoU=0.6808, F1=0.7705
- Epoch 10: Loss=0.0778, mIoU=0.6919, F1=0.7812

本轮结果: Round=12, Labeled=1157, Selection=best_val (epoch=4), mIoU=0.6963, F1=0.7856, peak_mIoU=0.6963

Round=12, Labeled=1157, mIoU=0.6963, F1=0.7856

## Round 13

Labeled Pool Size: 1245

- Epoch 1: Loss=0.2519, mIoU=0.6072, F1=0.6826
- Epoch 2: Loss=0.1231, mIoU=0.6359, F1=0.7184
- Epoch 3: Loss=0.1074, mIoU=0.6222, F1=0.7018
- Epoch 4: Loss=0.0966, mIoU=0.6361, F1=0.7188
- Epoch 5: Loss=0.0924, mIoU=0.6802, F1=0.7700
- Epoch 6: Loss=0.0867, mIoU=0.6791, F1=0.7693
- Epoch 7: Loss=0.0812, mIoU=0.6884, F1=0.7769
- Epoch 8: Loss=0.0803, mIoU=0.7038, F1=0.7938
- Epoch 9: Loss=0.0766, mIoU=0.6991, F1=0.7886
- Epoch 10: Loss=0.0749, mIoU=0.7008, F1=0.7907

本轮结果: Round=13, Labeled=1245, Selection=best_val (epoch=8), mIoU=0.7038, F1=0.7938, peak_mIoU=0.7038

Round=13, Labeled=1245, mIoU=0.7038, F1=0.7938

## Round 14

Labeled Pool Size: 1333

- Epoch 1: Loss=0.1870, mIoU=0.5427, F1=0.5880
- Epoch 2: Loss=0.1107, mIoU=0.6832, F1=0.7719
- Epoch 3: Loss=0.0957, mIoU=0.6390, F1=0.7221
- Epoch 4: Loss=0.0892, mIoU=0.6337, F1=0.7205
- Epoch 5: Loss=0.0861, mIoU=0.6688, F1=0.7562
- Epoch 6: Loss=0.0809, mIoU=0.7033, F1=0.7925
- Epoch 7: Loss=0.0805, mIoU=0.7099, F1=0.7987
- Epoch 8: Loss=0.0766, mIoU=0.6880, F1=0.7769
- Epoch 9: Loss=0.0731, mIoU=0.6506, F1=0.7365
- Epoch 10: Loss=0.0706, mIoU=0.7092, F1=0.7982

本轮结果: Round=14, Labeled=1333, Selection=best_val (epoch=7), mIoU=0.7099, F1=0.7987, peak_mIoU=0.7099

Round=14, Labeled=1333, mIoU=0.7099, F1=0.7987

## Round 15

Labeled Pool Size: 1421

- Epoch 1: Loss=0.2285, mIoU=0.5383, F1=0.5807
- Epoch 2: Loss=0.1131, mIoU=0.6083, F1=0.6839
- Epoch 3: Loss=0.0988, mIoU=0.6045, F1=0.6787
- Epoch 4: Loss=0.0903, mIoU=0.6495, F1=0.7345
- Epoch 5: Loss=0.0870, mIoU=0.6351, F1=0.7173
- Epoch 6: Loss=0.0813, mIoU=0.6717, F1=0.7593
- Epoch 7: Loss=0.0804, mIoU=0.7441, F1=0.8306
- Epoch 8: Loss=0.0763, mIoU=0.7470, F1=0.8326
- Epoch 9: Loss=0.0732, mIoU=0.7181, F1=0.8074
- Epoch 10: Loss=0.0710, mIoU=0.7005, F1=0.7896

本轮结果: Round=15, Labeled=1421, Selection=best_val (epoch=8), mIoU=0.7470, F1=0.8326, peak_mIoU=0.7470

Round=15, Labeled=1421, mIoU=0.7470, F1=0.8326

## Round 16

Labeled Pool Size: 1509

- Test-only mode: Round 16 loads Round 15 best_val checkpoint (epoch=8, val_mIoU=0.7470, val_F1=0.8326)

本轮结果: Round=16, Labeled=1509, Selection=prev_round_best_val (source_round=15, epoch=8), mIoU=0.7470, F1=0.8326, peak_mIoU=0.7470

Round=16, Labeled=1509, mIoU=0.7470, F1=0.8326


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421, 1509]
ALC(基于每轮选模val mIoU): 0.6084
最后一轮选模 mIoU(val): 0.7469824968211745
最后一轮选模 F1(val): 0.8326480291380034
最终报告 mIoU(test): 0.711696968667372
最终报告 F1(test): 0.8007055464317402
最终输出 mIoU: 0.7117 (source=final_report)
最终输出 F1: 0.8007 (source=final_report)
最终 Test Split: test
最终 Report: {'loss': 0.04039662178838625, 'mIoU': 0.711696968667372, 'f1_score': 0.8007055464317402}

--- [Checkpoint] 续跑开始时间: 2026-03-24T08:07:20.523732 ---


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421, 1509]
ALC(基于每轮选模val mIoU): 0.6084
最后一轮选模 mIoU(val): 0.7469824968211745
最后一轮选模 F1(val): 0.8326480291380034
最终报告 mIoU(None): None
最终报告 F1(None): None
最终输出 mIoU: 0.7470 (source=selected_val_fallback)
最终输出 F1: 0.8326 (source=selected_val_fallback)
最终 Test Split: None
最终 Report: None
