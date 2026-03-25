# 实验日志

实验名称: full_model_B_lambda_agent
描述: 对照B / AAL-SD (Agent-λ)：与 full_model_A_lambda_policy 完全一致，唯一区别：λ 由 LLM Agent 通过 set_lambda 显式决定，而非 policy 自动填充
开始时间: 2026-03-23T23:55:47.871470

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.5163, mIoU=0.5075, F1=0.5313
- Epoch 2: Loss=0.2815, mIoU=0.5337, F1=0.5803
- Epoch 3: Loss=0.1761, mIoU=0.5374, F1=0.5877
- Epoch 4: Loss=0.1323, mIoU=0.5270, F1=0.5723
- Epoch 5: Loss=0.1063, mIoU=0.5360, F1=0.5817
- Epoch 6: Loss=0.0970, mIoU=0.5441, F1=0.5932
- Epoch 7: Loss=0.0798, mIoU=0.5137, F1=0.5398
- Epoch 8: Loss=0.0744, mIoU=0.5173, F1=0.5481
- Epoch 9: Loss=0.0686, mIoU=0.5412, F1=0.5857
- Epoch 10: Loss=0.0650, mIoU=0.5553, F1=0.6090

本轮结果: Round=1, Labeled=189, Selection=best_val (epoch=10), mIoU=0.5553, F1=0.6090, peak_mIoU=0.5553

Round=1, Labeled=189, mIoU=0.5553, F1=0.6090

## Round 2

Labeled Pool Size: 277

- Epoch 1: Loss=0.3358, mIoU=0.5299, F1=0.5675
- Epoch 2: Loss=0.1992, mIoU=0.5534, F1=0.6051
- Epoch 3: Loss=0.1591, mIoU=0.5981, F1=0.6706
- Epoch 4: Loss=0.1340, mIoU=0.5516, F1=0.6022
- Epoch 5: Loss=0.1241, mIoU=0.6142, F1=0.6915
- Epoch 6: Loss=0.1120, mIoU=0.5620, F1=0.6183
- Epoch 7: Loss=0.1094, mIoU=0.5756, F1=0.6391
- Epoch 8: Loss=0.1089, mIoU=0.5780, F1=0.6426
- Epoch 9: Loss=0.1002, mIoU=0.5533, F1=0.6051
- Epoch 10: Loss=0.0982, mIoU=0.6071, F1=0.6824

本轮结果: Round=2, Labeled=277, Selection=best_val (epoch=5), mIoU=0.6142, F1=0.6915, peak_mIoU=0.6142

Round=2, Labeled=277, mIoU=0.6142, F1=0.6915

## Round 3

Labeled Pool Size: 365

- Epoch 1: Loss=0.4024, mIoU=0.5673, F1=0.6292
- Epoch 2: Loss=0.2104, mIoU=0.6113, F1=0.6881
- Epoch 3: Loss=0.1669, mIoU=0.5299, F1=0.5665
- Epoch 4: Loss=0.1487, mIoU=0.6022, F1=0.6759
- Epoch 5: Loss=0.1402, mIoU=0.5648, F1=0.6226
- Epoch 6: Loss=0.1311, mIoU=0.5834, F1=0.6498
- Epoch 7: Loss=0.1269, mIoU=0.6310, F1=0.7128
- Epoch 8: Loss=0.1152, mIoU=0.6307, F1=0.7122
- Epoch 9: Loss=0.1186, mIoU=0.5223, F1=0.5533
- Epoch 10: Loss=0.1157, mIoU=0.6211, F1=0.7002

本轮结果: Round=3, Labeled=365, Selection=best_val (epoch=7), mIoU=0.6310, F1=0.7128, peak_mIoU=0.6310

Round=3, Labeled=365, mIoU=0.6310, F1=0.7128

## Round 4

Labeled Pool Size: 453

- Epoch 1: Loss=0.3768, mIoU=0.5525, F1=0.6038
- Epoch 2: Loss=0.1929, mIoU=0.5899, F1=0.6591
- Epoch 3: Loss=0.1569, mIoU=0.5841, F1=0.6509
- Epoch 4: Loss=0.1382, mIoU=0.5567, F1=0.6102
- Epoch 5: Loss=0.1290, mIoU=0.6296, F1=0.7112
- Epoch 6: Loss=0.1248, mIoU=0.7222, F1=0.8103
- Epoch 7: Loss=0.1204, mIoU=0.6633, F1=0.7504
- Epoch 8: Loss=0.1163, mIoU=0.6183, F1=0.6972
- Epoch 9: Loss=0.1097, mIoU=0.6013, F1=0.6747
- Epoch 10: Loss=0.1072, mIoU=0.6689, F1=0.7563

本轮结果: Round=4, Labeled=453, Selection=best_val (epoch=6), mIoU=0.7222, F1=0.8103, peak_mIoU=0.7222

Round=4, Labeled=453, mIoU=0.7222, F1=0.8103

## Round 5

Labeled Pool Size: 541

- Epoch 1: Loss=0.3391, mIoU=0.5518, F1=0.6029
- Epoch 2: Loss=0.1808, mIoU=0.5360, F1=0.5785
- Epoch 3: Loss=0.1475, mIoU=0.5491, F1=0.5982
- Epoch 4: Loss=0.1382, mIoU=0.5148, F1=0.5400
- Epoch 5: Loss=0.1299, mIoU=0.5933, F1=0.6639
- Epoch 6: Loss=0.1186, mIoU=0.5542, F1=0.6065
- Epoch 7: Loss=0.1157, mIoU=0.6311, F1=0.7135
- Epoch 8: Loss=0.1124, mIoU=0.6265, F1=0.7079
- Epoch 9: Loss=0.1100, mIoU=0.6145, F1=0.6928
- Epoch 10: Loss=0.1047, mIoU=0.5836, F1=0.6515

本轮结果: Round=5, Labeled=541, Selection=best_val (epoch=7), mIoU=0.6311, F1=0.7135, peak_mIoU=0.6311

Round=5, Labeled=541, mIoU=0.6311, F1=0.7135

## Round 6

Labeled Pool Size: 629

- Epoch 1: Loss=0.3869, mIoU=0.5076, F1=0.5268
- Epoch 2: Loss=0.1785, mIoU=0.5426, F1=0.5878
- Epoch 3: Loss=0.1460, mIoU=0.5558, F1=0.6089
- Epoch 4: Loss=0.1301, mIoU=0.6435, F1=0.7277
- Epoch 5: Loss=0.1217, mIoU=0.6782, F1=0.7668
- Epoch 6: Loss=0.1181, mIoU=0.6547, F1=0.7405
- Epoch 7: Loss=0.1155, mIoU=0.7017, F1=0.7905
- Epoch 8: Loss=0.1085, mIoU=0.7019, F1=0.7909
- Epoch 9: Loss=0.1036, mIoU=0.7048, F1=0.7938
- Epoch 10: Loss=0.1017, mIoU=0.7038, F1=0.7925

本轮结果: Round=6, Labeled=629, Selection=best_val (epoch=9), mIoU=0.7048, F1=0.7938, peak_mIoU=0.7048

Round=6, Labeled=629, mIoU=0.7048, F1=0.7938

## Round 7

Labeled Pool Size: 717

- Epoch 1: Loss=0.3410, mIoU=0.5053, F1=0.5224
- Epoch 2: Loss=0.1676, mIoU=0.5115, F1=0.5339
- Epoch 3: Loss=0.1366, mIoU=0.5792, F1=0.6439
- Epoch 4: Loss=0.1248, mIoU=0.6783, F1=0.7663
- Epoch 5: Loss=0.1162, mIoU=0.5413, F1=0.5857
- Epoch 6: Loss=0.1099, mIoU=0.6070, F1=0.6822
- Epoch 7: Loss=0.1063, mIoU=0.6760, F1=0.7639
- Epoch 8: Loss=0.1013, mIoU=0.6791, F1=0.7688
- Epoch 9: Loss=0.0994, mIoU=0.6344, F1=0.7174
- Epoch 10: Loss=0.0963, mIoU=0.6900, F1=0.7787

本轮结果: Round=7, Labeled=717, Selection=best_val (epoch=10), mIoU=0.6900, F1=0.7787, peak_mIoU=0.6900

Round=7, Labeled=717, mIoU=0.6900, F1=0.7787

## Round 8

Labeled Pool Size: 805

- Epoch 1: Loss=0.3768, mIoU=0.5763, F1=0.6399
- Epoch 2: Loss=0.1614, mIoU=0.5782, F1=0.6426
- Epoch 3: Loss=0.1348, mIoU=0.5682, F1=0.6279
- Epoch 4: Loss=0.1230, mIoU=0.5851, F1=0.6604
- Epoch 5: Loss=0.1164, mIoU=0.6508, F1=0.7367
- Epoch 6: Loss=0.1112, mIoU=0.6216, F1=0.7011
- Epoch 7: Loss=0.1060, mIoU=0.6727, F1=0.7619
- Epoch 8: Loss=0.1029, mIoU=0.6684, F1=0.7559
- Epoch 9: Loss=0.1000, mIoU=0.6468, F1=0.7320
- Epoch 10: Loss=0.0973, mIoU=0.6759, F1=0.7644

本轮结果: Round=8, Labeled=805, Selection=best_val (epoch=10), mIoU=0.6759, F1=0.7644, peak_mIoU=0.6759

Round=8, Labeled=805, mIoU=0.6759, F1=0.7644

## Round 9

Labeled Pool Size: 893

- Epoch 1: Loss=0.3259, mIoU=0.5466, F1=0.5945
- Epoch 2: Loss=0.1462, mIoU=0.5939, F1=0.6646
- Epoch 3: Loss=0.1283, mIoU=0.6702, F1=0.7582
- Epoch 4: Loss=0.1199, mIoU=0.6227, F1=0.7025
- Epoch 5: Loss=0.1119, mIoU=0.5571, F1=0.6110
- Epoch 6: Loss=0.1074, mIoU=0.6638, F1=0.7522
- Epoch 7: Loss=0.1015, mIoU=0.6405, F1=0.7249
- Epoch 8: Loss=0.0990, mIoU=0.6336, F1=0.7159
- Epoch 9: Loss=0.0937, mIoU=0.5951, F1=0.6666
- Epoch 10: Loss=0.0919, mIoU=0.6187, F1=0.6981

本轮结果: Round=9, Labeled=893, Selection=best_val (epoch=3), mIoU=0.6702, F1=0.7582, peak_mIoU=0.6702

Round=9, Labeled=893, mIoU=0.6702, F1=0.7582

## Round 10

Labeled Pool Size: 981

- Epoch 1: Loss=0.2481, mIoU=0.5285, F1=0.5642
- Epoch 2: Loss=0.1358, mIoU=0.5910, F1=0.6605
- Epoch 3: Loss=0.1164, mIoU=0.6423, F1=0.7265
- Epoch 4: Loss=0.1090, mIoU=0.5679, F1=0.6279
- Epoch 5: Loss=0.1009, mIoU=0.6142, F1=0.6922
- Epoch 6: Loss=0.0981, mIoU=0.5929, F1=0.6633
- Epoch 7: Loss=0.0930, mIoU=0.5964, F1=0.6690
- Epoch 8: Loss=0.0939, mIoU=0.6770, F1=0.7657
- Epoch 9: Loss=0.0865, mIoU=0.6874, F1=0.7761
- Epoch 10: Loss=0.0835, mIoU=0.6765, F1=0.7658

本轮结果: Round=10, Labeled=981, Selection=best_val (epoch=9), mIoU=0.6874, F1=0.7761, peak_mIoU=0.6874

Round=10, Labeled=981, mIoU=0.6874, F1=0.7761

## Round 11

Labeled Pool Size: 1069

- Epoch 1: Loss=0.2202, mIoU=0.5133, F1=0.5373
- Epoch 2: Loss=0.1254, mIoU=0.6118, F1=0.6884
- Epoch 3: Loss=0.1092, mIoU=0.6151, F1=0.6933
- Epoch 4: Loss=0.1013, mIoU=0.6715, F1=0.7596
- Epoch 5: Loss=0.0970, mIoU=0.6690, F1=0.7579
- Epoch 6: Loss=0.0928, mIoU=0.6517, F1=0.7371
- Epoch 7: Loss=0.0871, mIoU=0.6823, F1=0.7714
- Epoch 8: Loss=0.0859, mIoU=0.6618, F1=0.7506
- Epoch 9: Loss=0.0828, mIoU=0.6806, F1=0.7689
- Epoch 10: Loss=0.0791, mIoU=0.6888, F1=0.7775

本轮结果: Round=11, Labeled=1069, Selection=best_val (epoch=10), mIoU=0.6888, F1=0.7775, peak_mIoU=0.6888

Round=11, Labeled=1069, mIoU=0.6888, F1=0.7775

## Round 12

Labeled Pool Size: 1157

- Epoch 1: Loss=0.2124, mIoU=0.5490, F1=0.5981
- Epoch 2: Loss=0.1183, mIoU=0.5706, F1=0.6313
- Epoch 3: Loss=0.1042, mIoU=0.5674, F1=0.6265
- Epoch 4: Loss=0.0989, mIoU=0.6127, F1=0.6898
- Epoch 5: Loss=0.0926, mIoU=0.6793, F1=0.7676
- Epoch 6: Loss=0.0883, mIoU=0.6518, F1=0.7388
- Epoch 7: Loss=0.0860, mIoU=0.6206, F1=0.7001
- Epoch 8: Loss=0.0845, mIoU=0.5898, F1=0.6590
- Epoch 9: Loss=0.0814, mIoU=0.6578, F1=0.7446
- Epoch 10: Loss=0.0804, mIoU=0.6944, F1=0.7831

本轮结果: Round=12, Labeled=1157, Selection=best_val (epoch=10), mIoU=0.6944, F1=0.7831, peak_mIoU=0.6944

Round=12, Labeled=1157, mIoU=0.6944, F1=0.7831

## Round 13

Labeled Pool Size: 1245

- Epoch 1: Loss=0.2511, mIoU=0.5851, F1=0.6524
- Epoch 2: Loss=0.1198, mIoU=0.6674, F1=0.7552
- Epoch 3: Loss=0.1039, mIoU=0.5394, F1=0.5826
- Epoch 4: Loss=0.0957, mIoU=0.6700, F1=0.7577
- Epoch 5: Loss=0.0933, mIoU=0.7056, F1=0.7943
- Epoch 6: Loss=0.0860, mIoU=0.6861, F1=0.7747
- Epoch 7: Loss=0.0819, mIoU=0.6763, F1=0.7660
- Epoch 8: Loss=0.0787, mIoU=0.6947, F1=0.7840
- Epoch 9: Loss=0.0762, mIoU=0.7174, F1=0.8059
- Epoch 10: Loss=0.0740, mIoU=0.6773, F1=0.7673

本轮结果: Round=13, Labeled=1245, Selection=best_val (epoch=9), mIoU=0.7174, F1=0.8059, peak_mIoU=0.7174

Round=13, Labeled=1245, mIoU=0.7174, F1=0.8059

## Round 14

Labeled Pool Size: 1333

- Epoch 1: Loss=0.1877, mIoU=0.5009, F1=0.5140
- Epoch 2: Loss=0.1116, mIoU=0.5976, F1=0.6695
- Epoch 3: Loss=0.1016, mIoU=0.5906, F1=0.6599
- Epoch 4: Loss=0.0913, mIoU=0.6330, F1=0.7158
- Epoch 5: Loss=0.0854, mIoU=0.6348, F1=0.7236
- Epoch 6: Loss=0.0809, mIoU=0.6900, F1=0.7792
- Epoch 7: Loss=0.0772, mIoU=0.7005, F1=0.7896
- Epoch 8: Loss=0.0757, mIoU=0.6675, F1=0.7547
- Epoch 9: Loss=0.0730, mIoU=0.6600, F1=0.7473
- Epoch 10: Loss=0.0714, mIoU=0.5908, F1=0.6610

本轮结果: Round=14, Labeled=1333, Selection=best_val (epoch=7), mIoU=0.7005, F1=0.7896, peak_mIoU=0.7005

Round=14, Labeled=1333, mIoU=0.7005, F1=0.7896

## Round 15

Labeled Pool Size: 1421

- Epoch 1: Loss=0.2249, mIoU=0.5163, F1=0.5428
- Epoch 2: Loss=0.1116, mIoU=0.5347, F1=0.5746
- Epoch 3: Loss=0.0994, mIoU=0.5908, F1=0.6604
- Epoch 4: Loss=0.0912, mIoU=0.6882, F1=0.7767
- Epoch 5: Loss=0.0869, mIoU=0.6584, F1=0.7451
- Epoch 6: Loss=0.0821, mIoU=0.6777, F1=0.7659
- Epoch 7: Loss=0.0779, mIoU=0.6836, F1=0.7718
- Epoch 8: Loss=0.0756, mIoU=0.6854, F1=0.7744
- Epoch 9: Loss=0.0723, mIoU=0.6919, F1=0.7812
- Epoch 10: Loss=0.0701, mIoU=0.6842, F1=0.7730

本轮结果: Round=15, Labeled=1421, Selection=best_val (epoch=9), mIoU=0.6919, F1=0.7812, peak_mIoU=0.6919

Round=15, Labeled=1421, mIoU=0.6919, F1=0.7812

## Round 16

Labeled Pool Size: 1509

- Test-only mode: Round 16 loads Round 15 best_val checkpoint (epoch=9, val_mIoU=0.6919, val_F1=0.7812)

本轮结果: Round=16, Labeled=1509, Selection=prev_round_best_val (source_round=15, epoch=9), mIoU=0.6919, F1=0.7812, peak_mIoU=0.6919

Round=16, Labeled=1509, mIoU=0.6919, F1=0.7812


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421, 1509]
ALC(基于每轮选模val mIoU): 0.5922
最后一轮选模 mIoU(val): 0.6919264522962254
最后一轮选模 F1(val): 0.7811746558560048
最终报告 mIoU(test): 0.6718474500326481
最终报告 F1(test): 0.7607796664182462
最终输出 mIoU: 0.6718 (source=final_report)
最终输出 F1: 0.7608 (source=final_report)
最终 Test Split: test
最终 Report: {'loss': 0.050185100622475144, 'mIoU': 0.6718474500326481, 'f1_score': 0.7607796664182462}

--- [Checkpoint] 续跑开始时间: 2026-03-24T08:07:20.573223 ---


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421, 1509]
ALC(基于每轮选模val mIoU): 0.5922
最后一轮选模 mIoU(val): 0.6919264522962254
最后一轮选模 F1(val): 0.7811746558560048
最终报告 mIoU(None): None
最终报告 F1(None): None
最终输出 mIoU: 0.6919 (source=selected_val_fallback)
最终输出 F1: 0.7812 (source=selected_val_fallback)
最终 Test Split: None
最终 Report: None
