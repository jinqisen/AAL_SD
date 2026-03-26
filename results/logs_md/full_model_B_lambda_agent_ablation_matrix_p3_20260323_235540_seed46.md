# 实验日志

实验名称: full_model_B_lambda_agent
描述: 对照B / AAL-SD (Agent-λ)：与 full_model_A_lambda_policy 完全一致，唯一区别：λ 由 LLM Agent 通过 set_lambda 显式决定，而非 policy 自动填充
开始时间: 2026-03-26T03:36:39.948571

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.4583, mIoU=0.4964, F1=0.5077
- Epoch 2: Loss=0.2551, mIoU=0.4921, F1=0.4971
- Epoch 3: Loss=0.1697, mIoU=0.5018, F1=0.5160
- Epoch 4: Loss=0.1252, mIoU=0.5110, F1=0.5332
- Epoch 5: Loss=0.1024, mIoU=0.5195, F1=0.5485
- Epoch 6: Loss=0.0863, mIoU=0.5177, F1=0.5451
- Epoch 7: Loss=0.0746, mIoU=0.5134, F1=0.5374
- Epoch 8: Loss=0.0664, mIoU=0.5081, F1=0.5277
- Epoch 9: Loss=0.0650, mIoU=0.5251, F1=0.5582
- Epoch 10: Loss=0.0598, mIoU=0.5212, F1=0.5514

本轮结果: Round=1, Labeled=189, Selection=best_val (epoch=9), mIoU=0.5251, F1=0.5582, peak_mIoU=0.5251

Round=1, Labeled=189, mIoU=0.5251, F1=0.5582

## Round 2

Labeled Pool Size: 277

- Epoch 1: Loss=0.5412, mIoU=0.5140, F1=0.5614
- Epoch 2: Loss=0.2674, mIoU=0.5218, F1=0.5526
- Epoch 3: Loss=0.1853, mIoU=0.5151, F1=0.5405
- Epoch 4: Loss=0.1498, mIoU=0.5089, F1=0.5292
- Epoch 5: Loss=0.1297, mIoU=0.5410, F1=0.5852
- Epoch 6: Loss=0.1214, mIoU=0.5465, F1=0.5943
- Epoch 7: Loss=0.1096, mIoU=0.5022, F1=0.5166
- Epoch 8: Loss=0.1038, mIoU=0.5403, F1=0.5840
- Epoch 9: Loss=0.1031, mIoU=0.5837, F1=0.6505
- Epoch 10: Loss=0.0954, mIoU=0.5038, F1=0.5196

本轮结果: Round=2, Labeled=277, Selection=best_val (epoch=9), mIoU=0.5837, F1=0.6505, peak_mIoU=0.5837

Round=2, Labeled=277, mIoU=0.5837, F1=0.6505

## Round 3

Labeled Pool Size: 365

- Epoch 1: Loss=0.4442, mIoU=0.5031, F1=0.5209
- Epoch 2: Loss=0.2048, mIoU=0.5230, F1=0.5550
- Epoch 3: Loss=0.1507, mIoU=0.5006, F1=0.5135
- Epoch 4: Loss=0.1288, mIoU=0.4987, F1=0.5101
- Epoch 5: Loss=0.1114, mIoU=0.5105, F1=0.5322
- Epoch 6: Loss=0.1056, mIoU=0.5632, F1=0.6201
- Epoch 7: Loss=0.0988, mIoU=0.6264, F1=0.7087
- Epoch 8: Loss=0.0935, mIoU=0.5370, F1=0.5786
- Epoch 9: Loss=0.0871, mIoU=0.5753, F1=0.6382
- Epoch 10: Loss=0.0833, mIoU=0.6742, F1=0.7625

本轮结果: Round=3, Labeled=365, Selection=best_val (epoch=10), mIoU=0.6742, F1=0.7625, peak_mIoU=0.6742

Round=3, Labeled=365, mIoU=0.6742, F1=0.7625

## Round 4

Labeled Pool Size: 453

- Epoch 1: Loss=0.4884, mIoU=0.5122, F1=0.5393
- Epoch 2: Loss=0.1954, mIoU=0.4956, F1=0.5040
- Epoch 3: Loss=0.1561, mIoU=0.5126, F1=0.5359
- Epoch 4: Loss=0.1374, mIoU=0.5113, F1=0.5336
- Epoch 5: Loss=0.1236, mIoU=0.5500, F1=0.5997
- Epoch 6: Loss=0.1233, mIoU=0.5197, F1=0.5487
- Epoch 7: Loss=0.1151, mIoU=0.5972, F1=0.6694
- Epoch 8: Loss=0.1116, mIoU=0.5705, F1=0.6311
- Epoch 9: Loss=0.1173, mIoU=0.5416, F1=0.5861
- Epoch 10: Loss=0.1085, mIoU=0.6665, F1=0.7537

本轮结果: Round=4, Labeled=453, Selection=best_val (epoch=10), mIoU=0.6665, F1=0.7537, peak_mIoU=0.6665

Round=4, Labeled=453, mIoU=0.6665, F1=0.7537

## Round 5

Labeled Pool Size: 541

- Epoch 1: Loss=0.4311, mIoU=0.5521, F1=0.6029
- Epoch 2: Loss=0.1819, mIoU=0.5290, F1=0.5650
- Epoch 3: Loss=0.1517, mIoU=0.6096, F1=0.6861
- Epoch 4: Loss=0.1356, mIoU=0.6339, F1=0.7161
- Epoch 5: Loss=0.1270, mIoU=0.6575, F1=0.7450
- Epoch 6: Loss=0.1211, mIoU=0.5780, F1=0.6422
- Epoch 7: Loss=0.1167, mIoU=0.5506, F1=0.6005
- Epoch 8: Loss=0.1115, mIoU=0.6381, F1=0.7222
- Epoch 9: Loss=0.1084, mIoU=0.6792, F1=0.7676
- Epoch 10: Loss=0.1006, mIoU=0.6715, F1=0.7593

本轮结果: Round=5, Labeled=541, Selection=best_val (epoch=9), mIoU=0.6792, F1=0.7676, peak_mIoU=0.6792

Round=5, Labeled=541, mIoU=0.6792, F1=0.7676

## Round 6

Labeled Pool Size: 629

- Epoch 1: Loss=0.3025, mIoU=0.5102, F1=0.5320
- Epoch 2: Loss=0.1630, mIoU=0.6166, F1=0.6945
- Epoch 3: Loss=0.1402, mIoU=0.5497, F1=0.5991
- Epoch 4: Loss=0.1257, mIoU=0.6229, F1=0.7026
- Epoch 5: Loss=0.1189, mIoU=0.6191, F1=0.6983
- Epoch 6: Loss=0.1133, mIoU=0.5824, F1=0.6487
- Epoch 7: Loss=0.1083, mIoU=0.5893, F1=0.6586
- Epoch 8: Loss=0.1025, mIoU=0.6970, F1=0.7857
- Epoch 9: Loss=0.1029, mIoU=0.6402, F1=0.7238
- Epoch 10: Loss=0.0989, mIoU=0.6587, F1=0.7452

本轮结果: Round=6, Labeled=629, Selection=best_val (epoch=8), mIoU=0.6970, F1=0.7857, peak_mIoU=0.6970

Round=6, Labeled=629, mIoU=0.6970, F1=0.7857

## Round 7

Labeled Pool Size: 717

- Epoch 1: Loss=0.2665, mIoU=0.5527, F1=0.6042
- Epoch 2: Loss=0.1493, mIoU=0.5747, F1=0.6373
- Epoch 3: Loss=0.1297, mIoU=0.6641, F1=0.7510
- Epoch 4: Loss=0.1233, mIoU=0.6482, F1=0.7330
- Epoch 5: Loss=0.1125, mIoU=0.6449, F1=0.7292
- Epoch 6: Loss=0.1063, mIoU=0.6576, F1=0.7438
- Epoch 7: Loss=0.1052, mIoU=0.7109, F1=0.7996
- Epoch 8: Loss=0.0989, mIoU=0.6576, F1=0.7442
- Epoch 9: Loss=0.0974, mIoU=0.6416, F1=0.7256
- Epoch 10: Loss=0.0940, mIoU=0.6659, F1=0.7532

本轮结果: Round=7, Labeled=717, Selection=best_val (epoch=7), mIoU=0.7109, F1=0.7996, peak_mIoU=0.7109

Round=7, Labeled=717, mIoU=0.7109, F1=0.7996

## Round 8

Labeled Pool Size: 805

- Epoch 1: Loss=0.2604, mIoU=0.5514, F1=0.6021
- Epoch 2: Loss=0.1461, mIoU=0.5414, F1=0.5859
- Epoch 3: Loss=0.1267, mIoU=0.5998, F1=0.6725
- Epoch 4: Loss=0.1150, mIoU=0.6678, F1=0.7552
- Epoch 5: Loss=0.1104, mIoU=0.6679, F1=0.7551
- Epoch 6: Loss=0.1048, mIoU=0.6767, F1=0.7650
- Epoch 7: Loss=0.1020, mIoU=0.7124, F1=0.8010
- Epoch 8: Loss=0.0956, mIoU=0.7212, F1=0.8095
- Epoch 9: Loss=0.0924, mIoU=0.7135, F1=0.8021
- Epoch 10: Loss=0.0900, mIoU=0.7053, F1=0.7943

本轮结果: Round=8, Labeled=805, Selection=best_val (epoch=8), mIoU=0.7212, F1=0.8095, peak_mIoU=0.7212

Round=8, Labeled=805, mIoU=0.7212, F1=0.8095

## Round 9

Labeled Pool Size: 893

- Epoch 1: Loss=0.2974, mIoU=0.5110, F1=0.5330
- Epoch 2: Loss=0.1443, mIoU=0.6238, F1=0.7035
- Epoch 3: Loss=0.1245, mIoU=0.6298, F1=0.7110
- Epoch 4: Loss=0.1131, mIoU=0.6750, F1=0.7629
- Epoch 5: Loss=0.1114, mIoU=0.6886, F1=0.7771
- Epoch 6: Loss=0.1026, mIoU=0.6299, F1=0.7111
- Epoch 7: Loss=0.0980, mIoU=0.6769, F1=0.7651
- Epoch 8: Loss=0.0962, mIoU=0.6614, F1=0.7487
- Epoch 9: Loss=0.0894, mIoU=0.6854, F1=0.7747
- Epoch 10: Loss=0.0878, mIoU=0.7023, F1=0.7915

本轮结果: Round=9, Labeled=893, Selection=best_val (epoch=10), mIoU=0.7023, F1=0.7915, peak_mIoU=0.7023

Round=9, Labeled=893, mIoU=0.7023, F1=0.7915

## Round 10

Labeled Pool Size: 981

- Epoch 1: Loss=0.2223, mIoU=0.5571, F1=0.6109
- Epoch 2: Loss=0.1336, mIoU=0.5590, F1=0.6139
- Epoch 3: Loss=0.1151, mIoU=0.6462, F1=0.7315
- Epoch 4: Loss=0.1065, mIoU=0.6460, F1=0.7305
- Epoch 5: Loss=0.0992, mIoU=0.6620, F1=0.7485
- Epoch 6: Loss=0.0955, mIoU=0.6673, F1=0.7549
- Epoch 7: Loss=0.0917, mIoU=0.6263, F1=0.7068
- Epoch 8: Loss=0.0884, mIoU=0.6993, F1=0.7880

--- [Checkpoint] 续跑开始时间: 2026-03-26T05:51:28.359192 ---

## Round 10

Labeled Pool Size: 981

- Epoch 9: Loss=0.0884, mIoU=0.6464, F1=0.7310
- Epoch 1: Loss=0.2214, mIoU=0.5469, F1=0.5947
- Epoch 10: Loss=0.0833, mIoU=0.6662, F1=0.7540

本轮结果: Round=10, Labeled=981, Selection=best_val (epoch=8), mIoU=0.6993, F1=0.7880, peak_mIoU=0.6993

Round=10, Labeled=981, mIoU=0.6993, F1=0.7880

- Epoch 2: Loss=0.1335, mIoU=0.5356, F1=0.5762
- Epoch 3: Loss=0.1140, mIoU=0.6316, F1=0.7140
- Epoch 4: Loss=0.1049, mIoU=0.6153, F1=0.6929
- Epoch 5: Loss=0.0981, mIoU=0.6323, F1=0.7143
- Epoch 6: Loss=0.0958, mIoU=0.6271, F1=0.7082
- Epoch 7: Loss=0.0907, mIoU=0.5579, F1=0.6122
- Epoch 8: Loss=0.0884, mIoU=0.6900, F1=0.7789
- Epoch 9: Loss=0.0888, mIoU=0.6769, F1=0.7657
- Epoch 10: Loss=0.0831, mIoU=0.6774, F1=0.7659

本轮结果: Round=10, Labeled=981, Selection=best_val (epoch=8), mIoU=0.6900, F1=0.7789, peak_mIoU=0.6900

Round=10, Labeled=981, mIoU=0.6900, F1=0.7789

## Round 11

Labeled Pool Size: 1069

- Epoch 1: Loss=0.2607, mIoU=0.5034, F1=0.5188
- Epoch 2: Loss=0.1309, mIoU=0.5989, F1=0.6715
- Epoch 3: Loss=0.1145, mIoU=0.5960, F1=0.6677
- Epoch 4: Loss=0.1057, mIoU=0.5610, F1=0.6170
- Epoch 5: Loss=0.0987, mIoU=0.6823, F1=0.7721
- Epoch 6: Loss=0.0938, mIoU=0.6723, F1=0.7601
## Round 11

Labeled Pool Size: 1069

- Epoch 7: Loss=0.0927, mIoU=0.6105, F1=0.6867
- Epoch 8: Loss=0.0885, mIoU=0.6275, F1=0.7082
- Epoch 1: Loss=0.2663, mIoU=0.5544, F1=0.6066
- Epoch 9: Loss=0.0859, mIoU=0.6436, F1=0.7293
- Epoch 2: Loss=0.1302, mIoU=0.6406, F1=0.7249
- Epoch 10: Loss=0.0835, mIoU=0.6515, F1=0.7378
- Epoch 3: Loss=0.1156, mIoU=0.5611, F1=0.6171

本轮结果: Round=11, Labeled=1069, Selection=best_val (epoch=5), mIoU=0.6823, F1=0.7721, peak_mIoU=0.6823

Round=11, Labeled=1069, mIoU=0.6823, F1=0.7721

- Epoch 4: Loss=0.1057, mIoU=0.6178, F1=0.6962
- Epoch 5: Loss=0.0982, mIoU=0.6812, F1=0.7699
- Epoch 6: Loss=0.0936, mIoU=0.6508, F1=0.7362
- Epoch 7: Loss=0.0903, mIoU=0.6889, F1=0.7780
- Epoch 8: Loss=0.0905, mIoU=0.6419, F1=0.7257
- Epoch 9: Loss=0.0857, mIoU=0.6896, F1=0.7787
## Round 12

Labeled Pool Size: 1157

- Epoch 10: Loss=0.0826, mIoU=0.7143, F1=0.8032

本轮结果: Round=11, Labeled=1069, Selection=best_val (epoch=10), mIoU=0.7143, F1=0.8032, peak_mIoU=0.7143

Round=11, Labeled=1069, mIoU=0.7143, F1=0.8032

- Epoch 1: Loss=0.2449, mIoU=0.5947, F1=0.6660
- Epoch 2: Loss=0.1237, mIoU=0.5296, F1=0.5660
- Epoch 3: Loss=0.1084, mIoU=0.6287, F1=0.7101
- Epoch 4: Loss=0.0996, mIoU=0.6409, F1=0.7262
## Round 12

Labeled Pool Size: 1157

- Epoch 5: Loss=0.0945, mIoU=0.7233, F1=0.8114
- Epoch 1: Loss=0.2527, mIoU=0.5536, F1=0.6056
- Epoch 6: Loss=0.0912, mIoU=0.7097, F1=0.7989
- Epoch 2: Loss=0.1250, mIoU=0.5959, F1=0.6677
- Epoch 7: Loss=0.0861, mIoU=0.6975, F1=0.7869
- Epoch 3: Loss=0.1062, mIoU=0.5639, F1=0.6213
- Epoch 8: Loss=0.0871, mIoU=0.6851, F1=0.7738
- Epoch 4: Loss=0.0992, mIoU=0.6649, F1=0.7529
- Epoch 9: Loss=0.0806, mIoU=0.7034, F1=0.7923
- Epoch 5: Loss=0.0944, mIoU=0.7012, F1=0.7902
- Epoch 10: Loss=0.0782, mIoU=0.6051, F1=0.6800

本轮结果: Round=12, Labeled=1157, Selection=best_val (epoch=5), mIoU=0.7233, F1=0.8114, peak_mIoU=0.7233

Round=12, Labeled=1157, mIoU=0.7233, F1=0.8114

- Epoch 6: Loss=0.0925, mIoU=0.7133, F1=0.8021
- Epoch 7: Loss=0.0866, mIoU=0.6959, F1=0.7854
- Epoch 8: Loss=0.0848, mIoU=0.7296, F1=0.8172
## Round 13

Labeled Pool Size: 1245

- Epoch 9: Loss=0.0791, mIoU=0.6636, F1=0.7508
- Epoch 1: Loss=0.2412, mIoU=0.5689, F1=0.6291
- Epoch 10: Loss=0.0769, mIoU=0.6182, F1=0.6970

本轮结果: Round=12, Labeled=1157, Selection=best_val (epoch=8), mIoU=0.7296, F1=0.8172, peak_mIoU=0.7296

Round=12, Labeled=1157, mIoU=0.7296, F1=0.8172

- Epoch 2: Loss=0.1194, mIoU=0.6590, F1=0.7456
- Epoch 3: Loss=0.1024, mIoU=0.6761, F1=0.7645
- Epoch 4: Loss=0.0965, mIoU=0.6007, F1=0.6740
- Epoch 5: Loss=0.0899, mIoU=0.6796, F1=0.7679
- Epoch 6: Loss=0.0874, mIoU=0.7069, F1=0.7957
## Round 13

Labeled Pool Size: 1245

- Epoch 7: Loss=0.0827, mIoU=0.6890, F1=0.7780
- Epoch 1: Loss=0.2351, mIoU=0.5233, F1=0.5551
- Epoch 8: Loss=0.0796, mIoU=0.7090, F1=0.7977
- Epoch 2: Loss=0.1161, mIoU=0.5483, F1=0.5972
- Epoch 9: Loss=0.0783, mIoU=0.7165, F1=0.8052
- Epoch 3: Loss=0.1035, mIoU=0.6499, F1=0.7350
- Epoch 10: Loss=0.0765, mIoU=0.6071, F1=0.6908

本轮结果: Round=13, Labeled=1245, Selection=best_val (epoch=9), mIoU=0.7165, F1=0.8052, peak_mIoU=0.7165

Round=13, Labeled=1245, mIoU=0.7165, F1=0.8052

- Epoch 4: Loss=0.0967, mIoU=0.6274, F1=0.7111
- Epoch 5: Loss=0.0907, mIoU=0.5996, F1=0.6723
- Epoch 6: Loss=0.0871, mIoU=0.6807, F1=0.7691
- Epoch 7: Loss=0.0831, mIoU=0.6378, F1=0.7210
- Epoch 8: Loss=0.0791, mIoU=0.6405, F1=0.7245
- Epoch 9: Loss=0.0764, mIoU=0.6682, F1=0.7560
## Round 14

Labeled Pool Size: 1333

- Epoch 10: Loss=0.0736, mIoU=0.6322, F1=0.7152

本轮结果: Round=13, Labeled=1245, Selection=best_val (epoch=6), mIoU=0.6807, F1=0.7691, peak_mIoU=0.6807

Round=13, Labeled=1245, mIoU=0.6807, F1=0.7691

- Epoch 1: Loss=0.2202, mIoU=0.5269, F1=0.5613
- Epoch 2: Loss=0.1137, mIoU=0.5767, F1=0.6402
- Epoch 3: Loss=0.0992, mIoU=0.5304, F1=0.5674
- Epoch 4: Loss=0.0903, mIoU=0.6752, F1=0.7630
- Epoch 5: Loss=0.0844, mIoU=0.6994, F1=0.7881
- Epoch 6: Loss=0.0802, mIoU=0.6357, F1=0.7225
- Epoch 7: Loss=0.0786, mIoU=0.6901, F1=0.7789
- Epoch 8: Loss=0.0754, mIoU=0.6720, F1=0.7613
## Round 14

Labeled Pool Size: 1333

- Epoch 9: Loss=0.0737, mIoU=0.6608, F1=0.7474
- Epoch 1: Loss=0.2185, mIoU=0.5111, F1=0.5333
- Epoch 10: Loss=0.0716, mIoU=0.6316, F1=0.7136

本轮结果: Round=14, Labeled=1333, Selection=best_val (epoch=5), mIoU=0.6994, F1=0.7881, peak_mIoU=0.6994

Round=14, Labeled=1333, mIoU=0.6994, F1=0.7881

- Epoch 2: Loss=0.1134, mIoU=0.6413, F1=0.7249
- Epoch 3: Loss=0.0977, mIoU=0.7043, F1=0.7935
## Round 15

Labeled Pool Size: 1421

- Epoch 4: Loss=0.0924, mIoU=0.6515, F1=0.7384
- Epoch 1: Loss=0.2340, mIoU=0.5831, F1=0.6495
- Epoch 5: Loss=0.0863, mIoU=0.6740, F1=0.7616
- Epoch 2: Loss=0.1114, mIoU=0.6497, F1=0.7349
- Epoch 6: Loss=0.0820, mIoU=0.6932, F1=0.7828
- Epoch 7: Loss=0.0787, mIoU=0.6938, F1=0.7831
- Epoch 3: Loss=0.0990, mIoU=0.6430, F1=0.7270
- Epoch 8: Loss=0.0756, mIoU=0.6780, F1=0.7670
- Epoch 4: Loss=0.0903, mIoU=0.6178, F1=0.7018
- Epoch 9: Loss=0.0734, mIoU=0.6737, F1=0.7617
- Epoch 5: Loss=0.0875, mIoU=0.6629, F1=0.7496
- Epoch 10: Loss=0.0714, mIoU=0.6346, F1=0.7177

本轮结果: Round=14, Labeled=1333, Selection=best_val (epoch=3), mIoU=0.7043, F1=0.7935, peak_mIoU=0.7043

Round=14, Labeled=1333, mIoU=0.7043, F1=0.7935

- Epoch 6: Loss=0.0811, mIoU=0.6856, F1=0.7745
- Epoch 7: Loss=0.0801, mIoU=0.6809, F1=0.7692
- Epoch 8: Loss=0.0778, mIoU=0.7318, F1=0.8191
## Round 15

Labeled Pool Size: 1421

- Epoch 9: Loss=0.0743, mIoU=0.7319, F1=0.8191
- Epoch 1: Loss=0.2306, mIoU=0.6266, F1=0.7072
- Epoch 10: Loss=0.0716, mIoU=0.7180, F1=0.8065

本轮结果: Round=15, Labeled=1421, Selection=best_val (epoch=9), mIoU=0.7319, F1=0.8191, peak_mIoU=0.7319

Round=15, Labeled=1421, mIoU=0.7319, F1=0.8191

- Epoch 2: Loss=0.1084, mIoU=0.6161, F1=0.6940
- Epoch 3: Loss=0.0945, mIoU=0.5523, F1=0.6034
## Round 16

Labeled Pool Size: 1509

- Test-only mode: Round 16 loads Round 15 best_val checkpoint (epoch=9, val_mIoU=0.7319, val_F1=0.8191)

本轮结果: Round=16, Labeled=1509, Selection=prev_round_best_val (source_round=15, epoch=9), mIoU=0.7319, F1=0.8191, peak_mIoU=0.7319

Round=16, Labeled=1509, mIoU=0.7319, F1=0.8191


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421, 1509]
ALC(基于每轮选模val mIoU): 0.6025
最后一轮选模 mIoU(val): 0.7319207415406184
最后一轮选模 F1(val): 0.8191272382600443
最终报告 mIoU(test): 0.731620456353631
最终报告 F1(test): 0.8191912294691078
最终输出 mIoU: 0.7316 (source=final_report)
最终输出 F1: 0.8192 (source=final_report)
最终 Test Split: test
最终 Report: {'loss': 0.03565651314333081, 'mIoU': 0.731620456353631, 'f1_score': 0.8191912294691078}
- Epoch 4: Loss=0.0872, mIoU=0.6292, F1=0.7105
- Epoch 5: Loss=0.0828, mIoU=0.7092, F1=0.7978
- Epoch 6: Loss=0.0804, mIoU=0.6393, F1=0.7226
- Epoch 7: Loss=0.0773, mIoU=0.6308, F1=0.7141
- Epoch 8: Loss=0.0746, mIoU=0.6429, F1=0.7278
- Epoch 9: Loss=0.0732, mIoU=0.7047, F1=0.7937
- Epoch 10: Loss=0.0698, mIoU=0.6880, F1=0.7769

本轮结果: Round=15, Labeled=1421, Selection=best_val (epoch=5), mIoU=0.7092, F1=0.7978, peak_mIoU=0.7092

Round=15, Labeled=1421, mIoU=0.7092, F1=0.7978

## Round 16

Labeled Pool Size: 1509

- Test-only mode: Round 16 loads Round 15 best_val checkpoint (epoch=5, val_mIoU=0.7092, val_F1=0.7978)

本轮结果: Round=16, Labeled=1509, Selection=prev_round_best_val (source_round=15, epoch=5), mIoU=0.7092, F1=0.7978, peak_mIoU=0.7092

Round=16, Labeled=1509, mIoU=0.7092, F1=0.7978


## 实验汇总

预算历史: [189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421, 1509]
ALC(基于每轮选模val mIoU): 0.6002
最后一轮选模 mIoU(val): 0.7091689049428538
最后一轮选模 F1(val): 0.7977621793659082
最终报告 mIoU(test): 0.7041864724956826
最终报告 F1(test): 0.7933782649728105
最终输出 mIoU: 0.7042 (source=final_report)
最终输出 F1: 0.7934 (source=final_report)
最终 Test Split: test
最终 Report: {'loss': 0.04127183911012253, 'mIoU': 0.7041864724956826, 'f1_score': 0.7933782649728105}
