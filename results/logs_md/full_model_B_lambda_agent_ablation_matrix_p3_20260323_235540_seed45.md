# 实验日志

实验名称: full_model_B_lambda_agent
描述: 对照B / AAL-SD (Agent-λ)：与 full_model_A_lambda_policy 完全一致，唯一区别：λ 由 LLM Agent 通过 set_lambda 显式决定，而非 policy 自动填充
开始时间: 2026-03-25T21:48:52.612852

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.4695, mIoU=0.4929, F1=0.5173
- Epoch 2: Loss=0.2509, mIoU=0.4989, F1=0.5112
- Epoch 3: Loss=0.1590, mIoU=0.5188, F1=0.5495
- Epoch 4: Loss=0.1139, mIoU=0.5027, F1=0.5182
- Epoch 5: Loss=0.0908, mIoU=0.5197, F1=0.5506
- Epoch 6: Loss=0.0772, mIoU=0.5013, F1=0.5150
- Epoch 7: Loss=0.0680, mIoU=0.5065, F1=0.5249
- Epoch 8: Loss=0.0599, mIoU=0.5329, F1=0.5719
- Epoch 9: Loss=0.0547, mIoU=0.6086, F1=0.6845
- Epoch 10: Loss=0.0531, mIoU=0.5621, F1=0.6188

本轮结果: Round=1, Labeled=189, Selection=best_val (epoch=9), mIoU=0.6086, F1=0.6845, peak_mIoU=0.6086

Round=1, Labeled=189, mIoU=0.6086, F1=0.6845


**[ERROR] Round 1 失败: LLM Agent failed at Round 1: LLMAPIError: Error calling API: HTTPSConnectionPool(host='api.siliconflow.cn', port=443): Max retries exceeded with url: /v1/chat/completions (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1032)')))**


--- [Legacy Log] 续跑开始时间: 2026-03-26T05:54:36.553358 ---

## Round 2

Labeled Pool Size: 189

- Epoch 1: Loss=0.4542, mIoU=0.4950, F1=0.5051
- Epoch 2: Loss=0.2494, mIoU=0.4992, F1=0.5115
- Epoch 3: Loss=0.1593, mIoU=0.5253, F1=0.5586
- Epoch 4: Loss=0.1164, mIoU=0.5175, F1=0.5448
- Epoch 5: Loss=0.0923, mIoU=0.5000, F1=0.5126
- Epoch 6: Loss=0.0768, mIoU=0.5016, F1=0.5157
- Epoch 7: Loss=0.0658, mIoU=0.5029, F1=0.5180
- Epoch 8: Loss=0.0589, mIoU=0.5200, F1=0.5495
- Epoch 9: Loss=0.0537, mIoU=0.5626, F1=0.6194
- Epoch 10: Loss=0.0512, mIoU=0.5005, F1=0.5134

本轮结果: Round=2, Labeled=189, Selection=best_val (epoch=9), mIoU=0.5626, F1=0.6194, peak_mIoU=0.5626

Round=2, Labeled=189, mIoU=0.5626, F1=0.6194

## Round 3

Labeled Pool Size: 277

- Epoch 1: Loss=0.5448, mIoU=0.5076, F1=0.5449
- Epoch 2: Loss=0.2713, mIoU=0.5022, F1=0.5192
- Epoch 3: Loss=0.1866, mIoU=0.4976, F1=0.5078
- Epoch 4: Loss=0.1492, mIoU=0.4999, F1=0.5122
- Epoch 5: Loss=0.1258, mIoU=0.5036, F1=0.5193
- Epoch 6: Loss=0.1141, mIoU=0.5046, F1=0.5218
- Epoch 7: Loss=0.1089, mIoU=0.5483, F1=0.5973
- Epoch 8: Loss=0.1031, mIoU=0.5830, F1=0.6504
- Epoch 9: Loss=0.0968, mIoU=0.5314, F1=0.5690
- Epoch 10: Loss=0.0961, mIoU=0.5148, F1=0.5401

本轮结果: Round=3, Labeled=277, Selection=best_val (epoch=8), mIoU=0.5830, F1=0.6504, peak_mIoU=0.5830

Round=3, Labeled=277, mIoU=0.5830, F1=0.6504

## Round 4

Labeled Pool Size: 365

- Epoch 1: Loss=0.4492, mIoU=0.5635, F1=0.6220
- Epoch 2: Loss=0.2220, mIoU=0.5198, F1=0.5489
- Epoch 3: Loss=0.1669, mIoU=0.5871, F1=0.6552
- Epoch 4: Loss=0.1474, mIoU=0.5330, F1=0.5717
- Epoch 5: Loss=0.1363, mIoU=0.5952, F1=0.6664
- Epoch 6: Loss=0.1201, mIoU=0.6194, F1=0.6984
- Epoch 7: Loss=0.1196, mIoU=0.6716, F1=0.7592
- Epoch 8: Loss=0.1107, mIoU=0.5317, F1=0.5695
- Epoch 9: Loss=0.1046, mIoU=0.6774, F1=0.7656
- Epoch 10: Loss=0.1032, mIoU=0.5784, F1=0.6428

本轮结果: Round=4, Labeled=365, Selection=best_val (epoch=9), mIoU=0.6774, F1=0.7656, peak_mIoU=0.6774

Round=4, Labeled=365, mIoU=0.6774, F1=0.7656

## Round 5

Labeled Pool Size: 453

- Epoch 1: Loss=0.4984, mIoU=0.5809, F1=0.6498
- Epoch 2: Loss=0.2236, mIoU=0.5883, F1=0.6569
- Epoch 3: Loss=0.1741, mIoU=0.5072, F1=0.5261
- Epoch 4: Loss=0.1514, mIoU=0.5438, F1=0.5898
- Epoch 5: Loss=0.1372, mIoU=0.6199, F1=0.6997
- Epoch 6: Loss=0.1297, mIoU=0.5598, F1=0.6149
- Epoch 7: Loss=0.1272, mIoU=0.6216, F1=0.7018
- Epoch 8: Loss=0.1211, mIoU=0.6309, F1=0.7125
- Epoch 9: Loss=0.1155, mIoU=0.6571, F1=0.7437
- Epoch 10: Loss=0.1115, mIoU=0.6725, F1=0.7606

本轮结果: Round=5, Labeled=453, Selection=best_val (epoch=10), mIoU=0.6725, F1=0.7606, peak_mIoU=0.6725

Round=5, Labeled=453, mIoU=0.6725, F1=0.7606

## Round 6

Labeled Pool Size: 541

- Epoch 1: Loss=0.4407, mIoU=0.5577, F1=0.6125
- Epoch 2: Loss=0.1880, mIoU=0.6493, F1=0.7356
- Epoch 3: Loss=0.1559, mIoU=0.5763, F1=0.6397
- Epoch 4: Loss=0.1389, mIoU=0.5762, F1=0.6397
- Epoch 5: Loss=0.1312, mIoU=0.6404, F1=0.7240
- Epoch 6: Loss=0.1314, mIoU=0.6088, F1=0.6847
- Epoch 7: Loss=0.1201, mIoU=0.6485, F1=0.7344
- Epoch 8: Loss=0.1158, mIoU=0.4858, F1=0.5512
- Epoch 9: Loss=0.1155, mIoU=0.6109, F1=0.6921
- Epoch 10: Loss=0.1079, mIoU=0.6451, F1=0.7321

本轮结果: Round=6, Labeled=541, Selection=best_val (epoch=2), mIoU=0.6493, F1=0.7356, peak_mIoU=0.6493

Round=6, Labeled=541, mIoU=0.6493, F1=0.7356

## Round 7

Labeled Pool Size: 629

- Epoch 1: Loss=0.3000, mIoU=0.5753, F1=0.6408
- Epoch 2: Loss=0.1627, mIoU=0.5617, F1=0.6180
- Epoch 3: Loss=0.1351, mIoU=0.6735, F1=0.7612
- Epoch 4: Loss=0.1243, mIoU=0.6698, F1=0.7574
- Epoch 5: Loss=0.1127, mIoU=0.6618, F1=0.7485
- Epoch 6: Loss=0.1092, mIoU=0.6750, F1=0.7630
- Epoch 7: Loss=0.1019, mIoU=0.6906, F1=0.7800
- Epoch 8: Loss=0.1016, mIoU=0.6753, F1=0.7637
- Epoch 9: Loss=0.0948, mIoU=0.6972, F1=0.7866
- Epoch 10: Loss=0.0928, mIoU=0.6918, F1=0.7805

本轮结果: Round=7, Labeled=629, Selection=best_val (epoch=9), mIoU=0.6972, F1=0.7866, peak_mIoU=0.6972

Round=7, Labeled=629, mIoU=0.6972, F1=0.7866

## Round 8

Labeled Pool Size: 717

- Epoch 1: Loss=0.2632, mIoU=0.4996, F1=0.5226
- Epoch 2: Loss=0.1487, mIoU=0.5488, F1=0.5983
- Epoch 3: Loss=0.1304, mIoU=0.5944, F1=0.6654
- Epoch 4: Loss=0.1187, mIoU=0.6079, F1=0.6834
- Epoch 5: Loss=0.1153, mIoU=0.5844, F1=0.6513
- Epoch 6: Loss=0.1091, mIoU=0.6807, F1=0.7691
- Epoch 7: Loss=0.1023, mIoU=0.7023, F1=0.7910
- Epoch 8: Loss=0.0999, mIoU=0.6882, F1=0.7775
- Epoch 9: Loss=0.0957, mIoU=0.6806, F1=0.7691
- Epoch 10: Loss=0.0941, mIoU=0.6508, F1=0.7362

本轮结果: Round=8, Labeled=717, Selection=best_val (epoch=7), mIoU=0.7023, F1=0.7910, peak_mIoU=0.7023

Round=8, Labeled=717, mIoU=0.7023, F1=0.7910

## Round 9

Labeled Pool Size: 805

- Epoch 1: Loss=0.2529, mIoU=0.5235, F1=0.5582
- Epoch 2: Loss=0.1437, mIoU=0.6085, F1=0.6846
- Epoch 3: Loss=0.1261, mIoU=0.6476, F1=0.7327
- Epoch 4: Loss=0.1152, mIoU=0.6765, F1=0.7647
- Epoch 5: Loss=0.1085, mIoU=0.6967, F1=0.7855
- Epoch 6: Loss=0.1031, mIoU=0.5951, F1=0.6770
- Epoch 7: Loss=0.0984, mIoU=0.6867, F1=0.7755
- Epoch 8: Loss=0.0962, mIoU=0.6998, F1=0.7886
- Epoch 9: Loss=0.0914, mIoU=0.6949, F1=0.7837
- Epoch 10: Loss=0.0887, mIoU=0.7009, F1=0.7900

本轮结果: Round=9, Labeled=805, Selection=best_val (epoch=10), mIoU=0.7009, F1=0.7900, peak_mIoU=0.7009

Round=9, Labeled=805, mIoU=0.7009, F1=0.7900

## Round 10

Labeled Pool Size: 893

- Epoch 1: Loss=0.3000, mIoU=0.5434, F1=0.5892
- Epoch 2: Loss=0.1424, mIoU=0.6223, F1=0.7024
- Epoch 3: Loss=0.1220, mIoU=0.5930, F1=0.6637
- Epoch 4: Loss=0.1126, mIoU=0.6018, F1=0.6752
- Epoch 5: Loss=0.1021, mIoU=0.5627, F1=0.6195
- Epoch 6: Loss=0.0995, mIoU=0.6588, F1=0.7474
- Epoch 7: Loss=0.0986, mIoU=0.6625, F1=0.7496
- Epoch 8: Loss=0.0910, mIoU=0.6264, F1=0.7070
- Epoch 9: Loss=0.0906, mIoU=0.6621, F1=0.7488
- Epoch 10: Loss=0.0880, mIoU=0.6881, F1=0.7772

本轮结果: Round=10, Labeled=893, Selection=best_val (epoch=10), mIoU=0.6881, F1=0.7772, peak_mIoU=0.6881

Round=10, Labeled=893, mIoU=0.6881, F1=0.7772

## Round 11

Labeled Pool Size: 981

- Epoch 1: Loss=0.2177, mIoU=0.5110, F1=0.5329
- Epoch 2: Loss=0.1283, mIoU=0.5517, F1=0.6024
- Epoch 3: Loss=0.1137, mIoU=0.6137, F1=0.6908
- Epoch 4: Loss=0.1051, mIoU=0.6866, F1=0.7753
- Epoch 5: Loss=0.1004, mIoU=0.5717, F1=0.6330
- Epoch 6: Loss=0.0952, mIoU=0.6796, F1=0.7682
- Epoch 7: Loss=0.0910, mIoU=0.6520, F1=0.7377
- Epoch 8: Loss=0.0867, mIoU=0.6542, F1=0.7404
- Epoch 9: Loss=0.0855, mIoU=0.6797, F1=0.7682
- Epoch 10: Loss=0.0851, mIoU=0.6960, F1=0.7848

本轮结果: Round=11, Labeled=981, Selection=best_val (epoch=10), mIoU=0.6960, F1=0.7848, peak_mIoU=0.6960

Round=11, Labeled=981, mIoU=0.6960, F1=0.7848

## Round 12

Labeled Pool Size: 1069

- Epoch 1: Loss=0.2672, mIoU=0.5740, F1=0.6365
- Epoch 2: Loss=0.1331, mIoU=0.6195, F1=0.6985
- Epoch 3: Loss=0.1147, mIoU=0.5896, F1=0.6590
- Epoch 4: Loss=0.1058, mIoU=0.6146, F1=0.6924
- Epoch 5: Loss=0.1020, mIoU=0.6190, F1=0.6980
- Epoch 6: Loss=0.0969, mIoU=0.6795, F1=0.7681
- Epoch 7: Loss=0.0959, mIoU=0.6573, F1=0.7459
- Epoch 8: Loss=0.0901, mIoU=0.6688, F1=0.7576
- Epoch 9: Loss=0.0846, mIoU=0.6688, F1=0.7564
- Epoch 10: Loss=0.0823, mIoU=0.6401, F1=0.7236

本轮结果: Round=12, Labeled=1069, Selection=best_val (epoch=6), mIoU=0.6795, F1=0.7681, peak_mIoU=0.6795

Round=12, Labeled=1069, mIoU=0.6795, F1=0.7681

## Round 13

Labeled Pool Size: 1157

- Epoch 1: Loss=0.2510, mIoU=0.5518, F1=0.6028
- Epoch 2: Loss=0.1259, mIoU=0.5491, F1=0.5984
- Epoch 3: Loss=0.1086, mIoU=0.5483, F1=0.5970
- Epoch 4: Loss=0.1006, mIoU=0.6593, F1=0.7458
- Epoch 5: Loss=0.0946, mIoU=0.6846, F1=0.7731
- Epoch 6: Loss=0.0895, mIoU=0.6966, F1=0.7859
- Epoch 7: Loss=0.0857, mIoU=0.6441, F1=0.7283
- Epoch 8: Loss=0.0814, mIoU=0.6563, F1=0.7423
- Epoch 9: Loss=0.0788, mIoU=0.6268, F1=0.7077
- Epoch 10: Loss=0.0768, mIoU=0.7059, F1=0.7947

本轮结果: Round=13, Labeled=1157, Selection=best_val (epoch=10), mIoU=0.7059, F1=0.7947, peak_mIoU=0.7059

Round=13, Labeled=1157, mIoU=0.7059, F1=0.7947

## Round 14

Labeled Pool Size: 1245

- Epoch 1: Loss=0.2431, mIoU=0.5745, F1=0.6374
- Epoch 2: Loss=0.1184, mIoU=0.6603, F1=0.7470
- Epoch 3: Loss=0.1030, mIoU=0.6359, F1=0.7227
- Epoch 4: Loss=0.0958, mIoU=0.6237, F1=0.7036
- Epoch 5: Loss=0.0894, mIoU=0.6557, F1=0.7416
- Epoch 6: Loss=0.0860, mIoU=0.6141, F1=0.6915
- Epoch 7: Loss=0.0838, mIoU=0.7166, F1=0.8054
- Epoch 8: Loss=0.0793, mIoU=0.6554, F1=0.7413
- Epoch 9: Loss=0.0770, mIoU=0.6928, F1=0.7824
- Epoch 10: Loss=0.0753, mIoU=0.6633, F1=0.7507

本轮结果: Round=14, Labeled=1245, Selection=best_val (epoch=7), mIoU=0.7166, F1=0.8054, peak_mIoU=0.7166

Round=14, Labeled=1245, mIoU=0.7166, F1=0.8054

## Round 15

Labeled Pool Size: 1333

- Epoch 1: Loss=0.2206, mIoU=0.5287, F1=0.5645
- Epoch 2: Loss=0.1136, mIoU=0.5922, F1=0.6621
- Epoch 3: Loss=0.1024, mIoU=0.6808, F1=0.7694
- Epoch 4: Loss=0.0926, mIoU=0.6547, F1=0.7406
- Epoch 5: Loss=0.0905, mIoU=0.6301, F1=0.7163
- Epoch 6: Loss=0.0854, mIoU=0.5621, F1=0.6391
- Epoch 7: Loss=0.0807, mIoU=0.7109, F1=0.7996
- Epoch 8: Loss=0.0763, mIoU=0.6799, F1=0.7689
- Epoch 9: Loss=0.0747, mIoU=0.6949, F1=0.7840
- Epoch 10: Loss=0.0727, mIoU=0.6899, F1=0.7787

本轮结果: Round=15, Labeled=1333, Selection=best_val (epoch=7), mIoU=0.7109, F1=0.7996, peak_mIoU=0.7109

Round=15, Labeled=1333, mIoU=0.7109, F1=0.7996

## Round 16

Labeled Pool Size: 1421

- Test-only mode: Round 16 loads Round 15 best_val checkpoint (epoch=7, val_mIoU=0.7109, val_F1=0.7996)

本轮结果: Round=16, Labeled=1421, Selection=prev_round_best_val (source_round=15, epoch=7), mIoU=0.7109, F1=0.7996, peak_mIoU=0.7109

Round=16, Labeled=1421, mIoU=0.7109, F1=0.7996


## 实验汇总

预算历史: [189, 189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421]
ALC(基于每轮选模val mIoU): 0.5985
最后一轮选模 mIoU(val): 0.7109182447982183
最后一轮选模 F1(val): 0.799646384140948
最终报告 mIoU(test): 0.6964753666194802
最终报告 F1(test): 0.7859815758016986
最终输出 mIoU: 0.6965 (source=final_report)
最终输出 F1: 0.7860 (source=final_report)
最终 Test Split: test
最终 Report: {'loss': 0.048748196843080224, 'mIoU': 0.6964753666194802, 'f1_score': 0.7859815758016986}
