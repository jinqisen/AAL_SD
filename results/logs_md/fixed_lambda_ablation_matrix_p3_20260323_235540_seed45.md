# 实验日志

实验名称: fixed_lambda
描述: 消融：Fixed λ=0.5；保留Agent，隔离三阶段λ policy贡献
开始时间: 2026-03-25T22:47:41.223891

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.4692, mIoU=0.4951, F1=0.5273
- Epoch 2: Loss=0.2521, mIoU=0.5024, F1=0.5171
- Epoch 3: Loss=0.1600, mIoU=0.5226, F1=0.5543
- Epoch 4: Loss=0.1160, mIoU=0.5176, F1=0.5451
- Epoch 5: Loss=0.0916, mIoU=0.5216, F1=0.5522
- Epoch 6: Loss=0.0760, mIoU=0.5009, F1=0.5142
- Epoch 7: Loss=0.0662, mIoU=0.5014, F1=0.5150
- Epoch 8: Loss=0.0589, mIoU=0.4975, F1=0.5075
- Epoch 9: Loss=0.0550, mIoU=0.5348, F1=0.5751
- Epoch 10: Loss=0.0523, mIoU=0.5281, F1=0.5637

本轮结果: Round=1, Labeled=189, Selection=best_val (epoch=9), mIoU=0.5348, F1=0.5751, peak_mIoU=0.5348

Round=1, Labeled=189, mIoU=0.5348, F1=0.5751


**[ERROR] Round 1 失败: LLM Agent failed at Round 1: LLMAPIError: Error calling API: HTTPSConnectionPool(host='api.siliconflow.cn', port=443): Max retries exceeded with url: /v1/chat/completions (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1032)')))**


--- [Legacy Log] 续跑开始时间: 2026-03-26T09:54:39.557095 ---

## Round 2

Labeled Pool Size: 189

- Epoch 1: Loss=0.4533, mIoU=0.5005, F1=0.5145
- Epoch 2: Loss=0.2497, mIoU=0.4966, F1=0.5060
- Epoch 3: Loss=0.1587, mIoU=0.5157, F1=0.5420
- Epoch 4: Loss=0.1163, mIoU=0.5077, F1=0.5271
- Epoch 5: Loss=0.0918, mIoU=0.5019, F1=0.5162
- Epoch 6: Loss=0.0754, mIoU=0.5026, F1=0.5177
- Epoch 7: Loss=0.0664, mIoU=0.5041, F1=0.5203
- Epoch 8: Loss=0.0596, mIoU=0.5265, F1=0.5611
- Epoch 9: Loss=0.0536, mIoU=0.5417, F1=0.5862
- Epoch 10: Loss=0.0508, mIoU=0.4998, F1=0.5121

本轮结果: Round=2, Labeled=189, Selection=best_val (epoch=9), mIoU=0.5417, F1=0.5862, peak_mIoU=0.5417

Round=2, Labeled=189, mIoU=0.5417, F1=0.5862

## Round 3

Labeled Pool Size: 277

- Epoch 1: Loss=0.5386, mIoU=0.5262, F1=0.5771
- Epoch 2: Loss=0.2628, mIoU=0.5123, F1=0.5365
- Epoch 3: Loss=0.1836, mIoU=0.5047, F1=0.5212
- Epoch 4: Loss=0.1481, mIoU=0.5071, F1=0.5258
- Epoch 5: Loss=0.1257, mIoU=0.5960, F1=0.6674
- Epoch 6: Loss=0.1205, mIoU=0.5633, F1=0.6203
- Epoch 7: Loss=0.1094, mIoU=0.4980, F1=0.5086
- Epoch 8: Loss=0.1072, mIoU=0.5070, F1=0.5256
- Epoch 9: Loss=0.1032, mIoU=0.5606, F1=0.6166
- Epoch 10: Loss=0.0980, mIoU=0.5938, F1=0.6644

本轮结果: Round=3, Labeled=277, Selection=best_val (epoch=5), mIoU=0.5960, F1=0.6674, peak_mIoU=0.5960

Round=3, Labeled=277, mIoU=0.5960, F1=0.6674

## Round 4

Labeled Pool Size: 365

- Epoch 1: Loss=0.4465, mIoU=0.5119, F1=0.5355
- Epoch 2: Loss=0.2159, mIoU=0.5057, F1=0.5232
- Epoch 3: Loss=0.1642, mIoU=0.4964, F1=0.5056
- Epoch 4: Loss=0.1379, mIoU=0.5078, F1=0.5271
- Epoch 5: Loss=0.1241, mIoU=0.5249, F1=0.5579
- Epoch 6: Loss=0.1173, mIoU=0.6186, F1=0.6973
- Epoch 7: Loss=0.1122, mIoU=0.5123, F1=0.5354
- Epoch 8: Loss=0.1050, mIoU=0.5859, F1=0.6533
- Epoch 9: Loss=0.0971, mIoU=0.5956, F1=0.6678
- Epoch 10: Loss=0.0934, mIoU=0.5332, F1=0.5720

本轮结果: Round=4, Labeled=365, Selection=best_val (epoch=6), mIoU=0.6186, F1=0.6973, peak_mIoU=0.6186

Round=4, Labeled=365, mIoU=0.6186, F1=0.6973

## Round 5

Labeled Pool Size: 453

- Epoch 1: Loss=0.4954, mIoU=0.5215, F1=0.5522
- Epoch 2: Loss=0.2099, mIoU=0.5554, F1=0.6085
- Epoch 3: Loss=0.1555, mIoU=0.5772, F1=0.6417
- Epoch 4: Loss=0.1336, mIoU=0.5371, F1=0.5788
- Epoch 5: Loss=0.1197, mIoU=0.5744, F1=0.6372
- Epoch 6: Loss=0.1176, mIoU=0.5690, F1=0.6293
- Epoch 7: Loss=0.1125, mIoU=0.6704, F1=0.7584
- Epoch 8: Loss=0.1081, mIoU=0.5359, F1=0.5768
- Epoch 9: Loss=0.1057, mIoU=0.5792, F1=0.6441
- Epoch 10: Loss=0.1000, mIoU=0.6251, F1=0.7062

本轮结果: Round=5, Labeled=453, Selection=best_val (epoch=7), mIoU=0.6704, F1=0.7584, peak_mIoU=0.6704

Round=5, Labeled=453, mIoU=0.6704, F1=0.7584

## Round 6

Labeled Pool Size: 541

- Epoch 1: Loss=0.4406, mIoU=0.5300, F1=0.5671
- Epoch 2: Loss=0.1912, mIoU=0.4963, F1=0.5052
- Epoch 3: Loss=0.1473, mIoU=0.5230, F1=0.5545
- Epoch 4: Loss=0.1342, mIoU=0.5849, F1=0.6520
- Epoch 5: Loss=0.1216, mIoU=0.5211, F1=0.5511
- Epoch 6: Loss=0.1150, mIoU=0.6201, F1=0.6992
- Epoch 7: Loss=0.1101, mIoU=0.6514, F1=0.7368
- Epoch 8: Loss=0.1063, mIoU=0.6782, F1=0.7676
- Epoch 9: Loss=0.1092, mIoU=0.6581, F1=0.7447
- Epoch 10: Loss=0.1005, mIoU=0.6322, F1=0.7141

本轮结果: Round=6, Labeled=541, Selection=best_val (epoch=8), mIoU=0.6782, F1=0.7676, peak_mIoU=0.6782

Round=6, Labeled=541, mIoU=0.6782, F1=0.7676

## Round 7

Labeled Pool Size: 629

- Epoch 1: Loss=0.3005, mIoU=0.5268, F1=0.5611
- Epoch 2: Loss=0.1607, mIoU=0.5448, F1=0.5914
- Epoch 3: Loss=0.1315, mIoU=0.5805, F1=0.6460
- Epoch 4: Loss=0.1202, mIoU=0.6340, F1=0.7168
- Epoch 5: Loss=0.1113, mIoU=0.6554, F1=0.7415
- Epoch 6: Loss=0.1079, mIoU=0.6745, F1=0.7631
- Epoch 7: Loss=0.1017, mIoU=0.6591, F1=0.7465
- Epoch 8: Loss=0.0980, mIoU=0.6432, F1=0.7279
- Epoch 9: Loss=0.0978, mIoU=0.6791, F1=0.7677
- Epoch 10: Loss=0.0931, mIoU=0.6197, F1=0.6990

本轮结果: Round=7, Labeled=629, Selection=best_val (epoch=9), mIoU=0.6791, F1=0.7677, peak_mIoU=0.6791

Round=7, Labeled=629, mIoU=0.6791, F1=0.7677

## Round 8

Labeled Pool Size: 717

- Epoch 1: Loss=0.2613, mIoU=0.5297, F1=0.5662
- Epoch 2: Loss=0.1470, mIoU=0.5623, F1=0.6192
- Epoch 3: Loss=0.1263, mIoU=0.5879, F1=0.6563
- Epoch 4: Loss=0.1159, mIoU=0.6539, F1=0.7404
- Epoch 5: Loss=0.1054, mIoU=0.6456, F1=0.7303
- Epoch 6: Loss=0.1009, mIoU=0.6451, F1=0.7295
- Epoch 7: Loss=0.1007, mIoU=0.6198, F1=0.6987
- Epoch 8: Loss=0.0968, mIoU=0.6143, F1=0.6924
- Epoch 9: Loss=0.0905, mIoU=0.6607, F1=0.7474
- Epoch 10: Loss=0.0864, mIoU=0.6631, F1=0.7503

本轮结果: Round=8, Labeled=717, Selection=best_val (epoch=10), mIoU=0.6631, F1=0.7503, peak_mIoU=0.6631

Round=8, Labeled=717, mIoU=0.6631, F1=0.7503

## Round 9

Labeled Pool Size: 805

- Epoch 1: Loss=0.2553, mIoU=0.5046, F1=0.5213
- Epoch 2: Loss=0.1425, mIoU=0.5286, F1=0.5644
- Epoch 3: Loss=0.1246, mIoU=0.5745, F1=0.6370
- Epoch 4: Loss=0.1153, mIoU=0.6659, F1=0.7542
- Epoch 5: Loss=0.1075, mIoU=0.6862, F1=0.7754
- Epoch 6: Loss=0.1028, mIoU=0.6020, F1=0.6758
- Epoch 7: Loss=0.0975, mIoU=0.6828, F1=0.7719
- Epoch 8: Loss=0.0936, mIoU=0.6692, F1=0.7568
- Epoch 9: Loss=0.0907, mIoU=0.6163, F1=0.6945
- Epoch 10: Loss=0.0867, mIoU=0.6815, F1=0.7703

本轮结果: Round=9, Labeled=805, Selection=best_val (epoch=5), mIoU=0.6862, F1=0.7754, peak_mIoU=0.6862

Round=9, Labeled=805, mIoU=0.6862, F1=0.7754

## Round 10

Labeled Pool Size: 893

- Epoch 1: Loss=0.2985, mIoU=0.5183, F1=0.5465
- Epoch 2: Loss=0.1448, mIoU=0.5744, F1=0.6372
- Epoch 3: Loss=0.1256, mIoU=0.5168, F1=0.5436
- Epoch 4: Loss=0.1153, mIoU=0.5940, F1=0.6646
- Epoch 5: Loss=0.1058, mIoU=0.6565, F1=0.7426
- Epoch 6: Loss=0.1019, mIoU=0.6560, F1=0.7419
- Epoch 7: Loss=0.0963, mIoU=0.6733, F1=0.7619
- Epoch 8: Loss=0.0928, mIoU=0.6241, F1=0.7044
- Epoch 9: Loss=0.0875, mIoU=0.6007, F1=0.6743
- Epoch 10: Loss=0.0866, mIoU=0.6630, F1=0.7504

本轮结果: Round=10, Labeled=893, Selection=best_val (epoch=7), mIoU=0.6733, F1=0.7619, peak_mIoU=0.6733

Round=10, Labeled=893, mIoU=0.6733, F1=0.7619

## Round 11

Labeled Pool Size: 981

- Epoch 1: Loss=0.2159, mIoU=0.5190, F1=0.5475
- Epoch 2: Loss=0.1271, mIoU=0.5797, F1=0.6446
- Epoch 3: Loss=0.1169, mIoU=0.5913, F1=0.6609
- Epoch 4: Loss=0.1051, mIoU=0.6948, F1=0.7837
- Epoch 5: Loss=0.0984, mIoU=0.6449, F1=0.7291
- Epoch 6: Loss=0.0934, mIoU=0.6853, F1=0.7742
- Epoch 7: Loss=0.0912, mIoU=0.6830, F1=0.7713
- Epoch 8: Loss=0.0880, mIoU=0.6908, F1=0.7801
- Epoch 9: Loss=0.0854, mIoU=0.6424, F1=0.7273
- Epoch 10: Loss=0.0821, mIoU=0.6312, F1=0.7132

本轮结果: Round=11, Labeled=981, Selection=best_val (epoch=4), mIoU=0.6948, F1=0.7837, peak_mIoU=0.6948

Round=11, Labeled=981, mIoU=0.6948, F1=0.7837

## Round 12

Labeled Pool Size: 1069

- Epoch 1: Loss=0.2646, mIoU=0.5440, F1=0.5901
- Epoch 2: Loss=0.1284, mIoU=0.5082, F1=0.5280
- Epoch 3: Loss=0.1089, mIoU=0.5494, F1=0.5986
- Epoch 4: Loss=0.0999, mIoU=0.6341, F1=0.7197
- Epoch 5: Loss=0.0944, mIoU=0.6331, F1=0.7158
- Epoch 6: Loss=0.0890, mIoU=0.5756, F1=0.6387
- Epoch 7: Loss=0.0891, mIoU=0.6587, F1=0.7461
- Epoch 8: Loss=0.0834, mIoU=0.6582, F1=0.7457
- Epoch 9: Loss=0.0794, mIoU=0.6380, F1=0.7220
- Epoch 10: Loss=0.0778, mIoU=0.6223, F1=0.7027

本轮结果: Round=12, Labeled=1069, Selection=best_val (epoch=7), mIoU=0.6587, F1=0.7461, peak_mIoU=0.6587

Round=12, Labeled=1069, mIoU=0.6587, F1=0.7461

## Round 13

Labeled Pool Size: 1157

- Epoch 1: Loss=0.2436, mIoU=0.5317, F1=0.5697
- Epoch 2: Loss=0.1181, mIoU=0.5520, F1=0.6032
- Epoch 3: Loss=0.1034, mIoU=0.5601, F1=0.6158
- Epoch 4: Loss=0.0935, mIoU=0.6856, F1=0.7751
- Epoch 5: Loss=0.0891, mIoU=0.6603, F1=0.7468
- Epoch 6: Loss=0.0835, mIoU=0.7174, F1=0.8059
- Epoch 7: Loss=0.0803, mIoU=0.6773, F1=0.7659
- Epoch 8: Loss=0.0758, mIoU=0.6737, F1=0.7626
- Epoch 9: Loss=0.0732, mIoU=0.7085, F1=0.7972
- Epoch 10: Loss=0.0715, mIoU=0.7100, F1=0.7988

本轮结果: Round=13, Labeled=1157, Selection=best_val (epoch=6), mIoU=0.7174, F1=0.8059, peak_mIoU=0.7174

Round=13, Labeled=1157, mIoU=0.7174, F1=0.8059

## Round 14

Labeled Pool Size: 1245

- Epoch 1: Loss=0.2317, mIoU=0.4975, F1=0.5076
- Epoch 2: Loss=0.1085, mIoU=0.6408, F1=0.7242
- Epoch 3: Loss=0.0954, mIoU=0.5766, F1=0.6413
- Epoch 4: Loss=0.0861, mIoU=0.6709, F1=0.7586
- Epoch 5: Loss=0.0808, mIoU=0.7009, F1=0.7896
- Epoch 6: Loss=0.0770, mIoU=0.5553, F1=0.6080
- Epoch 7: Loss=0.0719, mIoU=0.6786, F1=0.7668
- Epoch 8: Loss=0.0687, mIoU=0.5791, F1=0.6438
- Epoch 9: Loss=0.0682, mIoU=0.6668, F1=0.7545
- Epoch 10: Loss=0.0644, mIoU=0.6234, F1=0.7036

本轮结果: Round=14, Labeled=1245, Selection=best_val (epoch=5), mIoU=0.7009, F1=0.7896, peak_mIoU=0.7009

Round=14, Labeled=1245, mIoU=0.7009, F1=0.7896

## Round 15

Labeled Pool Size: 1333

- Epoch 1: Loss=0.2080, mIoU=0.5117, F1=0.5342
- Epoch 2: Loss=0.0983, mIoU=0.6641, F1=0.7509
- Epoch 3: Loss=0.0889, mIoU=0.6454, F1=0.7319
- Epoch 4: Loss=0.0799, mIoU=0.7296, F1=0.8174
- Epoch 5: Loss=0.0767, mIoU=0.6915, F1=0.7810
- Epoch 6: Loss=0.0732, mIoU=0.7034, F1=0.7924
- Epoch 7: Loss=0.0706, mIoU=0.6260, F1=0.7064
- Epoch 8: Loss=0.0693, mIoU=0.6864, F1=0.7753
- Epoch 9: Loss=0.0637, mIoU=0.6799, F1=0.7684
- Epoch 10: Loss=0.0625, mIoU=0.6949, F1=0.7839

本轮结果: Round=15, Labeled=1333, Selection=best_val (epoch=4), mIoU=0.7296, F1=0.8174, peak_mIoU=0.7296

Round=15, Labeled=1333, mIoU=0.7296, F1=0.8174

## Round 16

Labeled Pool Size: 1421

- Test-only mode: Round 16 loads Round 15 best_val checkpoint (epoch=4, val_mIoU=0.7296, val_F1=0.8174)

本轮结果: Round=16, Labeled=1421, Selection=prev_round_best_val (source_round=15, epoch=4), mIoU=0.7296, F1=0.8174, peak_mIoU=0.7296

Round=16, Labeled=1421, mIoU=0.7296, F1=0.8174


## 实验汇总

预算历史: [189, 189, 277, 365, 453, 541, 629, 717, 805, 893, 981, 1069, 1157, 1245, 1333, 1421]
ALC(基于每轮选模val mIoU): 0.5917
最后一轮选模 mIoU(val): 0.7296046714231962
最后一轮选模 F1(val): 0.8173753624482011
最终报告 mIoU(test): 0.7103268485008309
最终报告 F1(test): 0.7998486706283123
最终输出 mIoU: 0.7103 (source=final_report)
最终输出 F1: 0.7998 (source=final_report)
最终 Test Split: test
最终 Report: {'loss': 0.05240073774941265, 'mIoU': 0.7103268485008309, 'f1_score': 0.7998486706283123}
