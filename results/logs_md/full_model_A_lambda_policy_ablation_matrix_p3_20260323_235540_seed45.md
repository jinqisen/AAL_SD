# 实验日志

实验名称: full_model_A_lambda_policy
描述: 对照A / AAL-SD (Full)：基于 auto_opt_iter15_cand2_02（warmup+risk闭环 + 后期ramp + U-guardrail；train_holdout grad_probe）
开始时间: 2026-03-25T21:48:52.612990

## Round 1

Labeled Pool Size: 189

- Epoch 1: Loss=0.4697, mIoU=0.4893, F1=0.5104
- Epoch 2: Loss=0.2504, mIoU=0.4979, F1=0.5085
- Epoch 3: Loss=0.1569, mIoU=0.5105, F1=0.5323
- Epoch 4: Loss=0.1130, mIoU=0.5288, F1=0.5648
- Epoch 5: Loss=0.0893, mIoU=0.5242, F1=0.5568
- Epoch 6: Loss=0.0747, mIoU=0.5055, F1=0.5228
- Epoch 7: Loss=0.0656, mIoU=0.5015, F1=0.5153
- Epoch 8: Loss=0.0600, mIoU=0.5203, F1=0.5498
- Epoch 9: Loss=0.0536, mIoU=0.5179, F1=0.5456
- Epoch 10: Loss=0.0528, mIoU=0.5500, F1=0.5998

本轮结果: Round=1, Labeled=189, Selection=best_val (epoch=10), mIoU=0.5500, F1=0.5998, peak_mIoU=0.5500

Round=1, Labeled=189, mIoU=0.5500, F1=0.5998


**[ERROR] Round 1 失败: LLM Agent failed at Round 1: LLMAPIError: Error calling API: HTTPSConnectionPool(host='api.siliconflow.cn', port=443): Max retries exceeded with url: /v1/chat/completions (Caused by SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1032)')))**


--- [Legacy Log] 续跑开始时间: 2026-03-26T03:13:30.673459 ---

## Round 2

Labeled Pool Size: 189

- Epoch 1: Loss=0.4535, mIoU=0.4958, F1=0.5057
- Epoch 2: Loss=0.2459, mIoU=0.4997, F1=0.5121
- Epoch 3: Loss=0.1557, mIoU=0.5241, F1=0.5580
- Epoch 4: Loss=0.1135, mIoU=0.5170, F1=0.5451
- Epoch 5: Loss=0.0899, mIoU=0.5128, F1=0.5365
- Epoch 6: Loss=0.0765, mIoU=0.5297, F1=0.5663
- Epoch 7: Loss=0.0681, mIoU=0.5018, F1=0.5158
- Epoch 8: Loss=0.0609, mIoU=0.5301, F1=0.5671
- Epoch 9: Loss=0.0548, mIoU=0.5461, F1=0.5938
- Epoch 10: Loss=0.0504, mIoU=0.5182, F1=0.5464

本轮结果: Round=2, Labeled=189, Selection=best_val (epoch=9), mIoU=0.5461, F1=0.5938, peak_mIoU=0.5461

Round=2, Labeled=189, mIoU=0.5461, F1=0.5938


--- [Legacy Log] 续跑开始时间: 2026-03-26T03:19:40.377990 ---

## Round 3

Labeled Pool Size: 189

- Epoch 1: Loss=0.6046, mIoU=0.4705, F1=0.5177
- Epoch 2: Loss=0.3227, mIoU=0.5133, F1=0.5587
- Epoch 3: Loss=0.1918, mIoU=0.5273, F1=0.5738
- Epoch 4: Loss=0.1262, mIoU=0.5571, F1=0.6135
- Epoch 5: Loss=0.0935, mIoU=0.5528, F1=0.6054
- Epoch 6: Loss=0.0790, mIoU=0.5510, F1=0.6015
- Epoch 7: Loss=0.0702, mIoU=0.5061, F1=0.5242
- Epoch 8: Loss=0.0645, mIoU=0.5101, F1=0.5315
- Epoch 9: Loss=0.0604, mIoU=0.5539, F1=0.6059
- Epoch 10: Loss=0.0578, mIoU=0.5257, F1=0.5594

本轮结果: Round=3, Labeled=189, Selection=best_val (epoch=4), mIoU=0.5571, F1=0.6135, peak_mIoU=0.5571

Round=3, Labeled=189, mIoU=0.5571, F1=0.6135

## Round 4

Labeled Pool Size: 277

- Epoch 1: Loss=0.4841, mIoU=0.4977, F1=0.5100
- Epoch 2: Loss=0.2346, mIoU=0.4924, F1=0.4975
- Epoch 3: Loss=0.1540, mIoU=0.4932, F1=0.4992
- Epoch 4: Loss=0.1190, mIoU=0.4952, F1=0.5032
- Epoch 5: Loss=0.1010, mIoU=0.4963, F1=0.5052
- Epoch 6: Loss=0.0875, mIoU=0.5013, F1=0.5150
- Epoch 7: Loss=0.0815, mIoU=0.4990, F1=0.5105
- Epoch 8: Loss=0.0759, mIoU=0.5197, F1=0.5490
- Epoch 9: Loss=0.0753, mIoU=0.5264, F1=0.5605
- Epoch 10: Loss=0.0678, mIoU=0.5146, F1=0.5396

本轮结果: Round=4, Labeled=277, Selection=best_val (epoch=9), mIoU=0.5264, F1=0.5605, peak_mIoU=0.5264

Round=4, Labeled=277, mIoU=0.5264, F1=0.5605

## Round 5

Labeled Pool Size: 365

- Epoch 1: Loss=0.5444, mIoU=0.5023, F1=0.5196
- Epoch 2: Loss=0.2462, mIoU=0.5043, F1=0.5210
- Epoch 3: Loss=0.1696, mIoU=0.5408, F1=0.5850
- Epoch 4: Loss=0.1376, mIoU=0.5174, F1=0.5447
- Epoch 5: Loss=0.1221, mIoU=0.5522, F1=0.6031
- Epoch 6: Loss=0.1179, mIoU=0.5101, F1=0.5314
- Epoch 7: Loss=0.1065, mIoU=0.5919, F1=0.6620
- Epoch 8: Loss=0.1024, mIoU=0.5750, F1=0.6379
- Epoch 9: Loss=0.0980, mIoU=0.6707, F1=0.7584
- Epoch 10: Loss=0.0948, mIoU=0.6538, F1=0.7394

本轮结果: Round=5, Labeled=365, Selection=best_val (epoch=9), mIoU=0.6707, F1=0.7584, peak_mIoU=0.6707

Round=5, Labeled=365, mIoU=0.6707, F1=0.7584

## Round 6

Labeled Pool Size: 453

- Epoch 1: Loss=0.4776, mIoU=0.5060, F1=0.5239
- Epoch 2: Loss=0.2111, mIoU=0.5095, F1=0.5302
- Epoch 3: Loss=0.1620, mIoU=0.6098, F1=0.6858
- Epoch 4: Loss=0.1414, mIoU=0.6076, F1=0.6831
- Epoch 5: Loss=0.1329, mIoU=0.5576, F1=0.6117
- Epoch 6: Loss=0.1255, mIoU=0.6679, F1=0.7558
- Epoch 7: Loss=0.1207, mIoU=0.5534, F1=0.6051
- Epoch 8: Loss=0.1104, mIoU=0.6053, F1=0.6801
- Epoch 9: Loss=0.1062, mIoU=0.5519, F1=0.6027
- Epoch 10: Loss=0.1027, mIoU=0.5858, F1=0.6536

本轮结果: Round=6, Labeled=453, Selection=best_val (epoch=6), mIoU=0.6679, F1=0.7558, peak_mIoU=0.6679

Round=6, Labeled=453, mIoU=0.6679, F1=0.7558

## Round 7

Labeled Pool Size: 541

- Epoch 1: Loss=0.3217, mIoU=0.5144, F1=0.5395
- Epoch 2: Loss=0.1726, mIoU=0.5453, F1=0.5924
- Epoch 3: Loss=0.1452, mIoU=0.5447, F1=0.5912
- Epoch 4: Loss=0.1309, mIoU=0.5834, F1=0.6500
- Epoch 5: Loss=0.1255, mIoU=0.6718, F1=0.7596
- Epoch 6: Loss=0.1195, mIoU=0.6755, F1=0.7648
- Epoch 7: Loss=0.1109, mIoU=0.6231, F1=0.7029
- Epoch 8: Loss=0.1078, mIoU=0.6714, F1=0.7599
- Epoch 9: Loss=0.1111, mIoU=0.6815, F1=0.7702
- Epoch 10: Loss=0.1014, mIoU=0.6808, F1=0.7691

本轮结果: Round=7, Labeled=541, Selection=best_val (epoch=9), mIoU=0.6815, F1=0.7702, peak_mIoU=0.6815

Round=7, Labeled=541, mIoU=0.6815, F1=0.7702

## Round 8

Labeled Pool Size: 629

- Epoch 1: Loss=0.2734, mIoU=0.5274, F1=0.5621
- Epoch 2: Loss=0.1581, mIoU=0.5921, F1=0.6622
- Epoch 3: Loss=0.1346, mIoU=0.5167, F1=0.5434
- Epoch 4: Loss=0.1232, mIoU=0.6145, F1=0.6921
- Epoch 5: Loss=0.1142, mIoU=0.6190, F1=0.6976
- Epoch 6: Loss=0.1083, mIoU=0.6266, F1=0.7072
- Epoch 7: Loss=0.1009, mIoU=0.6836, F1=0.7722
- Epoch 8: Loss=0.1020, mIoU=0.6411, F1=0.7249
- Epoch 9: Loss=0.0967, mIoU=0.6598, F1=0.7472
- Epoch 10: Loss=0.0954, mIoU=0.6863, F1=0.7750

本轮结果: Round=8, Labeled=629, Selection=best_val (epoch=10), mIoU=0.6863, F1=0.7750, peak_mIoU=0.6863

Round=8, Labeled=629, mIoU=0.6863, F1=0.7750

## Round 9

Labeled Pool Size: 717

- Epoch 1: Loss=0.2631, mIoU=0.5236, F1=0.5555
- Epoch 2: Loss=0.1454, mIoU=0.6323, F1=0.7143
- Epoch 3: Loss=0.1304, mIoU=0.6497, F1=0.7351
- Epoch 4: Loss=0.1249, mIoU=0.5836, F1=0.6504
- Epoch 5: Loss=0.1133, mIoU=0.5576, F1=0.6117
- Epoch 6: Loss=0.1067, mIoU=0.6533, F1=0.7389
- Epoch 7: Loss=0.1029, mIoU=0.6951, F1=0.7839
- Epoch 8: Loss=0.1016, mIoU=0.6556, F1=0.7416
- Epoch 9: Loss=0.0971, mIoU=0.6850, F1=0.7745
- Epoch 10: Loss=0.0924, mIoU=0.6808, F1=0.7707

本轮结果: Round=9, Labeled=717, Selection=best_val (epoch=7), mIoU=0.6951, F1=0.7839, peak_mIoU=0.6951

Round=9, Labeled=717, mIoU=0.6951, F1=0.7839

## Round 10

Labeled Pool Size: 805

- Epoch 1: Loss=0.3163, mIoU=0.5284, F1=0.5639
- Epoch 2: Loss=0.1565, mIoU=0.5715, F1=0.6330
- Epoch 3: Loss=0.1296, mIoU=0.6469, F1=0.7317
- Epoch 4: Loss=0.1170, mIoU=0.6235, F1=0.7037
- Epoch 5: Loss=0.1090, mIoU=0.6969, F1=0.7863
- Epoch 6: Loss=0.1052, mIoU=0.6381, F1=0.7237
- Epoch 7: Loss=0.1008, mIoU=0.6179, F1=0.6993
- Epoch 8: Loss=0.0964, mIoU=0.7004, F1=0.7897
- Epoch 9: Loss=0.0929, mIoU=0.6913, F1=0.7799
- Epoch 10: Loss=0.0894, mIoU=0.7103, F1=0.7992

本轮结果: Round=10, Labeled=805, Selection=best_val (epoch=10), mIoU=0.7103, F1=0.7992, peak_mIoU=0.7103

Round=10, Labeled=805, mIoU=0.7103, F1=0.7992

## Round 11

Labeled Pool Size: 893

- Epoch 1: Loss=0.2310, mIoU=0.5063, F1=0.5243
- Epoch 2: Loss=0.1362, mIoU=0.5608, F1=0.6171
- Epoch 3: Loss=0.1186, mIoU=0.6582, F1=0.7449
- Epoch 4: Loss=0.1119, mIoU=0.6670, F1=0.7545
- Epoch 5: Loss=0.1072, mIoU=0.7066, F1=0.7958
- Epoch 6: Loss=0.0991, mIoU=0.5825, F1=0.6493
- Epoch 7: Loss=0.0958, mIoU=0.6090, F1=0.6849
- Epoch 8: Loss=0.0919, mIoU=0.6138, F1=0.6914
- Epoch 9: Loss=0.0877, mIoU=0.6415, F1=0.7263
- Epoch 10: Loss=0.0864, mIoU=0.6265, F1=0.7072

本轮结果: Round=11, Labeled=893, Selection=best_val (epoch=5), mIoU=0.7066, F1=0.7958, peak_mIoU=0.7066

Round=11, Labeled=893, mIoU=0.7066, F1=0.7958

## Round 12

Labeled Pool Size: 981

- Epoch 1: Loss=0.2767, mIoU=0.5372, F1=0.5790
- Epoch 2: Loss=0.1360, mIoU=0.6060, F1=0.6812
- Epoch 3: Loss=0.1191, mIoU=0.6805, F1=0.7691
- Epoch 4: Loss=0.1067, mIoU=0.6743, F1=0.7624
- Epoch 5: Loss=0.1039, mIoU=0.6458, F1=0.7303
- Epoch 6: Loss=0.0988, mIoU=0.6459, F1=0.7306
- Epoch 7: Loss=0.0956, mIoU=0.6758, F1=0.7638
- Epoch 8: Loss=0.0905, mIoU=0.6465, F1=0.7317
- Epoch 9: Loss=0.0878, mIoU=0.6502, F1=0.7361
- Epoch 10: Loss=0.0844, mIoU=0.6746, F1=0.7633

本轮结果: Round=12, Labeled=981, Selection=best_val (epoch=3), mIoU=0.6805, F1=0.7691, peak_mIoU=0.6805

Round=12, Labeled=981, mIoU=0.6805, F1=0.7691

## Round 13

Labeled Pool Size: 1069

- Epoch 1: Loss=0.2523, mIoU=0.5560, F1=0.6091
- Epoch 2: Loss=0.1312, mIoU=0.5883, F1=0.6567
- Epoch 3: Loss=0.1102, mIoU=0.7010, F1=0.7897
- Epoch 4: Loss=0.1063, mIoU=0.6878, F1=0.7766
- Epoch 5: Loss=0.0986, mIoU=0.6469, F1=0.7320
- Epoch 6: Loss=0.0932, mIoU=0.6528, F1=0.7388
- Epoch 7: Loss=0.0905, mIoU=0.6770, F1=0.7658
- Epoch 8: Loss=0.0864, mIoU=0.6499, F1=0.7353
- Epoch 9: Loss=0.0846, mIoU=0.6765, F1=0.7655
- Epoch 10: Loss=0.0803, mIoU=0.6872, F1=0.7762

本轮结果: Round=13, Labeled=1069, Selection=best_val (epoch=3), mIoU=0.7010, F1=0.7897, peak_mIoU=0.7010

Round=13, Labeled=1069, mIoU=0.7010, F1=0.7897

## Round 14

Labeled Pool Size: 1157

- Epoch 1: Loss=0.2533, mIoU=0.5640, F1=0.6215
