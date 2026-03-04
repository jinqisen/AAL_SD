# 实验日志

实验名称: agent_control_budget
描述: Agent控制消融：仅允许set_query_size
开始时间: 2026-02-14T01:26:42.555307

## Round 1

Labeled Pool Size: 151

- Epoch 1: Loss=0.6103, mIoU=0.4603, F1=0.5055
- Epoch 2: Loss=0.3448, mIoU=0.5045, F1=0.5331
- Epoch 3: Loss=0.2140, mIoU=0.4966, F1=0.5112
- Epoch 4: Loss=0.1528, mIoU=0.5209, F1=0.5557
- Epoch 5: Loss=0.1181, mIoU=0.5146, F1=0.5444
- Epoch 6: Loss=0.0937, mIoU=0.5908, F1=0.6646
- Epoch 7: Loss=0.0834, mIoU=0.5964, F1=0.6723
- Epoch 8: Loss=0.0745, mIoU=0.5352, F1=0.5796
- Epoch 9: Loss=0.0668, mIoU=0.5621, F1=0.6220
- Epoch 10: Loss=0.0606, mIoU=0.6094, F1=0.6880

当前轮次最佳结果: Round=1, Labeled=151, mIoU=0.6094, F1=0.6880


**[ERROR] Round 1 失败: LLM Agent failed at Round 1: LLMAPIError: Error calling API: HTTPSConnectionPool(host='api.siliconflow.cn', port=443): Max retries exceeded with url: /v1/chat/completions (Caused by ProxyError('Unable to connect to proxy', NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x17f21f4d0>: Failed to establish a new connection: [Errno 61] Connection refused')))**


--- [Legacy Log] 续跑开始时间: 2026-02-14T11:51:04.649009 ---

## Round 2

Labeled Pool Size: 151

- Epoch 1: Loss=0.6091, mIoU=0.4639, F1=0.5070
- Epoch 2: Loss=0.3330, mIoU=0.5074, F1=0.5349
- Epoch 3: Loss=0.1861, mIoU=0.5058, F1=0.5293
- Epoch 4: Loss=0.1295, mIoU=0.5341, F1=0.5787
- Epoch 5: Loss=0.0988, mIoU=0.5258, F1=0.5648
- Epoch 6: Loss=0.0811, mIoU=0.5487, F1=0.6018
- Epoch 7: Loss=0.0774, mIoU=0.5954, F1=0.6700
- Epoch 8: Loss=0.0696, mIoU=0.5249, F1=0.5622
- Epoch 9: Loss=0.0630, mIoU=0.6530, F1=0.7418
- Epoch 10: Loss=0.0591, mIoU=0.6089, F1=0.6872

当前轮次最佳结果: Round=2, Labeled=151, mIoU=0.6530, F1=0.7418

## Round 3

Labeled Pool Size: 239

- Epoch 1: Loss=0.3444, mIoU=0.5307, F1=0.5721
- Epoch 2: Loss=0.1997, mIoU=0.6174, F1=0.6983
- Epoch 3: Loss=0.1495, mIoU=0.6032, F1=0.6798
- Epoch 4: Loss=0.1304, mIoU=0.6477, F1=0.7344
- Epoch 5: Loss=0.1154, mIoU=0.6775, F1=0.7670
- Epoch 6: Loss=0.1087, mIoU=0.6745, F1=0.7639
- Epoch 7: Loss=0.1088, mIoU=0.6445, F1=0.7305
- Epoch 8: Loss=0.0978, mIoU=0.7059, F1=0.7963
- Epoch 9: Loss=0.0924, mIoU=0.6903, F1=0.7803
- Epoch 10: Loss=0.0891, mIoU=0.6890, F1=0.7805

当前轮次最佳结果: Round=3, Labeled=239, mIoU=0.7059, F1=0.7963

## Round 4

Labeled Pool Size: 327

- Epoch 1: Loss=0.4229, mIoU=0.5853, F1=0.6607
- Epoch 2: Loss=0.2092, mIoU=0.6333, F1=0.7179
- Epoch 3: Loss=0.1534, mIoU=0.6578, F1=0.7459
- Epoch 4: Loss=0.1311, mIoU=0.6289, F1=0.7122
- Epoch 5: Loss=0.1213, mIoU=0.6508, F1=0.7380
- Epoch 6: Loss=0.1175, mIoU=0.6586, F1=0.7466
- Epoch 7: Loss=0.1080, mIoU=0.6808, F1=0.7705
- Epoch 8: Loss=0.1060, mIoU=0.6779, F1=0.7674
- Epoch 9: Loss=0.0962, mIoU=0.7134, F1=0.8031
- Epoch 10: Loss=0.0952, mIoU=0.7020, F1=0.7919

当前轮次最佳结果: Round=4, Labeled=327, mIoU=0.7134, F1=0.8031

## Round 5

Labeled Pool Size: 415

- Epoch 1: Loss=0.3717, mIoU=0.5824, F1=0.6520
- Epoch 2: Loss=0.1856, mIoU=0.6550, F1=0.7427
- Epoch 3: Loss=0.1539, mIoU=0.6420, F1=0.7277
- Epoch 4: Loss=0.1362, mIoU=0.6879, F1=0.7779
- Epoch 5: Loss=0.1236, mIoU=0.6575, F1=0.7453
- Epoch 6: Loss=0.1225, mIoU=0.6983, F1=0.7883
- Epoch 7: Loss=0.1131, mIoU=0.7025, F1=0.7924
- Epoch 8: Loss=0.1093, mIoU=0.6711, F1=0.7601
- Epoch 9: Loss=0.1027, mIoU=0.7016, F1=0.7914
- Epoch 10: Loss=0.0990, mIoU=0.7186, F1=0.8078

当前轮次最佳结果: Round=5, Labeled=415, mIoU=0.7186, F1=0.8078

## Round 6

Labeled Pool Size: 503

- Epoch 1: Loss=0.3324, mIoU=0.6212, F1=0.7028
- Epoch 2: Loss=0.1725, mIoU=0.6763, F1=0.7659
- Epoch 3: Loss=0.1476, mIoU=0.6555, F1=0.7430
- Epoch 4: Loss=0.1283, mIoU=0.6730, F1=0.7621
- Epoch 5: Loss=0.1222, mIoU=0.7193, F1=0.8086
- Epoch 6: Loss=0.1158, mIoU=0.7034, F1=0.7932
- Epoch 7: Loss=0.1083, mIoU=0.7304, F1=0.8188
- Epoch 8: Loss=0.1070, mIoU=0.7426, F1=0.8298
- Epoch 9: Loss=0.0993, mIoU=0.7401, F1=0.8276
- Epoch 10: Loss=0.0978, mIoU=0.7430, F1=0.8302

当前轮次最佳结果: Round=6, Labeled=503, mIoU=0.7430, F1=0.8302

## Round 7

Labeled Pool Size: 591

- Epoch 1: Loss=0.3721, mIoU=0.6317, F1=0.7157
- Epoch 2: Loss=0.1606, mIoU=0.6677, F1=0.7565
- Epoch 3: Loss=0.1380, mIoU=0.7080, F1=0.7979
- Epoch 4: Loss=0.1261, mIoU=0.7114, F1=0.8013
- Epoch 5: Loss=0.1175, mIoU=0.7165, F1=0.8061
- Epoch 6: Loss=0.1101, mIoU=0.7350, F1=0.8228
- Epoch 7: Loss=0.1067, mIoU=0.7435, F1=0.8305
- Epoch 8: Loss=0.1020, mIoU=0.7415, F1=0.8287
- Epoch 9: Loss=0.1009, mIoU=0.7115, F1=0.8010
- Epoch 10: Loss=0.0964, mIoU=0.7387, F1=0.8263

当前轮次最佳结果: Round=7, Labeled=591, mIoU=0.7435, F1=0.8305

## Round 8

Labeled Pool Size: 679

- Epoch 1: Loss=0.3155, mIoU=0.6549, F1=0.7426
- Epoch 2: Loss=0.1549, mIoU=0.6862, F1=0.7760
- Epoch 3: Loss=0.1288, mIoU=0.7170, F1=0.8067
- Epoch 4: Loss=0.1189, mIoU=0.7246, F1=0.8139
- Epoch 5: Loss=0.1113, mIoU=0.6907, F1=0.7804
- Epoch 6: Loss=0.1085, mIoU=0.7332, F1=0.8213
- Epoch 7: Loss=0.1017, mIoU=0.7203, F1=0.8093
- Epoch 8: Loss=0.0966, mIoU=0.7449, F1=0.8317
- Epoch 9: Loss=0.0932, mIoU=0.6888, F1=0.7785
- Epoch 10: Loss=0.0910, mIoU=0.7542, F1=0.8397

当前轮次最佳结果: Round=8, Labeled=679, mIoU=0.7542, F1=0.8397

## Round 9

Labeled Pool Size: 767

- Epoch 1: Loss=0.3435, mIoU=0.6203, F1=0.7015
- Epoch 2: Loss=0.1493, mIoU=0.6902, F1=0.7804
- Epoch 3: Loss=0.1287, mIoU=0.7197, F1=0.8091
- Epoch 4: Loss=0.1177, mIoU=0.6328, F1=0.7165
- Epoch 5: Loss=0.1140, mIoU=0.7083, F1=0.7979
- Epoch 6: Loss=0.1117, mIoU=0.7293, F1=0.8176
- Epoch 7: Loss=0.1026, mIoU=0.7319, F1=0.8203
- Epoch 8: Loss=0.1004, mIoU=0.7455, F1=0.8322
- Epoch 9: Loss=0.0962, mIoU=0.7220, F1=0.8109
- Epoch 10: Loss=0.0948, mIoU=0.7557, F1=0.8412

当前轮次最佳结果: Round=9, Labeled=767, mIoU=0.7557, F1=0.8412

## Round 10

Labeled Pool Size: 855

- Epoch 1: Loss=0.3300, mIoU=0.6709, F1=0.7601
- Epoch 2: Loss=0.1421, mIoU=0.6647, F1=0.7532
- Epoch 3: Loss=0.1185, mIoU=0.7012, F1=0.7912
- Epoch 4: Loss=0.1089, mIoU=0.7347, F1=0.8229
- Epoch 5: Loss=0.1039, mIoU=0.7221, F1=0.8111
- Epoch 6: Loss=0.0979, mIoU=0.7439, F1=0.8312
- Epoch 7: Loss=0.0941, mIoU=0.7426, F1=0.8301
- Epoch 8: Loss=0.0891, mIoU=0.7443, F1=0.8311
- Epoch 9: Loss=0.0865, mIoU=0.7543, F1=0.8398
- Epoch 10: Loss=0.0823, mIoU=0.7588, F1=0.8438

当前轮次最佳结果: Round=10, Labeled=855, mIoU=0.7588, F1=0.8438

## Round 11

Labeled Pool Size: 943

- Epoch 1: Loss=0.2264, mIoU=0.6531, F1=0.7405
- Epoch 2: Loss=0.1199, mIoU=0.6960, F1=0.7860
- Epoch 3: Loss=0.1064, mIoU=0.6986, F1=0.7884
- Epoch 4: Loss=0.0970, mIoU=0.7193, F1=0.8084
- Epoch 5: Loss=0.0916, mIoU=0.7310, F1=0.8192
- Epoch 6: Loss=0.0889, mIoU=0.7407, F1=0.8279
- Epoch 7: Loss=0.0844, mIoU=0.7484, F1=0.8347
- Epoch 8: Loss=0.0803, mIoU=0.7440, F1=0.8308
- Epoch 9: Loss=0.0840, mIoU=0.7097, F1=0.7992
- Epoch 10: Loss=0.0794, mIoU=0.7566, F1=0.8416

当前轮次最佳结果: Round=11, Labeled=943, mIoU=0.7566, F1=0.8416

## Round 12

Labeled Pool Size: 1031


--- [Checkpoint] 续跑开始时间: 2026-02-14T15:30:54.796355 ---

## Round 12

Labeled Pool Size: 1031

- Epoch 1: Loss=0.2131, mIoU=0.6479, F1=0.7345
- Epoch 2: Loss=0.1160, mIoU=0.6887, F1=0.7785
- Epoch 3: Loss=0.1008, mIoU=0.7226, F1=0.8117
- Epoch 4: Loss=0.0950, mIoU=0.7169, F1=0.8061
- Epoch 5: Loss=0.0909, mIoU=0.7440, F1=0.8309
- Epoch 6: Loss=0.0842, mIoU=0.7403, F1=0.8276
- Epoch 7: Loss=0.0793, mIoU=0.7427, F1=0.8296
- Epoch 8: Loss=0.0756, mIoU=0.7372, F1=0.8247
- Epoch 9: Loss=0.0723, mIoU=0.7384, F1=0.8258
- Epoch 10: Loss=0.0703, mIoU=0.7485, F1=0.8347

当前轮次最佳结果: Round=12, Labeled=1031, mIoU=0.7485, F1=0.8347

## Round 13

Labeled Pool Size: 1119

- Epoch 1: Loss=0.1926, mIoU=0.6507, F1=0.7379
- Epoch 2: Loss=0.1072, mIoU=0.7263, F1=0.8151
- Epoch 3: Loss=0.0955, mIoU=0.7177, F1=0.8069
- Epoch 4: Loss=0.0875, mIoU=0.7092, F1=0.7987
- Epoch 5: Loss=0.0833, mIoU=0.7316, F1=0.8198
- Epoch 6: Loss=0.0798, mIoU=0.7401, F1=0.8274
- Epoch 7: Loss=0.0768, mIoU=0.7504, F1=0.8365
- Epoch 8: Loss=0.0732, mIoU=0.7528, F1=0.8387
- Epoch 9: Loss=0.0718, mIoU=0.7636, F1=0.8478
- Epoch 10: Loss=0.0674, mIoU=0.7619, F1=0.8463

当前轮次最佳结果: Round=13, Labeled=1119, mIoU=0.7636, F1=0.8478

## Round 14

Labeled Pool Size: 1207

- Epoch 1: Loss=0.2476, mIoU=0.6220, F1=0.7036
- Epoch 2: Loss=0.1085, mIoU=0.6785, F1=0.7679
- Epoch 3: Loss=0.0943, mIoU=0.7311, F1=0.8199
- Epoch 4: Loss=0.0876, mIoU=0.7106, F1=0.8001
- Epoch 5: Loss=0.0826, mIoU=0.7319, F1=0.8201
- Epoch 6: Loss=0.0792, mIoU=0.7256, F1=0.8142
- Epoch 7: Loss=0.0760, mIoU=0.7495, F1=0.8358
- Epoch 8: Loss=0.0733, mIoU=0.7591, F1=0.8440
- Epoch 9: Loss=0.0698, mIoU=0.7413, F1=0.8284
- Epoch 10: Loss=0.0688, mIoU=0.7480, F1=0.8343

当前轮次最佳结果: Round=14, Labeled=1207, mIoU=0.7591, F1=0.8440

## Round 15

Labeled Pool Size: 1295

- Epoch 1: Loss=0.1788, mIoU=0.6786, F1=0.7682
- Epoch 2: Loss=0.0996, mIoU=0.7063, F1=0.7963
- Epoch 3: Loss=0.0872, mIoU=0.7172, F1=0.8064
- Epoch 4: Loss=0.0792, mIoU=0.6818, F1=0.7713
- Epoch 5: Loss=0.0746, mIoU=0.7500, F1=0.8361
- Epoch 6: Loss=0.0725, mIoU=0.7517, F1=0.8375
- Epoch 7: Loss=0.0688, mIoU=0.7553, F1=0.8406
- Epoch 8: Loss=0.0661, mIoU=0.7538, F1=0.8392
- Epoch 9: Loss=0.0637, mIoU=0.7604, F1=0.8450
- Epoch 10: Loss=0.0615, mIoU=0.7463, F1=0.8328

当前轮次最佳结果: Round=15, Labeled=1295, mIoU=0.7604, F1=0.8450


## 实验汇总

预算历史: [151, 151, 239, 327, 415, 503, 591, 679, 767, 855, 943, 1031, 1119, 1207, 1295]
ALC: 0.6699
最终 mIoU: 0.7604
最终 F1: 0.8450
