# 实验日志

实验名称: agent_control_lambda
描述: Agent控制消融：仅允许set_lambda
开始时间: 2026-02-14T02:13:26.014221

## Round 1

Labeled Pool Size: 151

- Epoch 1: Loss=0.6114, mIoU=0.4731, F1=0.5148
- Epoch 2: Loss=0.3440, mIoU=0.5063, F1=0.5311
- Epoch 3: Loss=0.2125, mIoU=0.5087, F1=0.5340
- Epoch 4: Loss=0.1491, mIoU=0.5087, F1=0.5341
- Epoch 5: Loss=0.1133, mIoU=0.5165, F1=0.5480
- Epoch 6: Loss=0.0902, mIoU=0.6006, F1=0.6783
- Epoch 7: Loss=0.0784, mIoU=0.5873, F1=0.6597
- Epoch 8: Loss=0.0696, mIoU=0.5729, F1=0.6380
- Epoch 9: Loss=0.0648, mIoU=0.5932, F1=0.6664
- Epoch 10: Loss=0.0602, mIoU=0.5981, F1=0.6731

当前轮次最佳结果: Round=1, Labeled=151, mIoU=0.6006, F1=0.6783


**[ERROR] Round 1 失败: LLM Agent failed at Round 1: LLMAPIError: Error calling API: HTTPSConnectionPool(host='api.siliconflow.cn', port=443): Max retries exceeded with url: /v1/chat/completions (Caused by ProxyError('Unable to connect to proxy', NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x3080e5310>: Failed to establish a new connection: [Errno 61] Connection refused')))**


--- [Legacy Log] 续跑开始时间: 2026-02-14T17:21:58.747052 ---

## Round 2

Labeled Pool Size: 151

- Epoch 1: Loss=0.6101, mIoU=0.4714, F1=0.5111
- Epoch 2: Loss=0.3430, mIoU=0.5003, F1=0.5241
- Epoch 3: Loss=0.2023, mIoU=0.4995, F1=0.5170
- Epoch 4: Loss=0.1380, mIoU=0.5144, F1=0.5444
- Epoch 5: Loss=0.1011, mIoU=0.5618, F1=0.6221
- Epoch 6: Loss=0.0823, mIoU=0.5494, F1=0.6027
- Epoch 7: Loss=0.0760, mIoU=0.6009, F1=0.6774
- Epoch 8: Loss=0.0708, mIoU=0.5513, F1=0.6054
- Epoch 9: Loss=0.0643, mIoU=0.6489, F1=0.7369
- Epoch 10: Loss=0.0596, mIoU=0.6202, F1=0.7018

当前轮次最佳结果: Round=2, Labeled=151, mIoU=0.6489, F1=0.7369

## Round 3

Labeled Pool Size: 239

- Epoch 1: Loss=0.3431, mIoU=0.5238, F1=0.5607
- Epoch 2: Loss=0.1996, mIoU=0.5435, F1=0.5932
- Epoch 3: Loss=0.1538, mIoU=0.6269, F1=0.7097
- Epoch 4: Loss=0.1277, mIoU=0.6529, F1=0.7403
- Epoch 5: Loss=0.1186, mIoU=0.6728, F1=0.7625
- Epoch 6: Loss=0.1105, mIoU=0.6839, F1=0.7739
- Epoch 7: Loss=0.0997, mIoU=0.6838, F1=0.7737
- Epoch 8: Loss=0.0946, mIoU=0.7029, F1=0.7935
- Epoch 9: Loss=0.0914, mIoU=0.6464, F1=0.7326
- Epoch 10: Loss=0.0854, mIoU=0.7115, F1=0.8015

当前轮次最佳结果: Round=3, Labeled=239, mIoU=0.7115, F1=0.8015

## Round 4

Labeled Pool Size: 327

- Epoch 1: Loss=0.4217, mIoU=0.5601, F1=0.6192
- Epoch 2: Loss=0.2130, mIoU=0.6322, F1=0.7162
- Epoch 3: Loss=0.1665, mIoU=0.6115, F1=0.6904
- Epoch 4: Loss=0.1434, mIoU=0.6663, F1=0.7550
- Epoch 5: Loss=0.1305, mIoU=0.7061, F1=0.7961
- Epoch 6: Loss=0.1247, mIoU=0.6748, F1=0.7641
- Epoch 7: Loss=0.1196, mIoU=0.7028, F1=0.7927
- Epoch 8: Loss=0.1151, mIoU=0.7242, F1=0.8133
- Epoch 9: Loss=0.1113, mIoU=0.7159, F1=0.8054
- Epoch 10: Loss=0.1054, mIoU=0.7315, F1=0.8199

当前轮次最佳结果: Round=4, Labeled=327, mIoU=0.7315, F1=0.8199

## Round 5

Labeled Pool Size: 415

- Epoch 1: Loss=0.3784, mIoU=0.5460, F1=0.5971
- Epoch 2: Loss=0.1960, mIoU=0.6571, F1=0.7450
- Epoch 3: Loss=0.1537, mIoU=0.6886, F1=0.7786
- Epoch 4: Loss=0.1412, mIoU=0.6939, F1=0.7839
- Epoch 5: Loss=0.1310, mIoU=0.6807, F1=0.7704
- Epoch 6: Loss=0.1227, mIoU=0.7029, F1=0.7928
- Epoch 7: Loss=0.1186, mIoU=0.7035, F1=0.7935
- Epoch 8: Loss=0.1123, mIoU=0.7180, F1=0.8072
- Epoch 9: Loss=0.1068, mIoU=0.7258, F1=0.8147
- Epoch 10: Loss=0.1038, mIoU=0.6903, F1=0.7801

当前轮次最佳结果: Round=5, Labeled=415, mIoU=0.7258, F1=0.8147

## Round 6

Labeled Pool Size: 503

- Epoch 1: Loss=0.3266, mIoU=0.5773, F1=0.6444
- Epoch 2: Loss=0.1610, mIoU=0.6455, F1=0.7318
- Epoch 3: Loss=0.1304, mIoU=0.6573, F1=0.7451
- Epoch 4: Loss=0.1223, mIoU=0.6902, F1=0.7802
- Epoch 5: Loss=0.1164, mIoU=0.6820, F1=0.7717
- Epoch 6: Loss=0.1070, mIoU=0.7139, F1=0.8035
- Epoch 7: Loss=0.1020, mIoU=0.6969, F1=0.7871
- Epoch 8: Loss=0.1019, mIoU=0.7277, F1=0.8168
- Epoch 9: Loss=0.0948, mIoU=0.7324, F1=0.8207
- Epoch 10: Loss=0.0905, mIoU=0.7161, F1=0.8054

当前轮次最佳结果: Round=6, Labeled=503, mIoU=0.7324, F1=0.8207

## Round 7

Labeled Pool Size: 591

- Epoch 1: Loss=0.3706, mIoU=0.6307, F1=0.7144
- Epoch 2: Loss=0.1632, mIoU=0.6882, F1=0.7784
- Epoch 3: Loss=0.1355, mIoU=0.6767, F1=0.7662
- Epoch 4: Loss=0.1227, mIoU=0.6649, F1=0.7534
