# 数据集设置方案（官方 Train/Val/Test 对齐）与代码落盘修改方案

## 目标

- 数据集划分严格对齐 Landslide4Sense 官方目录结构：
  - Train = `TrainData`
  - Val = `ValidData`
  - Test = `TestData`
- Active Learning（AL）仅在 `TrainData` 上进行“标注池/未标注池”闭环；`ValidData` 仅用于闭环验证（控制信号）；`TestData` 仅用于最终一次性报告评估。
- 不再从 `TrainData` 里额外切 `val/test`（避免内部划分导致的口径不一致）。

## 数据集对应关系（代码级）

### 目录映射

| 语义 split | 官方目录 | 用途 |
| --- | --- | --- |
| train | `TrainData/img` + `TrainData/mask` | 训练数据源（AL 只从这里取样本） |
| val | `ValidData/img` + `ValidData/mask` | 每轮验证（闭环控制/早停/模型选择） |
| test | `TestData/img` (+ 可选 `TestData/mask`) | 最后一轮最终评估（仅一次） |

当前数据量（h5 文件数量）：
- TrainData：3799
- ValidData：245
- TestData：800

### AL 轮次与预算口径

采用“40% 标注预算 + rd15 + 1 次最终 test 评估”的口径：
- 预算：`TrainData` 的约 40% 作为最终累计标注量（约 1520 张）
- rd：15 轮用于 AL 选样（每轮新增标注）
- final：第 16 轮不再选样，仅训练/汇报，并执行最终 test 评估

建议配置的一组可复现参数（示例）：
- `N_ROUNDS = 16`（15 轮 AL + 1 轮 final/test）
- `INITIAL_LABELED_SIZE = 0.05`（约 190 张冷启动）
- `QUERY_SIZE = 88`（190 + 15*88 = 1510，约等于 40%）
- `TOTAL_BUDGET = 1510`（显式设置，保证预算与实际标注量一致）

## 推荐运行时配置（官方固定口径）

- 训练集来源固定为官方 `TrainData`（用于 AL 的 labeled/unlabeled 池）
- 验证集固定为官方 `ValidData`（每轮闭环验证）
- 最终报告评估固定为官方 `TestData`（仅最后一轮一次）
- pools 仅承载 `TrainData` 的 labeled/unlabeled（AL 闭环）
- `val_pool.csv` / `test_pool.csv` 仅作为占位文件（保持“4 个 CSV 文件都存在”的兼容约束）

## 需要落盘的代码修改点（建议最小改动集）

### 1) 最终评估固定走官方 TestData（不依赖 test_pool.csv）

文件：[main.py](file:///Users/anykong/AD-KUCS/AAL_SD/src/main.py)

修改要点：
- 初始化 pools 时允许 `test_pool.csv` 为空（占位文件）。
- 最后一轮固定执行 `report_eval`：构建 `Landslide4SenseDataset(DATA_DIR, split="test")` 的 DataLoader 进行评估。

预期效果：
- 训练闭环与验证完全不接触 TestData
- 最终报告的 test 指标来自官方 TestData（论文口径更标准）

### 2) 让 TestData 支持读取 mask（用于计算 mIoU/F1）

文件：[dataset.py](file:///Users/anykong/AD-KUCS/AAL_SD/src/core/dataset.py)

当前行为：
- `split="test"` 时 `mask_dir=None`，`__getitem__` 返回 `(image, img_name)`，无法走 `Trainer.evaluate(images, masks)`。

修改要点：
- 当 `TestData/mask` 存在时，将 `mask_dir` 指向该目录。
- `__getitem__` 对 test split：
  - 若 mask 可用：返回 `(image, mask)`（与 train/val 对齐）
  - 若 mask 不可用：保持返回 `(image, img_name)`，但此时不应调用 `Trainer.evaluate`（需要在 main 里做保护或改用无监督评估路径）。

预期效果：
- 官方 TestData 若提供 mask，则可直接计算分割指标并写入最终报告。

### 3) pools 生成逻辑与“官方 Val/Test”模式对齐

文件：[data_preprocessing.py](file:///Users/anykong/AD-KUCS/AAL_SD/src/core/data_preprocessing.py)

修改要点：
- `val_pool.csv` / `test_pool.csv` 固定写入空表（仅包含表头 `sample_id`），不参与训练或评估。
- labeled/unlabeled 仍按 `TrainData` 全量生成：
  - `labeled_pool` 初始占比由 `INITIAL_LABELED_SIZE` 控制
  - `unlabeled_pool = TrainData - labeled_pool`

预期效果：
- 不破坏现有“4 个 CSV 文件都存在”的兼容约束
- 同时避免内部 val/test 切分引入非官方口径

### 4) 回归测试建议（可选但强烈推荐）

新增/调整用例：
- 初始化不应因为 `val_pool.csv/test_pool.csv` 为空而失败
- 最终一轮应使用 `split="test"` 的官方数据集进行评估

## 学术严谨性说明（可直接写入论文/报告的方法部分）

- 训练/选样闭环严格限制在 TrainData：避免测试集泄漏，保证可复现性与可比性。
- ValidData 固定且每轮一致：作为控制信号的稳定参照，不参与训练池。
- TestData 仅最后一次评估：避免对测试集过拟合与多次试探。
