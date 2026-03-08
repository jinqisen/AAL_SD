# 模型数据加载整改方案 V1

## 1. 背景与目标

本方案覆盖从数据根目录解析、数据集切分（train/val/test + AL 的 labeled/unlabeled pools）、到 DataLoader 构建与一致性校验的全链路整改。

### 1.1 学术目标（严谨性）

- 严格保证 train/val/test 的数据边界，不允许任何隐式降级或“猜测式”目录回退导致的数据泄漏
- 初始标注集（cold start labeled pool）切分可复现、可审计、并在类别分布上尽可能稳定（避免极端样本集导致的无意义比较）
- 所有与数据切分与加载相关的关键元信息（数据指纹、切分策略、随机种子、版本）必须可追溯、可复验

### 1.2 工程目标（简洁可靠可扩展）

- 单一事实来源（Single Source of Truth）：一个地方定义数据布局与切分规则，其他模块只消费其产物
- 统一且可组合的“数据入口”：用明确的组件边界替代散落在 config / preprocessing / pipeline 的隐式规则
- 失败即失败（Fail Fast）：任何可能影响实验结论可靠性的异常都必须抛出并记录上下文，禁止吞异常

### 1.3 非目标（V1 不做）

- 不引入新的主动学习策略、模型结构或训练超参调度
- 不对训练过程（AMP、torch.compile、agent）做大规模架构重写（仅做与“实验可靠性/异常处理”强相关的最小修正建议）

---

## 2. 现状梳理（来自 src 的事实）

### 2.1 当前数据流摘要

- DATA_DIR 解析：优先环境变量，否则在若干候选路径中寻找；判定条件包含 `TrainData/img` 或 `images`
- 训练候选池生成：`DataPreprocessor.create_data_pools()` 扫描 `DATA_DIR/images`（优先）或回退到 `DATA_DIR/TrainData/img`，生成 `labeled_pool.csv/unlabeled_pool.csv`
- 训练数据：`full_dataset = Landslide4SenseDataset(DATA_DIR, split=train)` 只读取 `TrainData/img`，再用 `sample_id -> index` 映射得到 labeled subset
- 验证数据：`Landslide4SenseDataset(DATA_DIR, split=val)` 读取 `ValidData/*`
- 测试数据：最后一轮才构造 `Landslide4SenseDataset(DATA_DIR, split=test)`，并强制要求存在 `TestData/mask`

### 2.2 学术问题清单（严谨性风险）

1. **潜在数据泄漏入口：preprocessing 优先扫描 `DATA_DIR/images`**
   - 若数据目录里 `images/` 包含非 train 的样本（或包含 train/val/test 混合），将被纳入池并进入训练/选样循环
2. **初始 labeled pool 切分不分层（stratify）**
   - 语义分割任务存在类别极不均衡风险，随机 5% 可能抽到正类极少甚至为 0 的集合，导致冷启动比较缺乏可解释性
3. **切分产物缺少强审计与版本化**
   - 当前 pools 仅存 CSV，缺少“数据指纹 + 策略参数 + 代码版本”的强绑定，跨机器/跨次运行难以证明一致

### 2.3 工程问题清单（冗余/过度设计/一致性）

1. **数据布局规则不一致**
   - Config / Preprocessor 支持 `images/`，Dataset 只认 `TrainData/ValidData/TestData`，导致行为不闭环
2. **重复扫描与冗余字段**
   - pools CSV 存 `image_path/mask_path`，但训练主要依赖 `sample_id` 映射；同时 Dataset 再次扫目录构造文件列表
3. **异常处理不严肃**
   - 多处存在 `except Exception: pass` 或静默继续（例如 fresh 模式删除旧状态失败仍继续），会导致“混跑/脏状态”污染实验结论

---

## 3. V1 整改原则（强约束）

### 3.1 严禁隐式降级

- 禁止自动从 `images/` 回退到 `TrainData`（或相反）的“猜测式”策略
- 禁止在缺少关键资源时（例如目录缺失、mask 缺失、pool 不一致）继续训练或继续实验
- 任何兼容旧格式必须显式开关，并在日志/manifest 中记录启用原因与风险提示

### 3.2 异常处理政策（Fail Fast + 上下文）

- 禁止 `except Exception: pass` 出现在数据切分/加载/校验路径上
- 允许捕获异常的唯一理由：
  - 增补上下文（路径、split、计数、样本 id、配置项）后重新抛出
  - 将异常转为明确的领域异常（见 5.3）
- fresh 模式必须做到“清理失败则失败”，避免混用旧 checkpoint/status/trace

---

## 4. 目标架构（V1：简洁分层）

### 4.1 分层与职责

**(A) DatasetLayout（数据布局与发现）**
- 输入：`data_root`
- 输出：严格的 split 目录与文件列表（train/val/test），以及必要的统计（数量、文件扩展名一致性、mask 是否存在）
- 约束：只支持一种明确布局；若要支持多布局，用显式 `layout_type`，默认只允许标准布局

**(B) SplitPolicy（切分策略）**
- 负责：在 train 候选集中生成初始 labeled/unlabeled pools（以及未来扩展到 KFold/Group split）
- 约束：策略必须可复现（seed）且可审计（参数落盘）

**(C) PoolStore（池的持久化与原子写入）**
- 负责：读写 pools 与 manifest；强一致性校验（完整性、无重叠、计数匹配）
- 约束：写入必须原子；读取必须验证 schema/version

**(D) DatasetFactory / DataLoaderFactory（构建 Dataset 与 Loader）**
- 负责：只接受 Layout + Pools 的产物，不再自行“猜测目录”
- 约束：构建阶段做最后一道一致性校验（例如 subset indices 全覆盖、无越界）

### 4.2 依赖方向（避免循环依赖）

- `core/dataset.py` 只负责样本读取，不依赖 experiments/agent
- `core/data_preprocessing.py` 被拆解或降级为纯策略层（SplitPolicy），不直接扫多种目录
- `main.py` 只编排（orchestration），不承载布局推断与数据一致性规则

---

## 5. V1 具体整改方案（可落地条目）

### 5.1 统一并收紧数据布局（消除泄漏入口）

**变更点**
- 定义标准布局：`TrainData/img, TrainData/mask, ValidData/img, ValidData/mask, TestData/img, TestData/mask`
- V1 默认只允许标准布局；若目录不完整直接抛错

**替换/删除**
- 移除 preprocessing 对 `DATA_DIR/images` 的优先扫描逻辑（或仅在显式 `LAYOUT_TYPE=flat_images` 时启用）
- Config 的 `_looks_like_dataset_root` 判定应只认可标准布局（至少 train/val 必备）

**验收标准**
- 在任意机器上，若 `ValidData` 缺失：直接失败，并提示缺失路径
- 任何情况下，train pools 只能来源于 TrainData，不可跨 split

### 5.2 初始 labeled pool 的学术严格化（分层与审计）

**策略**
- 采用“按样本是否包含正类像素”的二元分层：
  - `has_positive = (mask.sum() > 0)` 或更稳健的阈值（例如正类像素比例 > eps）
- `train_test_split(..., stratify=has_positive, shuffle=True, random_state=seed)`

**代价与折中**
- 需要读取 mask 以计算 `has_positive`。V1 建议在第一次建池时离线计算并缓存为 `train_manifest.json`（见 5.4），避免每次训练重复扫描

**验收标准**
- labeled pool 与 unlabeled pool 在 `has_positive` 的比例上与全体 train 候选集偏差受控（可设置阈值并写入报告）

### 5.3 定义领域异常与错误码（严肃异常处理）

**建议异常类型（V1 至少覆盖）**
- `DatasetLayoutError`：目录结构不符合预期、文件缺失、mask 缺失
- `PoolIntegrityError`：labeled/unlabeled 重叠、并集不等于候选集、索引映射缺失
- `SplitPolicyError`：切分参数非法（比例不在 (0,1)、seed 为空等）
- `DataReadError`：样本文件损坏、h5 键缺失、shape 不符合预期

**政策**
- 捕获后必须携带上下文再抛出；日志必须包含 error code、split、路径、样本 id

### 5.4 引入强审计 manifest（单文件即可）

**V1 manifest 内容（建议 JSON，和 CSV 同目录）**
- dataset_fingerprint：train/val/test 的文件名哈希与计数（split 级别）
- layout_type：standard
- split_policy：名称 + 参数（initial_labeled_ratio、seed、stratify_key）
- pools_schema_version：整数
- created_at：时间戳

**验收标准**
- 任何 resume 都必须校验当前目录 fingerprint 与 manifest 一致；不一致则直接失败（除非显式 override）

### 5.5 去冗余：精简 pools 存储与索引映射

**两个可选方向（V1 推荐 A）**

**A. pools 仅存 `sample_id`（推荐）**
- DatasetLayout 提供稳定排序的 `sample_id <-> index` 映射来源
- PoolStore 写入 `labeled_ids.csv/unlabeled_ids.csv`（或 JSON）
- 训练侧不再需要保存 image_path/mask_path

**B. pools 存相对路径并直接读取**
- 训练 Dataset 从 pools 文件驱动样本列表，不再扫描目录
- 优点：完全消除“目录扫描与 sample_id 映射”的二义性
- 缺点：更改数据根目录/迁移时需要处理相对路径解析

### 5.6 fresh/resume 的状态一致性（避免脏状态）

**规则**
- fresh 模式下：删除 checkpoint/status/trace 任一失败都应抛错并终止
- resume 模式下：若检测到 pools 不完整、manifest 不匹配、或 pool_integrity 不通过，应直接失败

---

## 6. 扩展性设计（V1 预留接口，不堆叠实现）

### 6.1 SplitPolicy 插件化

- `InitialPoolSplitPolicy`（V1 仅实现这一种）
- 未来可加：
  - `KFoldSplitPolicy`（仅用于论文对比，不进入默认训练路径）
  - `GroupSplitPolicy`（按区域/事件分组，防止空间泄漏）

### 6.2 LayoutType 扩展（默认关闭）

- `standard`（默认、必须）
- `flat_images`（显式开启才允许，用于兼容某些数据集组织方式；必须提供 train/val/test 列表文件避免混入）

---

## 7. 测试与验收（V1 必须可自动化）

### 7.1 单元测试（建议新增/改造）

- Layout 校验：缺目录/缺 mask/扩展名不一致必须失败
- Pool 完整性：无重叠、并集覆盖、映射不丢样本
- Stratify：在构造的 toy 数据上验证分层生效

### 7.2 端到端验收

- 在同一数据目录上重复运行两次：manifest 一致、初始 pools 一致（seed 固定）
- 故意制造异常（例如删掉 ValidData/mask）：应立即失败并输出明确错误码与路径

---

## 8. 分阶段落地计划（V1）

1. 收紧 Layout：禁用 `images/` 隐式扫描与 DATA_DIR 判定回退
2. 引入 manifest：在 pool 生成时写入并在 resume 时强校验
3. 初始切分 stratify：实现 `has_positive` 分层并写入审计指标
4. 严格异常处理：清理 `except Exception: pass`（至少覆盖数据入口与 fresh/resume）
5. 精简 pools 存储：只保留必要字段，消除重复扫描/不一致源

---

## 9. 架构影响评估

- 影响等级：中
- 影响面：Config、数据预处理、主流程编排、测试用例
- 兼容性：对非标准数据目录将从“可能能跑”变为“明确失败”，符合严谨性目标

