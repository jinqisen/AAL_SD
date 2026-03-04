# Coding Plan: Async Agent（可选）& Python Pro Refactoring（已完成）

## Objectives
- 提供 `AsyncAgentManager`，用于并发/异步 LLM 交互（可选能力）。
- 将 Agent 相关代码按 “Python Pro” 标准重构（类型标注、结构清晰）。
- 在不引入新依赖的前提下，支持 Async/Await 的性能优化路径。
- 保持测试覆盖与可回归。

## Constraints
- Prioritize Python Standard Library.
- Use `asyncio`.
- Avoid adding new dependencies if possible (use `asyncio.to_thread` with `requests` if `aiohttp` is not available/desired).

## Current Status
- `src/agent/async_agent_manager.py` 与 `src/agent/utils.py` 已存在并通过单测回归。
- 训练主流程目前仍使用同步 `AgentManager`（`ActiveLearningPipeline` 调用路径未切换到 async），因此 async 能力属于“可用但未默认启用”。

## Steps

### Phase 1: Infrastructure & Common Logic
- [x] Extract message parsing and validation logic from `AgentManager` into a reusable module (e.g., `src/agent/utils.py`).
- [x] Ensure existing `AgentManager` uses these shared utilities.
- [x] Strengthen validation logic to strictly enforce ReAct format (reject trailing text).

### Phase 2: Async Agent Implementation
- [x] Create `src/agent/async_agent_manager.py`.
- [x] Implement `AsyncSiliconFlowClient` (using `asyncio.to_thread` wrapping `requests` to avoid new deps, or check if `aiohttp` is present).
- [x] Implement `AsyncAgentManager` with `run_cycle` as an async method.
- [x] Add Type Hints throughout.

### Phase 3: Integration & Testing
- [x] Create unit tests for validation logic in `src/tests/test_validation.py`.
- [ ] Create unit tests for `AsyncAgentManager` in `src/tests/test_async_agent.py`（可选）。
- [ ] 将训练入口增加可配置开关以选择 Sync/Async Agent（可选，涉及 `src/main.py` 与运行脚本）。

### Phase 4: Refactoring Existing Code
- [x] Review `src/agent/agent_manager.py` for "Python Pro" improvements (typing, docs).
- [x] Apply changes.

## Progress
- [x] Core refactoring and Async implementation completed.
- [x] Validation logic hardened and tested.
- [ ] Async Agent integration into training loop is optional and not enabled by default.

---

# Coding Plan: Overfit-Aware Controller Guard（进行中）

## Objectives
- 将 train_val_cos 等过拟合信号纳入 Agent 观测空间。
- 提供可开关的 overfit-aware 保护：在过拟合时限制 λ 与 query_size。
- 保持默认行为不变（关闭开关时与现有实验可复现）。

## Current Status
- 已在训练主循环聚合 train_val_cos，并写入 trace 事件 overfit_signal。
- 已在 get_system_status 返回 overfit 指标与阈值元信息。
- 已实现 overfit-aware λ/query_size guard（通过 OVERFIT_AWARE_GUARD 或实验 overfit_guard 开关启用）。

## Verification
- 运行 pytest 回归。
- 对现有 run trace 运行分析脚本，确认导出列与图生成正常。

---

# Coding Plan: Full Model λ Warmup + 风险闭环（已完成）

## Objectives
- R1 固定 λ=0.0（纯不确定性采样）。
- R2 固定 λ=0.20-0.25（默认 0.22）作为 warmup。
- 从 R3 起启用风险驱动闭环增减规则。
- full_model 路径移除 set_alpha/Sigmoid 依赖。
- 在低过拟合风险且 K 分布显著高于 U 时，允许 λ 小幅上调以利用 K。

## Implementation
- 在 full_model 配置中新增 `lambda_policy`，并关闭 `set_lambda` 权限。
- 在 Toolbox 内实现按 round 缓存的一次性 λ 策略应用，并写入 trace 事件 `lambda_policy_apply`。
- 在主流程对 AD-KUCS sampler 侧显式注入 round λ，避免回落到 Sigmoid 权重。

## Verification
- pytest 全量回归通过。

---

# Coding Plan: 实验方案审查与落地改进（进行中）

## Objectives
- 审查消融/基线设置与《AAL-SD 整合后完整实验方案》对齐度。
- 审查三层数据采集（L1-L3）与分析产物完备性。
- 输出两份文档：完备性分析报告 + 可落地的实验改进实施方案。

## Current Status
- 已完成：现有实验矩阵与主流程采集点盘点。
- 已完成：完备性分析报告与实施方案输出。
- 进行中：按实施方案改造代码（实验矩阵、采样基线、控制器、L3采集、分析脚本）。

## Steps
1. 扩展 ablation_config 与多 seed 运行清单，补齐 A2/A3/A4/A5/A7 与 B6/B7。
2. 新增 DIAL-style 与 Wang-style 采样基线并接入主流程。
3. 引入规则/随机 λ 控制器，支持无 LLM 的可复现实验组。
4. 增加 round_summary 与 L3 采集事件，输出统一 trace schema。
5. 更新分析脚本以读取 round_summary 与 l3 事件。
6. 补充采样器单测并回归 pytest。

## Progress
- [x] 输出完备性分析报告与实施方案。
- [x] 完成实验矩阵与基线实现。
- [x] 完成规则/随机控制器与 L3 数据采集。
- [x] 更新分析脚本与测试回归。
- [x] 调整批跑并发默认值为 agent=1 与 baseline=1 并保持双池并发。

---

# Coding Plan: Fullmodel性能优化方案文档输出（已完成）

## Objectives
- 输出完整、可落代码的 fullmodel 性能优化方案文档。
- 方案以独立实验形式存在，不影响既有 full_model 与消融。
- 严格对齐 registry 映射模式与可复现实验流程。

## Steps
1. 梳理 registry 映射模式与相关入口。
2. 整理方案模块与文件级改动清单。
3. 形成完整可落代码的 MD 文档。

## Progress
- [x] 文档输出与核对一致性。

---

# Coding Plan: Fullmodel性能优化方案代码落地（进行中）

## Objectives
- 以独立实验实现 U 校准、风险 CI 触发与 λ 平滑策略。
- 保持 registry 映射模式与既有实验不受影响。
- 完成最小必要改动并通过 pytest 回归。

## Steps
1. 扩展 ablation_config 新实验配置。
2. 扩展 toolbox 与 ADKUCSSampler 支持新策略配置。
3. 调整 sampler 构建与管线注入当前轮次。
4. 完成 pytest 回归与自检记录。

## Progress
- [x] 已新增独立实验配置 full_model_v5_calibrated_risk。
- [x] 已完成策略扩展与管线接入。
- [x] 已完成 pytest 回归与自检记录。

---

# Coding Plan: 采样阶段共享内存膨胀/内存泄漏修复（已完成）

## Objectives
- 降低采样阶段内存占用，避免 DataLoader worker 共享内存被大张量放大。
- 保持采样算法语义不变（U/K/BALD 的定义与排序逻辑不变）。

## Implementation
- 采样输入从“存整张 prob_map / mc_predictions”改为“存标量 uncertainty_score + feature（可选 pos_area）”。
- BALD 计算改为在线聚合互信息（不再堆叠全量 MC 预测数组）。
- candidate_constraints（pred_pos_area_quota）优先读取预计算的 pos_area，保留 prob_map 回退路径。

## Verification
- `python -m pytest -q` 回归：69 passed。
- `run_parallel_strict.py --resume run_src_full_model_with_baselines_seed45` 已成功续跑未完成实验。

---

# Coding Plan: monitor_and_recover 多 run 目录监听增强（已完成）

## Objectives
- 支持直接传入多个 run 目录（绝对/相对路径）进行监控。
- 提供多 run 的概览视图，便于快速判断各 run 进度与卡滞情况。
- 保持单 run 的输出与兼容性不受影响。

## Progress
- [x] 支持 --run-dirs 与 --run-ids 混合输入，并自动解析为 run_dir。
- [x] 增加 Multi-Run Overview 概览表格输出。
- [x] 用 seed42~46 的 run 目录做一次验证运行并记录关键输出。
