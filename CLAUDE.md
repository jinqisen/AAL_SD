# CLAUDE.md
本文件为 Claude Code 在本仓库内工作提供“长期记忆”。只写那些 Claude 仅靠读代码不容易稳定推断的内容；保持简短、可执行、可验证。

## 项目概览（AAL_SD）
- 目标：面向滑坡语义分割任务的主动学习（Active Learning）实验框架，并集成 LLM Agent 做策略/超参控制与消融对比。
- 主要入口：并发批量跑消融实验见 @src/run_parallel_strict.py；核心流水线类见 @src/main.py（ActiveLearningPipeline）。
- 数据与产物：数据池 CSV 在 data/pools/；运行日志与状态在 results/runs/；checkpoint 在 results/checkpoints/；共享日志在 logs/parallel/。
- 训练预算口径：当前默认每轮训练 epochs 固定（见 @src/config.py 的 FIX_EPOCHS_PER_ROUND/EPOCHS_PER_ROUND），Agent 不再动态调整 epochs。

## 目录结构速览
- src/
  - agent/：LLM Agent、工具箱、prompt 组合（重点：@src/agent/prompt_template.py、@src/agent/toolbox.py）
  - core/：数据预处理、Dataset、模型、训练器、采样器与 checkpoint（重点：@src/core/trainer.py、@src/core/sampler.py、@src/core/checkpoint.py）
  - baselines/：对比采样策略（random/entropy/coreset/bald/llm_*）
  - experiments/：消融配置与批量实验编排（重点：@src/experiments/ablation_config.py）
  - analysis/：画图与论文图表脚本（重点：@src/analysis/plot_paper_figures.py）
- 监控与恢复：@src/monitor_and_recover.py（读取 results/runs/<run_id> 下的 *_status.json / *_trace.jsonl）
- tests/ 与 src/tests/：单元与回归测试（pytest 可直接收集）
- data/：原始数据与缓存（体量大，避免无目的遍历/读取）
- results/：实验产物（同样体量大，除非排错否则不要全量读取）

## 工作方式（对 Claude 的约束）
- 优先在 src/ 与 tests/ 内定位问题与实现修复；除非任务明确要求，不要改动 data/ 与 results/ 中的文件。
- 不要对大目录做“全量读取/逐文件检查”；需要时用更小的范围与更精确的匹配（例如只看某个 run_id 目录或某个 trace 文件）。
- 任何涉及训练/实验流程的改动，都要明确“如何验证正确”：优先用现有测试与最小可复现实验（而不是长时间全量训练）。
- 处理 bug 时优先定位“可复现样例 + 最小失败用例”，再修复根因；不要用静默捕获异常掩盖问题。

## 常用命令（从仓库根目录执行）
- 跑测试（推荐）：python -m pytest -q
- 跑单测用例：python -m pytest -q tests/test_checkpoint.py::TestCheckpointManager::test_save_and_load
- 启动一轮严格并发批量实验（会生成新的 run_id）：python src/run_parallel_strict.py
- 续跑已有 run：python src/run_parallel_strict.py --resume <run_id>
- 多 seed 复现实验：python src/experiments/run_multi_seed.py --run_id <run_id> --start fresh
- 生成论文图/轨迹图：bash make_figures.sh <run_id>
- 检查某次 run 的 trace/log/pools/state 一致性：python inspect_integrity.py <run_id>
- 监控最新 run 的运行状态：python src/monitor_and_recover.py

## 变更后自检（默认期望）
- 改动涉及 Python 逻辑：至少跑一次 python -m pytest -q
- 改动涉及恢复/续跑/trace：跑一次 python inspect_integrity.py <run_id> 或对 tests/ 下相关用例补充覆盖

## LLM / Agent 配置（安全与可复现）
- LLM 配置默认从 @src/llm_config.json 读取；也可通过环境变量 AAL_SD_LLM_CONFIG_PATH（或 LLM_CONFIG_PATH）指定路径，详见 @src/config.py。
- 任何 API Key/Token 都必须来自环境变量或本地不入库配置文件；不要把密钥写入仓库文件、日志或测试里。
- 在严格模式下，部分 agent 消融会要求存在 LLM_API_KEY（例如 STRICT_INNOVATION 相关），缺失会直接报错（见 @src/main.py 与 @src/config.py）。
 
## 运行与复现（惯例）
- 如果脚本/文档中出现硬编码解释器路径（例如 resume 脚本），优先保持其可配置（环境变量或 python/python3），不要把本机绝对路径扩散到更多位置。
- 排查实验结果问题时，优先看：<exp>_status.json 与 <exp>_trace.jsonl，再看 <exp>.md 与 logs/parallel/。

## 产物与约定（便于排错）
- 每个 run_id 的运行目录：results/runs/<run_id>/
  - <exp>.md：实验报告（含“实验汇总”用于判定完成）
  - <exp>_status.json：进度与状态
  - <exp>_trace.jsonl：逐事件追踪（用于恢复/回滚判定）
  - reports/：监控脚本生成的阶段/异常报告（可选）
- 每个 run_id 的 checkpoint：results/checkpoints/<run_id>/<exp>_state.json
- 每个实验的数据池：data/pools/<run_id>/<exp>/labeled_pool.csv、unlabeled_pool.csv、test_pool.csv
- 文件写入与一致性：原子写/容错读/锁更新统一用 utils.atomic_write_json / utils.read_json_dict / utils.locked_update_json / utils.append_jsonl

## 关键代码入口（快速定位）
- 实验批跑与续跑：@src/run_parallel_strict.py
- 核心流水线：@src/main.py（ActiveLearningPipeline）
- 全局配置与路径/LLM 读取：@src/config.py
- 消融矩阵：@src/experiments/ablation_config.py
- 恢复/一致性检查脚本：@inspect_integrity.py

## 论文披露文件维护约定
- 论文需要公开的实现/实验细节统一维护在：@AAL-SD-Doc/survey/PAPER_DISCLOSURE_DETAILS.md
- 任何会影响论文结论解释的改动，都需要同步更新该文件，其中包括但不限于：
  - 主动学习协议（pool 定义、预算口径、round 内训练与选模规则）
  - 指标口径与 objective 映射（例如 ALC、val/test mIoU 的定义与解析逻辑）
  - LLM/Agent 行为与失败处理策略（包含 selection_guardrail、selection_repair 等系统性机制）
  - Auto-tuning 多保真/多 seed 复核的流程与 acceptance gate
- 在实现/修改上述逻辑时，除代码变更和测试外，默认还需要：
  - 检查 PAPER_DISCLOSURE_DETAILS.md 是否需要补充/修订
  - 让披露内容保持与当前实现严格一致，避免论文描述与仓库行为不符
