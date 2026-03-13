AAL-SD Fullmode AB 算法参数自动优化闭环框架设计稿

0. 现状摘要与目标

指标	当前最优	目标	差距
mIoU (单次最佳)	0.7221 (ab_tune_hi_ep10, seed43)	≥ 0.74	~0.018
mIoU (A策略均值)	~0.70-0.72 (seed间方差 0.02-0.04)	≥ 0.74 稳定	~0.02-0.04
最佳配置来源	full_model_A_lambda_policy_ab_tune_hi_ep10	—	—
关键观察：

seed 间方差约 0.02-0.04，说明单次 0.74 可能通过参数微调+运气达到，但稳定 0.74 需要系统性改进
ab_tune_hi 的核心差异：lambda_smoothing_alpha=1.0（无EMA平滑）、lambda_max_step=0.20（更大单步）、LAMBDA_DOWN_COOLING_ROUNDS=1
baseline wang_style 也达到 0.7216，说明 K 的代表性信号本身有价值，但当前 lambda 控制策略可能过于保守
1. Result Analysis：参数→λ→观测 的因果链

1.1 完整因果链路图

+-------------------------------------------------------------------+
|                    实验参数层 (Experiment Config)                    |
|                                                                     |
|  +--------------+  +--------------+  +-------------------------+   |
|  | lambda动力学  |  | 风险控制参数  |  | 训练/结构参数            |   |
|  |              |  |              |  |                         |   |
|  | DELTA_UP     |  | OVERFIT_     |  | epochs_per_round        |   |
|  | DELTA_DOWN   |  | RISK_HI/LO  |  | ALPHA (sigmoid)         |   |
|  | CLAMP_MIN/MAX|  | TVC_MIN_HI  |  | late_stage_ramp         |   |
|  | smoothing_a  |  | EMA_ALPHA   |  | selection_guardrail     |   |
|  | max_step     |  | COOLING_RDS |  | warmup_rounds/lambda    |   |
|  | risk_trigger |  | GAIN_THRESH |  | uncertainty_only_rds    |   |
|  +------+-------+  +------+------+  +-----------+-------------+   |
+---------|-----------------|-----------------------|----------------+
          |                 |                       |
          v                 v                       v
+-------------------------------------------------------------------+
|                    lambda决策层 (Lambda Policy Engine)               |
|                                                                     |
|  Phase1: uncertainty_only -> Phase2: warmup -> Phase3: risk_ctrl   |
|  -> Phase4: late_stage_ramp -> Phase5: EMA + max_step clamp        |
|                                                                     |
|  输入信号:                                                          |
|    rollback_flag   <- miou_delta < -adaptive_threshold             |
|    overfit_risk    <- tvc_neg_rate + 0.5*max(0,-tvc_min)           |
|                       + 0.5*max(0,-tvc_last)                       |
|    severe_overfit  <- (overfit_risk > RISK_HI) AND/OR              |
|                       (tvc_key < TVC_MIN_HI)                       |
|    low_gain_streak <- consecutive rounds with                      |
|                       miou_delta < GAIN_THRESH & risk < RISK_LO    |
|                                                                     |
|  输出: lambda_t in [CLAMP_MIN, CLAMP_MAX]                          |
+-----------------------------+---------------------------------------+
                              |
                              v
+-------------------------------------------------------------------+
|                    选样层 (Sample Selection)                        |
|                                                                     |
|  Score(x) = (1-lambda_t) * U_norm(x) + lambda_t * K_norm(x)       |
|  -> Top-k selection -> [optional] guardrail U底线检查               |
|                                                                     |
|  U(x): 像素熵均值聚合                                               |
|  K(x): 1 - d(x)/d_max (KMeans簇中心距离反转)                       |
+-----------------------------+---------------------------------------+
                              |
                              v
+-------------------------------------------------------------------+
|                    观测层 (Observable Metrics)                      |
|                                                                     |
|  性能: mIoU, F1, ALC, miou_delta                                   |
|  稳定性: rollback_count, rollback_flag, epoch_std                  |
|  风险: overfit_risk, tvc_mean/min/max/last/neg_rate                |
|  选样质量: U/K分布统计(mean/median/p10/p90), final_score分布        |
|  lambda轨迹: lambda_effective, lambda_source, lambda变化历史        |
|  预算: labeled_count, remaining_budget                              |
|  数据分布: mask正样本比例, has_positive分布 (需pools+mask离线分析)   |
+-------------------------------------------------------------------+
1.2 关键参数对 lambda 的影响路径

参数	影响路径	对 lambda 的效果	对 mIoU 的间接效果
LAMBDA_DELTA_UP	risk_control 阶段，低风险+低增益时 lambda 上调步长	越大→lambda 上升越快→K 权重增长越快	适度增大可加速代表性探索，过大导致不稳定
LAMBDA_DELTA_DOWN	rollback/severe 时 lambda 下调步长	越大→回撤时 lambda 下降越猛→更快回到 U 主导	过大导致 lambda 振荡，过小导致风险累积
LAMBDA_CLAMP_MIN/MAX	硬边界	限制 lambda 的活动范围	CLAMP_MAX 过低会限制 K 的贡献上限
lambda_smoothing_alpha	EMA 平滑	=1.0 无平滑（直接用新值），<1.0 惰性越大	无平滑允许更快响应但可能抖动
lambda_max_step	单轮 lambda 最大变化	限制突变幅度	过小限制策略响应速度
OVERFIT_RISK_HI	severe 判定门槛	越高→越难触发 severe→lambda 下调越少	过高可能忽略真实过拟合
OVERFIT_RISK_LO	允许 lambda 上调的风险门槛	越高→越难满足上调条件→lambda 上升越慢	过低可能在风险期上调 lambda
LAMBDA_DOWN_COOLING_ROUNDS	连续下调冷却期	>0 时抑制连续下调	减少振荡但可能延迟风险响应
risk_trigger="ci"	用历史分位数做自适应阈值	相对异常检测，影响 severe 命中率	CI 窗口/分位数影响敏感度
late_stage_ramp	后期 lambda 地板线性抬升	强制后期 K 权重不低于某值	可提升后期代表性但可能牺牲 U
selection_guardrail	U 底线安全阀	当选中样本 U 偏低时强制降 lambda	保护探索质量但可能限制 K
1.3 从 trace 数据提取的分析维度

事件类型	关键字段	分析用途
round_summary	mIoU, f1, labeled_size, lambda_effective, lambda_source, rollback_flag, overfit_risk, miou_delta, sampler.lambda_policy_mode	每轮核心指标汇总
epoch_end	loss, mIoU, f1, grad.train_val_cos, grad.total_norm.mean/std, grad.cos_consecutive.mean	epoch级训练曲线 + 梯度诊断
lambda_policy_apply	round, lambda, phase, reason	lambda 决策过程追踪
l3_selection	top_items[].uncertainty/knowledge_gain/final_score	选样质量分析 (U/K分布)
selection	selected_ids, expected, selected, context.decision_reason	选中样本ID + agent决策理由
overfit_signal	round, overfit_risk, tvc_last, tvc_min, severe	过拟合信号详情
controller_step	round, action, lambda_before/after	agent控制动作
1.4 需要 pools + mask 离线分析的维度

trace 中的 l3_selection.top_items 提供了选中样本的 U/K/score 分布，足以支持决策解释与失效模式定位。但以下"数据分布"类指标需要额外数据源：

分析维度	数据源	方法
mask 正像素面积比例	results/pools/<run_id>/_base/*.csv → mask_path → .h5	读取 mask 计算 positive_ratio
has_positive 分布	同上	统计每轮新增样本中含正样本的比例
极端样本占比	同上	正样本面积 < 1% 或 > 50% 的样本比例
选样偏置检测	trace selection.selected_ids + pools mask	对比选中 vs 未选中的 mask 分布差异
离线分析脚本入口：

python src/analysis/analyze_selection_mask_distribution.py \
    results/runs/<run_id>/<exp>_trace.jsonl \
    --output results/runs/<run_id>/_analysis
2. 实验参数全表（Fullmode 相关）

2.1 lambda 动力学参数

参数	当前最优值 (ab_tune_hi_ep10)	默认值 (AgentThresholds)	搜索范围建议
LAMBDA_DELTA_UP	0.15	0.05	[0.05, 0.25]
LAMBDA_DELTA_DOWN	0.10	0.10	[0.05, 0.20]
LAMBDA_CLAMP_MIN	0.05	0.20	[0.0, 0.15]
LAMBDA_CLAMP_MAX	0.80	0.65	[0.70, 0.95]
lambda_smoothing_alpha	1.0	0.85	[0.7, 1.0]
lambda_max_step	0.20	0.15	[0.10, 0.30]
2.2 风险控制参数

参数	当前最优值	默认值	搜索范围建议
OVERFIT_RISK_HI	1.2	0.9	[0.8, 1.5]
OVERFIT_RISK_LO	0.6	0.2	[0.3, 0.8]
OVERFIT_TVC_MIN_HI	0.8	0.6	[0.5, 0.9]
OVERFIT_RISK_EMA_ALPHA	0.6	1.0	[0.4, 1.0]
LAMBDA_DOWN_COOLING_ROUNDS	1	0	[0, 3]
MIOU_LOW_GAIN_THRESH	0.01	0.002	[0.002, 0.02]
MIOU_LOW_GAIN_STREAK	1	3	[1, 3]
2.3 策略结构参数

参数	当前最优值	搜索范围建议
uncertainty_only_rounds	2	[1, 3]
warmup_rounds	1	[1, 2]
warmup_lambda	0.2	[0.1, 0.3]
risk_control_start_round	4	[3, 5]
severe_logic	"and"	{"and", "or"}
risk_ci_window	6	[4, 8]
risk_ci_quantile	0.2	[0.1, 0.3]
2.4 后期策略与安全阀参数

参数	当前最优值	搜索范围建议
late_stage_ramp.start_round	(未启用)	[7, 10]
late_stage_ramp.end_round	—	[11, 14]
late_stage_ramp.start_lambda	—	[0.30, 0.50]
late_stage_ramp.end_lambda	—	[0.55, 0.80]
guardrail.u_median_min	(未启用)	[0.30, 0.50]
guardrail.fallback_quota_u_frac	—	[0.50, 0.80]
2.5 训练参数

参数	当前值	搜索范围建议
epochs_per_round	10	[8, 15]
rollback.std_factor	1.5	[1.0, 2.5]
rollback.tau_min	0.005	[0.003, 0.01]
3. 数据源分层体系

3.1 五大数据源

数据源	位置	格式	独有信息
*_trace.jsonl	runs/<run_id>/	JSONL (逐事件)	lambda 决策过程、agent 工具调用链、l3_selection 详情、epoch级梯度统计(grad.train_val_cos)
logs_md	results/logs_md/	Markdown	每 epoch 的 Loss/mIoU/F1、续跑断点标记、实验汇总段
pools	results/pools/<run_id>/<exp>/	CSV + JSON	最终 labeled/unlabeled 样本 ID 列表、pools_manifest（数据集指纹）
experiment_results.json	runs/<run_id>/	JSON	跨实验汇总：所有实验的 ALC/final_mIoU/budget_history/fallback_history
*_state.json (checkpoint)	checkpoints/<run_id>/	JSON	labeled_indices 完整列表、rng_states（可复现）、performance_history
3.2 Analyzer 数据源分层

Layer 1 (主数据源): *_trace.jsonl
  |- epoch_end: 每 epoch 的 loss/mIoU/F1/梯度统计
  |- round_summary: 每轮汇总 (mIoU/lambda/rollback/overfit_risk/TVC)
  |- lambda_policy_apply: lambda 决策过程
  |- l3_selection: 选样详情 (U/K/score 分布)
  |- selection: 选中样本 ID 列表
  '- controller_step: agent 控制动作

Layer 2 (补充数据源): experiment_results.json
  |- 跨实验快速对比 (ALC/final_mIoU/budget_history)
  '- fallback_history (agent 降级记录)

Layer 3 (快速判断): logs_md
  |- 实验是否完成 ("## 实验汇总" 存在性)
  '- 最终 mIoU/ALC (正则提取，用于快速筛选)

Layer 4 (Resume 基础设施): pools + checkpoint
  |- pools: branch_from_round 的输入 (labeled_pool.csv / unlabeled_pool.csv)
  |- checkpoint: resume 状态 + labeled_indices
  '- 不直接用于诊断分析，但 PoolResumeManager 需要

Layer 5 (离线数据分布分析): pools + .h5 mask
  |- mask 正样本比例 / has_positive 分布
  |- 选样偏置检测 (选中 vs 未选中的 mask 分布差异)
  '- 需要独立离线脚本 analyze_selection_mask_distribution.py
3.3 trace.jsonl 与 logs_md 的关系

trace.jsonl 的 epoch_end 事件已经覆盖了 logs_md 的 epoch 级数据，而且还多了梯度统计（grad.train_val_cos、grad.total_norm 等），这些是 logs_md 里没有的。logs_md 的价值在于：

快速判断实验是否完成（检查 ## 实验汇总 存在性）
快速提取最终 mIoU/ALC（正则匹配，避免解析整个 trace）
人类可读的实验叙事
logs_md 不作为主数据源。

4. 自动化优化闭环框架设计

4.1 整体架构

+---------------------------------------------------------------------+
|                    Tuning Orchestrator (主控)                         |
|                                                                      |
|  +-------------+   +------------------+   +------------------------+ |
|  | Analyzer    |-->| LLM Advisor      |-->| Proposer               | |
|  | (规则诊断)  |   | (Claude Opus)    |   | (规则+LLM->实验配置)   | |
|  |             |   |                  |   |                        | |
|  | trace解析   |   | 深度分析         |   | 参数白名单验证          | |
|  | 指标计算    |   | 多信号交叉       |   | 范围裁剪               | |
|  | 规则诊断    |   | 历史趋势学习     |   | 配置生成               | |
|  |             |   | [失败->跳过]     |   | [LLM无建议->纯规则]    | |
|  +-------------+   +------------------+   +----------+-------------+ |
|                                                       |              |
|                    +----------------------------------+              |
|                    v                                                 |
|  +---------------------------------------------------------------+  |
|  | Executor (实验执行)                                            |  |
|  |                                                               |  |
|  |  ResourceMonitor --> 并发数决策 (静态上限 + 动态内存检查)       |  |
|  |  PoolResumeManager --> 新run_id + 复制pools + resume          |  |
|  |  run_parallel_strict --> ProcessPoolExecutor 执行              |  |
|  +-----------------------------+---------------------------------+  |
|                                |                                     |
|                                v                                     |
|  +---------------------------------------------------------------+  |
|  | Collector (结果收集)                                           |  |
|  |                                                               |  |
|  |  trace.jsonl 解析 -> mIoU/ALC/lambda轨迹 -> 更新 best         |  |
|  |  iteration_history 追加 -> 供下轮 LLM Advisor 使用             |  |
|  +-----------------------------+---------------------------------+  |
|                                |                                     |
|                                v                                     |
|  +---------------------------------------------------------------+  |
|  | ConvergenceDetector                                            |  |
|  |  mIoU >= 0.74 -> STOP                                         |  |
|  |  plateau 3轮 -> STOP                                          |  |
|  |  max_iterations -> STOP                                       |  |
|  |  otherwise -> CONTINUE -> 回到 Analyzer                       |  |
|  +---------------------------------------------------------------+  |
|                                                                      |
|  +---------------------------------------------------------------+  |
|  | GitBranchManager (每轮迭代: 新分支 + 提交分析报告)              |  |
|  +---------------------------------------------------------------+  |
+---------------------------------------------------------------------+
4.2 模块 A：Analyzer（多数据源分析引擎）

输入：run_id + experiment_name

数据结构 ExperimentData（dataclass）：

@dataclass
class ExperimentData:
    trace: List[Dict]                    # 原始 trace 事件列表
    epoch_data: pd.DataFrame             # epoch_end 提取的 DataFrame
    round_data: pd.DataFrame             # round_summary 提取的 DataFrame
    lambda_history: List[Dict]           # lambda_policy_apply 事件
    selection_history: List[Dict]        # l3_selection 事件
    exp_result: Dict                     # experiment_results.json 中该实验的条目
    log_summary: Optional[Dict]          # logs_md 解析的摘要
compute_diagnostics() 返回 ~30 个指标：

类别	指标	来源
性能	final_miou, peak_miou, peak_round, alc, miou_trend(线性斜率), warmup_exit_miou(round 3), mid_stage_miou(round 8), late_stage_gain(final - round 8)	round_summary
稳定性	rollback_count, rollback_rate, max_consecutive_rollback, miou_cv	round_summary
Lambda	lambda_trajectory(list), lambda_mean, lambda_final, lambda_max_reached, lambda_volatility(mean abs diff)	round_summary + lambda_policy_apply
风险	severe_overfit_count(overfit_risk > 1.2), tvc_mean_avg, tvc_last_min	round_summary
Epoch级	epoch_miou_volatility(每轮内mIoU std均值), avg_loss_convergence_ratio(末epoch loss/首epoch loss), avg_best_val_epoch, tvc_sign_flip_rate(轮内TVC从正变负的比例)	epoch_end
选样质量	u_median_trajectory, u_median_trend, k_median_trajectory, k_median_trend	l3_selection
汇总	fallback_count, budget_history	experiment_results.json
完成状态	experiment_completed, test_miou, test_f1	logs_md
diagnose() 规则：

条件	诊断类型	严重度	建议
rollback_rate > 0.15	instability	high	降低 DELTA_UP, 增加 COOLING_ROUNDS
rollback_count == 0 AND lambda_max < 0.4	over_conservative	medium	增加 DELTA_UP, 降低 RISK_LO
late_stage_gain < 0.01	late_stage_plateau	high	启用 late_stage_ramp, 提高 CLAMP_MAX
lambda_volatility > 0.08	lambda_oscillation	medium	降低 smoothing_alpha, 增加 COOLING_ROUNDS
avg_best_val_epoch < 4	epochs_too_many	medium	减少 epochs_per_round
avg_best_val_epoch > 8.5	epochs_insufficient	medium	增加 epochs_per_round
tvc_sign_flip_rate > 0.3	intra_round_overfit	high	减少 epochs 或降低 LR
u_median_trend < -0.03	exploration_degradation	medium	启用 guardrail 或降低 CLAMP_MAX
fallback_count > 3	agent_unreliable	low	检查 LLM 配置或考虑纯 policy 模式
辅助方法：

load_all_experiments(run_id) → dict[str, float]：从 experiment_results.json 快速加载所有实验的 final_mIoU
parse_log_summary(run_id, exp_name) → Optional[dict]：从 logs_md 提取 completed/final_miou/alc
4.3 模块 B：LLM Advisor（Claude Opus 辅助分析）

角色定位：诊断顾问，不是决策者。

LLM 给出方向性建议 + 数值范围，Proposer 负责映射到合法配置
LLM 调用失败时，系统 fallback 到纯规则模式（不阻塞闭环）
LLM 客户端 TuningLLMClient：

OpenAI-compatible HTTP 客户端（复用 SiliconFlowClient 的 requests.post → /chat/completions 模式）
独立配置文件 tuning_llm_config.json（不影响实验内 agent 的 llm_config.json）
{
  "base_url": "https://code.ppchat.vip/v1",
  "api_key": "${TUNING_LLM_API_KEY}",
  "model": "claude-opus-4-6",
  "temperature": 0.3,
  "timeout": 120,
  "max_retries": 3,
  "retry_delay": 5.0,
  "retry_backoff": 2.0,
  "thinking_budget": 16000,
  "max_tokens": 8192,
  "fallback_on_failure": true,
  "log_requests": true,
  "log_dir": "results/tuning_llm_logs"
}
System prompt 包含：

完整参数因果链知识（PARAMETER_CAUSAL_KNOWLEDGE，即 1.2 节的表格内容）
JSON 输出格式要求
LLM 输出格式：

{
  "analysis": "对当前实验结果的深度分析（2-3句话）",
  "primary_bottleneck": "当前阻碍 mIoU 提升的主要瓶颈（一句话）",
  "suggestions": [
    {
      "direction": "方向名称（英文）",
      "description": "调整描述（中文）",
      "parameter_changes": {"参数名": "建议值"},
      "expected_effect": "预期效果（一句话）",
      "risk": "low/medium/high",
      "priority": 1
    }
  ],
  "branch_recommendation": {
    "should_branch": true,
    "branch_round": 7,
    "reason": "原因"
  },
  "warnings": ["需要注意的风险点"]
}
User prompt 包含：

当前实验诊断指标（diagnostics dict）
规则引擎诊断结果（issues list）
历史调优记录（最近 5 轮的 iteration_history）
目标差距
安全机制：

机制	实现
参数白名单	VALID_THRESHOLD_PARAMS（LAMBDA_DELTA_UP 等 13 个）、VALID_POLICY_PARAMS（uncertainty_only_rounds 等 10 个）、VALID_STRUCTURE_PARAMS（late_stage_ramp 等 3 个）
数值范围裁剪	PARAM_RANGES dict，如 LAMBDA_DELTA_UP: (0.02, 0.30)
失败降级	LLM 返回 None → Proposer fallback 纯规则
非法参数名	忽略不在白名单中的参数
JSON 解析容错	尝试直接解析 → markdown 代码块提取 → 正则匹配 {...}
4.4 模块 C：Proposer（实验方案生成器）

搜索策略：条件分支搜索（非 Bayesian/Grid），基于诊断结果的定向调整。

iteration 0:
  -> initial_exploration(): 基于当前最优的多方向扰动, 3-4个方向

iteration 1+:
  if LLM advice available:
    -> proposals_from_llm_advice(): 验证+裁剪LLM建议
  else:
    -> targeted_adjustment(): 基于诊断issues的定向调整
  if no issues:
    -> fine_grid_around_best(): 在最优配置周围 +/-5% 扰动
BASE_CONFIG 模板（从当前最优 ab_tune_hi_ep10 出发）：

BASE_CONFIG = {
    "agent_threshold_overrides": {
        "OVERFIT_RISK_HI": 1.2,
        "OVERFIT_RISK_LO": 0.6,
        "OVERFIT_TVC_MIN_HI": 0.8,
        "OVERFIT_RISK_LAMBDA_UP_MAX": 0.9,
        "MIOU_LOW_GAIN_THRESH": 0.01,
        "MIOU_LOW_GAIN_STREAK": 1,
        "LAMBDA_UP_K_U_GAP_MIN": 0.0,
        "LAMBDA_CLAMP_MIN": 0.05,
        "LAMBDA_CLAMP_MAX": 0.80,
        "LAMBDA_DELTA_UP": 0.15,
        "LAMBDA_DELTA_DOWN": 0.10,
        "OVERFIT_RISK_EMA_ALPHA": 0.6,
        "LAMBDA_DOWN_COOLING_ROUNDS": 1,
    },
    "lambda_policy": {
        "mode": "warmup_risk_closed_loop",
        "r1_lambda": 0.0,
        "uncertainty_only_rounds": 2,
        "warmup_start_round": 3,
        "warmup_rounds": 1,
        "warmup_lambda": 0.2,
        "risk_control_start_round": 4,
        "severe_logic": "and",
        "severe_tvc_key": "grad_train_val_cos_last",
        "risk_trigger": "ci",
        "risk_ci_window": 6,
        "risk_ci_quantile": 0.2,
        "risk_ci_min_samples": 3,
        "lambda_smoothing": "ema",
        "lambda_smoothing_alpha": 1.0,
        "lambda_max_step": 0.20,
    },
}
4.5 模块 D：Resource Monitor + 并发管理 + 实验预算

实测数据（基于 baseline_20260309_211601_seed42 的 full_model_A_lambda_policy trace）：

指标	值
单个 fullmode 实验总时长	~3.5h（16 轮）
平均每轮时长	~13 min
前 6 轮总时长	~50 min
后 10 轮总时长	~160 min
从 round 7 branch 的实验	~1.5-2h
机器配置	32GB RAM / 10 核 / MPS
单实验内存占用	~3-4GB（MPS 共享系统内存）
并发上限	约 2 个实验
ConcurrencyManager：两级并发控制

静态上限: min(3, floor((total_mem_gb - reserve_gb) / per_exp_gb))
  -> 32GB 机器: min(3, floor((32-6)/3.5)) = min(3, 7) = 3
  -> MPS 共享内存惩罚: 硬上限 3

动态检查: 每次启动新实验前
  -> vm_stat 获取 available_gb (free + inactive pages)
  -> available_gb < reserve_gb * 1.2 -> 暂停启动
  -> dynamic_max = floor((available_gb - reserve_gb) / per_exp_gb)
  -> actual_slots = min(static_max, dynamic_max) - running_count
ExperimentBudget：分层漏斗

Phase A (快速筛选): 单 seed, 3-4 个方向
  |- 1-2 个 branch 实验 (~1.5h each)
  |- 1-2 个 full 实验 (~3.5h each)
  |- 并发 2 -> 总耗时 ~4h

Phase B (验证): 仅最优 1-2 个方向
  |- 补跑 2 个 seed (42, 44)
  |- 并发 2 -> 总耗时 ~3.5h

单轮迭代总耗时: ~7.5h
执行流程：

Orchestrator 启动 Phase A:
|
|-- ConcurrencyManager.get_available_slots() -> 2
|
|-- 启动 exp_1 (branch, ~1.5h)
|-- 启动 exp_2 (branch, ~1.5h)
|   |-- 并发运行中...
|   |-- exp_1 完成 -> slot 释放
|   |-- ConcurrencyManager.get_available_slots() -> 1
|   |-- 启动 exp_3 (full, ~3.5h)
|   |-- exp_2 完成 -> slot 释放
|   '-- 等待 exp_3 完成
|
|-- Phase A 全部完成 -> 分析结果 -> 选最优
|
|-- Phase B: 对最优方向补跑 seed 42, 44
|   |-- 启动 exp_best_seed42
|   |-- 启动 exp_best_seed44
|   '-- 等待完成
|
'-- 收集结果 -> 更新 best -> 检查收敛
4.6 模块 E：Pool Resume Manager

关键设计：每轮迭代创建新的 run_id，复制源实验的 pools 和部分 round 数据到新 run_id 下，新实验在新 run_id 下 resume。避免每次从头跑 16 轮。

新 run_id 命名：autotune_iter{N:03d}_{timestamp}_{seed}

流程：

1. 确定 source: source_run_id + source_exp + branch_round
2. 从 source checkpoint 读取 performance_history
   -> 找到 branch_round 对应的 labeled_size
3. 将 source pools 的 labeled_pool.csv 截断到该 labeled_size
   -> 多余的 labeled 样本回退到 unlabeled_pool.csv
4. 为每个 target_exp 在新 run_id 下:
   -> 写入 pools (labeled_pool.csv + unlabeled_pool.csv)
   -> 写入 checkpoint (_state.json, 含 performance_history 截断到 branch_round)
   -> 复制 pools_manifest.json
5. 新实验以 resume 模式启动 (--start resume)
底层复用已有的 branch_experiment_from_round.py：

_resolve_source_perf()：从 checkpoint 或 result_json 读取 performance_history
_split_pools_at_labeled_size()：截断 labeled pool
_write_pools()：原子写入新 pools
_build_checkpoint_payload()：构建 resume checkpoint
Branch 策略决策：

实验类型	策略	理由
仅改后期参数（ramp, guardrail, epochs）	从 round 7 branch	前 6 轮结果可复用，节省 ~50 min
改前期参数（uncertainty_only_rounds, warmup_lambda）	全量跑	前期策略变了，必须从头
改风险控制参数（RISK_HI, DELTA_UP）	从 round 4 branch 或全量	取决于参数影响的起始 round
4.7 模块 F：Git Branch Manager

每轮迭代：

1. git checkout -b tuning/iter-{N:03d}-{YYYYMMDD_HHMM}
2. 写入分析报告:
   AAL-SD-Doc/tuning_reports/iter_{N:03d}_analysis.json
   内容: {iteration, timestamp, analysis, proposed_experiments, configs}
3. 如有代码改动 (如新增 auto_tune_configs.json 条目) 一并 git add
4. git commit -m "tuning iter {N}: {M} experiments proposed, best_miou={X}"
4.8 模块 G：Convergence Detector

update(iteration, best_miou) -> {action, reason}

条件1: best_miou >= 0.74
  -> STOP, reason="target_reached"

条件2: 连续 patience 轮 (default 3) 无改善 (< 0.002)
  -> STOP, reason="convergence_plateau"

条件3: iteration >= max_iterations (default 10)
  -> STOP, reason="max_iterations"

条件4: 最近 3 轮平均改善 < 0.001
  -> WARN, reason="diminishing_returns"

其他: CONTINUE
4.9 分析决策策略：滑动窗口 + 全局 best 追踪

不做全量分析，而是分层：

全局 best: 始终参与分析 (作为 baseline, 详细 trace 分析)

最近 window_size 轮 (default 2-3):
  -> 详细 trace 分析 (epoch 级 + round 级)
  -> 完整诊断
  -> LLM Advisor 咨询

更早的实验:
  -> 只保留摘要 (final_miou, config, direction)
  -> 不再解析 trace
LLM 上下文控制在 ~4000 tokens：

context = {
    "target": 0.74,
    "global_best": {miou, config},           # 全局最优
    "recent_experiments": [                    # 最近轮次, 完整诊断
        {name, miou, direction, diagnostics}
    ],
    "history_summary": [                       # 更早的, 只有方向和结果
        "aggressive_ramp: mIoU=0.7180",
        "high_clamp: mIoU=0.7250",
    ],
    "tried_directions": ["ramp", "high_clamp", ...]  # 防止重复建议
}
具体流程示例：

迭代 0: 分析初始最优 -> 详细 trace -> 3 个新方向
迭代 1: 分析 iter0 的 3 个 (详细) + 初始最优 (baseline)
         -> LLM 看到 4 个实验的完整诊断
迭代 2: 分析 iter1 的 3 个 (详细) + iter0 最优 (详细)
         + 初始最优 (摘要)
         -> LLM 看到 4 个详细 + 1 个摘要
迭代 3: 分析 iter2 的 3 个 (详细) + iter1 最优 (详细)
         + iter0 全部 (摘要) + 初始 (摘要)
         -> LLM 看到 4 个详细 + 4-5 个摘要
5. 推荐调优路径（优先级排序）

Phase 1：高优先级（预期收益最大）

优先级	方向	具体调整	预期机制	预期收益
P0	late_stage_ramp + 当前最优	在 ab_tune_hi_ep10 基础上加 late_stage_ramp(start=8, end=13, lambda: 0.35->0.70)	后期强制提升 K 权重，增加代表性样本覆盖	+0.01~0.02
P1	放宽 CLAMP_MAX	LAMBDA_CLAMP_MAX: 0.80->0.90~0.95	允许后期更多代表性驱动	+0.005~0.015
P2	增加 epochs_per_round	10->12~15	更充分训练，尤其后期数据量增大时	+0.005~0.01
P3	ramp + guardrail 组合	ramp 保证后期 K 权重 + guardrail 保护 U 底线	两者互补：ramp 推进，guardrail 防退化	+0.01~0.02
Phase 2：中优先级（精细调参）

优先级	方向	具体调整	预期机制
P4	更早引入 K	uncertainty_only_rounds: 2->1, warmup_start: 3->2, risk_control_start: 4->3	缩短纯 U 阶段
P5	放宽风险控制	OVERFIT_RISK_HI: 1.2->1.5, OVERFIT_RISK_LO: 0.6->0.8	减少 severe 触发
P6	DELTA_UP 激进化	LAMBDA_DELTA_UP: 0.15->0.20~0.25	加速 lambda 上升
P7	rollback 阈值放宽	std_factor: 1.5->2.0, tau_min: 0.005->0.008	减少误触发 rollback
Phase 3：低优先级（结构性探索）

优先级	方向	具体调整
P8	warmup_lambda 提高	0.2->0.3
P9	CI 窗口/分位数调整	risk_ci_window: 6->4, risk_ci_quantile: 0.2->0.15
P10	severe_logic 切换	"and"->"or"
6. 第一轮具体实验矩阵

3-4 个方向（单 seed=43 快速筛选）：

实验名	方向	关键变更	策略
auto_tune_iter00_ramp_on_best_00	P0	继承 ab_tune_hi_ep10 + late_stage_ramp(start=8, end=13, start_lambda=0.35, end_lambda=0.70)	从 round 7 branch
auto_tune_iter00_high_clamp_01	P1	继承 ab_tune_hi_ep10 + LAMBDA_CLAMP_MAX=0.92	全量跑
auto_tune_iter00_ramp_guard_02	P3	继承 ab_tune_hi_ep10 + ramp(0.35->0.65) + guardrail(u_median_min=0.40)	从 round 7 branch
auto_tune_iter00_ep12_03	P2	继承 ab_tune_hi_ep10 + epochs_per_round=12	全量跑
7. 完整工作流时序

Iteration 0 (初始)
|-- 分析已有最优实验 (ab_tune_hi_ep10, seed43, mIoU=0.7221)
|-- 诊断: 后期增益不足 + lambda 可能受限于 CLAMP_MAX
|-- 生成 3-4 个实验方案 (P0~P3)
|-- 创建新 run_id: autotune_iter000_{timestamp}_seed43
|-- 复制源实验 pools 到新 run_id (branch 实验从 round 7, full 实验复制 _base)
|-- git checkout -b tuning/iter-000-YYYYMMDD_HHMM
|-- 提交分析报告 + 实验配置
|-- 资源检查 -> 并发 2
|-- 执行实验 (Phase A, ~4h)
|-- 收集结果 -> 找到最优
'-- 检查: mIoU >= 0.74? -> 否 -> 继续

Iteration 1
|-- 分析 iter0 全部实验的 trace (详细, 滑动窗口内)
|-- 对 iter0 最优用 seed=42,44 验证 (Phase B)
|-- 创建新 run_id: autotune_iter001_{timestamp}
|-- 复制 iter0 最优的 pools 到新 run_id
|-- 基于诊断 + LLM Advisor 生成定向调整方案
|-- git checkout -b tuning/iter-001-...
|-- 提交报告 + 执行
'-- 检查: mIoU >= 0.74?

Iteration 2+
|-- 滑动窗口分析 (最近 2-3 轮详细 + 更早摘要)
|-- LLM Advisor 参与 (带历史 context + 已尝试方向)
|-- 精细网格或结构性探索
'-- 收敛检测 -> 决定继续/终止
8. 实现文件结构

AAL_SD/
|-- src/
|   |-- tuning/                            # 新增: 调优闭环框架
|   |   |-- __init__.py
|   |   |-- analyzer.py                    # 模块A: 多数据源分析 + 规则诊断
|   |   |-- llm_advisor.py                 # LLM客户端 (TuningLLMClient)
|   |   |-- advisor.py                     # 模块B: LLM Advisor (prompt构建+响应解析)
|   |   |-- proposer.py                    # 模块C: 实验方案生成 (规则+LLM->配置)
|   |   |-- resource_monitor.py            # 模块D: 内存监控 + 并发 + 预算
|   |   |-- pool_resume_manager.py         # 模块E: Pool复制 + branch resume
|   |   |-- git_branch_manager.py          # 模块F: Git分支 + 报告提交
|   |   |-- convergence.py                 # 模块G: 收敛检测
|   |   '-- orchestrator.py                # 主控闭环调度
|   |-- tuning_llm_config.json             # 调优LLM配置 (独立于实验agent)
|   |-- experiments/
|   |   |-- ablation_config.py             # 已有 (支持sidecar加载)
|   |   '-- auto_tune_configs.json         # 新增: 自动生成的实验配置
|   '-- analysis/
|       '-- analyze_selection_mask_distribution.py  # 新增: 离线mask分布分析
|-- AAL-SD-Doc/
|   |-- tuning_reports/                    # 新增: 每轮迭代分析报告
|   |   |-- iter_000_analysis.json
|   |   '-- ...
|   '-- auto_tuning_framework_design.md    # 本文档
'-- results/
    '-- tuning_llm_logs/                   # 新增: LLM调用日志
9. 关键设计决策汇总

决策点	选择	理由
搜索策略	条件分支搜索（非 Bayesian）	参数空间有明确因果结构，诊断驱动比黑盒搜索更高效
初筛 seed 数	单 seed (43) 快速筛选	seed 间方差 ~0.02-0.04，单 seed 足以判断方向性
验证 seed 数	3 seeds (42,43,44) 验证最优	确认稳定性后再投入多 seed 资源
分支策略	每轮新 run_id + 复制 pools + resume	避免从头跑，节省 ~40% 计算时间；新 run_id 隔离实验数据
配置注入方式	JSON sidecar 文件 (auto_tune_configs.json)	避免频繁修改 ablation_config.py，降低合并冲突风险
并发控制	静态上限 + 动态内存检查	MPS 共享系统内存，需要保守估计
每轮实验数	动态 3-4 个（ResourceMonitor 决定）	分 branch/full，Phase A 筛选 + Phase B 验证
LLM 角色	诊断顾问（不是决策者）	白名单 + 范围裁剪保证安全，失败时 fallback 规则
LLM 模型	Claude Opus 4.6 via ppchat	推理能力强，thinking 模式提升复杂推理质量
LLM 调用频率	每轮迭代 1 次	每轮实验耗时数小时，LLM 10-30s 延迟可忽略
分析范围	滑动窗口（最近 2-3 轮详细）	避免全量分析的 token 爆炸和注意力分散
主数据源	trace.jsonl	最丰富：epoch 级 + round 级 + lambda 决策 + 梯度统计
数据分布分析	独立离线脚本 (pools + .h5 mask)	trace 不含 mask 信息，需要额外数据源
单轮耗时	Phase A ~4h + Phase B ~3.5h ≈ 7.5h	32GB/MPS/并发 2 的实际约束
10. 风险与缓解

风险	概率	影响	缓解措施
0.74 在当前架构下不可达（K 的 KMeans 代表性上限）	中	高	如果 3 轮迭代后改善 < 0.005，考虑改进 K 的计算方式
seed 方差导致假阳性（单次达标但不稳定）	高	中	达标后必须 3-seed 验证
内存不足导致实验 OOM	低	中	ResourceMonitor 动态降低并发
参数组合爆炸	中	低	诊断驱动的条件搜索限制每轮 ≤4 个实验
late_stage_ramp 导致后期不稳定	中	中	配合 guardrail 使用，设置合理的 end_lambda 上限
LLM 幻觉出非法参数	中	低	白名单 + 范围裁剪
选样偏置（持续选低质量样本）	中	高	离线 mask 分布分析脚本检测；guardrail 保护 U 底线
11. 启动命令

# 基于已有最优实验启动自动调优
python src/tuning/orchestrator.py \
    --initial-run-id baseline_20260309_211601_seed43 \
    --initial-exp full_model_A_lambda_policy_ab_tune_hi_ep10 \
    --target-miou 0.74 \
    --max-iterations 10 \
    --seeds 43 \
    --max-concurrent 2

# 指定 LLM 配置
python src/tuning/orchestrator.py \
    --initial-run-id baseline_20260309_211601_seed43 \
    --initial-exp full_model_A_lambda_policy_ab_tune_hi_ep10 \
    --target-miou 0.74 \
    --llm-config src/tuning_llm_config.json

# 禁用 LLM Advisor（纯规则模式）
python src/tuning/orchestrator.py \
    --initial-run-id baseline_20260309_211601_seed43 \
    --initial-exp full_model_A_lambda_policy_ab_tune_hi_ep10 \
    --target-miou 0.74 \
    --no-llm

# 离线 mask 分布分析
python src/analysis/analyze_selection_mask_distribution.py \
    results/runs/<run_id>/<exp>_trace.jsonl \
    --output results/runs/<run_id>/_analysis
