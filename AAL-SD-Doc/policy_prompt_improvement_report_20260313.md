# AAL-SD Policy & Prompt 改进空间分析（基于 4 个 run，已去除超链接与引用块）

生成时间：2026-03-17  
范围（本文主结论只针对这 4 个 run）：
- baseline_20260311_201728_seed42
- baseline_20260311_201728_seed43
- baseline_20260309_211601_seed42
- baseline_20260309_211601_seed43

写作约束说明：
- 本文不使用超链接、不使用引用块（例如以 `>` 开头的引用文本）。
- 原文中“证据：xxx.md / xxx.json”的部分，改为：直接写出“数据文件路径 + 关键字段 + 关键数值/结论”，把引用内容转写进正文，便于离线审阅与论文式复核。

参考的调参/诊断框架（路径纯文本）：
- /Users/anykong/AD-KUCS/AAL_SD/AAL-SD-Doc/auto_tuning_framework_design.md  
  用途：指标体系、diagnose 规则、policy/prompt 调优逻辑映射。

---

## 全面科研研究报告（基于当前源码与 results 产物）

本节是本 session 最核心的交付：面向科研写作（开题/中期/论文方法与实验章节）的“完整研究报告正文”。后续章节中的“Policy & Prompt 改进空间分析（基于 4 个 run）”属于其中一个专题分析模块，用于解释当前策略/提示词的失效模式与可改进空间。

### 摘要

本项目研究遥感滑坡语义分割任务在标注预算受限场景下的主动学习（Active Learning, AL）策略。针对分割任务“像素级标签昂贵、类别极度稀疏、训练不稳定与过拟合风险高”的特点，我们构建了一个可审计、可恢复、可自动调参的闭环 AL 系统 AAL‑SD。系统以不确定性 U 与代表性/知识增益 K 的融合选样为核心，通过动态权重 λ 在探索与利用之间权衡；并引入风险闭环 λ policy、U 底线安全阀（selection guardrail）与回撤保护（rollback）以提升跨轮训练的稳定性与跨 seed 的稳健性。项目在 Landslide4Sense 数据集（14 通道输入、2 类分割）上，通过多种子实验、强基线对照与消融矩阵，评估学习效率指标 ALC（预算归一化学习曲线面积）与最终指标 final_mIoU/final‑F1。results 目录的当前实验进展表明：Full policy 闭环在 ALC 与稳定性方面具备优势，但后期 ramp 与 guardrail 的统计一致性与跨 seed 鲁棒性仍需进一步完善。

关键词：主动学习；语义分割；遥感滑坡；闭环控制；不确定性；代表性；LLM‑Agent

### 1. 研究背景与问题定义

#### 1.1 背景动机

遥感滑坡分割面临两个核心瓶颈：
- 标注成本高：像素级 mask 标注耗时且需要专业知识，导致可标注数据量受限。
- 数据稀疏与不稳定：滑坡区域通常面积占比小，训练早期容易出现类坍塌；后期易过拟合且跨轮策略引入可能导致回撤。

主动学习的目标是在固定预算下最大化模型性能与学习效率：同样的预算，尽早达到更高精度，并尽量减少“选错样本导致的回撤与训练失败”。

#### 1.2 问题定义（本项目采用的形式化）

给定：
- 初始标注集 L0（冷启动构建）与未标注池 U0；
- 总标注预算 B（可标注样本数上限）；
- 轮数 T（主动学习轮次）；
每轮 t：
1) 在当前标注集 Lt 上训练分割模型 ft；
2) 用验证集评估得到性能指标（mIoU/F1）与训练态势信号；
3) 根据选样策略从未标注池 Ut 中选出 qt 个样本加入标注集；
最终在预算用尽时得到模型 fT，并输出：
- 学习效率：ALC（预算归一化后的学习曲线面积）
- 最终性能：final_mIoU、final_F1（优先 test 口径）

#### 1.3 评价指标与口径（与源码/结果文件一致）

本项目明确区分两类口径：
- 轨迹口径（用于 learning curve 与 ALC）：每轮 performance_history[*].mIoU（通常为 val best 选模结果）
- 汇总口径（用于最终结论表）：final_miou / final_f1（优先来自 test split 的 final report；失败或不可用时可能回退）

### 2. 数据集、数据格式与数据处理流程

#### 2.1 数据集结构（工程约束）

数据目录结构必须满足：
- TrainData/img 与 TrainData/mask
- ValidData/img 与 ValidData/mask
- TestData/img（若存在 TestData/mask 且完整，则用于最终测试口径评估）

单样本文件格式（H5）：
- 图像键 img，shape (128, 128, 14)
- mask 键 mask，shape (128, 128)，类别 id 为 0/1

实现位置（路径纯文本）：
- 数据集读取与命名映射：/Users/anykong/AD-KUCS/AAL_SD/src/core/dataset.py
- 数据根目录与预算等默认设置：/Users/anykong/AD-KUCS/AAL_SD/src/config.py

#### 2.2 冷启动池构建（stratified initialization）

冷启动用分层抽样避免正类极稀疏导致训练不收敛：
- 定义 has_positive：mask 是否包含正类像素（mask>0 的存在性）
- 按 has_positive 将训练集分为正/负两层
- 分别抽取一定比例样本组成初始标注集 L0（默认约 5%），剩余作为未标注池 U0

实现位置（路径纯文本）：
- /Users/anykong/AD-KUCS/AAL_SD/src/core/data_preprocessing.py

#### 2.3 预算与每轮 query_size 的确定（可复现）

典型默认配置（与多数 results run 对齐）：
- 估计训练集总量约 3799
- 总预算 B ≈ 1519（约 40%）
- 总轮数约 16
- 每轮 query_size 约 88（由 (B - 初始标注估计)/(轮数-1) 推导得到）

实现位置（路径纯文本）：
- /Users/anykong/AD-KUCS/AAL_SD/src/config.py

### 3. 模型、训练协议与评估实现

#### 3.1 模型选择

本项目默认模型为 DeepLabV3+：
- encoder：ResNet50
- encoder_weights：imagenet
- in_channels：14
- classes：2

实现位置（路径纯文本）：
- /Users/anykong/AD-KUCS/AAL_SD/src/core/model.py

#### 3.2 每轮训练协议

默认训练设置：
- 优化器：Adam，学习率 1e-4
- batch_size：4
- 每轮 epoch：10（固定轮长）
- 支持 AMP 与 torch.compile（默认关闭）
- 设备：优先 CUDA/MPS，否则 CPU

实现位置（路径纯文本）：
- /Users/anykong/AD-KUCS/AAL_SD/src/core/trainer.py
- /Users/anykong/AD-KUCS/AAL_SD/src/config.py

#### 3.3 指标计算（mIoU/F1/ALC）

分割评估：
- 基于混淆矩阵统计每类的 intersection/union，并计算 macro mIoU 与 macro F1

学习效率指标 ALC：
- 将预算归一化到 [0,1]，对 (budget_ratio, mIoU) 曲线做 AUC
- 用于衡量“同预算下更早达到更高性能”的能力

实现位置（路径纯文本）：
- mIoU/F1：/Users/anykong/AD-KUCS/AAL_SD/src/core/trainer.py
- ALC：/Users/anykong/AD-KUCS/AAL_SD/src/utils/evaluation.py

### 4. 主动学习算法设计：AD‑KUCS（U、K 与融合权重 λ）

#### 4.1 选样基本思想

分割任务中，单纯依赖不确定性容易过度关注噪声与边界像素；单纯依赖代表性又容易忽视当前模型的决策盲区。因此采用融合：

Score(x) = (1 - λt) · U(x) + λt · K(x)

其中：
- U(x)：模型对样本 x 的不确定性（探索）
- K(x)：样本在未标注池中的代表性/知识增益（利用）
- λt：第 t 轮权衡系数（动态控制）

#### 4.2 不确定性 U 的理论依据与实现口径

默认采用像素熵聚合：
- 对每个像素的类别概率分布 p 计算熵：
  H(p) = - Σc p_c log2(p_c + 1e-10)
- 样本级不确定性 U(x) 取像素熵的均值（也支持“高熵像素聚合”以避免背景稀释）

可选 BALD（互信息）：
- 通过 MC Dropout 得到多个随机前向的预测分布，估计 epistemic 不确定性

实现位置（路径纯文本）：
- /Users/anykong/AD-KUCS/AAL_SD/src/core/sampler.py

#### 4.3 代表性/知识增益 K 的理论依据与实现口径

当前主实现口径：未标注池特征空间的 KMeans representativeness。
- 从 backbone 提取每个样本的特征向量 f_x（GAP 后得到定长向量）
- 在未标注池特征上做 KMeans，得到簇中心 c(x)
- 距离 d(x)=||f_x - c(x)||，以池内最大距离归一化
- 定义代表性得分 K(x)=1 - d(x)/max(d_max, 1e-12)

直观解释：
- 簇中心附近样本更“典型/代表”，加入标注后能让模型覆盖更广泛的常见模式；
- 与多样性策略结合时，可减少“重复标注相似样本”的浪费。

实现位置（路径纯文本）：
- /Users/anykong/AD-KUCS/AAL_SD/src/core/sampler.py

#### 4.4 融合权重 λ 的设计原则

λ 不是固定超参，而是“控制量”：
- 早期：模型尚未稳定，特征空间不可靠，宜小 λ（偏探索，依赖 U）
- 中期：探索与代表性并重，λ 在可控风险下逐步上调
- 后期：容易出现 plateau，若风险可控可 ramp 提高 λ（偏利用，强调覆盖与泛化）

### 5. 控制框架设计：感知信号、风险闭环与安全阀

本项目的核心区别在于：将 λ 视为闭环控制量，而不是“离线调参常数”。

#### 5.1 感知信号（Perception Design）

每轮形成 training_state，核心信号包括：
- 性能：mIoU、F1、miou_delta（相邻轮差值）
- 过拟合风险：overfit_risk（由梯度对齐与波动统计汇总）
- 梯度探针（TVC）：grad_train_val_cos_*（训练/验证梯度余弦相似度的统计量）
- 预算：当前标注量、剩余预算
- 选样统计：selected/top 的 U/K 分布统计（用于 guardrail 与一致性审计）

实现位置（路径纯文本）：
- training_state 构建与回撤判定：/Users/anykong/AD-KUCS/AAL_SD/src/main.py
- 梯度探针与统计：/Users/anykong/AD-KUCS/AAL_SD/src/core/trainer.py

#### 5.2 λ policy：warmup_risk_closed_loop（主方法）

策略结构（与实验消融配置一致）：
- uncertainty_only_rounds：前若干轮 λ 固定为 r1_lambda（通常 0）
- warmup：在 warmup_start_round 起持续 warmup_rounds，λ 设为 warmup_lambda
- risk_control：从 risk_control_start_round 起，基于风险信号触发 λ 上调/下调/保持
  - severe_logic：严重判定逻辑（and/or）
  - risk_trigger：阈值触发方式（例如 CI 分位数）
  - lambda_smoothing：λ 平滑（EMA）
  - lambda_max_step：单轮最大步长（抑制振荡）

典型参数（Full 默认实验）：
- uncertainty_only_rounds=2
- warmup_start_round=3，warmup_rounds=1，warmup_lambda=0.2
- risk_control_start_round=4
- risk_trigger=ci，risk_ci_window=6，risk_ci_quantile=0.2，risk_ci_min_samples=3
- lambda_smoothing=ema，lambda_smoothing_alpha=0.85，lambda_max_step=0.15
- clamp（工具层阈值覆盖）：LAMBDA_CLAMP_MIN=0.05，LAMBDA_CLAMP_MAX=0.80

实现位置（路径纯文本）：
- λ policy 细节与 guardrail：/Users/anykong/AD-KUCS/AAL_SD/src/agent/toolbox.py
- 消融参数表（权威来源）：/Users/anykong/AD-KUCS/AAL_SD/src/experiments/ablation_config.py

#### 5.3 selection guardrail：U 底线安全阀（防止“过推 K”）

当选中集合的不确定性质量过低（例如 U 中位数过低或低 U 比例过高）时：
- 逐步下调 λ（lambda_step_down），最多 max_steps 次
- 必要时启动“U 优先配额”fallback_quota_u_frac（例如 70% 直接从 U 排名挑选）

典型参数（u_guardrail / ramp_guardrail）：
- u_median_min=0.45
- u_low_thresh=0.40
- u_low_frac_max=0.20
- lambda_step_down=0.10
- max_steps=5
- fallback_quota_u_frac=0.70

#### 5.4 late-stage ramp：后期提升代表性以追求最终泛化

该模块用于解决后期 plateau：
- start_round=9，end_round=12
- start_lambda=0.40，end_lambda=0.70（也有 end_lambda=0.55 的分支用于隔离 ramp 强度影响）

关键注意点（已在 results 中体现为跨 seed 敏感性）：
- ramp 必须被风险信号门控；否则可能在某些 seed 上过推 K，导致性能退化或回撤。

#### 5.5 rollback：回撤保护（防止策略引发灾难性退化）

回撤判定采用自适应阈值：
- tau = max(tau_min, std_factor · std_epoch_mIoU)
- 若 miou_delta < -tau 则触发 rollback_flag

典型参数：
- std_factor=1.5
- tau_min=0.005

### 6. 算法整体流程（论文可直接用的伪代码描述）

输入：训练/验证/测试数据；初始标注比例；预算 B；轮数 T；每轮查询量 q；策略配置（sampler、λ policy、guardrail、rollback）。

1) 冷启动：按 has_positive 分层抽样构建 L0、U0，并持久化池文件。
2) 对 t=1..T：
   - 训练：在 Lt-1 上训练模型 ft（固定 epoch 或按配置）
   - 评估：在 val 上得到 mIoU/F1 与 best_val 轨迹
   - 感知：计算 miou_delta、TVC、overfit_risk、epoch 波动与回撤信号
   - 计算候选：对 Ut-1 中样本计算 U(x)、提取特征并计算 K(x)
   - 控制：由 λ policy 输出 λt（带 clamp、平滑、限步长）
   - 选样：按 Score(x) 排序并应用多样性/约束后处理，得到选中集合 Qt
   - 安全阀：若 guardrail 触发，调整 λt 并重算/重选（最多 max_steps）
   - 池更新：Lt = Lt-1 ∪ Qt，Ut = Ut-1 \ Qt，并持久化
3) 输出：performance_history、budget_history、ALC、final_miou/final_f1（尽可能基于 test split）

### 7. 实验设计：实验矩阵、参数空间与复现协议

#### 7.1 实验矩阵（强基线 + 主方法消融）

（1）强基线（sampler_type）
- random
- entropy
- coreset
- bald
- dial
- wang
- llm_us（LLM‑uncertainty）
- llm_rs（LLM‑random）

实现位置（路径纯文本）：
- sampler 构建入口：/Users/anykong/AD-KUCS/AAL_SD/src/experiments/components.py
- 具体实现：/Users/anykong/AD-KUCS/AAL_SD/src/baselines/*.py

（2）主方法消融（来自 ABLATION_SETTINGS，权威配置源）
建议论文主表至少包含以下实验（其余分支可作为补充材料）：
- full_model_A_lambda_policy（Full：warmup + risk closed-loop，禁止显式 set_lambda）
- full_model_A_lambda_policy_train_holdout（grad_probe 源为 train_holdout，强调学术严谨）
- full_model_A_lambda_policy_u_guardrail（仅加 guardrail）
- full_model_A_lambda_policy_ramp_guardrail（ramp + guardrail）
- full_model_A_lambda_policy_ramp_guardrail_train_probe（ramp + guardrail + train_holdout probe）
- full_model_A_lambda_policy_ramp_guardrail_train_probe_ramp055（仅降低 ramp 强度：end_lambda=0.55）
- full_model_A_lambda_policy_ramp_guardrail_train_probe_guardrail_u30（仅放宽 guardrail：u_median_min=0.30）
- full_model_A_lambda_policy_u_adaptive / full_model_A_lambda_policy_ramp_guardrail_train_probe_u_adaptive（引入 U 变化率自适应）

权威配置位置（路径纯文本）：
- /Users/anykong/AD-KUCS/AAL_SD/src/experiments/ablation_config.py

#### 7.2 参数空间（用于控制框架设计与 auto‑tuning）

建议按 6 组组织参数空间，便于写论文与做自动调参：

组 A：预算与轮设计
- INITIAL_LABELED_SIZE
- TOTAL_BUDGET / BUDGET_RATIO
- N_ROUNDS
- QUERY_SIZE

组 B：训练与稳定性
- LR、BATCH_SIZE、EPOCHS_PER_ROUND
- FIX_EPOCHS_PER_ROUND（固定轮长）
- AMP / torch.compile（开关）

组 C：不确定性 U
- uncertainty_method（entropy/bald）
- uncertainty_aggregation（mean 或高熵像素聚合）
- entropy_threshold（若启用）
- MC Dropout 次数（BALD）

组 D：代表性 K 与多样性
- 特征提取层（默认 layer4）
- KMeans 簇数（当前实现存在与 query_size 绑定的隐含假设，建议在论文中明确）
- diversity_postprocess（none 或 fps_feature）
- candidate_multiplier（多样性候选倍数）
- pred_pos_area_quota（可选约束）

组 E：λ policy（闭环核心）
- uncertainty_only_rounds、warmup_start_round、warmup_rounds、warmup_lambda
- risk_trigger、risk_ci_window、risk_ci_quantile、risk_ci_min_samples
- severe_logic、severe_tvc_key
- lambda_smoothing、lambda_smoothing_alpha、lambda_max_step
- clamp：LAMBDA_CLAMP_MIN/LAMBDA_CLAMP_MAX、delta_up/down、cooling
- late_stage_ramp（start/end rounds 与 start/end lambda）
- selection_guardrail（u_median_min、u_low_thresh、u_low_frac_max、step_down、max_steps、fallback_quota_u_frac）

组 F：rollback（回撤保护）
- std_factor、tau_min

auto‑tuning 的参数空间入口（路径纯文本）：
- /Users/anykong/AD-KUCS/AAL_SD/src/tuning_program.json

#### 7.3 实验复现协议（脚本与关键参数）

运行所有实验（同一 run_id 下可顺序/并行）：
- 脚本：/Users/anykong/AD-KUCS/AAL_SD/src/experiments/run_all_experiments.py
- 关键参数：
  - --experiments：指定实验名列表（默认跑全量）
  - --results_dir：结果输出目录（默认 results）
  - --config：可选，外部 config.py 路径
  - --start：resume 或 fresh
  - --run_id：建议科研显式指定，便于分组与复核
  - --n_rounds：覆盖轮数（可选）
  - --parallel_workers：同 run_id 内并行跑多个实验（注意输出竞态由单实验 result_*.json 规避）

运行多种子并聚合：
- 脚本：/Users/anykong/AD-KUCS/AAL_SD/src/experiments/run_multi_seed.py
- 输出：
  - results/runs/<base_run_id>/multi_seed_manifest.json
  - results/runs/<base_run_id>/multi_seed_summary.json
  - results/runs/<base_run_id>/multi_seed_report.md

### 8. 当前实验进展（结合 results 目录）

本项目 results 目录中已存在两类“可直接引用到论文结果章节”的产物：
- 单 run 的 experiment_results.json（每实验含 ALC、final 指标、轨迹）
- 多种子聚合 multi_seed_summary.json（含 mean/std/CI 与排名）

建议论文写法：
- 主结果表：multi_seed_summary.json 的 mean/std/CI（ALC 与 final_miou 分列）
- 过程解释：用单 run 的 performance_history 与 trace.jsonl 展示 λ 轨迹、guardrail 触发与回撤事件
- 失败分析：将 LLM 链路失败（网络/DNS）与训练/策略失败分开统计

#### 8.1 多种子聚合摘要（Full family：闭环控制相关）

数据文件路径（2 seeds 聚合）：
- /Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728/multi_seed_summary.json

该聚合覆盖 run_ids：
- baseline_20260311_201728_seed42
- baseline_20260311_201728_seed43

汇总表（mean ± std，括号内为 CI95 半径；n=2）：

| 实验 | ALC | final_miou | final_f1 |
|---|---:|---:|---:|
| full_model_A_lambda_policy | 0.605866 ± 0.001534 (0.013786) | 0.699876 ± 0.001121 (0.010068) | 0.789301 ± 0.001063 (0.009554) |
| full_model_A_lambda_policy_u_guardrail | 0.599851 ± 0.001754 (0.015758) | 0.692435 ± 0.015157 (0.136176) | 0.781777 ± 0.015490 (0.139174) |
| full_model_A_lambda_policy_ramp_guardrail | 0.606280 ± 0.006403 (0.057528) | 0.687115 ± 0.032113 (0.288515) | 0.775948 ± 0.033213 (0.298407) |
| full_model_A_lambda_policy_ramp_guardrail_train_probe | 0.597609 ± 0.008646 (0.077680) | 0.679059 ± 0.003865 (0.034729) | 0.768099 ± 0.003844 (0.034538) |

从科研表述角度的直接结论（严格对齐上述数值）：
- full_model_A_lambda_policy 的 final_miou 与 ALC 在这 2 个 seed 上方差最小，体现更强稳定性。
- ramp_guardrail 的 final_miou 均值被 seed43 显著拖低，std 与 CI95 明显增大，说明该策略在当前版本下存在跨 seed 不鲁棒风险。

#### 8.2 多种子聚合摘要（Baselines：强对照）

数据文件路径（4 seeds 聚合）：
- /Users/anykong/AD-KUCS/AAL_SD/results/runs/baselines_only_p3_20260313_212023/multi_seed_summary.json

该聚合覆盖 run_ids：
- baselines_only_p3_20260313_212023_seed42/seed43/seed44/seed45

汇总表（mean ± std，括号内为 CI95 半径；n=4）：

| Baseline | ALC | final_miou | final_f1 |
|---|---:|---:|---:|
| baseline_entropy | 0.600292 ± 0.002002 (0.003185) | 0.706251 ± 0.011669 (0.018566) | 0.794943 ± 0.011384 (0.018111) |
| baseline_bald | 0.599249 ± 0.002870 (0.004567) | 0.702165 ± 0.016934 (0.026941) | 0.791463 ± 0.016637 (0.026470) |
| baseline_dial_style | 0.598077 ± 0.005843 (0.009297) | 0.709403 ± 0.009951 (0.015832) | 0.798524 ± 0.009609 (0.015288) |
| baseline_wang_style | 0.600470 ± 0.006379 (0.010149) | 0.709344 ± 0.018161 (0.028895) | 0.798169 ± 0.017410 (0.027700) |
| baseline_random | 0.586203 ± 0.004103 (0.006528) | 0.705272 ± 0.017026 (0.027089) | 0.794085 ± 0.016167 (0.025722) |
| baseline_coreset | 0.588139 ± 0.003897 (0.006200) | 0.700787 ± 0.008052 (0.012810) | 0.790060 ± 0.007991 (0.012714) |

从科研表述角度的直接结论（严格对齐上述数值）：
- dial_style、wang_style、entropy 在 final_miou 上表现强且 CI95 相对可控，是论文中必须保留的强对照。
- random 的 final_miou 均值不低，但 ALC 明显偏低，符合“学习效率较差”的常识；因此论文应把 ALC 与 final_miou 分开讨论。

（与本文件“基于 4 个 run 的专题分析”一致的结论简述）
- Full policy 闭环在 ALC 上优势更稳定，但最终 final_miou 不保证领先所有强基线（需解释后期 plateau 与 test gap）。
- ramp_guardrail 能在部分 seed 提升 final_miou，但存在跨 seed 退化，提示 ramp 强度、风险门控与 guardrail 统计一致性仍需改进。

### 9. 威胁到有效性（Validity Threats）与待补全点清单

为保证“科研报告完整性”，需要在论文中明确或在后续实验中补齐的点包括：
- 数据增强：当前数据集类支持 transform，但工程内未提供明确的 augmentation pipeline；论文需说明是否使用增强、若无增强需解释。
- K 的口径一致性：当前主实现口径为 unlabeled KMeans representativeness；若 results/配置中出现其他 K 定义字段，需要在论文中明确“实际启用口径”以避免口径漂移。
- KMeans 簇数隐含假设：当前实现存在簇数与默认 query_size=88 的耦合，若更改预算/轮数导致 query_size 改变，需要明确如何设置簇数以保持公平比较。
- LLM 依赖：需要明确 LLM 不可用时的降级策略、重试策略与统计处理（否则可复现性会被质疑）。
- 失败 run 的统计：中途失败 run 应与完整预算 run 区分，必要时按预算段对齐比较 ALC/曲线。

---

## 0. 指标口径与数据源说明（本项目的“最终指标”为什么会不一致）

### 0.1 两类“最终指标”来源（必须区分）
在本项目 results 目录中，常见两类“最终指标”：
- 每轮轨迹指标（用于画 learning curve 与计算 ALC）
  - 数据来源：experiment_results.json 的 performance_history[*]
  - 口径：model_selection=best_val（每轮在 val 上选出的 checkpoint 指标）
  - 典型字段：round、labeled_size、mIoU、f1_score、best_val_mIoU、best_val_epoch、selected_epoch
- 汇总口径的 final_miou / final_f1（常被表格展示为“最终 mIoU”）
  - 数据来源：experiment_results.json 的 final_miou / final_f1（以及 status.json 的 result.*）
  - 口径：优先来自最后阶段的“final report”（通常是 test split 的评估）；当 test mask 不可用或该轮失败时可能回退到其他口径

因此会出现：round 16 的 mIoU（val 选模轨迹）高于 final_miou（汇总口径）的情况。这是口径差异导致的正常现象，不应直接判为数据错误。任何结论必须指明使用的是：
- 轨迹口径：performance_history[*].mIoU
- 汇总口径：final_miou（以及是否来自 test 的 final_report）

### 0.2 本文复核的原始数据文件（路径纯文本）
每个 run 至少用到这些文件：
- 结果汇总：/Users/anykong/AD-KUCS/AAL_SD/results/runs/<run_id>/experiment_results.json
  - 本文主要使用字段：alc、final_miou、final_f1、performance_history（用于解释 plateau/回撤/稳定性）
- 配置与环境：/Users/anykong/AD-KUCS/AAL_SD/results/runs/<run_id>/manifest.json
  - 本文主要使用字段：config（QUERY_SIZE、TOTAL_BUDGET、N_ROUNDS、MODEL_SELECTION、TRAIN/VAL/TEST split）
- 过程事件（必要时追根溯源）：/Users/anykong/AD-KUCS/AAL_SD/results/runs/<run_id>/*_trace.jsonl 与 *_status.json
- l3 选样离线回填统计（如果 run 内有 reports/）：/Users/anykong/AD-KUCS/AAL_SD/results/runs/<run_id>/reports/backfill_l3_selection_stats_*.csv

## 1. 结果总览：Full model vs Baseline（跨 seed，按 final_miou 与 ALC 同时阅读）

说明：
- Full model 指 full_model_A_lambda_policy（Agent + λ policy 闭环）。
- Baseline 指 random/entropy/coreset/bald/dial_style/wang_style 等。
- 下面每个表的数据都来自对应 run 的 experiment_results.json（字段 alc/final_miou/final_f1）。

### 1.1 seed42（baseline_20260309_211601_seed42）
数据文件路径：
- /Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed42/experiment_results.json

关键对比表（节选常用对照）：

| 实验 | ALC | final_miou | final_f1 | 结论备注 |
|---|---:|---:|---:|---|
| full_model_A_lambda_policy | 0.6101 | 0.6994 | 0.7891 | ALC 领先，但 final_miou 低于强 baseline |
| baseline_dial_style | 0.6012 | 0.7160 | 0.8046 | 本 run 的 best baseline（按 final_miou） |
| baseline_random | 0.5833 | 0.7042 | 0.7943 | ALC 明显更低，但 final_miou 不弱 |
| baseline_entropy | 0.6029 | 0.6938 | 0.7833 | ALC 接近 full，但 final_miou 较低 |
| baseline_bald | 0.5918 | 0.7021 | 0.7914 | final_miou 与 full 接近，ALC 略低 |

结论（按调优逻辑解释）：
- Full 的优势主要体现在学习效率（ALC）而不是最终一枪（final_miou）。
- 该 run 的现象更符合后期增长不足/late-stage plateau：早期靠 U（不确定性）增益快，但后期 K（代表性/多样性）引入强度与风险门控耦合不足，导致最终口径追不上 dial_style 的长尾收益。

### 1.2 seed43（baseline_20260309_211601_seed43）
数据文件路径：
- /Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed43/experiment_results.json

关键对比表（节选常用对照）：

| 实验 | ALC | final_miou | final_f1 | 结论备注 |
|---|---:|---:|---:|---|
| full_model_A_lambda_policy | 0.6112 | 0.7212 | 0.8096 | ALC 仍领先，但被随机 baseline 反超 |
| baseline_random | 0.5969 | 0.7390 | 0.8255 | 本 run 的 best baseline（按 final_miou），且差距显著 |
| baseline_dial_style | 0.5999 | 0.6904 | 0.7797 | 在该 seed 下并不强 |

解读（针对“跨 seed 不鲁棒”的表述应更严谨）：
- baseline_random 在该 seed 下异常强，说明策略收益高度受数据分布/seed 偶然性影响。
- Full 的优势仍是稳定的 ALC 与更一致的策略执行，但为了降低被偶然强 baseline 反超的概率，需要在后期策略上增强：late-stage ramp 的触发条件、风险联动、步长限制与 guardrail 的统计可靠性。

## 2. Full model vs 变体：有效方向与不鲁棒点

### 2.1 20260309：ab_tune 与 lambda_agent 的总体收益不明显

数据文件路径（两 seed 各一份）：
- /Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed42/experiment_results.json
- /Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed43/experiment_results.json

观察：
- `full_model_B_lambda_agent`（显式 set_lambda）在两个 seed 上均未体现优势
- `ab_tune_hi_ep10 / ab_tune_lo_ep10` 对 `final_miou` 与 ALC 的提升不稳定

推断：
- “放权给 agent 显式调 λ”在信号不足（如 K/U 轨迹、风险统计不全）时更容易出现抖动或保守，最终收益不如 policy 闭环稳定。

### 2.2 20260311：ramp+guardrail 在 seed42 有收益，但跨 seed 不鲁棒

数据文件路径（两 seed 各一份）：
- seed42：/Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed42/experiment_results.json
- seed43：/Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed43/experiment_results.json
如需追溯 ramp/guardrail 的实际参数与触发过程，可看对应 trace：
- /Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed42/full_model_A_lambda_policy_ramp_guardrail_trace.jsonl
- /Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed43/full_model_A_lambda_policy_ramp_guardrail_trace.jsonl

观察（按 final_miou，含 ALC/final_f1 以便复核）：
- seed42：
  - full_model_A_lambda_policy：ALC=0.6048，final_miou=0.6991，final_f1=0.7885
  - full_model_A_lambda_policy_ramp_guardrail：ALC=0.6108，final_miou=0.7098，final_f1=0.7994
  - full_model_A_lambda_policy_ramp_guardrail_train_probe：ALC=0.6037，final_miou=0.6818，final_f1=0.7708
- seed43：
  - full_model_A_lambda_policy：ALC=0.6070，final_miou=0.7007，final_f1=0.7901
  - full_model_A_lambda_policy_ramp_guardrail：ALC=0.6018，final_miou=0.6644，final_f1=0.7525
  - full_model_A_lambda_policy_ramp_guardrail_train_probe：ALC=0.5915，final_miou=0.6763，final_f1=0.7654

推断：
- `late_stage_ramp` 的强推策略能解决后期 plateau，但如果 ramp 与 overfit 风险/val→test gap 的联动不足，就会出现 seed 敏感（某些 seed 下过推 K、导致后期退化）。
- `train_probe` 改变 TVC/风险信号统计性质，容易导致风险门控误判，进而影响 λ 决策稳定性。

### 2.3 离线回填的 l3 统计：U/K 中位数轨迹（新增证据）

离线工具已在 `results/runs/<run_id>/reports/` 生成 `backfill_l3_selection_stats_*.csv`（每个实验一份），并在每个 run 下生成汇总：
- seed42（20260311）汇总：/Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed42/reports/backfill_l3_selection_stats_summary.csv
- seed43（20260311）汇总：/Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed43/reports/backfill_l3_selection_stats_summary.csv
- seed42（20260309）汇总：/Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed42/reports/backfill_l3_selection_stats_summary.csv
- seed43（20260309）汇总：/Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed43/reports/backfill_l3_selection_stats_summary.csv

关键发现 1（对之前推断的反证）：在 `ramp_guardrail` 的后期轮次，`selected_ids` 命中 `top_items` 的覆盖率显著下降，导致 `u_median_top/k_median_top` 以及基于 `selected_items` 的统计都不再能代表“全选中集合”
- 新增一致性指标：`coverage_selected_in_top = |selected_ids ∩ top_items| / |selected_ids|`，以及 `selected_scored_frac_u/k`（选中集合里能拿到 U/K 的比例）
- seed43 `ramp_guardrail`（文件：/Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed43/reports/backfill_l3_selection_stats_full_model_A_lambda_policy_ramp_guardrail.csv）：
  - round10 覆盖率=0.1818，selected_scored_frac_u=0.1818，selected_scored_frac_k=0.1818
  - round13 覆盖率=0.0，selected_scored_frac_u=0.0，selected_scored_frac_k=0.0（该轮 selected 的 U/K 统计为空）
  - round15 覆盖率=0.0455，selected_scored_frac_u=0.0455，selected_scored_frac_k=0.0455
- seed42 `ramp_guardrail`（文件：/Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed42/reports/backfill_l3_selection_stats_full_model_A_lambda_policy_ramp_guardrail.csv）：
  - round10 覆盖率≈0.2386
  - round11 覆盖率=0.0
  - round15 覆盖率≈0.0341

结论（更严格）：当覆盖率显著低于 1.0 时，`u_median_top/k_median_top` 只能说明“记录到的那部分 top_items 统计极端”，不能推导“全未标注池 topk 前沿退化”；同理，当 `selected_scored_frac_u/k` 很低时，`u_median_selected/k_median_selected` 只基于少量命中样本计算，不能代表全选中集合。

关键发现 2（对之前结论的校正）：不同实验的 `topk` 与 `selected` 统计一致性差异较大，说明“l3 日志的 top_items/selected_items 可能不是同一个排序空间”，因此它更像“调试视角的候选快照”而不是严格意义上的“全池 topk”
- 在部分 full_model 实验中，`u_median_top == u_median_selected` 且 `k_median_top == k_median_selected`（例如 seed43 full_model 的多个轮次），更像“候选 cache 被当作 topk”而非全量 pool 排序结果。
  - 可复核文件路径：/Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed43/reports/backfill_l3_selection_stats_full_model_A_lambda_policy.csv

解读（更严格）：在 agent 模式下，这组 l3 轨迹目前只能作为“近似信号”，适合做“异常侦测/触发器候选”，不适合作为定量阈值直接驱动 λ 策略闭环。若要让它可用于闭环，应首先保证两件事：
- 覆盖率：`coverage_selected_in_top` 接近 1（否则 top/selected 统计都会偏置）
- 选中集合可度量：能从真实 `selected_ids` 上直接计算 U/K 分布（而非通过 top_items 命中推断），例如强制记录 `selected_score_stats`（U/K 的 mean/p50/p75 等）

关键发现 3：20260309 的 baseline 覆盖不完整并非离线工具问题，而是原始 trace 未提供所需信号
- `baseline_random/bald/entropy/coreset` 这类策略无法回填 u/k 中位数（在汇总中常见 `rounds_with_stats=0`），因为它们原始 trace 中没有 l3 级别分解或缺少 K 指标。
- `baseline_dial_style/baseline_wang_style` 与 full_model 系列可回填（常见 `rounds_with_stats=15`），因为它们在实现里显式计算并记录了 U/K 相关结构。

## 3. Policy 改进空间（按优先级）

### P0：把 late-stage ramp 从“固定日程”改为“条件触发 + 风险联动”

依据：设计文档的 `diagnose()` 规则中已明确了 `late_stage_plateau`（late_stage_gain）、`over_conservative`（lambda_max）等判据。  
建议：
- 触发：`late_stage_gain` 低且 `overfit_risk` 低、TVC 稳定时才 ramp
- ramp 终点/斜率与风险耦合：风险偏高时降低 `end_lambda` 或延后 ramp 起点
- 修订：不要在现阶段用 `u_median_top/k_median_top` 做闭环触发（因为 ramp 的后期覆盖率显著下降，见 2.3），最多作为离线告警；闭环触发应基于可靠信号（late_stage_gain/overfit_risk/TVC 等）

实现状态（2026-03-13）：
- 已在 `Toolbox._compute_policy_lambda_for_round()` 中实现 `late_stage_ramp.conditional` 门控
- ramp 门控已接入 `overfit_risk`、`grad_train_val_cos_neg_rate`、`miou_delta/low_gain_streak`
- 门控结果写入 `lambda_policy_apply.diagnostics.late_stage_ramp`

### P1：把 guardrail 从“硬阈值”改为“自适应阈值 + 配额联动”

当前 guardrail 的 `u_median_min/u_low_frac_max` 作为常数，可能随 seed/轮次漂移。  
建议：
- `u_median_min` 使用最近 k 轮或 topk 分布的分位数动态估计
- 将 “fallback_quota_u_frac” 与 “lambda_step_down” 解耦，避免一触发就过度降 λ
- 修订：在修复覆盖率与“选中集合可度量”之前，不建议引入基于 topk 的硬约束；优先把 guardrail 做成“直接在 selected_ids 上计算 U 分布”的约束（以 `selection_guardrail.stats_before/stats_after` 为准），并将 topk 统计仅用于提示“是否需要检查排序空间一致性”

### P2：把轮内振荡（epoch 级）纳入风险与稳定性闭环

建议把 `epoch_miou_volatility`、`tvc_sign_flip_rate`（设计文档已定义）接入 diagnose→policy 映射：当轮内振荡显著时，下轮限制 λ 上升或减少 epochs（若 epochs 可调）。

实现状态（2026-03-13）：
- 已在训练侧补充 `epoch_miou_volatility` 与 `tvc_sign_flip_rate` 的采集与 trace 输出
- 已在 policy 闭环中加入稳定性门控：高波动/高翻转时禁止 λ 上调（`stability_risk_hold`）

### P3：将 guardrail 统计口径从“topk命中子集”改为“最终 selected_ids 全量”

依据：在 `ramp_guardrail` 后期，`coverage_selected_in_top` 在多轮接近 0，`selected_score_stats` 常出现 `n<<selected_count`，导致统计失真。  
建议：
- guardrail 触发后，直接在 `current_scores` 上对 `selected_ids_after` 计算 U/K 分布（mean/p50/p75/frac_u_lt）
- 将该全量统计写入 `selection_guardrail.stats_after_selected_all`
- `l3_selection` 中由 `top_items` 回推的统计仅保留为“告警信号”，不再用于闭环阈值

实现状态（2026-03-13）：
- 已新增 `stats_after_selected_all`（基于最终 `selected_ids_after` 全量统计）
- 已新增 guardrail 自适应阈值能力（历史分位数模式），并写入 `threshold_mode/threshold_adaptive`
- `l3_selection_stats` 已补 `coverage_selected_in_top` 与 `topk_stats_for_closed_loop=false`

## 4. Prompt 改进空间（前提：LLM 链路可用）

在多个报告中出现 “LLM Client not available”，且 trace 中 `context.llm_mode` 多为空，说明实验中存在 LLM 未参与决策的情况。prompt 优化只有在 LLM 链路稳定可用后才具备收益。

建议（prompt 结构层面）：
- 将输入拆为三块：`diagnostics`、`issues`、`recent_history`，并要求输出“可裁剪的参数建议”（按白名单与范围）
- 强制输出：主要瓶颈一句话 + 3 条按优先级排序的建议（每条含预期收益与风险）
- 将“禁止越权/禁止 set_lambda”等权限明确写入 system prompt，并要求给出“在权限内的替代动作”（例如只能通过 finalize_selection 间接影响 λ）

### Prompt 线落地前置（必须先做）

- 在 trace 中稳定记录 `agent_llm_request/agent_llm_response`，并携带 `messages_sha1/response_sha1`
- 每轮记录 `llm_calls_in_round`、`llm_transport_error_count`、`llm_invalid_format_count`
- 没有上述事件或计数缺失时，该轮默认判定为“LLM未参与决策”，不纳入 prompt 效果分析

实现状态（2026-03-13）：
- 已在 async agent 路径补齐 `agent_llm_request/agent_llm_response` 事件
- prompt 模板已结构化为 `diagnostics/issues/recent_history` 输入块
- prompt 输出已加入动作白名单、参数范围与权限边界约束

## 5. 可观测性缺口：l3_selection 与 lambda_effective 的结论与整改

### 5.1 l3_selection “看不到”多为检索/可视化问题

原因：
- `l3_selection` 单行 JSON 体积巨大（含 top_items/selected_items 数组），IDE/grep 输出经常截断，导致误以为未采集。

整改（降低检索成本，不改策略）：
- 在写入 `l3_selection` 的同时额外写入一条轻量事件 `l3_selection_stats`，包含：
  - `u_median_selected/k_median_selected`
  - `u_median_top/k_median_top`
  - `topk/selected_limit/source`

落点（代码审计路径）：/Users/anykong/AD-KUCS/AAL_SD/src/main.py 的 _append_l3_selection()

### 5.2 selection 事件中 lambda_effective=null：记录口径缺失（instrumentation gap）

原因：
- `selection` 事件的 `lambda_effective` 来自 `_last_ranking_metadata`；在 agent_finalize_selection 路径下，可能只补了 `_last_ranked_items`，未补 `_last_ranking_metadata`，导致 `_sampler_audit()` 写入 null。

整改（不改策略，只补 trace）：
- 在 `_sampler_audit()` 中，当 ranking_meta 缺失时，按优先级从同轮控制事件回填：
  1) `lambda_guard.lambda_after`
  2) `selection_guardrail.lambda_after`
  3) `lambda_override.applied`
  4) `lambda_policy_apply.applied`
  5) 最后兜底从 `toolbox.control_state.lambda_override_round` 取值

落点（代码审计路径）：/Users/anykong/AD-KUCS/AAL_SD/src/main.py 的 _sampler_audit()。

## 6. 本次实现落地与验证建议

已落地改动（不改变策略/训练，仅增强 trace 观测）：
- `selection` 事件补齐 `lambda_effective/lambda_source` 的回填逻辑
- 新增 `l3_selection_stats` 轻量事件（便于形成 u/k trajectory）
- 异步 agent 路径补齐 `agent_llm_request/agent_llm_response` 事件，确保 prompt 可审计
- policy 增加条件化 late-stage ramp 与稳定性门控（epoch 波动/符号翻转）
- guardrail 增加自适应阈值与 `selected_ids_after` 全量统计

验证建议：
- 运行单测：`python -m pytest -q`
- 对任意 run 的 trace，grep `l3_selection_stats` 应每轮可见；grep `selection` 的 `lambda_effective` 不应在 guardrail 触发轮次为 null
- 对异步实验 trace，grep `agent_llm_request|agent_llm_response` 应每轮可见并可按 `session_id+round+step` 配对

## 7. policy 与 prompt 两条线的可落地执行方案（两周）

### Week 1（先修证据链）

- Policy 线：
  - 在 `selection_guardrail` 事件补 `selected_ids_after` 的全量 U/K 统计字段
  - 给 `lambda_policy_apply` 增加 `lambda_before_policy/lambda_after_policy/ramp_phase` 字段
  - 产出一次 dry-run λ 轨迹（固定训练态输入）用于核对 ramp 与 guardrail 次序
- Prompt 线：
  - 强制打开 `enable_agent_prompt_logging`
  - 统计并输出每轮 LLM 可用性（calls、transport error、format invalid）
  - 形成 “LLM参与率” 指标：`rounds_with_llm_response / rounds_total`

### Week 2（做可证伪消融）

- Policy 线：
  - 协议统一为 `train_holdout` TVC（不使用 official val 标签梯度）
  - 跑 3 组：`full_model` / `ramp_guardrail` / `ramp_guardrail + selected_all_stats`
  - 每组至少 5 seeds，报告 mean±std 与 paired 差值
- Prompt 线：
  - 同 seed 严格 A/B：`prompt_v1` vs `prompt_v2`
  - 控制变量：同模型、同温度、同预算、同策略权限
  - 只在 `LLM参与率>=95%` 的轮次统计 prompt 影响

### 通过门槛（Go/No-Go）

- Policy：`ramp_guardrail` 在 5 seeds 上相对 `full_model` 的 `final_miou` 平均提升 > 0 且方差不扩大
- Prompt：在 LLM参与率达标前，不对 prompt 结论作因果陈述；达标后才报告 A/B 结果

## 8. 进展

- 2026-03-13：完成 trace 观测缺口整改（lambda_effective 回填、l3_selection_stats 事件）
- 2026-03-13：完成本报告初版输出
- 2026-03-13：基于离线回填 CSV（u/k 中位数轨迹）更新策略结论与风险点
- 2026-03-13：补充异步 agent 的 LLM request/response trace 事件
- 2026-03-13：补充 `selection_guardrail.lambda_after` 到 `lambda_effective` 回填优先级
- 2026-03-13：落地条件化 ramp、稳定性门控、自适应 guardrail 与 prompt 结构化约束

## 9. 证据对照表（关键论断 → 原始数据锚点）

本节把报告中的关键论断逐条映射到可复核的数据文件与字段/检索锚点，便于审稿式核查。

### 8.1 口径与指标

- 论断：`performance_history[*].mIoU`（每轮“选模 checkpoint 的 val mIoU”，也是 ALC 的计算口径）与 `final_miou`（最终输出口径）可能不一致，不能混称“最终 mIoU”  
  - 原始数据：四个 run 的 `experiment_results.json`  
    - ALC/曲线口径：`performance_history[*].mIoU`（以及 `labeled_size/model_selection/selected_epoch`）  
    - 选模口径补充：`performance_history[*].best_val_mIoU`、`best_val_epoch`（同轮训练过程内的 best-val）  
    - 最终输出口径：`final_miou`（优先来自最终报告 `final_report.mIoU@test_split`；若最终报告不可用则回退为最后一轮选模的 `performance_history[-1].mIoU@val`）  
    - seed42-0309：/Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed42/experiment_results.json  
    - seed43-0309：/Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed43/experiment_results.json  
    - seed42-0311：/Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed42/experiment_results.json  
    - seed43-0311：/Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed43/experiment_results.json  
  - 交叉印证：`detailed_results_report.md` 的“性能历史”表（每轮 best_val）与 `summary_report.md` 的“最终 mIoU”列  
  - 代码口径来源：/Users/anykong/AD-KUCS/AAL_SD/src/main.py 的结果聚合逻辑（ALC 使用 `performance_history[*].mIoU`；最终报告使用 `test_split` 的 `final_report`，并在不可用时回退）  

### 8.2 Full model vs Baseline（跨 seed）结论

- 论断：seed42（20260309）中 `baseline_dial_style(final_miou)` 高于 `full_model_A_lambda_policy(final_miou)`  
  - 原始数据：seed42-0309 /Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed42/summary_report.md（“最终 mIoU”表）  
  - 可复核字段：seed42-0309 /Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed42/experiment_results.json（各实验 `final_miou`）  

- 论断：seed43（20260309）中 `baseline_random(final_miou)` 显著高于 `full_model_A_lambda_policy(final_miou)`  
  - 原始数据：seed43-0309 /Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed43/summary_report.md  
  - 可复核字段：seed43-0309 /Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed43/experiment_results.json（各实验 `final_miou`）  

- 论断：Full model 的 ALC 在对应 run 内排名更靠前（但 final_miou 未必领先）  
  - 原始数据：对应 run 的 `summary_report.md`（“ALC 排名”）与 `experiment_results.json`（字段：`alc`/`final_miou`）  

### 8.3 Full model vs 变体结论

- 论断：20260309 中 `full_model_B_lambda_agent` 与 `ab_tune_*` 的收益不稳定/整体不显著  
  - 原始数据：seed42-0309 与 seed43-0309 的 summary_report.md（ALC 与 final_miou 对比）  
    - /Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed42/summary_report.md  
    - /Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed43/summary_report.md  
  - 可复核字段：对应 `experiment_results.json` 中 `full_model_B_lambda_agent`、`full_model_A_lambda_policy_ab_tune_*` 的 `final_miou/alc`  

- 论断：20260311 中 ramp_guardrail 在 seed42 有收益但 seed43 明显失败（跨 seed 不鲁棒）  
  - 原始数据：seed42-0311 与 seed43-0311 的 summary_report.md  
    - /Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed42/summary_report.md  
    - /Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed43/summary_report.md  
  - 可复核字段：对应 `experiment_results.json` 中 `full_model_A_lambda_policy` 与 `full_model_A_lambda_policy_ramp_guardrail` 的 `final_miou`  

### 8.4 离线回填 l3 统计（新增证据）与“需要审视”的依据

- 论断：离线回填会为每个实验生成 `backfill_l3_selection_stats_<exp>.csv`，并产出 `summary.csv`  
  - 原始数据（汇总入口）：  
    - seed42-0311：/Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed42/reports/backfill_l3_selection_stats_summary.csv  
    - seed43-0311：/Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed43/reports/backfill_l3_selection_stats_summary.csv  
    - seed42-0309：/Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed42/reports/backfill_l3_selection_stats_summary.csv  
    - seed43-0309：/Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260309_211601_seed43/reports/backfill_l3_selection_stats_summary.csv  

- 论断：ramp_guardrail 的 `u_median_top/k_median_top` 在部分后期轮次出现极端值（但不能直接等同“全池前沿退化”）  
  - 原始数据：  
    - seed42-0311 ramp_guardrail：/Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed42/reports/backfill_l3_selection_stats_full_model_A_lambda_policy_ramp_guardrail.csv（列：u_median_top/k_median_top/u_median_selected/k_median_selected）  
    - seed43-0311 ramp_guardrail：/Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728_seed43/reports/backfill_l3_selection_stats_full_model_A_lambda_policy_ramp_guardrail.csv  
  - 需要审视的证据点（报告中引用）：在 ramp_guardrail 的后期轮次，新增的一致性列 `coverage_selected_in_top` 显著下降（甚至为 0），因此 topk/selected 的中位数不能代表“全选中集合”或“全池前沿”  

- 论断：20260309 的部分 baseline（random/bald/entropy/coreset）回填为 0 并非离线工具失败，而是原始 trace 不含必要信号  
  - 原始数据：对应 run 的 `summary.csv` 里 `rounds_with_stats=0` 的实验行 + 该实验 trace 中缺少 `l3_selection`/K 指标事件  

### 8.5 Policy 改进项（每条建议对应的可验证证据）

- P0：late-stage ramp 从固定日程改为“条件触发 + 风险联动”  
  - 可验证信号（设计框架定义）：`late_stage_gain`（final - round8）、`overfit_risk`、`tvc_*`、`lambda_trajectory/lambda_volatility`（框架文件路径：/Users/anykong/AD-KUCS/AAL_SD/AAL-SD-Doc/auto_tuning_framework_design.md）  
  - 原始事件锚点：各实验 `*_trace.jsonl` 中的 `round_summary`、`controller_step.state`（字段含 `overfit_risk/grad_train_val_cos_*` 等）与 `lambda_policy_apply`  
  - 辅助证据：离线回填 CSV（仅作告警级别，不作硬约束）中的 `u_median_top/k_median_top` 极端波动轮次  

- P1：guardrail 改为自适应阈值 + 配额联动，并补“前沿退化监控”（先告警后闭环）  
  - 原始事件锚点：`*_trace.jsonl` 中 `selection_guardrail` 事件（字段：`stats_before/stats_after/thresholds`）  
  - 评估方式：对比 guardrail 触发轮次前后 `selected` 集合的 `u_median/frac_u_lt` 变化，以及对应轮次 `mIoU` 变化  
  - 离线回填支撑：`backfill_l3_selection_stats_*.csv` 新增列 `coverage_selected_in_top`、`selected_scored_frac_u/k`，用于判定 topk/selected 统计是否可用；当这些一致性指标较低时，topk 仅用于提示而不用于约束  

- P2：轮内振荡（epoch 级）纳入风险与稳定性闭环  
  - 可验证信号（设计框架定义）：`epoch_miou_volatility`、`tvc_sign_flip_rate`（框架文件路径：/Users/anykong/AD-KUCS/AAL_SD/AAL-SD-Doc/auto_tuning_framework_design.md）  
  - 原始事件锚点：`*_trace.jsonl` 中 `epoch_end`（轮内序列）与 `controller_step.state.grad_*`（若启用）  

### 8.6 Prompt 改进项的前置条件（可证伪的可用性判据）

- 论断：若 LLM 未参与决策，则 prompt 改动无法解释性能差异  
  - 原始证据：`reports/*anomaly_report*.md` 中 “LLM Client not available” 记录、以及 trace 中 `selection.context.llm_mode` / `llm_degraded` 事件（若触发）  
  - 可证伪条件：提供 LLM on/off 的严格消融（同 seed/同预算/同配置，仅开关 LLM 或 prompt）并在 trace 中出现明确的 LLM 调用/模式标记  

---

## 附录 A：基于当前源码（/Users/anykong/AD-KUCS/AAL_SD/src）的科研研究报告（详细版）

本附录面向“论文/开题/中期检查”写作需求，系统化说明：数据集、预处理、模型、主动学习算法设计、λ 控制闭环设计、实验矩阵与参数空间、以及与 results 目录对齐的当前实验进展。

### A.1 研究任务、约束与核心贡献点

研究任务：遥感滑坡语义分割（二分类：滑坡/非滑坡），输入为 14 通道多光谱/多源特征，输出为像素级 mask。

核心约束：
- 标注成本高，设置固定标注预算（预算比率约 40%），要求在预算轴上实现更高的“学习效率”与最终泛化。
- 训练过程存在不稳定与过拟合风险，需要闭环控制机制降低回撤与失败率。

核心贡献点（对应系统机制，而非单一技巧）：
- 统一的“U（不确定性）+ K（代表性）”融合选样框架，并由 λ 动态权衡探索/利用。
- 风险闭环的 λ policy：warmup → risk closed-loop → late-stage ramp（在风险可控时增强 K）。
- 安全阀与防崩溃机制：selection guardrail（U 底线）+ rollback（回撤保护）+ 可恢复的数据池与 checkpoint。
- 可审计与可复现：每个 run 输出 manifest、trace、status、池文件与聚合报告。

### A.2 数据集（Landslide4Sense）与数据格式

数据目录结构（强约束，必须存在对应子目录）：
- TrainData/img 与 TrainData/mask
- ValidData/img 与 ValidData/mask
- TestData/img（若存在 TestData/mask 且完整，则用于最终测试口径指标）

单样本文件格式（H5）：
- 图像键：img，shape 为 (128, 128, 14)
- mask 键：mask，shape 为 (128, 128)，值为类别 id（0/1）

命名对齐规则（避免图像与 mask 不匹配）：
- 若 mask 文件名形如 mask_XXX.h5，会映射到 image_XXX.h5
- Train/Val split 会严格检查 mask 对齐；Test split 允许缺 mask，但最终测试口径评估要求 mask 完整

### A.3 冷启动与数据池构建（Active Learning 的 round-0）

冷启动的目标：避免初始 labeled 过小或类别极度偏斜导致训练不收敛、特征不可用、K 代表性失效。

关键策略：
- 初始标注比例：约 5%（由 INITIAL_LABELED_SIZE 控制）
- 分层抽样：依据 has_positive（mask 是否包含正类像素）进行分层抽样，保证初始 labeled 与 unlabeled 都包含正/负样本
- 池文件落盘：每轮更新 labeled/unlabeled 列表并持久化，确保中断可恢复

预算与轮设计（默认口径，与多数 results run 对齐）：
- 估计训练集总量约 3799
- 总预算约 1519（约 40%）
- 总轮数约 16（含最后阶段可能进行 test 口径评估）
- 每轮查询量 query_size 约 88（由预算与轮数自动推导得到）

### A.4 模型与训练协议

模型：
- DeepLabV3+，encoder=ResNet50
- 输入通道=14，输出类别=2
- 可选 MC Dropout（用于 BALD）

训练协议（默认）：
- 优化器：Adam，学习率 1e-4
- batch size：4
- 每轮训练 epoch：10（固定轮长）
- 评估：每轮在验证集计算 mIoU 与 F1（基于混淆矩阵宏平均）

指标体系（论文建议写法）：
- 学习效率：ALC（Area under Learning Curve），在预算归一化后对 (budget, mIoU) 曲线做 AUC
- 最终指标：final_miou/final_f1（优先取 test split 的最终报告；若不可用需说明回退策略）

### A.5 主动学习算法设计（U/K/λ 的统一框架）

定义一：不确定性 U（默认 entropy，像素级聚合）
- 对每个像素的类别概率 p 计算熵 H(p) = -sum_c p_c log2(p_c + 1e-10)
- 以像素熵的均值作为样本级不确定性 U
- 可配置仅对高熵像素聚合，以降低大量背景像素稀释不确定性的问题

定义二：代表性/知识增益 K（当前主实现口径：unlabeled KMeans representativeness）
- 从网络 backbone 提取样本特征向量 f_x（通过 GAP 得到定长向量）
- 对 unlabeled 特征做 KMeans 聚类，得到每点所属簇中心 c(x)
- 距离 d(x)=||f_x - c(x)||，以池内最大距离归一化
- 定义 K(x)=1 - d(x)/max(d_max, 1e-12)，越接近簇中心越“代表性”

融合打分：
- Score(x) = (1-λ) * U(x) + λ * K(x)
- λ 越小越偏探索（采高不确定样本），λ 越大越偏利用（采更代表性/多样性样本）

多样性与约束后处理（可选）：
- 可对候选集合执行特征空间的最远点采样（FPS），提升 batch 内多样性
- 可基于预测正类面积等启发式约束，保证一定比例的疑似滑坡样本进入 labeled

### A.6 λ 控制策略与闭环控制框架（控制系统视角）

λ 控制分层：

层 1：基础调度（无 Agent 也可运行）
- 依据标注进度 progress=|L|/B 进行 logistic 调度，使后期逐步偏向 K
- 用途：在没有复杂闭环时提供合理默认行为

层 2：规则化闭环 λ policy（推荐作为论文主方法）
阶段化结构（强调可解释与可复核）：
1) uncertainty-only phase（前几轮）
   - λ 固定接近 0，避免冷启动时特征噪声误导 K
2) warmup（过渡阶段）
   - λ 取固定值或区间采样，使策略从纯探索平滑过渡
3) risk closed-loop（风险闭环阶段）
   - 使用训练态势信号（例如 miou_delta、overfit_risk、梯度对齐 TVC）触发 λ 的上调/下调/保持
   - 支持统计阈值（例如 CI 分位数）降低阈值手工敏感
   - 支持 λ 平滑与单轮最大步长限制，减少振荡
4) late-stage ramp（后期提升 K 的机制）
   - 在风险可控且进入后期时逐步提高 λ，解决后期 plateau，追求 final_miou
   - 必须与风险门控耦合，否则易出现 seed 敏感（在某些 seed 上过推 K 导致退化）

安全阀（必须写清楚触发条件与行为）：
- selection guardrail：当选中集合的不确定性质量过低（例如 U 中位数过低或低 U 比例过高），自动下调 λ 或启用配额式补救
- rollback：当 miou_delta 显著为负且超过自适应阈值（由 epoch 内波动 std 决定）时触发回撤标记，用于下一轮策略保守化

感知信号设计（用于把“控制”写得像控制论文而非经验调参）：
- 性能信号：mIoU、F1、miou_delta、最佳/最后/EMA
- 过拟合信号：训练-验证梯度对齐（TVC 相关统计）、overfit_risk 标量
- 稳定性信号：epoch_miou_volatility（轮内波动）、回撤触发标记
- 采样侧信号：selected/top 的 U/K 分布统计、coverage_selected_in_top、selected_scored_frac_u/k（一致性与可用性门控）

### A.7 实验矩阵设计（论文主表 + 消融表 + 稳健性表）

建议以三层矩阵组织：

1) Baseline 对照（证明方法必要性）
- random、entropy、coreset、bald、dial_style、wang_style（至少覆盖探索型与代表性型）

2) 主方法消融（证明每个机制贡献）
- Full：full_model_A_lambda_policy（warmup + risk closed-loop）
- + guardrail：验证安全阀是否减少失败与回撤
- + late-stage ramp：验证是否解决 plateau 并提升 final_miou
- train_holdout 梯度探针：验证风险信号的统计严谨性对闭环稳定的影响
- u_adaptive：验证二阶信号（选中集合 U 分布变化）是否能改善 guardrail 的可靠性

3) 稳健性与统计显著性（让结论可发表）
- multi-seed：至少 3~5 seeds（最好 paired 设计，同 seed 对比）
- 汇报：mean/std/CI（以及 paired 差值的置信区间）
- 指标拆分：ALC（效率）与 final_miou（最终）分开比较，并对中途失败 run 做预算对齐说明

### A.8 参数空间设计（控制/感知/策略/训练 四类）

建议把参数空间分组，便于做 auto-tuning 或消融：

组 1：预算与轮
- INITIAL_LABELED_SIZE、TOTAL_BUDGET 或 BUDGET_RATIO、N_ROUNDS、QUERY_SIZE

组 2：训练超参
- LR、BATCH_SIZE、EPOCHS_PER_ROUND、是否 AMP、是否固定轮长

组 3：U 计算
- uncertainty_method（entropy/bald）、MC 样本数、aggregation 方式、entropy_threshold

组 4：K 计算与多样性
- KMeans 的簇数（是否绑定 query_size）、特征层选择、FPS 多样性开关、候选倍数、正类面积配额

组 5：λ policy（闭环核心）
- warmup 起止与 λ 范围
- 风险阈值与统计窗口（CI 分位数、窗口长度、最小样本数）
- λ 平滑与最大步长
- late-stage ramp 的起止轮与目标 λ
- guardrail 阈值（u_median_min、u_low_thresh、u_low_frac_max）、回退配额与最大迭代步数

组 6：rollback
- std_factor、tau_min（回撤阈值自适应的保守程度）

---

## 附录 B：results 目录的“当前实验进展”补充（用于写论文结果章节的素材）

### B.1 本文 4 个 run 的关键数值回顾（便于快速引用）
四个 run 的原始数据入口都是：
- /Users/anykong/AD-KUCS/AAL_SD/results/runs/<run_id>/experiment_results.json

在本文第 1 节已经给出两条关键 seed（20260309）的对比表；第 2.2 节给出了 20260311 的 ramp_guardrail 跨 seed 退化现象（含 ALC/final_miou/final_f1 的可复核值）。

### B.2 多种子聚合结果（用于“统计显著性/稳健性”表）
多种子聚合文件示例（路径纯文本）：
- /Users/anykong/AD-KUCS/AAL_SD/results/runs/baseline_20260311_201728/multi_seed_summary.json
- /Users/anykong/AD-KUCS/AAL_SD/results/runs/baselines_only_p3_20260313_212023/multi_seed_summary.json

建议论文呈现方式：
- 主表：报告每个方法的 final_miou 与 ALC 的 mean±std（或 CI95）
- 稳健性表：对关键策略（full vs ramp_guardrail vs guardrail）给出 paired 差值统计
- 失败分析：将 LLM 相关失败（DNS/IO 等）与训练回撤失败分开统计，避免把“系统不可用”误判为“算法无效”

### B.3 Auto-tuning（自动调参）现状与论文可复现性注意点
auto-tune run 的 manifest/status/trace 会清晰记录搜索空间、候选配置与失败原因。

建议写进论文的工程性约束说明：
- LLM 不可用时的降级策略与重试策略
- 失败 run 的处理方式（是否纳入统计、如何做预算对齐）
- 使实验可复现的必要产物：manifest + pools + checkpoints + trace
