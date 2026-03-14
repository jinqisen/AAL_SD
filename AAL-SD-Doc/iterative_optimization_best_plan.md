# AAL-SD 迭代优化（最优改造方案）

本文给出一套面向 AAL-SD 自动调参/自动优化的“最优”工程化方案：以 **多保真（multi-fidelity）+ 约束/信赖域黑盒优化（TR-BO）+ 资源分配（Hyperband/Successive Halving）** 为核心，把当前“诊断→提案→跑实验→选最优→迭代”升级为可控、可复现、可停机、可量化的优化系统。

## 1. 目标与约束

### 1.1 目标

- 在有限训练预算下最大化最终指标（默认：test mIoU）
- 迭代过程中保持稳定（不频繁发散/崩溃/回滚）
- 在噪声（seed/训练随机性）下减少误判

### 1.2 约束

- 每次评估（一次实验）代价昂贵
- 目标函数不可微、非凸、噪声大
- 存在软/硬约束：过拟合风险、训练稳定性、资源限制（并发/显存/时间）

## 2. 总体架构（三层闭环）

### 2.1 外层：多保真 + 资源分配的黑盒优化器（收敛性与效率来源）

把一次“评估”拆成多层保真度（fidelity）：

- **F1（筛选）**：branch/resume（从中后期续跑），较小轮数（例如 N_ROUNDS=10），单 seed
- **F2（确认）**：同一 run_id 上继续 resume 到完整轮数（N_ROUNDS=16），单 seed
- **F3（统计复核）**：多 seed 复核（例如 seeds=[42,43,44]），用均值/置信区间决策

采用 Hyperband/Successive Halving 的资源分配策略：

- 先评估更多候选（便宜的 F1）
- 淘汰大部分候选
- 对 top-k 继续投入预算（F2/F3）

### 2.2 中层：LLM 作为“提案器/解释器”，不作为“裁判”

LLM 输出结构化提案：

- direction（方向）
- parameter_changes（参数改动）
- risk（风险）
- expected_gain（预期收益）
- constraints（硬约束触发条件）

提案是否执行/投入多少预算由外层优化器决定：

- 过滤不可行参数（范围/互斥/权限）
- 依据 acquisition（见 2.3）选择要跑的候选
- 依据统计准则更新 incumbent（当前最优）

### 2.3 内层：信赖域（Trust Region）+ 接受准则（稳定收敛性来源）

维护：

- incumbent θ\*（当前最佳参数向量）
- trust radius r（允许的改动半径）

规则：

- 若迭代提升显著：扩大 r（更大胆探索）
- 若多次无提升/退化：缩小 r（收敛到稳定区域）
- 只有当“统计上可信地更好”才替换 incumbent（避免噪声误判）

## 3. 学术依据（可直接借鉴的理论线索）

- **GP-UCB/BO 的 regret bound**：把优化形式化为带噪 bandit，可证明评估次数增加时 simple regret 下降（子线性后悔）  
  - Srinivas et al., 2012, GP-UCB regret bounds（IEEE TIT）
- **Successive Halving/Hyperband**：对未知收敛曲线的鲁棒资源分配，提供正确性/样本复杂度分析  
  - Jamieson & Talwalkar, 2016（Successive Halving analysis）  
  - Li et al., 2017（Hyperband, JMLR）
- **Constrained BO / Trust Region**：在可行域内做局部优化，避免大步跳变导致的震荡与不可行解  
  - 可从 Constrained BO 综述入手（IEEE Access 2025）

## 4. 本仓库落地实现（对应代码模块）

实现落点为 `src/tuning_opt/`，主要模块：

- `orchestrator.py`：主循环（iteration）、多保真调度、停止条件、incumbent 更新
- `llm_client.py`：读取 `src/tuning_llm_config.json`（默认），调用兼容 OpenAI Chat API 的服务
- `pool_resume.py`：准备 base pools、从指定 round 生成分支 pools + checkpoint
- `evaluator.py`：调用 `src/run_parallel_strict.py` 分阶段运行（F1/F2/F3），解析 md 产物得到指标
- `space.py`：参数空间定义、向量化、信赖域采样、范围裁剪

## 5. 使用方式（命令行）

### 5.1 默认启用 LLM（读取 tuning_llm_config.json）

先设置密钥（推荐仅使用环境变量）：

```bash
export TUNING_LLM_API_KEY="your-key"
```

启动：

```bash
cd /Users/anykong/AD-KUCS/AAL_SD

python src/tuning_opt/orchestrator.py \
  --initial-run-id baseline_20260311_201728_seed42 \
  --initial-exp full_model_A_lambda_policy \
  --target-miou 0.74 \
  --max-iterations 10 \
  --seeds 42,43,44 \
  --max-concurrent 1
```

### 5.2 纯规则模式（显式关闭 LLM）

```bash
python src/tuning_opt/orchestrator.py ... --no-llm
```

## 6. 收敛/停机判据（建议默认启用）

- target reached：best_miou ≥ target
- plateau：最近 N 次迭代提升均值 < ε 且置信度不足
- budget hit：达到 max_iterations 或 wall-clock 上限
- trust region collapse：r 下降到 r_min 且多次无显著提升

