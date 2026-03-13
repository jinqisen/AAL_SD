# AAL-SD 实验调优自动化逻辑框架

为实现“分析实验数据 -> 调整参数 -> 重新实验”的闭环自动化，建议采用以下逻辑架构。

## 1. 数据观测算子 (Observation)
脚本应从 `results/runs/<run_id>/experiment_results.json` 或 `*_trace.jsonl` 中提取以下核心信号：

| 信号分类 | 核心指标 | 定义/阈值 |
| :--- | :--- | :--- |
| **性能** | ALC (Area under Learning Curve) | 主目标函数，越高性能越好。 |
| **稳定性** | Rollback Count | 触发性能回撤的轮数比例，>15% 视为不稳定。 |
| **风险** | Severe Overfit Rate | 触发 `severe_overfit` 策略的轮次。 |
| **多样性**| U-Median Trend | 选中样本不确定性的中位数趋势，若后期过低说明探索不足。 |
| **质量** | TVC (Train-Val-Cos) Mean | 训练与验证梯度一致性，均值应 > 0.5。 |

---

## 2. 诊断与反馈逻辑 (Diagnosis & Feedback)

根据观测信号，脚本执行以下条件分支来生成参数建议：

### A. 稳定性优先分支 (Fixing Instability)
*   **触发条件**：`Rollback Count > 2` 或 `mIoU 波动变异系数 (CV) > 0.1`。
*   **调整逻辑**：
    1.  降低 **`LAMBDA_DELTA_UP`** (例如：0.15 -> 0.10)，减缓 K 的介入速度。
    2.  增加 **`LAMBDA_DELTA_DOWN`** (例如：0.10 -> 0.15)，加速回撤反馈。
    3.  降低 **`OVERFIT_RISK_HI`**，使风险抑制更敏感。
    4.  增加 **`LAMBDA_DOWN_COOLING_ROUNDS`** (例如：1 -> 2)，抑制振荡。

### B. 性能推进分支 (Boosting Performance)
*   **触发条件**：`ALC < Baseline` 且 `Rollback Count == 0` (系统过于保守)。
*   **调整逻辑**：
    1.  增加 **`LAMBDA_DELTA_UP`**，更激进地搜索代表性分布。
    2.  降低 **`MIOU_LOW_GAIN_THRESH`**，在更小的增益下即触发 λ 上调。
    3.  加阶梯 **`late_stage_ramp`**：如果最终轮次 mIoU 仍有追赶空间，提高 `end_lambda`。

### C. 泛化/对齐分支 (Generalization Tuning)
*   **触发条件**：`Val-mIoU` 与 `Test-mIoU` 差距拉大。
*   **调整逻辑**：
    1.  强化 **`late_stage_ramp`**：提前启动 Round 或增加斜率。
    2.  调整 **`uncertainty_only_rounds`**：若前期性能启动慢，尝试减少至 1 轮。

---

## 3. 自动化脚本实现伪代码

```python
def tune_experiment(results_path, current_config):
    # 1. 加载结果
    metrics = load_metrics(results_path)
    
    new_config = copy.deepcopy(current_config)
    
    # 2. 诊断
    if metrics['rollback_count'] >= 2:
        # 情况：系统不稳定
        new_config['agent_threshold_overrides']['LAMBDA_DELTA_UP'] *= 0.8
        new_config['agent_threshold_overrides']['OVERFIT_RISK_HI'] -= 0.1
        print("Diagnosis: Instability detected. Adjusting for stability.")
        
    elif metrics['alc'] < metrics['target_alc'] and metrics['rollback_count'] == 0:
        # 情况：系统过于保守
        new_config['agent_threshold_overrides']['LAMBDA_DELTA_UP'] += 0.05
        new_config['lambda_policy']['late_stage_ramp']['end_lambda'] += 0.05
        print("Diagnosis: Over-conservative. Boosting lambda progression.")
        
    # 3. 写入新的 ablation_config
    save_config(new_config, "full_model_A_tuned_v2")
    
    # 4. 触发 Shell 脚本重跑
    return "run_multiseed_ablation.sh --config full_model_A_tuned_v2"
```

## 4. 推荐调优实验路径 (The Learning Loop)

1.  **Baseline Exploration**：运行 `full_model_A_lambda_policy` 获得基准曲线。
2.  **Sensitivity Test**：固定 λ 观察 U 与 K 的贡献边界（单选 U 或单选 K）。
3.  **Closed-loop Tuning**：
    - 第一步：调优 `OVERFIT_RISK_HI` 找到回撤抑制的甜点区。
    - 第二步：调优 `LAMBDA_DELTA_UP` 寻找 ALC 增长最陡峭的步长。
    - 第三步：启用 `Ramp` 锁定最终性能上限。
