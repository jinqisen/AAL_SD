# Fullmodel 性能优化方案（独立实验，可落代码版）

## 目标与约束
- 目标：在不改动既有 full_model 与消融的前提下，通过独立实验提升最终 mIoU。
- 约束：遵循 AAL-SD registry 映射模式，配置可复现、实验可消融。
- 范围：仅新增实验与策略模块，不影响已有实验的默认行为。

## 设计原则
- 机制升级而非算法拼凑：优化信号估计、风险判定与闭环控制。
- 配置驱动：以 ablation_config registry 注册新实验。
- 风险隔离：新逻辑仅在新实验配置启用时生效。

## 方案总览
本方案在 AD-KUCS 框架内引入三项机制升级：
1. 不确定性校准（U-Calibration）
2. 风险置信区间触发（Risk-CI）
3. λ 调度平滑与后期偏 U 约束（Lambda-Smoothing + Late-U Bias）

## 新实验命名
- 实验名：full_model_v5_calibrated_risk
- 目标：保持 full_model 管线与采样器类型不变，仅增强内部策略机制。

## 代码改动清单
- src/experiments/ablation_config.py
  - 新增 full_model_v5_calibrated_risk 配置条目。
- src/agent/toolbox.py
  - 仅新增可选策略入口与配置解析，不改现有 full_model 行为。
- src/agent/policies/lambda_policy.py
  - 新增风险置信区间触发与 λ 平滑策略实现。
- src/agent/policies/uncertainty_calibration.py
  - 新增 U 校准实现（温度或分位数）。
- src/core/sampler.py
  - 在分数融合前对 U 进行校准（仅当配置启用）。

## ablation_config 示例配置（可直接落地）
```python
"full_model_v5_calibrated_risk": {
    "description": "完整模型V5：U校准 + 风险CI触发 + λ平滑与后期偏U约束",
    "use_agent": True,
    "sampler_type": "ad_kucs",
    "lambda_override": None,
    "enable_l3_logging": True,
    "l3_topk": 256,
    "l3_max_selected": 256,
    "lambda_policy": {
        "mode": "warmup_risk_closed_loop",
        "r1_lambda": 0.0,
        "uncertainty_only_rounds": 2,
        "warmup_start_round": 3,
        "warmup_rounds": 1,
        "warmup_lambda": 0.2,
        "risk_control_start_round": 4,
        "risk_trigger": "ci",
        "risk_ci_window": 3,
        "risk_ci_quantile": 0.1,
        "lambda_smoothing": "ema",
        "lambda_smoothing_alpha": 0.7,
        "lambda_max_step": 0.05,
        "late_u_bias_start_round": 10,
        "late_u_bias_strength": 0.3
    },
    "uncertainty_calibration": {
        "mode": "temperature",
        "tau": 0.7,
        "update_rounds": [6, 9, 12]
    },
    "risk_policy": {
        "trigger": "ci",
        "window": 3,
        "quantile": 0.1,
        "min_samples": 3
    },
    "rollback_config": {
        "mode": "adaptive_threshold",
        "std_factor": 1.5,
        "tau_min": 0.005
    },
    "overfit_guard": True,
    "control_permissions": {
        "set_lambda": False,
        "set_query_size": False,
        "set_epochs_per_round": False,
        "set_alpha": False
    }
}
```

## 策略模块接口设计（可落代码）
### 不确定性校准接口
```python
class UncertaintyCalibrator:
    def __init__(self, mode: str, tau: float, quantile_low: float, quantile_high: float):
        ...

    def fit(self, u_scores: np.ndarray) -> None:
        ...

    def transform(self, u_scores: np.ndarray) -> np.ndarray:
        ...
```

### 风险 CI 触发接口
```python
class RiskTrigger:
    def __init__(self, window: int, quantile: float, min_samples: int):
        ...

    def evaluate(self, series: List[float]) -> bool:
        ...
```

### λ 平滑调度接口
```python
class LambdaScheduler:
    def __init__(self, smoothing: str, alpha: float, max_step: float):
        ...

    def update(self, lambda_prev: float, lambda_candidate: float) -> float:
        ...
```

## 运行与实验设计
- 对照组：full_model、baseline_bald、full_model_v4_robust_severe_last。
- 预算一致：TOTAL_BUDGET、QUERY_SIZE、EPOCHS_PER_ROUND 不变。
- 多 seed 复现：使用 MULTI_SEED_DEFAULTS 或 run_multi_seed。
- 评估指标：最终 mIoU、ALC、R10–R15 同标注量曲线。

## 验证与回滚
- 若 final mIoU 未提升或波动增大，可禁用：
  - uncertainty_calibration
  - risk_policy
  - lambda_smoothing
- 保持原 full_model 路径不变，回滚仅需移除实验条目。

## 架构一致性检查
- registry 映射：新增实验名接入 ABLATION_SETTINGS，路径一致。
- 依赖方向：experiments → agent → core 不反转，无循环依赖。
- 可维护性：策略模块独立，Toolbox 仅负责编排与权限门控。
