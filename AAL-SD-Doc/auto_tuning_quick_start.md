# AAL-SD Auto-Tuning Quick Start Guide

## Prerequisites

1. Ensure you have a completed baseline experiment to start from
2. Set up LLM API key (if using LLM advisor):
   ```bash
   export TUNING_LLM_API_KEY="your-api-key-here"
   export TUNING_LLM_BASE_URL="https://code.ppchat.vip/v1"  # optional
   export TUNING_LLM_MODEL="claude-opus-4-6"  # optional
   ```

## Basic Usage

### Start Auto-Tuning (with LLM Advisor)

```bash
cd /Users/anykong/AD-KUCS/AAL_SD

python src/tuning/orchestrator.py \
    --initial-run-id baseline_20260309_211601_seed43 \
    --initial-exp full_model_A_lambda_policy_ab_tune_hi_ep10 \
    --target-miou 0.74 \
    --max-iterations 10 \
    --seeds 43 \
    --max-concurrent 2
```

### Start Auto-Tuning (Pure Rule Mode, No LLM)

```bash
python src/tuning/orchestrator.py \
    --initial-run-id baseline_20260309_211601_seed43 \
    --initial-exp full_model_A_lambda_policy_ab_tune_hi_ep10 \
    --target-miou 0.74 \
    --max-iterations 10 \
    --no-llm
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--initial-run-id` | Source run_id to start from | **Required** |
| `--initial-exp` | Source experiment name | **Required** |
| `--target-miou` | Target mIoU to reach | 0.74 |
| `--max-iterations` | Max tuning iterations | 10 |
| `--seeds` | Seeds for Phase A screening | [43] |
| `--max-concurrent` | Max concurrent experiments | 2 |
| `--llm-config` | Path to LLM config JSON | `src/tuning_llm_config.json` |
| `--no-llm` | Disable LLM advisor (pure rule mode) | False |
| `--results-dir` | Results directory | `results` |
| `--repo-dir` | Repository root | `.` |
| `--run-id-prefix` | Prefix for generated run IDs | `autotune` |

## What Happens During Auto-Tuning

Each iteration:

1. **Analyze** — Load trace data from best experiment, compute ~30 diagnostics, run rule-based diagnosis
2. **Propose** — Generate 3-4 parameter adjustment proposals (LLM-guided or rule-based)
3. **Classify** — Decide branch vs full-run per proposal based on which params changed
4. **Prepare** — Use PoolResumeManager to prepare pools/checkpoints for branch experiments
5. **Execute** — Run experiments via `run_parallel_strict.py` (Phase A: single seed screening)
6. **Collect** — Load results, compute diagnostics, find best experiment
7. **Converge** — Check if target reached, plateau detected, or max iterations hit
8. **Repeat** — Continue to next iteration with updated diagnosis

## Branch vs Full-Run Strategy

The orchestrator automatically decides:

- **Branch from round 7** — Only late-stage params changed (late_stage_ramp, selection_guardrail, epochs_per_round_override, LAMBDA_CLAMP_MAX)
- **Branch from round 4** — Only risk control params changed (OVERFIT_RISK_HI, LAMBDA_DELTA_UP, etc.)
- **Full run** — Early-stage params changed (uncertainty_only_rounds, warmup_*) or mixed changes

## Output Structure

```
results/
├── runs/
│   ├── autotune_iter000_20260313_1234/
│   │   ├── auto_tune_iter00_ramp_on_best_00_trace.jsonl
│   │   ├── auto_tune_iter00_high_clamp_01_trace.jsonl
│   │   ├── experiment_results.json
│   │   └── manifest.json
│   └── autotune_iter001_20260313_1456/
│       └── ...
├── checkpoints/
│   └── autotune_iter000_20260313_1234/
│       └── auto_tune_iter00_ramp_on_best_00_state.json
└── pools/
    └── autotune_iter000_20260313_1234/
        └── auto_tune_iter00_ramp_on_best_00/
            ├── labeled_pool.csv
            └── unlabeled_pool.csv

AAL-SD-Doc/
└── tuning_reports/
    ├── iter_000_analysis.json
    ├── iter_001_analysis.json
    └── ...

src/experiments/
└── auto_tune_configs.json  # Generated ablation configs

results/tuning_llm_logs/  # LLM request/response logs (if enabled)
└── llm_call_20260313_123456.json
```

## Convergence Conditions

Auto-tuning stops when:

1. **Target reached** — `best_miou >= target_miou`
2. **Plateau** — No improvement (< 0.002) for 3 consecutive iterations
3. **Max iterations** — Reached `max_iterations`
4. **Diminishing returns** (warning only) — Last 3 iterations avg improvement < 0.001

## Troubleshooting

### LLM Advisor Not Working

Check:
1. `TUNING_LLM_API_KEY` environment variable is set
2. API endpoint is reachable
3. Check logs in `results/tuning_llm_logs/`
4. Fallback: Use `--no-llm` for pure rule mode

### Experiments Failing

Check:
1. Source run_id and experiment exist in `results/runs/`
2. Pools and checkpoints are present
3. Check experiment logs in `results/runs/<run_id>/<exp>.md`
4. Verify `run_parallel_strict.py` works standalone

### Branch Experiments Not Resuming

Check:
1. Source experiment has valid checkpoint in `results/checkpoints/`
2. Pools exist in `results/pools/<source_run_id>/<source_exp>/`
3. `performance_history` in checkpoint has data for the branch round

### Memory Issues

Reduce `--max-concurrent` to 1 or adjust `per_experiment_gb` in orchestrator config.

## Advanced Configuration

Create `src/tuning_llm_config.json`:

```json
{
  "base_url": "https://code.ppchat.vip/v1",
  "api_key": "your-key-here",
  "model": "claude-opus-4-6",
  "temperature": 0.3,
  "timeout": 120,
  "max_retries": 3,
  "retry_delay": 5.0,
  "retry_backoff": 2.0,
  "thinking_budget": 16000,
  "log_requests": true,
  "log_dir": "results/tuning_llm_logs"
}
```

## Monitoring Progress

Watch the orchestrator output for:
- Iteration number and current best mIoU
- Proposed experiments and their directions
- Branch vs full-run decisions
- Experiment execution status
- Convergence warnings

Check Git branches:
```bash
git branch | grep tuning/iter
```

Each iteration creates a branch with the analysis report committed.

## Example Session

```bash
# Set up environment
export TUNING_LLM_API_KEY="sk-..."

# Start tuning
python src/tuning/orchestrator.py \
    --initial-run-id baseline_20260309_211601_seed43 \
    --initial-exp full_model_A_lambda_policy_ab_tune_hi_ep10 \
    --target-miou 0.74 \
    --max-iterations 5 \
    --seeds 43

# Output:
# === Tuning Iteration 0 | Best mIoU: 0.7221 | Target: 0.74 ===
# [sidecar] Wrote 4 configs to src/experiments/auto_tune_configs.json
# [pool] Branched auto_tune_iter00_ramp_on_best_00 from baseline/.../round 7
# [exec] Running 4 experiments (workers=2, mode=resume): [...]
# ...
# === Tuning Iteration 1 | Best mIoU: 0.7289 | Target: 0.74 ===
# ...
# STOP: target_reached
# 
# === Tuning Complete ===
# Iterations: 2
# Best mIoU:  0.7412
# Target reached: True
```

## Next Steps

After auto-tuning completes:

1. Review the best configuration in the final iteration's `experiment_results.json`
2. Run multi-seed validation on the best config
3. Analyze the tuning trajectory in `AAL-SD-Doc/tuning_reports/`
4. Merge the best config into `ablation_config.py` if desired

## Reference

- Design document: `AAL-SD-Doc/auto_tuning_framework_design.md`
- Tuning logic framework: `AAL-SD-Doc/tuning_logic_framework.md`
- Implementation: `src/tuning/`
