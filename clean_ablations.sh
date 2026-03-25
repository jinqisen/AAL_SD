#!/usr/bin/env bash
for seed in 42 43 44 45 46; do
  DIR="/Users/anykong/AD-KUCS/AAL_SD/results/runs/baselines_only_p3_20260313_212023_seed${seed}"
  if [ -d "$DIR" ]; then
    echo "Cleaning up non-baseline experiments in $DIR..."
    
    find "$DIR" -maxdepth 1 -type f -name "*_status.json" ! -name "baseline_*" ! -name "full_model_A_lambda_policy_status.json" -exec rm -f {} +
    find "$DIR" -maxdepth 1 -type f -name "*_trace.jsonl" ! -name "baseline_*" ! -name "full_model_A_lambda_policy_trace.jsonl" -exec rm -f {} +
    find "$DIR" -maxdepth 1 -type f -name "result_*.json" ! -name "result_baseline_*" ! -name "result_full_model_A_lambda_policy.json" -exec rm -f {} +
    
    rm -f "$DIR/full_model_A_lambda_policy_status.json"
    rm -f "$DIR/full_model_A_lambda_policy_trace.jsonl"
    rm -f "$DIR/result_full_model_A_lambda_policy.json"
    
    find "$DIR" -maxdepth 1 -type d -name "*_round_models" ! -name "baseline_*_round_models" -exec rm -rf {} +
    
    if [ -d "$DIR/reports" ]; then
        find "$DIR/reports" -type f ! -name "baseline_*" -exec rm -f {} +
    fi
  fi
done

DIR="/Users/anykong/AD-KUCS/AAL_SD/results/runs/baselines_only_p3_20260313_212023"
rm -f "$DIR/multi_seed_summary.json"
rm -f "$DIR/multi_seed_report.md"

echo "Cleanup complete."

