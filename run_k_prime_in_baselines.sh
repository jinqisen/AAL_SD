#!/usr/bin/env bash
cd /Users/anykong/AD-KUCS/AAL_SD

for seed in 42 43 44 45 46; do
  python src/main.py --run_id baselines_only_p3_20260313_212023_seed${seed} --experiment_name full_model_A_lambda_policy --seed "${seed}" --start resume --n_rounds 16 &
done

for seed in 42 43 44 45 46; do
  python src/main.py --run_id baselines_only_p3_20260313_212023_seed${seed} --experiment_name full_model_B_lambda_agent --seed "${seed}" --start resume --n_rounds 16 &
done

wait

python src/experiments/run_multi_seed.py --results_dir results --run_id baselines_only_p3_20260313_212023 --seeds "42,43,44,45,46" --start resume --experiments full_model_A_lambda_policy full_model_B_lambda_agent baseline_random baseline_entropy baseline_coreset baseline_bald baseline_dial_style baseline_wang_style

python src/analysis/plot_paper_figures.py --multi_seed_group_dir results/runs/baselines_only_p3_20260313_212023 --output_dir AAL-SD-Doc/figures

