import os
import json
import glob
import re
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict

RESULTS_DIR = "/Users/anykong/AD-KUCS/AAL_SD/results/runs"

params_to_track = [
    "agent_threshold_overrides.LAMBDA_DELTA_UP",
    "agent_threshold_overrides.LAMBDA_DELTA_DOWN",
    "lambda_policy.lambda_smoothing_alpha",
    "lambda_policy.lambda_max_step",
    "lambda_policy.risk_ci_window",
    "lambda_policy.risk_ci_quantile",
    "lambda_policy.late_stage_ramp.start_lambda",
    "lambda_policy.late_stage_ramp.end_lambda",
    "lambda_policy.selection_guardrail.guardrail_start_round",
    "lambda_policy.selection_guardrail.u_median_min",
    "lambda_policy.selection_guardrail.u_low_thresh",
    "lambda_policy.selection_guardrail.lambda_step_down"
]

def get_nested(d, path):
    keys = path.split('.')
    current = d
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return None
    return current

data = []

# Process all manifest.json in runs directories
for run_dir in glob.glob(os.path.join(RESULTS_DIR, "*")):
    manifest_path = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        continue
    
    with open(manifest_path, 'r') as f:
        try:
            manifest = json.load(f)
        except Exception as e:
            print(f"Error loading {manifest_path}: {e}")
            continue
            
    experiments = manifest.get("experiments", {})
    if isinstance(experiments, list):
        # old manifest format
        continue

    for exp_name, exp_cfg in experiments.items():
        # find the status file or md file to get mIoU
        # status file is like auto_opt_iter09_cand0_00_status.json
        status_path = os.path.join(run_dir, f"{exp_name}_status.json")
        miou = None
        if os.path.exists(status_path):
            with open(status_path, 'r') as f:
                try:
                    status = json.load(f)
                    if "result" in status and "final_mIoU" in status["result"]:
                        miou = status["result"]["final_mIoU"]
                except:
                    pass
        
        # If no status file, maybe check md file
        if miou is None:
            md_path = os.path.join(run_dir, f"{exp_name}.md")
            if os.path.exists(md_path):
                with open(md_path, 'r') as f:
                    text = f.read()
                    matches = re.findall(r'final_mIoU[\s\*\:]+([0-9\.]+)', text)
                    if matches:
                        miou = float(matches[-1])
                    else:
                        matches = re.findall(r'mIoU:\s*([0-9\.]+)', text)
                        if matches:
                            miou = float(matches[-1])
        
        if miou is not None:
            row = {'exp_name': exp_name, 'run_dir': os.path.basename(run_dir), 'miou': miou}
            for p in params_to_track:
                val = get_nested(exp_cfg, p)
                row[p] = val
            data.append(row)

df = pd.DataFrame(data)

# Print Top vs Bottom analysis
print("\n=== Top 20% vs Bottom 20% ===")
if len(df) > 5:
    top_20 = df.nlargest(int(len(df)*0.2), 'miou')
    bot_20 = df.nsmallest(int(len(df)*0.2), 'miou')
    for p in params_to_track:
        if p in df.columns:
            t_val = top_20[p].dropna().mean()
            b_val = bot_20[p].dropna().mean()
            print(f"{p:60s}: Top={t_val:7.4f}, Bot={b_val:7.4f}, Diff={t_val-b_val:7.4f}")

df.to_csv("param_analysis.csv", index=False)

# Analyze correlations
print(f"Total experiments found with mIoU: {len(df)}")
print("\nCorrelation with mIoU (Spearman):")
correlations = []
for p in params_to_track:
    valid_data = df.dropna(subset=[p, 'miou'])
    if len(valid_data) > 2:
        # Check if values are constant
        if valid_data[p].nunique() > 1:
            corr, pval = spearmanr(valid_data[p], valid_data['miou'])
            correlations.append((p, corr, pval, len(valid_data)))
        else:
            correlations.append((p, 0.0, 1.0, len(valid_data)))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)
for p, corr, pval, n in correlations:
    print(f"{p:60s}: r={corr:7.4f} (p={pval:7.4f}, n={n})")

# Feature importance using Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

X = df[params_to_track]
y = df['miou']
X = df[params_to_track].fillna(df[params_to_track].median())
X = X.fillna(0) # In case all are NaN

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

print("\nRandom Forest Feature Importance:")
importances = list(zip(params_to_track, rf.feature_importances_))
importances.sort(key=lambda x: x[1], reverse=True)
for p, imp in importances:
    print(f"{p:60s}: {imp:7.4f}")
