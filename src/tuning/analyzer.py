import os
import sys
import json
import re
import glob
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


RESULTS_DIR = "results"


@dataclass
class ExperimentData:
    trace: List[Dict] = field(default_factory=list)
    epoch_data: pd.DataFrame = None
    round_data: pd.DataFrame = None
    lambda_history: List[Dict] = field(default_factory=list)
    selection_history: List[Dict] = field(default_factory=list)
    exp_result: Dict = field(default_factory=dict)
    log_summary: Optional[Dict] = None


class ExperimentAnalyzer:
    def __init__(self, results_dir: str = RESULTS_DIR):
        self.results_dir = results_dir

    def load_experiment(self, run_id: str, exp_name: str) -> ExperimentData:
        trace = self._load_trace(run_id, exp_name)
        epoch_data = self._extract_epoch_data(trace) if trace else pd.DataFrame()
        round_data = self._extract_round_summaries(trace) if trace else pd.DataFrame()
        lambda_history = self._extract_lambda_history(trace) if trace else []
        selection_history = self._extract_selections(trace) if trace else []
        exp_result = self._load_exp_result(run_id, exp_name)
        log_summary = self._parse_log_summary(run_id, exp_name)
        return ExperimentData(
            trace=trace,
            epoch_data=epoch_data,
            round_data=round_data,
            lambda_history=lambda_history,
            selection_history=selection_history,
            exp_result=exp_result,
            log_summary=log_summary,
        )

    def _load_trace(self, run_id: str, exp_name: str) -> List[Dict]:
        trace_path = os.path.join(
            self.results_dir, "runs", run_id, f"{exp_name}_trace.jsonl"
        )
        if not os.path.exists(trace_path):
            return []
        events = []
        with open(trace_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events

    def _extract_epoch_data(self, trace: List[Dict]) -> pd.DataFrame:
        rows = []
        for ev in trace:
            if ev.get("type") != "epoch_end":
                continue
            grad = ev.get("grad", {})
            rows.append(
                {
                    "round": ev.get("round"),
                    "epoch": ev.get("epoch"),
                    "loss": ev.get("loss"),
                    "mIoU": ev.get("mIoU"),
                    "f1": ev.get("f1"),
                    "labeled_size": ev.get("labeled_size"),
                    "grad_total_norm_mean": grad.get("total_norm", {}).get("mean"),
                    "grad_total_norm_std": grad.get("total_norm", {}).get("std"),
                    "grad_train_val_cos": grad.get("train_val_cos"),
                    "grad_cos_consecutive_mean": grad.get("cos_consecutive", {}).get(
                        "mean"
                    ),
                }
            )
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _extract_round_summaries(self, trace: List[Dict]) -> pd.DataFrame:
        rows = []
        for ev in trace:
            if ev.get("type") != "round_summary":
                continue
            ts = ev.get("training_state", {})
            rows.append(
                {
                    "round": ev.get("round"),
                    "mIoU": ev.get("mIoU"),
                    "f1": ev.get("f1"),
                    "labeled_size": ev.get("labeled_size"),
                    "rollback_flag": ts.get("rollback_flag", False),
                    "overfit_risk": ts.get("overfit_risk", 0.0),
                    "miou_delta": ts.get("miou_delta", 0.0),
                    "lambda_effective": ev.get("sampler", {}).get("lambda_effective"),
                    "selected_epoch": ts.get("selected_epoch"),
                }
            )
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _extract_lambda_history(self, trace: List[Dict]) -> List[Dict]:
        return [ev for ev in trace if ev.get("type") == "lambda_policy_apply"]

    def _extract_selections(self, trace: List[Dict]) -> List[Dict]:
        return [ev for ev in trace if ev.get("type") == "l3_selection"]

    def _load_exp_result(self, run_id: str, exp_name: str) -> Dict:
        results_path = os.path.join(
            self.results_dir, "runs", run_id, "experiment_results.json"
        )
        if not os.path.exists(results_path):
            return {}
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(exp_name, {}) if isinstance(data, dict) else {}

    def _parse_log_summary(self, run_id: str, exp_name: str) -> Optional[Dict]:
        pattern = os.path.join(self.results_dir, "logs_md", f"{exp_name}_{run_id}.md")
        matches = glob.glob(pattern)
        if not matches:
            return None
        if len(matches) > 1:
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        with open(matches[0], "r", encoding="utf-8") as f:
            content = f.read()
        result = {"completed": "## 实验汇总" in content}
        miou_matches = re.findall(r"最终 mIoU: ([\d.]+)", content)
        if miou_matches:
            result["final_miou"] = float(miou_matches[-1])
        alc_matches = re.findall(r"ALC: ([\d.]+)", content)
        if alc_matches:
            result["alc"] = float(alc_matches[-1])
        return result

    def load_all_experiments(self, run_id: str) -> Dict[str, float]:
        results_path = os.path.join(
            self.results_dir, "runs", run_id, "experiment_results.json"
        )
        if not os.path.exists(results_path):
            return {}
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            exp: info.get("final_miou", 0)
            for exp, info in data.items()
            if isinstance(info, dict)
        }

    def compute_diagnostics(self, data: ExperimentData) -> Dict[str, Any]:
        diag = {}
        df = data.round_data
        if df.empty:
            return diag
        diag["final_miou"] = float(df.iloc[-1]["mIoU"])
        diag["peak_miou"] = float(df["mIoU"].max())
        diag["peak_round"] = int(df.loc[df["mIoU"].idxmax(), "round"])
        miou_arr = df["mIoU"].values
        if len(miou_arr) > 1:
            trend = np.polyfit(range(len(miou_arr)), miou_arr, 1)[0]
            diag["miou_trend"] = float(trend)
        else:
            diag["miou_trend"] = 0.0
        diag["rollback_count"] = int(df["rollback_flag"].sum())
        diag["rollback_rate"] = float(df["rollback_flag"].mean())
        max_consec = 0
        cur_consec = 0
        for flag in df["rollback_flag"].values:
            if flag:
                cur_consec += 1
                max_consec = max(max_consec, cur_consec)
            else:
                cur_consec = 0
        diag["max_consecutive_rollback"] = max_consec
        miou_std = float(df["mIoU"].std()) if len(df) > 1 else 0.0
        miou_mean = float(df["mIoU"].mean()) if len(df) > 0 else 1.0
        diag["miou_cv"] = miou_std / miou_mean if miou_mean > 0 else 0.0
        diag["lambda_trajectory"] = (
            df["lambda_effective"].dropna().tolist()
            if "lambda_effective" in df.columns
            else []
        )
        if diag["lambda_trajectory"]:
            diag["lambda_mean"] = float(np.mean(diag["lambda_trajectory"]))
            diag["lambda_final"] = float(diag["lambda_trajectory"][-1])
            diag["lambda_max_reached"] = float(np.max(diag["lambda_trajectory"]))
            lambdas = diag["lambda_trajectory"]
            if len(lambdas) > 1:
                diag["lambda_volatility"] = float(np.mean(np.abs(np.diff(lambdas))))
            else:
                diag["lambda_volatility"] = 0.0
        else:
            diag["lambda_mean"] = 0.0
            diag["lambda_final"] = 0.0
            diag["lambda_max_reached"] = 0.0
            diag["lambda_volatility"] = 0.0
        if len(df) >= 8:
            diag["warmup_exit_miou"] = (
                float(df.loc[df["round"] == 3, "mIoU"].values[0])
                if 3 in df["round"].values
                else float(df.iloc[0]["mIoU"])
            )
            diag["mid_stage_miou"] = (
                float(df.loc[df["round"] == 8, "mIoU"].values[0])
                if 8 in df["round"].values
                else float(df.iloc[-1]["mIoU"])
            )
            diag["late_stage_gain"] = float(
                df.iloc[-1]["mIoU"] - diag["mid_stage_miou"]
            )
        else:
            diag["warmup_exit_miou"] = (
                float(df.iloc[0]["mIoU"]) if not df.empty else 0.0
            )
            diag["mid_stage_miou"] = float(df.iloc[-1]["mIoU"]) if not df.empty else 0.0
            diag["late_stage_gain"] = 0.0
        edf = data.epoch_data
        if not edf.empty and "mIoU" in edf.columns:
            epoch_vol = edf.groupby("round")["mIoU"].std().mean()
            diag["epoch_miou_volatility"] = (
                float(epoch_vol) if not pd.isna(epoch_vol) else 0.0
            )
            best_epochs = []
            for rd, group in edf.groupby("round"):
                if "mIoU" in group.columns:
                    best_idx = group["mIoU"].idxmax()
                    best_epoch = (
                        group.loc[best_idx, "epoch"] if "epoch" in group.columns else 10
                    )
                    best_epochs.append(best_epoch)
            diag["avg_best_val_epoch"] = (
                float(np.mean(best_epochs)) if best_epochs else 10.0
            )
            if "grad_train_val_cos" in edf.columns:
                flip_rounds = 0
                total_rounds = 0
                for _, group in edf.groupby("round"):
                    tvc_vals = group["grad_train_val_cos"].dropna().values
                    if len(tvc_vals) >= 2:
                        total_rounds += 1
                        had_positive = False
                        flipped = False
                        for v in tvc_vals:
                            if v > 0:
                                had_positive = True
                            elif had_positive and v < 0:
                                flipped = True
                                break
                        if flipped:
                            flip_rounds += 1
                diag["tvc_sign_flip_rate"] = (
                    float(flip_rounds / total_rounds) if total_rounds > 0 else 0.0
                )
            else:
                diag["tvc_sign_flip_rate"] = 0.0
            if "loss" in edf.columns:
                ratios = []
                for _, group in edf.groupby("round"):
                    losses = group["loss"].dropna().values
                    if len(losses) >= 2 and losses[0] > 0:
                        ratios.append(float(losses[-1] / losses[0]))
                diag["avg_loss_convergence_ratio"] = (
                    float(np.mean(ratios)) if ratios else 1.0
                )
            else:
                diag["avg_loss_convergence_ratio"] = 1.0
        else:
            diag["epoch_miou_volatility"] = 0.0
            diag["avg_best_val_epoch"] = 10.0
            diag["tvc_sign_flip_rate"] = 0.0
            diag["avg_loss_convergence_ratio"] = 1.0
        if "overfit_risk" in df.columns:
            risk_hi = 1.2
            diag["severe_overfit_count"] = int((df["overfit_risk"] > risk_hi).sum())
        else:
            diag["severe_overfit_count"] = 0
        if not edf.empty and "grad_train_val_cos" in edf.columns:
            tvc_vals = edf["grad_train_val_cos"].dropna()
            diag["tvc_mean_avg"] = float(tvc_vals.mean()) if len(tvc_vals) > 0 else 0.0
            tvc_lasts = []
            for _, group in edf.groupby("round"):
                tvc_col = group["grad_train_val_cos"].dropna()
                if len(tvc_col) > 0:
                    tvc_lasts.append(float(tvc_col.iloc[-1]))
            diag["tvc_last_min"] = float(min(tvc_lasts)) if tvc_lasts else 0.0
        else:
            diag["tvc_mean_avg"] = 0.0
            diag["tvc_last_min"] = 0.0
        if data.exp_result:
            diag["alc"] = data.exp_result.get("alc")
            diag["fallback_count"] = len(data.exp_result.get("fallback_history", []))
            diag["budget_history"] = data.exp_result.get("budget_history", [])
        if data.log_summary:
            diag["experiment_completed"] = data.log_summary.get("completed", False)
            diag["test_miou"] = data.log_summary.get("test_miou")
            diag["test_f1"] = data.log_summary.get("test_f1")
        u_medians, k_medians = [], []
        for sel in data.selection_history:
            items = sel.get("top_items", [])
            if items:
                us = [
                    it.get("uncertainty", 0)
                    for it in items
                    if it.get("uncertainty") is not None
                ]
                ks = [
                    it.get("knowledge_gain", 0)
                    for it in items
                    if it.get("knowledge_gain") is not None
                ]
                if us:
                    u_medians.append(float(np.median(us)))
                if ks:
                    k_medians.append(float(np.median(ks)))
        if u_medians:
            diag["u_median_trajectory"] = u_medians
            diag["u_median_trend"] = (
                float(np.polyfit(range(len(u_medians)), u_medians, 1)[0])
                if len(u_medians) > 1
                else 0.0
            )
        else:
            diag["u_median_trajectory"] = []
            diag["u_median_trend"] = 0.0
        if k_medians:
            diag["k_median_trajectory"] = k_medians
            diag["k_median_trend"] = (
                float(np.polyfit(range(len(k_medians)), k_medians, 1)[0])
                if len(k_medians) > 1
                else 0.0
            )
        else:
            diag["k_median_trajectory"] = []
            diag["k_median_trend"] = 0.0
        return diag

    def diagnose(self, diagnostics: Dict) -> Dict:
        issues = []
        if diagnostics.get("rollback_rate", 0) > 0.15:
            issues.append(
                {
                    "type": "instability",
                    "severity": "high",
                    "evidence": f"rollback_rate={diagnostics['rollback_rate']:.2f}",
                    "suggestion": "降低 DELTA_UP, 增加 COOLING_ROUNDS",
                }
            )
        rollback_count = diagnostics.get("rollback_count", 0)
        lambda_max = diagnostics.get("lambda_max_reached", 0)
        if rollback_count == 0 and lambda_max < 0.4:
            issues.append(
                {
                    "type": "over_conservative",
                    "severity": "medium",
                    "evidence": f"lambda_max={lambda_max:.2f}, no rollbacks",
                    "suggestion": "增加 DELTA_UP, 降低 RISK_LO",
                }
            )
        if diagnostics.get("late_stage_gain", 0) < 0.01:
            issues.append(
                {
                    "type": "late_stage_plateau",
                    "severity": "high",
                    "evidence": f"late_gain={diagnostics.get('late_stage_gain', 0):.4f}",
                    "suggestion": "启用 late_stage_ramp, 提高 CLAMP_MAX",
                }
            )
        if diagnostics.get("lambda_volatility", 0) > 0.08:
            issues.append(
                {
                    "type": "lambda_oscillation",
                    "severity": "medium",
                    "evidence": f"volatility={diagnostics['lambda_volatility']:.3f}",
                    "suggestion": "降低 smoothing_alpha, 增加 COOLING_ROUNDS",
                }
            )
        avg_best = diagnostics.get("avg_best_val_epoch", 10)
        if avg_best < 4:
            issues.append(
                {
                    "type": "epochs_too_many",
                    "severity": "medium",
                    "evidence": f"avg_best_val_epoch={avg_best:.1f}",
                    "suggestion": "减少 epochs_per_round",
                }
            )
        if avg_best > 8.5:
            issues.append(
                {
                    "type": "epochs_insufficient",
                    "severity": "medium",
                    "evidence": f"avg_best_val_epoch={avg_best:.1f}",
                    "suggestion": "增加 epochs_per_round",
                }
            )
        if diagnostics.get("u_median_trend", 0) < -0.03:
            issues.append(
                {
                    "type": "exploration_degradation",
                    "severity": "medium",
                    "evidence": f"u_median_trend={diagnostics['u_median_trend']:.4f}",
                    "suggestion": "启用 guardrail 或降低 CLAMP_MAX",
                }
            )
        if diagnostics.get("tvc_sign_flip_rate", 0) > 0.3:
            issues.append(
                {
                    "type": "intra_round_overfit",
                    "severity": "high",
                    "evidence": f"tvc_sign_flip_rate={diagnostics['tvc_sign_flip_rate']:.2f}",
                    "suggestion": "减少 epochs 或降低 LR",
                }
            )
        if (
            diagnostics.get("miou_cv", 0) > 0.1
            and diagnostics.get("rollback_rate", 0) <= 0.15
        ):
            issues.append(
                {
                    "type": "instability",
                    "severity": "high",
                    "evidence": f"miou_cv={diagnostics['miou_cv']:.3f}",
                    "suggestion": "降低 DELTA_UP, 增加 COOLING_ROUNDS",
                }
            )
        if diagnostics.get("fallback_count", 0) > 3:
            issues.append(
                {
                    "type": "agent_unreliable",
                    "severity": "low",
                    "evidence": f"fallback_count={diagnostics['fallback_count']}",
                    "suggestion": "检查 LLM 配置或考虑纯 policy 模式",
                }
            )
        return {"issues": issues, "diagnostics": diagnostics}
