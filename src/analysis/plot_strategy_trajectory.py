import os
import re
import json
import argparse
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

def _parse_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _extract_distribution_hints(decision_reason: Optional[str]) -> Dict[str, Optional[float]]:
    text = decision_reason or ""

    def _find(pattern: str) -> Optional[float]:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            return None
        return _parse_float(m.group(1))

    return {
        "u_mean": _find(r"U\s*(?:分布)?\s*(?:均值|mean)\s*[^0-9\-]*\(?\s*([0-9]*\.?[0-9]+)\s*\)?"),
        "u_p75": _find(r"U\s*(?:分布)?\s*75\s*(?:分位|percentile)\s*[^0-9\-]*\(?\s*([0-9]*\.?[0-9]+)\s*\)?"),
        "k_mean": _find(r"K\s*(?:分布)?\s*(?:均值|mean)\s*[^0-9\-]*\(?\s*([0-9]*\.?[0-9]+)\s*\)?"),
        "k_p75": _find(r"K\s*(?:分布)?\s*75\s*(?:分位|percentile)\s*[^0-9\-]*\(?\s*([0-9]*\.?[0-9]+)\s*\)?"),
    }


def load_trace(trace_path: str) -> pd.DataFrame:
    if not os.path.exists(trace_path):
        raise FileNotFoundError(trace_path)

    per_round: Dict[int, Dict[str, Any]] = {}

    def _summarize(xs: List[float]) -> Dict[str, Optional[float]]:
        vals = [float(v) for v in xs if v is not None and math.isfinite(float(v))]
        if not vals:
            return {"mean": None, "p50": None, "p75": None}
        arr = np.array(vals, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "p50": float(np.quantile(arr, 0.5)),
            "p75": float(np.quantile(arr, 0.75)),
        }

    def _extract_values(items: Any, key: str, limit: Optional[int] = None) -> List[float]:
        if not isinstance(items, list) or not items:
            return []
        if limit is not None:
            limit = int(limit)
        out: List[float] = []
        seq = items[:limit] if limit is not None and limit > 0 else items
        for it in seq:
            if not isinstance(it, dict):
                raise TypeError(f"Expected dict item for key={key}, got {type(it).__name__}")
            v = it.get(key)
            if v is None:
                continue
            fv = float(v)
            if math.isfinite(fv):
                out.append(fv)
        return out

    def _row(r: Any) -> Dict[str, Any]:
        rr = int(r)
        if rr not in per_round:
            per_round[rr] = {"round": rr}
        return per_round[rr]

    with open(trace_path, "r") as f:
        for line_no, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSONL at {trace_path}:{line_no}")

            etype = entry.get("type")

            if etype == "epoch_end":
                r = entry.get("round")
                row = _row(r)
                row["final_epoch"] = entry.get("epoch")
                row["final_miou"] = entry.get("mIoU")
                row["final_f1"] = entry.get("f1")
                row["labeled_size"] = entry.get("labeled_size")
                grad = entry.get("grad") if isinstance(entry.get("grad"), dict) else {}
                row["grad_train_val_cos_last"] = grad.get("train_val_cos")

            elif etype == "selection":
                r = entry.get("round")
                row = _row(r)
                context = entry.get("context", {}) if isinstance(entry.get("context", {}), dict) else {}
                decision_reason = context.get("decision_reason")
                hints = _extract_distribution_hints(decision_reason)
                row.update(
                    {
                        "selected_count": entry.get("selected"),
                        "sampler_type": entry.get("sampler_type") or context.get("sampler") or "unknown",
                        "sampler_class": entry.get("sampler_class"),
                        "k_definition": entry.get("k_definition"),
                        "decision_reason": decision_reason,
                        "selected_ids": entry.get("selected_ids"),
                        "u_mean": hints.get("u_mean"),
                        "u_p75": hints.get("u_p75"),
                        "k_mean": hints.get("k_mean"),
                        "k_p75": hints.get("k_p75"),
                    }
                )

            elif etype == "controller_step":
                r = entry.get("round")
                row = _row(r)
                action = entry.get("action") if isinstance(entry.get("action"), dict) else {}
                state = entry.get("state") if isinstance(entry.get("state"), dict) else {}
                row.update(
                    {
                        "lambda_t": action.get("lambda"),
                        "epochs": action.get("epochs"),
                        "query_size": action.get("query_size"),
                        "rollback_flag": state.get("rollback_flag"),
                        "miou_delta": state.get("miou_delta"),
                        "last_miou": state.get("last_miou"),
                    }
                )

            elif etype == "lambda_override":
                r = entry.get("round")
                row = _row(r)
                row["lambda_override"] = entry.get("applied")
                row["lambda_override_suggested"] = entry.get("suggested_lambda")
                row["lambda_override_exceeds_suggested_range"] = entry.get(
                    "exceeds_suggested_range"
                )
                row["lambda_override_clamped"] = entry.get("clamped")
                row["lambda_override_clamp_reason"] = entry.get("clamp_reason")

            elif etype == "lambda_policy_apply":
                r = entry.get("round")
                row = _row(r)
                row["lambda_policy_apply"] = entry.get("applied")
                row["lambda_policy_rule"] = entry.get("rule")
                row["lambda_policy_base"] = entry.get("base")

            elif etype == "overfit_signal":
                r = entry.get("round")
                row = _row(r)
                row.update(
                    {
                        "grad_tvc_mean": entry.get("grad_train_val_cos_mean"),
                        "grad_tvc_min": entry.get("grad_train_val_cos_min"),
                        "grad_tvc_max": entry.get("grad_train_val_cos_max"),
                        "grad_tvc_last": entry.get("grad_train_val_cos_last"),
                        "grad_tvc_neg_rate": entry.get("grad_train_val_cos_neg_rate"),
                        "overfit_risk": entry.get("overfit_risk"),
                    }
                )

            elif etype == "lambda_guard":
                r = entry.get("round")
                row = _row(r)
                row["lambda_guard_cap"] = entry.get("cap")
                row["lambda_guard_before"] = entry.get("lambda_before")
                row["lambda_guard_after"] = entry.get("lambda_after")

            elif etype == "round_summary":
                r = entry.get("round")
                row = _row(r)
                selection = entry.get("selection") if isinstance(entry.get("selection"), dict) else {}
                ranking = entry.get("ranking") if isinstance(entry.get("ranking"), dict) else {}
                lambda_ctrl = entry.get("lambda_controller") if isinstance(entry.get("lambda_controller"), dict) else {}
                training_state = entry.get("training_state") if isinstance(entry.get("training_state"), dict) else {}
                row.update(
                    {
                        "final_miou": entry.get("mIoU"),
                        "final_f1": entry.get("f1"),
                        "labeled_size": entry.get("labeled_size"),
                        "selected_count": selection.get("selected"),
                        "sampler_type": entry.get("sampler", {}).get("sampler_type") if isinstance(entry.get("sampler"), dict) else None,
                        "sampler_class": entry.get("sampler", {}).get("sampler_class") if isinstance(entry.get("sampler"), dict) else None,
                        "k_definition": entry.get("sampler", {}).get("k_definition") if isinstance(entry.get("sampler"), dict) else None,
                        "avg_uncertainty": ranking.get("avg_uncertainty"),
                        "avg_knowledge_gain": ranking.get("avg_knowledge_gain"),
                        "lambda_effective": ranking.get("lambda_effective"),
                        "lambda_source": ranking.get("lambda_source"),
                        "lambda_controller": lambda_ctrl.get("lambda"),
                        "lambda_controller_mode": lambda_ctrl.get("mode"),
                        "rollback_flag": training_state.get("rollback_flag"),
                        "miou_delta": training_state.get("miou_delta"),
                    }
                )

            elif etype == "l3_selection":
                r = entry.get("round")
                row = _row(r)
                top_items = entry.get("top_items") if isinstance(entry.get("top_items"), list) else []
                selected_items = entry.get("selected_items") if isinstance(entry.get("selected_items"), list) else []
                row["l3_source"] = entry.get("source")
                row["l3_topk"] = entry.get("topk")
                row["l3_selected_limit"] = entry.get("selected_limit")
                row["l3_top_items"] = top_items
                row["l3_selected_items"] = selected_items

                sel_n = row.get("selected_count")
                sel_n_i = int(sel_n) if sel_n is not None else None

                sel_u = _extract_values(selected_items, "uncertainty")
                sel_k = _extract_values(selected_items, "knowledge_gain")
                sel_s = _extract_values(selected_items, "final_score")
                top_u = _extract_values(top_items, "uncertainty", limit=sel_n_i)
                top_k = _extract_values(top_items, "knowledge_gain", limit=sel_n_i)
                top_s = _extract_values(top_items, "final_score", limit=sel_n_i)

                su = _summarize(sel_u)
                sk = _summarize(sel_k)
                ss = _summarize(sel_s)
                tu = _summarize(top_u)
                tk = _summarize(top_k)
                ts = _summarize(top_s)

                row.update(
                    {
                        "l3_u_mean_selected": su.get("mean"),
                        "l3_u_p75_selected": su.get("p75"),
                        "l3_k_mean_selected": sk.get("mean"),
                        "l3_k_p75_selected": sk.get("p75"),
                        "l3_score_mean_selected": ss.get("mean"),
                        "l3_u_mean_topn": tu.get("mean"),
                        "l3_k_mean_topn": tk.get("mean"),
                        "l3_score_mean_topn": ts.get("mean"),
                    }
                )

    df = pd.DataFrame(list(per_round.values()))
    if df.empty:
        return df
    df = df.sort_values("round").reset_index(drop=True)

    if "selected_ids" in df.columns:
        df["selected_ids_count"] = df["selected_ids"].apply(lambda x: len(x) if isinstance(x, list) else None)

    for col in ["avg_uncertainty", "avg_knowledge_gain", "u_mean", "k_mean", "u_p75", "k_p75"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "l3_u_mean_topn" in df.columns:
        df["avg_uncertainty"] = df["avg_uncertainty"].fillna(pd.to_numeric(df["l3_u_mean_topn"], errors="coerce"))
    if "l3_k_mean_topn" in df.columns:
        df["avg_knowledge_gain"] = df["avg_knowledge_gain"].fillna(pd.to_numeric(df["l3_k_mean_topn"], errors="coerce"))
    if "l3_u_mean_selected" in df.columns:
        df["u_mean"] = df["u_mean"].fillna(pd.to_numeric(df["l3_u_mean_selected"], errors="coerce"))
    if "l3_k_mean_selected" in df.columns:
        df["k_mean"] = df["k_mean"].fillna(pd.to_numeric(df["l3_k_mean_selected"], errors="coerce"))
    if "l3_u_p75_selected" in df.columns:
        df["u_p75"] = df["u_p75"].fillna(pd.to_numeric(df["l3_u_p75_selected"], errors="coerce"))
    if "l3_k_p75_selected" in df.columns:
        df["k_p75"] = df["k_p75"].fillna(pd.to_numeric(df["l3_k_p75_selected"], errors="coerce"))

    return df

def load_trace_meta(trace_path: str) -> Dict[str, Any]:
    if not os.path.exists(trace_path):
        raise FileNotFoundError(trace_path)
    with open(trace_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            if entry.get("type") != "initialized":
                continue
            ab = entry.get("ablation") if isinstance(entry.get("ablation"), dict) else {}
            return {
                "lambda_policy": ab.get("lambda_policy") if isinstance(ab.get("lambda_policy"), dict) else None,
                "agent_threshold_overrides": ab.get("agent_threshold_overrides") if isinstance(ab.get("agent_threshold_overrides"), dict) else {},
            }
    return {"lambda_policy": None, "agent_threshold_overrides": {}}

def _tvc_series_for_policy_key(df: pd.DataFrame, tvc_key: str) -> pd.Series:
    mapping = {
        "grad_train_val_cos_last": "grad_tvc_last",
        "grad_train_val_cos_min": "grad_tvc_min",
        "grad_train_val_cos_mean": "grad_tvc_mean",
        "grad_train_val_cos_neg_rate": "grad_tvc_neg_rate",
    }
    col = mapping.get(str(tvc_key), None)
    if col and col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").astype(float)
    if "grad_tvc_last" in df.columns:
        return pd.to_numeric(df["grad_tvc_last"], errors="coerce").astype(float)
    return pd.Series([float("nan")] * len(df))

def add_lambda_policy_diagnostics(per_round_df: pd.DataFrame, trace_meta: Dict[str, Any]) -> pd.DataFrame:
    if per_round_df.empty:
        return per_round_df
    policy = trace_meta.get("lambda_policy") if isinstance(trace_meta, dict) else None
    if not isinstance(policy, dict) or not policy:
        return per_round_df

    df = per_round_df.copy()
    overrides = trace_meta.get("agent_threshold_overrides") if isinstance(trace_meta, dict) else {}
    if not isinstance(overrides, dict):
        overrides = {}

    q = float(policy.get("risk_ci_quantile", 0.2))
    window = int(policy.get("risk_ci_window", 6))
    min_samples = int(policy.get("risk_ci_min_samples", 3))
    severe_logic = str(policy.get("severe_logic", "or")).strip().lower()
    tvc_key = str(policy.get("severe_tvc_key", "grad_train_val_cos_last"))

    alpha = float(overrides.get("OVERFIT_RISK_EMA_ALPHA", 1.0))
    alpha = float(min(max(alpha, 0.0), 1.0))
    cooling = int(overrides.get("LAMBDA_DOWN_COOLING_ROUNDS", 0) or 0)

    df["policy_risk_ci_quantile"] = q
    df["policy_risk_ci_window"] = window
    df["policy_risk_ci_min_samples"] = min_samples
    df["policy_severe_logic"] = severe_logic
    df["policy_severe_tvc_key"] = tvc_key
    df["policy_overfit_risk_ema_alpha"] = alpha
    df["policy_lambda_down_cooling_rounds"] = cooling

    risk_raw = pd.to_numeric(df.get("overfit_risk", pd.Series([float("nan")] * len(df))), errors="coerce").astype(float)
    tvc = _tvc_series_for_policy_key(df, tvc_key)
    rule = df.get("lambda_policy_rule", pd.Series([None] * len(df)))

    risk_hist: List[float] = []
    tvc_hist: List[float] = []
    ema: Optional[float] = None
    last_down_round: Optional[int] = None

    rounds = pd.to_numeric(df.get("round"), errors="coerce").astype("Int64")

    out_ema: List[Optional[float]] = []
    out_risk_th: List[Optional[float]] = []
    out_risk_hit: List[Optional[bool]] = []
    out_tvc_th: List[Optional[float]] = []
    out_tvc_hit: List[Optional[bool]] = []
    out_severe: List[bool] = []
    out_in_cooling: List[bool] = []

    for i in range(len(df)):
        rr = risk_raw.iloc[i]
        tv = tvc.iloc[i]
        rd = rounds.iloc[i]
        rd_int = int(rd) if pd.notna(rd) else None

        if math.isfinite(float(rr)) if rr is not None else False:
            rrv = float(rr)
            ema = rrv if ema is None else (alpha * rrv + (1.0 - alpha) * float(ema))
            risk_hist.append(rrv)
        else:
            risk_hist.append(float("nan"))

        if math.isfinite(float(tv)) if tv is not None else False:
            tvc_hist.append(float(tv))
        else:
            tvc_hist.append(float("nan"))

        rh = [x for x in risk_hist[max(0, len(risk_hist) - window) :] if math.isfinite(float(x))]
        th = [x for x in tvc_hist[max(0, len(tvc_hist) - window) :] if math.isfinite(float(x))]

        risk_th = None
        risk_hit = None
        if ema is not None and len(rh) >= min_samples:
            risk_th = float(np.quantile(np.asarray(rh, dtype=float), 1.0 - q))
            risk_hit = bool(float(ema) >= float(risk_th))

        tvc_th = None
        tvc_hit = None
        if math.isfinite(float(tv)) and len(th) >= min_samples:
            tvc_th = float(np.quantile(np.asarray(th, dtype=float), q))
            tvc_hit = bool(float(tv) <= float(tvc_th))

        if severe_logic == "and":
            severe = bool((risk_hit is True) and (tvc_hit is True))
        else:
            severe = bool((risk_hit is True) or (tvc_hit is True))

        if isinstance(rule.iloc[i], str) and rule.iloc[i] in {
            "severe_overfit_lambda_down",
            "rollback_lambda_down",
        } and rd_int is not None:
            last_down_round = rd_int

        in_cooling = False
        if cooling > 0 and last_down_round is not None and rd_int is not None:
            in_cooling = bool((rd_int - last_down_round) <= cooling)

        out_ema.append(float(ema) if ema is not None else None)
        out_risk_th.append(risk_th)
        out_risk_hit.append(risk_hit)
        out_tvc_th.append(tvc_th)
        out_tvc_hit.append(tvc_hit)
        out_severe.append(severe)
        out_in_cooling.append(in_cooling)

    df["policy_overfit_risk_ema"] = pd.to_numeric(pd.Series(out_ema), errors="coerce").astype(float)
    df["policy_ci_risk_th_rawhist"] = pd.to_numeric(pd.Series(out_risk_th), errors="coerce").astype(float)
    df["policy_ci_risk_hit"] = pd.Series(out_risk_hit, dtype="boolean")
    df["policy_ci_tvc_th"] = pd.to_numeric(pd.Series(out_tvc_th), errors="coerce").astype(float)
    df["policy_ci_tvc_hit"] = pd.Series(out_tvc_hit, dtype="boolean")
    df["policy_ci_severe"] = pd.Series(out_severe, dtype=bool)
    df["policy_in_cooling"] = pd.Series(out_in_cooling, dtype=bool)
    return df

def plot_lambda_trajectory(df: pd.DataFrame, output_dir: str):
    lam = _effective_lambda(df)
    if lam.isna().all():
        print("Skipping Lambda Plot: No lambda_effective data found.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df["round"], lam, marker="o", linewidth=2.5)
    plt.title("Lambda Trajectory", fontsize=14)
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Lambda (effective)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.ylim(-0.05, 1.05)
    plt.savefig(os.path.join(output_dir, "strategy_lambda_trajectory.png"), dpi=300)
    plt.close()

def plot_run_lambda_compare(combined_round_df: pd.DataFrame, output_dir: str):
    if combined_round_df.empty:
        return
    if "experiment_name" not in combined_round_df.columns:
        return
    if "round" not in combined_round_df.columns:
        return

    df = combined_round_df.copy()
    df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    df = df[df["round"].notna()]
    if df.empty:
        return
    df["lambda_eff"] = _effective_lambda(df)
    if df["lambda_eff"].isna().all():
        return

    pivot = (
        df.pivot_table(index="round", columns="experiment_name", values="lambda_eff", aggfunc="mean")
        .sort_index()
    )
    if pivot.empty or pivot.shape[1] == 0:
        return

    plt.figure(figsize=(12, 7))
    for col in pivot.columns:
        y = pd.to_numeric(pivot[col], errors="coerce").astype(float)
        if y.isna().all():
            continue
        plt.plot(pivot.index.astype(int), y, marker="o", linewidth=2.0, alpha=0.9, label=str(col))
    plt.title("Lambda Trajectory Comparison (Run)", fontsize=14)
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Lambda (effective)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=8, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "run_lambda_trajectory_compare.png"), dpi=300)
    plt.close()

def plot_run_miou_compare(combined_round_df: pd.DataFrame, output_dir: str):
    if combined_round_df.empty:
        return
    if "experiment_name" not in combined_round_df.columns:
        return
    if "round" not in combined_round_df.columns:
        return
    if "final_miou" not in combined_round_df.columns:
        return

    df = combined_round_df.copy()
    df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    df = df[df["round"].notna()]
    if df.empty:
        return

    df["final_miou"] = pd.to_numeric(df["final_miou"], errors="coerce").astype(float)
    if df["final_miou"].isna().all():
        return

    pivot = (
        df.pivot_table(index="round", columns="experiment_name", values="final_miou", aggfunc="mean")
        .sort_index()
    )
    if pivot.empty or pivot.shape[1] == 0:
        return

    plt.figure(figsize=(12, 7))
    for col in pivot.columns:
        y = pd.to_numeric(pivot[col], errors="coerce").astype(float)
        if y.isna().all():
            continue
        plt.plot(pivot.index.astype(int), y, marker="o", linewidth=2.0, alpha=0.9, label=str(col))
    plt.title("mIoU Trajectory Comparison (Run)", fontsize=14)
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("mIoU", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=8, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "run_miou_trajectory_compare.png"), dpi=300)
    plt.close()

def export_run_experiment_summary(combined_round_df: pd.DataFrame, output_dir: str):
    if combined_round_df.empty:
        return
    if "experiment_name" not in combined_round_df.columns:
        return
    if "round" not in combined_round_df.columns:
        return

    df = combined_round_df.copy()
    df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    df = df[df["round"].notna()]
    if df.empty:
        return

    if "final_miou" in df.columns:
        df["final_miou"] = pd.to_numeric(df["final_miou"], errors="coerce").astype(float)
    if "final_f1" in df.columns:
        df["final_f1"] = pd.to_numeric(df["final_f1"], errors="coerce").astype(float)
    df["lambda_eff"] = _effective_lambda(df)

    rows: List[Dict[str, Any]] = []
    for name, g in df.groupby("experiment_name", dropna=False):
        g = g.sort_values("round")
        last_round = int(pd.to_numeric(g["round"], errors="coerce").max())
        last_miou = None
        if "final_miou" in g.columns:
            gm = g.dropna(subset=["final_miou"])
            if not gm.empty:
                last_miou = float(gm.sort_values("round")["final_miou"].iloc[-1])
        best_miou = None
        if "final_miou" in g.columns and pd.to_numeric(g["final_miou"], errors="coerce").notna().any():
            best_miou = float(pd.to_numeric(g["final_miou"], errors="coerce").max())
        last_lambda = None
        gl = g.dropna(subset=["lambda_eff"])
        if not gl.empty:
            last_lambda = float(pd.to_numeric(gl["lambda_eff"], errors="coerce").iloc[-1])
        rows.append(
            {
                "experiment_name": name,
                "n_rounds": int(g["round"].nunique()),
                "last_round": last_round,
                "last_miou": last_miou,
                "best_miou": best_miou,
                "lambda_mean": float(pd.to_numeric(g["lambda_eff"], errors="coerce").mean()) if pd.to_numeric(g["lambda_eff"], errors="coerce").notna().any() else None,
                "lambda_last": last_lambda,
                "overfit_risk_mean": float(pd.to_numeric(g.get("overfit_risk"), errors="coerce").mean()) if "overfit_risk" in g.columns else None,
                "tvc_neg_rate_mean": float(pd.to_numeric(g.get("grad_tvc_neg_rate"), errors="coerce").mean()) if "grad_tvc_neg_rate" in g.columns else None,
            }
        )

    if not rows:
        return
    out = pd.DataFrame(rows)
    out = out.sort_values(["last_miou", "best_miou"], ascending=False, na_position="last").reset_index(drop=True)
    out_path = os.path.join(output_dir, "run_experiment_summary.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

def plot_gradient_decomposition(df: pd.DataFrame, output_dir: str):
    if "avg_uncertainty" in df.columns and "avg_knowledge_gain" in df.columns:
        u_col, k_col = "avg_uncertainty", "avg_knowledge_gain"
    elif "u_mean" in df.columns and "k_mean" in df.columns:
        u_col, k_col = "u_mean", "k_mean"
    else:
        print("Skipping Decomposition Plot: Missing score components.")
        return

    if df[u_col].isnull().all() or df[k_col].isnull().all():
        print("Skipping Decomposition Plot: All score components are null.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(df["round"], df[u_col], label=u_col, marker="s", linestyle="-", color="#1f77b4")
    plt.plot(df["round"], df[k_col], label=k_col, marker="^", linestyle="-", color="#ff7f0e")
    plt.title("Component Evolution", fontsize=14)
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(output_dir, "strategy_gradient_components.png"), dpi=300)
    plt.close()

def plot_phase_space(df: pd.DataFrame, output_dir: str):
    if "avg_uncertainty" in df.columns and "avg_knowledge_gain" in df.columns:
        u_col, k_col = "avg_uncertainty", "avg_knowledge_gain"
        x_label, y_label = "Uncertainty", "Diversity"
    elif "u_mean" in df.columns and "k_mean" in df.columns:
        u_col, k_col = "u_mean", "k_mean"
        x_label, y_label = "U mean", "K mean"
    else:
        print("Skipping Phase Space Plot: Missing score components.")
        return

    if df[u_col].isnull().all() or df[k_col].isnull().all():
        print("Skipping Phase Space Plot: All score components are null.")
        return

    plt.figure(figsize=(8, 8))
    sc = plt.scatter(
        df[u_col],
        df[k_col],
        c=df["round"],
        cmap="viridis",
        s=100,
        edgecolors="k",
        zorder=2,
    )

    plt.plot(df[u_col], df[k_col], color="gray", alpha=0.5, zorder=1)
    cbar = plt.colorbar(sc)
    cbar.set_label("Round")

    plt.title("Phase Space Trajectory", fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.savefig(os.path.join(output_dir, "strategy_phase_space.png"), dpi=300)
    plt.close()

def load_epoch_end(trace_path: str) -> pd.DataFrame:
    if not os.path.exists(trace_path):
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    with open(trace_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "epoch_end":
                continue
            grad = entry.get("grad") if isinstance(entry.get("grad"), dict) else {}
            cos_to_mean = grad.get("cos_to_mean") if isinstance(grad.get("cos_to_mean"), dict) else {}
            cos_consecutive = grad.get("cos_consecutive") if isinstance(grad.get("cos_consecutive"), dict) else {}
            total_norm = grad.get("total_norm") if isinstance(grad.get("total_norm"), dict) else {}
            rows.append(
                {
                    "round": entry.get("round"),
                    "epoch": entry.get("epoch"),
                    "labeled_size": entry.get("labeled_size"),
                    "loss": entry.get("loss"),
                    "mIoU": entry.get("mIoU"),
                    "f1": entry.get("f1"),
                    "grad_train_val_cos": grad.get("train_val_cos"),
                    "grad_cos_to_mean_mean": cos_to_mean.get("mean"),
                    "grad_cos_consecutive_mean": cos_consecutive.get("mean"),
                    "grad_total_norm_mean": total_norm.get("mean"),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")
    for c in ["loss", "mIoU", "f1", "grad_train_val_cos", "grad_cos_to_mean_mean", "grad_cos_consecutive_mean", "grad_total_norm_mean"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df = df.sort_values(["round", "epoch"]).reset_index(drop=True)
    return df

def aggregate_gradients_per_round(epoch_df: pd.DataFrame) -> pd.DataFrame:
    if epoch_df.empty:
        return pd.DataFrame()

    def _last(series: pd.Series) -> Any:
        if series is None or series.empty:
            return None
        return series.iloc[-1]

    g = epoch_df.groupby("round", dropna=False)
    rounds = list(g.size().index)
    out = pd.DataFrame({"round": rounds})
    out = out.merge(g["grad_train_val_cos"].mean().rename("grad_tvc_mean"), left_on="round", right_index=True, how="left")
    out = out.merge(g["grad_train_val_cos"].median().rename("grad_tvc_median"), left_on="round", right_index=True, how="left")
    out = out.merge(g["grad_train_val_cos"].min().rename("grad_tvc_min"), left_on="round", right_index=True, how="left")
    out = out.merge(g["grad_train_val_cos"].max().rename("grad_tvc_max"), left_on="round", right_index=True, how="left")
    out = out.merge(g["grad_train_val_cos"].apply(_last).rename("grad_tvc_last"), left_on="round", right_index=True, how="left")
    out = out.merge(
        g["grad_train_val_cos"]
        .apply(lambda s: float((pd.to_numeric(s, errors="coerce") < 0).mean()) if len(s) else None)
        .rename("grad_tvc_neg_rate"),
        left_on="round",
        right_index=True,
        how="left",
    )
    out = out.merge(g.size().astype(int).rename("grad_epochs"), left_on="round", right_index=True, how="left")
    out["round"] = pd.to_numeric(out["round"], errors="coerce").astype("Int64")
    for c in ["grad_tvc_mean", "grad_tvc_median", "grad_tvc_min", "grad_tvc_max", "grad_tvc_last", "grad_tvc_neg_rate"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    out["grad_epochs"] = pd.to_numeric(out["grad_epochs"], errors="coerce").astype("Int64")
    out = out.sort_values("round").reset_index(drop=True)
    return out

def _effective_lambda(per_round_df: pd.DataFrame) -> pd.Series:
    n = len(per_round_df)
    base = pd.Series([pd.NA] * n)
    if "lambda_effective" in per_round_df.columns:
        base = base.combine_first(pd.to_numeric(per_round_df["lambda_effective"], errors="coerce"))
    if "lambda_policy_apply" in per_round_df.columns:
        base = base.combine_first(pd.to_numeric(per_round_df["lambda_policy_apply"], errors="coerce"))
    if "lambda_override" in per_round_df.columns:
        base = base.combine_first(pd.to_numeric(per_round_df["lambda_override"], errors="coerce"))
    if "lambda_t" in per_round_df.columns:
        base = base.combine_first(pd.to_numeric(per_round_df["lambda_t"], errors="coerce"))
    return pd.to_numeric(base, errors="coerce").astype(float)

def _corr(x: pd.Series, y: pd.Series) -> Optional[float]:
    try:
        xx = pd.to_numeric(x, errors="coerce").astype(float)
        yy = pd.to_numeric(y, errors="coerce").astype(float)
        mask = xx.notna() & yy.notna()
        if int(mask.sum()) < 3:
            return None
        return float(xx[mask].corr(yy[mask]))
    except Exception:
        return None

def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        return None
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])

def _rankdata(values: List[float]) -> List[float]:
    idx = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[idx[j + 1]] == values[idx[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = avg_rank
        i = j + 1
    return ranks

def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    if any((not math.isfinite(x)) for x in xs) or any((not math.isfinite(y)) for y in ys):
        return None
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    return _pearson(rx, ry)

def _fixed_effect_residuals(values: List[float], effects: Dict[str, List[Any]]) -> Optional[np.ndarray]:
    if not values:
        return None
    y = np.array(values, dtype=float)
    if np.any(~np.isfinite(y)):
        return None
    mats = []
    for _name, cats in effects.items():
        if len(cats) != len(values):
            return None
        d = pd.get_dummies(pd.Series(cats, dtype="category"), drop_first=True)
        if d.shape[1] > 0:
            mats.append(d.to_numpy(dtype=float))
    if mats:
        X = np.concatenate([np.ones((len(values), 1), dtype=float)] + mats, axis=1)
    else:
        X = np.ones((len(values), 1), dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X.dot(coef)
    return resid

def _partial_corr_fixed_effects(xs: List[float], ys: List[float], effects: Dict[str, List[Any]]) -> Dict[str, Optional[float]]:
    xr = _fixed_effect_residuals(xs, effects)
    yr = _fixed_effect_residuals(ys, effects)
    if xr is None or yr is None:
        return {"pearson": None, "spearman": None}
    xr_list = xr.tolist()
    yr_list = yr.tolist()
    return {"pearson": _pearson(xr_list, yr_list), "spearman": _spearman(xr_list, yr_list)}

def _parse_seed_from_run_id(run_id: str) -> Optional[int]:
    m = re.search(r"seed(\d+)", run_id)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _extract_round_miou_risk(trace_path: str) -> Tuple[Dict[int, float], Dict[int, float]]:
    miou_by_round: Dict[int, float] = {}
    risk_by_round: Dict[int, float] = {}
    with open(trace_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            t = ev.get("type")
            if t == "round_summary":
                r = ev.get("round")
                mi = _parse_float(ev.get("mIoU"))
                try:
                    rr = int(r)
                except Exception:
                    rr = None
                if rr is not None and mi is not None:
                    miou_by_round[rr] = float(mi)
                ts = ev.get("training_state") if isinstance(ev.get("training_state"), dict) else {}
                rrisk = _parse_float(ts.get("overfit_risk"))
                if rr is not None and rrisk is not None:
                    risk_by_round[rr] = float(rrisk)
            elif t == "overfit_signal":
                r = ev.get("round")
                rrisk = _parse_float(ev.get("overfit_risk"))
                try:
                    rr = int(r)
                except Exception:
                    rr = None
                if rr is not None and rrisk is not None:
                    risk_by_round[rr] = float(rrisk)
    return miou_by_round, risk_by_round

def deep_analyze_risk_miou(
    runs_root: str,
    run_ids: List[str],
    experiments: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run_id in run_ids:
        run_dir = os.path.join(runs_root, run_id)
        seed = _parse_seed_from_run_id(run_id)
        if not os.path.isdir(run_dir):
            continue
        for exp in experiments:
            trace_path = os.path.join(run_dir, f"{exp}_trace.jsonl")
            if not os.path.isfile(trace_path):
                continue
            miou_by_round, risk_by_round = _extract_round_miou_risk(trace_path)
            if not miou_by_round:
                continue
            rounds = sorted(miou_by_round.keys())
            last_round = int(rounds[-1])
            last_miou = float(miou_by_round[last_round])
            best_miou = float(max(miou_by_round.values()))

            risks = [risk_by_round[r] for r in sorted(risk_by_round.keys()) if r in risk_by_round]
            risk_mean = float(np.mean(risks)) if risks else None
            risk_max = float(np.max(risks)) if risks else None
            risk_last = float(risk_by_round.get(last_round)) if last_round in risk_by_round else None

            risk_round_corr = None
            if risks and len(risk_by_round) >= 3:
                rr = sorted(risk_by_round.keys())
                xs = [float(r) for r in rr]
                ys = [float(risk_by_round[r]) for r in rr]
                risk_round_corr = _pearson(xs, ys)

            deltas_next = []
            risks_t = []
            for r in rounds:
                if (r + 1) not in miou_by_round:
                    continue
                if r not in risk_by_round:
                    continue
                delta = float(miou_by_round[r + 1] - miou_by_round[r])
                deltas_next.append(delta)
                risks_t.append(float(risk_by_round[r]))
            corr_risk_delta_next = _pearson(risks_t, deltas_next) if len(risks_t) >= 3 else None
            spr_risk_delta_next = _spearman(risks_t, deltas_next) if len(risks_t) >= 3 else None

            rows.append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "experiment_name": exp,
                    "last_round": last_round,
                    "last_miou": last_miou,
                    "best_miou": best_miou,
                    "overfit_mean": risk_mean,
                    "overfit_max": risk_max,
                    "overfit_last": risk_last,
                    "corr_overfit_round": risk_round_corr,
                    "corr_overfit_to_next_delta_pearson": corr_risk_delta_next,
                    "corr_overfit_to_next_delta_spearman": spr_risk_delta_next,
                }
            )
    return pd.DataFrame(rows)

def print_deep_risk_miou_report(df: pd.DataFrame) -> None:
    if df.empty:
        print("No rows collected.")
        return

    d = df.copy()
    for c in ["last_miou", "best_miou", "overfit_mean", "overfit_max", "overfit_last"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)

    def _pooled_corr(xcol: str, ycol: str) -> Dict[str, Optional[float]]:
        sub = d[[xcol, ycol]].dropna()
        xs = sub[xcol].astype(float).tolist()
        ys = sub[ycol].astype(float).tolist()
        return {"n": int(len(xs)), "pearson": _pearson(xs, ys), "spearman": _spearman(xs, ys)}

    print("\nPooled correlations:")
    for xcol in ["overfit_mean", "overfit_max", "overfit_last"]:
        for ycol in ["last_miou", "best_miou"]:
            r = _pooled_corr(xcol, ycol)
            print(f"  {xcol:12s} vs {ycol:9s}  n={r['n']:3d}  pearson={r['pearson']}  spearman={r['spearman']}")

    print("\nFixed-effect partial correlations (controls):")
    base = d.dropna(subset=["overfit_mean", "last_miou", "experiment_name", "seed"])
    xs = base["overfit_mean"].astype(float).tolist()
    ys_last = base["last_miou"].astype(float).tolist()
    ys_best = base["best_miou"].astype(float).tolist() if base["best_miou"].notna().all() else None
    eff = {"seed": base["seed"].tolist(), "experiment_name": base["experiment_name"].tolist()}
    r_last = _partial_corr_fixed_effects(xs, ys_last, eff)
    print(f"  overfit_mean vs last_miou | seed+experiment  pearson={r_last['pearson']}  spearman={r_last['spearman']}  n={len(xs)}")
    if ys_best is not None:
        r_best = _partial_corr_fixed_effects(xs, ys_best, eff)
        print(f"  overfit_mean vs best_miou | seed+experiment  pearson={r_best['pearson']}  spearman={r_best['spearman']}  n={len(xs)}")

    print("\nWithin-seed (across experiments) Spearman:")
    for seed, g in d.dropna(subset=["seed", "overfit_mean", "last_miou"]).groupby("seed"):
        xs = g["overfit_mean"].astype(float).tolist()
        ys = g["last_miou"].astype(float).tolist()
        print(f"  seed{int(seed):02d}  n={len(xs)}  spearman(overfit_mean,last_miou)={_spearman(xs, ys)}  pearson={_pearson(xs, ys)}")

    print("\nPer-experiment (across seeds) Spearman:")
    for exp, g in d.dropna(subset=["experiment_name", "overfit_mean", "last_miou"]).groupby("experiment_name"):
        xs = g["overfit_mean"].astype(float).tolist()
        ys_last = g["last_miou"].astype(float).tolist()
        ys_best = g["best_miou"].astype(float).tolist()
        if len(xs) < 3:
            continue
        print(
            f"  {str(exp):18s} n={len(xs)}  spearman(mean,last)={_spearman(xs, ys_last)}  spearman(mean,best)={_spearman(xs, ys_best)}"
        )

    print("\nLagged relationship: overfit_risk(t) vs ΔmIoU(t→t+1):")
    lag = d.dropna(subset=["corr_overfit_to_next_delta_spearman", "corr_overfit_to_next_delta_pearson"])
    if lag.empty:
        print("  No lagged rows available.")
    else:
        print(
            f"  per-run corr distribution  n={len(lag)}  "
            f"mean_pearson={float(lag['corr_overfit_to_next_delta_pearson'].mean())}  "
            f"mean_spearman={float(lag['corr_overfit_to_next_delta_spearman'].mean())}"
        )
        by_exp = lag.groupby("experiment_name")[["corr_overfit_to_next_delta_pearson", "corr_overfit_to_next_delta_spearman"]].mean()
        print(by_exp.sort_index().to_string())

def _miou_gain(per_round_df: pd.DataFrame) -> pd.Series:
    if "final_miou" not in per_round_df.columns:
        return pd.Series([float("nan")] * len(per_round_df))
    m = pd.to_numeric(per_round_df["final_miou"], errors="coerce").astype(float)
    return m - m.shift(1)

def add_overfit_signals(
    per_round_df: pd.DataFrame,
    risk_w_neg_rate: float = 1.0,
    risk_w_min: float = 0.5,
    risk_w_last: float = 0.5,
) -> pd.DataFrame:
    if per_round_df.empty:
        return per_round_df

    df = per_round_df.copy()
    if "grad_tvc_neg_rate" not in df.columns:
        df["grad_tvc_neg_rate"] = float("nan")
    if "grad_tvc_min" not in df.columns:
        df["grad_tvc_min"] = float("nan")
    if "grad_tvc_last" not in df.columns:
        df["grad_tvc_last"] = float("nan")

    if "overfit_risk" not in df.columns:
        df["overfit_risk"] = float("nan")

    neg_rate = pd.to_numeric(df["grad_tvc_neg_rate"], errors="coerce").astype(float)
    tvc_min = pd.to_numeric(df["grad_tvc_min"], errors="coerce").astype(float)
    tvc_last = pd.to_numeric(df["grad_tvc_last"], errors="coerce").astype(float)

    risk_in_trace = pd.to_numeric(df["overfit_risk"], errors="coerce").astype(float)
    min_bad = (-tvc_min).clip(lower=0.0)
    last_bad = (-tvc_last).clip(lower=0.0)
    computed_risk = (risk_w_neg_rate * neg_rate) + (risk_w_min * min_bad) + (risk_w_last * last_bad)
    if risk_in_trace.notna().any():
        diff = (risk_in_trace - computed_risk).abs()
        bad = diff.notna() & (diff > 1e-6)
        if bool(bad.any()):
            raise RuntimeError("overfit_risk in trace is inconsistent with computed risk")
        df["overfit_risk"] = risk_in_trace
    else:
        df["overfit_risk"] = computed_risk

    df["miou_gain"] = _miou_gain(df)
    df["miou_gain_next"] = pd.to_numeric(df["miou_gain"], errors="coerce").shift(-1)

    lam = _effective_lambda(df)
    df["lambda_eff"] = lam
    df["overfit_risk_next"] = pd.to_numeric(df["overfit_risk"], errors="coerce").shift(-1)
    return df

def analyze_trace_with_gradients(trace_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_round = load_trace(trace_path)
    epoch = load_epoch_end(trace_path)
    grad_round = aggregate_gradients_per_round(epoch)
    if not per_round.empty and not grad_round.empty:
        per_round = per_round.merge(grad_round, on="round", how="left")
    return per_round, epoch

def export_lambda_diagnostics(per_round_df: pd.DataFrame, output_dir: str, name: str) -> Optional[str]:
    if per_round_df.empty:
        return None
    cols = [
        c
        for c in [
            "round",
            "lambda_eff",
            "lambda_policy_apply",
            "lambda_policy_rule",
            "lambda_policy_base",
            "lambda_override",
            "lambda_override_suggested",
            "lambda_override_exceeds_suggested_range",
            "lambda_override_clamped",
            "lambda_override_clamp_reason",
            "overfit_risk",
            "policy_overfit_risk_ema",
            "policy_ci_risk_th_rawhist",
            "policy_ci_risk_hit",
            "grad_tvc_last",
            "policy_ci_tvc_th",
            "policy_ci_tvc_hit",
            "policy_ci_severe",
            "policy_in_cooling",
            "miou_delta",
            "miou_gain",
            "miou_gain_next",
            "final_miou",
        ]
        if c in per_round_df.columns
    ]
    if not cols:
        return None
    out = per_round_df[cols].copy()
    out_path = os.path.join(output_dir, f"{name}_lambda_diagnostics.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return out_path

def plot_overfit_vs_gain(per_round_df: pd.DataFrame, output_dir: str, name: str):
    if per_round_df.empty:
        return
    if "overfit_risk" not in per_round_df.columns or "miou_gain" not in per_round_df.columns:
        return
    x = pd.to_numeric(per_round_df["overfit_risk"], errors="coerce").astype(float)
    y = pd.to_numeric(per_round_df["miou_gain"], errors="coerce").astype(float)
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 3:
        return
    plt.figure(figsize=(8, 6))
    plt.scatter(x[mask], y[mask], c=per_round_df.loc[mask, "round"], cmap="viridis", s=80, edgecolors="k")
    plt.xlabel("overfit_risk", fontsize=12)
    plt.ylabel("mIoU gain (this round)", fontsize=12)
    plt.title(f"overfit_risk vs mIoU gain ({name})", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.5)
    cbar = plt.colorbar()
    cbar.set_label("round")
    plt.savefig(os.path.join(output_dir, f"{name}_overfit_vs_miou_gain.png"), dpi=300)
    plt.close()

def plot_lambda_vs_tvc(per_round_df: pd.DataFrame, output_dir: str, name: str):
    if per_round_df.empty:
        return
    lam = _effective_lambda(per_round_df)
    if lam.isna().all():
        return
    if "grad_tvc_mean" not in per_round_df.columns and "grad_tvc_last" not in per_round_df.columns:
        return
    tvc = per_round_df["grad_tvc_mean"] if "grad_tvc_mean" in per_round_df.columns else per_round_df["grad_tvc_last"]
    tvc = pd.to_numeric(tvc, errors="coerce").astype(float)
    mask = lam.notna() & tvc.notna()
    if int(mask.sum()) < 3:
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(lam[mask], tvc[mask], c=per_round_df.loc[mask, "round"], cmap="viridis", s=80, edgecolors="k")
    plt.xlabel("lambda", fontsize=12)
    plt.ylabel("grad_train_val_cos (per-round agg)", fontsize=12)
    plt.title(f"lambda vs grad_train_val_cos ({name})", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.5)
    cbar = plt.colorbar()
    cbar.set_label("round")
    plt.savefig(os.path.join(output_dir, f"{name}_lambda_vs_tvc.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax1.plot(per_round_df["round"], lam, marker="o", linewidth=2.0, color="#1f77b4", label="lambda")
    ax1.set_xlabel("round", fontsize=12)
    ax1.set_ylabel("lambda", fontsize=12, color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(-0.05, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(per_round_df["round"], tvc, marker="s", linewidth=2.0, color="#d62728", label="tvc")
    ax2.set_ylabel("grad_train_val_cos", fontsize=12, color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.title(f"lambda & grad_train_val_cos over rounds ({name})", fontsize=13)
    plt.savefig(os.path.join(output_dir, f"{name}_lambda_tvc_trajectory.png"), dpi=300)
    plt.close()

def analyze_run_dir(run_dir: str, output_dir: str, experiment_name: Optional[str]) -> None:
    if not os.path.isdir(run_dir):
        raise NotADirectoryError(run_dir)

    traces = [
        fn
        for fn in os.listdir(run_dir)
        if fn.endswith("_trace.jsonl") and os.path.isfile(os.path.join(run_dir, fn))
    ]
    if experiment_name:
        want = f"{experiment_name}_trace.jsonl"
        traces = [t for t in traces if t == want]
    traces = sorted(traces)
    if not traces:
        raise FileNotFoundError("No *_trace.jsonl found")

    os.makedirs(output_dir, exist_ok=True)

    combined_round_rows: List[pd.DataFrame] = []
    combined_epoch_rows: List[pd.DataFrame] = []
    report_rows: List[Dict[str, Any]] = []
    overfit_report_rows: List[Dict[str, Any]] = []

    for fn in traces:
        trace_path = os.path.join(run_dir, fn)
        name = fn.replace("_trace.jsonl", "")
        per_round, epoch = analyze_trace_with_gradients(trace_path)
        if per_round.empty:
            continue
        per_round = add_overfit_signals(per_round)
        meta = load_trace_meta(trace_path)
        per_round = add_lambda_policy_diagnostics(per_round, meta)
        per_round["experiment_name"] = name
        if not epoch.empty:
            epoch["experiment_name"] = name
            combined_epoch_rows.append(epoch)
        combined_round_rows.append(per_round)

        export_path = os.path.join(output_dir, f"{name}_per_round_with_grad.csv")
        per_round.to_csv(export_path, index=False)
        export_lambda_diagnostics(per_round, output_dir, name)

        lam = _effective_lambda(per_round)
        corr_lam_tvc_mean = _corr(lam, per_round.get("grad_tvc_mean", pd.Series(dtype=float)))
        corr_lam_tvc_last = _corr(lam, per_round.get("grad_tvc_last", pd.Series(dtype=float)))
        corr_lam_neg_rate = _corr(lam, per_round.get("grad_tvc_neg_rate", pd.Series(dtype=float)))

        report_rows.append(
            {
                "experiment_name": name,
                "n_rounds": int(per_round["round"].notna().sum()) if "round" in per_round.columns else int(len(per_round)),
                "lambda_corr_tvc_mean": corr_lam_tvc_mean,
                "lambda_corr_tvc_last": corr_lam_tvc_last,
                "lambda_corr_tvc_neg_rate": corr_lam_neg_rate,
                "tvc_mean_all_rounds": float(pd.to_numeric(per_round.get("grad_tvc_mean"), errors="coerce").mean()) if "grad_tvc_mean" in per_round.columns else None,
                "tvc_neg_rate_all_rounds_mean": float(pd.to_numeric(per_round.get("grad_tvc_neg_rate"), errors="coerce").mean()) if "grad_tvc_neg_rate" in per_round.columns else None,
                "tvc_min_any": float(pd.to_numeric(per_round.get("grad_tvc_min"), errors="coerce").min()) if "grad_tvc_min" in per_round.columns else None,
                "tvc_max_any": float(pd.to_numeric(per_round.get("grad_tvc_max"), errors="coerce").max()) if "grad_tvc_max" in per_round.columns else None,
            }
        )

        overfit_report_rows.append(
            {
                "experiment_name": name,
                "n_rounds": int(per_round["round"].notna().sum()) if "round" in per_round.columns else int(len(per_round)),
                "overfit_risk_mean": float(pd.to_numeric(per_round.get("overfit_risk"), errors="coerce").mean()) if "overfit_risk" in per_round.columns else None,
                "overfit_risk_max": float(pd.to_numeric(per_round.get("overfit_risk"), errors="coerce").max()) if "overfit_risk" in per_round.columns else None,
                "corr_risk_miou_gain": _corr(per_round.get("overfit_risk", pd.Series(dtype=float)), per_round.get("miou_gain", pd.Series(dtype=float))),
                "corr_risk_miou_gain_next": _corr(per_round.get("overfit_risk", pd.Series(dtype=float)), per_round.get("miou_gain_next", pd.Series(dtype=float))),
                "corr_tvc_mean_miou_gain": _corr(per_round.get("grad_tvc_mean", pd.Series(dtype=float)), per_round.get("miou_gain", pd.Series(dtype=float))),
                "corr_tvc_mean_miou_gain_next": _corr(per_round.get("grad_tvc_mean", pd.Series(dtype=float)), per_round.get("miou_gain_next", pd.Series(dtype=float))),
            }
        )

        plot_lambda_vs_tvc(per_round, output_dir, name)
        plot_overfit_vs_gain(per_round, output_dir, name)

    if combined_round_rows:
        combined_round = pd.concat(combined_round_rows, ignore_index=True)
        combined_round_path = os.path.join(output_dir, "run_all_experiments_per_round_with_grad.csv")
        combined_round.to_csv(combined_round_path, index=False)
        print(f"Saved {combined_round_path}")
        plot_run_lambda_compare(combined_round, output_dir)
        plot_run_miou_compare(combined_round, output_dir)
        export_run_experiment_summary(combined_round, output_dir)

    if combined_epoch_rows:
        combined_epoch = pd.concat(combined_epoch_rows, ignore_index=True)
        combined_epoch_path = os.path.join(output_dir, "run_all_experiments_epoch_end_with_grad.csv")
        combined_epoch.to_csv(combined_epoch_path, index=False)
        print(f"Saved {combined_epoch_path}")

    if report_rows:
        report = pd.DataFrame(report_rows).sort_values("experiment_name").reset_index(drop=True)
        report_path = os.path.join(output_dir, "run_grad_alignment_report.csv")
        report.to_csv(report_path, index=False)
        print(f"Saved {report_path}")
        cols = [c for c in ["experiment_name", "n_rounds", "lambda_corr_tvc_mean", "lambda_corr_tvc_last", "tvc_mean_all_rounds", "tvc_neg_rate_all_rounds_mean", "tvc_min_any", "tvc_max_any"] if c in report.columns]
        if cols:
            print(report[cols].to_string(index=False))

    if overfit_report_rows:
        overfit = pd.DataFrame(overfit_report_rows).sort_values("experiment_name").reset_index(drop=True)
        overfit_path = os.path.join(output_dir, "run_overfit_risk_report.csv")
        overfit.to_csv(overfit_path, index=False)
        print(f"Saved {overfit_path}")
        cols = [
            c
            for c in [
                "experiment_name",
                "n_rounds",
                "overfit_risk_mean",
                "overfit_risk_max",
                "corr_risk_miou_gain",
                "corr_risk_miou_gain_next",
                "corr_tvc_mean_miou_gain",
                "corr_tvc_mean_miou_gain_next",
            ]
            if c in overfit.columns
        ]
        if cols:
            print(overfit[cols].to_string(index=False))

def _resolve_trace_path(experiment_dir_or_trace: str, experiment_name: Optional[str]) -> Optional[str]:
    if os.path.isfile(experiment_dir_or_trace) and experiment_dir_or_trace.endswith(".jsonl"):
        return experiment_dir_or_trace

    if not os.path.isdir(experiment_dir_or_trace):
        return None

    direct = os.path.join(experiment_dir_or_trace, "trace.jsonl")
    if os.path.exists(direct):
        return direct

    candidates = [
        fn
        for fn in os.listdir(experiment_dir_or_trace)
        if fn.endswith("_trace.jsonl") and os.path.isfile(os.path.join(experiment_dir_or_trace, fn))
    ]

    if not candidates:
        return None

    if experiment_name:
        expected = f"{experiment_name}_trace.jsonl"
        if expected in candidates:
            return os.path.join(experiment_dir_or_trace, expected)

    if len(candidates) == 1:
        return os.path.join(experiment_dir_or_trace, candidates[0])

    candidates = sorted(candidates)
    print("Error: multiple *_trace.jsonl found; please pass --experiment_name. Candidates:")
    for c in candidates:
        print(f"- {c}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Analyze Strategy from Trace Logs")
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to results/runs/<run_id>/ OR a specific *_trace.jsonl file",
    )
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name (used to pick <exp>_trace.jsonl)")
    parser.add_argument("--output", type=str, default=None, help="Output directory for plots/exports")
    parser.add_argument("--export_csv", action="store_true", help="Export per-round summary CSV")
    parser.add_argument("--grad_report", action="store_true", help="Analyze grad alignment across all *_trace.jsonl in a run dir")
    parser.add_argument("--risk_miou_deep", action="store_true", help="Deep analyze risk↔mIoU across multiple runs/seeds")
    parser.add_argument("--runs_root", type=str, default="results/runs", help="Root directory containing run_id folders")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Seed list for building run_ids")
    parser.add_argument("--run_id_template", type=str, default="run_src_full_model_with_baselines_seed{seed}", help="Template for run_id")
    parser.add_argument("--run_ids", type=str, nargs="*", default=None, help="Explicit run_ids (overrides --seeds)")
    parser.add_argument("--experiments", type=str, nargs="*", default=None, help="Experiment names to include")

    args = parser.parse_args()

    if args.risk_miou_deep:
        if args.run_ids:
            run_ids = list(args.run_ids)
        else:
            seeds = args.seeds if args.seeds else [42, 43, 44, 45, 46]
            run_ids = [args.run_id_template.format(seed=s) for s in seeds]
        experiments = (
            list(args.experiments)
            if args.experiments
            else [
                "baseline_random",
                "baseline_entropy",
                "baseline_bald",
                "baseline_coreset",
                "baseline_wang_style",
                "baseline_dial_style",
                "baseline_llm_rs",
                "baseline_llm_us",
                "full_model",
            ]
        )
        out_dir = args.output if args.output else os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        df = deep_analyze_risk_miou(args.runs_root, run_ids, experiments)
        csv_path = os.path.join(out_dir, "risk_miou_deep_rows.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")
        print_deep_risk_miou_report(df)
        return

    if args.grad_report:
        run_dir = args.experiment_dir
        output_dir = args.output if args.output else run_dir
        analyze_run_dir(run_dir, output_dir, args.experiment_name)
        return

    trace_path = _resolve_trace_path(args.experiment_dir, args.experiment_name)
    if not trace_path:
        raise FileNotFoundError(f"cannot resolve trace path from {args.experiment_dir}")

    df, _epoch = analyze_trace_with_gradients(trace_path)
    df = add_overfit_signals(df)
    meta = load_trace_meta(trace_path)
    df = add_lambda_policy_diagnostics(df, meta)

    if df.empty:
        raise RuntimeError("No per-round data found in trace")

    output_dir = args.output if args.output else (os.path.dirname(trace_path) if os.path.isfile(trace_path) else args.experiment_dir)
    os.makedirs(output_dir, exist_ok=True)

    cols = [
        c
        for c in [
            "round",
            "final_miou",
            "lambda_t",
            "lambda_override",
            "query_size",
            "selected_count",
            "selected_ids_count",
            "u_mean",
            "k_mean",
            "k_p75",
        ]
        if c in df.columns
    ]
    print(f"Loaded {len(df)} rounds from {os.path.basename(trace_path)}")
    if cols:
        print(df[cols].head(20).to_string(index=False))

    if args.export_csv:
        name = args.experiment_name or os.path.basename(trace_path).replace("_trace.jsonl", "")
        export_path = os.path.join(output_dir, f"{name}_per_round_summary.csv")
        df.to_csv(export_path, index=False)
        print(f"CSV exported to {export_path}")
        export_lambda_diagnostics(df, output_dir, name)

    plot_lambda_trajectory(df, output_dir)
    plot_gradient_decomposition(df, output_dir)
    plot_phase_space(df, output_dir)
    name = args.experiment_name or os.path.basename(trace_path).replace("_trace.jsonl", "")
    plot_lambda_vs_tvc(df, output_dir, name)
    plot_overfit_vs_gain(df, output_dir, name)

    print(f"Outputs saved to {output_dir}")

if __name__ == "__main__":
    main()
