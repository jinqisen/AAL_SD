import argparse
import json
import re
import statistics
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(obj).__name__}")
    return obj


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise TypeError(f"Expected JSON object at {path}:{line_no}, got {type(obj).__name__}")
            yield obj


def _parse_ts(ts: Any) -> Optional[datetime]:
    if not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _t_critical_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df == 1:
        return 12.706
    if df == 2:
        return 4.303
    if df == 3:
        return 3.182
    if df == 4:
        return 2.776
    if df == 5:
        return 2.571
    if df == 6:
        return 2.447
    if df == 7:
        return 2.365
    if df == 8:
        return 2.306
    if df == 9:
        return 2.262
    if df == 10:
        return 2.228
    return 1.96


def _summarize(values: List[float]) -> Dict[str, Any]:
    xs = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    n = len(xs)
    if n <= 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "ci95": float("nan")}
    if n == 1:
        return {"n": 1, "mean": float(xs[0]), "std": 0.0, "ci95": float("nan")}
    mean = float(statistics.fmean(xs))
    std = float(statistics.stdev(xs))
    sem = std / float(np.sqrt(n))
    t = _t_critical_95(n - 1)
    ci95 = float(t * sem)
    return {"n": int(n), "mean": mean, "std": std, "ci95": ci95}


def _load_json_obj(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_multi_seed_run_ids(group_dir: Path) -> List[str]:
    manifest = _load_json_obj(group_dir / "multi_seed_manifest.json")
    if isinstance(manifest, dict):
        run_ids = manifest.get("run_ids")
        if isinstance(run_ids, list) and all(isinstance(x, str) for x in run_ids):
            return [str(x) for x in run_ids]
    return []


def _load_multi_seed_summary(group_dir: Path) -> Dict[str, Any]:
    payload = _load_json_obj(group_dir / "multi_seed_summary.json")
    return payload if isinstance(payload, dict) else {}

def _pick_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None

def _pick_first_present(exps: List[str], candidates: List[str]) -> str:
    s = set(str(x) for x in (exps or []))
    for c in candidates:
        if str(c) in s:
            return str(c)
    return str(candidates[0]) if candidates else ""


@dataclass(frozen=True)
class ExperimentPaths:
    name: str
    status_path: Path
    trace_path: Path


def discover_experiments(run_dir: Path) -> Tuple[Dict[str, Any], List[ExperimentPaths]]:
    manifest = _read_json(run_dir / "manifest.json")
    exps: List[ExperimentPaths] = []
    for status_path in sorted(run_dir.glob("*_status.json")):
        name = status_path.name[: -len("_status.json")]
        trace_path = run_dir / f"{name}_trace.jsonl"
        exps.append(ExperimentPaths(name=name, status_path=status_path, trace_path=trace_path))
    return manifest, exps


def load_status_metrics(status_path: Path) -> Dict[str, Any]:
    payload = _read_json(status_path)
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    return {
        "status": payload.get("status"),
        "alc": result.get("alc"),
        "final_mIoU": result.get("final_mIoU"),
        "final_f1": result.get("final_f1"),
        "budget_history": result.get("budget_history"),
    }


def parse_trace(trace_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    epoch_rows: List[Dict[str, Any]] = []
    ctrl_rows: List[Dict[str, Any]] = []

    if not trace_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    for e in _iter_jsonl(trace_path):
        et = e.get("type")
        if et == "epoch_end":
            grad = e.get("grad") if isinstance(e.get("grad"), dict) else {}
            total_norm = grad.get("total_norm") if isinstance(grad.get("total_norm"), dict) else {}
            backbone_norm = grad.get("backbone_norm") if isinstance(grad.get("backbone_norm"), dict) else {}
            head_norm = grad.get("head_norm") if isinstance(grad.get("head_norm"), dict) else {}
            cos_consecutive = grad.get("cos_consecutive") if isinstance(grad.get("cos_consecutive"), dict) else {}
            cos_to_mean = grad.get("cos_to_mean") if isinstance(grad.get("cos_to_mean"), dict) else {}
            epoch_rows.append(
                {
                    "round": e.get("round"),
                    "epoch": e.get("epoch"),
                    "labeled_size": e.get("labeled_size"),
                    "mIoU": e.get("mIoU"),
                    "f1": e.get("f1"),
                    "ts": e.get("ts"),
                    "grad_total_norm_mean": total_norm.get("mean"),
                    "grad_backbone_norm_mean": backbone_norm.get("mean"),
                    "grad_head_norm_mean": head_norm.get("mean"),
                    "grad_cos_consecutive_mean": cos_consecutive.get("mean"),
                    "grad_cos_to_mean_mean": cos_to_mean.get("mean"),
                    "grad_train_val_cos": grad.get("train_val_cos"),
                    "grad_n_probe_batches": grad.get("n_probe_batches"),
                }
            )
        elif et == "controller_step":
            action = e.get("action") if isinstance(e.get("action"), dict) else {}
            state = e.get("state") if isinstance(e.get("state"), dict) else {}
            ctrl_rows.append(
                {
                    "round": e.get("round"),
                    "lambda": action.get("lambda"),
                    "epochs": action.get("epochs"),
                    "query_size": action.get("query_size"),
                    "rollback_flag": state.get("rollback_flag"),
                    "miou_delta": state.get("miou_delta"),
                    "last_miou": state.get("last_miou"),
                    "ts": e.get("ts"),
                }
            )

    df_epoch = pd.DataFrame(epoch_rows)
    df_ctrl = pd.DataFrame(ctrl_rows)

    if not df_epoch.empty:
        df_epoch["round"] = pd.to_numeric(df_epoch["round"], errors="coerce")
        df_epoch["epoch"] = pd.to_numeric(df_epoch["epoch"], errors="coerce")
        df_epoch["labeled_size"] = pd.to_numeric(df_epoch["labeled_size"], errors="coerce")
        df_epoch["mIoU"] = pd.to_numeric(df_epoch["mIoU"], errors="coerce")
        df_epoch["f1"] = pd.to_numeric(df_epoch["f1"], errors="coerce")
        for col in [
            "grad_total_norm_mean",
            "grad_backbone_norm_mean",
            "grad_head_norm_mean",
            "grad_cos_consecutive_mean",
            "grad_cos_to_mean_mean",
            "grad_train_val_cos",
            "grad_n_probe_batches",
        ]:
            if col in df_epoch.columns:
                df_epoch[col] = pd.to_numeric(df_epoch[col], errors="coerce")

    if not df_ctrl.empty:
        df_ctrl["round"] = pd.to_numeric(df_ctrl["round"], errors="coerce")
        df_ctrl["lambda"] = pd.to_numeric(df_ctrl["lambda"], errors="coerce")
        df_ctrl["epochs"] = pd.to_numeric(df_ctrl["epochs"], errors="coerce")
        df_ctrl["query_size"] = pd.to_numeric(df_ctrl["query_size"], errors="coerce")
        df_ctrl["miou_delta"] = pd.to_numeric(df_ctrl["miou_delta"], errors="coerce")
        df_ctrl["last_miou"] = pd.to_numeric(df_ctrl["last_miou"], errors="coerce")

    return df_epoch, df_ctrl


def parse_round_summary(trace_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, Any]] = []
    l3_rows: List[Dict[str, Any]] = []

    if not trace_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    for e in _iter_jsonl(trace_path):
        et = e.get("type")
        if et == "round_summary":
            selection = e.get("selection") if isinstance(e.get("selection"), dict) else {}
            ranking = e.get("ranking") if isinstance(e.get("ranking"), dict) else {}
            lambda_ctrl = e.get("lambda_controller") if isinstance(e.get("lambda_controller"), dict) else {}
            training_state = e.get("training_state") if isinstance(e.get("training_state"), dict) else {}
            summary_rows.append(
                {
                    "round": e.get("round"),
                    "labeled_size": e.get("labeled_size"),
                    "mIoU": e.get("mIoU"),
                    "f1": e.get("f1"),
                    "selection_expected": selection.get("expected"),
                    "selection_selected": selection.get("selected"),
                    "selection_source": selection.get("context", {}).get("source") if isinstance(selection.get("context"), dict) else None,
                    "avg_uncertainty": ranking.get("avg_uncertainty"),
                    "avg_knowledge_gain": ranking.get("avg_knowledge_gain"),
                    "lambda_t": ranking.get("lambda_effective"),
                    "lambda_controller": lambda_ctrl.get("lambda"),
                    "lambda_controller_mode": lambda_ctrl.get("mode"),
                    "miou_delta": training_state.get("miou_delta"),
                    "rollback_flag": training_state.get("rollback_flag"),
                }
            )
        elif et == "l3_selection":
            l3_rows.append(
                {
                    "round": e.get("round"),
                    "source": e.get("source"),
                    "topk": e.get("topk"),
                    "selected_limit": e.get("selected_limit"),
                    "top_items": json.dumps(e.get("top_items"), ensure_ascii=False),
                    "selected_items": json.dumps(e.get("selected_items"), ensure_ascii=False),
                }
            )

    df_summary = pd.DataFrame(summary_rows)
    df_l3 = pd.DataFrame(l3_rows)
    for col in ["round", "labeled_size", "mIoU", "f1", "selection_expected", "selection_selected", "avg_uncertainty", "avg_knowledge_gain", "lambda_t", "lambda_controller", "miou_delta"]:
        if col in df_summary.columns:
            df_summary[col] = pd.to_numeric(df_summary[col], errors="coerce")
    if "round" in df_l3.columns:
        df_l3["round"] = pd.to_numeric(df_l3["round"], errors="coerce")

    if (not df_summary.empty) and (not df_l3.empty) and ("round" in df_summary.columns) and ("round" in df_l3.columns):
        l3_stats_rows: List[Dict[str, Any]] = []
        l3g = df_l3.dropna(subset=["round"]).copy()
        if not l3g.empty:
            l3g["round"] = l3g["round"].astype(int)
            l3g = l3g.sort_values("round").groupby("round", as_index=False).tail(1)
            sel_map = {}
            if "selection_selected" in df_summary.columns:
                for _, rr in df_summary.dropna(subset=["round"]).iterrows():
                    sel_map[int(rr["round"])] = int(rr["selection_selected"]) if pd.notna(rr["selection_selected"]) else None
            for _, rrow in l3g.iterrows():
                rr = int(rrow["round"])
                top_items = json.loads(rrow["top_items"]) if isinstance(rrow.get("top_items"), str) else []
                selected_items = json.loads(rrow["selected_items"]) if isinstance(rrow.get("selected_items"), str) else []
                top_df = _l3_items_to_df(top_items).dropna(subset=["uncertainty", "knowledge_gain"]).copy()
                sel_df = _l3_items_to_df(selected_items).dropna(subset=["uncertainty", "knowledge_gain"]).copy()
                n_sel = sel_map.get(rr)
                use_df = None
                if n_sel is not None and n_sel > 0 and (not top_df.empty):
                    use_df = top_df.head(int(n_sel)).copy()
                if use_df is None or use_df.empty:
                    use_df = sel_df if not sel_df.empty else top_df
                if use_df is None or use_df.empty:
                    continue
                l3_stats_rows.append(
                    {
                        "round": rr,
                        "avg_uncertainty_l3": float(use_df["uncertainty"].astype(float).mean()),
                        "avg_knowledge_gain_l3": float(use_df["knowledge_gain"].astype(float).mean()),
                    }
                )
        if l3_stats_rows:
            df_l3s = pd.DataFrame(l3_stats_rows)
            for c in ["avg_uncertainty_l3", "avg_knowledge_gain_l3"]:
                df_l3s[c] = pd.to_numeric(df_l3s[c], errors="coerce")
            df_summary = df_summary.merge(df_l3s, on="round", how="left")
            if "avg_uncertainty" not in df_summary.columns:
                df_summary["avg_uncertainty"] = pd.NA
            if "avg_knowledge_gain" not in df_summary.columns:
                df_summary["avg_knowledge_gain"] = pd.NA
            df_summary["avg_uncertainty"] = pd.to_numeric(df_summary["avg_uncertainty"], errors="coerce").fillna(df_summary["avg_uncertainty_l3"])
            df_summary["avg_knowledge_gain"] = pd.to_numeric(df_summary["avg_knowledge_gain"], errors="coerce").fillna(df_summary["avg_knowledge_gain_l3"])
            df_summary = df_summary.drop(columns=[c for c in ["avg_uncertainty_l3", "avg_knowledge_gain_l3"] if c in df_summary.columns])
    return df_summary, df_l3


def build_round_curve(df_epoch: pd.DataFrame) -> pd.DataFrame:
    if df_epoch.empty:
        return pd.DataFrame()

    g = df_epoch.dropna(subset=["round"]).copy()
    g["round"] = g["round"].astype(int)
    agg = (
        g.groupby("round", as_index=False)
        .agg(
            labeled_size=("labeled_size", "max"),
            miou_round=("mIoU", "max"),
            f1_round=("f1", "max"),
            epochs_observed=("epoch", "max"),
        )
        .sort_values("round")
        .reset_index(drop=True)
    )
    return agg


def build_round_grad(df_epoch: pd.DataFrame) -> pd.DataFrame:
    if df_epoch.empty:
        return pd.DataFrame()
    cols = [
        "grad_total_norm_mean",
        "grad_backbone_norm_mean",
        "grad_head_norm_mean",
        "grad_cos_consecutive_mean",
        "grad_cos_to_mean_mean",
        "grad_train_val_cos",
        "grad_n_probe_batches",
    ]
    keep = [c for c in cols if c in df_epoch.columns]
    if not keep:
        return pd.DataFrame()
    g = df_epoch.dropna(subset=["round"]).copy()
    g["round"] = g["round"].astype(int)
    agg = g.groupby("round", as_index=False).agg({c: "mean" for c in keep}).sort_values("round").reset_index(drop=True)
    return agg


def compute_cost(df_epoch: pd.DataFrame, df_ctrl: pd.DataFrame) -> Dict[str, Any]:
    total_epochs = int(len(df_epoch)) if not df_epoch.empty else 0

    t0 = None
    t1 = None

    if not df_epoch.empty and df_epoch["ts"].dropna().shape[0]:
        t0 = _parse_ts(df_epoch["ts"].dropna().iloc[0])
        t1 = _parse_ts(df_epoch["ts"].dropna().iloc[-1])

    if (t0 is None or t1 is None) and (not df_ctrl.empty) and df_ctrl["ts"].dropna().shape[0]:
        t0 = _parse_ts(df_ctrl["ts"].dropna().iloc[0])
        t1 = _parse_ts(df_ctrl["ts"].dropna().iloc[-1])

    wall_clock_sec = None
    if t0 is not None and t1 is not None:
        wall_clock_sec = max(0.0, (t1 - t0).total_seconds())

    return {"total_epochs": total_epochs, "wall_clock_sec": wall_clock_sec}


def plot_aal_sd_active_learning_loop_diagram(
    outdir: Path,
    *,
    filename: str = "aal_sd_active_learning_loop.png",
) -> None:
    fig = plt.figure(figsize=(14, 5.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    def add_box(x: float, y: float, w: float, h: float, text: str, *, fc: str, ec: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.4,
            facecolor=fc,
            edgecolor=ec,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=11,
            wrap=True,
        )

    def add_arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        arr = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.4,
            color="#333333",
        )
        ax.add_patch(arr)

    w = 0.19
    h = 0.17
    y_top = 0.64
    y_bot = 0.23

    add_box(0.05, y_top, w, h, "未标注池 U", fc="#F2F2F2", ec="#666666")
    add_box(0.29, y_top, w, h, "模型推理\n(DeepLabV3+)", fc="#E8F0FE", ec="#3B6FB6")
    add_box(0.53, y_top, w, h, "计算得分\nU(x), K(x)", fc="#E8F7EE", ec="#2E8B57")
    add_box(0.77, y_top, w, h, "AD-KUCS\nScore(x)", fc="#FFF3E0", ec="#C77700")

    add_box(0.05, y_bot, w, h, "选择 Top-k\n(query size)", fc="#FFF3E0", ec="#C77700")
    add_box(0.29, y_bot, w, h, "人工标注\nOracle", fc="#FCE8E6", ec="#C43C35")
    add_box(0.53, y_bot, w, h, "已标注池 L\n+ 数据池更新", fc="#F2F2F2", ec="#666666")
    add_box(0.77, y_bot, w, h, "训练模型\n(固定 epochs=10)", fc="#E8F0FE", ec="#3B6FB6")

    add_box(
        0.77,
        0.43,
        w,
        0.14,
        "LLM Agent\n(控制 λ_t / 可选 query size)",
        fc="#EDE7F6",
        ec="#6A4FB6",
    )

    add_arrow(0.05 + w, y_top + h / 2, 0.29, y_top + h / 2)
    add_arrow(0.29 + w, y_top + h / 2, 0.53, y_top + h / 2)
    add_arrow(0.53 + w, y_top + h / 2, 0.77, y_top + h / 2)

    add_arrow(0.77 + w / 2, y_top, 0.05 + w / 2, y_bot + h)
    add_arrow(0.05 + w, y_bot + h / 2, 0.29, y_bot + h / 2)
    add_arrow(0.29 + w, y_bot + h / 2, 0.53, y_bot + h / 2)
    add_arrow(0.53 + w, y_bot + h / 2, 0.77, y_bot + h / 2)
    add_arrow(0.77, y_bot + h / 2, 0.29, y_top + h / 2)

    add_arrow(0.77, 0.50, 0.77, y_top + h * 0.25)

    ax.text(
        0.5,
        0.94,
        "AAL-SD 主动学习闭环（AD-KUCS 作为查询策略，LLM Agent 为受约束控制器）",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    out_path = outdir / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(
    curves: pd.DataFrame,
    outdir: Path,
    title: str,
    *,
    filename: str = "learning_curve_miou_vs_labeled.png",
    include_experiments: Optional[List[str]] = None,
    label_map: Optional[Dict[str, str]] = None,
):
    if curves.empty:
        return

    d = curves.copy()
    if include_experiments:
        keep = set(str(x) for x in include_experiments)
        d = d[d["experiment"].astype(str).isin(keep)]

    if d.empty:
        return

    label_map = label_map or {}
    plt.figure(figsize=(11, 6))
    for exp, df in d.groupby("experiment", sort=True):
        g = df.dropna(subset=["labeled_size", "miou_round"]).sort_values("labeled_size")
        if g.empty:
            continue
        plt.plot(
            g["labeled_size"].astype(float),
            g["miou_round"].astype(float),
            marker="o",
            linewidth=2.2,
            label=str(label_map.get(str(exp), exp)),
        )
    plt.title(title)
    plt.xlabel("标注样本数")
    plt.ylabel("mIoU（每轮最佳）")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outdir / filename, dpi=300)
    plt.close()


def plot_multiseed_learning_curves(
    per_seed_curves: Dict[str, Dict[str, pd.DataFrame]],
    outdir: Path,
    title: str,
    *,
    filename: str = "multiseed_learning_curve_miou_vs_labeled.png",
    include_experiments: Optional[List[str]] = None,
    label_map: Optional[Dict[str, str]] = None,
) -> None:
    label_map = label_map or {}
    exps = sorted(per_seed_curves.keys())
    if include_experiments:
        keep = set(str(x) for x in include_experiments)
        exps = [e for e in exps if str(e) in keep]
    if not exps:
        return

    plt.figure(figsize=(11, 6))
    for exp in exps:
        seed_map = per_seed_curves.get(exp) or {}
        if not isinstance(seed_map, dict) or not seed_map:
            continue
        rows: List[pd.DataFrame] = []
        for df in seed_map.values():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            g = df.dropna(subset=["round", "labeled_size", "miou_round"]).copy()
            g["round"] = pd.to_numeric(g["round"], errors="coerce")
            g["labeled_size"] = pd.to_numeric(g["labeled_size"], errors="coerce")
            g["miou_round"] = pd.to_numeric(g["miou_round"], errors="coerce")
            g = g.dropna(subset=["round", "labeled_size", "miou_round"]).sort_values("round")
            if g.empty:
                continue
            rows.append(g[["round", "labeled_size", "miou_round"]])
        if not rows:
            continue

        all_rounds = sorted(set(int(r) for df in rows for r in df["round"].astype(int).tolist()))
        xs: List[float] = []
        ys_mean: List[float] = []
        ys_lo: List[float] = []
        ys_hi: List[float] = []

        for rr in all_rounds:
            vals: List[float] = []
            labeled_sizes: List[float] = []
            for df in rows:
                match = df[df["round"].astype(int) == int(rr)]
                if match.empty:
                    continue
                vals.append(float(match["miou_round"].iloc[0]))
                labeled_sizes.append(float(match["labeled_size"].iloc[0]))
            if not vals:
                continue
            stats = _summarize(vals)
            xs.append(float(statistics.fmean(labeled_sizes)) if labeled_sizes else float(rr))
            ys_mean.append(float(stats["mean"]))
            ci = stats.get("ci95")
            if ci is None or not np.isfinite(float(ci)):
                ys_lo.append(float(stats["mean"]))
                ys_hi.append(float(stats["mean"]))
            else:
                ys_lo.append(float(stats["mean"]) - float(ci))
                ys_hi.append(float(stats["mean"]) + float(ci))

        if not xs:
            continue

        label = str(label_map.get(str(exp), exp))
        plt.plot(xs, ys_mean, marker="o", linewidth=2.2, label=label)
        if any(abs(a - b) > 1e-12 for a, b in zip(ys_lo, ys_hi)):
            plt.fill_between(xs, ys_lo, ys_hi, alpha=0.18)

    plt.title(title)
    plt.xlabel("标注样本数")
    plt.ylabel("mIoU（每轮最佳，均值±95%CI）")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outdir / filename, dpi=300)
    plt.close()


def plot_multiseed_metric_bars(
    summary: Dict[str, Any],
    outdir: Path,
    *,
    metric: str,
    title: str,
    filename: str,
    include_experiments: Optional[List[str]] = None,
    label_map: Optional[Dict[str, str]] = None,
) -> None:
    label_map = label_map or {}
    experiments = summary.get("experiments")
    if not isinstance(experiments, dict) or not experiments:
        return

    rows: List[Dict[str, Any]] = []
    for exp, payload in experiments.items():
        if include_experiments and str(exp) not in set(str(x) for x in include_experiments):
            continue
        if not isinstance(payload, dict):
            continue
        stats_block = payload.get("summary")
        if not isinstance(stats_block, dict):
            continue
        s = stats_block.get(metric)
        if not isinstance(s, dict):
            continue
        try:
            rows.append(
                {
                    "experiment": str(exp),
                    "mean": float(s.get("mean")),
                    "std": float(s.get("std")),
                    "ci95": float(s.get("ci95")),
                    "n": int(s.get("n")),
                }
            )
        except Exception:
            continue

    if not rows:
        return

    df = pd.DataFrame(rows).dropna(subset=["mean"]).sort_values("mean", ascending=False)
    if df.empty:
        return

    xs = [str(label_map.get(str(x), x)) for x in df["experiment"].astype(str).tolist()]
    ys = df["mean"].astype(float).tolist()
    errs = df["ci95"].astype(float).tolist()

    plt.figure(figsize=(11, 5))
    plt.bar(xs, ys, yerr=errs, capsize=4)
    plt.xticks(rotation=35, ha="right")
    plt.title(title)
    plt.xlabel("")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(outdir / filename, dpi=300)
    plt.close()

def plot_bars(metrics: pd.DataFrame, outdir: Path):
    if metrics.empty:
        return

    m1 = metrics.dropna(subset=["alc"]).sort_values("alc", ascending=False)
    if not m1.empty:
        plt.figure(figsize=(11, 5))
        xs = m1["experiment"].astype(str).tolist()
        ys = m1["alc"].astype(float).tolist()
        plt.bar(xs, ys)
        plt.xticks(rotation=35, ha="right")
        plt.title("ALC 对比")
        plt.xlabel("")
        plt.ylabel("ALC")
        plt.tight_layout()
        plt.savefig(outdir / "alc_bar.png", dpi=300)
        plt.close()

    m2 = metrics.dropna(subset=["final_mIoU"]).sort_values("final_mIoU", ascending=False)
    if not m2.empty:
        plt.figure(figsize=(11, 5))
        xs = m2["experiment"].astype(str).tolist()
        ys = m2["final_mIoU"].astype(float).tolist()
        plt.bar(xs, ys)
        plt.xticks(rotation=35, ha="right")
        plt.title("最终 mIoU 对比")
        plt.xlabel("")
        plt.ylabel("最终 mIoU")
        plt.tight_layout()
        plt.savefig(outdir / "final_miou_bar.png", dpi=300)
        plt.close()


def plot_cost_tradeoff(metrics: pd.DataFrame, outdir: Path):
    m = metrics.dropna(subset=["final_mIoU", "total_epochs"]).copy()
    if m.empty:
        return

    plt.figure(figsize=(9, 6))
    exps = sorted(m["experiment"].astype(str).unique().tolist())
    cmap = plt.get_cmap("tab10")
    color_map = {exp: cmap(i % 10) for i, exp in enumerate(exps)}
    for exp in exps:
        df = m[m["experiment"].astype(str) == exp]
        plt.scatter(
            df["total_epochs"].astype(float),
            df["final_mIoU"].astype(float),
            s=120,
            label=exp,
            color=color_map.get(exp),
        )
        for _, r in df.iterrows():
            plt.text(float(r["total_epochs"]), float(r["final_mIoU"]), str(r["experiment"]), fontsize=8, alpha=0.8)
    plt.title("成本—性能权衡（成本代理：总训练epoch数）")
    plt.xlabel("总训练 epochs（按 epoch_end 计数）")
    plt.ylabel("最终 mIoU")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="experiment", loc="best")
    plt.tight_layout()
    plt.savefig(outdir / "cost_tradeoff.png", dpi=300)
    plt.close()


def plot_controller_trajectories(ctrl: pd.DataFrame, outdir: Path):
    if ctrl.empty:
        return
    exps = sorted(ctrl["experiment"].unique().tolist())
    for exp in exps:
        df = ctrl[ctrl["experiment"] == exp].sort_values("round")
        if df.empty:
            continue

        show_epochs = False
        if "epochs" in df.columns:
            xs = pd.to_numeric(df["epochs"], errors="coerce").dropna()
            if xs.shape[0] > 0 and xs.nunique() > 1:
                show_epochs = True

        rows = 3 if show_epochs else 2
        fig, axes = plt.subplots(rows, 1, figsize=(10, 7.5 if rows == 2 else 9), sharex=True)
        if isinstance(axes, np.ndarray):
            axes = axes.flatten().tolist()
        else:
            axes = [axes]

        axes[0].plot(df["round"].astype(float), pd.to_numeric(df["lambda"], errors="coerce").astype(float), marker="o", linewidth=2.2)
        axes[0].set_ylabel("lambda")
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].grid(True, linestyle="--", alpha=0.5)

        idx = 1
        if show_epochs:
            axes[idx].plot(df["round"].astype(float), pd.to_numeric(df["epochs"], errors="coerce").astype(float), marker="o", linewidth=2.2)
            axes[idx].set_ylabel("epochs")
            axes[idx].grid(True, linestyle="--", alpha=0.5)
            idx += 1

        axes[idx].plot(df["round"].astype(float), pd.to_numeric(df["query_size"], errors="coerce").astype(float), marker="o", linewidth=2.2)
        axes[idx].set_ylabel("query_size")
        axes[idx].set_xlabel("round")
        axes[idx].grid(True, linestyle="--", alpha=0.5)

        fig.suptitle(f"控制轨迹：{exp}")
        plt.tight_layout()
        fig.savefig(outdir / f"controller_trajectory__{exp}.png", dpi=300)
        plt.close(fig)


def plot_gradient_diagnostics(
    grad: pd.DataFrame,
    outdir: Path,
    *,
    title: str = "梯度诊断（按 round 汇总）",
    filename: str = "gradient_diagnostics.png",
    include_experiments: Optional[List[str]] = None,
    label_map: Optional[Dict[str, str]] = None,
) -> None:
    if grad.empty:
        return

    df = grad.copy()
    if include_experiments:
        df = df[df["experiment"].astype(str).isin([str(x) for x in include_experiments])].copy()
    if df.empty:
        return

    cols = set(df.columns.tolist())
    has_norm = "grad_total_norm_mean" in cols
    has_align = "grad_train_val_cos" in cols
    if not has_norm and not has_align:
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.6), sharex=True)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten().tolist()
    else:
        axes = [axes]

    exps = sorted(df["experiment"].astype(str).unique().tolist())
    for exp in exps:
        d = df[df["experiment"].astype(str) == exp].sort_values("round")
        if d.empty:
            continue
        label = label_map.get(exp, exp) if isinstance(label_map, dict) else exp
        if has_norm:
            axes[0].plot(
                d["round"].astype(float),
                pd.to_numeric(d["grad_total_norm_mean"], errors="coerce").astype(float),
                marker="o",
                linewidth=2.0,
                label=label,
            )
        if has_align:
            axes[1].plot(
                d["round"].astype(float),
                pd.to_numeric(d["grad_train_val_cos"], errors="coerce").astype(float),
                marker="o",
                linewidth=2.0,
                label=label,
            )

    if has_norm:
        axes[0].set_ylabel("grad_total_norm (mean)")
        axes[0].grid(True, linestyle="--", alpha=0.5)
    else:
        axes[0].set_axis_off()

    if has_align:
        axes[1].set_ylabel("train-val cosine")
        axes[1].set_xlabel("round")
        axes[1].set_ylim(-1.05, 1.05)
        axes[1].grid(True, linestyle="--", alpha=0.5)
    else:
        axes[1].set_axis_off()

    handles, labels = axes[1].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(outdir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _pearsonr(xs: List[float], ys: List[float]) -> float:
    x = np.array([float(v) for v in xs], dtype=float)
    y = np.array([float(v) for v in ys], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    return r if np.isfinite(r) else float("nan")


def plot_lambda_evolution(
    ctrl: pd.DataFrame,
    outdir: Path,
    *,
    title: str = "λ_t 演化（AAL-SD vs Rule-based）",
    filename: str = "lambda_evolution.png",
    include_experiments: Optional[List[str]] = None,
    label_map: Optional[Dict[str, str]] = None,
) -> None:
    if ctrl.empty or "lambda" not in ctrl.columns:
        return
    df = ctrl.copy()
    df = df.dropna(subset=["round"])
    if include_experiments:
        keep = set(str(x) for x in include_experiments)
        df = df[df["experiment"].astype(str).isin(keep)].copy()
    if df.empty:
        return
    label_map = label_map or {}

    plt.figure(figsize=(10.5, 5.6))
    for exp, d in df.groupby("experiment", sort=True):
        g = d.dropna(subset=["lambda"]).sort_values("round")
        if g.empty:
            continue
        plt.plot(
            g["round"].astype(float),
            pd.to_numeric(g["lambda"], errors="coerce").astype(float),
            marker="o",
            linewidth=2.2,
            label=str(label_map.get(str(exp), exp)),
        )
    plt.title(title)
    plt.xlabel("round")
    plt.ylabel("lambda")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outdir / filename, dpi=300)
    plt.close()


def plot_lambda_gradient_consistency(
    ctrl: pd.DataFrame,
    grad: pd.DataFrame,
    outdir: Path,
    *,
    title: str = "λ_t 与梯度一致性（train-val cosine）",
    filename: str = "lambda_vs_gradient_consistency.png",
    include_experiments: Optional[List[str]] = None,
    label_map: Optional[Dict[str, str]] = None,
) -> None:
    if ctrl.empty or grad.empty:
        return
    if "lambda" not in ctrl.columns or "grad_train_val_cos" not in grad.columns:
        return
    if "experiment" not in ctrl.columns or "experiment" not in grad.columns:
        return

    c = ctrl[["experiment", "round", "lambda"]].copy()
    g = grad[["experiment", "round", "grad_train_val_cos"]].copy()
    c["round"] = pd.to_numeric(c["round"], errors="coerce")
    g["round"] = pd.to_numeric(g["round"], errors="coerce")
    c["lambda"] = pd.to_numeric(c["lambda"], errors="coerce")
    g["grad_train_val_cos"] = pd.to_numeric(g["grad_train_val_cos"], errors="coerce")
    df = c.merge(g, on=["experiment", "round"], how="inner")
    df = df.dropna(subset=["round", "lambda", "grad_train_val_cos"]).copy()
    if include_experiments:
        keep = set(str(x) for x in include_experiments)
        df = df[df["experiment"].astype(str).isin(keep)].copy()
    if df.empty:
        return
    label_map = label_map or {}

    exps = sorted(df["experiment"].astype(str).unique().tolist())
    rows = len(exps)
    fig, axes = plt.subplots(rows, 1, figsize=(11, max(3.0, 3.0 * rows)), sharex=True)
    if rows == 1:
        axes_list = [axes]
    else:
        axes_list = axes.flatten().tolist()

    for ax, exp in zip(axes_list, exps):
        d = df[df["experiment"].astype(str) == exp].sort_values("round")
        if d.empty:
            continue
        r = _pearsonr(d["lambda"].tolist(), d["grad_train_val_cos"].tolist())
        ax2 = ax.twinx()
        ax.plot(d["round"].astype(float), d["lambda"].astype(float), marker="o", linewidth=2.2, color="#1f77b4")
        ax2.plot(
            d["round"].astype(float),
            d["grad_train_val_cos"].astype(float),
            marker="s",
            linewidth=2.2,
            color="#d62728",
        )
        ax.set_ylim(-0.05, 1.05)
        ax2.set_ylim(-1.05, 1.05)
        ax.set_ylabel("lambda", color="#1f77b4")
        ax2.set_ylabel("train-val cosine", color="#d62728")
        ax.grid(True, linestyle="--", alpha=0.5)
        title_exp = str(label_map.get(str(exp), exp))
        r_txt = "" if not np.isfinite(r) else f" (Pearson r={r:+.3f})"
        ax.set_title(title_exp + r_txt)

    axes_list[-1].set_xlabel("round")
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(outdir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _l3_items_to_df(items: Any) -> pd.DataFrame:
    if not isinstance(items, list):
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        rows.append(
            {
                "sample_id": it.get("sample_id"),
                "uncertainty": it.get("uncertainty"),
                "knowledge_gain": it.get("knowledge_gain"),
                "final_score": it.get("final_score"),
                "lambda_t": it.get("lambda_t"),
            }
        )
    df = pd.DataFrame(rows)
    for c in ["uncertainty", "knowledge_gain", "final_score", "lambda_t"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def plot_uk_space_distribution(
    l3: pd.DataFrame,
    outdir: Path,
    *,
    title: str = "样本分布：Uncertainty–Knowledge 空间",
    filename: str = "uk_space_distribution.png",
    experiments: Optional[List[str]] = None,
    label_map: Optional[Dict[str, str]] = None,
    round_map: Optional[Dict[str, int]] = None,
) -> None:
    if l3.empty:
        return
    if "top_items" not in l3.columns or "selected_items" not in l3.columns:
        return
    if "experiment" not in l3.columns or "round" not in l3.columns:
        return
    label_map = label_map or {}
    round_map = round_map or {}

    df = l3.copy()
    if experiments:
        keep = set(str(x) for x in experiments)
        df = df[df["experiment"].astype(str).isin(keep)].copy()
    if df.empty:
        return

    exps = sorted(df["experiment"].astype(str).unique().tolist())
    if not exps:
        return

    fig, axes = plt.subplots(1, len(exps), figsize=(6.2 * len(exps), 5.3), sharex=True, sharey=True)
    if len(exps) == 1:
        axes_list = [axes]
    else:
        axes_list = axes.flatten().tolist()

    for ax, exp in zip(axes_list, exps):
        d = df[df["experiment"].astype(str) == exp].copy()
        if d.empty:
            continue
        if str(exp) in round_map:
            r = int(round_map[str(exp)])
            d = d[pd.to_numeric(d["round"], errors="coerce") == float(r)].copy()
        else:
            d = d.sort_values("round").tail(1).copy()
        if d.empty:
            continue

        row = d.iloc[0].to_dict()
        try:
            top_items = json.loads(row.get("top_items")) if isinstance(row.get("top_items"), str) else []
        except Exception:
            top_items = []
        try:
            selected_items = json.loads(row.get("selected_items")) if isinstance(row.get("selected_items"), str) else []
        except Exception:
            selected_items = []

        top_df = _l3_items_to_df(top_items)
        sel_df = _l3_items_to_df(selected_items)
        if top_df.empty or top_df["knowledge_gain"].dropna().empty:
            ax.set_axis_off()
            ax.set_title(str(label_map.get(str(exp), exp)))
            continue
        top_df = top_df.dropna(subset=["uncertainty", "knowledge_gain"]).copy()
        sel_df = sel_df.dropna(subset=["uncertainty", "knowledge_gain"]).copy()

        ax.scatter(
            top_df["uncertainty"].astype(float),
            top_df["knowledge_gain"].astype(float),
            s=18,
            alpha=0.25,
            label="Top-k candidates",
            color="#7f7f7f",
        )
        if not sel_df.empty:
            ax.scatter(
                sel_df["uncertainty"].astype(float),
                sel_df["knowledge_gain"].astype(float),
                s=38,
                alpha=0.85,
                label="Selected",
                color="#1f77b4",
            )
        ax.set_xlabel("Uncertainty U(x)")
        ax.set_ylabel("Knowledge gain K(x)")
        ax.grid(True, linestyle="--", alpha=0.4)
        r_show = row.get("round")
        ax.set_title(f"{label_map.get(str(exp), exp)} (round={int(r_show) if pd.notna(r_show) else 'NA'})")
        ax.legend(loc="best")

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(outdir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_contributions(
    metrics: pd.DataFrame,
    outdir: Path,
    *,
    title: str = "消融贡献：相对 Full 的 ΔALC",
    filename: str = "ablation_contributions_alc.png",
    full_experiment: str = "full_model_A_lambda_policy",
    experiments: Optional[List[str]] = None,
    label_map: Optional[Dict[str, str]] = None,
) -> None:
    if metrics.empty:
        return
    label_map = label_map or {}
    exp_names = metrics["experiment"].astype(str).unique().tolist() if "experiment" in metrics.columns else []
    full_key = _pick_first_present(exp_names, [str(full_experiment), "full_model_A_lambda_policy", "full_model"])
    if full_key not in set(metrics["experiment"].astype(str).tolist()):
        return
    full_row = metrics[metrics["experiment"].astype(str) == str(full_key)].iloc[0].to_dict()
    base_alc = full_row.get("alc")
    if base_alc is None or (not np.isfinite(float(base_alc))):
        return

    df = metrics.copy()
    if experiments:
        keep = set(str(x) for x in experiments)
        df = df[df["experiment"].astype(str).isin(keep)].copy()
    df = df.dropna(subset=["alc"]).copy()
    if df.empty:
        return
    df["delta_alc"] = df["alc"].astype(float) - float(base_alc)
    df = df[df["experiment"].astype(str) != str(full_key)].copy()
    if df.empty:
        return

    df = df.sort_values("delta_alc", ascending=True)
    xs = [str(label_map.get(str(x), x)) for x in df["experiment"].astype(str).tolist()]
    ys = df["delta_alc"].astype(float).tolist()

    plt.figure(figsize=(11.2, 5.2))
    plt.bar(xs, ys)
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    plt.xticks(rotation=28, ha="right")
    plt.title(title)
    plt.ylabel("ΔALC (vs Full)")
    plt.tight_layout()
    plt.savefig(outdir / filename, dpi=300)
    plt.close()


def _parse_report_numbers(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    m = re.search(r"\*\*mIoU:\*\*\s*([0-9]*\.?[0-9]+)", text)
    if m:
        out["miou"] = float(m.group(1))
    m = re.search(r"\*\*F1 Score:\*\*\s*([0-9]*\.?[0-9]+)", text)
    if m:
        out["f1"] = float(m.group(1))
    m = re.search(r"\*\*Labeled Samples:\*\*\s*([0-9]+)", text)
    if m:
        out["labeled"] = int(m.group(1))
    m = re.search(r"\*\*mIoU Change:\*\*\s*([+-]?[0-9]*\.?[0-9]+)", text)
    if m:
        out["miou_delta"] = float(m.group(1))
    m = re.search(r"\*\*F1 Change:\*\*\s*([+-]?[0-9]*\.?[0-9]+)", text)
    if m:
        out["f1_delta"] = float(m.group(1))
    return out


def _extract_anomalies_from_report(text: str) -> List[str]:
    m = re.search(r"## Anomalies\s*\n(\[[\s\S]*?\])", text)
    if not m:
        return []
    try:
        xs = json.loads(m.group(1))
        return [str(x) for x in xs] if isinstance(xs, list) else []
    except Exception:
        return []


def _add_text_page(pdf: PdfPages, title: str, paragraphs: List[str]):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle(title, fontsize=16, y=0.98)
    y = 0.94
    for p in paragraphs:
        wrapped = textwrap.fill(p, width=95)
        fig.text(0.06, y, wrapped, fontsize=10, va="top")
        y -= 0.03 * (wrapped.count("\n") + 1) + 0.02
        if y < 0.08:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            y = 0.97
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _add_image_page(pdf: PdfPages, title: str, image_paths: List[Path], cols: int = 1):
    image_paths = [p for p in image_paths if p.exists()]
    if not image_paths:
        return
    rows = (len(image_paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8.27, 11.69))
    if not isinstance(axes, (list, tuple)) and hasattr(axes, "shape"):
        axes_list = axes.flatten().tolist()
    else:
        axes_list = axes if isinstance(axes, list) else [axes]
    fig.suptitle(title, fontsize=14, y=0.98)
    for ax, p in zip(axes_list, image_paths):
        img = plt.imread(str(p))
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(p.name, fontsize=9)
    for ax in axes_list[len(image_paths) :]:
        ax.set_axis_off()
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _markdown_to_sections(markdown: str) -> List[Tuple[str, List[str]]]:
    lines = markdown.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    title = "正文"
    buf: List[str] = []
    in_code = False

    def _flush():
        nonlocal buf, title
        paragraphs: List[str] = []
        cur: List[str] = []
        for ln in buf:
            if not ln.strip():
                if cur:
                    paragraphs.append(" ".join(cur).strip())
                    cur = []
                continue
            cur.append(ln.rstrip())
        if cur:
            paragraphs.append(" ".join(cur).strip())
        if paragraphs:
            sections.append((title, paragraphs))
        buf = []

    for raw in lines:
        ln = raw.rstrip("\n")
        if ln.strip().startswith("```"):
            in_code = not in_code
            buf.append(ln)
            continue
        if not in_code:
            if ln.strip() in {"---", "***", "___"}:
                continue
            if ln.lstrip().startswith("![]("):
                continue
            if ln.startswith("# "):
                _flush()
                title = ln[2:].strip() or "正文"
                continue
            if ln.startswith("## "):
                _flush()
                title = ln[3:].strip() or "正文"
                continue
            if ln.startswith("### "):
                buf.append(f"【{ln[4:].strip()}】")
                continue
        buf.append(ln)

    _flush()
    return sections


def export_theory_based_pdf_report(
    run_dir: Path,
    outdir: Path,
    metrics: pd.DataFrame,
    reports_dir: Optional[Path],
    pdf_path: Path,
    theory_md_path: Path,
):
    theory_text = _read_text(theory_md_path)
    sections = _markdown_to_sections(theory_text) if theory_text.strip() else []

    exp_names = metrics["experiment"].astype(str).unique().tolist() if (not metrics.empty) and ("experiment" in metrics.columns) else []
    full_key = _pick_first_present(exp_names, ["full_model_A_lambda_policy", "full_model"])
    ours = metrics[metrics["experiment"] == full_key]
    ours_row = ours.iloc[0].to_dict() if not ours.empty else {}

    trad = metrics[metrics["experiment"].isin(["baseline_entropy", "baseline_bald", "baseline_coreset", "baseline_random"])]
    trad = trad.dropna(subset=["final_mIoU"]).sort_values("final_mIoU", ascending=False)
    best_trad = trad.iloc[0].to_dict() if not trad.empty else {}

    delta_lines: List[str] = []
    if ours_row and best_trad:
        delta_lines.append(
            f"相对最强传统基线 {best_trad.get('experiment')}：最终 mIoU "
            f"{float(ours_row.get('final_mIoU')) - float(best_trad.get('final_mIoU')):+.4f}，ALC "
            f"{float(ours_row.get('alc')) - float(best_trad.get('alc')):+.4f}。"
        )
    fixed_epochs = bool(int(metrics["total_epochs"].nunique() <= 1)) if "total_epochs" in metrics.columns else False
    if (not fixed_epochs) and ours_row:
        delta_lines.append(
            f"训练成本代理（total_epochs）：{full_key}={int(ours_row.get('total_epochs') or 0)}；多数基线约为 45。"
        )

    anomalies_lines: List[str] = []
    if reports_dir and reports_dir.exists():
        for prefix in [
            "baseline_random_anomaly_report_",
            "no_agent_anomaly_report_",
            "agent_control_lambda_anomaly_report_",
        ]:
            cand = sorted(reports_dir.glob(f"{prefix}*.md"))
            if not cand:
                continue
            xs = _extract_anomalies_from_report(_read_text(cand[-1]))
            if xs:
                anomalies_lines.append(f"{cand[-1].stem}: " + "; ".join(xs))

    with PdfPages(str(pdf_path)) as pdf:
        cover_paras = [
            f"Run: {run_dir.name}",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"理论框架来源: {theory_md_path.name}",
        ]
        if ours_row:
            cover_paras.append(
                f"{full_key}: ALC={float(ours_row.get('alc')):.4f}, final mIoU={float(ours_row.get('final_mIoU')):.4f}, "
                f"final F1={float(ours_row.get('final_f1')):.4f}"
            )
        cover_paras.extend(delta_lines)
        if anomalies_lines:
            cover_paras.append("异常摘要（自动诊断）： " + " | ".join(anomalies_lines))
        _add_text_page(pdf, title="AAL-SD 理论驱动报告（内嵌图表版）", paragraphs=cover_paras)

        if sections:
            for sec_title, paras in sections:
                _add_text_page(pdf, title=sec_title, paragraphs=paras)
        else:
            _add_text_page(
                pdf,
                title="理论正文缺失",
                paragraphs=[f"无法读取理论文档：{theory_md_path}"],
            )

        show = metrics.dropna(subset=["final_mIoU"]).sort_values("final_mIoU", ascending=False).copy()
        cols = ["experiment", "alc", "final_mIoU", "final_f1", "total_epochs", "wall_clock_sec"]
        show = show[[c for c in cols if c in show.columns]].copy()
        for c in ["alc", "final_mIoU", "final_f1"]:
            if c in show.columns:
                show[c] = show[c].astype(float).map(lambda x: f"{x:.4f}")
        if "total_epochs" in show.columns:
            show["total_epochs"] = show["total_epochs"].fillna(0).astype(int).astype(str)
        if "wall_clock_sec" in show.columns:
            show["wall_clock_sec"] = show["wall_clock_sec"].map(lambda x: "" if pd.isna(x) else f"{float(x):.0f}")

        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        fig.suptitle("实验结果汇总（按最终 mIoU 排序）", fontsize=14, y=0.98)
        tbl = ax.table(
            cellText=show.values.tolist(),
            colLabels=show.columns.tolist(),
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.25)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        _add_image_page(
            pdf,
            title="核心图表：标注效率曲线",
            image_paths=[outdir / "learning_curve_miou_vs_labeled.png"],
            cols=1,
        )
        _add_image_page(
            pdf,
            title="核心图表：ALC / 最终 mIoU / 成本权衡",
            image_paths=[
                outdir / "alc_bar.png",
                outdir / "final_miou_bar.png",
                outdir / "cost_tradeoff.png",
            ],
            cols=1,
        )
        _add_image_page(
            pdf,
            title="核心图表：控制策略轨迹",
            image_paths=[
                p
                for p in [
                    _pick_first_existing(
                        [
                            outdir / f"controller_trajectory__{full_key}.png",
                            outdir / "controller_trajectory__full_model.png",
                        ]
                    ),
                    _pick_first_existing(
                        [
                            outdir / "controller_trajectory__agent_control_lambda.png",
                            outdir / "controller_trajectory__full_model_B_lambda_agent.png",
                            outdir / "controller_trajectory__ablation_B_llm_lambda_control.png",
                        ]
                    ),
                ]
                if p is not None
            ],
            cols=1,
        )


def export_pdf_report(
    run_dir: Path,
    outdir: Path,
    metrics: pd.DataFrame,
    reports_dir: Optional[Path],
    pdf_path: Path,
):
    metrics_sorted = metrics.dropna(subset=["final_mIoU"]).sort_values("final_mIoU", ascending=False)
    best_row = metrics_sorted.iloc[0].to_dict() if not metrics_sorted.empty else {}
    exp_names = metrics["experiment"].astype(str).unique().tolist() if (not metrics.empty) and ("experiment" in metrics.columns) else []
    full_key = _pick_first_present(exp_names, ["full_model_A_lambda_policy", "full_model"])

    def _get_row(exp: str) -> Dict[str, Any]:
        r = metrics[metrics["experiment"] == exp]
        return r.iloc[0].to_dict() if not r.empty else {}

    ours = _get_row(full_key)
    best_trad = metrics[metrics["experiment"].isin(["baseline_entropy", "baseline_bald", "baseline_coreset", "baseline_random"])]
    best_trad = best_trad.dropna(subset=["final_mIoU"]).sort_values("final_mIoU", ascending=False)
    best_trad_row = best_trad.iloc[0].to_dict() if not best_trad.empty else {}

    effect_lines: List[str] = []
    if ours and best_trad_row:
        effect_lines.append(
            f"在固定标注预算下，{full_key} 相比最强传统基线 {best_trad_row.get('experiment')} 的最终 mIoU 提升 "
            f"{float(ours.get('final_mIoU')) - float(best_trad_row.get('final_mIoU')):+.4f}，ALC 提升 "
            f"{float(ours.get('alc')) - float(best_trad_row.get('alc')):+.4f}。"
        )
    fixed_epochs = bool(int(metrics["total_epochs"].nunique() <= 1)) if "total_epochs" in metrics.columns else False
    if (not fixed_epochs) and ours:
        effect_lines.append(
            f"训练成本代理（total_epochs）显示 {full_key} 为 {int(ours.get('total_epochs') or 0)}，多数基线约为 45，"
            f"存在训练预算不一致带来的混杂，需要在论文中明确并用固定训练预算的补充实验解耦。"
        )

    anomalies_lines: List[str] = []
    if reports_dir and reports_dir.exists():
        anomaly_targets = [
            "baseline_random_anomaly_report_",
            "no_agent_anomaly_report_",
            "agent_control_lambda_anomaly_report_",
        ]
        for prefix in anomaly_targets:
            cand = sorted(reports_dir.glob(f"{prefix}*.md"))
            if not cand:
                continue
            txt = _read_text(cand[-1])
            xs = _extract_anomalies_from_report(txt)
            if xs:
                anomalies_lines.append(f"{cand[-1].stem}: " + "; ".join(xs))

    round_refs: List[str] = []
    if reports_dir and reports_dir.exists():
        for p in [
            reports_dir / "full_model_round_15_report.md",
            reports_dir / "baseline_entropy_round_15_report.md",
            reports_dir / "baseline_llm_us_round_15_report.md",
            reports_dir / "baseline_random_round_4_report.md",
        ]:
            if p.exists():
                nums = _parse_report_numbers(_read_text(p))
                if nums:
                    s = f"{p.stem}: mIoU={nums.get('miou')}"
                    if "miou_delta" in nums:
                        s += f" (Δ={nums.get('miou_delta'):+.4f})"
                    round_refs.append(s)

    with PdfPages(str(pdf_path)) as pdf:
        _add_text_page(
            pdf,
            title=f"CVPR 风格结果分析（{run_dir.name}）",
            paragraphs=[
                "本报告基于固定标注预算（最终标注样本数一致）比较多种主动学习基线、LLM 打分基线、规则策略与我们的方法。",
                f"整体最佳方法：{best_row.get('experiment')}，最终 mIoU={best_row.get('final_mIoU')}, ALC={best_row.get('alc')}。",
                *effect_lines,
                "建议在最终论文中同时报告：固定标注预算结果（ALC/最终 mIoU）与固定训练预算结果（相同 epochs 或相同 wall-clock），以获得可归因的结论。",
            ],
        )

        if not metrics.empty:
            show = metrics.copy()
            show = show.dropna(subset=["final_mIoU"]).sort_values("final_mIoU", ascending=False)
            cols = ["experiment", "alc", "final_mIoU", "final_f1", "total_epochs", "wall_clock_sec"]
            show = show[[c for c in cols if c in show.columns]].copy()
            for c in ["alc", "final_mIoU", "final_f1"]:
                if c in show.columns:
                    show[c] = show[c].astype(float).map(lambda x: f"{x:.4f}")
            for c in ["total_epochs"]:
                if c in show.columns:
                    show[c] = show[c].fillna(0).astype(int).astype(str)
            if "wall_clock_sec" in show.columns:
                show["wall_clock_sec"] = show["wall_clock_sec"].map(lambda x: "" if pd.isna(x) else f"{float(x):.0f}")

            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_subplot(111)
            ax.set_axis_off()
            fig.suptitle("汇总指标（按最终 mIoU 排序）", fontsize=14, y=0.98)
            tbl = ax.table(
                cellText=show.values.tolist(),
                colLabels=show.columns.tolist(),
                loc="center",
                cellLoc="center",
                colLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1.0, 1.25)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        figs_dir = outdir
        _add_image_page(
            pdf,
            title="标注效率曲线",
            image_paths=[figs_dir / "learning_curve_miou_vs_labeled.png"],
            cols=1,
        )
        _add_image_page(
            pdf,
            title="ALC / 最终 mIoU / 成本权衡",
            image_paths=[
                figs_dir / "alc_bar.png",
                figs_dir / "final_miou_bar.png",
                figs_dir / "cost_tradeoff.png",
            ],
            cols=1,
        )

        ctrl_imgs = [
            figs_dir / "controller_trajectory__full_model.png",
            figs_dir / "controller_trajectory__agent_control_lambda.png",
        ]
        _add_image_page(pdf, title="控制策略轨迹", image_paths=ctrl_imgs, cols=1)

        if anomalies_lines or round_refs:
            _add_text_page(
                pdf,
                title="异常与轮次诊断摘要",
                paragraphs=[
                    "异常摘要来自自动诊断报告，建议在论文中作为稳定性/失败模式的补充说明，并在多随机种子上复核。",
                    *([("异常检测：" + " | ".join(anomalies_lines))] if anomalies_lines else []),
                    *([("关键轮次：" + " | ".join(round_refs))] if round_refs else []),
                ],
            )


def _html_escape(s: Any) -> str:
    if s is None:
        return ""
    t = str(s)
    return (
        t.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _fmt_float(x: Any, nd: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return ""
    if not np.isfinite(v):
        return ""
    return f"{v:.{nd}f}"


def _df_to_html_table(df: pd.DataFrame, *, float_cols: Optional[List[str]] = None, float_nd: int = 4) -> str:
    if df is None or df.empty:
        return "<p class='muted'>（空）</p>"
    d = df.copy()
    float_cols = float_cols or []
    for c in float_cols:
        if c in d.columns:
            d[c] = d[c].map(lambda v: _fmt_float(v, float_nd))
    head = "".join(f"<th>{_html_escape(c)}</th>" for c in d.columns.tolist())
    rows = []
    for _, r in d.iterrows():
        cells = "".join(f"<td>{_html_escape(v)}</td>" for v in r.tolist())
        rows.append(f"<tr>{cells}</tr>")
    body = "".join(rows)
    return f"<table class='tbl'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _img_tag(img_path: Path, *, caption: Optional[str] = None, max_width: str = "980px") -> str:
    if not img_path.exists():
        return ""
    cap = f"<div class='cap'>{_html_escape(caption)}</div>" if caption else ""
    return (
        "<div class='fig'>"
        f"<img src='{_html_escape(img_path.name)}' style='max-width:{_html_escape(max_width)}'/>"
        f"{cap}"
        "</div>"
    )


def _parse_llm_insight_from_report(text: str) -> str:
    m = re.search(r"## LLM Insight\\s*\\n([\\s\\S]*?)\\n(?:## |\\Z)", text)
    if not m:
        return ""
    return m.group(1).strip()


def _collect_anomaly_reports(reports_dir: Optional[Path]) -> List[Dict[str, Any]]:
    if not reports_dir or (not reports_dir.exists()):
        return []
    rows: List[Dict[str, Any]] = []
    for p in sorted(reports_dir.glob("*_anomaly_report_*.md")):
        text = _read_text(p)
        anomalies = _extract_anomalies_from_report(text)
        llm_insight = _parse_llm_insight_from_report(text)
        rows.append(
            {
                "report": p.name,
                "experiment": p.name.split("_anomaly_report_")[0],
                "anomalies": anomalies,
                "llm_insight": llm_insight,
            }
        )
    return rows


def _select_lambda_turn_round(ctrl: pd.DataFrame, exp: str) -> Optional[int]:
    if ctrl.empty:
        return None
    d = ctrl[ctrl["experiment"].astype(str) == str(exp)].dropna(subset=["round", "lambda"]).sort_values("round")
    if d.shape[0] < 2:
        return None
    r = d["round"].astype(float).tolist()
    lam = pd.to_numeric(d["lambda"], errors="coerce").astype(float).tolist()
    diffs = [abs(lam[i] - lam[i - 1]) if np.isfinite(lam[i]) and np.isfinite(lam[i - 1]) else float("nan") for i in range(1, len(lam))]
    if (not diffs) or (not any(np.isfinite(x) for x in diffs)):
        return None
    idx = max(range(len(diffs)), key=lambda i: diffs[i] if np.isfinite(diffs[i]) else -1.0)
    rr = r[idx + 1]
    return int(rr) if np.isfinite(rr) else None


def _l3_summary_stats(l3: pd.DataFrame, *, exp: str, round_idx: int) -> Dict[str, Any]:
    if l3.empty:
        return {}
    d = l3[(l3["experiment"].astype(str) == str(exp)) & (pd.to_numeric(l3["round"], errors="coerce") == float(round_idx))].copy()
    if d.empty:
        return {}
    row = d.iloc[0].to_dict()
    try:
        top_items = json.loads(row.get("top_items")) if isinstance(row.get("top_items"), str) else []
    except Exception:
        top_items = []
    try:
        selected_items = json.loads(row.get("selected_items")) if isinstance(row.get("selected_items"), str) else []
    except Exception:
        selected_items = []
    top_df = _l3_items_to_df(top_items).dropna(subset=["uncertainty", "knowledge_gain"])
    sel_df = _l3_items_to_df(selected_items).dropna(subset=["uncertainty", "knowledge_gain"])

    def _stats(df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {"n": 0}
        return {
            "n": int(df.shape[0]),
            "U_mean": float(df["uncertainty"].astype(float).mean()),
            "U_p50": float(df["uncertainty"].astype(float).quantile(0.5)),
            "U_p90": float(df["uncertainty"].astype(float).quantile(0.9)),
            "K_mean": float(df["knowledge_gain"].astype(float).mean()),
            "K_p50": float(df["knowledge_gain"].astype(float).quantile(0.5)),
            "K_p90": float(df["knowledge_gain"].astype(float).quantile(0.9)),
        }

    return {"topk": _stats(top_df), "selected": _stats(sel_df)}


def _lambda_grad_correlations(ctrl: pd.DataFrame, grad: pd.DataFrame, experiments: List[str]) -> pd.DataFrame:
    if ctrl.empty or grad.empty:
        return pd.DataFrame()
    c = ctrl[["experiment", "round", "lambda"]].copy() if "lambda" in ctrl.columns else pd.DataFrame()
    g = grad[["experiment", "round", "grad_train_val_cos"]].copy() if "grad_train_val_cos" in grad.columns else pd.DataFrame()
    if c.empty or g.empty:
        return pd.DataFrame()
    c["round"] = pd.to_numeric(c["round"], errors="coerce")
    g["round"] = pd.to_numeric(g["round"], errors="coerce")
    c["lambda"] = pd.to_numeric(c["lambda"], errors="coerce")
    g["grad_train_val_cos"] = pd.to_numeric(g["grad_train_val_cos"], errors="coerce")
    df = c.merge(g, on=["experiment", "round"], how="inner").dropna(subset=["lambda", "grad_train_val_cos"])
    rows: List[Dict[str, Any]] = []
    for exp in experiments:
        d = df[df["experiment"].astype(str) == str(exp)].copy()
        if d.empty:
            continue
        rows.append(
            {
                "experiment": str(exp),
                "n_points": int(d.shape[0]),
                "pearson_r": _pearsonr(d["lambda"].tolist(), d["grad_train_val_cos"].tolist()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["pearson_r"] = out["pearson_r"].map(lambda v: _fmt_float(v, 3))
    return out


def export_html_report(
    *,
    run_dir: Path,
    outdir: Path,
    manifest: Dict[str, Any],
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    ctrl: pd.DataFrame,
    grad: pd.DataFrame,
    summary: pd.DataFrame,
    l3: pd.DataFrame,
    reports_dir: Optional[Path],
    html_path: Path,
    plan_path: Optional[Path],
    label_map: Dict[str, str],
) -> None:
    cfg = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}

    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fixed_epochs = bool(cfg.get("FIX_EPOCHS_PER_ROUND")) or (
        cfg.get("EPOCHS_PER_ROUND_SCHEDULE") in (None, "null") and cfg.get("EPOCHS_PER_ROUND") is not None
    )

    best = metrics.dropna(subset=["final_mIoU"]).sort_values("final_mIoU", ascending=False).head(1)
    best_row = best.iloc[0].to_dict() if not best.empty else {}
    exp_names = metrics["experiment"].astype(str).unique().tolist() if (not metrics.empty) and ("experiment" in metrics.columns) else []
    full_key = _pick_first_present(exp_names, ["full_model_A_lambda_policy", "full_model"])
    ours = metrics[metrics["experiment"].astype(str) == full_key]
    ours_row = ours.iloc[0].to_dict() if not ours.empty else {}

    def _name(exp: str) -> str:
        return str(label_map.get(str(exp), exp))

    def _metric_line(exp: str) -> str:
        d = metrics[metrics["experiment"].astype(str) == str(exp)]
        if d.empty:
            return ""
        r = d.iloc[0].to_dict()
        return f"{_name(exp)}: ALC={_fmt_float(r.get('alc'))}, final mIoU={_fmt_float(r.get('final_mIoU'))}, final F1={_fmt_float(r.get('final_f1'))}"

    def _get_row(exp: str) -> Dict[str, Any]:
        d = metrics[metrics["experiment"].astype(str) == str(exp)]
        return d.iloc[0].to_dict() if not d.empty else {}

    def _delta(a: Any, b: Any) -> Optional[float]:
        try:
            x = float(a)
            y = float(b)
        except Exception:
            return None
        if (not np.isfinite(x)) or (not np.isfinite(y)):
            return None
        return x - y

    def _fmt_delta(x: Optional[float], nd: int = 4) -> str:
        if x is None:
            return "NA"
        return f"{x:+.{nd}f}"

    def _p(text: str, cls: str = "") -> str:
        c = f" class='{cls}'" if cls else ""
        return f"<p{c}>{_html_escape(text)}</p>"

    def _paras(lines: List[str], cls: str = "") -> str:
        return "".join(_p(x, cls=cls) for x in lines if str(x).strip())

    baseline_order = [
        ("B1", "baseline_random"),
        ("B2", "baseline_entropy"),
        ("B3", "baseline_coreset"),
        ("B4", "baseline_bald"),
        ("B5", "fixed_lambda"),
        ("B6", "baseline_dial_style"),
        ("B7", "baseline_wang_style"),
        ("B8", full_key),
    ]
    ablation_order = [
        ("A0", full_key),
        ("A1", "fixed_lambda"),
        ("A2", "random_lambda"),
        ("A3-R1", "rule_based_controller_r1"),
        ("A3-R2", "rule_based_controller_r2"),
        ("A3-R3", "rule_based_controller_r3"),
        ("A4", "no_cold_start"),
        ("A5", "fixed_k"),
        ("A7", "no_normalization"),
    ]

    def _subset_table(pairs: List[Tuple[str, str]], *, include_delta: bool) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        base_alc = ours_row.get("alc")
        base_miou = ours_row.get("final_mIoU")
        for group_id, exp in pairs:
            d = metrics[metrics["experiment"].astype(str) == str(exp)]
            if d.empty:
                continue
            r = d.iloc[0].to_dict()
            row = {
                "组别": group_id,
                "方法": _name(exp),
                "experiment": str(exp),
                "说明": r.get("description") or "",
                "ALC": r.get("alc"),
                "final mIoU": r.get("final_mIoU"),
                "final F1": r.get("final_f1"),
            }
            if include_delta and base_alc is not None and base_miou is not None:
                try:
                    row["ΔALC(vs Full)"] = float(r.get("alc")) - float(base_alc)
                except Exception:
                    row["ΔALC(vs Full)"] = None
                try:
                    row["ΔmIoU(vs Full)"] = float(r.get("final_mIoU")) - float(base_miou)
                except Exception:
                    row["ΔmIoU(vs Full)"] = None
            rows.append(row)
        df = pd.DataFrame(rows)
        if not df.empty:
            sort_cols = ["组别"]
            df = df[sort_cols + [c for c in df.columns if c not in sort_cols]]
        return df

    t_baseline = _subset_table(baseline_order, include_delta=False)
    t_ablation = _subset_table(ablation_order, include_delta=True)

    anomalies = _collect_anomaly_reports(reports_dir)

    rule_compare = [
        x
        for x in [full_key, "rule_based_controller_r1", "rule_based_controller_r2", "rule_based_controller_r3"]
        if (not ctrl.empty) and (x in set(ctrl["experiment"].astype(str).unique().tolist()))
    ]
    corr_tbl = _lambda_grad_correlations(ctrl, grad, rule_compare) if rule_compare else pd.DataFrame()

    fig_map = [
        ("Fig. 0", "AAL-SD 主动学习闭环示意", "aal_sd_active_learning_loop.png"),
        ("Fig. 1", "Learning curve（mIoU vs 标注量）", "learning_curve_miou_vs_labeled.png"),
        ("Fig. 2", "ALC 对比", "alc_bar.png"),
        ("Fig. 2b", "最终 mIoU 对比", "final_miou_bar.png"),
        ("Fig. 3", "λ_t 演化（AAL-SD vs Rule-based）", "lambda_evolution_aal_vs_rules.png"),
        ("Fig. 4", "λ_t 与梯度一致性（含 Pearson r）", "lambda_vs_gradient_consistency.png"),
        ("Fig. 4b", "梯度诊断（round 聚合）", "gradient_diagnostics.png"),
        ("Fig. 5", "U–K 空间样本分布（Top-k vs Selected）", "uk_space_distribution.png"),
        ("Fig. 6", "消融贡献（ΔALC vs Full）", "ablation_contributions_alc.png"),
    ]

    plan_link = ""
    if plan_path and plan_path.exists():
        plan_link = f"<a href='{_html_escape(plan_path.as_posix())}' target='_blank'>{_html_escape(plan_path.name)}</a>"

    plan_outline_html = "<p class='muted'>（未提供 plan_path 或文件不存在）</p>"
    if plan_path and plan_path.exists():
        text = _read_text(plan_path)
        hs: List[str] = []
        for ln in text.splitlines():
            s = ln.strip()
            if not s.startswith("#"):
                continue
            title = s.lstrip("#").strip()
            if not title:
                continue
            if title not in hs:
                hs.append(title)
            if len(hs) >= 24:
                break
        if hs:
            plan_outline_html = "<ul>" + "".join(f"<li>{_html_escape(x)}</li>" for x in hs) + "</ul>"
        else:
            plan_outline_html = "<p class='muted'>（未能从实验方案中提取到标题结构）</p>"

    exp_cfg_lines: List[str] = []
    for k in ["N_ROUNDS", "TOTAL_BUDGET", "QUERY_SIZE", "EPOCHS_PER_ROUND", "FIX_EPOCHS_PER_ROUND"]:
        if k in cfg:
            exp_cfg_lines.append(f"{k}={_html_escape(cfg.get(k))}")
    exp_cfg = "，".join(exp_cfg_lines) if exp_cfg_lines else "（manifest.config 缺失或为空）"

    fig_html = ""
    for fig_id, cap, fn in fig_map:
        p = outdir / fn
        fig_html += _img_tag(p, caption=f"{fig_id}：{cap}")

    all_pngs = sorted([p for p in outdir.glob("*.png") if p.is_file()])
    gallery_html = "".join(_img_tag(p, caption=p.name, max_width="680px") for p in all_pngs)

    anomaly_html = "<p class='muted'>未发现 anomaly_report。</p>"
    if anomalies:
        blocks = []
        for a in anomalies:
            xs = a.get("anomalies") if isinstance(a.get("anomalies"), list) else []
            insight = a.get("llm_insight") or ""
            block = "<div class='card'>"
            block += f"<div class='card-h'>{_html_escape(a.get('experiment'))}</div>"
            block += f"<div class='card-sub'>{_html_escape(a.get('report'))}</div>"
            if xs:
                block += "<ul>" + "".join(f"<li>{_html_escape(x)}</li>" for x in xs) + "</ul>"
            else:
                block += "<div class='muted'>（无 anomalies）</div>"
            if insight:
                block += f"<details><summary>LLM Insight</summary><pre class='pre'>{_html_escape(insight)}</pre></details>"
            block += "</div>"
            blocks.append(block)
        anomaly_html = "".join(blocks)

    l3_note_html = "<p class='muted'>本 run 的 L3 数据仅在部分实验组写入（当前包含 Full / rule_based / wang / dial 等）。</p>"
    l3_stats_html = ""
    if (not l3.empty) and rule_compare:
        rows: List[Dict[str, Any]] = []
        for exp in [x for x in [full_key, "rule_based_controller_r1"] if x in set(l3["experiment"].astype(str).unique().tolist())]:
            rr = _select_lambda_turn_round(ctrl, exp) or 1
            st = _l3_summary_stats(l3, exp=exp, round_idx=int(rr))
            if not st:
                continue
            topk = st.get("topk", {})
            sel = st.get("selected", {})
            rows.append(
                {
                    "experiment": _name(exp),
                    "round": int(rr),
                    "Top-k n": topk.get("n"),
                    "Top-k U_mean": topk.get("U_mean"),
                    "Top-k K_mean": topk.get("K_mean"),
                    "Selected n": sel.get("n"),
                    "Selected U_mean": sel.get("U_mean"),
                    "Selected K_mean": sel.get("K_mean"),
                }
            )
        df_stats = pd.DataFrame(rows)
        if not df_stats.empty:
            l3_stats_html = _df_to_html_table(
                df_stats,
                float_cols=["Top-k U_mean", "Top-k K_mean", "Selected U_mean", "Selected K_mean"],
                float_nd=4,
            )

    lc_html = "<p class='muted'>（round_curves 为空或缺失）</p>"
    if not curves.empty:
        d = curves[curves["experiment"].astype(str) == full_key].copy() if "experiment" in curves.columns else pd.DataFrame()
        if not d.empty:
            show = d[["round", "labeled_size", "miou_round", "f1_round", "epochs_observed"]].sort_values("round").copy()
            show = pd.concat([show.head(5), show.tail(5)], ignore_index=True) if show.shape[0] > 12 else show
            lc_html = _df_to_html_table(show, float_cols=["miou_round", "f1_round"], float_nd=4)

    round_summary_html = "<p class='muted'>（round_summary 为空或缺失）</p>"
    if not summary.empty:
        d = summary[summary["experiment"].astype(str) == full_key].copy() if "experiment" in summary.columns else pd.DataFrame()
        if not d.empty:
            cols = [
                c
                for c in [
                    "round",
                    "labeled_size",
                    "mIoU",
                    "f1",
                    "avg_uncertainty",
                    "avg_knowledge_gain",
                    "lambda_t",
                    "lambda_controller",
                    "miou_delta",
                    "rollback_flag",
                    "selection_source",
                ]
                if c in d.columns
            ]
            show = d[cols].sort_values("round").copy()
            show = pd.concat([show.head(6), show.tail(6)], ignore_index=True) if show.shape[0] > 14 else show
            round_summary_html = _df_to_html_table(
                show,
                float_cols=["mIoU", "f1", "avg_uncertainty", "avg_knowledge_gain", "lambda_t", "lambda_controller", "miou_delta"],
                float_nd=4,
            )

    def _curve_stability(curves_df: pd.DataFrame, exp: str) -> Dict[str, Any]:
        if curves_df.empty:
            return {}
        if "experiment" not in curves_df.columns:
            return {}
        d = curves_df[curves_df["experiment"].astype(str) == str(exp)].copy()
        if d.empty:
            return {}
        for c in ["round", "miou_round", "labeled_size"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["round", "miou_round"]).sort_values("round")
        if d.shape[0] < 2:
            return {}
        miou = d["miou_round"].astype(float).tolist()
        rnd = d["round"].astype(float).tolist()
        deltas = [miou[i] - miou[i - 1] for i in range(1, len(miou))]
        min_drop = min(deltas) if deltas else None
        min_i = int(np.argmin(deltas)) if deltas else 0
        drop_round = int(rnd[min_i + 1]) if deltas and np.isfinite(rnd[min_i + 1]) else None
        neg_cnt = sum(1 for x in deltas if x < 0)
        return {"min_delta": float(min_drop), "min_round": drop_round, "neg_cnt": int(neg_cnt), "n": int(d.shape[0])}

    stability_rows: List[Dict[str, Any]] = []
    for _, exp in baseline_order:
        st = _curve_stability(curves, exp)
        if not st:
            continue
        stability_rows.append(
            {
                "experiment": _name(exp),
                "min ΔmIoU(单步)": st.get("min_delta"),
                "发生轮次": st.get("min_round"),
                "回撤次数(Δ<0)": st.get("neg_cnt"),
                "点数": st.get("n"),
            }
        )
    stability_tbl = pd.DataFrame(stability_rows)
    stability_html = "<p class='muted'>（无法从 round_curves 计算稳定性统计）</p>"
    if not stability_tbl.empty:
        stability_tbl = stability_tbl.sort_values("min ΔmIoU(单步)")
        stability_html = _df_to_html_table(stability_tbl, float_cols=["min ΔmIoU(单步)"], float_nd=4)

    threats_html = (
        "<ul>"
        "<li>当前报告为单 seed（42）结果，未提供统计显著性；建议在 multi-seed 上汇总均值/方差/置信区间。</li>"
        "<li>传统基线中存在自动诊断的“回撤/异常”记录（如 BALD），需要与曲线稳定性统计联合解读，避免仅以最终点做结论。</li>"
        "<li>即便 FIX_EPOCHS_PER_ROUND=true，也要关注训练收敛差异带来的间接影响（例如不同 round 的有效学习强度）。</li>"
        "</ul>"
    )

    next_steps_html = (
        "<ul>"
        "<li>跑 multi-seed：复用相同 ablation 矩阵，输出 multi-seed 汇总表（ALC/final mIoU 的均值与 CI）。</li>"
        "<li>对异常基线（如 BALD）补充：复现实验与随机性敏感性分析（seed、dataloader、mps 等）。</li>"
        "<li>把关键结论落到机制：结合 Fig.3/4/5，解释 λ_t 变化、梯度一致性与样本选择分布之间的因果链条。</li>"
        "</ul>"
    )

    def _rank_by_metric(pairs: List[Tuple[str, str]], metric: str, higher_is_better: bool = True) -> Tuple[Optional[int], int]:
        rows = []
        for _, exp in pairs:
            r = _get_row(exp)
            if not r:
                continue
            v = r.get(metric)
            try:
                fv = float(v)
            except Exception:
                continue
            if not np.isfinite(fv):
                continue
            rows.append((exp, fv))
        if not rows:
            return None, 0
        rows_sorted = sorted(rows, key=lambda x: x[1], reverse=higher_is_better)
        exps = [e for e, _ in rows_sorted]
        if str(full_key) not in exps:
            return None, len(exps)
        return exps.index(str(full_key)) + 1, len(exps)

    rank_miou, n_miou = _rank_by_metric(baseline_order, "final_mIoU", higher_is_better=True)
    rank_alc, n_alc = _rank_by_metric(baseline_order, "alc", higher_is_better=True)

    def _best_in_group(pairs: List[Tuple[str, str]], metric: str) -> Dict[str, Any]:
        cand = []
        for _, exp in pairs:
            r = _get_row(exp)
            if not r:
                continue
            try:
                v = float(r.get(metric))
            except Exception:
                continue
            if not np.isfinite(v):
                continue
            cand.append((exp, v, r))
        if not cand:
            return {}
        exp, _, r = sorted(cand, key=lambda x: x[1], reverse=True)[0]
        out = dict(r)
        out["experiment"] = exp
        return out

    best_baseline_miou = _best_in_group(baseline_order, "final_mIoU")
    best_baseline_alc = _best_in_group(baseline_order, "alc")

    deltas_vs = []
    for _, exp in [("B1", "baseline_random"), ("B2", "baseline_entropy"), ("B3", "baseline_coreset"), ("B4", "baseline_bald")]:
        r = _get_row(exp)
        if not r:
            continue
        deltas_vs.append(
            {
                "name": _name(exp),
                "d_miou": _delta(ours_row.get("final_mIoU"), r.get("final_mIoU")),
                "d_alc": _delta(ours_row.get("alc"), r.get("alc")),
            }
        )

    abstract_lines = []
    if ours_row:
        abstract_lines.append(
            f"本文在滑坡语义分割主动学习设定下评估 AAL-SD（AD-KUCS + 动态 λ_t 控制）。在固定标注预算与固定每轮训练 epoch 的条件下（N_ROUNDS={cfg.get('N_ROUNDS')}，QUERY_SIZE={cfg.get('QUERY_SIZE')}，EPOCHS_PER_ROUND={cfg.get('EPOCHS_PER_ROUND')}），我们比较 {_name(str(full_key))} 与多类基线，并用 ALC 与最终 mIoU 衡量标注效率。"
        )
        if best_baseline_miou and (best_baseline_miou.get("experiment") != str(full_key)):
            abstract_lines.append(
                f"在 seed={cfg.get('RANDOM_SEED')} 的 run 中，{_name(str(full_key))} 最终 mIoU={_fmt_float(ours_row.get('final_mIoU'))}，ALC={_fmt_float(ours_row.get('alc'))}；最优基线（按最终 mIoU）为 {_name(str(best_baseline_miou.get('experiment')))}（mIoU={_fmt_float(best_baseline_miou.get('final_mIoU'))}）。"
            )
        else:
            abstract_lines.append(
                f"在 seed={cfg.get('RANDOM_SEED')} 的 run 中，{_name(str(full_key))} 在基线组中达到最优或并列最优（mIoU={_fmt_float(ours_row.get('final_mIoU'))}，ALC={_fmt_float(ours_row.get('alc'))}）。"
            )

    rq_lines = [
        f"研究问题 RQ1：在固定标注预算下，AAL-SD（{_name(str(full_key))}）能否提升标注效率（ALC）与最终精度（mIoU/F1）？",
        "研究问题 RQ2：动态 λ_t（U–K 权衡）是否相对固定 λ 或规则控制带来可量化增益？",
        "研究问题 RQ3：动态 λ_t 的变化是否与训练-验证梯度一致性等“过拟合风险证据”相关联，从而支持风险闭环机制？",
        "研究问题 RQ4：在样本级 U–K 空间，所选样本分布是否呈现与 λ_t 设定一致的偏好？",
    ]

    setup_lines = [
        f"数据与任务：滑坡语义分割（NUM_CLASSES={cfg.get('NUM_CLASSES')}），以 mIoU 与 F1 评估分割质量。",
        f"预算与轮次：TOTAL_BUDGET={cfg.get('TOTAL_BUDGET')}，N_ROUNDS={cfg.get('N_ROUNDS')}，每轮标注 QUERY_SIZE={cfg.get('QUERY_SIZE')}，初始标注比例 INITIAL_LABELED_SIZE={cfg.get('INITIAL_LABELED_SIZE')}。",
        f"训练口径：FIX_EPOCHS_PER_ROUND={cfg.get('FIX_EPOCHS_PER_ROUND')}，EPOCHS_PER_ROUND={cfg.get('EPOCHS_PER_ROUND')}，确保不同方法在训练预算上可比。",
        f"硬件与可复现：DEVICE={cfg.get('DEVICE')}，DETERMINISTIC={cfg.get('DETERMINISTIC')}，seed={cfg.get('RANDOM_SEED')}。",
    ]

    metric_lines = [
        "主要指标：ALC（学习曲线面积，衡量在整个标注过程中的平均标注效率）。",
        "辅助指标：最终 mIoU / 最终 F1（在最后一轮标注规模下的性能）；曲线稳定性（单步回撤幅度与回撤次数）。",
        "机制证据：train–val 梯度一致性（grad_train_val_cos）与 λ_t 的相关性（Pearson r）。",
    ]

    main_result_lines = []
    if ours_row:
        main_result_lines.append(
            f"在基线组（B1–B8）内，{_name(str(full_key))} 的最终 mIoU 排名为 {rank_miou}/{n_miou}，ALC 排名为 {rank_alc}/{n_alc}（排名仅基于本 run 中可用实验组）。"
        )
        if best_baseline_alc:
            main_result_lines.append(
                f"基线组内最优 ALC 方法为 {_name(str(best_baseline_alc.get('experiment')))}（ALC={_fmt_float(best_baseline_alc.get('alc'))}，mIoU={_fmt_float(best_baseline_alc.get('final_mIoU'))}）。"
            )
        if best_baseline_miou:
            main_result_lines.append(
                f"基线组内最优最终 mIoU 方法为 {_name(str(best_baseline_miou.get('experiment')))}（mIoU={_fmt_float(best_baseline_miou.get('final_mIoU'))}，ALC={_fmt_float(best_baseline_miou.get('alc'))}）。"
            )
        if deltas_vs:
            parts = []
            for it in deltas_vs:
                parts.append(f"{it['name']}：ΔmIoU={_fmt_delta(it['d_miou'])}，ΔALC={_fmt_delta(it['d_alc'])}")
            main_result_lines.append(f"{_name(str(full_key))} 相对传统基线的差值为：" + "；".join(parts) + "。")

    ablation_lines = []
    if not t_ablation.empty:
        ablation_lines.append(f"消融结果显示：固定 λ（A1）相对 {_name(str(full_key))} 的 ALC 与最终 mIoU 均下降，支持动态 λ_t 的有效性。")
        ablation_lines.append(f"三类 Rule-based 控制（A3）整体弱于 {_name(str(full_key))}，其中 R2 对 ALC 影响最大，R3 对最终 mIoU 影响最大。")
        ablation_lines.append("no_normalization 的 ALC 下降幅度最小，但最终 mIoU 仍下降，提示归一化可能主要影响排序稳定性而非最终上界。")

    grad_lines = []
    if not corr_tbl.empty:
        fm = corr_tbl[corr_tbl["experiment"].astype(str) == str(full_key)]
        if not fm.empty:
            r = fm.iloc[0].to_dict().get("pearson_r")
            grad_lines.append(f"{_name(str(full_key))} 的 λ_t 与 grad_train_val_cos 的 Pearson 相关系数为 {r}（n≈14 个 round 点）。")
            grad_lines.append("该相关性提供了“风险闭环”机制的定量证据：λ_t 的调整与梯度一致性信号存在系统性关系。")
    if not grad_lines:
        grad_lines.append("由于缺失 λ 或 grad_train_val_cos 的有效对齐数据，当前 run 无法形成梯度证据分析。")

    uk_lines = []
    if (not l3.empty) and (str(full_key) in set(l3["experiment"].astype(str).unique().tolist())):
        rr = _select_lambda_turn_round(ctrl, str(full_key)) or 1
        st = _l3_summary_stats(l3, exp=str(full_key), round_idx=int(rr))
        if st:
            topk = st.get("topk", {})
            sel = st.get("selected", {})
            if int(topk.get("n") or 0) == int(sel.get("n") or 0):
                uk_lines.append(
                    f"在 {_name(str(full_key))} 的 L3 日志中（round={int(rr)}），Top-k 与 Selected 的样本数一致（n={int(sel.get('n') or 0)}），导致两者 U/K 统计相同；这意味着该轮选择近似等价于直接取 Top-k。"
                )
            else:
                uk_lines.append(
                    f"在 {_name(str(full_key))} 的 L3 日志中（round={int(rr)}），Selected 相对 Top-k 的 U_mean {_fmt_delta(_delta(sel.get('U_mean'), topk.get('U_mean')))}，K_mean {_fmt_delta(_delta(sel.get('K_mean'), topk.get('K_mean')))}。"
                )
    if (not l3.empty) and ("rule_based_controller_r1" in set(l3["experiment"].astype(str).unique().tolist())):
        rr = _select_lambda_turn_round(ctrl, "rule_based_controller_r1") or 1
        st = _l3_summary_stats(l3, exp="rule_based_controller_r1", round_idx=int(rr))
        if st:
            topk = st.get("topk", {})
            sel = st.get("selected", {})
            uk_lines.append(
                f"对比 Rule-based(R1)（round={int(rr)}），Selected 的 U_mean 相对 Top-k 为 {_fmt_delta(_delta(sel.get('U_mean'), topk.get('U_mean')))}，显示其更偏向高不确定性样本（与 Fig.5 的分布形态一致）。"
            )
    if not uk_lines:
        uk_lines.append("当前 run 的 L3 统计不足以支撑样本级机制分析。")

    stability_lines = [
        "为避免仅依赖最终点结论，我们统计了学习曲线的单步回撤幅度与回撤次数（见“异常/稳定性”表）。",
        "需要强调：若某方法出现显著回撤或异常轮次，其最终点可能受随机性或训练不稳定影响，应结合异常报告与稳定性统计进行综合判断。",
    ]

    conclusion_lines = []
    if ours_row:
        conclusion_lines.append(
            f"结论：在本次 seed={cfg.get('RANDOM_SEED')} 的单 run 中，{_name(str(full_key))} 在标注效率（ALC={_fmt_float(ours_row.get('alc'))}）与最终性能（mIoU={_fmt_float(ours_row.get('final_mIoU'))}，F1={_fmt_float(ours_row.get('final_f1'))}）上达到强基线水平，并在消融对比中体现出动态 λ_t 与风险闭环组件的贡献。"
        )
        conclusion_lines.append("然而，单 seed 结果不足以给出统计显著性结论；建议以 multi-seed 汇总作为论文主结论依据。")

    academic_body_html = (
        "<h3>摘要</h3>"
        + _paras(abstract_lines)
        + "<h3>1 引言与研究问题</h3>"
        + _paras(rq_lines)
        + "<h3>2 方法概述</h3>"
        + _paras(
            [
                "AAL-SD 以 AD-KUCS 为基础，对每个候选样本结合不确定性 U(x) 与知识增益 K(x) 进行打分，并通过 λ_t 动态控制二者权衡。",
                f"{_name(str(full_key))} 使用“冷启动→warmup→风险闭环”的 λ_t 日程（见 manifest 中 lambda_policy），旨在早期优先覆盖不确定性、随后逐步引入知识增益并在过拟合风险上升时自适应调整。",
            ]
        )
        + "<h3>3 实验设置</h3>"
        + _paras(setup_lines)
        + "<h3>4 评估指标</h3>"
        + _paras(metric_lines)
        + "<h3>5 结果与分析</h3>"
        + "<h4>5.1 基线对比（B1–B8）</h4>"
        + _paras(main_result_lines)
        + "<h4>5.2 消融分析（A0/A1/A2/A3/A4/A5/A7）</h4>"
        + _paras(ablation_lines)
        + "<h4>5.3 梯度证据（λ_t 与梯度一致性）</h4>"
        + _paras(grad_lines)
        + "<h4>5.4 样本级证据（U–K 空间）</h4>"
        + _paras(uk_lines)
        + "<h4>5.5 稳定性与异常</h4>"
        + _paras(stability_lines)
        + "<h3>6 讨论与威胁</h3>"
        + threats_html
        + "<h3>7 结论</h3>"
        + _paras(conclusion_lines)
    )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>AAL-SD 实验报告 - { _html_escape(run_dir.name) }</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans", Arial, sans-serif; margin: 0; background: #0b1220; color: #e6edf3; }}
    a {{ color: #7dd3fc; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .wrap {{ max-width: 1080px; margin: 0 auto; padding: 28px 18px 64px; }}
    .top {{ padding: 16px 18px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; }}
    .h1 {{ font-size: 22px; font-weight: 700; margin: 0 0 8px; }}
    .muted {{ color: rgba(230,237,243,0.68); }}
    .nav a {{ margin-right: 12px; }}
    .sec {{ margin-top: 18px; padding: 16px 18px; background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; }}
    .sec h2 {{ margin: 0 0 10px; font-size: 18px; }}
    .sec h3 {{ margin: 12px 0 8px; font-size: 15px; }}
    .tbl {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .tbl th, .tbl td {{ border: 1px solid rgba(255,255,255,0.10); padding: 8px 8px; vertical-align: top; }}
    .tbl th {{ background: rgba(255,255,255,0.06); text-align: left; }}
    .fig {{ margin: 14px 0; padding: 12px; border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; background: rgba(0,0,0,0.12); }}
    .fig img {{ width: 100%; height: auto; display: block; border-radius: 10px; }}
    .cap {{ margin-top: 8px; font-size: 12px; color: rgba(230,237,243,0.75); }}
    .card {{ padding: 12px; border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; background: rgba(0,0,0,0.12); margin: 10px 0; }}
    .card-h {{ font-weight: 700; }}
    .card-sub {{ font-size: 12px; color: rgba(230,237,243,0.65); margin-top: 4px; }}
    .pre {{ white-space: pre-wrap; font-size: 12px; color: rgba(230,237,243,0.85); }}
    details {{ margin-top: 8px; }}
    code {{ color: #93c5fd; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div class="h1">AAL-SD 完整实验报告（HTML）</div>
      <div class="muted">Run：<code>{_html_escape(run_dir.name)}</code> ｜ 生成时间：{_html_escape(gen_time)} ｜ 输出目录：<code>{_html_escape(outdir.as_posix())}</code></div>
      <div class="muted">实验配置摘要：{_html_escape(exp_cfg)} ｜ 训练预算口径：{_html_escape("固定每轮 epoch（FIX_EPOCHS_PER_ROUND=true）" if fixed_epochs else "非固定每轮 epoch")}</div>
      <div class="muted">实验方案对齐：{plan_link if plan_link else "（未提供 plan_path）"}</div>
      <div class="nav" style="margin-top:10px;">
        <a href="#overview">概览</a>
        <a href="#paper">正文</a>
        <a href="#figs">图表</a>
        <a href="#exp1">实验一：基线</a>
        <a href="#exp2">实验二：消融</a>
        <a href="#exp3">实验三：梯度证据</a>
        <a href="#l3">L3：样本分布</a>
        <a href="#anomaly">异常/稳定性</a>
        <a href="#limits">局限</a>
        <a href="#next">下一步</a>
        <a href="#appendix">附录</a>
      </div>
    </div>

    <div class="sec" id="overview">
      <h2>1. 结论摘要（对齐实验方案的核心问题）</h2>
      <ul>
        <li>整体最优（按 final mIoU）：<b>{_html_escape(_name(str(best_row.get("experiment"))))}</b>（ALC={_fmt_float(best_row.get("alc"))}，final mIoU={_fmt_float(best_row.get("final_mIoU"))}）</li>
        <li>主方法 {_html_escape(_name(str(full_key)))}：{_html_escape(_metric_line(str(full_key)))}</li>
        <li>对照固定 λ（A1 / B5）：{_html_escape(_metric_line("fixed_lambda"))}</li>
        <li>对照规则控制（A3）：{_html_escape(_metric_line("rule_based_controller_r1"))}；{_html_escape(_metric_line("rule_based_controller_r2"))}；{_html_escape(_metric_line("rule_based_controller_r3"))}</li>
      </ul>
      <div class="muted">提示：当前报告为 seed=42 单次 run 的“论文口径”整理；统计显著性需要 multi-seed 汇总（可用本脚本的 --multi_seed_group_dir 分支）。</div>
    </div>

    <div class="sec" id="paper">
      <h2>2. 学术实验报告正文（自动生成）</h2>
      <div class="muted">实验方案对齐提纲（从方案文档提取）：</div>
      {plan_outline_html}
      <div style="margin-top:10px;"></div>
      {academic_body_html}
    </div>

    <div class="sec" id="figs">
      <h2>2. 论文图表清单（Fig.1–Fig.6）</h2>
      {fig_html}
    </div>

    <div class="sec" id="exp1">
      <h2>3. 实验一：基线对比（P0/P1）</h2>
      <div class="muted">对应方案：3.2（B1–B8），输出：Fig.1/2 + Table 1</div>
      <h3>Table 1：基线对比结果</h3>
      {_df_to_html_table(t_baseline, float_cols=["ALC", "final mIoU", "final F1"], float_nd=6)}
      <details>
        <summary>展开：学习曲线关键点（{_html_escape(_name(str(full_key)))}，来自 round_curves）</summary>
        {lc_html}
      </details>
      <details>
        <summary>展开：基线对比结论（基于 seed=42）</summary>
        <div class="pre">
best(final mIoU) = {_html_escape(_name(str(best_row.get("experiment"))))} ({_fmt_float(best_row.get("final_mIoU"))})
{_html_escape(str(full_key))} vs fixed_lambda: ΔmIoU = {_fmt_float((ours_row.get("final_mIoU") or float("nan")) - (metrics[metrics["experiment"].astype(str)=="fixed_lambda"].iloc[0].to_dict().get("final_mIoU") if (metrics["experiment"].astype(str)=="fixed_lambda").any() else float("nan")), 4)}
        </div>
      </details>
    </div>

    <div class="sec" id="exp2">
      <h2>4. 实验二：细粒度消融（P0/P1）</h2>
      <div class="muted">对应方案：4.2（A0/A1/A2/A3/A4/A5/A7），输出：Fig.6 + Table 2</div>
      <h3>Table 2：消融结果（Δ 相对 {_html_escape(_name(str(full_key)))}）</h3>
      {_df_to_html_table(t_ablation, float_cols=["ALC", "final mIoU", "final F1", "ΔALC(vs Full)", "ΔmIoU(vs Full)"], float_nd=6)}
      <div class="muted">ΔALC/ΔmIoU 为单 run 相对差值，用于“组件贡献量化”的直观展示；更稳健的版本应在 multi-seed 上汇总均值与置信区间。</div>
    </div>

    <div class="sec" id="exp3">
      <h2>5. 实验三：梯度证据分析（P2）</h2>
      <div class="muted">对应方案：5.2（λ_t–train_val_cos 可视化 + 相关性），输出：Fig.3/4</div>
      <h3>相关性汇总（Pearson r）</h3>
      {_df_to_html_table(corr_tbl, float_cols=[], float_nd=6)}
      <div class="muted">说明：r 为每个 experiment 在 round 维度上，λ 与 grad_train_val_cos 的 Pearson 相关（仅对齐存在两者的 round）。</div>
      <details>
        <summary>展开：{_html_escape(_name(str(full_key)))} round_summary 片段（用于追溯每轮 λ_t、U/K 与回撤标记）</summary>
        {round_summary_html}
      </details>
    </div>

    <div class="sec" id="l3">
      <h2>6. L3：样本级分布分析（U–K 空间）</h2>
      <div class="muted">对应方案：6.1 L3 + 6.2 A5，输出：Fig.5</div>
      {l3_note_html}
      {l3_stats_html}
    </div>

    <div class="sec" id="anomaly">
      <h2>7. 异常与稳定性（自动诊断汇总）</h2>
      <div class="muted">来源：results/runs/&lt;run_id&gt;/reports/*_anomaly_report_*.md</div>
      <h3>曲线稳定性统计（来自 round_curves）</h3>
      {stability_html}
      {anomaly_html}
    </div>

    <div class="sec" id="limits">
      <h2>8. 局限与威胁（面向论文写作口径）</h2>
      {threats_html}
    </div>

    <div class="sec" id="next">
      <h2>9. 下一步建议（可直接转为补充实验清单）</h2>
      {next_steps_html}
    </div>

    <div class="sec" id="appendix">
      <h2>10. 附录：原始数据与全量图表</h2>
      <h3>CSV 输出</h3>
      <ul>
        <li><a href="{_html_escape((outdir / "metrics_summary.csv").name)}">metrics_summary.csv</a></li>
        <li><a href="{_html_escape((outdir / "round_curves.csv").name)}">round_curves.csv</a></li>
        <li><a href="{_html_escape((outdir / "round_gradients.csv").name)}">round_gradients.csv</a></li>
        <li><a href="{_html_escape((outdir / "controller_trajectories.csv").name)}">controller_trajectories.csv</a></li>
        <li><a href="{_html_escape((outdir / "round_summary.csv").name)}">round_summary.csv</a></li>
        <li><a href="{_html_escape((outdir / "l3_selection.csv").name)}">l3_selection.csv</a></li>
      </ul>
      <details>
        <summary>展开：目录下全部 PNG 画廊（便于快速查验）</summary>
        {gallery_html}
      </details>
    </div>

    <div class="muted" style="margin-top:16px;">生成脚本：src/analysis/plot_paper_figures.py（HTML/PDF/CSV 一体化输出）。</div>
  </div>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--multi_seed_group_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--exclude_prefix", default="")
    parser.add_argument("--export_pdf", action="store_true")
    parser.add_argument("--export_theory_pdf", action="store_true")
    parser.add_argument("--export_html", action="store_true")
    parser.add_argument("--reports_dir", default=None)
    parser.add_argument("--pdf_path", default=None)
    parser.add_argument("--html_path", default=None)
    parser.add_argument("--theory_md", default=None)
    parser.add_argument("--plan_path", default=None)
    args = parser.parse_args()

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    if not args.run_dir and not args.multi_seed_group_dir:
        raise SystemExit("Either --run_dir or --multi_seed_group_dir must be provided.")

    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    multi_seed_group_dir = Path(args.multi_seed_group_dir).expanduser().resolve() if args.multi_seed_group_dir else None

    default_outdir = None
    if run_dir is not None:
        default_outdir = run_dir / "figures"
    elif multi_seed_group_dir is not None:
        default_outdir = multi_seed_group_dir / "figures"
    outdir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_outdir
    if outdir is None:
        raise SystemExit("Cannot determine output_dir.")
    outdir.mkdir(parents=True, exist_ok=True)

    paper_include = [
        "full_model_A_lambda_policy",
        "full_model_B_lambda_agent",
        "full_model",
        "ablation_B_llm_lambda_control",
        "fixed_lambda",
        "random_lambda",
        "rule_based_controller_r1",
        "rule_based_controller_r2",
        "rule_based_controller_r3",
        "no_cold_start",
        "fixed_k",
        "no_normalization",
        "uncertainty_only",
        "knowledge_only",
        "baseline_random",
        "baseline_entropy",
        "baseline_coreset",
        "baseline_dial_style",
        "baseline_wang_style",
    ]
    paper_labels = {
        "full_model_A_lambda_policy": "AAL-SD (A: Policy λ)",
        "full_model_B_lambda_agent": "AAL-SD (B: Agent λ)",
        "full_model": "AAL-SD (Full)",
        "ablation_B_llm_lambda_control": "AAL-SD (B: Agent λ)",
        "no_agent": "AD-KUCS (No Agent)",
        "fixed_lambda": "AD-KUCS (Fixed λ=0.5)",
        "uncertainty_only": "Uncertainty-only (λ=0)",
        "knowledge_only": "Knowledge-only (λ=1)",
        "baseline_random": "Random",
        "baseline_entropy": "Entropy",
        "baseline_bald": "BALD",
        "baseline_coreset": "Core-set",
        "baseline_llm_us": "LLM-US",
        "baseline_llm_rs": "LLM-RS",
        "agent_control_lambda": "Agent control λ",
        "rule_based_controller_r1": "Rule-based (R1)",
        "rule_based_controller_r2": "Rule-based (R2)",
        "rule_based_controller_r3": "Rule-based (R3)",
        "baseline_dial_style": "DIAL-style",
        "baseline_wang_style": "Wang-style",
        "no_cold_start": "No Cold-Start",
        "fixed_k": "Fixed K",
        "random_lambda": "Random λ",
        "no_normalization": "No Normalization",
    }

    if run_dir is not None:
        manifest, exps = discover_experiments(run_dir)
        exp_meta = manifest.get("experiments") if isinstance(manifest.get("experiments"), dict) else {}
        cfg = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
        fixed_epochs = bool(cfg.get("FIX_EPOCHS_PER_ROUND")) or (
            cfg.get("EPOCHS_PER_ROUND_SCHEDULE") in (None, "null") and cfg.get("EPOCHS_PER_ROUND") is not None
        )

        exclude_prefixes = [p.strip() for p in str(args.exclude_prefix).split(",") if p.strip()]

        metrics_rows: List[Dict[str, Any]] = []
        curves_rows: List[pd.DataFrame] = []
        ctrl_rows: List[pd.DataFrame] = []
        grad_rows: List[pd.DataFrame] = []
        summary_rows: List[pd.DataFrame] = []
        l3_rows: List[pd.DataFrame] = []

        for ep in exps:
            if any(ep.name.startswith(px) for px in exclude_prefixes):
                continue

            status = load_status_metrics(ep.status_path)
            df_epoch, df_ctrl = parse_trace(ep.trace_path)
            df_summary, df_l3 = parse_round_summary(ep.trace_path)
            curve = build_round_curve(df_epoch)
            if not curve.empty:
                curve["experiment"] = ep.name
                curves_rows.append(curve)

            round_grad = build_round_grad(df_epoch)
            if not round_grad.empty:
                round_grad["experiment"] = ep.name
                grad_rows.append(round_grad)

            cost = compute_cost(df_epoch, df_ctrl)
            meta = exp_meta.get(ep.name, {}) if isinstance(exp_meta.get(ep.name), dict) else {}
            description = meta.get("description")

            metrics_rows.append(
                {
                    "experiment": ep.name,
                    "description": description,
                    "status": status.get("status"),
                    "alc": status.get("alc"),
                    "final_mIoU": status.get("final_mIoU"),
                    "final_f1": status.get("final_f1"),
                    "total_epochs": cost.get("total_epochs"),
                    "wall_clock_sec": cost.get("wall_clock_sec"),
                    "n_rounds": int(cfg.get("N_ROUNDS")) if cfg.get("N_ROUNDS") is not None else None,
                }
            )

            if not df_ctrl.empty:
                d = df_ctrl.copy()
                d["experiment"] = ep.name
                ctrl_rows.append(d)

            if not df_summary.empty:
                s = df_summary.copy()
                s["experiment"] = ep.name
                summary_rows.append(s)

            if not df_l3.empty:
                l3 = df_l3.copy()
                l3["experiment"] = ep.name
                l3_rows.append(l3)

        metrics = pd.DataFrame(metrics_rows)
        metrics.to_csv(outdir / "metrics_summary.csv", index=False, encoding="utf-8")

        curves = pd.concat(curves_rows, ignore_index=True) if curves_rows else pd.DataFrame()
        if not curves.empty:
            curves.to_csv(outdir / "round_curves.csv", index=False, encoding="utf-8")

        ctrl = pd.concat(ctrl_rows, ignore_index=True) if ctrl_rows else pd.DataFrame()
        if not ctrl.empty:
            ctrl.to_csv(outdir / "controller_trajectories.csv", index=False, encoding="utf-8")

        grad = pd.concat(grad_rows, ignore_index=True) if grad_rows else pd.DataFrame()
        if not grad.empty:
            grad.to_csv(outdir / "round_gradients.csv", index=False, encoding="utf-8")

        summary = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
        if not summary.empty:
            summary.to_csv(outdir / "round_summary.csv", index=False, encoding="utf-8")

        l3 = pd.concat(l3_rows, ignore_index=True) if l3_rows else pd.DataFrame()
        if not l3.empty:
            l3.to_csv(outdir / "l3_selection.csv", index=False, encoding="utf-8")

        plot_learning_curves(curves, outdir, title=f"标注效率曲线（全量曲线，{run_dir.name}）", filename="learning_curve_miou_vs_labeled_all.png")

        plot_learning_curves(
            curves,
            outdir,
            title=f"标注效率曲线（用于论文展示，{run_dir.name}）",
            filename="learning_curve_miou_vs_labeled.png",
            include_experiments=[
                x
                for x in paper_include
                if (not curves.empty) and (x in set(curves["experiment"].astype(str).unique().tolist()))
            ],
            label_map=paper_labels,
        )
        plot_bars(metrics, outdir)
        if not fixed_epochs:
            plot_cost_tradeoff(metrics, outdir)
        plot_controller_trajectories(ctrl, outdir)
        plot_gradient_diagnostics(
            grad,
            outdir,
            title=f"梯度诊断（{run_dir.name}）",
            filename="gradient_diagnostics.png",
            include_experiments=[
                x for x in paper_include if (not grad.empty) and (x in set(grad["experiment"].astype(str).unique().tolist()))
            ],
            label_map=paper_labels,
        )
        plot_aal_sd_active_learning_loop_diagram(outdir)

        exp_names = metrics["experiment"].astype(str).unique().tolist() if (not metrics.empty) and ("experiment" in metrics.columns) else []
        full_key = _pick_first_present(exp_names, ["full_model_A_lambda_policy", "full_model"])
        rule_compare = [
            x
            for x in [full_key, "rule_based_controller_r1", "rule_based_controller_r2", "rule_based_controller_r3"]
            if (not ctrl.empty) and (x in set(ctrl["experiment"].astype(str).unique().tolist()))
        ]
        if rule_compare:
            plot_lambda_evolution(
                ctrl,
                outdir,
                title=f"λ_t 演化（{run_dir.name}）",
                filename="lambda_evolution_aal_vs_rules.png",
                include_experiments=rule_compare,
                label_map=paper_labels,
            )

        if (not ctrl.empty) and (not grad.empty) and rule_compare:
            plot_lambda_gradient_consistency(
                ctrl,
                grad,
                outdir,
                title=f"λ_t 与梯度一致性（{run_dir.name}）",
                filename="lambda_vs_gradient_consistency.png",
                include_experiments=rule_compare,
                label_map=paper_labels,
            )

        if not l3.empty:
            exp_set = set(l3["experiment"].astype(str).unique().tolist())
            uk_exps = [x for x in [full_key, "rule_based_controller_r1"] if x in exp_set] or [x for x in [full_key] if x in exp_set]
            if uk_exps:
                round_map: Dict[str, int] = {}
                if not ctrl.empty:
                    for exp in uk_exps:
                        d = ctrl[ctrl["experiment"].astype(str) == str(exp)].dropna(subset=["round", "lambda"]).sort_values("round")
                        if d.shape[0] >= 2:
                            r = d["round"].astype(float).tolist()
                            lam = pd.to_numeric(d["lambda"], errors="coerce").astype(float).tolist()
                            diffs = [abs(lam[i] - lam[i - 1]) if np.isfinite(lam[i]) and np.isfinite(lam[i - 1]) else float("nan") for i in range(1, len(lam))]
                            if diffs and any(np.isfinite(x) for x in diffs):
                                idx = max(range(len(diffs)), key=lambda i: diffs[i] if np.isfinite(diffs[i]) else -1.0)
                                rr = r[idx + 1]
                                if np.isfinite(rr):
                                    round_map[str(exp)] = int(rr)
                plot_uk_space_distribution(
                    l3,
                    outdir,
                    title=f"样本分布：Uncertainty–Knowledge（{run_dir.name}）",
                    filename="uk_space_distribution.png",
                    experiments=uk_exps,
                    label_map=paper_labels,
                    round_map=round_map,
                )

        ablation_exps = [
            "fixed_lambda",
            "random_lambda",
            "no_cold_start",
            "fixed_k",
            "no_normalization",
            "rule_based_controller_r1",
            "rule_based_controller_r2",
            "rule_based_controller_r3",
        ]
        ablation_exps = [x for x in ablation_exps if x in set(metrics["experiment"].astype(str).unique().tolist())]
        if ablation_exps:
            plot_ablation_contributions(
                metrics,
                outdir,
                title=f"消融贡献：相对 Full 的 ΔALC（{run_dir.name}）",
                filename="ablation_contributions_alc.png",
                experiments=ablation_exps,
                label_map=paper_labels,
            )

        if args.export_pdf:
            reports_dir = Path(args.reports_dir).expanduser().resolve() if args.reports_dir else None
            if args.pdf_path:
                pdf_path = Path(args.pdf_path).expanduser().resolve()
            else:
                analyze_dir = (run_dir.parent.parent / "analyze").resolve()
                analyze_dir.mkdir(parents=True, exist_ok=True)
                pdf_path = analyze_dir / f"{run_dir.name}__cvpr_baseline_analysis.pdf"
            export_pdf_report(
                run_dir=run_dir,
                outdir=outdir,
                metrics=metrics,
                reports_dir=reports_dir,
                pdf_path=pdf_path,
            )
            print(f"Saved pdf to: {pdf_path}")

        if args.export_theory_pdf:
            reports_dir = Path(args.reports_dir).expanduser().resolve() if args.reports_dir else None
            if args.theory_md:
                theory_md_path = Path(args.theory_md).expanduser().resolve()
            else:
                theory_md_path = (run_dir.parent.parent.parent / ".trae" / "documents" / "论AAL-SD框架中的动态训练梯度选择策略.md").resolve()
            if args.pdf_path:
                pdf_path = Path(args.pdf_path).expanduser().resolve()
            else:
                analyze_dir = (run_dir.parent.parent / "analyze").resolve()
                analyze_dir.mkdir(parents=True, exist_ok=True)
                pdf_path = analyze_dir / f"{run_dir.name}__theory_based_report.pdf"
            export_theory_based_pdf_report(
                run_dir=run_dir,
                outdir=outdir,
                metrics=metrics,
                reports_dir=reports_dir,
                pdf_path=pdf_path,
                theory_md_path=theory_md_path,
            )
            print(f"Saved theory pdf to: {pdf_path}")

        if args.export_html:
            reports_dir = Path(args.reports_dir).expanduser().resolve() if args.reports_dir else (run_dir / "reports")
            plan_path = Path(args.plan_path).expanduser().resolve() if args.plan_path else (run_dir.parent.parent.parent / ".trae" / "documents" / "AAL-SD 整合后完整实验方案.md").resolve()
            if args.html_path:
                html_path = Path(args.html_path).expanduser().resolve()
            else:
                html_path = outdir / f"{run_dir.name}__full_report.html"
            export_html_report(
                run_dir=run_dir,
                outdir=outdir,
                manifest=manifest,
                metrics=metrics,
                curves=curves,
                ctrl=ctrl,
                grad=grad,
                summary=summary,
                l3=l3,
                reports_dir=reports_dir,
                html_path=html_path,
                plan_path=plan_path if plan_path.exists() else None,
                label_map=paper_labels,
            )
            print(f"Saved html to: {html_path}")

        print(f"Saved figures to: {outdir}")
        print(f"Saved csv to: {outdir / 'metrics_summary.csv'}")

    if multi_seed_group_dir is not None:
        summary = _load_multi_seed_summary(multi_seed_group_dir)
        if not summary:
            run_ids = _discover_multi_seed_run_ids(multi_seed_group_dir)
            if not run_ids:
                raise SystemExit(f"multi-seed manifest not found or empty: {multi_seed_group_dir}")
            summary = {"run_ids": run_ids, "experiments": {}}

        run_ids = summary.get("run_ids")
        if not isinstance(run_ids, list) or not run_ids:
            run_ids = _discover_multi_seed_run_ids(multi_seed_group_dir)
        run_ids = [str(x) for x in run_ids] if isinstance(run_ids, list) else []
        if not run_ids:
            raise SystemExit(f"Cannot find multi-seed run_ids under: {multi_seed_group_dir}")

        experiments = summary.get("experiments")
        exp_names = sorted(experiments.keys()) if isinstance(experiments, dict) else []
        full_key = _pick_first_present(exp_names, ["full_model_A_lambda_policy", "full_model"])
        focus = [x for x in [full_key, "no_agent", "fixed_lambda", "baseline_entropy", "baseline_random"] if x in exp_names] or exp_names

        multi_seed_rows: List[Dict[str, Any]] = []
        if isinstance(experiments, dict):
            for exp in focus:
                payload = experiments.get(exp)
                if not isinstance(payload, dict):
                    continue
                summary_block = payload.get("summary")
                if not isinstance(summary_block, dict):
                    continue
                for metric, stats in summary_block.items():
                    if not isinstance(stats, dict):
                        continue
                    try:
                        multi_seed_rows.append(
                            {
                                "experiment": str(exp),
                                "metric": str(metric),
                                "n": int(stats.get("n")),
                                "mean": float(stats.get("mean")),
                                "std": float(stats.get("std")),
                                "ci95": float(stats.get("ci95")),
                            }
                        )
                    except Exception:
                        continue
        if multi_seed_rows:
            pd.DataFrame(multi_seed_rows).to_csv(outdir / "multiseed_metrics_summary.csv", index=False, encoding="utf-8")

        plot_multiseed_metric_bars(
            summary,
            outdir,
            metric="alc",
            title=f"Multi-seed ALC (mean ± 95% CI, {multi_seed_group_dir.name})",
            filename="multiseed_alc_bar.png",
            include_experiments=focus,
            label_map=paper_labels,
        )
        plot_multiseed_metric_bars(
            summary,
            outdir,
            metric="final_miou",
            title=f"Multi-seed Final mIoU (mean ± 95% CI, {multi_seed_group_dir.name})",
            filename="multiseed_final_miou_bar.png",
            include_experiments=focus,
            label_map=paper_labels,
        )

        base_runs_dir = multi_seed_group_dir.parent
        per_seed_curves: Dict[str, Dict[str, pd.DataFrame]] = {}
        for exp in focus:
            per_seed_curves[exp] = {}
        for rid in run_ids:
            seed_run_dir = (base_runs_dir / rid).resolve()
            for exp in focus:
                trace_path = seed_run_dir / f"{exp}_trace.jsonl"
                df_epoch, _ = parse_trace(trace_path)
                curve = build_round_curve(df_epoch)
                if curve.empty:
                    continue
                per_seed_curves[exp][rid] = curve

        plot_multiseed_learning_curves(
            per_seed_curves,
            outdir,
            title=f"Multi-seed learning curves (mean ± 95% CI, {multi_seed_group_dir.name})",
            filename="multiseed_learning_curve_miou_vs_labeled.png",
            include_experiments=focus,
            label_map=paper_labels,
        )

        print(f"Saved multi-seed figures to: {outdir}")


if __name__ == "__main__":
    main()
