import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _iter_jsonl(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except Exception:
        return


def _parse_ts(ts: Any) -> Optional[datetime]:
    if not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


@dataclass
class ExperimentPaths:
    name: str
    status_path: Path
    trace_path: Path


def discover_experiments(run_dir: Path) -> Tuple[Dict[str, Any], List[ExperimentPaths]]:
    manifest = _read_json(run_dir / "manifest.json")

    exps: List[ExperimentPaths] = []
    for p in sorted(run_dir.glob("*_status.json")):
        name = p.name[: -len("_status.json")]
        trace = run_dir / f"{name}_trace.jsonl"
        exps.append(ExperimentPaths(name=name, status_path=p, trace_path=trace))
    return manifest, exps


def load_status_metrics(status_path: Path) -> Dict[str, Any]:
    payload = _read_json(status_path)
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}

    out = {
        "status": payload.get("status"),
        "alc": result.get("alc"),
        "final_mIoU": result.get("final_mIoU"),
        "final_f1": result.get("final_f1"),
        "budget_history": result.get("budget_history"),
    }
    return out


def parse_trace(trace_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    epoch_rows: List[Dict[str, Any]] = []
    ctrl_rows: List[Dict[str, Any]] = []

    if not trace_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    for e in _iter_jsonl(trace_path):
        et = e.get("type")
        if et == "epoch_end":
            epoch_rows.append(
                {
                    "round": e.get("round"),
                    "epoch": e.get("epoch"),
                    "labeled_size": e.get("labeled_size"),
                    "mIoU": e.get("mIoU"),
                    "f1": e.get("f1"),
                    "ts": e.get("ts"),
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
    if not df_ctrl.empty:
        df_ctrl["round"] = pd.to_numeric(df_ctrl["round"], errors="coerce")
        df_ctrl["lambda"] = pd.to_numeric(df_ctrl["lambda"], errors="coerce")
        df_ctrl["epochs"] = pd.to_numeric(df_ctrl["epochs"], errors="coerce")
        df_ctrl["query_size"] = pd.to_numeric(df_ctrl["query_size"], errors="coerce")
        df_ctrl["miou_delta"] = pd.to_numeric(df_ctrl["miou_delta"], errors="coerce")
        df_ctrl["last_miou"] = pd.to_numeric(df_ctrl["last_miou"], errors="coerce")

    return df_epoch, df_ctrl


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


def compute_cost(df_epoch: pd.DataFrame, df_ctrl: pd.DataFrame) -> Dict[str, Any]:
    total_epochs = int(len(df_epoch)) if not df_epoch.empty else 0

    t0 = None
    t1 = None
    if not df_epoch.empty:
        ts0 = _parse_ts(df_epoch["ts"].dropna().iloc[0]) if df_epoch["ts"].dropna().shape[0] else None
        ts1 = _parse_ts(df_epoch["ts"].dropna().iloc[-1]) if df_epoch["ts"].dropna().shape[0] else None
        t0, t1 = ts0, ts1

    if t0 is None or t1 is None:
        if not df_ctrl.empty:
            ts0 = _parse_ts(df_ctrl["ts"].dropna().iloc[0]) if df_ctrl["ts"].dropna().shape[0] else None
            ts1 = _parse_ts(df_ctrl["ts"].dropna().iloc[-1]) if df_ctrl["ts"].dropna().shape[0] else None
            t0, t1 = ts0, ts1

    wall_clock_sec = None
    if t0 is not None and t1 is not None:
        wall_clock_sec = max(0.0, (t1 - t0).total_seconds())

    return {"total_epochs": total_epochs, "wall_clock_sec": wall_clock_sec}


def plot_learning_curves(curves: pd.DataFrame, outdir: Path, title: str):
    if curves.empty:
        return
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=curves, x="labeled_size", y="miou_round", hue="experiment", marker="o", linewidth=2.2)
    plt.title(title)
    plt.xlabel("Labeled size")
    plt.ylabel("mIoU (best within round)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outdir / "learning_curve_miou_vs_labeled.png", dpi=300)
    plt.close()


def plot_bars(metrics: pd.DataFrame, outdir: Path):
    if metrics.empty:
        return

    m1 = metrics.dropna(subset=["alc"]).sort_values("alc", ascending=False)
    if not m1.empty:
        plt.figure(figsize=(11, 5))
        sns.barplot(data=m1, x="experiment", y="alc")
        plt.xticks(rotation=35, ha="right")
        plt.title("ALC Comparison")
        plt.xlabel("")
        plt.ylabel("ALC")
        plt.tight_layout()
        plt.savefig(outdir / "alc_bar.png", dpi=300)
        plt.close()

    m2 = metrics.dropna(subset=["final_mIoU"]).sort_values("final_mIoU", ascending=False)
    if not m2.empty:
        plt.figure(figsize=(11, 5))
        sns.barplot(data=m2, x="experiment", y="final_mIoU")
        plt.xticks(rotation=35, ha="right")
        plt.title("Final mIoU Comparison")
        plt.xlabel("")
        plt.ylabel("Final mIoU")
        plt.tight_layout()
        plt.savefig(outdir / "final_miou_bar.png", dpi=300)
        plt.close()


def plot_cost_tradeoff(metrics: pd.DataFrame, outdir: Path):
    m = metrics.dropna(subset=["final_mIoU", "total_epochs"]).copy()
    if m.empty:
        return

    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=m, x="total_epochs", y="final_mIoU", hue="experiment", s=120)
    for _, r in m.iterrows():
        plt.text(float(r["total_epochs"]), float(r["final_mIoU"]), str(r["experiment"]), fontsize=8, alpha=0.8)
    plt.title("Cost–Performance Trade-off (proxy cost = total epoch_end count)")
    plt.xlabel("Total epochs (observed)")
    plt.ylabel("Final mIoU")
    plt.grid(True, linestyle="--", alpha=0.5)
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

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        sns.lineplot(data=df, x="round", y="lambda", ax=axes[0], marker="o")
        axes[0].set_ylabel("lambda")
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].grid(True, linestyle="--", alpha=0.5)

        sns.lineplot(data=df, x="round", y="epochs", ax=axes[1], marker="o")
        axes[1].set_ylabel("epochs")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        sns.lineplot(data=df, x="round", y="query_size", ax=axes[2], marker="o")
        axes[2].set_ylabel("query_size")
        axes[2].set_xlabel("round")
        axes[2].grid(True, linestyle="--", alpha=0.5)

        fig.suptitle(f"Controller Trajectory: {exp}")
        plt.tight_layout()
        fig.savefig(outdir / f"controller_trajectory__{exp}.png", dpi=300)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="e.g. /Users/anykong/AAL_SD/results/runs/20260204_111154_strict")
    parser.add_argument("--output_dir", default=None, help="default: <run_dir>/figures")
    parser.add_argument("--exclude_prefix", default="", help="comma separated prefixes to exclude, e.g. baseline_")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    outdir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (run_dir / "figures")
    outdir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    manifest, exps = discover_experiments(run_dir)
    exp_meta = manifest.get("experiments") if isinstance(manifest.get("experiments"), dict) else {}
    cfg = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}

    exclude_prefixes = [p.strip() for p in str(args.exclude_prefix).split(",") if p.strip()]

    metrics_rows: List[Dict[str, Any]] = []
    curves_rows: List[pd.DataFrame] = []
    ctrl_rows: List[pd.DataFrame] = []

    for ep in exps:
        if any(ep.name.startswith(px) for px in exclude_prefixes):
            continue

        status = load_status_metrics(ep.status_path)
        df_epoch, df_ctrl = parse_trace(ep.trace_path)

        curve = build_round_curve(df_epoch)
        if not curve.empty:
            curve["experiment"] = ep.name
            curves_rows.append(curve)

        cost = compute_cost(df_epoch, df_ctrl)
        metrics_rows.append(
            {
                "experiment": ep.name,
                "description": (exp_meta.get(ep.name, {}) or {}).get("description"),
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

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(outdir / "metrics_summary.csv", index=False, encoding="utf-8")

    curves = pd.concat(curves_rows, ignore_index=True) if curves_rows else pd.DataFrame()
    if not curves.empty:
        curves.to_csv(outdir / "round_curves.csv", index=False, encoding="utf-8")

    ctrl = pd.concat(ctrl_rows, ignore_index=True) if ctrl_rows else pd.DataFrame()
    if not ctrl.empty:
        ctrl.to_csv(outdir / "controller_trajectories.csv", index=False, encoding="utf-8")

    plot_learning_curves(curves, outdir, title=f"Label Efficiency ({run_dir.name})")
    plot_bars(metrics, outdir)
    plot_cost_tradeoff(metrics, outdir)
    plot_controller_trajectories(ctrl, outdir)

    print(f"Saved figures to: {outdir}")
    print(f"Saved csv to: {outdir / 'metrics_summary.csv'}")


if __name__ == "__main__":
    main()