
import os
import sys
import time
import glob
import json
import argparse
import logging
from datetime import datetime
import subprocess
from typing import Dict, List, Optional, Any, Union

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.agent_manager import SiliconFlowClient
from src.config import Config

MONITOR_INTERVAL = 30
STALL_THRESHOLD = 600
RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'runs')

# Setup basic logging
logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger("Monitor")

class TrainingMonitor:
    def __init__(
        self,
        run_id: Optional[str] = None,
        run_ids: Optional[Union[List[str], str]] = None,
        runs_dir: Optional[str] = None,
        monitor_interval: int = MONITOR_INTERVAL,
        stall_threshold: int = STALL_THRESHOLD,
        enable_llm: bool = False,
        enable_summary: bool = True,
        enable_process_log: bool = False,
        process_grace_seconds: int = 60,
    ):
        self.config = Config()
        self.run_id = run_id
        parsed_run_ids: List[str] = []
        if isinstance(run_ids, list):
            parsed_run_ids = [str(x).strip() for x in run_ids if str(x).strip()]
        elif isinstance(run_ids, str):
            parsed_run_ids = [x.strip() for x in run_ids.split(",") if x.strip()]
        if not parsed_run_ids and run_id:
            parsed_run_ids = [str(run_id).strip()]
        self.run_ids: Optional[List[str]] = parsed_run_ids or None
        self.runs_dir = runs_dir or RUNS_DIR
        self.monitor_interval = int(monitor_interval)
        self.stall_threshold = int(stall_threshold)
        self.llm_client: Optional[SiliconFlowClient] = self._setup_llm_client() if bool(enable_llm) else None
        self.enable_summary = bool(enable_summary)
        self.last_report_time: Dict[str, datetime] = {}  # exp_name -> timestamp (for LLM rate limit)
        self.last_round_seen: Dict[str, int] = {}        # exp_name -> round_number
        self.enable_process_log = bool(enable_process_log)
        self.process_grace_seconds = int(process_grace_seconds)
        self._last_main_process_seen_at: Optional[float] = None
        self._process_log_path: Optional[str] = None
        self._last_process_counts: Optional[Dict[str, int]] = None
        if self.enable_process_log and self.run_id:
            reports_dir = os.path.join(self.runs_dir, self.run_id, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            self._process_log_path = os.path.join(reports_dir, "process_watch.jsonl")

    def _count_pids_by_cmd_substring(self, needle: str) -> int:
        try:
            out = subprocess.run(
                ["pgrep", "-fl", needle],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            ).stdout
            lines = [ln for ln in str(out).splitlines() if ln.strip()]
            return int(len(lines))
        except Exception:
            return 0

    def _format_count_delta(self, key: str, value: int) -> str:
        try:
            prev = self._last_process_counts.get(key) if isinstance(self._last_process_counts, dict) else None
            if isinstance(prev, int):
                delta = int(value) - int(prev)
                sign = "+" if delta >= 0 else ""
                return f"{value}({sign}{delta})"
        except Exception:
            pass
        return str(value)

    def _format_memory_delta(self, key: str, value_mb: float) -> str:
        try:
            prev = self._last_process_counts.get(f"{key}_mem") if isinstance(self._last_process_counts, dict) else None
            if isinstance(prev, (int, float)):
                delta = value_mb - float(prev)
                sign = "+" if delta >= 0 else ""
                return f"{value_mb:.1f}MB({sign}{delta:.1f})"
        except Exception:
            pass
        return f"{value_mb:.1f}MB"

    def _print_process_health(self, snapshot: Dict[str, Any]) -> None:
        counts = snapshot.get("counts") if isinstance(snapshot, dict) else None
        memory = snapshot.get("memory") if isinstance(snapshot, dict) else {}
        
        if not isinstance(counts, dict):
            return
        main_running = bool(snapshot.get("main_running"))
        python_s = self._format_count_delta("python", int(counts.get("python") or 0))
        shm_s = self._format_count_delta("torch_shm_manager", int(counts.get("torch_shm_manager") or 0))
        spawn_s = self._format_count_delta("spawn_main", int(counts.get("spawn_main") or 0))
        tracker_s = self._format_count_delta("resource_tracker", int(counts.get("resource_tracker") or 0))
        main_s = self._format_count_delta("main", int(counts.get("main") or 0))
        
        python_mem_s = self._format_memory_delta("python", float(memory.get("python_mb") or 0))
        
        status_str = "Main Script Running" if main_running else "Main Script STOPPED"
        
        print(
            f"System Health: {status_str} | main={main_s} python={python_s} "
            f"spawn_main={spawn_s} resource_tracker={tracker_s} torch_shm_manager={shm_s} | "
            f"Mem(Python)={python_mem_s}"
        )

        try:
            self._last_process_counts = {k: int(v) for k, v in counts.items() if isinstance(v, int)}
            if memory:
                self._last_process_counts["python_mem"] = float(memory.get("python_mb") or 0)
        except Exception:
            self._last_process_counts = None

    def _setup_llm_client(self) -> Optional[SiliconFlowClient]:
        try:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llm_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    llm_cfg = json.load(f)
                return SiliconFlowClient(
                    api_key=llm_cfg.get("api_key", ""),
                    base_url=llm_cfg.get("base_url", ""),
                    model=llm_cfg.get("model", "Pro/deepseek-ai/DeepSeek-V3.2"),
                    temperature=0.1,
                    max_retries=int(llm_cfg.get("max_retries", 3)),
                    retry_delay=float(llm_cfg.get("retry_base_seconds", 5.0)),
                    retry_backoff=float(llm_cfg.get("retry_backoff", 2.0))
                )
        except Exception as e:
            logger.warning(f"Failed to setup LLM client: {e}")
        return None

    def _expand_multi_seed_group(self, group_dir: str) -> List[str]:
        manifest_path = os.path.join(group_dir, "multi_seed_manifest.json")
        if not os.path.exists(manifest_path):
            return []
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            run_ids = payload.get("run_ids") if isinstance(payload, dict) else None
            if not isinstance(run_ids, list):
                return []
            expanded = []
            for rid in run_ids:
                rid_str = str(rid).strip()
                if not rid_str:
                    continue
                candidate = os.path.join(self.runs_dir, rid_str)
                if os.path.isdir(candidate):
                    expanded.append(candidate)
            return expanded
        except Exception:
            return []

    def _resolve_run_dir(self, token: str) -> Optional[str]:
        raw = str(token).strip()
        if not raw:
            return None

        if os.path.isabs(raw) and os.path.isdir(raw):
            return os.path.normpath(raw)

        candidates: List[str] = []
        if any(sep in raw for sep in (os.sep, "/")) or raw.startswith("."):
            candidates.append(os.path.normpath(os.path.join(os.getcwd(), raw)))
            candidates.append(os.path.normpath(os.path.join(self.runs_dir, raw)))
        else:
            candidates.append(os.path.normpath(os.path.join(self.runs_dir, raw)))

        for c in candidates:
            if os.path.isdir(c):
                return c
        return None

    def _run_label(self) -> str:
        if self.run_id:
            return str(self.run_id)
        if self.run_ids:
            return str(self.run_ids[0])
        return "-"

    def find_latest_run_dirs(self) -> List[str]:
        try:
            if self.run_ids:
                dirs: List[str] = []
                for rid in self.run_ids:
                    candidate = self._resolve_run_dir(rid)
                    if not candidate:
                        continue
                    expanded = self._expand_multi_seed_group(candidate)
                    if expanded:
                        dirs.extend(expanded)
                    else:
                        dirs.append(candidate)
                seen = set()
                unique_dirs = []
                for d in sorted(dirs, key=lambda p: os.path.basename(p)):
                    if d in seen:
                        continue
                    seen.add(d)
                    unique_dirs.append(d)
                return unique_dirs

            candidates = []
            for p in sorted(glob.glob(os.path.join(self.runs_dir, "*"))):
                if not os.path.isdir(p):
                    continue
                if glob.glob(os.path.join(p, "*_status.json")):
                    candidates.append(p)
            if not candidates:
                return []
            candidates.sort(key=os.path.getmtime, reverse=True)
            return [candidates[0]]
        except Exception:
            return []

    def _read_planned_experiments(self, run_dir: str) -> Optional[List[str]]:
        manifest_path = os.path.join(run_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                return None
            experiments = payload.get("experiments")
            if isinstance(experiments, dict):
                names = [str(k).strip() for k in experiments.keys()]
                return [n for n in names if n]
            if isinstance(experiments, list):
                names: List[str] = []
                for item in experiments:
                    if isinstance(item, str):
                        name = item.strip()
                        if name:
                            names.append(name)
                        continue
                    if isinstance(item, dict):
                        raw = item.get("name")
                        name = str(raw).strip() if raw is not None else ""
                        if name:
                            names.append(name)
                return names or None
        except Exception:
            return None
        return None

    def _print_stage_summary(self, status_table: List[Dict[str, Any]], planned: Optional[List[str]]) -> None:
        completed = []
        running = []
        stalled = []
        failed = []
        unknown = []

        for row in status_table:
            status = str(row.get("status") or "unknown")
            if status in ("completed", "finished"):
                completed.append(row)
            elif status == "running":
                running.append(row)
            elif status == "stalled":
                stalled.append(row)
            elif status == "failed":
                failed.append(row)
            else:
                unknown.append(row)

        completed.sort(key=lambda r: str(r.get("name") or ""))
        running.sort(key=lambda r: str(r.get("name") or ""))
        stalled.sort(key=lambda r: str(r.get("name") or ""))
        failed.sort(key=lambda r: str(r.get("name") or ""))
        unknown.sort(key=lambda r: str(r.get("name") or ""))

        present_names = {str(r.get("name") or "").strip() for r in status_table}
        present_names = {n for n in present_names if n}
        not_started: Optional[List[str]] = None
        if planned:
            planned_clean = [str(x).strip() for x in planned if str(x).strip()]
            not_started = [x for x in planned_clean if x not in present_names]

        def _fmt_completed(r: Dict[str, Any]) -> str:
            name = str(r.get("name") or "-")
            round_str = str(r.get("round") or "-")
            return f"- {name} (Round {round_str} finished)"

        def _fmt_running(r: Dict[str, Any]) -> str:
            name = str(r.get("name") or "-")
            round_str = str(r.get("round") or "-")
            epoch_str = str(r.get("epoch") or "-")
            miou = r.get("mIoU")
            miou_str = f"{float(miou):.4f}" if isinstance(miou, (int, float)) else str(miou)
            updated = str(r.get("updated") or "-")
            return f"- {name}: Round {round_str} / Epoch {epoch_str}, mIoU≈{miou_str} (updated_at {updated})"

        print("\nStage Summary")
        print("-" * 100)
        print(f"已完成（{len(completed)} 个）")
        for r in completed:
            print(_fmt_completed(r))

        print(f"\n进行中（{len(running)} 个）")
        for r in running:
            print(_fmt_running(r))

        if stalled:
            print(f"\n已停滞（{len(stalled)} 个）")
            for r in stalled:
                print(_fmt_running(r))

        if failed:
            print(f"\n已失败（{len(failed)} 个）")
            for r in failed:
                name = str(r.get('name') or '-')
                updated = str(r.get('updated') or '-')
                print(f"- {name} (updated_at {updated})")

        if unknown:
            print(f"\n未知状态（{len(unknown)} 个）")
            for r in unknown:
                name = str(r.get('name') or '-')
                status = str(r.get('status') or '-')
                updated = str(r.get('updated') or '-')
                print(f"- {name}: {status} (updated_at {updated})")

        if not_started is not None:
            print(f"\n尚未开始/未生成 status（{len(not_started)} 个）")
            if not_started:
                print("- " + " / ".join(not_started))

    def parse_experiment_history(self, filepath: str, trace_path: Optional[str] = None) -> Dict[str, Any]:
        history: Dict[str, Any] = {
            "rounds": [],
            "epochs": [],
            "status": "unknown",
            "last_update": datetime.min,
            "progress": {},
        }

        try:
            mtime = os.path.getmtime(filepath)
            history["last_update"] = datetime.fromtimestamp(mtime)
            if filepath.endswith(".md"):
                import re

                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                current_round = 0
                for line in lines:
                    r_match = re.search(r"## Round (\d+)", line)
                    if r_match:
                        current_round = int(r_match.group(1))

                    e_match = re.search(r"Epoch (\d+): Loss=([\d\.]+), mIoU=([\d\.]+)", line)
                    if e_match:
                        epoch = int(e_match.group(1))
                        loss = float(e_match.group(2))
                        miou = float(e_match.group(3))
                        history["epochs"].append({
                            "round": current_round,
                            "epoch": epoch,
                            "loss": loss,
                            "mIoU": miou
                        })

                for line in lines:
                    s_match = re.search(r"Round=(\d+), Labeled=(\d+), mIoU=([\d\.]+), F1=([\d\.]+)", line)
                    if s_match:
                        history["rounds"].append({
                            "round": int(s_match.group(1)),
                            "labeled": int(s_match.group(2)),
                            "mIoU": float(s_match.group(3)),
                            "F1": float(s_match.group(4))
                        })

                content_tail = "".join(lines[-20:])
                if "实验汇总" in content_tail:
                    history["status"] = "finished"
                elif (datetime.now() - history["last_update"]).total_seconds() > self.stall_threshold:
                    history["status"] = "stalled"
                else:
                    history["status"] = "running"
            else:
                if trace_path and os.path.exists(trace_path):
                    try:
                        trace_mtime = os.path.getmtime(trace_path)
                        if isinstance(trace_mtime, (int, float)) and trace_mtime > float(mtime):
                            history["last_update"] = datetime.fromtimestamp(trace_mtime)
                    except Exception:
                        pass

                with open(filepath, "r", encoding="utf-8") as f:
                    status_payload = json.load(f)
                if isinstance(status_payload, dict):
                    history["status"] = str(status_payload.get("status") or "unknown")
                    progress = status_payload.get("progress")
                    if isinstance(progress, dict):
                        history["progress"] = dict(progress)

                if trace_path and os.path.exists(trace_path):
                    with open(trace_path, "r", encoding="utf-8") as f:
                        tail = f.readlines()[-400:]
                    events = []
                    for line in tail:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(obj, dict) and obj.get("type") == "epoch_end":
                            events.append(obj)

                    if events:
                        recent_epochs = events[-30:]
                        for e in recent_epochs:
                            try:
                                history["epochs"].append({
                                    "round": int(e.get("round")),
                                    "epoch": int(e.get("epoch")),
                                    "loss": float(e.get("loss")),
                                    "mIoU": float(e.get("mIoU")),
                                    "F1": float(e.get("f1")),
                                })
                            except Exception:
                                continue

                        round_best: Dict[int, Dict[str, Any]] = {}
                        for e in events:
                            try:
                                r = int(e.get("round"))
                                miou = float(e.get("mIoU"))
                                f1 = float(e.get("f1"))
                                labeled = int(e.get("labeled_size"))
                            except Exception:
                                continue
                            prev = round_best.get(r)
                            if prev is None or miou > float(prev.get("mIoU", -1.0)):
                                round_best[r] = {"round": r, "labeled": labeled, "mIoU": miou, "F1": f1}
                        history["rounds"] = [round_best[k] for k in sorted(round_best.keys())][-10:]

            if (datetime.now() - history["last_update"]).total_seconds() > self.stall_threshold and history["status"] == "running":
                history["status"] = "stalled"

        except Exception as e:
            logger.error(f"Error parsing {filepath}: {e}")

        return history

    def detect_anomalies(self, exp_name: str, history: Dict[str, Any]) -> List[str]:
        """Identifies potential issues in training history."""
        _ = exp_name
        anomalies = []
        
        epochs = history["epochs"]
        rounds = history["rounds"]
        
        if not epochs:
            return anomalies

        # 1. Loss Analysis (Epoch level)
        recent_epochs = epochs[-5:]
        if len(recent_epochs) >= 2:
            # Check for sudden drop (potential leakage)
            last_loss = recent_epochs[-1]["loss"]
            prev_loss = recent_epochs[-2]["loss"]
            if prev_loss > 0.1 and last_loss < prev_loss * 0.1:
                anomalies.append(f"Loss dropped suspiciously fast: {prev_loss:.4f} -> {last_loss:.4f}")
            
            # Check for loss explosion
            if last_loss > 10.0:
                anomalies.append(f"Loss explosion detected: {last_loss:.4f}")

        # 2. Performance Analysis (Round level)
        if len(rounds) >= 2:
            last_round = rounds[-1]
            prev_round = rounds[-2]
            
            # Check for catastrophic forgetting / performance drop
            if last_round["mIoU"] < prev_round["mIoU"] - 0.05:
                anomalies.append(f"Significant performance drop: mIoU {prev_round['mIoU']:.4f} -> {last_round['mIoU']:.4f}")
            
            # Check for stagnation (if we have > 3 rounds)
            if len(rounds) >= 4:
                recent_mious = [r["mIoU"] for r in rounds[-3:]]
                if max(recent_mious) - min(recent_mious) < 0.002:
                    anomalies.append("Performance stagnation detected (mIoU change < 0.002 over 3 rounds)")

        return anomalies

    def generate_stage_report(self, exp_name: str, history: Dict[str, Any], reports_dir: Optional[str] = None) -> None:
        """Generates a markdown report for the latest round comparison."""
        rounds = history["rounds"]
        if not rounds:
            return

        current_round = rounds[-1]
        round_num = current_round["round"]
        
        # Avoid duplicate reports for the same round
        if self.last_round_seen.get(exp_name) == round_num:
            return
            
        self.last_round_seen[exp_name] = round_num
        
        prev_round = rounds[-2] if len(rounds) >= 2 else None
        
        report_lines = [
            f"# Stage Report: {exp_name} - Round {round_num}",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Performance Metrics",
            f"- **mIoU:** {current_round['mIoU']:.4f}",
            f"- **F1 Score:** {current_round['F1']:.4f}",
            f"- **Labeled Samples:** {current_round['labeled']}",
            ""
        ]
        
        if prev_round:
            miou_diff = current_round['mIoU'] - prev_round['mIoU']
            f1_diff = current_round['F1'] - prev_round['F1']
            report_lines.extend([
                "## Comparison vs Previous Round",
                f"- **mIoU Change:** {miou_diff:+.4f} ({'IMPROVED' if miou_diff > 0 else 'DEGRADED'})",
                f"- **F1 Change:** {f1_diff:+.4f}",
                ""
            ])
            
        if not reports_dir:
            reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        report_path = os.path.join(reports_dir, f"{exp_name}_round_{round_num}_report.md")
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(report_lines))
            logger.info(f"Generated stage report for {exp_name} Round {round_num}")
        except Exception as e:
            logger.error(f"Failed to write stage report: {e}")

    def generate_llm_report(self, exp_name: str, history: Dict[str, Any], anomalies: List[str]) -> str:
        """Uses LLM to analyze the situation and generate a report."""
        if not self.llm_client:
            return "LLM Client not available."

        # Construct Prompt
        epochs_summary = history["epochs"][-5:] if history["epochs"] else []
        rounds_summary = history["rounds"][-5:] if history["rounds"] else []
        
        prompt = f"""
        Analyze the following training log for Active Learning experiment '{exp_name}'.
        
        Status: {history['status']}
        Last Update: {history['last_update']}
        Detected Anomalies: {anomalies}
        
        Recent Epoch Data (Last 5):
        {json.dumps(epochs_summary, indent=2)}
        
        Recent Round Data (Last 5):
        {json.dumps(rounds_summary, indent=2)}
        
        Please provide:
        1. Assessment of training health.
        2. Possible causes for any anomalies.
        3. Recommendations for adjustment (learning rate, batch size, query strategy, etc.).
        
        Keep it concise.
        """
        
        try:
            logger.info(f"Requesting LLM analysis for {exp_name}...")
            response = self.llm_client.chat([{"role": "user", "content": prompt}])
            return response
        except Exception as e:
            return f"LLM Analysis Failed: {e}"

    def run_cycle(self):
        run_dirs = self.find_latest_run_dirs()
        if not run_dirs:
            print("No active run directory found.")
            self._check_processes()
            return
        self._check_processes()

        by_run: List[Dict[str, Any]] = []

        for current_run_dir in run_dirs:
            run_id = os.path.basename(current_run_dir)
            reports_dir = os.path.join(current_run_dir, "reports")
            status_files = sorted(glob.glob(os.path.join(current_run_dir, "*_status.json")))
            planned_experiments = self._read_planned_experiments(current_run_dir)

            status_table: List[Dict[str, Any]] = []

            for status_path in status_files:
                exp_name = os.path.basename(status_path).replace("_status.json", "")
                trace_path = os.path.join(current_run_dir, f"{exp_name}_trace.jsonl")
                history = self.parse_experiment_history(status_path, trace_path)
                anomalies = self.detect_anomalies(exp_name, history)

                progress_epoch = history.get("progress", {}).get("epoch")
                if progress_epoch == "finished" or history.get("status") in ("completed", "failed"):
                    self.generate_stage_report(exp_name, history, reports_dir=reports_dir)

                last_epoch = history["epochs"][-1] if history["epochs"] else {"round": "-", "epoch": "-", "loss": "-", "mIoU": "-"}
                last_round_metric = history["rounds"][-1]["mIoU"] if history["rounds"] else "-"
                last_f1_metric = history["rounds"][-1]["F1"] if history["rounds"] else "-"
                updated = history.get("last_update")
                updated_str = updated.strftime("%H:%M:%S") if isinstance(updated, datetime) else "-"

                status_table.append({
                    "name": exp_name,
                    "round": last_epoch["round"],
                    "epoch": last_epoch["epoch"],
                    "loss": last_epoch["loss"],
                    "mIoU": last_epoch["mIoU"],
                    "round_mIoU": last_round_metric,
                    "round_F1": last_f1_metric,
                    "status": history["status"],
                    "updated": updated_str,
                    "anomalies": anomalies
                })

                if anomalies:
                    last_report = self.last_report_time.get(exp_name, datetime.min)
                    if (datetime.now() - last_report).total_seconds() > 3600:
                        print(f"  > Analyzing anomalies for {exp_name}...")
                        report = self.generate_llm_report(exp_name, history, anomalies)

                        os.makedirs(reports_dir, exist_ok=True)
                        report_path = os.path.join(reports_dir, f"{exp_name}_anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
                        with open(report_path, 'w', encoding='utf-8') as f:
                            f.write(f"# Anomaly Analysis: {exp_name}\n\n")
                            f.write(f"## Anomalies\n{json.dumps(anomalies, indent=2)}\n\n")
                            f.write(f"## LLM Insight\n{report}\n")

                        print(f"  > Anomaly report saved to {os.path.basename(report_path)}")
                        self.last_report_time[exp_name] = datetime.now()

            by_run.append({
                "run_id": run_id,
                "run_dir": current_run_dir,
                "reports_dir": reports_dir,
                "planned_experiments": planned_experiments,
                "status_table": status_table,
            })

        if len(by_run) > 1:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Multi-Run Overview")
            print("=" * 100)
            print(f"{'Run':<42} | {'Done':<4} | {'Run':<4} | {'Stal':<4} | {'Fail':<4} | {'Unk':<4} | {'Latest':<8}")
            print("-" * 100)
            for r in sorted(by_run, key=lambda x: str(x.get("run_id") or "")):
                status_table = r.get("status_table") if isinstance(r.get("status_table"), list) else []
                counts = {"completed": 0, "finished": 0, "running": 0, "stalled": 0, "failed": 0, "unknown": 0}
                latest = "-"
                try:
                    times = []
                    for row in status_table:
                        st = str(row.get("status") or "unknown")
                        if st not in counts:
                            st = "unknown"
                        counts[st] += 1
                        upd = row.get("updated")
                        if isinstance(upd, str) and upd.strip() and upd.strip() != "-":
                            times.append(upd.strip())
                    if times:
                        latest = sorted(times)[-1]
                except Exception:
                    pass
                done_n = counts["completed"] + counts["finished"]
                print(
                    f"{str(r.get('run_id') or '-'):<42} | {done_n:<4} | {counts['running']:<4} | "
                    f"{counts['stalled']:<4} | {counts['failed']:<4} | {counts['unknown']:<4} | {latest:<8}"
                )

        for r in sorted(by_run, key=lambda x: str(x.get("run_id") or "")):
            current_run_dir = str(r.get("run_dir") or "")
            run_id = str(r.get("run_id") or "-")
            reports_dir = str(r.get("reports_dir") or "-")
            planned_experiments = r.get("planned_experiments")
            status_table = r.get("status_table") if isinstance(r.get("status_table"), list) else []

            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Monitoring Run: {run_id}")
            print("=" * 100)
            print(f"Run: {run_id} | Reports: {reports_dir}")
            print(f"{'Experiment':<25} | {'Rnd':<4} | {'Ep':<8} | {'Loss':<10} | {'mIoU(E)':<8} | {'mIoU(R)':<8} | {'F1(R)':<8} | {'Upd':<8} | {'Status':<10} | {'Warnings'}")
            print("-" * 140)
            for row in status_table:
                warn_flag = "!" if row.get("anomalies") else ""
                anomalies_n = len(row.get("anomalies") or []) if isinstance(row.get("anomalies"), list) else 0
                print(
                    f"{str(row.get('name')):<25} | {str(row.get('round')):<4} | {str(row.get('epoch')):<8} | "
                    f"{str(row.get('loss')):<10} | {str(row.get('mIoU')):<8} | {str(row.get('round_mIoU')):<8} | "
                    f"{str(row.get('round_F1')):<8} | {str(row.get('updated')):<8} | {str(row.get('status')):<10} | "
                    f"{warn_flag} {anomalies_n} issues"
                )

            if self.enable_summary:
                self._print_stage_summary(status_table, planned_experiments)

    def _check_processes(self):
        snapshot: Optional[Dict[str, Any]] = None
        try:
            result = subprocess.run(
                ["ps", "-axo", "pid,ppid,rss,command"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError("ps_failed")

            lines = [ln.rstrip("\n") for ln in str(result.stdout).splitlines() if ln.strip()]
            rows: List[Dict[str, Any]] = []
            
            total_python_rss_kb = 0
            
            for ln in lines[1:]:
                parts = ln.strip().split(None, 3)
                if len(parts) < 4:
                    continue
                pid_s, ppid_s, rss_s, cmd = parts[0], parts[1], parts[2], parts[3]
                try:
                    pid = int(pid_s)
                    ppid = int(ppid_s)
                    rss = int(rss_s)
                except ValueError:
                    continue
                rows.append({"pid": pid, "ppid": ppid, "rss": rss, "cmd": cmd})
                
                if "python" in cmd:
                    total_python_rss_kb += rss

            main_markers = ("run_parallel_strict.py", "src/main.py")
            main_pids = {r["pid"] for r in rows if any(m in r["cmd"] for m in main_markers)}
            python_pids = {r["pid"] for r in rows if "python" in r["cmd"]}
            torch_shm = [r for r in rows if "torch_shm_manager" in r["cmd"]]
            resource_tracker = [r for r in rows if "multiprocessing.resource_tracker" in r["cmd"]]
            spawn_main = [r for r in rows if "multiprocessing.spawn" in r["cmd"] or "spawn_main" in r["cmd"]]

            main_running = bool(main_pids)
            now_ts = time.time()
            if main_running:
                self._last_main_process_seen_at = now_ts

            grace_ok = True
            if not main_running and self._last_main_process_seen_at is not None:
                grace_ok = (now_ts - float(self._last_main_process_seen_at)) <= float(self.process_grace_seconds)

            orphan_shm = []
            if (not main_running) and (not grace_ok) and torch_shm:
                for r in torch_shm:
                    if int(r["ppid"]) not in python_pids:
                        orphan_shm.append(r)

            snapshot = {
                "ts": datetime.now().isoformat(),
                "run_id": self._run_label(),
                "main_running": main_running,
                "counts": {
                    "python": int(len(python_pids)),
                    "main": int(len(main_pids)),
                    "torch_shm_manager": int(len(torch_shm)),
                    "resource_tracker": int(len(resource_tracker)),
                    "spawn_main": int(len(spawn_main)),
                },
                "memory": {
                    "python_mb": total_python_rss_kb / 1024.0
                },
                "orphan_torch_shm_manager": [
                    {"pid": r["pid"], "ppid": r["ppid"], "cmd": r["cmd"]}
                    for r in orphan_shm[:50]
                ],
            }
        except Exception:
            snapshot = {
                "ts": datetime.now().isoformat(),
                "run_id": self._run_label(),
                "main_running": bool(self._count_pids_by_cmd_substring("run_parallel_strict.py") or self._count_pids_by_cmd_substring("src/main.py")),
                "counts": {
                    "python": int(self._count_pids_by_cmd_substring("python")),
                    "main": int(self._count_pids_by_cmd_substring("run_parallel_strict.py") + self._count_pids_by_cmd_substring("src/main.py")),
                    "torch_shm_manager": int(self._count_pids_by_cmd_substring("torch_shm_manager")),
                    "resource_tracker": int(self._count_pids_by_cmd_substring("multiprocessing.resource_tracker")),
                    "spawn_main": int(self._count_pids_by_cmd_substring("multiprocessing.spawn")),
                },
                "memory": {"python_mb": 0.0},
                "orphan_torch_shm_manager": [],
            }

        if self._process_log_path and snapshot:
            try:
                with open(self._process_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
            except Exception:
                pass

        if snapshot:
            self._print_process_health(snapshot)
            try:
                if (not bool(snapshot.get("main_running"))) and snapshot.get("orphan_torch_shm_manager"):
                    orphan_n = len(snapshot.get("orphan_torch_shm_manager") or [])
                    if orphan_n:
                        print(
                            f"WARNING: Orphaned torch_shm_manager detected (n={orphan_n}). "
                            f"Suggested cleanup: pkill -x torch_shm_manager"
                        )
            except Exception:
                pass

def main():
    parser = argparse.ArgumentParser(description="Monitor and report training status for a specific run_id")
    parser.add_argument("--run-id", type=str, default=None, help="Monitor a specific run_id under results/runs/")
    parser.add_argument("--run-ids", type=str, default="", help="Comma-separated run_ids to monitor")
    parser.add_argument("--run-dirs", type=str, default="", help="Comma-separated run directories to monitor (absolute or relative paths)")
    parser.add_argument("--runs-dir", type=str, default=RUNS_DIR, help="Base runs directory")
    parser.add_argument("--interval", type=int, default=MONITOR_INTERVAL, help="Seconds between monitoring cycles")
    parser.add_argument("--stall-threshold", type=int, default=STALL_THRESHOLD, help="Seconds without update to mark stalled")
    parser.add_argument("--enable-llm", action="store_true")
    parser.add_argument("--summary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--proc-log", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--proc-grace", type=int, default=60)
    args = parser.parse_args()
    run_ids = [x.strip() for x in str(args.run_ids).split(",") if x.strip()]
    run_dirs = [x.strip() for x in str(args.run_dirs).split(",") if x.strip()]
    targets = run_dirs + run_ids

    monitor = TrainingMonitor(
        run_id=args.run_id,
        run_ids=targets or None,
        runs_dir=args.runs_dir,
        monitor_interval=args.interval,
        stall_threshold=args.stall_threshold,
        enable_llm=bool(args.enable_llm),
        enable_summary=bool(args.summary),
        enable_process_log=bool(args.proc_log),
        process_grace_seconds=int(args.proc_grace),
    )
    print("Starting Advanced Training Monitor...")
    print(f"Runs Directory: {monitor.runs_dir}")
    if monitor.run_ids:
        print(f"Target Run IDs: {', '.join(monitor.run_ids)}")
    elif monitor.run_id:
        print(f"Target Run ID: {monitor.run_id}")
    
    while True:
        try:
            monitor.run_cycle()
            time.sleep(monitor.monitor_interval)
        except KeyboardInterrupt:
            print("Stopping...")
            break
        except Exception as e:
            logger.error(f"Monitor Loop Error: {e}")
            time.sleep(monitor.monitor_interval)

if __name__ == "__main__":
    main()
