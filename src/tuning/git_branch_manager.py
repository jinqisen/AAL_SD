import os
import subprocess
import json
from datetime import datetime
from typing import Dict, List, Optional


class GitBranchManager:
    def __init__(self, repo_dir: str):
        self.repo_dir = repo_dir
        self._enabled = self._check_git_available()

    def _check_git_available(self) -> bool:
        try:
            subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _run_git(self, *args) -> Optional[str]:
        if not self._enabled:
            return None
        try:
            result = subprocess.run(
                ["git"] + list(args),
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"  [git] Command failed: git {' '.join(args)}: {e.stderr.strip()}")
            return None
        except FileNotFoundError:
            print("  [git] git executable not found")
            self._enabled = False
            return None

    def create_tuning_branch(
        self, iteration: int, base_branch: str = "main"
    ) -> Optional[str]:
        if not self._enabled:
            print("  [git] Skipping branch creation (git not available)")
            return None
        branch_name = (
            f"tuning/iter-{iteration:03d}-{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        existing = self._run_git("branch", "--list", branch_name)
        if existing and branch_name in existing:
            print(f"  [git] Branch {branch_name} already exists, reusing")
            self._run_git("checkout", branch_name)
            return branch_name
        checkout_base = self._run_git("checkout", base_branch)
        if checkout_base is None:
            print(
                f"  [git] Could not checkout {base_branch}, staying on current branch"
            )
        result = self._run_git("checkout", "-b", branch_name)
        if result is None:
            print(f"  [git] Failed to create branch {branch_name}")
            return None
        return branch_name

    def commit_iteration_report(
        self,
        iteration: int,
        analysis_report: Dict,
        proposed_configs: List[Dict],
        code_changes: Optional[List[str]] = None,
    ) -> Optional[str]:
        if not self._enabled:
            print("  [git] Skipping commit (git not available)")
            return None
        report_dir = os.path.join(
            self.repo_dir,
            "AAL-SD-Doc",
            "tuning_reports",
        )
        report_path = os.path.join(report_dir, f"iter_{iteration:03d}_analysis.json")
        try:
            os.makedirs(report_dir, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "iteration": iteration,
                        "timestamp": datetime.now().isoformat(),
                        "analysis": analysis_report,
                        "proposed_experiments": [
                            p.get("experiment_name", p.get("direction", ""))
                            for p in proposed_configs
                        ],
                        "configs": proposed_configs,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except OSError as e:
            print(f"  [git] Failed to write report: {e}")
            return None

        self._run_git("add", report_path)
        if code_changes:
            for path in code_changes:
                self._run_git("add", path)
        best_miou = analysis_report.get("best_miou", "N/A")
        msg = f"tuning iter {iteration}: {len(proposed_configs)} experiments proposed, best_miou={best_miou}"
        commit_result = self._run_git("commit", "-m", msg)
        if commit_result is None:
            print(f"  [git] Commit failed for iteration {iteration}")
            return None
        return self._run_git("rev-parse", "HEAD")
