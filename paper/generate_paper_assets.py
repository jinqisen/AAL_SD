from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_RUN_IDS = [
    "baseline_20260228_124857_seed42",
    "baseline_20260228_124857_seed43",
    "baseline_20260228_124857_seed44",
    "baseline_20260228_124857_seed45",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_dir", default=None)
    parser.add_argument("--run_dirs", nargs="*", default=None)
    args = parser.parse_args()

    paper_dir = (
        Path(args.paper_dir).expanduser().resolve()
        if args.paper_dir
        else Path(__file__).resolve().parent
    )
    repo_root = paper_dir.parent
    script_path = (
        repo_root / "src" / "analysis" / "build_multiseed_paper_assets.py"
    ).resolve()
    run_dirs = args.run_dirs or [
        str((repo_root / "results" / "runs" / run_id).resolve())
        for run_id in DEFAULT_RUN_IDS
    ]

    cmd = [
        sys.executable,
        str(script_path),
        "--paper_dir",
        str(paper_dir),
        "--run_dirs",
        *run_dirs,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
