import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _copy(src: Path, dst: Path, overwrite: bool) -> bool:
    if not src.exists():
        return False
    if dst.exists() and not overwrite:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--multi_seed_group_dir", default=None)
    parser.add_argument("--paper_dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paper_dir = Path(args.paper_dir).expanduser().resolve() if args.paper_dir else Path(__file__).resolve().parent
    repo_root = paper_dir.parent
    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else (repo_root / "results" / "runs" / "20260207_paper").resolve()
    )

    plot_script = (repo_root / "src" / "analysis" / "plot_paper_figures.py").resolve()
    if not plot_script.exists():
        raise FileNotFoundError(str(plot_script))

    intermediate_dir = (paper_dir / "_intermediate").resolve()
    subprocess.run(
        [sys.executable, str(plot_script), "--run_dir", str(run_dir), "--output_dir", str(intermediate_dir)],
        check=True,
    )

    _copy(
        intermediate_dir / "aal_sd_active_learning_loop.png",
        paper_dir / "Figure1b_AAL_SD_Active_Learning_Loop.png",
        overwrite=args.overwrite,
    )
    _copy(
        intermediate_dir / "controller_trajectory__full_model.png",
        paper_dir / "Figure3_Controller_Trajectory_Full_Model.png",
        overwrite=args.overwrite,
    )
    _copy(intermediate_dir / "alc_bar.png", paper_dir / "Figure4_ALC_Bar.png", overwrite=args.overwrite)
    _copy(intermediate_dir / "final_miou_bar.png", paper_dir / "Figure5_Final_mIoU_Bar.png", overwrite=args.overwrite)
    _copy(
        intermediate_dir / "learning_curve_miou_vs_labeled.png",
        paper_dir / "Figure2_Learning_Curves_generated.png",
        overwrite=args.overwrite,
    )
    _copy(
        intermediate_dir / "gradient_diagnostics.png",
        paper_dir / "Figure6_Gradient_Diagnostics.png",
        overwrite=args.overwrite,
    )
    _copy(intermediate_dir / "metrics_summary.csv", paper_dir / "metrics_summary.csv", overwrite=True)
    _copy(intermediate_dir / "round_curves.csv", paper_dir / "round_curves.csv", overwrite=True)
    _copy(intermediate_dir / "controller_trajectories.csv", paper_dir / "controller_trajectories.csv", overwrite=True)
    _copy(intermediate_dir / "round_gradients.csv", paper_dir / "round_gradients.csv", overwrite=True)

    if args.multi_seed_group_dir:
        group_dir = Path(args.multi_seed_group_dir).expanduser().resolve()
        intermediate_ms = (paper_dir / "_intermediate_multiseed").resolve()
        subprocess.run(
            [
                sys.executable,
                str(plot_script),
                "--multi_seed_group_dir",
                str(group_dir),
                "--output_dir",
                str(intermediate_ms),
            ],
            check=True,
        )

        _copy(
            intermediate_ms / "multiseed_learning_curve_miou_vs_labeled.png",
            paper_dir / "Figure7_MultiSeed_Learning_Curves.png",
            overwrite=args.overwrite,
        )
        _copy(
            intermediate_ms / "multiseed_alc_bar.png",
            paper_dir / "Figure8_MultiSeed_ALC_Bar.png",
            overwrite=args.overwrite,
        )
        _copy(
            intermediate_ms / "multiseed_final_miou_bar.png",
            paper_dir / "Figure9_MultiSeed_Final_mIoU_Bar.png",
            overwrite=args.overwrite,
        )
        _copy(intermediate_ms / "multiseed_metrics_summary.csv", paper_dir / "multiseed_metrics_summary.csv", overwrite=True)


if __name__ == "__main__":
    main()
