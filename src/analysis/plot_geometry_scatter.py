import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
from analyze_geometry_rho_evolution import collect_round_geometry

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_files", nargs="+", type=Path)
    parser.add_argument("--out_dir", type=Path, default=Path("results/analysis"))
    return parser.parse_args()

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for tf in args.trace_files:
        try:
            for r in collect_round_geometry(tf):
                r['source'] = tf.parent.name.split('_')[-1]
                all_rows.append(r)
        except Exception as e:
            print(f"Failed {tf}: {e}")
    if not all_rows: return 1
    
    valid_dk = [r for r in all_rows if r.get('D_K') is not None and r.get('sens_up') is not None]
    if valid_dk:
        plt.figure()
        for src in set(r['source'] for r in valid_dk):
            pts = [r for r in valid_dk if r['source'] == src]
            plt.scatter([r['D_K'] for r in pts], [r['sens_up'] for r in pts], label=src, alpha=0.7)
        plt.xlabel("D_K"); plt.ylabel("sens_up"); plt.legend(); plt.grid(True)
        plt.savefig(args.out_dir / "dk_vs_sens_up.png")
        plt.close()
        
    valid_rho = [r for r in all_rows if r.get('rho_uk') is not None and r.get('sens_up') is not None]
    if valid_rho:
        plt.figure()
        for src in set(r['source'] for r in valid_rho):
            pts = [r for r in valid_rho if r['source'] == src]
            plt.scatter([r['rho_uk'] for r in pts], [r['sens_up'] for r in pts], label=src, alpha=0.7)
        plt.xlabel("rho_uk"); plt.ylabel("sens_up"); plt.legend(); plt.grid(True)
        plt.savefig(args.out_dir / "rho_vs_sens_up.png")
        plt.close()
    return 0

if __name__ == "__main__":
    main()
