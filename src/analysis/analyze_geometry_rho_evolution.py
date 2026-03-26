import argparse
import json
from pathlib import Path


def iter_trace_entries(trace_path: Path):
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def collect_round_geometry(trace_path: Path):
    rows = {}
    for entry in iter_trace_entries(trace_path):
        entry_type = entry.get("type")
        if entry_type == "round_summary":
            round_idx = entry.get("round")
            ranking = entry.get("ranking")
            if not isinstance(ranking, dict):
                continue
            geometry = ranking.get("selection_geometry")
            if not isinstance(geometry, dict):
                continue
            rows[int(round_idx)] = {
                "round": int(round_idx),
                "lambda_effective": ranking.get("lambda_effective"),
                "rho_uk": geometry.get("spearman_rho_uk"),
                "rho_pvalue": geometry.get("spearman_pvalue_uk"),
                "D_U": geometry.get("boundary_u_std"),
                "D_K": geometry.get("boundary_k_std"),
                "sens_up": geometry.get("sens_up"),
                "sens_down": geometry.get("sens_down"),
                "asymmetry_ratio": geometry.get("asymmetry_ratio"),
                "crossing_density": geometry.get("crossing_density"),
                "boundary_n": geometry.get("boundary_n"),
            }
        elif entry_type == "score_snapshot":
            round_idx = entry.get("round")
            rows.setdefault(
                int(round_idx),
                {
                    "round": int(round_idx),
                    "lambda_effective": None,
                    "rho_uk": None,
                    "rho_pvalue": None,
                    "D_U": None,
                    "D_K": None,
                    "sens_up": None,
                    "sens_down": None,
                    "asymmetry_ratio": None,
                    "crossing_density": None,
                    "boundary_n": None,
                },
            )
            boundary_rows = entry.get("boundary_rows") or []
            boundary_u = [
                float(r["uncertainty"])
                for r in boundary_rows
                if isinstance(r, dict) and r.get("uncertainty") is not None
            ]
            boundary_k = [
                float(r["knowledge_gain"])
                for r in boundary_rows
                if isinstance(r, dict) and r.get("knowledge_gain") is not None
            ]
            rows[int(round_idx)].update(
                {
                    "lambda_effective": rows[int(round_idx)]["lambda_effective"]
                    or (entry.get("rows") or [{}])[0].get("lambda_t"),
                    "D_U": (sum((x - (sum(boundary_u) / len(boundary_u))) ** 2 for x in boundary_u) / len(boundary_u)) ** 0.5
                    if boundary_u
                    else None,
                    "D_K": (sum((x - (sum(boundary_k) / len(boundary_k))) ** 2 for x in boundary_k) / len(boundary_k)) ** 0.5
                    if boundary_k
                    else None,
                    "boundary_n": len(boundary_rows),
                }
            )
    return [rows[k] for k in sorted(rows)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_file", type=Path)
    args = parser.parse_args()

    rows = collect_round_geometry(args.trace_file)
    if not rows:
        print("No round_summary entries with selection_geometry found.")
        print(
            "This script requires traces produced after enabling score snapshot / selection geometry logging."
        )
        return 1

    print("round,lambda_effective,rho_uk,rho_pvalue,D_U,D_K,sens_up,sens_down,asymmetry_ratio,crossing_density,boundary_n")
    for row in rows:
        print(
            ",".join(
                [
                    str(row["round"]),
                    str(row["lambda_effective"]),
                    str(row["rho_uk"]),
                    str(row["rho_pvalue"]),
                    str(row["D_U"]),
                    str(row["D_K"]),
                    str(row["sens_up"]),
                    str(row["sens_down"]),
                    str(row["asymmetry_ratio"]),
                    str(row["crossing_density"]),
                    str(row["boundary_n"]),
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
