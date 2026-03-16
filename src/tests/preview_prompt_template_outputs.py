import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

# Ensure we can import src/* as top-level modules (agent.*, core.*, etc.)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from agent.prompt_template import PromptBuilder  # noqa: E402


def _find_section(prompt: str, start_key: str, end_key: str) -> str:
    lines = prompt.splitlines()
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if start_idx is None and start_key in line:
            start_idx = i
        if start_idx is not None and end_key in line:
            end_idx = i
            break
    if start_idx is None:
        return ""
    if end_idx is None:
        end_idx = len(lines)
    return "\n".join(lines[start_idx:end_idx])


def _grep_lines(prompt: str, keys: List[str]) -> List[str]:
    out = []
    for line in prompt.splitlines():
        if any(k in line for k in keys):
            out.append(line)
    return out


def _case(
    name: str,
    *,
    rollback_mode: Optional[str],
    rollback_threshold: Optional[float],
    total_iterations: int = 1000,
    current_iteration: int = 100,
    last_miou: float = 0.55,
    lambda_t: float = 0.6,
) -> Tuple[str, str]:
    builder = PromptBuilder()
    control_permissions: Dict[str, bool] = {
        "get_system_status": True,
        "get_top_k_samples": True,
        "get_sample_details": True,
        "get_score_distribution": True,
        "finalize_selection": True,
        "set_lambda": True,
        "set_query_size": True,
        "set_epochs_per_round": False,
        "set_alpha": True,
    }
    prompt = builder.build_system_prompt(
        total_iterations=total_iterations,
        current_iteration=current_iteration,
        last_miou=last_miou,
        lambda_t=lambda_t,
        rollback_threshold=rollback_threshold,
        rollback_mode=rollback_mode,
        control_permissions=control_permissions,
    )
    return name, prompt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full",
        action="store_true",
        help="Print full prompt for every case (otherwise only key sections).",
    )
    ap.add_argument(
        "--only",
        type=str,
        default=None,
        help="Only print a specific case name (exact match).",
    )
    args = ap.parse_args()

    cases = [
        _case(
            "A_adaptive_threshold__K_coreset_to_labeled",
            rollback_mode="adaptive_threshold",
            rollback_threshold=-0.03,
        )
    ]

    for name, prompt in cases:
        if args.only and name != args.only:
            continue

        print("\n" + "=" * 100)
        print(f"CASE: {name}")
        print("-" * 100)

        if args.full:
            print(prompt)
            continue

        # 1) Show the AD-KUCS scoring section and threshold lines
        k_section = _find_section(prompt, "AD-KUCS 评分逻辑", "量化阈值")
        t_section = _find_section(prompt, "量化阈值", "控制与评价机制")

        if k_section:
            print("[AD-KUCS Section]")
            print(k_section.strip())
            print()

        if t_section:
            print("[Threshold Section]")
            print(t_section.strip())
            print()

        # 2) Grep key lines to quickly verify substitution
        keys = ["回撤阈值", "coreset-to-labeled"]
        hits = _grep_lines(prompt, keys)
        if hits:
            print("[Key Lines]")
            for line in hits:
                print(line.rstrip())
        else:
            print("[Key Lines] (no matches)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
