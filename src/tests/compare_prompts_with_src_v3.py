import argparse
import difflib
import json
import os
import subprocess
import sys
import zipfile
from typing import Any, Dict, List, Tuple


def _compute_lambda_schedule(
    rounds: List[int],
    *,
    r1_lambda: float = 0.0,
    uncertainty_only_rounds: int = 2,
    warmup_start_round: int = 3,
    warmup_rounds: int = 1,
    warmup_lambda: float = 0.2,
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for r in rounds:
        if r <= max(1, int(uncertainty_only_rounds)):
            out[int(r)] = float(r1_lambda)
        elif int(warmup_rounds) > 0 and int(warmup_start_round) <= r < (int(warmup_start_round) + int(warmup_rounds)):
            out[int(r)] = float(warmup_lambda)
        else:
            out[int(r)] = float(warmup_lambda)
    return out


def _emit_prompts(
    *,
    target: str,
    zip_path: str,
    exp_name: str,
    rounds: List[int],
    total_rounds: int,
    last_miou: float,
    rollback_threshold: float,
    rollback_mode: str,
    k_definition: str,
    control_permissions: Dict[str, bool],
    lambda_schedule: Dict[int, float],
    query_size: int,
    labeled_base: int,
    unlabeled_base: int,
) -> Dict[str, str]:
    if target == "current":
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
    elif target == "zip_v3":
        sys.path[:] = [f"{zip_path}/src"] + [p for p in sys.path if p]
    else:
        raise ValueError(f"unknown target: {target}")

    from agent.prompt_template import PromptBuilder  # noqa: E402
    from experiments.ablation_config import ABLATION_SETTINGS, EXPERIMENT_NAME_ALIASES  # noqa: E402

    requested = str(exp_name)
    canonical = str(EXPERIMENT_NAME_ALIASES.get(requested, requested))
    exp_cfg = ABLATION_SETTINGS.get(canonical) if isinstance(ABLATION_SETTINGS, dict) else None
    if isinstance(exp_cfg, dict):
        cp = exp_cfg.get("control_permissions")
        if isinstance(cp, dict):
            control_permissions = dict(cp)
        rc = exp_cfg.get("rollback_config")
        if isinstance(rc, dict) and rc.get("mode") is not None:
            rollback_mode = str(rc.get("mode"))
        if exp_cfg.get("k_definition") is not None:
            k_definition = str(exp_cfg.get("k_definition"))
        pol = exp_cfg.get("lambda_policy")
        if isinstance(pol, dict):
            lambda_schedule = _compute_lambda_schedule(
                rounds,
                r1_lambda=float(pol.get("r1_lambda", 0.0)),
                uncertainty_only_rounds=int(pol.get("uncertainty_only_rounds", 2)),
                warmup_start_round=int(pol.get("warmup_start_round", 3)),
                warmup_rounds=int(pol.get("warmup_rounds", 1)),
                warmup_lambda=float(pol.get("warmup_lambda", 0.2)),
            )

    builder = PromptBuilder()
    out: Dict[str, str] = {}
    for r in rounds:
        system_prompt = builder.build_system_prompt(
            total_iterations=int(total_rounds),
            current_iteration=int(r),
            last_miou=float(last_miou),
            lambda_t=float(lambda_schedule[int(r)]),
            rollback_threshold=float(rollback_threshold),
            rollback_mode=str(rollback_mode),
            k_definition=str(k_definition),
            control_permissions=dict(control_permissions),
        )
        labeled_size = int(labeled_base) + (int(r) - 1) * int(query_size)
        unlabeled_size = int(unlabeled_base) - (int(r) - 1) * int(query_size)
        user_prompt = builder.build_user_prompt(
            labeled_size=labeled_size,
            unlabeled_size=max(0, unlabeled_size),
            query_size=int(query_size),
        )
        out[str(r)] = "### SYSTEM PROMPT\n" + str(system_prompt) + "\n\n### USER PROMPT\n" + str(user_prompt)
    return out


def _run_emit_subprocess(script_path: str, target: str, zip_path: str, payload: Dict) -> Dict[str, str]:
    cmd = [
        sys.executable,
        script_path,
        "--emit",
        "--target",
        target,
        "--zip",
        zip_path,
        "--payload",
        json.dumps(payload, ensure_ascii=False),
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out)
    if not isinstance(data, dict):
        raise RuntimeError("emit output is not a dict")
    return data


def _unified_diff(a: str, b: str, *, from_name: str, to_name: str) -> str:
    a_lines = a.splitlines(keepends=True)
    b_lines = b.splitlines(keepends=True)
    diff = difflib.unified_diff(a_lines, b_lines, fromfile=from_name, tofile=to_name, n=3)
    return "".join(diff)


def _stable_json(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2, sort_keys=True)


def _read_zip_text(zip_path: str, inner_path: str) -> str:
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read(inner_path)
    return raw.decode("utf-8", errors="ignore")


def _read_local_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _deep_diff(a: Any, b: Any, path: str = "") -> List[Dict[str, Any]]:
    diffs: List[Dict[str, Any]] = []
    if type(a) is not type(b):
        diffs.append({"path": path or "<root>", "type": "type_mismatch", "a": str(type(a)), "b": str(type(b))})
        return diffs
    if isinstance(a, dict):
        a_keys = set(a.keys())
        b_keys = set(b.keys())
        for k in sorted(a_keys - b_keys, key=lambda x: str(x)):
            diffs.append({"path": f"{path}.{k}" if path else str(k), "type": "only_in_a", "a": a[k]})
        for k in sorted(b_keys - a_keys, key=lambda x: str(x)):
            diffs.append({"path": f"{path}.{k}" if path else str(k), "type": "only_in_b", "b": b[k]})
        for k in sorted(a_keys & b_keys, key=lambda x: str(x)):
            diffs.extend(_deep_diff(a[k], b[k], f"{path}.{k}" if path else str(k)))
        return diffs
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            diffs.append({"path": path or "<root>", "type": "len_mismatch", "a_len": len(a), "b_len": len(b)})
        n = min(len(a), len(b))
        for i in range(n):
            diffs.extend(_deep_diff(a[i], b[i], f"{path}[{i}]"))
        return diffs
    if a != b:
        diffs.append({"path": path or "<root>", "type": "value_mismatch", "a": a, "b": b})
    return diffs


def _emit_experiment_config(*, target: str, zip_path: str, exp_name: str) -> Dict[str, Any]:
    if target == "current":
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
    elif target == "zip_v3":
        sys.path[:] = [f"{zip_path}/src"] + [p for p in sys.path if p]
    else:
        raise ValueError(f"unknown target: {target}")
    from experiments.ablation_config import ABLATION_SETTINGS  # noqa: E402

    cfg = None
    if isinstance(ABLATION_SETTINGS, dict):
        cfg = ABLATION_SETTINGS.get(str(exp_name))
    if not isinstance(cfg, dict):
        return {}
    return cfg


def _emit_agent_thresholds(*, target: str, zip_path: str) -> Dict[str, Any]:
    if target == "current":
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
    elif target == "zip_v3":
        sys.path[:] = [f"{zip_path}/src"] + [p for p in sys.path if p]
    else:
        raise ValueError(f"unknown target: {target}")
    from agent.config import AgentThresholds, AgentConstraints  # noqa: E402

    def _dump_class(cls):
        out = {}
        for k, v in cls.__dict__.items():
            if k.startswith("_"):
                continue
            if callable(v):
                continue
            out[k] = v
        return out

    return {"AgentThresholds": _dump_class(AgentThresholds), "AgentConstraints": _dump_class(AgentConstraints)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=str, default="/Users/anykong/AAL_SD/src_v3.zip")
    ap.add_argument("--current-exp", type=str, default="full_model_A_lambda_policy")
    ap.add_argument("--zip-exp", type=str, default="full_model_v2_fixed_warmup_relaxed_risk")
    ap.add_argument("--rounds", type=str, default="1,2,3,4")
    ap.add_argument("--total-rounds", type=int, default=15)
    ap.add_argument("--last-miou", type=float, default=0.55)
    ap.add_argument("--rollback-threshold", type=float, default=-0.03)
    ap.add_argument("--rollback-mode", type=str, default="adaptive_threshold")
    ap.add_argument("--k-definition", type=str, default="coreset_to_labeled")
    ap.add_argument("--query-size", type=int, default=88)
    ap.add_argument("--labeled-base", type=int, default=0)
    ap.add_argument("--unlabeled-base", type=int, default=1000)
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--deep-mode", type=str, choices=("all", "config", "files", "prompts"), default="all")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--target", type=str, choices=("current", "zip_v3"), default="current")
    ap.add_argument("--payload", type=str, default=None)
    args = ap.parse_args()

    if str(args.current_exp).strip().lower() == "fullmodel":
        args.current_exp = "full_model_A_lambda_policy"

    rounds = [int(x.strip()) for x in str(args.rounds).split(",") if x.strip()]
    rounds = sorted({int(r) for r in rounds})
    if not rounds:
        raise SystemExit("no rounds provided")

    control_permissions: Dict[str, bool] = {
        "get_system_status": True,
        "get_top_k_samples": True,
        "get_sample_details": True,
        "get_score_distribution": True,
        "finalize_selection": True,
        "set_lambda": False,
        "set_query_size": False,
        "set_epochs_per_round": False,
        "set_alpha": False,
    }

    lambda_schedule = _compute_lambda_schedule(rounds)

    payload = {
        "current_exp": str(args.current_exp),
        "zip_exp": str(args.zip_exp),
        "rounds": rounds,
        "total_rounds": int(args.total_rounds),
        "last_miou": float(args.last_miou),
        "rollback_threshold": float(args.rollback_threshold),
        "rollback_mode": str(args.rollback_mode),
        "k_definition": str(args.k_definition),
        "query_size": int(args.query_size),
        "labeled_base": int(args.labeled_base),
        "unlabeled_base": int(args.unlabeled_base),
        "control_permissions": control_permissions,
        "lambda_schedule": lambda_schedule,
        "deep": bool(args.deep),
        "deep_mode": str(args.deep_mode),
    }
    if args.payload:
        payload = json.loads(args.payload)

    if args.emit:
        if payload.get("deep"):
            exp_name = str(payload["current_exp"] if args.target == "current" else payload["zip_exp"])
            out: Dict[str, Any] = {}
            out["experiment_config"] = _emit_experiment_config(target=str(args.target), zip_path=str(args.zip), exp_name=exp_name)
            out["agent_thresholds"] = _emit_agent_thresholds(target=str(args.target), zip_path=str(args.zip))
            deep_mode = str(payload.get("deep_mode") or "all")
            if deep_mode in ("all", "prompts"):
                out["prompts"] = _emit_prompts(
                    target=str(args.target),
                    zip_path=str(args.zip),
                    exp_name=exp_name,
                    rounds=[int(x) for x in payload["rounds"]],
                    total_rounds=int(payload["total_rounds"]),
                    last_miou=float(payload["last_miou"]),
                    rollback_threshold=float(payload["rollback_threshold"]),
                    rollback_mode=str(payload["rollback_mode"]),
                    k_definition=str(payload["k_definition"]),
                    control_permissions=dict(payload["control_permissions"]),
                    lambda_schedule={int(k): float(v) for k, v in payload["lambda_schedule"].items()},
                    query_size=int(payload["query_size"]),
                    labeled_base=int(payload["labeled_base"]),
                    unlabeled_base=int(payload["unlabeled_base"]),
                )
            sys.stdout.write(json.dumps(out, ensure_ascii=False))
        else:
            exp_name = str(payload["current_exp"] if args.target == "current" else payload["zip_exp"])
            prompts = _emit_prompts(
                target=str(args.target),
                zip_path=str(args.zip),
                exp_name=exp_name,
                rounds=[int(x) for x in payload["rounds"]],
                total_rounds=int(payload["total_rounds"]),
                last_miou=float(payload["last_miou"]),
                rollback_threshold=float(payload["rollback_threshold"]),
                rollback_mode=str(payload["rollback_mode"]),
                k_definition=str(payload["k_definition"]),
                control_permissions=dict(payload["control_permissions"]),
                lambda_schedule={int(k): float(v) for k, v in payload["lambda_schedule"].items()},
                query_size=int(payload["query_size"]),
                labeled_base=int(payload["labeled_base"]),
                unlabeled_base=int(payload["unlabeled_base"]),
            )
            sys.stdout.write(json.dumps(prompts, ensure_ascii=False))
        return 0

    script_path = os.path.abspath(__file__)
    zip_path = os.path.abspath(str(args.zip))
    current_payload = dict(payload)
    v3_payload = dict(payload)
    current_payload["current_exp"] = payload.get("current_exp")
    current_payload["zip_exp"] = payload.get("zip_exp")
    v3_payload["current_exp"] = payload.get("current_exp")
    v3_payload["zip_exp"] = payload.get("zip_exp")
    current_out = _run_emit_subprocess(script_path, "current", zip_path, current_payload)
    v3_out = _run_emit_subprocess(script_path, "zip_v3", zip_path, v3_payload)

    deep_mode = str(args.deep_mode or "all")
    if args.deep:
        current_cfg = current_out.get("experiment_config") if isinstance(current_out, dict) else None
        v3_cfg = v3_out.get("experiment_config") if isinstance(v3_out, dict) else None
        current_thr = current_out.get("agent_thresholds") if isinstance(current_out, dict) else None
        v3_thr = v3_out.get("agent_thresholds") if isinstance(v3_out, dict) else None
        current_prompts = current_out.get("prompts", {}) if isinstance(current_out, dict) else {}
        v3_prompts = v3_out.get("prompts", {}) if isinstance(v3_out, dict) else {}

        if deep_mode in ("all", "config"):
            print("\n" + "#" * 100)
            print("EXPERIMENT CONFIG (CURRENT)")
            print("#" * 100)
            print(_stable_json(current_cfg or {}))
            print("\n" + "#" * 100)
            print("EXPERIMENT CONFIG (SRC_V3_ZIP)")
            print("#" * 100)
            print(_stable_json(v3_cfg or {}))
            print("\n" + "#" * 100)
            print("EXPERIMENT CONFIG DIFF (JSON)")
            print("#" * 100)
            print(
                _unified_diff(
                    _stable_json(current_cfg or {}),
                    _stable_json(v3_cfg or {}),
                    from_name=f"current:{payload.get('current_exp')}",
                    to_name=f"src_v3.zip:{payload.get('zip_exp')}",
                ).rstrip()
                or "(no differences)"
            )
            print("\n" + "#" * 100)
            print("EXPERIMENT CONFIG DIFF (STRUCTURED)")
            print("#" * 100)
            structured = _deep_diff(current_cfg or {}, v3_cfg or {})
            print(_stable_json(structured))

            print("\n" + "#" * 100)
            print("AGENT THRESHOLDS/CONSTRAINTS DIFF (JSON)")
            print("#" * 100)
            print(
                _unified_diff(
                    _stable_json(current_thr or {}),
                    _stable_json(v3_thr or {}),
                    from_name="current:agent.config",
                    to_name="src_v3.zip:agent.config",
                ).rstrip()
                or "(no differences)"
            )

        if deep_mode in ("all", "files"):
            compare_files = [
                "src/agent/prompt_template.py",
                "src/agent/toolbox.py",
                "src/agent/agent_manager.py",
                "src/agent/config.py",
                "src/experiments/ablation_config.py",
            ]
            for inner in compare_files:
                local_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", inner))
                local_text = _read_local_text(local_path)
                zip_text = _read_zip_text(zip_path, inner)
                diff = _unified_diff(local_text, zip_text, from_name=f"current:{inner}", to_name=f"src_v3.zip:{inner}")
                print("\n" + "#" * 100)
                print(f"FILE DIFF: {inner}")
                print("#" * 100)
                if diff.strip():
                    print(diff.rstrip())
                else:
                    print("(no differences)")
    else:
        current_prompts = current_out
        v3_prompts = v3_out

    if args.deep and deep_mode in ("config", "files"):
        return 0

    for r in [str(x) for x in rounds]:
        a = current_prompts.get(r, "") if isinstance(current_prompts, dict) else ""
        b = v3_prompts.get(r, "") if isinstance(v3_prompts, dict) else ""
        diff = _unified_diff(
            a,
            b,
            from_name=f"current:R{r}",
            to_name=f"src_v3.zip:R{r}",
        )
        print("\n" + "=" * 100)
        print(f"ROUND R{r}")
        print("-" * 100)
        print("\n" + "=" * 40)
        print(f"CURRENT ({payload.get('current_exp')}):R{r}")
        print("=" * 40)
        print(a.rstrip())
        print("\n" + "=" * 40)
        print(f"SRC_V3_ZIP ({payload.get('zip_exp')}):R{r}")
        print("=" * 40)
        print(b.rstrip())
        print("\n" + "=" * 40)
        print("DIFF")
        print("=" * 40)
        if diff.strip():
            print(diff.rstrip())
        else:
            print("(no differences)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
