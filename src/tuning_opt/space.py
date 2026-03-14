from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class Param:
    key: str
    lo: float
    hi: float
    kind: str = "float"

    def clip(self, value: float) -> float:
        return max(self.lo, min(self.hi, float(value)))


def _get_nested(d: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _set_nested(d: Dict[str, Any], path: List[str], value: Any) -> None:
    cur: Any = d
    for p in path[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[path[-1]] = value


def _path_from_key(key: str) -> List[str]:
    return [p for p in key.split(".") if p]


class ParameterSpace:
    def __init__(self, params: List[Param]):
        self.params = list(params)
        self._index = {p.key: i for i, p in enumerate(self.params)}

    @staticmethod
    def default() -> "ParameterSpace":
        return ParameterSpace(
            [
                Param("agent_threshold_overrides.LAMBDA_DELTA_UP", 0.02, 0.30),
                Param("agent_threshold_overrides.LAMBDA_DELTA_DOWN", 0.02, 0.25),
                Param("agent_threshold_overrides.LAMBDA_CLAMP_MIN", 0.0, 0.30),
                Param("agent_threshold_overrides.LAMBDA_CLAMP_MAX", 0.50, 1.00),
                Param("agent_threshold_overrides.OVERFIT_RISK_HI", 0.5, 2.0),
                Param("agent_threshold_overrides.OVERFIT_RISK_LO", 0.1, 1.0),
                Param("agent_threshold_overrides.OVERFIT_TVC_MIN_HI", 0.4, 0.95),
                Param("agent_threshold_overrides.OVERFIT_RISK_EMA_ALPHA", 0.2, 0.95),
                Param("agent_threshold_overrides.LAMBDA_DOWN_COOLING_ROUNDS", 0.0, 5.0),
                Param("lambda_policy.lambda_smoothing_alpha", 0.5, 1.0),
                Param("lambda_policy.lambda_max_step", 0.05, 0.35),
                Param("lambda_policy.risk_ci_window", 3.0, 10.0),
                Param("lambda_policy.risk_ci_quantile", 0.05, 0.50),
                Param("lambda_policy.late_stage_ramp.start_round", 6.0, 12.0),
                Param("lambda_policy.late_stage_ramp.end_round", 8.0, 16.0),
                Param("lambda_policy.late_stage_ramp.start_lambda", 0.05, 0.70),
                Param("lambda_policy.late_stage_ramp.end_lambda", 0.30, 0.95),
                Param("lambda_policy.selection_guardrail.u_median_min", 0.20, 0.60),
                Param("lambda_policy.selection_guardrail.u_low_thresh", 0.10, 0.60),
                Param("lambda_policy.selection_guardrail.u_low_frac_max", 0.05, 0.60),
                Param("lambda_policy.selection_guardrail.lambda_step_down", 0.02, 0.25),
                Param("epochs_per_round_override", 5.0, 20.0),
            ]
        )

    def flatten_from_ablation_cfg(self, cfg: Mapping[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for p in self.params:
            path = _path_from_key(p.key)
            v = _get_nested(dict(cfg), path)
            if v is None:
                continue
            try:
                out[p.key] = float(v)
            except Exception:
                continue
        return out

    def apply_overrides(
        self, base_cfg: Mapping[str, Any], overrides: Mapping[str, Any]
    ) -> Dict[str, Any]:
        merged = copy.deepcopy(dict(base_cfg))
        for k, v in overrides.items():
            if k not in self._index:
                continue
            spec = self.params[self._index[k]]
            try:
                fv = float(v)
            except Exception:
                continue
            fv = spec.clip(fv)
            if spec.kind == "int":
                fv = int(round(fv))
            path = _path_from_key(k)
            if path and path[0] == "epochs_per_round_override":
                merged["epochs_per_round_override"] = int(round(fv))
            else:
                _set_nested(
                    merged, path, int(round(fv)) if k.endswith("_round") else fv
                )
        return merged

    def trust_region_sample(
        self,
        *,
        center: Mapping[str, float],
        radius: float,
        rng,
        n: int,
        n_active: int = 4,
    ) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for _ in range(int(n)):
            cand: Dict[str, float] = {}
            active = set(
                p.key
                for p in rng.sample(self.params, k=min(int(n_active), len(self.params)))
            )
            for p in self.params:
                c = float(center.get(p.key, (p.lo + p.hi) / 2))
                if p.key in active:
                    width = (p.hi - p.lo) * float(radius)
                    v = c + rng.uniform(-width, width)
                    v = p.clip(v)
                else:
                    v = c
                if p.key.endswith(".start_round") or p.key.endswith(".end_round"):
                    v = int(round(v))
                if p.key == "epochs_per_round_override":
                    v = int(round(v))
                cand[p.key] = float(v)
            out.append(cand)
        return out

    def deduplicate(
        self, candidates: Iterable[Mapping[str, Any]]
    ) -> List[Dict[str, float]]:
        seen = set()
        unique: List[Dict[str, float]] = []
        for c in candidates:
            frozen = tuple(sorted((k, float(v)) for k, v in c.items()))
            if frozen in seen:
                continue
            seen.add(frozen)
            unique.append({k: float(v) for k, v in c.items()})
        return unique
