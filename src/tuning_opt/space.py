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
                Param("stage_aware_scales.early_up_scale", 0.5, 2.0),
                Param("stage_aware_scales.late_down_scale", 0.5, 3.0),
                Param("stage_aware_scales.late_up_scale", 0.2, 1.0),
            ]
        )

    @staticmethod
    def from_config(cfg: Mapping[str, Any]) -> "ParameterSpace":
        if not cfg or "parameter_space" not in cfg:
            return ParameterSpace.default()
        
        params = []
        for p_def in cfg["parameter_space"]:
            key = p_def.get("key")
            lo = p_def.get("lo")
            hi = p_def.get("hi")
            kind = p_def.get("kind", "float")
            if key and lo is not None and hi is not None:
                params.append(Param(key, float(lo), float(hi), kind))
                
        if not params:
            return ParameterSpace.default()
            
        return ParameterSpace(params)

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
        ignored_keys = []
        for k, v in overrides.items():
            if k not in self._index:
                ignored_keys.append(k)
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
            _set_nested(merged, path, int(round(fv)) if k.endswith("_round") else fv)

        agent_overrides = merged.get("agent_threshold_overrides")
        if isinstance(agent_overrides, dict):
            lo = agent_overrides.get("OVERFIT_RISK_LO")
            hi = agent_overrides.get("OVERFIT_RISK_HI")
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                if float(lo) >= float(hi):
                    agent_overrides["OVERFIT_RISK_LO"] = max(0.0, float(hi) - 0.05)

            clamp_min = agent_overrides.get("LAMBDA_CLAMP_MIN")
            clamp_max = agent_overrides.get("LAMBDA_CLAMP_MAX")
            if isinstance(clamp_min, (int, float)) and isinstance(clamp_max, (int, float)):
                if float(clamp_min) >= float(clamp_max):
                    agent_overrides["LAMBDA_CLAMP_MIN"] = max(0.0, float(clamp_max) - 0.05)

        lp = merged.get("lambda_policy")
        if not isinstance(lp, dict):
            lp = {}
            merged["lambda_policy"] = lp
        if isinstance(lp, dict):
            ramp = lp.get("late_stage_ramp")
            if isinstance(ramp, dict):
                sr = ramp.get("start_round")
                er = ramp.get("end_round")
                if isinstance(sr, int) and isinstance(er, int):
                    if int(er) <= int(sr):
                        ramp["end_round"] = int(sr) + 1

                sl = ramp.get("start_lambda")
                el = ramp.get("end_lambda")
                if isinstance(sl, (int, float)) and isinstance(el, (int, float)):
                    if float(el) < float(sl):
                        ramp["end_lambda"] = float(sl)

            guard = lp.get("selection_guardrail")
            if isinstance(guard, dict):
                u_min = guard.get("u_median_min")
                u_low = guard.get("u_low_thresh")
                if isinstance(u_min, (int, float)) and isinstance(u_low, (int, float)):
                    if float(u_low) > float(u_min):
                        guard["u_low_thresh"] = float(u_min)

        scales = merged.pop("stage_aware_scales", None)
        if isinstance(scales, dict) and isinstance(agent_overrides, dict):
            base_up = float(agent_overrides.get("LAMBDA_DELTA_UP", 0.10))
            base_down = float(agent_overrides.get("LAMBDA_DELTA_DOWN", 0.10))

            lp["stage_aware"] = True
            if not isinstance(lp.get("stage_boundaries"), list):
                lp["stage_boundaries"] = [5, 10]
            lp["stage_deltas"] = {
                "early": {
                    "delta_up": base_up * float(scales.get("early_up_scale", 1.0)),
                    "delta_down": base_down,
                },
                "mid": {
                    "delta_up": base_up,
                    "delta_down": base_down,
                },
                "late": {
                    "delta_up": base_up * float(scales.get("late_up_scale", 1.0)),
                    "delta_down": base_down * float(scales.get("late_down_scale", 1.0)),
                }
            }

        if ignored_keys:
            merged["_tuning_ignored_keys"] = ignored_keys

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
