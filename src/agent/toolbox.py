import json
import hashlib
import numpy as np
from typing import Dict, List, Optional, Any, Union
from .config import AgentThresholds, AgentConstraints
from .exceptions import (
    StateError,
    DataError,
    InvalidParameterError,
    ConstraintViolationError,
)


class Toolbox:
    def __init__(self, agent_controller: Any, query_strategy: Any, model: Any) -> None:
        """
        Agent 工具集，封装了 Agent 可以调用的所有 Python 函数
        Args:
            agent_controller: ActiveLearningAgent 实例 (包含 dataset, indices 等)
            query_strategy: QueryStrategy 实例 (负责计算 U/K)
            model: 当前训练好的模型
        """
        self.controller = agent_controller
        self.strategy = query_strategy
        self.model = model
        self.current_scores: Dict[str, Dict[str, float]] = {}
        self.candidates_cache: List[Dict[str, Any]] = []
        self.training_state: Dict[str, Any] = {}
        cfg = getattr(agent_controller, "config", None)
        self.alpha: float = float(
            getattr(cfg, "ALPHA", 5.0) if cfg is not None else 5.0
        )
        self.control_state: Dict[str, Any] = {}
        self.control_meta: Dict[str, Any] = {}
        self.control_permissions: Dict[str, bool] = {
            "set_lambda": False,
            "set_query_size": False,
            "set_epochs_per_round": False,
            "set_alpha": False,
            "get_top_k_samples": True,
            "get_sample_details": True,
            "get_score_distribution": True,
            "get_system_status": True,
            "finalize_selection": True,
        }
        self.SCHEMA_VERSION: int = 1
        self.last_score_error: Optional[Dict[str, Any]] = None
        self._miou_low_gain_streak: int = 0
        self._last_lambda_applied: Optional[float] = None
        self._ema_overfit_risk: Optional[float] = None
        self._last_ema_update_round: int = -999
        self._last_overfit_risk_raw: Optional[float] = None
        self._last_lambda_down_round: int = -999
        self._geometry_down_streak: int = 0
        self._signal_history: Dict[str, List[float]] = {}
        # P2: U/K 中位数历史（用于自适应 lambda 调整）
        self._u_median_history: List[tuple] = []
        self._k_median_history: List[tuple] = []

    def _success_response(self, result: Any, meta: Optional[Dict] = None) -> str:
        return json.dumps({"status": "success", "result": result, "meta": meta or {}})

    def _current_round(self) -> int:
        try:
            return int(getattr(self.controller, "current_round", 0) or 0)
        except Exception:
            return 0

    def _safe_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            out = float(value)
        except Exception:
            return None
        if np.isnan(out) or np.isinf(out):
            return None
        return float(out)

    def _lambda_policy_config(self) -> Optional[Dict[str, Any]]:
        controller = getattr(self, "controller", None)
        exp_cfg = getattr(controller, "exp_config", None)
        if isinstance(exp_cfg, dict):
            pol = exp_cfg.get("lambda_policy")
            if isinstance(pol, dict):
                return dict(pol)
        return None

    def _current_selection_geometry(self) -> Optional[Dict[str, Any]]:
        state = self.training_state if isinstance(self.training_state, dict) else {}
        geometry = state.get("selection_geometry")
        if isinstance(geometry, dict):
            return dict(geometry)
        controller = getattr(self, "controller", None)
        ranking_meta = getattr(controller, "_last_ranking_metadata", None)
        if isinstance(ranking_meta, dict):
            geometry = ranking_meta.get("selection_geometry")
            if isinstance(geometry, dict):
                return dict(geometry)
        return None

    def _resolve_geometry_lambda_cap(
        self, cfg: Dict[str, Any], default_cap: float
    ) -> Dict[str, Any]:
        cap = self._safe_float(cfg.get("lambda_cap"))
        cap_source = "lambda_cap"
        progress = None
        state = self.training_state if isinstance(self.training_state, dict) else {}
        current_labeled = self._safe_float(state.get("current_labeled_count"))
        total_budget = self._safe_float(state.get("total_budget"))
        if (
            current_labeled is not None
            and total_budget is not None
            and float(total_budget) > 0.0
        ):
            progress = float(current_labeled) / float(total_budget)
        schedule = cfg.get("progressive_caps")
        if isinstance(schedule, list) and progress is not None:
            points = []
            for item in schedule:
                if not isinstance(item, dict):
                    continue
                progress_cap = self._safe_float(
                    item.get("max_progress", item.get("progress_lte"))
                )
                lambda_cap = self._safe_float(item.get("lambda_max"))
                if progress_cap is None or lambda_cap is None:
                    continue
                points.append((float(progress_cap), float(lambda_cap)))
            points = sorted(points, key=lambda x: x[0])
            if points:
                interpolate = bool(cfg.get("interpolate_progressive_caps", True))
                chosen_cap = float(points[-1][1])
                chosen_progress_cap = float(points[-1][0])
                interpolated = False
                if float(progress) <= float(points[0][0]):
                    chosen_cap = float(points[0][1])
                    chosen_progress_cap = float(points[0][0])
                else:
                    for idx in range(1, len(points)):
                        prev_progress, prev_cap = points[idx - 1]
                        curr_progress, curr_cap = points[idx]
                        if float(progress) <= float(curr_progress):
                            chosen_progress_cap = float(curr_progress)
                            if interpolate and float(curr_progress) > float(prev_progress):
                                alpha = (float(progress) - float(prev_progress)) / (
                                    float(curr_progress) - float(prev_progress)
                                )
                                alpha = float(min(max(alpha, 0.0), 1.0))
                                chosen_cap = float(
                                    prev_cap + alpha * (curr_cap - prev_cap)
                                )
                                interpolated = True
                            else:
                                chosen_cap = float(curr_cap)
                            break
                cap = float(chosen_cap)
                cap_source = "progressive_caps_interp" if interpolated else "progressive_caps"
                return {
                    "cap": float(min(default_cap, cap)),
                    "source": cap_source,
                    "progress": progress,
                    "progress_cap": chosen_progress_cap,
                    "interpolated": bool(interpolated),
                }
        if cap is None:
            cap = float(default_cap)
            cap_source = "clamp_max"
        return {
            "cap": float(min(default_cap, cap)),
            "source": str(cap_source),
            "progress": progress,
            "progress_cap": None,
            "interpolated": False,
        }

    def _geometry_control_decision(
        self,
        *,
        round_num: int,
        policy: Dict[str, Any],
        base: float,
        clamp_min: float,
        clamp_max: float,
        delta_up: float,
        delta_down: float,
    ) -> Dict[str, Any]:
        cfg = policy.get("geometry_controller")
        if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
            return {"enabled": False, "handled": False, "diagnostics": {"enabled": False}}
        start_round = int(cfg.get("start_round", policy.get("risk_control_start_round", 1)))
        diagnostics: Dict[str, Any] = {
            "enabled": True,
            "start_round": int(start_round),
        }
        if int(round_num) < int(start_round):
            diagnostics["active"] = False
            diagnostics["reason"] = "before_start_round"
            return {"enabled": True, "handled": False, "diagnostics": diagnostics}
        geometry = self._current_selection_geometry()
        diagnostics["active"] = True
        diagnostics["geometry_available"] = isinstance(geometry, dict)
        if not isinstance(geometry, dict):
            diagnostics["reason"] = "missing_selection_geometry"
            return {"enabled": True, "handled": False, "diagnostics": diagnostics}
        sens_up = self._safe_float(geometry.get("sens_up"))
        sens_down = self._safe_float(geometry.get("sens_down"))
        asymmetry_ratio = self._safe_float(geometry.get("asymmetry_ratio"))
        if asymmetry_ratio is None and sens_up is not None and sens_down is not None:
            if abs(float(sens_down)) > 1e-12:
                asymmetry_ratio = float(sens_up) / float(sens_down)
        sens_threshold = self._safe_float(
            cfg.get("sens_up_threshold", cfg.get("tau", 0.12))
        )
        if sens_threshold is None:
            sens_threshold = 0.12
        step_up_base = self._safe_float(cfg.get("step_up", delta_up))
        if step_up_base is None:
            step_up_base = float(delta_up)
        step_down_value = self._safe_float(cfg.get("step_down", step_up_base))
        if step_down_value is None:
            step_down_value = float(step_up_base)
        asymmetry_floor = self._safe_float(cfg.get("asymmetry_floor", 1.0))
        if asymmetry_floor is None:
            asymmetry_floor = 1.0
        asymmetry_boost_cap = self._safe_float(cfg.get("asymmetry_boost_cap", 2.0))
        if asymmetry_boost_cap is None:
            asymmetry_boost_cap = 2.0
        asymmetry_eps = 1e-12
        asymmetry_mode = "neutral"
        asymmetry_scale = 1.0
        asymmetry_for_step = float(asymmetry_floor)
        if asymmetry_ratio is not None:
            if float(asymmetry_ratio) >= float(asymmetry_floor):
                asymmetry_for_step = max(float(asymmetry_ratio), float(asymmetry_floor))
                asymmetry_scale = 1.0 / float(asymmetry_for_step)
                asymmetry_mode = "shrink_up"
            elif float(asymmetry_ratio) > asymmetry_eps:
                asymmetry_for_step = float(asymmetry_ratio)
                asymmetry_scale = min(
                    1.0 / float(asymmetry_ratio), float(asymmetry_boost_cap)
                )
                asymmetry_mode = "boost_up"
        step_up_value = float(step_up_base) * float(asymmetry_scale)
        cap_info = self._resolve_geometry_lambda_cap(cfg, clamp_max)
        effective_cap = float(cap_info.get("cap", clamp_max))
        max_consecutive_down = int(cfg.get("max_consecutive_down", 2) or 0)
        current_down_streak = int(getattr(self, "_geometry_down_streak", 0) or 0)
        diagnostics.update(
            {
                "selection_geometry": {
                    "sens_up": sens_up,
                    "sens_down": sens_down,
                    "asymmetry_ratio": asymmetry_ratio,
                    "crossing_density": self._safe_float(geometry.get("crossing_density")),
                    "lambda_effective": self._safe_float(geometry.get("lambda_effective")),
                },
                "sens_up_threshold": float(sens_threshold),
                "step_up_base": float(step_up_base),
                "step_up_effective": float(step_up_value),
                "step_down": float(step_down_value),
                "asymmetry_floor": float(asymmetry_floor),
                "asymmetry_boost_cap": float(asymmetry_boost_cap),
                "asymmetry_mode": str(asymmetry_mode),
                "asymmetry_scale": float(asymmetry_scale),
                "asymmetry_for_step": float(asymmetry_for_step),
                "lambda_cap": float(effective_cap),
                "lambda_cap_source": cap_info.get("source"),
                "progress_ratio": cap_info.get("progress"),
                "progress_cap": cap_info.get("progress_cap"),
                "cap_interpolated": bool(cap_info.get("interpolated", False)),
                "max_consecutive_down": int(max_consecutive_down),
                "current_down_streak": int(current_down_streak),
                "fallback_to_risk_policy": bool(cfg.get("fallback_to_risk_policy", True)),
            }
        )
        if sens_up is None:
            diagnostics["reason"] = "missing_sens_up"
            return {"enabled": True, "handled": False, "diagnostics": diagnostics}
        if float(sens_up) > float(sens_threshold):
            if int(max_consecutive_down) > 0 and int(current_down_streak) >= int(
                max_consecutive_down
            ):
                diagnostics["direction"] = "hold"
                diagnostics["reason"] = "down_streak_cap"
                return {
                    "enabled": True,
                    "handled": True,
                    "applied": float(min(max(float(base), float(clamp_min)), float(effective_cap))),
                    "rule": "geometry_sensitive_hold_streak_cap",
                    "diagnostics": diagnostics,
                }
            applied = float(
                min(
                    max(float(base) - float(step_down_value), float(clamp_min)),
                    float(effective_cap),
                )
            )
            diagnostics["direction"] = "down"
            return {
                "enabled": True,
                "handled": True,
                "applied": float(applied),
                "rule": "geometry_sensitive_down",
                "diagnostics": diagnostics,
            }
        applied = float(
            min(float(base) + float(step_up_value), float(clamp_max), float(effective_cap))
        )
        diagnostics["direction"] = "up"
        diagnostics["cap_hit"] = bool(float(applied) >= float(effective_cap) - 1e-12)
        if float(applied) < float(base) - 1e-12:
            rule = "geometry_cap_down"
        else:
            rule = "geometry_safe_up_capped" if diagnostics["cap_hit"] else "geometry_safe_up"
        return {
            "enabled": True,
            "handled": True,
            "applied": float(applied),
            "rule": str(rule),
            "diagnostics": diagnostics,
        }

    def _selection_guardrail_config(self) -> Optional[Dict[str, Any]]:
        policy = self._lambda_policy_config()
        if not isinstance(policy, dict):
            return None
        cfg = policy.get("selection_guardrail")
        if not isinstance(cfg, dict):
            return None
        if not bool(cfg.get("enabled", False)):
            return None
        return dict(cfg)

    def _guardrail_u_stats(
        self, sample_ids: List[str], u_low_thresh: float
    ) -> Dict[str, Any]:
        u_vals: List[float] = []
        missing: List[str] = []
        for sid in sample_ids:
            v = None
            if isinstance(self.current_scores, dict) and sid in self.current_scores:
                v = (self.current_scores.get(sid) or {}).get("U")
            if v is None:
                try:
                    alt = str(int(sid))
                except Exception:
                    alt = None
                if (
                    alt
                    and isinstance(self.current_scores, dict)
                    and alt in self.current_scores
                ):
                    v = (self.current_scores.get(alt) or {}).get("U")
            if v is None:
                missing.append(str(sid))
                continue
            try:
                u_vals.append(float(v))
            except Exception:
                missing.append(str(sid))
        n = int(len(u_vals))
        if n <= 0:
            return {"n": 0, "u_median": None, "frac_u_lt": None, "missing": missing}
        u_median = float(np.median(np.array(u_vals, dtype=np.float32)))
        frac_u_lt = float(
            sum(1 for x in u_vals if float(x) < float(u_low_thresh))
        ) / float(n)
        return {
            "n": n,
            "u_median": u_median,
            "frac_u_lt": frac_u_lt,
            "missing": missing,
        }

    def _score_stats_for_ids(
        self, sample_ids: List[str], u_low_thresh: float
    ) -> Dict[str, Any]:
        u_vals: List[float] = []
        k_vals: List[float] = []
        missing: List[str] = []
        for sid in sample_ids:
            raw = None
            if isinstance(self.current_scores, dict):
                raw = self.current_scores.get(str(sid))
                if raw is None:
                    try:
                        alt = str(int(sid))
                    except Exception:
                        alt = None
                    if alt is not None:
                        raw = self.current_scores.get(alt)
            if not isinstance(raw, dict):
                missing.append(str(sid))
                continue
            u_val = raw.get("U")
            k_val = raw.get("K")
            if u_val is None and k_val is None:
                missing.append(str(sid))
                continue
            try:
                if u_val is not None:
                    u_vals.append(float(u_val))
            except Exception:
                pass
            try:
                if k_val is not None:
                    k_vals.append(float(k_val))
            except Exception:
                pass
        u_arr = (
            np.asarray(u_vals, dtype=float) if u_vals else np.asarray([], dtype=float)
        )
        k_arr = (
            np.asarray(k_vals, dtype=float) if k_vals else np.asarray([], dtype=float)
        )
        u_n = int(u_arr.size)
        k_n = int(k_arr.size)
        out: Dict[str, Any] = {
            "selected_count": int(len(sample_ids or [])),
            "u_n": int(u_n),
            "k_n": int(k_n),
            "missing_count": int(len(missing)),
            "missing_sample_ids": list(missing),
            "u_mean": float(np.mean(u_arr)) if u_n > 0 else None,
            "u_median": float(np.median(u_arr)) if u_n > 0 else None,
            "u_p75": float(np.quantile(u_arr, 0.75)) if u_n > 0 else None,
            "k_mean": float(np.mean(k_arr)) if k_n > 0 else None,
            "k_median": float(np.median(k_arr)) if k_n > 0 else None,
            "k_p75": float(np.quantile(k_arr, 0.75)) if k_n > 0 else None,
        }
        if u_n > 0:
            out["frac_u_lt"] = float(np.mean(u_arr < float(u_low_thresh)))
        else:
            out["frac_u_lt"] = None
        return out

    def _dynamic_guardrail_thresholds(
        self,
        cfg: Dict[str, Any],
        *,
        base_u_median_min: float,
        base_u_low_frac_max: float,
    ) -> Dict[str, Any]:
        adaptive_cfg = cfg.get("adaptive_thresholds")
        enabled = bool(cfg.get("adaptive", False))
        if isinstance(adaptive_cfg, dict):
            enabled = bool(adaptive_cfg.get("enabled", enabled))
        if not enabled:
            return {
                "u_median_min": float(base_u_median_min),
                "u_low_frac_max": float(base_u_low_frac_max),
                "mode": "fixed",
            }
        window = int(
            (adaptive_cfg or {}).get("window", cfg.get("adaptive_window", 5)) or 5
        )
        min_samples = int(
            (adaptive_cfg or {}).get("min_samples", cfg.get("adaptive_min_samples", 3))
            or 3
        )
        q_u = float(
            (adaptive_cfg or {}).get(
                "u_median_quantile", cfg.get("adaptive_u_median_quantile", 0.3)
            )
            or 0.3
        )
        q_u = min(max(q_u, 0.0), 1.0)
        q_frac = float(
            (adaptive_cfg or {}).get(
                "u_low_frac_quantile", cfg.get("adaptive_u_low_frac_quantile", 0.8)
            )
            or 0.8
        )
        q_frac = min(max(q_frac, 0.0), 1.0)
        hist_u = [
            float(x)
            for x in self._get_signal_history(
                "guardrail_selected_u_median", window=window
            )
        ]
        hist_frac = [
            float(x)
            for x in self._get_signal_history(
                "guardrail_selected_frac_u_lt", window=window
            )
        ]
        if len(hist_u) < int(min_samples) or len(hist_frac) < int(min_samples):
            return {
                "u_median_min": float(base_u_median_min),
                "u_low_frac_max": float(base_u_low_frac_max),
                "mode": "fixed_insufficient_history",
                "window": int(window),
                "min_samples": int(min_samples),
                "history_u_n": int(len(hist_u)),
                "history_frac_n": int(len(hist_frac)),
            }
        dyn_u = float(np.quantile(np.asarray(hist_u, dtype=float), q_u))
        dyn_frac = float(np.quantile(np.asarray(hist_frac, dtype=float), q_frac))
        floor_u = (adaptive_cfg or {}).get(
            "u_median_floor", cfg.get("u_median_min_floor", base_u_median_min)
        )
        ceil_frac = (adaptive_cfg or {}).get(
            "u_low_frac_ceiling", cfg.get("u_low_frac_max_ceiling", base_u_low_frac_max)
        )
        if floor_u is not None:
            dyn_u = max(float(dyn_u), float(floor_u))
        if ceil_frac is not None:
            dyn_frac = min(float(dyn_frac), float(ceil_frac))
        return {
            "u_median_min": float(dyn_u),
            "u_low_frac_max": float(dyn_frac),
            "mode": "adaptive_quantile",
            "window": int(window),
            "min_samples": int(min_samples),
            "u_median_quantile": float(q_u),
            "u_low_frac_quantile": float(q_frac),
            "history_u_n": int(len(hist_u)),
            "history_frac_n": int(len(hist_frac)),
            "u_median_floor": None if floor_u is None else float(floor_u),
            "u_low_frac_ceiling": None if ceil_frac is None else float(ceil_frac),
        }

    def _rank_candidate_ids_by_lambda(self, lambda_value: float) -> List[str]:
        items: List[tuple[float, str]] = []
        for sid, scores in (self.current_scores or {}).items():
            if not isinstance(scores, dict):
                continue
            try:
                u = float(scores.get("U", 0.0))
                k = float(scores.get("K", 0.0))
            except Exception:
                continue
            score = (1.0 - float(lambda_value)) * u + float(lambda_value) * k
            items.append((score, str(sid)))
        items.sort(key=lambda x: (float(x[0]), x[1]), reverse=True)
        return [sid for _, sid in items]

    def _pick_top_k(
        self, ranked_ids: List[str], k: int, exclude: Optional[set[str]] = None
    ) -> List[str]:
        out: List[str] = []
        ex = exclude or set()
        for sid in ranked_ids:
            if sid in ex:
                continue
            out.append(str(sid))
            if len(out) >= int(k):
                break
        return out

    def _apply_selection_guardrail(self, sample_ids: List[str]) -> Dict[str, Any]:
        cfg = self._selection_guardrail_config()
        if cfg is None:
            return {"applied": False, "sample_ids": list(sample_ids)}
        if not isinstance(self.current_scores, dict) or not self.current_scores:
            return {"applied": False, "sample_ids": list(sample_ids)}

        # --- Warmup-phase exemption ---
        # Data analysis shows early guardrail triggers (warmup rounds) strongly
        # hurt final mIoU (r=-0.576). Skip guardrail until risk_control phase
        # is active, controlled by guardrail_start_round or risk_control_start_round.
        round_num = self._current_round()
        policy = self._lambda_policy_config() or {}
        guardrail_start = cfg.get("guardrail_start_round")
        if guardrail_start is None:
            guardrail_start = policy.get(
                "risk_control_start_round",
                int(policy.get("warmup_start_round", 2))
                + int(policy.get("warmup_rounds", 1)),
            )
        if int(round_num) < int(guardrail_start):
            return {
                "applied": False,
                "sample_ids": list(sample_ids),
                "skipped_reason": "warmup_phase_exemption",
                "round": int(round_num),
                "guardrail_start_round": int(guardrail_start),
            }

        controller_cfg = getattr(self.controller, "config", None)
        expected = int(getattr(controller_cfg, "QUERY_SIZE", 0) or 0)
        if expected <= 0:
            expected = int(len(sample_ids))
        expected = min(expected, int(len(self.current_scores)))
        if expected <= 0:
            return {"applied": False, "sample_ids": list(sample_ids)}

        base_u_median_min = float(cfg.get("u_median_min", 0.45))
        u_low_thresh = float(cfg.get("u_low_thresh", 0.4))
        base_u_low_frac_max = float(cfg.get("u_low_frac_max", 0.2))
        max_steps = int(cfg.get("max_steps", 5) or 0)
        lambda_step_down = float(cfg.get("lambda_step_down", 0.1))
        fallback_u_frac = float(cfg.get("fallback_quota_u_frac", 0.7))
        fallback_u_frac = float(min(max(fallback_u_frac, 0.0), 1.0))
        threshold_meta = self._dynamic_guardrail_thresholds(
            cfg,
            base_u_median_min=base_u_median_min,
            base_u_low_frac_max=base_u_low_frac_max,
        )
        u_median_min = float(threshold_meta.get("u_median_min", base_u_median_min))
        u_low_frac_max = float(
            threshold_meta.get("u_low_frac_max", base_u_low_frac_max)
        )

        clamp_min = float(
            self._agent_threshold("LAMBDA_CLAMP_MIN", AgentConstraints.LAMBDA_MIN)
        )
        clamp_max = float(
            self._agent_threshold("LAMBDA_CLAMP_MAX", AgentConstraints.LAMBDA_MAX)
        )

        lambda_before = self.control_state.get("lambda_override_round")
        if lambda_before is None:
            lambda_before = self._default_lambda()
        lambda_before = float(lambda_before)
        lambda_before = float(min(max(lambda_before, clamp_min), clamp_max))

        orig = [str(x) for x in (sample_ids or [])][:expected]
        stats0 = self._guardrail_u_stats(orig, u_low_thresh=u_low_thresh)
        ok0 = (
            stats0.get("n", 0) > 0
            and stats0.get("u_median") is not None
            and stats0.get("frac_u_lt") is not None
            and float(stats0["u_median"]) >= float(u_median_min)
            and float(stats0["frac_u_lt"]) <= float(u_low_frac_max)
        )
        if ok0:
            return {
                "applied": False,
                "sample_ids": orig,
                "lambda_before": lambda_before,
                "lambda_after": lambda_before,
                "stats": stats0,
            }

        lam = float(lambda_before)
        chosen = None
        method = "lambda_step_down"
        ranked_cache: Dict[float, List[str]] = {}
        for _ in range(max(0, int(max_steps))):
            lam = float(max(float(lam) - abs(float(lambda_step_down)), clamp_min))
            if lam in ranked_cache:
                ranked = ranked_cache[lam]
            else:
                ranked = self._rank_candidate_ids_by_lambda(lam)
                ranked_cache[lam] = ranked
            cand = self._pick_top_k(ranked, expected)
            st = self._guardrail_u_stats(cand, u_low_thresh=u_low_thresh)
            ok = (
                st.get("n", 0) > 0
                and st.get("u_median") is not None
                and st.get("frac_u_lt") is not None
                and float(st["u_median"]) >= float(u_median_min)
                and float(st["frac_u_lt"]) <= float(u_low_frac_max)
            )
            if ok:
                chosen = {"sample_ids": cand, "stats": st, "lambda_after": lam}
                break

        if chosen is None:
            method = "quota_u"
            u_k = int(round(float(expected) * float(fallback_u_frac)))
            u_k = max(0, min(int(expected), int(u_k)))
            ranked_u = self._rank_candidate_ids_by_lambda(0.0)
            ranked_h = self._rank_candidate_ids_by_lambda(lam)
            first = self._pick_top_k(ranked_u, u_k)
            ex = set(first)
            second = self._pick_top_k(
                ranked_h, int(expected) - int(len(first)), exclude=ex
            )
            cand = list(first) + list(second)
            st = self._guardrail_u_stats(cand, u_low_thresh=u_low_thresh)
            chosen = {"sample_ids": cand, "stats": st, "lambda_after": lam}

        lambda_after = float(chosen.get("lambda_after", lambda_before))
        lambda_after = float(min(max(lambda_after, clamp_min), clamp_max))
        final_ids = list(chosen.get("sample_ids") or orig)
        final_stats = dict(chosen.get("stats") or {})
        final_stats_selected_all = self._score_stats_for_ids(
            final_ids, u_low_thresh=u_low_thresh
        )
        self._append_signal_history(
            "guardrail_selected_u_median",
            final_stats_selected_all.get("u_median"),
            max_len=32,
        )
        self._append_signal_history(
            "guardrail_selected_frac_u_lt",
            final_stats_selected_all.get("frac_u_lt"),
            max_len=32,
        )

        self.control_state["lambda_override_round"] = float(lambda_after)
        self.control_meta["lambda_guardrail"] = {
            "method": method,
            "lambda_before": float(lambda_before),
            "lambda_after": float(lambda_after),
            "threshold_mode": threshold_meta.get("mode"),
            "thresholds": {
                "u_median_min": float(u_median_min),
                "u_low_thresh": float(u_low_thresh),
                "u_low_frac_max": float(u_low_frac_max),
            },
            "threshold_adaptive": threshold_meta,
            "stats_before": stats0,
            "stats_after": final_stats,
            "stats_after_selected_all": final_stats_selected_all,
        }

        if hasattr(self.controller, "_append_trace"):
            try:
                self.controller._append_trace(
                    {
                        "type": "selection_guardrail",
                        "round": int(self._current_round()),
                        "applied": True,
                        "method": method,
                        "lambda_before": float(lambda_before),
                        "lambda_after": float(lambda_after),
                        "thresholds": {
                            "u_median_min": float(u_median_min),
                            "u_low_thresh": float(u_low_thresh),
                            "u_low_frac_max": float(u_low_frac_max),
                        },
                        "threshold_mode": threshold_meta.get("mode"),
                        "threshold_adaptive": threshold_meta,
                        "stats_before": stats0,
                        "stats_after": final_stats,
                        "stats_after_selected_all": final_stats_selected_all,
                        "selected_ids_before": orig,
                        "selected_ids_after": final_ids,
                    }
                )
                self.controller._append_trace(
                    {
                        "type": "lambda_guard",
                        "round": int(self._current_round()),
                        "cap": float(clamp_max),
                        "lambda_before": float(lambda_before),
                        "lambda_after": float(lambda_after),
                        "method": method,
                        "thresholds": {
                            "u_median_min": float(u_median_min),
                            "u_low_thresh": float(u_low_thresh),
                            "u_low_frac_max": float(u_low_frac_max),
                        },
                        "threshold_mode": threshold_meta.get("mode"),
                        "threshold_adaptive": threshold_meta,
                        "stats_before": stats0,
                        "stats_after": final_stats,
                        "stats_after_selected_all": final_stats_selected_all,
                    }
                )
            except Exception:
                pass

        if hasattr(self.controller, "_selection_context") and isinstance(
            getattr(self.controller, "_selection_context", None), dict
        ):
            ctx = dict(getattr(self.controller, "_selection_context") or {})
            ctx["policy"] = "agent_finalize_selection_guardrail"
            ctx["guardrail"] = {
                "method": method,
                "lambda_before": float(lambda_before),
                "lambda_after": float(lambda_after),
                "u_median_before": stats0.get("u_median"),
                "u_median_after": final_stats.get("u_median"),
                "frac_u_lt_before": stats0.get("frac_u_lt"),
                "frac_u_lt_after": final_stats.get("frac_u_lt"),
                "threshold_mode": threshold_meta.get("mode"),
            }
            self.controller._selection_context = ctx

        return {
            "applied": True,
            "method": method,
            "lambda_before": float(lambda_before),
            "lambda_after": float(lambda_after),
            "stats_before": stats0,
            "stats_after": final_stats,
            "stats_after_selected_all": final_stats_selected_all,
            "sample_ids": final_ids,
        }

    def _require_explicit_lambda(self) -> bool:
        controller = getattr(self, "controller", None)
        exp_cfg = getattr(controller, "exp_config", None)
        if isinstance(exp_cfg, dict):
            return bool(exp_cfg.get("require_explicit_lambda"))
        return False

    def _risk_policy_config(self) -> Optional[Dict[str, Any]]:
        controller = getattr(self, "controller", None)
        exp_cfg = getattr(controller, "exp_config", None)
        if isinstance(exp_cfg, dict):
            pol = exp_cfg.get("risk_policy")
            if isinstance(pol, dict):
                return dict(pol)
        return None

    def _append_signal_history(self, key: str, value: Any, max_len: int = 20) -> None:
        if value is None:
            return
        try:
            val = float(value)
        except Exception:
            return
        if not np.isfinite(val):
            return
        history = self._signal_history.setdefault(str(key), [])
        history.append(val)
        if len(history) > int(max_len):
            del history[: len(history) - int(max_len)]

    def _get_signal_history(self, key: str, window: int | None = None) -> List[float]:
        history = list(self._signal_history.get(str(key), []))
        if window is None:
            return history
        w = max(int(window), 0)
        if w <= 0:
            return history
        return history[-w:]

    def _agent_threshold(self, name: str, fallback: Any) -> Any:
        controller = getattr(self, "controller", None)
        exp_cfg = getattr(controller, "exp_config", None)
        overrides = (
            exp_cfg.get("agent_threshold_overrides")
            if isinstance(exp_cfg, dict)
            else None
        )
        if isinstance(overrides, dict) and name in overrides:
            return overrides[name]
        return getattr(AgentThresholds, name, fallback)

    def _get_delta_adaptive(
        self, round_num: int, policy: Dict[str, Any], risk: float | None
    ):
        if not bool(policy.get("u_adaptive", False)):
            return None, None
        u_history = self.training_state.get("train_u_median_history", [])
        if not isinstance(u_history, list) or len(u_history) < 2:
            return None, None
        u_curr = u_history[-1][1]
        u_prev = u_history[-2][1]
        u_delta = float(u_curr) - float(u_prev)
        u_delta_down_threshold = float(policy.get("u_delta_down_threshold", -0.3))
        u_delta_up_threshold = float(policy.get("u_delta_up_threshold", 0.1))
        if u_delta < u_delta_down_threshold:
            delta = -0.2
            rule = f"u_quality_degradation_aggressive_down_u_delta={u_delta:.3f}"
            return float(delta), str(rule)
        if u_delta > u_delta_up_threshold:
            delta = +0.1
            rule = f"u_quality_improvement_aggressive_up_u_delta={u_delta:.3f}"
            return float(delta), str(rule)
        return None, None

    def _compute_policy_lambda_for_round(
        self, round_num: int, policy: Dict[str, Any]
    ) -> Dict[str, Any]:
        mode = str(policy.get("mode") or "")
        if mode != "warmup_risk_closed_loop":
            raise ValueError(f"Unsupported lambda_policy mode: {mode}")

        r1_lambda = float(policy.get("r1_lambda", 0.0))
        warmup_start_round = int(policy.get("warmup_start_round", 2))
        warmup_rounds = int(policy.get("warmup_rounds", 1))
        warmup_lambda = policy.get("warmup_lambda", 0.22)
        warmup_lambda_range = policy.get("warmup_lambda_range")
        warmup_floor = None
        if isinstance(warmup_lambda_range, (list, tuple)) and len(warmup_lambda_range) == 2:
            try:
                lo = float(warmup_lambda_range[0])
                hi = float(warmup_lambda_range[1])
                warmup_floor = float(min(lo, hi))
            except Exception:
                warmup_floor = None
        risk_control_start_round = int(
            policy.get("risk_control_start_round", warmup_start_round + warmup_rounds)
        )
        uncertainty_only_rounds = int(policy.get("uncertainty_only_rounds", 1))

        clamp_min = float(
            self._agent_threshold("LAMBDA_CLAMP_MIN", AgentConstraints.LAMBDA_MIN)
        )
        clamp_max = float(
            self._agent_threshold("LAMBDA_CLAMP_MAX", AgentConstraints.LAMBDA_MAX)
        )
        delta_down = float(self._agent_threshold("LAMBDA_DELTA_DOWN", 0.1) or 0.0)
        delta_up = float(self._agent_threshold("LAMBDA_DELTA_UP", 0.05) or 0.0)
        risk_hi = float(self._agent_threshold("OVERFIT_RISK_HI", 0.8) or 0.0)
        risk_lo = float(self._agent_threshold("OVERFIT_RISK_LO", 0.2) or 0.0)
        tvc_min_hi = float(self._agent_threshold("OVERFIT_TVC_MIN_HI", 0.5) or 0.0)
        streak_need = int(self._agent_threshold("MIOU_LOW_GAIN_STREAK", 0) or 0)

        stage_aware = bool(policy.get("stage_aware", False))
        stage_meta = None
        if stage_aware:
            boundaries = policy.get("stage_boundaries", [])
            deltas = policy.get("stage_deltas", {})
            stage = None
            b = (
                [int(x) for x in boundaries]
                if isinstance(boundaries, (list, tuple))
                else []
            )
            b = [int(x) for x in b if x is not None]
            b = sorted(b)
            if len(b) >= 1 and int(round_num) <= int(b[0]):
                stage = "early"
            elif len(b) >= 2 and int(round_num) <= int(b[1]):
                stage = "mid"
            else:
                stage = "late"
            if isinstance(deltas, dict):
                sd = deltas.get(stage)
                if isinstance(sd, dict):
                    if sd.get("delta_up") is not None:
                        delta_up = float(sd.get("delta_up"))
                    if sd.get("delta_down") is not None:
                        delta_down = float(sd.get("delta_down"))
            stage_meta = {
                "enabled": True,
                "stage": str(stage),
                "stage_boundaries": b,
                "stage_deltas": deltas if isinstance(deltas, dict) else None,
            }

        diagnostics = {
            "round": int(round_num),
            "policy_mode": str(mode),
            "uncertainty_only_rounds": int(uncertainty_only_rounds),
            "warmup_start_round": int(warmup_start_round),
            "warmup_rounds": int(warmup_rounds),
            "warmup_lambda": None if warmup_lambda is None else float(warmup_lambda),
            "warmup_lambda_range": warmup_lambda_range,
            "risk_control_start_round": int(risk_control_start_round),
            "r1_lambda": float(r1_lambda),
            "thresholds": {
                "lambda_clamp_min": float(clamp_min),
                "lambda_clamp_max": float(clamp_max),
                "lambda_delta_down": float(delta_down),
                "lambda_delta_up": float(delta_up),
                "overfit_risk_hi": float(risk_hi),
                "overfit_risk_lo": float(risk_lo),
                "overfit_tvc_min_hi": float(tvc_min_hi),
                "miou_low_gain_streak_need": int(streak_need),
            },
            "stage_aware": stage_meta if stage_meta is not None else {"enabled": False},
            "phase_aware": stage_meta if stage_meta is not None else {"enabled": False},
        }

        if round_num <= max(1, int(uncertainty_only_rounds)):
            applied = float(r1_lambda)
            diagnostics.update(
                {
                    "phase": "uncertainty_only",
                    "base": None,
                    "applied": float(applied),
                    "rule": "uncertainty_only_phase",
                }
            )
            return {
                "applied": applied,
                "bounds": {"min": applied, "max": applied},
                "rule": "uncertainty_only_phase",
                "base": None,
                "diagnostics": diagnostics,
            }

        if warmup_rounds > 0 and warmup_start_round <= round_num < (
            warmup_start_round + warmup_rounds
        ):
            applied = None
            rule = "warmup_fixed_lambda"
            if warmup_lambda_range is not None:
                if not (
                    isinstance(warmup_lambda_range, (list, tuple))
                    and len(warmup_lambda_range) == 2
                ):
                    raise ValueError(
                        "warmup_lambda_range must be a 2-item list/tuple when provided"
                    )
                lo = float(warmup_lambda_range[0])
                hi = float(warmup_lambda_range[1])
                if lo > hi:
                    lo, hi = hi, lo
                seed = getattr(self.controller, "seed", None)
                if seed is None:
                    seed = getattr(
                        getattr(self.controller, "config", None), "RANDOM_SEED", None
                    )
                seed = str(seed if seed is not None else "")
                # Proposal A: Remove run_id and exp_name to ensure consistent lambda sequence
                salt = f"fixed_warmup|{seed}|round={int(round_num)}"
                h = hashlib.sha256(salt.encode("utf-8", errors="ignore")).digest()
                u = int.from_bytes(h, "big") / float(2 ** (8 * len(h)) - 1)
                applied = float(lo + (hi - lo) * u)
                rule = "warmup_seeded_uniform"
            if applied is None:
                applied = float(warmup_lambda)
            applied = float(min(max(applied, clamp_min), clamp_max))
            diagnostics.update(
                {
                    "phase": "warmup",
                    "base": None,
                    "applied": float(applied),
                    "rule": str(rule),
                }
            )
            return {
                "applied": applied,
                "bounds": {"min": clamp_min, "max": clamp_max},
                "rule": rule,
                "base": None,
                "diagnostics": diagnostics,
            }

        post_warmup_start = int(warmup_start_round) + int(max(warmup_rounds, 0))
        if warmup_floor is not None and int(round_num) >= int(post_warmup_start):
            clamp_min = float(max(float(clamp_min), float(warmup_floor)))

        base = self._last_lambda_applied
        if base is None:
            base = (
                float(warmup_lambda) if warmup_lambda is not None else float(clamp_min)
            )
        base = float(base)

        applied = float(base)
        rule = "hold"

        rollback_flag = bool(self.training_state.get("rollback_flag", False))
        risk_raw = self.training_state.get("overfit_risk")
        risk = risk_raw

        ema_alpha = float(self._agent_threshold("OVERFIT_RISK_EMA_ALPHA", 1.0))
        diagnostics["thresholds"]["overfit_risk_ema_alpha"] = float(
            min(max(ema_alpha, 0.0), 1.0)
        )
        if risk is not None:
            risk = float(risk)
            if (
                self._last_ema_update_round != round_num
                or self._last_overfit_risk_raw != risk
            ):
                if self._ema_overfit_risk is None:
                    self._ema_overfit_risk = risk
                else:
                    self._ema_overfit_risk = (
                        ema_alpha * risk + (1.0 - ema_alpha) * self._ema_overfit_risk
                    )
                self._last_ema_update_round = round_num
                self._last_overfit_risk_raw = risk
            if self._ema_overfit_risk is not None:
                risk = self._ema_overfit_risk

        tvc_key = policy.get("severe_tvc_key")
        if not isinstance(tvc_key, str) or not tvc_key:
            tvc_key = "grad_train_val_cos_min"
        tvc_val = self.training_state.get(tvc_key)
        risk_policy = self._risk_policy_config() or {}
        risk_trigger = (
            str(policy.get("risk_trigger", risk_policy.get("trigger", "")) or "")
            .strip()
            .lower()
        )
        risk_ci_window = policy.get("risk_ci_window", risk_policy.get("window"))
        risk_ci_quantile = policy.get("risk_ci_quantile", risk_policy.get("quantile"))
        risk_ci_min_samples = policy.get(
            "risk_ci_min_samples", risk_policy.get("min_samples", 3)
        )
        logic = str(policy.get("severe_logic", "or")).strip().lower()
        risk_hit = False
        tvc_hit = False
        risk_thresh = None
        tvc_thresh = None
        if risk is not None:
            if risk_trigger == "ci":
                history = self._get_signal_history(
                    "overfit_risk", window=risk_ci_window
                )
                if len(history) >= int(risk_ci_min_samples or 0):
                    q = float(risk_ci_quantile if risk_ci_quantile is not None else 0.1)
                    q = min(max(q, 0.0), 1.0)
                    thresh = float(np.quantile(history, 1.0 - q))
                    risk_thresh = float(thresh)
                    risk_hit = float(risk) >= float(thresh)
            else:
                risk_hit = float(risk) >= float(risk_hi)
        if tvc_val is not None:
            if risk_trigger == "ci":
                history = self._get_signal_history(tvc_key, window=risk_ci_window)
                if len(history) >= int(risk_ci_min_samples or 0):
                    q = float(risk_ci_quantile if risk_ci_quantile is not None else 0.1)
                    q = min(max(q, 0.0), 1.0)
                    thresh = float(np.quantile(history, q))
                    tvc_thresh = float(thresh)
                    tvc_hit = float(tvc_val) <= float(thresh)
            else:
                tvc_hit = float(tvc_val) <= -float(tvc_min_hi)
        if logic == "and":
            severe = bool(risk_hit and tvc_hit)
        else:
            severe = bool(risk_hit or tvc_hit)

        diagnostics.update(
            {
                "phase": "risk_control"
                if int(round_num) >= int(risk_control_start_round)
                else "pre_risk_control",
                "base": float(base),
                "inputs": {
                    "rollback_flag": bool(rollback_flag),
                    "miou_delta": self.training_state.get("miou_delta"),
                    "miou_low_gain_streak": self.training_state.get(
                        "miou_low_gain_streak"
                    ),
                    "overfit_risk_raw": None if risk_raw is None else float(risk_raw),
                    "overfit_risk_ema": None if risk is None else float(risk),
                    "tvc_key": str(tvc_key),
                    "tvc_value": None if tvc_val is None else float(tvc_val),
                    "epoch_miou_volatility": self.training_state.get(
                        "epoch_miou_volatility"
                    ),
                    "tvc_sign_flip_rate": self.training_state.get("tvc_sign_flip_rate"),
                },
                "risk_control": {
                    "severe_logic": str(logic),
                    "risk_trigger": str(risk_trigger),
                    "risk_ci_window": None
                    if risk_ci_window is None
                    else int(risk_ci_window),
                    "risk_ci_quantile": None
                    if risk_ci_quantile is None
                    else float(risk_ci_quantile),
                    "risk_ci_min_samples": None
                    if risk_ci_min_samples is None
                    else int(risk_ci_min_samples),
                    "risk_hit": bool(risk_hit),
                    "tvc_hit": bool(tvc_hit),
                    "risk_thresh": risk_thresh,
                    "tvc_thresh": tvc_thresh,
                    "severe": bool(severe),
                },
            }
        )
        if bool(policy.get("u_adaptive", False)) and int(round_num) >= int(
            risk_control_start_round
        ):
            u_history = self.training_state.get("train_u_median_history", [])
            u_hist_len = len(u_history) if isinstance(u_history, list) else 0
            if u_hist_len >= 2:
                u_curr = u_history[-1][1]
                u_prev = u_history[-2][1]
                u_delta = float(u_curr) - float(u_prev)
            else:
                u_curr = None
                u_prev = None
                u_delta = None
            diagnostics["u_adaptive"] = {
                "enabled": True,
                "history_len": int(u_hist_len),
                "u_curr": u_curr,
                "u_prev": u_prev,
                "u_delta": u_delta,
                "threshold_down": float(policy.get("u_delta_down_threshold", -0.3)),
                "threshold_up": float(policy.get("u_delta_up_threshold", 0.1)),
            }
            adaptive_delta, adaptive_rule = self._get_delta_adaptive(
                round_num, policy, risk
            )
            if adaptive_delta is not None and adaptive_rule is not None:
                applied = float(
                    min(
                        max(float(applied) + float(adaptive_delta), clamp_min),
                        clamp_max,
                    )
                )
                rule = str(adaptive_rule)
                diagnostics["u_adaptive"]["triggered"] = True
                diagnostics["u_adaptive"]["applied_delta"] = float(adaptive_delta)
                diagnostics["u_adaptive"]["rule"] = str(adaptive_rule)
                return {
                    "applied": float(applied),
                    "bounds": {"min": float(clamp_min), "max": float(clamp_max)},
                    "rule": str(rule),
                    "base": float(base),
                    "diagnostics": diagnostics,
                }
            diagnostics["u_adaptive"]["triggered"] = False
        else:
            diagnostics["u_adaptive"] = {"enabled": False}

        geometry_control = self._geometry_control_decision(
            round_num=int(round_num),
            policy=policy,
            base=float(applied),
            clamp_min=float(clamp_min),
            clamp_max=float(clamp_max),
            delta_up=float(delta_up),
            delta_down=float(delta_down),
        )
        diagnostics["geometry_controller"] = geometry_control.get("diagnostics")
        geometry_fallback = bool(
            (geometry_control.get("diagnostics") or {}).get("fallback_to_risk_policy", True)
        )

        if round_num >= risk_control_start_round:
            epoch_vol = self.training_state.get("epoch_miou_volatility")
            tvc_flip = self.training_state.get("tvc_sign_flip_rate")
            vol_hi = float(
                self._agent_threshold("EPOCH_MIOU_VOLATILITY_HI", 0.03) or 0.03
            )
            flip_hi = float(
                self._agent_threshold("TVC_SIGN_FLIP_RATE_HI", 0.65) or 0.65
            )
            vol_hit = False
            flip_hit = False
            try:
                if epoch_vol is not None:
                    vol_hit = float(epoch_vol) >= float(vol_hi)
            except Exception:
                vol_hit = False
            try:
                if tvc_flip is not None:
                    flip_hit = float(tvc_flip) >= float(flip_hi)
            except Exception:
                flip_hit = False
            diagnostics["thresholds"]["epoch_miou_volatility_hi"] = float(vol_hi)
            diagnostics["thresholds"]["tvc_sign_flip_rate_hi"] = float(flip_hi)
            diagnostics["risk_control"]["volatility_hit"] = bool(vol_hit)
            diagnostics["risk_control"]["sign_flip_hit"] = bool(flip_hit)
            if rollback_flag:
                if delta_down > 0:
                    applied = float(max(float(applied) - float(delta_down), clamp_min))
                    self._last_lambda_down_round = round_num
                rule = "rollback_lambda_down"
            elif severe:
                cooling_rounds = int(
                    self._agent_threshold("LAMBDA_DOWN_COOLING_ROUNDS", 0)
                )
                in_cooling = (
                    round_num - self._last_lambda_down_round
                ) <= cooling_rounds
                diagnostics["thresholds"]["lambda_down_cooling_rounds"] = int(
                    cooling_rounds
                )
                diagnostics["risk_control"]["in_cooling"] = bool(in_cooling)

                if in_cooling:
                    rule = "severe_overfit_lambda_down_blocked_cooling"
                elif delta_down > 0:
                    applied = float(max(float(applied) - float(delta_down), clamp_min))
                    self._last_lambda_down_round = round_num
                    rule = "severe_overfit_lambda_down"
                else:
                    rule = "severe_overfit_lambda_down_no_delta"
            elif bool(geometry_control.get("handled")):
                applied = float(geometry_control.get("applied"))
                rule = str(geometry_control.get("rule"))
            elif bool(geometry_control.get("enabled")) and (not geometry_fallback):
                rule = "geometry_hold_missing_signal"
            else:
                allow_up = False
                allow_up_reason = None

                u_mean = None
                k_mean = None
                if self.current_scores:
                    try:
                        n = 0
                        su = 0.0
                        sk = 0.0
                        for v in self.current_scores.values():
                            su += float(v.get("U", 0.0))
                            sk += float(v.get("K", 0.0))
                            n += 1
                        if n > 0:
                            u_mean = float(su) / float(n)
                            k_mean = float(sk) / float(n)
                    except Exception:
                        u_mean = None
                        k_mean = None

                miou_delta = self.training_state.get("miou_delta")
                low_gain_thresh = float(
                    self._agent_threshold("MIOU_LOW_GAIN_THRESH", 0.0) or 0.0
                )
                risk_up_max = float(
                    self._agent_threshold("OVERFIT_RISK_LAMBDA_UP_MAX", 0.0) or 0.0
                )
                k_u_gap_min = float(
                    self._agent_threshold("LAMBDA_UP_K_U_GAP_MIN", 0.0) or 0.0
                )
                diagnostics["thresholds"]["miou_low_gain_thresh"] = float(
                    low_gain_thresh
                )
                diagnostics["thresholds"]["overfit_risk_lambda_up_max"] = float(
                    risk_up_max
                )
                diagnostics["thresholds"]["lambda_up_k_u_gap_min"] = float(k_u_gap_min)
                diagnostics["inputs"]["u_mean"] = u_mean
                diagnostics["inputs"]["k_mean"] = k_mean
                if u_mean is not None and k_mean is not None:
                    diagnostics["inputs"]["k_u_gap"] = float(k_mean) - float(u_mean)

                if bool(vol_hit) or bool(flip_hit):
                    allow_up = False
                    allow_up_reason = "stability_risk_block_up"
                elif (
                    risk is not None
                    and miou_delta is not None
                    and u_mean is not None
                    and k_mean is not None
                    and float(risk) <= float(risk_up_max)
                    and float(miou_delta) <= float(low_gain_thresh)
                    and (float(k_mean) - float(u_mean)) >= float(k_u_gap_min)
                ):
                    allow_up = True
                    allow_up_reason = "low_risk_k_dominant_up"

                if (not allow_up) and (not (bool(vol_hit) or bool(flip_hit))):
                    if (
                        risk is not None
                        and float(risk) <= float(risk_lo)
                        and int(self._miou_low_gain_streak) >= int(streak_need)
                    ):
                        allow_up = True
                        allow_up_reason = "low_risk_low_gain_small_up"

                if allow_up and delta_up > 0:
                    applied = float(min(float(applied) + float(delta_up), clamp_max))
                    rule = str(allow_up_reason or "low_risk_up")
                else:
                    if bool(vol_hit) or bool(flip_hit):
                        rule = "stability_risk_hold"
                    else:
                        rule = "hold"
        else:
            rule = "pre_risk_control_hold"

        ramp = policy.get("late_stage_ramp")
        if isinstance(ramp, dict):
            try:
                start = int(ramp.get("start_round", 0) or 0)
                end = int(ramp.get("end_round", start) or start)
                v0 = float(
                    ramp.get("start_lambda", ramp.get("lambda_start", 0.0)) or 0.0
                )
                v1 = float(ramp.get("end_lambda", ramp.get("lambda_end", v0)) or v0)
                if end < start:
                    start, end = end, start
                    v0, v1 = v1, v0
                if round_num >= start and start > 0:
                    ramp_conditional = bool(ramp.get("conditional", True))
                    risk_for_ramp_max = ramp.get("risk_max_for_ramp")
                    if risk_for_ramp_max is None:
                        risk_for_ramp_max = self._agent_threshold(
                            "OVERFIT_RISK_FOR_RAMP_MAX", 0.4
                        )
                    tvc_neg_rate_max = ramp.get("tvc_neg_rate_max_for_ramp")
                    if tvc_neg_rate_max is None:
                        tvc_neg_rate_max = self._agent_threshold(
                            "TVC_NEG_RATE_FOR_RAMP_MAX", 0.5
                        )
                    miou_delta_max = ramp.get("miou_delta_max_for_ramp")
                    if miou_delta_max is None:
                        miou_delta_max = self._agent_threshold(
                            "MIOU_LOW_GAIN_THRESH", 0.0
                        )
                    require_low_gain = bool(ramp.get("require_low_gain", True))
                    require_stable_tvc = bool(ramp.get("require_stable_tvc", True))
                    ramp_risk_ok = True
                    if risk is not None and risk_for_ramp_max is not None:
                        ramp_risk_ok = float(risk) <= float(risk_for_ramp_max)
                    ramp_low_gain_ok = True
                    if require_low_gain:
                        md = self.training_state.get("miou_delta")
                        if md is None:
                            ramp_low_gain_ok = int(self._miou_low_gain_streak) >= max(
                                int(streak_need), 1
                            )
                        else:
                            ramp_low_gain_ok = float(md) <= float(miou_delta_max)
                    ramp_tvc_ok = True
                    if require_stable_tvc:
                        tvc_neg = self.training_state.get("grad_train_val_cos_neg_rate")
                        if tvc_neg is not None:
                            ramp_tvc_ok = float(tvc_neg) <= float(tvc_neg_rate_max)
                    ramp_gate_ok = bool(
                        ramp_risk_ok and ramp_low_gain_ok and ramp_tvc_ok
                    )
                    diagnostics["late_stage_ramp"] = {
                        "conditional": bool(ramp_conditional),
                        "risk_max_for_ramp": None
                        if risk_for_ramp_max is None
                        else float(risk_for_ramp_max),
                        "tvc_neg_rate_max_for_ramp": None
                        if tvc_neg_rate_max is None
                        else float(tvc_neg_rate_max),
                        "miou_delta_max_for_ramp": None
                        if miou_delta_max is None
                        else float(miou_delta_max),
                        "require_low_gain": bool(require_low_gain),
                        "require_stable_tvc": bool(require_stable_tvc),
                        "risk_ok": bool(ramp_risk_ok),
                        "low_gain_ok": bool(ramp_low_gain_ok),
                        "tvc_ok": bool(ramp_tvc_ok),
                        "gate_ok": bool(ramp_gate_ok),
                    }
                    if end == start:
                        floor = float(v1)
                    else:
                        t = float(round_num - start) / float(end - start)
                        t = float(min(max(t, 0.0), 1.0))
                        floor = float(v0 + (v1 - v0) * t)
                    floor = float(min(max(floor, 0.0), 1.0))
                    if (not ramp_conditional or ramp_gate_ok) and rule not in (
                        "rollback_lambda_down",
                        "severe_overfit_lambda_down",
                        "severe_overfit_lambda_down_blocked_cooling",
                        "severe_overfit_lambda_down_no_delta",
                    ):
                        if float(applied) < float(floor):
                            applied = float(floor)
                            rule = f"{rule}+late_stage_ramp"
                    elif ramp_conditional and (not ramp_gate_ok):
                        diagnostics["late_stage_ramp"]["blocked"] = True
            except Exception:
                pass

        smoothing = str(policy.get("lambda_smoothing", "") or "").strip().lower()
        diagnostics["smoothing"] = {"mode": smoothing or None}
        if smoothing == "ema" and self._last_lambda_applied is not None:
            alpha = float(policy.get("lambda_smoothing_alpha", 0.7))
            alpha = min(max(alpha, 0.0), 1.0)
            diagnostics["smoothing"]["alpha"] = float(alpha)
            applied = alpha * float(applied) + (1.0 - alpha) * float(
                self._last_lambda_applied
            )
        max_step = policy.get("lambda_max_step")
        diagnostics["smoothing"]["max_step"] = (
            None if max_step is None else float(max_step)
        )
        if max_step is not None and self._last_lambda_applied is not None:
            step = abs(float(max_step))
            if step > 0:
                delta = float(applied) - float(self._last_lambda_applied)
                if abs(delta) > step:
                    applied = float(self._last_lambda_applied) + step * (
                        1.0 if delta > 0 else -1.0
                    )
        late_start = int(policy.get("late_u_bias_start_round", 0) or 0)
        late_strength = float(policy.get("late_u_bias_strength", 0.0) or 0.0)
        diagnostics["late_u_bias"] = {
            "start_round": int(late_start),
            "strength": float(late_strength),
        }
        if late_strength > 0 and round_num >= late_start:
            late_strength = min(max(late_strength, 0.0), 1.0)
            applied = (1.0 - late_strength) * float(applied) + late_strength * float(
                clamp_min
            )
        applied = float(min(max(applied, clamp_min), clamp_max))
        diagnostics.update({"applied": float(applied), "rule": str(rule)})
        return {
            "applied": applied,
            "bounds": {"min": clamp_min, "max": clamp_max},
            "rule": rule,
            "base": float(base),
            "diagnostics": diagnostics,
        }

    def apply_round_lambda_policy(self) -> Optional[float]:
        policy = self._lambda_policy_config()
        if not isinstance(policy, dict):
            return None
        if "lambda_override_round" in (self.control_state or {}):
            return float(self.control_state.get("lambda_override_round"))
        round_num = self._current_round()
        payload = self._compute_policy_lambda_for_round(round_num, policy)
        applied = float(payload.get("applied"))
        diagnostics = payload.get("diagnostics") if isinstance(payload, dict) else None
        self.control_state["lambda_override_round"] = applied
        self.control_meta["lambda_override_round"] = {
            "clamped": True,
            "clamp_reason": payload.get("rule"),
            "policy": {
                "mode": str(policy.get("mode")),
                "bounds": payload.get("bounds"),
                "base": payload.get("base"),
            },
        }
        rule = str(payload.get("rule") or "")
        if rule == "geometry_sensitive_down":
            self._geometry_down_streak = int(getattr(self, "_geometry_down_streak", 0)) + 1
        else:
            self._geometry_down_streak = 0
        self._last_lambda_applied = float(applied)
        if hasattr(self.controller, "_append_trace"):
            ts = self.training_state if isinstance(self.training_state, dict) else {}
            u_hist = ts.get("train_u_median_history")
            k_hist = ts.get("train_k_median_history")
            ts_snapshot = {
                "grad_probe_source": ts.get("grad_probe_source"),
                "u_median_selected": ts.get("u_median_selected"),
                "k_median_selected": ts.get("k_median_selected"),
                "u_median_top": ts.get("u_median_top"),
                "k_median_top": ts.get("k_median_top"),
                "selection_geometry": dict(ts.get("selection_geometry"))
                if isinstance(ts.get("selection_geometry"), dict)
                else None,
                "train_u_median_selected": ts.get("train_u_median_selected"),
                "train_k_median_selected": ts.get("train_k_median_selected"),
                "train_u_median_history": list(u_hist)
                if isinstance(u_hist, list)
                else [],
                "train_k_median_history": list(k_hist)
                if isinstance(k_hist, list)
                else [],
                "max_history_length": ts.get("max_history_length"),
            }
            self.controller._append_trace(
                {
                    "type": "lambda_policy_apply",
                    "round": int(round_num),
                    "applied": float(applied),
                    "rule": payload.get("rule"),
                    "base": payload.get("base"),
                    "bounds": payload.get("bounds"),
                    "policy_mode": str(policy.get("mode")),
                    "diagnostics": diagnostics
                    if isinstance(diagnostics, dict)
                    else None,
                    "training_state": ts_snapshot,
                }
            )
        return float(applied)

    def _error_response(
        self, error_type: str, message: str, meta: Optional[Dict] = None
    ) -> str:
        return json.dumps(
            {
                "status": "error",
                "error_type": error_type,
                "message": message,
                "meta": meta or {},
            }
        )

    def _parse_response(self, response_str: str) -> Dict[str, Any]:
        """
        兼容解析层：处理新旧两种返回结构
        Args:
            response_str: JSON string from tool
        Returns:
            dict with unified structure: {status, result/error_type/message, meta}
        """
        try:
            parsed = json.loads(response_str)
        except Exception:
            return {
                "status": "error",
                "error_type": "ParseError",
                "message": "Invalid JSON response",
                "meta": {},
            }

        if not isinstance(parsed, dict):
            return {
                "status": "error",
                "error_type": "ParseError",
                "message": "Response is not a dict",
                "meta": {},
            }

        if "status" in parsed:
            return parsed

        return {"status": "success", "result": parsed, "meta": {}}

    def precalculate_scores(self) -> None:
        """
        在每一轮 AL 开始前调用，预计算所有未标注样本的得分
        """
        self.last_score_error = None
        if hasattr(self.controller, "_last_score_precalc_error"):
            self.controller._last_score_precalc_error = None

        print("Pre-calculating scores for all unlabeled samples...")
        cfg = getattr(self.controller, "config", None)
        dataset = getattr(self.controller, "dataset", None) or getattr(
            self.controller, "full_dataset", None
        )
        if hasattr(self.strategy, "set_round"):
            self.strategy.set_round(self._current_round())
        if dataset is None:
            self.current_scores = {}
            self.controller._last_score_precalc_error = {
                "phase": "score_precalc",
                "exception": "DatasetMissing",
                "message": "Dataset not found in controller",
            }
            if hasattr(self.controller, "_append_trace"):
                self.controller._append_trace(
                    {
                        "type": "ranking_degraded",
                        "round": int(getattr(self.controller, "current_round", 0) or 0),
                        "degraded": {
                            "source": "toolbox",
                            "reason_type": "dataset_missing",
                            "policy": "no_scores",
                        },
                    }
                )
            raise RuntimeError("Dataset not found in controller")
        try:
            u_norm, k_norm = self.strategy.calculate_scores(
                self.model,
                dataset,
                self.controller.unlabeled_indices,
                self.controller.labeled_indices,
            )
        except Exception as e:
            err = {
                "phase": "score_precalc",
                "exception": e.__class__.__name__,
                "message": str(e),
            }
            self.last_score_error = dict(err)
            if hasattr(self.controller, "_last_score_precalc_error"):
                self.controller._last_score_precalc_error = dict(err)
            self.current_scores = {}
            if hasattr(self.controller, "_append_trace"):
                self.controller._append_trace(
                    {
                        "type": "ranking_degraded",
                        "round": int(getattr(self.controller, "current_round", 0) or 0),
                        "degraded": {
                            "source": "toolbox",
                            "reason_type": "score_precalc_failed",
                            "policy": "no_scores",
                            "exception": e.__class__.__name__,
                            "message": str(e),
                        },
                    }
                )
            raise

        self.current_scores = {}
        for i, idx in enumerate(self.controller.unlabeled_indices):
            self.current_scores[str(idx)] = {
                "U": float(u_norm[i]),
                "K": float(k_norm[i]),
            }
        print("Score calculation completed.")

    def get_system_status(self) -> str:
        """
        Tool: 获取当前系统状态（唯一权威状态工具）
        Returns: JSON string with unified structure
        """
        if not self._check_permission("get_system_status"):
            raise RuntimeError("PermissionDenied: get_system_status")
        try:
            t = len(self.controller.labeled_indices)
            dataset = getattr(self.controller, "dataset", None) or getattr(
                self.controller, "full_dataset", None
            )
            if dataset is None:
                raise StateError(
                    "Dataset not found in controller",
                    {"controller_type": type(self.controller).__name__},
                )
            total = len(dataset)
            total_budget = getattr(self.controller, "config", None)
            t_max = (
                getattr(total_budget, "TOTAL_BUDGET", None)
                if total_budget is not None
                else None
            )
            if t_max in (None, 0):
                t_max = int(total * 0.1)
            if t_max == 0:
                t_max = 1

            progress = t / t_max
            policy = self._lambda_policy_config()
            if isinstance(policy, dict):
                if (
                    self._check_permission("set_lambda")
                    and self._require_explicit_lambda()
                ):
                    payload = self._compute_policy_lambda_for_round(
                        self._current_round(), policy
                    )
                    lambda_t = float(payload.get("applied"))
                else:
                    lambda_t = self.apply_round_lambda_policy()
                    if lambda_t is None:
                        raise RuntimeError(
                            "lambda_policy is enabled but apply_round_lambda_policy returned None"
                        )
            else:
                lambda_t = AgentThresholds.calculate_lambda_t(progress, self.alpha)
            late_stage_ratio = float(
                self._agent_threshold(
                    "LATE_STAGE_RATIO",
                    getattr(AgentThresholds, "LATE_STAGE_RATIO", 0.2),
                )
            )

            last_miou = self.training_state.get("last_miou", 0.0)
            prev_miou = self.training_state.get("prev_miou", 0.0)
            miou_delta = last_miou - prev_miou

            if np.isnan(miou_delta) or np.isinf(miou_delta):
                raise RuntimeError("miou_delta is NaN/Inf")

            rollback_flag = self.training_state.get("rollback_flag", False)
            rollback_mode = self.training_state.get("rollback_mode", None)
            rollback_threshold = self.training_state.get("rollback_threshold", None)
            remaining_budget = max(0, t_max - t)

            status = {
                "current_labeled_count": t,
                "total_budget": t_max,
                "progress_ratio": round(progress, 4),
                "lambda_t": round(lambda_t, 4),
                "stage_description": "Early Stage"
                if progress < (1 - late_stage_ratio)
                else ("Late Stage" if progress > late_stage_ratio else "Middle Stage"),
                "last_miou": round(last_miou, 4),
                "prev_miou": round(prev_miou, 4),
                "miou_delta": round(miou_delta, 4),
                "rollback_flag": bool(rollback_flag),
                "rollback_mode": rollback_mode,
                "rollback_threshold": rollback_threshold,
                "remaining_budget": int(remaining_budget),
                "grad_train_val_cos_mean": self.training_state.get(
                    "grad_train_val_cos_mean"
                ),
                "grad_train_val_cos_min": self.training_state.get(
                    "grad_train_val_cos_min"
                ),
                "grad_train_val_cos_max": self.training_state.get(
                    "grad_train_val_cos_max"
                ),
                "grad_train_val_cos_last": self.training_state.get(
                    "grad_train_val_cos_last"
                ),
                "grad_train_val_cos_neg_rate": self.training_state.get(
                    "grad_train_val_cos_neg_rate"
                ),
                "overfit_risk": self.training_state.get("overfit_risk"),
            }

            meta = {
                "schema_version": self.SCHEMA_VERSION,
                "observation_contract": {
                    "version": 1,
                    "raw_image_access": False,
                    "sample_details_fields": [
                        "id",
                        "U_score",
                        "K_score",
                        "image_name",
                        "split",
                        "image_mean",
                        "image_std",
                        "mask_positive_ratio",
                        "in_labeled_pool",
                        "in_unlabeled_pool",
                    ],
                },
                "overfit_thresholds": {
                    "tvc_warn": float(
                        self._agent_threshold(
                            "OVERFIT_TVC_WARN",
                            getattr(AgentThresholds, "OVERFIT_TVC_WARN", 0.0),
                        )
                    ),
                    "tvc_severe": float(
                        self._agent_threshold(
                            "OVERFIT_TVC_SEVERE",
                            getattr(AgentThresholds, "OVERFIT_TVC_SEVERE", 0.0),
                        )
                    ),
                    "lambda_cap_warn": float(
                        self._agent_threshold(
                            "OVERFIT_LAMBDA_CAP_WARN",
                            getattr(AgentThresholds, "OVERFIT_LAMBDA_CAP_WARN", 0.0),
                        )
                    ),
                    "lambda_cap_severe": float(
                        self._agent_threshold(
                            "OVERFIT_LAMBDA_CAP_SEVERE",
                            getattr(AgentThresholds, "OVERFIT_LAMBDA_CAP_SEVERE", 0.0),
                        )
                    ),
                    "query_ratio_warn": float(
                        self._agent_threshold(
                            "OVERFIT_QUERY_RATIO_WARN",
                            getattr(AgentThresholds, "OVERFIT_QUERY_RATIO_WARN", 0.0),
                        )
                    ),
                    "query_ratio_severe": float(
                        self._agent_threshold(
                            "OVERFIT_QUERY_RATIO_SEVERE",
                            getattr(AgentThresholds, "OVERFIT_QUERY_RATIO_SEVERE", 0.0),
                        )
                    ),
                },
            }

            return self._success_response(status, meta)
        except StateError as e:
            raise RuntimeError(f"{e.error_type}: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"TOOL_FAILURE: 获取系统状态失败: {str(e)}")

    def reset_round_controls(self) -> None:
        self.control_state: Dict[str, Any] = {}
        self.control_meta: Dict[str, Any] = {}

    def set_control_permissions(self, permissions: Dict[str, bool]) -> None:
        """
        设置控制动作权限（用于消融实验）
        Args:
            permissions: Dict[str, bool] - Key为工具名，Value为是否允许
        """
        # Setter 类默认为 False (需显式开启)
        setter_keys = {
            "set_lambda",
            "set_query_size",
            "set_epochs_per_round",
            "set_alpha",
        }
        # Getter 类默认为 True (需显式关闭)
        getter_keys = {
            "get_top_k_samples",
            "get_sample_details",
            "get_score_distribution",
            "get_system_status",
            "finalize_selection",
        }

        valid_keys = setter_keys | getter_keys

        # 重置为默认状态
        self.control_permissions = {}
        for k in setter_keys:
            self.control_permissions[k] = False
        for k in getter_keys:
            self.control_permissions[k] = True

        # 应用用户配置
        for key, value in (permissions or {}).items():
            if key in valid_keys:
                self.control_permissions[key] = bool(value)

    def _check_permission(self, action_name: str) -> bool:
        """
        检查控制动作权限
        Args:
            action_name: 控制动作名称
        Returns:
            bool: 是否允许执行该动作
        """
        return self.control_permissions.get(action_name, False)

    def _is_alpha_allowed(self) -> bool:
        return bool(self.control_permissions.get("set_alpha", False))

    def _default_lambda(self) -> float:
        policy = self._lambda_policy_config()
        if isinstance(policy, dict):
            v = self.apply_round_lambda_policy()
            if v is None:
                raise RuntimeError(
                    "lambda_policy is enabled but apply_round_lambda_policy returned None"
                )
            return float(v)
        try:
            labeled = len(getattr(self.controller, "labeled_indices", []) or [])
            dataset = getattr(self.controller, "dataset", None) or getattr(
                self.controller, "full_dataset", None
            )
            total = len(dataset) if dataset is not None else 0
            cfg = getattr(self.controller, "config", None)
            t_max = getattr(cfg, "TOTAL_BUDGET", None) if cfg is not None else None
            if t_max in (None, 0):
                t_max = int(total * 0.1) if total > 0 else 0
            if t_max == 0:
                t_max = 1
            progress = labeled / t_max
            return float(AgentThresholds.calculate_lambda_t(progress, self.alpha))
        except Exception:
            return 0.5

    def get_candidate_samples(
        self, top_k: int = 10, sort_by: str = "U", lambda_param: Optional[float] = None
    ) -> str:
        """
        Tool: 获取候选样本列表（已弃用，请使用 get_top_k_samples）
        Args:
            top_k: 返回前 K 个
            sort_by: 'U' (Uncertainty), 'K' (Knowledge), 'Hybrid' (Legacy, defaults to lambda=0.5)
            lambda_param: float (0.0 - 1.0). If provided, overrides sort_by.
                          Score = (1 - lambda) * U + lambda * K
        """
        # Determine lambda
        final_lambda = 0.0
        if not self._check_permission("set_lambda"):
            final_lambda = self._default_lambda()
        elif lambda_param is not None:
            final_lambda = float(lambda_param)
        else:
            if sort_by == "U":
                final_lambda = 0.0
            elif sort_by == "K":
                final_lambda = 1.0
            elif sort_by == "Hybrid":
                final_lambda = 0.5
            else:
                final_lambda = 0.5

        # 将 scores 转为 list 用于排序
        score_list = []
        for idx, scores in self.current_scores.items():
            item = {"id": idx, "U": scores["U"], "K": scores["K"]}
            # Score = (1 - lambda) * U + lambda * K
            item["score"] = (1 - final_lambda) * scores["U"] + final_lambda * scores[
                "K"
            ]
            score_list.append(item)

        # 降序排列
        score_list.sort(key=lambda x: x["score"], reverse=True)

        candidates = score_list[: int(top_k)]
        self.candidates_cache = candidates  # 缓存，供后续选择验证
        meta = {
            "schema_version": self.SCHEMA_VERSION,
            "deprecated": True,
            "replacement": "get_top_k_samples",
        }
        return self._success_response(candidates, meta)

    def get_top_k_samples(
        self, k: int = 5, lambda_param: Optional[float] = None
    ) -> str:
        if not self._check_permission("get_top_k_samples"):
            raise RuntimeError("PermissionDenied: get_top_k_samples")
        if not self.current_scores:
            raise RuntimeError(
                "No scores available: call precalculate_scores() before get_top_k_samples"
            )

        if not self._check_permission("set_lambda"):
            final_lambda = self._default_lambda()
        else:
            if lambda_param is not None:
                _ = self.set_lambda(lambda_param, scope="round")
                final_lambda = float(self.control_state.get("lambda_override_round"))
            else:
                existing = self.control_state.get("lambda_override_round")
                if existing is None:
                    raise RuntimeError(
                        "lambda_param is required unless set_lambda has been called in this round"
                    )
                final_lambda = float(existing)

        clamp_min = float(
            self._agent_threshold("LAMBDA_CLAMP_MIN", AgentConstraints.LAMBDA_MIN)
        )
        clamp_max = float(
            self._agent_threshold("LAMBDA_CLAMP_MAX", AgentConstraints.LAMBDA_MAX)
        )
        if not self._check_permission("set_lambda"):
            policy = self._lambda_policy_config()
            if isinstance(policy, dict):
                payload = self._compute_policy_lambda_for_round(
                    self._current_round(), policy
                )
                bounds = payload.get("bounds")
                if (
                    not isinstance(bounds, dict)
                    or ("min" not in bounds)
                    or ("max" not in bounds)
                ):
                    raise RuntimeError(
                        "lambda_policy bounds missing: expected dict with min/max"
                    )
                b_min = float(bounds.get("min"))
                b_max = float(bounds.get("max"))
                if b_min > b_max:
                    b_min, b_max = b_max, b_min
                b_min = float(max(0.0, b_min))
                b_max = float(min(1.0, b_max))
                clamp_min = float(b_min)
                clamp_max = float(b_max)
        lambda_before = float(final_lambda)
        lambda_clamped = False
        if float(final_lambda) < float(clamp_min):
            final_lambda = float(clamp_min)
            lambda_clamped = True
        if float(final_lambda) > float(clamp_max):
            final_lambda = float(clamp_max)
            lambda_clamped = True

        items = []
        for idx, scores in self.current_scores.items():
            u = float(scores["U"])
            k_score = float(scores["K"])
            s = (1 - float(final_lambda)) * u + float(final_lambda) * k_score
            items.append(
                {"id": idx, "U_score": u, "K_score": k_score, "final_score": float(s)}
            )

        items.sort(key=lambda x: x["final_score"], reverse=True)
        result = items[: int(k)]
        self.candidates_cache = result
        meta = {
            "schema_version": self.SCHEMA_VERSION,
            "lambda_param": float(final_lambda),
            "lambda_before_clamp": float(lambda_before),
            "lambda_clamped": bool(lambda_clamped),
            "k": int(k),
        }
        return self._success_response(result, meta)

    def set_lambda(self, lambda_value: Union[float, str], scope: str = "round") -> str:
        if not self._check_permission("set_lambda"):
            raise RuntimeError("PermissionDenied: set_lambda")
        v_req = float(lambda_value)
        existing = self.control_state.get("lambda_override_round")
        if existing is not None and str(scope or "round") == "round":
            try:
                if abs(float(existing) - float(v_req)) <= 1e-12:
                    return json.dumps(
                        {
                            "status": "success",
                            "applied": float(existing),
                            "requested": float(v_req),
                            "scope": "round",
                            "clamped": False,
                            "clamp_reason": None,
                            "duplicate": True,
                        },
                        ensure_ascii=False,
                    )
            except Exception:
                pass

        suggested_lambda = None
        delta_from_suggested = None
        exceeds_suggested_range = None
        t = len(getattr(self.controller, "labeled_indices", []) or [])
        total_budget = int(self.training_state.get("total_budget") or 0)
        if total_budget <= 0:
            total_budget = int(
                getattr(getattr(self.controller, "config", None), "TOTAL_BUDGET", 0)
                or 0
            )
        if total_budget > 0:
            progress = float(t) / float(max(1, total_budget))
            suggested_lambda = float(
                AgentThresholds.calculate_lambda_t(progress, self.alpha)
            )
            delta_from_suggested = float(v_req) - float(suggested_lambda)
            exceeds_suggested_range = bool(
                abs(delta_from_suggested)
                > float(
                    self._agent_threshold(
                        "LAMBDA_ADJUST_RANGE",
                        getattr(AgentThresholds, "LAMBDA_ADJUST_RANGE", 0.0),
                    )
                )
            )

        if v_req < 0.0:
            raise ValueError(f"lambda {v_req} < min 0.0")
        if v_req > 1.0:
            raise ValueError(f"lambda {v_req} > max 1.0")

        v = float(v_req)
        clamped = False
        clamp_reason = None
        clamp_min = float(
            self._agent_threshold("LAMBDA_CLAMP_MIN", AgentConstraints.LAMBDA_MIN)
        )
        clamp_max = float(
            self._agent_threshold("LAMBDA_CLAMP_MAX", AgentConstraints.LAMBDA_MAX)
        )
        policy = self._lambda_policy_config()
        in_uncertainty_phase = False
        if isinstance(policy, dict) and str(policy.get("mode") or "") == "warmup_risk_closed_loop":
            round_num = int(getattr(self.controller, "current_round", 0) or 0)
            try:
                payload = self._compute_policy_lambda_for_round(int(round_num), policy)
                bounds = payload.get("bounds") if isinstance(payload, dict) else None
                if isinstance(bounds, dict) and ("min" in bounds) and ("max" in bounds):
                    b_min = float(bounds.get("min"))
                    b_max = float(bounds.get("max"))
                    if b_min > b_max:
                        b_min, b_max = b_max, b_min
                    b_min = float(max(0.0, b_min))
                    b_max = float(min(1.0, b_max))
                    clamp_min = float(b_min)
                    clamp_max = float(b_max)
                uncertainty_only_rounds = int(policy.get("uncertainty_only_rounds", 0) or 0)
                if round_num > 0 and round_num <= max(1, uncertainty_only_rounds):
                    in_uncertainty_phase = True
            except Exception:
                pass
        if v < clamp_min:
            v = float(clamp_min)
            clamped = True
            clamp_reason = f"policy_clamp_min:{clamp_min}"
        if v > clamp_max:
            v = float(clamp_max)
            clamped = True
            clamp_reason = f"policy_clamp_max:{clamp_max}"

        adjust_range = float(
            self._agent_threshold(
                "LAMBDA_ADJUST_RANGE",
                getattr(AgentThresholds, "LAMBDA_ADJUST_RANGE", 0.0) or 0.0,
            )
            or 0.0
        )
        if (
            (not in_uncertainty_phase)
            and suggested_lambda is not None
            and np.isfinite(float(suggested_lambda))
            and float(adjust_range) > 0
        ):
            lo = float(suggested_lambda) - float(adjust_range)
            hi = float(suggested_lambda) + float(adjust_range)
            if lo > hi:
                lo, hi = hi, lo
            v_before = float(v)
            v = float(min(max(float(v), float(lo)), float(hi)))
            if abs(float(v) - float(v_before)) > 1e-12:
                clamped = True
                clamp_reason = clamp_reason or "suggested_adjust_range"

        overfit_guard = {
            "enabled": True,
            "tvc_last": None,
            "tvc_min": None,
            "overfit_risk": None,
            "tvc_key": None,
            "tvc_value": None,
            "risk_trigger": None,
            "severe_logic": None,
            "risk_hit": None,
            "tvc_hit": None,
            "rollback_flag": None,
            "cooling": None,
            "rule": None,
            "applied": False,
            "delta": None,
        }
        last = self._last_lambda_applied
        tvc_last = self.training_state.get("grad_train_val_cos_last")
        tvc_min = self.training_state.get("grad_train_val_cos_min")
        risk = self.training_state.get("overfit_risk")
        rollback_flag = bool(self.training_state.get("rollback_flag", False))
        try:
            overfit_guard["tvc_last"] = None if tvc_last is None else float(tvc_last)
            overfit_guard["tvc_min"] = None if tvc_min is None else float(tvc_min)
            overfit_guard["overfit_risk"] = None if risk is None else float(risk)
            overfit_guard["rollback_flag"] = bool(rollback_flag)
        except Exception:
            overfit_guard["tvc_last"] = tvc_last
            overfit_guard["tvc_min"] = tvc_min
            overfit_guard["overfit_risk"] = risk
            overfit_guard["rollback_flag"] = rollback_flag

        signals_available = (
            (tvc_last is not None) or (tvc_min is not None) or (risk is not None)
        )
        if signals_available:
            severe = False
            risk_hit = False
            tvc_hit = False
            in_cooling = False
            tvc_key = None
            tvc_val = None
            risk_trigger = ""
            logic = "or"
            try:
                policy_cfg = self._lambda_policy_config() or {}
                risk_policy = self._risk_policy_config() or {}
                tvc_key = (
                    str(
                        policy_cfg.get("severe_tvc_key") or "grad_train_val_cos_last"
                    ).strip()
                    or "grad_train_val_cos_last"
                )
                tvc_val = self.training_state.get(tvc_key)
                risk_trigger = (
                    str(
                        policy_cfg.get("risk_trigger", risk_policy.get("trigger", ""))
                        or ""
                    )
                    .strip()
                    .lower()
                )
                risk_ci_window = policy_cfg.get(
                    "risk_ci_window", risk_policy.get("window")
                )
                risk_ci_quantile = policy_cfg.get(
                    "risk_ci_quantile", risk_policy.get("quantile")
                )
                risk_ci_min_samples = policy_cfg.get(
                    "risk_ci_min_samples", risk_policy.get("min_samples", 3)
                )
                logic = (
                    str(policy_cfg.get("severe_logic", "or") or "or").strip().lower()
                )

                overfit_guard["tvc_key"] = tvc_key
                overfit_guard["tvc_value"] = None if tvc_val is None else float(tvc_val)
                overfit_guard["risk_trigger"] = risk_trigger
                overfit_guard["severe_logic"] = logic

                risk_hi = float(
                    self._agent_threshold(
                        "OVERFIT_RISK_HI",
                        getattr(AgentThresholds, "OVERFIT_RISK_HI", 0.0),
                    )
                    or 0.0
                )
                tvc_min_hi = float(
                    self._agent_threshold(
                        "OVERFIT_TVC_MIN_HI",
                        getattr(AgentThresholds, "OVERFIT_TVC_MIN_HI", 0.0),
                    )
                    or 0.0
                )

                if risk is not None:
                    if risk_trigger == "ci":
                        history = self._get_signal_history(
                            "overfit_risk", window=risk_ci_window
                        )
                        if len(history) >= int(risk_ci_min_samples or 0):
                            q = float(
                                risk_ci_quantile
                                if risk_ci_quantile is not None
                                else 0.2
                            )
                            q = min(max(q, 0.0), 1.0)
                            thresh = float(np.quantile(history, 1.0 - q))
                            risk_hit = float(risk) >= float(thresh)
                        else:
                            risk_hit = float(risk) >= float(risk_hi)
                    else:
                        risk_hit = float(risk) >= float(risk_hi)

                if tvc_val is not None:
                    if risk_trigger == "ci":
                        history = self._get_signal_history(
                            tvc_key, window=risk_ci_window
                        )
                        if len(history) >= int(risk_ci_min_samples or 0):
                            q = float(
                                risk_ci_quantile
                                if risk_ci_quantile is not None
                                else 0.2
                            )
                            q = min(max(q, 0.0), 1.0)
                            thresh = float(np.quantile(history, q))
                            tvc_hit = float(tvc_val) <= float(thresh)
                        else:
                            tvc_hit = float(tvc_val) <= -float(tvc_min_hi)
                    else:
                        tvc_hit = float(tvc_val) <= -float(tvc_min_hi)

                if logic == "and":
                    severe = bool(risk_hit and tvc_hit)
                else:
                    severe = bool(risk_hit or tvc_hit)
            except Exception:
                severe = False
                risk_hit = False
                tvc_hit = False
                tvc_key = None
                tvc_val = None
                risk_trigger = ""
                logic = "or"

            overfit_guard["risk_hit"] = bool(risk_hit)
            overfit_guard["tvc_hit"] = bool(tvc_hit)

            delta_down = float(
                self._agent_threshold(
                    "LAMBDA_DELTA_DOWN",
                    getattr(AgentThresholds, "LAMBDA_DELTA_DOWN", 0.0),
                )
                or 0.0
            )
            delta_up = float(
                self._agent_threshold(
                    "LAMBDA_DELTA_UP", getattr(AgentThresholds, "LAMBDA_DELTA_UP", 0.0)
                )
                or 0.0
            )
            allow_up = bool(
                (overfit_guard.get("overfit_risk") is not None)
                and float(overfit_guard.get("overfit_risk"))
                <= float(
                    self._agent_threshold(
                        "OVERFIT_RISK_LO",
                        getattr(AgentThresholds, "OVERFIT_RISK_LO", 0.0),
                    )
                    or 0.0
                )
                and int(self._miou_low_gain_streak)
                >= int(
                    self._agent_threshold(
                        "MIOU_LOW_GAIN_STREAK",
                        getattr(AgentThresholds, "MIOU_LOW_GAIN_STREAK", 0),
                    )
                    or 0
                )
            )

            cooling_rounds = int(
                self._agent_threshold(
                    "LAMBDA_DOWN_COOLING_ROUNDS",
                    getattr(AgentThresholds, "LAMBDA_DOWN_COOLING_ROUNDS", 0) or 0,
                )
                or 0
            )
            round_num = int(getattr(self.controller, "current_round", 0) or 0)
            in_cooling = bool(
                round_num > 0
                and (round_num - int(getattr(self, "_last_lambda_down_round", 0) or 0))
                <= int(cooling_rounds)
            )
            overfit_guard["cooling"] = bool(in_cooling)

            if rollback_flag and last is not None and float(v) > float(last):
                v = float(last)
                overfit_guard["rule"] = "rollback_no_increase"
                overfit_guard["applied"] = True
                overfit_guard["delta"] = 0.0
            elif severe:
                if in_cooling:
                    if last is not None and float(v) > float(last):
                        v = float(last)
                        overfit_guard["rule"] = "severe_overfit_no_increase_cooling"
                        overfit_guard["applied"] = True
                        overfit_guard["delta"] = 0.0
                elif delta_down > 0:
                    base = float(last) if last is not None else float(v)
                    target = float(base) - float(delta_down)
                    v = float(min(v, target))
                    v = float(max(v, clamp_min))
                    self._last_lambda_down_round = int(round_num)
                    overfit_guard["rule"] = "severe_overfit_lambda_down"
                    overfit_guard["applied"] = True
                    overfit_guard["delta"] = -float(delta_down)
                    if clamp_reason is None:
                        clamp_reason = "policy_overfit_down"
            else:
                if last is not None and float(v) > float(last):
                    if allow_up and (not rollback_flag) and delta_up > 0:
                        v = float(min(v, float(last) + float(delta_up)))
                        overfit_guard["rule"] = "low_risk_allow_small_up"
                        overfit_guard["applied"] = True
                        overfit_guard["delta"] = float(delta_up)
                    else:
                        v = float(last)
                        overfit_guard["rule"] = "no_increase_when_not_allowed"
                        overfit_guard["applied"] = True
                        overfit_guard["delta"] = 0.0
        else:
            overfit_guard["rule"] = "no_signal"

        if v < clamp_min:
            v = float(clamp_min)
            clamped = True
            clamp_reason = clamp_reason or f"policy_clamp_min:{clamp_min}"
        if v > clamp_max:
            v = float(clamp_max)
            clamped = True
            clamp_reason = clamp_reason or f"policy_clamp_max:{clamp_max}"

        self._last_lambda_applied = float(v)
        self.control_state["lambda_override_round"] = float(v)
        self.control_meta["lambda_override_round"] = {
            "clamped": bool(clamped),
            "clamp_reason": clamp_reason,
            "suggested_lambda": suggested_lambda,
            "delta_from_suggested": delta_from_suggested,
            "suggested_adjust_range": float(
                self._agent_threshold(
                    "LAMBDA_ADJUST_RANGE",
                    getattr(AgentThresholds, "LAMBDA_ADJUST_RANGE", 0.0),
                )
            ),
            "exceeds_suggested_range": exceeds_suggested_range,
            "overfit_guard": overfit_guard,
        }
        if hasattr(self.controller, "_append_trace"):
            self.controller._append_trace(
                {
                    "type": "lambda_override",
                    "round": int(getattr(self.controller, "current_round", 0) or 0),
                    "scope": str(scope or "round"),
                    "applied": float(v),
                    "requested": float(v_req),
                    "suggested_lambda": suggested_lambda,
                    "delta_from_suggested": delta_from_suggested,
                    "suggested_adjust_range": float(
                        self._agent_threshold(
                            "LAMBDA_ADJUST_RANGE",
                            getattr(AgentThresholds, "LAMBDA_ADJUST_RANGE", 0.0),
                        )
                    ),
                    "exceeds_suggested_range": exceeds_suggested_range,
                    "clamped": bool(clamped),
                    "clamp_reason": clamp_reason,
                    "overfit_guard": overfit_guard,
                }
            )
        payload = {
            "status": "success",
            "applied": float(v),
            "requested": float(v_req),
            "scope": scope,
            "clamped": bool(clamped),
            "clamp_reason": clamp_reason,
            "suggested_lambda": suggested_lambda,
            "delta_from_suggested": delta_from_suggested,
            "suggested_adjust_range": float(
                self._agent_threshold(
                    "LAMBDA_ADJUST_RANGE",
                    getattr(AgentThresholds, "LAMBDA_ADJUST_RANGE", 0.0),
                )
            ),
            "exceeds_suggested_range": exceeds_suggested_range,
            "overfit_guard": overfit_guard,
        }
        return json.dumps(payload)

    def set_query_size(self, query_size: Union[int, str], scope: str = "round") -> str:
        if not self._check_permission("set_query_size"):
            raise RuntimeError("PermissionDenied: set_query_size")
        q = int(query_size)
        total_budget = int(self.training_state.get("total_budget") or 0)
        current_labeled = int(self.training_state.get("current_labeled_count") or 0)
        if total_budget <= 0:
            raise ValueError(f"Invalid total_budget in training_state: {total_budget}")
        if current_labeled < 0:
            raise ValueError(
                f"Invalid current_labeled_count in training_state: {current_labeled}"
            )
        remaining = max(0, total_budget - current_labeled)
        if q < AgentConstraints.QUERY_SIZE_MIN:
            raise ValueError(f"query_size {q} < min {AgentConstraints.QUERY_SIZE_MIN}")

        clamped = False
        clamp_reason = None
        if q > remaining:
            if int(remaining) >= int(AgentConstraints.QUERY_SIZE_MIN):
                q = int(remaining)
                clamped = True
                clamp_reason = f"remaining_budget_cap:{int(remaining)}"
            else:
                raise ValueError(f"query_size {q} > remaining_budget {remaining}")

        overfit_guard = {
            "enabled": True,
            "applied": False,
        }

        self.control_state["query_size_round"] = q
        self.control_meta["query_size_round"] = {
            "clamped": bool(clamped),
            "clamp_reason": clamp_reason,
            "overfit_guard": overfit_guard,
        }
        if hasattr(self.controller, "config"):
            self.controller.config.QUERY_SIZE = int(q)
        payload = {
            "status": "success",
            "applied": int(q),
            "scope": scope,
            "remaining_budget": int(remaining) if remaining is not None else None,
            "clamped": bool(clamped),
            "clamp_reason": clamp_reason,
            "overfit_guard": overfit_guard,
        }
        return json.dumps(payload)

    def set_epochs_per_round(
        self, epochs: Union[int, str], scope: str = "round"
    ) -> str:
        if not self._check_permission("set_epochs_per_round"):
            raise RuntimeError("PermissionDenied: set_epochs_per_round")
        e = int(epochs)
        if e < AgentConstraints.EPOCHS_MIN:
            raise ValueError(f"epochs {e} < min {AgentConstraints.EPOCHS_MIN}")
        if e > AgentConstraints.EPOCHS_MAX:
            raise ValueError(f"epochs {e} > max {AgentConstraints.EPOCHS_MAX}")
        self.control_state["epochs_round"] = e
        self.control_meta["epochs_round"] = {"clamped": False, "clamp_reason": None}
        if hasattr(self.controller, "config"):
            self.controller.config.EPOCHS_PER_ROUND = int(e)
        payload = {
            "status": "success",
            "applied": int(e),
            "scope": scope,
            "clamped": False,
            "clamp_reason": None,
        }
        return json.dumps(payload)

    def get_score_distribution(
        self, n_bins: int = 10, quantiles: Optional[List[float]] = None
    ) -> str:
        if not self._check_permission("get_score_distribution"):
            raise RuntimeError("PermissionDenied: get_score_distribution")
        if quantiles is None:
            qts = [0.25, 0.5, 0.75]
        else:
            if not isinstance(quantiles, list) or not quantiles:
                raise TypeError("quantiles must be a non-empty list")
            qts = quantiles
        if not self.current_scores:
            raise RuntimeError(
                "No scores available: call precalculate_scores() before get_score_distribution"
            )
        u_vals: List[float] = []
        k_vals: List[float] = []
        for v in self.current_scores.values():
            u_vals.append(float(v["U"]))
            k_vals.append(float(v["K"]))
        h_vals: List[float] = []
        for u, k in zip(u_vals, k_vals):
            h_vals.append((u + k) * 0.5)
        import numpy as np

        def _stats(arr: List[float]) -> Dict[str, Any]:
            a = np.asarray(arr, dtype=float)
            if a.size == 0:
                return {
                    "min": None,
                    "max": None,
                    "mean": None,
                    "std": None,
                    "quantiles": {},
                }
            qs = {str(p): float(np.quantile(a, p)) for p in qts}
            return {
                "min": float(np.min(a)),
                "max": float(np.max(a)),
                "mean": float(np.mean(a)),
                "std": float(np.std(a)),
                "quantiles": qs,
            }

        def _hist(arr: List[float]) -> Dict[str, Any]:
            a = np.asarray(arr, dtype=float)
            if a.size == 0:
                return {"bins": [], "counts": []}
            counts, bins = np.histogram(a, bins=int(n_bins))
            return {
                "bins": [float(x) for x in bins.tolist()],
                "counts": [int(x) for x in counts.tolist()],
            }

        n_unlabeled = int(len(getattr(self.controller, "unlabeled_indices", []) or []))
        payload = {
            "U_stats": _stats(u_vals),
            "K_stats": _stats(k_vals),
            "H_stats": _stats(h_vals),
            "histograms": {"U": _hist(u_vals), "K": _hist(k_vals), "H": _hist(h_vals)},
            "n_unlabeled": n_unlabeled,
        }
        meta = {
            "schema_version": self.SCHEMA_VERSION,
            "n_bins": int(n_bins),
            "quantiles": qts,
        }
        return self._success_response(payload, meta)

    def _to_numpy(self, value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        dataset = getattr(self.controller, "dataset", None)
        if dataset is None:
            return info
        if idx < 0 or idx >= len(dataset):
            return info
        try:
            item = dataset[idx]
        except (IndexError, KeyError, TypeError) as ex:
            return info

        image = None
        mask = None
        name = None

        if isinstance(item, dict):
            image = item.get("image")
            mask = item.get("mask")
            name = item.get("image_name")
            if mask is not None and hasattr(mask, "numel") and int(mask.numel()) == 0:
                mask = None
        elif isinstance(item, (tuple, list)):
            if len(item) >= 1:
                image = item[0]
            if len(item) >= 2:
                if isinstance(item[1], str):
                    name = item[1]
                else:
                    mask = item[1]
        else:
            image = item

        if name is None and hasattr(dataset, "images"):
            try:
                name = dataset.images[idx]
            except (IndexError, KeyError, TypeError) as ex:
                name = None

        if name is not None:
            info["image_name"] = name

        if hasattr(dataset, "split"):
            info["split"] = dataset.split

        if image is not None:
            img_np = self._to_numpy(image)
            info["image_mean"] = float(np.mean(img_np))
            info["image_std"] = float(np.std(img_np))

        if mask is not None:
            mask_np = self._to_numpy(mask)
            info["mask_positive_ratio"] = float(np.mean(mask_np > 0))

        info["in_labeled_pool"] = idx in self.controller.labeled_indices
        info["in_unlabeled_pool"] = idx in self.controller.unlabeled_indices
        return info

    def get_sample_details(self, sample_id: Union[int, str]) -> str:
        if not self._check_permission("get_sample_details"):
            raise RuntimeError("PermissionDenied: get_sample_details")
        sid = str(sample_id)
        info = {"id": sid}
        if sid in self.current_scores:
            scores = self.current_scores[sid]
            info.update({"U_score": scores["U"], "K_score": scores["K"]})
        try:
            idx = int(sample_id)
        except (ValueError, TypeError) as ex:
            idx = None
        if idx is not None:
            info.update(self._get_sample_metadata(idx))
        info["raw_image_access"] = False
        return json.dumps({"status": "success", "result": info})

    def get_training_status(self) -> str:
        return json.dumps(
            {"status": "success", "result": {"training_status": self.training_state}}
        )

    def set_hyperparameter(self, alpha: Union[float, str]) -> str:
        if not self._check_permission("set_alpha"):
            raise RuntimeError("PermissionDenied: set_alpha")
        alpha_val = float(alpha)
        if alpha_val < AgentConstraints.ALPHA_MIN:
            raise ValueError(f"alpha {alpha_val} < min {AgentConstraints.ALPHA_MIN}")
        if alpha_val > AgentConstraints.ALPHA_MAX:
            raise ValueError(f"alpha {alpha_val} > max {AgentConstraints.ALPHA_MAX}")

        self.alpha = float(alpha_val)
        if hasattr(self.strategy, "alpha"):
            self.strategy.alpha = float(alpha_val)
        if hasattr(self.controller, "config") and hasattr(
            self.controller.config, "ALPHA"
        ):
            self.controller.config.ALPHA = float(alpha_val)
        self.control_state["alpha"] = float(alpha_val)
        self.control_meta["alpha"] = {"clamped": False, "clamp_reason": None}
        return json.dumps(
            {
                "status": "success",
                "alpha": self.alpha,
                "clamped": False,
                "clamp_reason": None,
            }
        )

    def set_training_state(self, state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise TypeError("training_state must be a dict")
        for field in (
            "last_miou",
            "prev_miou",
            "rollback_flag",
            "current_labeled_count",
            "total_budget",
        ):
            if field not in state:
                raise KeyError(f"training_state missing required field: {field}")
        self.training_state = dict(state)
        for key in (
            "overfit_risk",
            "grad_train_val_cos_min",
            "grad_train_val_cos_last",
        ):
            self._append_signal_history(key, self.training_state.get(key))
        try:
            miou_delta = self.training_state.get("miou_delta")
            overfit_risk = self.training_state.get("overfit_risk")
            low_gain = False
            low_gain_thresh = float(
                self._agent_threshold(
                    "MIOU_LOW_GAIN_THRESH",
                    getattr(AgentThresholds, "MIOU_LOW_GAIN_THRESH", 0.0) or 0.0,
                )
                or 0.0
            )
            if miou_delta is not None and float(miou_delta) <= float(low_gain_thresh):
                low_gain = True
            low_risk = False
            risk_lo = float(
                self._agent_threshold(
                    "OVERFIT_RISK_LO",
                    getattr(AgentThresholds, "OVERFIT_RISK_LO", 0.0) or 0.0,
                )
                or 0.0
            )
            if overfit_risk is not None and float(overfit_risk) <= float(risk_lo):
                low_risk = True
            if low_gain and low_risk:
                self._miou_low_gain_streak = int(self._miou_low_gain_streak) + 1
            else:
                self._miou_low_gain_streak = 0
            self.training_state["miou_low_gain_streak"] = self._miou_low_gain_streak
        except Exception:
            self._miou_low_gain_streak = 0
            self.training_state["miou_low_gain_streak"] = 0

    def finalize_selection(
        self, sample_ids: List[str], reason: str, thought: Optional[str] = None
    ) -> str:
        """
        Tool: 提交最终选择
        Args:
            sample_ids: list of strings (sample indices)
            reason: string
            thought: string (Agent的推理过程)
        """
        # 真正更新 Agent 状态
        result: Dict[str, Any] = {
            "status": "error",
            "selected_count": 0,
            "reason_recorded": reason,
        }
        if not hasattr(self.controller, "update"):
            raise RuntimeError("Controller has no update()")
        if hasattr(self.controller, "_selection_context"):
            ctx = {
                "source": "agent",
                "policy": "agent_finalize_selection",
                "sampler_type": getattr(self.controller, "sampler_type", None),
                "experiment_name": getattr(self.controller, "experiment_name", None),
                "current_round": getattr(self.controller, "current_round", None),
                "decision_reason": reason,
                "llm_mode": "mock_no_api_key"
                if str(reason) == "mock_no_api_key"
                else None,
            }
            self.controller._selection_context = ctx
        if str(reason) == "mock_no_api_key" and hasattr(
            self.controller, "_append_trace"
        ):
            self.controller._append_trace(
                {
                    "type": "llm_degraded",
                    "round": int(getattr(self.controller, "current_round", 0) or 0),
                    "degraded": {
                        "source": "agent",
                        "reason_type": "missing_api_key",
                        "policy": "mock_no_api_key",
                    },
                }
            )
        if str(reason) == "mock_no_api_key":
            raise RuntimeError(
                "LLM_API_KEY is missing: agent fallback is forbidden (reason=mock_no_api_key)."
            )
        if not getattr(self.controller, "_last_ranked_items", None) and isinstance(
            self.current_scores, dict
        ):
            items = []
            lambda_t = self.control_state.get("lambda_override_round")
            if lambda_t is None:
                lambda_t = self._default_lambda()
            for sid, scores in self.current_scores.items():
                u = float(scores["U"])
                k_score = float(scores["K"])
                s = (1 - float(lambda_t)) * u + float(lambda_t) * k_score
                try:
                    sid_int = int(sid)
                except Exception:
                    sid_int = str(sid)
                items.append(
                    {
                        "sample_id": sid_int,
                        "final_score": float(s),
                        "uncertainty": u,
                        "knowledge_gain": k_score,
                        "lambda_t": float(lambda_t),
                    }
                )
            if items:
                items.sort(key=lambda x: x["final_score"], reverse=True)
                self.controller._last_ranked_items = items
                # Compute ranking metadata to ensure geometry metrics are populated
                controller_cfg = getattr(self.controller, "config", None)
                query_size = int(getattr(controller_cfg, "QUERY_SIZE", 0) or 0)
                if query_size <= 0:
                    query_size = int(len(sample_ids or []))
                query_size = max(1, query_size)
                
                if hasattr(self.controller, "_compute_ranking_metadata"):
                    base_meta = self.controller._compute_ranking_metadata(items, query_size)
                    if isinstance(base_meta, dict):
                        if not isinstance(getattr(self.controller, "_last_ranking_metadata", None), dict):
                            self.controller._last_ranking_metadata = {}
                        self.controller._last_ranking_metadata.update(base_meta)
        controller_cfg = getattr(self.controller, "config", None)
        expected = int(getattr(controller_cfg, "QUERY_SIZE", 0) or 0)
        if expected <= 0:
            expected = int(len(sample_ids or []))
        expected = max(1, expected)

        def _to_int(x: Any) -> Optional[int]:
            if isinstance(x, int):
                return int(x)
            if isinstance(x, float) and int(x) == x:
                return int(x)
            if isinstance(x, str):
                s = x.strip()
                if s and (s.isdigit() or (s.startswith("-") and s[1:].isdigit())):
                    try:
                        return int(s)
                    except Exception:
                        return None
            return None

        unlabeled = getattr(self.controller, "unlabeled_indices", None)
        unlabeled_int_set = None
        unlabeled_str_set = None
        if isinstance(unlabeled, list):
            ints = []
            strs = []
            for x in unlabeled:
                if isinstance(x, int):
                    ints.append(int(x))
                    strs.append(str(int(x)))
                    continue
                if isinstance(x, str):
                    s = x.strip()
                    if not s:
                        continue
                    strs.append(s)
                    try:
                        ints.append(int(s))
                    except Exception:
                        pass
                    continue
                v = _to_int(x)
                if v is not None:
                    ints.append(int(v))
                    strs.append(str(int(v)))
            unlabeled_int_set = set(ints)
            unlabeled_str_set = set(strs)

        def _dedup_keep_order(xs: List[str]) -> List[str]:
            out: List[str] = []
            seen: set[str] = set()
            for v in xs:
                if v in seen:
                    continue
                seen.add(v)
                out.append(v)
            return out

        def _fallback_ranked_ids() -> List[str]:
            ranked: List[str] = []
            items = getattr(self.controller, "_last_ranked_items", None)
            if isinstance(items, list) and items:
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    sid = _to_int(it.get("sample_id"))
                    if sid is None:
                        continue
                    ranked.append(str(int(sid)))
                ranked = _dedup_keep_order(ranked)
            if ranked:
                return ranked

            cache = (
                self.candidates_cache if isinstance(self.candidates_cache, list) else []
            )
            scored: List[tuple] = []
            for raw in cache:
                if not isinstance(raw, dict):
                    continue
                sid = _to_int(raw.get("id", raw.get("sample_id")))
                if sid is None:
                    continue
                score = raw.get("final_score")
                try:
                    score_f = float(score) if score is not None else float("-inf")
                except Exception:
                    score_f = float("-inf")
                scored.append((score_f, str(int(sid))))
            scored.sort(key=lambda x: (float(x[0]), int(x[1])), reverse=True)
            ranked = [str(sid) for _, sid in scored]
            return _dedup_keep_order(ranked)

        def _is_valid_unlabeled(sid: str) -> bool:
            if unlabeled_int_set is None and unlabeled_str_set is None:
                return True
            if unlabeled_str_set is not None and sid in unlabeled_str_set:
                return True
            if unlabeled_int_set is not None:
                try:
                    return int(sid) in unlabeled_int_set
                except Exception:
                    return False
            return False

        def _repair_to_expected(ids_in: List[Any], *, tag: str) -> List[str]:
            raw_ids: List[str] = []
            for x in list(ids_in or []):
                v = _to_int(x)
                if v is None:
                    continue
                raw_ids.append(str(int(v)))
            raw_ids = _dedup_keep_order(raw_ids)
            valid = [v for v in raw_ids if _is_valid_unlabeled(v)]

            before_n = int(len(valid))
            if before_n < expected:
                ranked = _fallback_ranked_ids()
                chosen = set(valid)
                for v in ranked:
                    if v in chosen:
                        continue
                    if not _is_valid_unlabeled(v):
                        continue
                    valid.append(str(v))
                    chosen.add(str(v))
                    if len(valid) >= expected:
                        break

            if len(valid) < expected and isinstance(unlabeled, list):
                chosen = set(valid)
                for v in unlabeled:
                    if isinstance(v, int):
                        sid = str(int(v))
                    elif isinstance(v, str) and v.strip():
                        sid = v.strip()
                    else:
                        vi = _to_int(v)
                        if vi is None:
                            continue
                        sid = str(int(vi))
                    if sid in chosen:
                        continue
                    valid.append(str(sid))
                    chosen.add(str(sid))
                    if len(valid) >= expected:
                        break

            repaired = valid[:expected]
            if hasattr(self.controller, "_selection_context") and isinstance(
                getattr(self.controller, "_selection_context", None), dict
            ):
                ctx = dict(getattr(self.controller, "_selection_context") or {})
                if before_n < expected:
                    ctx["selection_repair"] = {
                        "applied": True,
                        "tag": str(tag),
                        "expected": int(expected),
                        "valid_before": int(before_n),
                        "valid_after": int(len(repaired)),
                    }
                self.controller._selection_context = ctx
            if hasattr(self.controller, "_append_trace") and before_n < expected:
                try:
                    self.controller._append_trace(
                        {
                            "type": "selection_repair",
                            "round": int(self._current_round()),
                            "tag": str(tag),
                            "expected": int(expected),
                            "valid_before": int(before_n),
                            "valid_after": int(len(repaired)),
                        }
                    )
                except Exception:
                    pass
            return repaired

        repaired_ids = _repair_to_expected(list(sample_ids or []), tag="agent_raw")
        guardrail = self._apply_selection_guardrail([str(x) for x in repaired_ids])
        guarded = guardrail.get("sample_ids") if isinstance(guardrail, dict) else None
        final_ids = _repair_to_expected(
            list(guarded or repaired_ids), tag="after_guardrail"
        )

        update_result = self.controller.update(final_ids)
        if not (
            isinstance(update_result, dict) and update_result.get("status") == "success"
        ):
            raise RuntimeError(f"update_failed: {update_result}")
        result.update(update_result)
        applied: Dict[str, Any] = {
            "lambda": self.control_state.get("lambda_override_round"),
            "query_size": self.control_state.get("query_size_round"),
            "epochs": self.control_state.get("epochs_round"),
            "alpha": self.control_state.get("alpha"),
        }
        violations: Dict[str, Any] = dict(self.control_meta or {})
        result["decision_reason"] = reason
        result["control_applied"] = applied
        result["violations"] = violations
        constraints: Dict[str, Any] = {
            "remaining_budget": int(self.training_state.get("total_budget") or 0)
            - int(self.training_state.get("current_labeled_count") or 0)
            if self.training_state
            else None,
            "epochs_cap": AgentThresholds.EPOCHS_CAP,
        }
        action: Dict[str, Any] = {
            "selected_ids": result.get("selected_ids") or final_ids,
            "query_size": applied.get("query_size"),
            "epochs": applied.get("epochs"),
            "lambda": applied.get("lambda"),
        }
        event: Dict[str, Any] = {
            "type": "controller_step",
            "round": int(getattr(self.controller, "current_round", 0) or 0),
            "state": dict(self.training_state or {}),
            "action": action,
            "constraints": constraints,
            "outcome_ref": {
                "expected": result.get("expected_count"),
                "selected": result.get("selected_count"),
            },
            "reasoning": thought,
        }
        self.controller._append_trace(event)
        return json.dumps(result)

    def reset(self) -> None:
        """
        Reset toolbox state to prevent contamination between experiments.
        Clears cached candidates, scores, and training state.
        """
        self.candidates_cache = []
        self.current_scores = {}
        self.training_state = {}
