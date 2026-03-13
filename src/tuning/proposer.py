from typing import Dict, List, Optional, Any
from .advisor import TuningAdvisor


BASE_CONFIG = {
    "agent_threshold_overrides": {
        "OVERFIT_RISK_HI": 1.2,
        "OVERFIT_RISK_LO": 0.6,
        "OVERFIT_TVC_MIN_HI": 0.8,
        "OVERFIT_RISK_LAMBDA_UP_MAX": 0.9,
        "MIOU_LOW_GAIN_THRESH": 0.01,
        "MIOU_LOW_GAIN_STREAK": 1,
        "LAMBDA_UP_K_U_GAP_MIN": 0.0,
        "LAMBDA_CLAMP_MIN": 0.05,
        "LAMBDA_CLAMP_MAX": 0.80,
        "LAMBDA_DELTA_UP": 0.15,
        "LAMBDA_DELTA_DOWN": 0.10,
        "OVERFIT_RISK_EMA_ALPHA": 0.6,
        "LAMBDA_DOWN_COOLING_ROUNDS": 1,
    },
    "lambda_policy": {
        "mode": "warmup_risk_closed_loop",
        "r1_lambda": 0.0,
        "uncertainty_only_rounds": 2,
        "warmup_start_round": 3,
        "warmup_rounds": 1,
        "warmup_lambda": 0.2,
        "risk_control_start_round": 4,
        "severe_logic": "and",
        "severe_tvc_key": "grad_train_val_cos_last",
        "risk_trigger": "ci",
        "risk_ci_window": 6,
        "risk_ci_quantile": 0.2,
        "risk_ci_min_samples": 3,
        "lambda_smoothing": "ema",
        "lambda_smoothing_alpha": 1.0,
        "lambda_max_step": 0.20,
    },
}


VALID_THRESHOLD_PARAMS = {
    "LAMBDA_DELTA_UP",
    "LAMBDA_DELTA_DOWN",
    "LAMBDA_CLAMP_MIN",
    "LAMBDA_CLAMP_MAX",
    "OVERFIT_RISK_HI",
    "OVERFIT_RISK_LO",
    "OVERFIT_TVC_MIN_HI",
    "OVERFIT_RISK_LAMBDA_UP_MAX",
    "MIOU_LOW_GAIN_THRESH",
    "MIOU_LOW_GAIN_STREAK",
    "LAMBDA_UP_K_U_GAP_MIN",
    "OVERFIT_RISK_EMA_ALPHA",
    "LAMBDA_DOWN_COOLING_ROUNDS",
}

VALID_POLICY_PARAMS = {
    "uncertainty_only_rounds",
    "warmup_start_round",
    "warmup_rounds",
    "warmup_lambda",
    "risk_control_start_round",
    "severe_logic",
    "lambda_smoothing_alpha",
    "lambda_max_step",
    "risk_ci_window",
    "risk_ci_quantile",
}

VALID_STRUCTURE_PARAMS = {
    "epochs_per_round_override",
    "late_stage_ramp",
    "selection_guardrail",
}

PARAM_RANGES = {
    "LAMBDA_DELTA_UP": (0.02, 0.30),
    "LAMBDA_DELTA_DOWN": (0.02, 0.25),
    "LAMBDA_CLAMP_MIN": (0.0, 0.30),
    "LAMBDA_CLAMP_MAX": (0.50, 1.0),
    "OVERFIT_RISK_HI": (0.5, 2.0),
    "OVERFIT_RISK_LO": (0.1, 1.0),
    "lambda_smoothing_alpha": (0.5, 1.0),
    "lambda_max_step": (0.05, 0.35),
    "epochs_per_round_override": (5, 20),
    "uncertainty_only_rounds": (0, 4),
    "warmup_lambda": (0.05, 0.40),
}


class ExperimentProposer:
    def __init__(self, advisor: Optional["TuningAdvisor"] = None):
        self.advisor = advisor

    def propose(
        self, diagnosis: Dict, iteration: int, history: Optional[List[Dict]] = None
    ) -> List[Dict]:
        llm_advice = None
        if self.advisor is not None:
            llm_advice = self.advisor.advise(
                diagnostics=diagnosis.get("diagnostics", {}),
                rule_diagnosis=diagnosis,
                history=history,
            )
        if llm_advice is not None:
            proposals = self._proposals_from_llm_advice(llm_advice, iteration)
            if proposals:
                return self._deduplicate(proposals, history)
        if iteration == 0:
            return self._initial_exploration()
        proposals = self._rule_based_proposals(diagnosis)
        return self._deduplicate(proposals, history)

    def _deduplicate(
        self, proposals: List[Dict], history: Optional[List[Dict]]
    ) -> List[Dict]:
        if not history:
            return proposals
        tried = {
            str(h.get("direction", "")).lower().strip()
            for h in history
            if h.get("direction")
        }
        if not tried:
            return proposals
        filtered = []
        for p in proposals:
            direction = str(p.get("direction", "")).lower().strip()
            if direction and direction in tried:
                print(
                    f"  [proposer] Skipping duplicate direction: {p.get('direction')}"
                )
                continue
            filtered.append(p)
        return filtered if filtered else proposals[:1]

    def _proposals_from_llm_advice(self, advice: Dict, iteration: int) -> List[Dict]:
        proposals = []
        suggestions = advice.get("suggestions", [])
        suggestions.sort(key=lambda s: s.get("priority", 99))
        for i, suggestion in enumerate(suggestions[:6]):
            direction = suggestion.get("direction", f"llm_suggestion_{i}")
            param_changes = suggestion.get("parameter_changes", {})
            validated = self._validate_param_changes(param_changes)
            if not validated:
                continue
            proposal = self._variant(direction, validated)
            proposal["llm_metadata"] = {
                "description": suggestion.get("description", ""),
                "expected_effect": suggestion.get("expected_effect", ""),
                "risk": suggestion.get("risk", "unknown"),
                "priority": suggestion.get("priority", 99),
            }
            proposal["experiment_name"] = (
                f"auto_tune_iter{iteration:02d}_{direction}_{i:02d}"
            )
            proposals.append(proposal)
        branch_rec = advice.get("branch_recommendation", {})
        if branch_rec.get("should_branch"):
            for p in proposals:
                if any(k in p.get("direction", "") for k in ["ramp", "late", "epoch"]):
                    p["resume_strategy"] = {
                        "type": "branch",
                        "branch_round": branch_rec.get("branch_round", 7),
                    }
        return proposals

    def _validate_param_changes(self, changes: Dict) -> Dict:
        validated = {}
        all_valid = (
            VALID_THRESHOLD_PARAMS | VALID_POLICY_PARAMS | VALID_STRUCTURE_PARAMS
        )
        for key, value in changes.items():
            if key not in all_valid:
                continue
            if key in ("late_stage_ramp", "selection_guardrail"):
                if isinstance(value, dict):
                    validated[key] = value
                continue
            if key in PARAM_RANGES:
                lo, hi = PARAM_RANGES[key]
                try:
                    value = max(lo, min(hi, float(value)))
                except (TypeError, ValueError):
                    continue
            validated[key] = value
        return validated

    def _variant(self, direction: str, overrides: Dict) -> Dict:
        config = {"direction": direction}
        config.update(overrides)
        return config

    def _initial_exploration(self) -> List[Dict]:
        return [
            self._variant(
                "ramp_on_best",
                {
                    "late_stage_ramp": {
                        "start_round": 8,
                        "end_round": 13,
                        "start_lambda": 0.35,
                        "end_lambda": 0.70,
                    }
                },
            ),
            self._variant("high_clamp", {"LAMBDA_CLAMP_MAX": 0.92}),
            self._variant(
                "ramp_guard",
                {
                    "late_stage_ramp": {
                        "start_round": 8,
                        "end_round": 13,
                        "start_lambda": 0.35,
                        "end_lambda": 0.65,
                    },
                    "selection_guardrail": {"enabled": True, "u_median_min": 0.40},
                },
            ),
            self._variant("more_epochs", {"epochs_per_round_override": 12}),
        ]

    def _rule_based_proposals(self, diagnosis: Dict) -> List[Dict]:
        issues = diagnosis.get("issues", [])
        proposals = []
        for issue in issues:
            ptype = issue.get("type", "")
            if ptype == "late_stage_plateau":
                proposals.append(
                    self._variant(
                        "ramp_fix",
                        {
                            "late_stage_ramp": {
                                "start_round": 8,
                                "end_round": 13,
                                "start_lambda": 0.35,
                                "end_lambda": 0.70,
                            }
                        },
                    )
                )
            elif ptype == "over_conservative":
                proposals.append(
                    self._variant(
                        "push_lambda",
                        {"LAMBDA_DELTA_UP": 0.20, "LAMBDA_CLAMP_MAX": 0.90},
                    )
                )
            elif ptype == "instability":
                proposals.append(
                    self._variant(
                        "stabilize",
                        {
                            "lambda_smoothing_alpha": 0.85,
                            "LAMBDA_DOWN_COOLING_ROUNDS": 2,
                        },
                    )
                )
            elif ptype == "intra_round_overfit":
                proposals.append(
                    self._variant(
                        "reduce_epochs",
                        {"epochs_per_round_override": 8},
                    )
                )
            elif ptype == "exploration_degradation":
                proposals.append(
                    self._variant(
                        "guardrail_protect",
                        {
                            "selection_guardrail": {
                                "enabled": True,
                                "u_median_min": 0.40,
                            }
                        },
                    )
                )
        if not proposals:
            return self._fine_grid_around_best(diagnosis)
        return proposals[:4]

    def _fine_grid_around_best(self, diagnosis: Dict) -> List[Dict]:
        diag = diagnosis.get("diagnostics", {})
        base_delta_up = BASE_CONFIG["agent_threshold_overrides"]["LAMBDA_DELTA_UP"]
        base_clamp_max = BASE_CONFIG["agent_threshold_overrides"]["LAMBDA_CLAMP_MAX"]
        base_smoothing = BASE_CONFIG["lambda_policy"]["lambda_smoothing_alpha"]
        delta = 0.15
        return [
            self._variant(
                "grid_delta_up_hi",
                {
                    "LAMBDA_DELTA_UP": round(
                        min(
                            base_delta_up * (1 + delta),
                            PARAM_RANGES["LAMBDA_DELTA_UP"][1],
                        ),
                        4,
                    )
                },
            ),
            self._variant(
                "grid_delta_up_lo",
                {
                    "LAMBDA_DELTA_UP": round(
                        max(
                            base_delta_up * (1 - delta),
                            PARAM_RANGES["LAMBDA_DELTA_UP"][0],
                        ),
                        4,
                    )
                },
            ),
            self._variant(
                "grid_clamp_max_hi",
                {
                    "LAMBDA_CLAMP_MAX": round(
                        min(
                            base_clamp_max * (1 + delta),
                            PARAM_RANGES["LAMBDA_CLAMP_MAX"][1],
                        ),
                        4,
                    )
                },
            ),
            self._variant(
                "grid_smoothing_lo",
                {
                    "lambda_smoothing_alpha": round(
                        max(
                            base_smoothing * (1 - delta),
                            PARAM_RANGES["lambda_smoothing_alpha"][0],
                        ),
                        4,
                    )
                },
            ),
        ]
