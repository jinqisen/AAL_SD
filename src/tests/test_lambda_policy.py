import types

from src.agent.toolbox import Toolbox
from src.main import ActiveLearningPipeline


class _Cfg:
    RANDOM_SEED = 42


class _Controller:
    def __init__(self):
        self.config = _Cfg()
        self.current_round = 2
        self.seed = 42
        self.exp_config = {}


def test_data_isolation():
    labeled = list(range(1, 41))
    train_ids, probe_ids = ActiveLearningPipeline._split_labeled_indices_for_grad_probe(
        labeled, frac=0.10, min_count=8, seed=42
    )
    assert set(train_ids).isdisjoint(set(probe_ids))
    assert sorted(train_ids + probe_ids) == sorted(labeled)
    assert len(probe_ids) >= 8


def test_stage_aware_deltas():
    controller = _Controller()
    tools = Toolbox(controller, types.SimpleNamespace(), model=None)
    tools._last_lambda_applied = 0.5
    tools.training_state = {
        "last_miou": 0.7,
        "prev_miou": 0.7,
        "miou_delta": 0.0,
        "rollback_flag": False,
        "current_labeled_count": 10,
        "total_budget": 100,
        "overfit_risk": 0.1,
        "grad_train_val_cos_min": 0.0,
        "grad_train_val_cos_last": 0.0,
    }
    policy = {
        "mode": "warmup_risk_closed_loop",
        "uncertainty_only_rounds": 0,
        "warmup_rounds": 0,
        "risk_control_start_round": 1,
        "stage_aware": True,
        "stage_boundaries": [5, 10],
        "stage_deltas": {
            "early": {"delta_up": 0.1, "delta_down": 0.05},
            "mid": {"delta_up": 0.05, "delta_down": 0.1},
            "late": {"delta_up": 0.02, "delta_down": 0.15},
        },
    }
    payload = tools._compute_policy_lambda_for_round(2, policy)
    diag = payload.get("diagnostics") or {}
    thresholds = (diag.get("thresholds") or {}) if isinstance(diag, dict) else {}
    assert float(thresholds.get("lambda_delta_up")) == 0.1
    assert float(thresholds.get("lambda_delta_down")) == 0.05
    stage = (diag.get("stage_aware") or {}) if isinstance(diag, dict) else {}
    assert stage.get("enabled") is True
    assert stage.get("stage") == "early"


def test_u_adaptive_delta():
    controller = _Controller()
    tools = Toolbox(controller, types.SimpleNamespace(), model=None)
    tools._last_lambda_applied = 0.5
    tools.training_state = {
        "last_miou": 0.7,
        "prev_miou": 0.7,
        "miou_delta": 0.0,
        "rollback_flag": False,
        "current_labeled_count": 10,
        "total_budget": 100,
        "overfit_risk": 10.0,
        "grad_train_val_cos_min": -10.0,
        "grad_train_val_cos_last": -10.0,
        "train_u_median_history": [(1, 0.9), (2, 0.5)],
        "train_k_median_history": [(1, 0.1), (2, 0.1)],
        "max_history_length": 5,
    }
    policy = {
        "mode": "warmup_risk_closed_loop",
        "uncertainty_only_rounds": 0,
        "warmup_rounds": 0,
        "risk_control_start_round": 1,
        "u_adaptive": True,
        "u_delta_down_threshold": -0.3,
        "u_delta_up_threshold": 0.1,
    }
    payload = tools._compute_policy_lambda_for_round(2, policy)
    assert str(payload.get("rule", "")).startswith(
        "u_quality_degradation_aggressive_down"
    )
    diag = payload.get("diagnostics") or {}
    u_ad = (diag.get("u_adaptive") or {}) if isinstance(diag, dict) else {}
    assert u_ad.get("enabled") is True
    assert u_ad.get("triggered") is True
    assert float(u_ad.get("applied_delta")) < 0.0


def test_geometry_controller_scales_step_up_and_applies_progress_cap():
    controller = _Controller()
    tools = Toolbox(controller, types.SimpleNamespace(), model=None)
    tools._last_lambda_applied = 0.2
    tools.training_state = {
        "last_miou": 0.7,
        "prev_miou": 0.7,
        "miou_delta": 0.0,
        "rollback_flag": False,
        "current_labeled_count": 20,
        "total_budget": 100,
        "overfit_risk": 0.1,
        "grad_train_val_cos_min": 0.0,
        "grad_train_val_cos_last": 0.0,
        "selection_geometry": {
            "sens_up": 0.04,
            "sens_down": 0.02,
            "asymmetry_ratio": 2.0,
        },
    }
    policy = {
        "mode": "warmup_risk_closed_loop",
        "uncertainty_only_rounds": 0,
        "warmup_rounds": 0,
        "risk_control_start_round": 1,
        "geometry_controller": {
            "enabled": True,
            "sens_up_threshold": 0.1,
            "step_up": 0.08,
            "step_down": 0.12,
            "progressive_caps": [
                {"max_progress": 0.3, "lambda_max": 0.23},
                {"max_progress": 1.0, "lambda_max": 0.6},
            ],
        },
    }
    payload = tools._compute_policy_lambda_for_round(2, policy)
    assert float(payload.get("applied")) == 0.23
    assert payload.get("rule") == "geometry_safe_up_capped"
    diag = payload.get("diagnostics") or {}
    geom = (diag.get("geometry_controller") or {}) if isinstance(diag, dict) else {}
    assert geom.get("direction") == "up"
    assert float(geom.get("step_up_effective")) == 0.04
    assert float(geom.get("lambda_cap")) == 0.23


def test_geometry_controller_reduces_lambda_when_sens_up_is_high():
    controller = _Controller()
    tools = Toolbox(controller, types.SimpleNamespace(), model=None)
    tools._last_lambda_applied = 0.35
    tools.training_state = {
        "last_miou": 0.7,
        "prev_miou": 0.7,
        "miou_delta": 0.0,
        "rollback_flag": False,
        "current_labeled_count": 40,
        "total_budget": 100,
        "overfit_risk": 0.1,
        "grad_train_val_cos_min": 0.0,
        "grad_train_val_cos_last": 0.0,
        "selection_geometry": {
            "sens_up": 0.18,
            "sens_down": 0.05,
            "asymmetry_ratio": 3.6,
        },
    }
    policy = {
        "mode": "warmup_risk_closed_loop",
        "uncertainty_only_rounds": 0,
        "warmup_rounds": 0,
        "risk_control_start_round": 1,
        "geometry_controller": {
            "enabled": True,
            "sens_up_threshold": 0.1,
            "step_up": 0.08,
            "step_down": 0.12,
        },
    }
    payload = tools._compute_policy_lambda_for_round(2, policy)
    assert abs(float(payload.get("applied")) - 0.23) < 1e-12
    assert payload.get("rule") == "geometry_sensitive_down"
