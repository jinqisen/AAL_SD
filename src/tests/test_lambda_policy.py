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
