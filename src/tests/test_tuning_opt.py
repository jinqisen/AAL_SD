import os
import sys
import json
import tempfile
from pathlib import Path

# Add src to sys.path
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def test_tuning_llm_config_expands_env_var():
    from tuning_opt.llm_config import TuningLLMConfig

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.json"
        p.write_text(
            json.dumps(
                {
                    "provider": "ppchat_anthropic",
                    "base_url": "https://code.ppchat.vip/v1",
                    "api_key": "${TUNING_LLM_API_KEY}",
                    "model": "claude-opus-4-6",
                }
            ),
            encoding="utf-8",
        )
        old = os.environ.get("TUNING_LLM_API_KEY")
        os.environ["TUNING_LLM_API_KEY"] = "sk-test"
        try:
            cfg = TuningLLMConfig.load(p)
            assert cfg.api_key == "sk-test"
        finally:
            if old is None:
                del os.environ["TUNING_LLM_API_KEY"]
            else:
                os.environ["TUNING_LLM_API_KEY"] = old


def test_parse_final_miou_from_md():
    from tuning_opt.evaluator import parse_final_miou_from_md, parse_last_val_miou_from_md

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.md"
        p.write_text("最终报告 mIoU(test): 0.7189\n", encoding="utf-8")
        v = parse_final_miou_from_md(p)
        assert abs(v - 0.7189) < 1e-9

        p.write_text("最后一轮选模 mIoU(val): 0.7124753599\n", encoding="utf-8")
        vv = parse_last_val_miou_from_md(p)
        assert abs(vv - 0.7124753599) < 1e-9

def test_load_incumbent_objective_uses_val_not_test(monkeypatch):
    from tuning_opt.orchestrator import _load_incumbent_objective
    from unittest.mock import MagicMock
    
    mock_status = MagicMock(return_value=0.55)
    mock_md = MagicMock(return_value=0.50)
    monkeypatch.setattr('tuning_opt.evaluator.parse_objective_from_status', mock_status)
    monkeypatch.setattr('tuning_opt.evaluator.parse_objective_miou_from_md', mock_md)
    
    val = _load_incumbent_objective(Path('/dummy'), 'run1', 'exp1', 'val')
    assert val == 0.55
    mock_status.assert_called_with(Path('/dummy/results/runs/run1/exp1_status.json'), 'val')

def test_acceptance_gate_mean_only():
    from tuning_opt.orchestrator import _acceptance_gate
    ok, res = _acceptance_gate(
        candidate_mean=0.55,
        incumbent_mean=0.50,
        candidate_std=None,
        n_seeds=1,
    )
    assert ok
    assert res["mean_improvement"]
    assert res["min_seed_gate"]
    assert res["guardrail_gate"]

def test_acceptance_gate_rejects_low_min_seed():
    from tuning_opt.orchestrator import _acceptance_gate, AcceptanceConfig
    cfg = AcceptanceConfig(min_seed_epsilon=0.01)
    ok, res = _acceptance_gate(
        candidate_mean=0.60,
        incumbent_mean=0.50,
        candidate_std=0.01,
        n_seeds=3,
        candidate_min=0.40,
        incumbent_min=0.48,
        candidate_guardrail_frac=0.1,
        config=cfg
    )
    assert not ok
    assert not res["min_seed_gate"]
    assert res["mean_improvement"]
    assert res["guardrail_gate"]

def test_acceptance_gate_rejects_guardrail_overload():
    from tuning_opt.orchestrator import _acceptance_gate, AcceptanceConfig
    cfg = AcceptanceConfig(max_guardrail_frac=0.5)
    ok, res = _acceptance_gate(
        candidate_mean=0.60,
        incumbent_mean=0.50,
        candidate_std=0.01,
        n_seeds=3,
        candidate_min=0.48,
        incumbent_min=0.48,
        candidate_guardrail_frac=0.6,
        config=cfg
    )
    assert not ok
    assert not res["guardrail_gate"]
    assert res["mean_improvement"]
    assert res["min_seed_gate"]

def test_load_incumbent_objective_uses_val_not_test(monkeypatch):
    from tuning_opt.orchestrator import _load_incumbent_objective
    from unittest.mock import MagicMock
    
    mock_status = MagicMock(return_value=0.55)
    mock_md = MagicMock(return_value=0.50)
    monkeypatch.setattr('tuning_opt.evaluator.parse_objective_from_status', mock_status)
    monkeypatch.setattr('tuning_opt.evaluator.parse_objective_miou_from_md', mock_md)
    
    val = _load_incumbent_objective(Path('/dummy'), 'run1', 'exp1', 'val')
    assert val == 0.55
    mock_status.assert_called_with(Path('/dummy/results/runs/run1/exp1_status.json'), 'val')

def test_acceptance_gate_mean_only():
    from tuning_opt.orchestrator import _acceptance_gate
    ok, res = _acceptance_gate(
        candidate_mean=0.55,
        incumbent_mean=0.50,
        candidate_std=None,
        n_seeds=1,
    )
    assert ok
    assert res["mean_improvement"]
    assert res["min_seed_gate"]
    assert res["guardrail_gate"]

def test_acceptance_gate_rejects_low_min_seed():
    from tuning_opt.orchestrator import _acceptance_gate, AcceptanceConfig
    cfg = AcceptanceConfig(min_seed_epsilon=0.01)
    ok, res = _acceptance_gate(
        candidate_mean=0.60,
        incumbent_mean=0.50,
        candidate_std=0.01,
        n_seeds=3,
        candidate_min=0.40,
        incumbent_min=0.48,
        candidate_guardrail_frac=0.1,
        config=cfg
    )
    assert not ok
    assert not res["min_seed_gate"]
    assert res["mean_improvement"]
    assert res["guardrail_gate"]

def test_acceptance_gate_rejects_guardrail_overload():
    from tuning_opt.orchestrator import _acceptance_gate, AcceptanceConfig
    cfg = AcceptanceConfig(max_guardrail_frac=0.5)
    ok, res = _acceptance_gate(
        candidate_mean=0.60,
        incumbent_mean=0.50,
        candidate_std=0.01,
        n_seeds=3,
        candidate_min=0.48,
        incumbent_min=0.48,
        candidate_guardrail_frac=0.6,
        config=cfg
    )
    assert not ok
    assert not res["guardrail_gate"]
    assert res["mean_improvement"]
    assert res["min_seed_gate"]

def test_apply_overrides_logs_ignored_keys():
    from tuning_opt.space import ParameterSpace, Param
    space = ParameterSpace([
        Param("a", 0.0, 1.0)
    ])
    res = space.apply_overrides({"a": 0.5, "b": 1.0}, {"b": 2.0, "c": 3.0, "a": 0.8})
    assert res["a"] == 0.8
    assert res["b"] == 1.0  # untouched
    assert "_tuning_ignored_keys" in res
    assert set(res["_tuning_ignored_keys"]) == {"b", "c"}

def test_llm_prompt_keys_subset_of_space():
    from tuning_opt.proposer import _SYSTEM_PROMPT
    from tuning_opt.space import ParameterSpace
    
    space = ParameterSpace.default()
    valid_keys = set(p.key for p in space.params)
    
    # Extract keys from prompt looking for bullet points starting with '  - '
    import re
    prompt_keys = set()
    for line in _SYSTEM_PROMPT.split('\n'):
        m = re.match(r'^\s+-\s+([a-zA-Z0-9_\.]+)', line)
        if m:
            prompt_keys.add(m.group(1))
            
    # Remove 'epochs_per_round_override' from expectation if it was removed in prompt
    # and LAMBDA_CLAMP_MIN/MAX as well.
    # We just want to ensure that NO prompt key is OUTSIDE the space (unless explicitly known like direction etc)
    # The prompt bullet points are precisely the parameter_changes keys
    # ensure prompt_keys is a subset of valid_keys
    unknown = prompt_keys - valid_keys
    assert not unknown, f"Found keys in LLM prompt not in ParameterSpace: {unknown}"

def test_default_llm_prompt_targets_val():
    from tuning_opt.proposer import _SYSTEM_PROMPT

    assert "最终 val mIoU" in _SYSTEM_PROMPT

def test_stage_aware_scales_generate_stage_deltas():
    from tuning_opt.space import ParameterSpace
    
    space = ParameterSpace.default()
    base_cfg = {
        "agent_threshold_overrides": {
            "LAMBDA_DELTA_UP": 0.20,
            "LAMBDA_DELTA_DOWN": 0.15
        }
    }
    
    overrides = {
        "stage_aware_scales.early_up_scale": 1.5,
        "stage_aware_scales.late_down_scale": 2.0,
        "stage_aware_scales.late_up_scale": 0.5
    }
    
    merged = space.apply_overrides(base_cfg, overrides)

    policy = merged.get("lambda_policy", {})
    assert isinstance(policy, dict)
    assert policy.get("stage_aware") is True
    assert "stage_aware_scales" not in merged

    assert policy.get("stage_boundaries") == [5, 10]
    stage_deltas = policy.get("stage_deltas", {})
    assert "early" in stage_deltas
    assert "mid" in stage_deltas
    assert "late" in stage_deltas
    
    # Base UP = 0.20, Base DOWN = 0.15
    assert stage_deltas["early"]["delta_up"] == 0.20 * 1.5
    assert stage_deltas["early"]["delta_down"] == 0.15
    assert stage_deltas["mid"]["delta_up"] == 0.20
    assert stage_deltas["mid"]["delta_down"] == 0.15
    assert stage_deltas["late"]["delta_up"] == 0.20 * 0.5
    assert stage_deltas["late"]["delta_down"] == 0.15 * 2.0

def test_load_tuning_program_config():
    from tuning_opt.space import ParameterSpace
    
    cfg = {
        "version": "1.0",
        "parameter_space": [
            {"key": "lambda_policy.lambda_max_step", "lo": 0.1, "hi": 0.5}
        ],
        "acceptance_gate": {
            "max_guardrail_frac": 0.2
        }
    }
    
    space = ParameterSpace.from_config(cfg)
    assert len(space.params) == 1
    assert space.params[0].key == "lambda_policy.lambda_max_step"
    assert space.params[0].lo == 0.1
    assert space.params[0].hi == 0.5
    
    # Test fallback
    space2 = ParameterSpace.from_config({})
    assert len(space2.params) > 10  # default params
