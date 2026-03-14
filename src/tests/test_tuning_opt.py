import os
import json
import tempfile
from pathlib import Path


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
