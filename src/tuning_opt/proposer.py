from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .llm_client import ChatMessage, TuningLLMClient


@dataclass(frozen=True)
class Proposal:
    direction: str
    parameter_changes: Dict[str, Any]
    description: str = ""
    priority: int = 99
    risk: str = ""
    expected_gain: float = 0.0
    constraints: Optional[Dict[str, Any]] = None


_SYSTEM_PROMPT = """你是 AAL-SD 自动调参助手。你只能输出严格 JSON，不能输出多余文本。
目标：提高最终 val mIoU，同时保持训练稳定，避免过拟合风险升高。
输出最多 6 个建议，每条建议包含：
- direction: 简短方向名（字符串）
- priority: 越小越优先（整数）
- description: 一句话说明（字符串）
- risk: 该方向的潜在风险描述（字符串，例如"可能导致过拟合"）
- expected_gain: 预期 mIoU 提升量（浮点数，例如 0.005）
- constraints: 硬约束触发条件（对象，可为空 {}，例如 {"overfit_risk_max": 1.5}）
- parameter_changes: 参数改动（对象），只允许以下 key（可以是子集）：
  - agent_threshold_overrides.LAMBDA_DELTA_UP
  - agent_threshold_overrides.LAMBDA_DELTA_DOWN
  - agent_threshold_overrides.OVERFIT_RISK_HI
  - agent_threshold_overrides.OVERFIT_RISK_LO
  - agent_threshold_overrides.OVERFIT_TVC_MIN_HI
  - agent_threshold_overrides.OVERFIT_RISK_EMA_ALPHA
  - agent_threshold_overrides.LAMBDA_DOWN_COOLING_ROUNDS
  - lambda_policy.lambda_smoothing_alpha
  - lambda_policy.lambda_max_step
  - lambda_policy.risk_ci_window
  - lambda_policy.risk_ci_quantile
  - lambda_policy.late_stage_ramp.start_round
  - lambda_policy.late_stage_ramp.end_round
  - lambda_policy.late_stage_ramp.start_lambda
  - lambda_policy.late_stage_ramp.end_lambda
  - lambda_policy.selection_guardrail.u_median_min
  - lambda_policy.selection_guardrail.u_low_thresh
  - lambda_policy.selection_guardrail.u_low_frac_max
  - lambda_policy.selection_guardrail.lambda_step_down
数值必须是数字（int/float）。"""


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    s = (text or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


class LLMProposer:
    def __init__(self, client: TuningLLMClient, system_prompt: Optional[str] = None):
        self.client = client
        self.system_prompt = system_prompt or _SYSTEM_PROMPT

    def propose(self, *, context: Dict[str, Any]) -> List[Proposal]:
        if not self.client.is_available():
            return []
        user_payload = json.dumps(context, ensure_ascii=False, indent=2)
        content = self.client.chat(
            messages=[
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=user_payload),
            ]
        )
        parsed = _try_parse_json(content)
        if not parsed:
            return []
        suggestions = parsed.get("suggestions")
        if not isinstance(suggestions, list):
            return []
        out: List[Proposal] = []
        for s in suggestions[:6]:
            if not isinstance(s, dict):
                continue
            direction = str(s.get("direction", "") or "").strip() or "unknown"
            desc = str(s.get("description", "") or "").strip()
            risk = str(s.get("risk", "") or "").strip()
            try:
                priority = int(s.get("priority", 99) or 99)
            except Exception:
                priority = 99
            try:
                expected_gain = float(s.get("expected_gain", 0.0) or 0.0)
            except Exception:
                expected_gain = 0.0
            constraints = s.get("constraints")
            if not isinstance(constraints, dict):
                constraints = {}
            changes = s.get("parameter_changes")
            if not isinstance(changes, dict):
                changes = {}
            out.append(
                Proposal(
                    direction=direction,
                    parameter_changes=dict(changes),
                    description=desc,
                    priority=priority,
                    risk=risk,
                    expected_gain=expected_gain,
                    constraints=dict(constraints),
                )
            )
        out.sort(key=lambda p: p.priority)
        return out
