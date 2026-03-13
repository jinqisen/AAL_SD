from typing import Dict, List, Optional, Any
from .llm_advisor import TuningLLMClient


PARAMETER_CAUSAL_KNOWLEDGE = """
## AD-KUCS λ 参数因果链

Score(x) = (1-λ)·U(x) + λ·K(x)

### 参数 → λ 的影响路径

| 参数 | 影响路径 | 效果 |
|---|---|---|
| LAMBDA_DELTA_UP | risk_control 阶段低风险时 λ 上调步长 | 越大→λ 上升越快 |
| LAMBDA_DELTA_DOWN | rollback/severe 时 λ 下调步长 | 越大→回撤时 λ 下降越猛 |
| LAMBDA_CLAMP_MIN/MAX | 硬边界 | 限制 λ 活动范围 |
| lambda_smoothing_alpha | EMA 平滑系数 | =1.0 无平滑，<1.0 惰性越大 |
| lambda_max_step | 单轮 λ 最大变化 | 限制突变幅度 |
| OVERFIT_RISK_HI | severe 判定门槛 | 越高→越难触发降 λ |
| OVERFIT_RISK_LO | 允许 λ 上调的风险门槛 | 越高→越难上调 λ |
| LAMBDA_DOWN_COOLING_ROUNDS | 连续下调冷却期 | >0 抑制连续下调 |
| late_stage_ramp | 后期 λ 地板线性抬升 | 强制后期 K 权重 |
| selection_guardrail | U 底线安全阀 | 选中样本 U 偏低时降 λ |
"""


class TuningAdvisor:
    def __init__(self, llm_config: Optional[Dict] = None, enabled: bool = True):
        self.enabled = enabled
        self.client = TuningLLMClient(llm_config) if enabled else None

    def advise(
        self,
        diagnostics: Dict[str, Any],
        rule_diagnosis: Dict[str, Any],
        history: Optional[List[Dict]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled or self.client is None:
            return None
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(diagnostics, rule_diagnosis, history)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw = self.client.chat(messages)
        if raw is None:
            return None
        return self._parse_response(raw)

    def _build_system_prompt(self) -> str:
        return f"""你是 AAL-SD 主动学习系统的参数调优顾问。你的任务：基于实验诊断数据，给出参数调整建议，目标是将 fullmode 实验的 mIoU 推到 0.74 以上。

{PARAMETER_CAUSAL_KNOWLEDGE}

## 输出格式要求

输出一个 JSON 对象，包含以下字段：
{{
  "analysis": "对当前实验结果的深度分析（2-3句话）",
  "primary_bottleneck": "当前阻碍 mIoU 提升的主要瓶颈（一句话）",
  "suggestions": [
    {{
      "direction": "方向名称（英文）",
      "description": "调整描述（中文）",
      "parameter_changes": {{"参数名": 建议值}},
      "expected_effect": "预期效果（一句话）",
      "risk": "low/medium/high",
      "priority": 1
    }}
  ],
  "branch_recommendation": {{
    "should_branch": true/false,
    "branch_round": 7,
    "reason": "原因"
  }},
  "warnings": ["需要注意的风险点"]
}}

## 约束
- suggestions 最多 4 个方向
- parameter_changes 中只能使用已知参数名
- 数值必须在合理范围内
- priority 1=最高优先级
- 如果历史数据显示某个方向已经尝试过且效果不佳，不要重复建议"""

    def _build_user_prompt(
        self, diagnostics: Dict, rule_diagnosis: Dict, history: Optional[List[Dict]]
    ) -> str:
        parts = []
        parts.append("## 当前实验诊断指标\n")
        import json

        parts.append(json.dumps(diagnostics, indent=2, ensure_ascii=False))
        parts.append("\n\n## 规则引擎诊断\n")
        issues = rule_diagnosis.get("issues", [])
        if issues:
            for issue in issues:
                parts.append(
                    f"- [{issue['severity']}] {issue['type']}: {issue['evidence']} → {issue['suggestion']}"
                )
        else:
            parts.append("- 无明显问题（规则引擎未触发任何诊断）")

        if history:
            tried_dirs = set()
            for h in history:
                d = h.get("direction", "")
                if d:
                    tried_dirs.add(d)

            recent_window = min(3, len(history))
            older_window = (
                len(history) - recent_window if len(history) > recent_window else 0
            )

            if older_window > 0:
                parts.append("\n\n## 历史调优记录（摘要）\n")
                for h in history[:older_window]:
                    parts.append(
                        f"Iter {h.get('iteration', '?')}: {h.get('direction', 'unknown')} → mIoU={h.get('best_miou', 'N/A')}"
                    )

            if len(history) > 0:
                parts.append("\n\n## 最近调优记录（详细）\n")
                for h in history[-recent_window:]:
                    parts.append(
                        f"Iter {h.get('iteration', '?')}: best_miou={h.get('best_miou', 'N/A')}, direction={h.get('direction', 'N/A')}"
                    )

            if tried_dirs:
                parts.append(f"\n\n## 已尝试方向\n{', '.join(sorted(tried_dirs))}")
                parts.append("请避免重复建议已尝试的方向。")

        gap = 0.74 - diagnostics.get("final_miou", 0.72)
        parts.append(f"\n\n## 目标\n距离 mIoU 0.74 还差 {gap:.4f}。")
        parts.append("请分析瓶颈并给出最多 4 个参数调整方向。")
        return "\n".join(parts)

    def _parse_response(self, raw: str) -> Optional[Dict]:
        import json
        import re

        text = raw.strip()
        parsed = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            pass
        if parsed is None:
            patterns = [
                r"```json\s*\n(.*?)\n```",
                r"```\s*\n(.*?)\n```",
                r"\{[\s\S]*\}",
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    try:
                        candidate = (
                            match.group(1) if match.lastindex else match.group(0)
                        )
                        parsed = json.loads(candidate)
                        break
                    except (json.JSONDecodeError, IndexError):
                        continue
        if not isinstance(parsed, dict):
            return None
        if not isinstance(parsed.get("suggestions"), list):
            parsed["suggestions"] = []
        if not isinstance(parsed.get("branch_recommendation"), dict):
            parsed["branch_recommendation"] = {}
        if not isinstance(parsed.get("warnings"), list):
            parsed["warnings"] = []
        valid_suggestions = []
        for s in parsed["suggestions"]:
            if not isinstance(s, dict):
                continue
            if "direction" not in s and "parameter_changes" not in s:
                continue
            s.setdefault("direction", "unknown")
            s.setdefault("parameter_changes", {})
            s.setdefault("priority", 99)
            s.setdefault("risk", "unknown")
            valid_suggestions.append(s)
        parsed["suggestions"] = valid_suggestions
        return parsed
