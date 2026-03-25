from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env_vars(value: str) -> str:
    def repl(m: re.Match[str]) -> str:
        return os.environ.get(m.group(1), "")

    return _ENV_VAR_PATTERN.sub(repl, value)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


@dataclass(frozen=True)
class TuningLLMConfig:
    provider: str
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.3
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 5.0
    retry_backoff: float = 2.0
    thinking_budget: Optional[int] = None
    max_tokens: Optional[int] = None
    log_requests: bool = False
    log_dir: str = "results/tuning_llm_logs"

    @staticmethod
    def default_path(repo_root: Path) -> Path:
        return repo_root / "src" / "tuning_llm_config.json"

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> "TuningLLMConfig":
        provider = str(m.get("provider", "") or "").strip()
        base_url = str(m.get("base_url", "") or "").strip().rstrip("/")
        model = str(m.get("model", "") or "").strip()
        raw_key = str(m.get("api_key", "") or "").strip()
        api_key = _expand_env_vars(raw_key) if raw_key else ""
        if not api_key:
            api_key = str(os.environ.get("TUNING_LLM_API_KEY", "") or "").strip()

        thinking_budget = m.get("thinking_budget")
        if thinking_budget is not None:
            try:
                thinking_budget = int(thinking_budget)
            except Exception:
                thinking_budget = None

        max_tokens = m.get("max_tokens")
        if max_tokens is not None:
            try:
                max_tokens = int(max_tokens)
            except Exception:
                max_tokens = None

        return cls(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=float(m.get("temperature", 0.3) or 0.3),
            timeout=int(m.get("timeout", 120) or 120),
            max_retries=int(m.get("max_retries", 3) or 3),
            retry_delay=float(m.get("retry_delay", 5.0) or 5.0),
            retry_backoff=float(m.get("retry_backoff", 2.0) or 2.0),
            thinking_budget=thinking_budget,
            max_tokens=max_tokens,
            log_requests=bool(m.get("log_requests", False)),
            log_dir=str(m.get("log_dir", "results/tuning_llm_logs") or "results/tuning_llm_logs"),
        )

    @classmethod
    def load(cls, path: Path) -> "TuningLLMConfig":
        payload = _load_json(path)
        return cls.from_mapping(payload)

