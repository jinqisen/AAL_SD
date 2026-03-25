from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm_config import TuningLLMConfig


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class TuningLLMClient:
    def __init__(self, config: TuningLLMConfig):
        self.config = config

    def is_available(self) -> bool:
        return bool(self.config.base_url and self.config.model and self.config.api_key)

    def chat(self, *, messages: List[ChatMessage]) -> str:
        if not self.is_available():
            raise RuntimeError("LLM config is missing base_url/model/api_key")

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": float(self.config.temperature),
        }
        if self.config.max_tokens is not None:
            payload["max_tokens"] = int(self.config.max_tokens)
        if self.config.thinking_budget is not None:
            payload["thinking_budget"] = int(self.config.thinking_budget)

        url = f"{self.config.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        last_err: Optional[Exception] = None
        for attempt in range(int(self.config.max_retries) + 1):
            try:
                t0 = time.time()
                resp = _http_post_json(
                    url,
                    headers=headers,
                    payload=payload,
                    timeout_seconds=int(self.config.timeout),
                )
                dt = time.time() - t0
                if self.config.log_requests:
                    self._log_call(payload=payload, response=resp, latency_s=dt)
                return _extract_chat_content(resp)
            except Exception as e:
                last_err = e
                if attempt >= int(self.config.max_retries):
                    break
                delay = float(self.config.retry_delay) * (float(self.config.retry_backoff) ** attempt)
                time.sleep(delay)
        raise RuntimeError(f"LLM call failed after retries: {last_err}") from last_err

    def _log_call(self, *, payload: Dict[str, Any], response: Dict[str, Any], latency_s: float) -> None:
        try:
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = log_dir / f"tuning_llm_{ts}_{int(time.time() * 1000)}.json"
            path.write_text(
                json.dumps(
                    {"request": payload, "response": response, "latency_s": latency_s},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            return


def _http_post_json(url: str, *, headers: Dict[str, str], payload: Dict[str, Any], timeout_seconds: int) -> Dict[str, Any]:
    try:
        import requests

        r = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
        r.raise_for_status()
        obj = r.json()
        return obj if isinstance(obj, dict) else {"raw": obj}
    except ImportError:
        import urllib.request

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {"raw": obj}


def _extract_chat_content(obj: Dict[str, Any]) -> str:
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                return content
    raise RuntimeError(f"Unexpected LLM response shape: keys={sorted(obj.keys())}")

