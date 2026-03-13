from typing import Dict, List, Optional, Tuple
import os
import json
import logging
import time
import requests

logger = logging.getLogger(__name__)


class TuningLLMClient:
    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = self._default_config()
        raw_key = config.get("api_key", "")
        self.api_key = os.path.expandvars(raw_key) if raw_key else ""
        if not self.api_key or self.api_key.startswith("${"):
            self.api_key = os.environ.get("TUNING_LLM_API_KEY", "")
        self.base_url = (config.get("base_url", "") or "").rstrip("/")
        self.model = config.get("model", "claude-opus-4-6")
        self.temperature = float(config.get("temperature", 0.3))
        self.timeout = int(config.get("timeout", 120))
        self.max_retries = int(config.get("max_retries", 3))
        self.retry_delay = float(config.get("retry_delay", 5.0))
        self.retry_backoff = float(config.get("retry_backoff", 2.0))
        self.thinking_budget = config.get("thinking_budget")
        self.log_requests = bool(config.get("log_requests", False))
        self.log_dir = config.get("log_dir", "results/tuning_llm_logs")

    @staticmethod
    def _default_config() -> Dict:
        return {
            "base_url": os.environ.get(
                "TUNING_LLM_BASE_URL", "https://code.ppchat.vip/v1"
            ),
            "api_key": os.environ.get("TUNING_LLM_API_KEY", ""),
            "model": os.environ.get("TUNING_LLM_MODEL", "claude-opus-4-6"),
            "temperature": 0.3,
            "timeout": 120,
            "max_retries": 3,
            "retry_delay": 5.0,
            "retry_backoff": 2.0,
        }

    def chat(self, messages: List[Dict[str, str]]) -> Optional[str]:
        if not self.api_key:
            return None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
            "max_tokens": 8192,
        }
        if self.thinking_budget:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": int(self.thinking_budget),
            }
        endpoint = f"{self.base_url}/chat/completions"
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    endpoint, headers=headers, json=payload, timeout=self.timeout
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    if self.log_requests:
                        self._write_log(messages, content)
                    return content
                if resp.status_code in (429, 500, 502, 503, 504):
                    logger.warning(
                        "LLM call attempt %d/%d got status %d",
                        attempt + 1,
                        self.max_retries + 1,
                        resp.status_code,
                    )
                    if attempt < self.max_retries:
                        delay = self.retry_delay * (self.retry_backoff**attempt)
                        time.sleep(delay)
                        continue
                logger.error("LLM call failed with status %d", resp.status_code)
                return None
            except Exception as exc:
                logger.warning(
                    "LLM call attempt %d/%d failed: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                if attempt < self.max_retries:
                    delay = self.retry_delay * (self.retry_backoff**attempt)
                    time.sleep(delay)
                    continue
                logger.error("LLM call exhausted all retries")
                return None
        return None

    def _write_log(self, messages: List[Dict], response: str) -> None:
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(self.log_dir, f"llm_call_{ts}.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"timestamp": ts, "messages": messages, "response": response},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            pass
