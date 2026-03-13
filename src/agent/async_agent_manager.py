import asyncio
import json
import logging
import os
import hashlib
import requests
import re
import time
from typing import Dict, List, Optional, Any, Tuple, Callable

from .prompt_template import PromptBuilder
from .toolbox import Toolbox
from .utils import (
    is_llm_transport_error,
    validate_response,
    extract_thought,
    extract_action_dict
)

# Setup logger
logger = logging.getLogger(__name__)

class AsyncSiliconFlowClient:
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float = 0.0, timeout: int = 60):
        self.api_key = api_key
        self.base_url = (base_url or "").rstrip('/')
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    async def chat_async(self, messages: List[Dict[str, str]]) -> str:
        """
        Interacts with the SiliconFlow API asynchronously (using thread pool for requests).
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
            "stop": ["\nObservation:", "\nObservation："]
        }
        
        def _make_request():
            if not self.api_key:
                # Mock response logic for missing API key (simplified from sync client)
                # In async mode we assume valid config, or replicate the mock logic if needed.
                # For brevity/cleanliness, we return a standard error or mock if key is missing.
                return "Error calling API: Missing API Key"
                
            endpoint = f"{self.base_url}/chat/completions"
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    data = response.json()
                    return data['choices'][0]['message']['content']
                else:
                    return f"Error: API returned status code {response.status_code}. Response: {response.text}"
            except Exception as e:
                return f"Error calling API: {str(e)}"

        # Run blocking request in a separate thread
        return await asyncio.to_thread(_make_request)


class AsyncAgentManager:
    def __init__(
        self,
        tools: Toolbox,
        client: AsyncSiliconFlowClient,
        verbose: bool = True,
        llm_max_retries: int = 3,
        llm_retry_base_seconds: float = 5.0,
        llm_retry_backoff: float = 2.0,
        llm_retry_max_seconds: float = 60.0,
    ):
        """
        Async LLM Agent Controller that manages the ReAct loop.
        """
        self.tools = tools
        self.client = client
        self.verbose = verbose
        self.max_steps = 10
        self.history: List[Dict[str, str]] = []
        self.prompt_builder = PromptBuilder()
        self.llm_max_retries = int(llm_max_retries or 0)
        self.llm_retry_base_seconds = float(llm_retry_base_seconds or 0.0)
        self.llm_retry_backoff = float(llm_retry_backoff or 1.0)
        self.llm_retry_max_seconds = float(llm_retry_max_seconds or 0.0)
        try:
            self.session_id = f"{int(time.time() * 1000)}-{int(os.getpid())}"
        except Exception:
            self.session_id = None

    async def run_cycle_async(self) -> Dict[str, Any]:
        """
        Executes one Active Learning selection cycle asynchronously.
        """
        # 1. Pre-calculate scores (run in thread if CPU intensive)
        if hasattr(self.tools, "precalculate_scores"):
            await asyncio.to_thread(self.tools.precalculate_scores)

        # 2. Initialize conversation
        # Accessing attributes is fast, no need for async
        controller = getattr(self.tools, "controller", None)
        labeled_size = len(getattr(controller, "labeled_indices", []) or []) if controller else None
        unlabeled_size = len(getattr(controller, "unlabeled_indices", []) or []) if controller else None
        
        query_size = None
        if controller and hasattr(controller, "config"):
            query_size = getattr(controller.config, "QUERY_SIZE", None)
            
        user_prompt = self.prompt_builder.build_user_prompt(labeled_size, unlabeled_size, query_size)

        status_payload = None
        try:
            status_json = await asyncio.to_thread(self.tools.get_system_status)
            status_payload = json.loads(status_json)
        except Exception:
            status_payload = None
            
        total_iterations = None
        current_iteration = None
        lambda_t = None
        last_miou = None
        rollback_threshold = None
        rollback_mode = None
        k_definition = None
        
        if isinstance(status_payload, dict):
            result = status_payload.get("result", {})
            if isinstance(result, dict):
                total_iterations = result.get("total_budget")
                current_iteration = result.get("current_labeled_count")
                lambda_t = result.get("lambda_t")
                last_miou = result.get("last_miou")
                rollback_threshold = result.get("rollback_threshold")
                rollback_mode = result.get("rollback_mode")
                k_definition = result.get("k_definition")

        self.history = [
            {
                "role": "system",
                "content": self.prompt_builder.build_system_prompt(
                    total_iterations=total_iterations,
                    current_iteration=current_iteration,
                    last_miou=last_miou,
                    lambda_t=lambda_t,
                    rollback_threshold=rollback_threshold,
                    rollback_mode=rollback_mode,
                    k_definition=k_definition,
                    control_permissions=getattr(self.tools, "control_permissions", None)
                )
            },
            {"role": "user", "content": user_prompt}
        ]

        step = 0
        while step < self.max_steps:
            if self.verbose:
                print(f"\n--- Step {step + 1} ---")
            self._current_step = int(step + 1)

            response_text = await self._call_llm_with_retries_async(self.history, validator=validate_response)

            if response_text is None or str(response_text).strip() == "":
                return {
                    "status": "error",
                    "error_type": "LLMEmptyResponse",
                    "message": "LLM returned empty response"
                }

            if is_llm_transport_error(response_text):
                return {
                    "status": "error",
                    "error_type": "LLMAPIError",
                    "message": str(response_text)
                }
            
            # Clean up response
            if "Observation:" in response_text:
                response_text = response_text.split("Observation:")[0].strip()
            if "Observation：" in response_text:
                response_text = response_text.split("Observation：")[0].strip()
            
            if self.verbose:
                print(f"LLM Response:\n{response_text}")

            self.history.append({"role": "assistant", "content": response_text})

            # 4. Parse Action
            observation, done = await self._handle_action_async(response_text)

            if self.verbose:
                print(f"Observation: {observation}")

            self.history.append({"role": "user", "content": f"Observation: {observation}"})
            
            try:
                payload = json.loads(observation)
            except Exception:
                payload = None
                
            if isinstance(payload, dict) and payload.get("error_type") == "InvalidAction":
                self.history.append({
                    "role": "user",
                    "content": "请严格按 ReAct 格式重试：只输出两行 Thought 与 Action。Action 行必须以 'Action:' 开头，后面紧跟单个 JSON 对象（不要代码块、不要多余文本、不要输出 Observation）。"
                })
                
            if done:
                try:
                    return json.loads(observation)
                except Exception:
                    return {"status": "success"}
            step += 1

        return {"status": "failed", "reason": "Max steps reached"}

    def _trace_thought_enabled(self) -> bool:
        controller = getattr(self.tools, "controller", None)
        cfg = getattr(controller, "config", None) if controller is not None else None
        if cfg is not None:
            try:
                return bool(getattr(cfg, "TRACE_AGENT_THOUGHT", False))
            except Exception:
                pass
        text = str(os.getenv("AAL_SD_TRACE_AGENT_THOUGHT", "") or "").strip().lower()
        return text in ("1", "true", "yes", "y", "on")

    def _trace_prompt_enabled(self) -> bool:
        controller = getattr(self.tools, "controller", None)
        cfg = getattr(controller, "config", None) if controller is not None else None
        if cfg is not None:
            try:
                return bool(getattr(cfg, "TRACE_AGENT_PROMPT", False))
            except Exception:
                pass
        try:
            exp_cfg = getattr(controller, "exp_config", None) if controller is not None else None
            if isinstance(exp_cfg, dict) and bool(exp_cfg.get("enable_agent_prompt_logging")):
                return True
        except Exception:
            pass
        try:
            runtime = getattr(controller, "experiment_runtime", None) if controller is not None else None
            opts = getattr(runtime, "trace_options", None) if runtime is not None else None
            if opts is not None and bool(getattr(opts, "enable_agent_prompt_logging", False)):
                return True
        except Exception:
            pass
        text = str(os.getenv("AAL_SD_TRACE_AGENT_PROMPT", "") or "").strip().lower()
        return text in ("1", "true", "yes", "y", "on")

    def _redact_text(self, text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        s = str(text)
        s = re.sub(r"Bearer\s+[A-Za-z0-9\-\._~\+\/]+=*", "Bearer [REDACTED]", s)
        s = re.sub(r"\bsk-[A-Za-z0-9]{10,}\b", "sk-[REDACTED]", s)
        return s

    def _sha1(self, text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        try:
            return hashlib.sha1(str(text).encode("utf-8", errors="ignore")).hexdigest()
        except Exception:
            return None

    def _safe_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        safe = []
        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            safe.append(
                {
                    "role": msg.get("role"),
                    "content": self._redact_text(msg.get("content")),
                }
            )
        return safe

    async def _call_llm_with_retries_async(self, messages: List[Dict[str, str]], validator: Optional[Callable[[str], bool]] = None) -> str:
        attempts = max(1, 1 + self.llm_max_retries)
        last_text = None
        
        for attempt in range(attempts):
            if (
                self._trace_prompt_enabled()
                and hasattr(self.tools, "controller")
                and hasattr(self.tools.controller, "_append_trace")
            ):
                try:
                    controller = self.tools.controller
                    round_num = int(getattr(controller, "current_round", 0) or 0)
                    step_num = int(getattr(self, "_current_step", 0) or 0)
                    safe_messages = self._safe_messages(messages or [])
                    self.tools.controller._append_trace(
                        {
                            "type": "agent_llm_request",
                            "session_id": self.session_id,
                            "round": round_num,
                            "step": step_num,
                            "attempt": int(attempt + 1),
                            "messages": safe_messages,
                            "messages_sha1": self._sha1(
                                json.dumps(safe_messages, ensure_ascii=False)
                            ),
                        }
                    )
                except Exception:
                    pass
            text = await self.client.chat_async(messages)
            last_text = text
            if (
                self._trace_prompt_enabled()
                and hasattr(self.tools, "controller")
                and hasattr(self.tools.controller, "_append_trace")
            ):
                try:
                    controller = self.tools.controller
                    round_num = int(getattr(controller, "current_round", 0) or 0)
                    step_num = int(getattr(self, "_current_step", 0) or 0)
                    safe_text = self._redact_text(text)
                    extracted = self._redact_text(extract_thought(text or ""))
                    self.tools.controller._append_trace(
                        {
                            "type": "agent_llm_response",
                            "session_id": self.session_id,
                            "round": round_num,
                            "step": step_num,
                            "attempt": int(attempt + 1),
                            "response": safe_text,
                            "response_sha1": self._sha1(safe_text),
                            "thought": extracted,
                            "thought_sha1": self._sha1(extracted),
                        }
                    )
                except Exception:
                    pass
            
            # 1. Transport Error Check
            if is_llm_transport_error(text):
                if attempt >= attempts - 1:
                    break
                delay = self._compute_retry_delay(attempt)
                if self.verbose:
                    print(f"LLM transport error. Retry {attempt + 1}/{attempts - 1} after {delay:.1f}s")
                if delay > 0:
                    await asyncio.sleep(delay)
                continue

            # 2. Content Validation Check
            if validator and not validator(text):
                if attempt >= attempts - 1:
                    break
                delay = self._compute_retry_delay(attempt)
                if self.verbose:
                    preview = (text[:100].replace('\n', ' ') + "...") if text and len(text) > 100 else (text or "").replace('\n', ' ')
                    print(f"LLM validation failed (Invalid format). Retry {attempt + 1}/{attempts - 1} after {delay:.1f}s. Response preview: {preview}")
                if delay > 0:
                    await asyncio.sleep(delay)
                continue

            return text
        return last_text

    def _compute_retry_delay(self, attempt_index: int) -> float:
        base = max(0.0, self.llm_retry_base_seconds)
        backoff = max(1.0, self.llm_retry_backoff)
        delay = base * (backoff ** attempt_index)
        max_delay = max(0.0, self.llm_retry_max_seconds)
        if max_delay > 0:
            delay = min(delay, max_delay)
        return delay

    async def _handle_action_async(self, response_text: str) -> Tuple[str, bool]:
        thought = extract_thought(response_text)
        action, error_message = extract_action_dict(response_text)
        
        if action is None:
            return json.dumps({"status": "error", "error_type": "InvalidAction", "message": error_message or "Missing Action"}, ensure_ascii=False), False

        tool_name = action.get("tool_name")
        params = action.get("parameters") or {}
        round_num = int(getattr(self.tools.controller, "current_round", 0) or 0) if getattr(self.tools, "controller", None) is not None else 0
        step_num = int(getattr(self, "_current_step", 0) or 0)
        event_id = f"agent_step:{round_num}:{step_num}:{str(tool_name)}"

        if tool_name == "Final Answer":
            sample_id = params.get("selected_sample_ids", params.get("selected_sample_id"))
            reason = params.get("reasoning", "")
            if isinstance(sample_id, list):
                sample_ids = [str(x) for x in sample_id]
            else:
                sample_ids = [] if sample_id is None else [str(sample_id)]
            
            # finalize_selection might be blocking, run in thread
            if self._trace_thought_enabled() and hasattr(self.tools.controller, "_append_trace"):
                try:
                    safe_thought = self._redact_text(thought)
                    self.tools.controller._append_trace({
                        "type": "agent_step_intent",
                        "event_id": event_id,
                        "session_id": self.session_id,
                        "round": round_num,
                        "step": step_num,
                        "tool": str(tool_name),
                        "parameters": params,
                        "thought": safe_thought,
                        "thought_sha1": self._sha1(safe_thought),
                    })
                except Exception:
                    pass
            result = await asyncio.to_thread(self.tools.finalize_selection, sample_ids, reason, thought)
            if self._trace_thought_enabled() and hasattr(self.tools.controller, "_append_trace"):
                try:
                    safe_obs = self._redact_text(result)
                    self.tools.controller._append_trace({
                        "type": "agent_step_result",
                        "event_id": event_id,
                        "session_id": self.session_id,
                        "round": round_num,
                        "step": step_num,
                        "tool": str(tool_name),
                        "observation": safe_obs,
                        "observation_sha1": self._sha1(safe_obs),
                    })
                except Exception:
                    pass
            return result, True

        if tool_name == "finalize_selection":
            sample_ids = params.get("sample_ids", [])
            reason = params.get("reason", "")
            if self._trace_thought_enabled() and hasattr(self.tools.controller, "_append_trace"):
                try:
                    safe_thought = self._redact_text(thought)
                    self.tools.controller._append_trace({
                        "type": "agent_step_intent",
                        "event_id": event_id,
                        "session_id": self.session_id,
                        "round": round_num,
                        "step": step_num,
                        "tool": str(tool_name),
                        "parameters": params,
                        "thought": safe_thought,
                        "thought_sha1": self._sha1(safe_thought),
                    })
                except Exception:
                    pass
            result = await asyncio.to_thread(self.tools.finalize_selection, sample_ids, reason, thought)
            if self._trace_thought_enabled() and hasattr(self.tools.controller, "_append_trace"):
                try:
                    safe_obs = self._redact_text(result)
                    self.tools.controller._append_trace({
                        "type": "agent_step_result",
                        "event_id": event_id,
                        "session_id": self.session_id,
                        "round": round_num,
                        "step": step_num,
                        "tool": str(tool_name),
                        "observation": safe_obs,
                        "observation_sha1": self._sha1(safe_obs),
                    })
                except Exception:
                    pass
            return result, True

        if self._trace_thought_enabled() and hasattr(self.tools.controller, "_append_trace"):
            try:
                safe_thought = self._redact_text(thought)
                self.tools.controller._append_trace({
                    "type": "agent_step_intent",
                    "event_id": event_id,
                    "session_id": self.session_id,
                    "round": round_num,
                    "step": step_num,
                    "tool": str(tool_name),
                    "parameters": params,
                    "thought": safe_thought,
                    "thought_sha1": self._sha1(safe_thought),
                })
            except Exception:
                pass
        result_json = await self._execute_tool_async(tool_name, params)
        if self._trace_thought_enabled() and hasattr(self.tools.controller, "_append_trace"):
            try:
                safe_obs = self._redact_text(result_json)
                self.tools.controller._append_trace({
                    "type": "agent_step_result",
                    "event_id": event_id,
                    "session_id": self.session_id,
                    "round": round_num,
                    "step": step_num,
                    "tool": str(tool_name),
                    "observation": safe_obs,
                    "observation_sha1": self._sha1(safe_obs),
                })
            except Exception:
                pass
        return result_json, False

    async def _execute_tool_async(self, tool_name: str, params: Dict[str, Any]) -> str:
        try:
            if not hasattr(self.tools, tool_name):
                return json.dumps({"status": "error", "error_type": "ToolNotFound", "message": f"Unknown tool '{tool_name}'"}, ensure_ascii=False)

            method = getattr(self.tools, tool_name)
            
            # Execute in thread pool to avoid blocking event loop
            result = await asyncio.to_thread(method, **params)
            
            if hasattr(self.tools, '_parse_response'):
                parsed = self.tools._parse_response(result)
                return json.dumps(parsed, ensure_ascii=False)
            
            payload = result
            if isinstance(result, str):
                try:
                    payload = json.loads(result)
                except Exception:
                    payload = result
            return json.dumps({"status": "success", "result": payload}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "error_type": type(e).__name__, "message": str(e)}, ensure_ascii=False)

    def reset(self) -> None:
        """
        Reset agent state.
        """
        self.history = []
        if hasattr(self.tools, 'reset'):
            self.tools.reset()
