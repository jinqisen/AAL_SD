
import json
import os
import hashlib
import requests
import re
import time
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from .prompt_template import PromptBuilder
from .toolbox import Toolbox
from .config import AgentConstraints
from .utils import (
    is_llm_transport_error,
    validate_response,
    extract_thought,
    extract_action_dict
)

class SiliconFlowClient:
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float = 0.0, timeout: int = 60, max_retries: int = 3, retry_delay: float = 5.0, retry_backoff: float = 2.0):
        self.api_key = api_key
        self.base_url = (base_url or "").rstrip('/')
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Interacts with the SiliconFlow API (OpenAI compatible).
        """
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY is missing (agent experiments require a valid API key).")

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

        endpoint = f"{self.base_url}/chat/completions"
        
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout)
                
                if response.status_code == 200:
                    data = response.json()
                    content = data['choices'][0]['message']['content']
                    return content
                
                # Retry on server errors (5xx) or Rate Limit (429)
                if response.status_code in [429, 500, 502, 503, 504]:
                    error_msg = f"API returned status code {response.status_code}. Response: {response.text}"
                    last_error = error_msg
                    if attempt < self.max_retries:
                        delay = self.retry_delay * (self.retry_backoff ** attempt)
                        time.sleep(delay)
                        continue
                    return f"Error: {error_msg}"
                else:
                    # Client error (4xx except 429) - Do not retry
                    return f"Error: API returned status code {response.status_code}. Response: {response.text}"
                    
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    delay = self.retry_delay * (self.retry_backoff ** attempt)
                    time.sleep(delay)
                    continue
                return f"Error calling API: {str(e)}"
            except Exception as e:
                return f"Error calling API: {str(e)}"
                
        return f"Error: Max retries exceeded. Last error: {last_error}"

class AgentManager:
    def __init__(
        self,
        tools: Toolbox,
        client: SiliconFlowClient,
        verbose: bool = True,
        llm_max_retries: int = 3,
        llm_retry_base_seconds: float = 5.0,
        llm_retry_backoff: float = 2.0,
        llm_retry_max_seconds: float = 60.0,
    ):
        """
        LLM Agent Controller that manages the ReAct loop.
        Args:
            tools: Instance of AgentTools/Toolbox.
            client: LLM client object with a .chat(messages) method.
            verbose: Whether to print the agent's thought process.
        """
        self.tools = tools
        self.client = client
        self.verbose = verbose
        self.max_steps = int(getattr(AgentConstraints, "MAX_STEPS", 10) or 10)
        self.history = []
        self.prompt_builder = PromptBuilder()
        self.llm_max_retries = int(llm_max_retries or 0)
        self.llm_retry_base_seconds = float(llm_retry_base_seconds or 0.0)
        self.llm_retry_backoff = float(llm_retry_backoff or 1.0)
        self.llm_retry_max_seconds = float(llm_retry_max_seconds or 0.0)
        try:
            self.session_id = f"{int(time.time() * 1000)}-{int(os.getpid())}"
        except Exception:
            self.session_id = None

    def run_cycle(self) -> Dict[str, Any]:
        """
        Executes one Active Learning selection cycle.
        """
        # 1. Pre-calculate scores (handled by tools, but good to ensure)
        self.tools.precalculate_scores()
        
        # Reset per-cycle tool usage stats (Trace Logging)
        self.tool_usage_stats = {}

        # 2. Initialize conversation
        controller = getattr(self.tools, "controller", None)
        labeled_size = None
        unlabeled_size = None
        if controller is not None:
            labeled_size = len(getattr(controller, "labeled_indices", []) or [])
            unlabeled_size = len(getattr(controller, "unlabeled_indices", []) or [])
        query_size = None
        if controller is not None and hasattr(controller, "config"):
            query_size = getattr(controller.config, "QUERY_SIZE", None)
        user_prompt = self.prompt_builder.build_user_prompt(labeled_size, unlabeled_size, query_size)

        status_payload = None
        try:
            status_payload = json.loads(self.tools.get_system_status())
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

        require_explicit_lambda = False
        controller_cfg = getattr(getattr(self.tools, "controller", None), "exp_config", None)
        if isinstance(controller_cfg, dict):
            require_explicit_lambda = bool(controller_cfg.get("require_explicit_lambda", False))

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
                    control_permissions=getattr(self.tools, "control_permissions", None),
                    require_explicit_lambda=bool(require_explicit_lambda),
                    miou_low_gain_streak=int(self.tools.training_state.get("miou_low_gain_streak", 0)),
                )
            },
            {"role": "user", "content": user_prompt}
        ]

        step = 0
        while step < self.max_steps:
            if self.verbose:
                print(f"\n--- Step {step + 1} ---")
            self._current_step = int(step + 1)

            if step == self.max_steps - 1:
                self.history.append({
                    "role": "user",
                    "content": "最后一步：你必须调用 finalize_selection 提交样本（sample_ids 与 reason）。不要再调用 get_system_status/get_score_distribution，也不要再进行控参。"
                })

            # 3. Call LLM with validation
            response_text = self._call_llm_with_retries(self.history, validator=validate_response)

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
            
            # Clean up response (just in case the stop sequence didn't work or wasn't supported)
            if "Observation:" in response_text:
                response_text = response_text.split("Observation:")[0].strip()
            if "Observation：" in response_text:
                response_text = response_text.split("Observation：")[0].strip()
            
            if self.verbose:
                print(f"LLM Response:\n{response_text}")

            self.history.append({"role": "assistant", "content": response_text})

            # 4. Parse Action
            observation, done = self._handle_action(response_text)

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
            elif isinstance(payload, dict) and payload.get("error_type") == "PermissionDenied":
                self.history.append({
                    "role": "user",
                    "content": "注意：你刚刚调用的工具在当前消融配置中被禁止（PermissionDenied）。不要再次调用该工具；按执行顺序继续下一阶段（选样 get_top_k_samples → 提交 finalize_selection）。"
                })
            elif isinstance(payload, dict) and payload.get("error_type") == "ConstraintViolationError":
                self.history.append({
                    "role": "user",
                    "content": "注意：你刚刚的动作违反约束（ConstraintViolation）。不要重复同一无效动作；请改用默认值/保守策略并继续到 finalize_selection。"
                })
            if done:
                try:
                    return json.loads(observation)
                except Exception:
                    preview = str(observation)
                    if len(preview) > 400:
                        preview = preview[:400] + "..."
                    return {
                        "status": "error",
                        "error_type": "InvalidToolObservation",
                        "message": "Final tool observation is not valid JSON",
                        "observation_preview": preview,
                    }
            step += 1

        return {"status": "error", "error_type": "AgentMaxSteps", "message": "Max steps reached"}

    def _trace_thought_enabled(self) -> bool:
        controller = getattr(self.tools, "controller", None)
        cfg = getattr(controller, "config", None) if controller is not None else None
        if cfg is not None:
            try:
                return bool(getattr(cfg, "TRACE_AGENT_THOUGHT", False))
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

    def _call_llm_with_retries(self, messages: List[Dict[str, str]], validator: Optional[Callable[[str], bool]] = None) -> str:
        attempts = max(1, 1 + self.llm_max_retries)
        last_text = None
        for attempt in range(attempts):
            if self._trace_prompt_enabled() and hasattr(self.tools, "controller") and hasattr(self.tools.controller, "_append_trace"):
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
                            "messages_sha1": self._sha1(json.dumps(safe_messages, ensure_ascii=False)),
                        }
                    )
                except Exception:
                    pass
            text = self.client.chat(messages)
            last_text = text
            if self._trace_prompt_enabled() and hasattr(self.tools, "controller") and hasattr(self.tools.controller, "_append_trace"):
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
                    time.sleep(delay)
                continue

            # 2. Content Validation Check
            if validator and not validator(text):
                if attempt >= attempts - 1:
                    break
                delay = self._compute_retry_delay(attempt)
                if self.verbose:
                    # Log a snippet of the invalid response to help debugging
                    preview = (text[:100].replace('\n', ' ') + "...") if text and len(text) > 100 else (text or "").replace('\n', ' ')
                    print(f"LLM validation failed (Invalid format). Retry {attempt + 1}/{attempts - 1} after {delay:.1f}s. Response preview: {preview}")
                if delay > 0:
                    time.sleep(delay)
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

    def _handle_action(self, response_text: str) -> Tuple[Optional[str], Optional[str]]:
        thought = extract_thought(response_text)
        action, error_message = extract_action_dict(response_text)
        if action is None:
            return json.dumps({"status": "error", "error_type": "InvalidAction", "message": error_message or "Missing Action"}, ensure_ascii=False), False

        tool_name = action.get("tool_name")
        params = action.get("parameters") or {}
        round_num = int(getattr(self.tools.controller, "current_round", 0) or 0) if getattr(self.tools, "controller", None) is not None else 0
        step_num = int(getattr(self, "_current_step", 0) or 0)
        event_id = f"agent_step:{round_num}:{step_num}:{str(tool_name)}"

        # Trace Logging: Count tool usage
        if tool_name:
            self.tool_usage_stats[tool_name] = self.tool_usage_stats.get(tool_name, 0) + 1

        limited_tools = (
            "get_system_status",
            "get_score_distribution",
            "set_query_size",
            "set_lambda",
            "set_hyperparameter",
        )
        if tool_name in limited_tools:
            if int(self.tool_usage_stats.get(tool_name, 0) or 0) > 2:
                return json.dumps({
                    "status": "error",
                    "error_type": "InvalidAction",
                    "message": f"{tool_name} called too many times in one cycle; proceed to get_top_k_samples and finalize_selection."
                }, ensure_ascii=False), False

        if tool_name == "Final Answer":
            sample_id = params.get("selected_sample_ids", params.get("selected_sample_id"))
            reason = params.get("reasoning", "")
            if isinstance(sample_id, list):
                sample_ids = [str(x) for x in sample_id]
            else:
                sample_ids = [] if sample_id is None else [str(sample_id)]
            
            # Append usage stats to the final result context for analysis
            if hasattr(self.tools.controller, "_append_trace"):
                try:
                    self.tools.controller._append_trace({
                        "type": "tool_usage_stats",
                        "round": int(getattr(self.tools.controller, "current_round", 0) or 0),
                        "stats": dict(self.tool_usage_stats)
                    })
                except Exception:
                    pass

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
            result = self.tools.finalize_selection(sample_ids, reason, thought)
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
            
            # Append usage stats to the final result context for analysis
            if hasattr(self.tools.controller, "_append_trace"):
                try:
                    self.tools.controller._append_trace({
                        "type": "tool_usage_stats",
                        "round": int(getattr(self.tools.controller, "current_round", 0) or 0),
                        "stats": dict(self.tool_usage_stats)
                    })
                except Exception:
                    pass
            
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
            result = self.tools.finalize_selection(sample_ids, reason, thought)
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
        result_json = self._execute_tool(tool_name, params)

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
        
        # Auto-save trace log if loop count is high (Warning)
        if self.tool_usage_stats.get(tool_name, 0) > 5:
            if hasattr(self.tools.controller, "_append_trace"):
                try:
                    self.tools.controller._append_trace({
                        "type": "tool_usage_warning",
                        "round": int(getattr(self.tools.controller, "current_round", 0) or 0),
                        "tool": tool_name,
                        "count": self.tool_usage_stats[tool_name],
                        "message": "High frequency tool usage detected"
                    })
                except Exception:
                    pass

        return result_json, False

    def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> str:
        try:
            if not hasattr(self.tools, tool_name):
                return json.dumps({"status": "error", "error_type": "ToolNotFound", "message": f"Unknown tool '{tool_name}'"}, ensure_ascii=False)

            method = getattr(self.tools, tool_name)
            result = method(**params)
            
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
        Reset agent state to prevent contamination between experiments.
        Clears conversation history and resets toolbox state.
        """
        self.history = []
        if hasattr(self.tools, 'reset'):
            self.tools.reset()
