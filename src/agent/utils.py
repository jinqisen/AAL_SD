import json
import re
from typing import Optional, Dict, Any, Tuple, List

def is_llm_transport_error(response_text: str) -> bool:
    """
    Check if the response text indicates a transport/API error.
    """
    text = str(response_text).strip()
    if not text:
        return False
    prefixes = (
        "Error calling API:",
        "Error: API returned status code",
    )
    return text.startswith(prefixes)

def validate_response(text: str) -> bool:
    """
    Validate if the LLM response contains a valid ReAct Action.
    Checks for 'Action:' marker and valid JSON structure, rejecting trailing garbage.
    """
    if not text:
        return False
    
    # 1. Check for Action keyword
    marker = None
    if "Action:" in text:
        marker = "Action:"
    elif "Action：" in text:
        marker = "Action："
    
    if not marker:
        return False
    
    # 2. Extract Action part and check for trailing garbage
    try:
        parts = text.rsplit(marker, 1)
        if len(parts) < 2:
            return False
        
        candidate = parts[-1].strip()
        start_idx = candidate.find("{")
        if start_idx == -1:
            return False
        
        json_str = candidate[start_idx:]
        decoder = json.JSONDecoder()
        obj, end_idx = decoder.raw_decode(json_str)
        
        # Check remainder after JSON
        remainder = json_str[end_idx:].strip()
        # Strict check: prompt forbids any text after JSON
        if remainder:
            return False
            
    except Exception:
        # JSON parse error or other error -> Invalid
        return False

    return True

def extract_thought(response_text: str) -> Optional[str]:
    """
    Extract the 'Thought' part from the response.
    """
    text = (response_text or "").strip()
    if not text:
        return None
    markers = ["Thought:", "Thought：", "思考:", "思考："]
    for marker in markers:
        if marker in text:
            thought_part = text.split(marker, 1)[1].strip()
            action_markers = ["Action:", "Action：", "行动:", "行动："]
            for action_marker in action_markers:
                if action_marker in thought_part:
                    thought_part = thought_part.split(action_marker, 1)[0].strip()
            return thought_part
    return None

def extract_action_dict(response_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Extract the Action JSON object from the response.
    Returns (action_dict, error_message).
    """
    text = (response_text or "").strip()
    if not text:
        return None, "Missing Action"

    # Remove code blocks
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()

    markers = ["Action:", "Action：", "行动:", "行动："]
    segments = []
    for marker in markers:
        if marker in text:
            segments.append(text.split(marker, 1)[1].strip())

    candidates = segments if segments else [text]
    decoder = json.JSONDecoder()

    for candidate in candidates:
        brace_index = candidate.find("{")
        if brace_index == -1:
            continue
        payload_text = candidate[brace_index:].lstrip()
        try:
            obj, _ = decoder.raw_decode(payload_text)
        except Exception:
            continue
        if isinstance(obj, dict) and "tool_name" in obj:
            return obj, None

    if segments:
        return None, "Action is not valid JSON"
    return None, "Missing Action"
