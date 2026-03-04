from typing import Optional, Any, Dict


class AgentToolError(Exception):
    """Base exception for all Agent tool errors"""
    def __init__(self, message: str, error_type: str = "AgentToolError", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}


class ToolNotFoundError(AgentToolError):
    """Raised when a requested tool does not exist"""
    def __init__(self, tool_name: str):
        super().__init__(
            f"Unknown tool '{tool_name}'",
            error_type="ToolNotFound",
            details={"tool_name": tool_name}
        )


class InvalidParameterError(AgentToolError):
    """Raised when a tool receives invalid parameters"""
    def __init__(self, parameter_name: str, value: Any, reason: str = ""):
        super().__init__(
            f"Invalid parameter '{parameter_name}': {value}. {reason}",
            error_type="InvalidParameter",
            details={"parameter": parameter_name, "value": value, "reason": reason}
        )


class StateError(AgentToolError):
    """Raised when the system is in an invalid state"""
    def __init__(self, message: str, state_info: Optional[Dict[str, Any]] = None):
        super().__init__(
            message,
            error_type="StateError",
            details={"state_info": state_info}
        )


class DataError(AgentToolError):
    """Raised when data is missing or corrupted"""
    def __init__(self, message: str, data_info: Optional[Dict[str, Any]] = None):
        super().__init__(
            message,
            error_type="DataError",
            details={"data_info": data_info}
        )


class ConstraintViolationError(AgentToolError):
    """Raised when a constraint is violated"""
    def __init__(self, constraint_name: str, value: Any, limit: Any):
        super().__init__(
            f"Constraint '{constraint_name}' violated: {value} exceeds limit {limit}",
            error_type="ConstraintViolation",
            details={"constraint": constraint_name, "value": value, "limit": limit}
        )


class LLMTransportError(AgentToolError):
    """Raised when LLM API communication fails"""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(
            message,
            error_type="LLMTransportError",
            details={"status_code": status_code}
        )


class LLMResponseError(AgentToolError):
    """Raised when LLM response is invalid or malformed"""
    def __init__(self, message: str, response_text: Optional[str] = None):
        super().__init__(
            message,
            error_type="LLMResponseError",
            details={"response_preview": response_text[:200] if response_text else None}
        )


class ParseError(AgentToolError):
    """Raised when parsing fails"""
    def __init__(self, message: str, parse_target: Optional[str] = None):
        super().__init__(
            message,
            error_type="ParseError",
            details={"parse_target": parse_target}
        )
