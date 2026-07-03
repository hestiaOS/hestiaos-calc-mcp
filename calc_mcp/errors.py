"""Structured error codes and error objects.

Every error is returned as a structured JSON object:
  {"ok": false, "error": {"code": "<CODE>", "message": "<short>", "detail": {...}}}
Never a raw stacktrace, path leak, or fabricated value.
"""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    """Canonical error codes for the calculator MCP."""

    PARSE_ERROR = "PARSE_ERROR"
    UNKNOWN_SYMBOL = "UNKNOWN_SYMBOL"
    DIV_BY_ZERO = "DIV_BY_ZERO"
    DOMAIN_ERROR = "DOMAIN_ERROR"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    INTERNAL = "INTERNAL"
    UNSUPPORTED_OP = "UNSUPPORTED_OP"
    SYMBOLIC_FAILED = "SYMBOLIC_FAILED"
    INCOMPATIBLE_UNITS = "INCOMPATIBLE_UNITS"
    UNKNOWN_UNIT = "UNKNOWN_UNIT"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    TIMEOUT = "TIMEOUT"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    UNSUPPORTED_OPTION = "UNSUPPORTED_OPTION"
    WORKER_FAILURE = "WORKER_FAILURE"
    NUMERIC_OVERFLOW = "NUMERIC_OVERFLOW"


class CalculatorError(Exception):
    """Calculator error with structured payload.

    Never contains stacktraces, file paths, or secrets.
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        detail: dict | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.detail = detail or {}
        super().__init__(self.message)

    def to_dict(self) -> dict:
        return {
            "ok": False,
            "error": {
                "code": self.code.value,
                "message": self.message,
                "detail": self.detail,
            },
        }

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"
