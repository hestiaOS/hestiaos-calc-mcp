#!/usr/bin/env python3
"""One-shot plot worker — isolated mpmath sampling via stdio line protocol.

Protocol:
  Parent → Worker: exactly one UTF-8 JSON object followed by "\n"
  Worker → Parent: exactly one UTF-8 JSON object followed by "\n"

The worker reads at most MAX_PLOT_REQUEST_BYTES + 1 bytes from stdin,
processes exactly one request, outputs exactly one response, and exits.

Usage:
  python3 -m calc_mcp.plot_worker
"""

from __future__ import annotations

import json
import sys
from typing import Any

from calc_mcp.errors import CalculatorError, ErrorCode
from calc_mcp.limits import MAX_PLOT_REQUEST_BYTES
from calc_mcp.plot_contract import (
    parse_plot_request,
    public_top_level_error_message,
)
from calc_mcp.plot import sample_plot


def _make_error(code: str, message: str) -> str:
    return json.dumps({
        "ok": False,
        "error": {"code": code, "message": message},
    }, separators=(",", ":"))


def _make_success(payload: dict) -> str:
    return json.dumps({"ok": True, "result": payload}, separators=(",", ":"))


def _safe_error(code: str) -> str:
    """Return a worker-safe error response with the public template."""
    return _make_error(code, public_top_level_error_message(code))


def main() -> None:
    """Read one request from stdin, write one response to stdout, exit."""
    # --- Read bounded request from stdin ---
    raw = sys.stdin.buffer.read(MAX_PLOT_REQUEST_BYTES + 1)
    if not raw:
        sys.stdout.write(_safe_error("INVALID_ARGUMENT"))
        sys.stdout.write("\n")
        sys.stdout.flush()
        sys.exit(0)

    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        sys.stdout.write(_safe_error("INVALID_ARGUMENT"))
        sys.stdout.write("\n")
        sys.stdout.flush()
        sys.exit(0)

    if len(raw) > MAX_PLOT_REQUEST_BYTES:
        sys.stdout.write(_safe_error("LIMIT_EXCEEDED"))
        sys.stdout.write("\n")
        sys.stdout.flush()
        sys.exit(0)

    # --- Check for multiple lines / trailing data ---
    lines = decoded.split("\n")
    # If there are more than 2 lines (request line + empty trailing newline),
    # or the first line itself contains a newline within MAX_PLOT_REQUEST_BYTES
    if len(lines) > 2 or (len(lines) == 2 and lines[1] != ""):
        # Reject multiple messages
        sys.stdout.write(_safe_error("INVALID_ARGUMENT"))
        sys.stdout.write("\n")
        sys.stdout.flush()
        sys.exit(0)

    request_line = lines[0].strip()
    if not request_line:
        sys.stdout.write(_safe_error("INVALID_ARGUMENT"))
        sys.stdout.write("\n")
        sys.stdout.flush()
        sys.exit(0)

    # --- JSON parsing ---
    try:
        data: dict[str, Any] = json.loads(request_line)
    except json.JSONDecodeError:
        sys.stdout.write(_safe_error("INVALID_ARGUMENT"))
        sys.stdout.write("\n")
        sys.stdout.flush()
        sys.exit(0)

    if not isinstance(data, dict):
        sys.stdout.write(_safe_error("INVALID_ARGUMENT"))
        sys.stdout.write("\n")
        sys.stdout.flush()
        sys.exit(0)

    # --- Unknown field rejection ---
    allowed_fields = frozenset({
        "expression", "x_min", "x_max",
        "sample_count", "precision", "rounding",
    })
    extra = set(data.keys()) - allowed_fields
    if extra:
        sys.stdout.write(_safe_error("INVALID_ARGUMENT"))
        sys.stdout.write("\n")
        sys.stdout.flush()
        sys.exit(0)

    # --- Parse and validate request via plot_contract ---
    try:
        req = parse_plot_request(
            expression=data.get("expression", ""),
            x_min=data.get("x_min", ""),
            x_max=data.get("x_max", ""),
            sample_count=data.get("sample_count", 0),
            precision=data.get("precision", 0),
            rounding=data.get("rounding", "ROUND_HALF_EVEN"),
        )
    except CalculatorError as exc:
        # Map to worker-safe codes (never TIMEOUT/WORKER_FAILURE/INTERNAL)
        code_map = {
            ErrorCode.INVALID_ARGUMENT: "INVALID_ARGUMENT",
            ErrorCode.LIMIT_EXCEEDED: "LIMIT_EXCEEDED",
            ErrorCode.UNSUPPORTED_OPTION: "UNSUPPORTED_OPTION",
            ErrorCode.PARSE_ERROR: "PARSE_ERROR",
            ErrorCode.UNKNOWN_SYMBOL: "UNKNOWN_SYMBOL",
        }
        mapped = code_map.get(exc.code, "INVALID_ARGUMENT")
        sys.stdout.write(_safe_error(mapped))
        sys.stdout.write("\n")
        sys.stdout.flush()
        sys.exit(0)

    # --- Execute plot sampling ---
    try:
        result = sample_plot(req)
    except CalculatorError as exc:
        code_map = {
            ErrorCode.INVALID_ARGUMENT: "INVALID_ARGUMENT",
            ErrorCode.LIMIT_EXCEEDED: "LIMIT_EXCEEDED",
            ErrorCode.UNSUPPORTED_OPTION: "UNSUPPORTED_OPTION",
            ErrorCode.PARSE_ERROR: "PARSE_ERROR",
            ErrorCode.UNKNOWN_SYMBOL: "UNKNOWN_SYMBOL",
        }
        mapped = code_map.get(exc.code, "INVALID_ARGUMENT")
        sys.stdout.write(_safe_error(mapped))
        sys.stdout.write("\n")
        sys.stdout.flush()
        sys.exit(0)

    # --- Success ---
    out = _make_success(result)
    sys.stdout.write(out)
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
