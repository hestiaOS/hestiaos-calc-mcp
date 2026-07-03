"""Pure, deterministic plot contract layer for sample_function.

This module contains only contract validation, canonicalization, and grid
generation logic. It imports NO runtime/I/O modules (subprocess, threading,
asyncio, os, select, errno, time).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal

from calc_mcp.errors import CalculatorError, ErrorCode
from calc_mcp.limits import (
    GRID_GUARD_DIGITS,
    GUARD_DIGITS,
    MAX_PLOT_REQUEST_BYTES,
    MAX_PLOT_RESPONSE_BYTES,
    MAX_PLOT_X_ABS,
    MAX_PLOT_X_CHARS,
    MAX_RESULT_EXPONENT,
    MAX_SAMPLE_COUNT,
    MAX_SAMPLE_EXPR_LEN,
)

# ---------------------------------------------------------------------------
# Domain literal grammar (ASCII fixed-point only)
# ---------------------------------------------------------------------------
_DOMAIN_RE = re.compile(r"^[+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")


def parse_domain_literal(raw: str) -> str:
    """Validate and canonicalize a domain coordinate string.

    1. raw must be a string matching the ASCII fixed-point grammar exactly.
    2. Canonicalize lexically: remove leading '+', normalise integer leading
       zeros, strip fractional trailing zeros after the decimal point.
    3. No strip(), no Decimal.normalize().
    """
    if not isinstance(raw, str):
        raise CalculatorError(
            ErrorCode.INVALID_ARGUMENT,
            "Domain value must be a string.",
        )
    if not _DOMAIN_RE.match(raw):
        raise CalculatorError(
            ErrorCode.INVALID_ARGUMENT,
            f"Domain value {raw!r} is not a valid fixed-point decimal.",
        )

    # Lexical canonicalization (no strip, no Decimal.normalize)
    s = raw
    # Remove leading '+'
    if s.startswith("+"):
        s = s[1:]
    # Normalise leading zeros in integer part
    if "." in s:
        int_part, frac_part = s.split(".", 1)
        if int_part == "" or int_part == "+" or int_part == "-":
            pass  # keep as-is
        else:
            sign = ""
            rest = int_part
            if rest.startswith("-"):
                sign = "-"
                rest = rest[1:]
            elif rest.startswith("+"):
                sign = ""
                rest = rest[1:]
            if rest:
                stripped = rest.lstrip("0")
                int_part = sign + (stripped if stripped else "0")
            else:
                int_part = sign + "0"
        # Remove trailing zeros in fractional part
        frac_part = frac_part.rstrip("0")
        if frac_part:
            s = int_part + "." + frac_part
        else:
            s = int_part
    else:
        # Integer: normalise leading zeros
        sign = ""
        rest = s
        if rest.startswith("-"):
            sign = "-"
            rest = rest[1:]
        elif rest.startswith("+"):
            rest = rest[1:]
        if rest:
            stripped = rest.lstrip("0")
            s = sign + (stripped if stripped else "0")
        else:
            s = "0"

    # Normalise zero variants
    if s in ("-0", "+0", "0"):
        s = "0"

    return s


# ---------------------------------------------------------------------------
# PlotRequest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlotRequest:
    """Immutable, validated plot request."""
    expression: str
    x_min: str  # canonical domain string
    x_max: str  # canonical domain string
    sample_count: int
    precision: int
    rounding: str = "ROUND_HALF_EVEN"


def parse_plot_request(
    expression: str,
    x_min: str,
    x_max: str,
    sample_count: int,
    precision: int,
    rounding: str = "ROUND_HALF_EVEN",
) -> PlotRequest:
    """Validate and canonicalize a plot request. Returns a PlotRequest or raises."""

    # --- Type checks ---
    if not isinstance(expression, str):
        raise CalculatorError(ErrorCode.INVALID_ARGUMENT, "Expression must be a string.")
    if not isinstance(x_min, str) or not isinstance(x_max, str):
        raise CalculatorError(ErrorCode.INVALID_ARGUMENT, "Domain values must be strings.")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise CalculatorError(ErrorCode.INVALID_ARGUMENT, "sample_count must be an integer.")
    if isinstance(precision, bool) or not isinstance(precision, int):
        raise CalculatorError(ErrorCode.INVALID_ARGUMENT, "precision must be an integer.")

    # --- Expression limits ---
    if len(expression) > MAX_SAMPLE_EXPR_LEN:
        raise CalculatorError(
            ErrorCode.LIMIT_EXCEEDED,
            f"Expression exceeds maximum length of {MAX_SAMPLE_EXPR_LEN}.",
        )

    # --- Rounding ---
    if rounding != "ROUND_HALF_EVEN":
        raise CalculatorError(
            ErrorCode.UNSUPPORTED_OPTION,
            "Only ROUND_HALF_EVEN is supported for plot sampling.",
        )

    # --- Domain canonicalization ---
    x_min_canon = parse_domain_literal(x_min)
    x_max_canon = parse_domain_literal(x_max)

    # --- Domain comparison (x_min < x_max) ---
    if Decimal(x_min_canon) >= Decimal(x_max_canon):
        raise CalculatorError(
            ErrorCode.INVALID_ARGUMENT,
            "x_min must be less than x_max.",
        )

    # --- Domain absolute limits ---
    if abs(Decimal(x_min_canon)) > MAX_PLOT_X_ABS or abs(Decimal(x_max_canon)) > MAX_PLOT_X_ABS:
        raise CalculatorError(
            ErrorCode.LIMIT_EXCEEDED,
            f"Domain value exceeds maximum absolute value of {MAX_PLOT_X_ABS}.",
        )

    # --- Sample count ---
    if sample_count < 2:
        raise CalculatorError(
            ErrorCode.INVALID_ARGUMENT,
            "sample_count must be at least 2.",
        )
    if sample_count > MAX_SAMPLE_COUNT:
        raise CalculatorError(
            ErrorCode.LIMIT_EXCEEDED,
            f"sample_count exceeds maximum of {MAX_SAMPLE_COUNT}.",
        )

    # --- Precision ---
    if precision < 2:
        raise CalculatorError(
            ErrorCode.INVALID_ARGUMENT,
            "precision must be at least 2.",
        )
    if precision > 1000:  # NUMERIC_DPS_MAX
        raise CalculatorError(
            ErrorCode.LIMIT_EXCEEDED,
            "precision exceeds maximum of 1000.",
        )

    # --- Response budget estimation ---
    budget = estimated_response_bytes(sample_count, precision)
    if budget > MAX_PLOT_RESPONSE_BYTES:
        raise CalculatorError(
            ErrorCode.LIMIT_EXCEEDED,
            f"Estimated response size ({budget} bytes) exceeds limit "
            f"({MAX_PLOT_RESPONSE_BYTES} bytes). Reduce sample_count or precision.",
        )

    return PlotRequest(
        expression=expression,
        x_min=x_min_canon,
        x_max=x_max_canon,
        sample_count=sample_count,
        precision=precision,
        rounding=rounding,
    )


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------


def decimal_literal_significance(s: str) -> int:
    """Count significant digits in a validated decimal literal string.

    >>> decimal_literal_significance("0.0001") -> 1
    >>> decimal_literal_significance("12.0300") -> 4
    >>> decimal_literal_significance("0") -> 1
    """
    s = s.lstrip("-+")
    if "." in s:
        s = s.replace(".", "")
    s = s.lstrip("0")
    s = s.rstrip("0")
    return max(len(s), 1)


def grid_work_precision(x_min: str, x_max: str, sample_count: int) -> int:
    """Determine the Decimal context precision needed for grid generation."""
    significance = max(
        decimal_literal_significance(x_min),
        decimal_literal_significance(x_max),
    )
    depth = len(str(sample_count - 1))
    return significance + depth + GRID_GUARD_DIGITS


def build_canonical_grid(request: PlotRequest) -> list[str]:
    """Generate a deterministic canonical grid for the given PlotRequest.

    Returns a list of canonically formatted x-coordinate strings.
    The list length is exactly request.sample_count.
    All strings are canonical: no trailing zeros after decimal point,
    no leading '+' sign, no '-0', exactly one non-zero digit before
    decimal point if non-zero.
    """
    work_prec = grid_work_precision(request.x_min, request.x_max, request.sample_count)

    x_min_dec = Decimal(request.x_min)
    x_max_dec = Decimal(request.x_max)
    step = (x_max_dec - x_min_dec) / Decimal(request.sample_count - 1)

    grid: list[str] = []
    for i in range(request.sample_count):
        x_val = x_min_dec + Decimal(i) * step
        x_str = _canonicalize_grid_coordinate(x_val)
        grid.append(x_str)

    # Monotonicity check
    for i in range(1, len(grid)):
        if Decimal(grid[i]) <= Decimal(grid[i - 1]):
            raise CalculatorError(
                ErrorCode.LIMIT_EXCEEDED,
                "Grid coordinates collapsed after canonicalization — "
                "cannot produce distinct sample points.",
            )

    # Length check
    for x_str in grid:
        if len(x_str) > MAX_PLOT_X_CHARS:
            raise CalculatorError(
                ErrorCode.LIMIT_EXCEEDED,
                "A grid coordinate exceeded the maximum allowed string length.",
            )

    return grid


def _canonicalize_grid_coordinate(val: Decimal) -> str:
    """Canonicalize a Decimal coordinate for display.

    - -0 → "0"
    - Strip trailing zeros after decimal point
    - Remove decimal point if no fractional part remains
    """
    if val.is_zero():
        return "0"
    s = format(val, "f")
    # Strip trailing zeros
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


# ---------------------------------------------------------------------------
# Response budget estimation
# ---------------------------------------------------------------------------


def estimated_response_bytes(sample_count: int, precision: int) -> int:
    """Conservative estimate of response JSON size in bytes.

    Each point includes x, y, index, state, JSON framing overhead.
    For worst-case rows (error objects included), add error block size.
    """
    # Approximate maximum per-point size (bytes)
    MAX_COORD_LEN = MAX_PLOT_X_CHARS  # x value as string
    MAX_Y_LEN = precision + 32  # y value with precision + sign/exp/overhead
    PER_POINT_OVERHEAD = 200  # JSON framing, field names, index, state
    PER_POINT_ERROR = (MAX_ERROR_CODE_CHARS + MAX_ERROR_MSG_CHARS + 100)  # error block
    MAX_POINT_BYTES = MAX_COORD_LEN + MAX_Y_LEN + PER_POINT_OVERHEAD + PER_POINT_ERROR
    META_BYTES = 4096
    return sample_count * MAX_POINT_BYTES + META_BYTES


# ---------------------------------------------------------------------------
# Error code allowlists and public message templates
# ---------------------------------------------------------------------------

_TOP_LEVEL_ERROR_CODES = frozenset({
    "INVALID_ARGUMENT",
    "LIMIT_EXCEEDED",
    "UNSUPPORTED_OPTION",
    "PARSE_ERROR",
    "UNKNOWN_SYMBOL",
})

_PARENT_ORIGIN_CODES = frozenset({
    "TIMEOUT",
    "WORKER_FAILURE",
    "INTERNAL",
})

_POINT_ERROR_CODES = frozenset({
    "DIV_BY_ZERO",
    "DOMAIN_ERROR",
    "NUMERIC_OVERFLOW",
})

_WORKER_MESSAGES = {
    "INVALID_ARGUMENT":   "Invalid argument in plot request.",
    "LIMIT_EXCEEDED":     "Plot request exceeds resource limits.",
    "UNSUPPORTED_OPTION": "This option is not supported by the plot tier.",
    "PARSE_ERROR":        "Could not parse the expression.",
    "UNKNOWN_SYMBOL":     "Expression contains an unrecognized symbol.",
}

_PARENT_MESSAGES = {
    "TIMEOUT":        "The plot operation exceeded its time limit.",
    "WORKER_FAILURE": "The plot worker could not complete the request.",
    "INTERNAL":       "The plot operation could not be completed.",
}

_POINT_MESSAGES = {
    "DIV_BY_ZERO":        "Function is undefined at this sample point.",
    "DOMAIN_ERROR":        "Function is undefined at this sample point.",
    "NUMERIC_OVERFLOW":    "Function is undefined at this sample point.",
}


def public_top_level_error_message(code: str) -> str:
    """Return a fixed, safe public error message for a top-level error code."""
    return _WORKER_MESSAGES.get(code) or _PARENT_MESSAGES.get(code) or "An internal error occurred."


def public_point_error_message(code: str) -> str:
    return _POINT_MESSAGES.get(code, "An internal error occurred.")


# ---------------------------------------------------------------------------
# String limit constants used by the response budget estimator
# ---------------------------------------------------------------------------
MAX_ERROR_CODE_CHARS = 64
MAX_ERROR_MSG_CHARS = 128


# ---------------------------------------------------------------------------
# Worker response validation
# ---------------------------------------------------------------------------

_ALLOWED_TOP_LEVEL_CODES: frozenset[str] = frozenset({
    "INVALID_ARGUMENT",
    "LIMIT_EXCEEDED",
    "UNSUPPORTED_OPTION",
    "PARSE_ERROR",
    "UNKNOWN_SYMBOL",
})
_ALLOWED_POINT_STATES: frozenset[str] = frozenset({"defined", "undefined"})
_ALLOWED_POINT_ERROR_CODES: frozenset[str] = frozenset({
    "DIV_BY_ZERO",
    "DOMAIN_ERROR",
    "NUMERIC_OVERFLOW",
})
_PARENT_CODES: frozenset[str] = frozenset({
    "TIMEOUT", "WORKER_FAILURE", "INTERNAL",
})


def validate_plot_response(
    data: dict,
    expected_grid: list[str],
    expected_request: PlotRequest,
) -> dict:
    """Strictly validate a worker response. Returns validated data or raises WORKER_FAILURE."""
    if not isinstance(data, dict):
        _raise_worker_failure("Response is not a dict")

    # Top-level fields
    allowed_top = frozenset({"ok", "result", "error"})
    extra = set(data.keys()) - allowed_top
    if extra:
        _raise_worker_failure(f"Unexpected top-level fields: {extra}")

    ok_val = data.get("ok")
    if not isinstance(ok_val, bool):
        _raise_worker_failure("ok must be bool")

    if not ok_val:
        return _validate_error_response(data)
    return _validate_success_response(data, expected_grid, expected_request)


def _validate_error_response(data: dict) -> dict:
    err = data.get("error")
    if not isinstance(err, dict):
        _raise_worker_failure("error must be object")
    allowed_err = frozenset({"code", "message"})
    extra = set(err.keys()) - allowed_err
    if extra:
        _raise_worker_failure(f"Unexpected error fields: {extra}")
    code = err.get("code", "")
    if not isinstance(code, str) or code not in _ALLOWED_TOP_LEVEL_CODES:
        _raise_worker_failure(f"Forbidden error code: {code}")
    if code in _PARENT_CODES:
        _raise_worker_failure(f"Worker returned parent-only code: {code}")
    msg = err.get("message", "")
    expected_msg = public_top_level_error_message(code)
    if msg != expected_msg:
        _raise_worker_failure("Error message does not match public template")
    return data


def _validate_success_response(
    data: dict, expected_grid: list[str], expected_request: PlotRequest,
) -> dict:
    result = data.get("result")
    if not isinstance(result, dict):
        _raise_worker_failure("result must be object")
    allowed_result = frozenset({"points", "meta"})
    extra = set(result.keys()) - allowed_result
    if extra:
        _raise_worker_failure(f"Unexpected result fields: {extra}")

    # Validate meta
    meta = result.get("meta", {})
    if not isinstance(meta, dict):
        _raise_worker_failure("meta must be object")
    _validate_meta(meta, expected_request)

    # Validate points against expected grid
    points = result.get("points", [])
    if not isinstance(points, list) or len(points) != len(expected_grid):
        _raise_worker_failure(f"Expected {len(expected_grid)} points, got {len(points) if isinstance(points, list) else 'not a list'}")

    allowed_point = frozenset({"index", "x", "y", "state", "error"})
    for i, pt in enumerate(points):
        if not isinstance(pt, dict):
            _raise_worker_failure(f"Point {i} not a dict")
        extra = set(pt.keys()) - allowed_point
        if extra:
            _raise_worker_failure(f"Point {i} has unknown fields: {extra}")

        idx = pt.get("index")
        if idx != i:
            _raise_worker_failure(f"Point {i} index mismatch: {idx}")

        x_val = pt.get("x", "")
        if x_val != expected_grid[i]:
            _raise_worker_failure(f"Point {i} x={x_val!r} != expected {expected_grid[i]!r}")

        state = pt.get("state")
        if state not in _ALLOWED_POINT_STATES:
            _raise_worker_failure(f"Point {i} invalid state: {state}")

        y_val = pt.get("y")
        if state == "defined":
            if not isinstance(y_val, str) or not y_val:
                _raise_worker_failure(f"Point {i} defined but invalid y")
            if "error" in pt:
                _raise_worker_failure(f"Point {i} defined but has error")
        else:
            if y_val is not None:
                _raise_worker_failure(f"Point {i} undefined but y not null")
            pt_err = pt.get("error")
            if not isinstance(pt_err, dict):
                _raise_worker_failure(f"Point {i} undefined but missing error")
            allowed_pt_err = frozenset({"code", "message"})
            extra = set(pt_err.keys()) - allowed_pt_err
            if extra:
                _raise_worker_failure(f"Point {i} error has unknown fields: {extra}")
            pt_code = pt_err.get("code", "")
            if pt_code not in _ALLOWED_POINT_ERROR_CODES:
                _raise_worker_failure(f"Point {i} forbidden error code: {pt_code}")

    return data


def _validate_meta(meta: dict, request: PlotRequest) -> None:
    allowed_meta = frozenset({
        "contract_version", "expression", "requested_domain", "effective_domain",
        "sample_count", "precision", "rounding", "sampling", "engine",
        "assurance_level", "warnings",
    })
    extra = set(meta.keys()) - allowed_meta
    if extra:
        _raise_worker_failure(f"Unexpected meta fields: {extra}")
    if meta.get("sample_count") != request.sample_count:
        _raise_worker_failure("sample_count mismatch in meta")
    if meta.get("precision") != request.precision:
        _raise_worker_failure("precision mismatch in meta")


def _raise_worker_failure(detail: str) -> None:
    raise CalculatorError(
        ErrorCode.WORKER_FAILURE,
        public_top_level_error_message("WORKER_FAILURE"),
        detail={"reason": detail},
    )


# ---------------------------------------------------------------------------
# Canonical numeric result validation
# ---------------------------------------------------------------------------

_CANONICAL_FP_RE = re.compile(r'^-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?$')
_CANONICAL_SCI_RE = re.compile(
    r'^-?[1-9](?:\.[0-9]*[1-9])?e-?(?:0|[1-9][0-9]*)$'
)


def is_canonical_numeric_result(s: str) -> bool:
    if not isinstance(s, str):
        return False
    if s == "0":
        return True
    if s in ("-0", "0.0", "-0.0"):
        return False
    fp_match = _CANONICAL_FP_RE.match(s)
    sci_match = _CANONICAL_SCI_RE.match(s)
    if not fp_match and not sci_match:
        return False
    from decimal import Decimal
    try:
        d = Decimal(s)
    except Exception:
        return False
    if d.is_zero():
        return False
    abs_d = abs(d)
    if fp_match:
        return Decimal("1e-6") <= abs_d < Decimal("1e13")
    else:
        return abs_d < Decimal("1e-6") or abs_d >= Decimal("1e13")


def format_numeric_result(value_mpf, significant_digits: int) -> str:
    """Format a numeric mpmath value as a canonical numeric string.

    - Fixed-point for 1e-6 <= abs(value) < 1e13
    - Scientific notation otherwise
    - No trailing zeros after decimal point
    - -0 → "0"

    Deterministic: same input + same significant_digits → same output.
    """
    from decimal import (
        Decimal,
        InvalidOperation,
        ROUND_HALF_EVEN,
        Overflow,
        localcontext,
    )
    import decimal

    if value_mpf == 0:
        return "0"

    work_dps = significant_digits + GUARD_DIGITS
    import mpmath as mp

    with mp.workdps(work_dps):
        s = mp.nstr(value_mpf, work_dps)
        d = Decimal(s)

    if d.is_zero():
        return "0"

    exp = d.adjusted() - significant_digits + 1
    quantum = Decimal(1).scaleb(exp)

    with localcontext() as ctx:
        ctx.prec = significant_digits + GUARD_DIGITS
        try:
            rounded = d.quantize(quantum, rounding=ROUND_HALF_EVEN)
        except (InvalidOperation, decimal.Overflow):
            raise CalculatorError(
                ErrorCode.NUMERIC_OVERFLOW,
                "Numeric overflow in result formatting.",
            )

    if abs(rounded.adjusted()) > 1000:  # MAX_RESULT_EXPONENT
        raise CalculatorError(
            ErrorCode.NUMERIC_OVERFLOW,
            "Result exponent exceeds limit.",
        )

    abs_val = abs(rounded)
    if Decimal("1e-6") <= abs_val < Decimal("1e13"):
        s = format(rounded, "f")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s
    else:
        sign = "-" if rounded < 0 else ""
        coeff = abs(rounded)
        exp_adj = coeff.adjusted()
        mantissa_val = coeff.scaleb(-exp_adj)
        mantissa_str = format(mantissa_val, "f")
        if "." in mantissa_str:
            mantissa_str = mantissa_str.rstrip("0").rstrip(".")
        return f"{sign}{mantissa_str}e{exp_adj}"