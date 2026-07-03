"""FastMCP server for calculator operations.

4 tools — structured JSON in/out, all errors as structured objects.

Tool surface:
  - calculate(expression, precision, rounding)  — exact + numeric routing
  - symbolic(op, expression, variable)           — lazy SymPy tier
  - convert_units(value, from_unit, to_unit)     — lazy Pint tier
  - capabilities()                               — agent discovery

Logging discipline: all diagnostics → stderr. stdout = protocol only.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from calc_mcp.ast_nodes import VALID_BINARY_OPS, VALID_CONSTANTS, VALID_FUNCTIONS
from calc_mcp.errors import CalculatorError, ErrorCode
from calc_mcp.limits import (
    CALL_TIMEOUT_MS,
    MAX_INPUT_LEN,
    MAX_OUTPUT_BYTES,
    NUMERIC_DPS_DEFAULT,
    NUMERIC_DPS_MAX,
    SYMBOLIC_TIMEOUT_MS,
)
from calc_mcp.parser import parse
from mcp.server.fastmcp import FastMCP

log = logging.getLogger("calc-mcp")

# ---------------------------------------------------------------------------
# Rounded decimal rendering (exact path)
# ---------------------------------------------------------------------------

_ROUNDING_MODES: dict[str, str] = {
    "ROUND_HALF_EVEN": "ROUND_HALF_EVEN",
    "ROUND_HALF_UP": "ROUND_HALF_UP",
    "ROUND_HALF_DOWN": "ROUND_HALF_DOWN",
    "ROUND_UP": "ROUND_UP",
    "ROUND_DOWN": "ROUND_DOWN",
    "ROUND_CEILING": "ROUND_CEILING",
    "ROUND_FLOOR": "ROUND_FLOOR",
}

_rounding_map: dict[str, str] = _ROUNDING_MODES  # alias for import


def _rational_to_decimal(rational: str, precision: int, rounding: str) -> str:
    """Render an exact rational string to a decimal with given precision/rounding."""
    import decimal
    from decimal import Decimal, ROUND_HALF_EVEN

    round_mode = _ROUNDING_MODES.get(rounding, "ROUND_HALF_EVEN")
    dec_rounding = getattr(Decimal, round_mode, ROUND_HALF_EVEN)

    if "/" in rational:
        parts = rational.split("/")
        d = Decimal(parts[0]) / Decimal(parts[1])
    else:
        d = Decimal(rational)

    # Fast path: exact integer — no quantize needed, just format with trailing zeros.
    # This avoids decimal.InvalidOperation on huge integers (P1 crash: 10^100+1).
    if d == d.to_integral_value():
        fmt = f"{{:.{precision}f}}"
        return fmt.format(d)

    # Non-integer path: quantize with rounding and guard digits.
    quantum = Decimal(10) ** -precision
    with decimal.localcontext() as ctx:
        size = len(d.normalize().as_tuple().digits) + precision + 10
        ctx.prec = max(size, 100)
        try:
            rounded = d.quantize(quantum, rounding=dec_rounding)
        except (decimal.InvalidOperation, decimal.Overflow, ValueError):
            raise CalculatorError(
                ErrorCode.LIMIT_EXCEEDED,
                "Result too large to display with the requested precision.",
            )
    fmt = f"{{:.{precision}f}}"
    return fmt.format(rounded)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

mcp = FastMCP("calc-mcp", log_level="ERROR")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(data: dict) -> str:
    """Wrap a successful result as JSON."""
    return json.dumps({"ok": True, "result": data})


def _err(code: ErrorCode, message: str, detail: dict | None = None) -> str:
    """Wrap a structured error as JSON."""
    return json.dumps(CalculatorError(code, message, detail).to_dict())


def _safe_json(data: dict) -> str:
    """Serialize to JSON, enforcing output size limit."""
    dumped = json.dumps(data)
    if len(dumped.encode("utf-8")) > MAX_OUTPUT_BYTES:
        return _err(
            ErrorCode.LIMIT_EXCEEDED,
            f"Response exceeds {MAX_OUTPUT_BYTES} byte limit",
        )
    return dumped


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def calculate(
    expression: str,
    precision: int = NUMERIC_DPS_DEFAULT,
    rounding: str = "ROUND_HALF_EVEN",
) -> str:
    """Evaluate an arithmetic expression with exact rationals or numeric mpmath.

    - Exact rational arithmetic (+, -, *, /, ^ with integer exponents):
      returns rational and decimal forms.
    - Transcendental functions (sqrt, sin, cos, tan, ln, log, exp) and
      irrational constants (pi, e) are evaluated numerically via mpmath
      with controlled precision.

    Args:
        expression: Arithmetic expression string (e.g., '1/2+1/3', 'sqrt(2)').
        precision: Decimal places for output (default 50, max 1000).
        rounding: Rounding mode. One of: ROUND_HALF_EVEN, ROUND_HALF_UP,
            ROUND_HALF_DOWN, ROUND_UP, ROUND_DOWN, ROUND_CEILING, ROUND_FLOOR.

    Returns:
        JSON with exact status, rational form, and decimal string.
    """
    # Validate expression length
    if len(expression) > MAX_INPUT_LEN:
        return _err(
            ErrorCode.LIMIT_EXCEEDED,
            f"Expression exceeds {MAX_INPUT_LEN} character limit",
        )

    # Parse
    try:
        ast = parse(expression)
    except CalculatorError as exc:
        return json.dumps(exc.to_dict())

    from calc_mcp.numeric import eval_ast_numeric, has_transcendental

    try:
        if has_transcendental(ast):
            # Numeric path
            result = eval_ast_numeric(ast, dps=precision)
            return _ok({
                "exact": False,
                "decimal": result,
                "precision": precision,
                "rounding": rounding,
            })
        else:
            # Exact path: core supervisor
            from calc_mcp.core import exact_eval

            rational = exact_eval(ast)
            decimal = _rational_to_decimal(rational, precision, rounding)
            return _ok({
                "exact": True,
                "rational": rational,
                "decimal": decimal,
                "precision": precision,
                "rounding": rounding,
            })
    except CalculatorError as exc:
        return json.dumps(exc.to_dict())


@mcp.tool()
async def symbolic(
    op: str,
    expression: str,
    variable: str = "x",
) -> str:
    """Apply a symbolic mathematics operation via SymPy (lazy loaded, preemptive timeout).

    Supported operations: simplify, solve, differentiate, integrate,
    factor, expand.

    The operation runs in an isolated subprocess that can be preemptively
    killed on timeout — no lingering compute threads.

    Args:
        op: Operation name from the supported list above.
        expression: Mathematical expression in SymPy-compatible syntax.
        variable: Symbol/dependent variable (default 'x').

    Returns:
        JSON with operation result including LaTeX output.
    """
    try:
        result = _symbolic_subprocess(op, expression, variable)
        return _ok(result)
    except CalculatorError as exc:
        return json.dumps(exc.to_dict())


def _symbolic_subprocess(op: str, expression: str, variable: str) -> dict:
    """Run symbolic operation in a killable subprocess with timeout."""
    import subprocess as _subprocess

    script = """\
import sys, json
sys.path.insert(0, {repo!r})
from calc_mcp.tiers.symbolic import symbolic
from calc_mcp.errors import CalculatorError
try:
    result = symbolic({op!r}, {expression!r}, {variable!r})
    print(json.dumps(result))
except CalculatorError as e:
    print(json.dumps({{"_error": e.message, "_code": e.code.value}}))
    sys.exit(1)
except Exception as e:
    print(json.dumps({{"_error": str(e), "_code": "SYMBOLIC_FAILED"}}))
    sys.exit(1)
""".format(
        repo=str(Path(__file__).resolve().parent.parent),
        op=op,
        expression=expression,
        variable=variable,
    )

    proc = _subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=_subprocess.PIPE,
        stderr=_subprocess.PIPE,
        cwd=Path(__file__).resolve().parent.parent,
    )

    try:
        stdout, stderr = proc.communicate(timeout=SYMBOLIC_TIMEOUT_MS / 1000.0)
    except _subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)
        raise CalculatorError(
            ErrorCode.TIMEOUT,
            f"Symbolic {op!r} timed out after {SYMBOLIC_TIMEOUT_MS}ms",
        )

    if proc.returncode != 0:
        try:
            err_data = json.loads(stdout.decode())
            code = err_data.get("_code", "SYMBOLIC_FAILED")
            msg = err_data.get("_error", stdout.decode()[:200])
        except (json.JSONDecodeError, ValueError):
            code = "SYMBOLIC_FAILED"
            msg = stdout.decode()[:200] or "(no output)"
        # Map known error codes
        code_map = {
            "UNSUPPORTED_OP": ErrorCode.UNSUPPORTED_OP,
            "SYMBOLIC_FAILED": ErrorCode.SYMBOLIC_FAILED,
            "TIMEOUT": ErrorCode.TIMEOUT,
        }
        mapped = code_map.get(code, ErrorCode.SYMBOLIC_FAILED)
        raise CalculatorError(mapped, msg)

    result = json.loads(stdout.decode())
    return result


@mcp.tool()
async def convert_units(
    value: str,
    from_unit: str,
    to_unit: str,
) -> str:
    """Convert a value between measurement units via Pint (lazy loaded).

    Supports SI, imperial, and many other unit systems.

    Args:
        value: Numeric value as a string (e.g., '10', '3.14').
        from_unit: Source unit (e.g., 'meter', 'kg', 'degC').
        to_unit: Target unit (e.g., 'foot', 'pound', 'degF').

    Returns:
        JSON with converted value and unit.
    """
    try:
        from calc_mcp.tiers.units import convert as convert_fn
        result = convert_fn(value, from_unit, to_unit)
        return _ok(result)
    except CalculatorError as exc:
        return json.dumps(exc.to_dict())


@mcp.tool()
async def capabilities() -> str:
    """Return the server's self-description for agent discovery.

    Lists available tiers, operators, functions, constants, and limits.
    No computation is performed — purely descriptive.
    """
    return _ok({
        "name": "calc-mcp",
        "version": "0.1.0",
        "tiers": ["exact", "numeric", "symbolic", "units"],
        "operators": sorted(VALID_BINARY_OPS),
        "unary_operators": ["-"],
        "functions": sorted(VALID_FUNCTIONS),
        "constants": sorted(VALID_CONSTANTS),
        "rounding_modes": list(_ROUNDING_MODES.keys()),
        "limits": {
            "max_input_len": MAX_INPUT_LEN,
            "max_ast_depth": 64,
            "max_ast_nodes": 1024,
            "max_int_pow_exp": 1000,
            "max_result_digits": 100_000,
            "max_output_bytes": MAX_OUTPUT_BYTES,
            "call_timeout_ms": CALL_TIMEOUT_MS,
            "symbolic_timeout_ms": SYMBOLIC_TIMEOUT_MS,
            "numeric_dps_default": NUMERIC_DPS_DEFAULT,
            "numeric_dps_max": NUMERIC_DPS_MAX,
        },
        "plot": {
            "description": "2D numeric function sampling over literal x only. "
                           "Uniform sampling. Exact canonical x coordinates. "
                           "Point-level undefined states for sampled mathematical "
                           "domain errors. No symbolic plotting, no adaptive "
                           "sampling, no inferred discontinuities.",
            "max_sample_count": 10_000,
            "sampling_modes": ["uniform"],
            "assurance_levels": ["numeric-tier"],
            "variable": "x",
        },
    })


@mcp.tool()
async def sample_function(
    expression: str,
    x_min: str = "-10",
    x_max: str = "10",
    sample_count: int = 401,
    precision: int = 50,
    rounding: str = "ROUND_HALF_EVEN",
) -> str:
    """Sample a real-valued 2D function over a uniform grid (numeric tier).

    - Reellwertige 2D-Funktion einer einzelnen Variablen x
    - Deterministisches uniformes Sampling auf endlichem Intervall
    - Exakte kanonische x-Koordinaten
    - Punktweise undefined-States bei mathematischen Domain-Fehlern
    - Kein symbolisches Plotten, kein adaptives Sampling

    Args:
        expression: Function expression with literal x (e.g. 'sin(x)/x').
        x_min: Lower domain bound as canonical decimal string.
        x_max: Upper domain bound as canonical decimal string.
        sample_count: Number of sample points (2 to 10,000).
        precision: Significant decimal digits for y-values.
        rounding: Rounding mode (V1: only ROUND_HALF_EVEN).

    Returns:
        JSON with ok/result: {points: [...], meta: {...}}
        or ok/error on validation failure.
    """
    try:
        from calc_mcp.plot_contract import parse_plot_request
        req = parse_plot_request(
            expression=expression,
            x_min=x_min,
            x_max=x_max,
            sample_count=sample_count,
            precision=precision,
            rounding=rounding,
        )
    except CalculatorError as exc:
        return json.dumps(exc.to_dict())

    try:
        from calc_mcp.core import run_isolated_plot_worker
        result = run_isolated_plot_worker(req)
        return json.dumps(result)
    except CalculatorError as exc:
        return json.dumps(exc.to_dict())


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the FastMCP server over stdio.

    Logging goes to stderr; stdout is the MCP protocol channel.
    """
    # Configure our logger explicitly (root may already have handlers from MCP lib)
    log.setLevel(logging.INFO)
    if not log.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        ))
        log.addHandler(handler)
    log.info("calc-mcp starting (stdio transport)")
    sys.stderr.flush()
    try:
        mcp.run(transport="stdio")
    except Exception:
        log.exception("Fatal server error")
        sys.exit(1)


if __name__ == "__main__":
    main()
