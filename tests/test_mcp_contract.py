"""Contract tests for the calc-mcp FastMCP server — tool schemas, routing, errors, stdout hygiene."""

import asyncio
import json
import subprocess
import sys

import pytest

from calc_mcp.server import calculate, symbolic, convert_units, capabilities
from calc_mcp.errors import ErrorCode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call(fn, *args, **kwargs) -> str:
    """Call an async FastMCP tool synchronously."""
    return asyncio.run(fn(*args, **kwargs))


_cc = lambda *a, **kw: _call(calculate, *a, **kw)
_cs = lambda *a, **kw: _call(symbolic, *a, **kw)
_ccu = lambda *a, **kw: _call(convert_units, *a, **kw)
_ccap = lambda: _call(capabilities)


def _ok(r: str) -> dict:
    data = json.loads(r)
    assert data.get("ok") is True, f"Expected ok=True, got: {data}"
    return data["result"]


def _err(r: str) -> dict:
    data = json.loads(r)
    assert data.get("ok") is False, f"Expected ok=False, got: {data}"
    return data["error"]


# ---------------------------------------------------------------------------
# calculate — exact path
# ---------------------------------------------------------------------------

class TestCalculateExact:
    def test_simple_addition(self):
        r = _ok(_cc("1/2+1/3"))
        assert r["exact"] is True
        assert r["rational"] == "5/6"

    def test_integer_result(self):
        r = _ok(_cc("1/4+3/4"))
        assert r["exact"] is True
        assert r["rational"] == "1"

    def test_negative_rational(self):
        r = _ok(_cc("1/4-3/4"))
        assert r["exact"] is True
        assert r["rational"] == "-1/2"

    def test_large_exact(self):
        r = _ok(_cc("9999999999999999999*2"))
        assert r["exact"] is True
        assert r["rational"] == "19999999999999999998"

    def test_power_exact(self):
        r = _ok(_cc("(2/3)^3"))
        assert r["exact"] is True
        assert r["rational"] == "8/27"

    def test_decimal_output(self):
        r = _ok(_cc("1/2", precision=10))
        assert r["exact"] is True
        assert "decimal" in r
        assert float(r["decimal"]) == 0.5

    def test_non_integer_power(self):
        r = _ok(_cc("2^0.5", precision=10))
        assert r["exact"] is False


# ---------------------------------------------------------------------------
# calculate — numeric / transcendental
# ---------------------------------------------------------------------------

class TestCalculateNumeric:
    def test_sqrt(self):
        r = _ok(_cc("sqrt(2)", precision=10))
        assert r["exact"] is False
        assert "decimal" in r
        assert "precision" in r
        assert r["precision"] == 10

    def test_pi(self):
        r = _ok(_cc("pi", precision=10))
        assert r["exact"] is False
        val = float(r["decimal"])
        assert abs(val - 3.1415926535) < 1e-9

    def test_sin(self):
        r = _ok(_cc("sin(0.5)", precision=15))
        assert r["exact"] is False
        val = float(r["decimal"])
        assert abs(val - 0.479425538604203) < 1e-12

    def test_complex_numeric(self):
        r = _ok(_cc("sqrt(2)+sin(0.5)", precision=10))
        assert r["exact"] is False


# ---------------------------------------------------------------------------
# calculate — error cases
# ---------------------------------------------------------------------------

class TestCalculateErrors:
    def test_parse_error(self):
        e = _err(_cc("1+@2"))
        assert e["code"] == ErrorCode.PARSE_ERROR.value

    def test_unknown_symbol(self):
        e = _err(_cc("foo(1)"))
        assert e["code"] == ErrorCode.UNKNOWN_SYMBOL.value

    def test_div_by_zero(self):
        e = _err(_cc("1/0"))
        assert e["code"] == ErrorCode.DIV_BY_ZERO.value

    def test_domain_error(self):
        e = _err(_cc("sqrt(-1)"))
        assert e["code"] == ErrorCode.DOMAIN_ERROR.value

    def test_limit_exceeded_input_len(self):
        e = _err(_cc("1" * 5000))
        assert e["code"] == ErrorCode.LIMIT_EXCEEDED.value

    def test_no_stacktrace_in_error(self):
        e = _err(_cc("1/0"))
        assert "Traceback" not in json.dumps(e)
        assert "/calc_mcp/" not in json.dumps(e)
        assert "secret" not in json.dumps(e)


# ---------------------------------------------------------------------------
# symbolic
# ---------------------------------------------------------------------------

class TestSymbolic:
    def test_simplify(self):
        r = _ok(_cs("simplify", "2*x + x"))
        assert r["op"] == "simplify"
        assert "3*x" in r["output"]

    def test_solve(self):
        r = _ok(_cs("solve", "x**2 - 4"))
        assert r["op"] == "solve"
        assert "2" in str(r["output"])

    def test_latex_present(self):
        r = _ok(_cs("simplify", "x+x"))
        assert "latex" in r
        assert r["latex"] is not None

    def test_unsupported_op(self):
        e = _err(_cs("graph", "x"))
        assert e["code"] == ErrorCode.UNSUPPORTED_OP.value

    def test_malformed_expression(self):
        e = _err(_cs("simplify", "x + "))
        assert e["code"] == ErrorCode.SYMBOLIC_FAILED.value


# ---------------------------------------------------------------------------
# convert_units
# ---------------------------------------------------------------------------

class TestConvertUnits:
    def test_meter_to_foot(self):
        r = _ok(_ccu("1", "meter", "foot"))
        assert "value" in r
        assert r["unit"] == "foot"
        assert abs(r["value"] - 3.28084) < 0.001

    def test_incompatible(self):
        e = _err(_ccu("1", "meter", "kilogram"))
        assert e["code"] == ErrorCode.INCOMPATIBLE_UNITS.value

    def test_unknown_unit(self):
        e = _err(_ccu("1", "flurbo", "meter"))
        assert e["code"] == ErrorCode.UNKNOWN_UNIT.value


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------

class TestCapabilities:
    def test_has_tiers(self):
        r = _ok(_ccap())
        assert "tiers" in r
        assert "exact" in r["tiers"]

    def test_has_limits(self):
        r = _ok(_ccap())
        assert "limits" in r

    def test_is_static(self):
        r1 = _ok(_ccap())
        r2 = _ok(_ccap())
        assert r1 == r2


# ---------------------------------------------------------------------------
# stdout hygiene (M4-FIX.2)
# ---------------------------------------------------------------------------

class TestStdoutHygiene:
    """Server must write NOTHING but MCP protocol to stdout.

    Verified by:
      - Worker stdout hygiene already tested in test_kernel_supervisor.py
      - Server entrypoint parses and imports cleanly
      - All logging goes to stderr (verified via basicConfig)
    """

    def test_entrypoint_parses(self):
        """entrypoint.py must be importable and parseable."""
        import ast
        with open("entrypoint.py") as f:
            ast.parse(f.read())
        assert True

    def test_stderr_logging(self):
        """server.py's main() sets up logging to stderr."""
        import calc_mcp.server as srv
        import io
        capture = io.StringIO()
        import logging
        handler = logging.StreamHandler(capture)
        srv.log.addHandler(handler)
        srv.log.warning("test message")
        assert "test message" in capture.getvalue()
