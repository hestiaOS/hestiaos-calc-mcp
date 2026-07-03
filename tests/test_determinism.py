"""Determinism tests — same input → identical output.

Covers exact kernel, numeric path, and MCP server tool outputs.
"""

import json

import pytest

from calc_mcp.parser import parse
from calc_mcp.core import exact_eval
from calc_mcp.numeric import eval_ast_numeric, has_transcendental
from calc_mcp.errors import CalculatorError
from calc_mcp.server import calculate, symbolic, convert_units, capabilities


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import asyncio

def _tool(fn, *args, **kwargs) -> str:
    return asyncio.run(fn(*args, **kwargs))

_cc = lambda *a, **kw: _tool(calculate, *a, **kw)
_cs = lambda *a, **kw: _tool(symbolic, *a, **kw)


# ---------------------------------------------------------------------------
# Exact kernel determinism
# ---------------------------------------------------------------------------

class TestExactDeterminism:
    """Exact kernel returns identical string for same input over repeats."""

    EXPRESSIONS = [
        "1/2+1/3",
        "(3/4)^5",
        "1/1000000-1/3",
        "9999999999999999999*2",
        "-5",
        "--5",
        "1/2*-2/3",
    ]

    def test_exact_kernel(self):
        for expr in self.EXPRESSIONS:
            ast = parse(expr)
            first = exact_eval(ast)
            for _ in range(20):
                assert exact_eval(ast) == first, f"Not deterministic: {expr}"

    def test_exact_via_server(self):
        for expr in self.EXPRESSIONS:
            first = _cc(expr)
            for _ in range(10):
                assert _cc(expr) == first, f"Server not deterministic: {expr}"


# ---------------------------------------------------------------------------
# Numeric determinism
# ---------------------------------------------------------------------------

class TestNumericDeterminism:
    """Numeric path returns identical string for same input+dps."""

    EXPRESSIONS = [
        "sqrt(2)",
        "sin(0.5)",
        "pi",
        "e",
        "ln(2)",
        "cos(0.25)+sin(0.25)",
        "sqrt(2)^2",
        "1.5+2.5",
    ]

    def test_numeric_path(self):
        for expr in self.EXPRESSIONS:
            ast = parse(expr)
            first = eval_ast_numeric(ast, dps=50)
            for _ in range(10):
                result = eval_ast_numeric(ast, dps=50)
                assert result == first, f"Numeric not deterministic: {expr}"

    def test_numeric_via_server(self):
        for expr in self.EXPRESSIONS:
            first = _cc(expr, precision=50)
            for _ in range(5):
                assert _cc(expr, precision=50) == first, (
                    f"Server numeric not deterministic: {expr}"
                )


# ---------------------------------------------------------------------------
# Symbolic determinism
# ---------------------------------------------------------------------------

class TestSymbolicDeterminism:
    """Symbolic path returns identical output for same input."""

    def test_simplify(self):
        first = _cs("simplify", "2*x + x")
        for _ in range(5):
            assert _cs("simplify", "2*x + x") == first

    def test_capabilities(self):
        first = _tool(capabilities)
        for _ in range(5):
            assert _tool(capabilities) == first


# ---------------------------------------------------------------------------
# Cross-path consistency (exact vs numeric for exact-only expressions)
# ---------------------------------------------------------------------------

class TestCrossPath:
    """For purely exact expressions, numeric and exact paths must agree."""

    def test_exact_rational_matches_numeric(self):
        expr = "1/2+1/3"
        ast = parse(expr)
        assert not has_transcendental(ast)

        exact_r = exact_eval(ast)
        numeric_r = eval_ast_numeric(ast, dps=50)

        exact_decimal = float(exact_r) if "/" not in exact_r else (
            int(exact_r.split("/")[0]) / int(exact_r.split("/")[1])
        )
        numeric_val = float(numeric_r)
        assert abs(exact_decimal - numeric_val) < 1e-15, (
            f"Exact {exact_r} ({exact_decimal}) != numeric {numeric_val} for {expr}"
        )
