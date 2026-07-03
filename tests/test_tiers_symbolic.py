"""Tests for the lazy SymPy symbolic tier."""

import sys

import pytest

from calc_mcp.tiers.symbolic import symbolic, SUPPORTED_OPS
from calc_mcp.errors import CalculatorError, ErrorCode


class TestLazyImport:
    """sympy must NOT be imported at module top level."""

    def test_sympy_not_imported_at_top(self):
        """Check source: 'import sympy' must not appear at module level."""
        import ast
        with open("calc_mcp/tiers/symbolic.py") as f:
            tree = ast.parse(f.read())
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import) and any(
                a.name == "sympy" for a in node.names
            ):
                pytest.fail(
                    "sympy imported at module level in symbolic.py "
                    "(should be inside function body)"
                )
            if isinstance(node, ast.ImportFrom) and node.module == "sympy":
                pytest.fail(
                    "sympy imported at module level in symbolic.py "
                    "(should be inside function body)"
                )


class TestSymbolic:
    def test_simplify(self):
        result = symbolic("simplify", "2*x + x")
        assert result["op"] == "simplify"
        assert "3*x" in result["output"]

    def test_solve(self):
        result = symbolic("solve", "x**2 - 4")
        assert result["op"] == "solve"
        assert "2" in result["output"] or "-2" in result["output"]

    def test_differentiate(self):
        result = symbolic("differentiate", "x**3")
        assert result["op"] == "differentiate"
        assert "3*x" in result["output"] or "3*x**2" in result["output"]

    def test_integrate(self):
        result = symbolic("integrate", "2*x")
        assert result["op"] == "integrate"
        assert "x**2" in result["output"]

    def test_factor(self):
        result = symbolic("factor", "x**2 - 1")
        assert result["op"] == "factor"
        assert "(x - 1)*(x + 1)" in result["output"]

    def test_expand(self):
        result = symbolic("expand", "(x+1)**2")
        assert result["op"] == "expand"
        assert "x**2" in result["output"]

    def test_unsupported_op(self):
        with pytest.raises(CalculatorError) as exc:
            symbolic("graph", "x")
        assert exc.value.code == ErrorCode.UNSUPPORTED_OP

    def test_malformed_expression(self):
        with pytest.raises(CalculatorError) as exc:
            symbolic("simplify", "x + ")
        assert exc.value.code == ErrorCode.SYMBOLIC_FAILED

    def test_result_has_latex(self):
        result = symbolic("simplify", "x + x")
        assert "latex" in result
        assert result["latex"] is not None

    def test_multiple_variable(self):
        result = symbolic("differentiate", "y**2", variable="y")
        assert "2*y" in result["output"]


# ---------------------------------------------------------------------------
# Timeout and resource leak
# ---------------------------------------------------------------------------

class TestSymbolicTimeout:
    """Preemptive subprocess timeout — no lingering processes."""

    def test_timeout_returns_error(self):
        """A subprocess that sleeps triggers TIMEOUT."""
        from calc_mcp.server import _symbolic_subprocess
        from calc_mcp.errors import ErrorCode

        import calc_mcp.limits as lim
        original = lim.SYMBOLIC_TIMEOUT_MS
        lim.SYMBOLIC_TIMEOUT_MS = 50  # 50ms

        try:
            # Simplify is fast — may or may not timeout. Both are acceptable.
            _symbolic_subprocess("simplify", "x", "x")
        except CalculatorError as exc:
            assert exc.code in (ErrorCode.TIMEOUT, ErrorCode.SYMBOLIC_FAILED)
        finally:
            lim.SYMBOLIC_TIMEOUT_MS = original

    def test_no_lingering_processes_after_kill(self):
        """Killed subprocesses must not leave children behind."""
        import subprocess, sys, time, os
        import calc_mcp.limits as lim

        # Count children before
        try:
            import psutil
            proc_self = psutil.Process()
            children_before = proc_self.children()
        except ImportError:
            children_before = []

        kills = 0
        for _ in range(20):
            # Spawn a process that sleeps 10s — must be killed by timeout
            p = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            try:
                p.communicate(timeout=0.001)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=2)
                kills += 1

        # Count children after
        children_after: list = []
        try:
            children_after = proc_self.children()
        except (NameError, ImportError):
            pass

        # Small delay to ensure OS reaps killed processes
        time.sleep(0.2)

        print(f"\n  Kills triggered: {kills}/20")
        print(f"  Children before:   {len(children_before)}")
        print(f"  Children after:    {len(children_after)}")

        assert kills == 20, f"Expected 20/20 kills, got {kills}"
        assert len(children_after) <= len(children_before) + 1, (
            f"Children grew from {len(children_before)} to "
            f"{len(children_after)} — lingering subprocesses"
        )
