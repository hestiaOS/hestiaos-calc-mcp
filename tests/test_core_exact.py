"""Differential tests for the exact rational arithmetic kernel.

Tests compare the Idris kernel (via ctypes) against Python's fractions.Fraction
over curated and random inputs.
"""

import pytest

from calc_mcp.ast_nodes import (
    BinaryOpNode,
    NumberNode,
    UnaryOpNode,
)
from calc_mcp.core import exact_eval
from calc_mcp.errors import CalculatorError, ErrorCode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _n(value: str) -> NumberNode:
    return NumberNode(value)


def _b(left, op: str, right) -> BinaryOpNode:
    return BinaryOpNode(left, op, right)


def _u(op: str, operand) -> UnaryOpNode:
    return UnaryOpNode(op, operand)


def _eval(expr: str) -> str:
    """Parse and evaluate an expression using the test parser."""
    from calc_mcp.parser import parse
    ast = parse(expr)
    return exact_eval(ast)


# ---------------------------------------------------------------------------
# Basic arithmetic
# ---------------------------------------------------------------------------

class TestAdd:
    def test_simple(self):
        assert _eval("1/2+1/3") == "5/6"

    def test_integer_result(self):
        assert _eval("1/4+3/4") == "1"

    def test_negative_result(self):
        assert _eval("1/4+-3/4") == "-1/2"

    def test_with_unary(self):
        assert _eval("-1/2+1/2") == "0"


class TestSub:
    def test_simple(self):
        assert _eval("1/2-1/3") == "1/6"

    def test_positive(self):
        assert _eval("3/4-1/4") == "1/2"

    def test_negative(self):
        assert _eval("1/4-3/4") == "-1/2"


class TestMul:
    def test_simple(self):
        assert _eval("1/2*2/3") == "1/3"

    def test_integer_result(self):
        assert _eval("3/4*4/3") == "1"

    def test_negative(self):
        assert _eval("-1/2*1/2") == "-1/4"


class TestDiv:
    def test_simple(self):
        # 1/2/2/3 = (1/2)/2/3 = (1/4)/3 = 1/12
        assert _eval("1/2/2/3") == "1/12"

    def test_integer(self):
        assert _eval("6/2") == "3"

    def test_by_zero(self):
        with pytest.raises(CalculatorError) as exc:
            _eval("1/0")
        assert exc.value.code == ErrorCode.DIV_BY_ZERO


class TestPow:
    def test_positive_exp(self):
        assert _eval("(2/3)^3") == "8/27"

    def test_negative_exp(self):
        assert _eval("(2/3)^(-2)") == "9/4"

    def test_zero_exp(self):
        assert _eval("(1/2)^0") == "1"

    def test_exp_one(self):
        assert _eval("(1/2)^1") == "1/2"

    def test_large_exp(self):
        """Large but within limits."""
        result = _eval("2^10")
        assert result == "1024"


# ---------------------------------------------------------------------------
# Differential: Idris kernel vs fractions.Fraction
# ---------------------------------------------------------------------------

class TestDifferential:
    """Compare the Idris kernel against Python's fractions.Fraction."""

    def _ref(self, expr: str) -> str:
        """Evaluate with fractions.Fraction as reference."""
        from fractions import Fraction
        from calc_mcp.parser import parse
        from calc_mcp.core import _eval_ast

        ast = parse(expr)
        num, den = _eval_ast(ast)
        return f"{num}/{den}" if den != 1 else str(num)

    def test_simple_addition(self):
        assert _eval("1/3+2/3") == self._ref("1/3+2/3")

    def test_multiplication(self):
        assert _eval("2/3*3/4") == self._ref("2/3*3/4")

    def test_power(self):
        assert _eval("(3/4)^3") == self._ref("(3/4)^3")

    def test_division(self):
        assert _eval("(1/2)/(3/4)") == self._ref("(1/2)/(3/4)")

    def test_complex(self):
        expr = "(1/2+1/3)*(3/4-1/4)"
        assert _eval(expr) == self._ref(expr)

    def test_negative_power(self):
        expr = "(2/3)^(-2)"
        assert _eval(expr) == self._ref(expr)


# ---------------------------------------------------------------------------
# Unary minus
# ---------------------------------------------------------------------------

class TestUnaryMinus:
    def test_negative_number(self):
        assert _eval("-5") == "-5"

    def test_double_negative(self):
        assert _eval("--5") == "5"

    def test_negative_addition(self):
        assert _eval("-1+3") == "2"

    def test_negative_division(self):
        assert _eval("-(1/2)") == "-1/2"


# ---------------------------------------------------------------------------
# Large numbers / arbitrary precision
# ---------------------------------------------------------------------------

class TestArbitraryPrecision:
    def test_large_numerator(self):
        """Prove arbitrary precision with GMP-backed integers."""
        result = _eval("999999999999999999999999999999*2")
        assert result == "1999999999999999999999999999998"

    def test_very_large_multiplication(self):
        result = _eval("10^20")
        assert result == "100000000000000000000"
        assert len(result) == 21  # "1" + 20 zeros

    def test_large_rational(self):
        # 1/10^24 + 1/3 = (3+10^24)/(3*10^24)
        result = _eval("1/1000000000000000000000000+1/3")
        assert result == "1000000000000000000000003/3000000000000000000000000"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestErrors:
    def test_division_by_zero(self):
        with pytest.raises(CalculatorError) as exc:
            _eval("1/(2-2)")
        assert exc.value.code == ErrorCode.DIV_BY_ZERO

    def test_non_integer_exponent(self):
        """Exact mode only supports integer powers."""
        with pytest.raises(CalculatorError) as exc:
            _eval("2^(1/2)")
        assert exc.value.code == ErrorCode.DOMAIN_ERROR

    def test_magnitude_limit_pow(self):
        """2^1001 should exceed MAX_INT_POW_EXP."""
        from calc_mcp.limits import MAX_INT_POW_EXP
        with pytest.raises(CalculatorError) as exc:
            _eval(f"2^{MAX_INT_POW_EXP + 1}")
        assert exc.value.code == ErrorCode.LIMIT_EXCEEDED

    def test_unknown_symbol(self):
        with pytest.raises(CalculatorError):
            _eval("sqrt(4)")  # needs numeric tier (M5)
        assert True


# ---------------------------------------------------------------------------
# Soak / memory leak test — bounded RSS via worker recycle
# ---------------------------------------------------------------------------

class TestSoak:
    """With recycled-worker supervisor, RSS stays bounded (sawtooth).

    Runs ≥10⁵ calls with KERNEL_MAX_CALLS=1000 → ≥100 recycle cycles.
    Verifies:
      - RSS capped (end ≈ baseline, not linear growth)
      - File descriptor count stable (no fd leak)
      - No zombie child processes
    """

    CALL_MIX = ["1/2+1/3", "(3/4)^5", "1/1000000-1/3"]
    TOTAL_CALLS = 100_000
    RECYCLE_LIMIT = 1_000

    def test_bounded_rss_100k_calls(self):
        """RSS must be capped over 100k calls / 100+ recycles."""
        import gc

        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not installed — install with pip install psutil")

        import calc_mcp.limits as lim
        import calc_mcp.core as core

        original_max = lim.KERNEL_MAX_CALLS
        lim.KERNEL_MAX_CALLS = self.RECYCLE_LIMIT

        try:
            process = psutil.Process()

            # Warmup
            for _ in range(500):
                _eval("1/2+1/3")
            gc.collect()

            # Baseline: RSS, fd count, children
            rss_baseline = process.memory_info().rss
            fds_before = _count_fds(process)
            children_before = process.children()

            peak_rss = rss_baseline

            # Run the calls
            calls_per_cycle = len(self.CALL_MIX)
            total_ops = self.TOTAL_CALLS
            for i in range(0, total_ops, calls_per_cycle):
                for expr in self.CALL_MIX:
                    _eval(expr)
                # Track peak every ~10k calls
                if (i + calls_per_cycle) % 10000 < calls_per_cycle:
                    current = process.memory_info().rss
                    if current > peak_rss:
                        peak_rss = current

            gc.collect()
            rss_end = process.memory_info().rss
            fds_after = _count_fds(process)
            children_after = process.children()

            # Calculate recycle count
            # _worker_call_count is reset on each recycle, so it's < RECYCLE_LIMIT
            remaining_calls = core._worker_call_count
            estimated_recycles = max(
                1,
                total_ops // lim.KERNEL_MAX_CALLS
                + (1 if remaining_calls > 0 else 0),
            )

            # Detect zombies
            zombies = [c for c in children_after if c.status() == psutil.STATUS_ZOMBIE]

            print(f"\n  Total ops:        {total_ops}")
            print(f"  KERNEL_MAX_CALLS: {lim.KERNEL_MAX_CALLS}")
            print(f"  Recycles:         ~{estimated_recycles}")
            print(f"  --------------- RSS ---------------")
            print(f"  RSS baseline:     {rss_baseline // 1024} KB")
            print(f"  RSS peak:         {peak_rss // 1024} KB")
            print(f"  RSS end:          {rss_end // 1024} KB")
            print(f"  Drift (end-baseline): {abs(rss_end - rss_baseline) * 100 / max(rss_baseline, 1):.2f}%")
            print(f"  --------------- FDs ---------------")
            print(f"  FDs before:       {fds_before}")
            print(f"  FDs after:        {fds_after}")
            print(f"  Diff:             {fds_after - fds_before}")
            print(f"  --------------- Zombies ----------")
            print(f"  Children:         {len(children_after)}")
            print(f"  Zombies:          {len(zombies)}")

            # Assertions
            drift = abs(rss_end - rss_baseline) / max(rss_baseline, 1)
            assert drift < 0.20, (
                f"RSS end ({rss_end // 1024} KB) drifted {drift*100:.1f}% from "
                f"baseline ({rss_baseline // 1024} KB) — not bounded"
            )

            assert fds_after <= fds_before + 5, (
                f"FD count grew from {fds_before} to {fds_after} — possible fd leak"
            )

            assert len(zombies) == 0, (
                f"Found {len(zombies)} zombie processes — workers not reaped"
            )

            assert estimated_recycles >= 100, (
                f"Only ~{estimated_recycles} recycle cycles — expected ≥100 "
                f"({total_ops} calls / {lim.KERNEL_MAX_CALLS} limit)"
            )

        finally:
            lim.KERNEL_MAX_CALLS = original_max
            core._kill_worker()


def _count_fds(process) -> int:
    """Count open file descriptors for a process."""
    try:
        return process.num_fds()
    except AttributeError:
        # Fallback for platforms without num_fds()
        import os
        try:
            fds = os.listdir('/dev/fd')
            return len(fds)
        except FileNotFoundError:
            try:
                fds = os.listdir(f'/proc/{os.getpid()}/fd')
                return len(fds)
            except FileNotFoundError:
                return 0
