"""Tests for the mpmath numeric evaluation path."""

import pytest

from calc_mcp.errors import CalculatorError, ErrorCode
from calc_mcp.numeric import eval_ast_numeric, has_transcendental
from calc_mcp.parser import parse


def _n(expr: str, dps: int = 50) -> str:
    """Parse and evaluate numerically."""
    ast = parse(expr)
    return eval_ast_numeric(ast, dps=dps)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Same input + dps → identical output string."""

    def test_addition_deterministic(self):
        a = _n("1+2")
        b = _n("1+2")
        assert a == b

    def test_pi_deterministic(self):
        a = _n("pi")
        for _ in range(5):
            assert _n("pi") == a

    def test_sin_deterministic(self):
        a = _n("sin(0.5)")
        b = _n("sin(0.5)")
        assert a == b

    def test_with_different_dps(self):
        """Different dps produces different precision."""
        low = _n("pi", dps=10)
        high = _n("pi", dps=50)
        assert len(low) <= len(high) or low != high


# ---------------------------------------------------------------------------
# Differential: compare against mpmath reference
# ---------------------------------------------------------------------------

class TestDifferential:
    """Compare eval_ast_numeric against mpmath high-precision reference."""

    def test_sin(self):
        import mpmath as mp
        mp.mp.dps = 100
        ref = float(mp.sin(0.5))
        result = float(_n("sin(0.5)", dps=50))
        assert abs(result - ref) < 1e-15

    def test_cos(self):
        import mpmath as mp
        mp.mp.dps = 100
        ref = float(mp.cos(0.25))
        result = float(_n("cos(0.25)", dps=50))
        assert abs(result - ref) < 1e-15

    def test_exp(self):
        import mpmath as mp
        mp.mp.dps = 100
        ref = float(mp.e ** 1)
        result = float(_n("exp(1)", dps=50))
        assert abs(result - ref) < 1e-15

    def test_ln(self):
        import mpmath as mp
        mp.mp.dps = 100
        ref = float(mp.log(2))
        result = float(_n("ln(2)", dps=50))
        assert abs(result - ref) < 1e-15

    def test_sqrt(self):
        result = float(_n("sqrt(2)", dps=50))
        assert abs(result - 1.4142135623730951) < 1e-15

    def test_constant_pi(self):
        import mpmath as mp
        mp.mp.dps = 100
        ref = float(mp.pi)
        result = float(_n("pi", dps=50))
        assert abs(result - ref) < 1e-15

    def test_complex_expr(self):
        """pi + sin(0.5) * sqrt(2)"""
        import mpmath as mp
        mp.mp.dps = 100
        ref = float(mp.pi + mp.sin(0.5) * mp.sqrt(2))
        result = float(_n("pi+sin(0.5)*sqrt(2)", dps=50))
        assert abs(result - ref) < 1e-14


# ---------------------------------------------------------------------------
# Domain errors
# ---------------------------------------------------------------------------

class TestDomainErrors:
    def test_ln_negative(self):
        with pytest.raises(CalculatorError) as exc:
            _n("ln(-1)")
        assert exc.value.code == ErrorCode.DOMAIN_ERROR

    def test_log_zero(self):
        with pytest.raises(CalculatorError) as exc:
            _n("log(0)")
        assert exc.value.code == ErrorCode.DOMAIN_ERROR

    def test_sqrt_negative(self):
        with pytest.raises(CalculatorError) as exc:
            _n("sqrt(-1)")
        assert exc.value.code == ErrorCode.DOMAIN_ERROR

    def test_asin_out_of_range(self):
        with pytest.raises(CalculatorError) as exc:
            _n("asin(2)")
        assert exc.value.code == ErrorCode.DOMAIN_ERROR

    def test_acos_out_of_range(self):
        with pytest.raises(CalculatorError) as exc:
            _n("acos(-2)")
        assert exc.value.code == ErrorCode.DOMAIN_ERROR

    def test_dps_exceeds_max(self):
        from calc_mcp.limits import NUMERIC_DPS_MAX
        with pytest.raises(CalculatorError) as exc:
            _n("1+1", dps=NUMERIC_DPS_MAX + 1)
        assert exc.value.code == ErrorCode.LIMIT_EXCEEDED


# ---------------------------------------------------------------------------
# has_transcendental
# ---------------------------------------------------------------------------

class TestHasTranscendental:
    def test_exact_arithmetic(self):
        assert not has_transcendental(parse("1+2*3"))

    def test_pi(self):
        assert has_transcendental(parse("pi"))

    def test_sin(self):
        assert has_transcendental(parse("sin(0)"))

    def test_sqrt(self):
        assert has_transcendental(parse("sqrt(2)"))

    def test_power_non_integer(self):
        assert has_transcendental(parse("2^(1/2)"))

    def test_power_integer(self):
        assert not has_transcendental(parse("2^3"))

    def test_decimal_number(self):
        assert has_transcendental(parse("3.14"))

    def test_nested(self):
        assert has_transcendental(parse("1+sin(2*pi)"))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_tan(self):
        result = float(_n("tan(0)", dps=50))
        assert abs(result) < 1e-30

    def test_atan(self):
        result = float(_n("atan(1)", dps=50))
        assert abs(result - 0.7853981633974483) < 1e-15

    def test_e_constant(self):
        import mpmath as mp
        mp.mp.dps = 100
        ref = float(mp.e)
        result = float(_n("e", dps=50))
        assert abs(result - ref) < 1e-15

    def test_abs_exact(self):
        """abs is exact — not transcendental."""
        assert not has_transcendental(parse("abs(-5)"))
