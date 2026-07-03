"""Golden tests — curated inputs with expected outcomes.

Covers edge cases and boundary conditions for all calculation paths.
"""

import json
import pytest

from calc_mcp.errors import CalculatorError, ErrorCode
from calc_mcp.parser import parse
from calc_mcp.core import exact_eval
from calc_mcp.numeric import eval_ast_numeric, has_transcendental
from calc_mcp.tiers.units import convert
from calc_mcp.tiers.symbolic import symbolic


# ---------------------------------------------------------------------------
# Exact kernel — golden
# ---------------------------------------------------------------------------

class TestGoldenExact:
    def test_div_by_zero(self):
        with pytest.raises(CalculatorError) as exc:
            ast = parse("1/0")
            exact_eval(ast)
        assert exc.value.code == ErrorCode.DIV_BY_ZERO

    def test_very_large_number(self):
        ast = parse("10^20")
        r = exact_eval(ast)
        assert r == "100000000000000000000"

    def test_very_small_rational(self):
        """1/1000000 is exact."""
        ast = parse("1/1000000")
        r = exact_eval(ast)
        assert r == "1/1000000"

    def test_negative_exponent(self):
        ast = parse("(1/2)^(-2)")
        r = exact_eval(ast)
        assert r == "4"

    def test_zero_numerator(self):
        ast = parse("0/5")
        r = exact_eval(ast)
        assert r == "0"

    def test_negative_denominator(self):
        ast = parse("1/-2+1/2")
        r = exact_eval(ast)
        assert r == "0"

    def test_magnitude_limit_exceeded(self):
        from calc_mcp.limits import MAX_INT_POW_EXP
        with pytest.raises(CalculatorError) as exc:
            ast = parse(f"2^{MAX_INT_POW_EXP + 1}")
            exact_eval(ast)
        assert exc.value.code == ErrorCode.LIMIT_EXCEEDED


# ---------------------------------------------------------------------------
# Numeric — golden
# ---------------------------------------------------------------------------

class TestGoldenNumeric:
    def test_sqrt_negative_domain(self):
        with pytest.raises(CalculatorError) as exc:
            ast = parse("sqrt(-1)")
            eval_ast_numeric(ast)
        assert exc.value.code == ErrorCode.DOMAIN_ERROR

    def test_ln_zero_domain(self):
        with pytest.raises(CalculatorError) as exc:
            ast = parse("ln(0)")
            eval_ast_numeric(ast)
        assert exc.value.code == ErrorCode.DOMAIN_ERROR

    def test_log_negative(self):
        with pytest.raises(CalculatorError) as exc:
            ast = parse("log(-5)")
            eval_ast_numeric(ast)
        assert exc.value.code == ErrorCode.DOMAIN_ERROR

    def test_asin_out_of_range(self):
        with pytest.raises(CalculatorError) as exc:
            ast = parse("asin(2)")
            eval_ast_numeric(ast)
        assert exc.value.code == ErrorCode.DOMAIN_ERROR

    def test_pi_value(self):
        ast = parse("pi")
        s = eval_ast_numeric(ast, dps=10)
        val = float(s)
        assert abs(val - 3.1415926535) < 1e-9

    def test_e_value(self):
        ast = parse("e")
        s = eval_ast_numeric(ast, dps=10)
        val = float(s)
        assert abs(val - 2.7182818284) < 1e-9

    def test_large_precision(self):
        ast = parse("pi")
        from calc_mcp.limits import NUMERIC_DPS_MAX
        with pytest.raises(CalculatorError) as exc:
            eval_ast_numeric(ast, dps=NUMERIC_DPS_MAX + 1)
        assert exc.value.code == ErrorCode.LIMIT_EXCEEDED


# ---------------------------------------------------------------------------
# Parser — golden
# ---------------------------------------------------------------------------

class TestGoldenParser:
    def test_empty_input(self):
        with pytest.raises(CalculatorError) as exc:
            parse("")
        assert exc.value.code == ErrorCode.PARSE_ERROR

    def test_only_whitespace(self):
        with pytest.raises(CalculatorError) as exc:
            parse("   ")
        assert exc.value.code == ErrorCode.PARSE_ERROR

    def test_trailing_garbage(self):
        with pytest.raises(CalculatorError) as exc:
            parse("1+2abc")
        assert exc.value.code in (ErrorCode.PARSE_ERROR, ErrorCode.UNKNOWN_SYMBOL)

    def test_unicode_rejected(self):
        with pytest.raises(CalculatorError) as exc:
            parse("π")  # Greek pi != identifier "pi"
        assert exc.value.code in (ErrorCode.PARSE_ERROR, ErrorCode.UNKNOWN_SYMBOL)

    def test_decimal_notation(self):
        """Decimal numbers are valid but route to numeric tier."""
        ast = parse("3.14")
        assert has_transcendental(ast)  # decimals are "transcendental" for routing


# ---------------------------------------------------------------------------
# Units — golden
# ---------------------------------------------------------------------------

class TestGoldenUnits:
    def test_incompatible_units(self):
        with pytest.raises(CalculatorError) as exc:
            convert("1", "meter", "kilogram")
        assert exc.value.code == ErrorCode.INCOMPATIBLE_UNITS

    def test_unknown_from_unit(self):
        with pytest.raises(CalculatorError) as exc:
            convert("1", "flurbo", "meter")
        assert exc.value.code == ErrorCode.UNKNOWN_UNIT

    def test_meter_to_foot(self):
        r = convert("1", "meter", "foot")
        assert abs(r["value"] - 3.28084) < 0.001

    def test_celsius_fahrenheit(self):
        r = convert("0", "degC", "degF")
        assert abs(r["value"] - 32.0) < 0.001


# ---------------------------------------------------------------------------
# Symbolic — golden
# ---------------------------------------------------------------------------

class TestGoldenSymbolic:
    def test_unsupported_op(self):
        with pytest.raises(CalculatorError) as exc:
            symbolic("graph", "x")
        assert exc.value.code == ErrorCode.UNSUPPORTED_OP

    def test_malformed_expression(self):
        with pytest.raises(CalculatorError) as exc:
            symbolic("simplify", "x + ")
        assert exc.value.code == ErrorCode.SYMBOLIC_FAILED

    def test_solve_quadratic(self):
        r = symbolic("solve", "x**2 - 4")
        assert "2" in str(r["output"])
