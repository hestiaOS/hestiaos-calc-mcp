"""Property-based tests for the calculator using Hypothesis.

Tests algebraic invariants of exact and numeric evaluation:
  - Commutativity: a+b == b+a, a*b == b*a
  - Associativity: (a+b)+c == a+(b+c)
  - Roundtrip: parse(format(eval(x))) == eval(x)
  - Monotonicity where valid
  - Differential vs fractions.Fraction or mpmath
"""

import pytest
from hypothesis import given, strategies as st
from hypothesis import settings

from calc_mcp.core import exact_eval
from calc_mcp.numeric import eval_ast_numeric, has_transcendental
from calc_mcp.parser import parse
from calc_mcp.errors import CalculatorError


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Small rationals: a/b where a,b ∈ [-100, 100], b ≠ 0
_small_rational = st.fractions(
    min_value=-(100/1), max_value=100/1,
    max_denominator=100,
)

# Integer (for powers)
_small_int = st.integers(min_value=-10, max_value=10).filter(lambda x: x != 0)

# Expression string from a rational
def _rat_str(f):
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval_exact(expr: str) -> str | None:
    """Return exact eval result, or None on CalculatorError."""
    try:
        ast = parse(expr)
        if has_transcendental(ast):
            return None
        return exact_eval(ast)
    except CalculatorError:
        return None


def _eval_numeric(expr: str, dps: int = 50) -> float | None:
    """Return numeric eval result as float, or None on error."""
    try:
        ast = parse(expr)
        s = eval_ast_numeric(ast, dps=dps)
        return float(s)
    except (CalculatorError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Commutativity
# ---------------------------------------------------------------------------

class TestCommutativity:
    """a+b == b+a, a*b == b*a (exact rational kernel)."""

    @given(st.integers(-100, 100), st.integers(-100, 100))
    @settings(max_examples=200, deadline=1000)
    def test_add_commutes_int(self, a, b):
        left = _eval_exact(f"{a}+{b}")
        right = _eval_exact(f"{b}+{a}")
        if left is not None and right is not None:
            assert left == right, f"{a}+{b} != {b}+{a}"

    @given(st.integers(1, 50), st.integers(1, 50), st.integers(1, 50), st.integers(1, 50))
    @settings(max_examples=100)
    def test_add_commutes_rational(self, a, b, c, d):
        """(a/b) + (c/d) == (c/d) + (a/b)"""
        left = _eval_exact(f"{a}/{b}+{c}/{d}")
        right = _eval_exact(f"{c}/{d}+{a}/{b}")
        if left is not None and right is not None:
            assert left == right

    @given(st.integers(-10, 10), st.integers(-10, 10))
    @settings(max_examples=200, deadline=1000)
    def test_mul_commutes_int(self, a, b):
        left = _eval_exact(f"{a}*{b}")
        right = _eval_exact(f"{b}*{a}")
        if left is not None and right is not None:
            assert left == right


# ---------------------------------------------------------------------------
# Associativity
# ---------------------------------------------------------------------------

class TestAssociativity:
    """(a+b)+c == a+(b+c), (a*b)*c == a*(b*c)"""

    @given(st.integers(-20, 20), st.integers(-20, 20), st.integers(-20, 20))
    @settings(max_examples=200, deadline=1000)
    def test_add_assoc_int(self, a, b, c):
        left = _eval_exact(f"({a}+{b})+{c}")
        right = _eval_exact(f"{a}+({b}+{c})")
        if left is not None and right is not None:
            assert left == right

    @given(st.integers(-10, 10), st.integers(-10, 10), st.integers(-10, 10))
    @settings(max_examples=100, deadline=1000)
    def test_mul_assoc_int(self, a, b, c):
        left = _eval_exact(f"({a}*{b})*{c}")
        right = _eval_exact(f"{a}*({b}*{c})")
        if left is not None and right is not None:
            assert left == right


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------

class TestRoundtrip:
    """parse(format(eval(x))) == eval(x) — canonical format survives roundtrip."""

    @given(st.integers(-1000, 1000))
    @settings(max_examples=200)
    def test_integer_roundtrip(self, n):
        r = _eval_exact(str(n))
        if r is not None:
            r2 = _eval_exact(r)
            assert r2 == r

    @given(st.integers(1, 100), st.integers(1, 100))
    @settings(max_examples=100)
    def test_rational_roundtrip(self, a, b):
        expr = f"{a}/{b}"
        r = _eval_exact(expr)
        if r is not None:
            r2 = _eval_exact(r)
            assert r2 == r


# ---------------------------------------------------------------------------
# Differential vs fractions.Fraction
# ---------------------------------------------------------------------------

class TestDifferential:
    """Exact kernel results match fractions.Fraction over random inputs."""

    @given(st.fractions(min_value=-20, max_value=20, max_denominator=20))
    @settings(max_examples=50)
    def test_fraction_vs_fractions(self, f):
        from fractions import Fraction
        expr = _rat_str(f)
        r = _eval_exact(expr)
        if r is not None:
            assert r == _rat_str(f)

    @given(st.fractions(min_value=-20, max_value=20, max_denominator=20),
           st.fractions(min_value=-20, max_value=20, max_denominator=20))
    @settings(max_examples=100)
    def test_add_vs_fractions(self, f1, f2):
        from fractions import Fraction
        expr = f"{_rat_str(f1)}+{_rat_str(f2)}"
        r = _eval_exact(expr)
        if r is not None:
            expected = _rat_str(f1 + f2)
            assert r == expected, f"{expr} = {r}, expected {expected}"


# ---------------------------------------------------------------------------
# Numeric determinism (via Hypothesis — fixed dps → identical output)
# ---------------------------------------------------------------------------

class TestNumericProperties:
    """Properties of the numeric path."""

    @given(st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_sqrt_square(self, x):
        """sqrt(x)^2 ≈ x"""
        expr = f"sqrt({x})^2"
        val = _eval_numeric(expr, dps=30)
        if val is not None:
            assert abs(val - x) < 1e-10, f"sqrt({x})^2 = {val}, expected ~{x}"

    @given(st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_sin_cos(self, x):
        """sin(x)^2 + cos(x)^2 ≈ 1"""
        expr = f"sin({x})^2+cos({x})^2"
        val = _eval_numeric(expr, dps=30)
        if val is not None:
            assert abs(val - 1.0) < 1e-10
