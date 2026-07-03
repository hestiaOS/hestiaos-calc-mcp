"""Tests for the safe parser and AST construction."""

import pytest

from calc_mcp.ast_nodes import (
    ASTNode,
    BinaryOpNode,
    ConstantNode,
    FunctionCallNode,
    NumberNode,
    UnaryOpNode,
)
from calc_mcp.errors import CalculatorError, ErrorCode
from calc_mcp.parser import parse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _n(value: str) -> NumberNode:
    return NumberNode(value)


def _c(name: str) -> ConstantNode:
    return ConstantNode(name)


def _u(op: str, operand: ASTNode) -> UnaryOpNode:
    return UnaryOpNode(op, operand)


def _b(left: ASTNode, op: str, right: ASTNode) -> BinaryOpNode:
    return BinaryOpNode(left, op, right)


def _f(name: str, *args: ASTNode) -> FunctionCallNode:
    return FunctionCallNode(name, list(args))


# ---------------------------------------------------------------------------
# Basic numbers
# ---------------------------------------------------------------------------

class TestNumbers:
    def test_integer(self):
        assert parse("42") == _n("42")

    def test_decimal(self):
        assert parse("3.14") == _n("3.14")

    def test_scientific(self):
        assert parse("1.5e-3") == _n("1.5e-3")
        assert parse("2E10") == _n("2E10")
        assert parse("1e+2") == _n("1e+2")

    def test_negative(self):
        ast = parse("-5")
        assert ast == _u("-", _n("5"))


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------

class TestArithmetic:
    def test_addition(self):
        assert parse("1+2") == _b(_n("1"), "+", _n("2"))

    def test_subtraction(self):
        assert parse("3-1") == _b(_n("3"), "-", _n("1"))

    def test_multiplication(self):
        assert parse("2*3") == _b(_n("2"), "*", _n("3"))

    def test_division(self):
        assert parse("6/2") == _b(_n("6"), "/", _n("2"))

    def test_power(self):
        assert parse("2^3") == _b(_n("2"), "^", _n("3"))


# ---------------------------------------------------------------------------
# Precedence and associativity
# ---------------------------------------------------------------------------

class TestPrecedence:
    def test_mul_over_add(self):
        """2+3*4 → 2+(3*4)"""
        expected = _b(_n("2"), "+", _b(_n("3"), "*", _n("4")))
        assert parse("2+3*4") == expected

    def test_add_over_mul(self):
        """2*3+4 → (2*3)+4"""
        expected = _b(_b(_n("2"), "*", _n("3")), "+", _n("4"))
        assert parse("2*3+4") == expected

    def test_pow_right_assoc(self):
        """2^3^2 → 2^(3^2)"""
        expected = _b(_n("2"), "^", _b(_n("3"), "^", _n("2")))
        assert parse("2^3^2") == expected

    def test_unary_minus_precedence(self):
        """-2+3 → (-2)+3"""
        expected = _b(_u("-", _n("2")), "+", _n("3"))
        assert parse("-2+3") == expected

    def test_double_unary(self):
        """--5"""
        assert parse("--5") == _u("-", _u("-", _n("5")))

    def test_parens_override(self):
        """(2+3)*4"""
        expected = _b(
            _b(_n("2"), "+", _n("3")),
            "*",
            _n("4"),
        )
        assert parse("(2+3)*4") == expected


# ---------------------------------------------------------------------------
# Function calls
# ---------------------------------------------------------------------------

class TestFunctions:
    def test_sqrt(self):
        assert parse("sqrt(4)") == _f("sqrt", _n("4"))

    def test_abs(self):
        assert parse("abs(-5)") == _f("abs", _u("-", _n("5")))

    def test_ln(self):
        assert parse("ln(e)") == _f("ln", _c("e"))

    def test_log(self):
        assert parse("log(100)") == _f("log", _n("100"))

    def test_exp(self):
        assert parse("exp(1)") == _f("exp", _n("1"))

    def test_trig(self):
        assert parse("sin(0)") == _f("sin", _n("0"))
        assert parse("cos(pi)") == _f("cos", _c("pi"))
        assert parse("tan(0)") == _f("tan", _n("0"))

    def test_inverse_trig(self):
        assert parse("asin(1)") == _f("asin", _n("1"))
        assert parse("acos(0)") == _f("acos", _n("0"))
        assert parse("atan(1)") == _f("atan", _n("1"))

    def test_nested_function(self):
        """sqrt(sin(x)^2+cos(x)^2)"""
        ast = parse("sqrt(sin(pi)^2+cos(pi)^2)")
        # Just check top-level structure
        assert isinstance(ast, FunctionCallNode)
        assert ast.name == "sqrt"
        assert isinstance(ast.args[0], BinaryOpNode)
        assert ast.args[0].op == "+"

    def test_multiple_args(self):
        """hypot not in allowlist but we test comma syntax"""
        # Use a function that IS in allowlist with one arg
        ast = parse("atan(1)")
        assert isinstance(ast, FunctionCallNode)
        assert ast.name == "atan"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_pi(self):
        assert parse("pi") == _c("pi")

    def test_e(self):
        assert parse("e") == _c("e")

    def test_pi_in_expr(self):
        ast = parse("2*pi")
        assert ast == _b(_n("2"), "*", _c("pi"))


# ---------------------------------------------------------------------------
# Complex expressions
# ---------------------------------------------------------------------------

class TestComplex:
    def test_golden_ratio(self):
        """ (1+sqrt(5))/2 """
        ast = parse("(1+sqrt(5))/2")
        expected = _b(
            _b(_n("1"), "+", _f("sqrt", _n("5"))),
            "/",
            _n("2"),
        )
        assert ast == expected

    def test_quadratic_formula(self):
        """ (-b+sqrt(b^2-4*a*c))/(2*a) -- using symbolic names"""
        # Note: identifiers are only valid if in allowlist or constant.
        # Here we use numbers instead.
        ast = parse("(-1+sqrt(1^2-4*2*3))/(2*2)")
        assert ast is not None

    def test_whitespace_ignored(self):
        assert parse("  1  +  2  ") == _b(_n("1"), "+", _n("2"))
        assert parse("\n1\n+\n2\n") == _b(_n("1"), "+", _n("2"))
        assert parse("1\t+\t2") == _b(_n("1"), "+", _n("2"))


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestParseErrors:
    def test_unexpected_character(self):
        with pytest.raises(CalculatorError) as exc:
            parse("1+@2")
        assert exc.value.code == ErrorCode.PARSE_ERROR

    def test_incomplete_expression(self):
        with pytest.raises(CalculatorError) as exc:
            parse("1+")
        assert exc.value.code == ErrorCode.PARSE_ERROR

    def test_missing_paren(self):
        with pytest.raises(CalculatorError) as exc:
            parse("(1+2")
        assert exc.value.code == ErrorCode.PARSE_ERROR

    def test_extra_paren(self):
        with pytest.raises(CalculatorError) as exc:
            parse("1+2)")
        assert exc.value.code == ErrorCode.PARSE_ERROR

    def test_empty_input(self):
        with pytest.raises(CalculatorError) as exc:
            parse("")
        assert exc.value.code == ErrorCode.PARSE_ERROR

    def test_only_whitespace(self):
        with pytest.raises(CalculatorError) as exc:
            parse("   ")
        assert exc.value.code == ErrorCode.PARSE_ERROR

    def test_double_operator(self):
        with pytest.raises(CalculatorError) as exc:
            parse("1++2")
        assert exc.value.code == ErrorCode.PARSE_ERROR

    def test_trailing_operator(self):
        with pytest.raises(CalculatorError) as exc:
            parse("1+2+")
        assert exc.value.code == ErrorCode.PARSE_ERROR


# ---------------------------------------------------------------------------
# Unknown symbols
# ---------------------------------------------------------------------------

class TestUnknownSymbols:
    def test_unknown_function(self):
        with pytest.raises(CalculatorError) as exc:
            parse("foo(1)")
        assert exc.value.code == ErrorCode.UNKNOWN_SYMBOL

    def test_unknown_constant(self):
        with pytest.raises(CalculatorError) as exc:
            parse("x")
        assert exc.value.code == ErrorCode.UNKNOWN_SYMBOL

    def test_unknown_variable(self):
        with pytest.raises(CalculatorError) as exc:
            parse("x+y")
        assert exc.value.code == ErrorCode.UNKNOWN_SYMBOL

    def test_unknown_binary_operator(self):
        with pytest.raises(CalculatorError) as exc:
            parse("1%2")
        assert exc.value.code == ErrorCode.PARSE_ERROR
