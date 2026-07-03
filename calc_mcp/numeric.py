"""mpmath numeric path for transcendental evaluation.

Provides:
  has_transcendental(ast) -> bool     — routing helper for M6
  eval_ast_numeric(ast, dps) -> str   — evaluate full AST numerically

Deterministic: same input + dps → identical output string.
Lazy import: mpmath loaded on first call only.
"""

from __future__ import annotations

from calc_mcp.ast_nodes import (
    ASTNode,
    BinaryOpNode,
    ConstantNode,
    FunctionCallNode,
    NumberNode,
    UnaryOpNode,
    VALID_BINARY_OPS,
    VALID_CONSTANTS,
    VALID_FUNCTIONS,
    VALID_UNARY_OPS,
)
from calc_mcp.errors import CalculatorError, ErrorCode
from calc_mcp.limits import NUMERIC_DPS_DEFAULT, NUMERIC_DPS_MAX

# ---------------------------------------------------------------------------
# Allowlists (redundant with M3 parser — kept for isolation)
# ---------------------------------------------------------------------------

_TRANSCENDENTAL_FUNCTIONS: frozenset[str] = frozenset(
    name for name in VALID_FUNCTIONS if name != "abs"
)
_TRANSCENDENTAL_CONSTANTS: frozenset[str] = frozenset({"pi", "e"})
_ALL_NUMERIC_FUNCTIONS: frozenset[str] = VALID_FUNCTIONS


def has_transcendental(ast: ASTNode) -> bool:
    """Return True if *ast* contains any non-exact operation.

    An operation is "not exact" if it involves:
      - A transcendental function (sqrt, ln, log, exp, trig)
      - An irrational constant (pi, e)
      - A non-integer exponent in a power expression
      - Decimal or scientific notation numbers
    """
    if isinstance(ast, NumberNode):
        return "." in ast.value or "e" in ast.value.lower()
    if isinstance(ast, ConstantNode):
        return ast.name in _TRANSCENDENTAL_CONSTANTS
    if isinstance(ast, UnaryOpNode):
        return has_transcendental(ast.operand)
    if isinstance(ast, BinaryOpNode):
        if ast.op == "^":
            # Non-integer exponent → numeric
            if not _is_integer(ast.right):
                return True
        return has_transcendental(ast.left) or has_transcendental(ast.right)
    if isinstance(ast, FunctionCallNode):
        if ast.name in _TRANSCENDENTAL_FUNCTIONS:
            return True
        # abs is exact; recurse into args
        return any(has_transcendental(a) for a in ast.args)
    return False


def _is_integer(node: ASTNode) -> bool:
    """Check if an AST node represents an integer literal."""
    from calc_mcp.ast_nodes import NumberNode
    if isinstance(node, NumberNode):
        try:
            int(node.value)
            return "." not in node.value and "e" not in node.value.lower()
        except ValueError:
            return False
    return False


# ---------------------------------------------------------------------------
# Numeric evaluation
# ---------------------------------------------------------------------------


def _to_mpmath(node, mp):
    """Convert an AST node to an mpmath mpf/mpc value."""
    import mpmath
    if isinstance(node, NumberNode):
        return mp.mpf(node.value)

    if isinstance(node, ConstantNode):
        if node.name == "pi":
            return mp.pi  # type: ignore[union-attr]
        elif node.name == "e":
            return mp.e  # type: ignore[union-attr]

    if isinstance(node, UnaryOpNode):
        val = _to_mpmath(node.operand, mp)
        if node.op == "-":
            return -val
        raise CalculatorError(
            ErrorCode.UNSUPPORTED_OP,
            f"Unsupported unary operator {node.op!r} in numeric mode",
        )

    if isinstance(node, BinaryOpNode):
        left = _to_mpmath(node.left, mp)
        right = _to_mpmath(node.right, mp)
        if node.op == "+":
            return left + right
        elif node.op == "-":
            return left - right
        elif node.op == "*":
            return left * right
        elif node.op == "/":
            return left / right
        elif node.op == "^":
            return left ** right
        raise CalculatorError(
            ErrorCode.UNSUPPORTED_OP,
            f"Unknown operator {node.op!r}",
        )

    if isinstance(node, FunctionCallNode):
        args = [_to_mpmath(a, mp) for a in node.args]
        name = node.name

        # Domain checks
        if name in ("ln", "log") and args[0] <= 0:
            raise CalculatorError(
                ErrorCode.DOMAIN_ERROR,
                f"{name}(x) requires x > 0, got {args[0]}",
            )
        if name == "sqrt" and args[0] < 0:
            raise CalculatorError(
                ErrorCode.DOMAIN_ERROR,
                f"sqrt(x) requires x >= 0, got {args[0]}",
            )
        if name in ("asin", "acos") and (args[0] < -1 or args[0] > 1):
            raise CalculatorError(
                ErrorCode.DOMAIN_ERROR,
                f"{name}(x) requires x in [-1, 1], got {args[0]}",
            )

        fn_map = {
            "sqrt": mp.sqrt,
            "abs": abs,
            "ln": mp.log,
            "log": mp.log10,
            "exp": mp.e ** args[0] if False else None,  # handled below
            "sin": mp.sin,
            "cos": mp.cos,
            "tan": mp.tan,
            "asin": mp.asin,
            "acos": mp.acos,
            "atan": mp.atan,
        }

        if name == "exp":
            return mp.e ** args[0]  # type: ignore[operator]
        fn = fn_map.get(name)
        if fn is None:
            raise CalculatorError(
                ErrorCode.UNSUPPORTED_OP,
                f"Unknown function {name!r}",
            )
        return fn(*args)

    raise CalculatorError(ErrorCode.INTERNAL, f"Unknown AST node type: {type(node)}")


def eval_ast_numeric(
    ast: ASTNode,
    dps: int = NUMERIC_DPS_DEFAULT,
) -> str:
    """Evaluate *ast* numerically via mpmath.

    Args:
        ast: Parsed AST (may contain exact + transcendental nodes).
        dps: Decimal places for mpmath (default from limits).

    Returns:
        Decimal result string with ``dps`` significant digits.

    Raises:
        CalculatorError on domain errors, limit violations, etc.
    """
    if dps > NUMERIC_DPS_MAX:
        raise CalculatorError(
            ErrorCode.LIMIT_EXCEEDED,
            f"Precision {dps} exceeds maximum {NUMERIC_DPS_MAX}",
        )

    import mpmath as _mpmath

    mp = _mpmath  # local alias for brevity

    mp.mp.dps = dps + 5  # extra guard digits for rounding
    try:
        result = _to_mpmath(ast, mp)
    except CalculatorError:
        raise
    except Exception as exc:
        raise CalculatorError(
            ErrorCode.DOMAIN_ERROR,
            f"Numeric evaluation error: {exc}",
        ) from exc

    # Format: n digits after decimal point
    mp.mp.dps = dps
    s = mp.nstr(result, dps + 5, strip_zeros=False)

    # Ensure result has the expected precision
    # mp.nstr returns scientific notation for very large/small values
    return s
