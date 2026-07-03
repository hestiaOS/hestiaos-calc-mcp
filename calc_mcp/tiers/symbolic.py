"""Symbolic mathematics tier (SymPy, lazy import).

Loaded on first symbolic call only. Covers:
  simplify, solve, differentiate, integrate, factor, expand

Returns structured dict with {op, input, output, latex?}.
Fehler: UNSUPPORTED_OP, SYMBOLIC_FAILED.
"""

from __future__ import annotations

from calc_mcp.errors import CalculatorError, ErrorCode

SUPPORTED_OPS: frozenset[str] = frozenset({
    "simplify", "solve", "differentiate", "integrate", "factor", "expand",
})


def symbolic(op: str, expression: str, variable: str = "x") -> dict:
    """Apply a symbolic operation.

    Args:
        op: Operation name (from SUPPORTED_OPS).
        expression: String expression in SymPy syntax.
        variable: Symbol name (default 'x').

    Returns:
        Dict with keys: op, input, output, latex (optional).

    Raises:
        CalculatorError on unsupported op or evaluation failure.
    """
    if op not in SUPPORTED_OPS:
        raise CalculatorError(
            ErrorCode.UNSUPPORTED_OP,
            f"Unsupported symbolic operation {op!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_OPS))}",
        )

    # Lazy import: sympy loaded only on first call
    import sympy as _sympy

    sp = _sympy

    try:
        x = sp.Symbol(variable)
        expr = sp.sympify(expression)

        if op == "simplify":
            result = sp.simplify(expr)
        elif op == "solve":
            result = sp.solve(expr, x)
        elif op == "differentiate":
            result = sp.diff(expr, x)
        elif op == "integrate":
            result = sp.integrate(expr, x)
        elif op == "factor":
            result = sp.factor(expr)
        elif op == "expand":
            result = sp.expand(expr)
        else:
            raise CalculatorError(
                ErrorCode.UNSUPPORTED_OP,
                f"Unsupported symbolic operation {op!r}",
            )

        output = str(result)
        latex = sp.latex(result) if result is not None else None

        return {
            "op": op,
            "input": expression,
            "output": output,
            "latex": latex,
        }

    except CalculatorError:
        raise
    except Exception as exc:
        raise CalculatorError(
            ErrorCode.SYMBOLIC_FAILED,
            f"Symbolic {op} failed: {exc}",
        ) from exc
