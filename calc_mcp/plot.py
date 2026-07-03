"""Plot evaluation path — mpmath-based sampling with XNode binding.

Provides:
  sample_plot(request: PlotRequest) -> dict
    Full end-to-end plot sampling: parse → grid → evaluate → structured points.
    Each point has canonical x/y strings, state, and per-point error objects.

  eval_plot_ast(ast, x_value_mpf, work_dps) -> mpf
    Evaluate an AST with XNode bound to the given x value.
"""

from __future__ import annotations


from calc_mcp.ast_nodes import (
    ASTNode,
    BinaryOpNode,
    ConstantNode,
    FunctionCallNode,
    NumberNode,
    UnaryOpNode,
    XNode,
)
from calc_mcp.errors import CalculatorError, ErrorCode
from calc_mcp.numeric import _to_mpmath  # same-package evaluator
from calc_mcp.plot_contract import (
    PlotRequest,
    build_canonical_grid,
    format_numeric_result,
    public_point_error_message,
)
from calc_mcp.limits import GUARD_DIGITS


# ---------------------------------------------------------------------------
# Plot-specific AST evaluator (XNode-aware)
# ---------------------------------------------------------------------------


def eval_plot_ast(ast: ASTNode, x_value_mpf, work_dps: int) -> float:
    """Evaluate *ast* with *x* bound to the given mpmath mpf value.

    Handles XNode by returning the bound x value. All other node types
    delegate to the existing _to_mpmath evaluator from calc_mcp.numeric.

    Args:
        ast: Parsed AST (may contain XNode).
        x_value_mpf: The current x sample value as an mpmath mpf.
        work_dps: mpmath working precision.

    Returns:
        mpmath mpf result value.

    Raises:
        CalculatorError on domain errors, division by zero, etc.
    """
    import mpmath as mp

    with mp.workdps(work_dps):
        # Check for XNode at the top level — if present, substitute
        # the current x value. Otherwise, delegate to the existing
        # numeric evaluator which doesn't know about XNode internally.
        if _has_xnode(ast):
            return _eval_with_xnode(ast, x_value_mpf, mp, work_dps)
        else:
            # No XNode — standard numeric eval
            return _to_mpmath(ast, mp)


def _has_xnode(node: ASTNode) -> bool:
    """Return True if the AST contains any XNode."""
    if isinstance(node, XNode):
        return True
    if isinstance(node, UnaryOpNode):
        return _has_xnode(node.operand)
    if isinstance(node, BinaryOpNode):
        return _has_xnode(node.left) or _has_xnode(node.right)
    if isinstance(node, FunctionCallNode):
        return any(_has_xnode(a) for a in node.args)
    return False


def _eval_with_xnode(node: ASTNode, x_value_mpf, mp, work_dps: int):
    """Evaluate AST containing XNode(s) with x bound.

    Mirrors _to_mpmath() logic but substitutes XNode.
    Domain checks are identical to calc_mcp.numeric._to_mpmath.
    """
    import mpmath

    if isinstance(node, XNode):
        return x_value_mpf

    if isinstance(node, NumberNode):
        return mp.mpf(node.value)

    if isinstance(node, ConstantNode):
        if node.name == "pi":
            return mp.pi
        elif node.name == "e":
            return mp.e

    if isinstance(node, UnaryOpNode):
        val = _eval_with_xnode(node.operand, x_value_mpf, mp, work_dps)
        if node.op == "-":
            return -val
        raise CalculatorError(
            ErrorCode.UNSUPPORTED_OP,
            f"Unsupported unary operator {node.op!r}",
        )

    if isinstance(node, BinaryOpNode):
        left = _eval_with_xnode(node.left, x_value_mpf, mp, work_dps)
        right = _eval_with_xnode(node.right, x_value_mpf, mp, work_dps)
        if node.op == "+":
            return left + right
        elif node.op == "-":
            return left - right
        elif node.op == "*":
            return left * right
        elif node.op == "/":
            try:
                return left / right
            except (ZeroDivisionError, ValueError, Exception) as div_err:
                # mpmath raises mpmath.libmp.libmpf.ZeroDivisionError (not built-in)
                if "ZeroDivision" in type(div_err).__name__:
                    raise CalculatorError(
                        ErrorCode.DIV_BY_ZERO,
                        public_point_error_message("DIV_BY_ZERO"),
                    ) from div_err
                raise
        elif node.op == "^":
            return left ** right
        raise CalculatorError(
            ErrorCode.UNSUPPORTED_OP,
            f"Unknown operator {node.op!r}",
        )

    if isinstance(node, FunctionCallNode):
        args = [_eval_with_xnode(a, x_value_mpf, mp, work_dps) for a in node.args]
        name = node.name

        # Domain checks (identical to calc_mcp.numeric._to_mpmath)
        if name in ("ln", "log") and args[0] <= 0:
            raise CalculatorError(
                ErrorCode.DOMAIN_ERROR,
                public_point_error_message("DOMAIN_ERROR"),
            )
        if name == "sqrt" and args[0] < 0:
            raise CalculatorError(
                ErrorCode.DOMAIN_ERROR,
                public_point_error_message("DOMAIN_ERROR"),
            )
        if name in ("asin", "acos") and (args[0] < -1 or args[0] > 1):
            raise CalculatorError(
                ErrorCode.DOMAIN_ERROR,
                public_point_error_message("DOMAIN_ERROR"),
            )

        fn_map = {
            "sqrt": mp.sqrt,
            "abs": abs,
            "ln": mp.log,
            "log": mp.log10,
            "exp": None,  # handled below
            "sin": mp.sin,
            "cos": mp.cos,
            "tan": mp.tan,
            "asin": mp.asin,
            "acos": mp.acos,
            "atan": mp.atan,
        }

        if name == "exp":
            return mp.e ** args[0]
        fn = fn_map.get(name)
        if fn is None:
            raise CalculatorError(
                ErrorCode.UNSUPPORTED_OP,
                f"Unknown function {name!r}",
            )
        return fn(*args)

    raise CalculatorError(
        ErrorCode.INTERNAL,
        f"Unknown AST node type in plot evaluator: {type(node)}",
    )


# ---------------------------------------------------------------------------
# Full plot sampling
# ---------------------------------------------------------------------------


def sample_plot(request: PlotRequest) -> dict:
    """Execute a full plot sampling run for the given PlotRequest.

    1. Parse expression via parse_for_plot()
    2. Build canonical grid
    3. For each point: evaluate with XNode binding
    4. Format y values with canonical numeric formatting
    5. Return structured points + metadata

    Args:
        request: A validated PlotRequest.

    Returns:
        dict with "points" and "meta" keys per the V1 contract.
    """
    import mpmath as mp
    from calc_mcp.parser import parse_for_plot

    work_dps = request.precision + GUARD_DIGITS

    # 1. Parse
    ast = parse_for_plot(request.expression)

    # 2. Grid
    grid = build_canonical_grid(request)

    # 3. Evaluate each point
    points: list[dict] = []
    for i, x_str in enumerate(grid):
        x_mpf = mp.mpf(x_str)
        try:
            with mp.workdps(work_dps):
                y_mpf = eval_plot_ast(ast, x_mpf, work_dps)
            y_str = format_numeric_result(y_mpf, request.precision)
            points.append({
                "index": i,
                "x": x_str,
                "y": y_str,
                "state": "defined",
            })
        except CalculatorError as exc:
            # Per-point error → structured undefined point
            point = {
                "index": i,
                "x": x_str,
                "y": None,
                "state": "undefined",
            }
            # Map internal codes to point-level codes
            code_map = {
                ErrorCode.DIV_BY_ZERO: "DIV_BY_ZERO",
                ErrorCode.DOMAIN_ERROR: "DOMAIN_ERROR",
                ErrorCode.LIMIT_EXCEEDED: "NUMERIC_OVERFLOW",
            }
            mapped_code = code_map.get(exc.code, "DOMAIN_ERROR")
            point["error"] = {
                "code": mapped_code,
                "message": public_point_error_message(mapped_code),
            }
            points.append(point)
        except Exception as exc:
            # Any non-CalculatorError (mpmath ZeroDivisionError, etc.)
            err_code = "DIV_BY_ZERO" if "zero" in str(exc).lower() else "DOMAIN_ERROR"
            points.append({
                "index": i,
                "x": x_str,
                "y": None,
                "state": "undefined",
                "error": {
                    "code": err_code,
                    "message": public_point_error_message(err_code),
                },
            })

    # 4. Metadata
    meta = {
        "contract_version": "1",
        "expression": request.expression,
        "requested_domain": {"x_min": request.x_min, "x_max": request.x_max},
        "effective_domain": {"x_min": grid[0], "x_max": grid[-1]},
        "sample_count": request.sample_count,
        "precision": request.precision,
        "rounding": request.rounding,
        "sampling": "uniform",
        "engine": "numeric-worker",
        "assurance_level": "numeric-tier",
        "warnings": [],
    }

    return {"points": points, "meta": meta}
