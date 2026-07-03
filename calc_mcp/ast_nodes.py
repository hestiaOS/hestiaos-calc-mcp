"""Typed AST node types for calculator expressions.

Every expression is parsed into a validated, normalised AST. No eval/exec.

Node hierarchy:
  - NumberNode(value: str)         # integer or decimal literal
  - ConstantNode(name: str)        # pi, e
  - UnaryOpNode(op: str, operand)  # unary minus only
  - BinaryOpNode(left, op, right)  # +, -, *, /, ^
  - FunctionCallNode(name, args)   # sqrt, sin, exp, …
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class NumberNode:
    """A numeric literal (integer or decimal string)."""
    value: str


@dataclass(frozen=True)
class ConstantNode:
    """A named constant (pi, e)."""
    name: str


@dataclass(frozen=True)
class UnaryOpNode:
    """A unary operator application. Currently only '-'."""
    op: str
    operand: ASTNode


@dataclass(frozen=True)
class BinaryOpNode:
    """A binary operator application."""
    left: ASTNode
    op: str
    right: ASTNode


@dataclass(frozen=True)
class FunctionCallNode:
    """A function call with positional arguments."""
    name: str
    args: List[ASTNode] = field(default_factory=list)


@dataclass(frozen=True)
class XNode:
    """The literal variable x — used exclusively by the plot path (sample_function).

    NOT a generic VariableNode. No string name, no attribute access,
    no free symbol resolution. Created only by parse_for_plot().
    """


# Union type for all valid AST nodes
ASTNode = NumberNode | ConstantNode | UnaryOpNode | BinaryOpNode | FunctionCallNode | XNode


# ---------------------------------------------------------------------------
# Allowlists
# ---------------------------------------------------------------------------

VALID_BINARY_OPS: frozenset[str] = frozenset({"+", "-", "*", "/", "^"})

VALID_UNARY_OPS: frozenset[str] = frozenset({"-"})

VALID_FUNCTIONS: frozenset[str] = frozenset({
    "sqrt", "abs",
    "ln", "log", "exp",
    "sin", "cos", "tan",
    "asin", "acos", "atan",
})

VALID_CONSTANTS: frozenset[str] = frozenset({"pi", "e"})
