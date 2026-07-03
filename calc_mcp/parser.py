"""Safe tokenizer + Pratt parser → typed AST. No eval/exec/compile.

Grammar (Pratt parser, allowlist-validated):

  expression     → term (("+" | "-") term)*
  term           → unary (("*" | "/") unary)*
  unary          → "-" unary | power
  power          → call ("^" power)?
  call           → primary ("(" args? ")")?
  primary        → NUMBER | IDENTIFIER | "(" expression ")"
  args           → expression ("," expression)*

Security guarantees:
  - No eval/exec/compile
  - Strict allowlist for operators, functions, constants
  - Hard limits on input length, AST depth, node count
  - Null-byte rejection
  - Attribute/dunder access is impossible by design (no __ built)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Generator

from calc_mcp.ast_nodes import (
    ASTNode,
    BinaryOpNode,
    ConstantNode,
    FunctionCallNode,
    NumberNode,
    UnaryOpNode,
    XNode,
    VALID_BINARY_OPS,
    VALID_CONSTANTS,
    VALID_FUNCTIONS,
    VALID_UNARY_OPS,
)
from calc_mcp.errors import CalculatorError, ErrorCode
from calc_mcp.limits import MAX_AST_DEPTH, MAX_AST_NODES, MAX_INPUT_LEN

# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

TOKEN_NUMBER = "NUMBER"
TOKEN_IDENTIFIER = "IDENTIFIER"
TOKEN_OPERATOR = "OPERATOR"
TOKEN_LPAREN = "LPAREN"
TOKEN_RPAREN = "RPAREN"
TOKEN_COMMA = "COMMA"
TOKEN_EOF = "EOF"


@dataclass(frozen=True)
class Token:
    type: str
    value: str
    pos: int  # character offset in input


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Pattern for numbers (int, decimal, scientific)
_NUMBER_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")

# Identifier pattern (letters only, no digits — enforce allowlist at parse time)
_IDENTIFIER_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def tokenize(source: str) -> Generator[Token, None, None]:
    """Yield tokens from *source*.

    Raises CalculatorError(PARSE_ERROR) on invalid characters.
    """
    # Security: reject null bytes
    if "\x00" in source:
        raise CalculatorError(
            ErrorCode.PARSE_ERROR,
            "Null bytes are not allowed in expressions",
        )

    # Security: input length check
    if len(source) > MAX_INPUT_LEN:
        raise CalculatorError(
            ErrorCode.LIMIT_EXCEEDED,
            f"Input exceeds maximum length of {MAX_INPUT_LEN} characters",
            detail={"max_length": MAX_INPUT_LEN, "actual": len(source)},
        )

    pos = 0
    length = len(source)

    while pos < length:
        ch = source[pos]

        # Skip whitespace
        if ch in " \t\n\r":
            pos += 1
            continue

        # Operators
        if ch in "+-*/^":
            yield Token(TOKEN_OPERATOR, ch, pos)
            pos += 1
            continue

        # Parens
        if ch == "(":
            yield Token(TOKEN_LPAREN, ch, pos)
            pos += 1
            continue
        if ch == ")":
            yield Token(TOKEN_RPAREN, ch, pos)
            pos += 1
            continue

        # Comma
        if ch == ",":
            yield Token(TOKEN_COMMA, ch, pos)
            pos += 1
            continue

        # Number
        m = _NUMBER_RE.match(source, pos)
        if m:
            yield Token(TOKEN_NUMBER, m.group(), pos)
            pos = m.end()
            continue

        # Identifier
        m = _IDENTIFIER_RE.match(source, pos)
        if m:
            yield Token(TOKEN_IDENTIFIER, m.group(), pos)
            pos = m.end()
            continue

        # Invalid character
        raise CalculatorError(
            ErrorCode.PARSE_ERROR,
            f"Unexpected character {ch!r} at position {pos}",
            detail={"pos": pos, "char": ch},
        )

    yield Token(TOKEN_EOF, "", length)


# ---------------------------------------------------------------------------
# Precedence levels
# ---------------------------------------------------------------------------

_PREC_SUM = 1     # +, -
_PREC_PROD = 2    # *, /
_PREC_POW = 3     # ^
_PREC_PREFIX = 4  # unary -
_PREC_CALL = 5    # function call


def _precedence(op: str) -> int:
    return {
        "+": _PREC_SUM,
        "-": _PREC_SUM,
        "*": _PREC_PROD,
        "/": _PREC_PROD,
        "^": _PREC_POW,
    }.get(op, 0)


# ---------------------------------------------------------------------------
# Pratt parser
# ---------------------------------------------------------------------------


def _expect(tokens: list[Token], pos: int, *types: str) -> tuple[Token, int]:
    """Consume a token of one of *types* or raise."""
    if pos >= len(tokens) or tokens[pos].type not in types:
        got = tokens[pos].value if pos < len(tokens) else "EOF"
        expected = "/".join(types)
        raise CalculatorError(
            ErrorCode.PARSE_ERROR,
            f"Expected {expected}, got {got!r}",
            detail={"expected": types, "got": got, "pos": tokens[pos].pos if pos < len(tokens) else -1},
        )
    tok = tokens[pos]
    return tok, pos + 1


def _parse_expression(
    tokens: list[Token],
    pos: int,
    min_prec: int,
    depth: int,
    node_count: list[int],
    xnode_allowed: bool = False,
) -> tuple[ASTNode, int]:
    """Parse an expression using the Pratt algorithm.

    *depth* tracks parse-tree depth for LIMIT_EXCEEDED.
    *node_count* is a mutable list with one element (list[0]) for LIMIT_EXCEEDED.
    Returns (node, next_pos).
    """

    # --- Depth limit ---
    if depth > MAX_AST_DEPTH:
        raise CalculatorError(
            ErrorCode.LIMIT_EXCEEDED,
            f"Expression exceeds maximum AST depth of {MAX_AST_DEPTH}",
            detail={"max_depth": MAX_AST_DEPTH},
        )

    # --- Node count limit ---
    if node_count[0] > MAX_AST_NODES:
        raise CalculatorError(
            ErrorCode.LIMIT_EXCEEDED,
            f"Expression exceeds maximum of {MAX_AST_NODES} nodes",
            detail={"max_nodes": MAX_AST_NODES},
        )

    # --- Prefix parsing ---
    if pos >= len(tokens) or tokens[pos].type == TOKEN_EOF:
        raise CalculatorError(
            ErrorCode.PARSE_ERROR,
            "Unexpected end of expression",
        )

    tok = tokens[pos]

    if tok.type == TOKEN_NUMBER:
        node: ASTNode = NumberNode(tok.value)
        node_count[0] += 1
        pos += 1

    elif tok.type == TOKEN_IDENTIFIER:
        name = tok.value
        pos += 1

        # Plot mode: x is a variable
        if xnode_allowed and name == "x":
            node = XNode()
            node_count[0] += 1

        # Check if it's a function call
        elif pos < len(tokens) and tokens[pos].type == TOKEN_LPAREN:
            # Function call
            if name not in VALID_FUNCTIONS:
                raise CalculatorError(
                    ErrorCode.UNKNOWN_SYMBOL,
                    f"Unknown function {name!r}",
                    detail={"symbol": name},
                )
            pos += 1  # consume LPAREN
            args: list[ASTNode] = []
            if pos < len(tokens) and tokens[pos].type != TOKEN_RPAREN:
                arg, pos = _parse_expression(tokens, pos, 0, depth + 1, node_count, xnode_allowed)
                args.append(arg)
                while pos < len(tokens) and tokens[pos].type == TOKEN_COMMA:
                    pos += 1
                    arg, pos = _parse_expression(tokens, pos, 0, depth + 1, node_count, xnode_allowed)
                    args.append(arg)
            tok, pos = _expect(tokens, pos, TOKEN_RPAREN)
            node = FunctionCallNode(name, args)
            node_count[0] += 1
        else:
            # Constant reference
            if name not in VALID_CONSTANTS:
                raise CalculatorError(
                    ErrorCode.UNKNOWN_SYMBOL,
                    f"Unknown constant {name!r}",
                    detail={"symbol": name},
                )
            node = ConstantNode(name)
            node_count[0] += 1

    elif tok.type == TOKEN_OPERATOR and tok.value in VALID_UNARY_OPS:
        pos += 1  # consume '-'
        operand, pos = _parse_expression(tokens, pos, _PREC_PREFIX, depth + 1, node_count, xnode_allowed)
        node = UnaryOpNode(tok.value, operand)
        node_count[0] += 1

    elif tok.type == TOKEN_LPAREN:
        pos += 1  # consume '('
        node, pos = _parse_expression(tokens, pos, 0, depth + 1, node_count, xnode_allowed)
        tok, pos = _expect(tokens, pos, TOKEN_RPAREN)
        # node count already accounted for

    else:
        raise CalculatorError(
            ErrorCode.PARSE_ERROR,
            f"Unexpected token {tok.value!r} at position {tok.pos}",
            detail={"pos": tok.pos, "token": tok.value},
        )

    # --- Infix parsing ---
    while pos < len(tokens):
        tok = tokens[pos]

        if tok.type == TOKEN_EOF or tok.type in (TOKEN_RPAREN, TOKEN_COMMA):
            break

        if tok.type == TOKEN_OPERATOR:
            op = tok.value
            prec = _precedence(op)
            if prec == 0 or prec < min_prec:
                break

            # Validate operator is allowed
            if op not in VALID_BINARY_OPS:
                raise CalculatorError(
                    ErrorCode.PARSE_ERROR,
                    f"Unknown operator {op!r}",
                    detail={"operator": op},
                )

            pos += 1  # consume operator

            if op == "^":
                # Right-associative: use prec (not prec + 1)
                next_min = prec
            else:
                next_min = prec + 1

            right, pos = _parse_expression(tokens, pos, next_min, depth + 1, node_count, xnode_allowed)
            node = BinaryOpNode(node, op, right)
            node_count[0] += 1

        else:
            break

    return node, pos


def parse(source: str) -> ASTNode:
    """Parse *source* into a typed AST.

    Raises CalculatorError on invalid input, unknown symbols, or limit violations.
    """
    tokens = list(tokenize(source))
    node_count: list[int] = [0]
    node, pos = _parse_expression(tokens, 0, min_prec=0, depth=0, node_count=node_count)

    # After parsing, we must be at the end
    if pos < len(tokens) and tokens[pos].type != TOKEN_EOF:
        tok = tokens[pos]
        raise CalculatorError(
            ErrorCode.PARSE_ERROR,
            f"Unexpected token {tok.value!r} after expression at position {tok.pos}",
            detail={"pos": tok.pos, "token": tok.value},
        )

    return node


def parse_for_plot(source: str) -> ASTNode:
    """Parse *source* into a typed AST, allowing `x` as a plot variable.

    Identical to parse() except that the unbound identifier `x` produces
    an XNode instead of UNKNOWN_SYMBOL. All other identifiers remain
    restricted to the existing function/constant allowlists.
    """
    tokens = list(tokenize(source))
    node_count: list[int] = [0]
    node, pos = _parse_expression(
        tokens, 0, min_prec=0, depth=0, node_count=node_count,
        xnode_allowed=True,
    )

    if pos < len(tokens) and tokens[pos].type != TOKEN_EOF:
        tok = tokens[pos]
        raise CalculatorError(
            ErrorCode.PARSE_ERROR,
            f"Unexpected token {tok.value!r} after expression at position {tok.pos}",
            detail={"pos": tok.pos, "token": tok.value},
        )

    return node
