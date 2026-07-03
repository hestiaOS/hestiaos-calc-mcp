# Calculator-MCP — Subsystem Scope

## What this subsystem is

A standalone FastMCP server (`calc-mcp`) that provides deterministic arithmetic
evaluation to any hestiaOS agent. It runs over stdio and exposes 4 tools:
`calculate`, `symbolic`, `convert_units`, `capabilities`.

## Boundaries

- **IN scope:** exact rational arithmetic, numeric evaluation of transcendentals,
  symbolic manipulation (SymPy), unit conversion (Pint), agent discovery.
- **OUT of scope:** graph plotting, random number generation, stateful computation,
  file I/O, network access, any IO beyond stdio.

## Dependencies

- `mcp` (runtime required)
- `mpmath`, `sympy` (optional: `[symbolic]`)
- `pint` (optional: `[units]`)
- `pytest`, `hypothesis` (dev)

## Key files

| File | Role |
|---|---|
| `calc_mcp/server.py` | FastMCP tool definitions, I/O |
| `calc_mcp/parser.py` | Safe tokenizer + Pratt parser |
| `calc_mcp/ast_nodes.py` | Typed AST node types |
| `calc_mcp/core.py` | Exact kernel (Idris via ctypes or Python fallback) |
| `calc_mcp/numeric.py` | mpmath numeric path |
| `calc_mcp/tiers/symbolic.py` | SymPy tier (lazy) |
| `calc_mcp/tiers/units.py` | Pint tier (lazy) |
| `calc_mcp/limits.py` | Resource limits |
| `calc_mcp/errors.py` | Structured error codes |
| `entrypoint.py` | stdio entrypoint |
