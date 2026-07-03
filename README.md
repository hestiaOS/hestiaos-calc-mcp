# calc-mcp &nbsp;·&nbsp; v0.9.0-math-ui

Build passing locally: 335 tests after the documented native build.

**Why?** LLMs are unreliable at arithmetic — they approximate rather than
compute. calc-mcp solves this by routing every operation through a verified
Idris kernel, a deterministic mpmath evaluator, or an isolated worker process.
Results are always correct, structured, and never guessed.

## Quick Start

1. **Install the MCP library:** `pip install mcp`
2. **Build the native Idris kernel** (optional — see [idris/BUILD.md](idris/BUILD.md)):
   follow the documented step-by-step instructions to produce `libcalc.dylib`.
3. **Run the test suite:** `python3 -m pytest -q` (expect 335 passing)

## What this is NOT — out of scope

- **Not a Python REPL or code runner** — the parser is a strict Pratt parser
  with allowlists. No `eval`, `exec`, `compile`, or dynamic attribute access.
- **Not a cloud service** — calc-mcp runs locally over stdio. No external
  dependencies, no telemetry, no API keys required.
- **Not a plotting library** — `sample_function` provides deterministic data
  points; rendering and graph assembly are client responsibilities.
- **Not a persistent computation engine** — no state, no session, no history
  is kept between tool calls.

## Architecture

```
Agent → FastMCP (stdio) → Safe Parser (no eval) → Exact Kernel (Idris/.py)
                                                   → Numeric (mpmath, lazy)
                                                   → Symbolic (SymPy, lazy)
                                                   → Units (Pint, lazy)
                                                   → structured JSON
```

## Setup

```bash
pip install mcp
# Optional tiers:
pip install ".[symbolic,units]"   # sympy, mpmath, pint
# Development:
pip install ".[dev]"              # pytest, hypothesis
```

## Tools

| Tool | Description |
|---|---|
| `calculate(expr, precision, rounding)` | Arithmetic + numeric evaluation |
| `symbolic(op, expr, variable)` | Symbolic math (SymPy) |
| `convert_units(value, from, to)` | Unit conversion (Pint) |
| `sample_function(expr, x_min, x_max, n, precision)` | 2D numeric function sampling over literal `x` |
| `capabilities()` | Self-description for agent discovery |

## Security

- **No `eval`/`exec`** — custom Pratt parser with strict allowlist
- **Hard resource limits** — timeout, AST depth, input length, exponent cap
- **Structured errors only** — never raw stacktraces or path leaks

## Precision Model

- **Exact:** rational arithmetic over arbitrary-precision integers (gcd-normalised)
- **Numeric:** transcendentals via mpmath at fixed precision (deterministic)
- **Rounding:** explicit mode per call (default `ROUND_HALF_EVEN`)

## Limits (defaults in `calc_mcp/limits.py`)

| Limit | Default |
|---|---|
| Input length | 4,096 chars |
| AST depth | 64 |
| AST nodes | 1,024 |
| Integer exponent | ≤1,000 |
| Result digits | ≤100,000 |
| Output size | ≤1 MiB |
| Call timeout | 2,000 ms |
| Symbolic timeout | 5,000 ms |
| Numeric precision | 50 dps (max 1,000) |

## Building from Source

The optional Idris kernel provides exact rational arithmetic via a shared
library (`libcalc.dylib` / `libcalc.so`). To build it locally, follow the
step-by-step instructions in [idris/BUILD.md](idris/BUILD.md).

Without the Idris kernel, the server degrades transparently to a pure-Python
`fractions.Fraction` fallback — all tools remain functional.

## License

Licensed under the Apache License, Version 2.0.
See [LICENSE](LICENSE), [NOTICE](NOTICE), and [CLEANROOM.md](CLEANROOM.md) for details.

## contributing

Contributions are welcome. Please open an issue or submit a pull request with
small, tested changes. For substantial modifications, discuss the design first
to ensure alignment with the project's scope and security model.

## Related

- [Model Context Protocol](https://modelcontextprotocol.io) — the protocol calc-mcp implements
- [Idris 2](https://idris2.readthedocs.io) — the verified functional language behind the exact kernel
