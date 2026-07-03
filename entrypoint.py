#!/usr/bin/env python3
"""Stdio entrypoint for the calc-mcp FastMCP server.

Usage:
    python3 entrypoint.py          # start stdio MCP server
    python3 -m calc_mcp.server     # equivalent

The sys.path fix ensures the package is importable from the repo root.
"""

import sys
from pathlib import Path

# Add repo root to sys.path so calc_mcp is importable
_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from calc_mcp.server import main

if __name__ == "__main__":
    main()
