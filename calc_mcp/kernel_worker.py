#!/usr/bin/env python3
"""Kernel worker subprocess — Idris RefC via ctypes, stdio-line IPC.

Protocol:
  Request:  OP<TAB>ARG\\n
  Response: RESULT\\n

  OP ∈ {add_rat, sub_rat, mul_rat, div_rat, intpow_rat}
  RESULT  = canonical rational string, DIV_BY_ZERO, PARSE_ERROR, or
            INTERNAL_ERROR:<msg>

The worker redirects all Idris-init stdout to stderr during startup so the
stdout pipe carries only protocol responses (M4-FIX.2 hygiene).
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Redirect stdout BEFORE loading libcalc (Idris putStrLn at init)
# ---------------------------------------------------------------------------
_saved_stdout = os.dup(1)
_DEVNULL = os.open(os.devnull, os.O_WRONLY)
os.dup2(_DEVNULL, 1)
os.close(_DEVNULL)

# ---------------------------------------------------------------------------
# Load Idris kernel library
# ---------------------------------------------------------------------------
_BUF_SIZE = 2 * 100_000 + 32  # MAX_RESULT_DIGITS + safety

repo_root = Path(__file__).resolve().parent.parent
_lib_path: str | None = None
for candidate in [
    repo_root / "idris" / "libcalc.dylib",
    repo_root / "idris" / "libcalc.so",
    Path("/usr/local/lib") / "libcalc.dylib",
    Path("/usr/local/lib") / "libcalc.so",
    Path("/opt/homebrew/lib") / "libcalc.dylib",
]:
    if candidate.exists():
        _lib_path = str(candidate)
        break

if _lib_path is None:
    print("INTERNAL_ERROR:Lib not found", flush=True)
    sys.exit(1)

_lib = ctypes.CDLL(_lib_path)

# Configure all 5 buffer-function signatures
_OP_FUNCS: dict[str, ctypes.CFUNCTYPE] = {}
for name in ("add_rat_buf", "sub_rat_buf", "mul_rat_buf", "div_rat_buf", "intpow_rat_buf"):
    fn = getattr(_lib, name)
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    _OP_FUNCS[name.replace("_buf", "")] = fn

# ---------------------------------------------------------------------------
# Trigger Idris init WHILE stdout is still redirected to /dev/null,
# so the init-time all_results/putStrLn output is discarded.
# ---------------------------------------------------------------------------
sys.stdout.flush()

_BUF = ctypes.create_string_buffer(_BUF_SIZE)

_warmup_fn = _OP_FUNCS.get("add_rat")
if _warmup_fn is not None:
    _warmup_fn(b"1/1,1/1", _BUF, ctypes.c_int(_BUF_SIZE))

# Restore stdout for protocol — warmup is done, no more init output
os.dup2(_saved_stdout, 1)
os.close(_saved_stdout)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

_BUF = ctypes.create_string_buffer(_BUF_SIZE)


def _handle(op: str, arg: str) -> str:
    fn = _OP_FUNCS.get(op)
    if fn is None:
        return f"INTERNAL_ERROR:Unknown op {op}"

    # Clear buffer
    _BUF.value = b""

    written = fn(arg.encode("utf-8"), _BUF, ctypes.c_int(_BUF_SIZE))
    if written < 0:
        return "INTERNAL_ERROR:Buffer overflow"
    return _BUF.value.decode("utf-8")


def main() -> None:
    """Read stdin, dispatch, write result to stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        op = parts[0]
        arg = parts[1] if len(parts) > 1 else ""
        result = _handle(op, arg)
        sys.stdout.write(result + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
