# Idris Kernel Build Guide

## Architecture

```
Agent → FastMCP (M6) → calc_mcp.core (Supervisor)
                         ├── kernel_worker.py (subprocess) → libcalc.dylib (Idris RefC)
                         └── degraded path → fractions.Fraction
```

The Idris `.dylib` runs in a persistent subprocess (`kernel_worker.py`). The
supervisor (`calc_mcp/core.py`) communicates via stdio line protocol and
**recycles** the worker after `KERNEL_MAX_CALLS` (default 10,000) to bound
the RefC runtime's reference-counted heap growth (sawtooth memory profile).
On timeout or crash the worker is killed and respawned transparently.

## Prerequisites

- **Idris 2** (v0.8.0+): `brew install idris2`
- **GMP** (GNU Multiple Precision): `brew install gmp`
- **Python 3.10+**, **ctypes** (stdlib)
- **gcc/clang** (for the C wrapper)

## Build Steps

### 1. Compile the Idris kernel to RefC

```bash
cd idris/
CPPFLAGS="-I/opt/homebrew/include" LDFLAGS="-L/opt/homebrew/lib" \
  idris2 libcalc_main.idr --codegen refc -o libcalc
```

This generates `build/exec/libcalc.o` (compiled Idris code with all 5 exports).

> **Note:** `libcalc_main.idr` duplicates the kernel inline under `module Main`
> and calls all 5 functions in `main()` to prevent dead code elimination.
> A `putStrLn` in `main` forces the exports into the compiled output.

### 2. Compile the C wrapper

```bash
REFCDIR="$(brew --prefix idris2)/libexec/idris2-0.8.0/support/refc"
SUPPORTCDIR="$(brew --prefix idris2)/libexec/idris2-0.8.0/support/c"

gcc -c -I"$REFCDIR" -I"$SUPPORTCDIR" -I/opt/homebrew/include \
    -fPIC calc_wrapper.c -o calc_wrapper.o
```

### 3. Link the shared library

```bash
REFC_A="$REFCDIR/libidris2_refc.a"
SUPPORT_DYLIB="$(brew --prefix idris2)/libexec/idris2-0.8.0/lib/libidris2_support.dylib"

gcc -dynamiclib -o libcalc.dylib \
    calc_wrapper.o build/exec/libcalc.o \
    "$REFC_A" "$SUPPORT_DYLIB" \
    -L/opt/homebrew/lib -lgmp \
    -install_name @rpath/libcalc.dylib
```

### 4. Verify

```bash
# Test via the compiled Idris executable
./build/exec/libcalc
# Expected output (from putStrLn in main): 5/6,1/6,1/3,3/4,8/27

# Test via Python ctypes
python3 -c "
import ctypes
lib = ctypes.CDLL('libcalc.dylib')
lib.add_rat.restype = ctypes.c_char_p
print(lib.add_rat(b'1/2,1/3').decode())
"
# Expected: 5/6
```

### 5. Run the full test suite

```bash
cd ..  # repo root
python3 -m pytest tests/test_core_exact.py -v
```

## Platform Notes

- **macOS (arm64):** builds produce `.dylib`. The build commands above assume
  Homebrew at `/opt/homebrew`. Adjust paths if using Intel or a different prefix.
- **Linux:** replace `.dylib` with `.so` and use `-fPIC`. The `-install_name`
  flag is macOS-specific (omit on Linux).
- **Python path:** `core.py` searches for `libcalc.dylib` relative to the repo
  root (`idris/`). On Linux it searches for `libcalc.so`.

## Memory Model

- **Recycled subprocess:** the Idris RefC `.dylib` runs in a persistent
  worker subprocess (`kernel_worker.py`). The supervisor tracks per-process
  call counts and recycles the worker after `KERNEL_MAX_CALLS` (default
  10,000). This bounds the RefC runtime's reference-counted heap growth
  to a sawtooth pattern — no linear accumulation.
- **stdout hygiene:** the worker redirects Idris-internal `putStrLn` output
  to stderr during startup; the stdout pipe carries only protocol responses.
- **Crash/respawn:** on timeout or crash the worker is killed and respawned
  transparently. After one retry, degraded mode (pure Python `fractions`)
  kicks in.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `gmp.h` not found | `brew install gmp`; set `CPPFLAGS=-I/opt/homebrew/include` |
| `_Main_add` not in `nm` output | Idris dead-code eliminated the function; ensure `main()` references all exports |
| `Abort trap: 6` in Python | (Legacy) old `strdup`-return ABI — rebuild with `calc_wrapper.c` buffer-based ABI |
| Library not loaded | Add `entrypoint.py`'s directory to `DYLD_LIBRARY_PATH` or use absolute path in `CDLL()` |
