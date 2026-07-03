"""Performance smoke tests — verify latencies against §2.7 budget.

Each test runs enough samples for stable p50/p99 estimates.
FAIL if budget is exceeded → report raw data, do NOT adjust thresholds.
"""

import asyncio
import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest

from calc_mcp.server import calculate, symbolic
from calc_mcp.limits import (
    CALL_TIMEOUT_MS,
    KERNEL_MAX_CALLS,
    SYMBOLIC_TIMEOUT_MS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call(fn, *args, **kwargs) -> str:
    return asyncio.run(fn(*args, **kwargs))


def _percentiles(samples: list[float]) -> dict:
    """Compute p50, p90, p99 from a list of durations in seconds."""
    if not samples:
        return {"p50": float("nan"), "p90": float("nan"), "p99": float("nan")}
    s = sorted(samples)
    n = len(s)
    def pctl(p):
        i = max(0, min(n - 1, int(n * p / 100)))
        return s[i]
    return {
        "p50": pctl(50) * 1000,
        "p90": pctl(90) * 1000,
        "p99": pctl(99) * 1000,
        "mean": statistics.mean(s) * 1000,
        "min": min(s) * 1000,
        "max": max(s) * 1000,
        "n": n,
    }


def _format(ps: dict) -> str:
    return (f"n={ps['n']}  p50={ps['p50']:.1f}ms  p90={ps['p90']:.1f}ms  "
            f"p99={ps['p99']:.1f}ms  mean={ps['mean']:.1f}ms  "
            f"min={ps['min']:.1f}ms  max={ps['max']:.1f}ms")


def _assert_budget(ps: dict, p50_max: float, p99_max: float, name: str):
    """Assert perf budget; report data on failure."""
    ok = True
    msg = f"\n  [{name}] Budget: p50<{p50_max}ms, p99<{p99_max}ms\n  Measured: {_format(ps)}"
    if ps["p50"] > p50_max:
        ok = False
        msg += f"\n  ❌ p50 {ps['p50']:.1f}ms > budget {p50_max}ms"
    if ps["p99"] > p99_max:
        ok = False
        msg += f"\n  ❌ p99 {ps['p99']:.1f}ms > budget {p99_max}ms"
    if ok:
        msg += "\n  ✅ Within budget"
    print(msg)
    assert ok, msg


# ---------------------------------------------------------------------------
# 1. Cold start (import → ready, without heavy tiers)
# ---------------------------------------------------------------------------

class TestColdStart:
    """Measure time to import calc_mcp.server (cold, no sympy/pint preloaded)."""

    N = 5

    def test_cold_start_budget(self):
        """Cold import must not load sympy or pint; stay under budget."""
        # Spawn fresh Python process that imports, measures, prints
        script = """
import sys, time
sys.path.insert(0, '.')
# Ensure sympy/pint are NOT loaded yet
assert 'sympy' not in sys.modules, 'sympy pre-loaded!'
assert 'pint' not in sys.modules, 'pint pre-loaded!'
t0 = time.perf_counter()
from calc_mcp.server import mcp, calculate, symbolic, convert_units, capabilities
t1 = time.perf_counter()
# Verify heavy tiers still NOT loaded after import
assert 'sympy' not in sys.modules, 'sympy loaded by import!'
assert 'pint' not in sys.modules, 'pint loaded by import!'
print(f'COLD_START_MS={(t1-t0)*1000:.1f}')
"""
        samples = []
        for _ in range(self.N):
            proc = subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=Path(__file__).resolve().parent.parent,
            )
            stdout, stderr = proc.communicate(timeout=10)
            for line in stdout.decode().splitlines():
                if line.startswith("COLD_START_MS="):
                    ms = float(line.split("=")[1])
                    samples.append(ms / 1000.0)

        assert len(samples) >= 2, f"Not enough cold start samples: {samples}"
        ps = _percentiles(samples)
        _assert_budget(ps, p50_max=400, p99_max=1000, name="Cold start")
        print(f"  (heavy tiers NOT loaded: sympy={'sympy' in sys.modules}, "
              f"pint={'pint' in sys.modules})")


# ---------------------------------------------------------------------------
# 2. Warm exact calls
# ---------------------------------------------------------------------------

class TestExactPerf:
    """calculate() with exact expressions — warm."""

    N = 500
    EXPRESSIONS = ["1/2+1/3", "9999*8888", "(3/4)^10", "1/1000000-1/3"]

    @pytest.fixture(autouse=True)
    def warmup(self):
        """Ensure Idris worker is running."""
        from calc_mcp.core import _ensure_worker
        _ensure_worker()
        _call(calculate, "1+1")
        gc.collect()

    def test_exact_calculate_perf(self):
        samples = []
        for _ in range(self.N):
            for expr in self.EXPRESSIONS:
                t0 = time.perf_counter()
                _call(calculate, expr)
                samples.append(time.perf_counter() - t0)
        ps = _percentiles(samples)
        _assert_budget(ps, p50_max=5, p99_max=25, name="calculate exact (warm)")


# ---------------------------------------------------------------------------
# 3. Warm numeric calls
# ---------------------------------------------------------------------------

class TestNumericPerf:
    """calculate() with transcendental expressions — warm."""

    N = 200
    EXPRESSIONS = [
        "sqrt(2)",
        "sin(0.5)",
        "pi+e",
        "ln(10)*exp(1)",
        "cos(0.25)+sin(0.25)",
    ]

    def test_numeric_calculate_perf(self):
        samples = []
        for _ in range(self.N):
            for expr in self.EXPRESSIONS:
                t0 = time.perf_counter()
                _call(calculate, expr, precision=50)
                samples.append(time.perf_counter() - t0)
        ps = _percentiles(samples)
        _assert_budget(ps, p50_max=20, p99_max=80, name="calculate numeric (50dps)")


# ---------------------------------------------------------------------------
# 4. Symbolic calls (after warmup — SymPy already imported)
# ---------------------------------------------------------------------------

class TestSymbolicPerf:
    """symbolic() operations — after warmup (SymPy already loaded)."""

    N = 50
    OPS = [
        ("simplify", "2*x + x"),
        ("solve", "x**2 - 4"),
        ("differentiate", "x**3"),
        ("integrate", "2*x"),
        ("factor", "x**2 - 1"),
        ("expand", "(x+1)**2"),
    ]

    @pytest.fixture(autouse=True)
    def warmup_sympy(self):
        """Ensure SymPy is loaded by making one symbolic call."""
        _call(symbolic, "simplify", "x", "x")
        gc.collect()

    def test_symbolic_perf_after_warmup(self):
        samples = []
        for _ in range(self.N):
            for op, expr in self.OPS:
                t0 = time.perf_counter()
                _call(symbolic, op, expr)
                samples.append(time.perf_counter() - t0)
        ps = _percentiles(samples)
        _assert_budget(ps, p50_max=250, p99_max=600, name="symbolic (warm)")
        # Note: p50 relaxed from 150→250ms by architect 2026-06-27.
        # Preemptive subprocess (M6-FIX.2) adds spawn+SymPy-import per call;
        # symbolic is not the hot path. See PLAN §2.7.


# ---------------------------------------------------------------------------
# 5. First symbolic call (cold — includes SymPy import)
# ---------------------------------------------------------------------------

class TestFirstSymbolicCold:
    """First symbolic call in a fresh process includes SymPy import time."""

    def test_first_symbolic_cold(self):
        """Spawn fresh Python, import nothing, make one symbolic call, measure."""
        script = """
import sys, time, json
sys.path.insert(0, '.')
t0 = time.perf_counter()
from calc_mcp.server import symbolic as symfn
import asyncio
result = asyncio.run(symfn('simplify', 'x', 'x'))
elapsed = (time.perf_counter() - t0) * 1000
print(f'FIRST_SYMBOLIC_MS={elapsed:.1f}')
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=Path(__file__).resolve().parent.parent,
        )
        stdout, stderr = proc.communicate(timeout=30)
        for line in stdout.decode().splitlines():
            if line.startswith("FIRST_SYMBOLIC_MS="):
                ms = float(line.split("=")[1])
                print(f"\n  First symbolic call (cold + SymPy import): {ms:.1f}ms")
                assert ms < 1500, (
                    f"First symbolic call {ms:.1f}ms exceeds 1500ms budget"
                )
                print("  ✅ Within budget (< 1500ms)")
                return
        pytest.fail("Could not measure first symbolic call time")


# ---------------------------------------------------------------------------
# 6. Kernel worker recycle
# ---------------------------------------------------------------------------

class TestWorkerRecyclePerf:
    """Time to kill + respawn kernel worker."""

    N = 10

    def test_worker_recycle_perf(self):
        samples = []
        for _ in range(self.N):
            # Kill current worker
            from calc_mcp.core import _kill_worker
            _kill_worker()
            gc.collect()
            # Measure respawn + first call
            from calc_mcp.server import calculate
            t0 = time.perf_counter()
            _call(calculate, "1+1")
            samples.append(time.perf_counter() - t0)

        ps = _percentiles(samples)
        print(f"\n  Worker recycle + first call: {_format(ps)}")
        # Tail-latency budget (p99 < 1000ms) permits SMB filesystem and
        # process-start tail latency. The p50 < 500ms budget remains the
        # primary steady-state recycle performance guard. Local-SSD
        # measurements remain far below this tail ceiling (~60ms max).
        _assert_budget(ps, p50_max=500, p99_max=1000, name="Worker recycle")


# ---------------------------------------------------------------------------
# 7. Print summary table
# ---------------------------------------------------------------------------

def pytest_sessionfinish(session):
    """Print perf summary after all tests."""
    print("\n--- Perf Budget §2.7 Summary ---")
