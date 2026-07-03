"""Exact rational arithmetic kernel — supervised worker-subprocess.

Architecture (M4-FIX.3):
  - A persistent subprocess (kernel_worker.py) hosts the Idris RefC .dylib.
  - The supervisor tracks calls and recycles the process after
    KERNEL_MAX_CALLS to bound the linear RefC heap growth.
  - Timeout: if the worker doesn't respond within CALL_TIMEOUT_MS,
    the process is killed, respawned, and a TIMEOUT error is returned.
  - Crash resilience: broken pipe → respawn once + retry.
  - Degraded fallback: if the worker can't start, pure-Python
    fractions.Fraction is used.
  - Public API (exact_eval) unchanged.

Public API:
  exact_eval(ast: ASTNode) -> str
    Returns canonical result string 'p/q' (or 'p' if denom == 1).
    Raises CalculatorError on errors.
"""

from __future__ import annotations

import json
import logging
import os
import select
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from calc_mcp.ast_nodes import (
    ASTNode,
    BinaryOpNode,
    ConstantNode,
    FunctionCallNode,
    NumberNode,
    UnaryOpNode,
)
from calc_mcp.errors import CalculatorError, ErrorCode
from calc_mcp.limits import (
    CALL_TIMEOUT_MS,
    CLEANUP_GRACE_MS,
    KERNEL_MAX_CALLS,
    MAX_INT_POW_EXP,
    MAX_PLOT_REQUEST_BYTES,
    MAX_PLOT_RESPONSE_BYTES,
    MAX_RESULT_DIGITS,
    PLOT_TOTAL_TIMEOUT_MS,
    MAX_INFLIGHT_PLOT_WORKERS,
)

log = logging.getLogger("calc_mcp.core")


def _plot_public_message(code: str) -> str:
    """Lazy-import public message from plot_contract.py (pure, no I/O)."""
    from calc_mcp.plot_contract import public_top_level_error_message
    return public_top_level_error_message(code)

# ---------------------------------------------------------------------------
# Operator → worker function name map
# ---------------------------------------------------------------------------
_OP_MAP: dict[str, str] = {
    "+": "add_rat",
    "-": "sub_rat",
    "*": "mul_rat",
    "/": "div_rat",
}
_POW_FN = "intpow_rat"

# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

_worker_lock = threading.Lock()
_worker_proc: subprocess.Popen | None = None
_worker_call_count = 0
_worker_degraded = False


def _worker_script() -> str:
    """Return path to kernel_worker.py."""
    return str(Path(__file__).resolve().parent / "kernel_worker.py")


def _spawn_worker() -> subprocess.Popen:
    """Start the kernel worker subprocess.

    Returns the Popen handle. Raises CalculatorError on failure.
    """
    script = _worker_script()
    if not Path(script).exists():
        raise CalculatorError(
            ErrorCode.INTERNAL,
            f"Kernel worker script not found: {script}",
        )
    try:
        proc = subprocess.Popen(
            [sys.executable, script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise CalculatorError(
            ErrorCode.INTERNAL,
            f"Failed to spawn kernel worker: {exc}",
        )
    return proc


def _ensure_worker() -> subprocess.Popen | None:
    """Get or create worker process (thread-safe)."""
    global _worker_proc, _worker_call_count, _worker_degraded
    with _worker_lock:
        # Recycle if call limit reached
        if _worker_proc is not None and _worker_call_count >= KERNEL_MAX_CALLS:
            _kill_worker()
            _worker_call_count = 0

        if _worker_proc is None:
            try:
                _worker_proc = _spawn_worker()
                _worker_degraded = False
            except CalculatorError:
                # Degraded: fall back to pure Python
                if not _worker_degraded:
                    log.warning("Kernel worker unavailable — using degraded Python fallback")
                    _worker_degraded = True
                _worker_proc = None
        return _worker_proc


def _kill_worker() -> None:
    """Terminate the worker process and close pipes."""
    global _worker_proc
    if _worker_proc is None:
        return
    try:
        _worker_proc.terminate()
        _worker_proc.wait(timeout=2)
    except Exception:
        try:
            _worker_proc.kill()
            _worker_proc.wait(timeout=1)
        except Exception:
            pass
    finally:
        # Close pipes to prevent fd leak
        for pipe in (_worker_proc.stdin, _worker_proc.stdout, _worker_proc.stderr):
            try:
                if pipe is not None:
                    pipe.close()
            except Exception:
                pass
        _worker_proc = None


def _worker_call(op: str, arg: str) -> str:
    """Send a request to the worker and read the response.

    Handles timeout, crash-respawn, and wraps in CalculatorError.
    Returns the result string, or raises CalculatorError.
    """
    proc = _ensure_worker()

    # Send request
    request = f"{op}\t{arg}\n"
    try:
        proc.stdin.write(request.encode("utf-8"))  # type: ignore[union-attr]
        proc.stdin.flush()  # type: ignore[union-attr]
    except (BrokenPipeError, OSError):
        # Worker crashed — respawn once
        _kill_worker()
        proc = _ensure_worker()
        if proc is None:
            # Degraded path
            return _degraded_call(op, arg)
        try:
            proc.stdin.write(request.encode("utf-8"))  # type: ignore[union-attr]
            proc.stdin.flush()  # type: ignore[union-attr]
        except (BrokenPipeError, OSError):
            return _degraded_call(op, arg)

    # Read response with timeout
    deadline = time.monotonic() + CALL_TIMEOUT_MS / 1000.0
    line = b""
    try:
        while time.monotonic() < deadline:
            # Try reading one byte at a time (line-buffered)
            chunk = proc.stdout.read(1)  # type: ignore[union-attr]
            if not chunk:
                # EOF — process died
                _kill_worker()
                _ensure_worker()
                raise CalculatorError(
                    ErrorCode.INTERNAL,
                    "Kernel worker died during request",
                )
            if chunk == b"\n":
                break
            line += chunk
    except CalculatorError:
        raise
    except Exception:
        _kill_worker()
        _ensure_worker()
        raise CalculatorError(
            ErrorCode.INTERNAL,
            "Kernel worker communication error",
        )

    if time.monotonic() >= deadline:
        # Timeout — kill and respawn
        _kill_worker()
        _ensure_worker()
        raise CalculatorError(
            ErrorCode.TIMEOUT,
            f"Kernel worker timed out after {CALL_TIMEOUT_MS}ms",
        )

    return line.decode("utf-8").strip()


def _degraded_call(op: str, arg: str) -> str:
    """Pure-Python fractions.Fraction fallback when worker is unavailable."""
    from fractions import Fraction

    parts = arg.split(",")
    if len(parts) != 2:
        return "PARSE_ERROR"

    try:
        f1 = Fraction(parts[0]) if "/" in parts[0] else Fraction(int(parts[0]), 1)
        f2 = Fraction(parts[1]) if "/" in parts[1] else Fraction(int(parts[1]), 1)
    except (ValueError, ZeroDivisionError):
        return "PARSE_ERROR"

    if op == "add_rat":
        result = f1 + f2
    elif op == "sub_rat":
        result = f1 - f2
    elif op == "mul_rat":
        result = f1 * f2
    elif op == "div_rat":
        if f2 == 0:
            return "DIV_BY_ZERO"
        result = f1 / f2
    elif op == "intpow_rat":
        n = int(parts[1])
        result = f1 ** n
    else:
        return f"INTERNAL_ERROR:Unknown op {op}"

    if result.denominator == 1:
        return str(result.numerator)
    return f"{result.numerator}/{result.denominator}"


# ---------------------------------------------------------------------------
# AST evaluation
# ---------------------------------------------------------------------------


def _eval_ast(node: ASTNode) -> tuple[int, int]:
    """Walk the AST and evaluate via the supervised worker or degraded path."""
    if isinstance(node, NumberNode):
        val = node.value
        if "." in val or "e" in val.lower():
            raise CalculatorError(
                ErrorCode.NOT_IMPLEMENTED,
                f"Decimal/scientific notation ({val!r}) needs numeric tier (M5)",
            )
        try:
            n = int(val)
        except ValueError:
            raise CalculatorError(
                ErrorCode.PARSE_ERROR,
                f"Cannot parse number {val!r}",
            )
        return (n, 1)

    if isinstance(node, ConstantNode):
        name = node.name
        if name in ("pi", "e"):
            raise CalculatorError(
                ErrorCode.NOT_IMPLEMENTED,
                f"{name!r} needs numeric tier (M5)",
            )
        raise CalculatorError(
            ErrorCode.UNKNOWN_SYMBOL,
            f"Unknown constant {name!r}",
        )

    if isinstance(node, UnaryOpNode):
        if node.op == "-":
            num, den = _eval_ast(node.operand)
            return (-num, den)
        raise CalculatorError(
            ErrorCode.UNSUPPORTED_OP,
            f"Unsupported unary operator {node.op!r}",
        )

    if isinstance(node, BinaryOpNode):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)

        if node.op in _OP_MAP:
            left_str = _format_rat(*left)
            right_str = _format_rat(*right)
            result_str = _idris_call(_OP_MAP[node.op], f"{left_str},{right_str}")
            return _parse_rat_result(result_str)

        if node.op == "^":
            right_num, right_den = right
            if right_den != 1:
                raise CalculatorError(
                    ErrorCode.DOMAIN_ERROR,
                    "Non-integer exponent not supported in exact mode",
                )
            exp = right_num
            if abs(exp) > MAX_INT_POW_EXP:
                raise CalculatorError(
                    ErrorCode.LIMIT_EXCEEDED,
                    f"Exponent {exp} exceeds maximum {MAX_INT_POW_EXP}",
                )
            left_str = _format_rat(*left)
            result_str = _idris_call(_POW_FN, f"{left_str},{exp}")
            return _parse_rat_result(result_str)

        raise CalculatorError(
            ErrorCode.UNSUPPORTED_OP,
            f"Unknown operator {node.op!r}",
        )

    if isinstance(node, FunctionCallNode):
        if node.name == "abs":
            num, den = _eval_ast(node.args[0])
            return (abs(num), den)
        raise CalculatorError(
            ErrorCode.NOT_IMPLEMENTED,
            f"Function {node.name!r} needs numeric tier (M5)",
        )

    raise CalculatorError(ErrorCode.INTERNAL, f"Unknown AST node type: {type(node)}")


def _format_rat(num: int, den: int) -> str:
    """Format as 'p/q' (or 'p' if q==1)."""
    return str(num) if den == 1 else f"{num}/{den}"


def _parse_rat_result(s: str) -> tuple[int, int]:
    """Parse Idris result string back to (num, den)."""
    if s == "DIV_BY_ZERO":
        raise CalculatorError(ErrorCode.DIV_BY_ZERO, "Division by zero")
    if s == "PARSE_ERROR":
        raise CalculatorError(
            ErrorCode.PARSE_ERROR,
            "Internal error in Idris kernel: malformed input",
        )
    if s.startswith("INTERNAL_ERROR"):
        raise CalculatorError(ErrorCode.INTERNAL, s)
    if "/" in s:
        parts = s.split("/")
        if len(parts) != 2:
            raise CalculatorError(ErrorCode.INTERNAL, f"Unexpected result: {s}")
        return (int(parts[0]), int(parts[1]))
    try:
        return (int(s), 1)
    except ValueError:
        raise CalculatorError(ErrorCode.INTERNAL, f"Unexpected result: {s}")


# ---------------------------------------------------------------------------
# Public API — idris_call for use by _eval_ast
# ---------------------------------------------------------------------------


def _idris_call(op: str, arg: str) -> str:
    """High-level call: tracks count, delegates to worker or degraded."""
    global _worker_call_count, _worker_degraded

    if _worker_degraded:
        return _degraded_call(op, arg)

    try:
        result = _worker_call(op, arg)
        with _worker_lock:
            _worker_call_count += 1
        return result
    except CalculatorError as exc:
        if exc.code == ErrorCode.TIMEOUT:
            raise
        # Degrade on persistent worker failures
        with _worker_lock:
            _worker_degraded = True
        log.warning("Worker error — degrading to Python fallback: %s", exc)
        return _degraded_call(op, arg)


# ---------------------------------------------------------------------------
# Public API — exact_eval
# ---------------------------------------------------------------------------


def exact_eval(ast: ASTNode) -> str:
    """Evaluate an exact arithmetic AST.

    Returns canonical result string 'p/q' (or 'p' if denom == 1).
    Raises CalculatorError on errors.
    """
    num, den = _eval_ast(ast)

    num_digits = len(str(abs(num)))
    den_digits = len(str(abs(den)))
    if max(num_digits, den_digits) > MAX_RESULT_DIGITS:
        raise CalculatorError(
            ErrorCode.LIMIT_EXCEEDED,
            f"Result exceeds maximum of {MAX_RESULT_DIGITS} digits",
        )

    return _format_rat(num, den)


# ---------------------------------------------------------------------------
# Plot worker runner — isolated one-shot subprocess
# ---------------------------------------------------------------------------

_plot_inflight_slots = threading.Semaphore(MAX_INFLIGHT_PLOT_WORKERS)
_plot_lease_registry: dict[int, _WorkerLease] = {}
_plot_registry_lock = threading.Lock()
_plot_reaper_started = False


@dataclass
class _WorkerReservation:
    _lock: threading.Lock = field(default_factory=threading.Lock)
    released: bool = False

    def release_once(self) -> None:
        with self._lock:
            if self.released:
                return
            self.released = True
        _plot_inflight_slots.release()


@dataclass
class _WorkerLease:
    reservation: _WorkerReservation
    proc: subprocess.Popen | None = None


def _build_plot_worker_env() -> dict[str, str]:
    repo_root = str(Path(__file__).resolve().parent.parent)
    return {
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": repo_root,
        "LC_ALL": "C",
        "LANG": "C",
    }


def _register_plot_lease(lease: _WorkerLease) -> None:
    with _plot_registry_lock:
        _plot_lease_registry[lease.proc.pid] = lease


def _remove_plot_lease(pid: int) -> None:
    with _plot_registry_lock:
        _plot_lease_registry.pop(pid, None)


def _start_plot_reaper() -> None:
    global _plot_reaper_started
    if not _plot_reaper_started:
        t = threading.Thread(target=_plot_reaper_loop, daemon=True)
        t.start()
        _plot_reaper_started = True


def _plot_reaper_loop() -> None:
    while True:
        time.sleep(2.0)
        with _plot_registry_lock:
            for pid, lease in list(_plot_lease_registry.items()):
                proc = lease.proc
                if proc.poll() is not None:
                    try:
                        proc.wait(timeout=0.5)
                    except Exception:
                        pass
                    lease.reservation.release_once()
                    _plot_lease_registry.pop(pid, None)


def _kill_and_reap_plot(proc: subprocess.Popen, reservation: _WorkerReservation) -> None:
    """Kill process and try bounded reap. Falls back to reaper if slow."""
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=CLEANUP_GRACE_MS / 1000.0)
        reservation.release_once()
    except subprocess.TimeoutExpired:
        _register_emergency_plot(proc, reservation)
    _remove_plot_lease(proc.pid)


def _register_emergency_plot(proc: subprocess.Popen, reservation: _WorkerReservation) -> None:
    """Register an un-reapable process with the reaper loop."""
    lease = _WorkerLease(reservation=reservation, proc=proc)
    _register_plot_lease(lease)
    _start_plot_reaper()


def _run_plot_worker_inner(
    request_bytes: bytes,
    expected_grid: list[str],
    plot_request,  # PlotRequest — imported lazily
) -> dict:
    """Internal runner: spawn, write, read, validate. Returns validated dict."""
    deadline = time.monotonic() + PLOT_TOTAL_TIMEOUT_MS / 1000.0

    # 1. Reserve inflight slot
    reservation = _WorkerReservation()
    if not _plot_inflight_slots.acquire(blocking=False):
        reservation.release_once()
        raise CalculatorError(
            ErrorCode.LIMIT_EXCEEDED,
            _plot_public_message("LIMIT_EXCEEDED"),
        )

    proc = None
    try:
        # 2. Spawn
        proc = subprocess.Popen(
            [sys.executable, "-m", "calc_mcp.plot_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            close_fds=True,
            cwd=str(Path(__file__).resolve().parent.parent),
            env=_build_plot_worker_env(),
        )
        lease = _WorkerLease(reservation=reservation, proc=proc)
        _register_plot_lease(lease)
        _start_plot_reaper()
    except BaseException:
        if proc is not None:
            _kill_and_reap_plot(proc, reservation)
        else:
            reservation.release_once()
        raise CalculatorError(
            ErrorCode.WORKER_FAILURE,
            _plot_public_message("WORKER_FAILURE"),
        )

    try:
        # 3. Bounded write
        _bounded_plot_write(proc, request_bytes, deadline)

        # 4. Bounded read
        raw = _bounded_plot_read(proc, deadline)

        # 5. Wait for exit
        remaining = deadline - time.monotonic() + CLEANUP_GRACE_MS / 1000.0
        if remaining > 0:
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                _kill_and_reap_plot(proc, reservation)
                raise CalculatorError(
                    ErrorCode.TIMEOUT,
                    _plot_public_message("TIMEOUT"),
                )

        if proc.returncode != 0:
            _kill_and_reap_plot(proc, reservation)
            raise CalculatorError(
                ErrorCode.WORKER_FAILURE,
                _plot_public_message("WORKER_FAILURE"),
            )

        # 6. Parse + validate
        try:
            decoded = raw.decode("utf-8")
            data = json.loads(decoded)
            from calc_mcp.plot_contract import validate_plot_response
            validated = validate_plot_response(data, expected_grid, plot_request)
            # If worker returned ok:false, raise as CalculatorError
            if not validated.get("ok", True):
                err = validated.get("error", {})
                code = err.get("code", "WORKER_FAILURE")
                msg = err.get("message", _plot_public_message("WORKER_FAILURE"))
                _kill_and_reap_plot(proc, reservation)
                raise CalculatorError(
                    getattr(ErrorCode, code, ErrorCode.WORKER_FAILURE),
                    msg,
                )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            _kill_and_reap_plot(proc, reservation)
            raise CalculatorError(
                ErrorCode.WORKER_FAILURE,
                _plot_public_message("WORKER_FAILURE"),
            )
        except CalculatorError:
            _kill_and_reap_plot(proc, reservation)
            raise

        # 7. Clean release
        _remove_plot_lease(proc.pid)
        reservation.release_once()
        return validated

    except BaseException:
        _kill_and_reap_plot(proc, reservation)
        raise


def _bounded_plot_write(
    proc: subprocess.Popen, data: bytes, deadline: float,
) -> None:
    """Write bounded request bytes via non-blocking pipe."""
    import errno
    stdin_fd = proc.stdin.fileno()
    os.set_blocking(stdin_fd, False)
    written = 0
    while written < len(data):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CalculatorError(
                ErrorCode.TIMEOUT,
                _plot_public_message("TIMEOUT"),
            )
        _, w, _ = select.select([], [stdin_fd], [], remaining)
        if not w:
            raise CalculatorError(
                ErrorCode.TIMEOUT,
                _plot_public_message("TIMEOUT"),
            )
        try:
            n = os.write(stdin_fd, data[written:])
        except (BrokenPipeError, OSError) as e:
            if getattr(e, "errno", None) in (errno.EAGAIN, errno.EWOULDBLOCK):
                continue
            raise CalculatorError(
                ErrorCode.WORKER_FAILURE,
                _plot_public_message("WORKER_FAILURE"),
            )
        if n == 0:
            raise CalculatorError(
                ErrorCode.WORKER_FAILURE,
                _plot_public_message("WORKER_FAILURE"),
            )
        written += n
    proc.stdin.close()


def _bounded_plot_read(proc: subprocess.Popen, deadline: float) -> bytes:
    """Read bounded response via non-blocking pipe. Max MAX_PLOT_RESPONSE_BYTES + 1."""
    import errno
    stdout_fd = proc.stdout.fileno()
    os.set_blocking(stdout_fd, False)
    chunks: list[bytes] = []
    total = 0
    saw_newline = False
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        r, _, _ = select.select([stdout_fd], [], [], remaining)
        if not r:
            raise CalculatorError(
                ErrorCode.TIMEOUT,
                _plot_public_message("TIMEOUT"),
            )
        try:
            chunk = os.read(stdout_fd, 65536)
        except (BlockingIOError, OSError) as e:
            if getattr(e, "errno", None) in (errno.EAGAIN, errno.EWOULDBLOCK):
                continue
            raise CalculatorError(
                ErrorCode.WORKER_FAILURE,
                _plot_public_message("WORKER_FAILURE"),
            )
        if not chunk:
            break
        # Check for newline
        newline_idx = chunk.find(b"\n")
        if newline_idx >= 0:
            chunks.append(chunk[:newline_idx])
            saw_newline = True
            total += newline_idx
            # Check for trailing bytes after newline
            if len(chunk) > newline_idx + 1:
                _trim_trailing = chunk[newline_idx + 1:]
                if _trim_trailing.strip():
                    raise CalculatorError(
                        ErrorCode.WORKER_FAILURE,
                        _plot_public_message("WORKER_FAILURE"),
                    )
            break
        else:
            chunks.append(chunk)
            total += len(chunk)
        if total > MAX_PLOT_RESPONSE_BYTES:
            raise CalculatorError(
                ErrorCode.LIMIT_EXCEEDED,
                _plot_public_message("LIMIT_EXCEEDED"),
            )
    if not saw_newline or total == 0:
        raise CalculatorError(
            ErrorCode.WORKER_FAILURE,
            _plot_public_message("WORKER_FAILURE"),
        )
    return b"".join(chunks)


def run_isolated_plot_worker(plot_request) -> dict:
    """Run one isolated plot worker call.

    Args:
        plot_request: A validated PlotRequest instance.

    Returns:
        A strictly validated dict with "result": {"points": [...], "meta": {...}}
        or raises CalculatorError on any failure.
    """
    import json
    from dataclasses import asdict
    from calc_mcp.plot_contract import build_canonical_grid

    # 1. Serialize and budget-check
    serialized = json.dumps(asdict(plot_request), separators=(",", ":"))
    request_bytes = serialized.encode("utf-8") + b"\n"
    if len(request_bytes) > MAX_PLOT_REQUEST_BYTES:
        raise CalculatorError(
            ErrorCode.LIMIT_EXCEEDED,
            _plot_public_message("LIMIT_EXCEEDED"),
        )

    # 2. Build expected grid for validation
    expected_grid = build_canonical_grid(plot_request)

    # 3. Delegate to inner runner
    return _run_plot_worker_inner(request_bytes, expected_grid, plot_request)
