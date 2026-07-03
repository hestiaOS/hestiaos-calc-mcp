"""Resource limits for calculator evaluation.

All limits are constants. Violation raises CalculatorError(LIMIT_EXCEEDED).
"""

# ---------------------------------------------------------------------------
# Input limits
# ---------------------------------------------------------------------------
MAX_INPUT_LEN: int = 4096  # characters in a single expression

# ---------------------------------------------------------------------------
# AST limits
# ---------------------------------------------------------------------------
MAX_AST_DEPTH: int = 64  # maximum parse tree depth
MAX_AST_NODES: int = 1024  # maximum total nodes in a single expression

# ---------------------------------------------------------------------------
# Numeric limits
# ---------------------------------------------------------------------------
MAX_INT_POW_EXP: int = 1000  # maximum integer exponent for ** (absolute)
MAX_RESULT_DIGITS: int = 100_000  # maximum digits in a single result
MAX_OUTPUT_BYTES: int = 1_048_576  # 1 MiB — maximum output size

# ---------------------------------------------------------------------------
# Timeouts (milliseconds)
# ---------------------------------------------------------------------------
CALL_TIMEOUT_MS: int = 2000  # wall-clock timeout for calculate
SYMBOLIC_TIMEOUT_MS: int = 5000  # timeout for symbolic operations

# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------
NUMERIC_DPS_DEFAULT: int = 50  # default decimal places for mpmath
NUMERIC_DPS_MAX: int = 1000  # maximum allowed decimal places

# ---------------------------------------------------------------------------
# Kernel worker
# ---------------------------------------------------------------------------
KERNEL_MAX_CALLS: int = 10_000  # calls before recycling the Idris worker

# ---------------------------------------------------------------------------
# Plot / sample_function limits (provisional safety ceilings)
# ---------------------------------------------------------------------------
MAX_SAMPLE_COUNT: int = 10_000
MAX_SAMPLE_EXPR_LEN: int = 256
MAX_PLOT_X_CHARS: int = 256
MAX_PLOT_X_ABS: int = 10_000_000_000_000
MAX_PLOT_REQUEST_BYTES: int = 4096
MAX_PLOT_RESPONSE_BYTES: int = 10 * 1024 * 1024
PLOT_TOTAL_TIMEOUT_MS: int = 10_000
CLEANUP_GRACE_MS: int = 500
READ_CHUNK_BYTES: int = 65_536
MAX_INFLIGHT_PLOT_WORKERS: int = 10
GRID_GUARD_DIGITS: int = 10
GUARD_DIGITS: int = 10
MAX_RESULT_EXPONENT: int = 1_000
MAX_ERROR_CODE_CHARS: int = 64
MAX_ERROR_MSG_CHARS: int = 128
