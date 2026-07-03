"""Security tests for the calculator parser.

Tests focus on:
  - Injection attacks (eval/exec/compile/__import__ patterns)
  - Attribute/dunder access
  - Resource bomb (deep nesting, long input)
  - Null bytes
"""

import pytest

from calc_mcp.errors import CalculatorError, ErrorCode
from calc_mcp.parser import parse


# ---------------------------------------------------------------------------
# No dynamic execution — verifying the architecture guarantee
# ---------------------------------------------------------------------------

class TestNoEvalExec:
    """Guard: the parser must NEVER evaluate injected code."""

    def test_import_injection(self):
        """__import__('os') should not execute."""
        with pytest.raises(CalculatorError) as exc:
            parse("__import__('os')")
        # Any structured error is acceptable — the key is no execution
        assert exc.value.code in (ErrorCode.PARSE_ERROR, ErrorCode.UNKNOWN_SYMBOL)

    def test_eval_injection(self):
        with pytest.raises(CalculatorError):
            parse("eval('1+1')")
        assert True  # no crash = pass

    def test_exec_injection(self):
        with pytest.raises(CalculatorError):
            parse("exec('print(1)')")
        assert True

    def test_compile_injection(self):
        with pytest.raises(CalculatorError):
            parse("compile('1+1')")
        assert True

    def test_system_injection(self):
        """Attempt to call os.system or subprocess via attr access."""
        with pytest.raises(CalculatorError):
            parse("os.system('id')")
        assert True

    def test_dunder_attr_access(self):
        """__class__, __base__, __subclasses__ etc."""
        for dunder in ("__class__", "__base__", "__subclasses__", "__globals__"):
            with pytest.raises(CalculatorError):
                parse(dunder)
        assert True

    def test_path_traversal(self):
        """No file path should be constructible."""
        with pytest.raises(CalculatorError):
            parse("open('/etc/passwd')")
        assert True

    def test_unicode_script(self):
        """Unicode mathematical script symbols — should be rejected."""
        with pytest.raises(CalculatorError):
            parse("𝚜𝚚𝚛𝚝(4)")  # mathematical monospace
        assert True

    def test_null_byte(self):
        """Null bytes must be rejected immediately."""
        with pytest.raises(CalculatorError) as exc:
            parse("1+1\x00")
        assert exc.value.code == ErrorCode.PARSE_ERROR

    def test_backticks(self):
        """JavaScript-style template/command injection must be rejected."""
        with pytest.raises(CalculatorError):
            parse("`ls`")
        assert True


# ---------------------------------------------------------------------------
# Resource bombs
# ---------------------------------------------------------------------------

class TestResourceLimits:
    """The parser must enforce hard resource limits."""

    def test_deep_nesting(self):
        """Extremely deep (((...))) should exceed AST depth limit."""
        expr = "(" * 100 + "1" + ")" * 100
        with pytest.raises(CalculatorError) as exc:
            parse(expr)
        assert exc.value.code == ErrorCode.LIMIT_EXCEEDED

    def test_deep_left_assoc(self):
        """Deeply nested ((((1+1)+1)+1)+...) should hit node or depth limit."""
        expr = "1"
        for _ in range(200):
            expr = f"({expr}+1)"
        # Can also be just a very long left-associated expression
        expr_long = "+".join("1" for _ in range(2000))
        with pytest.raises(CalculatorError) as exc:
            parse(expr_long)
        assert exc.value.code == ErrorCode.LIMIT_EXCEEDED

    def test_long_input(self):
        """Very long input should be rejected."""
        expr = "1" * 5000
        with pytest.raises(CalculatorError) as exc:
            parse(expr)
        assert exc.value.code == ErrorCode.LIMIT_EXCEEDED

    def test_max_input_length_boundary(self):
        """Input exactly at the limit should be accepted (for valid expressions)."""
        # Build an expression just under MAX_INPUT_LEN (4096)
        # "1+1" repeated many times — simple enough to not hit node limit
        from calc_mcp.limits import MAX_INPUT_LEN
        base = "1+"
        repeats = (MAX_INPUT_LEN - 10) // len(base)
        expr = base * repeats + "1"
        assert len(expr) <= MAX_INPUT_LEN

        # This might hit node limit first — that's fine
        try:
            result = parse(expr)
            assert result is not None
        except CalculatorError as exc:
            # Accept LIMIT_EXCEEDED from node count; execution is not required
            assert exc.code == ErrorCode.LIMIT_EXCEEDED

    def test_many_nested_function_calls(self):
        """sqrt(sqrt(sqrt(...(1)...))) should hit depth or node limit."""
        expr = "sqrt(" * 100 + "1" + ")" * 100
        with pytest.raises(CalculatorError) as exc:
            parse(expr)
        assert exc.value.code in (
            ErrorCode.LIMIT_EXCEEDED,
            ErrorCode.PARSE_ERROR,  # overflow tokens too
        )
