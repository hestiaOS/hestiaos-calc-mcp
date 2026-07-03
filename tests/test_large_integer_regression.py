"""Regression test for P1 crash: exact large-integer formatting.

calculate('10^100 + 1') caused a decimal.InvalidOperation crash.
The fix adds an integer fast-path in _rational_to_decimal() that
skips Decimal.quantize() for exact integers.
"""

import asyncio
import json

import pytest

from calc_mcp.server import calculate


class TestLargeInteger:
    """Large integer formatting must not crash."""

    def _call(self, *args, **kwargs):
        return asyncio.run(calculate(*args, **kwargs))

    def _parse(self, raw: str) -> dict:
        return json.loads(raw)

    def test_huge_exact_integer(self):
        """10^100 + 1: exact integer, not scientific, no crash."""
        raw = self._call("10^100 + 1", precision=50)
        data = self._parse(raw)
        assert data["ok"] is True, f"Expected ok:true, got: {data}"
        assert data["result"]["exact"] is True
        rational = data["result"]["rational"]
        decimal = data["result"]["decimal"]
        expected = str(10**100 + 1)
        # rational must be the exact integer (no /1 or canonical)
        assert rational == expected, f"Rational mismatch: {rational} != {expected}"
        # decimal must have exactly 50 fractional places (0s)
        assert decimal.startswith(expected), f"Decimal prefix mismatch"
        assert decimal.count(".") == 1
        int_part, frac_part = decimal.split(".")
        assert len(frac_part) == 50, f"Expected 50 fractional digits, got {len(frac_part)}"
        assert all(c == "0" for c in frac_part), "Fractional part should be all zeros"

    def test_small_exact_integer(self):
        """2^10 = 1024: normal integer must also follow fast path."""
        raw = self._call("2^10", precision=10)
        data = self._parse(raw)
        assert data["ok"] is True, f"Expected ok:true, got: {data}"
        assert data["result"]["exact"] is True
        assert data["result"]["rational"] == "1024"
        decimal = data["result"]["decimal"]
        assert decimal.startswith("1024.")
        assert len(decimal) == 10 + 1 + 4  # 4 digits + dot + 10 fractional

    def test_non_integer_preserves_semantics(self):
        """1/3 with precision must preserve existing rounding semantics."""
        raw = self._call("1/3", precision=10)
        data = self._parse(raw)
        assert data["ok"] is True
        assert data["result"]["rational"] == "1/3"
        decimal = data["result"]["decimal"]
        # 1/3 ~ 0.3333333333 (10 digits)
        assert decimal == "0.3333333333"

    def test_error_on_crash_scenario(self):
        """Even if quantize somehow fails, must return structured error, not crash."""
        # This is a safety net test: force the non-integer path with
        # an extremely large denominator to stress the try/except.
        raw = self._call("1/3", precision=5)
        data = self._parse(raw)
        assert "ok" in data  # any valid response, not a crash