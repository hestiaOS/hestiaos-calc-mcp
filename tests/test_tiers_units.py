"""Tests for the lazy Pint unit conversion tier."""

import sys

import pytest

from calc_mcp.tiers.units import convert
from calc_mcp.errors import CalculatorError, ErrorCode


class TestLazyImport:
    """pint must NOT be imported at module top level."""

    def test_pint_not_imported_at_top(self):
        """Check source: 'import pint' must not appear at module level."""
        import ast
        with open("calc_mcp/tiers/units.py") as f:
            tree = ast.parse(f.read())
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import) and any(
                a.name == "pint" for a in node.names
            ):
                pytest.fail(
                    "pint imported at module level in units.py "
                    "(should be inside function body)"
                )
            if isinstance(node, ast.ImportFrom) and node.module == "pint":
                pytest.fail(
                    "pint imported at module level in units.py "
                    "(should be inside function body)"
                )


class TestUnits:
    def test_meter_to_foot(self):
        result = convert("1", "meter", "foot")
        assert result["unit"] == "foot"
        assert abs(result["value"] - 3.28084) < 0.001

    def test_kilogram_to_pound(self):
        result = convert("1", "kilogram", "pound")
        assert result["unit"] == "pound"
        assert abs(result["value"] - 2.20462) < 0.001

    def test_celsius_to_fahrenheit(self):
        result = convert("0", "degC", "degF")
        assert abs(result["value"] - 32.0) < 0.001

    def test_incompatible_units(self):
        with pytest.raises(CalculatorError) as exc:
            convert("1", "meter", "kilogram")
        assert exc.value.code == ErrorCode.INCOMPATIBLE_UNITS

    def test_unknown_from_unit(self):
        with pytest.raises(CalculatorError) as exc:
            convert("1", "flurbo", "meter")
        assert exc.value.code == ErrorCode.UNKNOWN_UNIT

    def test_unknown_to_unit(self):
        with pytest.raises(CalculatorError) as exc:
            convert("1", "meter", "flurbo")
        assert exc.value.code == ErrorCode.UNKNOWN_UNIT
