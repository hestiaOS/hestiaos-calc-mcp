"""Unit conversion tier (Pint, lazy import).

Loaded on first convert_units call only.
"""

from __future__ import annotations

from calc_mcp.errors import CalculatorError, ErrorCode


def convert(value: str, from_unit: str, to_unit: str) -> dict:
    """Convert a value between units.

    Args:
        value: Numeric value string (e.g., '10', '3.14').
        from_unit: Source unit string (e.g., 'meter', 'kg').
        to_unit: Target unit string (e.g., 'foot', 'pound').

    Returns:
        Dict with keys: value, unit, exact.

    Raises:
        CalculatorError on incompatible or unknown units.
    """
    # Lazy import: pint loaded only on first call
    import pint as _pint

    ureg = _pint.UnitRegistry()

    try:
        qty = ureg.Quantity(float(value), from_unit)
    except _pint.errors.UndefinedUnitError as exc:
        raise CalculatorError(
            ErrorCode.UNKNOWN_UNIT,
            f"Unknown unit {from_unit!r}: {exc}",
        ) from exc

    try:
        converted = qty.to(to_unit)
    except _pint.errors.UndefinedUnitError as exc:
        raise CalculatorError(
            ErrorCode.UNKNOWN_UNIT,
            f"Unknown unit {to_unit!r}: {exc}",
        ) from exc
    except _pint.errors.DimensionalityError as exc:
        raise CalculatorError(
            ErrorCode.INCOMPATIBLE_UNITS,
            f"Incompatible units: {from_unit!r} → {to_unit!r}: {exc}",
        ) from exc

    return {
        "value": float(converted.magnitude),
        "unit": str(converted.units),
        "exact": False,
    }
