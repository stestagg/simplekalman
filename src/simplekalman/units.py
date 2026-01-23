"""Unit registry, parsing, and conversion for Pint-based unit handling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pint

ureg = pint.UnitRegistry()

# Define custom dimension for per-sample deltas
ureg.define("sample = [sample_count]")

# Internal base units for each field kind
INTERNAL_BASE_UNITS = {
    "distance": ureg.meter,
    "angle": ureg.radian,
}


class Semantics(Enum):
    """Interpretation of a measurement value."""

    ABSOLUTE = "absolute"
    RATE_PER_S = "rate_per_s"
    DELTA_PER_SAMPLE = "delta_per_sample"


@dataclass(frozen=True)
class UnitDescriptor:
    """Parsed unit description and conversion to internal units."""

    raw_unit: str
    base_unit: str
    semantics: Semantics
    scale: float
    internal_unit: str


class UnitSystem:
    """Unit parsing and conversion to internal units using Pint."""

    @staticmethod
    def parse(unit_str: str, kind: str | None = None) -> UnitDescriptor:
        """Parse a unit string into a UnitDescriptor.

        Args:
            unit_str: The unit string to parse (e.g., "km", "deg/s", "m/sample")
            kind: The field kind for validation (e.g., "distance", "angle")

        Returns:
            UnitDescriptor with parsed unit information and scale factor

        Raises:
            ValueError: If the unit is invalid or incompatible with the kind
        """
        normalized = unit_str.strip()

        try:
            parsed_unit = ureg.Unit(normalized)
        except pint.UndefinedUnitError as e:
            raise ValueError(f"Invalid unit: {unit_str}") from e

        # Detect semantics from dimensionality
        dimensionality = parsed_unit.dimensionality
        semantics = Semantics.ABSOLUTE
        rate_suffix = ""

        if "[sample_count]" in dimensionality and dimensionality["[sample_count]"] == -1:
            semantics = Semantics.DELTA_PER_SAMPLE
            rate_suffix = "/sample"
        elif "[time]" in dimensionality and dimensionality["[time]"] == -1:
            semantics = Semantics.RATE_PER_S
            rate_suffix = "/s"

        # Extract base unit by multiplying out the denominator
        if semantics == Semantics.DELTA_PER_SAMPLE:
            base_quantity = 1 * parsed_unit * ureg.sample
        elif semantics == Semantics.RATE_PER_S:
            base_quantity = 1 * parsed_unit * ureg.second
        else:
            base_quantity = 1 * parsed_unit

        base_dimensionality = base_quantity.dimensionality

        # Determine internal unit based on kind or infer from dimensionality
        if kind is not None:
            if kind not in INTERNAL_BASE_UNITS:
                raise ValueError(f"Unknown kind: {kind}")
            internal_base_unit = INTERNAL_BASE_UNITS[kind]
            # Validate compatibility
            if not base_quantity.is_compatible_with(internal_base_unit):
                raise ValueError(
                    f"Unit '{unit_str}' is not compatible with kind '{kind}'"
                )
        else:
            # Infer kind from dimensionality
            if "[length]" in base_dimensionality:
                internal_base_unit = ureg.meter
            elif base_dimensionality == {} or "[angle]" in str(base_dimensionality):
                # Pint treats angles as dimensionless, check explicitly
                try:
                    base_quantity.to(ureg.radian)
                    internal_base_unit = ureg.radian
                except pint.DimensionalityError:
                    # Truly dimensionless, default to keeping as-is
                    internal_base_unit = base_quantity.units
            else:
                internal_base_unit = base_quantity.units

        # Calculate scale factor
        try:
            converted = base_quantity.to(internal_base_unit)
            scale = converted.magnitude
        except pint.DimensionalityError as e:
            raise ValueError(
                f"Cannot convert '{unit_str}' to internal units"
            ) from e

        # Build internal unit string
        internal_unit_str = f"{internal_base_unit:~}".replace(" ", "")
        if rate_suffix:
            internal_unit_str = f"{internal_unit_str}{rate_suffix}"

        # Determine base unit name for the descriptor
        base_unit_str = f"{internal_base_unit:~}".replace(" ", "")

        return UnitDescriptor(
            raw_unit=unit_str,
            base_unit=base_unit_str,
            semantics=semantics,
            scale=scale,
            internal_unit=internal_unit_str,
        )
