"""Sensor configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from .late_data import LateData
from .outliers import Outliers


@dataclass
class Sensor:
    """Configuration for a sensor input to the Kalman filter."""

    name: str
    measures: str
    units: str | dict[str, str]
    standard_deviation: float | dict[str, float]
    outliers: Outliers = field(default=Outliers.ACCEPT_ALL)
    late_data: LateData = field(default_factory=lambda: LateData.IGNORE)
