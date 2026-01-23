"""Main KalmanFilter class."""

from __future__ import annotations

from .conceptual import FilterCompiler, FilterProgram
from .motion_estimate import MotionEstimate
from .sensor import Sensor


class KalmanFilter:
    """A configurable Kalman filter for state estimation."""

    def __init__(
        self,
        estimate: MotionEstimate,
        sensors: list[Sensor],
    ) -> None:
        self.estimate = estimate
        self.sensors = sensors
        self._sensor_registry: dict[str, Sensor] = {s.name: s for s in sensors}
        self._compiler = FilterCompiler()

    def compile_program(self) -> FilterProgram:
        """Compile configuration into a conceptual filter program."""
        return self._compiler.compile(self.estimate, self.sensors)
