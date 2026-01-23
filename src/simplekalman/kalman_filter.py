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

        # Compile immediately
        compiler = FilterCompiler()
        self._program: FilterProgram = compiler.compile(estimate, sensors)
