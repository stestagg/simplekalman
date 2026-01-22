"""Main KalmanFilter class."""

from __future__ import annotations

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
