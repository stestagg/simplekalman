"""Main KalmanFilter class."""

from __future__ import annotations

from .conceptual import FilterCompiler, FilterProgram, Observation
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

    def observe(
        self,
        sensor: str,
        time: float,
        values: dict[str, float],
        accuracy: float | dict[str, float] | None = None,
    ) -> None:
        """Ingest an observation and advance the filter state."""
        observation = Observation(sensor=sensor, time=time, values=values, accuracy=accuracy)
        self._program.ingest(observation)
        self._program.process()

    def predict(self, t: float) -> object | None:
        """Advance the filter prediction to a future time."""
        self._program.predict_to(t)
        return self._program.state.belief

    @property
    def prediction(self) -> object | None:
        """Return the current belief state."""
        return self._program.state.belief
