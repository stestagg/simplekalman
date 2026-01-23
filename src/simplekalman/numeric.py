"""Numeric primitives for Kalman filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .conceptual import MeasurementPlan, NormalizedObservation, ProcessPreset, StateDescriptor


@dataclass
class LinearBelief:
    """Linear belief state represented by mean/covariance."""

    state: np.ndarray
    covariance: np.ndarray
    index: dict[str, int]

    def as_dict(self) -> dict[str, float]:
        """Return state values as a mapping."""
        return {name: float(self.state[idx]) for name, idx in self.index.items()}


def initialize_belief(
    descriptor: StateDescriptor, *, initial_variance: float = 1_000.0
) -> LinearBelief:
    """Create an initial belief with large covariance."""
    size = len(descriptor.components)
    state = np.zeros(size, dtype=float)
    covariance = np.eye(size, dtype=float) * initial_variance
    index = {component.name: i for i, component in enumerate(descriptor.components)}
    return LinearBelief(state=state, covariance=covariance, index=index)


def predict(
    belief: LinearBelief, dt: float, process: ProcessPreset, descriptor: StateDescriptor
) -> LinearBelief:
    """Predict state forward using a constant-velocity model."""
    transition = build_transition_matrix(descriptor, belief.index, dt)
    noise = build_process_noise(descriptor, process, dt)
    state = transition @ belief.state
    covariance = transition @ belief.covariance @ transition.T + noise
    return LinearBelief(state=state, covariance=covariance, index=belief.index)


def update(
    belief: LinearBelief,
    plan: MeasurementPlan,
    observation: NormalizedObservation,
) -> LinearBelief:
    """Apply a measurement update to the belief."""
    state = belief.state
    measurement, matrix, noise = build_measurement_matrices(
        belief, plan, observation
    )
    residual = measurement - matrix @ state
    innovation = matrix @ belief.covariance @ matrix.T + noise
    kalman_gain = belief.covariance @ matrix.T @ np.linalg.inv(innovation)
    state = state + kalman_gain @ residual
    covariance = (np.eye(state.shape[0]) - kalman_gain @ matrix) @ belief.covariance
    return LinearBelief(state=state, covariance=covariance, index=belief.index)


def build_transition_matrix(
    descriptor: StateDescriptor, index: dict[str, int], dt: float
) -> np.ndarray:
    """Build a linear transition matrix for the configured state."""
    size = len(descriptor.components)
    transition = np.eye(size, dtype=float)
    _assign_transition(transition, index, "position.x", "velocity.vx", dt)
    _assign_transition(transition, index, "position.y", "velocity.vy", dt)
    _assign_transition(transition, index, "heading.yaw", "heading_rate.yaw_rate", dt)
    return transition


def _assign_transition(
    transition: np.ndarray,
    index: dict[str, int],
    position_key: str,
    rate_key: str,
    dt: float,
) -> None:
    if position_key in index and rate_key in index:
        transition[index[position_key], index[rate_key]] = dt


def build_process_noise(
    descriptor: StateDescriptor, process: ProcessPreset, dt: float
) -> np.ndarray:
    """Build a diagonal process noise matrix."""
    size = len(descriptor.components)
    base = _process_noise_scale(process)
    scale = base * max(dt, 1e-6)
    return np.eye(size, dtype=float) * scale


def _process_noise_scale(process: ProcessPreset) -> float:
    preset = process.name.upper()
    if preset == "UNCHANGING":
        return 1e-4
    if preset == "SMOOTH":
        return 5e-3
    if preset == "AGILE":
        return 2e-2
    if preset == "INERTIAL":
        return 1e-3
    return 1e-3


def build_measurement_matrices(
    belief: LinearBelief,
    plan: MeasurementPlan,
    observation: NormalizedObservation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct measurement vector, matrix, and noise covariance."""
    size = len(plan.fields)
    state_size = belief.state.shape[0]
    measurement = np.zeros(size, dtype=float)
    matrix = np.zeros((size, state_size), dtype=float)
    noise = np.zeros((size, size), dtype=float)
    for row, field in enumerate(plan.fields):
        target = field.target.removeprefix("state.")
        if target not in belief.index:
            raise ValueError(f"Unknown target '{target}' in measurement plan")
        idx = belief.index[target]
        value = observation.values[field.field_name]
        sigma = observation.sigmas[field.field_name]
        if field.action == "apply_delta":
            measurement[row] = belief.state[idx] + value
        else:
            measurement[row] = value
        matrix[row, idx] = 1.0
        noise[row, row] = sigma**2
    return measurement, matrix, noise
