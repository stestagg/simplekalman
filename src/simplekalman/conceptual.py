"""Conceptual layer structures for Kalman filter compilation and runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import pi
from typing import Callable, Iterable, Protocol

from .late_data import LateData
from .motion_estimate import MotionEstimate
from .outliers import Outliers
from .sensor import Sensor


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


@dataclass(frozen=True)
class FieldSchema:
    """Schema definition for a single measurement field."""

    name: str
    kind: str
    default_unit: str


@dataclass(frozen=True)
class MeasurementSchema:
    """Measurement schema for a named measure."""

    name: str
    fields: tuple[FieldSchema, ...]

    def field_names(self) -> tuple[str, ...]:
        """Return the field names in this schema."""
        return tuple(field.name for field in self.fields)


DEFAULT_SCHEMAS: dict[str, MeasurementSchema] = {
    "POSITION_2D": MeasurementSchema(
        name="POSITION_2D",
        fields=(
            FieldSchema(name="x", kind="distance", default_unit="m"),
            FieldSchema(name="y", kind="distance", default_unit="m"),
        ),
    ),
    "POSE_2D": MeasurementSchema(
        name="POSE_2D",
        fields=(
            FieldSchema(name="x", kind="distance", default_unit="m"),
            FieldSchema(name="y", kind="distance", default_unit="m"),
            FieldSchema(name="yaw", kind="angle", default_unit="rad"),
        ),
    ),
    "YAW": MeasurementSchema(
        name="YAW",
        fields=(FieldSchema(name="yaw", kind="angle", default_unit="rad"),),
    ),
}


@dataclass(frozen=True)
class FieldSpec:
    """Fully resolved field specification."""

    name: str
    kind: str
    unit: str
    semantics: Semantics
    sigma: float
    outlier_policy: Outliers
    late_policy: LateData
    internal_unit: str
    to_internal: Callable[[float], float]
    from_internal: Callable[[float], float]


@dataclass(frozen=True)
class SensorSpec:
    """Resolved sensor specification."""

    name: str
    measures: str
    fields: tuple[FieldSpec, ...]

    def field_map(self) -> dict[str, FieldSpec]:
        """Return a mapping of field name to FieldSpec."""
        return {field.name: field for field in self.fields}


@dataclass(frozen=True)
class StateComponent:
    """A named state component with internal units."""

    name: str
    unit: str


@dataclass(frozen=True)
class StateDescriptor:
    """Descriptor of the estimate state layout."""

    components: tuple[StateComponent, ...]
    preset: str


@dataclass(frozen=True)
class ProcessPreset:
    """Process model preset for motion between samples."""

    name: str


@dataclass(frozen=True)
class MeasurementPlanField:
    """Mapping between a sensor field and a state target."""

    field_name: str
    target: str
    semantics: Semantics
    action: str


@dataclass(frozen=True)
class MeasurementPlan:
    """Plan describing how a sensor measurement affects the state."""

    sensor_name: str
    fields: tuple[MeasurementPlanField, ...]


@dataclass(frozen=True)
class Observation:
    """An incoming observation from a sensor."""

    sensor: str
    time: float
    values: dict[str, float]
    accuracy: float | dict[str, float] | None = None


@dataclass(frozen=True)
class NormalizedObservation:
    """Validated and normalized observation with internal units."""

    sensor: str
    time: float
    values: dict[str, float]
    sigmas: dict[str, float]


@dataclass
class RuntimeState:
    """Runtime state for the conceptual filter layer."""

    belief: object | None = None
    last_time: float | None = None


class Kernel(Protocol):
    """Opaque numeric kernel interface."""

    def predict(self, belief: object, dt: float, process: ProcessPreset) -> object:
        """Advance the belief forward in time."""

    def update(
        self,
        belief: object,
        plan: MeasurementPlan,
        observation: NormalizedObservation,
    ) -> object:
        """Update the belief with a measurement."""


class UnitSystem:
    """Unit parsing and conversion to internal units."""

    @staticmethod
    def parse(unit: str) -> UnitDescriptor:
        """Parse a unit string into a UnitDescriptor."""
        normalized = unit.strip().lower()
        semantics = Semantics.ABSOLUTE
        base = normalized
        suffix = ""
        if "/s" in normalized or "per_s" in normalized or "per_sec" in normalized:
            semantics = Semantics.RATE_PER_S
            base = normalized.split("/")[0].split("per_")[0]
            suffix = "/s"
        elif "/sample" in normalized or "per_sample" in normalized:
            semantics = Semantics.DELTA_PER_SAMPLE
            base = normalized.split("/")[0].split("per_")[0]
            suffix = "/sample"

        base = base.strip()
        scale = 1.0
        internal_base = base
        if base in {"deg", "degree", "degrees"}:
            internal_base = "rad"
            scale = pi / 180.0
        elif base in {"rad", "radian", "radians"}:
            internal_base = "rad"
        elif base in {"m", "meter", "meters"}:
            internal_base = "m"

        internal_unit = f"{internal_base}{suffix}"
        return UnitDescriptor(
            raw_unit=unit,
            base_unit=internal_base,
            semantics=semantics,
            scale=scale,
            internal_unit=internal_unit,
        )


class SensorCompiler:
    """Compile sensor configuration into explicit specs."""

    def __init__(self, schemas: dict[str, MeasurementSchema] | None = None) -> None:
        self._schemas = schemas or DEFAULT_SCHEMAS

    def compile(self, sensor: Sensor) -> SensorSpec:
        """Compile a Sensor into a SensorSpec."""
        if sensor.measures not in self._schemas:
            raise ValueError(f"Unknown schema: {sensor.measures}")
        schema = self._schemas[sensor.measures]
        units = self._expand_per_field(sensor.units, schema.field_names())
        sigmas = self._expand_per_field(sensor.standard_deviation, schema.field_names())
        fields: list[FieldSpec] = []
        for field in schema.fields:
            unit_str = units[field.name]
            sigma = sigmas[field.name]
            unit_desc = UnitSystem.parse(unit_str)
            to_internal = self._scale_to_internal(unit_desc.scale)
            from_internal = self._scale_from_internal(unit_desc.scale)
            fields.append(
                FieldSpec(
                    name=field.name,
                    kind=field.kind,
                    unit=unit_str,
                    semantics=unit_desc.semantics,
                    sigma=sigma,
                    outlier_policy=sensor.outliers,
                    late_policy=sensor.late_data,
                    internal_unit=unit_desc.internal_unit,
                    to_internal=to_internal,
                    from_internal=from_internal,
                )
            )
        return SensorSpec(name=sensor.name, measures=sensor.measures, fields=tuple(fields))

    @staticmethod
    def _expand_per_field(
        value: str | dict[str, str] | float | dict[str, float],
        field_names: Iterable[str],
    ) -> dict[str, str] | dict[str, float]:
        if isinstance(value, dict):
            missing = [name for name in field_names if name not in value]
            if missing:
                raise ValueError(f"Missing field values for {missing}")
            return value
        return {name: value for name in field_names}

    @staticmethod
    def _scale_to_internal(scale: float) -> Callable[[float], float]:
        return lambda value: value * scale

    @staticmethod
    def _scale_from_internal(scale: float) -> Callable[[float], float]:
        if scale == 0:
            return lambda value: value
        return lambda value: value / scale


class EstimateCompiler:
    """Compile MotionEstimate into state descriptors."""

    def compile(self, estimate: MotionEstimate) -> tuple[StateDescriptor, ProcessPreset]:
        preset = estimate.motion_between_samples.upper()
        components = [
            StateComponent(name="position.x", unit="m"),
            StateComponent(name="position.y", unit="m"),
            StateComponent(name="heading.yaw", unit="rad"),
        ]
        if preset in {"SMOOTH", "AGILE", "INERTIAL"}:
            components.extend(
                [
                    StateComponent(name="velocity.vx", unit="m/s"),
                    StateComponent(name="velocity.vy", unit="m/s"),
                    StateComponent(name="heading_rate.yaw_rate", unit="rad/s"),
                ]
            )
        descriptor = StateDescriptor(components=tuple(components), preset=preset)
        return descriptor, ProcessPreset(name=preset)


class PlanCompiler:
    """Compile measurement plans for sensors."""

    @staticmethod
    def compile(sensor_specs: Iterable[SensorSpec]) -> dict[str, MeasurementPlan]:
        plans: dict[str, MeasurementPlan] = {}
        for spec in sensor_specs:
            fields = []
            for field in spec.fields:
                target = PlanCompiler._map_target(field.name, field.semantics)
                action = PlanCompiler._action_for_semantics(field.semantics)
                fields.append(
                    MeasurementPlanField(
                        field_name=field.name,
                        target=target,
                        semantics=field.semantics,
                        action=action,
                    )
                )
            plans[spec.name] = MeasurementPlan(sensor_name=spec.name, fields=tuple(fields))
        return plans

    @staticmethod
    def _map_target(field_name: str, semantics: Semantics) -> str:
        if field_name in {"x", "y"} and semantics == Semantics.DELTA_PER_SAMPLE:
            return f"state.position.{field_name}"
        if field_name in {"x", "y"}:
            return f"state.position.{field_name}"
        if field_name in {"vx", "vy"}:
            return f"state.velocity.{field_name}"
        if field_name == "yaw":
            return "state.heading.yaw"
        if field_name == "yaw_rate":
            return "state.heading_rate.yaw_rate"
        return f"state.{field_name}"

    @staticmethod
    def _action_for_semantics(semantics: Semantics) -> str:
        if semantics == Semantics.ABSOLUTE:
            return "observe"
        if semantics == Semantics.RATE_PER_S:
            return "integrate_rate"
        return "apply_delta"


class ObservationNormalizer:
    """Validate and normalize incoming observations."""

    def normalize(self, observation: Observation, spec: SensorSpec) -> NormalizedObservation:
        field_map = spec.field_map()
        missing = [name for name in field_map if name not in observation.values]
        extra = [name for name in observation.values if name not in field_map]
        if missing or extra:
            raise ValueError(f"Field mismatch missing={missing} extra={extra}")
        sigmas = self._resolve_sigmas(observation, field_map)
        values: dict[str, float] = {}
        for name, field_spec in field_map.items():
            values[name] = field_spec.to_internal(observation.values[name])
        return NormalizedObservation(
            sensor=observation.sensor,
            time=observation.time,
            values=values,
            sigmas=sigmas,
        )

    @staticmethod
    def _resolve_sigmas(
        observation: Observation, field_map: dict[str, FieldSpec]
    ) -> dict[str, float]:
        if observation.accuracy is None:
            return {name: field.sigma for name, field in field_map.items()}
        if isinstance(observation.accuracy, dict):
            return {name: observation.accuracy.get(name, field.sigma) for name, field in field_map.items()}
        return {name: float(observation.accuracy) for name in field_map}


class EventQueue:
    """Queue that handles time ordering and late data policies."""

    def __init__(self) -> None:
        self._queue: list[NormalizedObservation] = []

    def insert(self, observation: NormalizedObservation, policy: LateData, current_time: float | None) -> None:
        if current_time is not None and observation.time < current_time:
            if policy.policy == "ignore":
                return
            if policy.policy == "reorder_within" and policy.seconds is not None:
                if observation.time < current_time - policy.seconds:
                    return
        self._queue.append(observation)
        self._queue.sort(key=lambda obs: obs.time)

    def pop_next(self) -> NormalizedObservation | None:
        if not self._queue:
            return None
        return self._queue.pop(0)


@dataclass
class FilterProgram:
    """Runtime program for the conceptual layer."""

    sensor_specs: dict[str, SensorSpec]
    state_descriptor: StateDescriptor
    process_preset: ProcessPreset
    plans: dict[str, MeasurementPlan]
    kernel: Kernel | None = None
    state: RuntimeState = field(default_factory=RuntimeState)
    queue: EventQueue = field(default_factory=EventQueue)
    normalizer: ObservationNormalizer = field(default_factory=ObservationNormalizer)

    def ingest(self, observation: Observation) -> None:
        """Normalize and enqueue an observation."""
        spec = self.sensor_specs[observation.sensor]
        normalized = self.normalizer.normalize(observation, spec)
        self.queue.insert(normalized, spec.fields[0].late_policy, self.state.last_time)

    def predict_to(self, time: float) -> None:
        """Advance the runtime to the requested time."""
        if self.state.last_time is None:
            self.state.last_time = time
            return
        dt = time - self.state.last_time
        if dt <= 0:
            self.state.last_time = time
            return
        if self.kernel is None:
            raise NotImplementedError("Kernel is required to predict.")
        if self.state.belief is None:
            raise NotImplementedError("Belief initialization not implemented.")
        self.state.belief = self.kernel.predict(self.state.belief, dt, self.process_preset)
        self.state.last_time = time

    def apply_measurement(self, observation: NormalizedObservation) -> None:
        """Apply a measurement update."""
        plan = self.plans[observation.sensor]
        if self.kernel is None:
            raise NotImplementedError("Kernel is required to apply measurements.")
        if self.state.belief is None:
            raise NotImplementedError("Belief initialization not implemented.")
        self.state.belief = self.kernel.update(self.state.belief, plan, observation)

    def process(self) -> None:
        """Process all queued observations in time order."""
        while True:
            obs = self.queue.pop_next()
            if obs is None:
                return
            self.predict_to(obs.time)
            self.apply_measurement(obs)


class FilterCompiler:
    """Compile configuration into a FilterProgram."""

    def __init__(self) -> None:
        self._sensor_compiler = SensorCompiler()
        self._estimate_compiler = EstimateCompiler()

    def compile(self, estimate: MotionEstimate, sensors: Iterable[Sensor]) -> FilterProgram:
        sensor_specs = [self._sensor_compiler.compile(sensor) for sensor in sensors]
        state_descriptor, process_preset = self._estimate_compiler.compile(estimate)
        plans = PlanCompiler.compile(sensor_specs)
        return FilterProgram(
            sensor_specs={spec.name: spec for spec in sensor_specs},
            state_descriptor=state_descriptor,
            process_preset=process_preset,
            plans=plans,
        )
