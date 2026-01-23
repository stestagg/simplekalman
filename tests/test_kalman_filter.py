"""Tests for KalmanFilter initialization and automatic compilation."""

import math

import pytest

from simplekalman import FilterProgram, KalmanFilter, MotionEstimate, Sensor
from simplekalman.numeric import LinearBelief
from simplekalman.units import Semantics, UnitSystem


def test_program_created_on_init():
    """Basic compilation happens in __init__."""
    kf = KalmanFilter(
        estimate=MotionEstimate(motion_between_samples="UNCHANGING"),
        sensors=[Sensor(name="gps", measures="POSITION_2D", units="m", standard_deviation=1.0)],
    )
    assert isinstance(kf._program, FilterProgram)


def test_sensor_spec_structure():
    """Sensor specs are correctly compiled."""
    kf = KalmanFilter(
        estimate=MotionEstimate(motion_between_samples="SMOOTH"),
        sensors=[Sensor(name="gps", measures="POSITION_2D", units="m", standard_deviation=2.5)],
    )
    assert "gps" in kf._program.sensor_specs
    spec = kf._program.sensor_specs["gps"]
    assert spec.name == "gps"
    assert spec.measures == "POSITION_2D"
    assert len(spec.fields) == 2  # x, y
    field_names = {f.name for f in spec.fields}
    assert field_names == {"x", "y"}


def test_pose_2d_fields():
    """POSE_2D creates x, y, yaw fields."""
    kf = KalmanFilter(
        estimate=MotionEstimate(motion_between_samples="SMOOTH"),
        sensors=[
            Sensor(
                name="arkit",
                measures="POSE_2D",
                units={"distance": "m", "angle": "rad"},
                standard_deviation={"distance": 0.1, "angle": 0.5},
            )
        ],
    )
    spec = kf._program.sensor_specs["arkit"]
    field_names = {f.name for f in spec.fields}
    assert field_names == {"x", "y", "yaw"}


def test_degree_to_radian_conversion():
    """Degree units convert to radians."""
    kf = KalmanFilter(
        estimate=MotionEstimate(motion_between_samples="SMOOTH"),
        sensors=[Sensor(name="gyro", measures="YAW", units="deg", standard_deviation=1.0)],
    )
    spec = kf._program.sensor_specs["gyro"]
    yaw_field = spec.fields[0]
    assert yaw_field.internal_unit == "rad"
    # 180 deg should convert to pi radians
    assert abs(yaw_field.to_internal(180.0) - 3.14159) < 0.001


def test_rate_semantics():
    """Rate semantics detected from units."""
    kf = KalmanFilter(
        estimate=MotionEstimate(motion_between_samples="SMOOTH"),
        sensors=[Sensor(name="gyro", measures="YAW", units="deg/s", standard_deviation=1.0)],
    )
    spec = kf._program.sensor_specs["gyro"]
    assert spec.fields[0].semantics == Semantics.RATE_PER_S


def test_delta_semantics():
    """Delta semantics detected from units."""
    kf = KalmanFilter(
        estimate=MotionEstimate(motion_between_samples="SMOOTH"),
        sensors=[
            Sensor(name="odom", measures="POSITION_2D", units="m/sample", standard_deviation=0.01)
        ],
    )
    spec = kf._program.sensor_specs["odom"]
    assert spec.fields[0].semantics == Semantics.DELTA_PER_SAMPLE


def test_unchanging_state_components():
    """UNCHANGING motion creates 3 state components."""
    kf = KalmanFilter(
        estimate=MotionEstimate(motion_between_samples="UNCHANGING"),
        sensors=[Sensor(name="gps", measures="POSITION_2D", units="m", standard_deviation=1.0)],
    )
    components = kf._program.state_descriptor.components
    assert len(components) == 3  # position.x, position.y, heading.yaw


def test_smooth_state_components():
    """SMOOTH motion adds velocity components (6 total)."""
    kf = KalmanFilter(
        estimate=MotionEstimate(motion_between_samples="SMOOTH"),
        sensors=[Sensor(name="gps", measures="POSITION_2D", units="m", standard_deviation=1.0)],
    )
    components = kf._program.state_descriptor.components
    assert len(components) == 6  # position + heading + velocity + heading_rate
    component_names = {c.name for c in components}
    assert "velocity.vx" in component_names
    assert "velocity.vy" in component_names


def test_measurement_plan_actions():
    """Measurement plans use correct actions."""
    kf = KalmanFilter(
        estimate=MotionEstimate(motion_between_samples="SMOOTH"),
        sensors=[
            Sensor(name="gps", measures="POSITION_2D", units="m", standard_deviation=1.0),
            Sensor(name="gyro", measures="YAW", units="deg/s", standard_deviation=1.0),
            Sensor(name="odom", measures="POSITION_2D", units="m/sample", standard_deviation=0.01),
        ],
    )
    # Absolute -> observe
    gps_plan = kf._program.plans["gps"]
    assert gps_plan.fields[0].action == "observe"

    # Rate -> integrate_rate
    gyro_plan = kf._program.plans["gyro"]
    assert gyro_plan.fields[0].action == "integrate_rate"

    # Delta -> apply_delta
    odom_plan = kf._program.plans["odom"]
    assert odom_plan.fields[0].action == "apply_delta"


def test_multiple_sensors():
    """Multiple sensors all appear in program."""
    kf = KalmanFilter(
        estimate=MotionEstimate(motion_between_samples="SMOOTH"),
        sensors=[
            Sensor(name="gps", measures="POSITION_2D", units="m", standard_deviation=2.5),
            Sensor(name="gyro", measures="YAW", units="deg/s", standard_deviation=1.0),
            Sensor(
                name="arkit",
                measures="POSE_2D",
                units={"distance": "m/sample", "angle": "deg/sample"},
                standard_deviation={"distance": 0.05, "angle": 0.5},
            ),
        ],
    )
    assert set(kf._program.sensor_specs.keys()) == {"gps", "gyro", "arkit"}
    assert set(kf._program.plans.keys()) == {"gps", "gyro", "arkit"}


# --- Pint-based unit conversion tests ---


def test_km_to_meter_scale():
    """km converts to meter with scale=1000."""
    desc = UnitSystem.parse("km", kind="distance")
    assert desc.scale == 1000.0
    assert desc.base_unit == "m"
    assert desc.internal_unit == "m"
    assert desc.semantics == Semantics.ABSOLUTE


def test_km_per_hour_to_meter_per_second():
    """km/h converts to m/s with correct scale."""
    desc = UnitSystem.parse("km/h", kind="distance")
    # 1 km/h = 1000m / 3600s ≈ 0.2778 m/s
    expected_scale = 1000.0 / 3600.0
    assert abs(desc.scale - expected_scale) < 0.001
    assert desc.semantics == Semantics.RATE_PER_S
    assert desc.internal_unit == "m/s"


def test_deg_per_sample_conversion():
    """deg/sample converts with deg→rad scale."""
    desc = UnitSystem.parse("deg/sample", kind="angle")
    expected_scale = math.pi / 180.0
    assert abs(desc.scale - expected_scale) < 1e-9
    assert desc.semantics == Semantics.DELTA_PER_SAMPLE
    assert desc.base_unit == "rad"


def test_invalid_unit_raises_value_error():
    """Invalid unit string raises ValueError."""
    with pytest.raises(ValueError, match="Invalid unit"):
        UnitSystem.parse("not_a_unit", kind="distance")


def test_incompatible_kind_raises_value_error():
    """Unit incompatible with kind raises ValueError."""
    with pytest.raises(ValueError, match="not compatible with kind"):
        UnitSystem.parse("kg", kind="distance")


def test_feet_to_meter():
    """Feet converts to meters correctly."""
    desc = UnitSystem.parse("ft", kind="distance")
    # 1 ft = 0.3048 m
    assert abs(desc.scale - 0.3048) < 0.0001


def test_observe_updates_belief_state():
    """Observations update the numeric belief state."""
    kf = KalmanFilter(
        estimate=MotionEstimate(motion_between_samples="UNCHANGING"),
        sensors=[Sensor(name="gps", measures="POSITION_2D", units="m", standard_deviation=1.0)],
    )
    kf.observe("gps", time=1.0, values={"x": 2.0, "y": -1.0})
    belief = kf.prediction
    assert isinstance(belief, LinearBelief)
    values = belief.as_dict()
    assert values["position.x"] == pytest.approx(2.0, abs=0.5)
    assert values["position.y"] == pytest.approx(-1.0, abs=0.5)


def test_arcmin_to_radian():
    """Arcminutes convert to radians."""
    desc = UnitSystem.parse("arcmin", kind="angle")
    # 1 arcmin = (1/60) degree = pi/(180*60) radian
    expected_scale = math.pi / (180.0 * 60.0)
    assert abs(desc.scale - expected_scale) < 1e-10
    assert desc.base_unit == "rad"


def test_mph_rate_conversion():
    """Miles per hour rate converts correctly."""
    desc = UnitSystem.parse("mph", kind="distance")
    # 1 mph = 1609.344 m / 3600 s ≈ 0.44704 m/s
    expected_scale = 1609.344 / 3600.0
    assert abs(desc.scale - expected_scale) < 0.0001
    assert desc.semantics == Semantics.RATE_PER_S
    assert desc.internal_unit == "m/s"


def test_backward_compat_meter():
    """Existing 'm' unit still works."""
    desc = UnitSystem.parse("m", kind="distance")
    assert desc.scale == 1.0
    assert desc.base_unit == "m"
    assert desc.semantics == Semantics.ABSOLUTE


def test_backward_compat_deg():
    """Existing 'deg' unit still works."""
    desc = UnitSystem.parse("deg", kind="angle")
    expected_scale = math.pi / 180.0
    assert abs(desc.scale - expected_scale) < 1e-9
    assert desc.base_unit == "rad"


def test_backward_compat_rad():
    """Existing 'rad' unit still works."""
    desc = UnitSystem.parse("rad", kind="angle")
    assert desc.scale == 1.0
    assert desc.base_unit == "rad"


def test_backward_compat_m_per_s():
    """Existing 'm/s' unit still works."""
    desc = UnitSystem.parse("m/s", kind="distance")
    assert desc.scale == 1.0
    assert desc.semantics == Semantics.RATE_PER_S
    assert desc.internal_unit == "m/s"


def test_backward_compat_deg_per_s():
    """Existing 'deg/s' unit still works."""
    desc = UnitSystem.parse("deg/s", kind="angle")
    expected_scale = math.pi / 180.0
    assert abs(desc.scale - expected_scale) < 1e-9
    assert desc.semantics == Semantics.RATE_PER_S
    assert desc.internal_unit == "rad/s"


def test_backward_compat_m_per_sample():
    """Existing 'm/sample' unit still works."""
    desc = UnitSystem.parse("m/sample", kind="distance")
    assert desc.scale == 1.0
    assert desc.semantics == Semantics.DELTA_PER_SAMPLE
    assert desc.internal_unit == "m/sample"
