# simplekalman — a high-level Kalman filter library for Python

## Philosophy

Kalman filters are *conceptually simple*:

* sensors produce readings over time
* we combine those readings into a best-guess
* we keep a confidence score for that guess
* we can predict forward when no new readings arrive

The reason people struggle isn’t the algorithm — it’s the **mixing of concerns**:

* domain semantics (GPS vs AR vs IMU)
* timing & ordering (late/out-of-order samples)
* frames & units
* noise / trust tuning
* opaque paper-derived variable names (hx, F, H, Q, R, P)
* and finally… matrices

**simplekalman**'s design goals are to separate a kalman filter into three layers:

1. The **user-oriented layer** - Understanding at this level should not require knowledge of Kalman filter internals or terminology. Users should be able to express their problem in terms of sensors, observations, and desired estimates.
2. The **conceptual implementation layer** - This layer handles data primitives, timing, data conversion, and has the logic of the filter, but expressed as operations on objects, rather than the underlying math.
3. The **numeric kernel layer** - This layer implements the actual mathematical operations (matrices, Jacobians, etc.).


Why each layer is important:
 *  **user** - Even if you understand all of the detail of Kalman Filters, it's easy to get trivial things wrong when dealing with conversions, normalization etc.  An abstracted layer that deals with the semantics of the problem makes it easier to get the base setup and inputs/outputs correct.
  * **conceptual** - The intended logic of the implemenation can be hard to understand when it's encoded as a series of matrix operations.  This layer should, (assuming the underlying kernels/operations are correct) be understandable and easily verifiablel
  * **numeric** - Individual operations and matrix operations are well understood, and are easy to verify in isolation.


> **Sensors → Observations → Filter → Predictions**

No matrices. No Jacobians. No “hx() / F / H / Q / R” leaking into user code.

We want an API where a user describes their problem in *plain measurement terms* and the library does the numeric work underneath.

---

## Core concepts (the user-facing vocabulary)

### Sensor

A **Sensor** is a configured source of information:

* it has an identity (`name`)
* it declares what it measures (e.g. `"YAW"`, `"POSITION_2D"`, `"POSE_2D"`)
* it declares its units (including delta/rate semantics)
* it declares how noisy it is (standard deviation)
* it optionally declares how to handle outliers / late data

A sensor is not “a physical thing” (gyro, GPS chip).
It is an **abstract measurement stream**.

Examples:

* `"YAW"` in `deg/s` (yaw rate)
* `"POSE_2D"` in `m/sample` + `deg/sample` (delta pose per sample)
* `"POSITION_2D"` in `m` (absolute position)

---

### Observation

An **Observation** is one timestamped reading from a Sensor.

It's just data + metadata:

* `time`
* sensor name
* measurement `values`
* optional per-sample `accuracy` (can override sensor's default standard deviation)

Observations are *events*. You pass them into the filter as they arrive.

---

### Estimate

An **Estimate** is what the user wants out: the current best guess of motion.

We treat “prediction” as **a kind of estimate**:

* an estimate produced by integrating forward between samples
* or refined by incorporating observations

So users interact with **one runtime object**:

* `kf.prediction` (the current estimate snapshot)

The estimate is a *noun*: “this is what we believe right now”.

---

### KalmanFilter

`KalmanFilter` is the thing you use.

* You configure it with an Estimate type (e.g. `MotionEstimate`)
* You declare your sensors
* You feed it observations
* You read out predictions (estimates)

It owns:

* current prediction
* sensor registry
* timing/orchestration behavior
* robustness policies

Users do **not** supply mathematical models.
They supply **sensor semantics + noise** and select a **motion-between-samples assumption**.

---

## Design goals

### 1) Human-first interface

Users should not need to know KF internals to use the library correctly.

The API must support a mental model like:

> “I have sensors. They produce readings. I want a motion estimate.”

### 2) Noise as real-world “trust”

Noise parameters represent:

> the standard deviation (1σ) of observation error vs reality
> in the same units as the measurement

This keeps tuning intuitive.

### 3) Units encode meaning

We prefer a single abstraction for “is this absolute, delta, or rate?”:

✅ **Units strings carry baseline semantics**

* `"deg"` → absolute yaw at a time
* `"deg/s"` → yaw rate
* `"deg/sample"` → yaw delta per sample
* `"m"` → absolute position
* `"m/sample"` → translation delta per sample
* `"m/s"` → velocity

So users don’t have to learn a separate `form=ABSOLUTE|DELTA|RATE`.

(Internally the library may decode these to the appropriate handling.)

### 4) Motion-between-samples is explicit

A Kalman filter must assume *something* happens between observations.
This is unavoidable, so we expose it directly in the most human terms possible:

> “How should motion behave between samples?”

Presets:

* `UNCHANGING` — stays where it was unless corrected
* `SMOOTH` — tends to keep moving steadily
* `AGILE` — can change quickly
* `INERTIAL` — rate/delta sensors can drive between-sample motion

This setting belongs to the **Estimate** configuration, not the whole library.

### 5) Policies can be per-sensor

It makes sense to treat different sensors differently:

* GPS might need aggressive outlier rejection
* AR deltas might be mostly reliable but occasionally catastrophic
* yaw-rate streams should often be accepted continuously

So:

* `outlier_handling` can live on sensors
* `late_data_policy` can live on sensors

Some late-data modes (reorder/replay) require filter history support, but the *declaration* can still be sensor-level.

### 6) Diagnostics are optional and pluggable

Diagnostics are useful, but should not distract from the core API.

If included, they should be:

* off by default
* implemented as a lightweight protocol / callback interface
* called by the filter at key events (predict/update/reject/reset)

---

# Sensor specification design

## Canonical internal model (what everything compiles to)

User input compiles down to a fully explicit spec:

```python
SensorSpec(
  name="arkit_delta_pose",
  fields={
    "x": FieldSpec(kind="distance", semantics="delta_per_sample", unit="m/sample", sigma=0.05),
    "y": FieldSpec(kind="distance", semantics="delta_per_sample", unit="m/sample", sigma=0.05),
    "yaw": FieldSpec(kind="angle", semantics="delta_per_sample", unit="rad/sample", sigma=0.5),
  }
)
```

Key: **field names exist even if the user didn't specify them**. For `measures="POSITION_2D"` you get default fields `x, y`. For `POSE_2D` default fields are `x, y, yaw`.

The "sensor gives deltas" semantics isn't something the user must restate for each field — it's implied by the **measure schema** (and optionally reinforced by unit strings).

## Measure schemas

Each measure schema declares its fields with a `kind` per field:

| Measure Schema | Fields |
|----------------|--------|
| `POSITION_2D` | `x(distance), y(distance)` |
| `POSE_2D` | `x(distance), y(distance), yaw(angle)` |
| `YAW` | `yaw(angle)` |

Rate vs absolute vs delta is determined by units (e.g. `deg` vs `deg/s` vs `deg/sample`), not by the measure schema.

This makes domain-map mode robust and keeps user code clean.

## Three user input forms (compile-time sugar)

### A) Single units + single sigma (broadcast)

Good for homogeneous components:

```python
Sensor(name="gps", measures="POSITION_2D", units="m", standard_deviation=2.5)
```

Rules:

* expand units to each field (`x, y`)
* expand sigma to each field
* semantics for each field comes from `measures` (absolute vs delta vs rate)

### B) Domain map (distance vs angle, etc.)

Good for mixed-dimension observations like pose:

```python
Sensor(
  name="arkit_delta_pose",
  measures="POSE_2D",
  units={"distance": "m/sample", "angle": "rad/sample"},
  standard_deviation={"distance": 0.05, "angle": 0.5},
)
```

Rules:

* each field has a `kind` (distance/angle/…); map applies by `kind`
* if a field's kind is missing from the dict → error (or require a fallback)

This is the sweet spot for "don't make me repeat it" while still supporting mixed-dimension observations.

### C) Full per-field spec

Power users / unusual sensors:

```python
Sensor(
  name="arkit_delta_pose",
  measures="POSE_2D",
  fields={
    "x": {"units": "m/sample", "standard_deviation": 0.03},
    "y": {"units": "m/sample", "standard_deviation": 0.06},
    "yaw": {"units": "rad/sample", "standard_deviation": 0.4},
  },
)
```

Rules:

* explicit wins
* you can still allow partial field specs with fallback to (A) or (B)

## Validation / resolution logic

Resolution precedence (simple, deterministic):

1. per-field override (`fields["x"].units`)
2. domain map (`units["distance"]`)
3. scalar broadcast (`units="m"`)

Same precedence applies for standard deviation.

---

# High-level API (as it stands)

## Creating a filter

Bob wants to fuse three sensors into one `MotionEstimate`.

```python
from simplekalman import KalmanFilter
from simplekalman.estimates import MotionEstimate
from simplekalman.policies import Outliers, LateData
from simplekalman import Sensor

kf = KalmanFilter(
    estimate=MotionEstimate(
        motion_between_samples="SMOOTH",   # UNCHANGING | SMOOTH | AGILE | INERTIAL
        stale_after_s=2.0,                 # optional freshness threshold
    ),
    sensors=[
        Sensor(
            name="gps",
            measures="POSITION_2D",
            units="m",
            standard_deviation=2.5,
            outliers=Outliers.REJECT_LIKELY,
            late_data=LateData.REORDER_WITHIN(seconds=0.25),
        ),
        Sensor(
            name="yaw_rate",
            measures="YAW",
            units="deg/s",
            standard_deviation=1.0,
            outliers=Outliers.ACCEPT_ALL,
            late_data=LateData.IGNORE,
        ),
        Sensor(
            name="arkit_delta_pose",
            measures="POSE_2D",
            units={"distance": "m/sample", "angle": "deg/sample"},
            standard_deviation={"distance": 0.05, "angle": 0.5},
            outliers=Outliers.REJECT_LIKELY,
            late_data=LateData.IGNORE,
        ),
    ],
)
```

Notes:

* `Sensor(...)` is intentionally *abstract* (“what is measured”) not physical (“gyro”, “GPS module”).
* Units communicate whether the observation is absolute, per-second, or per-sample.

---

## Feeding observations

User code pushes observations as data arrives:

```python
kf.observe("gps", time=1000.00, values={"x": 1.2, "y": -3.4})
kf.observe("yaw_rate", time=1000.01, values={"yaw": 12.0})
kf.observe("arkit_delta_pose", time=1000.02, values={"x": 0.03, "y": 0.00, "yaw": 0.2})
```

Optional per-sample quality can override sensor defaults:

```python
kf.observe("gps", time=1000.00, values={"x": 1.2, "y": -3.4}, accuracy={"x": 1.0, "y": 1.0})
```

(Exact fields depend on `measures=...`; the library validates them.)

Key rule:

* observations are timestamped
* filter moves forward in time automatically as needed

---

## Getting predictions / estimates

The primary output is always an estimate snapshot:

```python
motion = kf.prediction

print(motion.time)
print(motion.position)
print(motion.velocity)
print(motion.heading)
print(motion.confidence)   # shape TBD
```

Optional convenience:

```python
future = kf.predict(t=1001.00)
```

---

# Roles and ownership

## Sensor owns

* identity
* what it measures
* units
* standard deviation (trust)
* outlier and late-data policy

It does **not** own the math.

## MotionEstimate owns

* what fields exist (position/velocity/heading etc.)
* motion_between_samples preset
* freshness rules (`stale_after_s` / per-field later)

It does **not** own the fusion logic.

## KalmanFilter owns

* runtime state (“current belief”)
* sensor registry
* observation ingestion
* ordering rules
* prediction step execution
* update step execution
* choosing internal filter method (EKF/UKF/etc.) *without exposing it*

---

# Things explicitly *not* exposed (by design)

Users do not write:

* matrices
* Jacobians
* `Q` / `R` / `P` directly
* measurement functions like `hx(x)`
* frame transforms as math primitives

Instead users provide:

* **meaning**
* **units**
* **standard deviation**
* and choose a motion style preset

Internals can still be very sophisticated — but that sophistication stays below the surface.

---

# Next steps (what this document enables)

This manifesto locks in the high-level architecture:

✅ clean separation of sensor semantics from math
✅ estimate/prediction as a single user-visible “belief” snapshot
✅ units-driven observation interpretation
✅ motion-between-samples as an intuitive choice
✅ per-sensor policies for robustness

From here, we can design the lower layers:

1. **Mid-layer orchestration**

* “predict then update” flow
* outlier gating hooks
* late observation handling

2. **Bottom numeric kernel**

* vector/covariance operations
* angle wrapping
* stable linear algebra
* UKF/EKF implementation details hidden behind primitives

This doc is the baseline: any lower-level design must preserve the user story:

> *Configure sensors and an estimate, feed observations, read predictions — without learning Kalman math.*
