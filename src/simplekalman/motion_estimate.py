"""Motion estimation configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MotionEstimate:
    """Configuration for how the filter estimates motion between samples."""

    motion_between_samples: str
    stale_after_s: float | None = None
