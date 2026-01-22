"""Late data handling policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class LateData:
    """Policy for handling late-arriving measurements."""

    policy: str
    seconds: float | None = None

    IGNORE: ClassVar[LateData]

    @classmethod
    def REORDER_WITHIN(cls, seconds: float) -> LateData:
        """Create a policy that reorders data within a time window."""
        return cls(policy="reorder_within", seconds=seconds)


LateData.IGNORE = LateData(policy="ignore")
