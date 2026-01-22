"""Outliers handling policy enum."""

from enum import Enum


class Outliers(Enum):
    """Policy for handling outlier measurements."""

    ACCEPT_ALL = "accept_all"
    REJECT_LIKELY = "reject_likely"
