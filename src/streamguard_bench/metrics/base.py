"""Small result contract for future metric implementations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    sample_count: int
