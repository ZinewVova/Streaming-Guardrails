"""Metric interfaces and future implementations."""

from streamguard_bench.metrics.base import MetricResult

__all__ = ["MetricResult"]
from .streaming import compute_streaming_metrics

__all__ = ["compute_streaming_metrics"]
