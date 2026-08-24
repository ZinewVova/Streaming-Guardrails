"""Shared, model-agnostic data contracts used across the project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RawStreamSafeRecord:
    """An untouched source record with provenance."""

    source_file: str
    source_split: str
    source_row: int
    raw_data: dict[str, Any]


@dataclass(frozen=True)
class StreamingTrace:
    """Canonical trace contract to be populated in the normalization stage."""

    trace_id: str
    prompt: str
    response: str
    final_label: str
    harm_categories: tuple[str, ...]
    language: str
    unsafe_start_character: int | None


@dataclass(frozen=True)
class GuardDecision:
    """One model decision made on a generated prefix."""

    label: str
    risk_score: float | None
    generated_tokens: int
    checked_tokens: int
    latency_ms: float


@dataclass(frozen=True)
class EvaluationRecord:
    """A model decision enriched with experiment context."""

    trace_id: str
    model_id: str
    mode: str
    generated_tokens: int
    released_tokens: int
    label: str
    risk_score: float | None
    blocked: bool
    latency_ms: float
