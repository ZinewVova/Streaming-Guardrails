"""Typed contracts shared by dataset, inference, replay, and reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkExample:
    trace_id: str
    query: str
    response: str
    query_label: str
    response_label: str
    unsafe_start_character: int
    safe_prefix: str
    unsafe_start_token: int
    response_token_count: int
    dataset_revision: str


@dataclass(frozen=True)
class PromptDecision:
    trace_id: str
    query_ground_truth: str
    risk_label: str
    risk_categories: tuple[str, ...]
    confidence: float | None
    latency_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResponseTokenDecision:
    token_index: int
    token_id: int
    end_character: int
    risk_label: str
    risk_categories: tuple[str, ...] = ()
    confidence: float | None = None
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        normalized = self.risk_label.lower()
        if normalized not in {"safe", "controversial", "unsafe"}:
            raise ValueError(f"Unsupported guard label: {self.risk_label!r}")
        object.__setattr__(self, "risk_label", normalized)


@dataclass(frozen=True)
class GuardTrace:
    trace_id: str
    prompt_decision: PromptDecision
    decisions: tuple[ResponseTokenDecision, ...]
    model_id: str
    model_revision: str | None
    tokenizer_id: str
    tokenizer_revision: str | None
    device: str
    total_latency_ms: float
    schema_version: int = 2

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["decisions"] = [asdict(item) for item in self.decisions]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GuardTrace:
        if value.get("schema_version") != 2:
            raise ValueError("Checkpoint schema is obsolete; rerun this trace")
        data = dict(value)
        data["prompt_decision"] = PromptDecision(**data["prompt_decision"])
        data["decisions"] = tuple(ResponseTokenDecision(**item) for item in data["decisions"])
        return cls(**data)


@dataclass(frozen=True)
class InterventionResult:
    trace_id: str
    response_ground_truth: str
    mode: str
    policy: str
    blocked: bool
    signal_token: int | None
    intervention_token: int | None
    released_tokens: int
    safe_withheld_tokens: int | None
    leakage_tokens: int | None
    normalized_leakage: float | None
    signal_offset_tokens: int | None
    premature_block: bool
    early_lead_tokens: int | None
    detection_delay_tokens: int | None
    post_signal_buffer_delay_tokens: int | None
    checks: int
    guard_time_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
