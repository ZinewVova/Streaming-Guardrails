"""Model-agnostic contracts for token-level guard traces and buffer simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class BufferMode(StrEnum):
    """When buffered response tokens may be released to a user."""

    TOKEN = "token"
    CHUNK_8 = "chunk_8"
    CHUNK_16 = "chunk_16"
    CHUNK_32 = "chunk_32"
    SENTENCE = "sentence"
    FULL_BUFFERED = "full_buffered"


class SafetyPolicy(StrEnum):
    """How the three Qwen3Guard labels map to a binary intervention."""

    STRICT = "strict"
    CONSERVATIVE = "conservative"

    @property
    def blocking_labels(self) -> frozenset[str]:
        if self is SafetyPolicy.CONSERVATIVE:
            return frozenset({"unsafe", "controversial"})
        return frozenset({"unsafe"})


@dataclass(frozen=True)
class TokenizedResponse:
    """One immutable tokenization of an assistant response."""

    token_ids: tuple[int, ...]
    end_characters: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.end_characters):
            raise ValueError("token_ids and end_characters must have equal length")
        if tuple(sorted(self.end_characters)) != self.end_characters:
            raise ValueError("end_characters must be monotonically non-decreasing")


@dataclass(frozen=True)
class TokenDecision:
    """The guard decision returned for one new assistant token."""

    token_index: int
    token_id: int
    end_character: int
    risk_label: str
    risk_categories: tuple[str, ...] = ()
    label_confidence: float | None = None
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        normalized = self.risk_label.lower()
        if normalized not in {"safe", "controversial", "unsafe"}:
            raise ValueError(f"Unsupported guard label: {self.risk_label!r}")
        object.__setattr__(self, "risk_label", normalized)


@dataclass(frozen=True)
class GuardTrace:
    """All raw token decisions produced by one model pass over one trace."""

    trace_id: str
    decisions: tuple[TokenDecision, ...]
    model_id: str
    model_revision: str | None
    tokenizer_id: str
    tokenizer_revision: str | None
    device: str
    total_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "decisions": [asdict(item) for item in self.decisions]}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GuardTrace:
        data = dict(value)
        data["decisions"] = tuple(TokenDecision(**item) for item in data["decisions"])
        return cls(**data)


@dataclass(frozen=True)
class InterventionResult:
    """One trace after applying a buffer mode and a safety policy."""

    trace_id: str
    source_split: str
    ground_truth_label: str
    harm_categories: tuple[str, ...]
    mode: str
    policy: str
    blocked: bool
    signal_token: int | None
    intervention_token: int | None
    released_tokens: int
    leakage_min: int
    leakage_max: int
    detection_delay_min: int | None
    detection_delay_max: int | None
    checks: int
    guard_time_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_MODES: tuple[BufferMode, ...] = (
    BufferMode.TOKEN,
    BufferMode.CHUNK_8,
    BufferMode.CHUNK_16,
    BufferMode.CHUNK_32,
    BufferMode.SENTENCE,
    BufferMode.FULL_BUFFERED,
)

DEFAULT_POLICIES: tuple[SafetyPolicy, ...] = (
    SafetyPolicy.STRICT,
    SafetyPolicy.CONSERVATIVE,
)
