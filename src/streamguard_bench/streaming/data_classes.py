"""Model-agnostic streaming configuration and tokenization contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from streamguard_bench.contracts import (
    GuardTrace,
    InterventionResult,
    ResponseTokenDecision,
)


class BufferMode(StrEnum):
    TOKEN = "token"
    CHUNK_8 = "chunk_8"
    CHUNK_16 = "chunk_16"
    CHUNK_32 = "chunk_32"
    SENTENCE = "sentence"
    FULL_BUFFERED = "full_buffered"


class SafetyPolicy(StrEnum):
    """Explicit label sets; names alone are never used to infer behavior."""

    STRICT = "strict"
    CONSERVATIVE = "conservative"

    @property
    def blocking_labels(self) -> frozenset[str]:
        if self is SafetyPolicy.STRICT:
            return frozenset({"unsafe", "controversial"})
        return frozenset({"unsafe"})


class TriggerMode(StrEnum):
    """How flagged tokens add up to a stop decision; `trigger_count` is the k in each rule."""

    CUMULATIVE = "cumulative"  # k flagged tokens anywhere in the response so far
    CONSECUTIVE = "consecutive"  # k flagged tokens in a row
    WINDOW = "window"  # k flagged tokens among the last `trigger_window` tokens
    BUFFER = "buffer"  # k flagged tokens inside the current release buffer


@dataclass(frozen=True)
class TokenizedResponse:
    token_ids: tuple[int, ...]
    end_characters: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.end_characters):
            raise ValueError("token_ids and end_characters must have equal length")
        if tuple(sorted(self.end_characters)) != self.end_characters:
            raise ValueError("end_characters must be monotonically non-decreasing")


TokenDecision = ResponseTokenDecision
DEFAULT_MODES = tuple(BufferMode)
DEFAULT_POLICIES = tuple(SafetyPolicy)

__all__ = [
    "DEFAULT_MODES",
    "DEFAULT_POLICIES",
    "BufferMode",
    "GuardTrace",
    "InterventionResult",
    "ResponseTokenDecision",
    "SafetyPolicy",
    "TokenDecision",
    "TokenizedResponse",
    "TriggerMode",
]
