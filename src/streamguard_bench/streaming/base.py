"""Stable interface for future token, chunk, and sentence policies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class StreamingPolicy(Protocol):
    """Select token positions at which a guard decision is allowed."""

    def checkpoints(self, token_ids: Sequence[int], decoded_text: str) -> Sequence[int]:
        """Return monotonically increasing, one-based token positions."""
