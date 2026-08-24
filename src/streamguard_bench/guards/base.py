"""Stable interface for future guard-model integrations."""

from __future__ import annotations

from typing import Protocol

from streamguard_bench.contracts import GuardDecision


class Guard(Protocol):
    """A guard that can score a complete prefix without exposing model details."""

    def reset(self, prompt: str) -> None:
        """Reset conversation state before processing a new response."""

    def score_prefix(self, prefix: str) -> GuardDecision:
        """Return a safety decision for the currently observed response prefix."""
