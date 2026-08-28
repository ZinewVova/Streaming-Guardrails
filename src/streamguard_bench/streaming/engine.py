"""Replay token-level guard decisions under different release-buffer policies."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from .data_classes import (
    BufferMode,
    GuardTrace,
    InterventionResult,
    SafetyPolicy,
    TokenDecision,
)

_TERMINAL_PUNCTUATION = re.compile(r"[.!?。！？]+[\"'»”’)*\]]*$")
_COMMON_ABBREVIATIONS = frozenset(
    {
        "dr.",
        "mr.",
        "mrs.",
        "ms.",
        "prof.",
        "sr.",
        "jr.",
        "st.",
        "vs.",
        "etc.",
        "e.g.",
        "i.e.",
    }
)


def simulate_intervention(
    *,
    trace: GuardTrace,
    response: str,
    source_split: str,
    ground_truth_label: str,
    harm_categories: Sequence[str],
    mode: BufferMode | str,
    policy: SafetyPolicy | str,
    onset_lower_token: int | None,
    onset_upper_token: int | None,
    max_sentence_tokens: int = 128,
) -> InterventionResult:
    """Apply one release policy to an already-computed guard trace."""

    selected_mode = BufferMode(mode)
    selected_policy = SafetyPolicy(policy)
    decisions = trace.decisions
    blocking = selected_policy.blocking_labels
    checkpoints = _buffer_checkpoints(
        decisions,
        response,
        selected_mode,
        max_sentence_tokens=max_sentence_tokens,
    )

    signal_token = next(
        (item.token_index for item in decisions if item.risk_label in blocking),
        None,
    )
    released = 0
    intervention = None
    checks = 0

    for start, end in checkpoints:
        checks += 1
        window = decisions[start:end]
        if any(item.risk_label in blocking for item in window):
            intervention = end
            break
        released = end

    blocked = intervention is not None
    processed_end = intervention if intervention is not None else len(decisions)
    guard_time = sum(item.latency_ms for item in decisions[:processed_end])
    leakage_min, leakage_max = _leakage_interval(
        released,
        onset_lower_token,
        onset_upper_token,
        ground_truth_label,
    )
    delay_min, delay_max = _delay_interval(
        intervention,
        onset_lower_token,
        onset_upper_token,
        ground_truth_label,
    )

    return InterventionResult(
        trace_id=trace.trace_id,
        source_split=source_split,
        ground_truth_label=ground_truth_label,
        harm_categories=tuple(harm_categories),
        mode=selected_mode.value,
        policy=selected_policy.value,
        blocked=blocked,
        signal_token=signal_token,
        intervention_token=intervention,
        released_tokens=released,
        leakage_min=leakage_min,
        leakage_max=leakage_max,
        detection_delay_min=delay_min,
        detection_delay_max=delay_max,
        checks=checks,
        guard_time_ms=guard_time,
    )


def simulate_all(
    *,
    trace: GuardTrace,
    response: str,
    source_split: str,
    ground_truth_label: str,
    harm_categories: Sequence[str],
    modes: Iterable[BufferMode | str],
    policies: Iterable[SafetyPolicy | str],
    onset_lower_token: int | None,
    onset_upper_token: int | None,
    max_sentence_tokens: int = 128,
) -> list[InterventionResult]:
    """Replay every requested mode-policy combination for one trace."""

    return [
        simulate_intervention(
            trace=trace,
            response=response,
            source_split=source_split,
            ground_truth_label=ground_truth_label,
            harm_categories=harm_categories,
            mode=mode,
            policy=policy,
            onset_lower_token=onset_lower_token,
            onset_upper_token=onset_upper_token,
            max_sentence_tokens=max_sentence_tokens,
        )
        for mode in modes
        for policy in policies
    ]


def _buffer_checkpoints(
    decisions: Sequence[TokenDecision],
    response: str,
    mode: BufferMode,
    *,
    max_sentence_tokens: int,
) -> list[tuple[int, int]]:
    count = len(decisions)
    if not count:
        return []
    if mode is BufferMode.TOKEN:
        return [(index, index + 1) for index in range(count)]
    if mode is BufferMode.FULL_BUFFERED:
        return [(0, count)]
    if mode.value.startswith("chunk_"):
        size = int(mode.value.rsplit("_", maxsplit=1)[1])
        return [(start, min(start + size, count)) for start in range(0, count, size)]
    return _sentence_checkpoints(
        decisions,
        response,
        max_sentence_tokens=max_sentence_tokens,
    )


def _sentence_checkpoints(
    decisions: Sequence[TokenDecision],
    response: str,
    *,
    max_sentence_tokens: int,
) -> list[tuple[int, int]]:
    if max_sentence_tokens < 1:
        raise ValueError("max_sentence_tokens must be positive")

    checkpoints: list[tuple[int, int]] = []
    start = 0
    previous_character = 0
    for end, decision in enumerate(decisions, start=1):
        prefix = response[: decision.end_character]
        new_text = response[previous_character : decision.end_character]
        forced = end - start >= max_sentence_tokens
        boundary = _ends_sentence(prefix) and (bool(new_text.strip()) or "\n" in new_text)
        if forced or boundary or end == len(decisions):
            checkpoints.append((start, end))
            start = end
        previous_character = decision.end_character
    return checkpoints


def _ends_sentence(prefix: str) -> bool:
    """Conservative online boundary detector that never reads future text."""

    stripped = prefix.rstrip()
    if not stripped:
        return False
    if prefix.endswith("\n"):
        return True
    last_word = stripped.rsplit(maxsplit=1)[-1].lower()
    if last_word in _COMMON_ABBREVIATIONS:
        return False
    return bool(_TERMINAL_PUNCTUATION.search(stripped))


def _leakage_interval(
    released: int,
    lower: int | None,
    upper: int | None,
    ground_truth_label: str,
) -> tuple[int, int]:
    if ground_truth_label != "unsafe":
        return 0, 0
    if lower is None or upper is None:
        return 0, released
    return max(0, released - upper), max(0, released - lower)


def _delay_interval(
    intervention: int | None,
    lower: int | None,
    upper: int | None,
    ground_truth_label: str,
) -> tuple[int | None, int | None]:
    if ground_truth_label != "unsafe" or intervention is None:
        return None, None
    if lower is None or upper is None:
        return None, None
    return intervention - upper, intervention - lower
