from dataclasses import FrozenInstanceError

import pytest

from streamguard_bench.contracts import GuardDecision, RawStreamSafeRecord, StreamingTrace


def test_raw_record_preserves_payload() -> None:
    payload = {"response": "unchanged"}
    record = RawStreamSafeRecord("fixture.json", "train", 0, payload)
    assert record.raw_data is payload


def test_streaming_trace_is_immutable() -> None:
    trace = StreamingTrace("t1", "prompt", "response", "safe", (), "en", None)
    with pytest.raises(FrozenInstanceError):
        trace.final_label = "unsafe"  # type: ignore[misc]


def test_guard_decision_keeps_numeric_context() -> None:
    decision = GuardDecision("safe", 0.1, 10, 10, 2.5)
    assert decision.generated_tokens == decision.checked_tokens == 10
