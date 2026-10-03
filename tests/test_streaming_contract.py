import pytest

from streamguard_bench.contracts import GuardTrace, PromptDecision, ResponseTokenDecision
from streamguard_bench.streaming import SafetyPolicy, relabel_trace, simulate_intervention


def trace(labels):
    return GuardTrace(
        trace_id="t",
        prompt_decision=PromptDecision("t", "unsafe", "unsafe", (), None, 1),
        decisions=tuple(
            ResponseTokenDecision(i, i, i, label, latency_ms=1) for i, label in enumerate(labels, 1)
        ),
        model_id="fake",
        model_revision=None,
        tokenizer_id="fake",
        tokenizer_revision=None,
        device="cpu",
        total_latency_ms=len(labels),
    )


def test_policy_label_sets_are_not_inverted():
    assert SafetyPolicy.STRICT.blocking_labels == {"unsafe", "controversial"}
    assert SafetyPolicy.CONSERVATIVE.blocking_labels == {"unsafe"}


def test_exact_leakage_and_premature_are_separate_from_delay():
    result = simulate_intervention(
        trace=trace(["safe", "controversial", "safe", "unsafe"]),
        response="abcd",
        response_ground_truth="unsafe",
        unsafe_start_token=3,
        response_token_count=4,
        mode="token",
        policy="strict",
    )
    assert result.released_tokens == 1
    assert result.leakage_tokens == 0
    assert result.signal_offset_tokens == -1
    assert result.premature_block
    assert result.early_lead_tokens == 1
    assert result.detection_delay_tokens is None


def test_late_signal_has_detection_delay():
    result = simulate_intervention(
        trace=trace(["safe", "safe", "safe", "unsafe"]),
        response="abcd",
        response_ground_truth="unsafe",
        unsafe_start_token=3,
        response_token_count=4,
        mode="token",
        policy="conservative",
    )
    assert result.detection_delay_tokens == 1
    assert not result.premature_block


def replay(labels, *, mode="token", trigger_count=1, onset=1):
    return simulate_intervention(
        trace=trace(labels),
        response="x" * len(labels),
        response_ground_truth="unsafe",
        unsafe_start_token=onset,
        response_token_count=len(labels),
        mode=mode,
        policy="conservative",
        trigger_count=trigger_count,
    )


def test_trigger_count_counts_non_consecutive_flags():
    labels = ["unsafe", "safe", "unsafe", "safe", "unsafe", "safe"]
    first = replay(labels)
    third = replay(labels, trigger_count=3)
    assert (first.signal_token, first.intervention_token) == (1, 1)
    assert (third.signal_token, third.intervention_token) == (5, 5)
    assert third.released_tokens == 4
    assert third.trigger_count == 3


def test_trigger_count_accumulates_across_chunk_boundaries():
    labels = ["safe"] * 7 + ["unsafe"] + ["safe"] * 2 + ["unsafe"] + ["safe"] * 5
    result = replay(labels, mode="chunk_8", trigger_count=2)
    assert result.signal_token == 11
    assert result.intervention_token == 16
    assert result.released_tokens == 8
    assert result.post_signal_buffer_delay_tokens == 5


def test_trigger_count_not_reached_leaves_response_unblocked():
    result = replay(["unsafe", "safe", "unsafe"], trigger_count=3)
    assert not result.blocked
    assert result.signal_token is None
    assert result.released_tokens == 3


def test_trigger_count_must_be_positive():
    with pytest.raises(ValueError, match="trigger_count"):
        replay(["safe"], trigger_count=0)


def test_relabel_trace_applies_threshold_to_stored_scores():
    scored = trace(["safe", "safe", "safe"])
    scores = (0.2, 0.7, 0.9)
    scored = GuardTrace(
        **{
            **scored.__dict__,
            "decisions": tuple(
                ResponseTokenDecision(i, i, i, "safe", unsafe_score=score)
                for i, score in enumerate(scores, 1)
            ),
        }
    )
    labels = [item.risk_label for item in relabel_trace(scored, threshold=0.7).decisions]
    assert labels == ["safe", "unsafe", "unsafe"]
    with pytest.raises(ValueError, match="unsafe_score"):
        relabel_trace(trace(["safe"]), threshold=0.5)
