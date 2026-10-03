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


def test_leakage_in_characters_and_words_follows_the_released_text():
    response = "ok. bad word here now"
    ends = (2, 3, 7, 12, 17, 21)
    labels = ["safe", "safe", "safe", "safe", "unsafe", "safe"]
    decisions = tuple(
        ResponseTokenDecision(i, i, end, label)
        for i, (end, label) in enumerate(zip(ends, labels, strict=True), 1)
    )
    base = trace(labels)
    scored = GuardTrace(**{**base.__dict__, "decisions": decisions})
    result = simulate_intervention(
        trace=scored,
        response=response,
        response_ground_truth="unsafe",
        unsafe_start_token=3,
        response_token_count=6,
        mode="token",
        policy="conservative",
        unsafe_start_character=4,
    )
    # Four tokens were released; the unsafe part starts at character 4: "bad word".
    assert result.released_tokens == 4
    assert result.leakage_tokens == 2
    assert result.leakage_characters == len("bad word")
    assert result.leakage_words == 2

    without_onset = replay(labels)
    assert without_onset.leakage_words is None


def rule(labels, mode, *, trigger_mode, trigger_count, trigger_window=None):
    result = simulate_intervention(
        trace=trace(labels),
        response="x" * len(labels),
        response_ground_truth="unsafe",
        unsafe_start_token=1,
        response_token_count=len(labels),
        mode=mode,
        policy="conservative",
        trigger_count=trigger_count,
        trigger_mode=trigger_mode,
        trigger_window=trigger_window,
    )
    return result.signal_token, result.intervention_token


U, S = "unsafe", "safe"


def test_consecutive_rule_needs_an_unbroken_run():
    labels = [U, S, U, S, U, U, S]
    assert rule(labels, "token", trigger_mode="cumulative", trigger_count=2) == (3, 3)
    assert rule(labels, "token", trigger_mode="consecutive", trigger_count=2) == (6, 6)
    assert rule(labels, "token", trigger_mode="consecutive", trigger_count=3) == (None, None)


def test_window_rule_counts_only_recent_tokens():
    labels = [U, S, S, S, U, S, U]
    # Flags at 1 and 5 are four tokens apart: outside a window of 3, inside a window of 5.
    assert rule(labels, "token", trigger_mode="window", trigger_count=2, trigger_window=3) == (7, 7)
    assert rule(labels, "token", trigger_mode="window", trigger_count=2, trigger_window=5) == (5, 5)


def test_buffer_rule_restarts_with_every_release_buffer():
    labels = [S] * 7 + [U, U] + [S] * 6 + [U]
    # The two flags sit on both sides of the chunk boundary at token 8.
    assert rule(labels, "chunk_8", trigger_mode="buffer", trigger_count=2) == (16, 16)
    assert rule(labels, "chunk_16", trigger_mode="buffer", trigger_count=2) == (9, 16)
    # A one-token buffer can never hold two flags.
    assert rule(labels, "token", trigger_mode="buffer", trigger_count=2) == (None, None)


def test_window_arguments_are_validated():
    with pytest.raises(ValueError, match="trigger_window"):
        rule([U], "token", trigger_mode="window", trigger_count=2, trigger_window=1)
    with pytest.raises(ValueError, match="trigger_window"):
        rule([U], "token", trigger_mode="cumulative", trigger_count=1, trigger_window=4)


def test_relabel_can_add_the_controversial_probability():
    base = trace([S, S])
    decisions = (
        ResponseTokenDecision(1, 1, 1, S, unsafe_score=0.3, controversial_score=0.3),
        ResponseTokenDecision(2, 2, 2, S, unsafe_score=0.6, controversial_score=0.1),
    )
    scored = GuardTrace(**{**base.__dict__, "decisions": decisions})

    def labels(**kwargs):
        return [
            item.risk_label for item in relabel_trace(scored, threshold=0.55, **kwargs).decisions
        ]

    assert labels() == [S, U]
    assert labels(include_controversial=True) == [U, U]
