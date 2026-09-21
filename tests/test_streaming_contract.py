from streamguard_bench.contracts import GuardTrace, PromptDecision, ResponseTokenDecision
from streamguard_bench.streaming import SafetyPolicy, simulate_intervention


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
