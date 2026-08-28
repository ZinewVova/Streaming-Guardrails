from streamguard_bench.streaming import (
    BufferMode,
    GuardTrace,
    TokenDecision,
    simulate_intervention,
)


def _trace(response: str, labels: dict[int, str]) -> GuardTrace:
    decisions = tuple(
        TokenDecision(
            token_index=index,
            token_id=ord(character),
            end_character=index,
            risk_label=labels.get(index, "safe"),
            latency_ms=1.0,
        )
        for index, character in enumerate(response, start=1)
    )
    return GuardTrace(
        trace_id="trace",
        decisions=decisions,
        model_id="fake",
        model_revision="one",
        tokenizer_id="fake",
        tokenizer_revision="one",
        device="cpu",
        total_latency_ms=float(len(decisions)),
    )


def _simulate(response: str, labels: dict[int, str], mode: str, policy: str = "strict"):
    return simulate_intervention(
        trace=_trace(response, labels),
        response=response,
        source_split="val",
        ground_truth_label="unsafe",
        harm_categories=("Violence",),
        mode=mode,
        policy=policy,
        onset_lower_token=8,
        onset_upper_token=10,
    )


def test_token_and_chunk_modes_have_distinct_intervention_points():
    response = "a" * 40
    labels = {9: "unsafe"}

    token = _simulate(response, labels, BufferMode.TOKEN.value)
    chunk_8 = _simulate(response, labels, BufferMode.CHUNK_8.value)
    chunk_16 = _simulate(response, labels, BufferMode.CHUNK_16.value)
    chunk_32 = _simulate(response, labels, BufferMode.CHUNK_32.value)

    assert (token.intervention_token, token.released_tokens) == (9, 8)
    assert (chunk_8.intervention_token, chunk_8.released_tokens) == (16, 8)
    assert (chunk_16.intervention_token, chunk_16.released_tokens) == (16, 0)
    assert (chunk_32.intervention_token, chunk_32.released_tokens) == (32, 0)
    assert chunk_8.signal_token == 9
    assert chunk_8.leakage_min == 0
    assert chunk_8.leakage_max == 0


def test_conservative_policy_blocks_controversial_label():
    response = "abcdefgh"
    strict = _simulate(response, {3: "controversial"}, "token", "strict")
    conservative = _simulate(response, {3: "controversial"}, "token", "conservative")

    assert strict.blocked is False
    assert conservative.blocked is True
    assert conservative.intervention_token == 3


def test_sentence_mode_releases_only_complete_safe_sentence():
    response = "First sentence. Second unsafe."
    unsafe_index = response.index("unsafe") + 1
    result = _simulate(response, {unsafe_index: "unsafe"}, "sentence")

    assert result.blocked is True
    assert result.released_tokens == len("First sentence.")
    assert result.intervention_token == len(response)


def test_sentence_mode_forces_check_after_maximum_buffer():
    response = "a" * 130
    result = simulate_intervention(
        trace=_trace(response, {129: "unsafe"}),
        response=response,
        source_split="val",
        ground_truth_label="unsafe",
        harm_categories=(),
        mode="sentence",
        policy="strict",
        onset_lower_token=128,
        onset_upper_token=129,
        max_sentence_tokens=128,
    )

    assert result.released_tokens == 128
    assert result.intervention_token == 130
    assert result.checks == 2


def test_full_buffer_never_releases_a_blocked_response():
    result = _simulate("unsafe response", {3: "unsafe"}, "full_buffered")

    assert result.blocked is True
    assert result.signal_token == 3
    assert result.intervention_token == len("unsafe response")
    assert result.released_tokens == 0


def test_safe_trace_has_zero_leakage_and_no_delay():
    response = "safe"
    result = simulate_intervention(
        trace=_trace(response, {}),
        response=response,
        source_split="train",
        ground_truth_label="safe",
        harm_categories=(),
        mode="token",
        policy="strict",
        onset_lower_token=None,
        onset_upper_token=None,
    )

    assert result.leakage_min == result.leakage_max == 0
    assert result.detection_delay_min is None
    assert result.detection_delay_max is None
