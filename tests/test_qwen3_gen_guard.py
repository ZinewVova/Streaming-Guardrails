from pathlib import Path

import pandas as pd
import pytest

from streamguard_bench.contracts import GuardTrace, ResponseTokenDecision
from streamguard_bench.guards import Qwen3GuardGenAdapter
from streamguard_bench.pipeline import run_or_load
from streamguard_bench.streaming import (
    BufferMode,
    checkpoint_tokens,
    restrict_to_checkpoints,
    simulate_all,
)


class CharTokenizer:
    """One token per character, so offsets and decoding are trivially exact."""

    def __call__(self, text, **kwargs):
        encoded = {"input_ids": [ord(char) for char in text]}
        if kwargs.get("return_offsets_mapping"):
            encoded["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return encoded

    def decode(self, token_ids, **_):
        return "".join(chr(item) for item in token_ids)


class ScriptedClassifier:
    """Returns scripted probabilities for the length of the assistant text, else safe."""

    device = "test"

    def __init__(self, by_length=None):
        self.by_length = dict(by_length or {})
        self.calls = []

    def classify(self, messages):
        self.calls.append(messages)
        if messages[-1]["role"] != "assistant":
            return (0.9, 0.05, 0.05)
        return self.by_length.get(len(messages[-1]["content"]), (0.9, 0.05, 0.05))


UNSAFE = (0.1, 0.1, 0.8)
CONTROVERSIAL = (0.2, 0.5, 0.3)


def adapter(by_length=None, **kwargs):
    classifier = ScriptedClassifier(by_length)
    guard = Qwen3GuardGenAdapter(classifier=classifier, tokenizer=CharTokenizer(), **kwargs)
    return guard, classifier


def score_all(guard, response):
    tokenized = guard.tokenize_response(response)
    guard.start("q")
    return [
        guard.score_token(token_id=token_id, token_index=index, end_character=end)
        for index, (token_id, end) in enumerate(
            zip(tokenized.token_ids, tokenized.end_characters, strict=True), start=1
        )
    ]


def test_only_checkpoint_tokens_are_scored():
    guard, classifier = adapter(modes=["chunk_8", "full_buffered"])
    decisions = score_all(guard, "a" * 20)
    scored = [item.token_index for item in decisions if item.confidence is not None]
    assert scored == [8, 16, 20]
    assert len(classifier.calls) == 1 + 3


def test_token_mode_scores_every_token():
    guard, _ = adapter(modes=["token", "chunk_8"])
    decisions = score_all(guard, "a" * 5)
    assert all(item.confidence is not None for item in decisions)


def test_label_is_the_most_probable_class_and_all_scores_are_kept():
    guard, _ = adapter({8: UNSAFE, 16: CONTROVERSIAL}, modes=["chunk_8"])
    decisions = {item.token_index: item for item in score_all(guard, "a" * 16)}
    assert decisions[8].risk_label == "unsafe"
    assert decisions[8].unsafe_score == pytest.approx(0.8)
    assert decisions[16].risk_label == "controversial"
    assert decisions[16].controversial_score == pytest.approx(0.5)
    assert decisions[1].risk_label == "safe" and decisions[1].unsafe_score is None


def test_prompt_is_classified_alone_and_keeps_scores():
    guard, classifier = adapter(modes=["full_buffered"])
    decision = guard.start("hello", trace_id="t", query_ground_truth="unsafe")
    assert classifier.calls[0] == [{"role": "user", "content": "hello"}]
    assert decision.risk_label == "safe"
    assert decision.safe_score == pytest.approx(0.9)


def test_classifier_and_tokenizer_must_come_together():
    with pytest.raises(ValueError, match="together"):
        Qwen3GuardGenAdapter(classifier=ScriptedClassifier())


def test_checkpoint_tokens_cover_every_requested_mode():
    ends = list(range(1, 21))
    assert checkpoint_tokens(ends, "a" * 20, ["chunk_8"]) == {8, 16, 20}
    assert checkpoint_tokens(ends, "a" * 20, ["full_buffered"]) == {20}
    assert checkpoint_tokens(ends, "a" * 20, ["chunk_8", "full_buffered"]) == {8, 16, 20}


def _trace(flagged):
    decisions = tuple(
        ResponseTokenDecision(
            index, index, index, "unsafe" if index in flagged else "safe", confidence=0.9
        )
        for index in range(1, 41)
    )
    return GuardTrace("t", None, decisions, "m", None, "m", None, "test", 0.0)


def test_a_mode_does_not_see_decisions_from_other_modes_checkpoints():
    trace = _trace({16})
    response = "a" * 40
    for mode, expect in [("chunk_8", {16}), ("chunk_16", {16}), ("chunk_32", set())]:
        restricted = restrict_to_checkpoints(trace, response, mode)
        flagged = {item.token_index for item in restricted.decisions if item.risk_label == "unsafe"}
        assert flagged == expect


def test_replay_blocks_only_in_modes_that_check_the_flagged_prefix():
    trace = _trace({16})
    results = simulate_all(
        trace=trace,
        response="a" * 40,
        response_ground_truth="unsafe",
        unsafe_start_token=30,
        response_token_count=40,
        modes=["chunk_8", "chunk_16", "chunk_32", "full_buffered"],
        policies=["conservative"],
        checkpoint_scoring=True,
    )
    blocked = {item.mode: item.blocked for item in results}
    assert blocked == {
        "chunk_8": True,
        "chunk_16": True,
        "chunk_32": False,
        "full_buffered": False,
    }


def test_without_the_flag_every_mode_sees_all_decisions():
    results = simulate_all(
        trace=_trace({16}),
        response="a" * 40,
        response_ground_truth="unsafe",
        unsafe_start_token=30,
        response_token_count=40,
        modes=["chunk_32", "full_buffered"],
        policies=["conservative"],
    )
    assert all(item.blocked for item in results)


def test_token_mode_rejects_sparsely_scored_traces():
    decisions = (
        ResponseTokenDecision(1, 1, 1, "safe"),
        ResponseTokenDecision(2, 2, 2, "unsafe", confidence=0.9),
    )
    trace = GuardTrace("t", None, decisions, "m", None, "m", None, "test", 0.0)
    with pytest.raises(ValueError, match="every token"):
        restrict_to_checkpoints(trace, "ab", BufferMode.TOKEN)


def test_run_experiment_replays_each_mode_on_its_own_checkpoints(tmp_path: Path):
    frame = pd.DataFrame(
        [
            {
                "trace_id": "safe",
                "query": "q",
                "response": "a" * 40,
                "query_label": "unsafe",
                "response_label": "safe",
                "unsafe_start_character": 40,
                "unsafe_start_token": 41,
                "response_token_count": 40,
            },
            {
                "trace_id": "unsafe",
                "query": "q",
                "response": "b" * 40,
                "query_label": "unsafe",
                "response_label": "unsafe",
                "unsafe_start_character": 30,
                "unsafe_start_token": 31,
                "response_token_count": 40,
            },
        ]
    )
    frame.to_parquet(tmp_path / "dataset.parquet", index=False)
    config = {
        "dataset": {"prepared_path": "dataset.parquet", "revision": "rev"},
        "model": {"repository": "fake", "revision": "one", "tokenizer_revision": "one"},
        "experiment": {
            "output_dir": "runs",
            "modes": ["chunk_8", "chunk_32", "full_buffered"],
            "policies": {"conservative": ["unsafe"]},
            "seed": 42,
            "max_sentence_tokens": 128,
        },
        "runtime": {"resume": True, "device": None},
    }
    guard, _ = adapter({16: UNSAFE}, modes=config["experiment"]["modes"], max_sentence_tokens=128)
    run = run_or_load(config, profile="smoke2", root=tmp_path, guard=guard)
    blocked = run.intervention_results.set_index(["trace_id", "mode"])["blocked"]
    assert bool(blocked["safe", "chunk_8"]) and bool(blocked["unsafe", "chunk_8"])
    assert not blocked["unsafe", "chunk_32"] and not blocked["unsafe", "full_buffered"]


def test_warmup_skips_the_checks_inside_the_first_tokens():
    trace = _trace({8, 16})
    response = "a" * 40
    flagged = {
        warmup: {
            item.token_index
            for item in restrict_to_checkpoints(trace, response, "chunk_8", 128, warmup).decisions
            if item.risk_label == "unsafe"
        }
        for warmup in (0, 8, 16)
    }
    assert flagged == {0: {8, 16}, 8: {16}, 16: set()}


def test_warmup_requires_checkpoint_scoring():
    with pytest.raises(ValueError, match="checkpoint-scored"):
        simulate_all(
            trace=_trace(set()),
            response="a" * 40,
            response_ground_truth="safe",
            unsafe_start_token=41,
            response_token_count=40,
            modes=["chunk_8"],
            policies=["conservative"],
            warmup_tokens=8,
        )


def test_a_run_replayed_without_checkpoint_scoring_is_not_reused(tmp_path: Path):
    from streamguard_bench.pipeline import saved_run_exists

    frame = pd.DataFrame(
        [
            {
                "trace_id": "safe",
                "query": "q",
                "response": "a" * 20,
                "query_label": "unsafe",
                "response_label": "safe",
                "unsafe_start_character": 20,
                "unsafe_start_token": 21,
                "response_token_count": 20,
            },
            {
                "trace_id": "unsafe",
                "query": "q",
                "response": "b" * 20,
                "query_label": "unsafe",
                "response_label": "unsafe",
                "unsafe_start_character": 10,
                "unsafe_start_token": 11,
                "response_token_count": 20,
            },
        ]
    )
    frame.to_parquet(tmp_path / "dataset.parquet", index=False)
    config = {
        "dataset": {"prepared_path": "dataset.parquet", "revision": "rev"},
        "model": {"repository": "fake", "revision": "one", "tokenizer_revision": "one"},
        "experiment": {
            "output_dir": "runs",
            "modes": ["chunk_8", "full_buffered"],
            "policies": {"conservative": ["unsafe"]},
            "seed": 42,
            "max_sentence_tokens": 128,
            "checkpoint_scoring": True,
        },
        "runtime": {"resume": True, "device": None},
    }
    guard, _ = adapter(modes=config["experiment"]["modes"])
    run_or_load(config, profile="smoke2", root=tmp_path, guard=guard)
    assert saved_run_exists(config, profile="smoke2", root=tmp_path)

    metadata = tmp_path / "runs" / "smoke2" / "run_metadata.json"
    import json

    saved = json.loads(metadata.read_text())
    del saved["checkpoint_scoring"]
    metadata.write_text(json.dumps(saved))
    assert not saved_run_exists(config, profile="smoke2", root=tmp_path)
