import pytest

from streamguard_bench.guards import SCMAdapter
from streamguard_bench.guards.scm_guard import PROMPT_NOT_EVALUATED


class CharTokenizer:
    """One token per character, so offsets and decoding are trivially exact."""

    def __call__(self, text, **kwargs):
        encoded = {"input_ids": [ord(char) for char in text]}
        if kwargs.get("return_offsets_mapping"):
            encoded["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return encoded

    def decode(self, token_ids, **_):
        return "".join(chr(item) for item in token_ids)


class ScriptedScorer:
    device = "test"

    def __init__(self, scores):
        self.scores = dict(scores)
        self.prompt_ids = None
        self.closed = 0

    def reset(self, prompt_ids):
        self.prompt_ids = prompt_ids

    def step(self, token_id):
        return self.scores.get(chr(token_id), 0.0)

    def close(self):
        self.closed += 1


def adapter(scores=(), **kwargs):
    scorer = ScriptedScorer(scores)
    return SCMAdapter(scorer=scorer, tokenizer=CharTokenizer(), **kwargs), scorer


def test_prompt_is_context_only_and_uses_the_training_separator():
    guard, scorer = adapter()
    decision = guard.start("hi", trace_id="t", query_ground_truth="unsafe")
    assert "".join(map(chr, scorer.prompt_ids)) == "hi\n\n"
    assert decision.risk_label == PROMPT_NOT_EVALUATED
    assert decision.query_ground_truth == "unsafe"


def test_threshold_turns_scores_into_labels_and_keeps_the_raw_score():
    guard, _ = adapter({"a": 0.69, "b": 0.7}, threshold=0.7)
    guard.start("q")
    below = guard.score_token(token_id=ord("a"), token_index=1, end_character=1)
    at = guard.score_token(token_id=ord("b"), token_index=2, end_character=2)
    assert (below.risk_label, below.unsafe_score) == ("safe", 0.69)
    assert (at.risk_label, at.unsafe_score) == ("unsafe", 0.7)


def test_scoring_requires_an_active_stream():
    guard, scorer = adapter()
    with pytest.raises(RuntimeError, match="start"):
        guard.score_token(token_id=1, token_index=1, end_character=1)
    guard.start("q")
    guard.close()
    guard.close()
    assert scorer.closed == 1
    with pytest.raises(RuntimeError, match="start"):
        guard.score_token(token_id=1, token_index=1, end_character=1)


def test_response_tokenization_keeps_character_offsets():
    guard, _ = adapter()
    tokenized = guard.tokenize_response("abc")
    assert tokenized.end_characters == (1, 2, 3)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_degenerate_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        adapter(threshold=threshold)


def test_cached_scoring_equals_one_pass_on_a_tiny_random_model():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from streamguard_bench.guards.scm_guard import TorchTokenScorer, scm_model_class

    torch.manual_seed(0)
    config = transformers.Qwen2Config(
        vocab_size=50,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_token_labels=2,
        num_sequence_labels=2,
    )
    scorer = TorchTokenScorer(scm_model_class()(config).eval(), device="cpu")
    prompt_ids, response_ids = [3, 7, 11], [5, 9, 13, 2, 40]

    scorer.reset(prompt_ids)
    stepped = [scorer.step(token_id) for token_id in response_ids]
    one_pass = scorer.score_sequence(prompt_ids, response_ids)

    assert stepped == pytest.approx(one_pass, abs=1e-5)
    assert len(set(round(value, 6) for value in stepped)) > 1
