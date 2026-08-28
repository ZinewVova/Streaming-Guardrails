from streamguard_bench.guards import Qwen3GuardStreamAdapter


class FakeTokenizer:
    init_kwargs = {"_commit_hash": "tokenizer-revision"}

    def apply_chat_template(self, messages, **kwargs):
        assert messages[0]["role"] == "user"
        assert kwargs == {"tokenize": True, "add_generation_prompt": False}
        return [100, 101]

    def __call__(self, text, **kwargs):
        assert kwargs["add_special_tokens"] is False
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token_id) for token_id in token_ids)


class FakeConfig:
    _commit_hash = "model-revision"


class FakeModel:
    config = FakeConfig()

    def __init__(self):
        self.calls = []
        self.closed = []

    def stream_moderate_from_ids(self, token_ids, *, role, stream_state):
        self.calls.append((list(token_ids), role, stream_state))
        state = object() if stream_state is None else stream_state
        if role == "user":
            return {"risk_level": ["Safe"]}, state
        return {
            "risk_level": ["Controversial"],
            "risk_prob": [0.73],
            "category": ["Violent"],
        }, state

    def close_stream(self, state):
        self.closed.append(state)


def _adapter():
    model = FakeModel()
    adapter = Qwen3GuardStreamAdapter(
        model=model,
        tokenizer=FakeTokenizer(),
        device="test",
    )
    return adapter, model


def test_adapter_tokenizes_once_and_preserves_offsets():
    adapter, _ = _adapter()
    tokenized = adapter.tokenize_response("Aé")

    assert tokenized.token_ids == (ord("A"), ord("é"))
    assert tokenized.end_characters == (1, 2)


def test_adapter_passes_only_the_new_token_and_preserves_three_class_output():
    adapter, model = _adapter()
    adapter.start("prompt")
    decision = adapter.score_token(token_id=65, token_index=1, end_character=1)

    assert model.calls[0][0] == [100, 101]
    assert model.calls[0][1] == "user"
    assert model.calls[1][0] == [65]
    assert model.calls[1][1] == "assistant"
    assert decision.risk_label == "controversial"
    assert decision.label_confidence == 0.73
    assert decision.risk_categories == ("Violent",)


def test_start_and_close_release_previous_stream_state():
    adapter, model = _adapter()
    adapter.start("first")
    first_state = adapter.stream_state
    adapter.start("second")

    assert model.closed == [first_state]
    adapter.close()
    assert len(model.closed) == 2
    assert adapter.stream_state is None
