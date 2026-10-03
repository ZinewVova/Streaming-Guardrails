import pytest

from streamguard_bench.guards import GuardOutputError, parse_guard_output


@pytest.mark.parametrize(
    "value",
    [{}, {"risk_level": None}, {"risk_level": []}, {"risk_level": ""}, {"risk_level": "mystery"}],
)
def test_invalid_risk_level_is_protocol_error(value):
    with pytest.raises(GuardOutputError):
        parse_guard_output(value)


@pytest.mark.parametrize("label", ["Safe", "Controversial", "Unsafe"])
def test_documented_labels_are_parsed(label):
    assert parse_guard_output({"risk_level": [label]})["risk_label"] == label.lower()


def test_stream_adapter_keeps_every_risk_probability():
    torch = pytest.importorskip("torch")
    from streamguard_bench.guards import Qwen3GuardStreamAdapter

    class Tokenizer:
        def apply_chat_template(self, messages, **_):
            return [1, 2]

    class Model:
        device = "cpu"
        response_risk_level_map = {0: "Safe", 1: "Controversial", 2: "Unsafe"}
        query_risk_level_map = response_risk_level_map
        response_category_map = query_category_map = {0: "None", 1: "Violent"}

        def stream_generate(self, token_ids):
            length = len(token_ids)
            while True:
                # The last position decides: token id 9 is unsafe, anything else is safe.
                last = (
                    torch.tensor([0.0, 1.0, 3.0])
                    if int(token_ids[-1]) == 9
                    else torch.tensor([3.0, 1.0, 0.0])
                )
                risk = torch.zeros(1, length, 3)
                risk[0, -1] = last
                category = torch.zeros(1, length, 2)
                category[0, -1, 1] = 1.0
                received = yield (risk, category, risk, category)
                if received is None:
                    return
                token_ids = torch.cat([token_ids, received])
                length += 1

        def close_stream(self, state):
            state.close()

    guard = Qwen3GuardStreamAdapter(model=Model(), tokenizer=Tokenizer())
    guard._torch = torch
    guard.device = "cpu"
    prompt = guard.start("q")
    decision = guard.score_token(token_id=9, token_index=1, end_character=1)
    guard.close()

    assert prompt.risk_label == "safe"
    assert decision.risk_label == "unsafe"
    assert decision.risk_categories == ("Violent",)
    total = decision.safe_score + decision.controversial_score + decision.unsafe_score
    assert total == pytest.approx(1.0)
    assert decision.unsafe_score == pytest.approx(decision.confidence)
    assert decision.unsafe_score > decision.controversial_score > decision.safe_score
