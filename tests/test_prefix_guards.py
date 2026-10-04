import json

import pytest
from test_experiment_runner_new import dataset
from test_scm_guard import CharTokenizer

from streamguard_bench.contracts import GuardTrace
from streamguard_bench.experiments import run_experiment, sweep_decision_rules
from streamguard_bench.guards import SentGuardAdapter
from streamguard_bench.guards.sent_guard import parse_sentguard_output
from streamguard_bench.pipeline import build_guard, load_config


class Generator:
    device = "test"

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, instruction, **kwargs):
        self.calls.append((instruction, kwargs))
        return next(self.outputs)


def make_guard(cls=SentGuardAdapter, outputs=("<answer>safe</answer>",)):
    generator = Generator(outputs)
    return cls(generator=generator, tokenizer=CharTokenizer()), generator


def score(guard, response):
    tokens = guard.tokenize_response(response)
    guard.start("query")
    return [
        guard.score_token(token_id=t, token_index=i, end_character=e)
        for i, (t, e) in enumerate(zip(tokens.token_ids, tokens.end_characters, strict=True), 1)
    ]


@pytest.mark.parametrize(
    ("raw", "label"),
    [
        ("<answer>safe</answer>", "safe"),
        ("<evidence>unsafe</evidence><answer>safe</answer>", "safe"),
        ("<answer> uncertain </answer>", "controversial"),
        ("<answer>UNSAFE</answer>", "unsafe"),
        ("unsafe\n<evidence>safe</evidence>", "unsafe"),
    ],
)
def test_sentguard_parser(raw, label):
    assert parse_sentguard_output(raw) == label


@pytest.mark.parametrize("raw", ["", "invalid", "safe or unsafe", "<evidence>refusal</evidence>"])
def test_bad_sentguard_output_is_not_safe(raw):
    with pytest.raises(ValueError):
        parse_sentguard_output(raw)


def test_sentence_checks_have_no_future_text_and_flush_final_tail():
    guard, generator = make_guard(outputs=["<answer>safe</answer>"] * 3)
    decisions = score(guard, "Hi. Next! tail")
    assert [d.token_index for d in decisions if d.evaluated] == [3, 9, 14]
    for (instruction, kwargs), prefix in zip(
        generator.calls,
        ["Hi.", "Hi. Next!", "Hi. Next! tail"],
        strict=True,
    ):
        assert f"Agent: {prefix}\n<END CONVERSATION>" in instruction
        assert kwargs == {"chat": True, "max_new_tokens": 512}
    assert all(d.raw_output is None and d.latency_ms == 0 for d in decisions if not d.evaluated)
    assert all(d.unsafe_score is None for d in decisions)


def test_chinese_boundaries_and_empty_response():
    guard, generator = make_guard(outputs=["<answer>safe</answer>"] * 2)
    assert [d.token_index for d in score(guard, "好。好！") if d.evaluated] == [2, 4]
    guard.close()
    assert score(guard, "") == []
    assert len(generator.calls) == 2


def test_lifecycle_and_token_order():
    guard, _ = make_guard()
    with pytest.raises(RuntimeError):
        guard.score_token(token_id=1, token_index=1, end_character=1)
    guard.tokenize_response("a")
    guard.start("q")
    with pytest.raises(ValueError, match="order"):
        guard.score_token(token_id=ord("a"), token_index=2, end_character=1)
    guard.close()
    guard.close()
    with pytest.raises(RuntimeError):
        guard.score_token(token_id=ord("a"), token_index=1, end_character=1)


def test_real_adapter_pipeline_save_resume_replay_and_metrics(tmp_path):
    guard, generator = make_guard(outputs=["<answer>safe</answer>", "<answer>uncertain</answer>"])
    run = run_experiment(dataset=dataset(), guard=guard, output_dir=tmp_path)
    assert run.errors.empty
    assert len(run.token_decisions) == 4
    assert run.token_decisions["evaluated"].sum() == 2
    strict = run.intervention_results.query("policy == 'strict' and mode == 'token'")
    assert strict.set_index("trace_id")["blocked"].to_dict() == {"safe": False, "unsafe": True}
    checkpoint = run.output_dir / "checkpoints/unsafe.json"
    trace = GuardTrace.from_dict(json.loads(checkpoint.read_text()))
    assert trace.decisions[-1].raw_output
    legacy = trace.to_dict()
    for decision in legacy["decisions"]:
        decision["response_refusal"] = None
    assert GuardTrace.from_dict(legacy) == trace
    calls = len(generator.calls)
    run_experiment(dataset=dataset(), guard=guard, output_dir=tmp_path)
    assert len(generator.calls) == calls
    grid = sweep_decision_rules(
        dataset=dataset(),
        output_dir=tmp_path,
        profile="smoke2",
        policy="strict",
        rules=[{}, {"threshold": 0.7}],
    )
    assert len(grid) == 1
    assert grid.iloc[0]["fn"] == 0


def test_malformed_output_is_saved_as_error_not_safe(tmp_path):
    guard, _ = make_guard(outputs=["bad", "<answer>safe</answer>"])
    run = run_experiment(dataset=dataset(), guard=guard, output_dir=tmp_path)
    assert list(run.errors["trace_id"]) == ["safe"]
    assert not (run.output_dir / "checkpoints/safe.json").exists()


def test_config_and_factory(monkeypatch):
    name, cls = "sentguard", SentGuardAdapter
    import streamguard_bench.guards as guards

    captured = {}

    def factory(model, **kwargs):
        captured.update(model=model, **kwargs)
        return name

    monkeypatch.setattr(guards, cls.__name__, factory)
    config = load_config("configs/sentguard_4b.yaml")
    assert build_guard(config) == name
    assert len(captured["revision"]) == 40
    assert config["experiment"]["trigger_count"] == 1


def test_hf_runtime_loads_and_generates_from_local_tiny_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from streamguard_bench.guards.prefix_guard import HFGenerator

    backend = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "hello": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        eos_token="[EOS]",
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    tokenizer.save_pretrained(tmp_path)
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=3,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        eos_token_id=1,
    )
    transformers.Qwen3ForCausalLM(config).save_pretrained(tmp_path)
    runtime = HFGenerator(str(tmp_path), revision=None, tokenizer_revision=None, device="cpu")
    for chat in (False, True):
        assert isinstance(runtime.generate("hello", chat=chat, max_new_tokens=2), str)
    with pytest.raises(ValueError, match="context limit"):
        runtime.generate("hello " * 32, chat=False, max_new_tokens=2)


def test_nf4_has_distinct_scoring_identity_and_rejects_unknown_format(tmp_path):
    generator = Generator(["<answer>safe</answer>", "<answer>unsafe</answer>"])
    guard = SentGuardAdapter(generator=generator, tokenizer=CharTokenizer(), quantization="nf4")
    assert guard.inference_settings["quantization"] == "nf4"
    run_experiment(dataset=dataset(), guard=guard, output_dir=tmp_path)
    original, _ = make_guard()
    with pytest.raises(ValueError, match="different configuration"):
        run_experiment(dataset=dataset(), guard=original, output_dir=tmp_path)
    with pytest.raises(ValueError, match="quantization"):
        SentGuardAdapter(generator=generator, tokenizer=CharTokenizer(), quantization="unknown")


def test_nf4_config_keeps_original_checkpoint_and_separate_results():
    original = load_config("configs/sentguard_4b.yaml")
    quantized = load_config("configs/sentguard_4b_nf4.yaml")
    assert quantized["model"]["revision"] == original["model"]["revision"]
    assert quantized["model"]["quantization"] == "nf4"
    assert quantized["experiment"]["output_dir"] != original["experiment"]["output_dir"]
