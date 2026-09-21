from pathlib import Path

import pandas as pd

from streamguard_bench.contracts import PromptDecision, ResponseTokenDecision
from streamguard_bench.experiments import run_experiment
from streamguard_bench.streaming import TokenizedResponse


class FakeGuard:
    model_id = tokenizer_id = "fake"
    model_revision = tokenizer_revision = "one"
    device = "cpu"

    def __init__(self):
        self.scored = 0

    def tokenize_response(self, response):
        return TokenizedResponse(tuple(map(ord, response)), tuple(range(1, len(response) + 1)))

    def start(self, prompt, *, trace_id, query_ground_truth):
        return PromptDecision(trace_id, query_ground_truth, "unsafe", (), 0.9, 1)

    def score_token(self, *, token_id, token_index, end_character):
        self.scored += 1
        return ResponseTokenDecision(
            token_index, token_id, end_character, "unsafe" if chr(token_id) == "!" else "safe"
        )

    def close(self):
        pass


def dataset():
    return pd.DataFrame(
        [
            {
                "trace_id": "safe",
                "query": "q",
                "response": "ok",
                "query_label": "unsafe",
                "response_label": "safe",
                "unsafe_start_character": 2,
                "unsafe_start_token": 3,
                "response_token_count": 2,
            },
            {
                "trace_id": "unsafe",
                "query": "q",
                "response": "a!",
                "query_label": "unsafe",
                "response_label": "unsafe",
                "unsafe_start_character": 1,
                "unsafe_start_token": 2,
                "response_token_count": 2,
            },
        ]
    )


def test_fake_end_to_end_saves_prompt_and_continues_after_unsafe_prompt(tmp_path: Path):
    guard = FakeGuard()
    run = run_experiment(dataset=dataset(), guard=guard, output_dir=tmp_path)
    assert len(run.prompt_decisions) == 2
    assert guard.scored == 4
    assert (run.output_dir / "prompt_decisions.parquet").exists()
    scored = guard.scored
    resumed = run_experiment(dataset=dataset(), guard=guard, output_dir=tmp_path)
    assert guard.scored == scored
    assert len(resumed.intervention_results) == 24


def test_old_checkpoint_schema_is_rejected():
    from streamguard_bench.contracts import GuardTrace

    try:
        GuardTrace.from_dict({"schema_version": 1})
    except ValueError as error:
        assert "obsolete" in str(error)
    else:
        raise AssertionError("old checkpoint must be rejected")
