from pathlib import Path

import pandas as pd

from streamguard_bench.experiments import run_experiment, select_profile_traces
from streamguard_bench.streaming import TokenDecision, TokenizedResponse


class FakeGuard:
    model_id = "fake-guard"
    model_revision = "model-one"
    tokenizer_id = "fake-tokenizer"
    tokenizer_revision = "tokenizer-one"
    device = "cpu"

    def __init__(self):
        self.scored_tokens = 0
        self.close_calls = 0

    def tokenize_response(self, response):
        return TokenizedResponse(
            token_ids=tuple(ord(character) for character in response),
            end_characters=tuple(range(1, len(response) + 1)),
        )

    def start(self, prompt):
        self.prompt = prompt

    def score_token(self, *, token_id, token_index, end_character):
        self.scored_tokens += 1
        character = chr(token_id)
        label = "unsafe" if character == "!" else "controversial" if character == "?" else "safe"
        return TokenDecision(
            token_index=token_index,
            token_id=token_id,
            end_character=end_character,
            risk_label=label,
            latency_ms=1.0,
        )

    def close(self):
        self.close_calls += 1


def _dataset():
    rows = []
    for index in range(5):
        rows.append(
            {
                "prompt": f"safe prompt {index}",
                "response": "safe text.",
                "label": "safe",
                "harm_categories": [],
                "trace_id": f"safe-{index}",
                "source_split": "train",
                "last_safe_end_qwen_token": None,
                "first_unsafe_end_qwen_token": None,
                "first_unsafe_end_character": None,
            }
        )
    onset_positions = [2, 3, 5, 8, 9]
    for index, onset in enumerate(onset_positions):
        response = "a" * (onset - 1) + "!" + "a" * (9 - onset) + "."
        rows.append(
            {
                "prompt": f"unsafe prompt {index}",
                "response": response,
                "label": "unsafe",
                "harm_categories": [f"category-{index}"],
                "trace_id": f"unsafe-{index}",
                "source_split": "val",
                "last_safe_end_qwen_token": onset - 1,
                "first_unsafe_end_qwen_token": onset,
                "first_unsafe_end_character": onset,
            }
        )
    return pd.DataFrame(rows)


def test_smoke_selection_is_balanced_and_deterministic():
    first = select_profile_traces(_dataset(), "smoke")
    second = select_profile_traces(_dataset(), "smoke")

    assert first["trace_id"].tolist() == second["trace_id"].tolist()
    assert first["label"].value_counts().to_dict() == {"safe": 5, "unsafe": 5}


def test_runner_produces_120_rows_and_resumes_without_rescoring(tmp_path: Path):
    guard = FakeGuard()
    run = run_experiment(
        dataset=_dataset(),
        guard=guard,
        profile="smoke",
        output_dir=tmp_path,
    )
    scored_once = guard.scored_tokens

    assert len(run.results) == 120
    assert run.results["error"].isna().all()
    assert run.results[["trace_id", "mode", "policy"]].duplicated().sum() == 0
    assert "prompt" not in run.results.columns
    assert "response" not in run.results.columns
    assert (run.output_dir / "results.csv").exists()
    assert (run.output_dir / "token_decisions.parquet").exists()

    resumed = run_experiment(
        dataset=_dataset(),
        guard=guard,
        profile="smoke",
        output_dir=tmp_path,
        resume=True,
    )
    assert len(resumed.results) == 120
    assert guard.scored_tokens == scored_once
