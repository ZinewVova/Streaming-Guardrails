from pathlib import Path

import pandas as pd
import pytest

from streamguard_bench.contracts import PromptDecision, ResponseTokenDecision
from streamguard_bench.pipeline import load_config, run_or_load, saved_run_exists
from streamguard_bench.streaming import TokenizedResponse


class CountingGuard:
    model_id = tokenizer_id = "fake"
    model_revision = tokenizer_revision = "one"
    device = "cpu"

    def __init__(self) -> None:
        self.scored = 0
        self.traces = 0

    def tokenize_response(self, response):
        return TokenizedResponse(tuple(map(ord, response)), tuple(range(1, len(response) + 1)))

    def start(self, prompt, *, trace_id, query_ground_truth):
        self.traces += 1
        return PromptDecision(trace_id, query_ground_truth, "unsafe", (), 0.9, 1)

    def score_token(self, *, token_id, token_index, end_character):
        self.scored += 1
        label = "unsafe" if chr(token_id) == "!" else "safe"
        return ResponseTokenDecision(token_index, token_id, end_character, label)

    def close(self) -> None:
        pass


class ForbiddenGuard:
    def __getattr__(self, name):
        raise AssertionError("the guard must not be touched when a saved run exists")


def _project(tmp_path: Path) -> dict:
    frame = pd.DataFrame(
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
    frame.to_parquet(tmp_path / "dataset.parquet", index=False)
    return {
        "dataset": {"prepared_path": "dataset.parquet", "revision": "rev"},
        "model": {"repository": "fake", "revision": "one", "tokenizer_revision": "one"},
        "experiment": {
            "output_dir": "runs",
            "modes": ["token", "chunk_8"],
            "policies": {"strict": ["controversial", "unsafe"], "conservative": ["unsafe"]},
            "seed": 42,
            "max_sentence_tokens": 128,
        },
        "runtime": {"resume": True, "device": None},
    }


def test_first_call_scores_and_second_call_reads_the_saved_run(tmp_path: Path):
    config = _project(tmp_path)
    assert not saved_run_exists(config, profile="smoke2", root=tmp_path)

    guard = CountingGuard()
    first = run_or_load(config, profile="smoke2", root=tmp_path, guard=guard)
    assert guard.traces == 2
    assert saved_run_exists(config, profile="smoke2", root=tmp_path)

    second = run_or_load(config, profile="smoke2", root=tmp_path, guard=ForbiddenGuard())
    assert len(second.intervention_results) == len(first.intervention_results)
    assert second.selected_trace_ids == first.selected_trace_ids


def test_force_rebuilds_the_tables_without_scoring_again(tmp_path: Path):
    config = _project(tmp_path)
    guard = CountingGuard()
    run_or_load(config, profile="smoke2", root=tmp_path, guard=guard)
    scored = guard.scored

    run_or_load(config, profile="smoke2", root=tmp_path, guard=guard, force=True)
    assert guard.scored == scored


def test_unknown_profile_fails_before_any_data_is_read(tmp_path: Path):
    config = _project(tmp_path)
    with pytest.raises(ValueError, match="Unknown profile"):
        run_or_load(config, profile="unknown", root=tmp_path, guard=ForbiddenGuard())


def test_repository_configuration_matches_the_pipeline_contract():
    config = load_config()
    assert set(config["experiment"]["policies"]) == {"strict", "conservative"}
    assert config["experiment"]["output_dir"] == "data/interim/qwen3guard"
    assert set(config["experiment"]["modes"]) >= {"token", "full_buffered"}
