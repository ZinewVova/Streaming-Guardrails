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
    assert config["experiment"]["output_dir"] == "data/interim/qwen3guard_stream_0_6b"
    assert set(config["experiment"]["modes"]) >= {"token", "full_buffered"}


def test_scm_configuration_selects_the_scm_adapter_and_harm_at_k_rule():
    config = load_config("configs/scm_0_5b.yaml")
    assert config["model"]["adapter"] == "scm"
    assert 0 < config["model"]["threshold"] < 1
    assert config["experiment"]["trigger_count"] >= 1
    assert config["experiment"]["output_dir"] != load_config()["experiment"]["output_dir"]


def test_unknown_adapter_is_rejected():
    from streamguard_bench.pipeline import build_guard

    config = {
        "model": {"adapter": "nope", "repository": "x", "revision": "r", "tokenizer_revision": "r"},
        "runtime": {"device": None},
    }
    with pytest.raises(ValueError, match="Unknown guard adapter"):
        build_guard(config)


def test_changing_the_stop_rule_replays_saved_traces_without_scoring(tmp_path: Path):
    config = _project(tmp_path)
    config["experiment"]["trigger_count"] = 2
    guard = CountingGuard()
    run = run_or_load(config, profile="smoke2", root=tmp_path, guard=guard)
    assert set(run.intervention_results["trigger_count"]) == {2}
    # One flagged token is not enough for k = 2, so nothing is blocked.
    assert not run.intervention_results["blocked"].any()
    scored = guard.scored

    config["experiment"]["trigger_count"] = 1
    # Tables saved under k = 2 must not be read as the answer for k = 1.
    assert not saved_run_exists(config, profile="smoke2", root=tmp_path)
    replayed = run_or_load(config, profile="smoke2", root=tmp_path, guard=guard)
    assert guard.scored == scored
    assert set(replayed.intervention_results["trigger_count"]) == {1}
    assert replayed.intervention_results["blocked"].any()
    assert saved_run_exists(config, profile="smoke2", root=tmp_path)


def test_another_model_revision_cannot_reuse_the_run_directory(tmp_path: Path):
    config = _project(tmp_path)
    run_or_load(config, profile="smoke2", root=tmp_path, guard=CountingGuard())
    other = CountingGuard()
    other.model_revision = "two"
    with pytest.raises(ValueError, match="different configuration"):
        run_or_load(config, profile="smoke2", root=tmp_path, guard=other, force=True)


def test_tokenization_mismatch_is_reported_as_a_trace_error(tmp_path: Path):
    config = _project(tmp_path)
    frame = pd.read_parquet(tmp_path / "dataset.parquet")
    frame.loc[frame["trace_id"] == "safe", "response_token_count"] = 99
    frame.to_parquet(tmp_path / "dataset.parquet", index=False)
    run = run_or_load(config, profile="smoke2", root=tmp_path, guard=CountingGuard())
    assert list(run.errors["trace_id"]) == ["safe"]
    assert "tokenization" in run.errors["error"].iloc[0]


def test_saved_runs_are_loaded_per_config_under_the_first_policy(tmp_path: Path):
    import yaml

    from streamguard_bench.pipeline import load_saved_runs

    config = _project(tmp_path)
    config["model"]["repository"] = "org/first-guard"
    guard = CountingGuard()
    guard.model_id = "org/first-guard"
    run_or_load(config, profile="smoke2", root=tmp_path, guard=guard)
    unfinished = {**config, "experiment": {**config["experiment"], "output_dir": "missing"}}
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "a.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (tmp_path / "configs" / "b.yaml").write_text(yaml.safe_dump(unfinished, sort_keys=False))

    runs = load_saved_runs(profile="smoke2", root=tmp_path)
    assert [item.name for item in runs] == ["first-guard"]
    assert runs[0].policy == "strict"
    assert set(runs[0].results["policy"]) == {"strict"}
    assert set(runs[0].results["model"]) == {"first-guard"}


def test_tables_are_rebuilt_from_saved_traces_without_the_model(tmp_path: Path, monkeypatch):
    from streamguard_bench import pipeline

    config = _project(tmp_path)
    run_or_load(config, profile="smoke2", root=tmp_path, guard=CountingGuard())
    monkeypatch.setattr(pipeline, "build_guard", ForbiddenGuard())
    config["experiment"]["modes"] = ["token"]

    rebuilt = run_or_load(config, profile="smoke2", root=tmp_path)
    assert set(rebuilt.intervention_results["mode"]) == {"token"}


def test_a_run_made_by_another_model_revision_is_not_reused(tmp_path: Path):
    config = _project(tmp_path)
    run_or_load(config, profile="smoke2", root=tmp_path, guard=CountingGuard())
    config["model"]["revision"] = "two"
    assert not saved_run_exists(config, profile="smoke2", root=tmp_path)


def test_the_baseline_guard_comes_first_whatever_its_file_name(tmp_path: Path):
    import yaml

    from streamguard_bench.pipeline import DEFAULT_CONFIG, load_saved_runs

    config = _project(tmp_path)
    run_or_load(config, profile="smoke2", root=tmp_path, guard=CountingGuard())
    (tmp_path / "configs").mkdir()
    for name, label in (("a.yaml", "earlier"), (DEFAULT_CONFIG.name, "baseline")):
        named = {**config, "model": {**config["model"], "display_name": label}}
        (tmp_path / "configs" / name).write_text(yaml.safe_dump(named, sort_keys=False))

    runs = load_saved_runs(profile="smoke2", root=tmp_path)
    assert [item.name for item in runs] == ["baseline", "earlier"]
