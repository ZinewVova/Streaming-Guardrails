from pathlib import Path

import pandas as pd

from streamguard_bench.data.inspect_schema import (
    align_streamsafe_prefixes,
    dataset_overview,
    distribution_for_role,
    duplicate_summary,
    infer_column_roles,
    length_statistics,
    prefix_transition_summary,
    schema_summary,
    streamsafe_trace_summary,
)
from streamguard_bench.data.load_streamsafe import read_data_file

FIXTURE = Path(__file__).parents[1] / "data" / "fixtures" / "synthetic_streamsafe_sample.json"


def fixture_tables():
    return {"synthetic": read_data_file(FIXTURE)}


def test_role_inference() -> None:
    roles = infer_column_roles(["prompt", "response", "safety_label", "harm_categories"])
    assert roles["prompt"] == ["prompt"]
    assert roles["response"] == ["response"]
    assert "safety_label" in roles["label"]
    assert "harm_categories" in roles["category"]


def test_schema_summary_is_column_level() -> None:
    tables = fixture_tables()
    summary = schema_summary(tables)
    assert len(summary) == len(tables["synthetic"].columns)
    assert set(summary["table"]) == {"synthetic"}
    assert dataset_overview(tables).loc[0, "rows"] == 12


def test_distributions_and_lengths_do_not_return_raw_text() -> None:
    tables = fixture_tables()
    labels = distribution_for_role(tables, "label")
    lengths = length_statistics(tables)
    duplicates = duplicate_summary(tables)
    assert {"safe", "unsafe", "needs_review"}.issubset(set(labels["value"]))
    assert set(lengths["measure"]) == {
        "characters",
        "bytes_utf8",
        "words",
        "sentences_approx",
    }
    assert not duplicates.empty


def test_prefix_transition_summary_detects_changes() -> None:
    transitions = prefix_transition_summary(fixture_tables())
    trace = transitions[transitions["trace_id"] == "trace-003"].iloc[0]
    assert trace["prefix_count"] == 3
    assert trace["label_transitions"] >= 1


def test_streamsafe_alignment_marks_ambiguous_matches() -> None:
    full = pd.DataFrame(
        {
            "query": ["q", "q"],
            "response": ["safe start unsafe end", "safe start different end"],
            "response_mode": ["unsafe", "benign"],
            "answer": ["unsafe", "safe"],
        }
    )
    partial = pd.DataFrame(
        {
            "query": ["q", "q"],
            "response": ["safe start", "safe start unsafe"],
            "response_mode": ["unknown", "unsafe"],
            "answer": ["safe", "unsafe"],
        }
    )
    alignment = align_streamsafe_prefixes(full, partial)
    assert alignment.loc[0, "status"] == "ambiguous"
    assert alignment.loc[1, "status"] == "unique"
    traces = streamsafe_trace_summary(alignment)
    assert len(traces) == 1
    assert traces.loc[0, "first_unsafe_prefix_index"] == 1
