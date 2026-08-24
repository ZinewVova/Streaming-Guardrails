from __future__ import annotations

import pandas as pd

from streamguard_bench.contracts import HarmOnset, NormalizedTrace
from streamguard_bench.data.overlap import analyze_split_overlap, canonical_test_response


def _trace(trace_id: str, split: str, prompt: str, response: str, label: str):
    onset = None
    if label == "unsafe":
        onset = HarmOnset(2, 5, 2, 6, 1, 2, 5, "fixture")
    return NormalizedTrace(
        trace_id,
        split,
        prompt,
        response,
        label,
        label,
        ("Violent",) if label == "unsafe" else (),
        (),
        onset,
        "en",
        (0,),
        "fixture",
        None,
    )


def test_overlap_reports_hashes_and_identifiers_but_not_source_text():
    traces = [
        _trace("train-id", "train", "shared", "same", "safe"),
        _trace("val-id", "val", "shared", "different", "safe"),
    ]
    test = pd.DataFrame(
        [{"query": "other", "response": "<think>x</think><output>same</output>"}]
    )
    summary, details = analyze_split_overlap(traces, test, {"train-id"})
    assert int(summary["shared_values"].sum()) > 0
    assert {"value_hash", "left_trace_ids", "right_trace_ids"}.issubset(details.columns)
    assert "prompt" not in details.columns
    assert "response" not in details.columns
    assert canonical_test_response("<think>x</think><output>same</output>") == "xsame"
