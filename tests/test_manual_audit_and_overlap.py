from __future__ import annotations

import pandas as pd

from streamguard_bench.contracts import HarmOnset, NormalizedTrace
from streamguard_bench.data.manual_audit import validate_completed_audit
from streamguard_bench.data.overlap import analyze_split_overlap, canonical_test_response


def _trace(trace_id: str, split: str, prompt: str, response: str, label: str):
    onset = None
    if label == "unsafe":
        onset = HarmOnset(2, 5, 2, 6, 1, 2, 5, None, None, None, "fixture")
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


def test_manual_exact_position_uses_utf8_and_reports_interval_conflict():
    trace = _trace("t1", "train", "p", "aαunsafe", "unsafe")
    audit = pd.DataFrame(
        [
            {
                "trace_id": "t1",
                "reviewer": "reviewer",
                "review_date": "2026-08-24",
                "final_label_correct": True,
                "category_correct": True,
                "onset_interval_correct": False,
                "exact_unsafe_start_character": 6,
                "text_roundtrip_correct": True,
                "cross_split_issue": False,
                "suggested_label": "",
                "suggested_categories": "",
                "notes": "fixture",
            }
        ]
    )
    completed, issues = validate_completed_audit(audit, [trace])
    assert completed.loc[0, "exact_unsafe_start_byte"] == len("aαunsa".encode())
    assert "exact_onset_outside_interval" in set(issues["code"])


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


def test_public_audit_validation_never_adds_source_text_columns():
    trace = _trace("t1", "train", "private prompt", "private response", "safe")
    audit = pd.DataFrame(
        [
            {
                "trace_id": "t1",
                "reviewer": "reviewer",
                "review_date": "2026-08-24",
                "final_label_correct": True,
                "category_correct": True,
                "onset_interval_correct": True,
                "exact_unsafe_start_character": "",
                "text_roundtrip_correct": True,
                "cross_split_issue": False,
                "suggested_label": "",
                "suggested_categories": "",
                "notes": "",
            }
        ]
    )
    completed, issues = validate_completed_audit(audit, [trace])
    assert issues.empty
    assert "prompt" not in completed.columns
    assert "response" not in completed.columns
