from __future__ import annotations

import pandas as pd

from streamguard_bench.data.normalize_streamsafe import (
    normalize_categories,
    normalize_label,
    normalize_streamsafe_tables,
    stable_trace_id,
)


class CharacterTokenMapper:
    """Искусственный токенизатор: один Unicode-символ равен одному токену."""

    def map_interval(self, prompt, response, lower_character, upper_character):
        return lower_character, upper_character


def _tables(full_train: pd.DataFrame, partial_train: pd.DataFrame):
    full_val = pd.DataFrame(
        [
            {
                "query": "validation prompt",
                "response": "Validation response.",
                "response_mode": "benign",
                "answer": "safe",
                "violated_categories": [],
            }
        ]
    )
    partial_val = full_val.copy()
    return {
        "full_response_train": full_train,
        "partial_response_train": partial_train,
        "full_response_val": full_val,
        "partial_response_val": partial_val,
    }


def test_normalization_preserves_text_merges_prefixes_and_builds_unicode_coordinates():
    response = "Safe α. Unsafe ending."
    safe_prefix = "Safe α."
    full = pd.DataFrame(
        [
            {
                "query": "prompt",
                "response": response,
                "response_mode": "unsafe",
                "answer": "unsafe",
                "violated_categories": ["Violent", "Violent", "Unethical Acts"],
            }
        ]
    )
    partial = pd.DataFrame(
        [
            {
                "query": "prompt",
                "response": safe_prefix,
                "response_mode": "unsafe",
                "answer": "safe",
                "violated_categories": [],
            },
            {
                "query": "prompt",
                "response": safe_prefix,
                "response_mode": "unsafe",
                "answer": "safe",
                "violated_categories": [],
            },
            {
                "query": "prompt",
                "response": response,
                "response_mode": "unsafe",
                "answer": "unsafe",
                "violated_categories": ["Violent", "Unethical Acts"],
            },
        ]
    )

    traces, issues = normalize_streamsafe_tables(
        _tables(full, partial), dataset_revision="revision", token_mapper=CharacterTokenMapper()
    )
    trace = next(trace for trace in traces if trace.source_split == "train")

    assert issues.empty
    assert trace.prompt == "prompt"
    assert trace.response == response
    assert trace.trace_id == stable_trace_id("prompt", response)
    assert trace.harm_categories == ("Violent", "Unethical Acts")
    assert trace.prefix_annotations[0].source_rows == (0, 1)
    assert trace.harm_onset is not None
    assert trace.harm_onset.lower_character == len(safe_prefix)
    assert trace.harm_onset.lower_byte == len(safe_prefix.encode("utf-8"))
    assert trace.harm_onset.lower_qwen_token == len(safe_prefix)


def test_conflicting_prefix_labels_exclude_trace_without_silent_deletion():
    full = pd.DataFrame(
        [
            {
                "query": "prompt",
                "response": "One. Two.",
                "response_mode": "unsafe",
                "answer": "unsafe",
                "violated_categories": ["Violent"],
            }
        ]
    )
    partial = pd.DataFrame(
        [
            {
                "query": "prompt",
                "response": "One.",
                "response_mode": "unsafe",
                "answer": label,
                "violated_categories": [],
            }
            for label in ("safe", "unsafe")
        ]
    )
    traces, issues = normalize_streamsafe_tables(
        _tables(full, partial), dataset_revision="revision"
    )
    trace = next(trace for trace in traces if trace.source_split == "train")
    assert trace.exclusion_reason == "conflicting_prefix_labels"
    assert "conflicting_prefix_labels" in set(issues["code"])


def test_label_and_category_rules_are_explicit():
    assert normalize_label("safe") == ("safe", "safe", None)
    assert normalize_label("uncertain") == (None, "uncertain", "uncertain_label")
    assert normalize_label("new-label") == (None, "new-label", "unknown_label")
    assert normalize_label(None) == (None, "", "missing_label")
    assert normalize_categories(["A", "B", "A", ""]) == ("A", "B")


def test_stable_identifier_depends_on_unchanged_prompt_and_response():
    first = stable_trace_id("prompt", "response")
    assert first == stable_trace_id("prompt", "response")
    assert first != stable_trace_id("prompt ", "response")
