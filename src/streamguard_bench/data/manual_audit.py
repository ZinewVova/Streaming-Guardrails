"""Deterministic manual-review sampling and validation without public source text."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

import pandas as pd

from streamguard_bench.contracts import NormalizedTrace
from streamguard_bench.data.normalize_streamsafe import TokenCoordinateMapper

AUDIT_COLUMNS = (
    "trace_id",
    "reviewer",
    "review_date",
    "final_label_correct",
    "category_correct",
    "onset_interval_correct",
    "exact_unsafe_start_character",
    "text_roundtrip_correct",
    "cross_split_issue",
    "suggested_label",
    "suggested_categories",
    "notes",
)


def build_manual_audit_sample(
    subset: list[NormalizedTrace], *, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a public blank form and a private text-bearing review table."""

    safe = [trace for trace in subset if trace.binary_label == "safe"]
    unsafe = [trace for trace in subset if trace.binary_label == "unsafe"]
    safe_selected = _select_safe(safe, seed)
    unsafe_selected = _select_unsafe(unsafe, seed)
    selected = sorted(safe_selected + unsafe_selected, key=lambda trace: trace.trace_id)
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for trace in selected:
        onset = trace.harm_onset
        public_rows.append(
            {
                "trace_id": trace.trace_id,
                "source_split": trace.source_split,
                "binary_label": trace.binary_label,
                "harm_categories": "|".join(trace.harm_categories),
                "onset_bucket": _onset_bucket(trace),
                "onset_lower_character": onset.lower_character if onset else None,
                "onset_upper_character": onset.upper_character if onset else None,
                **{column: "" for column in AUDIT_COLUMNS if column != "trace_id"},
            }
        )
        private_rows.append(
            {
                "trace_id": trace.trace_id,
                "prompt": trace.prompt,
                "response": trace.response,
                "review_segment": (
                    trace.response[onset.lower_character : onset.upper_character]
                    if onset
                    else ""
                ),
            }
        )
    return pd.DataFrame(public_rows), pd.DataFrame(private_rows)


def validate_completed_audit(
    audit: pd.DataFrame,
    subset: list[NormalizedTrace],
    *,
    token_mapper: TokenCoordinateMapper | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate human fields and derive exact byte/token coordinates when completed."""

    traces = {trace.trace_id: trace for trace in subset}
    result = audit.copy()
    issues: list[dict[str, Any]] = []
    result["exact_unsafe_start_byte"] = pd.NA
    result["exact_unsafe_start_qwen_token"] = pd.NA
    result["exact_inside_automatic_interval"] = pd.NA
    for index, row in result.iterrows():
        trace_id = row.get("trace_id")
        trace = traces.get(trace_id)
        if trace is None:
            issues.append({"trace_id": trace_id, "code": "unknown_trace_id", "details": ""})
            continue
        for field in (
            "reviewer",
            "review_date",
            "final_label_correct",
            "category_correct",
            "onset_interval_correct",
            "text_roundtrip_correct",
            "cross_split_issue",
        ):
            if _blank(row.get(field)):
                issues.append(
                    {"trace_id": trace_id, "code": "missing_manual_field", "details": field}
                )
        if trace.binary_label != "unsafe":
            continue
        exact = row.get("exact_unsafe_start_character")
        if _blank(exact):
            issues.append(
                {"trace_id": trace_id, "code": "missing_exact_onset", "details": ""}
            )
            continue
        try:
            exact_character = int(exact)
        except (TypeError, ValueError):
            issues.append(
                {"trace_id": trace_id, "code": "invalid_exact_onset", "details": str(exact)}
            )
            continue
        if not 0 <= exact_character < len(trace.response):
            issues.append(
                {
                    "trace_id": trace_id,
                    "code": "exact_onset_out_of_response",
                    "details": str(exact_character),
                }
            )
            continue
        result.at[index, "exact_unsafe_start_byte"] = len(
            trace.response[:exact_character].encode("utf-8")
        )
        if trace.harm_onset:
            inside = (
                trace.harm_onset.lower_character
                <= exact_character
                < trace.harm_onset.upper_character
            )
            result.at[index, "exact_inside_automatic_interval"] = inside
            if not inside:
                issues.append(
                    {
                        "trace_id": trace_id,
                        "code": "exact_onset_outside_interval",
                        "details": str(exact_character),
                    }
                )
        if token_mapper is not None:
            token, _ = token_mapper.map_interval(
                trace.prompt, trace.response, exact_character, exact_character + 1
            )
            result.at[index, "exact_unsafe_start_qwen_token"] = token
    return result, pd.DataFrame(issues, columns=["trace_id", "code", "details"])


def audit_summary(audit: pd.DataFrame, issues: pd.DataFrame) -> pd.DataFrame:
    completed = audit["reviewer"].astype("string").str.strip().ne("").sum()
    return pd.DataFrame(
        [
            {"metric": "selected_records", "value": len(audit)},
            {"metric": "completed_records", "value": int(completed)},
            {"metric": "safe_records", "value": int((audit["binary_label"] == "safe").sum())},
            {
                "metric": "unsafe_records",
                "value": int((audit["binary_label"] == "unsafe").sum()),
            },
            {"metric": "validation_issues", "value": len(issues)},
        ]
    )


def _select_safe(traces: list[NormalizedTrace], seed: int) -> list[NormalizedTrace]:
    lengths = pd.Series([len(trace.response) for trace in traces])
    q33, q66 = lengths.quantile([1 / 3, 2 / 3]).tolist()
    buckets = {
        "short": [trace for trace in traces if len(trace.response) <= q33],
        "medium": [trace for trace in traces if q33 < len(trace.response) <= q66],
        "long": [trace for trace in traces if len(trace.response) > q66],
    }
    targets = {"short": 17, "medium": 17, "long": 16}
    result: list[NormalizedTrace] = []
    for offset, bucket in enumerate(("short", "medium", "long")):
        result.extend(_sample(buckets[bucket], targets[bucket], seed + offset))
    return result


def _select_unsafe(traces: list[NormalizedTrace], seed: int) -> list[NormalizedTrace]:
    targets = {"early": 15, "middle": 15, "late": 20}
    category_counts = Counter(category for trace in traces for category in trace.harm_categories)
    result: list[NormalizedTrace] = []
    for offset, bucket in enumerate(("early", "middle", "late")):
        candidates = [trace for trace in traces if _onset_bucket(trace) == bucket]

        def key(trace: NormalizedTrace) -> tuple[float, str]:
            rarity = sum(
                1 / category_counts[category]
                for category in trace.harm_categories
                if category_counts[category]
            )
            return -rarity, trace.trace_id

        ordered = sorted(candidates, key=key)
        random.Random(seed + offset).shuffle(ordered)
        ordered.sort(key=key)
        if len(ordered) < targets[bucket]:
            raise ValueError(f"Not enough {bucket} unsafe traces for manual audit")
        result.extend(ordered[: targets[bucket]])
    return result


def _sample(traces: list[NormalizedTrace], target: int, seed: int) -> list[NormalizedTrace]:
    ordered = sorted(traces, key=lambda trace: trace.trace_id)
    random.Random(seed).shuffle(ordered)
    if len(ordered) < target:
        raise ValueError(f"Need {target} audit candidates, found {len(ordered)}")
    return ordered[:target]


def _onset_bucket(trace: NormalizedTrace) -> str | None:
    if not trace.harm_onset or not trace.response:
        return None
    fraction = trace.harm_onset.upper_character / len(trace.response)
    return "early" if fraction <= 0.33 else "middle" if fraction <= 0.66 else "late"


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or not str(value).strip()
