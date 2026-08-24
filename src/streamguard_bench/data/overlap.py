"""Exact cross-split overlap checks that never expose source text in reports."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from itertools import combinations
from typing import Any

import pandas as pd

from streamguard_bench.contracts import NormalizedTrace
from streamguard_bench.data.normalize_streamsafe import stable_trace_id


def analyze_split_overlap(
    traces: list[NormalizedTrace], test_frame: pd.DataFrame, selected_ids: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare train, validation, and test using hashes of exact and comparison-only text."""

    records = [_trace_record(trace, selected_ids) for trace in traces]
    for source_row, row in test_frame.reset_index(drop=True).iterrows():
        prompt = row.get("query")
        response = row.get("response")
        if not isinstance(prompt, str) or not isinstance(response, str):
            continue
        records.append(
            {
                "split": "test",
                "trace_id": stable_trace_id(prompt, response),
                "selected": False,
                "prompt_hash": _hash(prompt),
                "response_hash": _hash(response),
                "pair_hash": _hash(f"{prompt}\0{response}"),
                "canonical_response_hash": _hash(canonical_test_response(response)),
                "source_row": int(source_row),
            }
        )
    frame = pd.DataFrame(records)
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for left, right in combinations(("train", "val", "test"), 2):
        left_frame = frame[frame["split"] == left]
        right_frame = frame[frame["split"] == right]
        for field in ("prompt_hash", "response_hash", "pair_hash", "canonical_response_hash"):
            left_values = set(left_frame[field])
            right_values = set(right_frame[field])
            shared = left_values & right_values
            summary_rows.append(
                {
                    "left_split": left,
                    "right_split": right,
                    "comparison": field,
                    "shared_values": len(shared),
                    "left_rows": int(left_frame[field].isin(shared).sum()),
                    "right_rows": int(right_frame[field].isin(shared).sum()),
                }
            )
            for value in sorted(shared):
                left_matches = left_frame[left_frame[field] == value]
                right_matches = right_frame[right_frame[field] == value]
                detail_rows.append(
                    {
                        "left_split": left,
                        "right_split": right,
                        "comparison": field,
                        "value_hash": value,
                        "left_trace_ids": "|".join(sorted(left_matches["trace_id"])),
                        "right_trace_ids": "|".join(sorted(right_matches["trace_id"])),
                        "selected_trace_ids": "|".join(
                            sorted(
                                set(left_matches.loc[left_matches["selected"], "trace_id"])
                                | set(right_matches.loc[right_matches["selected"], "trace_id"])
                            )
                        ),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def canonical_test_response(text: str) -> str:
    """Remove only documented test wrappers for comparison; never mutate stored text."""

    without_tags = re.sub(r"</?(?:think|output)>", "", text, flags=re.IGNORECASE)
    return without_tags.strip()


def _trace_record(trace: NormalizedTrace, selected_ids: set[str]) -> dict[str, Any]:
    return {
        "split": trace.source_split,
        "trace_id": trace.trace_id,
        "selected": trace.trace_id in selected_ids,
        "prompt_hash": _hash(trace.prompt),
        "response_hash": _hash(trace.response),
        "pair_hash": _hash(f"{trace.prompt}\0{trace.response}"),
        "canonical_response_hash": _hash(canonical_test_response(trace.response)),
        "source_row": trace.source_rows[0] if trace.source_rows else None,
    }


def overlap_ids_by_trace(details: pd.DataFrame) -> dict[str, set[str]]:
    """Index overlap comparison names by trace identifier."""

    result: dict[str, set[str]] = defaultdict(set)
    for row in details.itertuples():
        for field in (row.left_trace_ids, row.right_trace_ids):
            for trace_id in str(field).split("|"):
                if trace_id:
                    result[trace_id].add(row.comparison)
    return result


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
