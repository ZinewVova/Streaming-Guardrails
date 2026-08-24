"""Schema and exploratory summaries that do not expose raw text."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

ROLE_PATTERNS = {
    "prompt": ("prompt", "query", "question", "instruction", "user"),
    "response": ("response", "completion", "assistant", "output"),
    "label": ("answer", "label", "safety", "safe", "harmful", "unsafe", "risk", "severity"),
    "category": ("category", "categories", "harm_type", "risk_type", "taxonomy"),
    "trace_id": ("trace_id", "conversation_id", "sample_id", "example_id", "id"),
    "prefix_index": ("prefix_index", "sentence_index", "turn_index", "step", "index"),
}


def infer_column_roles(columns: Iterable[Any]) -> dict[str, list[str]]:
    result = {role: [] for role in ROLE_PATTERNS}
    for raw_column in columns:
        column = str(raw_column)
        normalized = re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_")
        for role, patterns in ROLE_PATTERNS.items():
            if role in {"prompt", "response"} and normalized.endswith(
                ("_mode", "_level", "_label", "_type", "_category", "_categories")
            ):
                continue
            if normalized in patterns or any(pattern in normalized for pattern in patterns):
                result[role].append(column)
    return result


def schema_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for table_name, frame in tables.items():
        roles = infer_column_roles(frame.columns)
        role_lookup = {
            column: ",".join(role for role, matches in roles.items() if column in matches)
            for column in frame.columns
        }
        for column in frame.columns:
            series = frame[column]
            non_null = series.dropna()
            rows.append(
                {
                    "table": table_name,
                    "rows": len(frame),
                    "columns": len(frame.columns),
                    "column": str(column),
                    "dtype": str(series.dtype),
                    "missing_count": int(series.isna().sum()),
                    "missing_percent": round(float(series.isna().mean() * 100), 4),
                    "unique_count": _safe_nunique(non_null),
                    "nested_types": _nested_types(non_null.head(100)),
                    "possible_roles": role_lookup.get(column, ""),
                }
            )
    return pd.DataFrame(rows)


def dataset_overview(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "table": name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "memory_mb": round(float(frame.memory_usage(deep=True).sum() / 1024**2), 3),
                "column_names": " | ".join(map(str, frame.columns)),
            }
            for name, frame in tables.items()
        ]
    )


def missing_values_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summary = schema_summary(tables)
    return summary[
        ["table", "column", "dtype", "missing_count", "missing_percent", "unique_count"]
    ]


def distribution_for_role(
    tables: dict[str, pd.DataFrame], role: str, *, top_n: int = 100
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for table_name, frame in tables.items():
        candidates = infer_column_roles(frame.columns).get(role, [])
        for column in candidates:
            exploded = _explode_values(frame[column])
            counts = exploded.astype("string").fillna("<missing>").value_counts().head(top_n)
            for value, count in counts.items():
                rows.append(
                    {
                        "table": table_name,
                        "column": column,
                        "value": str(value),
                        "count": int(count),
                    }
                )
    return pd.DataFrame(rows, columns=["table", "column", "value", "count"])


def length_statistics(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for table_name, frame in tables.items():
        roles = infer_column_roles(frame.columns)
        for role in ("prompt", "response"):
            for column in roles[role]:
                strings = frame[column].dropna()
                strings = strings[strings.map(lambda value: isinstance(value, str))]
                if strings.empty:
                    continue
                measures = {
                    "characters": strings.str.len(),
                    "bytes_utf8": strings.map(lambda value: len(value.encode("utf-8"))),
                    "words": strings.str.split().str.len(),
                    "sentences_approx": strings.map(_approx_sentence_count),
                }
                for measure, values in measures.items():
                    rows.append(_describe_values(table_name, column, role, measure, values))
    return pd.DataFrame(rows)


def duplicate_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for table_name, frame in tables.items():
        roles = infer_column_roles(frame.columns)
        for role in ("prompt", "response", "trace_id"):
            for column in roles[role]:
                series = frame[column]
                hashable = series.map(_stable_cell_repr)
                rows.append(
                    {
                        "table": table_name,
                        "role": role,
                        "column": column,
                        "non_null_rows": int(series.notna().sum()),
                        "duplicate_rows": int(hashable.duplicated(keep=False).sum()),
                        "duplicate_values": int(
                            hashable[hashable.duplicated(keep=False)].nunique()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def prefix_transition_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize label transitions when trace, order, and label columns are discoverable."""

    rows: list[dict[str, Any]] = []
    for table_name, frame in tables.items():
        roles = infer_column_roles(frame.columns)
        if not roles["trace_id"] or not roles["label"]:
            continue
        trace_column = _preferred_column(
            roles["trace_id"], ("trace_id", "conversation_id", "sample_id", "example_id", "id")
        )
        label_column = _preferred_column(
            roles["label"], ("safety_label", "label", "unsafe", "harmful", "safe")
        )
        order_column = (
            _preferred_column(
                roles["prefix_index"],
                ("prefix_index", "sentence_index", "turn_index", "step", "index"),
            )
            if roles["prefix_index"]
            else None
        )
        selected_columns = [trace_column, label_column] + (
            [order_column] if order_column else []
        )
        working = frame[selected_columns].copy()
        if order_column:
            working = working.sort_values([trace_column, order_column], kind="stable")
        for trace_id, group in working.groupby(trace_column, dropna=False, sort=False):
            labels = group[label_column].astype("string").fillna("<missing>").tolist()
            transitions = sum(
                left != right for left, right in pairwise(labels)
            )
            rows.append(
                {
                    "table": table_name,
                    "trace_id": _redact_identifier(trace_id),
                    "prefix_count": len(labels),
                    "label_transitions": transitions,
                    "first_label": labels[0] if labels else None,
                    "last_label": labels[-1] if labels else None,
                }
            )
    return pd.DataFrame(rows)


def align_streamsafe_prefixes(
    full_responses: pd.DataFrame,
    partial_responses: pd.DataFrame,
) -> pd.DataFrame:
    """Align StreamSafe prefixes to full responses without guessing ambiguous matches.

    StreamSafe does not expose an explicit trace identifier. A prefix is linked only when
    exactly one full response for the same query and response mode starts with that prefix.
    The returned table contains identifiers and lengths, never source text.
    """

    required = {"query", "response"}
    for name, frame in (("full", full_responses), ("partial", partial_responses)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} table is missing required columns: {sorted(missing)}")

    candidates_by_query: dict[str, list[dict[str, Any]]] = {}
    for full_row, record in full_responses.reset_index(drop=True).iterrows():
        query = record["query"]
        response = record["response"]
        if not isinstance(query, str) or not isinstance(response, str):
            continue
        trace_key = hashlib.sha256(f"{query}\0{response}".encode()).hexdigest()[:20]
        candidates_by_query.setdefault(query, []).append(
            {
                "full_row": int(full_row),
                "full_response": response,
                "full_response_characters": len(response),
                "full_answer": record.get("answer"),
                "response_mode": record.get("response_mode"),
                "trace_key": trace_key,
            }
        )

    rows: list[dict[str, Any]] = []
    for partial_row, record in partial_responses.reset_index(drop=True).iterrows():
        query = record["query"]
        prefix = record["response"]
        query_candidates = candidates_by_query.get(query, []) if isinstance(query, str) else []
        candidates = [
            candidate
            for candidate in query_candidates
            if isinstance(prefix, str) and candidate["full_response"].startswith(prefix)
        ]
        response_mode = record.get("response_mode")
        same_mode = [
            candidate for candidate in candidates if candidate["response_mode"] == response_mode
        ]
        if same_mode:
            candidates = same_mode

        status = "unique" if len(candidates) == 1 else "ambiguous" if candidates else "unmatched"
        match = candidates[0] if len(candidates) == 1 else None
        prefix_characters = len(prefix) if isinstance(prefix, str) else None
        full_characters = match["full_response_characters"] if match else None
        rows.append(
            {
                "partial_row": int(partial_row),
                "status": status,
                "candidate_count": len(candidates),
                "trace_key": match["trace_key"] if match else None,
                "full_row": match["full_row"] if match else None,
                "prefix_characters": prefix_characters,
                "full_response_characters": full_characters,
                "prefix_fraction": (
                    prefix_characters / full_characters
                    if prefix_characters is not None and full_characters
                    else None
                ),
                "prefix_answer": record.get("answer"),
                "full_answer": match["full_answer"] if match else None,
                "response_mode": response_mode,
            }
        )
    return pd.DataFrame(rows)


def streamsafe_trace_summary(alignment: pd.DataFrame) -> pd.DataFrame:
    """Aggregate uniquely aligned prefixes into model-independent trace statistics."""

    if alignment.empty or "status" not in alignment:
        return pd.DataFrame()
    unique = alignment[alignment["status"] == "unique"].copy()
    if unique.empty:
        return pd.DataFrame()
    unique = unique.sort_values(["trace_key", "prefix_characters"], kind="stable")
    rows: list[dict[str, Any]] = []
    for trace_key, group in unique.groupby("trace_key", sort=False):
        labels = group["prefix_answer"].astype("string").fillna("<missing>").tolist()
        unsafe_mask = group["prefix_answer"].astype("string").str.lower().eq("unsafe")
        first_unsafe = group.loc[unsafe_mask].head(1)
        first_unsafe_position = None
        if unsafe_mask.any():
            first_unsafe_position = int(np.flatnonzero(unsafe_mask.to_numpy())[0] + 1)
        rows.append(
            {
                "trace_key": trace_key,
                "prefix_count": len(group),
                "label_transitions": sum(
                    left != right for left, right in pairwise(labels)
                ),
                "first_label": labels[0],
                "last_prefix_label": labels[-1],
                "full_answer": group["full_answer"].iloc[0],
                "full_response_characters": int(group["full_response_characters"].iloc[0]),
                "first_unsafe_prefix_index": first_unsafe_position,
                "first_unsafe_characters": (
                    int(first_unsafe["prefix_characters"].iloc[0])
                    if not first_unsafe.empty
                    else None
                ),
                "first_unsafe_fraction": (
                    float(first_unsafe["prefix_fraction"].iloc[0])
                    if not first_unsafe.empty
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _safe_nunique(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return int(series.map(_stable_cell_repr).nunique(dropna=True))


def _nested_types(series: pd.Series) -> str:
    return ",".join(sorted({type(value).__name__ for value in series}))


def _explode_values(series: pd.Series) -> pd.Series:
    values = series.map(
        lambda value: value if isinstance(value, list | tuple | set) else [value]
    )
    return values.explode()


def _approx_sentence_count(text: str) -> int:
    parts = [part for part in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if part]
    return max(1, len(parts)) if text else 0


def _describe_values(
    table: str, column: str, role: str, measure: str, values: pd.Series
) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "table": table,
        "column": column,
        "role": role,
        "measure": measure,
        "count": len(numeric),
        "mean": round(float(numeric.mean()), 3),
        "std": round(float(numeric.std(ddof=1)), 3) if len(numeric) > 1 else 0.0,
        "min": int(numeric.min()),
        "p25": round(float(numeric.quantile(0.25)), 3),
        "median": round(float(numeric.median()), 3),
        "p75": round(float(numeric.quantile(0.75)), 3),
        "p95": round(float(numeric.quantile(0.95)), 3),
        "max": int(numeric.max()),
    }


def _stable_cell_repr(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "<missing>"
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return str(value)


def _redact_identifier(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 24 else f"{text[:12]}…{text[-8:]}"


def _preferred_column(candidates: list[str], priorities: tuple[str, ...]) -> str:
    normalized = {candidate.lower(): candidate for candidate in candidates}
    for priority in priorities:
        if priority in normalized:
            return normalized[priority]
    return candidates[0]
