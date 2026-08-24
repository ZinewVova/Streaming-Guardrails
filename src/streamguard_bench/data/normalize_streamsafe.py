"""Normalize StreamSafe full responses and sentence prefixes without changing source text."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from streamguard_bench.contracts import HarmOnset, NormalizedTrace, PrefixAnnotation

KNOWN_LABELS = {"safe", "unsafe", "uncertain"}
KNOWN_CATEGORIES = (
    "Non-violent Illegal Acts",
    "Unethical Acts",
    "Violent",
    "Politically Sensitive Topics",
    "Personally Identifiable Information",
    "Copyright Violation",
    "Sexual Content or Sexual Acts",
    "Suicide & Self-Harm",
)


class TokenCoordinateMapper(Protocol):
    """Map response character intervals to response-only token coordinates."""

    def map_interval(
        self, prompt: str, response: str, lower_character: int, upper_character: int
    ) -> tuple[int, int]: ...


def stable_trace_id(prompt: str, response: str) -> str:
    """Return a deterministic identifier derived from unchanged source text."""

    return hashlib.sha256(f"{prompt}\0{response}".encode()).hexdigest()[:20]


def normalize_label(value: Any) -> tuple[str | None, str, str | None]:
    """Map a source label to the binary benchmark label and an exclusion reason."""

    original = "" if value is None else str(value).strip().lower()
    if original in {"safe", "unsafe"}:
        return original, original, None
    if original == "uncertain":
        return None, original, "uncertain_label"
    if not original:
        return None, original, "missing_label"
    return None, original, "unknown_label"


def normalize_categories(value: Any) -> tuple[str, ...]:
    """Preserve category names and their first-seen order while removing duplicates."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ()
    values = value if isinstance(value, list | tuple | set) else [value]
    result: list[str] = []
    for item in values:
        category = str(item).strip()
        if category and category not in result:
            result.append(category)
    return tuple(result)


def normalize_streamsafe_tables(
    tables: dict[str, pd.DataFrame],
    *,
    dataset_revision: str,
    token_mapper: TokenCoordinateMapper | None = None,
) -> tuple[list[NormalizedTrace], pd.DataFrame]:
    """Build trace-level records and a complete non-silent exclusion log."""

    traces: list[NormalizedTrace] = []
    issues: list[dict[str, Any]] = []
    for split in ("train", "val"):
        full_name = _find_table(tables, "full_response", split)
        partial_name = _find_table(tables, "partial_response", split)
        split_traces, split_issues = _normalize_split(
            tables[full_name],
            tables[partial_name],
            split=split,
            dataset_revision=dataset_revision,
            token_mapper=token_mapper,
        )
        traces.extend(split_traces)
        issues.extend(split_issues)
    return traces, pd.DataFrame(
        issues,
        columns=["split", "table_kind", "source_row", "trace_id", "code", "details"],
    )


def _normalize_split(
    full: pd.DataFrame,
    partial: pd.DataFrame,
    *,
    split: str,
    dataset_revision: str,
    token_mapper: TokenCoordinateMapper | None,
) -> tuple[list[NormalizedTrace], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    full_candidates: dict[str, list[dict[str, Any]]] = {}
    for full_row, record in full.reset_index(drop=True).iterrows():
        prompt = record.get("query")
        response = record.get("response")
        if not isinstance(prompt, str) or not isinstance(response, str) or not response:
            issues.append(
                _issue(split, "full", int(full_row), None, "invalid_full_text", "")
            )
            continue
        trace_id = stable_trace_id(prompt, response)
        full_candidates.setdefault(prompt, []).append(
            {
                "trace_id": trace_id,
                "full_row": int(full_row),
                "prompt": prompt,
                "response": response,
                "response_mode": record.get("response_mode"),
                "original_label": record.get("answer"),
                "categories": normalize_categories(record.get("violated_categories")),
            }
        )

    matched: dict[str, list[dict[str, Any]]] = {}
    matched_full: dict[str, dict[str, Any]] = {}
    for partial_row, record in partial.reset_index(drop=True).iterrows():
        prompt = record.get("query")
        prefix = record.get("response")
        if not isinstance(prompt, str) or not isinstance(prefix, str):
            issues.append(
                _issue(split, "partial", int(partial_row), None, "invalid_prefix_text", "")
            )
            continue
        candidates = [
            candidate
            for candidate in full_candidates.get(prompt, [])
            if candidate["response"].startswith(prefix)
        ]
        mode = record.get("response_mode")
        if _has_value(mode):
            candidates = [
                candidate
                for candidate in candidates
                if candidate["response_mode"] == mode
            ]
        if len(candidates) != 1:
            code = "ambiguous_prefix" if candidates else "unmatched_prefix"
            issues.append(
                _issue(
                    split,
                    "partial",
                    int(partial_row),
                    None,
                    code,
                    f"candidate_count={len(candidates)}",
                )
            )
            continue
        candidate = candidates[0]
        trace_id = candidate["trace_id"]
        matched_full[trace_id] = candidate
        matched.setdefault(trace_id, []).append(
            {
                "source_row": int(partial_row),
                "prefix": prefix,
                "original_label": record.get("answer"),
                "categories": normalize_categories(record.get("violated_categories")),
            }
        )

    traces: list[NormalizedTrace] = []
    for trace_id in sorted(matched):
        candidate = matched_full[trace_id]
        trace, trace_issues = _build_trace(
            candidate,
            matched[trace_id],
            split=split,
            dataset_revision=dataset_revision,
            token_mapper=token_mapper,
        )
        traces.append(trace)
        issues.extend(trace_issues)
    return traces, issues


def _build_trace(
    full: dict[str, Any],
    prefixes: list[dict[str, Any]],
    *,
    split: str,
    dataset_revision: str,
    token_mapper: TokenCoordinateMapper | None,
) -> tuple[NormalizedTrace, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    trace_id = full["trace_id"]
    response = full["response"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for prefix in prefixes:
        grouped.setdefault(prefix["prefix"], []).append(prefix)

    annotations: list[PrefixAnnotation] = []
    conflicting = False
    for prefix_text, records in sorted(grouped.items(), key=lambda item: (len(item[0]), item[0])):
        signatures = {
            (str(record["original_label"]).strip().lower(), record["categories"])
            for record in records
        }
        if len(signatures) != 1:
            conflicting = True
            issues.append(
                _issue(
                    split,
                    "trace",
                    min(record["source_row"] for record in records),
                    trace_id,
                    "conflicting_prefix_labels",
                    f"end_character={len(prefix_text)}",
                )
            )
            continue
        binary, original, _ = normalize_label(records[0]["original_label"])
        annotations.append(
            PrefixAnnotation(
                prefix_index=0,
                end_character=len(prefix_text),
                end_byte=len(prefix_text.encode("utf-8")),
                end_sentence=_sentence_count(prefix_text),
                binary_label=binary,
                original_label=original,
                harm_categories=records[0]["categories"],
                source_rows=tuple(sorted(record["source_row"] for record in records)),
            )
        )

    final_binary, final_original, exclusion_reason = normalize_label(full["original_label"])
    if conflicting:
        exclusion_reason = "conflicting_prefix_labels"
    if not annotations or annotations[-1].end_character != len(response):
        annotations.append(
            PrefixAnnotation(
                prefix_index=0,
                end_character=len(response),
                end_byte=len(response.encode("utf-8")),
                end_sentence=_sentence_count(response),
                binary_label=final_binary,
                original_label=final_original,
                harm_categories=full["categories"],
                source_rows=(full["full_row"],),
            )
        )
    annotations = [
        replace(annotation, prefix_index=index)
        for index, annotation in enumerate(annotations)
    ]
    onset = _derive_harm_onset(
        full["prompt"], response, tuple(annotations), final_binary, token_mapper
    )
    if final_binary == "unsafe" and onset is None and exclusion_reason is None:
        exclusion_reason = "unsafe_without_unsafe_prefix"
    if final_binary == "unsafe" and not full["categories"]:
        issues.append(
            _issue(split, "trace", full["full_row"], trace_id, "unsafe_without_category", "")
        )
    for category in full["categories"]:
        if category not in KNOWN_CATEGORIES:
            issues.append(
                _issue(split, "trace", full["full_row"], trace_id, "unknown_category", category)
            )
    source_rows = (full["full_row"],) + tuple(
        sorted({row for annotation in annotations for row in annotation.source_rows})
    )
    return (
        NormalizedTrace(
            trace_id=trace_id,
            source_split=split,
            prompt=full["prompt"],
            response=response,
            binary_label=final_binary,
            original_label=final_original,
            harm_categories=full["categories"],
            prefix_annotations=tuple(annotations),
            harm_onset=onset,
            language="en",
            source_rows=tuple(dict.fromkeys(source_rows)),
            dataset_revision=dataset_revision,
            exclusion_reason=exclusion_reason,
        ),
        issues,
    )


def _derive_harm_onset(
    prompt: str,
    response: str,
    annotations: tuple[PrefixAnnotation, ...],
    final_label: str | None,
    token_mapper: TokenCoordinateMapper | None,
) -> HarmOnset | None:
    if final_label != "unsafe":
        return None
    for index, annotation in enumerate(annotations):
        if annotation.binary_label != "unsafe":
            continue
        lower = annotations[index - 1].end_character if index else 0
        upper = annotation.end_character
        lower_token = upper_token = None
        if token_mapper is not None:
            lower_token, upper_token = token_mapper.map_interval(prompt, response, lower, upper)
        return HarmOnset(
            lower_character=lower,
            upper_character=upper,
            lower_byte=len(response[:lower].encode("utf-8")),
            upper_byte=len(response[:upper].encode("utf-8")),
            sentence_index=annotation.end_sentence,
            lower_qwen_token=lower_token,
            upper_qwen_token=upper_token,
            exact_character=None,
            exact_byte=None,
            exact_qwen_token=None,
            source="streamsafe_sentence_interval",
        )
    return None


def traces_to_frame(traces: list[NormalizedTrace]) -> pd.DataFrame:
    """Serialize traces to a Parquet-friendly table with explicit JSON nested fields."""

    rows: list[dict[str, Any]] = []
    for trace in traces:
        rows.append(
            {
                "trace_id": trace.trace_id,
                "source_split": trace.source_split,
                "prompt": trace.prompt,
                "response": trace.response,
                "binary_label": trace.binary_label,
                "original_label": trace.original_label,
                "harm_categories_json": json.dumps(
                    trace.harm_categories, ensure_ascii=False
                ),
                "prefix_annotations_json": json.dumps(
                    [asdict(annotation) for annotation in trace.prefix_annotations],
                    ensure_ascii=False,
                ),
                "harm_onset_json": (
                    json.dumps(asdict(trace.harm_onset), ensure_ascii=False)
                    if trace.harm_onset
                    else None
                ),
                "language": trace.language,
                "source_rows_json": json.dumps(trace.source_rows),
                "dataset_revision": trace.dataset_revision,
                "exclusion_reason": trace.exclusion_reason,
                "response_characters": len(trace.response),
                "response_bytes": len(trace.response.encode("utf-8")),
                "prefix_count": len(trace.prefix_annotations),
            }
        )
    return pd.DataFrame(rows)


def frame_to_traces(frame: pd.DataFrame) -> list[NormalizedTrace]:
    """Restore dataclasses from a table produced by :func:`traces_to_frame`."""

    traces: list[NormalizedTrace] = []
    for record in frame.to_dict(orient="records"):
        onset_data = _json_or_none(record.get("harm_onset_json"))
        prefix_data = json.loads(record["prefix_annotations_json"])
        traces.append(
            NormalizedTrace(
                trace_id=record["trace_id"],
                source_split=record["source_split"],
                prompt=record["prompt"],
                response=record["response"],
                binary_label=_none_if_nan(record.get("binary_label")),
                original_label=record["original_label"],
                harm_categories=tuple(json.loads(record["harm_categories_json"])),
                prefix_annotations=tuple(PrefixAnnotation(**item) for item in prefix_data),
                harm_onset=HarmOnset(**onset_data) if onset_data else None,
                language=record["language"],
                source_rows=tuple(json.loads(record["source_rows_json"])),
                dataset_revision=record["dataset_revision"],
                exclusion_reason=_none_if_nan(record.get("exclusion_reason")),
            )
        )
    return traces


def write_trace_parquet(traces: list[NormalizedTrace], path: str | Path) -> str:
    """Write traces and return the SHA-256 checksum of the resulting file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    traces_to_frame(traces).to_parquet(path, index=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_trace_parquet(path: str | Path) -> list[NormalizedTrace]:
    return frame_to_traces(pd.read_parquet(path))


def _find_table(tables: dict[str, pd.DataFrame], prefix: str, split: str) -> str:
    matches = [name for name in tables if name.startswith(prefix) and name.endswith(split)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {prefix} {split} table, found {matches}")
    return matches[0]


def _sentence_count(text: str) -> int:
    parts = [part for part in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if part]
    return max(1, len(parts)) if text else 0


def _issue(
    split: str,
    table_kind: str,
    source_row: int,
    trace_id: str | None,
    code: str,
    details: str,
) -> dict[str, Any]:
    return {
        "split": split,
        "table_kind": table_kind,
        "source_row": source_row,
        "trace_id": trace_id,
        "code": code,
        "details": details,
    }


def _has_value(value: Any) -> bool:
    return (
        value is not None
        and not (isinstance(value, float) and pd.isna(value))
        and str(value) != ""
    )


def _json_or_none(value: Any) -> dict[str, Any] | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return json.loads(value)


def _none_if_nan(value: Any) -> Any:
    return None if isinstance(value, float) and pd.isna(value) else value
