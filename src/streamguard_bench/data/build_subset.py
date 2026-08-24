"""Deterministic construction and validation of the first 500-trace subset."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from streamguard_bench.contracts import NormalizedTrace


@dataclass(frozen=True)
class SubsetSpecification:
    seed: int = 42
    train_safe: int = 200
    train_unsafe: int = 200
    val_safe: int = 50
    val_unsafe: int = 50
    minimum_safe_per_length_bucket: int = 60
    minimum_unsafe_early: int = 120
    minimum_unsafe_middle: int = 50
    minimum_unsafe_late: int = 40
    minimum_category_examples: int = 5


def build_balanced_subset(
    traces: list[NormalizedTrace], specification: SubsetSpecification
) -> tuple[list[NormalizedTrace], pd.DataFrame]:
    """Select an exact, reproducible subset and fail when any declared quota is impossible."""

    raw_eligible = [
        trace
        for trace in traces
        if trace.exclusion_reason is None and trace.binary_label in {"safe", "unsafe"}
    ]
    grouped: dict[str, list[NormalizedTrace]] = {}
    for trace in raw_eligible:
        grouped.setdefault(trace.trace_id, []).append(trace)
    eligible = [
        sorted(group, key=lambda trace: (trace.source_split != "val", trace.source_split))[0]
        for group in grouped.values()
    ]
    metadata = _selection_metadata(eligible)
    by_id = {trace.trace_id: trace for trace in eligible}
    selected_ids: set[str] = set()

    safe_targets = {
        ("train", "short"): 67,
        ("train", "medium"): 67,
        ("train", "long"): 66,
        ("val", "short"): 17,
        ("val", "medium"): 17,
        ("val", "long"): 16,
    }
    for cell, target in safe_targets.items():
        candidates = metadata[
            (metadata["binary_label"] == "safe")
            & (metadata["source_split"] == cell[0])
            & (metadata["length_bucket"] == cell[1])
        ]
        chosen = _deterministic_ids(candidates["trace_id"].tolist(), target, specification.seed)
        selected_ids.update(chosen)

    unsafe_minimum_cells = {
        ("train", "early"): 95,
        ("train", "middle"): 32,
        ("train", "late"): 33,
        ("val", "early"): 25,
        ("val", "middle"): 18,
        ("val", "late"): 7,
    }
    category_counts = Counter(
        category
        for trace in eligible
        if trace.binary_label == "unsafe"
        for category in trace.harm_categories
    )
    for cell, target in unsafe_minimum_cells.items():
        candidates = [
            trace
            for trace in eligible
            if trace.binary_label == "unsafe"
            and trace.source_split == cell[0]
            and metadata.loc[trace.trace_id, "onset_bucket"] == cell[1]
        ]
        chosen = _category_aware_ids(candidates, target, category_counts, specification.seed)
        selected_ids.update(chosen)

    train_unsafe_selected = sum(
        by_id[trace_id].source_split == "train"
        and by_id[trace_id].binary_label == "unsafe"
        for trace_id in selected_ids
    )
    remaining_train = [
        trace
        for trace in eligible
        if trace.source_split == "train"
        and trace.binary_label == "unsafe"
        and trace.trace_id not in selected_ids
    ]
    selected_ids.update(
        _category_aware_ids(
            remaining_train,
            specification.train_unsafe - train_unsafe_selected,
            category_counts,
            specification.seed + 1,
        )
    )

    selected = [by_id[trace_id] for trace_id in sorted(selected_ids)]
    validation = validate_subset(selected, eligible, specification)
    failures = validation[validation["status"] == "fail"]
    if not failures.empty:
        messages = "; ".join(f"{row.check}: {row.details}" for row in failures.itertuples())
        raise ValueError(f"Subset quotas are not satisfiable: {messages}")
    return selected, validation


def validate_subset(
    subset: list[NormalizedTrace],
    pool: list[NormalizedTrace],
    specification: SubsetSpecification,
) -> pd.DataFrame:
    """Return explicit pass/fail rows for every public subset guarantee."""

    metadata = _selection_metadata(subset)
    pool_category_counts = Counter(
        category
        for trace in pool
        if trace.binary_label == "unsafe"
        for category in trace.harm_categories
    )
    checks: list[dict[str, Any]] = []

    def add(check: str, actual: int, expected: int, operator: str = "eq") -> None:
        passed = actual == expected if operator == "eq" else actual >= expected
        checks.append(
            {
                "check": check,
                "status": "pass" if passed else "fail",
                "actual": actual,
                "expected": expected,
                "details": f"required {operator} {expected}, observed {actual}",
            }
        )

    add("total_traces", len(subset), 500)
    add("unique_trace_ids", len({trace.trace_id for trace in subset}), 500)
    for split, label, expected in (
        ("train", "safe", specification.train_safe),
        ("train", "unsafe", specification.train_unsafe),
        ("val", "safe", specification.val_safe),
        ("val", "unsafe", specification.val_unsafe),
    ):
        actual = int(
            ((metadata["source_split"] == split) & (metadata["binary_label"] == label)).sum()
        )
        add(f"{split}_{label}", actual, expected)
    for bucket in ("short", "medium", "long"):
        actual = int(
            ((metadata["binary_label"] == "safe") & (metadata["length_bucket"] == bucket)).sum()
        )
        add(
            f"safe_length_{bucket}",
            actual,
            specification.minimum_safe_per_length_bucket,
            "ge",
        )
    onset_minimums = {
        "early": specification.minimum_unsafe_early,
        "middle": specification.minimum_unsafe_middle,
        "late": specification.minimum_unsafe_late,
    }
    for bucket, expected in onset_minimums.items():
        actual = int(
            (
                (metadata["binary_label"] == "unsafe")
                & (metadata["onset_bucket"] == bucket)
            ).sum()
        )
        add(f"unsafe_onset_{bucket}", actual, expected, "ge")
    selected_category_counts = Counter(
        category
        for trace in subset
        if trace.binary_label == "unsafe"
        for category in trace.harm_categories
    )
    for category, available in sorted(pool_category_counts.items()):
        expected = min(available, specification.minimum_category_examples)
        add(f"category::{category}", selected_category_counts[category], expected, "ge")
    return pd.DataFrame(checks)


def subset_distribution(traces: list[NormalizedTrace]) -> pd.DataFrame:
    metadata = _selection_metadata(traces).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for columns in (["source_split", "binary_label"], ["binary_label", "length_bucket"]):
        grouped = metadata.groupby(columns, dropna=False).size().reset_index(name="count")
        grouped.insert(0, "dimension", "+".join(columns))
        rows.extend(grouped.to_dict(orient="records"))
    onset = metadata[metadata["binary_label"] == "unsafe"]
    grouped = onset.groupby(["onset_bucket"], dropna=False).size().reset_index(name="count")
    grouped.insert(0, "dimension", "unsafe_onset_bucket")
    rows.extend(grouped.to_dict(orient="records"))
    for trace in traces:
        for category in trace.harm_categories:
            rows.append(
                {
                    "dimension": "harm_category",
                    "harm_category": category,
                    "count": 1,
                }
            )
    frame = pd.DataFrame(rows)
    category_mask = frame["dimension"] == "harm_category"
    categories = (
        frame[category_mask]
        .groupby(["dimension", "harm_category"], as_index=False)["count"]
        .sum()
    )
    return pd.concat([frame[~category_mask], categories], ignore_index=True)


def selection_manifest(
    traces: list[NormalizedTrace], specification: SubsetSpecification, checksum: str
) -> dict[str, Any]:
    ids = "\n".join(sorted(trace.trace_id for trace in traces)).encode()
    return {
        "specification": asdict(specification),
        "trace_count": len(traces),
        "trace_ids_sha256": hashlib.sha256(ids).hexdigest(),
        "parquet_sha256": checksum,
        "dataset_revision": traces[0].dataset_revision if traces else None,
    }


def _selection_metadata(traces: list[NormalizedTrace]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    safe_lengths = pd.Series(
        [len(trace.response) for trace in traces if trace.binary_label == "safe"], dtype=float
    )
    q33 = float(safe_lengths.quantile(1 / 3)) if not safe_lengths.empty else 0
    q66 = float(safe_lengths.quantile(2 / 3)) if not safe_lengths.empty else 0
    for trace in traces:
        length = len(trace.response)
        length_bucket = "short" if length <= q33 else "medium" if length <= q66 else "long"
        onset_fraction = None
        onset_bucket = None
        if trace.harm_onset and length:
            onset_fraction = trace.harm_onset.upper_character / length
            if onset_fraction <= 0.33:
                onset_bucket = "early"
            elif onset_fraction <= 0.66:
                onset_bucket = "middle"
            else:
                onset_bucket = "late"
        rows.append(
            {
                "trace_id": trace.trace_id,
                "source_split": trace.source_split,
                "binary_label": trace.binary_label,
                "length_bucket": length_bucket,
                "onset_fraction": onset_fraction,
                "onset_bucket": onset_bucket,
            }
        )
    return pd.DataFrame(rows).set_index("trace_id", drop=False)


def _deterministic_ids(ids: list[str], target: int, seed: int) -> list[str]:
    if len(ids) < target:
        raise ValueError(f"Need {target} candidates, found {len(ids)}")
    ordered = sorted(ids)
    random.Random(seed).shuffle(ordered)
    return ordered[:target]


def _category_aware_ids(
    traces: list[NormalizedTrace], target: int, counts: Counter[str], seed: int
) -> list[str]:
    if len(traces) < target:
        raise ValueError(f"Need {target} unsafe candidates, found {len(traces)}")
    random_rank = list(range(len(traces)))
    random.Random(seed).shuffle(random_rank)
    ordered_traces = sorted(traces, key=lambda trace: trace.trace_id)
    rank_by_id = {
        trace.trace_id: rank
        for trace, rank in zip(ordered_traces, random_rank, strict=True)
    }

    def key(trace: NormalizedTrace) -> tuple[float, int, str]:
        rarity = sum(1 / counts[category] for category in trace.harm_categories if counts[category])
        return (-rarity, rank_by_id[trace.trace_id], trace.trace_id)

    return [trace.trace_id for trace in sorted(traces, key=key)[:target]]
