"""Loading, validation, preparation, and manifesting for SingStreamBench."""

from __future__ import annotations

import hashlib
import json
from numbers import Integral
from pathlib import Path
from typing import Any

import pandas as pd

from streamguard_bench.data.token_coordinates import unsafe_start_token

REPOSITORY = "inclusionAI/SingStreamBench"
REVISION = "957b703ca028af6aa834b95ec0893b81e4d24a71"
SOURCE_COLUMNS = (
    "Query",
    "Response",
    "Query_Label",
    "Response_Label",
    "Unsafe_Start_Index",
    "Safe_Prefix",
)


def load_dataset_frame(
    repository: str = REPOSITORY,
    *,
    revision: str = REVISION,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    from datasets import load_dataset

    loaded = load_dataset(repository, revision=revision, cache_dir=cache_dir)
    split = loaded["train"] if isinstance(loaded, dict) else loaded
    return split.to_pandas() if hasattr(split, "to_pandas") else pd.DataFrame(split)


def validate_source(frame: pd.DataFrame, *, expected_rows: int | None = 210) -> pd.DataFrame:
    checks: list[tuple[str, bool, str]] = []
    missing = sorted(set(SOURCE_COLUMNS) - set(frame.columns))
    checks.append(("required_columns", not missing, ", ".join(missing)))
    if missing:
        return _report(checks)
    checks.append(
        ("row_count", expected_rows is None or len(frame) == expected_rows, str(len(frame)))
    )
    labels = {"safe", "unsafe"}
    checks.append(("query_labels", set(frame["Query_Label"]) <= labels, ""))
    checks.append(("response_labels", set(frame["Response_Label"]) <= labels, ""))
    strings_ok = all(
        isinstance(value, str) and bool(value)
        for column in ("Query", "Response")
        for value in frame[column]
    )
    checks.append(("nonempty_text", strings_ok, ""))
    onset_type = all(
        isinstance(value, Integral) and not isinstance(value, bool)
        for value in frame["Unsafe_Start_Index"]
    )
    checks.append(("onset_integer", onset_type, ""))
    boundary_ok = onset_type and all(
        0 <= onset <= len(response)
        for onset, response in zip(frame["Unsafe_Start_Index"], frame["Response"], strict=True)
    )
    checks.append(("onset_range", boundary_ok, ""))
    if boundary_ok:
        prefix_matches = [
            prefix == response[:onset]
            for prefix, response, onset in zip(
                frame["Safe_Prefix"],
                frame["Response"],
                frame["Unsafe_Start_Index"],
                strict=True,
            )
        ]
        mismatching_prefixes = prefix_matches.count(False)
        prefix_ok = mismatching_prefixes == 0
        prefix_detail = (
            f"{len(prefix_matches) - mismatching_prefixes} matching, "
            f"{mismatching_prefixes} mismatching"
        )
    else:
        prefix_ok = False
        prefix_detail = "not evaluated: invalid onset"
    checks.append(("safe_prefix_exact", prefix_ok, prefix_detail))
    label_onset_ok = boundary_ok and all(
        (label == "safe" and onset == len(response))
        or (label == "unsafe" and onset < len(response))
        for label, onset, response in zip(
            frame["Response_Label"], frame["Unsafe_Start_Index"], frame["Response"], strict=True
        )
    )
    checks.append(("label_onset_consistency", label_onset_ok, ""))
    checks.append(
        ("unicode_unchanged", all(value == str(value) for value in frame["Response"]), "")
    )
    return _report(checks)


def prepare_frame(
    frame: pd.DataFrame,
    tokenizer: Any,
    *,
    revision: str = REVISION,
    expected_rows: int | None = 210,
) -> pd.DataFrame:
    report = validate_source(frame, expected_rows=expected_rows)
    failed = report.loc[~report["passed"], "check"].tolist()
    if failed:
        raise ValueError(f"SingStreamBench validation failed: {failed}")
    rows = []
    for position, row in enumerate(frame.to_dict(orient="records")):
        encoded = tokenizer(
            row["Response"],
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
        token_ids = _flatten(encoded["input_ids"])
        offsets = [tuple(map(int, pair)) for pair in _flatten_offsets(encoded["offset_mapping"])]
        rows.append(
            {
                "trace_id": f"singstreambench-{position:03d}",
                "query": row["Query"],
                "response": row["Response"],
                "query_label": row["Query_Label"],
                "response_label": row["Response_Label"],
                "unsafe_start_character": int(row["Unsafe_Start_Index"]),
                "safe_prefix": row["Safe_Prefix"],
                "unsafe_start_token": unsafe_start_token(
                    unsafe_start_character=int(row["Unsafe_Start_Index"]),
                    response=row["Response"],
                    offsets=offsets,
                ),
                "response_token_count": len(token_ids),
                "dataset_revision": revision,
            }
        )
    prepared = pd.DataFrame(rows)
    if prepared["trace_id"].duplicated().any():
        raise ValueError("Duplicate trace_id values")
    return prepared


def build_manifest(
    frame: pd.DataFrame, parquet_path: Path, *, revision: str = REVISION
) -> dict[str, Any]:
    return {
        "repository": REPOSITORY,
        "revision": revision,
        "rows": len(frame),
        "query_labels": frame["query_label"].value_counts().sort_index().to_dict(),
        "response_labels": frame["response_label"].value_counts().sort_index().to_dict(),
        "sha256": hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
    }


def write_manifest(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _report(checks: list[tuple[str, bool, str]]) -> pd.DataFrame:
    return pd.DataFrame(checks, columns=["check", "passed", "detail"])


def _flatten(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while (
        isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple))
    ):
        value = value[0]
    return list(value)


def _flatten_offsets(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while value and len(value) == 1 and isinstance(value[0][0], (list, tuple)):
        value = value[0]
    return list(value)
