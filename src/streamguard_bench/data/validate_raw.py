"""Non-destructive quality checks for raw StreamSafe tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from streamguard_bench.data.inspect_schema import infer_column_roles


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    table: str
    code: str
    count: int
    details: str


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]

    @property
    def has_fatal(self) -> bool:
        return any(issue.severity == "fatal" for issue in self.issues)

    def to_frame(self) -> pd.DataFrame:
        columns = ["severity", "table", "code", "count", "details"]
        return pd.DataFrame([asdict(issue) for issue in self.issues], columns=columns)


def validate_tables(tables: dict[str, pd.DataFrame]) -> ValidationReport:
    issues: list[ValidationIssue] = []
    if not tables:
        return ValidationReport(
            (ValidationIssue("fatal", "<all>", "no_tables", 0, "No source tables loaded"),)
        )

    for table_name, frame in tables.items():
        if frame.empty:
            issues.append(ValidationIssue("fatal", table_name, "empty_table", 0, "Table is empty"))
            continue

        roles = infer_column_roles(frame.columns)
        text_candidates = list(dict.fromkeys(roles["prompt"] + roles["response"]))
        if not text_candidates:
            issues.append(
                ValidationIssue(
                    "fatal",
                    table_name,
                    "no_text_candidate",
                    len(frame),
                    "No prompt- or response-like column was detected",
                )
            )

        for column in text_candidates:
            series = frame[column]
            missing = int(series.isna().sum())
            empty = int(
                series.map(lambda value: isinstance(value, str) and not value.strip()).sum()
            )
            non_string = int(series.dropna().map(lambda value: not isinstance(value, str)).sum())
            if missing:
                issues.append(
                    ValidationIssue("warning", table_name, "missing_text", missing, column)
                )
            if empty:
                issues.append(ValidationIssue("warning", table_name, "empty_text", empty, column))
            if non_string:
                issues.append(
                    ValidationIssue("warning", table_name, "non_string_text", non_string, column)
                )

        duplicate_columns = text_candidates or list(frame.columns)
        duplicated = _duplicate_count(frame, duplicate_columns)
        if duplicated:
            issues.append(
                ValidationIssue(
                    "warning",
                    table_name,
                    "duplicate_rows",
                    duplicated,
                    "Duplicates measured on detected text columns",
                )
            )

        for label_column in roles["label"]:
            values = frame[label_column].dropna().map(_display_value).value_counts()
            if len(values) > 50:
                issues.append(
                    ValidationIssue(
                        "info",
                        table_name,
                        "high_cardinality_label",
                        int(len(values)),
                        label_column,
                    )
                )

        if not any(issue.table == table_name for issue in issues):
            issues.append(ValidationIssue("info", table_name, "ok", len(frame), "No issues found"))

    return ValidationReport(tuple(issues))


def _duplicate_count(frame: pd.DataFrame, columns: list[Any]) -> int:
    comparable = frame[columns].copy()
    for column in comparable.columns:
        comparable[column] = comparable[column].map(_display_value)
    return int(comparable.duplicated(keep=False).sum())


def _display_value(value: Any) -> str:
    if isinstance(value, dict):
        return repr(sorted(value.items()))
    if isinstance(value, list | tuple | set):
        return repr(list(value))
    return str(value)
