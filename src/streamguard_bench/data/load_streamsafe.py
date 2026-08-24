"""Load StreamSafe without coercing incompatible source schemas."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi, snapshot_download

SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl", ".parquet"}


@dataclass(frozen=True)
class StreamSafeBundle:
    """Loaded source tables and their immutable provenance."""

    repository: str
    requested_revision: str | None
    resolved_revision: str
    loader_mode: str
    tables: dict[str, pd.DataFrame]
    source_files: dict[str, str]
    snapshot_path: str | None = None

    def manifest(self) -> dict[str, Any]:
        return {
            "repository": self.repository,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "loader_mode": self.loader_mode,
            "snapshot_path": self.snapshot_path,
            "tables": {
                name: {
                    "rows": len(frame),
                    "columns": [str(column) for column in frame.columns],
                    "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
                    "source_file": self.source_files.get(name),
                }
                for name, frame in self.tables.items()
            },
        }


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    return config


def resolve_revision(repository: str, revision: str | None = None) -> str:
    """Resolve a branch or tag to an immutable dataset commit hash."""

    info = HfApi().dataset_info(repo_id=repository, revision=revision)
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a revision for {repository}")
    return info.sha


def load_streamsafe(
    *,
    repository: str,
    revision: str | None,
    cache_dir: str | Path,
    allow_remote_code: bool = False,
) -> StreamSafeBundle:
    """Load StreamSafe through datasets, with a file-by-file fallback.

    The fallback intentionally preserves separate source tables. It never concatenates
    files with different schemas.
    """

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    resolved = resolve_revision(repository, revision)

    try:
        loaded = _load_with_datasets(
            repository=repository,
            revision=resolved,
            cache_dir=cache_path,
            allow_remote_code=allow_remote_code,
        )
        if loaded:
            return StreamSafeBundle(
                repository=repository,
                requested_revision=revision,
                resolved_revision=resolved,
                loader_mode="datasets",
                tables=loaded,
                source_files={name: f"datasets://{name}" for name in loaded},
            )
    except Exception as primary_error:  # noqa: BLE001 - fallback is intentional
        fallback_reason = repr(primary_error)
    else:
        fallback_reason = "datasets returned no tables"

    snapshot_path = Path(
        snapshot_download(
            repo_id=repository,
            repo_type="dataset",
            revision=resolved,
            local_dir=cache_path,
        )
    )
    tables, source_files = load_local_snapshot(snapshot_path)
    if not tables:
        raise RuntimeError(
            f"No supported StreamSafe files found in {snapshot_path}; "
            f"primary loader failure: {fallback_reason}"
        )
    return StreamSafeBundle(
        repository=repository,
        requested_revision=revision,
        resolved_revision=resolved,
        loader_mode=f"snapshot_fallback: {fallback_reason}",
        tables=tables,
        source_files=source_files,
        snapshot_path=str(snapshot_path),
    )


def _load_with_datasets(
    *,
    repository: str,
    revision: str,
    cache_dir: Path,
    allow_remote_code: bool,
) -> dict[str, pd.DataFrame]:
    kwargs = {
        "path": repository,
        "revision": revision,
        "cache_dir": str(cache_dir / "datasets_cache"),
    }
    try:
        loaded = load_dataset(**kwargs, trust_remote_code=allow_remote_code)
    except TypeError:
        loaded = load_dataset(**kwargs)

    if isinstance(loaded, DatasetDict):
        return {str(split): dataset.to_pandas() for split, dataset in loaded.items()}
    if hasattr(loaded, "to_pandas"):
        return {"dataset": loaded.to_pandas()}
    raise TypeError(f"Unsupported datasets result: {type(loaded).__name__}")


def load_local_snapshot(root: str | Path) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Read every supported data file as an independent table."""

    root_path = Path(root)
    files = sorted(
        path
        for path in root_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and path.name != "manifest.json"
        and not any(part.startswith(".") for part in path.relative_to(root_path).parts)
    )
    tables: dict[str, pd.DataFrame] = {}
    source_files: dict[str, str] = {}
    for path in files:
        name = _unique_table_name(path, root_path, existing=set(tables))
        tables[name] = read_data_file(path)
        source_files[name] = str(path.relative_to(root_path))
    return tables, source_files


def read_data_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                return pd.DataFrame(payload)
            if isinstance(payload, dict):
                for key in ("data", "records", "examples"):
                    if isinstance(payload.get(key), list):
                        return pd.DataFrame(payload[key])
                return pd.json_normalize(payload)
        except json.JSONDecodeError:
            return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported data file: {path}")


def _unique_table_name(path: Path, root: Path, *, existing: set[str]) -> str:
    relative = path.relative_to(root).with_suffix("")
    base = re.sub(r"[^a-z0-9]+", "_", "_".join(relative.parts).lower()).strip("_")
    base = base or "table"
    candidate = base
    counter = 2
    while candidate in existing:
        candidate = f"{base}_{counter}"
        counter += 1
    return candidate
