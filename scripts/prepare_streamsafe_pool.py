#!/usr/bin/env python3
"""Normalize full StreamSafe responses and prefixes into trace-level Parquet records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from huggingface_hub import HfApi

from streamguard_bench.data.load_streamsafe import load_streamsafe, load_yaml_config
from streamguard_bench.data.normalize_streamsafe import (
    normalize_streamsafe_tables,
    traces_to_frame,
    write_trace_parquet,
)
from streamguard_bench.data.token_coordinates import QwenTokenCoordinateMapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_subset_v1.yaml")
    parser.add_argument(
        "--without-tokenizer",
        action="store_true",
        help="Development-only mode: leave Qwen token coordinates empty.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.config).resolve()
    config = load_yaml_config(config_path)
    tokenizer_config = config["tokenizer"]
    mapper = None
    if not args.without_tokenizer:
        revision = tokenizer_config.get("revision")
        if not revision:
            info = HfApi().model_info(tokenizer_config["repository"])
            if not info.sha:
                raise RuntimeError("Hugging Face did not return a tokenizer revision")
            revision = info.sha
            config["tokenizer"]["revision"] = revision
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
            print(f"Зафиксирована ревизия токенизатора: {revision}")
        mapper = QwenTokenCoordinateMapper.from_pretrained(
            tokenizer_config["repository"], revision
        )

    dataset = config["dataset"]
    bundle = load_streamsafe(
        repository=dataset["repository"],
        revision=dataset["revision"],
        cache_dir=root / dataset["cache_dir"],
    )
    traces, issues = normalize_streamsafe_tables(
        bundle.tables,
        dataset_revision=bundle.resolved_revision,
        token_mapper=mapper,
    )
    paths = config["paths"]
    pool_path = root / paths["pool"]
    checksum = write_trace_parquet(traces, pool_path)
    issues_path = root / paths["normalization_issues"]
    issues_path.parent.mkdir(parents=True, exist_ok=True)
    issues.to_csv(issues_path, index=False)
    issue_summary_path = root / paths["normalization_issue_summary"]
    issue_summary_path.parent.mkdir(parents=True, exist_ok=True)
    issue_summary = (
        issues.groupby(["split", "table_kind", "code"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["split", "table_kind", "code"])
    )
    issue_summary.to_csv(issue_summary_path, index=False)
    frame = traces_to_frame(traces)
    manifest = {
        "dataset_repository": dataset["repository"],
        "dataset_revision": bundle.resolved_revision,
        "tokenizer_repository": tokenizer_config["repository"],
        "tokenizer_revision": config["tokenizer"].get("revision"),
        "token_coordinates_complete": not args.without_tokenizer,
        "trace_count": len(traces),
        "split_counts": frame["source_split"].value_counts().sort_index().to_dict(),
        "label_counts": frame["binary_label"].fillna("excluded").value_counts().to_dict(),
        "exclusion_counts": frame["exclusion_reason"].fillna("included").value_counts().to_dict(),
        "issue_counts": issues["code"].value_counts().to_dict(),
        "pool_sha256": checksum,
        "trace_ids_sha256": hashlib.sha256(
            "\n".join(sorted(frame["trace_id"])).encode()
        ).hexdigest(),
    }
    manifest_path = root / paths["pool_manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Нормализовано трасс: {len(traces):,}")
    print(f"Локальный пул: {pool_path}")
    print(f"Полный локальный журнал проблем: {issues_path}")
    print(f"Публичная сводка проблем: {issue_summary_path}")


if __name__ == "__main__":
    main()
