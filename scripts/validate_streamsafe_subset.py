#!/usr/bin/env python3
"""Validate the subset and, when completed, enrich manual-review annotations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from streamguard_bench.data.build_subset import SubsetSpecification, validate_subset
from streamguard_bench.data.load_streamsafe import load_yaml_config
from streamguard_bench.data.manual_audit import (
    audit_summary,
    validate_completed_audit,
)
from streamguard_bench.data.normalize_streamsafe import read_trace_parquet
from streamguard_bench.data.token_coordinates import QwenTokenCoordinateMapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_subset_v1.yaml")
    parser.add_argument("--skip-manual", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(root / args.config)
    paths = config["paths"]
    pool = read_trace_parquet(root / paths["pool"])
    subset = read_trace_parquet(root / paths["subset"])
    validation = validate_subset(subset, pool, SubsetSpecification(**config["subset"]))
    validation.to_csv(root / paths["subset_validation"], index=False)
    failed = validation[validation["status"] == "fail"]
    if not failed.empty:
        raise SystemExit(f"Subset validation failed:\n{failed.to_string(index=False)}")

    if args.skip_manual:
        print("Поднабор прошёл проверки; ручной аудит пропущен.")
        return
    audit_path = root / paths["audit_template"]
    audit = pd.read_csv(audit_path, keep_default_na=False)
    tokenizer = config["tokenizer"]
    if not tokenizer.get("revision"):
        raise SystemExit("Tokenizer revision is not pinned")
    mapper = QwenTokenCoordinateMapper.from_pretrained(
        tokenizer["repository"], tokenizer["revision"]
    )
    enriched, issues = validate_completed_audit(audit, subset, token_mapper=mapper)
    enriched.to_csv(root / paths["audit_annotations_public"], index=False)
    issues.to_csv(root / paths["audit_conflicts"], index=False)
    audit_summary(enriched, issues).to_csv(root / paths["audit_summary"], index=False)
    if not issues.empty:
        raise SystemExit(
            "Subset is valid, but manual audit is incomplete or conflicting; "
            f"see {root / paths['audit_conflicts']}"
        )
    print("Поднабор и ручной аудит прошли все проверки.")


if __name__ == "__main__":
    main()
