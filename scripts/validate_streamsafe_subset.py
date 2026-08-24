#!/usr/bin/env python3
"""Validate the deterministic 500-trace subset."""

from __future__ import annotations

import argparse
from pathlib import Path

from streamguard_bench.data.build_subset import SubsetSpecification, validate_subset
from streamguard_bench.data.load_streamsafe import load_yaml_config
from streamguard_bench.data.normalize_streamsafe import read_trace_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_subset_v1.yaml")
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

    print("Поднабор из 500 трасс прошёл все проверки.")


if __name__ == "__main__":
    main()
