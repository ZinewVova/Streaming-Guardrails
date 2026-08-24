#!/usr/bin/env python3
"""Load StreamSafe and persist raw schema and quality reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from streamguard_bench.data.inspect_schema import (
    dataset_overview,
    schema_summary,
)
from streamguard_bench.data.load_streamsafe import load_streamsafe, load_yaml_config
from streamguard_bench.data.validate_raw import validate_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_streamsafe.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(project_root / args.config)
    dataset_config = config["dataset"]
    bundle = load_streamsafe(
        repository=dataset_config["repository"],
        revision=dataset_config.get("revision"),
        cache_dir=project_root / dataset_config["cache_dir"],
        allow_remote_code=bool(config.get("loading", {}).get("allow_remote_code", False)),
    )

    tables_dir = project_root / config["reports"]["tables_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    schema_summary(bundle.tables).to_csv(tables_dir / "raw_schema_summary.csv", index=False)
    dataset_overview(bundle.tables).to_csv(tables_dir / "dataset_overview.csv", index=False)
    report = validate_tables(bundle.tables)
    report.to_frame().to_csv(tables_dir / "raw_validation.csv", index=False)

    print(f"Saved reports to {tables_dir}")
    print(report.to_frame().to_string(index=False))
    if report.has_fatal:
        raise SystemExit("Fatal raw-data validation issues were detected")


if __name__ == "__main__":
    main()
