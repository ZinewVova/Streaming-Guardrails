#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from streamguard_bench.data import load_dataset_frame, validate_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report", type=Path, default=Path("reports/tables/dataset_validation.csv")
    )
    args = parser.parse_args()
    source = load_dataset_frame(cache_dir="data/cache/huggingface")
    report = validate_source(source)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report, index=False)
    source.groupby(["Query_Label", "Response_Label"]).size().rename("traces").reset_index().to_csv(
        args.report.parent / "dataset_summary.csv", index=False
    )
    if not report["passed"].all():
        raise SystemExit("Dataset validation failed")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
