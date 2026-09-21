#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from streamguard_bench.experiments import load_experiment_run
from streamguard_bench.metrics import (
    compute_prompt_metrics,
    compute_response_policy_metrics,
    compute_streaming_metrics,
    paired_mode_differences,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke2", "full"], default="full")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = load_experiment_run(profile=args.profile)
    output = args.output or (
        Path("reports/tables") if args.profile == "full" else run.output_dir / "reports"
    )
    output.mkdir(parents=True, exist_ok=True)
    compute_prompt_metrics(run.prompt_decisions).to_csv(output / "prompt_metrics.csv", index=False)
    compute_response_policy_metrics(run.intervention_results).to_csv(
        output / "response_policy_metrics.csv", index=False
    )
    compute_streaming_metrics(run.intervention_results).to_csv(
        output / "streaming_metrics.csv", index=False
    )
    paired_mode_differences(run.intervention_results).to_csv(
        output / "paired_mode_differences.csv", index=False
    )
    prepared = Path("data/processed/singstreambench.parquet")
    if prepared.exists():
        frame = pd.read_parquet(prepared)
        frame.groupby(["query_label", "response_label"]).size().rename(
            "traces"
        ).reset_index().to_csv(output / "dataset_summary.csv", index=False)


if __name__ == "__main__":
    main()
