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
from streamguard_bench.pipeline import DEFAULT_CONFIG, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke2", "full"], default="full")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    run_name = Path(config["experiment"]["output_dir"]).name
    run = load_experiment_run(config["experiment"]["output_dir"], profile=args.profile)
    # The Qwen3Guard baseline keeps its original location; other guards get a subfolder.
    full_reports = Path("reports/tables")
    if run_name != "qwen3guard":
        full_reports = full_reports / run_name
    output = args.output or (full_reports if args.profile == "full" else run.output_dir / "reports")
    output.mkdir(parents=True, exist_ok=True)
    policies = tuple(config["experiment"]["policies"])
    evaluated = run.prompt_decisions[run.prompt_decisions["risk_label"] != "not_evaluated"]
    if len(evaluated):
        compute_prompt_metrics(evaluated, policies).to_csv(
            output / "prompt_metrics.csv", index=False
        )
    compute_response_policy_metrics(run.intervention_results).to_csv(
        output / "response_policy_metrics.csv", index=False
    )
    compute_streaming_metrics(run.intervention_results).to_csv(
        output / "streaming_metrics.csv", index=False
    )
    paired_mode_differences(run.intervention_results).to_csv(
        output / "paired_mode_differences.csv", index=False
    )
    prepared = Path(config["dataset"]["prepared_path"])
    if prepared.exists():
        frame = pd.read_parquet(prepared)
        frame.groupby(["query_label", "response_label"]).size().rename(
            "traces"
        ).reset_index().to_csv(output / "dataset_summary.csv", index=False)


if __name__ == "__main__":
    main()
