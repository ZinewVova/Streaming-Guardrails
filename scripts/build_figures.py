#!/usr/bin/env python
"""Render the figures shown in README from the saved full runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from streamguard_bench import plots  # noqa: E402
from streamguard_bench.experiments import sweep_decision_rules  # noqa: E402
from streamguard_bench.metrics import compute_streaming_metrics, stopped_within  # noqa: E402
from streamguard_bench.pipeline import load_saved_runs  # noqa: E402

THRESHOLDS = (None, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95)
COUNTS = (1, 2, 3, 4, 6, 8, 10)
RULES = [
    {"threshold": threshold, "trigger_count": count, "trigger_mode": mode}
    for threshold in THRESHOLDS
    for count in COUNTS
    for mode in ("cumulative", "consecutive")
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    runs = load_saved_runs(profile="full")
    if not runs:
        raise SystemExit("No saved full runs; run `make full MODEL=...` first.")
    dataset = pd.read_parquet(runs[0].config["dataset"]["prepared_path"])
    args.output.mkdir(parents=True, exist_ok=True)

    grids = []
    for item in runs:
        experiment = item.config["experiment"]
        grid = sweep_decision_rules(
            dataset=dataset,
            output_dir=experiment["output_dir"],
            profile="full",
            policy=item.policy,
            rules=RULES,
        )
        grid["default"] = (
            grid["threshold"].isna()
            & (grid["trigger_count"] == experiment.get("trigger_count", 1))
            & (grid["trigger_mode"] == experiment.get("trigger_mode", "cumulative"))
        )
        grids.append(grid.assign(model=item.name))
    curves = pd.concat(
        [stopped_within(item.results, range(0, 61)).assign(model=item.name) for item in runs],
        ignore_index=True,
    )
    streaming = pd.concat(
        [compute_streaming_metrics(item.results).assign(model=item.name) for item in runs],
        ignore_index=True,
    )
    figures = {
        "rule_frontier": plots.plot_rule_frontier(pd.concat(grids, ignore_index=True)),
        "stopping_curves": plots.plot_stopping_curves(curves),
        "leakage_by_buffer": plots.plot_model_leakage(streaming, unit="words"),
    }
    for name, figure in figures.items():
        figure.savefig(args.output / f"{name}.png", dpi=130)
        print(args.output / f"{name}.png")


if __name__ == "__main__":
    main()
