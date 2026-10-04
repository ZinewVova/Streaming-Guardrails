#!/usr/bin/env python
"""Render the figures shown in README from the saved full runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from streamguard_bench import plots  # noqa: E402
from streamguard_bench.metrics import (  # noqa: E402
    compute_response_policy_metrics,
    stopped_within,
)
from streamguard_bench.pipeline import load_saved_runs, sweep_saved_run  # noqa: E402

# The buffer mode in which the guards are compared with each other.
MODE = "sentence"
THRESHOLDS = (None, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95)
COUNTS = (1, 2, 3, 4, 6, 8, 10)
RULES = [
    {"threshold": threshold, "trigger_count": count, "trigger_mode": trigger}
    for threshold in THRESHOLDS
    for count in COUNTS
    for trigger in ("cumulative", "consecutive")
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    runs = load_saved_runs(profile="full")
    if not runs:
        raise SystemExit("No saved full runs; run `make full MODEL=...` first.")
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(runs[0].config["dataset"]["prepared_path"])
    rule_grid = pd.concat(
        [
            sweep_saved_run(item, dataset=dataset, rules=RULES, mode=MODE, profile="full")
            for item in runs
        ],
        ignore_index=True,
    )

    errors = pd.concat(
        [
            compute_response_policy_metrics(item.results, by_mode=True).assign(model=item.name)
            for item in runs
        ],
        ignore_index=True,
    )
    curves = pd.concat(
        [
            stopped_within(item.results, range(0, 61), mode=MODE).assign(model=item.name)
            for item in runs
        ],
        ignore_index=True,
    )
    cost = pd.concat(
        [
            pd.read_csv(
                Path("reports/tables")
                / Path(item.config["experiment"]["output_dir"]).name
                / "guard_cost.csv"
            )
            .query("mode == @MODE and policy == @item.policy")
            .assign(model=item.name)
            for item in runs
        ],
        ignore_index=True,
    ).merge(errors[errors["mode"] == MODE], on="model")
    figures = {
        "error_tradeoff": plots.plot_error_tradeoff(
            errors[errors["mode"] == MODE], subtitle=f"режим {MODE}; ближе к нулю — лучше; 95% CI"
        ),
        "errors_by_buffer": plots.plot_model_mode_errors(errors),
        "rule_frontier": plots.plot_rule_frontier(rule_grid),
        "stopping_curves": plots.plot_stopping_curves(
            curves, subtitle=f"режим {MODE}; выше и левее — лучше"
        ),
        "cost_quality": plots.plot_cost_quality(
            cost,
            column="tflops",
            label="вычислений на ответ, TFLOPs (логарифмическая шкала)",
            log=True,
        ),
    }
    for name, figure in figures.items():
        figure.savefig(args.output / f"{name}.png", dpi=130)
        print(args.output / f"{name}.png")


if __name__ == "__main__":
    main()
