"""One entry point for running the benchmark from a script or from a notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from streamguard_bench.experiments import (
    PROFILES,
    RESULT_FILES,
    ExperimentRun,
    load_experiment_run,
    run_experiment,
)

DEFAULT_CONFIG = Path("configs/qwen3guard_baseline.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG, *, root: str | Path = ".") -> dict[str, Any]:
    """Read the YAML configuration that pins dataset, model, modes and policies."""
    return yaml.safe_load((Path(root) / path).read_text())


def results_dir(config: dict[str, Any], *, profile: str, root: str | Path = ".") -> Path:
    return Path(root) / config["experiment"]["output_dir"] / profile


def saved_run_exists(config: dict[str, Any], *, profile: str, root: str | Path = ".") -> bool:
    """True when the profile already has a complete saved run on disk."""
    directory = results_dir(config, profile=profile, root=root)
    return all((directory / name).exists() for name in RESULT_FILES.values())


def run_or_load(
    config: dict[str, Any],
    *,
    profile: str,
    root: str | Path = ".",
    force: bool = False,
    guard: Any | None = None,
) -> ExperimentRun:
    """Return the saved run for this profile, or score the traces when none exists.

    `force` ignores the saved tables and goes through scoring again. Individual traces
    still come from their checkpoints while `runtime.resume` stays true, so a forced
    call rebuilds the tables without paying for inference twice.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    root = Path(root)
    output_dir = root / config["experiment"]["output_dir"]
    if not force and saved_run_exists(config, profile=profile, root=root):
        return load_experiment_run(output_dir=output_dir, profile=profile)
    experiment = config["experiment"]
    return run_experiment(
        dataset=pd.read_parquet(root / config["dataset"]["prepared_path"]),
        guard=guard if guard is not None else build_guard(config),
        profile=profile,
        modes=tuple(experiment["modes"]),
        policies=tuple(experiment["policies"]),
        output_dir=output_dir,
        resume=config["runtime"]["resume"],
        seed=experiment["seed"],
        max_sentence_tokens=experiment["max_sentence_tokens"],
        dataset_revision=config["dataset"]["revision"],
    )


def build_guard(config: dict[str, Any]) -> Any:
    """Instantiate the guard named by the configuration; needs the `models` extra."""
    from streamguard_bench.guards import Qwen3GuardStreamAdapter

    model = config["model"]
    return Qwen3GuardStreamAdapter(
        model["repository"],
        revision=model["revision"],
        tokenizer_revision=model["tokenizer_revision"],
        device=config["runtime"].get("device"),
    )
