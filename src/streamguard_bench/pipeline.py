"""One entry point for running the benchmark from a script or from a notebook."""

from __future__ import annotations

import json
from dataclasses import dataclass
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
    """True when the profile has a complete saved run replayed under the configured rule.

    Tables replayed under another stop rule, policy set or mode list do not count: they
    would be read as if they answered the current configuration.
    """
    directory = results_dir(config, profile=profile, root=root)
    if not all((directory / name).exists() for name in RESULT_FILES.values()):
        return False
    metadata_path = directory / "run_metadata.json"
    if not metadata_path.exists():
        return True
    saved = json.loads(metadata_path.read_text())
    model = config.get("model", {})
    if model.get("adapter") == "sentguard":
        settings = saved.get("inference_settings", {})
        if (
            saved.get("model_id") != model["repository"]
            or saved.get("model_revision") != model["revision"]
            or settings.get("max_new_tokens") != model["max_new_tokens"]
            or settings.get("quantization") != model.get("quantization")
            or settings.get("generation_tokenizer_revision")
            != (model["tokenizer_revision"] or model["revision"])
        ):
            return False
    experiment = config["experiment"]
    return (
        saved["modes"] == list(experiment["modes"])
        and [item["name"] for item in saved["policies"]] == list(experiment["policies"])
        and saved["max_sentence_tokens"] == experiment["max_sentence_tokens"]
        and saved.get("trigger_count", 1) == experiment.get("trigger_count", 1)
        and saved.get("trigger_mode", "cumulative") == experiment.get("trigger_mode", "cumulative")
        and saved.get("trigger_window") == experiment.get("trigger_window")
        and saved.get("checkpoint_scoring", False) == experiment.get("checkpoint_scoring", False)
    )


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
        trigger_count=experiment.get("trigger_count", 1),
        trigger_mode=experiment.get("trigger_mode", "cumulative"),
        trigger_window=experiment.get("trigger_window"),
    )


@dataclass(frozen=True)
class SavedRun:
    """One guard's saved run together with the decision rule it is compared under."""

    name: str
    config: dict[str, Any]
    run: ExperimentRun
    policy: str

    @property
    def results(self) -> pd.DataFrame:
        """Replay results under the guard's primary policy, tagged with the guard name."""
        frame = self.run.intervention_results
        return frame[frame["policy"] == self.policy].assign(model=self.name)


def load_saved_runs(
    *, profile: str, root: str | Path = ".", config_dir: str | Path = "configs"
) -> list[SavedRun]:
    """Load every configured guard that already has a complete saved run.

    Guards come in config file name order. A guard is compared under the first policy its
    config lists, so each guard contributes exactly one decision rule.
    """
    root = Path(root)
    runs = []
    for path in sorted((root / config_dir).glob("*.yaml")):
        config = yaml.safe_load(path.read_text())
        if not saved_run_exists(config, profile=profile, root=root):
            continue
        runs.append(
            SavedRun(
                name=config["model"].get("display_name")
                or config["model"]["repository"].rsplit("/", 1)[-1],
                config=config,
                run=load_experiment_run(
                    output_dir=root / config["experiment"]["output_dir"], profile=profile
                ),
                policy=next(iter(config["experiment"]["policies"])),
            )
        )
    return runs


def build_guard(config: dict[str, Any]) -> Any:
    """Instantiate the guard named by the configuration; needs the `models` extra."""
    model = config["model"]
    adapter = model.get("adapter", "qwen3guard_stream")
    common = {
        "revision": model["revision"],
        "tokenizer_revision": model["tokenizer_revision"],
        "device": config["runtime"].get("device"),
    }
    if adapter == "qwen3guard_stream":
        from streamguard_bench.guards import Qwen3GuardStreamAdapter

        return Qwen3GuardStreamAdapter(model["repository"], **common)
    if adapter == "scm":
        from streamguard_bench.guards import SCMAdapter

        return SCMAdapter(model["repository"], threshold=model["threshold"], **common)
    if adapter == "qwen3guard_gen":
        from streamguard_bench.guards import Qwen3GuardGenAdapter

        return Qwen3GuardGenAdapter(
            model["repository"],
            modes=config["experiment"]["modes"],
            max_sentence_tokens=config["experiment"]["max_sentence_tokens"],
            **common,
        )
    if adapter == "sentguard":
        from streamguard_bench.guards import SentGuardAdapter

        return SentGuardAdapter(
            model["repository"],
            max_new_tokens=model["max_new_tokens"],
            quantization=model.get("quantization"),
            **common,
        )
    raise ValueError(f"Unknown guard adapter: {adapter}")
