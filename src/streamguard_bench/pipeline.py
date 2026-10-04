"""One entry point for running the benchmark from a script or from a notebook."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import yaml

from streamguard_bench.experiments import (
    PROFILES,
    RESULT_FILES,
    ExperimentRun,
    load_experiment_run,
    run_experiment,
    sweep_decision_rules,
)

# The default guard is also the baseline that the other guards are compared with.
DEFAULT_CONFIG = Path("configs/qwen3guard_stream_0_6b.yaml")
# Model settings that change the decisions and are recorded with a run when present.
INFERENCE_SETTINGS = ("max_new_tokens", "quantization")


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
    if not _same_model(saved, config.get("model", {})):
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


def _same_model(saved: dict[str, Any], model: dict[str, Any]) -> bool:
    """Whether the saved decisions were made by the model the config describes."""
    if "repository" in model and saved.get("model_id") != model["repository"]:
        return False
    if model.get("revision") and saved.get("model_revision") != model["revision"]:
        return False
    if saved.get("threshold") != model.get("threshold"):
        return False
    settings = saved.get("inference_settings", {})
    return all(settings.get(key) == model.get(key) for key in INFERENCE_SETTINGS)


def _saved_guard(config: dict[str, Any], directory: Path) -> Any | None:
    """Stand-in for the guard when every trace of the run is already scored.

    Rebuilding the tables under another stop rule or mode list then needs no model weights.
    """
    metadata_path = directory / "run_metadata.json"
    ids_path = directory / RESULT_FILES["ids"]
    if not metadata_path.exists() or not ids_path.exists():
        return None
    saved = json.loads(metadata_path.read_text())
    scored = all(
        (directory / "checkpoints" / f"{trace_id}.json").exists()
        for trace_id in json.loads(ids_path.read_text())
    )
    if not scored or not _same_model(saved, config.get("model", {})):
        return None
    return SimpleNamespace(
        model_id=saved["model_id"],
        model_revision=saved.get("model_revision"),
        tokenizer_id=saved["tokenizer_id"],
        tokenizer_revision=saved.get("tokenizer_revision"),
        device=saved["device"],
        threshold=saved.get("threshold"),
        inference_settings=saved.get("inference_settings"),
        close=lambda: None,
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
    if guard is None and config["runtime"]["resume"]:
        guard = _saved_guard(config, output_dir / profile)
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
        checkpoint_scoring=experiment.get("checkpoint_scoring"),
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
        return self.results_for(self.policy)

    @property
    def policies(self) -> tuple[str, ...]:
        return tuple(self.config["experiment"]["policies"])

    @property
    def modes(self) -> tuple[str, ...]:
        return tuple(self.config["experiment"]["modes"])

    @property
    def checkpoint_scoring(self) -> bool:
        """True for a guard asked only at buffer checkpoints: its errors depend on the mode."""
        return bool(self.config["experiment"].get("checkpoint_scoring", False))

    def results_for(self, policy: str, mode: str | None = None) -> pd.DataFrame:
        """Replay results under one policy, optionally in one buffer mode."""
        frame = self.run.intervention_results
        frame = frame[frame["policy"] == policy]
        if mode is not None:
            frame = frame[frame["mode"] == mode]
        return frame.assign(model=self.name)


def load_saved_runs(
    *, profile: str, root: str | Path = ".", config_dir: str | Path = "configs"
) -> list[SavedRun]:
    """Load every configured guard that already has a complete saved run.

    The baseline (`DEFAULT_CONFIG`) comes first, the rest follow in config file name order.
    The primary policy of a guard is the first one its config lists.
    """
    root = Path(root)
    runs = []
    paths = sorted(
        (root / config_dir).glob("*.yaml"),
        key=lambda path: (path.name != DEFAULT_CONFIG.name, path.name),
    )
    for path in paths:
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


def sweep_saved_run(
    item: SavedRun,
    *,
    dataset: pd.DataFrame,
    rules: list[dict[str, Any]],
    mode: str,
    profile: str,
    root: str | Path = ".",
) -> pd.DataFrame:
    """Replay one guard under many stop rules and every policy it has; one row per rule.

    A probability threshold replaces the labels of the guard, so it is tried under the
    primary policy only. `default` marks the rule the guard is configured with.
    """
    experiment = item.config["experiment"]
    label_rules = [rule for rule in rules if rule.get("threshold") is None]
    grids = []
    for policy in item.policies:
        grid = sweep_decision_rules(
            dataset=dataset,
            output_dir=Path(root) / experiment["output_dir"],
            profile=profile,
            policy=policy,
            rules=rules if policy == item.policy else label_rules,
            mode=mode,
            max_sentence_tokens=experiment["max_sentence_tokens"],
            checkpoint_scoring=item.checkpoint_scoring,
        )
        grid["default"] = (
            (policy == item.policy)
            & grid["threshold"].isna()
            & (grid["trigger_count"] == experiment.get("trigger_count", 1))
            & (grid["trigger_mode"] == experiment.get("trigger_mode", "cumulative"))
        )
        grids.append(grid.assign(model=item.name, policy=policy))
    return pd.concat(grids, ignore_index=True)


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
