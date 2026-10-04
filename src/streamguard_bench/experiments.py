"""Checkpointed prompt scoring, response scoring, and buffer replay."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from streamguard_bench.contracts import GuardTrace
from streamguard_bench.data import REPOSITORY
from streamguard_bench.metrics import compute_response_policy_metrics
from streamguard_bench.streaming import (
    DEFAULT_MODES,
    DEFAULT_POLICIES,
    BufferMode,
    SafetyPolicy,
    relabel_trace,
    simulate_all,
)

PROFILES = {"smoke2": 2, "full": 210}
SCHEMA_VERSION = 2
RESULT_FILES = {
    "prompt": "prompt_decisions.parquet",
    "token": "token_decisions.parquet",
    "intervention": "intervention_results.parquet",
    "errors": "errors.csv",
    "ids": "selected_trace_ids.json",
}


@dataclass(frozen=True)
class ExperimentRun:
    prompt_decisions: pd.DataFrame
    token_decisions: pd.DataFrame
    intervention_results: pd.DataFrame
    errors: pd.DataFrame
    selected_trace_ids: tuple[str, ...]
    output_dir: Path


def select_profile_traces(dataset: Any, profile: str) -> pd.DataFrame:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    frame = _frame(dataset).sort_values("trace_id", kind="stable").reset_index(drop=True)
    _validate_columns(frame)
    if frame["trace_id"].duplicated().any():
        raise ValueError("Dataset contains duplicate trace_id values")
    if profile == "full":
        if len(frame) != 210:
            raise ValueError(f"Full profile requires 210 traces; received {len(frame)}")
        return frame
    safe = frame[frame["response_label"] == "safe"].head(1)
    unsafe = frame[
        (frame["response_label"] == "unsafe") & (frame["unsafe_start_character"] > 0)
    ].head(1)
    if len(safe) != 1 or len(unsafe) != 1:
        raise ValueError("smoke2 requires one safe and one unsafe trace with a safe prefix")
    return pd.concat([safe, unsafe], ignore_index=True).sort_values("trace_id", kind="stable")


def run_experiment(
    *,
    dataset: Any,
    guard: Any,
    profile: str = "smoke2",
    modes: tuple[str, ...] | list[str] = tuple(item.value for item in DEFAULT_MODES),
    policies: tuple[str, ...] | list[str] = tuple(item.value for item in DEFAULT_POLICIES),
    output_dir: str | Path = "data/interim/qwen3guard_stream_0_6b",
    resume: bool = True,
    seed: int = 42,
    max_sentence_tokens: int = 128,
    dataset_revision: str | None = None,
    trigger_count: int = 1,
    trigger_mode: str = "cumulative",
    trigger_window: int | None = None,
    checkpoint_scoring: bool | None = None,
) -> ExperimentRun:
    """Score the selected traces and replay them under the configured modes and stop rule.

    `checkpoint_scoring` says that the guard was asked only at buffer checkpoints, so each
    mode is replayed on the decisions made at its own checkpoints. By default the guard
    declares it itself.
    """
    if checkpoint_scoring is None:
        checkpoint_scoring = bool(getattr(guard, "checkpoint_scoring", False))
    selected = select_profile_traces(dataset, profile)
    selected_modes = tuple(BufferMode(item) for item in modes)
    selected_policies = tuple(SafetyPolicy(item) for item in policies)
    run_dir = Path(output_dir) / profile
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = tuple(selected["trace_id"].astype(str))
    metadata = _metadata(
        guard,
        profile,
        selected_ids,
        selected_modes,
        selected_policies,
        seed,
        max_sentence_tokens,
        dataset_revision,
        trigger_count,
        trigger_mode,
        trigger_window,
        checkpoint_scoring,
    )
    _validate_or_write(run_dir / "run_metadata.json", metadata, resume)
    _write_json(run_dir / "selected_trace_ids.json", list(selected_ids))
    traces: dict[str, GuardTrace] = {}
    errors: list[dict[str, str]] = []
    for row in selected.to_dict(orient="records"):
        trace_id = str(row["trace_id"])
        checkpoint = checkpoint_dir / f"{trace_id}.json"
        if resume and checkpoint.exists():
            try:
                traces[trace_id] = GuardTrace.from_dict(json.loads(checkpoint.read_text()))
                continue
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
        try:
            trace = _score_trace(guard, row)
            _write_json(checkpoint, trace.to_dict())
            traces[trace_id] = trace
        except Exception as error:  # noqa: BLE001
            errors.append(
                {"trace_id": trace_id, "error_type": type(error).__name__, "error": str(error)}
            )
        finally:
            guard.close()
    prompt_rows = [trace.prompt_decision.to_dict() for trace in traces.values()]
    ground_truth = selected.set_index("trace_id")["query_label"].astype(str).to_dict()
    prompt_rows.extend(
        {
            "trace_id": item["trace_id"],
            "query_ground_truth": ground_truth[item["trace_id"]],
            "risk_label": None,
            "risk_categories": (),
            "confidence": None,
            "latency_ms": 0.0,
            "error": item["error"],
        }
        for item in errors
    )
    prompt_frame = pd.DataFrame(prompt_rows)
    token_frame = pd.DataFrame(
        [
            {"trace_id": trace_id, **asdict(decision)}
            for trace_id, trace in sorted(traces.items())
            for decision in trace.decisions
        ]
    )
    intervention_frame = _replay(
        selected,
        traces,
        errors,
        selected_modes,
        selected_policies,
        max_sentence_tokens,
        trigger_count,
        trigger_mode,
        trigger_window,
        checkpoint_scoring=checkpoint_scoring,
    )
    error_frame = pd.DataFrame(errors, columns=["trace_id", "error_type", "error"])
    prompt_frame.to_parquet(run_dir / "prompt_decisions.parquet", index=False)
    token_frame.to_parquet(run_dir / "token_decisions.parquet", index=False)
    intervention_frame.to_parquet(run_dir / "intervention_results.parquet", index=False)
    error_frame.to_csv(run_dir / "errors.csv", index=False)
    return ExperimentRun(
        prompt_frame, token_frame, intervention_frame, error_frame, selected_ids, run_dir
    )


def load_experiment_run(
    output_dir: str | Path = "data/interim/qwen3guard_stream_0_6b", *, profile: str = "smoke2"
) -> ExperimentRun:
    run_dir = Path(output_dir) / profile
    paths = {key: run_dir / name for key, name in RESULT_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Incomplete saved run: " + ", ".join(missing))
    return ExperimentRun(
        pd.read_parquet(paths["prompt"]),
        pd.read_parquet(paths["token"]),
        pd.read_parquet(paths["intervention"]),
        pd.read_csv(paths["errors"]),
        tuple(json.loads(paths["ids"].read_text())),
        run_dir,
    )


def replay_saved_traces(
    *,
    dataset: Any,
    output_dir: str | Path,
    profile: str,
    modes: tuple[str, ...] | list[str] = tuple(item.value for item in DEFAULT_MODES),
    policies: tuple[str, ...] | list[str] = tuple(item.value for item in DEFAULT_POLICIES),
    trigger_count: int = 1,
    threshold: float | None = None,
    max_sentence_tokens: int = 128,
    trigger_mode: str = "cumulative",
    trigger_window: int | None = None,
    include_controversial: bool = False,
    checkpoint_scoring: bool = False,
    warmup_tokens: int = 0,
) -> pd.DataFrame:
    """Replay checkpointed traces under another decision rule without scoring them again.

    `checkpoint_scoring` is for guards that were scored only at buffer checkpoints: each mode
    then sees only the decisions made at its own checkpoints. `warmup_tokens` additionally
    skips the checks within the first tokens, which are then released unchecked.
    """

    selected = select_profile_traces(dataset, profile)
    traces = _apply_threshold(
        _load_traces(selected, output_dir, profile), threshold, include_controversial
    )
    return _replay(
        selected,
        traces,
        [],
        tuple(BufferMode(item) for item in modes),
        tuple(SafetyPolicy(item) for item in policies),
        max_sentence_tokens,
        trigger_count,
        trigger_mode,
        trigger_window,
        checkpoint_scoring,
        warmup_tokens,
    )


def sweep_decision_rules(
    *,
    dataset: Any,
    output_dir: str | Path,
    profile: str,
    policy: str,
    rules: list[dict[str, Any]],
    mode: str = "token",
    max_sentence_tokens: int = 128,
    checkpoint_scoring: bool = False,
) -> pd.DataFrame:
    """Summarise saved traces under many stop rules; one row per rule.

    Probability thresholds are omitted for label-only guards, which cannot be
    re-thresholded. A rule is a dict with `threshold`, `include_controversial`, `trigger_count`,
    `trigger_mode` and `trigger_window`. Without a threshold the stored labels are used.
    """
    selected = select_profile_traces(dataset, profile)
    saved = _load_traces(selected, output_dir, profile)
    has_scores = bool(saved) and all(
        any(item.unsafe_score is not None for item in trace.decisions) for trace in saved.values()
    )
    rows = []
    for rule in rules:
        if rule.get("threshold") is not None and not has_scores:
            continue
        traces = _apply_threshold(
            saved, rule.get("threshold"), rule.get("include_controversial", False)
        )
        results = _replay(
            selected,
            traces,
            [],
            (BufferMode(mode),),
            (SafetyPolicy(policy),),
            max_sentence_tokens,
            rule.get("trigger_count", 1),
            rule.get("trigger_mode", "cumulative"),
            rule.get("trigger_window"),
            checkpoint_scoring,
        )
        metrics = compute_response_policy_metrics(results).iloc[0]
        harmful = results[results["response_ground_truth"] == "unsafe"]
        rows.append(
            {
                "threshold": rule.get("threshold"),
                "trigger_count": rule.get("trigger_count", 1),
                "trigger_mode": rule.get("trigger_mode", "cumulative"),
                "trigger_window": rule.get("trigger_window"),
                "fp": int(metrics["fp"]),
                "fn": int(metrics["fn"]),
                "false_positive_rate": metrics["false_positive_rate"],
                "false_negative_rate": metrics["false_negative_rate"],
                "leakage_words_mean": harmful["leakage_words"].mean(),
                "detection_delay_median": harmful["detection_delay_tokens"].median(),
                "premature_block_rate": harmful["premature_block"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _load_traces(selected: pd.DataFrame, output_dir: str | Path, profile: str) -> dict:
    checkpoint_dir = Path(output_dir) / profile / "checkpoints"
    traces = {}
    for trace_id in selected["trace_id"].astype(str):
        checkpoint = checkpoint_dir / f"{trace_id}.json"
        if checkpoint.exists():
            traces[trace_id] = GuardTrace.from_dict(json.loads(checkpoint.read_text()))
    return traces


def _apply_threshold(traces: dict, threshold: float | None, include_controversial: bool) -> dict:
    if threshold is None:
        return traces
    return {
        trace_id: relabel_trace(
            trace, threshold=threshold, include_controversial=include_controversial
        )
        for trace_id, trace in traces.items()
    }


def _score_trace(guard: Any, row: dict[str, Any]) -> GuardTrace:
    tokenized = guard.tokenize_response(str(row["response"]))
    if len(tokenized.token_ids) != int(row["response_token_count"]):
        # unsafe_start_token was computed with the dataset tokenizer; a guard that
        # tokenizes differently would be judged against the wrong onset.
        raise ValueError("Guard tokenization differs from the prepared dataset")
    started = time.perf_counter()
    prompt = guard.start(
        str(row["query"]), trace_id=str(row["trace_id"]), query_ground_truth=str(row["query_label"])
    )
    decisions = tuple(
        guard.score_token(token_id=token_id, token_index=index, end_character=end_character)
        for index, (token_id, end_character) in enumerate(
            zip(tokenized.token_ids, tokenized.end_characters, strict=True), start=1
        )
    )
    return GuardTrace(
        trace_id=str(row["trace_id"]),
        prompt_decision=prompt,
        decisions=decisions,
        model_id=str(guard.model_id),
        model_revision=getattr(guard, "model_revision", None),
        tokenizer_id=str(guard.tokenizer_id),
        tokenizer_revision=getattr(guard, "tokenizer_revision", None),
        device=str(guard.device),
        total_latency_ms=(time.perf_counter() - started) * 1000,
    )


def _replay(
    selected,
    traces,
    errors,
    modes,
    policies,
    max_sentence_tokens,
    trigger_count=1,
    trigger_mode="cumulative",
    trigger_window=None,
    checkpoint_scoring=False,
    warmup_tokens=0,
):
    records = []
    by_id = {str(row["trace_id"]): row for row in selected.to_dict(orient="records")}
    for trace_id, trace in traces.items():
        row = by_id[trace_id]
        records.extend(
            item.to_dict()
            for item in simulate_all(
                trace=trace,
                response=str(row["response"]),
                response_ground_truth=str(row["response_label"]),
                unsafe_start_token=int(row["unsafe_start_token"]),
                response_token_count=int(row["response_token_count"]),
                modes=modes,
                policies=policies,
                max_sentence_tokens=max_sentence_tokens,
                trigger_counts=(trigger_count,),
                trigger_mode=trigger_mode,
                trigger_window=trigger_window,
                unsafe_start_character=int(row["unsafe_start_character"]),
                checkpoint_scoring=checkpoint_scoring,
                warmup_tokens=warmup_tokens,
            )
        )
    for failure in errors:
        row = by_id[failure["trace_id"]]
        records.extend(
            {
                "trace_id": failure["trace_id"],
                "response_ground_truth": str(row["response_label"]),
                "mode": mode.value,
                "policy": policy.value,
                "blocked": False,
                "signal_token": None,
                "intervention_token": None,
                "released_tokens": 0,
                "safe_withheld_tokens": None,
                "leakage_tokens": None,
                "normalized_leakage": None,
                "signal_offset_tokens": None,
                "premature_block": False,
                "early_lead_tokens": None,
                "detection_delay_tokens": None,
                "post_signal_buffer_delay_tokens": None,
                "checks": 0,
                "guard_time_ms": 0.0,
                "error": failure["error"],
                "trigger_count": trigger_count,
                "leakage_characters": None,
                "leakage_words": None,
                "trigger_mode": trigger_mode,
                "trigger_window": trigger_window,
            }
            for mode in modes
            for policy in policies
        )
    return pd.DataFrame(records)


def _frame(dataset: Any) -> pd.DataFrame:
    if isinstance(dataset, pd.DataFrame):
        return dataset.copy()
    if hasattr(dataset, "to_pandas"):
        return dataset.to_pandas()
    return pd.DataFrame(dataset)


def _validate_columns(frame: pd.DataFrame) -> None:
    required = {
        "trace_id",
        "query",
        "response",
        "query_label",
        "response_label",
        "unsafe_start_character",
        "unsafe_start_token",
        "response_token_count",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def _metadata(
    guard,
    profile,
    ids,
    modes,
    policies,
    seed,
    max_sentence_tokens,
    dataset_revision,
    trigger_count,
    trigger_mode="cumulative",
    trigger_window=None,
    checkpoint_scoring=False,
):
    value = {
        "schema_version": SCHEMA_VERSION,
        "dataset_repository": REPOSITORY,
        "dataset_revision": dataset_revision,
        "profile": profile,
        "selected_trace_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "model_id": str(guard.model_id),
        "model_revision": getattr(guard, "model_revision", None),
        "tokenizer_id": str(guard.tokenizer_id),
        "tokenizer_revision": getattr(guard, "tokenizer_revision", None),
        "device": str(guard.device),
        "modes": [item.value for item in modes],
        "policies": [
            {"name": item.value, "blocking_labels": sorted(item.blocking_labels)}
            for item in policies
        ],
        "seed": seed,
        "max_sentence_tokens": max_sentence_tokens,
    }
    # Recorded only when set, so runs made before these options stay resumable.
    if trigger_count != 1:
        value["trigger_count"] = trigger_count
    if trigger_mode != "cumulative":
        value["trigger_mode"] = trigger_mode
    if trigger_window is not None:
        value["trigger_window"] = trigger_window
    if getattr(guard, "threshold", None) is not None:
        value["threshold"] = guard.threshold
    if checkpoint_scoring:
        value["checkpoint_scoring"] = True
    if getattr(guard, "inference_settings", None) is not None:
        value["inference_settings"] = guard.inference_settings
    value["configuration_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True).encode()
    ).hexdigest()
    return value


# Settings that only affect the replay. Checkpoints hold per-token decisions, which do not
# depend on them, so a run may be replayed under another stop rule without rescoring.
REPLAY_SETTINGS = frozenset(
    {
        "modes",
        "policies",
        "max_sentence_tokens",
        "trigger_count",
        "trigger_mode",
        "trigger_window",
        "checkpoint_scoring",
        "configuration_sha256",
    }
)


def _validate_or_write(path: Path, metadata: dict[str, Any], resume: bool) -> None:
    if path.exists():
        saved = json.loads(path.read_text())
        if _scoring_identity(saved) != _scoring_identity(metadata):
            raise ValueError("Existing run uses a different configuration or checkpoint schema")
        if not resume:
            raise FileExistsError("Run directory exists and resume=False")
    _write_json(path, metadata)


def _scoring_identity(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key not in REPLAY_SETTINGS}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
