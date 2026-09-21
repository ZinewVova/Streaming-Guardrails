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
from streamguard_bench.streaming import (
    DEFAULT_MODES,
    DEFAULT_POLICIES,
    BufferMode,
    SafetyPolicy,
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
    output_dir: str | Path = "data/interim/qwen3guard",
    resume: bool = True,
    seed: int = 42,
    max_sentence_tokens: int = 128,
    dataset_revision: str | None = None,
) -> ExperimentRun:
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
    output_dir: str | Path = "data/interim/qwen3guard", *, profile: str = "smoke2"
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


def _score_trace(guard: Any, row: dict[str, Any]) -> GuardTrace:
    tokenized = guard.tokenize_response(str(row["response"]))
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


def _replay(selected, traces, errors, modes, policies, max_sentence_tokens):
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


def _metadata(guard, profile, ids, modes, policies, seed, max_sentence_tokens, dataset_revision):
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
    value["configuration_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True).encode()
    ).hexdigest()
    return value


def _validate_or_write(path: Path, metadata: dict[str, Any], resume: bool) -> None:
    if path.exists():
        if json.loads(path.read_text()) != metadata:
            raise ValueError("Existing run uses a different configuration or checkpoint schema")
        if not resume:
            raise FileExistsError("Run directory exists and resume=False")
    else:
        _write_json(path, metadata)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
