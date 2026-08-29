"""Reproducible Qwen3Guard baseline runner used by the experiment notebook."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from streamguard_bench.streaming import (
    DEFAULT_MODES,
    DEFAULT_POLICIES,
    BufferMode,
    GuardTrace,
    SafetyPolicy,
    simulate_all,
)

DATASET_REPOSITORY = "Vovarus12go/streamsafe-500-boundaries"
PROFILES = {
    "smoke": {"limit": 10, "safe": 5, "unsafe": 5},
    "full": {"limit": None, "safe": 250, "unsafe": 250},
}


@dataclass(frozen=True)
class ExperimentRun:
    """Notebook-friendly result of a completed or partially completed run."""

    results: pd.DataFrame
    token_decisions: pd.DataFrame
    errors: pd.DataFrame
    selected_trace_ids: tuple[str, ...]
    output_dir: Path


def load_boundary_dataset(
    repository: str = DATASET_REPOSITORY,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Download the published 500-trace release and return one DataFrame."""

    from datasets import load_dataset

    kwargs: dict[str, Any] = {}
    if revision:
        kwargs["revision"] = revision
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)
    return _dataset_to_frame(load_dataset(repository, **kwargs))


def load_experiment_run(
    output_dir: str | Path = "data/interim/qwen3guard_baseline",
    *,
    profile: str = "smoke",
) -> ExperimentRun:
    """Load a previously checkpointed run without loading the dataset or model."""

    run_dir = Path(output_dir) / profile
    required = {
        "results": run_dir / "results.csv",
        "token_decisions": run_dir / "token_decisions.parquet",
        "errors": run_dir / "errors.csv",
        "selected_trace_ids": run_dir / "selected_trace_ids.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Saved experiment is incomplete. Missing files: " + ", ".join(missing)
        )
    selected_ids = tuple(
        str(item)
        for item in json.loads(required["selected_trace_ids"].read_text(encoding="utf-8"))
    )
    return ExperimentRun(
        results=pd.read_csv(required["results"]),
        token_decisions=pd.read_parquet(required["token_decisions"]),
        errors=pd.read_csv(required["errors"]),
        selected_trace_ids=selected_ids,
        output_dir=run_dir,
    )


def select_profile_traces(
    dataset: Any,
    profile: str,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Select the fixed 5-safe/5-unsafe smoke set or all 500 release traces."""

    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; expected one of {sorted(PROFILES)}")
    frame = _dataset_to_frame(dataset)
    _validate_columns(frame)
    frame = frame.sort_values("trace_id", kind="stable").reset_index(drop=True)
    if frame["trace_id"].duplicated().any():
        raise ValueError("Dataset contains duplicate trace_id values")

    expected = PROFILES[profile]
    if profile == "full":
        counts = frame["label"].value_counts().to_dict()
        if len(frame) != 500 or counts.get("safe") != 250 or counts.get("unsafe") != 250:
            raise ValueError(f"Full profile requires 500 balanced traces; received {counts}")
        return frame

    safe = _stable_sample(frame[frame["label"] == "safe"], expected["safe"], seed)
    unsafe = _select_unsafe_smoke(frame[frame["label"] == "unsafe"], seed=seed)
    selected = pd.concat([safe, unsafe], ignore_index=True)
    return selected.sort_values("trace_id", kind="stable").reset_index(drop=True)


def run_experiment(
    *,
    dataset: Any,
    guard: Any,
    profile: str = "smoke",
    modes: list[str] | tuple[str, ...] = tuple(item.value for item in DEFAULT_MODES),
    policies: list[str] | tuple[str, ...] = tuple(
        item.value for item in DEFAULT_POLICIES
    ),
    output_dir: str | Path = "data/interim/qwen3guard_baseline",
    resume: bool = True,
    seed: int = 42,
    max_sentence_tokens: int = 128,
    dataset_revision: str | None = None,
) -> ExperimentRun:
    """Run one model pass per trace, checkpoint it, then replay all buffer policies."""

    selected = select_profile_traces(dataset, profile, seed=seed)
    selected_modes = tuple(BufferMode(item) for item in modes)
    selected_policies = tuple(SafetyPolicy(item) for item in policies)
    run_dir = Path(output_dir) / profile
    traces_dir = run_dir / "token_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = tuple(selected["trace_id"].astype(str))
    metadata = _run_metadata(
        guard=guard,
        profile=profile,
        selected_ids=selected_ids,
        modes=selected_modes,
        policies=selected_policies,
        seed=seed,
        max_sentence_tokens=max_sentence_tokens,
        dataset_revision=dataset_revision,
    )
    _validate_or_write_metadata(run_dir / "run_metadata.json", metadata, resume=resume)
    _write_json(run_dir / "selected_trace_ids.json", list(selected_ids))

    traces: dict[str, GuardTrace] = {}
    errors: list[dict[str, str]] = []
    for row in selected.to_dict(orient="records"):
        trace_id = str(row["trace_id"])
        checkpoint = traces_dir / f"{trace_id}.json"
        if resume and checkpoint.exists():
            try:
                trace = GuardTrace.from_dict(json.loads(checkpoint.read_text(encoding="utf-8")))
                if trace.trace_id == trace_id and _complete_trace(trace):
                    traces[trace_id] = trace
                    continue
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass

        try:
            trace = _score_trace(guard, row)
            _write_json(checkpoint, trace.to_dict())
            traces[trace_id] = trace
        except Exception as error:  # noqa: BLE001 - every failed trace must be logged
            errors.append(
                {
                    "trace_id": trace_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        finally:
            guard.close()

    results = _simulate_results(
        selected,
        traces,
        errors,
        modes=selected_modes,
        policies=selected_policies,
        max_sentence_tokens=max_sentence_tokens,
    )
    token_decisions = _token_decision_frame(traces)
    errors_frame = pd.DataFrame(errors, columns=["trace_id", "error_type", "error"])
    results.to_csv(run_dir / "results.csv", index=False)
    token_decisions.to_parquet(run_dir / "token_decisions.parquet", index=False)
    errors_frame.to_csv(run_dir / "errors.csv", index=False)
    return ExperimentRun(
        results=results,
        token_decisions=token_decisions,
        errors=errors_frame,
        selected_trace_ids=selected_ids,
        output_dir=run_dir,
    )


def _score_trace(guard: Any, row: dict[str, Any]) -> GuardTrace:
    response = str(row["response"])
    tokenized = guard.tokenize_response(response)
    decisions = []
    started = time.perf_counter()
    guard.start(str(row["prompt"]))
    for index, (token_id, end_character) in enumerate(
        zip(tokenized.token_ids, tokenized.end_characters, strict=True),
        start=1,
    ):
        decisions.append(
            guard.score_token(
                token_id=token_id,
                token_index=index,
                end_character=end_character,
            )
        )
    total_latency_ms = (time.perf_counter() - started) * 1000
    return GuardTrace(
        trace_id=str(row["trace_id"]),
        decisions=tuple(decisions),
        model_id=str(guard.model_id),
        model_revision=getattr(guard, "model_revision", None),
        tokenizer_id=str(guard.tokenizer_id),
        tokenizer_revision=getattr(guard, "tokenizer_revision", None),
        device=str(guard.device),
        total_latency_ms=total_latency_ms,
    )


def _simulate_results(
    selected: pd.DataFrame,
    traces: dict[str, GuardTrace],
    errors: list[dict[str, str]],
    *,
    modes: tuple[BufferMode, ...],
    policies: tuple[SafetyPolicy, ...],
    max_sentence_tokens: int,
) -> pd.DataFrame:
    error_map = {item["trace_id"]: item["error"] for item in errors}
    records: list[dict[str, Any]] = []
    for row in selected.to_dict(orient="records"):
        trace_id = str(row["trace_id"])
        trace = traces.get(trace_id)
        if trace is None:
            records.extend(_error_result_rows(row, modes, policies, error_map.get(trace_id)))
            continue
        lower = _optional_int(row.get("last_safe_end_qwen_token"))
        upper = _optional_int(row.get("first_unsafe_end_qwen_token"))
        categories = _categories(row.get("harm_categories"))
        results = simulate_all(
            trace=trace,
            response=str(row["response"]),
            source_split=str(row["source_split"]),
            ground_truth_label=str(row["label"]),
            harm_categories=categories,
            modes=modes,
            policies=policies,
            onset_lower_token=lower,
            onset_upper_token=upper,
            max_sentence_tokens=max_sentence_tokens,
        )
        records.extend(item.to_dict() for item in results)
    columns = [
        "trace_id",
        "source_split",
        "ground_truth_label",
        "harm_categories",
        "mode",
        "policy",
        "blocked",
        "signal_token",
        "intervention_token",
        "released_tokens",
        "leakage_min",
        "leakage_max",
        "detection_delay_min",
        "detection_delay_max",
        "checks",
        "guard_time_ms",
        "error",
    ]
    return pd.DataFrame(records, columns=columns)


def _error_result_rows(
    row: dict[str, Any],
    modes: tuple[BufferMode, ...],
    policies: tuple[SafetyPolicy, ...],
    error: str | None,
) -> list[dict[str, Any]]:
    return [
        {
            "trace_id": str(row["trace_id"]),
            "source_split": str(row["source_split"]),
            "ground_truth_label": str(row["label"]),
            "harm_categories": _categories(row.get("harm_categories")),
            "mode": mode.value,
            "policy": policy.value,
            "blocked": False,
            "signal_token": None,
            "intervention_token": None,
            "released_tokens": 0,
            "leakage_min": 0,
            "leakage_max": 0,
            "detection_delay_min": None,
            "detection_delay_max": None,
            "checks": 0,
            "guard_time_ms": 0.0,
            "error": error or "Trace did not produce a complete checkpoint",
        }
        for mode in modes
        for policy in policies
    ]


def _token_decision_frame(traces: dict[str, GuardTrace]) -> pd.DataFrame:
    rows = []
    for trace_id, trace in sorted(traces.items()):
        for decision in trace.decisions:
            rows.append({"trace_id": trace_id, **asdict(decision)})
    return pd.DataFrame(rows)


def _dataset_to_frame(dataset: Any) -> pd.DataFrame:
    if isinstance(dataset, pd.DataFrame):
        return dataset.copy()
    if hasattr(dataset, "to_pandas"):
        return dataset.to_pandas()
    if isinstance(dataset, dict):
        frames = []
        for split, value in dataset.items():
            frame = _dataset_to_frame(value)
            if "source_split" not in frame.columns:
                frame["source_split"] = "val" if split == "validation" else split
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(dataset)


def _validate_columns(frame: pd.DataFrame) -> None:
    required = {
        "prompt",
        "response",
        "label",
        "harm_categories",
        "trace_id",
        "source_split",
        "last_safe_end_qwen_token",
        "first_unsafe_end_qwen_token",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    labels = set(frame["label"].dropna().astype(str))
    if not labels <= {"safe", "unsafe"}:
        raise ValueError(f"Dataset contains unsupported labels: {sorted(labels)}")


def _stable_sample(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if len(frame) < count:
        raise ValueError(f"Requested {count} rows, but only {len(frame)} are available")
    ordered = frame.sort_values("trace_id", kind="stable")
    return ordered.sample(n=count, random_state=seed, replace=False)


def _select_unsafe_smoke(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    candidates = frame.copy()
    candidates["_onset_bucket"] = candidates.apply(_onset_bucket, axis=1)
    targets = {"early": 2, "middle": 1, "late": 2}
    selected_indices: list[int] = []
    seen_categories: set[str] = set()
    rng = random.Random(seed)
    for bucket, target in targets.items():
        pool = list(candidates[candidates["_onset_bucket"] == bucket].index)
        rng.shuffle(pool)
        for _ in range(target):
            if not pool:
                break
            best = max(
                pool,
                key=lambda index: len(
                    set(_categories(candidates.at[index, "harm_categories"]))
                    - seen_categories
                ),
            )
            pool.remove(best)
            selected_indices.append(best)
            seen_categories.update(_categories(candidates.at[best, "harm_categories"]))

    if len(selected_indices) < 5:
        remaining = [index for index in candidates.index if index not in selected_indices]
        rng.shuffle(remaining)
        selected_indices.extend(remaining[: 5 - len(selected_indices)])
    if len(selected_indices) != 5:
        raise ValueError("Could not select five unsafe smoke traces")
    return candidates.loc[selected_indices].drop(columns="_onset_bucket")


def _onset_bucket(row: pd.Series) -> str:
    length = max(1, len(str(row["response"])))
    upper = _optional_int(row.get("first_unsafe_end_character"))
    fraction = 1.0 if upper is None else upper / length
    if fraction <= 0.33:
        return "early"
    if fraction <= 0.66:
        return "middle"
    return "late"


def _run_metadata(
    *,
    guard: Any,
    profile: str,
    selected_ids: tuple[str, ...],
    modes: tuple[BufferMode, ...],
    policies: tuple[SafetyPolicy, ...],
    seed: int,
    max_sentence_tokens: int,
    dataset_revision: str | None,
) -> dict[str, Any]:
    metadata = {
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_revision": dataset_revision,
        "profile": profile,
        "selected_trace_ids_sha256": hashlib.sha256(
            "\n".join(selected_ids).encode("utf-8")
        ).hexdigest(),
        "model_id": str(guard.model_id),
        "model_revision": getattr(guard, "model_revision", None),
        "tokenizer_id": str(guard.tokenizer_id),
        "tokenizer_revision": getattr(guard, "tokenizer_revision", None),
        "device": str(guard.device),
        "modes": [item.value for item in modes],
        "policies": [item.value for item in policies],
        "seed": seed,
        "max_sentence_tokens": max_sentence_tokens,
    }
    canonical = json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode("utf-8")
    metadata["configuration_sha256"] = hashlib.sha256(canonical).hexdigest()
    return metadata


def _validate_or_write_metadata(path: Path, metadata: dict[str, Any], *, resume: bool) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise ValueError(
                "Existing run uses a different configuration; choose another output directory"
            )
        if not resume:
            raise FileExistsError("Run directory already exists and resume=False")
        return
    _write_json(path, metadata)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _complete_trace(trace: GuardTrace) -> bool:
    return all(
        item.token_index == expected
        for expected, item in enumerate(trace.decisions, start=1)
    )


def _categories(value: Any) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ()
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            if isinstance(decoded, list):
                value = decoded
            else:
                return (value,)
        except json.JSONDecodeError:
            return (value,)
    return tuple(str(item) for item in value)


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)
