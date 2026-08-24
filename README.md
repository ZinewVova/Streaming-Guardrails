# Streaming Guardrails Benchmark

This repository studies safety moderation for streaming generation by large language
models. In a streaming interface, a response is shown token by token or in small chunks,
so a guardrail must make decisions before the complete response is available. The project
measures how moderation granularity affects false blocks, missed unsafe content, leakage
before blocking, and computational delay.

## Current status

Stage 1 is complete: the repository provides reproducible ingestion, raw-data validation,
and exploratory analysis of
[StreamSafe](https://huggingface.co/datasets/Solitude0630/StreamSafe) at revision
`16d0ff1f42e980bb99bd36125583361b15c664e3`. The source schemas are preserved; creation of
the final normalized benchmark subset remains follow-up work.

## Planned data flow

```text
StreamSafe
    ↓
schema inspection and validation
    ↓
model-independent streaming traces
    ↓
token / fixed-chunk / sentence policies
    ↓
guard model
    ↓
leakage, error, and latency metrics
```

## Quick start

Python 3.11 is required.

```bash
python -m pip install -e ".[analysis,dev]"
python scripts/download_streamsafe.py
python scripts/run_data_validation.py
jupyter notebook notebooks/01_streamsafe_eda.ipynb
pytest
```

The first download pins an immutable Hugging Face dataset revision in
`configs/data_streamsafe.yaml`. Raw files are cached under `data/raw/` and are not tracked
by Git.

## Repository layout

- `configs/`: versioned data and experiment settings.
- `notebooks/`: explanatory analysis, not reusable transformation logic.
- `scripts/`: thin command-line entry points.
- `src/streamguard_bench/data/`: dataset loading, inspection, and validation.
- `src/streamguard_bench/guards/`: stable interface for guard-model adapters.
- `src/streamguard_bench/streaming/`: stable interface for buffering policies.
- `src/streamguard_bench/metrics/`: stable interface for benchmark metrics.
- `data/fixtures/`: synthetic records used by offline tests.
- `reports/`: public aggregate tables, figures, and written findings.
- `tests/`: tests that do not require network access.

## Reproducing the data inspection

Download and write a provenance manifest:

```bash
python scripts/download_streamsafe.py --config configs/data_streamsafe.yaml
```

Generate the raw schema and validation reports:

```bash
python scripts/run_data_validation.py --config configs/data_streamsafe.yaml
```

Generated raw data remain local. Small aggregate reports are written to `reports/tables/`.

The executed analysis is available in `notebooks/01_streamsafe_eda.ipynb`. Its standalone
summary is `reports/eda_summary.md`; public plots and aggregate tables are stored under
`reports/figures/` and `reports/tables/`.

## Sensitive-content policy

StreamSafe contains harmful and policy-violating examples. Do not paste full records into
GitHub issues, pull requests, logs, figures, or persisted notebook outputs. Public reports
should contain aggregate statistics, record identifiers, and redacted excerpts only.

Never commit:

- raw datasets;
- model weights;
- Hugging Face access tokens;
- environment files containing secrets;
- unredacted harmful examples copied from source data.

## Interfaces for follow-up work

Shared contracts live in `src/streamguard_bench/contracts.py`. Future contributions should:

1. normalize source records into `StreamingTrace` objects;
2. implement token, chunk, and sentence checkpoints through `StreamingPolicy`;
3. integrate Qwen3Guard-Stream behind the `Guard` protocol;
4. persist decisions as `EvaluationRecord` objects;
5. compute false-block, miss, leakage, and latency metrics from those records.

## Licensing and attribution

Project code is released under the MIT License. StreamSafe is licensed separately under
Creative Commons Attribution 4.0; this repository does not redistribute its raw contents.
