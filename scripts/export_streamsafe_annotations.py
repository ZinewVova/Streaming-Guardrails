#!/usr/bin/env python3
"""Export the public StreamSafe 500 boundary dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from streamguard_bench.data.load_streamsafe import load_yaml_config
from streamguard_bench.data.normalize_streamsafe import read_trace_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_subset_v1.yaml")
    parser.add_argument(
        "--output-dir",
        default="data/processed/huggingface/streamsafe-500-boundaries",
    )
    parser.add_argument(
        "--include-source-text",
        action="store_true",
        help="Include the original prompt and full response in the public files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(root / args.config)
    output_dir = root / args.output_dir
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    traces = read_trace_parquet(root / config["paths"]["subset"])
    frame = pd.DataFrame(
        [
            _public_record(trace, config, include_source_text=args.include_source_text)
            for trace in traces
        ]
    )
    forbidden = {"prompt", "response", "prefix_text", "review_segment"}
    leaked_columns = forbidden & set(frame.columns)
    allowed_text_columns = {"prompt", "response"} if args.include_source_text else set()
    unexpected_text_columns = leaked_columns - allowed_text_columns
    if unexpected_text_columns:
        raise RuntimeError(f"Sensitive text columns reached public export: {leaked_columns}")

    checksums: dict[str, str] = {}
    for split, filename in (("train", "train.parquet"), ("val", "validation.parquet")):
        path = data_dir / filename
        frame[frame["source_split"] == split].to_parquet(path, index=False)
        checksums[str(path.relative_to(output_dir))] = _sha256(path)

    validation_source = root / config["paths"]["subset_validation"]
    shutil.copy2(validation_source, output_dir / "subset_validation.csv")
    checksums["subset_validation.csv"] = _sha256(output_dir / "subset_validation.csv")

    manifest = {
        "name": "streamsafe-500-boundaries",
        "description": "A 500-trace streaming-boundary subset derived from StreamSafe.",
        "source_repository": config["dataset"]["repository"],
        "source_revision": config["dataset"]["revision"],
        "tokenizer_repository": config["tokenizer"]["repository"],
        "tokenizer_revision": config["tokenizer"]["revision"],
        "records": len(frame),
        "split_counts": frame["source_split"].value_counts().sort_index().to_dict(),
        "label_counts": frame["label"].value_counts().sort_index().to_dict(),
        "contains_source_text": args.include_source_text,
        "files_sha256": checksums,
    }
    manifest_path = output_dir / "metadata.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    card_path = output_dir / "README.md"
    card_path.write_text(_dataset_card(manifest), encoding="utf-8")
    print(f"Публичный выпуск подготовлен: {output_dir}")
    print(f"Строк: {len(frame)}; столбцов: {len(frame.columns)}")
    print(
        "Исходные запросы и ответы включены."
        if args.include_source_text
        else "Исходные запросы и ответы отсутствуют."
    )


def _public_record(
    trace: Any, config: dict[str, Any], *, include_source_text: bool
) -> dict[str, Any]:
    onset = trace.harm_onset
    split_suffix = "train" if trace.source_split == "train" else "val"
    record: dict[str, Any] = {}
    if include_source_text:
        record["prompt"] = trace.prompt
        record["response"] = trace.response
    record.update(
        {
        "label": trace.binary_label,
        "harm_categories": list(trace.harm_categories),
        "last_safe_end_character": onset.lower_character if onset else None,
        "first_unsafe_end_character": onset.upper_character if onset else None,
        "last_safe_end_byte_utf8": onset.lower_byte if onset else None,
        "first_unsafe_end_byte_utf8": onset.upper_byte if onset else None,
        "first_unsafe_sentence": onset.sentence_index if onset else None,
        "last_safe_end_qwen_token": onset.lower_qwen_token if onset else None,
        "first_unsafe_end_qwen_token": onset.upper_qwen_token if onset else None,
        "trace_id": trace.trace_id,
        "source_split": trace.source_split,
        "full_response_source_file": f"full_response_mix2.1-8k-{split_suffix}.jsonl",
        "full_response_source_row": trace.source_rows[0],
        "prefix_end_characters": [item.end_character for item in trace.prefix_annotations],
        "prefix_end_bytes_utf8": [item.end_byte for item in trace.prefix_annotations],
        "prefix_end_sentences": [item.end_sentence for item in trace.prefix_annotations],
        "prefix_labels": [item.original_label for item in trace.prefix_annotations],
        "prefix_harm_categories": [
            list(item.harm_categories) for item in trace.prefix_annotations
        ],
        }
    )
    return record


def _dataset_card(manifest: dict[str, Any]) -> str:
    contains_source_text = manifest["contains_source_text"]
    text_description = (
        "It includes the unchanged source prompt and full assistant response so it can be "
        "used without separately downloading StreamSafe. Prefix strings are reconstructed "
        "as `response[:end_character]`."
        if contains_source_text
        else "It does not redistribute prompts, responses, or prefix text."
    )
    reconstruction = (
        "Use `prompt` and `response` directly. Reconstruct any labeled prefix with "
        "`response[:prefix_end_characters[i]]`."
        if contains_source_text
        else "Download the pinned StreamSafe revision and use the source locator fields to "
        "rebuild the text-bearing traces."
    )
    return f"""---
license: cc-by-4.0
language:
- en
task_categories:
- text-classification
tags:
- safety
- streaming
- guardrail
- annotations
pretty_name: StreamSafe 500 Boundary Annotations
size_categories:
- n<1K
---

# StreamSafe 500 Boundary Annotations

This is a 500-trace streaming-boundary dataset derived from
[StreamSafe](https://huggingface.co/datasets/Solitude0630/StreamSafe). It contains
deterministic trace identifiers, prefix boundary positions, labels, harm categories, and
sentence-level unsafe-onset intervals. {text_description}

## Content warning

This dataset intentionally contains safety-sensitive assistant responses, including material
related to illegal activity, violence, privacy, sexual content, and self-harm. It is intended
only for safety evaluation, guardrail research, and controlled red-teaming. Do not display
raw examples in public logs, screenshots, issues, or model demonstrations.

## Source and attribution

- Source dataset: `Solitude0630/StreamSafe`
- Related paper: [SentGuard: Sentence-Level Streaming Guardrails for Large Language Models](https://arxiv.org/abs/2606.02041)
- Source license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Derived project: [Streaming Guardrails](https://github.com/ZinewVova/Streaming-Guardrails)

Changes made by this project: strict full-response/prefix matching, deterministic sampling
of 500 traces, label normalization, boundary-coordinate calculation, and cross-split checks.
The original authors do not endorse this derived release.

## Composition

- 500 records: 250 `safe`, 250 `unsafe`.
- 400 records from StreamSafe `train`, 100 from `validation`.
- The StreamSafe test split is not included because it has no labeled prefix sequence.

## Columns

The first four columns are `prompt`, `response`, `label`, and `harm_categories`. They are
followed by the unsafe-onset interval in character, UTF-8 byte, sentence, and Qwen-token
coordinates. `label` is the final binary `safe` or `unsafe` judgment. `prefix_labels`
preserves the three original prefix states: `safe`, `uncertain`, and `unsafe`. Constant
provenance and tokenizer values are stored once in `metadata.json` rather than repeated
in every row.

## Boundary semantics

For an unsafe trace:

- `last_safe_end_character` is the end of the previous available labeled prefix;
- `first_unsafe_end_character` is the end of the first available prefix labeled `unsafe`;
- the unsafe onset lies somewhere in the half-open interval
  `[last_safe_end_character, first_unsafe_end_character)`;
- `first_unsafe_end_character` is a confirmed sentence-level unsafe boundary, **not an exact
  character-level onset**.

Safe traces have null onset fields. Character and UTF-8 byte positions are zero-based.
Token coordinates refer only to the assistant response and use
`Qwen/Qwen3Guard-Stream-0.6B`. Exact technical versions are stored in `metadata.json`.

## Reconstruction

{reconstruction} The `trace_id` is a deterministic SHA-256-derived identifier of the
unchanged prompt and full response.

## Limitations

- StreamSafe publishes sampled sentence-boundary prefixes, not exact harmful spans.
- The interval may cover more than one sentence when intermediate boundaries were not sampled.
- Prefix labels may contain annotation noise.
- Categories are multi-label and imbalanced.
- Strict matching excludes ambiguous and unmatched source prefixes rather than guessing.
- The interval is a sentence-level annotation boundary, not an exact character-level harmful
  onset.

## License

Released under CC BY 4.0. Attribution to StreamSafe and its authors must be retained, the
source license must be linked, and the modifications described above must remain indicated.
Other rights, including privacy and third-party rights, may still apply to the source material.
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
