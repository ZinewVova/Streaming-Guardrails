#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from streamguard_bench.data import REVISION, build_manifest, load_dataset_frame, prepare_frame
from streamguard_bench.data.singstreambench import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/singstreambench.parquet")
    )
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3Guard-Stream-0.6B")
    parser.add_argument("--tokenizer-revision", default="419364a715de9840d47b1457982f64ff37f90ed4")
    args = parser.parse_args()
    source = load_dataset_frame(revision=args.revision, cache_dir="data/cache/huggingface")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=args.tokenizer_revision, trust_remote_code=True
    )
    prepared = prepare_frame(source, tokenizer, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(args.output, index=False)
    write_manifest(
        args.output.with_suffix(".manifest.json"),
        build_manifest(prepared, args.output, revision=args.revision),
    )
    print(f"Prepared {len(prepared)} traces at {args.output}")


if __name__ == "__main__":
    main()
