#!/usr/bin/env python3
"""Download StreamSafe, preserve source tables, and write a provenance manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from streamguard_bench.data.load_streamsafe import load_streamsafe, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_streamsafe.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = (project_root / args.config).resolve()
    config = load_yaml_config(config_path)
    dataset_config = config["dataset"]
    loading_config = config.get("loading", {})
    cache_dir = (project_root / dataset_config["cache_dir"]).resolve()

    bundle = load_streamsafe(
        repository=dataset_config["repository"],
        revision=dataset_config.get("revision"),
        cache_dir=cache_dir,
        allow_remote_code=bool(loading_config.get("allow_remote_code", False)),
    )
    manifest_path = cache_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(bundle.manifest(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not dataset_config.get("revision"):
        config["dataset"]["revision"] = bundle.resolved_revision
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        print(f"Pinned revision in {config_path}: {bundle.resolved_revision}")

    print(f"Loader mode: {bundle.loader_mode}")
    print(f"Manifest: {manifest_path}")
    for name, frame in bundle.tables.items():
        print(f"- {name}: {len(frame):,} rows, {len(frame.columns)} columns")


if __name__ == "__main__":
    main()
