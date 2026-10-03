#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from streamguard_bench.pipeline import DEFAULT_CONFIG, load_config, run_or_load, saved_run_exists


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke2", "full"], default="smoke2")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Score the profile again instead of reading the saved run.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if saved_run_exists(config, profile=args.profile) and not args.force:
        print(f"Saved run found for {args.profile}; reading it instead of scoring.")
    run = run_or_load(config, profile=args.profile, force=args.force)
    print(f"Completed {len(run.prompt_decisions)} traces; failures: {len(run.errors)}")
    print(f"Results: {run.output_dir}")


if __name__ == "__main__":
    main()
