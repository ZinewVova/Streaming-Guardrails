#!/usr/bin/env python
"""Count the tokens every guard processes per trace and save the summary as CSV.

Needs the tokenizers of the guards (the `models` extra), not their weights. The result
goes to reports/tables/<model>/guard_cost.csv and is read by the comparison notebook.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from transformers import AutoTokenizer

from streamguard_bench.experiments import load_experiment_run
from streamguard_bench.guards.qwen3_gen_guard import ANSWER_PREFIX
from streamguard_bench.guards.sent_guard import sentguard_instruction
from streamguard_bench.metrics.cost import prefix_cost, streaming_cost, summarise_cost
from streamguard_bench.pipeline import saved_run_exists

PROFILE = "full"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--output", type=Path, default=Path("reports/tables"))
    args = parser.parse_args()
    for path in sorted(args.configs.glob("*.yaml")):
        config = yaml.safe_load(path.read_text())
        if not saved_run_exists(config, profile=PROFILE):
            continue
        summary = summarise_cost(_cost(config), parameters=float(config["model"]["parameters"]))
        target = args.output / Path(config["experiment"]["output_dir"]).name / "guard_cost.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        summary.round(3).to_csv(target, index=False)
        print(target)


def _cost(config: dict) -> pd.DataFrame:
    experiment, model = config["experiment"], config["model"]
    run = load_experiment_run(experiment["output_dir"], profile=PROFILE)
    dataset = pd.read_parquet(config["dataset"]["prepared_path"])
    queries = dataset.set_index("trace_id")["query"].astype(str).to_dict()
    adapter = model.get("adapter", "qwen3guard_stream")
    if adapter in ("qwen3guard_stream", "scm"):
        # Both guards share the vocabulary the dataset coordinates were built with.
        count = _counter(_metadata(run)["tokenizer_id"], _metadata(run)["tokenizer_revision"])
        query_tokens = {trace_id: count(query) for trace_id, query in queries.items()}
        return streaming_cost(run.intervention_results, query_tokens)
    decisions = run.token_decisions
    responses = dataset.set_index("trace_id")["response"].astype(str).to_dict()
    if adapter == "qwen3guard_gen":
        tokenizer = AutoTokenizer.from_pretrained(
            model["repository"], revision=model["tokenizer_revision"]
        )
        count = _count_with(tokenizer)

        def render(query: str, prefix: str | None) -> str:
            messages = [{"role": "user", "content": query}]
            if prefix is not None:
                messages.append({"role": "assistant", "content": prefix})
            return tokenizer.apply_chat_template(messages, tokenize=False) + ANSWER_PREFIX

        scored = decisions[decisions["confidence"].notna()].assign(generated_tokens=0)
        query_check = {trace_id: count(render(query, None)) for trace_id, query in queries.items()}
    else:
        # SentGuard is a Qwen3 fine-tune: the same vocabulary and the ChatML wrapper.
        count = _counter(_metadata(run)["tokenizer_id"], _metadata(run)["tokenizer_revision"])

        def render(query: str, prefix: str | None) -> str:
            instruction = sentguard_instruction(query, prefix)
            return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

        scored = decisions[decisions["evaluated"]]
        scored = scored.assign(generated_tokens=scored["raw_output"].map(count) + 1)
        query_check = None
    rows = []
    for record in scored.to_dict("records"):
        prefix = responses[record["trace_id"]][: record["end_character"]]
        text = render(queries[record["trace_id"]], prefix)
        split = text.rfind(prefix.strip()) + len(prefix.strip())
        rows.append(
            {
                "trace_id": record["trace_id"],
                "token_index": record["token_index"],
                "head_tokens": count(text[:split]),
                "tail_tokens": count(text[split:]),
                "generated_tokens": record["generated_tokens"],
            }
        )
    return prefix_cost(
        run.intervention_results,
        pd.DataFrame(rows),
        dataset,
        decisions,
        query_check_tokens=query_check,
        max_sentence_tokens=experiment["max_sentence_tokens"],
    )


def _metadata(run) -> dict:
    return json.loads((run.output_dir / "run_metadata.json").read_text())


def _counter(repository: str, revision: str | None):
    return _count_with(AutoTokenizer.from_pretrained(repository, revision=revision))


def _count_with(tokenizer):
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


if __name__ == "__main__":
    main()
