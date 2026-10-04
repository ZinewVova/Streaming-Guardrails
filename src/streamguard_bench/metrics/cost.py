"""Hardware-independent cost of a guard: how many tokens the model has to process.

Wall-clock time depends on the machine a run was made on, so runs from different machines
cannot be compared by it. The number of processed tokens can: it follows from the saved
decisions alone. Multiplied by the model size it gives an estimate in FLOPs.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from streamguard_bench.streaming import checkpoint_tokens

# A forward pass costs about two floating point operations per parameter per token.
FLOPS_PER_PARAMETER_TOKEN = 2


def streaming_cost(results: pd.DataFrame, query_tokens: Mapping[str, int]) -> pd.DataFrame:
    """Cost of a guard that reads the stream once, token by token, with a KV cache.

    The guard reads the query and then every response token up to the point where the
    stream is stopped. One row per row of `results`.
    """
    read = _processed_end(results)
    tokens = results["trace_id"].map(query_tokens) + read
    return results[["trace_id", "mode", "policy"]].assign(
        model_calls=read, tokens=tokens, tokens_cached=tokens
    )


def prefix_cost(
    results: pd.DataFrame,
    checks: pd.DataFrame,
    dataset: pd.DataFrame,
    token_decisions: pd.DataFrame,
    *,
    query_check_tokens: Mapping[str, int] | None = None,
    max_sentence_tokens: int = 128,
) -> pd.DataFrame:
    """Cost of a guard that reads the whole prefix again at every check.

    `checks` has one row per scored position (`trace_id`, `token_index`) with
    `head_tokens` (everything up to the end of the prefix), `tail_tokens` (the rest of the
    prompt) and `generated_tokens`. `tokens` is what the adapter does now: every check
    reads head and tail from scratch. `tokens_cached` is what a KV cache would leave: the
    head is read once up to the last check, and only the tail is repeated.

    `query_check_tokens` adds the separate check of the query, for guards that make one.
    """
    responses = dataset.set_index("trace_id")["response"].astype(str).to_dict()
    ends = {
        trace_id: group.sort_values("token_index")["end_character"].tolist()
        for trace_id, group in token_decisions.groupby("trace_id")
    }
    by_trace = {trace_id: group for trace_id, group in checks.groupby("trace_id")}
    stop = _processed_end(results)
    rows = []
    for record, last in zip(results.to_dict("records"), stop, strict=True):
        trace_id = record["trace_id"]
        points = checkpoint_tokens(
            ends[trace_id], responses[trace_id], [record["mode"]], max_sentence_tokens
        )
        made = by_trace.get(trace_id)
        if made is not None:
            made = made[made["token_index"].isin(points) & (made["token_index"] <= last)]
        if made is None or made.empty:
            calls = tokens = cached = 0
        else:
            repeated = made["tail_tokens"].sum() + made["generated_tokens"].sum()
            calls = len(made)
            tokens = made["head_tokens"].sum() + repeated
            cached = made["head_tokens"].max() + repeated
        if query_check_tokens is not None:
            calls += 1
            tokens += query_check_tokens[trace_id]
            cached += query_check_tokens[trace_id]
        rows.append(
            {
                "trace_id": trace_id,
                "mode": record["mode"],
                "policy": record["policy"],
                "model_calls": calls,
                "tokens": tokens,
                "tokens_cached": cached,
            }
        )
    return pd.DataFrame(rows)


def summarise_cost(cost: pd.DataFrame, *, parameters: float) -> pd.DataFrame:
    """Mean cost per trace for every mode and policy, with FLOPs in units of 10^12."""
    summary = (
        cost.groupby(["mode", "policy"], sort=False)[["model_calls", "tokens", "tokens_cached"]]
        .mean()
        .reset_index()
    )
    scale = FLOPS_PER_PARAMETER_TOKEN * parameters / 1e12
    return summary.assign(
        tflops=summary["tokens"] * scale, tflops_cached=summary["tokens_cached"] * scale
    )


def _processed_end(results: pd.DataFrame) -> pd.Series:
    """Number of response tokens generated before the stream stops."""
    return results["intervention_token"].fillna(results["released_tokens"]).astype(int)
