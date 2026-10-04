import pandas as pd

from streamguard_bench.metrics import (
    compute_response_policy_metrics,
    prefix_cost,
    streaming_cost,
    summarise_cost,
)


def _results(**overrides):
    row = {
        "trace_id": "t",
        "mode": "chunk_8",
        "policy": "strict",
        "intervention_token": None,
        "released_tokens": 20,
    }
    return pd.DataFrame([{**row, **overrides}])


def test_a_streaming_guard_reads_the_query_and_every_token_until_the_stop():
    passed = streaming_cost(_results(), {"t": 5}).iloc[0]
    stopped = streaming_cost(_results(intervention_token=8, released_tokens=0), {"t": 5}).iloc[0]
    assert (passed["model_calls"], passed["tokens"]) == (20, 25)
    assert (stopped["model_calls"], stopped["tokens"]) == (8, 13)


def test_a_prefix_guard_pays_for_the_whole_prefix_at_every_check():
    dataset = pd.DataFrame([{"trace_id": "t", "response": "a" * 20}])
    decisions = pd.DataFrame(
        {"trace_id": "t", "token_index": range(1, 21), "end_character": range(1, 21)}
    )
    checks = pd.DataFrame(
        {
            "trace_id": "t",
            "token_index": [8, 16, 20],
            "head_tokens": [18, 26, 30],
            "tail_tokens": 4,
            "generated_tokens": 1,
        }
    )
    cost = prefix_cost(_results(), checks, dataset, decisions, query_check_tokens={"t": 12})
    row = cost.iloc[0]
    assert row["model_calls"] == 3 + 1
    assert row["tokens"] == (18 + 26 + 30) + 3 * 5 + 12
    # With a cache the prefix is read once; only the tail and the answer are repeated.
    assert row["tokens_cached"] == 30 + 3 * 5 + 12

    stopped = prefix_cost(
        _results(intervention_token=16, released_tokens=8), checks, dataset, decisions
    ).iloc[0]
    assert (stopped["model_calls"], stopped["tokens"]) == (2, 18 + 26 + 2 * 5)


def test_cost_summary_converts_tokens_to_flops():
    cost = streaming_cost(_results(), {"t": 5})
    summary = summarise_cost(cost, parameters=1e9).iloc[0]
    assert summary["tflops"] == 2 * 1e9 * 25 / 1e12


def test_block_errors_can_be_reported_per_mode():
    results = pd.DataFrame(
        [
            {"trace_id": "s", "mode": "chunk_8", "policy": "strict", "blocked": True},
            {"trace_id": "s", "mode": "full_buffered", "policy": "strict", "blocked": False},
        ]
    ).assign(response_ground_truth="safe", error=None)
    by_mode = compute_response_policy_metrics(results, by_mode=True).set_index("mode")
    assert by_mode.loc["chunk_8", "fp"] == 1
    assert by_mode.loc["full_buffered", "fp"] == 0
