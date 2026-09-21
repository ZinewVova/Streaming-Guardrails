import pandas as pd

from streamguard_bench.metrics import (
    bootstrap_ci,
    compute_response_policy_metrics,
    compute_streaming_metrics,
    paired_mode_differences,
    wilson_interval,
)


def test_response_classification_is_not_duplicated_by_mode():
    rows = []
    for mode in ["token", "chunk_8"]:
        rows.extend(
            [
                {
                    "trace_id": "s",
                    "mode": mode,
                    "policy": "strict",
                    "response_ground_truth": "safe",
                    "blocked": True,
                    "error": None,
                },
                {
                    "trace_id": "u",
                    "mode": mode,
                    "policy": "strict",
                    "response_ground_truth": "unsafe",
                    "blocked": False,
                    "error": None,
                },
            ]
        )
    metrics = compute_response_policy_metrics(pd.DataFrame(rows))
    assert len(metrics) == 1
    assert metrics.iloc[0][["fp", "fn"]].tolist() == [1, 1]


def test_wilson_bootstrap_and_paired_bootstrap_are_deterministic():
    low, high = wilson_interval(5, 10)
    assert 0 < low < 0.5 < high < 1
    assert bootstrap_ci([1, 2, 3], seed=7, resamples=200) == bootstrap_ci(
        [1, 2, 3], seed=7, resamples=200
    )
    frame = pd.DataFrame(
        [
            {"trace_id": trace, "policy": "strict", "mode": mode, "leakage_tokens": value}
            for trace, a, b in [("a", 3, 1), ("b", 4, 2)]
            for mode, value in [("token", a), ("chunk_8", b)]
        ]
    )
    paired = paired_mode_differences(frame, resamples=100)
    assert abs(paired.iloc[0]["mean_difference"]) == 2


def test_streaming_metrics_report_median_p90_and_exclude_failures():
    rows = []
    for trace_id, leakage, delay in [("a", 0, 0), ("b", 2, 1), ("c", 10, 5)]:
        rows.append(
            {
                "trace_id": trace_id,
                "mode": "token",
                "policy": "strict",
                "response_ground_truth": "unsafe",
                "released_tokens": leakage,
                "safe_withheld_tokens": None,
                "leakage_tokens": leakage,
                "normalized_leakage": leakage / 10,
                "signal_token": 1,
                "signal_offset_tokens": delay,
                "premature_block": False,
                "early_lead_tokens": None,
                "detection_delay_tokens": delay,
                "post_signal_buffer_delay_tokens": 0,
                "checks": 1,
                "guard_time_ms": 1,
                "error": None,
            }
        )
    rows.append({**rows[0], "trace_id": "failed", "error": "protocol error"})
    metrics = compute_streaming_metrics(pd.DataFrame(rows), resamples=100).iloc[0]
    assert metrics["traces"] == 3
    assert metrics["failed_traces"] == 1
    assert metrics["leakage_median"] == 2
    assert metrics["leakage_p90"] > metrics["leakage_median"]
