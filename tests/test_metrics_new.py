import pandas as pd

from streamguard_bench.metrics import (
    bootstrap_ci,
    compute_response_policy_metrics,
    compute_streaming_metrics,
    paired_mode_differences,
    paired_run_differences,
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


def test_paired_run_differences_compare_the_same_traces_per_mode():
    def frame(values):
        return pd.DataFrame(
            [
                {"trace_id": trace_id, "mode": "token", "leakage_tokens": value}
                for trace_id, value in values.items()
            ]
        )

    left = frame({"a": 10.0, "b": 4.0, "only_left": 100.0})
    right = frame({"a": 6.0, "b": 4.0, "c": None})
    result = paired_run_differences(left, right, resamples=50).iloc[0]
    assert result["traces"] == 2
    assert result["mean_difference"] == 2.0
    assert result["ci_low"] <= 2.0 <= result["ci_high"]

    try:
        paired_run_differences(pd.concat([left, left]), right, resamples=50)
    except ValueError as error:
        assert "single policy" in str(error)
    else:
        raise AssertionError("ambiguous frames must be rejected")


def test_streaming_metrics_report_word_leakage_and_tolerate_older_runs():
    rows = [
        {
            "trace_id": trace_id,
            "mode": "token",
            "policy": "strict",
            "response_ground_truth": "unsafe",
            "blocked": True,
            "signal_token": 3,
            "released_tokens": 4,
            "safe_withheld_tokens": None,
            "leakage_tokens": tokens,
            "normalized_leakage": 0.5,
            "signal_offset_tokens": 1,
            "premature_block": False,
            "early_lead_tokens": None,
            "detection_delay_tokens": 1,
            "post_signal_buffer_delay_tokens": 0,
            "checks": 4,
            "guard_time_ms": 1.0,
            "error": None,
            "leakage_words": words,
            "leakage_characters": words * 5,
        }
        for trace_id, tokens, words in [("a", 4, 2), ("b", 8, 6)]
    ]
    frame = pd.DataFrame(rows)
    metrics = compute_streaming_metrics(frame, resamples=50).iloc[0]
    assert metrics["leakage_words_mean"] == 4.0
    assert metrics["leakage_characters_mean"] == 20.0

    older = compute_streaming_metrics(
        frame.drop(columns=["leakage_words", "leakage_characters"]), resamples=50
    ).iloc[0]
    assert pd.isna(older["leakage_words_mean"])
    assert older["leakage_mean"] == 6.0
