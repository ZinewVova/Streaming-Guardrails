import pandas as pd

from streamguard_bench.metrics import compute_streaming_metrics


def test_metrics_compute_binary_rates_and_exclude_errors():
    frame = pd.DataFrame(
        [
            _row("safe", False),
            _row("safe", True),
            _row("unsafe", True),
            _row("unsafe", False),
            _row("unsafe", False, error="failed"),
        ]
    )
    metrics = compute_streaming_metrics(frame)
    row = metrics.iloc[0]

    assert row["traces"] == 4
    assert row["failed_traces"] == 1
    assert row["false_positive_rate"] == 0.5
    assert row["false_negative_rate"] == 0.5


def _row(label, blocked, error=None):
    return {
        "trace_id": f"{label}-{blocked}-{error}",
        "mode": "token",
        "policy": "strict",
        "ground_truth_label": label,
        "blocked": blocked,
        "released_tokens": 2,
        "leakage_min": 0,
        "leakage_max": 1,
        "signal_token": 1,
        "intervention_token": 2,
        "detection_delay_min": 0,
        "detection_delay_max": 1,
        "checks": 2,
        "guard_time_ms": 3.0,
        "error": error,
    }
