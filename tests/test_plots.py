import pandas as pd
import pytest

from streamguard_bench.metrics import (
    compute_response_policy_metrics,
    compute_streaming_metrics,
)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plots = pytest.importorskip("streamguard_bench.plots")

MODES = ("token", "chunk_8", "full_buffered")


def _results() -> pd.DataFrame:
    rows = []
    for trace_id, truth, onset in [("a", "unsafe", 10), ("b", "unsafe", 20), ("c", "safe", None)]:
        for index, mode in enumerate(MODES):
            released = index * 6
            unsafe = truth == "unsafe"
            leakage = max(0, released - onset + 1) if unsafe else None
            signal = onset - 2 if unsafe else 30
            rows.append(
                {
                    "trace_id": trace_id,
                    "response_ground_truth": truth,
                    "mode": mode,
                    "policy": "strict",
                    "blocked": True,
                    "signal_token": signal,
                    "intervention_token": signal + index,
                    "released_tokens": released,
                    "safe_withheld_tokens": None if unsafe else 40 - released,
                    "leakage_tokens": leakage,
                    "normalized_leakage": None if leakage is None else leakage / 20,
                    "signal_offset_tokens": -2.0 if unsafe else None,
                    "premature_block": unsafe,
                    "early_lead_tokens": 2.0 if unsafe else None,
                    "detection_delay_tokens": None,
                    "post_signal_buffer_delay_tokens": index,
                    "checks": 40 // (index + 1),
                    "guard_time_ms": 100.0 * (index + 1),
                    "error": None,
                }
            )
    return pd.DataFrame(rows)


def _token_decisions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trace_id": "a",
                "token_index": index,
                "token_id": index,
                "end_character": index * 3,
                "risk_label": "safe" if index < 10 else "unsafe",
                "confidence": 0.9,
                "latency_ms": 10.0,
            }
            for index in range(1, 21)
        ]
    )


def test_every_figure_renders_without_empty_axes():
    results = _results()
    streaming = compute_streaming_metrics(results, resamples=50)
    classification = compute_response_policy_metrics(results)
    figures = [
        plots.plot_confusion_matrices(classification, title="response"),
        plots.plot_error_rates(classification, title="response"),
        plots.plot_signal_offset(results, policy="strict"),
        plots.plot_leakage_intervals(streaming),
    ]
    assert all(figure.axes for figure in figures)
    assert len(figures[0].axes) == len(classification)


def test_trace_timeline_uses_shared_token_axis():
    figure = plots.plot_trace_timeline(
        _token_decisions(), _results(), trace_id="a", policy="strict", onset_token=10
    )
    top, bottom = figure.axes
    assert top.get_xlim() == bottom.get_xlim()
    assert [label.get_text() for label in bottom.get_yticklabels()] == list(MODES)


def test_paired_difference_is_oriented_against_the_reference_mode():
    paired = pd.DataFrame(
        [
            {
                "policy": "strict",
                "mode_a": "token",
                "mode_b": "chunk_8",
                "metric": "leakage_tokens",
                "mean_difference": -2.0,
                "ci_low": -3.0,
                "ci_high": -1.0,
            }
        ]
    )
    axis = plots.plot_paired_differences(paired, reference_mode="token").axes[0]
    points = [line.get_xdata()[0] for line in axis.lines if line.get_marker() == "o"]
    assert points == [2.0]
