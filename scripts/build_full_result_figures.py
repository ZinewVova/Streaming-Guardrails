"""Build compact public figures from aggregate Qwen3Guard full-run tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
MODE_ORDER = ["token", "chunk_8", "chunk_16", "chunk_32", "sentence", "full_buffered"]


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    _plot_policy_errors()
    _plot_streaming_tradeoff()


def _plot_policy_errors() -> None:
    frame = pd.read_csv(TABLES / "qwen3guard_full_policy_metrics.csv")
    positions = list(range(len(frame)))
    width = 0.34

    figure, axis = plt.subplots(figsize=(8, 4.8))
    false_positive = axis.bar(
        [position - width / 2 for position in positions],
        frame["false_positive_rate"] * 100,
        width,
        label="Ложные блокировки (FPR)",
        color="#d95f5f",
    )
    false_negative = axis.bar(
        [position + width / 2 for position in positions],
        frame["false_negative_rate"] * 100,
        width,
        label="Пропуски опасных ответов (FNR)",
        color="#4c78a8",
    )
    axis.bar_label(false_positive, fmt="%.1f%%", padding=3)
    axis.bar_label(false_negative, fmt="%.1f%%", padding=3)
    axis.set_xticks(positions, frame["policy"])
    axis.set_ylabel("Доля ответов, %")
    axis.set_title("Ошибки классификации зависят от политики, а не от буфера")
    axis.set_ylim(0, 16)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(FIGURES / "qwen3guard_policy_errors.png", dpi=180)
    plt.close(figure)


def _plot_streaming_tradeoff() -> None:
    frame = pd.read_csv(TABLES / "qwen3guard_full_streaming_metrics.csv")
    frame["mode"] = pd.Categorical(frame["mode"], categories=MODE_ORDER, ordered=True)
    frame = frame.sort_values("mode")
    positions = list(range(len(frame)))

    figure, (leakage_axis, delay_axis) = plt.subplots(1, 2, figsize=(13, 5.2))

    leakage_axis.vlines(
        positions,
        frame["leakage_min_mean"],
        frame["leakage_max_mean"],
        color="#8c6bb1",
        linewidth=7,
        alpha=0.45,
    )
    leakage_axis.scatter(
        positions,
        frame["leakage_min_mean"],
        color="#2c7fb8",
        label="Минимально возможная",
        zorder=3,
    )
    leakage_axis.scatter(
        positions,
        frame["leakage_max_mean"],
        color="#d95f5f",
        label="Максимально возможная",
        zorder=3,
    )
    leakage_axis.set_xticks(positions, frame["mode"], rotation=28, ha="right")
    leakage_axis.set_ylabel("Средняя утечка, токенов")
    leakage_axis.set_title("Диапазон возможной утечки")
    leakage_axis.legend(frameon=False)
    leakage_axis.spines[["top", "right"]].set_visible(False)

    bars = delay_axis.bar(
        positions,
        frame["post_signal_buffer_delay_mean"],
        color="#f2a65a",
    )
    delay_axis.bar_label(bars, fmt="%.1f", padding=3)
    delay_axis.set_xticks(positions, frame["mode"], rotation=28, ha="right")
    delay_axis.set_ylabel("Токенов после сигнала до вмешательства")
    delay_axis.set_title("Дополнительная задержка буфера")
    delay_axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle("Компромисс потоковых режимов на 250 опасных трассах", y=1.02)
    figure.tight_layout()
    figure.savefig(FIGURES / "qwen3guard_streaming_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
