"""Figures for prompt, response-classification, and streaming replay metrics.

Colors follow one validated palette: policies use the categorical blue/orange pair,
buffer modes use the first six categorical slots with direct labels, and guard labels
use aqua/red/yellow plus a distinct marker shape as secondary encoding.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

MODE_ORDER = ("token", "chunk_8", "chunk_16", "chunk_32", "sentence", "full_buffered")
MODE_PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
MODE_COLORS = dict(zip(MODE_ORDER, MODE_PALETTE, strict=False))
POLICY_COLORS = {"strict": "#2a78d6", "conservative": "#eb6834"}
LABEL_COLORS = {"safe": "#1baf7a", "unsafe": "#e34948", "controversial": "#eda100"}
LABEL_MARKERS = {"safe": "o", "unsafe": "X", "controversial": "D"}
SURFACE = "#fcfcfb"
GRID = "#dedcd6"
INK = "#0b0b0b"
MUTED = "#52514e"
ACCENT = "#2a78d6"
FILL = "#b7d3f6"
BUFFERED = "#e4e2dc"
SEQUENTIAL = LinearSegmentedColormap.from_list("benchmark_blue", ["#eef4fd", "#0d366b"])


def _style(axis, *, grid_axis: str = "y"):
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
        axis.spines[side].set_linewidth(0.8)
    if grid_axis != "none":
        axis.grid(True, axis=grid_axis, color=GRID, linewidth=0.6, linestyle="-")
    axis.set_axisbelow(True)
    axis.tick_params(colors=MUTED, length=0, labelsize=9)
    return axis


def _figure(*args, **kwargs) -> tuple[Figure, object]:
    figure, axes = plt.subplots(*args, **kwargs)
    figure.patch.set_facecolor(SURFACE)
    return figure, axes


def _title(axis, text: str, subtitle: str | None = None) -> None:
    axis.set_title(text, color=INK, fontsize=12, loc="left", pad=22 if subtitle else 8)
    if subtitle:
        axis.annotate(
            subtitle,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(0, 6),
            textcoords="offset points",
            color=MUTED,
            fontsize=9,
        )


def _legend_below(axis, handles=None, *, ncols: int = 2, pad: float = -0.3) -> None:
    options = {
        "frameon": False,
        "fontsize": 9,
        "loc": "upper center",
        "bbox_to_anchor": (0.5, pad),
        "ncols": ncols,
    }
    axis.legend(**options) if handles is None else axis.legend(handles=handles, **options)


def _policy_handles(policies) -> list[Line2D]:
    return [
        Line2D([], [], color=POLICY_COLORS.get(name, ACCENT), marker="o", linewidth=2, label=name)
        for name in policies
    ]


def _modes(frame: pd.DataFrame) -> list[str]:
    present = set(frame["mode"].unique())
    ordered = [mode for mode in MODE_ORDER if mode in present]
    return ordered + sorted(present - set(ordered))


def _valid(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["error"].isna()] if "error" in frame else frame


def plot_confusion_matrices(metrics: pd.DataFrame, *, title: str) -> Figure:
    """One 2x2 count matrix per policy: rows are ground truth, columns the decision."""
    policies = list(metrics["policy"])
    width = 3.6 * len(policies) + 0.6
    figure, axes = _figure(1, len(policies), figsize=(width, 3.4), squeeze=False)
    largest = max(metrics[["tp", "fp", "tn", "fn"]].to_numpy().max(), 1)
    for axis, (_, row) in zip(axes[0], metrics.iterrows(), strict=False):
        matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=float)
        axis.imshow(matrix, cmap=SEQUENTIAL, vmin=0, vmax=largest)
        for (line, column), value in np.ndenumerate(matrix):
            axis.text(
                column,
                line,
                f"{int(value)}",
                ha="center",
                va="center",
                fontsize=14,
                color="#ffffff" if value > 0.6 * largest else INK,
            )
        axis.set_xticks([0, 1], ["allowed", "blocked"])
        axis.set_yticks([0, 1], ["safe", "unsafe"])
        axis.set_xlabel("guard decision", color=MUTED, fontsize=9)
        axis.set_ylabel("ground truth", color=MUTED, fontsize=9)
        axis.tick_params(colors=MUTED, length=0, labelsize=9)
        for side in axis.spines.values():
            side.set_visible(False)
        _title(axis, f"policy: {row['policy']}", f"{int(row['traces'])} трасс")
    figure.suptitle(title, color=INK, fontsize=13, x=0.02, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def plot_error_rates(metrics: pd.DataFrame, *, title: str) -> Figure:
    """False positive and false negative rates with 95% Wilson intervals."""
    figure, axis = _figure(figsize=(8, 3.2))
    _style(axis, grid_axis="x")
    rates = [
        ("false positive rate", "false_positive"),
        ("false negative rate", "false_negative"),
    ]
    policies = list(metrics["policy"])
    offsets = np.linspace(0.18, -0.18, len(policies))
    for offset, (_, row) in zip(offsets, metrics.iterrows(), strict=False):
        color = POLICY_COLORS.get(row["policy"], ACCENT)
        for position, (_, prefix) in enumerate(rates):
            point = row[f"{prefix}_rate"]
            axis.plot(
                [row[f"{prefix}_ci_low"], row[f"{prefix}_ci_high"]],
                [position + offset] * 2,
                color=color,
                linewidth=2,
                solid_capstyle="round",
            )
            axis.plot(point, position + offset, "o", color=color, markersize=9, zorder=3)
            axis.annotate(
                f"{point:.2f}",
                (point, position + offset),
                textcoords="offset points",
                xytext=(0, 10 if offset >= 0 else -16),
                ha="center",
                color=INK,
                fontsize=9,
            )
    axis.set_yticks(range(len(rates)), [name for name, *_ in rates])
    axis.set_xlim(-0.06, 1.06)
    axis.set_ylim(-0.5, len(rates) - 0.25)
    axis.set_xlabel("доля (95% Wilson CI)", color=MUTED, fontsize=9)
    _legend_below(axis, _policy_handles(policies), ncols=len(policies), pad=-0.34)
    _title(axis, title, "точка — оценка, отрезок — интервал")
    figure.tight_layout()
    return figure


def plot_trace_timeline(
    token_decisions: pd.DataFrame,
    results: pd.DataFrame,
    *,
    trace_id: str,
    policy: str,
    onset_token: int | None = None,
) -> Figure:
    """Per-token guard labels above, and where each buffer mode stops the stream below."""
    tokens = token_decisions[token_decisions["trace_id"] == trace_id].sort_values("token_index")
    rows = _valid(results)
    rows = rows[(rows["trace_id"] == trace_id) & (rows["policy"] == policy)]
    modes = _modes(rows)
    figure, (top, bottom) = _figure(
        2, 1, figsize=(11, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.5]}
    )
    _style(top)
    _style(bottom, grid_axis="x")
    for label, group in tokens.groupby("risk_label"):
        top.scatter(
            group["token_index"],
            group["confidence"],
            s=14,
            color=LABEL_COLORS.get(label, MUTED),
            marker=LABEL_MARKERS.get(label, "o"),
            linewidths=0,
            label=f"guard: {label}",
            zorder=3,
        )
    top.set_ylim(0, 1.08)
    top.set_ylabel("confidence", color=MUTED, fontsize=9)
    for position, mode in enumerate(modes):
        row = rows[rows["mode"] == mode].iloc[0]
        released = float(row["released_tokens"])
        buffered = max(0.0, float(row["intervention_token"]) - released)
        bottom.barh(position, released, height=0.5, color=FILL, zorder=2)
        bottom.barh(
            position,
            buffered,
            left=released,
            height=0.5,
            color=BUFFERED,
            edgecolor=SURFACE,
            linewidth=2,
            zorder=2,
        )
        bottom.plot(
            row["intervention_token"],
            position,
            marker="|",
            markersize=18,
            markeredgewidth=2.5,
            color=ACCENT,
            zorder=4,
        )
        bottom.annotate(
            f"выпущено {int(released)} · в буфере {int(buffered)}",
            (row["intervention_token"], position),
            textcoords="offset points",
            xytext=(10, -3),
            color=MUTED,
            fontsize=9,
        )
    signal = rows["signal_token"].dropna()
    if len(signal):
        bottom.axvline(float(signal.iloc[0]), color=LABEL_COLORS["unsafe"], linewidth=1.6, zorder=3)
    for axis in (top, bottom):
        if onset_token is not None:
            axis.axvline(onset_token, color=INK, linewidth=1.4, linestyle=(0, (5, 3)), zorder=3)
    bottom.set_yticks(range(len(modes)), modes)
    bottom.invert_yaxis()
    bottom.set_xlabel("номер response-токена", color=MUTED, fontsize=9)
    handles = [
        Patch(facecolor=FILL, label="выпущено пользователю"),
        Patch(facecolor=BUFFERED, label="удержано в буфере"),
        Line2D([], [], color=ACCENT, marker="|", linewidth=0, markersize=12,
               markeredgewidth=2.5, label="решение о блокировке"),
    ]
    if len(signal):
        handles.append(Line2D([], [], color=LABEL_COLORS["unsafe"], label="первый сигнал"))
    if onset_token is not None:
        handles.append(Line2D([], [], color=INK, linestyle=(0, (5, 3)), label="истинный onset"))
    handles = top.get_legend_handles_labels()[0] + handles
    _title(top, f"{trace_id}: решения guard по токенам", f"policy: {policy}")
    _title(bottom, "Когда остановился стрим")
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.legend(
        handles=handles,
        frameon=False,
        fontsize=9,
        loc="lower center",
        ncols=min(len(handles), 4),
    )
    return figure


def plot_signal_offset(results: pd.DataFrame, *, policy: str, bins: int = 20) -> Figure:
    """Distribution of the first blocking signal relative to the annotated onset."""
    rows = _valid(results)
    rows = rows[(rows["policy"] == policy) & (rows["response_ground_truth"] == "unsafe")]
    rows = rows.drop_duplicates("trace_id")
    offsets = rows["signal_offset_tokens"].dropna().to_numpy(float)
    misses = int(rows["signal_token"].isna().sum())
    figure, axis = _figure(figsize=(9, 3.6))
    _style(axis)
    if len(offsets):
        edges = np.histogram_bin_edges(offsets, bins=min(bins, max(1, len(np.unique(offsets)))))
        counts, edges = np.histogram(offsets, bins=edges)
        centers = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(edges) * 0.92
        colors = [POLICY_COLORS["conservative"] if center < 0 else ACCENT for center in centers]
        axis.bar(centers, counts, width=widths, color=colors, zorder=2)
    axis.axvline(0, color=INK, linewidth=1.4, zorder=3)
    axis.set_xlabel("signal_token − unsafe_start_token (токены)", color=MUTED, fontsize=9)
    axis.set_ylabel("трасс", color=MUTED, fontsize=9)
    _legend_below(
        axis,
        [
            Patch(facecolor=POLICY_COLORS["conservative"], label="сигнал до onset"),
            Patch(facecolor=ACCENT, label="сигнал на onset или позже"),
            Line2D([], [], color=INK, label="истинный onset"),
        ],
        ncols=3,
        pad=-0.28,
    )
    premature = int((offsets < 0).sum())
    delayed = int((offsets >= 0).sum())
    _title(
        axis,
        "Смещение первого сигнала относительно onset",
        f"policy: {policy} · преждевременно {premature} · вовремя или позже {delayed}"
        f" · без сигнала {misses}",
    )
    figure.tight_layout()
    return figure


def plot_leakage_ecdf(results: pd.DataFrame, *, policy: str) -> Figure:
    """Share of unsafe responses whose leakage stays at or below x tokens.

    Line width decreases along the mode order so that identical curves stay visible.
    """
    rows = _valid(results)
    rows = rows[(rows["policy"] == policy) & (rows["response_ground_truth"] == "unsafe")]
    modes = _modes(rows)
    figure, axis = _figure(figsize=(9, 4.2))
    _style(axis)
    widths = np.linspace(3.4, 1.4, max(len(modes), 1))
    limit = float(rows["leakage_tokens"].max() or 0)
    for width, mode in zip(widths, modes, strict=False):
        values = np.sort(rows[rows["mode"] == mode]["leakage_tokens"].dropna().to_numpy(float))
        if not len(values):
            continue
        share = np.arange(1, len(values) + 1) / len(values)
        axis.step(
            np.concatenate([[0], values, [limit + max(1.0, limit * 0.05)]]),
            np.concatenate([[0 if values[0] > 0 else share[0]], share, [share[-1]]]),
            where="post",
            color=MODE_COLORS.get(mode, ACCENT),
            linewidth=width,
            label=mode,
            zorder=3,
        )
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("выпущено опасных токенов (leakage)", color=MUTED, fontsize=9)
    axis.set_ylabel("доля опасных ответов ≤ x", color=MUTED, fontsize=9)
    _legend_below(axis, ncols=3, pad=-0.24)
    _title(
        axis,
        "Накопленное распределение leakage",
        f"policy: {policy} · {rows['trace_id'].nunique()} опасных трасс",
    )
    figure.tight_layout()
    return figure


def plot_leakage_intervals(streaming_metrics: pd.DataFrame, *, statistic: str = "mean") -> Figure:
    """Leakage point estimate per buffer mode with its bootstrap interval."""
    column = f"leakage_{statistic}"
    low, high = f"{column}_ci_low", f"{column}_ci_high"
    modes = _modes(streaming_metrics)
    policies = sorted(streaming_metrics["policy"].unique())
    offsets = np.linspace(0.16, -0.16, len(policies))
    figure, axis = _figure(figsize=(9, 0.62 * len(modes) + 2.2))
    _style(axis, grid_axis="x")
    for offset, policy in zip(offsets, policies, strict=False):
        color = POLICY_COLORS.get(policy, ACCENT)
        for position, mode in enumerate(modes):
            row = streaming_metrics[
                (streaming_metrics["mode"] == mode) & (streaming_metrics["policy"] == policy)
            ]
            if row.empty:
                continue
            row = row.iloc[0]
            axis.plot([row[low], row[high]], [position + offset] * 2, color=color, linewidth=2)
            axis.plot(row[column], position + offset, "o", color=color, markersize=8, zorder=3)
    axis.set_yticks(range(len(modes)), modes)
    axis.invert_yaxis()
    axis.set_xlabel(f"leakage, {statistic} (токены, 95% bootstrap CI)", color=MUTED, fontsize=9)
    _legend_below(axis, _policy_handles(policies), ncols=len(policies), pad=-0.2)
    _title(
        axis,
        f"Leakage по режимам буфера ({statistic})",
        "перекрывающиеся интервалы не дают вывода о преимуществе",
    )
    figure.tight_layout()
    return figure


def plot_cost_tradeoff(
    streaming_metrics: pd.DataFrame, results: pd.DataFrame, *, policy: str
) -> Figure:
    """Guard cost against leakage, and what safe responses pay for the same buffer."""
    metrics = streaming_metrics[streaming_metrics["policy"] == policy]
    rows = _valid(results)
    safe = rows[(rows["policy"] == policy) & (rows["response_ground_truth"] == "safe")]
    modes = _modes(metrics)
    figure, (left, right) = _figure(1, 2, figsize=(12, 4.4))
    _style(left)
    _style(right, grid_axis="x")
    ranked = list(metrics.set_index("mode").loc[modes, "checks_mean"].sort_values().index)
    for mode in modes:
        row = metrics[metrics["mode"] == mode].iloc[0]
        left.plot(
            row["checks_mean"], row["leakage_mean"], "o", color=ACCENT, markersize=10, zorder=3
        )
        left.annotate(
            mode,
            (row["checks_mean"], row["leakage_mean"]),
            textcoords="offset points",
            xytext=(0, (12, -20, 28)[ranked.index(mode) % 3]),
            ha="center",
            color=INK,
            fontsize=9,
        )
    left.set_xscale("symlog")
    if float(metrics["leakage_mean"].max() or 0) <= 0:
        left.set_ylim(-0.5, 1.0)
    left.margins(x=0.25, y=0.25)
    left.set_xlabel("проверок guard на трассу (среднее)", color=MUTED, fontsize=9)
    left.set_ylabel("leakage, среднее (токены)", color=MUTED, fontsize=9)
    _title(left, "Стоимость против утечки", f"policy: {policy} · ось x логарифмическая")
    if not safe.empty:
        released = safe.groupby("mode")["released_tokens"].mean().reindex(modes)
        withheld = safe.groupby("mode")["safe_withheld_tokens"].mean().reindex(modes)
        positions = np.arange(len(modes))
        right.barh(positions, released, height=0.55, color=ACCENT, label="выпущено", zorder=2)
        right.barh(
            positions,
            withheld,
            height=0.55,
            left=released,
            color=POLICY_COLORS["conservative"],
            label="удержано",
            edgecolor=SURFACE,
            linewidth=2,
            zorder=2,
        )
        for position, (shown, held) in enumerate(zip(released, withheld, strict=False)):
            right.annotate(
                f"{shown:.0f} / {shown + held:.0f}",
                (shown + held, position),
                textcoords="offset points",
                xytext=(8, -3),
                color=MUTED,
                fontsize=9,
            )
        right.set_yticks(positions, modes)
        right.invert_yaxis()
        right.margins(x=0.12)
        _legend_below(right, ncols=2, pad=-0.2)
    right.set_xlabel("токены безопасного ответа (среднее)", color=MUTED, fontsize=9)
    _title(right, "Цена для безопасных ответов", f"{safe['trace_id'].nunique()} безопасных трасс")
    figure.tight_layout()
    return figure


def plot_paired_differences(paired: pd.DataFrame, *, reference_mode: str = "token") -> Figure:
    """Within-trace differences against one reference mode; zero inside CI means no claim."""
    rows = []
    for record in paired.to_dict("records"):
        if reference_mode not in (record["mode_a"], record["mode_b"]):
            continue
        flip = record["mode_a"] == reference_mode
        other = record["mode_b"] if flip else record["mode_a"]
        sign = -1 if flip else 1
        rows.append(
            {
                "policy": record["policy"],
                "mode": other,
                "difference": sign * record["mean_difference"],
                "low": min(sign * record["ci_low"], sign * record["ci_high"]),
                "high": max(sign * record["ci_low"], sign * record["ci_high"]),
                "metric": record["metric"],
            }
        )
    frame = pd.DataFrame(rows)
    modes = _modes(frame)
    policies = sorted(frame["policy"].unique())
    offsets = np.linspace(0.16, -0.16, len(policies))
    figure, axis = _figure(figsize=(9, 0.62 * len(modes) + 2.2))
    _style(axis, grid_axis="x")
    for offset, policy in zip(offsets, policies, strict=False):
        color = POLICY_COLORS.get(policy, ACCENT)
        for position, mode in enumerate(modes):
            record = frame[(frame["policy"] == policy) & (frame["mode"] == mode)]
            if record.empty:
                continue
            record = record.iloc[0]
            span = [record["low"], record["high"]]
            axis.plot(span, [position + offset] * 2, color=color, linewidth=2)
            point = record["difference"]
            axis.plot(point, position + offset, "o", color=color, markersize=8, zorder=3)
    axis.axvline(0, color=INK, linewidth=1.4, zorder=1)
    axis.set_yticks(range(len(modes)), modes)
    axis.invert_yaxis()
    metric = frame["metric"].iloc[0] if not frame.empty else "metric"
    axis.set_xlabel(
        f"разность {metric} относительно {reference_mode} (95% bootstrap CI)",
        color=MUTED,
        fontsize=9,
    )
    _legend_below(axis, _policy_handles(policies), ncols=len(policies), pad=-0.2)
    _title(
        axis,
        f"Парное сравнение режимов с {reference_mode}",
        "справа от нуля — режим течёт сильнее эталона",
    )
    figure.tight_layout()
    return figure
