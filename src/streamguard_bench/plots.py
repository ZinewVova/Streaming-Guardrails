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
    """One 2x2 count matrix per policy: rows are ground truth, columns the decision.

    Precision and recall of the blocking decision are printed under each matrix.
    """
    policies = list(metrics["policy"])
    width = 3.6 * len(policies) + 0.6
    figure, axes = _figure(1, len(policies), figsize=(width, 3.9), squeeze=False)
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
        blocked, unsafe = row["tp"] + row["fp"], row["tp"] + row["fn"]
        precision = f"{row['tp'] / blocked:.1%}" if blocked else "—"
        recall = f"{row['tp'] / unsafe:.1%}" if unsafe else "—"
        axis.annotate(
            f"precision {precision}  ·  recall {recall}",
            xy=(0.5, 0),
            xycoords="axes fraction",
            xytext=(0, -44),
            textcoords="offset points",
            ha="center",
            color=INK,
            fontsize=10,
        )
    figure.suptitle(title, color=INK, fontsize=13, x=0.02, ha="left")
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
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
    threshold: float | None = None,
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
    score_label = "confidence" if threshold is None else "unsafe score"
    top.set_ylabel(score_label, color=MUTED, fontsize=9)
    if threshold is not None:
        top.axhline(threshold, color=MUTED, linewidth=1, linestyle=(0, (2, 2)), zorder=2)
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
        Line2D(
            [],
            [],
            color=ACCENT,
            marker="|",
            linewidth=0,
            markersize=12,
            markeredgewidth=2.5,
            label="решение о блокировке",
        ),
    ]
    if len(signal):
        handles.append(Line2D([], [], color=LABEL_COLORS["unsafe"], label="правило сработало"))
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


def plot_signal_offset(
    results: pd.DataFrame, *, policy: str, early_bin: int = 25, late_bin: int = 10
) -> Figure:
    """When the stop rule fired relative to the annotated onset.

    Early and late signals get separate panels with their own bin width, because early
    signals spread over hundreds of tokens while delays stay within tens. Traces where the
    rule never fired have no offset and are shown as one separate bar.
    """
    rows = _valid(results)
    rows = rows[(rows["policy"] == policy) & (rows["response_ground_truth"] == "unsafe")]
    rows = rows.drop_duplicates("trace_id")
    offsets = rows["signal_offset_tokens"].dropna().to_numpy(float)
    misses = int(rows["signal_token"].isna().sum())
    early, late = -offsets[offsets < 0], offsets[offsets >= 0]
    early_color, miss_color = POLICY_COLORS["conservative"], LABEL_COLORS["unsafe"]

    figure, (left, right, missed) = _figure(
        1, 3, figsize=(10, 3.9), sharey=True, gridspec_kw={"width_ratios": [3, 3, 0.7]}
    )
    for axis in (left, right, missed):
        _style(axis)

    def bars(axis, values, width, color):
        if not len(values):
            return 0
        edges = np.arange(0, values.max() + width, width)
        if len(edges) < 2:
            edges = np.array([0, width])
        counts, edges = np.histogram(values, bins=edges)
        axis.bar(edges[:-1], counts, width=width * 0.92, align="edge", color=color, zorder=2)
        return counts.max()

    tallest = max(bars(left, early, early_bin, early_color), bars(right, late, late_bin, ACCENT))
    missed.bar(0, misses, width=0.6, color=miss_color, zorder=2)
    missed.annotate(
        str(misses),
        (0, misses),
        textcoords="offset points",
        xytext=(0, 4),
        ha="center",
        color=INK,
        fontsize=10,
    )
    left.set_ylim(0, max(tallest, misses, 1) * 1.15)

    left.invert_xaxis()
    left.set_xlim(left=max(early.max(), early_bin) + early_bin if len(early) else early_bin)
    left.set_xlim(right=0)
    right.set_xlim(left=0)
    right.axvline(0, color=INK, linewidth=1.4, zorder=3)
    missed.set_xlim(-0.6, 0.6)
    missed.set_xticks([0], ["без сигнала"])
    left.set_ylabel("трасс", color=MUTED, fontsize=9)
    left.set_xlabel(f"токенов до onset (столбец = {early_bin})", color=MUTED, fontsize=9)
    right.set_xlabel(f"токенов после onset (столбец = {late_bin})", color=MUTED, fontsize=9)

    exact = int((offsets == 0).sum())
    # The title belongs to the whole figure: on one panel it would squeeze the others.
    figure.text(0.015, 0.95, "Смещение срабатывания относительно onset", color=INK, fontsize=12)
    figure.text(
        0.015,
        0.885,
        f"policy: {policy} · до onset {len(early)} · точно в onset {exact}"
        f" · позже {len(late) - exact} · без сигнала {misses}",
        color=MUTED,
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.86))
    figure.legend(
        handles=[
            Patch(facecolor=early_color, label="сработало до onset"),
            Patch(facecolor=ACCENT, label="сработало на onset или позже"),
            Patch(facecolor=miss_color, label="не сработало"),
            Line2D([], [], color=INK, label="истинный onset"),
        ],
        frameon=False,
        fontsize=9,
        loc="lower center",
        ncols=4,
    )
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


def plot_decision_rule_grid(
    grid: pd.DataFrame, *, value: str, title: str, subtitle: str | None = None, percent: bool = True
) -> Figure:
    """One metric over the Harm@k rule: thresholds in rows, trigger counts in columns."""
    table = grid.pivot(index="threshold", columns="trigger_count", values=value).sort_index()
    figure, axis = _figure(figsize=(0.72 * len(table.columns) + 2.4, 0.55 * len(table) + 1.9))
    matrix = table.to_numpy(float)
    low, high = np.nanmin(matrix), np.nanmax(matrix)
    axis.imshow(matrix, cmap=SEQUENTIAL, aspect="auto", vmin=low, vmax=high)
    for (line, column), cell in np.ndenumerate(matrix):
        axis.text(
            column,
            line,
            f"{cell:.0%}" if percent else f"{cell:.1f}",
            ha="center",
            va="center",
            fontsize=9,
            color="#ffffff" if cell > (low + high) / 2 else INK,
        )
    axis.set_xticks(range(len(table.columns)), [str(item) for item in table.columns])
    axis.set_yticks(range(len(table)), [f"{item:g}" for item in table.index])
    axis.set_xlabel("k — сколько флагов нужно для остановки", color=MUTED, fontsize=9)
    axis.set_ylabel("порог θ", color=MUTED, fontsize=9)
    axis.tick_params(colors=MUTED, length=0, labelsize=9)
    for side in axis.spines.values():
        side.set_visible(False)
    _title(axis, title, subtitle)
    figure.tight_layout()
    return figure


def plot_run_differences(paired: pd.DataFrame, *, name_a: str, name_b: str) -> Figure:
    """Per-mode difference between two guards on the same traces; zero inside CI: no claim."""
    modes = _modes(paired)
    figure, axis = _figure(figsize=(9, 0.62 * len(modes) + 2.2))
    _style(axis, grid_axis="x")
    for position, mode in enumerate(modes):
        record = paired[paired["mode"] == mode].iloc[0]
        axis.plot([record["ci_low"], record["ci_high"]], [position] * 2, color=ACCENT, linewidth=2)
        axis.plot(record["mean_difference"], position, "o", color=ACCENT, markersize=8, zorder=3)
    axis.axvline(0, color=INK, linewidth=1.4, zorder=1)
    axis.set_yticks(range(len(modes)), modes)
    axis.invert_yaxis()
    metric = paired["metric"].iloc[0] if not paired.empty else "metric"
    axis.set_xlabel(
        f"разность {metric}: {name_a} − {name_b} (95% bootstrap CI)", color=MUTED, fontsize=9
    )
    _title(
        axis,
        f"{name_a} против {name_b} на одних и тех же трассах",
        f"справа от нуля — значение у {name_a} больше",
    )
    figure.tight_layout()
    return figure


def _model_colors(models) -> dict[str, str]:
    return {name: MODE_PALETTE[index % len(MODE_PALETTE)] for index, name in enumerate(models)}


def _model_handles(colors: dict[str, str]) -> list[Line2D]:
    return [
        Line2D([], [], color=color, marker="o", linewidth=2, label=name)
        for name, color in colors.items()
    ]


def plot_error_tradeoff(metrics: pd.DataFrame) -> Figure:
    """False blocks against misses, one point per guard, with Wilson intervals on both axes."""
    colors = _model_colors(metrics["model"])
    figure, axis = _figure(figsize=(6.4, 5.2))
    _style(axis, grid_axis="both")
    for record in metrics.to_dict("records"):
        color = colors[record["model"]]
        x, y = record["false_positive_rate"], record["false_negative_rate"]
        axis.plot(
            [record["false_positive_ci_low"], record["false_positive_ci_high"]],
            [y, y],
            color=color,
            linewidth=2,
        )
        axis.plot(
            [x, x],
            [record["false_negative_ci_low"], record["false_negative_ci_high"]],
            color=color,
            linewidth=2,
        )
        axis.plot(x, y, "o", color=color, markersize=9, zorder=3)
    limit = max(
        metrics["false_positive_ci_high"].max(), metrics["false_negative_ci_high"].max(), 0.1
    )
    axis.set_xlim(0, limit * 1.08)
    axis.set_ylim(0, limit * 1.08)
    axis.set_xlabel("доля заблокированных безопасных ответов", color=MUTED, fontsize=9)
    axis.set_ylabel("доля пропущенных вредных ответов", color=MUTED, fontsize=9)
    _legend_below(axis, _model_handles(colors), ncols=min(len(colors), 3), pad=-0.16)
    _title(axis, "Два вида ошибок блокировки ответа", "ближе к началу координат — лучше; 95% CI")
    figure.tight_layout()
    return figure


def plot_model_leakage(
    streaming_metrics: pd.DataFrame, *, statistic: str = "mean", unit: str = "tokens"
) -> Figure:
    """Leakage per buffer mode for several guards, each with its bootstrap interval.

    Use `unit="words"` when the guards tokenize differently.
    """
    column = f"leakage_{statistic}" if unit == "tokens" else f"leakage_{unit}_{statistic}"
    unit_name = {"tokens": "токены", "words": "слова"}.get(unit, unit)
    low, high = f"{column}_ci_low", f"{column}_ci_high"
    modes = _modes(streaming_metrics)
    colors = _model_colors(streaming_metrics["model"].drop_duplicates())
    offsets = np.linspace(-0.2, 0.2, len(colors)) if len(colors) > 1 else [0.0]
    figure, axis = _figure(figsize=(9, 0.72 * len(modes) + 2.2))
    _style(axis, grid_axis="x")
    for offset, (model, color) in zip(offsets, colors.items(), strict=True):
        for position, mode in enumerate(modes):
            row = streaming_metrics[
                (streaming_metrics["mode"] == mode) & (streaming_metrics["model"] == model)
            ]
            if row.empty:
                continue
            row = row.iloc[0]
            axis.plot([row[low], row[high]], [position + offset] * 2, color=color, linewidth=2)
            axis.plot(row[column], position + offset, "o", color=color, markersize=8, zorder=3)
    axis.set_yticks(range(len(modes)), modes)
    axis.invert_yaxis()
    axis.set_xlabel(
        f"leakage, {statistic} ({unit_name}, 95% bootstrap CI)", color=MUTED, fontsize=9
    )
    _legend_below(axis, _model_handles(colors), ncols=min(len(colors), 3), pad=-0.2)
    _title(
        axis,
        f"Leakage по режимам буфера ({statistic})",
        "перекрывающиеся интервалы не дают вывода о преимуществе",
    )
    figure.tight_layout()
    return figure


def plot_rule_frontier(grid: pd.DataFrame) -> Figure:
    """Every stop rule as a point; the line joins the rules no other rule beats on both axes.

    `grid` needs `model`, both error rates and a boolean `default` marking each guard's
    own rule, which is drawn as a large outlined marker.
    """
    colors = _model_colors(grid["model"].drop_duplicates())
    figure, axis = _figure(figsize=(7.2, 5.4))
    _style(axis, grid_axis="both")
    for model, color in colors.items():
        points = grid[grid["model"] == model]
        axis.scatter(
            points["false_positive_rate"],
            points["false_negative_rate"],
            s=14,
            color=color,
            alpha=0.3,
            linewidths=0,
            zorder=2,
        )
        ordered = points.sort_values(["false_positive_rate", "false_negative_rate"])
        misses = ordered["false_negative_rate"]
        best = ordered[misses < misses.cummin().shift(1, fill_value=np.inf)]
        axis.plot(
            best["false_positive_rate"],
            best["false_negative_rate"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
            zorder=3,
        )
        default = points[points["default"]]
        axis.scatter(
            default["false_positive_rate"],
            default["false_negative_rate"],
            s=150,
            facecolor=color,
            edgecolor=INK,
            linewidths=1.6,
            marker="*",
            zorder=4,
        )
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("доля заблокированных безопасных ответов", color=MUTED, fontsize=9)
    axis.set_ylabel("доля пропущенных вредных ответов", color=MUTED, fontsize=9)
    handles = _model_handles(colors) + [
        Line2D(
            [], [], color=MUTED, marker="*", markersize=12, markeredgecolor=INK, linewidth=0,
            label="правило по умолчанию",
        )
    ]
    _legend_below(axis, handles, ncols=min(len(handles), 3), pad=-0.16)
    _title(axis, "Все правила остановки", "линия — лучшие правила; ближе к нулю — лучше")
    figure.tight_layout()
    return figure
