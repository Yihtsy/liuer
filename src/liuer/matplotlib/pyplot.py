"""Plotting helpers built on top of :mod:`matplotlib.pyplot`."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def radar_chart(labels, values, *, ax=None, fill=True, close=True, **plot_kwargs):
    """Draw a single-series radar chart."""
    labels = list(labels)
    values = list(values)
    if len(labels) != len(values):
        raise ValueError("labels and values must have the same length")
    if not labels:
        raise ValueError("labels must not be empty")

    if close:
        values = values + values[:1]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    plot_angles = angles + angles[:1] if close else angles

    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})

    alpha = plot_kwargs.pop("alpha", 0.25)
    ax.plot(plot_angles, values, **plot_kwargs)
    if fill:
        ax.fill(plot_angles, values, alpha=alpha)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    return ax.figure, ax


def standardize_columns(df, participant_col=None, group_col=None):
    """Rename selected columns to ``Participant`` and ``Group`` in place."""
    rename_map = {}
    if participant_col is not None:
        rename_map[participant_col] = "Participant"
    if group_col is not None:
        rename_map[group_col] = "Group"
    df.rename(columns=rename_map, inplace=True)


def bland_altman_plot(x, y, *, ax=None, sd_limit=1.96, scatter_kwargs=None, line_kwargs=None):
    """Draw a Bland-Altman agreement plot."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    means = (x + y) / 2
    diffs = y - x
    mean_diff = np.mean(diffs)
    sd_diff = np.std(diffs, ddof=1)

    if ax is None:
        _, ax = plt.subplots()

    scatter_kwargs = {"s": 28, "alpha": 0.75, **(scatter_kwargs or {})}
    line_kwargs = {"color": "black", "linewidth": 1, **(line_kwargs or {})}

    ax.scatter(means, diffs, **scatter_kwargs)
    ax.axhline(mean_diff, **line_kwargs)
    ax.axhline(mean_diff + sd_limit * sd_diff, linestyle="--", **line_kwargs)
    ax.axhline(mean_diff - sd_limit * sd_diff, linestyle="--", **line_kwargs)
    ax.set_xlabel("Mean")
    ax.set_ylabel("Difference")
    return ax.figure, ax


def grouped_bar(labels, series, *, ax=None, width=0.8, **bar_kwargs):
    """Draw a grouped bar chart.

    ``series`` should be a mapping of series name to values.
    """
    labels = list(labels)
    names = list(series.keys())
    values = [list(series[name]) for name in names]
    if any(len(item) != len(labels) for item in values):
        raise ValueError("each series must match labels length")

    if ax is None:
        _, ax = plt.subplots()

    x = np.arange(len(labels))
    bar_width = width / max(len(names), 1)
    offset_start = -width / 2 + bar_width / 2
    for index, (name, item) in enumerate(zip(names, values)):
        ax.bar(x + offset_start + index * bar_width, item, bar_width, label=name, **bar_kwargs)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    return ax.figure, ax


def slope_chart(labels, before, after, *, ax=None, before_label="Before", after_label="After"):
    """Draw a two-point slope chart."""
    labels = list(labels)
    before = list(before)
    after = list(after)
    if not (len(labels) == len(before) == len(after)):
        raise ValueError("labels, before, and after must have the same length")

    if ax is None:
        _, ax = plt.subplots()

    for label, left, right in zip(labels, before, after):
        ax.plot([0, 1], [left, right], marker="o")
        ax.text(-0.03, left, label, ha="right", va="center")
        ax.text(1.03, right, label, ha="left", va="center")

    lower = min(before + after)
    upper = max(before + after)
    padding = (upper - lower) * 0.08 if not math.isclose(upper, lower) else 1
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(lower - padding, upper + padding)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([before_label, after_label])
    return ax.figure, ax


def plot_jitter_boxplot_with_significance(
    df,
    target_features,
    p_values_dict,
    ordered_labels,
    *,
    group_col="Group",
    col_wrap=4,
    height=5,
    palette="Set2",
    significance_level=0.05,
    show=True,
):
    """Draw faceted boxplots with jittered points and significance markers."""
    import seaborn as sns

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    plt.rcParams.update({"font.family": "Arial", "font.size": 24})
    df_melted = df.melt(
        id_vars=[group_col],
        value_vars=target_features,
        var_name="target_feat",
        value_name="value",
    )
    grid = sns.FacetGrid(df_melted, col="target_feat", col_wrap=col_wrap, height=height, sharey=False)
    grid.map_dataframe(
        sns.boxplot,
        x=group_col,
        y="value",
        order=ordered_labels,
        palette=palette,
        fliersize=0,
        legend=False,
    )
    grid.map_dataframe(
        sns.stripplot,
        x=group_col,
        y="value",
        order=ordered_labels,
        jitter=True,
        color="black",
        size=6,
        edgecolor="black",
        legend=False,
    )
    grid.set_titles(col_template="{col_name}", fontsize=24)
    grid.set_axis_labels(group_col, "Feature Value", fontsize=24)

    for ax in grid.axes.flat:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)

        feature = ax.get_title().split(" = ")[-1]
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min, y_max + y_range * 0.1)

        p_value = p_values_dict.get(feature)
        if p_value is None or p_value >= significance_level:
            continue

        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        else:
            significance = "*"

        group_positions = [0, 1]
        y_line = y_max + y_range * 0.01
        y_text = y_line - y_range * 0.02
        ax.plot(
            [group_positions[0], group_positions[0]],
            [y_line - y_range * 0.02, y_line],
            color="black",
            linewidth=1.5,
        )
        ax.plot(
            [group_positions[1], group_positions[1]],
            [y_line - y_range * 0.02, y_line],
            color="black",
            linewidth=1.5,
        )
        ax.plot(group_positions, [y_line, y_line], color="black", linewidth=1.5)
        ax.text(
            sum(group_positions) / 2,
            y_text,
            significance,
            ha="center",
            va="bottom",
            color="black",
            fontsize=24,
        )

    plt.tight_layout()
    if show:
        plt.show()
    return grid
