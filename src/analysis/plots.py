"""
plots.py
--------
Shared plotting utilities for all analysis scripts.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


PALETTE = {
    "blue":   "#2E86AB",
    "red":    "#E84855",
    "yellow": "#F9C74F",
    "green":  "#90BE6D",
    "purple": "#7B2D8B",
    "orange": "#F8961E",
}


def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "#F8F9FA",
        "axes.grid":        True,
        "grid.color":       "#DDDDDD",
        "grid.linewidth":   0.5,
        "font.family":      "sans-serif",
        "axes.spines.top":  False,
        "axes.spines.right": False,
    })


def travel_time_cdf(
    series_dict: dict[str, pd.Series],
    ax: plt.Axes | None = None,
    title: str = "Travel Time CDF",
) -> plt.Axes:
    """Plot empirical CDFs for one or more travel-time series."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    colors = list(PALETTE.values())
    for (label, series), color in zip(series_dict.items(), colors):
        sorted_vals = np.sort(series.dropna())
        cdf         = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, label=label, color=color, linewidth=2)
        p90 = np.percentile(sorted_vals, 90)
        ax.axvline(p90, color=color, linestyle=":", linewidth=1,
                   label=f"p90={p90:.1f}m")
    ax.set_xlabel("Travel Time (minutes)")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    return ax


def queue_heatmap(
    edge_df: pd.DataFrame,
    value_col: str = "density",
    top_n: int = 15,
    ax: plt.Axes | None = None,
    title: str = "Queue / Density Heatmap",
) -> plt.Axes:
    top_edges = (
        edge_df.groupby("edge_id")[value_col].mean()
        .nlargest(top_n).index.tolist()
    )
    sub = edge_df[edge_df["edge_id"].isin(top_edges)]
    pivot = sub.pivot_table(index="edge_id", columns="time_bin_h",
                             values=value_col, aggfunc="mean")

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Hour")
    ax.set_xticks(range(0, len(pivot.columns), 4))
    ax.set_xticklabels([f"{h:.0f}" for h in pivot.columns[::4]], fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([e[:22] for e in pivot.index], fontsize=6)
    plt.colorbar(im, ax=ax, label=value_col)
    return ax


def bar_bottleneck_rank(
    bottleneck_df: pd.DataFrame,
    baseline_df: pd.DataFrame | None = None,
    ax: plt.Axes | None = None,
    title: str = "Top Bottlenecks — Congested Minutes",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    labels = [e[:25] for e in bottleneck_df["edge_id"]]
    values = bottleneck_df["total_congested_min"].values
    bars = ax.barh(labels[::-1], values[::-1], color=PALETTE["red"], alpha=0.85)
    if baseline_df is not None:
        base_vals = (
            baseline_df.set_index("edge_id")["total_congested_min"]
            .reindex(bottleneck_df["edge_id"]).values
        )
        ax.barh(labels[::-1], base_vals[::-1], color=PALETTE["blue"],
                alpha=0.4, label="Baseline")
        ax.legend()
    ax.set_xlabel("Total Congested Minutes")
    ax.set_title(title)
    return ax
