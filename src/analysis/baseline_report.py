"""
baseline_report.py
------------------
Generates a reproducible baseline analysis report:
  - Corridor KPI summary table
  - Congestion heatmap (edge × time bin)
  - Travel time distribution (overall + peak)
  - Bottleneck ranking table
  - Before/after placeholder section

Usage
-----
    python src/analysis/baseline_report.py --config src/config/scenario.yaml
    python src/analysis/baseline_report.py --scenario baseline --compare adaptive
"""

from __future__ import annotations
import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_parquet(path: str, label: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"[report] WARNING: {label} not found at {path}")
        return None
    return pd.read_parquet(path)


def load_kpis(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


# ── Figure 1 — Travel time distribution ──────────────────────────────────────

def fig_travel_time_dist(trip_df: pd.DataFrame, peaks: dict, out_path: str) -> None:
    completed = trip_df[trip_df["completed"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Travel Time Distribution — Baseline", fontsize=14, fontweight="bold")

    # Overall distribution
    ax = axes[0]
    ax.hist(completed["travel_time_min"], bins=60, color="#2E86AB", edgecolor="white", linewidth=0.3)
    p50 = completed["travel_time_min"].median()
    p90 = completed["travel_time_min"].quantile(0.90)
    ax.axvline(p50, color="#E84855", linestyle="--", linewidth=1.5, label=f"p50 = {p50:.1f} min")
    ax.axvline(p90, color="#F9C74F", linestyle="--", linewidth=1.5, label=f"p90 = {p90:.1f} min")
    ax.set_xlabel("Travel Time (minutes)")
    ax.set_ylabel("Vehicles")
    ax.set_title("All-day Distribution")
    ax.legend()

    # By peak period
    ax2 = axes[1]
    colors = ["#2E86AB", "#E84855", "#F9C74F", "#90BE6D"]
    for (pk_name, pk), color in zip(peaks.items(), colors):
        mask = (
            (trip_df["depart"] >= pk["start"]) &
            (trip_df["depart"] < pk["end"]) &
            trip_df["completed"]
        )
        sub = trip_df[mask]["travel_time_min"]
        if sub.empty:
            continue
        ax2.hist(sub, bins=40, alpha=0.7, color=color,
                 label=f"{pk_name} (n={len(sub)}, p50={sub.median():.1f}m)")
    ax2.set_xlabel("Travel Time (minutes)")
    ax2.set_title("By Peak Period")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[report] Figure → {out_path}")


# ── Figure 2 — Congestion heatmap ────────────────────────────────────────────

def fig_congestion_heatmap(edge_df: pd.DataFrame, out_path: str, top_n: int = 20) -> None:
    if edge_df is None or edge_df.empty:
        return

    # Select top_n most congested edges
    mean_speed = edge_df.groupby("edge_id")["speed"].mean()
    top_edges  = mean_speed.nsmallest(top_n).index.tolist()
    sub = edge_df[edge_df["edge_id"].isin(top_edges)].copy()

    pivot = sub.pivot_table(
        index="edge_id", columns="time_bin_h", values="speed", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(16, 7))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="RdYlGn",
        vmin=0, vmax=14,   # 0 – 50 km/h
        interpolation="nearest",
    )
    ax.set_title(f"Speed Heatmap — Top {top_n} Most Congested Edges", fontsize=13)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Edge ID")
    ax.set_xticks(range(0, len(pivot.columns), 4))
    ax.set_xticklabels([f"{h:.0f}:00" for h in pivot.columns[::4]], rotation=45, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([e[:25] for e in pivot.index], fontsize=6)
    plt.colorbar(im, ax=ax, label="Speed (m/s)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[report] Figure → {out_path}")


# ── Figure 3 — Hourly throughput and delay ───────────────────────────────────

def fig_hourly_profile(trip_df: pd.DataFrame, out_path: str) -> None:
    hourly = (
        trip_df[trip_df["completed"]]
        .groupby(trip_df["depart_hour"].astype(int))
        .agg(vehicles=("id", "count"), mean_loss=("loss_min", "mean"))
        .reset_index()
    )
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.bar(hourly["depart_hour"], hourly["vehicles"], color="#2E86AB", alpha=0.7, label="Vehicles")
    ax2.plot(hourly["depart_hour"], hourly["mean_loss"], color="#E84855", linewidth=2, label="Mean Delay (min)")
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Vehicles Departed")
    ax2.set_ylabel("Mean Time Loss (minutes)")
    ax1.set_title("Hourly Traffic Profile — Vehicles & Delay")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[report] Figure → {out_path}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(kpis: dict, bottlenecks_path: str) -> None:
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║          BASELINE SIMULATION — KPI SUMMARY          ║")
    print("╠══════════════════════════════════════════════════════╣")
    if kpis:
        rows = [
            ("Completed vehicles",        f"{kpis.get('n_completed', 0):,}"),
            ("Completion rate",            f"{kpis.get('completion_rate', 0):.1%}"),
            ("Mean travel time",           f"{kpis.get('mean_travel_min', 0):.1f} min"),
            ("Median travel time (p50)",   f"{kpis.get('median_travel_min', 0):.1f} min"),
            ("p90 travel time",            f"{kpis.get('p90_travel_min', 0):.1f} min"),
            ("Total delay (network)",      f"{kpis.get('total_delay_h', 0):.1f} veh·h"),
            ("Mean waiting time",          f"{kpis.get('mean_waiting_min', 0):.1f} min"),
        ]
        for label, val in rows:
            print(f"║  {label:35s} {val:>15s}  ║")
    print("╠══════════════════════════════════════════════════════╣")

    if os.path.exists(bottlenecks_path):
        bn = pd.read_csv(bottlenecks_path)
        print("║  TOP BOTTLENECK EDGES                                ║")
        print(f"║  {'Rank':>4}  {'Edge ID':30s}  {'Cong.min':>8}  ║")
        for _, row in bn.head(5).iterrows():
            eid = str(row["edge_id"])[:28]
            print(f"║  {int(row['rank']):>4}  {eid:30s}  {row['total_congested_min']:>7.0f}m  ║")
    print("╚══════════════════════════════════════════════════════╝")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="src/config/scenario.yaml")
    parser.add_argument("--compare", default=None,
                        help="Name of comparison run outputs dir for before/after")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = cfg["outputs"]["dir"]

    trip_df = load_parquet(os.path.join(out, "baseline_metrics.parquet"), "trip metrics")
    edge_df = load_parquet(os.path.join(out, "edgedata_parsed.parquet"),  "edge data")
    kpis    = load_kpis(os.path.join(out, "corridor_kpis.yaml"))

    if trip_df is None:
        print("[report] No data found. Run metrics.py first.")
        sys.exit(1)

    peaks = cfg["simulation"]["peak_hours"]

    fig_travel_time_dist(trip_df, peaks,
                         os.path.join(out, "fig_travel_time_dist.png"))

    if edge_df is not None:
        fig_congestion_heatmap(edge_df,
                               os.path.join(out, "fig_congestion_heatmap.png"))

    fig_hourly_profile(trip_df,
                       os.path.join(out, "fig_hourly_profile.png"))

    print_summary(kpis, os.path.join(out, "baseline_bottlenecks.csv"))

    print(f"\n[report] All outputs written to: {out}/")


if __name__ == "__main__":
    main()
