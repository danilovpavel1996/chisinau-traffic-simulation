"""
compare_runs.py
---------------
Compares two simulation runs (e.g., baseline vs adaptive) and produces:
  - Side-by-side KPI table
  - Travel time distribution overlay
  - Delay reduction by hour
  - Bottleneck improvement chart

Usage
-----
    python src/analysis/compare_runs.py \\
        --baseline data/outputs/baseline \\
        --compare  data/outputs/adaptive \\
        --out      data/outputs/comparison
"""

from __future__ import annotations
import argparse
import os
import yaml
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_trip_df(run_dir: str) -> pd.DataFrame | None:
    p = os.path.join(run_dir, "baseline_metrics.parquet")
    if not os.path.exists(p):
        p = os.path.join(run_dir, "metrics.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)
    return None


def load_kpis(run_dir: str) -> dict:
    p = os.path.join(run_dir, "corridor_kpis.yaml")
    if os.path.exists(p):
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}


def load_bottlenecks(run_dir: str) -> pd.DataFrame | None:
    p = os.path.join(run_dir, "baseline_bottlenecks.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return None


# ── KPI comparison table ──────────────────────────────────────────────────────

def print_comparison_table(kpis_base: dict, kpis_cmp: dict, label_base: str, label_cmp: str):
    keys = [
        ("Completion rate",          "completion_rate",       "pct"),
        ("Mean travel time (min)",   "mean_travel_min",       "float"),
        ("Median travel time (min)", "median_travel_min",     "float"),
        ("p90 travel time (min)",    "p90_travel_min",        "float"),
        ("Total delay (veh·h)",      "total_delay_h",         "float"),
        ("Mean waiting (min)",       "mean_waiting_min",      "float"),
    ]
    print(f"\n{'Metric':35s}  {label_base:>12s}  {label_cmp:>12s}  {'Δ':>10s}  {'Δ%':>8s}")
    print("─" * 84)
    for label, key, fmt in keys:
        base_val = kpis_base.get(key)
        cmp_val  = kpis_cmp.get(key)
        if base_val is None or cmp_val is None:
            continue
        delta    = cmp_val - base_val
        delta_pct = (delta / base_val * 100) if base_val != 0 else 0
        sign     = "▼" if delta < 0 else ("▲" if delta > 0 else "")
        if fmt == "pct":
            print(f"{label:35s}  {base_val:>11.1%}  {cmp_val:>11.1%}  {delta:>+9.1%}  {delta_pct:>+7.1f}%")
        else:
            print(f"{label:35s}  {base_val:>11.2f}  {cmp_val:>11.2f}  {delta:>+9.2f}  {sign}{abs(delta_pct):>6.1f}%")


# ── Figure — Travel time overlay ──────────────────────────────────────────────

def fig_tt_overlay(df_base: pd.DataFrame, df_cmp: pd.DataFrame,
                   label_base: str, label_cmp: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for df, label, color in [(df_base, label_base, "#2E86AB"),
                              (df_cmp,  label_cmp,  "#E84855")]:
        c = df[df["completed"]]["travel_time_min"]
        ax.hist(c, bins=60, alpha=0.6, color=color,
                label=f"{label} (p50={c.median():.1f}, p90={c.quantile(.9):.1f})")
        ax.axvline(c.median(), color=color, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Travel Time (minutes)")
    ax.set_ylabel("Vehicles")
    ax.set_title("Travel Time Distribution — Before vs After")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[compare] Figure → {out_path}")


# ── Figure — Hourly delay delta ───────────────────────────────────────────────

def fig_hourly_delay_delta(df_base: pd.DataFrame, df_cmp: pd.DataFrame,
                            label_base: str, label_cmp: str, out_path: str) -> None:
    def hourly_delay(df):
        return (
            df[df["completed"]]
            .groupby(df["depart_hour"].astype(int))["loss_min"]
            .mean()
        )

    h_base = hourly_delay(df_base)
    h_cmp  = hourly_delay(df_cmp)
    delta  = h_cmp.subtract(h_base, fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(h_base.index, h_base.values, label=label_base, color="#2E86AB", linewidth=2)
    ax.plot(h_cmp.index,  h_cmp.values,  label=label_cmp,  color="#E84855", linewidth=2)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Time Loss (min)")
    ax.set_title("Mean Delay by Hour")
    ax.legend()

    ax2 = axes[1]
    colors = ["#E84855" if v > 0 else "#90BE6D" for v in delta.values]
    ax2.bar(delta.index, delta.values, color=colors)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Δ Delay (min)  [positive = worse]")
    ax2.set_title(f"Delay Change: {label_cmp} − {label_base}")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[compare] Figure → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="data/outputs",      help="Baseline run outputs dir")
    parser.add_argument("--compare",  required=True,               help="Comparison run outputs dir")
    parser.add_argument("--out",      default="data/outputs/comparison")
    parser.add_argument("--label-baseline", default="Baseline (fixed-time)")
    parser.add_argument("--label-compare",  default="Adaptive (max-pressure)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df_base = load_trip_df(args.baseline)
    df_cmp  = load_trip_df(args.compare)
    kpis_b  = load_kpis(args.baseline)
    kpis_c  = load_kpis(args.compare)

    if df_base is None or df_cmp is None:
        print("[compare] One or both metric files missing. Run metrics.py for each scenario.")
        return

    print_comparison_table(kpis_b, kpis_c, args.label_baseline, args.label_compare)

    fig_tt_overlay(df_base, df_cmp, args.label_baseline, args.label_compare,
                   os.path.join(args.out, "fig_tt_overlay.png"))

    fig_hourly_delay_delta(df_base, df_cmp, args.label_baseline, args.label_compare,
                           os.path.join(args.out, "fig_hourly_delay.png"))

    print(f"\n[compare] Outputs → {args.out}/")


if __name__ == "__main__":
    main()
