"""
metrics.py
----------
Parses SUMO output files and computes corridor + network KPIs.

Inputs
------
  tripinfo.xml  → per-vehicle travel stats
  edgedata.xml  → per-edge speed/occupancy/flow per 15-min interval

Outputs
-------
  baseline_metrics.parquet   (vehicle-level data)
  baseline_bottlenecks.csv   (top-10 congested edges / time bins)

Usage
-----
    python src/simulation/metrics.py --config src/config/scenario.yaml
    python src/simulation/metrics.py --tripinfo data/outputs/tripinfo.xml \\
                                      --edgedata data/outputs/edgedata.xml
"""

from __future__ import annotations
import argparse
import os
import yaml
import numpy as np
import pandas as pd
from lxml import etree


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_tripinfo(path: str) -> pd.DataFrame:
    """Parse tripinfo.xml → DataFrame of per-vehicle KPIs."""
    records = []
    for event, el in etree.iterparse(path, tag="tripinfo"):
        try:
            records.append({
                "id":             el.get("id"),
                "depart":         float(el.get("depart", 0)),
                "arrival":        float(el.get("arrival", -1)),
                "duration":       float(el.get("duration", 0)),
                "routeLength":    float(el.get("routeLength", 0)),
                "waitingTime":    float(el.get("waitingTime", 0)),
                "timeLoss":       float(el.get("timeLoss", 0)),
                "departLane":     el.get("departLane", ""),
                "arrivalLane":    el.get("arrivalLane", ""),
                "vType":          el.get("vType", "passenger"),
            })
        except (TypeError, ValueError):
            pass
        el.clear()
    df = pd.DataFrame(records)
    df["completed"] = df["arrival"] > 0
    df["travel_time_min"] = df["duration"] / 60.0
    df["waiting_min"]     = df["waitingTime"] / 60.0
    df["loss_min"]        = df["timeLoss"] / 60.0
    df["depart_hour"]     = df["depart"] / 3600.0
    return df


def parse_edgedata(path: str) -> pd.DataFrame:
    """Parse edgedata.xml → DataFrame of per-edge per-interval stats."""
    records = []
    current_interval_begin = 0.0

    for event, el in etree.iterparse(path, tag=["interval", "edge"]):
        if el.tag == "interval":
            current_interval_begin = float(el.get("begin", 0))
        elif el.tag == "edge":
            try:
                records.append({
                    "time_bin":    current_interval_begin,
                    "edge_id":     el.get("id"),
                    "speed":       float(el.get("speed",       0)),
                    "density":     float(el.get("density",     0)),
                    "occupancy":   float(el.get("occupancy",   0)),
                    "flow":        float(el.get("entered",     0)),
                    "waiting":     float(el.get("waitingTime", 0)),
                    "traveltime":  float(el.get("traveltime",  0)),
                })
            except (TypeError, ValueError):
                pass
        el.clear()

    df = pd.DataFrame(records)
    df["time_bin_h"] = df["time_bin"] / 3600.0
    return df


# ── Corridor KPIs ─────────────────────────────────────────────────────────────

def corridor_kpis(trip_df: pd.DataFrame) -> dict:
    completed = trip_df[trip_df["completed"]]
    if completed.empty:
        return {}
    tt = completed["travel_time_min"]
    return {
        "n_completed":         int(len(completed)),
        "n_total":             int(len(trip_df)),
        "completion_rate":     len(completed) / len(trip_df),
        "mean_travel_min":     float(tt.mean()),
        "median_travel_min":   float(tt.median()),
        "p90_travel_min":      float(tt.quantile(0.90)),
        "p95_travel_min":      float(tt.quantile(0.95)),
        "total_delay_h":       float(completed["loss_min"].sum() / 60),
        "mean_waiting_min":    float(completed["waiting_min"].mean()),
    }


def peak_kpis(trip_df: pd.DataFrame, peaks: dict) -> dict:
    result = {}
    for pk_name, pk in peaks.items():
        mask = (trip_df["depart"] >= pk["start"]) & (trip_df["depart"] < pk["end"])
        sub  = trip_df[mask & trip_df["completed"]]
        if sub.empty:
            continue
        tt = sub["travel_time_min"]
        result[pk_name] = {
            "n":           int(len(sub)),
            "mean_min":    float(tt.mean()),
            "p90_min":     float(tt.quantile(0.90)),
            "total_delay_h": float(sub["loss_min"].sum() / 60),
        }
    return result


# ── Bottleneck ranking ────────────────────────────────────────────────────────

def bottleneck_edges(
    edge_df: pd.DataFrame,
    top_n: int = 10,
    congestion_speed_ms: float = 5.0,   # < 18 km/h = congested
) -> pd.DataFrame:
    """
    Identify top-N congested edges by total congested minutes.
    """
    congested = edge_df[edge_df["speed"] < congestion_speed_ms].copy()
    congested["interval_min"] = 15   # bin width

    agg = (
        congested.groupby("edge_id")
        .agg(
            congested_intervals=("interval_min", "count"),
            total_congested_min=("interval_min", "sum"),
            mean_speed_ms=("speed", "mean"),
            max_waiting=("waiting", "max"),
            peak_flow=("flow", "max"),
        )
        .reset_index()
        .sort_values("total_congested_min", ascending=False)
        .head(top_n)
    )
    agg["rank"] = range(1, len(agg) + 1)
    return agg


# ── Save outputs ──────────────────────────────────────────────────────────────

def save_metrics(trip_df: pd.DataFrame, edge_df: pd.DataFrame, cfg: dict) -> dict:
    out_dir = cfg["outputs"]["dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Vehicle metrics
    metrics_file = os.path.join(out_dir, "baseline_metrics.parquet")
    trip_df.to_parquet(metrics_file, index=False)
    print(f"[metrics] Saved vehicle metrics → {metrics_file}")

    # Edge metrics
    edge_file = os.path.join(out_dir, "edgedata_parsed.parquet")
    edge_df.to_parquet(edge_file, index=False)

    # Bottlenecks
    bottleneck_file = os.path.join(out_dir, "baseline_bottlenecks.csv")
    bn = bottleneck_edges(edge_df)
    bn.to_csv(bottleneck_file, index=False)
    print(f"[metrics] Saved bottlenecks → {bottleneck_file}")

    # KPIs
    kpis = corridor_kpis(trip_df)
    kpi_file = os.path.join(out_dir, "corridor_kpis.yaml")
    import yaml as _yaml
    with open(kpi_file, "w") as f:
        _yaml.dump(kpis, f, default_flow_style=False)
    print(f"[metrics] Saved KPIs → {kpi_file}")

    return kpis


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="src/config/scenario.yaml")
    parser.add_argument("--tripinfo", help="Override tripinfo.xml path")
    parser.add_argument("--edgedata", help="Override edgedata.xml path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tripinfo_path = args.tripinfo or cfg["outputs"]["tripinfo"]
    edgedata_path = args.edgedata or cfg["outputs"]["edgedata"]

    print(f"[metrics] Parsing tripinfo: {tripinfo_path}")
    trip_df = parse_tripinfo(tripinfo_path)
    print(f"[metrics] Parsed {len(trip_df)} vehicles")

    print(f"[metrics] Parsing edgedata: {edgedata_path}")
    edge_df = parse_edgedata(edgedata_path)
    print(f"[metrics] Parsed {len(edge_df)} edge-interval rows")

    kpis = save_metrics(trip_df, edge_df, cfg)

    print("\n── Corridor KPIs ──────────────────────────────────────")
    for k, v in kpis.items():
        print(f"  {k:30s}: {v:.3f}" if isinstance(v, float) else f"  {k:30s}: {v}")
    print("────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
