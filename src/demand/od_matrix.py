"""
od_matrix.py
------------
Builds an Origin–Destination matrix per 15-minute time bin
using a gravity model calibrated on zone population/job weights.

Outputs
-------
    data/demand/od_matrix.csv  — columns: time_bin, origin, destination, trips

Usage
-----
    python src/demand/od_matrix.py --config src/config/scenario.yaml
"""

from __future__ import annotations
import argparse
import math
import os
import yaml
import numpy as np
import pandas as pd

from zones import ZoneRegistry


def haversine_km(lon1, lat1, lon2, lat2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_gravity_od(
    registry: ZoneRegistry,
    total_daily: int,
    beta: float = 0.8,
) -> pd.DataFrame:
    """
    Gravity model:
        T(i→j) ∝  P_i * A_j / d(i,j)^beta

    Normalised so that sum(T) == total_daily.
    Returns a DataFrame of (origin, destination, daily_trips).
    """
    zones = registry.all_zones()
    n = len(zones)
    ids = [z.id for z in zones]

    # Distance matrix (km)
    dist = np.zeros((n, n))
    for i, zi in enumerate(zones):
        for j, zj in enumerate(zones):
            if i == j:
                dist[i, j] = 0.5   # intra-zonal, 0.5 km stub
            else:
                dist[i, j] = haversine_km(*zi.centroid, *zj.centroid)

    # Raw gravity flows
    prod = np.array([z.production for z in zones])
    attr = np.array([z.attraction for z in zones])
    raw  = np.outer(prod, attr) / (dist ** beta)
    np.fill_diagonal(raw, 0)   # suppress intra-zonal

    # Scale to total daily trips
    scale = total_daily / raw.sum()
    T = raw * scale

    records = []
    for i, oi in enumerate(ids):
        for j, dj in enumerate(ids):
            if i != j and T[i, j] > 0:
                records.append({"origin": oi, "destination": dj, "daily_trips": T[i, j]})

    return pd.DataFrame(records)


def temporal_distribution(
    od_daily: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    Disaggregate daily OD into 15-min bins using peak multipliers.
    Returns (time_bin_start_sec, origin, destination, trips).
    """
    sim       = cfg["simulation"]
    begin     = sim["begin"]
    end       = sim["end"]
    bin_sec   = cfg["demand"]["time_bin_minutes"] * 60
    peaks     = sim["peak_hours"]
    mults     = cfg["demand"]["peak_multipliers"]

    bins = list(range(begin, end, bin_sec))
    n_bins = len(bins)

    def multiplier_for(t_start):
        for pk_name, pk in peaks.items():
            if pk["start"] <= t_start < pk["end"]:
                return mults[pk_name]
        return mults["offpeak"]

    weights = np.array([multiplier_for(b) for b in bins], dtype=float)
    weights /= weights.sum()   # normalise → fraction of daily trips per bin

    records = []
    for _, row in od_daily.iterrows():
        daily = row["daily_trips"]
        for i, b in enumerate(bins):
            trips_in_bin = daily * weights[i]
            if trips_in_bin >= 0.5:   # skip negligible flows
                records.append({
                    "time_bin": b,
                    "origin":      row["origin"],
                    "destination": row["destination"],
                    "trips": int(round(trips_in_bin)),
                })

    return pd.DataFrame(records)


def save_od(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[od_matrix] Saved {len(df)} OD rows → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config/scenario.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    registry = ZoneRegistry.from_config(args.config)
    total    = cfg["demand"]["total_daily_vehicles"]
    od_out   = cfg["demand"]["od_file"]

    od_daily   = build_gravity_od(registry, total_daily=total)
    od_temporal = temporal_distribution(od_daily, cfg)
    save_od(od_temporal, od_out)

    print("\n── OD Summary ──────────────────────────────────")
    print(f"  Daily OD pairs      : {len(od_daily)}")
    print(f"  Total daily trips   : {od_daily['daily_trips'].sum():.0f}")
    print(f"  Time-bin rows       : {len(od_temporal)}")
    print(f"  Total temporal trips: {od_temporal['trips'].sum()}")
    print("────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
