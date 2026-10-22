"""
street_stats.py
---------------
Traffic measures per street and per traffic signal, from one drive's position
trace (position.py) and what the camera saw (perception.py):

    stops       every standstill ≥ MIN_STOP_S: where, how long, street, the next
                OSM traffic signal AHEAD on the driven route (metres, seconds until
                passing it), red light seen (daylight only), vehicles visible ahead
                in our lane, cause (queue for signal ahead / red light / other)
    streets     per street: distance, time, median speed while moving, share of
                time standing, number of stops
    signals     per OSM traffic signal (and drive): stops in its queue, time standing,
                where the queue started (metres before the signal)

and the same tables combined over all processed drives (data/video/aggregate/),
the input for calibrating the city simulation.

Usage:
    .venv-video/bin/python -m src.video.street_stats <drive>      # one drive + aggregate
    .venv-video/bin/python -m src.video.street_stats --aggregate  # aggregate only
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.video import drives
from src.video.drives import PROCESSED, ROOT
from src.video.traffic import VEHICLES, dominant_light

MIN_STOP_S = 3
SIGNAL_PASS_M = 25       # the car "passes" an OSM signal when it comes this close
QUEUE_MAX_M = 1000       # a stop belongs to the queue of the next signal ahead within this distance
LANE_X = (0.38, 0.62)    # image band (fraction of width) ≈ our lane ahead
AGGREGATE = ROOT / "data" / "video" / "aggregate"
FEATURES = ROOT / "data" / "osm" / "junction_features.parquet"


def stop_events(tr: pd.DataFrame) -> pd.DataFrame:
    w = tr[tr.in_window].reset_index(drop=True)
    still = ~w.moving
    seg = (still != still.shift()).cumsum()
    rows = []
    for _, g in w[still].groupby(seg[still]):
        if len(g) >= MIN_STOP_S:
            rows.append(dict(t_start=g.video_s.iat[0], t_end=g.video_s.iat[-1], duration_s=len(g),
                             clock=g.clock.iat[0], lat=g.lat.mean(), lon=g.lon.mean(), street=g.street.mode().iat[0]))
    return pd.DataFrame(rows)


def run(drv: dict, quiet: bool = False) -> dict:
    folder = drv["folder"]
    tr = pd.read_parquet(folder / "trace.parquet")
    meta = json.loads((folder / "meta.json").read_text())
    W, H = meta["width"], meta["height"]
    det = pd.read_parquet(folder / "detections.parquet")
    light = dominant_light(det).merge(det[["frame", "t"]].drop_duplicates(), on="frame")
    veh = det[det.cls.isin(VEHICLES) & ((det.y1 + det.y2) / 2 < H * (1 - drv["camera"]["hood"]))].copy()
    veh["cx"] = (veh.x1 + veh.x2) / 2 / W
    frame_t = det[["frame", "t"]].drop_duplicates().sort_values("t")

    sig = pd.read_parquet(FEATURES).query("kind == 'traffic_signals'") if FEATURES.exists() else pd.DataFrame()
    kx = 111_320 * np.cos(np.radians(tr.lat.mean()))
    tree = cKDTree(np.column_stack([sig.lat * 111_320, sig.lon * kx])) if len(sig) else None

    st = stop_events(tr)
    if len(st):
        reds, queues, sig_d, sig_id, sig_t = [], [], [], [], []
        w = tr.reset_index(drop=True)
        near = np.zeros(len(w), dtype=int) - 1        # index of the signal the car is passing at each second
        if tree is not None:
            dist, idx = tree.query(np.column_stack([w.lat * 111_320, w.lon * kx]))
            near = np.where(dist <= SIGNAL_PASS_M, idx, -1)
        cum_m = np.cumsum(w.speed_kmh.to_numpy() / 3.6)
        for r in st.itertuples():
            during = light[(light.t >= r.t_start) & (light.t <= r.t_end)]
            reds.append(bool((during.light == "red").any()))
            # queue: vehicles in our lane band ~2 s after stopping (the car ahead has stopped too)
            f = frame_t.frame.iat[int(np.searchsorted(frame_t.t, min(r.t_start + 2, r.t_end)))
                                  if np.searchsorted(frame_t.t, min(r.t_start + 2, r.t_end)) < len(frame_t) else -1]
            ahead = veh[(veh.frame == f) & veh.cx.between(*LANE_X)]
            queues.append(int(len(ahead)))
            # next signal AHEAD along the driven route (not merely the nearest one)
            i0 = int(np.searchsorted(w.video_s, r.t_end))
            ahead_i = np.flatnonzero(near[i0:] >= 0)
            if len(ahead_i) and cum_m[i0 + ahead_i[0]] - cum_m[i0] <= QUEUE_MAX_M:
                k = i0 + ahead_i[0]
                sig_d.append(round(float(cum_m[k] - cum_m[i0]))); sig_id.append(int(sig.osm_id.iat[near[k]]))
                sig_t.append(round(float(w.video_s.iat[k] - r.t_start)))
            else:
                sig_d.append(np.nan); sig_id.append(None); sig_t.append(np.nan)
        st["red_light_seen"], st["vehicles_ahead"] = reds, queues
        st["next_signal_m"], st["signal_osm_id"], st["s_until_signal"] = sig_d, sig_id, sig_t
        st["cause"] = np.where(st.next_signal_m.notna(), "queue for signal ahead",
                               np.where(st.red_light_seen, "red light", "congestion / other"))
    st["drive"] = drv["key"]

    w = tr[tr.in_window]
    w = w.assign(metres=w.speed_kmh / 3.6)
    streets = (w.groupby("street")
                .agg(seconds=("video_s", "size"), km=("metres", lambda m: round(m.sum() / 1000, 2)),
                     median_kmh_moving=("speed_kmh", lambda v: round(float(v[v > 3].median()), 1) if (v > 3).any() else 0),
                     standing_share=("moving", lambda m: round(1 - m.mean(), 2)),
                     osm_lanes=("lanes", lambda x: x.mode().iat[0] if len(x.mode()) else ""))
                .sort_values("seconds", ascending=False))
    streets["stops"] = st.groupby("street").size().reindex(streets.index).fillna(0).astype(int) if len(st) else 0
    streets["drive"] = drv["key"]
    streets["source"] = tr.source.iat[0]

    st.to_csv(folder / "stops_detail.csv", index=False)
    streets.to_csv(folder / "streets.csv")
    if not quiet:
        print(f"\n── {drv['key']} ({tr.source.iat[0]}) — {len(st)} stops ≥ {MIN_STOP_S} s ──")
        if len(st):
            print(st[["clock", "duration_s", "street", "cause", "next_signal_m", "s_until_signal", "red_light_seen",
                      "vehicles_ahead"]]
                  .to_string(index=False))
        print("\nper street:")
        print(streets.drop(columns=["drive", "source"]).to_string())
    return dict(stops=st, streets=streets)


def aggregate():
    """All processed drives → data/video/aggregate/{stops,streets,signals}.csv."""
    stops, streets = [], []
    for f in sorted(PROCESSED.glob("*/stops_detail.csv")):
        stops.append(pd.read_csv(f))
        streets.append(pd.read_csv(f.with_name("streets.csv")))
    if not stops:
        return
    AGGREGATE.mkdir(parents=True, exist_ok=True)
    st, sr = pd.concat(stops), pd.concat(streets)
    st.to_csv(AGGREGATE / "stops.csv", index=False)
    by_street = (sr.groupby("street")
                   .apply(lambda g: pd.Series(dict(
                       drives=g.drive.nunique(), seconds=g.seconds.sum(), km=round(g.km.sum(), 2),
                       median_kmh_moving=round(float(np.average(g.median_kmh_moving, weights=g.seconds)), 1),
                       standing_share=round(float(np.average(g.standing_share, weights=g.seconds)), 2),
                       stops=int(g.stops.sum()))), include_groups=False)
                   .sort_values("seconds", ascending=False))
    by_street.to_csv(AGGREGATE / "streets.csv")
    at_sig = st.dropna(subset=["signal_osm_id"])
    signals = (at_sig.groupby(["signal_osm_id", "drive"])
                 .agg(approach=("street", lambda s: s.mode().iat[0]), stops=("duration_s", "size"),
                      standing_s=("duration_s", "sum"), queue_start_m=("next_signal_m", "max"),
                      longest_stop_s=("duration_s", "max"), mean_vehicles_ahead=("vehicles_ahead", "mean"))
                 .round(1).reset_index().sort_values("standing_s", ascending=False))
    sigpos = pd.read_parquet(FEATURES).set_index("osm_id")[["lat", "lon"]] if FEATURES.exists() else None
    if sigpos is not None:
        signals = signals.join(sigpos, on="signal_osm_id")
    signals.to_csv(AGGREGATE / "signals.csv")
    print(f"\n── all drives: {st.drive.nunique()} drives, {len(st)} stops, {len(by_street)} streets, "
          f"{len(signals)} signals with waits → {AGGREGATE}")
    print(signals.head(10).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", nargs="?", help="drive key in configs/drives.yaml")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.drive:
        run(drives.load(a.drive))
    aggregate()


if __name__ == "__main__":
    main()
