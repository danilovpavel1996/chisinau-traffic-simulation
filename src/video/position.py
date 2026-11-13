"""
position.py
-----------
Where the car is, every second — the single position trace all traffic
measures are attached to:

    dashcam with burnt-in GPS   gps.parquet (overlay.py)       source = "gps"
    otherwise                   match.json (localize.py)        source = "video match"

Each second is snapped to an OSM street edge: nearest edge within 30 m whose
direction agrees with the direction of travel, preferring to stay on the same
edge (GPS jitter must not hop between parallel streets or the two carriageways
of a boulevard).

Usage:
    .venv-video/bin/python -m src.video.position <drive>

Output: data/video/processed/<drive>/trace.parquet
    video_s, clock, lat, lon, speed_kmh, heading, moving, edge (u-v-k), street,
    highway, lanes (OSM), source
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.video import drives
from src.video.roads import Roads, load_graph, street_name, wrap

SNAP_M = 30.0          # search radius around the position
HEADING_M = 15.0       # metres of penalty per 45° of direction disagreement
STAY_M = 6.0           # metres of bonus for staying on the previous edge


def raw_positions(drv: dict) -> pd.DataFrame:
    """Per video second: lat, lon, speed (km/h), source."""
    gps = drives.gps_track(drv)
    if gps is not None:
        s = np.arange(int(np.ceil(gps.video_s.min())), int(gps.video_s.max()) + 1)
        lat = pd.Series(np.interp(s, gps.video_s, gps.lat)).rolling(3, center=True, min_periods=1).median()
        lon = pd.Series(np.interp(s, gps.video_s, gps.lon)).rolling(3, center=True, min_periods=1).median()
        return pd.DataFrame(dict(video_s=s, lat=lat, lon=lon, speed_kmh=np.interp(s, gps.video_s, gps.speed_kmh),
                                 source="gps"))
    m = json.loads((drv["folder"] / "match.json").read_text())
    tr = pd.DataFrame(m["trace"])
    kx = 111_320 * np.cos(np.radians(tr.lat.mean()))
    v = np.hypot(np.gradient(tr.lon) * kx, np.gradient(tr.lat) * 111_320) * 3.6
    return pd.DataFrame(dict(video_s=tr.video_s, lat=tr.lat, lon=tr.lon,
                             speed_kmh=pd.Series(v).rolling(5, center=True, min_periods=1).mean(),
                             source="video match"))


def snap(df: pd.DataFrame, roads: Roads) -> pd.DataFrame:
    """Attach the OSM edge each second is on (direction-aware, sticky)."""
    keys, lat, lon, brg = [], [], [], []
    kx0 = 111_320 * np.cos(np.radians(df.lat.mean()))
    edges = [(u, v, k, coords) for u, lst in roads.out.items() for v, k, _, coords in lst]
    for i, (u, v, k, coords) in enumerate(edges):
        c = np.array(coords)
        for a, b in zip(c, c[1:]):                                   # sample every ~5 m
            n = max(1, int(np.hypot((b[0] - a[0]) * 111_320, (b[1] - a[1]) * kx0) / 5))
            for f in np.linspace(0, 1, n, endpoint=False):
                p = a + (b - a) * f
                keys.append(i); lat.append(p[0]); lon.append(p[1])
                brg.append(np.degrees(np.arctan2((b[1] - a[1]) * kx0, (b[0] - a[0]) * 111_320)) % 360)
    keys, brg = np.array(keys), np.array(brg)
    kx = 111_320 * np.cos(np.radians(df.lat.mean()))
    tree = cKDTree(np.column_stack([np.array(lat) * 111_320, np.array(lon) * kx]))

    head = np.degrees(np.arctan2(np.gradient(df.lon) * kx, np.gradient(df.lat) * 111_320)) % 360
    head = pd.Series(head).where(df.speed_kmh > 4).ffill().bfill().to_numpy()   # unreliable when standing
    out, prev = [], None
    for la, lo, h in zip(df.lat, df.lon, head):
        idx = tree.query_ball_point([la * 111_320, lo * kx], r=SNAP_M)
        best, cost = None, np.inf
        for j in idx:
            d = np.hypot(tree.data[j][0] - la * 111_320, tree.data[j][1] - lo * kx)
            c = d + HEADING_M * abs(wrap(brg[j] - h)) / 45 - (STAY_M if keys[j] == prev else 0)
            if c < cost:
                best, cost = keys[j], c
        prev = best if best is not None else prev
        out.append(best)
    df = df.copy()
    df["heading"] = head.round(0)
    rows = [edges[i] if i is not None else None for i in out]
    df["edge"] = [f"{r[0]}-{r[1]}-{r[2]}" if r else None for r in rows]
    data = [roads.data((r[0], r[1], r[2])) if r else {} for r in rows]
    df["street"] = [street_name(d) if d else "(off map)" for d in data]
    df["highway"] = [str(d.get("highway", "")) for d in data]
    df["lanes"] = [str(d.get("lanes", "")) for d in data]
    return df


def run(drv: dict):
    df = raw_positions(drv)
    roads = Roads(load_graph())
    df = snap(df, roads)
    df["clock"] = drives.clock(drv, df.video_s).strftime("%H:%M:%S")
    df["moving"] = df.speed_kmh > 3
    t0, t1 = drv.get("street_window_s", [df.video_s.min(), df.video_s.max()])
    df["in_window"] = df.video_s.between(t0, t1)
    out = drv["folder"] / "trace.parquet"
    df.to_parquet(out, index=False)
    on = df[df.in_window]
    seq = on.street[on.street != on.street.shift()]
    print(f"→ {out}: {len(df)} s from {df.source.iat[0]}, {(on.street != '(off map)').mean():.0%} on a mapped street; "
          f"route: {' → '.join(dict.fromkeys(s.removeprefix('Strada ') for s in seq if s not in ('(unnamed)', '(off map)')))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    run(drives.load(ap.parse_args().drive))


if __name__ == "__main__":
    main()
