"""
side_streets.py
---------------
A junction cue from the calibrated ground plan (bev.Geometry): a side street is road
surface that suddenly reaches much further sideways than this street's road edge.

Per dense frame (5 per second), for the left and the right side:
    reach(Z)   how far drivable surface (road, lane markings, crosswalk) continues sideways,
               from our lane outward, in each 1-m band Z = 6–25 m ahead
    edge       this street's usual reach on that side (running median over EDGE_S seconds)
    mouth      reach ≥ edge + WIDER_M over at least MOUTH_M of length → a side street there

A mouth seen at distance Z is passed Z / v seconds later (v = the video's own speed);
detections that predict the same passing moment (± MERGE_S) are one side-street event.

Only video: no GPS, no map. The GPS answer key is used in evaluate() only.

Usage:
    .venv-video/bin/python -m src.video.side_streets <drive>      # detect + evaluate (GPS drives)

Output: data/video/processed/<drive>/side_streets.parquet (t, side, z_first, frames, width_m)
"""

from __future__ import annotations

import argparse
import json
import zlib

import numpy as np
import pandas as pd

from src.video import drives
from src.video.bev import Geometry
from src.video.scene import CODES

DRIVABLE = ("road", "lane_marking", "crosswalk", "bike_lane")   # (counting vehicles as road: rows of parked cars
                                                                 # became "road" — precision 21 % → 7 %; dropped)
PARKED_MAX = 2                           # more parked cars than this in the opening → a car park, not a street
CELL_M = 0.25
X_MAX_M = 20.0
Z_BANDS = np.arange(6.0, 25.0, 1.0)      # near edge of each 1-m band
EDGE_S = 12.0                            # running window for the street's usual road edge
WIDER_M = 3.0                            # a mouth reaches at least this much further than the edge …
MOUTH_M = 5                              # … over at least this many 1-m bands (a street is ≥ 5 m wide)
MIN_FRAMES = 3                           # seen in at least this many frames
MERGE_S = 2.0
MIN_SPEED_MS = 1.0                       # standing still: nothing approaches


def reach_profiles(drv: dict, geo: Geometry) -> pd.DataFrame:
    """Per frame and side: sideways reach of drivable ground (metres) in each Z band."""
    dense = pd.read_parquet(drv["folder"] / "scene_dense.parquet").sort_values("t").reset_index(drop=True)
    (W, H), valid, rr, cc = geo.grid(1 / CELL_M, (-X_MAX_M, X_MAX_M), (0.0, float(Z_BANDS[-1] + 1)))
    drivable = np.array([c in DRIVABLE for c in CODES])
    X = -X_MAX_M + (np.arange(W) + 0.5) * CELL_M
    Z = Z_BANDS[-1] + 1 - (np.arange(H) + 0.5) * CELL_M
    centre = W // 2
    rows = []
    for r in dense.to_dict("records"):
        codes = np.frombuffer(zlib.decompress(r["mask"]), np.uint8).reshape(json.loads(r["mask_shape"]))
        road = valid & drivable[codes[rr, cc]]
        rec = dict(t=r["t"])
        for side, cols in (("right", np.arange(centre, W)), ("left", np.arange(centre - 1, -1, -1))):
            reach = []
            for z0 in Z_BANDS:
                band = road[(Z >= z0) & (Z < z0 + 1)][:, cols].any(axis=0)       # any row of the band
                # continuous road from our lane outward (allow 1-m gaps: lane markings, small objects)
                gaps = np.flatnonzero(~band)
                stop = len(band)
                run = 0
                for g in range(len(band)):
                    run = run + 1 if not band[g] else 0
                    if run > int(1.0 / CELL_M):
                        stop = g - run + 1
                        break
                reach.append(abs(X[cols[min(stop, len(cols)) - 1]]) if stop > 0 else 0.0)
            rec[side] = reach
        rows.append(rec)
    return pd.DataFrame(rows)


def detect(drv: dict) -> pd.DataFrame:
    from src.video.surround import EgoMotion
    dense = pd.read_parquet(drv["folder"] / "scene_dense.parquet").iloc[:1]
    geo = Geometry(drv, dense.iloc[0])
    prof = reach_profiles(drv, geo)
    ego = EgoMotion(drv)
    dt = float(np.median(np.diff(prof.t)))
    win = max(3, int(EDGE_S / dt))
    hits = []
    for side in ("left", "right"):
        R = np.array(prof[side].tolist())                                   # frames × bands
        edge = pd.DataFrame(R).rolling(win, center=True, min_periods=win // 3).median().to_numpy()
        wide = R >= edge + WIDER_M
        for i, t in enumerate(prof.t):
            row = wide[i]
            run, best, longest = 0, None, 0
            for b in range(len(row)):                                       # longest run of wide bands
                run = run + 1 if row[b] else 0
                if run > longest:
                    longest, best = run, b - run + 1
            if longest < MOUTH_M:
                continue
            v = ego.between(t - 0.5, t + 0.5)[0]                            # metres per second, video speed
            if v < MIN_SPEED_MS:
                continue
            z = Z_BANDS[best] + MOUTH_M / 2
            hits.append(dict(t_seen=t, side=side, z=z, t_pass=t + z / v, mouth_m=longest,
                             width=float(np.nanmedian(R[i, best:best + MOUTH_M] - edge[i, best:best + MOUTH_M]))))
    h = pd.DataFrame(hits)
    if h.empty:
        return pd.DataFrame(columns=["t", "side", "z_first", "frames", "width_m"])
    events = []
    for side, g in h.sort_values("t_pass").groupby("side"):
        grp = (g.t_pass.diff() > MERGE_S).cumsum()
        for _, c in g.groupby(grp):
            if len(c) >= MIN_FRAMES:
                events.append(dict(t=round(float(c.t_pass.median()), 1), t_seen=float(c.t_seen.min()),
                                   side=side, z_first=round(float(c.z.max()), 1),
                                   frames=len(c), width_m=round(float(c.width.median()), 1),
                                   mouth_m=float(c.mouth_m.max())))
    out = pd.DataFrame(events).sort_values("t").reset_index(drop=True)
    out["parked"] = parked_inside(drv, out)          # kept for inspection; as a filter it removed real side
                                                      # streets too (parked cars line their mouths): recall 71 % → 14 %
    t0, t1 = drv.get("street_window_s", [0, 1e9])
    return out[(out.t >= t0) & (out.t <= t1)].reset_index(drop=True)


def parked_inside(drv: dict, ev: pd.DataFrame) -> list[int]:
    """Parked cars seen in each opening (beyond 3 m to that side) while approaching it — a car
    park entrance is a real opening the street map does not list (2026_10_23 at 10:12:55)."""
    f = drv["folder"] / "surround_data.js"
    tracks = drv["folder"] / "tracks.parquet"
    if not f.exists() or not tracks.exists():
        return [0] * len(ev)
    data = json.loads(f.read_text().split("=", 1)[1].rstrip().rstrip(";"))
    parked = set(pd.read_parquet(tracks).query("label == 'parked'").track_id)
    out = []
    for e in ev.itertuples():
        ids = set()
        for t, rows in data["frames"]:
            if e.t_seen <= t <= e.t:
                for o in rows:
                    X = o[2]
                    if o[0] in parked and (X > 3 if e.side == "right" else X < -3):
                        ids.add(o[0])
        out.append(len(ids))
    return out


def truth(drv: dict) -> pd.DataFrame:
    """GPS answer key: junctions passed (node within 15 m) with the sides streets join on."""
    from scipy.spatial import cKDTree
    from src.video.event_match import RouteMap
    from src.video.roads import load_graph
    g = drives.gps_track(drv).sort_values("video_s")
    R = RouteMap(load_graph())
    t0, t1 = drv.get("street_window_s", [g.video_s.min(), g.video_s.max()])
    secs = np.arange(int(t0), int(t1) + 1)
    la, lo = np.interp(secs, g.video_s, g.lat), np.interp(secs, g.video_s, g.lon)
    J = [n for n in R.nodes if R.deg.get(n, 0) >= 3]
    jt = cKDTree([[R.G.nodes[n]["y"] * 111_320, R.G.nodes[n]["x"] * R.kx] for n in J])
    d, idx = jt.query(np.column_stack([la * 111_320, lo * R.kx]))
    rows, last = [], None
    for k in np.flatnonzero(d < 15):
        n = J[idx[k]]
        if n == last:
            continue
        last = n
        t = secs[k]
        a = (np.interp(t - 3, g.video_s, g.lat), np.interp(t - 3, g.video_s, g.lon))
        b = (np.interp(t + 3, g.video_s, g.lat), np.interp(t + 3, g.video_s, g.lon))
        heading = np.degrees(np.arctan2((b[1] - a[1]) * R.kx, (b[0] - a[0]) * 111_320))
        sides = set()
        for m in set(R.G.successors(n)) | set(R.G.predecessors(n)):
            brg = np.degrees(np.arctan2((R.G.nodes[m]["x"] - R.G.nodes[n]["x"]) * R.kx,
                                        (R.G.nodes[m]["y"] - R.G.nodes[n]["y"]) * 111_320))
            rel = (brg - heading + 180) % 360 - 180
            if 30 < abs(rel) < 150:
                sides.add("right" if rel > 0 else "left")
        rows.append(dict(t=float(t), node=n, sides="+".join(sorted(sides))))
    return pd.DataFrame(rows)


def evaluate(drv: dict, ev: pd.DataFrame, tol_s: float = 4.0) -> dict:
    tr = truth(drv)
    tr = tr[tr.sides != ""]
    hit = [any(abs(e.t - x.t) <= tol_s and e.side in x.sides for x in tr.itertuples()) for e in ev.itertuples()]
    found = [any(abs(e.t - x.t) <= tol_s and e.side in x.sides for e in ev.itertuples()) for x in tr.itertuples()]
    return dict(detections=len(ev), correct=int(sum(hit)), precision=f"{np.mean(hit):.0%}" if len(ev) else "–",
                side_streets_passed=len(tr), found=int(sum(found)), recall=f"{np.mean(found):.0%}" if len(tr) else "–")


def run(drv: dict) -> pd.DataFrame:
    ev = detect(drv)
    ev.to_parquet(drv["folder"] / "side_streets.parquet", index=False)
    print(f"→ {drv['folder'] / 'side_streets.parquet'}: {len(ev)} side-street events "
          f"({(ev.side == 'left').sum()} left, {(ev.side == 'right').sum()} right)", flush=True)
    if drives.gps_track(drv) is not None:
        print("   vs GPS (±4 s, right side):", evaluate(drv, ev), flush=True)
    return ev


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    run(drives.load(ap.parse_args().drive))


if __name__ == "__main__":
    main()
