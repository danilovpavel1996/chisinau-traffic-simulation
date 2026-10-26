"""
lane_count.py
-------------
How many lanes a street has, from the video, where OpenStreetMap does not say: only 44 % of
Chișinău's car roads carry a `lanes` tag (residential 15 %, secondary 74 %), and without one SUMO
gives every street one lane per direction — Strada Mircești (secondary, 2026_10_26_10_02) got 1 + 1
where the driver counts 1 + 2.

Per analysed second inside the street window:
    street   the OSM way we are on (map_pose + the SUMO lanes beside us running our way)
    road     the kerbs / road edges the video sees left and right (lanes_world.parquet) → width
    lanes    width ÷ lane width (the spacing of the lane lines seen, else LANE_W_M) → total lanes
    split    which of those lanes carry traffic our way: our own lane does, and each lane takes the
             direction of most vehicles seen in it (surround objects: heading ±90° from ours)

Per way: the median over every second on it, kept only with MIN_S seconds of evidence. The table is
city-wide (data/video/lane_counts.parquet, one row per way and drive) so later drives add up;
`counts()` combines them. map_layer.LaneIndex uses it for ways OSM has no lanes for.

Usage:
    .venv-video/bin/python -m src.video.lane_count <drive> [<drive> …]
"""

from __future__ import annotations

import argparse
import io
import json
import contextlib

import numpy as np
import pandas as pd

from src.video import drives
from src.video.drives import ROOT

OUT = ROOT / "data" / "video" / "lane_counts.parquet"
WAYS = ROOT / "data" / "osm" / "way_lanes.parquet"
LANE_W_M = 3.3               # lane width where the lines do not show it
LANE_W_RANGE = (2.7, 3.9)
MAX_LANES = 6
LINE_GAP_M = 0.8             # lane-line readings closer than this are one line
EDGE_Z = 15.0                # kerbs / edges read within this far ahead
MIN_S = 5                    # seconds on a way before its count is used
MIN_DRIVES = 2               # drives agreeing on a way's lane total before the map uses it
USED_S = 10                  # seconds in a lane before we count it as one we drove in
STEP_S = 1.0


def way_of(edge: str) -> int | None:
    if edge.startswith(":"):
        return None
    try:
        return int(edge.lstrip("-").split("#")[0])
    except ValueError:
        return None


def per_second(drv: dict) -> pd.DataFrame:
    """One row per second: way, forward (our edge's direction), road width, lanes, lanes our way."""
    from src.video import map_layer as M, surround as SR
    from src.video.bev import Geometry
    folder = drv["folder"]
    mp = pd.read_parquet(folder / "map_pose.parquet")
    em = json.loads((folder / "event_match.json").read_text())["route"]
    e, n = M.utm().transform(em["lon"], em["lat"])
    li = M.LaneIndex(np.column_stack([e, n]), M.LAYER_M, video_lanes=False)
    lw = pd.read_parquet(folder / "lanes_world.parquet")
    lw = lw[lw.z <= EDGE_Z]
    dense = pd.read_parquet(folder / "scene_dense.parquet").sort_values("t")
    with contextlib.redirect_stdout(io.StringIO()):
        frames = SR.objects(drv, Geometry(drv, dense.iloc[0]))          # video only: no map lanes
    light = SR.KINDS.index("traffic_light")
    veh = pd.DataFrame([(t, o[0], o[2], o[3], o[7]) for t, r in frames for o in r
                        if o[1] != light and len(o) > 7 and o[6] != 1],
                       columns=["t", "tid", "X", "Z", "h"])
    # direction: the track's image-motion label where it has one (traffic.py — the heading from box
    # shape and ground motion read most oncoming cars on Mircești as ours: 10_06, 17_08), else the heading
    labels = pd.read_parquet(folder / "tracks.parquet").set_index("track_id").label
    lab = veh.tid.map(labels)
    veh["same"] = np.where(lab == "oncoming", False, np.where(lab.isin(["same_dir", "overtook_ego"]), True,
                                                            np.abs((veh.h + 180) % 360 - 180) < 90))
    veh = veh[lab.fillna("") != "parked"]
    veh = veh[(veh.Z > 3) & (veh.Z < 40)]
    by_t = {t: g for t, g in lw.groupby("t")}
    lt = np.array(sorted(by_t))
    rows = []
    for t0 in np.arange(mp.t.min(), mp.t.max(), STEP_S):
        r = mp.iloc[int(np.argmin(np.abs(mp.t - t0)))]
        X, Z, _, own, rel = li.car_frame(r.E, r.N, r.heading, radius=40)
        m = (Z > -5) & (Z < 10) & (np.abs(rel) < 30)
        if not m.any() or not len(lt):
            continue
        edge = li.lanes.edge.iat[own[m][np.argmin(np.abs(X[m]))]]
        way = way_of(edge)
        k = int(np.argmin(np.abs(lt - t0)))
        if way is None or abs(lt[k] - t0) > 0.5:
            continue
        g = by_t[lt[k]]
        right = g[g.kind.isin(["edge_right", "curb"]) & (g.x > 0.5) & (g.x < 9)].x
        left = g[g.kind.isin(["edge_left", "curb"]) & (g.x < -0.5) & (g.x > -20)].x
        if len(right) < 2 or len(left) < 2:
            continue
        xr, xl = float(right.median()), float(left.median())
        # lane lines between the kerbs (one per 5-m band each: clustered); they count the lanes even
        # where the far kerb is read too near (Mircești at 2:20: lines at −1.3 and −3.2 m, a solid centre
        # line and a dashed one between two oncoming lanes, but the left kerb read at −5.5 m)
        xs = np.sort(g[(g.kind == "line") & (g.x > xl + 0.8) & (g.x < xr - 0.8)].x.to_numpy())[::-1]
        lines = []
        for x in xs:
            if not lines or lines[-1][-1] - x > LINE_GAP_M:
                lines.append([x])
            else:
                lines[-1].append(x)
        lines = [float(np.median(c)) for c in lines if len(c) >= 2]          # seen in ≥ 2 bands
        width = xr - xl
        bounds = [xr] + lines + [xl]
        gaps = -np.diff(bounds)
        inner = gaps[1:-1] if len(gaps) > 2 else gaps[:0]
        lw_m = float(np.median(inner)) if len(inner) else (float(gaps[0]) if len(lines) else LANE_W_M)
        lw_m = float(np.clip(lw_m, *LANE_W_RANGE))
        n_l = max(len(lines) + 1, int(max(1, round(width / lw_m))))
        while len(bounds) - 1 < n_l:                                     # lanes no line shows: by width
            bounds.insert(-1, bounds[-2] - lw_m)
        bounds = np.array(bounds[:n_l] + [min(bounds[n_l], bounds[-1])])
        # who drives in each lane (0 = rightmost): our lane is ours; the rest by vehicle headings
        v = veh[(veh.t > t0 - 2) & (veh.t < t0 + 2)]
        vx = v.X.to_numpy()
        lane_of = np.searchsorted(-bounds, -vx) - 1                       # bounds descend
        inside = (lane_of >= 0) & (lane_of < n_l)
        same = v.same.to_numpy(bool)
        ours = int(np.searchsorted(-bounds, 0.0) - 1)
        row = dict(t=round(float(t0), 1), edge=edge, way=way, forward=not edge.startswith("-"),
                   width=round(width, 2), lane_w=round(lw_m, 2), lanes=n_l, ours=ours)
        for i in range(MAX_LANES):                                        # vehicles seen per lane, each way
            row[f"same{i}"] = int(np.sum(same & inside & (lane_of == i)))
            row[f"on{i}"] = int(np.sum(~same & inside & (lane_of == i)))
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(drv: dict, sec: pd.DataFrame) -> pd.DataFrame:
    """Per way on this drive: total lanes, which direction we drove it, the lanes we drove in, and
    what the vehicles seen in each lane suggest (lanes our way, contiguous from the right)."""
    W = pd.read_parquet(WAYS).set_index("way")
    out = []
    for way, g in sec.groupby("way"):
        if len(g) < MIN_S:
            continue
        total = int(min(np.median(g.lanes), MAX_LANES))
        g = g[g.lanes == total] if (g.lanes == total).sum() >= MIN_S else g
        used = int(max([i + 1 for i in range(total) if (g.ours == i).sum() >= USED_S] or [1]))
        sm = np.array([g[f"same{i}"].sum() for i in range(total)])
        on = np.array([g[f"on{i}"].sum() for i in range(total)])
        k_cars = 0
        for i in range(total):
            if i >= used and on[i] > sm[i]:
                break
            k_cars = i + 1
        out.append(dict(drive=drv["key"], way=int(way), name=W.name.get(way), highway=W.highway.get(way),
                        osm_lanes=W.lanes.get(way), oneway=W.oneway.get(way) in ("yes", "1", "-1"),
                        forward=bool(g.forward.mean() >= 0.5), seconds=len(g),
                        width_m=round(float(g.width.median()), 1), lane_w_m=round(float(g.lane_w.median()), 2),
                        lanes=total, used=used, k_cars=k_cars, oncoming_seen=int(on.sum())))
    return pd.DataFrame(out)


def fuse(g: pd.DataFrame) -> pd.Series:
    """One way from every drive on it. Right-hand traffic: every lane we drove in is our direction's —
    drives both ways settle the split (Mircești: towards the mall only the right lane of 3 was driven,
    towards home the right two → 1 + 2). Lanes no one drove in go by the vehicles seen in them; a two-way
    street whose far lanes all carried our direction is divided: the other side mirrored."""
    w = g.seconds.to_numpy(float)
    mode = lambda c: int(g[c].to_numpy()[np.argmax([w[g[c] == v].sum() for v in g[c]])])
    n = mode("lanes")
    fw, bw = g[g.forward], g[~g.forward]
    if bool(g.oneway.iat[0]):
        f, b = n, 0
    else:
        f = int(fw.used.max()) if len(fw) else 0
        b = int(bw.used.max()) if len(bw) else 0
        n = max(n, f + b)
        if f + b < n:
            if len(fw) and len(bw):                                       # both ways driven: cars decide
                f = int(np.clip(mode("k_cars") if len(fw) == len(g) else int(np.median(fw.k_cars)), f, n - b))
                b = n - f
            else:
                side = fw if len(fw) else bw
                k = int(np.clip(int(np.median(side.k_cars)), side.used.max(), n))
                if k >= n and side.oncoming_seen.sum() == 0:              # divided: the other side alike
                    mine, other = n, n
                elif k >= n:                                              # no lane left: the far half
                    mine = max(int(side.used.max()), n - n // 2); other = n - mine
                else:
                    mine, other = k, n - k
                f, b = (mine, other) if len(fw) else (other, mine)
    return pd.Series(dict(lanes_forward=f, lanes_backward=b, lane_w_m=float(np.average(g.lane_w_m, weights=w)),
                          seconds=int(w.sum()), drives=g.drive.nunique()))


def run(drv: dict) -> pd.DataFrame:
    sec = per_second(drv)
    S = summarise(drv, sec)
    old = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=S.columns)
    old = old[(old.drive != drv["key"]) & old.columns.isin(S.columns).all()] if "used" in old else old.iloc[:0]
    pd.concat([old, S], ignore_index=True).to_parquet(OUT, index=False)
    print(f"→ {OUT}: {len(S)} streets of {drv['key']} measured ({S.osm_lanes.isna().sum()} without an OSM lanes tag)", flush=True)
    C = counts(all_ways=True).set_index("way")
    for r in S.sort_values("seconds", ascending=False).head(12).itertuples():
        c = C.loc[r.way]
        print(f"   {str(r.name)[:28]:28s} {r.highway:12s} OSM lanes {r.osm_lanes or '–':>3}  video {r.lanes}, we drove "
              f"{r.used} of them ({r.seconds} s) → all drives: {c.lanes_forward} forward + {c.lanes_backward} backward", flush=True)
    return S


def counts(all_ways: bool = False) -> pd.DataFrame:
    """Per way, from every drive (fuse): lanes forward / backward, lane width — for ways OSM has no
    lane count for (all_ways: every measured way, to compare with OSM where it has one)."""
    cols = ["way", "lanes_forward", "lanes_backward", "lane_w_m", "seconds", "drives"]
    if not OUT.exists():
        return pd.DataFrame(columns=cols)
    C = pd.read_parquet(OUT)
    if "used" not in C:
        return pd.DataFrame(columns=cols)
    if not all_ways:
        C = C[C.osm_lanes.isna()]
    if not len(C):
        return pd.DataFrame(columns=cols)
    F = C.groupby("way").apply(fuse).reset_index()
    if all_ways:
        return F
    # used only where it is not one drive's guess: on OSM-tagged streets the method gets a carriageway's
    # lane count right 33 % of the time, the split 11 % (2026-10-27) — at least MIN_DRIVES drives that
    # agree on the lane total
    agree = C.groupby("way").lanes.agg(lambda s: s.value_counts().iat[0])
    return F[F.way.map(agree).fillna(0) >= MIN_DRIVES].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drives", nargs="+", help="drive keys in configs/drives.yaml")
    for k in ap.parse_args().drives:
        run(drives.load(k))


if __name__ == "__main__":
    main()
