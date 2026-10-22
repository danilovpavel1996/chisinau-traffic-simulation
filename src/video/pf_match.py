"""
pf_match.py
-----------
Map-matching a GPS-less video with a particle filter on the OSM street graph.

The video gives, every second, two odometry signals (from analyze.ego_states):
    heading  H(t)  — integrated rotation (deg), relative: true heading + unknown offset
    speed    v(t)  — speed index, ∝ m/s up to an unknown scale
That alone is a dead-reckoned trajectory: the *shape* of the drive.

Each particle is one hypothesis
    (edge, offset along it, scale s [m per speed unit], align a [deg])
and every second
    move     offset += s · v(t)·dt; at a junction choose the next street —
             proposal favours streets whose direction fits the video heading a few
             seconds ahead (offline, so we may look ahead)
    weigh    change of direction over the last WINDOW s: the particle's own map
             path vs the video (curves, turns, roundabouts all count; immune to
             the slow heading drift of vision-only rotation);
             when the car comes to a stop, particles just before a junction gain
             (lights, give-way); implied speeds above V_MAX are penalised
    jitter   s and a random-walk slowly (speed-scale error, gyro-like drift)
    resample when the effective particle count drops
The best final particle is traced back through its ancestors → the route, and
its position every second.

Usage:
    .venv-video/bin/python -m src.video.pf_match <drive>     # key in src/config/drives.yaml

Output: data/video/processed/<drive>/match.json — per-second trace (lat, lon,
street, clock, error vs truth), true route, video-only shape, accuracy.
View it with src/video/viewer.py.
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from src.video import drives
from src.video.analyze import ego_states
from src.video.route_match import GRAPH, Roads, video_anchor

N_PARTICLES = 4000
STEP_M = 5.0           # resolution of the per-edge lookup tables
SIGMA_HEAD = 20.0      # deg, heading-agreement likelihood width
SIGMA_PROP = 35.0      # deg, junction proposal width
LOOKAHEAD = 3          # s, video heading used to choose the next street
WINDOW = 8             # s; heading is compared as CHANGE over this window. Video
                       # rotation is reliable short-term (turns) but drifts on
                       # straights (−97° on straight Mihai Viteazul) → absolute
                       # heading must not be trusted, short-term change can be
SIGMA_DELTA = 20.0     # deg, likelihood width for the windowed heading change
SIGMA_ABS = 30.0       # deg, absolute heading (with per-particle drift) — pins the street
RATE_SIGMA = 0.5       # deg/s, prior spread of the heading drift rate
RATE_WALK = 0.03       # deg/s per √s, drift rate may change (street types differ)
ALIGN_WALK = 0.4       # deg/√s random walk of the heading offset (drift)
SCALE_WALK = 0.01     # log-scale random walk per √s: lets the scale adapt to street type (wide vs narrow streets differ ~2×; 0.02 loses the route)
SCALE_PRIOR = 0.35     # log-scale spread of the initial scale around V_TYPICAL
V_MAX = 25.0           # m/s; faster implied speed is implausible in the city
STOP_SIGMA = 40.0      # m; a stop is most likely within this distance before a junction
STOP_FLOOR = 0.2       # mid-block stops (queues) keep this much likelihood
V_TYPICAL = 7.0        # m/s; centre of the scale prior
CONTRA_LOGW = -1.0     # entering an edge against its OSM one-way direction
START_RADIUS = 300     # m around the GPS anchor


def wrap(a):
    return (np.asarray(a) + 180) % 360 - 180


# ── Edge tables ───────────────────────────────────────────────────────────────
class EdgeTable:
    """Flat arrays of every directed edge sampled every STEP_M metres."""

    def __init__(self, roads: Roads):
        self.roads = roads
        self.key, self.length, self.contra = [], [], []
        self.start, self.n = [], []
        lat, lon, brg = [], [], []
        idx = {}
        for u, lst in roads.out.items():
            for v, k, length, *_r, coords in lst:
                c = np.array(coords)
                seg = np.array([ox.distance.great_circle(*a, *b) for a, b in zip(c, c[1:])])
                cum = np.concatenate([[0], np.cumsum(seg)])
                s = np.arange(0, max(cum[-1], 1e-3) + STEP_M, STEP_M)
                la, lo = np.interp(s, cum, c[:, 0]), np.interp(s, cum, c[:, 1])
                b = np.degrees(np.arctan2((np.roll(lo, -1) - lo) * np.cos(np.radians(la)),
                                          np.roll(la, -1) - la)) % 360
                b[-1] = b[-2] if len(b) > 1 else 0
                idx[(u, v, k)] = len(self.key)
                self.key.append((u, v, k)); self.length.append(max(cum[-1], 1.0))
                self.contra.append((u, v, k) in roads.contra)
                self.start.append(len(lat)); self.n.append(len(s))
                lat.extend(la); lon.extend(lo); brg.extend(b)
        self.idx = idx
        self.length, self.contra = np.array(self.length), np.array(self.contra)
        self.start, self.n = np.array(self.start), np.array(self.n)
        self.lat, self.lon, self.brg = np.array(lat), np.array(lon), np.array(brg)
        self.out = {u: [idx[(u, v, k)] for v, k, *_ in lst] for u, lst in roads.out.items()}

    def at(self, e: np.ndarray, off: np.ndarray):
        j = self.start[e] + np.minimum((off / STEP_M).astype(int), self.n[e] - 1)
        return self.lat[j], self.lon[j], self.brg[j]


# ── Odometry ──────────────────────────────────────────────────────────────────
def odometry(folder: Path, hfov: float, t_from: float, t_to: float) -> pd.DataFrame:
    """Per-second video heading (compass sense, + = clockwise) and speed index."""
    meta = json.loads((folder / "meta.json").read_text())
    e = ego_states(pd.read_parquet(folder / "ego.parquet"), meta["width"], hfov, meta.get("hood", 0.15))
    dt = e.t.diff().fillna(0)
    e["head"] = -(e.yaw_deg_s * dt).cumsum()              # yaw + = left → compass −
    e["dist"] = np.where(e.moving, e.speed_idx.fillna(0) * dt, 0.0)
    e = e[(e.t >= t_from) & (e.t <= t_to)]
    sec = e.assign(s=e.t.astype(int)).groupby("s").agg(head=("head", "last"), dist=("dist", "sum"),
                                                        moving=("moving", "mean"))
    sec["head"] -= sec["head"].iloc[0]
    # speed noise: single-second spikes of 3–5× (a car cannot do 30 → 190 km/h
    # in a second). 3 s median then 3 s mean; stops (zeros) are preserved.
    d = sec["dist"].rolling(3, center=True, min_periods=1).median()
    sec["dist"] = d.rolling(3, center=True, min_periods=1).mean().where(sec["dist"] > 0, 0.0)
    return sec.reset_index()


# ── Filter ────────────────────────────────────────────────────────────────────
def run_filter(T: EdgeTable, od: pd.DataFrame, anchor, end_point=None, seed: int = 0):
    rng = np.random.default_rng(seed)
    G = T.roads.G
    lat0, lon0 = anchor
    near = [n for n in G.nodes
            if ox.distance.great_circle(lat0, lon0, G.nodes[n]["y"], G.nodes[n]["x"]) < START_RADIUS]
    edges0 = np.array([e for n in near for e in T.out.get(n, [])])
    N = N_PARTICLES
    e = rng.choice(edges0, N)
    off = rng.uniform(0, 1, N) * T.length[e] * 0.5
    moving_dist = od.dist[od.moving > 0.5]
    s0 = V_TYPICAL / max(np.median(moving_dist), 1e-6)       # dist index per moving second
    scale = s0 * np.exp(rng.normal(0, SCALE_PRIOR, N))
    align = rng.uniform(0, 360, N)                           # video heading = map heading + align
    rate = rng.normal(0, RATE_SIGMA, N)                      # align drifts at this rate (deg/s)
    logw = np.zeros(N)

    hist_e, hist_off, hist_parent = [], [], []
    evidence = 0.0                                # log marginal likelihood: picks the best run, no truth needed
    brg_hist = np.full((N, WINDOW + 1), np.nan)   # each particle's map bearing, last WINDOW s
    head = od["head"].to_numpy(); dist = od["dist"].to_numpy()
    stopped = (od["moving"] < 0.2).to_numpy()
    stop_start = stopped & ~np.concatenate([[False], stopped[:-1]])
    for i in range(len(od)):
        off = off + scale * dist[i]
        # junction transitions
        over = np.flatnonzero(off > T.length[e])
        target_head = head[min(i + LOOKAHEAD, len(head) - 1)]
        for p in over:
            for _ in range(20):                               # may cross several short edges
                if off[p] <= T.length[e[p]]:
                    break
                u, v, k = T.key[e[p]]
                cands = [c for c in T.out.get(v, []) if T.key[c][1] != u] or T.out.get(v, [])
                if not cands:                                 # dead end
                    off[p] = T.length[e[p]]; logw[p] -= 5
                    break
                b = np.array([T.brg[T.start[c]] for c in cands])
                want = (target_head - align[p]) % 360
                pr = np.exp(-0.5 * (wrap(b - want) / SIGMA_PROP) ** 2) + 0.02
                c = cands[rng.choice(len(cands), p=pr / pr.sum())]
                off[p] -= T.length[e[p]]
                e[p] = c
                if T.contra[c]:
                    logw[p] += CONTRA_LOGW
        # weigh: change of direction over the last WINDOW s, map vs video
        _, _, brg = T.at(e, off)
        brg_hist = np.roll(brg_hist, -1, axis=1); brg_hist[:, -1] = brg
        if dist[i] > 0 and i >= WINDOW:
            d_video = head[i] - head[i - WINDOW]
            d_map = wrap(brg - brg_hist[:, 0])
            r = wrap(d_map - d_video)
            logw += np.where(np.isnan(r), 0.0, -0.5 * (r / SIGMA_DELTA) ** 2)
        if dist[i] > 0:
            r_abs = wrap((head[i] - align) - brg)
            logw += -0.5 * (r_abs / SIGMA_ABS) ** 2
        # implausible speed
        excess = np.clip(scale * dist[i] - V_MAX, 0, None)       # soft: by how much, not a flat hit
        logw -= 0.5 * (excess / 5.0) ** 2
        # a new stop: cars stop at junctions / lights (or in a queue behind them)
        if stop_start[i]:
            to_junction = T.length[e] - off
            logw += np.log(STOP_FLOOR + np.exp(-0.5 * (to_junction / STOP_SIGMA) ** 2))
        # drift
        align = align + rate * (dist[i] > 0) + rng.normal(0, ALIGN_WALK, N)   # drift only while moving
        rate = rate + rng.normal(0, RATE_WALK, N)
        scale = scale * np.exp(rng.normal(0, SCALE_WALK, N))
        # resample
        w = np.exp(logw - logw.max()); w /= w.sum()
        parent = np.arange(N)
        if 1.0 / np.sum(w ** 2) < N / 2:
            evidence += logw.max() + np.log(np.mean(np.exp(logw - logw.max())))
            parent = systematic(w, rng)
            e, off, scale, align, rate = e[parent], off[parent], scale[parent], align[parent], rate[parent]
            brg_hist = brg_hist[parent]
            logw = np.zeros(N)
        hist_e.append(e.copy()); hist_off.append(off.copy()); hist_parent.append(parent)

    if end_point is not None:
        la, lo, _ = T.at(e, off)
        d = np.array([ox.distance.great_circle(*end_point, a, b) for a, b in zip(la, lo)])
        logw += -0.5 * (d / 200) ** 2
    evidence += logw.max() + np.log(np.mean(np.exp(logw - logw.max())))
    best = int(np.argmax(logw))
    # trace back
    path_e, path_off = [], []
    j = best
    for i in range(len(od) - 1, -1, -1):
        path_e.append(hist_e[i][j]); path_off.append(hist_off[i][j])
        j = hist_parent[i][j]
    path_e, path_off = np.array(path_e[::-1]), np.array(path_off[::-1])
    lat, lon, _ = T.at(path_e, path_off)
    return pd.DataFrame(dict(video_s=od.s, lat=lat, lon=lon, edge=path_e)), float(scale[best]), evidence


def systematic(w: np.ndarray, rng) -> np.ndarray:
    N = len(w)
    pos = (rng.random() + np.arange(N)) / N
    return np.minimum(np.searchsorted(np.cumsum(w), pos), N - 1)


# ── Dead reckoning (video only, for display) ──────────────────────────────────
def dead_reckoning(od: pd.DataFrame, anchor, scale: float, align: float) -> pd.DataFrame:
    lat0, lon0 = anchor
    b = np.radians((od["head"] - align) % 360)
    d = od["dist"] * scale
    north, east = np.cumsum(d * np.cos(b)), np.cumsum(d * np.sin(b))
    return pd.DataFrame(dict(lat=lat0 + north / 111_320,
                             lon=lon0 + east / (111_320 * np.cos(np.radians(lat0)))))


# ── Ground truth + accuracy ───────────────────────────────────────────────────
def truth_line(G, waypoints) -> list[tuple[float, float]]:
    nodes = [min(G.nodes, key=lambda n: ox.distance.great_circle(la, lo, G.nodes[n]["y"], G.nodes[n]["x"]))
             for la, lo in waypoints]
    pts = []
    for a, b in zip(nodes, nodes[1:]):
        path = nx.shortest_path(G, a, b, weight="length")
        for u, v in zip(path, path[1:]):
            d = G.edges[u, v, min(G[u][v], key=lambda k: G[u][v][k]["length"])]
            if "geometry" in d:
                xs, ys = d["geometry"].xy
                pts.extend(zip(ys, xs))
            else:
                pts.extend([(G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])])
    return pts


def accuracy(trace: pd.DataFrame, truth) -> dict:
    t = np.array(truth)
    # densify truth to ~5 m
    dense = [t[0]]
    for a, b in zip(t, t[1:]):
        n = max(1, int(ox.distance.great_circle(*a, *b) / 5))
        dense.extend(a + (b - a) * np.linspace(0, 1, n + 1)[1:, None])
    dense = np.array(dense)
    kx = 111_320 * np.cos(np.radians(dense[:, 0].mean()))
    d = np.array([np.min(np.hypot((dense[:, 0] - la) * 111_320, (dense[:, 1] - lo) * kx))
                  for la, lo in zip(trace.lat, trace.lon)])
    # coverage: share of the true route that the trajectory passes within 30 m of
    # (a route that stops early can be "on the road" yet miss most of the drive)
    tl, to = trace.lat.to_numpy(), trace.lon.to_numpy()
    cover = np.array([np.min(np.hypot((tl - la) * 111_320, (to - lo) * kx)) < 30 for la, lo in dense[::4]])
    return dict(median_m=round(float(np.median(d))), p90_m=round(float(np.percentile(d, 90))),
                within_30m=f"{np.mean(d < 30):.0%}", coverage=f"{cover.mean():.0%}", errors=d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in src/config/drives.yaml")
    ap.add_argument("--seeds", type=int, default=5, help="independent runs; the most probable one is kept")
    a = ap.parse_args()

    drv = drives.load(a.drive)
    folder, cam = drv["folder"], drv["camera"]
    t_from, t_to = drv.get("street_window_s", [0, 1e9])
    lat0, lon0, start = video_anchor(drv["video"])
    od = odometry(folder, cam["hfov_deg"], t_from, t_to)
    G = ox.load_graphml(GRAPH)
    T = EdgeTable(Roads(G))
    runs = []
    for seed in range(a.seeds):
        tr, scale, ev = run_filter(T, od, (lat0, lon0), None, seed)
        runs.append((ev, seed, tr, scale))
        print(f"  seed {seed}: evidence {ev:9.1f}")
    ev, seed, tr, scale = max(runs, key=lambda r: r[0])
    print(f"kept seed {seed} (highest evidence — chosen without looking at the truth)")
    tr["clock"] = [(start + timedelta(seconds=int(s))).strftime("%H:%M:%S") for s in tr.video_s]
    tr["street"] = [street_name(T.roads.data(T.key[e])) for e in tr.edge]

    # video-only shape (no map), oriented like the match over its first minute
    d0 = tr.iloc[: min(60, len(tr))]
    b_map = np.degrees(np.arctan2((d0.lon.iloc[-1] - d0.lon.iloc[0]) * np.cos(np.radians(lat0)),
                                  d0.lat.iloc[-1] - d0.lat.iloc[0])) % 360
    i60 = min(60, len(od) - 1)
    b_vid = np.degrees(np.arctan2(*[np.sum(od["dist"][:i60] * f(np.radians(od["head"][:i60])))
                                    for f in (np.sin, np.cos)])) % 360
    dr = dead_reckoning(od, (lat0, lon0), scale, float(b_vid - b_map))

    truth, stats = None, None
    if drv.get("truth"):
        truth = truth_line(G, [tuple(w) for w in drv["truth"]["waypoints"]])
        stats = accuracy(tr, truth)
        tr["error_m"] = stats.pop("errors").round(1)
        print(f"accuracy vs truth: median {stats['median_m']} m, 90th pct {stats['p90_m']} m, "
              f"within 30 m {stats['within_30m']} of the time, covers {stats['coverage']} of the true route")

    km = float(np.sum(np.hypot(np.diff(tr.lat) * 111_320,
                               np.diff(tr.lon) * 111_320 * np.cos(np.radians(lat0))))) / 1000
    out = dict(drive=a.drive, method="particle filter", seed=seed, evidence=round(ev, 1), start_time=start.isoformat(),
               anchor=[lat0, lon0], street_window_s=[t_from, t_to], scale=scale, km=round(km, 2),
               accuracy=stats, truth=[[round(a, 6), round(b, 6)] for a, b in truth] if truth else None,
               dead_reckoning=dr.round(6).values.tolist(),
               trace=tr.drop(columns="edge").round(6).to_dict(orient="list"))
    (folder / "match.json").write_text(json.dumps(out))
    print(f"matched trajectory {km:.1f} km → {folder / 'match.json'}")


def street_name(d: dict) -> str:
    n = d.get("name", "")
    return ", ".join(n) if isinstance(n, list) else (str(n) or "(unnamed)")


if __name__ == "__main__":
    main()
