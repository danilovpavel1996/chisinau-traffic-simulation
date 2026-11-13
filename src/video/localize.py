"""
localize.py
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
    .venv-video/bin/python -m src.video.localize <drive>     # key in configs/drives.yaml

Output: data/video/processed/<drive>/match.json — per-second trace (lat, lon,
street, clock, error vs truth), true route, video-only shape, accuracy.
View it in the dashboard (src.video.dashboard.build).
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd

from src.video import drives
from src.video.drives import ROOT
from src.video.motion import ego_states
from src.video.roads import Roads, load_graph, street_name, wrap
from src.video.truth import route_from_streets

N_PARTICLES = 4000
RESAMPLE_ESS = 0.1     # resample only when the effective particle count < 10 % (was 50 %):
                       # keeps competing routes alive until later evidence (e.g. a bridge) decides
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
TURN_RESEED = 45.0     # deg in WINDOW s: a clear turn; if few hypotheses can follow it…
RESEED_OK = 0.2        # …(share agreeing within 30°) re-seed RESEED_SHARE of the particles
RESEED_SHARE = 0.25    # at junctions within RESEED_RADIUS where such a turn is possible
RESEED_RADIUS = 1500.0 # m searched AHEAD along the road (distance is undercounted, not overcounted)
RESEED_NEAR = 400.0    # m; nearer matching junctions are preferred (weight exp(−d / RESEED_NEAR))
STRETCH_BACK = 120     # s; a re-seeded route is stretched along the skipped road over at most this
RESEED_COOLDOWN = 20   # s; one re-seed per turn (the 8 s window still holds the turn afterwards,
                       # and repeated re-seeds pushed hypotheses one junction too far)
RESEED_AFTER = 60      # s into the street window; before that hypotheses still surround the GPS
                       # start, and leaving a parking lot produces turns no street explains
RATE_SIGMA = 0.5       # deg/s, prior spread of the heading drift rate
RATE_WALK = 0.03       # deg/s per √s, drift rate may change (street types differ)
ALIGN_WALK = 0.4       # deg/√s random walk of the heading offset (drift)
SCALE_WALK = 0.01     # log-scale random walk per √s: lets the scale adapt to street type (wide vs narrow streets differ ~2×; 0.02 loses the route)
SCALE_PRIOR = 0.35     # log-scale spread of the initial scale around V_TYPICAL
V_MAX = 25.0           # m/s; faster implied speed is implausible in the city
STOP_SIGMA = 40.0      # m; a stop is most likely within this distance before a junction
STOP_FLOOR = 0.2       # mid-block stops (queues) keep this much likelihood
V_TYPICAL = 7.0        # m/s; centre of the scale prior
CONTRA_LOGW = -5.0     # entering an edge against its OSM one-way direction: allowed (tags can be
                       # outdated) but strongly penalised. It was −1 while I believed the driver
                       # had entered Dosoftei against its one-way — the truth was wrong, OSM right.
GUARD_FRAC = 0.04      # guard-rail share of the image above which the video "shows guard rails"
P_GUARD = {True: 0.9, False: 0.15}   # P(guard rails seen | on an OSM bridge / not on a bridge)
SITE_SIGMA = 25.0      # m; a crosswalk under the car is most likely within this of a mapped site
SITE_FLOOR = 0.15      # OSM misses crossings and the detector misses some → never zero
FEATURES = ROOT / "data" / "osm" / "junction_features.parquet"   # OSM crossings + signals
START_RADIUS = 300     # m around the GPS anchor


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
            for v, k, length, coords in lst:
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
        self.sites = self._junction_sites()
        G = roads.G
        deg = {n: len(set(G.successors(n)) | set(G.predecessors(n))) for n in G.nodes}
        self.start_junction = np.array([deg.get(u, 0) >= 3 for u, v, k in self.key])
        self.bridge = np.array([str(roads.data(key).get("bridge", "")) in ("yes", "viaduct", "True")
                                for key in self.key])
        self.first_brg = self.brg[self.start]                      # direction leaving the start node

    def _junction_sites(self) -> list[np.ndarray]:
        """
        Per edge: offsets (m) of junction-like places — the end/start node if it is
        a real junction (≥3 streets), plus OSM crossings / traffic signals projected
        onto the edge (they sit between junctions, which the simplified graph drops).
        """
        from scipy.spatial import cKDTree
        G = self.roads.G
        deg = {n: len(set(G.successors(n)) | set(G.predecessors(n))) for n in G.nodes}
        sites = [[] for _ in self.key]
        for e, (u, v, k) in enumerate(self.key):
            if deg.get(u, 0) >= 3:
                sites[e].append(0.0)
            if deg.get(v, 0) >= 3:
                sites[e].append(float(self.length[e]))
        if FEATURES.exists():
            f = pd.read_parquet(FEATURES)
            kx = 111_320 * np.cos(np.radians(self.lat.mean()))
            tree = cKDTree(np.column_stack([self.lat * 111_320, self.lon * kx]))
            sample_edge = np.repeat(np.arange(len(self.key)), self.n)
            for la, lo in zip(f.lat, f.lon):
                for j in tree.query_ball_point([la * 111_320, lo * kx], r=15.0):
                    e = sample_edge[j]
                    sites[e].append(float((j - self.start[e]) * STEP_M))
        return [np.unique(np.round(s)) if s else np.array([]) for s in sites]

    def dist_to_site(self, e: np.ndarray, off: np.ndarray) -> np.ndarray:
        """Distance (m) from each particle to the nearest junction-like site on its edge."""
        out = np.full(len(e), np.inf)
        for i, (ee, oo) in enumerate(zip(e, off)):
            s = self.sites[ee]
            if len(s):
                out[i] = np.min(np.abs(s - oo))
        return out

    def at(self, e: np.ndarray, off: np.ndarray):
        j = self.start[e] + np.minimum((off / STEP_M).astype(int), self.n[e] - 1)
        return self.lat[j], self.lon[j], self.brg[j]


# ── Odometry ──────────────────────────────────────────────────────────────────
def odometry(folder: Path, hfov: float, t_from: float, t_to: float) -> pd.DataFrame:
    """Per-second video heading (compass sense, + = clockwise) and speed index."""
    meta = json.loads((folder / "meta.json").read_text())
    e = ego_states(pd.read_parquet(folder / "ego.parquet"), meta["width"], hfov, meta["hood"])
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
def guard_rail_seconds(folder: Path) -> dict[int, bool]:
    """Video second → guard rails visible (scene segmentation samples, ~every 2 s)."""
    f = folder / "scene.parquet"
    if not f.exists():
        return {}
    sc = pd.read_parquet(f)
    return {int(t): bool(g > GUARD_FRAC) for t, g in zip(sc.t, sc.guard_rail)}


def crossing_events(folder: Path, min_gap: float = 4.0) -> np.ndarray:
    """
    Video seconds at which the car drives over a crosswalk (segmentation: crosswalk
    pixels reaching the bottom of the above-bonnet image). Empty without scene.parquet.
    """
    f = folder / "scene.parquet"
    if not f.exists():
        return np.array([])
    sc = pd.read_parquet(f).sort_values("t")
    hits = sc[(sc.crosswalk > 0.003) & (sc.crosswalk_bottom > 0.8)].t.to_numpy()
    ev = []
    for t in hits:
        if not ev or t - ev[-1] > min_gap:
            ev.append(t)
    return np.array(ev)


def run_filter(T: EdgeTable, od: pd.DataFrame, anchor, end_point=None, seed: int = 0,
               crossings: np.ndarray = np.array([]), guard: dict | None = None):
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
    crossing_now = np.isin(od.s.to_numpy(), np.floor(crossings).astype(int))
    moved_s = np.inf             # seconds driven since the last crosswalk that was used
    reseeds = []                 # (video second, particles re-seeded)
    jumps = {}                   # step → {particle: [(lat, lon, edge), …] road skipped by re-seeding}
    hist_src = []
    last_reseed = -np.inf
    for i in range(len(od)):
        src = np.arange(N)       # which particle of step i−1 each particle continues
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
            if i >= RESEED_AFTER and i - last_reseed >= RESEED_COOLDOWN and abs(d_video) > TURN_RESEED and np.mean(np.abs(np.nan_to_num(r, nan=180)) < 30) < RESEED_OK:
                placed = reseed(T, e, off, align, scale, rate, logw, brg_hist, head[i], d_video, rng)
                for k, (source, conn) in placed.items():
                    src[k] = source
                jumps[i] = {k: conn for k, (_, conn) in placed.items()}
                reseeds.append((int(od.s.iat[i]), len(placed)))
                last_reseed = i
        if dist[i] > 0:
            r_abs = wrap((head[i] - align) - brg)
            logw += -0.5 * (r_abs / SIGMA_ABS) ** 2
        # implausible speed
        excess = np.clip(scale * dist[i] - V_MAX, 0, None)       # soft: by how much, not a flat hit
        logw -= 0.5 * (excess / 5.0) ** 2
        # a new stop: cars stop at junctions / lights (or in a queue behind them)
        moved_s += dist[i] > 0
        if crossing_now[i] and moved_s >= 3:                  # driving over a crosswalk right now
            d = T.dist_to_site(e, off)                        # (once: standing at one re-detects it)
            logw += np.log(SITE_FLOOR + np.exp(-0.5 * (d / SITE_SIGMA) ** 2))
            moved_s = 0
        g = (guard or {}).get(int(od.s.iat[i]))
        if g is not None and dist[i] > 0:                     # guard rails ↔ OSM bridge (relative evidence)
            on_bridge = T.bridge[e]
            p_bridge, p_other = (P_GUARD[True], P_GUARD[False]) if g else (1 - P_GUARD[True], 1 - P_GUARD[False])
            logw += np.log(np.where(on_bridge, p_bridge, p_other))
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
        if 1.0 / np.sum(w ** 2) < N * RESAMPLE_ESS:
            evidence += logw.max() + np.log(np.mean(np.exp(logw - logw.max())))
            parent = systematic(w, rng)
            e, off, scale, align, rate = e[parent], off[parent], scale[parent], align[parent], rate[parent]
            brg_hist = brg_hist[parent]
            logw = np.zeros(N)
        hist_e.append(e.copy()); hist_off.append(off.copy()); hist_parent.append(parent); hist_src.append(src)

    if end_point is not None:
        la, lo, _ = T.at(e, off)
        d = np.array([ox.distance.great_circle(*end_point, a, b) for a, b in zip(la, lo)])
        logw += -0.5 * (d / 200) ** 2
    evidence += logw.max() + np.log(np.mean(np.exp(logw - logw.max())))
    if reseeds:
        print(f"    re-seeded at turns: {', '.join(f'{t // 60}:{t % 60:02d} ({n})' for t, n in reseeds)}")
    best = int(np.argmax(logw))
    # trace back (k = index before resampling at step i; src[k] = particle it continues)
    path_e, path_off, path_jumps = [], [], []
    j = best
    for i in range(len(od) - 1, -1, -1):
        path_e.append(hist_e[i][j]); path_off.append(hist_off[i][j])
        k = hist_parent[i][j]
        if i in jumps and k in jumps[i]:
            path_jumps.append((i, jumps[i][k]))
        j = hist_src[i][k]
    path_e, path_off = np.array(path_e[::-1]), np.array(path_off[::-1])
    lat, lon, _ = T.at(path_e, path_off)
    tr = pd.DataFrame(dict(video_s=od.s, lat=lat, lon=lon, edge=path_e))
    for i, conn in sorted(path_jumps):
        stretch(tr, i, conn, dist)
    tr.attrs["reseeds"] = reseeds
    return tr, float(scale[best]), evidence


def stretch(tr: pd.DataFrame, i: int, conn: list, dist: np.ndarray):
    """
    A re-seeded hypothesis skipped ahead along the road: the car was really there,
    the distance was undercounted. Re-lay the last ≤STRETCH_BACK s of the route
    along (route so far + skipped road), proportional to measured motion → no jump.
    """
    a = max(0, i - STRETCH_BACK)
    pts = ([(r.lat, r.lon, r.edge) for r in tr.iloc[a:i].itertuples()] + conn
           + [(tr.lat.iat[i], tr.lon.iat[i], tr.edge.iat[i])])
    la, lo, ed = (np.array(x) for x in zip(*pts))
    seg = np.hypot(np.diff(la) * 111_320, np.diff(lo) * 111_320 * np.cos(np.radians(la.mean())))
    arc = np.r_[0, np.cumsum(seg)]
    m = np.cumsum(np.r_[0, dist[a + 1:i + 1]])
    frac = m / m[-1] if m[-1] > 0 else np.linspace(0, 1, len(m))
    target = frac * arc[-1]
    idx = np.clip(np.searchsorted(arc, target), 0, len(arc) - 1)
    tr.loc[tr.index[a:i + 1], "lat"] = np.interp(target, arc, la)
    tr.loc[tr.index[a:i + 1], "lon"] = np.interp(target, arc, lo)
    tr.loc[tr.index[a:i + 1], "edge"] = ed[idx]


def reseed(T: EdgeTable, e, off, align, scale, rate, logw, brg_hist, head_now: float,
           d_video: float, rng, n_sources: int = 30) -> dict:
    """
    Replace the least likely RESEED_SHARE of particles with hypotheses that have
    just made the observed turn. Candidates come from following the road AHEAD of
    likely particles (no jumping across blocks) to junctions where a turn into the
    direction the video now heads exists; nearer junctions are preferred.
    Arrays are modified in place. Returns {particle: (source particle, skipped road)}.
    """
    w = np.exp(logw - logw.max()); w /= w.sum()
    a_med = float(np.degrees(np.angle(np.sum(w * np.exp(1j * np.radians(align)))))) % 360
    want = (head_now - a_med) % 360
    sources = np.unique(rng.choice(len(e), n_sources, p=w))
    pool = []                                   # (weight, source, candidate edge, skipped road)
    for sp in sources:
        for c, d, conn in forward_turns(T, int(e[sp]), float(off[sp]), want):
            pool.append((w[sp] * np.exp(-d / RESEED_NEAR), sp, c, conn))
    if not pool:
        return {}
    pw = np.array([x[0] for x in pool]); pw /= pw.sum()
    k = int(RESEED_SHARE * len(e))
    worst = np.argsort(logw)[:k]
    placed = {}
    for p, choice in zip(worst, rng.choice(len(pool), k, p=pw)):
        _, sp, c, conn = pool[choice]
        e[p], off[p] = c, rng.uniform(0, 15)
        align[p] = a_med + rng.normal(0, 5)
        rate[p], scale[p] = 0.0, scale[sp]
        brg_hist[p] = T.first_brg[c] - d_video          # they just turned by d_video
        placed[int(p)] = (int(sp), conn)
    logw[worst] = np.median(logw)
    return placed


def forward_turns(T: EdgeTable, e0: int, off0: float, want: float, max_turn_on_road: float = 40):
    """
    Follow the road ahead from (e0, off0) — continuing through junctions while the
    direction changes < max_turn_on_road — and yield (edge, metres ahead, skipped
    road as [(lat, lon, edge)]) for every junction exit that turns towards `want`.
    """
    def samples(ee, from_off=0.0):
        j0 = T.start[ee] + min(int(from_off / STEP_M), T.n[ee] - 1)
        j1 = T.start[ee] + T.n[ee]
        return [(T.lat[j], T.lon[j], ee) for j in range(j0, j1)]

    stack = [(e0, T.length[e0] - off0, samples(e0, off0))]
    seen = {e0}
    while stack:
        ee, d, road = stack.pop()
        if d > RESEED_RADIUS:
            continue
        u, v, _ = T.key[ee]
        in_brg = T.brg[T.start[ee] + max(T.n[ee] - 2, 0)]
        for c in T.out.get(v, []):
            if T.key[c][1] == u:
                continue                                       # no U-turn
            turn = abs(wrap(T.first_brg[c] - in_brg))
            if turn > 30 and abs(wrap(T.first_brg[c] - want)) < 35:
                yield c, d, road
            if turn < max_turn_on_road and c not in seen:
                seen.add(c)
                stack.append((c, d + T.length[c], road + samples(c)))


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
def accuracy(trace: pd.DataFrame, truth) -> dict:
    """
    Distance of every matched position to the true route (sampled every 5 m →
    the metric itself has ~2.5 m resolution), and coverage: the share of the
    true route the match passes within 30 m of.
    """
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


def run(drv: dict, seeds: int = 10):
    """Match the drive to the map; keep the most probable of `seeds` runs; write match.json."""
    folder, cam = drv["folder"], drv["camera"]
    t_from, t_to = drv.get("street_window_s", [0, 1e9])
    lat0, lon0, start = drives.anchor(drv)
    od = odometry(folder, cam["hfov_deg"], t_from, t_to)
    crossings = crossing_events(folder)
    print(f"{len(crossings)} crosswalk events from the video" if len(crossings) else "no scene.parquet: crosswalks not used")
    G = load_graph()
    T = EdgeTable(Roads(G))
    runs = []
    for seed in range(seeds):
        tr, scale, ev = run_filter(T, od, (lat0, lon0), None, seed, crossings, guard_rail_seconds(folder))
        runs.append((ev, seed, tr, scale))
        print(f"  seed {seed}: evidence {ev:9.1f}")
    ev, seed, tr, scale = max(runs, key=lambda r: r[0])
    print(f"kept seed {seed} (highest evidence — chosen without looking at the truth)")
    tr["clock"] = drives.clock(drv, tr.video_s).strftime("%H:%M:%S")
    tr["street"] = [street_name(T.roads.data(T.key[e])) for e in tr.edge]

    # video-only shape (no map), oriented like the match over its first minute
    d0 = tr.iloc[: min(60, len(tr))]
    b_map = np.degrees(np.arctan2((d0.lon.iloc[-1] - d0.lon.iloc[0]) * np.cos(np.radians(lat0)),
                                  d0.lat.iloc[-1] - d0.lat.iloc[0])) % 360
    i60 = min(60, len(od) - 1)
    b_vid = np.degrees(np.arctan2(*[np.sum(od["dist"][:i60] * f(np.radians(od["head"][:i60])))
                                    for f in (np.sin, np.cos)])) % 360
    dr = dead_reckoning(od, (lat0, lon0), scale, float(b_vid - b_map))

    truth, stats, truth_kind = None, None, None
    gps = drives.gps_track(drv)
    if gps is not None:                       # dashcam overlay: position at every second
        truth, truth_kind = list(zip(gps.lat, gps.lon)), "GPS (dashcam overlay)"
    elif drv.get("truth"):
        t = drv["truth"]
        truth = route_from_streets(G, (lat0, lon0), t["streets"], tuple(t["end"]))
        truth_kind = "driver's directions"
    if truth:
        stats = accuracy(tr, truth)
        tr["error_m"] = stats.pop("errors").round(1)
        if gps is not None:                   # same moment in time, not just the same road
            la = np.interp(tr.video_s, gps.video_s, gps.lat); lo = np.interp(tr.video_s, gps.video_s, gps.lon)
            d = np.hypot((tr.lat - la) * 111_320, (tr.lon - lo) * 111_320 * np.cos(np.radians(la)))
            tr["error_m"] = d.round(1)
            stats.update(time_median_m=round(float(np.median(d))), time_within_50m=f"{np.mean(d < 50):.0%}")
        print(f"accuracy vs {truth_kind}: median {stats['median_m']} m, 90th pct {stats['p90_m']} m, "
              f"within 30 m {stats['within_30m']} of the time, covers {stats['coverage']} of the true route"
              + (f"; at the same moment: median {stats['time_median_m']} m, within 50 m {stats['time_within_50m']}"
                 if gps is not None else ""))

    km = float(np.sum(np.hypot(np.diff(tr.lat) * 111_320,
                               np.diff(tr.lon) * 111_320 * np.cos(np.radians(lat0))))) / 1000
    out = dict(drive=drv["key"], method="particle filter", seed=seed, evidence=round(ev, 1), start_time=start.isoformat(),
               anchor=[lat0, lon0], street_window_s=[t_from, t_to], scale=scale, km=round(km, 2),
               accuracy=stats, truth_kind=truth_kind,
               truth=[[round(a, 6), round(b, 6)] for a, b in truth] if truth else None,
               reseeds=tr.attrs.get("reseeds", []),
               dead_reckoning=dr.round(6).values.tolist(),
               trace=tr.drop(columns="edge").round(6).to_dict(orient="list"))
    (folder / "match.json").write_text(json.dumps(out))
    print(f"matched trajectory {km:.1f} km → {folder / 'match.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    ap.add_argument("--seeds", type=int, default=10, help="independent runs; the most probable one is kept")
    a = ap.parse_args()
    run(drives.load(a.drive), a.seeds)


if __name__ == "__main__":
    main()
