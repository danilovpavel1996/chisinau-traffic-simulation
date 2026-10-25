"""
route_search.py
---------------
Candidate routes found from the video's events, not from route lengths.

event_match.py lists routes start → destination shortest first, up to 1.5× the shortest. A
real drive can be a long detour: 2026_10_24_17_08 drove 13.2 km where the shortest route is
8.1 km (1.63×) — and between those lengths there are far too many routes to list, so the true
one was never scored (13 of 54 junctions on the chosen route).

Here the route is built junction by junction along the street graph, guided by the video:

    events     the video's junction events (events.py), each at a metric distance driven
               (speed.py: flow × metric depth; 5.14 km vs 5.13 km by GPS on 2026_10_23_10_06)
    states     at event j the car makes a manoeuvre at a junction v (in from u, out to w);
               candidates are the junctions reachable from the previous state's position at
               about the distance driven since then (Dijkstra from there)
    cost       event_match's costs: the manoeuvre vs what the video saw (turn direction and
               size, signal, crosswalk, side opening), the distance along the streets vs the
               distance driven (REL_SIGMA, relative to the stretch), junctions passed on the
               way with no event (a missed turn costs a lot, a missed straight junction little),
               events left unmatched (false detections)
    search     Viterbi with a beam: the best BEAM states per event; up to LOOKBACK events may be
               skipped between two matches; at the end the way to the destination pin

The driven distance is metric but not exact (0.84–1.19× GPS across the drives), so the search
runs with a few overall scale factors (SCALES) and pools the routes. The routes found go to
event_match's scoring together with the shortest ones: this only proposes, it does not decide.

Only video + map: no GPS.
"""

from __future__ import annotations

import heapq
from collections import namedtuple

import numpy as np
import pandas as pd

from src.video.event_match import (ABS_SIGMA_M, FALSE, MISS, REL_SIGMA, SITE_NEAR, TURN, RouteMap,
                                   false_cost, match_cost)
from src.video.roads import wrap

BEAM = 150              # states kept per event
LOOKBACK = 4            # a match may follow up to this many events back (the ones between unmatched)
SLACK = 1.35            # search radius: this × the distance driven since the previous match …
SLACK_M = 150           # … + this
SCALES = (0.85, 1.0, 1.18)
N_ROUTES = 40           # distinct routes returned per scale
BEARING_M = 25          # direction of a street at a junction: over its first / last 25 m

Site = namedtuple("Site", "turn roundabout junction signal crossing")


class Graph:
    """Street-graph helpers with caches: bearings at junctions, OSM features at nodes."""

    def __init__(self, R: RouteMap):
        self.R, self.D = R, R.D
        self._b, self._feat = {}, {}

    def length(self, u, v) -> float:
        return self.D[u][v]["length"]

    def bearing(self, u, v, at_end: bool) -> float:
        """Direction of edge u → v near v (at_end) or near u."""
        key = (u, v, at_end)
        if key not in self._b:
            c = np.array(self.R.coords(u, v))
            if at_end:
                c = c[::-1]
            seg = np.hypot(np.diff(c[:, 0]) * 111_320, np.diff(c[:, 1]) * self.R.kx)
            k = int(np.searchsorted(np.cumsum(seg), BEARING_M)) + 1
            a, b = c[0], c[min(k, len(c) - 1)]
            if at_end:
                a, b = b, a
            self._b[key] = np.degrees(np.arctan2((b[1] - a[1]) * self.R.kx, (b[0] - a[0]) * 111_320)) % 360
        return self._b[key]

    def turn(self, u, v, w) -> float:
        return float(wrap(self.bearing(v, w, False) - self.bearing(u, v, True)))

    def features(self, v) -> tuple[bool, bool]:
        if v not in self._feat:
            n = self.R.G.nodes[v]
            idx = self.R.feat_tree.query_ball_point(self.R.xy(n["y"], n["x"]), r=SITE_NEAR)
            kinds = set(self.R.feat.kind.iloc[idx]) if idx else set()
            self._feat[v] = ("traffic_signals" in kinds, bool(kinds - {"traffic_signals"}))
        return self._feat[v]

    def site(self, u, v, w) -> Site:
        ring = lambda a, b: self.R.edge(a, b).get("junction") in ("roundabout", "circular")
        signal, crossing = self.features(v)
        return Site(self.turn(u, v, w), ring(u, v) or ring(v, w), True, signal, crossing)

    def junction(self, v) -> bool:
        return self.R.deg.get(v, 0) >= 3

    def reach(self, prev, x, cutoff: float) -> tuple[dict, dict]:
        """Dijkstra from x (not straight back to prev) up to cutoff metres: distances and predecessors."""
        dist, pred = {x: 0.0}, {x: prev}
        heap = [(0.0, x)]
        while heap:
            d, n = heapq.heappop(heap)
            if d > dist.get(n, np.inf) or d > cutoff:
                continue
            for m in self.D.successors(n):
                if m == pred[n]:                       # no U-turn on the spot
                    continue
                nd = d + self.length(n, m)
                if nd < dist.get(m, np.inf) and nd <= cutoff:
                    dist[m], pred[m] = nd, n
                    heapq.heappush(heap, (nd, m))
        return dist, pred


def path_to(pred: dict, x, v) -> list:
    """Nodes from x to v along the Dijkstra predecessors (x first)."""
    out = [v]
    while out[-1] != x:
        out.append(pred[out[-1]])
    return out[::-1]


def miss_along(g: Graph, before, nodes: list) -> float:
    """Junctions passed with no event: inner nodes of `nodes` (entered from `before`)."""
    c = 0.0
    seq = [before] + nodes
    for k in range(1, len(seq) - 1):
        n = seq[k]
        if g.junction(n) and seq[k - 1] is not None:
            c += MISS["turn"] if abs(g.turn(seq[k - 1], n, seq[k + 1])) > TURN else MISS["junction"]
    return c


def timing(d_odo: float, d_map: float) -> float:
    sigma = REL_SIGMA * max(d_odo, d_map) + ABS_SIGMA_M
    return ((d_odo - d_map) / sigma) ** 2


def search(R: RouteMap, ev: pd.DataFrame, o: np.ndarray, total: float, start, dest, g: Graph | None = None) -> list[tuple[float, list]]:
    """Best routes (cost, node path from the start pin's head node to the destination pin's tail node).
    o: metric distance driven at each event; total: at the destination."""
    g = g or Graph(R)
    E = list(ev.itertuples())
    n = len(E)
    fc = np.array([false_cost(e) for e in E])
    FC = np.r_[0, np.cumsum(fc)]                      # FC[i] = false costs of events < i
    # start: the pin on edge (a, b) heading to b; at b the odometer reads the pin → b distance
    R.pin_nodes(start, True)
    starts = []
    for b in R.pin_nodes(start, True):
        c, a = R.partial_edge(start, b, True)
        d0 = float(np.sum(np.hypot(np.diff(np.array(c)[:, 0]) * 111_320, np.diff(np.array(c)[:, 1]) * R.kx)))
        starts.append(dict(cost=0.0, prev=a, x=b, dx=d0, i=-1, nodes=[b], back=None))
    ends = {}
    for a in R.pin_nodes(dest, False):
        c, b = R.partial_edge(dest, a, False)
        ends[a] = float(np.sum(np.hypot(np.diff(np.array(c)[:, 0]) * 111_320, np.diff(np.array(c)[:, 1]) * R.kx)))

    layers: list[list[dict]] = []
    reach_cache: dict = {}

    def reach(s, cutoff):
        key = (s["prev"], s["x"])
        hit = reach_cache.get(key)
        if hit is None or hit[0] < cutoff:
            reach_cache[key] = (cutoff, *g.reach(s["prev"], s["x"], cutoff))
        return reach_cache[key][1], reach_cache[key][2]

    for j in range(n):
        e = E[j]
        best: dict = {}
        sources = [(s, -1) for s in starts] if j < LOOKBACK else []
        for i in range(max(0, j - LOOKBACK), j):
            sources += [(s, i) for s in layers[i]]
        for s, i in sources:
            drive = o[j] - s["dx"]
            if drive < -ABS_SIGMA_M:
                continue
            cutoff = max(drive, 0) * SLACK + SLACK_M
            dist, pred = reach(s, cutoff)
            skipped = FC[j] - FC[i + 1]
            for v, L in dist.items():
                if v == s["x"] and s["prev"] is None:
                    continue
                if not g.junction(v) or abs(L - drive) > REL_SIGMA * max(L, drive) * 2.5 + 2 * ABS_SIGMA_M:
                    continue
                u = pred[v]
                if u is None:
                    continue
                nodes = path_to(pred, s["x"], v)
                base = s["cost"] + skipped + timing(drive, L) + miss_along(g, s["prev"], nodes[:-1] + [v])
                for w in g.D.successors(v):
                    if w == u:
                        continue
                    c = base + match_cost(e, g.site(u, v, w))
                    key = (v, w)
                    if c < best.get(key, (np.inf,))[0]:
                        best[key] = (c, dict(cost=c, prev=v, x=w, dx=o[j] + g.length(v, w), i=j,
                                             nodes=nodes[1:] + [w], back=s))
        layers.append([st for _, st in sorted(best.values(), key=lambda z: z[0])[:BEAM]])

    # to the destination: from any state of the last LOOKBACK events, or the start
    finals = []
    sources = [s for i in range(max(0, n - LOOKBACK), n) for s in layers[i]] + (starts if n <= LOOKBACK else [])
    for s in sources:
        drive = total - s["dx"]
        cutoff = max(drive, 0) * SLACK + SLACK_M
        dist, pred = reach(s, cutoff)
        for a, tail in ends.items():
            if a not in dist:
                continue
            L = dist[a] + tail
            nodes = path_to(pred, s["x"], a)
            c = (s["cost"] + (FC[n] - FC[s["i"] + 1]) + timing(drive, L)
                 + miss_along(g, s["prev"], nodes[:-1] + [a]))
            finals.append((c, s, nodes[1:]))
    finals.sort(key=lambda z: z[0])
    routes, seen = [], set()
    for c, s, tail_nodes in finals:
        path, cur = list(tail_nodes), s
        while cur is not None:
            path = cur["nodes"] + path
            cur = cur["back"]
        path = [p for k, p in enumerate(path) if k == 0 or p != path[k - 1]]
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        routes.append((float(c), path))
        if len(routes) >= N_ROUTES:
            break
    return routes


def metric_odometer(drv: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Metric distance driven over the street window, from speed.py (None if not run)."""
    f = drv["folder"] / "speed.parquet"
    if not f.exists():
        return None
    s = pd.read_parquet(f).dropna(subset=["kmh_raw"])
    t0, t1 = drv.get("street_window_s", [s.t.min(), s.t.max()])
    s = s[(s.t >= t0) & (s.t <= t1)]
    v = s.kmh_raw.clip(lower=0).to_numpy() / 3.6
    return s.t.to_numpy(), np.cumsum(v * s.t.diff().fillna(0).to_numpy())


def event_routes(drv: dict, R: RouteMap, ev: pd.DataFrame) -> list[list]:
    """Routes proposed by the event-guided search, pooled over SCALES (empty without speed.py)."""
    odo = metric_odometer(drv)
    if odo is None:
        return []
    tt, mm = odo
    o_ev = np.interp(ev.t, tt, mm)
    g = Graph(R)
    found, seen = [], set()
    for k in SCALES:
        for c, path in search(R, ev, o_ev * k, mm[-1] * k, tuple(drv["start"]), tuple(drv["destination"]), g):
            if tuple(path) not in seen:
                seen.add(tuple(path))
                found.append(path)
    return found
