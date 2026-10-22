"""
event_match.py
--------------
Event-based route matching: which route from START to DESTINATION explains the
sequence of junction events seen in the video?

    1. candidate routes   shortest routes start → destination on the OSM graph (one-way
                          respected), up to MAX_DETOUR × the shortest length
    2. route sites        along each route, in order: real junctions (≥3 streets) with
                          the turn made there (roundabouts collapsed to one manoeuvre
                          with its net angle), OSM traffic signals and crossings
                          (mid-block crossings are sites too)
    3. alignment          sequence alignment (Needleman–Wunsch) of video events with
                          route sites: matches cost little when turn direction/size,
                          crosswalk, signal and roundabout agree and the timing fits;
                          unmatched sites (missed by the video) and unmatched events
                          (false detections) cost according to how detectable they are
    4. result             the cheapest route; every matched site gets a video time, and
                          the position at every second follows by interpolating along
                          the route while the car is moving

Robust to drift by construction: a wrong detection costs one event, it cannot drag
the position off the map.

Usage:
    .venv-video/bin/python -m src.video.event_match <drive>

Output: data/video/processed/<drive>/event_match.json (route, matched sites,
per-second trace, scores; with GPS: junction matching rate).
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.video import drives
from src.video.drives import ROOT
from src.video.motion import ego_states
from src.video.roads import load_graph, street_name, wrap

MAX_DETOUR = 1.5        # candidate routes up to this × the shortest route length
MAX_ROUTES = 300
TURN = 45               # deg; a turn larger than this is a "turn", else straight
SITE_NEAR = 30          # m; OSM signal / crossing within this of a junction belongs to it
FEATURES = ROOT / "data" / "osm" / "junction_features.parquet"

# detection reliability (daylight) → costs
MISS = dict(turn=3.0, roundabout=3.0, signal=1.0, crossing=0.6, junction=0.15)
FALSE = dict(turn=2.5, crosswalk=1.2, signal=0.8, small_turn=0.5)


# ── Map side ──────────────────────────────────────────────────────────────────
class RouteMap:
    def __init__(self, G):
        self.G = G
        D = nx.DiGraph()
        for u, v, k, d in G.edges(keys=True, data=True):
            if not D.has_edge(u, v) or d["length"] < D[u][v]["length"]:
                D.add_edge(u, v, length=d["length"], key=k)
        self.D = D
        self.deg = {n: len(set(G.successors(n)) | set(G.predecessors(n))) for n in G.nodes}
        self.kx = 111_320 * np.cos(np.radians(47.03))
        nodes = list(G.nodes)
        self.nodes = nodes
        self.node_tree = cKDTree([[G.nodes[n]["y"] * 111_320, G.nodes[n]["x"] * self.kx] for n in nodes])
        f = pd.read_parquet(FEATURES)
        self.feat = f
        self.feat_tree = cKDTree(np.column_stack([f.lat * 111_320, f.lon * self.kx]))

    def xy(self, lat, lon):
        return np.array([lat * 111_320, lon * self.kx])

    def nearest_nodes(self, p, k=6, r=250):
        d, i = self.node_tree.query(self.xy(*p), k=k)
        return [self.nodes[j] for dd, j in zip(np.atleast_1d(d), np.atleast_1d(i)) if dd < r]

    def edge(self, u, v):
        return self.G.edges[u, v, self.D[u][v]["key"]]

    def coords(self, u, v):
        d = self.edge(u, v)
        if "geometry" in d:
            xs, ys = d["geometry"].xy
            return list(zip(ys, xs))
        return [(self.G.nodes[u]["y"], self.G.nodes[u]["x"]), (self.G.nodes[v]["y"], self.G.nodes[v]["x"])]

    def candidate_routes(self, start, dest):
        best = []
        for a, b in itertools.product(self.nearest_nodes(start), self.nearest_nodes(dest)):
            try:
                best.append((nx.shortest_path_length(self.D, a, b, weight="length"), a, b))
            except nx.NetworkXNoPath:
                pass
        if not best:
            raise RuntimeError("no route between start and destination on the street graph")
        L0, a, b = min(best)
        routes = []
        for path in nx.shortest_simple_paths(self.D, a, b, weight="length"):
            L = sum(self.D[u][v]["length"] for u, v in zip(path, path[1:]))
            if L > MAX_DETOUR * L0 or len(routes) >= MAX_ROUTES:
                break
            routes.append(path)
        return routes, L0

    def sites(self, path) -> pd.DataFrame:
        """Junctions (with the turn made) and OSM signals / crossings along a path, in order."""
        pts, cum, brg_in, brg_out, node_at = [], [0.0], {}, {}, {}
        for u, v in zip(path, path[1:]):
            c = self.coords(u, v)
            if pts:
                c = c[1:]
            node_at[u] = len(pts) - 1 if pts else 0
            for p in c:
                if pts:
                    a = pts[-1]
                    cum.append(cum[-1] + np.hypot((p[0] - a[0]) * 111_320, (p[1] - a[1]) * self.kx))
                pts.append(p)
        node_at[path[-1]] = len(pts) - 1
        pts = np.array(pts)
        cum = np.array(cum)

        def bearing_at(i, ahead):              # direction over ~25 m before / after point i
            j = i
            while 0 < j < len(pts) - 1 and abs(cum[j] - cum[i]) < 25:
                j += 1 if ahead else -1
            a, b = (pts[i], pts[j]) if ahead else (pts[j], pts[i])
            return np.degrees(np.arctan2((b[1] - a[1]) * self.kx, (b[0] - a[0]) * 111_320)) % 360

        rows, ring = [], None
        for k, n in enumerate(path[1:-1], start=1):
            u, v = path[k - 1], path[k + 1]
            in_ring = self.edge(u, n).get("junction") in ("roundabout", "circular")
            out_ring = self.edge(n, v).get("junction") in ("roundabout", "circular")
            i = node_at[n]
            if out_ring and not in_ring:                  # entering a roundabout
                ring = dict(i=i, b_in=bearing_at(i, False))
                continue
            if in_ring and out_ring:
                continue
            if in_ring and not out_ring and ring:         # leaving it: one manoeuvre, net angle
                mid = (ring["i"] + i) // 2
                rows.append(dict(d=cum[mid], node=n, turn=float(wrap(bearing_at(i, True) - ring["b_in"])),
                                 roundabout=True, junction=True))
                ring = None
                continue
            if self.deg.get(n, 0) >= 3:
                rows.append(dict(d=cum[i], node=n, turn=float(wrap(bearing_at(i, True) - bearing_at(i, False))),
                                 roundabout=False, junction=True))
        s = pd.DataFrame(rows, columns=["d", "node", "turn", "roundabout", "junction"])
        # OSM signals / crossings: attach to a junction within SITE_NEAR, else a site of their own
        s["signal"], s["crossing"] = False, False
        tree = cKDTree(np.column_stack([pts[:, 0] * 111_320, pts[:, 1] * self.kx]))
        near = self.feat_tree.query_ball_point(np.column_stack([pts[:, 0] * 111_320, pts[:, 1] * self.kx]), r=12)
        seen, extra = set(), []
        for idx in set(itertools.chain.from_iterable(near)):
            if idx in seen:
                continue
            seen.add(idx)
            f = self.feat.iloc[idx]
            dd, pi = tree.query(self.xy(f.lat, f.lon))
            d_along = cum[pi]
            col = "signal" if f.kind == "traffic_signals" else "crossing"
            j = (s.d - d_along).abs().idxmin() if len(s) else None
            if j is not None and abs(s.d[j] - d_along) <= SITE_NEAR:
                s.loc[j, col] = True
            else:
                extra.append(dict(d=d_along, node=None, turn=0.0, roundabout=False, junction=False,
                                  signal=col == "signal", crossing=col == "crossing"))
        if extra:                                         # merge nearby extra features into one site
            e = pd.DataFrame(extra).sort_values("d")
            e["g"] = (e.d.diff() > SITE_NEAR).cumsum()
            e = e.groupby("g").agg(d=("d", "mean"), signal=("signal", "any"), crossing=("crossing", "any")).reset_index(drop=True)
            e["node"], e["turn"], e["roundabout"], e["junction"] = None, 0.0, False, False
            s = pd.concat([s, e], ignore_index=True)
        s = s.sort_values("d").reset_index(drop=True)
        s.attrs["length"], s.attrs["pts"], s.attrs["cum"] = float(cum[-1]), pts, cum
        return s


# ── Alignment ─────────────────────────────────────────────────────────────────
def site_kind(s) -> str:
    if s.roundabout:
        return "roundabout"
    if abs(s.turn) > TURN:
        return "turn"
    if s.signal:
        return "signal"
    if s.crossing:
        return "crossing"
    return "junction"


def miss_cost(s) -> float:
    return MISS[site_kind(s)]


def false_cost(e) -> float:
    if abs(e.turn_deg) > TURN:
        return FALSE["turn"]
    if e.crosswalk:
        return FALSE["crosswalk"]
    if e.signal:
        return FALSE["signal"]
    return FALSE["small_turn"]


def match_cost(e, s, m_obs: float, m_exp: float, sigma: float) -> float:
    c = ((m_obs - m_exp) / sigma) ** 2                                  # timing
    et = 0 if abs(e.turn_deg) <= TURN else np.sign(e.turn_deg)
    st = 0 if abs(s.turn) <= TURN else np.sign(s.turn)
    if et == st == 0:
        pass
    elif et == st:
        c += ((abs(e.turn_deg) - abs(s.turn)) / 60) ** 2               # same direction: size check
    elif st == 0:
        c += 2.5                                                         # video turned, map straight
    elif et == 0:
        c += 3.0                                                         # map turns, video saw no turn
    else:
        c += 6.0                                                         # opposite directions
    c += -1.0 if (e.roundabout and s.roundabout) else (0.5 if e.roundabout != s.roundabout else 0)
    if e.crosswalk:
        c += -1.0 if s.crossing else 0.8                                 # OSM misses crossings → mild
    if e.signal:
        c += -1.0 if s.signal else 0.5
    if not s.junction and not e.crosswalk and not e.signal:
        c += 1.0                                                         # a bare crossing site needs a crossing cue
    return c


def align(ev: pd.DataFrame, sites: pd.DataFrame, m_ev: np.ndarray, m_total: float):
    """Needleman–Wunsch: cost and matched pairs (event index, site index)."""
    L = sites.attrs["length"]
    n, m = len(ev), len(sites)
    sigma = 0.12 * m_total                          # 12 % of the whole drive's odometer
    m_exp = sites.d.to_numpy() / L * m_total
    E, S = list(ev.itertuples()), list(sites.itertuples())
    fc = np.array([false_cost(e) for e in E]); mc = np.array([miss_cost(s) for s in S])
    D = np.zeros((n + 1, m + 1)); B = np.zeros((n + 1, m + 1), dtype=np.int8)
    D[1:, 0] = np.cumsum(fc); B[1:, 0] = 1
    D[0, 1:] = np.cumsum(mc); B[0, 1:] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            opts = (D[i - 1, j - 1] + match_cost(E[i - 1], S[j - 1], m_ev[i - 1], m_exp[j - 1], sigma),
                    D[i - 1, j] + fc[i - 1], D[i, j - 1] + mc[j - 1])
            k = int(np.argmin(opts)); D[i, j] = opts[k]; B[i, j] = k
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        k = B[i, j]
        if k == 0:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    return float(D[n, m]), pairs[::-1]


# ── Main ──────────────────────────────────────────────────────────────────────
def odometer(drv: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Cumulative video speed index over the street window — distance up to an unknown
    scale (no GPS). Proportional *distance*, not moving time: creeping in a queue at
    8 km/h and cruising at 50 km/h must not count the same.
    """
    folder, cam = drv["folder"], drv["camera"]
    meta = json.loads((folder / "meta.json").read_text())
    e = ego_states(pd.read_parquet(folder / "ego.parquet"), meta["width"], cam["hfov_deg"], cam["hood"])
    t0, t1 = drv.get("street_window_s", [e.t.min(), e.t.max()])
    e = e[(e.t >= t0) & (e.t <= t1)]
    v = np.where(e.moving, e.speed_idx.fillna(0).clip(lower=0), 0.0)
    v = pd.Series(v).rolling(15, center=True, min_periods=1).median().to_numpy()   # ~3 s: kill spikes
    return e.t.to_numpy(), np.cumsum(v * e.t.diff().fillna(0).to_numpy())


def run(drv: dict):
    t_start = time.time()
    ev = pd.read_parquet(drv["folder"] / "events.parquet")
    tt, mm = odometer(drv)
    m_ev, m_total = np.interp(ev.t, tt, mm), mm[-1]
    R = RouteMap(load_graph())
    routes, L0 = R.candidate_routes(tuple(drv["start"]), tuple(drv["destination"]))
    print(f"{len(ev)} video events · {len(routes)} candidate routes (shortest {L0 / 1000:.2f} km, ≤ {MAX_DETOUR}×) "
          f"· {tt[-1] - tt[0]:.0f} s of video", flush=True)
    scored = []
    for path in routes:
        s = R.sites(path)
        cost, pairs = align(ev, s, m_ev, m_total)
        scored.append((cost, path, s, pairs))
    scored.sort(key=lambda x: x[0])
    cost, path, s, pairs = scored[0]

    # per-second position: anchors (video time, distance along route) from matched sites
    L = s.attrs["length"]
    anchors = [(tt[0], 0.0)] + [(ev.t.iat[i], s.d.iat[j]) for i, j in pairs] + [(tt[-1], L)]
    anchors = sorted(anchors)
    a_t, a_d = np.array([a[0] for a in anchors]), np.maximum.accumulate([a[1] for a in anchors])
    secs = np.arange(int(tt[0]), int(tt[-1]) + 1)
    m_s = np.interp(secs, tt, mm); m_a = np.interp(a_t, tt, mm)
    d_s = np.interp(m_s, m_a, a_d)                     # between anchors, distance follows the odometer
    pts, cum = s.attrs["pts"], s.attrs["cum"]
    lat, lon = np.interp(d_s, cum, pts[:, 0]), np.interp(d_s, cum, pts[:, 1])
    names = []
    for u, v in zip(path, path[1:]):
        n = street_name(R.edge(u, v))
        if not names or names[-1] != n:
            names.append(n)
    matched = [dict(t=float(ev.t.iat[i]), clock=ev.clock.iat[i], cues=ev.cues.iat[i], video_turn=int(ev.turn_deg.iat[i]),
                    site_kind=site_kind(s.iloc[j]), map_turn=round(float(s.turn.iat[j])), d=round(float(s.d.iat[j])),
                    node=None if pd.isna(s.node.iat[j]) else int(s.node.iat[j])) for i, j in pairs]
    out = dict(method="event-based route matching", cost=round(cost, 2), routes_considered=len(routes),
               route_km=round(L / 1000, 2), streets=[n for n in names if not n.startswith("(")],
               events=len(ev), sites=len(s), matched=matched,
               trace=dict(video_s=secs.tolist(), lat=np.round(lat, 6).tolist(), lon=np.round(lon, 6).tolist()),
               runner_up_cost=round(scored[1][0], 2) if len(scored) > 1 else None)
    out["evaluation"] = evaluate(drv, R, path, matched, secs, lat, lon)
    (drv["folder"] / "event_match.json").write_text(json.dumps(out, default=float))
    print(f"best route: {L / 1000:.2f} km, cost {cost:.1f} (runner-up {out['runner_up_cost']}); "
          f"{len(pairs)}/{len(ev)} events matched to {len(s)} route sites")
    print("  " + " → ".join(n.removeprefix("Strada ") for n in out["streets"]))
    for k, v in out["evaluation"].items():
        print(f"  {k}: {v}")
    print(f"  ({time.time() - t_start:.0f} s)")
    return out


def evaluate(drv, R: RouteMap, path, matched, secs, lat, lon) -> dict:
    """Against GPS (junction matching rate, position error) or the driver's directions (route)."""
    res = {}
    gps = drives.gps_track(drv)
    if gps is not None:
        la = np.interp(secs, gps.video_s, gps.lat); lo = np.interp(secs, gps.video_s, gps.lon)
        err = np.hypot((lat - la) * 111_320, (lon - lo) * R.kx)
        res["position error at the same moment"] = f"median {np.median(err):.0f} m, within 50 m {np.mean(err < 50):.0%}"
        # true junctions passed: GPS within 15 m of a ≥3-street node
        J = [n for n in R.nodes if R.deg.get(n, 0) >= 3]
        jt = cKDTree([[R.G.nodes[n]["y"] * 111_320, R.G.nodes[n]["x"] * R.kx] for n in J])
        dist, idx = jt.query(np.column_stack([la * 111_320, lo * R.kx]))
        true, last = [], None
        for k in np.flatnonzero(dist < 15):
            n = J[idx[k]]
            if n != last:
                true.append((secs[k], n)); last = n
        on_route = set(path)
        got = [t for t, n in true if any(m["node"] == n and abs(m["t"] - t) <= 8 for m in matched)]
        on = [t for t, n in true if n in on_route]
        res["true junctions passed (GPS)"] = len(true)
        res["junctions on the chosen route"] = f"{len(on)}/{len(true)} ({len(on) / max(len(true), 1):.0%})"
        res["junction matching rate (right junction, ±8 s)"] = f"{len(got)}/{len(true)} ({len(got) / max(len(true), 1):.0%})"
    elif drv.get("truth"):
        from src.video.localize import accuracy
        from src.video.truth import route_from_streets
        truth = route_from_streets(R.G, tuple(drv["start"]), drv["truth"]["streets"], tuple(drv["truth"]["end"]))
        a = accuracy(pd.DataFrame(dict(lat=lat, lon=lon)), truth)
        res["route vs driver's directions"] = (f"median {a['median_m']} m, within 30 m {a['within_30m']}, "
                                               f"covers {a['coverage']} of the true route")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    run(drives.load(ap.parse_args().drive))


if __name__ == "__main__":
    main()
