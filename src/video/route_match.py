"""
route_match.py
--------------
Reconstruct the driven route of a GPS-less video from:
  - one anchor: the GPS point + timestamp where recording started (iPhone
    writes both into the .MOV metadata)
  - the sequence of ego turns (direction, approximate angle, time) and the
    ego-motion signal, both from analyze.py

Method — beam search over the OSM drivable graph:
  between two detected turns the car goes "straight" (follows the road
  through intersections, bends < STRAIGHT_DEG); at some intersection it takes
  a turn whose direction/angle must match the next detected turn.
  Every candidate path is scored by
      angle mismatch            Σ ((detected − map angle) / 30°)²
      skipped detections        a detected "turn" with no map turn (lane change,
                                parking manoeuvre, road curve) costs SKIP_COST
      distance consistency      distance on the map between turns ∝ ∫ ego-motion
                                over the same time: one scale k must fit all
  The best paths are kept per step; the winner also gives position at every
  second (cumulative ego-motion × k along the path).

Usage:
    .venv-video/bin/python -m src.video.route_match data/video/processed/<name> \
        --video data/video/raw/<name>.MOV [--hfov 70] [--from 95 --to 850]
        [--end 47.0,28.8]

Outputs (same folder): route_candidates.geojson, trace.csv (per-second
lat/lon/clock time for the best route), route_map.html
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd

from src.video.analyze import HFOV_DEG, ego_states, stops, turns

GRAPH = Path("data/osm/chisinau_drive.graphml")

STRAIGHT_DEG = 35     # a direction change below this at a junction counts as "straight"
BEND_MAX = 70         # deg; max heading change of "straight" map driving between two
                      # detections (stops chains of gentle ramp bends adding up to a turn
                      # the camera never saw)
ANGLE_TOL = 30        # deg, scale of the angle-mismatch penalty
SKIP_COST = 1.5       # cost of treating a detected turn as a false detection
SMALL_TURN = 30       # detected turns below this (deg) are curves / lane changes: cheap to skip
V_MAX = 22.0          # m/s, upper bound on speed between turns (~80 km/h)
DIST_WEIGHT = 2.0     # weight of the distance-consistency term
BEAM = 150            # states kept per "detections consumed" bucket
V_TYPICAL = 7.0       # m/s (25 km/h): centre of the speed prior while moving
V_SIGMA = 0.6         # log-scale spread of the speed prior (≈ ×0.55 … ×1.8 is cheap)
ONEWAY_COST = 1.0     # driving against an OSM one-way: allowed (map tags can be wrong or
                      # outdated) but it must be supported by the video to win
END_RADIUS = 250      # m; end-point penalty scale ((distance / END_RADIUS)²)


# ── Video anchor ──────────────────────────────────────────────────────────────
def video_anchor(video: Path) -> tuple[float, float, datetime]:
    """Start GPS point and local start time from QuickTime metadata."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags",
                          "-of", "json", str(video)], capture_output=True, text=True, check=True)
    tags = json.loads(out.stdout)["format"]["tags"]
    iso = tags["com.apple.quicktime.location.ISO6709"]          # +47.0586+028.8141+133.146/
    nums = [float(x) for x in iso.replace("+", " +").replace("-", " -").split()[:2]]
    start = datetime.fromisoformat(tags["com.apple.quicktime.creationdate"])
    return nums[0], nums[1], start


# ── Graph helpers ─────────────────────────────────────────────────────────────
def bearing(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def end_bearing(coords, reverse: bool = False, span: float = 25.0) -> float:
    """
    Direction of a road at its start, measured over its first ~`span` metres:
    the single first geometry segment can be 2–3 m long and 20° off.
    With reverse=True (coords given end-first) it is the direction at the end.
    """
    a, d, b = coords[0], 0.0, coords[1]
    for p, q in zip(coords, coords[1:]):
        d += ox.distance.great_circle(*p, *q)
        b = q
        if d >= span:
            break
    h = bearing(*a, *b)
    return (h + 180) % 360 if reverse else h


def wrap(a: float) -> float:
    """Angle difference to (-180, 180]; positive = clockwise = right turn."""
    return (a + 180) % 360 - 180


class Roads:
    """
    Graph wrapper. A move is (v, edges, length, b_out, turn): normally one edge;
    entering a roundabout it is the whole ring arc to one exit plus that exit,
    and `turn` is the NET heading change entry → exit (what the camera sees as
    a right-left-right wiggle). Without this, each ring segment was judged on
    its own and the matcher drove past the right exit.
    """

    def __init__(self, G):
        self.G = G
        self.out = {}   # node -> list of (v, key, length, b_in, b_out, coords)
        self.ring = {(u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
                     if d.get("junction") in ("roundabout", "circular")}
        self.contra = set()  # (u, v, k) = one-way edge u←v driven against its OSM direction
        for u, v, k, d in G.edges(keys=True, data=True):
            if "geometry" in d:
                xs, ys = d["geometry"].xy
                coords = list(zip(ys, xs))
            else:
                coords = [(G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])]
            self._add(u, v, k, d["length"], coords)
            oneway = str(d.get("oneway", "False")) == "True"
            if oneway and (u, v, k) not in self.ring and not G.has_edge(v, u):
                self._add(v, u, k, d["length"], coords[::-1])
                self.contra.add((v, u, k))

    def _add(self, u, v, k, length, coords):
        b_in = end_bearing(coords)                       # heading when entering the edge
        b_out = end_bearing(coords[::-1], reverse=True)  # heading when leaving it
        self.out.setdefault(u, []).append((v, k, length, b_in, b_out, coords))

    def length(self, edge) -> float:
        u, v, k = edge
        return next(e[2] for e in self.out[u] if e[0] == v and e[1] == k)

    def data(self, edge) -> dict:
        """OSM attributes of an edge (of the original one for a contra-flow edge)."""
        u, v, k = edge
        return self.G.edges[(v, u, k) if edge in self.contra else edge]

    def _ring_exits(self, node, first):
        """Walk a roundabout from its first ring edge; yield (arc edges, arc length, exit edge)."""
        arc, length, u, edge = [], 0.0, node, first
        for _ in range(50):
            v, k, ln = edge[0], edge[1], edge[2]
            arc.append((u, v, k)); length += ln; u = v
            nxt_ring = None
            for e in self.out.get(u, []):
                if (u, e[0], e[1]) in self.ring:
                    nxt_ring = e
                else:
                    yield list(arc), length, e
            if nxt_ring is None or u == node:
                return
            edge = nxt_ring

    def moves(self, node, heading, came_from):
        """Possible moves from node with their turn angle relative to the current heading."""
        for e in self.out.get(node, []):
            v, k, length, b_in, b_out, coords = e
            if v == came_from and len(self.out.get(node, [])) > 1:
                continue  # no U-turns back along the same street
            if (node, v, k) in self.ring and heading is not None:
                for arc, arc_len, (xv, xk, xlen, xb_in, xb_out, _) in self._ring_exits(node, e):
                    if xv == came_from:
                        continue  # exiting back where we came from = U-turn
                    x_edge = (arc[-1][1], xv, xk)
                    yield (xv, arc + [x_edge], arc_len + xlen, xb_out, wrap(xb_out - heading),
                           ONEWAY_COST if x_edge in self.contra else 0.0)
                continue
            turn = 0.0 if heading is None else wrap(b_in - heading)
            yield (v, [(node, v, k)], length, b_out, turn,
                   ONEWAY_COST if (node, v, k) in self.contra else 0.0)


# ── Observations ──────────────────────────────────────────────────────────────
def observations(folder: Path, hfov: float, t_from: float | None, t_to: float | None):
    """
    Turns and ego-motion inside [t_from, t_to]. Outside that window (phone being
    mounted, manoeuvring in a parking lot) nothing is on the street graph.
    """
    meta = json.loads((folder / "meta.json").read_text())
    e = ego_states(pd.read_parquet(folder / "ego.parquet"), meta["width"], hfov, meta.get("hood", 0.15))
    t_from = e.t.min() if t_from is None else t_from
    t_to = e.t.max() if t_to is None else t_to
    tn = turns(e)
    tn = tn[(tn.t_start >= t_from) & (tn.t_end <= t_to)].reset_index(drop=True)
    tn["t_mid"] = (tn.t_start + tn.t_end) / 2
    tn["signed"] = np.where(tn.direction == "right", 1, -1) * tn.angle_deg_approx
    dt = e.t.diff().fillna(0)
    inside = (e.t >= t_from) & (e.t <= t_to)
    e["moving"] = e.moving & inside
    e["motion"] = np.where(e.moving, e.speed_idx.fillna(0) * dt, 0.0)
    e["cum_motion"] = e.motion.cumsum()
    return meta, e, tn, t_from, t_to


class Motion:
    """(∫ ego-motion, moving seconds) between two video times, via running totals."""

    def __init__(self, e: pd.DataFrame):
        dt = e.t.diff().fillna(0).to_numpy()
        self.t = e.t.to_numpy()
        self.motion = np.concatenate([[0], np.cumsum(e.motion.to_numpy())])
        self.moving = np.concatenate([[0], np.cumsum(dt * e.moving.to_numpy())])

    def __call__(self, t0: float, t1: float) -> tuple[float, float]:
        i0, i1 = np.searchsorted(self.t, [t0, t1], side="right")
        return float(self.motion[i1] - self.motion[i0]), float(self.moving[i1] - self.moving[i0])


# ── Beam search ───────────────────────────────────────────────────────────────
@dataclass(order=True)
class State:
    cost: float
    node: int = field(compare=False)
    heading: float | None = field(compare=False)
    came_from: int | None = field(compare=False)
    ev: int = field(compare=False)                 # index of the next detected turn
    t_last: float = field(compare=False)           # video time of the last matched event
    dist: float = field(compare=False)             # metres since the last matched event
    segs: tuple = field(compare=False)             # ((metres, motion), …) per matched segment
    path: tuple = field(compare=False)             # ((u, v, key), …)
    matched: tuple = field(compare=False)          # ((event idx, node, map angle), …)
    bend: float = field(default=0.0, compare=False)  # map heading change since the last detection


def dist_cost(segs) -> float:
    """
    segs: ((metres, motion, moving seconds), …) between consecutive matched turns.
    1) one scale k (metres per motion unit) should fit every segment: penalise
       the spread of log(metres / motion);
    2) speed prior: metres / moving seconds should look like city driving.
       Without it, ending the route early (skipping all turns) was free.
    Short segments are clamped, not exempt.
    """
    r = [math.log(max(m, 15) / f) for m, f, _ in segs if f > 0.5]
    spread = DIST_WEIGHT * float(np.var(r)) * len(r) if len(r) >= 2 else 0.0
    prior = sum((math.log(max(m, 15) / mv / V_TYPICAL) / V_SIGMA) ** 2
                for m, _, mv in segs if mv > 10)
    return spread + 0.5 * prior


def search(roads: Roads, start_nodes, tn: pd.DataFrame, e: pd.DataFrame,
           t_from: float, t_end: float, end_point=None):
    """
    Beam search. States are bucketed by how many detections they have consumed
    and the best BEAM of each bucket survive, so routes that are still driving
    straight don't crowd out routes that already matched turns. Cost only —
    no reward per matched turn (that made forced loops look good).
    """
    events = tn.to_dict("records")
    motion_between = Motion(e)
    beam = [State(0.0, n, None, None, 0, t_from, 0.0, (), (), ()) for n in start_nodes]
    finished = {}

    for _ in range(10_000):
        nxt = []
        for s in beam:
            done = s.ev >= len(events)
            t_next = t_end if done else events[s.ev]["t_mid"]
            _, moving_s = motion_between(s.t_last, t_next)
            budget = V_MAX * moving_s + 80

            if done:                                  # every detection consumed
                key = (s.node, s.came_from)
                if key not in finished or s.cost < finished[key].cost:
                    finished[key] = s
            else:                                     # this detection is noise, not a junction turn
                cost = SKIP_COST * (0.3 if events[s.ev]["angle_deg_approx"] < SMALL_TURN else 1)
                nxt.append(State(s.cost + cost, s.node, s.heading, s.came_from, s.ev + 1,
                                 s.t_last, s.dist, s.segs, s.path, s.matched, 0.0))
                # ↑ the camera rotated here but it is not a junction turn (parking lot,
                #   lane change, wiggle) → re-sync: map and camera may differ by it

            for v, edges, length, b_out, turn, penalty in roads.moves(s.node, s.heading, s.came_from):
                if s.dist + length > budget:
                    continue
                prev = edges[-1][0]
                if s.heading is None or abs(turn) < STRAIGHT_DEG:      # continue straight
                    bend = 0.0 if s.heading is None else s.bend + wrap(b_out - s.heading)
                    if abs(bend) > BEND_MAX:
                        continue  # the map turns here but the camera saw no turn
                    nxt.append(State(s.cost + penalty, v, b_out, prev, s.ev, s.t_last, s.dist + length,
                                     s.segs, s.path + tuple(edges), s.matched, bend))
                    continue
                if done:
                    continue
                ev = events[s.ev]
                if np.sign(turn) != np.sign(ev["signed"]):
                    continue
                motion, mv = motion_between(s.t_last, ev["t_mid"])
                segs = s.segs + ((s.dist, motion, mv),)
                c = (s.cost + penalty + ((abs(turn) - ev["angle_deg_approx"]) / ANGLE_TOL) ** 2
                     + dist_cost(segs) - dist_cost(s.segs))
                nxt.append(State(c, v, b_out, prev, s.ev + 1, ev["t_mid"], length, segs,
                                 s.path + tuple(edges),
                                 s.matched + ((s.ev, s.node, round(turn)),),
                                 wrap(b_out - s.heading) - turn))  # bend within the turn edge
        if not nxt:
            break
        best = {}
        for st in sorted(nxt):                        # same position + progress → cheapest
            best.setdefault((st.node, st.came_from, st.ev), st)
        buckets = {}
        for st in sorted(best.values()):
            buckets.setdefault(st.ev, []).append(st)
        beam = [st for b in buckets.values() for st in b[:BEAM]]

    out = list(finished.values())
    for st in out:                                    # final straight segment + end anchor
        motion, mv = motion_between(st.t_last, t_end)
        st.segs = st.segs + ((st.dist, motion, mv),)
        st.cost += dist_cost(st.segs) - dist_cost(st.segs[:-1])
        if end_point is not None:
            G = roads.G
            d = ox.distance.great_circle(*end_point, G.nodes[st.node]["y"], G.nodes[st.node]["x"])
            st.cost += (d / END_RADIUS) ** 2
    return sorted(out)


# ── Outputs ───────────────────────────────────────────────────────────────────
def path_coords(roads: Roads, path) -> list[tuple[float, float]]:
    pts = []
    for u, v, k in path:
        for vv, kk, _, _, _, coords in roads.out[u]:
            if vv == v and kk == k:
                pts.extend(coords if not pts else coords[1:])
                break
    return pts


def trace(coords, e: pd.DataFrame, start: datetime, segs) -> pd.DataFrame:
    """Per-second position: cumulative ego-motion × k, laid along the path."""
    ks = [m / f for m, f, _ in segs if m > 30 and f > 1]
    k = float(np.median(ks)) if ks else 1.0
    seg_len = [ox.distance.great_circle(*a, *b) for a, b in zip(coords, coords[1:])]
    cum = np.concatenate([[0], np.cumsum(seg_len)])
    sec = e.assign(sec=e.t.astype(int)).groupby("sec").cum_motion.last()
    d = np.clip((sec - sec.iloc[0]) * k, 0, cum[-1])
    lat = np.interp(d, cum, [c[0] for c in coords])
    lon = np.interp(d, cum, [c[1] for c in coords])
    return pd.DataFrame(dict(video_s=sec.index, clock=[(start + timedelta(seconds=int(s))).strftime("%H:%M:%S")
                                                        for s in sec.index],
                             dist_m=d.round(1).to_numpy(), lat=lat, lon=lon,
                             speed_kmh=(np.gradient(d) * 3.6).round(1)))


def street_list(roads, path, tr: pd.DataFrame) -> pd.DataFrame:
    """Collapse the edge path into consecutive named streets with entry clock time."""
    rows, d = [], 0.0
    for edge in path:
        ed = roads.data(edge)
        name = ed.get("name", "(unnamed)")
        name = ", ".join(name) if isinstance(name, list) else str(name)
        if not rows or rows[-1]["street"] != name:
            i = int(np.searchsorted(tr.dist_m.to_numpy(), d))
            rows.append(dict(street=name + (" (against OSM one-way)" if edge in roads.contra else ""),
                             enter=tr.clock.iloc[min(i, len(tr) - 1)], length_m=0.0))
        rows[-1]["length_m"] += ed["length"]
        d += ed["length"]
    out = pd.DataFrame(rows)
    out["length_m"] = out.length_m.round()
    return out


def write_map(folder: Path, cands, roads: Roads, anchor, tr: pd.DataFrame, st: pd.DataFrame):
    import folium
    lat0, lon0, start = anchor
    # OSM's tile servers reject pages opened from disk (no Referer) and CARTO now
    # needs a key → Esri tiles; satellite helps check against what the video shows
    m = folium.Map(location=[lat0, lon0], zoom_start=14, tiles=None)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
                     attr="© Esri, OpenStreetMap contributors", name="Streets").add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="© Esri", name="Satellite", show=False).add_to(m)
    colors = ["#E84855", "#2E86AB", "#90BE6D"]
    for i, (s, coords) in reversed(list(enumerate(cands))):
        folium.PolyLine(coords, color=colors[i % 3], weight=6 if i == 0 else 3,
                        opacity=0.9 if i == 0 else 0.6,
                        tooltip=f"candidate {i + 1}: cost {s.cost:.2f}, "
                                f"{len(s.matched)} turns matched").add_to(m)
    folium.Marker([lat0, lon0], tooltip=f"start {start:%H:%M:%S} (GPS)",
                  icon=folium.Icon(color="green")).add_to(m)
    G = roads.G
    for ev, node, ang in cands[0][0].matched:
        folium.CircleMarker([G.nodes[node]["y"], G.nodes[node]["x"]], radius=5, color="#333",
                            fill=True, tooltip=f"turn {'right' if ang > 0 else 'left'} {abs(ang)}°").add_to(m)
    for _, r in st.iterrows():
        p = tr.iloc[(tr.video_s - r.t_start).abs().argmin()]
        folium.CircleMarker([p.lat, p.lon], radius=7, color="#F8961E", fill=True,
                            tooltip=f"stop {p.clock}, {r.duration_s:.0f} s ({r.cause})").add_to(m)
    folium.LayerControl().add_to(m)
    m.save(folder / "route_map.html")


def run(folder: Path, video: Path, hfov: float, top: int,
        t_from: float | None, t_to: float | None, end_point):
    lat0, lon0, start = video_anchor(video)
    meta, e, tn, t_from, t_to = observations(folder, hfov, t_from, t_to)
    print(f"street-driving window: {t_from:.0f}–{t_to:.0f} s of video")
    print(f"start {start:%Y-%m-%d %H:%M:%S}, {len(tn)} detected turns:")
    print(tn[["t_mid", "direction", "angle_deg_approx", "duration_s"]].to_string(index=False))

    G = ox.load_graphml(GRAPH)
    roads = Roads(G)
    # recording may start in a parking lot: seed with every junction within 300 m
    dist0 = {n: ox.distance.great_circle(lat0, lon0, G.nodes[n]["y"], G.nodes[n]["x"]) for n in G.nodes}
    start_nodes = [n for n, d in dist0.items() if d < 300] or [min(dist0, key=dist0.get)]
    cands = search(roads, start_nodes, tn, e, t_from, t_to, end_point)
    if not cands:
        print("no route found"); return

    uniq, seen = [], set()
    for s in cands:
        sig = tuple(p[1] for p in s.path)
        if sig not in seen:
            seen.add(sig); uniq.append(s)
        if len(uniq) == top:
            break
    cc = [(s, path_coords(roads, s.path)) for s in uniq]

    tr = trace(cc[0][1], e, start, cc[0][0].segs)
    st = stops(e.assign(light=None))
    tr.to_csv(folder / "trace.csv", index=False)
    feats = [dict(type="Feature", properties=dict(rank=i + 1, cost=round(s.cost, 2),
                                                  turns_matched=len(s.matched),
                                                  length_m=round(sum(roads.length(x) for x in s.path))),
                  geometry=dict(type="LineString", coordinates=[[lo, la] for la, lo in c]))
             for i, (s, c) in enumerate(cc)]
    (folder / "route_candidates.geojson").write_text(json.dumps(dict(type="FeatureCollection",
                                                                     features=feats)))
    write_map(folder, cc, roads, (lat0, lon0, start), tr, st)
    for f in feats:
        print(f["properties"])
    print("\nbest route, street by street (clock time when entering):")
    print(street_list(roads, cc[0][0].path, tr).to_string(index=False))
    print(f"best route: {tr.dist_m.iloc[-1] / 1000:.1f} km, "
          f"mean speed {tr.dist_m.iloc[-1] / (tr.video_s.iloc[-1] - tr.video_s.iloc[0]) * 3.6:.1f} km/h")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--video", type=Path, required=True, help="original video (for GPS anchor + time)")
    ap.add_argument("--hfov", type=float, default=HFOV_DEG)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--from", dest="t_from", type=float,
                    help="video second when street driving starts (skip mounting / parking lot)")
    ap.add_argument("--to", dest="t_to", type=float,
                    help="video second when street driving ends (skip parking at destination)")
    ap.add_argument("--end", help="known destination 'lat,lon' (second anchor)")
    a = ap.parse_args()
    end = tuple(float(x) for x in a.end.split(",")) if a.end else None
    run(a.folder, a.video, a.hfov, a.top, a.t_from, a.t_to, end)


if __name__ == "__main__":
    main()
