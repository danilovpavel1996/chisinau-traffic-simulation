"""
map_layer.py
------------
The map around the car for the 3D view: what the camera cannot see (side streets round a corner,
lanes behind other cars, the far side of a junction), from the SUMO network the simulation uses —
lane by lane, with junction shapes, pedestrian crossings and traffic-light junctions.

    network    data/sumo_net/network.net.xml (netconvert from OpenStreetMap) → compact tables,
               once: lanes.parquet (vehicle lanes: edge, index, width, shape), crossings.parquet,
               junctions.parquet (shape, traffic light or not). Lane counts come from OSM where
               mapped (lanes=* on ~41 % of Chișinău's streets), else netconvert's default per road type
    placement  the map position from the video alone (event_match: route + time at each matched
               junction) is right street, but tens of metres off along it at times. Our own path in
               the ground frame (world.Pose) has the right shape (turns within ~10 %); fitting that
               shape onto the matched route, window by window (rigid: rotation + shift), places the
               map around the car more precisely (pose_on_map)

Coordinates: UTM zone 35 (metres, east / north), as SUMO's projection.

Usage:
    .venv-video/bin/python -m src.video.map_layer --extract        # once
    .venv-video/bin/python -m src.video.map_layer <drive>           # placement vs GPS (answer key)
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from src.video import drives
from src.video.drives import ROOT

NET = ROOT / "data" / "sumo_net" / "network.net.xml"
OUT = ROOT / "data" / "sumo_net"
VEHICLE = {"passenger", "bus", "truck", "taxi", "delivery", "trolleybus", "motorcycle", "evehicle", "private"}


def _shape(s: str) -> list[list[float]]:
    return [[round(float(a), 2) for a in p.split(",")[:2]] for p in s.split()]


def _vehicle_lane(allow: str | None, disallow: str | None) -> bool:
    if allow:
        return bool(VEHICLE & set(allow.split()))
    if disallow:
        return "passenger" not in disallow.split()
    return True


def extract():
    """network.net.xml → lanes / crossings / junctions parquet (UTM metres)."""
    offset = None
    lanes, crossings, junctions = [], [], []
    edge = None
    for ev, el in ET.iterparse(NET, events=("start", "end")):
        if ev == "start":
            if el.tag == "location":
                offset = [float(v) for v in el.get("netOffset").split(",")]
            elif el.tag == "edge":
                edge = dict(id=el.get("id"), function=el.get("function"), name=el.get("name"), type=el.get("type"),
                            frm=el.get("from"), to=el.get("to"))
            continue
        if el.tag == "lane" and edge is not None:
            fn = edge["function"]
            row = dict(edge=edge["id"], lane=el.get("id"), index=int(el.get("index")),
                       width=float(el.get("width", 3.2)), shape=json.dumps(_shape(el.get("shape"))))
            if fn is None and _vehicle_lane(el.get("allow"), el.get("disallow")):
                lanes.append(dict(row, name=edge["name"], type=edge["type"], frm=edge["frm"], to=edge["to"]))
            elif fn == "crossing":
                crossings.append(dict(row, frm=edge["frm"]))
        elif el.tag == "edge":
            edge = None
        elif el.tag == "junction" and el.get("type") != "internal" and el.get("shape"):
            junctions.append(dict(id=el.get("id"), type=el.get("type"), x=float(el.get("x")), y=float(el.get("y")),
                                  shape=json.dumps(_shape(el.get("shape")))))
        el.clear()
    ox, oy = offset
    for rows in (lanes, crossings, junctions):                     # net coordinates → UTM
        for r in rows:
            pts = np.array(json.loads(r["shape"])) - [ox, oy]
            r["shape"] = json.dumps(np.round(pts, 2).tolist())
            if "x" in r:
                r["x"], r["y"] = r["x"] - ox, r["y"] - oy
    for name, rows in (("lanes", lanes), ("crossings", crossings), ("junctions", junctions)):
        pd.DataFrame(rows).to_parquet(OUT / f"{name}.parquet", index=False)
    tl = sum(j["type"] == "traffic_light" for j in junctions)
    print(f"→ {OUT}: {len(lanes)} vehicle lanes, {len(crossings)} crossings, {len(junctions)} junctions "
          f"({tl} with traffic lights)", flush=True)


# ── placement ───────────────────────────────────────────────────────────────


def utm():
    from pyproj import Transformer
    return Transformer.from_crs("EPSG:4326", "EPSG:32635", always_xy=True)


def _rigid(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
    """Rotation φ (world → map: E = x cos φ + z sin φ, N = −x sin φ + z cos φ) and shift T minimising
    Σ w |R a + T − b|² (2-D, weighted)."""
    w = w / w.sum()
    ca, cb = (a * w[:, None]).sum(0), (b * w[:, None]).sum(0)
    A, B = a - ca, b - cb
    sxx = (w * (A[:, 0] * B[:, 0] + A[:, 1] * B[:, 1])).sum()
    sxy = (w * (A[:, 1] * B[:, 0] - A[:, 0] * B[:, 1])).sum()
    phi = float(np.arctan2(sxy, sxx))
    return phi, cb - _rot(ca[None], phi)[0]


def _rot(a: np.ndarray, phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    return np.column_stack([a[:, 0] * c + a[:, 1] * s, -a[:, 0] * s + a[:, 1] * c])


TURN_DEG = 35.0             # a corner / turn: heading change at least this …
TURN_SPAN_M = 20.0           # … over ±this along the route, ±TURN_SPAN_S along our path
TURN_SPAN_S = 6.0
TURN_MATCH_DEG = 35.0        # a path turn and a route corner match when their angles agree this well …
TURN_GAP = 0.35              # … and the distance between consecutive anchors agrees within ±35 %
SCALE_CLIP = (0.85, 1.15)


def _headings(xy: np.ndarray) -> np.ndarray:
    d = np.gradient(xy, axis=0)
    return np.degrees(np.arctan2(d[:, 0], d[:, 1]))


def _wrap(a):
    return (np.asarray(a) + 180) % 360 - 180


def route_corners(route: np.ndarray, cum: np.ndarray) -> list[tuple[float, float]]:
    """(distance along the route, signed heading change) of its corners."""
    d = np.arange(0, cum[-1], 2.0)
    pts = np.column_stack([np.interp(d, cum, route[:, 0]), np.interp(d, cum, route[:, 1])])
    h = np.unwrap(np.radians(_headings(pts)))
    k = int(TURN_SPAN_M / 2)
    change = np.degrees(np.r_[np.zeros(k), h[2 * k:] - h[:-2 * k], np.zeros(k)])
    return _peaks(d, change)


def _peaks(x: np.ndarray, change: np.ndarray) -> list[tuple[float, float]]:
    big = np.abs(change) >= TURN_DEG
    out, i = [], 0
    while i < len(x):
        if big[i]:
            j = i
            while j < len(x) and big[j] and np.sign(change[j]) == np.sign(change[i]):
                j += 1
            m = i + int(np.argmax(np.abs(change[i:j])))
            out.append((float(x[m]), float(change[m])))
            i = j
        else:
            i += 1
    return out


def path_turns(pose, t: np.ndarray) -> list[tuple[float, float]]:
    """(video time, signed heading change) of our turns, from world.Pose."""
    psi = np.degrees(pose.at(t)[2])
    k = int(TURN_SPAN_S / np.median(np.diff(t)))
    change = np.r_[np.zeros(k), psi[2 * k:] - psi[:-2 * k], np.zeros(k)]
    return _peaks(t, change)


def match_turns(turns, corners, odo_at) -> list[tuple[float, float]]:
    """In-order matching of our turns to the route's corners (direction and size agree, the distance
    driven between consecutive anchors agrees with the route's): (time, route distance) anchors."""
    n, m = len(turns), len(corners)
    best = {}                                                   # (i, j) → (count, prev)
    for i, (ti, ai) in enumerate(turns):
        for j, (dj, bj) in enumerate(corners):
            if abs(_wrap(ai - bj)) > TURN_MATCH_DEG:
                continue
            cand = (1, None)
            for (pi, pj), (cnt, _) in best.items():
                if pi < i and pj < j:
                    dt_odo = odo_at(ti) - odo_at(turns[pi][0])
                    dd = dj - corners[pj][0]
                    if dd > 0 and abs(dt_odo - dd) <= TURN_GAP * dd + 30 and cnt + 1 > cand[0]:
                        cand = (cnt + 1, (pi, pj))
            best[(i, j)] = cand
    if not best:
        return []
    cur = max(best, key=lambda k: best[k][0])
    chain = []
    while cur is not None:
        chain.append((turns[cur[0]][0], corners[cur[1]][0]))
        cur = best[cur][1]
    return chain[::-1]


PROFILE_M = 2.0              # heading profiles sampled every 2 m of distance …
CHANGE_M = 20.0              # … compared as heading change over ±this (immune to slow heading drift)
BAND_M = 400.0               # alignment searched within ±this of the odometer's guess
STRETCH_COST = 3.0           # degrees of cost for a step that stretches / squeezes distance
# A mostly straight route gives the heading profile nothing to align to: 2026_10_24_15_48 (91 % straight,
# Șos. Muncești) was put 200 m ahead in its first 10 s although its odometer was 2 % right. There, stay
# near the odometer's guess on the straight parts. City routes (48–60 % straight) are left alone — on
# them the pull made things worse (night drive 15.8 → 24 m: its odometer drifts more than its bends do).
ODO_PULL = 2.0               # degrees of cost per step per 100 m away from the odometer's guess …
STRAIGHT_ROUTE = 0.8         # … on routes at least this share straight …
FLAT_M = 100.0               # … where there is no bend within ±this
FLAT_DEG = 10.0              # (heading change below this)


def _change(h_deg: np.ndarray, k: int) -> np.ndarray:
    h = np.unwrap(np.radians(h_deg))
    return np.degrees(np.r_[np.zeros(k), h[2 * k:] - h[:-2 * k], np.zeros(k)])


def align_profiles(path_change: np.ndarray, route_change: np.ndarray) -> np.ndarray:
    """Monotone alignment (dynamic time warping) of our heading-change profile to the route's, from
    the start pin to the destination pin; returns the route index for each path index."""
    n, m = len(path_change), len(route_change)
    scale = m / max(n, 1)
    band = int(BAND_M / PROFILE_M)
    w = int(FLAT_M / PROFILE_M)
    bend = np.abs(route_change) >= FLAT_DEG
    flat = ~(np.convolve(bend, np.ones(2 * w + 1), "same") > 0)          # no bend nearby: shape says nothing
    pull = ODO_PULL if flat.mean() >= STRAIGHT_ROUTE else 0.0
    INF = 1e18
    D = np.full((n, m), INF, np.float32)
    back = np.zeros((n, m), np.int8)                      # 0 diagonal, 1 from (i-1, j), 2 from (i, j-1)
    for i in range(n):
        lo, hi = max(0, int(i * scale) - band), min(m, int(i * scale) + band + 1)
        cost = np.abs(path_change[i] - route_change[lo:hi])
        if pull:                                           # a straight road says nothing: stay near the odometer
            cost = cost + pull * flat[lo:hi] * np.abs(np.arange(lo, hi) - i * scale) * PROFILE_M / 100
        cost = cost.astype(np.float32)
        if i == 0:
            row = np.full(hi - lo, INF, np.float32)
            row[0] = cost[0] if lo == 0 else INF
            for j in range(1, hi - lo):                    # moving along the route only (squeeze)
                row[j] = row[j - 1] + cost[j] + STRETCH_COST
            D[0, lo:hi] = row
            back[0, lo + 1:hi] = 2
            continue
        diag = np.r_[INF, D[i - 1, lo:hi - 1]] if lo == 0 else D[i - 1, lo - 1:hi - 1]
        up = D[i - 1, lo:hi] + STRETCH_COST
        best = np.minimum(diag, up)
        which = np.where(diag <= up, 0, 1).astype(np.int8)
        row = best + cost
        for j in range(1, hi - lo):                        # (i, j-1) → (i, j): route advances alone
            alt = row[j - 1] + cost[j] + STRETCH_COST
            if alt < row[j]:
                row[j], which[j] = alt, 2
        D[i, lo:hi], back[i, lo:hi] = row, which
    j = m - 1 if D[n - 1, m - 1] < INF else int(np.argmin(D[n - 1]))
    out = np.empty(n, int)
    i = n - 1
    while i >= 0:
        out[i] = j
        b = back[i, j]
        if b == 0:
            i, j = i - 1, j - 1
        elif b == 1:
            i -= 1
        else:
            j -= 1
        if j < 0:
            out[:i + 1] = 0
            break
    return out


def pose_on_map(drv: dict) -> pd.DataFrame:
    """
    Per analysed frame: our position (E, N, UTM metres) and heading (degrees from north) on the map,
    from the video alone. The matched route is right street, but its timing (positions between
    matched junctions) is 40–75 m off in the median: on a straight road neither junction events nor a
    path's shape say how far along it we are. Here our whole heading profile — every bend and turn,
    as heading change against metres driven (world.Pose: speed.py's metric speed and rotation) — is
    aligned to the route's own profile from the start pin to the destination pin (align_profiles).
    """
    from src.video.world import Pose
    em = json.loads((drv["folder"] / "event_match.json").read_text())
    rt = em["route"]
    to = utm()
    re_, rn = to.transform(rt["lon"], rt["lat"])
    route = np.column_stack([re_, rn])
    cum = np.r_[0, np.cumsum(np.hypot(*np.diff(route, axis=0).T))]
    pose = Pose(drv)
    win = drv.get("street_window_s", [pose.t.min(), pose.t.max()])
    t = pose.t[(pose.t >= win[0]) & (pose.t <= win[1])]
    px, pz, psi = pose.at(t)
    odo = np.r_[0, np.cumsum(np.hypot(np.diff(px), np.diff(pz)))]
    k = int(CHANGE_M / PROFILE_M)
    o_s = np.arange(0, odo[-1], PROFILE_M)                       # our profile, per 2 m driven
    path_change = _change(np.degrees(np.interp(o_s, odo, np.unwrap(psi))), k)
    d_s = np.arange(0, cum[-1], PROFILE_M)                       # the route's, per 2 m
    pts = np.column_stack([np.interp(d_s, cum, route[:, 0]), np.interp(d_s, cum, route[:, 1])])
    route_change = _change(_headings(pts), k)
    j = align_profiles(path_change, route_change)
    d = np.interp(odo, o_s, d_s[j])
    n_anchors = 0
    if USE_ANCHORS:
        pose.odo = np.interp(pose.t, t, odo)
        events = [(crosswalk_passes(drv, pose), route_features(route, cum, "crossing")),
                  (light_passes(drv, pose), route_features(route, cum, "traffic_light"))]
        for _ in range(2):                                       # re-predict with the corrected positions
            d, n_anchors = anchor(t, d, odo, events, cum[-1])
    E, N = np.interp(d, cum, route[:, 0]), np.interp(d, cum, route[:, 1])
    ahead, back = np.clip(d + 8, 0, cum[-1]), np.clip(d - 8, 0, cum[-1])
    hd = np.degrees(np.arctan2(np.interp(ahead, cum, route[:, 0]) - np.interp(back, cum, route[:, 0]),
                               np.interp(ahead, cum, route[:, 1]) - np.interp(back, cum, route[:, 1]))) % 360
    out = pd.DataFrame(dict(t=t, E=E, N=N, heading=hd, d=d))
    out.attrs["anchors"] = n_anchors
    return out


# ── anchors: landmarks seen in the video, matched to the map ────────────────
# Off: the video's crosswalk / light passings are not precise enough yet — at the true position the
# nearest OSM crossing was a median 21–30 m away, and 60–170 light "passes" met 7–9 signalised
# junctions on the route (several heads per junction, pedestrian and false lights). Median error
# with anchors: 10_06 25 → 22 m, 19_26 46 → 24 m, but 17_08 28 → 47 m.
USE_ANCHORS = False
CROSS_Z = (4.0, 16.0)        # a crosswalk measured this far ahead …
CROSS_MIN_PX = 40            # … with at least this many class-map pixels in that band
GROUP_S = 3.0                # passing estimates closer than this are one crosswalk
MATCH_M = 45.0               # a landmark is matched to a map feature this near the current estimate …
UNIQUE_M = 20.0              # … when the next-nearest is this much farther
CONSIST = 0.2                # consecutive anchors: distance along the route vs driven within ±20 % + 15 m


def crosswalk_passes(drv: dict, pose) -> np.ndarray:
    """Video seconds at which we drive over a crosswalk: every analysed frame's crosswalk pixels
    projected onto the road (bev.Geometry) give its distance ahead; plus our speed, the moment we reach it."""
    import zlib
    from src.video.bev import Geometry
    from src.video.scene import CODES
    dense = pd.read_parquet(drv["folder"] / "scene_dense.parquet").sort_values("t")
    geo = Geometry(drv, dense.iloc[0])
    cw = CODES.index("crosswalk")
    est = []
    for r in dense.to_dict("records"):
        m = np.frombuffer(zlib.decompress(r["mask"]), np.uint8).reshape(json.loads(r["mask_shape"]))
        rr, cc = np.nonzero(m == cw)
        if len(rr) < CROSS_MIN_PX:
            continue
        y = r["top"] + (rr + 0.5) * r["step"]
        x = (cc + 0.5) * r["step"]
        xu, yu = geo.undistort(x, y)
        ok = yu > geo.y_h + 3
        Z = geo.fz * geo.h / (yu[ok] - geo.y_h)
        X = (xu[ok] - geo.cx) * geo.h / (yu[ok] - geo.y_h)
        sel = (Z > CROSS_Z[0]) & (Z < CROSS_Z[1]) & (np.abs(X) < 4)        # across our own lane
        if sel.sum() < CROSS_MIN_PX:
            continue
        z_near = float(np.percentile(Z[sel], 5))
        t = float(r["t"])
        # when have we driven z_near metres from here?
        d0 = np.interp(t, pose.t, pose.odo)
        est.append(float(np.interp(d0 + z_near, pose.odo, pose.t)))
    est = np.sort(np.array(est))
    out, grp = [], []
    for e in est:
        if grp and e - grp[-1] > GROUP_S:
            if len(grp) >= 3:
                out.append(float(np.median(grp)))
            grp = []
        grp.append(e)
    if len(grp) >= 3:
        out.append(float(np.median(grp)))
    return np.array(out)


def light_passes(drv: dict, pose) -> np.ndarray:
    """Video seconds at which we pass traffic-light landmarks (world.lights) that stand beside or over
    our road (within 12 m sideways): the moment their distance ahead reaches zero."""
    from src.video import world
    from src.video.bev import Geometry
    dense = pd.read_parquet(drv["folder"] / "scene_dense.parquet").sort_values("t")
    marks, _ = world.lights(drv, Geometry(drv, dense.iloc[0]), pose)
    out = []
    for m in marks.itertuples():
        ts = np.arange(m.t_first, m.t_last + 15, 0.2)
        X, Z = pose.to_car(ts, m.x, m.z)
        k = np.flatnonzero((Z[:-1] > 0) & (Z[1:] <= 0))
        if len(k) and abs(X[k[0]]) < 12:
            out.append(float(ts[k[0]]))
    return np.sort(np.array(out))


def route_features(route: np.ndarray, cum: np.ndarray, kind: str) -> np.ndarray:
    """Distances along the route of OSM crossings / signalised junctions on it."""
    from scipy.spatial import cKDTree
    to = utm()
    if kind == "crossing":
        f = pd.read_parquet(ROOT / "data" / "osm" / "junction_features.parquet")
        f = f[f.kind != "traffic_signals"]
        fe, fn = to.transform(f.lon.to_numpy(), f.lat.to_numpy())
        pts = np.column_stack([fe, fn])
    else:
        j = pd.read_parquet(OUT / "junctions.parquet")
        j = j[j.type == "traffic_light"]
        pts = j[["x", "y"]].to_numpy()
    dd = np.arange(0, cum[-1], 1.0)
    line = np.column_stack([np.interp(dd, cum, route[:, 0]), np.interp(dd, cum, route[:, 1])])
    dist, idx = cKDTree(line).query(pts)
    near = dist < (12 if kind == "crossing" else 25)
    return np.unique(np.round(dd[idx[near]] / 5) * 5)


def anchor(t: np.ndarray, d: np.ndarray, odo: np.ndarray, events: list[tuple[np.ndarray, np.ndarray]],
           length: float) -> tuple[np.ndarray, int]:
    """Correct d(t) with landmark anchors: each (event times, feature distances) pair is matched by
    nearest distance; inconsistent anchors are dropped; between anchors the odometer, stretched."""
    anchors = []
    for times, feats in events:
        if not len(times) or not len(feats):
            continue
        for tc in times:
            dc = float(np.interp(tc, t, d))
            gap = np.sort(np.abs(feats - dc))
            if gap[0] <= MATCH_M and (len(gap) == 1 or gap[1] - gap[0] >= UNIQUE_M):
                anchors.append((float(tc), float(feats[np.argmin(np.abs(feats - dc))])))
    anchors.sort()
    odo_at = lambda x: float(np.interp(x, t, odo))
    changed = True                                              # drop anchors that disagree with their neighbours
    while changed and len(anchors) > 2:
        changed = False
        a = [(t[0], d[0])] + anchors + [(t[-1], d[-1])]
        bad = []
        for k in range(1, len(a) - 1):
            ok_prev = abs((a[k][1] - a[k - 1][1]) - (odo_at(a[k][0]) - odo_at(a[k - 1][0]))) <= CONSIST * abs(a[k][1] - a[k - 1][1]) + 15
            ok_next = abs((a[k + 1][1] - a[k][1]) - (odo_at(a[k + 1][0]) - odo_at(a[k][0]))) <= CONSIST * abs(a[k + 1][1] - a[k][1]) + 15
            if not (ok_prev or ok_next) or a[k][1] < a[k - 1][1]:
                bad.append(k - 1)
        if bad:
            anchors = [x for i, x in enumerate(anchors) if i not in bad]
            changed = True
    a = [(t[0], d[0])] + anchors + [(t[-1], d[-1])]
    at, ad = np.array([x[0] for x in a]), np.maximum.accumulate([x[1] for x in a])
    out = d.copy()
    for k in range(len(a) - 1):
        m = (t >= at[k]) & (t <= at[k + 1])
        o0, o1 = odo_at(at[k]), odo_at(at[k + 1])
        sc = (ad[k + 1] - ad[k]) / max(o1 - o0, 1e-3)
        out[m] = ad[k] + (odo[m] - o0) * np.clip(sc, 0.7, 1.3)
    return np.clip(out, 0, length), len(anchors)


# ── registration: the road surface seen vs the map's ───────────────────────
# Tried and dropped (not called): started from the GPS position it wandered 7–9 m off, started 25 m
# off it stayed 25 m off — the road surface is too uniform along a street, and the network's lane
# centre lines too approximate, to fix the position along the road.
REG_CELL = 0.5               # m per raster cell
REG_HALF = 70.0              # raster half-size around the window's centre position
REG_WINDOW_S = 12.0          # road seen during ±this (12 s at 40 km/h ≈ ±130 m of road)
REG_STEP_S = 4.0
REG_SHIFT = 45.0             # search shifts up to this far
REG_MIN_PEAK = 1.25          # best overlap must beat the median of the search area by this factor


def frame_road_points(drv: dict, mp: pd.DataFrame, pose) -> tuple[np.ndarray, list]:
    """Every analysed frame's road pixels (road, markings, crosswalks) in map coordinates, placed with
    the current estimate (mp: t, E, N, heading); world.Pose gives where each was relative to us."""
    from src.video import ground_world as gw
    from src.video.bev import Geometry
    from src.video.scene import CODES
    dense = pd.read_parquet(drv["folder"] / "scene_dense.parquet").sort_values("t").reset_index(drop=True)
    geo = Geometry(drv, dense.iloc[0])
    lut = np.full(len(CODES), -1, np.int8)
    for name in ("road", "lane_marking", "crosswalk", "bike_lane"):
        lut[CODES.index(name)] = 0
    gw_sample, gw.SAMPLE = gw.SAMPLE, 4
    ts, pts = [], []
    for r in dense.to_dict("records"):
        t = float(r["t"])
        if t < mp.t.iat[0] or t > mp.t.iat[-1]:
            continue
        wx, wz, _ = gw.frame_points(geo, r, pose, t, lut)
        px, pz, psi = pose.at(t)
        E0, N0, hd = (np.interp(t, mp.t, mp[c]) for c in ("E", "N", "heading"))
        phi = np.radians(hd) - psi                                  # world → map rotation at t
        dx, dz = wx - px, wz - pz
        c, s_ = np.cos(phi), np.sin(phi)
        ts.append(t)
        pts.append(np.column_stack([E0 + dx * c + dz * s_, N0 - dx * s_ + dz * c]).astype(np.float32))
    gw.SAMPLE = gw_sample
    return np.array(ts), pts


def map_raster(lanes: pd.DataFrame, centre: np.ndarray, n: int) -> np.ndarray:
    import cv2
    img = np.zeros((n, n), np.uint8)
    for r in lanes.itertuples():
        p = (np.array(json.loads(r.shape)) - centre + REG_HALF) / REG_CELL
        p = np.column_stack([p[:, 0], n - 1 - p[:, 1]]).round().astype(np.int32)
        cv2.polylines(img, [p], False, 1, thickness=max(1, int(round(r.width / REG_CELL))))
    return img


def register(drv: dict, mp: pd.DataFrame, pose) -> tuple[pd.DataFrame, dict]:
    """Shift the estimate, window by window, to where the road surface seen best overlaps the map's
    (FFT cross-correlation); along the road at junctions / side streets, sideways everywhere."""
    from scipy.signal import fftconvolve
    from scipy.spatial import cKDTree
    lanes = pd.read_parquet(OUT / "lanes.parquet")
    first = np.array([json.loads(s_)[0] for s_ in lanes["shape"]])
    tree = cKDTree(first)
    ts, pts = frame_road_points(drv, mp, pose)
    n = int(2 * REG_HALF / REG_CELL)
    sh = int(REG_SHIFT / REG_CELL)
    fixes = []
    for c in np.arange(mp.t.iat[0], mp.t.iat[-1], REG_STEP_S):
        sel = [p for t, p in zip(ts, pts) if abs(t - c) <= REG_WINDOW_S]
        if not sel:
            continue
        P = np.vstack(sel)
        centre = np.array([np.interp(c, mp.t, mp.E), np.interp(c, mp.t, mp.N)])
        q = ((P - centre + REG_HALF) / REG_CELL).astype(int)
        ok = (q >= 0).all(1) & (q < n).all(1)
        seen = np.zeros((n, n), np.float32)
        np.add.at(seen, (n - 1 - q[ok, 1], q[ok, 0]), 1)
        seen = (seen > 0).astype(np.float32)
        near = tree.query_ball_point(centre, REG_HALF * 1.5)
        if not near or seen.sum() < 200:
            continue
        ref = map_raster(lanes.iloc[near], centre, n).astype(np.float32)
        corr = fftconvolve(ref, seen[::-1, ::-1], mode="same")      # corr[y, x]: overlap with seen shifted
        cy, cx = n // 2, n // 2
        win = corr[cy - sh:cy + sh + 1, cx - sh:cx + sh + 1]
        k = np.unravel_index(int(np.argmax(win)), win.shape)
        peak, med = float(win[k]), float(np.median(win))
        if med <= 0 or peak / med < REG_MIN_PEAK:
            continue
        dy, dx = (k[0] - sh) * REG_CELL, (k[1] - sh) * REG_CELL
        fixes.append((c, dx, -dy, peak / med))                      # image rows grow southwards
    F = pd.DataFrame(fixes, columns=["t", "dE", "dN", "q"])
    if len(F) < 3:
        return mp, dict(windows=len(F))
    for col in ("dE", "dN"):                                         # robust, smooth over time
        F[col] = F[col].rolling(5, center=True, min_periods=1).median()
    out = mp.copy()
    out["E"] = mp.E + np.interp(mp.t, F.t, F.dE)
    out["N"] = mp.N + np.interp(mp.t, F.t, F.dN)
    return out, dict(windows=len(F), shift_median_m=round(float(np.median(np.hypot(F.dE, F.dN))), 1))


# ── Mapillary pins: our traffic lights vs Mapillary's ──────────────────────
PIN_R = 60.0                 # Mapillary lights this near one of ours (as currently placed) are candidates …
PIN_SIDE = 15.0              # … if no farther sideways than this (sideways is locked to the lane lines)
PIN_WINDOW_S = 40.0          # offsets pooled over ±this
PIN_STEP_S = 10.0
PIN_BW = 4.0                 # kernel width of the offset vote, metres
PIN_SUPPORT = 3              # at least this many of our lights must agree (within ±6 m of the peak)
PIN_MAX = 60.0
PIN_ROUNDS = 1
# Off: Mapillary's light heads are not points to pin on — 866 along the 8.9-km city route (spacing 1–2 m),
# clustered into 8–19 spread-out "spots"; at the true position only 16–17 % of our passings were within
# 15 m of one. Median error: light-to-light votes 25 → 34 m (10_06), 28 → 34 m (17_08); spot-to-spot
# matching 25 → 32 m, 28 → 55 m.
USE_PINS = False


def our_lights_on_map(drv: dict, mp: pd.DataFrame, pose) -> pd.DataFrame:
    """Our verified light landmarks placed on the map with the current estimate, at the moment we pass
    them: t, E, N, and the route direction there (ux, uy)."""
    L = pd.read_parquet(drv["folder"] / "lights.parquet")
    L = L[L.real]
    rows = []
    for m in L.itertuples():
        ts = np.arange(m.t_first, m.t_last + 20, 0.2)
        X, Z = pose.to_car(ts, m.x, m.z)
        k = int(np.argmin(np.abs(Z - 5)))                         # nearly beside us: placement least sensitive to heading
        t = float(ts[k])
        if t < mp.t.iat[0] or t > mp.t.iat[-1]:
            continue
        r = mp.iloc[int(np.argmin(np.abs(mp.t - t)))]
        th = np.radians(r.heading)
        rows.append(dict(t=t, E=r.E + X[k] * np.cos(th) + Z[k] * np.sin(th), N=r.N - X[k] * np.sin(th) + Z[k] * np.cos(th),
                         ux=np.sin(th), uy=np.cos(th)))
    return pd.DataFrame(rows)


CLUSTER_GAP_M = 30.0         # Mapillary lights closer than this along the route: one signalised spot
OUR_GAP_S = 4.0              # our verified lights passed within this: one spot
SPOT_SIDE_M = 25.0           # Mapillary lights at most this far from the route
SPOT_MATCH_M = 70.0          # one of our spots matches a Mapillary spot this near the current estimate …
SPOT_CONSIST = 0.15          # … in order, distances between consecutive matches agreeing within ±15 % + 15 m


def mapillary_spots(drv: dict, route: np.ndarray, cum: np.ndarray) -> np.ndarray:
    """Distances along the route of Mapillary's signalised spots (its light heads, clustered)."""
    from scipy.spatial import cKDTree
    from src.video import mapillary
    ML = mapillary.route_features(drv)
    if not len(ML):
        return np.array([])
    me, mn = utm().transform(ML.lon.to_numpy(), ML.lat.to_numpy())
    dd = np.arange(0, cum[-1], 1.0)
    line = np.column_stack([np.interp(dd, cum, route[:, 0]), np.interp(dd, cum, route[:, 1])])
    side, idx = cKDTree(line).query(np.column_stack([me, mn]))
    along = np.sort(dd[idx[side <= SPOT_SIDE_M]])
    if not len(along):
        return along
    groups = np.split(along, np.flatnonzero(np.diff(along) > CLUSTER_GAP_M) + 1)
    return np.array([np.median(g) for g in groups if len(g) >= 2])         # a single head: too weak


def our_spots(drv: dict, pose) -> np.ndarray:
    """Moments we pass one of our verified lights (beside / over our road), clustered per spot."""
    L = pd.read_parquet(drv["folder"] / "lights.parquet")
    L = L[L.real & (L.x_pass.abs() <= SPOT_SIDE_M)]
    ts = []
    for m in L.itertuples():
        tt = np.arange(m.t_first, m.t_last + 20, 0.2)
        _, Z = pose.to_car(tt, m.x, m.z)
        k = np.flatnonzero(Z <= 0)
        if len(k):
            ts.append(float(tt[k[0]]))
    ts = np.sort(ts)
    if not len(ts):
        return ts
    groups = np.split(ts, np.flatnonzero(np.diff(ts) > OUR_GAP_S) + 1)
    return np.array([np.median(g) for g in groups])


def mapillary_pins(drv: dict, mp: pd.DataFrame, pose, route: np.ndarray, cum: np.ndarray) -> tuple[pd.DataFrame, int]:
    """Pin how far along the route we are where we pass a signalised spot that Mapillary has too: our spots
    and Mapillary's matched in order (current estimate ± SPOT_MATCH_M, consistent distances between
    matches); between pins the odometer, stretched to fit."""
    if not (drv["folder"] / "lights.parquet").exists():
        return mp, 0
    spots = mapillary_spots(drv, route, cum)
    ours = our_spots(drv, pose)
    ours = ours[(ours >= mp.t.iat[0]) & (ours <= mp.t.iat[-1])]
    if not len(spots) or not len(ours):
        return mp, 0
    t = mp.t.to_numpy()
    px, pz, _ = pose.at(t)
    odo = np.r_[0, np.cumsum(np.hypot(np.diff(px), np.diff(pz)))]
    odo_at = lambda x: float(np.interp(x, t, odo))
    dhat = np.interp(ours, t, mp.d)
    best = {}                                                   # (i, j) → (count, previous)
    for i, ti in enumerate(ours):
        for j in np.flatnonzero(np.abs(spots - dhat[i]) <= SPOT_MATCH_M):
            cand = (1, None)
            for (pi, pj), (cnt, _) in best.items():
                if pi < i and pj < j:
                    dd, do = spots[j] - spots[pj], odo_at(ti) - odo_at(ours[pi])
                    if abs(dd - do) <= SPOT_CONSIST * dd + 15 and cnt + 1 > cand[0]:
                        cand = (cnt + 1, (pi, pj))
            best[(i, int(j))] = cand
    if not best:
        return mp, 0
    cur = max(best, key=lambda k_: best[k_][0])
    pins = []
    while cur is not None:
        pins.append((float(ours[cur[0]]), float(spots[cur[1]])))
        cur = best[cur][1]
    pins = pins[::-1]
    if len(pins) < 2:
        return mp, 0
    a = [(t[0], float(mp.d.iat[0]))] + pins + [(t[-1], float(mp.d.iat[-1]))]
    at = np.array([x[0] for x in a]); ad = np.maximum.accumulate([x[1] for x in a])
    d = mp.d.to_numpy().copy()
    for k in range(len(a) - 1):
        m = (t >= at[k]) & (t <= at[k + 1])
        o0, o1 = odo_at(at[k]), odo_at(at[k + 1])
        sc = np.clip((ad[k + 1] - ad[k]) / max(o1 - o0, 1e-3), 0.7, 1.3)
        d[m] = ad[k] + (odo[m] - o0) * sc
    d = np.maximum.accumulate(np.clip(d, 0, cum[-1]))
    out = mp.copy()
    out["d"] = d
    out["E"], out["N"] = np.interp(d, cum, route[:, 0]), np.interp(d, cum, route[:, 1])
    ahead, back = np.clip(d + 8, 0, cum[-1]), np.clip(d - 8, 0, cum[-1])
    out["heading"] = np.degrees(np.arctan2(np.interp(ahead, cum, route[:, 0]) - np.interp(back, cum, route[:, 0]),
                                           np.interp(ahead, cum, route[:, 1]) - np.interp(back, cum, route[:, 1]))) % 360
    return out, len(pins)


# ── photo pins (vpr.py) ─────────────────────────────────────────────────────
USE_VPR = True
VPR_EVERY_S = 10.0           # one pin per this many seconds: the median of the confident photo matches
VPR_INLIERS = 100          # verified photo positions with this many consistent point matches (9 m median, 10_06)
VPR_MIN = 3                  # … if at least this many in the stretch


def vpr_pins(mp: pd.DataFrame, v: pd.DataFrame, route: np.ndarray, cum: np.ndarray) -> tuple[pd.DataFrame, int]:
    """Pin how far along the route we are on the photo alignment (confident frames, a robust median
    per VPR_EVERY_S); between pins the estimate's own course (odometer), shifted to fit."""
    if "inliers" in v and (v.inliers >= VPR_INLIERS).sum() >= VPR_MIN:   # verified by point matching (vpr.verify)
        c = v[v.inliers >= VPR_INLIERS].assign(d=lambda x: x.d_photo)
    else:
        c = v[v.confident]
    pins = []
    for t0 in np.arange(c.t.min() if len(c) else 0, (c.t.max() if len(c) else 0) + VPR_EVERY_S, VPR_EVERY_S):
        w = c[(c.t >= t0) & (c.t < t0 + VPR_EVERY_S)]
        if len(w) >= VPR_MIN:
            off = float(np.median(w.d - np.interp(w.t, mp.t, mp.d)))      # photo position − estimate
            pins.append((float(w.t.median()), off))
    if len(pins) < 2:
        return mp, 0
    pt, po = np.array(pins).T
    po = pd.Series(po).rolling(3, center=True, min_periods=1).median().to_numpy()
    d = np.maximum.accumulate(np.clip(mp.d.to_numpy() + np.interp(mp.t, pt, po), 0, cum[-1]))
    out = mp.copy()
    out["d"] = d
    out["E"], out["N"] = np.interp(d, cum, route[:, 0]), np.interp(d, cum, route[:, 1])
    ahead, back = np.clip(d + 8, 0, cum[-1]), np.clip(d - 8, 0, cum[-1])
    out["heading"] = np.degrees(np.arctan2(np.interp(ahead, cum, route[:, 0]) - np.interp(back, cum, route[:, 0]),
                                           np.interp(ahead, cum, route[:, 1]) - np.interp(back, cum, route[:, 1]))) % 360
    return out, len(pins)


# ── sideways lock, and the layer for the 3D view ────────────────────────────
LANE_STEP_M = 2.0            # map lane centre lines resampled to points this far apart
NEAR_M = 30.0                # map lanes this near are compared with the video's lines …
LOCK_Z = (3.0, 15.0)         # … over this stretch ahead
LOCK_SHIFT = np.arange(-12.0, 12.01, 0.1)   # the route runs on the street's centre line: on a 3 + 3 boulevard our lane is ~7 m off it
LOCK_SMOOTH_S = 3.0
# Lane lines repeat every lane width, so matching them alone is ambiguous by a whole lane (10_02 at 10:11
# on Bd. Mihai Viteazul: in the right lane, locked into the middle one; the map's right edge 2.5 m beyond
# the kerb the video saw). The right road edge is not: it is matched too (`EDGE_WEIGHT` × its miss).
EDGE_WEIGHT = 1.0
EDGE_MAX_M = 6.0             # a right edge / curb seen this far right at most
LAYER_M = 60.0               # map lanes within this of our route go into the 3D view


USE_VIDEO_LANES = True


def _offset(p: np.ndarray, right: float) -> np.ndarray:
    """A polyline moved sideways, `right` metres to the right of its direction."""
    d = np.gradient(p, axis=0)
    n = np.column_stack([d[:, 1], -d[:, 0]]) / np.maximum(np.hypot(*d.T), 1e-6)[:, None]
    return p + n * right


def with_video_lanes(lanes: pd.DataFrame) -> pd.DataFrame:
    """SUMO lanes, with the lanes of streets OSM has no `lanes` for replaced by the count the video
    measured (lane_count.py): rebuilt off the street's centre line, lane 0 the rightmost."""
    from src.video import lane_count
    C = lane_count.counts()
    if not len(C):
        return lanes
    C = C.set_index("way")
    W = pd.read_parquet(lane_count.WAYS).set_index("way")
    keep, new = [], []
    for edge, g in lanes.groupby("edge", sort=False):
        way = lane_count.way_of(edge)
        if way is None or way not in C.index:
            keep.append(g)
            continue
        c = C.loc[way]
        k = int(c.lanes_backward if edge.startswith("-") else c.lanes_forward)
        n = len(g)
        if k < 1 or k == n:
            keep.append(g)
            continue
        oneway = W.oneway.get(way) in ("yes", "1", "-1")
        g = g.sort_values("index")
        w0, w = float(g.width.iat[0]), float(c.lane_w_m)
        lane0 = np.array(json.loads(g["shape"].iat[0]))
        centre = _offset(lane0, -(((n - 1) / 2) * w0 if oneway else (n - 0.5) * w0))
        for i in range(k):
            off = ((k - 1) / 2 - i) * w if oneway else (k - i - 0.5) * w
            r = g.iloc[0].to_dict()
            r.update(lane=f"{edge}_{i}", index=i, width=round(w, 2),
                     shape=json.dumps(np.round(_offset(centre, off), 2).tolist()))
            new.append(r)
    out = pd.concat(keep + [pd.DataFrame(new)], ignore_index=True) if new else lanes
    out.attrs["video_lanes"] = len(new)
    return out


class LaneIndex:
    """Vehicle lanes near a route as resampled points (E, N, lane row, direction)."""

    def __init__(self, route: np.ndarray, margin: float, video_lanes: bool = True):
        from scipy.spatial import cKDTree
        lanes = pd.read_parquet(OUT / "lanes.parquet")
        first = np.array([json.loads(x)[0] for x in lanes["shape"]])
        lo, hi = route.min(0) - margin - 300, route.max(0) + margin + 300
        lanes = lanes[(first >= lo).all(1) & (first <= hi).all(1)].reset_index(drop=True)
        if video_lanes and USE_VIDEO_LANES:                      # streets OSM gives no lane count: the video's
            lanes = with_video_lanes(lanes)
        rt = cKDTree(route)
        pts, own, dirs = [], [], []
        for i, r in enumerate(lanes.itertuples()):
            p = np.array(json.loads(r.shape))
            seg = np.hypot(*np.diff(p, axis=0).T)
            if not len(seg) or seg.sum() < 1:
                continue
            cum = np.r_[0, np.cumsum(seg)]
            d = np.arange(0, cum[-1] + 1e-6, LANE_STEP_M)
            q = np.column_stack([np.interp(d, cum, p[:, 0]), np.interp(d, cum, p[:, 1])])
            if rt.query(q)[0].min() > margin:
                continue
            h = np.degrees(np.arctan2(*np.gradient(q, axis=0).T[::1])) if len(q) > 1 else np.zeros(1)
            pts.append(q); own += [i] * len(q); dirs.append(h)
        self.lanes = lanes
        self.pts = np.vstack(pts)
        self.own = np.array(own)
        self.dir = np.concatenate(dirs) % 360
        self.tree = cKDTree(self.pts)

    def car_frame(self, E, N, heading, radius=NEAR_M):
        """Nearby lane points in our car frame: X right, Z ahead; their lane rows and relative directions."""
        idx = np.array(self.tree.query_ball_point([E, N], radius), int)
        if not len(idx):
            return np.zeros(0), np.zeros(0), idx, idx, np.zeros(0)
        th = np.radians(heading)
        dx, dy = self.pts[idx, 0] - E, self.pts[idx, 1] - N
        X = dx * np.cos(th) - dy * np.sin(th)
        Z = dx * np.sin(th) + dy * np.cos(th)
        rel = (self.dir[idx] - heading + 180) % 360 - 180
        return X, Z, idx, self.own[idx], rel


def lateral_lock(drv: dict, mp: pd.DataFrame, li: LaneIndex) -> pd.DataFrame:
    """Shift the map position sideways so the map's lane boundaries beside us fall on the lane lines
    the video measures (lanes_world.parquet); with no lines seen, into the middle of a map lane."""
    lw = drv["folder"] / "lanes_world.parquet"
    lines = pd.read_parquet(lw) if lw.exists() else pd.DataFrame(columns=["t", "z", "kind", "x"])
    near = lines[lines.z <= LOCK_Z[1]]
    lines = near[near.kind == "line"]
    by_t = {t: g.x.to_numpy() for t, g in lines.groupby("t")}
    edge = near[near.kind.isin(["edge_right", "curb"]) & (near.x > 0.5) & (near.x < EDGE_MAX_M)]
    edge_t = edge.groupby("t").x.median()                              # the road's right edge, per frame
    lt = np.array(sorted(by_t))
    shifts = []
    for r in mp.iloc[::5].itertuples():                                # once per second
        X, Z, _, own, rel = li.car_frame(r.E, r.N, r.heading)
        m = (Z >= LOCK_Z[0]) & (Z <= LOCK_Z[1]) & (np.abs(rel) < 30)
        if not m.any():
            shifts.append((r.t, np.nan))
            continue
        cen = pd.Series(X[m]).groupby(own[m]).median()
        widths = li.lanes.width.to_numpy()[cen.index.to_numpy()]
        bounds = np.sort(np.r_[cen.to_numpy() - widths / 2, cen.to_numpy() + widths / 2])
        seen = by_t[lt[np.argmin(np.abs(lt - r.t))]] if len(lt) and np.min(np.abs(lt - r.t)) < 0.5 else np.array([])
        ke = int(np.argmin(np.abs(edge_t.index.to_numpy() - r.t))) if len(edge_t) else -1
        right = float(edge_t.iat[ke]) if ke >= 0 and abs(edge_t.index[ke] - r.t) < 0.5 else None
        if len(seen) or right is not None:
            cost = np.array([np.minimum(np.abs(seen[:, None] - (bounds[None, :] + s_)).min(1), 1.5).mean() if len(seen) else 0.0
                             for s_ in LOCK_SHIFT])                    # mean: one reading per 5-m band, not per line
            if right is not None:                                      # the map's right edge on the kerb
                cost = cost + EDGE_WEIGHT * np.minimum(np.abs(bounds.max() + LOCK_SHIFT - right), 3.0)
            shifts.append((r.t, float(LOCK_SHIFT[int(np.argmin(cost))])))
        else:                                                          # into the middle of the nearest lane
            c = cen.to_numpy()
            shifts.append((r.t, float(-c[np.argmin(np.abs(c))])))
    S = pd.DataFrame(shifts, columns=["t", "s"]).interpolate(limit_direction="both").fillna(0)
    S["s"] = S.s.rolling(int(LOCK_SMOOTH_S) * 2 + 1, center=True, min_periods=1).median()
    sft = np.interp(mp.t, S.t, S.s)
    th = np.radians(mp.heading)
    out = mp.copy()
    out["E"] = mp.E - sft * np.cos(th)                                # map moves right by s = we move left
    out["N"] = mp.N + sft * np.sin(th)
    out["lateral_shift"] = sft
    return out


def layer(drv: dict, mp: pd.DataFrame, li: LaneIndex) -> dict:
    """The map for the 3D view: lane edges and signalised junction outlines near the route (metres from
    an origin, three.js layout x = east, z = −north), and our map pose per frame."""
    E0, N0 = float(mp.E.iat[0]), float(mp.N.iat[0])
    edges = []
    for r in li.lanes.iloc[np.unique(li.own)].itertuples():          # only lanes near the route
        p = np.array(json.loads(r.shape))
        if len(p) < 2:
            continue
        d = np.gradient(p, axis=0)
        n = np.column_stack([d[:, 1], -d[:, 0]]) / np.maximum(np.hypot(*d.T), 1e-6)[:, None]   # right normal
        for side in (-0.5, 0.5):
            q = p + n * r.width * side
            edges.append(np.round(np.column_stack([q[:, 0] - E0, -(q[:, 1] - N0)]), 2).ravel().tolist())
    j = pd.read_parquet(OUT / "junctions.parquet")
    j = j[j.type == "traffic_light"]
    near = np.hypot(j.x.to_numpy()[:, None] - li.pts[::20, 0][None], j.y.to_numpy()[:, None] - li.pts[::20, 1][None]).min(1) < LAYER_M
    tl = [np.round(np.column_stack([np.array(json.loads(sh))[:, 0] - E0, -(np.array(json.loads(sh))[:, 1] - N0)]), 2).ravel().tolist()
          for sh in j[near]["shape"]]
    pose = np.column_stack([mp.t, mp.E - E0, mp.N - N0, mp.heading])[::1]
    houses = []
    bf = OUT / "buildings.parquet"
    if bf.exists():                                                  # OSM buildings near the route
        from scipy.spatial import cKDTree
        b = pd.read_parquet(bf)
        dist, _ = cKDTree(li.pts[::5]).query(b[["cx", "cy"]].to_numpy())
        for r in b[dist < LAYER_M].itertuples():
            q = np.array(json.loads(r.shape))
            houses.append([r.height, np.round(np.column_stack([q[:, 0] - E0, q[:, 1] - N0]), 1).ravel().tolist()])
    return dict(lanes=edges, tl_junctions=tl, buildings=houses, pose=np.round(pose, 2).tolist())


def build(drv: dict, full: bool = False):
    """map_pose.parquet (placed + locked sideways) and the 3D view's map layer (with full: also the
    pose table and lane index, for putting vehicles on map lanes)."""
    mp = pose_on_map(drv)
    em = json.loads((drv["folder"] / "event_match.json").read_text())["route"]
    to = utm()
    re_, rn = to.transform(em["lon"], em["lat"])
    route = np.column_stack([re_, rn])
    vf = drv["folder"] / "vpr.parquet"
    if USE_VPR and vf.exists():                                   # where Mapillary photos say we are
        cum = np.r_[0, np.cumsum(np.hypot(*np.diff(route, axis=0).T))]
        mp, n_vpr = vpr_pins(mp, pd.read_parquet(vf), route, cum)
        mp.attrs["vpr_pins"] = n_vpr
    if USE_PINS:
        from src.video.world import Pose
        cum = np.r_[0, np.cumsum(np.hypot(*np.diff(route, axis=0).T))]
        for _ in range(PIN_ROUNDS):                                # re-place our lights with the corrected position
            mp, n_pins = mapillary_pins(drv, mp, Pose(drv), route, cum)
        mp.attrs["pins"] = n_pins
    li = LaneIndex(route, LAYER_M)
    mp = lateral_lock(drv, mp, li)
    mp.to_parquet(drv["folder"] / "map_pose.parquet", index=False)
    lay = layer(drv, mp, li)
    return (lay, mp, li) if full else lay


SNAP_M = 2.5                 # a vehicle within this of a map lane's centre line (sideways) is on it
SNAP_DIR = 45.0              # … if the lane runs its way (± this)
# Per track, not per frame: snapped frame by frame a car between two lanes flipped between them (city
# drive 17_08: 1.1 lane switches per track, nearly all flicker). Each track's lanes are chosen for the
# whole track at once (Viterbi): a lane is "the k-th from the kerb in this direction" (SUMO's lane
# index — lane ids change at every edge even straight on); staying costs nothing, a lane change
# LANE_CHANGE, being off every lane OFF_LANE per frame. The car is then put on that lane's centre line
# and turned along it.
TRACK_LANES = True
LANE_SIGMA_M = 0.8           # sideways measurement noise of a vehicle's position
LANE_CHANGE = 25.0           # cost of a lane change: big sideways jumps 20.8 → 8.4 per 1000 steps (17_08), 5 % fewer on a lane (50: 5.8, 13 % fewer)
OFF_LANE = (SNAP_M / LANE_SIGMA_M) ** 2


def _lane_candidates(frames, mp, li, light_kind, labels, axis_max):
    """(frame index, object index) → {lane index k: (X of its centre line, lane row, lane direction)}."""
    mt = mp.t.to_numpy()
    cand = {}
    for fi, (t, objs) in enumerate(frames):
        if t < mt[0] or t > mt[-1]:
            continue
        r = mp.iloc[int(np.argmin(np.abs(mt - t)))]
        X, Z, _, own, rel = li.car_frame(r.E, r.N, r.heading, radius=60)
        if not len(X):
            continue
        for oi, o in enumerate(objs):
            if o[1] == light_kind or len(o) < 8 or labels.get(o[0]) == "parked":
                continue
            h = o[7]
            if abs((h + 90) % 180 - 90) > axis_max:                       # crossing / turning: not along our road
                continue
            m = np.flatnonzero((np.abs(Z - o[3]) < LANE_STEP_M * 1.5) & (np.abs((rel - h + 180) % 360 - 180) < SNAP_DIR))
            if not len(m):
                continue
            ks = li.lanes["index"].to_numpy()[own[m]]
            c = {}
            for k in np.unique(ks):
                mk = m[ks == k]
                j = mk[np.argmin(np.abs(X[mk] - o[2]))]
                c[int(k)] = (float(X[j]), int(own[j]), float(rel[j]))
            cand[(fi, oi)] = c
    return cand


def _viterbi(xs: list[float], cands: list[dict]) -> list:
    """Lane index per observation (None = off the map lanes), the cheapest sequence."""
    states = sorted({k for c in cands for k in c}) + [None]
    INF = 1e18
    cost = {s: 0.0 for s in states}
    back = []
    for x, c in zip(xs, cands):
        emit = {s: (OFF_LANE if s is None else (((c[s][0] - x) / LANE_SIGMA_M) ** 2 if s in c else INF)) for s in states}
        new, bk = {}, {}
        for s in states:
            prev = min(states, key=lambda q: cost[q] + (0 if q == s else LANE_CHANGE))
            new[s] = cost[prev] + (0 if prev == s else LANE_CHANGE) + min(emit[s], INF)
            bk[s] = prev
        cost = new
        back.append(bk)
    s = min(states, key=lambda q: cost[q])
    out = [s]
    for bk in back[:0:-1]:
        s = bk[s]
        out.append(s)
    return out[::-1]


def snap_vehicles(frames: list, mp: pd.DataFrame, li: LaneIndex, light_kind: int, labels: dict,
                  axis_max: float = 30.0) -> pd.DataFrame:
    """Vehicles running along the road onto a map lane of their direction: moved to its centre line
    sideways (and, per track, turned along it), the SUMO lane recorded (t, track_id, lane, edge)."""
    cand = _lane_candidates(frames, mp, li, light_kind, labels, axis_max)
    rows = []
    if not TRACK_LANES:
        for (fi, oi), c in cand.items():
            o = frames[fi][1][oi]
            k = min(c, key=lambda q: abs(c[q][0] - o[2]))
            if abs(c[k][0] - o[2]) <= SNAP_M:
                o[2] = round(c[k][0], 2)
                lane = li.lanes.iloc[c[k][1]]
                rows.append(dict(t=frames[fi][0], track_id=o[0], lane=lane.lane, edge=lane.edge))
        return pd.DataFrame(rows, columns=["t", "track_id", "lane", "edge"])
    by_track = {}
    for key in sorted(cand):
        by_track.setdefault(frames[key[0]][1][key[1]][0], []).append(key)
    for tid, keys in by_track.items():
        objs = [frames[fi][1][oi] for fi, oi in keys]
        path = _viterbi([o[2] for o in objs], [cand[k] for k in keys])
        for (fi, oi), o, k in zip(keys, objs, path):
            if k is None:
                continue
            x, row, rel = cand[(fi, oi)][k]
            o[2] = round(x, 2)
            o[7] = round(rel, 1)                                           # along its lane
            lane = li.lanes.iloc[row]
            rows.append(dict(t=frames[fi][0], track_id=tid, lane=lane.lane, edge=lane.edge, index=k))
    return pd.DataFrame(rows, columns=["t", "track_id", "lane", "edge", "index"])


def evaluate(drv: dict, mp: pd.DataFrame) -> dict:
    """Against the GPS answer key: placement error, and the matcher's own trace for comparison."""
    gps = drives.gps_track(drv)
    to = utm()
    ge, gn = to.transform(np.interp(mp.t, gps.video_s, gps.lon), np.interp(mp.t, gps.video_s, gps.lat))
    err = np.hypot(mp.E - ge, mp.N - gn)
    em = json.loads((drv["folder"] / "event_match.json").read_text())["trace"]
    te, tn = to.transform(np.interp(mp.t, em["video_s"], em["lon"]), np.interp(mp.t, em["video_s"], em["lat"]))
    e0 = np.hypot(te - ge, tn - gn)
    return dict(placed_median_m=round(float(np.median(err)), 1), placed_within_10m=f"{np.mean(err < 10):.0%}",
                placed_within_30m=f"{np.mean(err < 30):.0%}", matcher_median_m=round(float(np.median(e0)), 1),
                matcher_within_10m=f"{np.mean(e0 < 10):.0%}", matcher_within_30m=f"{np.mean(e0 < 30):.0%}")


# ── buildings (OpenStreetMap) ────────────────────────────────────────────────
OSM_FULL = ROOT / "data" / "osm" / "chisinau_full.osm"
LEVEL_M = 3.0                # metres per floor (building:levels)
DEFAULT_H = {"house": 6.0, "detached": 6.0, "garage": 3.0, "garages": 3.0, "shed": 3.0, "kiosk": 3.0,
             "apartments": 15.0, "residential": 9.0, "commercial": 9.0, "retail": 6.0, "industrial": 8.0,
             "church": 12.0, "school": 12.0}
DEFAULT_OTHER_H = 9.0


def extract_buildings():
    """OSM building outlines (UTM) with a height: height=*, else building:levels × 3 m, else a default by
    type — buildings.parquet (69k in Chișinău; 2.7k with a height, 6.3k with floors)."""
    ways, need = [], set()
    for ev, el in ET.iterparse(OSM_FULL, events=("end",)):
        if el.tag == "way":
            tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
            if "building" in tags:
                refs = [int(n.get("ref")) for n in el.findall("nd")]
                if len(refs) >= 4:
                    ways.append((int(el.get("id")), refs, tags))
                    need.update(refs)
            el.clear()
        elif el.tag in ("node", "relation"):
            if el.tag == "relation":
                el.clear()
    coords = {}
    for ev, el in ET.iterparse(OSM_FULL, events=("end",)):
        if el.tag == "node":
            i = int(el.get("id"))
            if i in need:
                coords[i] = (float(el.get("lon")), float(el.get("lat")))
            el.clear()
        elif el.tag in ("way", "relation"):
            el.clear()
    to = utm()
    rows = []
    for wid, refs, tags in ways:
        pts = [coords[r] for r in refs if r in coords]
        if len(pts) < 4:
            continue
        e, n = to.transform([p[0] for p in pts], [p[1] for p in pts])
        h, src = None, "default"
        try:
            if "height" in tags:
                h, src = float(tags["height"].split()[0].replace(",", ".")), "height"
            elif "building:levels" in tags:
                h, src = float(tags["building:levels"].replace(",", ".")) * LEVEL_M, "levels"
        except ValueError:
            pass
        if h is None or not 2 <= h <= 200:
            h, src = DEFAULT_H.get(tags["building"], DEFAULT_OTHER_H), "default"
        rows.append(dict(id=wid, kind=tags["building"], height=round(h, 1), height_source=src,
                         cx=float(np.mean(e)), cy=float(np.mean(n)),
                         shape=json.dumps(np.round(np.column_stack([e, n]), 1).tolist())))
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "buildings.parquet", index=False)
    print(f"→ {OUT / 'buildings.parquet'}: {len(df)} buildings; height from height=* {np.mean(df.height_source == 'height'):.0%}, "
          f"floors {np.mean(df.height_source == 'levels'):.0%}, default {np.mean(df.height_source == 'default'):.0%}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", nargs="?", help="drive key in configs/drives.yaml")
    ap.add_argument("--extract", action="store_true", help="network.net.xml → parquet tables (once)")
    ap.add_argument("--buildings", action="store_true", help="OSM buildings → buildings.parquet (once)")
    a = ap.parse_args()
    if a.extract:
        extract()
    if a.buildings:
        extract_buildings()
    if a.drive:
        drv = drives.load(a.drive)
        lay = build(drv)
        mp = pd.read_parquet(drv["folder"] / "map_pose.parquet")
        print(f"→ {drv['folder'] / 'map_pose.parquet'}: {len(mp)} frames; map layer: {len(lay['lanes'])} lane edges, "
              f"{len(lay['tl_junctions'])} signalised junctions; sideways shift median "
              f"{np.median(np.abs(mp.lateral_shift)):.1f} m", flush=True)
        if drives.gps_track(drv) is not None:
            print("   vs GPS (answer key):", evaluate(drv, mp), flush=True)

if __name__ == "__main__":
    main()
