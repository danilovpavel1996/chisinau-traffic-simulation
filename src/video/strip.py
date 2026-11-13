"""
strip.py
--------
The stitched top-down view: the whole drive as one long, straightened strip of road.

Every half second the sharpest part of the spatial view — the first NEAR_M metres of
ground beyond the bonnet, where one class-map pixel is a few centimetres — is painted
into the strip at the distance along the route the car is believed to be (video-only
position, event_match.py). Rows are metres along the route, columns metres left /
right of the car; each cell keeps the class seen most often there. Curves are
straightened, so a side street shows up as a gap in the curb line at a distance.

On top, what the map expects along the route (white): each junction with stubs for
its side streets (left / right, at the angle they leave), crossings (yellow) and
traffic signals (red). Where a curb gap in the strip and a map side street sit at
different distances, the video-only position is off there by that much.

Usage:
    .venv-video/bin/python -m src.video.strip <drive>

Output: data/video/processed/<drive>/strip.png (route end at the top, one pixel =
M_PER_PX metres) and strip.json (its scale and length, for the dashboard).
"""

from __future__ import annotations

import argparse
import json
import zlib

import cv2
import numpy as np
import pandas as pd

from src.video import drives
from src.video.bev import PALETTE, UNSEEN, Geometry
from src.video.compute import progress
from src.video.roads import load_graph
from src.video.scene import CODES

M_PER_PX = 0.25          # strip resolution
HALF_W = 16.0            # m, left / right of the car
NEAR_M = 14.0            # m of ground beyond the nearest visible ground used from each frame (≥ 0.5 s at 80 km/h)
STUB_M = 12.0            # m, length of a side-street stub on the map layer


def near_cells(geo: Geometry):
    """Ground cells of the sharp near band: (X, Z) in metres and their class-map index."""
    X = np.arange(-HALF_W, HALF_W, M_PER_PX) + M_PER_PX / 2
    Z = np.arange(geo.z_min, geo.z_min + NEAR_M, M_PER_PX) + M_PER_PX / 2
    XX, ZZ = np.meshgrid(X, Z)
    y = geo.y_h + geo.fz * geo.h / ZZ
    x = geo.cx + XX * (y - geo.y_h) / geo.h
    r = np.floor((y - geo.top) / geo.step).astype(int)
    c = np.floor(x / geo.step).astype(int)
    ok = (y < geo.bonnet) & (r >= 0) & (r < geo.shape[0]) & (c >= 0) & (c < geo.shape[1])
    return XX[ok], ZZ[ok], r[ok], c[ok]


def side_streets(G, route_nodes: list[int], route_d: dict[int, float], route_xy, kx: float):
    """Per route junction: (d, angle) of each street leaving it that the route does not take
    (angle relative to the route direction, + = right)."""
    out = []
    pos = {n: i for i, n in enumerate(route_nodes)}
    lat, lon, cum = route_xy
    for n, d in route_d.items():
        i = pos.get(n)
        if i is None or i == 0 or i == len(route_nodes) - 1:
            continue
        on_route = {route_nodes[i - 1], route_nodes[i + 1]}
        a_lat, a_lon = np.interp(d - 10, cum, lat), np.interp(d - 10, cum, lon)
        b_lat, b_lon = np.interp(d + 10, cum, lat), np.interp(d + 10, cum, lon)
        heading = np.degrees(np.arctan2((b_lon - a_lon) * kx, (b_lat - a_lat) * 111_320))
        for m in set(G.successors(n)) | set(G.predecessors(n)):
            if m in on_route:
                continue
            brg = np.degrees(np.arctan2((G.nodes[m]["x"] - G.nodes[n]["x"]) * kx, (G.nodes[m]["y"] - G.nodes[n]["y"]) * 111_320))
            out.append((d, (brg - heading + 180) % 360 - 180))
    return out


def run(drv: dict):
    folder = drv["folder"]
    dense = pd.read_parquet(folder / "scene_dense.parquet").sort_values("t").reset_index(drop=True)
    geo = Geometry(drv, dense.iloc[0])
    em = json.loads((folder / "event_match.json").read_text())
    tr, rt = em["trace"], em["route"]
    L = rt["d"][-1]
    H, W = int(L / M_PER_PX) + int(30 / M_PER_PX), int(2 * HALF_W / M_PER_PX)
    X, Z, r, c = near_cells(geo)
    col = np.clip(((X + HALF_W) / M_PER_PX).astype(int), 0, W - 1)
    ground = [i for i, name in enumerate(CODES) if name in PALETTE]
    votes = np.zeros((H, W, len(ground)), np.uint16)
    slot = np.full(len(CODES), -1)
    slot[ground] = np.arange(len(ground))
    t0, t1 = tr["video_s"][0], tr["video_s"][-1]
    bar = progress(len(dense), "stitching the strip")
    for rec in dense.to_dict("records"):
        bar.update(1)
        if not t0 <= rec["t"] <= t1:
            continue
        d = float(np.interp(rec["t"], tr["video_s"], tr["d"]))
        codes = np.frombuffer(zlib.decompress(rec["mask"]), np.uint8).reshape(json.loads(rec["mask_shape"]))
        k = slot[codes[r, c]]
        ok = k >= 0
        row = ((d + Z[ok]) / M_PER_PX).astype(int)
        inside = row < H
        np.add.at(votes, (row[inside], col[ok][inside], k[ok][inside]), 1)
    bar.close()

    # colour: the class seen most often in each cell; route end at the top
    lut = np.array([PALETTE[CODES[g]] for g in ground], np.uint8)
    img = np.where((votes.sum(axis=2) > 0)[..., None], lut[votes.argmax(axis=2)], np.array(UNSEEN, np.uint8))
    img = np.ascontiguousarray(img[::-1])
    y_of = lambda d: int(H - 1 - d / M_PER_PX)
    x_of = lambda X: int((X + HALF_W) / M_PER_PX)

    # the map's expectation along the route
    G = load_graph()
    kx = 111_320 * np.cos(np.radians(np.mean(rt["lat"])))
    sites = em.get("route_sites", [])
    junction_d = {s["node"]: s["d"] for s in sites if s["node"] is not None}
    for d, ang in side_streets(G, rt["nodes"], junction_d, (rt["lat"], rt["lon"], rt["d"]), kx):
        a = np.radians(ang)
        side = 1 if ang > 0 else -1
        x0, y0 = x_of(side * 4), y_of(d)
        x1 = x_of(side * 4 + np.sin(a) * STUB_M)
        y1 = y_of(d + np.cos(a) * STUB_M)
        cv2.line(img, (x0, y0), (x1, y1), (255, 255, 255), 2, cv2.LINE_AA)
    for s in sites:
        y = y_of(s["d"])
        if s["crossing"]:
            cv2.line(img, (x_of(-6), y), (x_of(6), y), (0, 210, 255), 1)
        if s["signal"]:
            cv2.circle(img, (x_of(7), y), 4, (60, 60, 235), -1, cv2.LINE_AA)
    for d in range(0, int(L) + 1, 100):                                    # distance ticks
        cv2.line(img, (0, y_of(d)), (6, y_of(d)), (150, 150, 150), 1)
        cv2.putText(img, f"{d}", (8, y_of(d) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)

    cv2.imwrite(str(folder / "strip.png"), img)
    (folder / "strip.json").write_text(json.dumps(dict(m_per_px=M_PER_PX, length_m=L, height_px=H, width_px=W,
                                                       half_width_m=HALF_W)))
    seen = (votes.sum(axis=2) > 0).any(axis=1).mean()
    print(f"→ {folder / 'strip.png'}: {W}×{H} px ({L:.0f} m of route, {M_PER_PX} m/px), ground seen along {seen:.0%} "
          f"of it; map: {len(sites)} sites", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    run(drives.load(ap.parse_args().drive))


if __name__ == "__main__":
    main()
