"""
bev.py
------
The spatial view: a bird's-eye plan of the road ahead of the car, every half
second, with two layers on top of each other:

    seen      what the camera sees, projected onto the ground: the segmentation
              (road, lane markings, crosswalks, sidewalk, curbs…) from the dense
              class maps (scene.py --colab --dense), and detected vehicles where
              their wheels touch the road
    expected  what the map says should be there, drawn from the video-only position
              and heading (event_match.py): OSM streets with their width (lanes),
              crossings, traffic signals. With GPS, the streets as seen from the
              true position are drawn too, as thin green lines.

Where the two layers disagree, the video-only position is wrong.

Geometry (flat road, pinhole camera): a ground point Z metres ahead and X metres to
the right appears at image row  y = y_h + f·h / Z  and column  x = c + f·X / Z,
y_h = horizon, f from the horizontal field of view, h = camera height, self-calibrated
from the cars seen: a car is ~CAR_W wide, so h = CAR_W · (bottom row − y_h) / box width
(2026_10_23: 1.28 m over 1,576 cars). Lateral distances depend on h only; depth on f·h —
and f from the nominal field of view is too small for this dashcam (turn angles come out
~1.5× too large vs GPS, depths too short): to be calibrated against the map next
(distance between matched junctions). The horizon is where the road visibly ends straight ahead on open
road (10th percentile over frames driving straight; cars ahead and bends only push it
down), minus the few pixels to infinity — the motion-based focus of expansion sat ~25 px
too high on 2026_10_23 and projected the tree line as road 10–40 m ahead.
Only ground classes are projected: upright things (cars, poles, signs) would smear into
long streaks; vehicles are drawn from their detection boxes instead.
Reliable to ~30–40 m; beyond that one class-map row spans several metres. A wide
dashcam lens is not a pinhole: lateral positions far to the side are compressed.

Usage:
    .venv-video/bin/python -m src.video.bev <drive>

Output: data/video/processed/<drive>/bev.mp4 (2 frames/s, one key frame each, so the
dashboard can seek it in sync with the video).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import zlib

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.video import drives
from src.video.compute import progress
from src.video.drives import ROOT
from src.video.event_match import lanes_of
from src.video.motion import ego_states
from src.video.roads import load_graph, street_name
from src.video.scene import CODES, DENSE_FPS, WIDTH

CAM_H = 1.3                  # m, camera height when no cars are seen to calibrate it
CAR_W = 1.8                  # m, typical car width (calibrates the camera height)
X_RANGE = (-16.0, 16.0)      # m, left … right of the car
Z_RANGE = (-4.0, 40.0)       # m, behind … ahead of the camera
PX_PER_M = 10                # canvas resolution → 320 × 440 px
LANE_M = 3.0                 # m per lane, for the map's road width
NEAR_M = 70                  # map streets drawn within this distance of the car
FEATURES = ROOT / "data" / "osm" / "junction_features.parquet"

ROAD_VISIBLE_M = 60          # the road "ends" in the image this far ahead, not at infinity
PALETTE = {                  # BGR; ground classes only, + vegetation dimmed (grass verges: where the road ends).
    # Guard rails, barriers, buildings stand upright: in a bend they smeared over the road ahead.
    "road": (72, 72, 72), "bike_lane": (110, 80, 60), "lane_marking": (235, 235, 235),
    "crosswalk": (0, 210, 255), "sidewalk": (140, 165, 185), "curb": (40, 130, 230), "curb_cut": (40, 130, 230),
    "vegetation": (40, 70, 40),
}
UNSEEN = (28, 28, 28)
VEHICLES = {"car", "bus", "truck", "motorcycle"}


def horizon_from_road(drv: dict, f: float) -> float:
    """Inference-image row of the horizon: where the road ends straight ahead on open road."""
    folder, cam = drv["folder"], drv["camera"]
    meta = json.loads((folder / "meta.json").read_text())
    e = ego_states(pd.read_parquet(folder / "ego.parquet"), meta["width"], cam["hfov_deg"], cam["hood"])
    dense = pd.read_parquet(folder / "scene_dense.parquet")
    road = [CODES.index(c) for c in ("road", "lane_marking", "crosswalk")]
    straight = (np.interp(dense.t, e.t, e.moving.astype(float)) > 0.5) & (np.interp(dense.t, e.t, e.turning.astype(float)) < 0.2)
    tops = []
    for r, ok in zip(dense.to_dict("records"), straight):
        if not ok:
            continue
        m = np.frombuffer(zlib.decompress(r["mask"]), np.uint8).reshape(json.loads(r["mask_shape"]))
        c = m.shape[1] // 2
        rows = np.flatnonzero(np.isin(m[:, c - 12:c + 12], road).mean(axis=1) > 0.5)
        if len(rows):
            tops.append(r["top"] + rows.min() * r["step"])
    return float(np.percentile(tops, 10)) - f * CAM_H / ROAD_VISIBLE_M   # a few px: nominal h is fine


def crosswalk_depths(drv: dict, geo: "Geometry", driven: tuple | None = None) -> np.ndarray:
    """
    (distance shown by the spatial view, distance driven) for frames approaching a crosswalk
    straight ahead: the view's distance to its near edge vs the distance driven until the
    car reaches it — `driven` = (times, cumulative metres); default the video odometer, in
    metres through the map (the whole drive = the route length).
    """
    f = drv["folder"]
    dense = pd.read_parquet(f / "scene_dense.parquet").sort_values("t").reset_index(drop=True)
    if driven is None:
        from src.video.event_match import odometer
        tt, mm = odometer(drv)
        driven = (tt, mm / mm[-1] * json.loads((f / "event_match.json").read_text())["route"]["d"][-1])
    tt, cum = driven
    cw = CODES.index("crosswalk")
    near = []
    for r in dense.to_dict("records"):
        m = np.frombuffer(zlib.decompress(r["mask"]), np.uint8).reshape(json.loads(r["mask_shape"]))
        c = m.shape[1] // 2
        rows = np.flatnonzero((m[:, c - 15:c + 15] == cw).mean(axis=1) > 0.3)
        y = r["top"] + (rows.max() + 1) * r["step"] if len(rows) else 0
        near.append((r["t"], geo.fz * geo.h / (y - geo.y_h) if y > geo.y_h + 2 else np.nan))
    n = pd.DataFrame(near, columns=["t", "Z"])
    out = []
    for _, run in n.dropna().groupby((n.Z.isna() | (n.Z.diff() > 3)).cumsum()):
        if len(run) < 4 or run.Z.min() > geo.z_min + 2.5 or run.Z.max() < 10:   # a full approach, to the bonnet
            continue
        t_end, z_end = run.t.iloc[-1], run.Z.iloc[-1]
        for t, z in zip(run.t, run.Z):
            if z >= 6:
                out.append((z, np.interp(t_end, tt, cum) - np.interp(t, tt, cum) + z_end))
    return np.array(out).reshape(-1, 2)


def camera_height(drv: dict, y_h: float, k: float) -> float:
    """Camera height from cars near the image centre: h = CAR_W · (bottom − horizon) / width."""
    meta = json.loads((drv["folder"] / "meta.json").read_text())
    d = pd.read_parquet(drv["folder"] / "detections.parquet")
    d = d[(d.cls == "car") & (d.conf > 0.5) & (d.y2 < meta["height"] * (1 - drv["camera"]["hood"]))]
    x, dy, w = (d.x1 + d.x2) / 2 * k, d.y2 * k - y_h, (d.x2 - d.x1) * k
    ok = (abs(x - WIDTH / 2) < WIDTH / 8) & (dy > 12) & (w > 25)
    return float(np.median(CAR_W * dy[ok] / w[ok])) if ok.sum() > 50 else CAM_H


class Geometry:
    """Ground (X, Z) ↔ inference-image pixels, and the BEV-cell → class-map lookup."""

    def __init__(self, drv: dict, dense_row: pd.Series):
        folder, cam = drv["folder"], drv["camera"]
        meta = json.loads((folder / "meta.json").read_text())
        self.k = WIDTH / meta["width"]                                 # full frame → inference px
        self.f = (WIDTH / 2) / np.tan(np.radians(cam["hfov_deg"] / 2))
        cal = folder / "calibration.json"
        c = json.loads(cal.read_text()) if cal.exists() else {}
        # depth: Z = fz·h / (y − y_h); lateral X = (x − c)·h / (y − y_h) needs no focal length.
        # Best: horizon and scale fz·h measured from parked cars + the overlay speed (ground.py);
        # else the horizon where the road ends and fz = f × the crosswalk depth factor (calibrate.py)
        measured = c.get("ground_status") == "calibrated"
        self.y_h = c["horizon_px"] if measured else horizon_from_road(drv, self.f)
        self.h = camera_height(drv, self.y_h, self.k)
        self.depth_scale = c.get("depth_scale", 1.0)
        self.fz = c["ground_scale"] / self.h if measured else self.f * self.depth_scale
        self.source = "parked cars + speed overlay" if measured else "map (turns, crosswalks)"
        self.cx = WIDTH / 2
        self.bonnet = meta["height"] * (1 - cam["hood"]) * self.k
        self.top, self.step = int(dense_row.top), int(dense_row.step)
        self.shape = json.loads(dense_row.mask_shape)
        # one class-map index per BEV cell (same for every frame)
        W = int((X_RANGE[1] - X_RANGE[0]) * PX_PER_M)
        H = int((Z_RANGE[1] - Z_RANGE[0]) * PX_PER_M)
        self.size = (W, H)
        X = X_RANGE[0] + (np.arange(W) + 0.5) / PX_PER_M
        Z = Z_RANGE[1] - (np.arange(H) + 0.5) / PX_PER_M
        XX, ZZ = np.meshgrid(X, Z)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = self.y_h + self.fz * self.h / ZZ
            x = self.cx + XX * (y - self.y_h) / self.h
        r = np.floor((y - self.top) / self.step).astype(int)
        c = np.floor(x / self.step).astype(int)
        self.valid = (ZZ > 0.5) & (y < self.bonnet) & (r >= 0) & (r < self.shape[0]) & (c >= 0) & (c < self.shape[1])
        self.r, self.c = np.where(self.valid, r, 0), np.where(self.valid, c, 0)
        self.z_min = self.fz * self.h / (self.bonnet - self.y_h)

    def ground(self, x_full: float, y_full: float) -> tuple[float, float] | None:
        """Full-frame pixel on the road → (X, Z) metres, or None above the horizon."""
        y, x = y_full * self.k, x_full * self.k
        if y <= self.y_h + 2:
            return None
        return (x - self.cx) * self.h / (y - self.y_h), self.fz * self.h / (y - self.y_h)

    def px(self, X, Z):
        """Ground metres → canvas pixel (int arrays)."""
        return (np.round((np.asarray(X) - X_RANGE[0]) * PX_PER_M).astype(int),
                np.round((Z_RANGE[1] - np.asarray(Z)) * PX_PER_M).astype(int))


def ascii_name(name: str) -> str:
    """'Domniței' → 'Domnitei' (the drawing font has no diacritics)."""
    import unicodedata
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()


class MapLayer:
    """OSM streets, crossings and signals around a pose, in the car's frame."""

    def __init__(self, G):
        self.G = G
        self.kx = 111_320 * np.cos(np.radians(47.03))
        pts, own = [], []
        self.edges = []
        for u, v, d in G.edges(data=True):
            if "geometry" in d:
                xs, ys = d["geometry"].xy
                c = np.column_stack([ys, xs])
            else:
                c = np.array([[G.nodes[u]["y"], G.nodes[u]["x"]], [G.nodes[v]["y"], G.nodes[v]["x"]]])
            oneway = bool(d.get("oneway", False))
            width = LANE_M * (lanes_of(d) if oneway else max(lanes_of(d), 2))     # lanes: whole street
            self.edges.append((c, width, oneway, street_name(d)))
            for p in c:
                pts.append(p); own.append(len(self.edges) - 1)
        pts = np.array(pts)
        self.tree = cKDTree(np.column_stack([pts[:, 0] * 111_320, pts[:, 1] * self.kx]))
        self.own = np.array(own)
        f = pd.read_parquet(FEATURES) if FEATURES.exists() else pd.DataFrame(columns=["lat", "lon", "kind"])
        self.feat = f
        self.feat_tree = cKDTree(np.column_stack([f.lat * 111_320, f.lon * self.kx])) if len(f) else None

    def local(self, latlon: np.ndarray, pose) -> tuple[np.ndarray, np.ndarray]:
        lat0, lon0, hd = pose
        dN = (latlon[:, 0] - lat0) * 111_320
        dE = (latlon[:, 1] - lon0) * self.kx
        h = np.radians(hd)
        return dE * np.cos(h) - dN * np.sin(h), dE * np.sin(h) + dN * np.cos(h)

    def near(self, pose):
        lat0, lon0, _ = pose
        idx = self.tree.query_ball_point([lat0 * 111_320, lon0 * self.kx], r=NEAR_M)
        return sorted(set(self.own[idx].tolist()))

    def offset(self, pose) -> float:
        """Lateral shift of the map so the car sits in the right half of a two-way street."""
        lat0, lon0, _ = pose
        _, i = self.tree.query([lat0 * 111_320, lon0 * self.kx])
        c, width, oneway, _ = self.edges[self.own[i]]
        return 0.0 if oneway else width / 4

    def draw(self, img, geo: Geometry, pose, color, filled: bool):
        off = self.offset(pose)
        band = img.copy()
        names = []
        for i in self.near(pose):
            c, width, oneway, name = self.edges[i]
            X, Z = self.local(c, pose)
            X = X - off
            u, v = geo.px(X, Z)
            line = np.column_stack([u, v]).astype(np.int32)
            if filled:
                cv2.polylines(band, [line], False, (255, 255, 255), max(1, int(width * PX_PER_M)), cv2.LINE_AA)
            cv2.polylines(img, [line], False, color, 1, cv2.LINE_AA)
            if filled and not name.startswith("("):
                j = int(np.argmin(np.hypot(X, Z - 15)))                 # label near 15 m ahead
                if -14 < X[j] < 14 and 2 < Z[j] < 38:
                    names.append((ascii_name(name.removeprefix("Strada ")), int(u[j]), int(v[j])))
        if filled:
            cv2.addWeighted(band, 0.22, img, 0.78, 0, img)
            if self.feat_tree is not None:
                lat0, lon0, _ = pose
                for j in self.feat_tree.query_ball_point([lat0 * 111_320, lon0 * self.kx], r=50):
                    f = self.feat.iloc[j]
                    X, Z = self.local(np.array([[f.lat, f.lon]]), pose)
                    u, v = geo.px(X - off, Z)
                    if f.kind == "traffic_signals":
                        cv2.circle(img, (int(u[0]), int(v[0])), 5, (60, 60, 235), -1, cv2.LINE_AA)
                    else:
                        cv2.rectangle(img, (int(u[0]) - 6, int(v[0]) - 3), (int(u[0]) + 6, int(v[0]) + 3),
                                      (0, 210, 255), 1)
            seen = set()
            for name, u, v in names:
                if name not in seen:
                    seen.add(name)
                    cv2.putText(img, name, (u + 4, v), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)


def poses(drv: dict, secs: np.ndarray):
    """Video-only pose (lat, lon, heading) per time from event_match.json, and the GPS pose if any."""
    out = {}
    em = drv["folder"] / "event_match.json"
    if em.exists():
        m = json.loads(em.read_text())
        tr, rt = m["trace"], m["route"]
        kx = 111_320 * np.cos(np.radians(47.03))
        r_lat, r_lon, r_d = map(np.asarray, (rt["lat"], rt["lon"], rt["d"]))
        d = np.interp(secs, tr["video_s"], tr["d"])
        lat, lon = np.interp(d, r_d, r_lat), np.interp(d, r_d, r_lon)
        a_lat, a_lon = np.interp(d - 8, r_d, r_lat), np.interp(d - 8, r_d, r_lon)
        b_lat, b_lon = np.interp(d + 8, r_d, r_lat), np.interp(d + 8, r_d, r_lon)
        hd = np.degrees(np.arctan2((b_lon - a_lon) * kx, (b_lat - a_lat) * 111_320)) % 360
        out["video"] = np.column_stack([lat, lon, hd])
    gps = drives.gps_track(drv)
    if gps is not None:
        g = gps.sort_values("video_s")
        kx = 111_320 * np.cos(np.radians(g.lat.mean()))
        lat, lon = np.interp(secs, g.video_s, g.lat), np.interp(secs, g.video_s, g.lon)
        a_lat, a_lon = np.interp(secs - 2, g.video_s, g.lat), np.interp(secs - 2, g.video_s, g.lon)
        b_lat, b_lon = np.interp(secs + 2, g.video_s, g.lat), np.interp(secs + 2, g.video_s, g.lon)
        dist = np.hypot((b_lat - a_lat) * 111_320, (b_lon - a_lon) * kx)
        hd = pd.Series(np.where(dist > 3, np.degrees(np.arctan2((b_lon - a_lon) * kx, (b_lat - a_lat) * 111_320)) % 360,
                                np.nan)).ffill().bfill().to_numpy()                # standing: keep last heading
        out["gps"] = np.column_stack([lat, lon, hd])
    return out


def render(geo: Geometry, codes: np.ndarray | None, boxes, pose_v, pose_g, layer: MapLayer, label: str) -> np.ndarray:
    W, H = geo.size
    lut = np.array([PALETTE.get(c, UNSEEN) for c in CODES], dtype=np.uint8)
    img = np.full((H, W, 3), UNSEEN, np.uint8)
    if codes is not None:
        cls = codes[geo.r, geo.c]
        img[geo.valid] = lut[cls[geo.valid]]
    for d in range(10, int(Z_RANGE[1]) + 1, 10):                        # distance rings
        _, v = geo.px(0, d)
        cv2.line(img, (0, int(v)), (W, int(v)), (90, 90, 90), 1)
        cv2.putText(img, f"{d} m", (4, int(v) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
    for X, Z in boxes:                                                   # vehicles where they touch the road
        u0, v0 = geo.px(X - 0.9, Z)
        u1, v1 = geo.px(X + 0.9, Z + 4.2)
        cv2.rectangle(img, (int(u0), int(v1)), (int(u1), int(v0)), (255, 200, 60), 1)
    if pose_g is not None:
        layer.draw(img, geo, pose_g, (90, 200, 90), filled=False)        # streets as seen from the GPS position
    if pose_v is not None:
        layer.draw(img, geo, pose_v, (255, 255, 255), filled=True)       # the map at the video-only position
    u0, v0 = geo.px(-0.9, 0.8)                                           # our car
    u1, v1 = geo.px(0.9, -3.4)
    cv2.rectangle(img, (int(u0), int(v0)), (int(u1), int(v1)), (60, 60, 230), -1)
    cv2.putText(img, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def run(drv: dict):
    folder = drv["folder"]
    dense = pd.read_parquet(folder / "scene_dense.parquet").sort_values("t").reset_index(drop=True)
    geo = Geometry(drv, dense.iloc[0])
    layer = MapLayer(load_graph())
    meta = json.loads((folder / "meta.json").read_text())
    times = np.round(np.arange(0, meta["end_s"], 1 / DENSE_FPS), 1)
    P = poses(drv, times)
    clock = drives.clock(drv, times).strftime("%H:%M:%S")
    det = pd.read_parquet(folder / "detections.parquet")
    det = det[det.cls.isin(VEHICLES) & (det.y2 < meta["height"] * (1 - drv["camera"]["hood"]))]
    det_t = np.sort(det.t.unique())
    by_t = {t: g for t, g in det.groupby("t")}
    dense_t = dense.t.to_numpy()

    W, H = geo.size
    out = folder / "bev.mp4"
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", str(DENSE_FPS), "-i", "-", "-c:v", "libx264", "-g", "1", "-crf", "23",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    t0, t1 = drv.get("street_window_s", [0, meta["end_s"]])
    bar = progress(len(times), "spatial view")
    for i, t in enumerate(times):
        j = int(np.argmin(np.abs(dense_t - t)))
        codes = None
        if abs(dense_t[j] - t) <= 0.3:
            r = dense.iloc[j]
            codes = np.frombuffer(zlib.decompress(r["mask"]), np.uint8).reshape(json.loads(r["mask_shape"]))
        boxes = []
        if len(det_t):
            k = int(np.argmin(np.abs(det_t - t)))
            if abs(det_t[k] - t) <= 0.25:
                for b in by_t[det_t[k]].itertuples():
                    g = geo.ground((b.x1 + b.x2) / 2, b.y2)
                    if g and g[1] < Z_RANGE[1]:
                        boxes.append(g)
        in_win = t0 <= t <= t1
        pose_v = P["video"][i] if "video" in P and in_win else None
        pose_g = P["gps"][i] if "gps" in P else None
        img = render(geo, codes, boxes, pose_v, pose_g, layer,
                     f"{clock[i]}  " + ("map at video-only position" if pose_v is not None else "outside street window"))
        ff.stdin.write(img.tobytes())
        bar.update(1)
    bar.close()
    ff.stdin.close()
    ff.wait()
    print(f"→ {out}: {len(times)} frames ({W}×{H}), horizon at row {geo.y_h:.0f}/{WIDTH * meta['height'] // meta['width']}, "
          f"camera height {geo.h:.2f} m (from cars), nearest visible ground {geo.z_min:.1f} m", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    run(drives.load(ap.parse_args().drive))


if __name__ == "__main__":
    main()
