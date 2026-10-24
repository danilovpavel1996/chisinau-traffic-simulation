"""
surround.py
-----------
Data for the Tesla-style surround view in the dashboard: the road around the car
and every detected vehicle and traffic light, in metres, 5 times a second.

    ground   surround_ground.mp4 — the ground plan (5 frames/s) in a clean style: dark
             road, white lane markings and crosswalks, dim sidewalk and curbs; undistorted,
             calibrated geometry (bev.Geometry), no map overlay. With memory: the ground
             seen in earlier frames is carried along with our own motion, so road hidden
             behind a car (or under ours) shows what was seen there a moment ago; it fades
             after MEMORY_S seconds without being seen again (faster while turning)
    objects  per detection frame: vehicles placed where their box touches the road (or by
             depth when the bottom is hidden by the bonnet), width from the box; traffic
             lights by depth (metres, depth.metric) with their height and colour; each vehicle's
             heading from its own track (positions over ~1 s, our own motion removed)

Car frame: X metres to the right, Z ahead of the camera, Y up.

Usage:
    .venv-video/bin/python -m src.video.surround <drive>

Output: processed/<drive>/surround_ground.mp4, surround_data.js (window.SURROUND).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import zlib

import cv2
import numpy as np
import pandas as pd

from src.video import depth as depth_mod
from src.video import drives
from src.video.bev import Geometry
from src.video.compute import progress
from src.video.scene import CODES, DENSE_FPS, WIDTH

PX_PER_M = 32            # ground texture: 1024 × 1536 px
X_RANGE = (-16.0, 16.0)
Z_RANGE = (-4.0, 44.0)
STYLE = {                     # BGR, Tesla-like: dark scene, white paint
    "road": (70, 64, 60), "bike_lane": (78, 70, 64), "lane_marking": (235, 235, 235), "crosswalk": (215, 215, 215),
    "sidewalk": (78, 72, 68), "curb": (120, 115, 110), "curb_cut": (120, 115, 110), "vegetation": (30, 34, 26),
}
DARK = (20, 16, 14)
KINDS = ["car", "bus", "truck", "motorcycle", "traffic_light"]
LIGHTS = {"red": 1, "yellow": 2, "green": 3}
WIDTH_M = {"car": (1.6, 2.1), "bus": (2.4, 2.6), "truck": (2.2, 2.6), "motorcycle": (0.6, 1.0)}
DEPTH_SYNC_S = 0.26           # use a depth map at most this far from the detection frame
MEMORY_S = 8.0                # ground not seen again for this long fades out …
FADE_S = 3.0                  # … over this many seconds
TURN_AGE = 0.15               # extra memory age per second per deg/s of rotation: the rotation
                              # scale is uncertain without a lens calibration (night) → shorter memory
MEMORY_MAX_Z = 20.0           # m: ground remembered only from this close (see ground_video)
HEADING_WINDOW_S = 0.6        # vehicle heading from its positions ± this long
MEASURED_MIN_MS = 4.0         # a measured heading only for vehicles clearly moving (see add_headings)


class EgoMotion:
    """Our car's motion from the video alone: distance (video speed, metres through the map
    route) and rotation (yaw), cumulative over time."""

    def __init__(self, drv: dict):
        from src.video.event_match import odometer
        from src.video.motion import ego_states
        folder, cam = drv["folder"], drv["camera"]
        meta = json.loads((folder / "meta.json").read_text())
        e = ego_states(pd.read_parquet(folder / "ego.parquet"), meta["width"], cam["hfov_deg"], cam["hood"])
        tt, mm = odometer(drv)
        em = folder / "event_match.json"
        L = json.loads(em.read_text())["route"]["d"][-1] if em.exists() else None
        k = L / mm[-1] if L and mm[-1] > 0 else 0.0                   # metres per (flow · second)
        v = np.where(e.moving, e.flow_smooth.fillna(0).clip(lower=0), 0.0)
        v = pd.Series(v).rolling(15, center=True, min_periods=1).median().to_numpy() * k
        dt = e.t.diff().fillna(0).to_numpy()
        self.t = e.t.to_numpy()
        self.dist = np.cumsum(v * dt)                                   # metres
        self.turn = np.cumsum(np.radians(-e.yaw_deg_s.to_numpy()) * dt)  # radians, + = to the right
        self.yaw_abs = np.abs(e.yaw_smooth.to_numpy())

    def between(self, t0: float, t1: float) -> tuple[float, float]:
        """(metres forward, radians turned right) from t0 to t1."""
        d = np.interp(t1, self.t, self.dist) - np.interp(t0, self.t, self.dist)
        a = np.interp(t1, self.t, self.turn) - np.interp(t0, self.t, self.turn)
        return float(d), float(a)

    def yaw_at(self, t: float) -> float:
        return float(np.interp(t, self.t, self.yaw_abs))


def to_earlier(X1, Z1, d: float, a: float):
    """Car-frame position at a later time → the car frame of an earlier time, the car having
    driven d metres and turned a radians to the right in between (along a short arc)."""
    X1, Z1 = np.asarray(X1, float), np.asarray(Z1, float)
    X = X1 * np.cos(a) + Z1 * np.sin(a) + d * np.sin(a / 2)
    Z = -X1 * np.sin(a) + Z1 * np.cos(a) + d * np.cos(a / 2)
    return X, Z


def ground_video(drv: dict, geo: Geometry) -> int:
    folder = drv["folder"]
    meta = json.loads((folder / "meta.json").read_text())
    dense = pd.read_parquet(folder / "scene_dense.parquet").sort_values("t").reset_index(drop=True)
    (W, H), valid, rr, cc = geo.grid(PX_PER_M, X_RANGE, Z_RANGE)
    near_rows = int((Z_RANGE[1] - (geo.z_min + 0.5)) * PX_PER_M)        # first row nearer than what is seen
    lut = np.array([STYLE.get(c, DARK) for c in CODES], np.uint8)
    times = np.round(np.arange(0, meta["end_s"], 1 / DENSE_FPS), 1)
    dt = dense.t.to_numpy()
    ego = EgoMotion(drv)
    X = X_RANGE[0] + (np.arange(W) + 0.5) / PX_PER_M                     # cell centres, metres
    Z = Z_RANGE[1] - (np.arange(H) + 0.5) / PX_PER_M
    XX, ZZ = np.meshgrid(X, Z)
    to_px = lambda Xm, Zm: (((Xm - X_RANGE[0]) * PX_PER_M - 0.5).astype(np.float32),
                            ((Z_RANGE[1] - Zm) * PX_PER_M - 0.5).astype(np.float32))
    ground_codes = np.array([c in STYLE for c in CODES])
    memory = np.full((H, W, 3), DARK, np.uint8)
    age = np.full((H, W), 1e9, np.float32)
    t_prev = None
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", str(DENSE_FPS), "-i", "-", "-c:v", "libx264", "-g", "1", "-crf", "20",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(folder / "surround_ground.mp4")],
                          stdin=subprocess.PIPE)
    bar = progress(len(times), "surround ground")
    for t in times:
        if t_prev is not None:                         # carry the remembered ground along with our motion
            d, a = ego.between(t_prev, t)
            mx, my = to_px(*to_earlier(XX, ZZ, d, a))
            memory = cv2.remap(memory, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=DARK)
            age = cv2.remap(age, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1e9)
            age += (t - t_prev) * (1 + TURN_AGE * ego.yaw_at(t))
        t_prev = t
        j = int(np.argmin(np.abs(dt - t)))
        if abs(dt[j] - t) <= 0.3:
            r = dense.iloc[j]
            codes = np.frombuffer(zlib.decompress(r["mask"]), np.uint8).reshape(json.loads(r["mask_shape"]))
            cls = codes[rr, cc]
            seen = valid & ground_codes[cls]          # ground seen now (not cars, poles, sky)
            sharp = seen & (ZZ <= MEMORY_MAX_Z)        # only the sharp near field is remembered: far ground is
            memory[sharp] = lut[cls[sharp]]            # coarse and smears into streaks when carried close
            age[sharp] = 0
        else:
            seen = np.zeros((H, W), bool)
        fade = np.clip((age - (MEMORY_S - FADE_S)) / FADE_S, 0, 1)[..., None]
        img = (memory * (1 - fade) + np.array(DARK) * fade).astype(np.uint8)
        far = seen & (ZZ > MEMORY_MAX_Z)                # far ground: as seen right now
        img[far] = lut[cls[far]]
        never = age > 1e8                              # under / just ahead of our car, never seen yet:
        img[near_rows:][never[near_rows:]] = img[near_rows - 1][None].repeat(H - near_rows, 0)[never[near_rows:]]
        ff.stdin.write(img.tobytes())
        bar.update(1)
    bar.close()
    ff.stdin.close()
    ff.wait()
    return len(times)


def objects(drv: dict, geo: Geometry) -> list:
    """[[t, [[track_id, kind, X, Z, width, Y, extra(, heading)], …]], …]  (extra: crossing for vehicles,
    colour for lights; heading in degrees for vehicles, see add_headings)."""
    folder, cam = drv["folder"], drv["camera"]
    meta = json.loads((folder / "meta.json").read_text())
    k = WIDTH / meta["width"]
    hood_y = meta["height"] * (1 - cam["hood"]) - 4
    det = pd.read_parquet(folder / "detections.parquet")
    labels = pd.read_parquet(folder / "tracks.parquet").set_index("track_id").label.to_dict()
    td, maps = depth_mod.load(drv) if (folder / "depth.parquet").exists() else (np.array([]), [])
    metric = {}

    def depth_at(t, x1, y1, x2, y2):
        if not len(td):
            return None
        j = int(np.argmin(np.abs(td - t)))
        if abs(td[j] - t) > DEPTH_SYNC_S:
            return None
        if j not in metric:
            metric.clear()
            metric[j] = depth_mod.metric(drv, maps[j])
        D = metric[j]
        r0, r1 = int(y1 * k) // depth_mod.DOWN, int(y2 * k) // depth_mod.DOWN + 1
        c0, c1 = int(x1 * k) // depth_mod.DOWN, int(x2 * k) // depth_mod.DOWN + 1
        patch = D[max(r0, 0):min(r1, D.shape[0]), max(c0, 0):min(c1, D.shape[1])]
        return float(np.median(patch)) if patch.size else None

    out = []
    bar = progress(det.frame.nunique(), "surround objects")
    for t, g in det.sort_values("t").groupby("t"):
        rows = []
        for b in g.itertuples():
            kind = KINDS.index(b.cls)
            xc = (b.x1 + b.x2) / 2
            if b.cls == "traffic_light":
                Z = depth_at(t, b.x1, b.y1, b.x2, b.y2)
                if Z is None or not 3 < Z < 70:
                    continue
                xu, yu = geo.undistort(xc * k, (b.y1 + b.y2) / 2 * k)
                X, Y = (xu - geo.cx) * Z / geo.fz, geo.h - (yu - geo.y_h) * Z / geo.fz
                if 1.5 < Y < 9:
                    rows.append([int(b.track_id), kind, round(X, 2), round(Z, 2), 0.4, round(Y, 2), LIGHTS.get(b.light, 0)])
                continue
            if b.y2 < hood_y and (gp := geo.ground(xc, b.y2)) is not None:            # where it touches the road
                X, Z = gp
                left, right = geo.ground(b.x1, b.y2), geo.ground(b.x2, b.y2)
                w = (right[0] - left[0]) if left and right else 1.8
            else:                                                                      # bottom hidden: depth
                Z = depth_at(t, b.x1, (b.y1 + b.y2) / 2, b.x2, b.y2)
                if Z is None:
                    continue
                xu, _ = geo.undistort(xc * k, b.y2 * k)
                X, w = (xu - geo.cx) * Z / geo.fz, (b.x2 - b.x1) * k * Z / geo.fz
            if not (1.5 < Z < 70 and abs(X) < 30):
                continue
            lo, hi = WIDTH_M[b.cls]
            rows.append([int(b.track_id), kind, round(X, 2), round(Z, 2), round(float(np.clip(w, lo, hi)), 2), 0,
                         int(labels.get(b.track_id) == "crossing")])
        out.append([round(float(t), 2), rows])
        bar.update(1)
    bar.close()
    add_headings(out, EgoMotion(drv), labels)
    return out


def add_headings(frames: list, ego: EgoMotion, labels: dict):
    """
    Each vehicle's heading (degrees, 0 = our direction, + = to the right), appended as the 8th
    value of its row. From the traffic step's labels, which compare the vehicle's image motion
    with the road beside it (robust): ahead / overtaking / parked → along our road, oncoming →
    180°, crossing → ±90° by the way it moves across the image. Only turning or unlabelled
    vehicles get a measured heading (track positions ± HEADING_WINDOW_S, our motion removed),
    and only when clearly moving (≥ MEASURED_MIN_MS): our own speed from the video is off by up
    to ~3 m/s over a fifth of a second, so slower apparent motion is noise — measuring every
    vehicle this way gave parked cars headings and oncoming cars a median of 30° (2026_10_23).
    """
    tracks: dict[int, list] = {}
    for t, rows in frames:
        for o in rows:
            if o[1] != KINDS.index("traffic_light"):
                tracks.setdefault(o[0], []).append((t, o))
    for tid, pts in tracks.items():
        label = labels.get(tid, "")
        ts = np.array([p[0] for p in pts])
        xs = np.array([p[1][2] for p in pts])
        fixed = {"oncoming": 180.0, "same_dir": 0.0, "overtook_ego": 0.0, "passed_by_ego": 0.0, "parked": 0.0}
        if label == "crossing":
            fixed["crossing"] = 90.0 if np.polyfit(ts, xs, 1)[0] > 0 else -90.0 if len(ts) > 1 else 90.0
        for i, (t, o) in enumerate(pts):
            heading = fixed.get(label)
            if heading is None:
                heading = 0.0
                j0 = int(np.searchsorted(ts, t - HEADING_WINDOW_S))
                j1 = int(np.searchsorted(ts, t + HEADING_WINDOW_S, side="right")) - 1
                if j1 > j0 and ts[j1] - ts[j0] >= 0.4:
                    (ta, a), (tb, b) = pts[j0], pts[j1]
                    d, turn = ego.between(ta, tb)
                    Xb, Zb = to_earlier(b[2], b[3], d, turn)
                    dx, dz = float(Xb) - a[2], float(Zb) - a[3]
                    if np.hypot(dx, dz) / (tb - ta) >= MEASURED_MIN_MS:
                        _, since = ego.between(ta, t)
                        heading = float(np.degrees(np.arctan2(dx, dz) - since))
            o.append(round((heading + 180) % 360 - 180, 1))

def run(drv: dict, objects_only: bool = False):
    folder = drv["folder"]
    dense = pd.read_parquet(folder / "scene_dense.parquet").sort_values("t")
    geo = Geometry(drv, dense.iloc[0])
    n = "kept" if objects_only else ground_video(drv, geo)
    frames = objects(drv, geo)
    data = dict(ground=dict(video=f"processed/{drv['key']}/surround_ground.mp4", x_range=X_RANGE, z_range=Z_RANGE),
                kinds=KINDS, camera_height=round(geo.h, 2), frames=frames)
    (folder / "surround_data.js").write_text("window.SURROUND = " + json.dumps(data, separators=(",", ":")) + ";\n")
    n_obj = sum(len(f[1]) for f in frames)
    print(f"→ {folder / 'surround_ground.mp4'} ({n} frames) and surround_data.js: {len(frames)} detection frames, "
          f"{n_obj} objects ({(folder / 'surround_data.js').stat().st_size / 1e6:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    ap.add_argument("--objects", action="store_true", help="only the vehicles / lights (keep the ground video)")
    a = ap.parse_args()
    run(drives.load(a.drive), a.objects)


if __name__ == "__main__":
    main()
