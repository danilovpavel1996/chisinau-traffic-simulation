"""
surround.py
-----------
Data for the Tesla-style surround view in the dashboard: the road around the car
and every detected vehicle and traffic light, in metres, 5 times a second.

    ground   surround_ground.mp4 — the ground plan (2 frames/s) in a clean style: dark
             road, white lane markings and crosswalks, dim sidewalk and curbs; undistorted,
             calibrated geometry (bev.Geometry), no map overlay
    objects  per detection frame: vehicles placed where their box touches the road (or by
             depth when the bottom is hidden by the bonnet), width from the box; traffic
             lights by depth (metres, depth.metric) with their height and colour

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

import numpy as np
import pandas as pd

from src.video import depth as depth_mod
from src.video import drives
from src.video.bev import Geometry
from src.video.compute import progress
from src.video.scene import CODES, DENSE_FPS, WIDTH

PX_PER_M = 16
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


def ground_video(drv: dict, geo: Geometry) -> int:
    folder = drv["folder"]
    meta = json.loads((folder / "meta.json").read_text())
    dense = pd.read_parquet(folder / "scene_dense.parquet").sort_values("t").reset_index(drop=True)
    (W, H), valid, rr, cc = geo.grid(PX_PER_M, X_RANGE, Z_RANGE)
    lut = np.array([STYLE.get(c, DARK) for c in CODES], np.uint8)
    times = np.round(np.arange(0, meta["end_s"], 1 / DENSE_FPS), 1)
    dt = dense.t.to_numpy()
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", str(DENSE_FPS), "-i", "-", "-c:v", "libx264", "-g", "1", "-crf", "20",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(folder / "surround_ground.mp4")],
                          stdin=subprocess.PIPE)
    bar = progress(len(times), "surround ground")
    for t in times:
        img = np.full((H, W, 3), DARK, np.uint8)
        j = int(np.argmin(np.abs(dt - t)))
        if abs(dt[j] - t) <= 0.3:
            r = dense.iloc[j]
            codes = np.frombuffer(zlib.decompress(r["mask"]), np.uint8).reshape(json.loads(r["mask_shape"]))
            img[valid] = lut[codes[rr, cc][valid]]
        ff.stdin.write(img.tobytes())
        bar.update(1)
    bar.close()
    ff.stdin.close()
    ff.wait()
    return len(times)


def objects(drv: dict, geo: Geometry) -> list:
    """[[t, [[track_id, kind, X, Z, width, Y, extra], …]], …]  (extra: crossing for vehicles, colour for lights)."""
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
    return out


def run(drv: dict):
    folder = drv["folder"]
    dense = pd.read_parquet(folder / "scene_dense.parquet").sort_values("t")
    geo = Geometry(drv, dense.iloc[0])
    n = ground_video(drv, geo)
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
    run(drives.load(ap.parse_args().drive))


if __name__ == "__main__":
    main()
