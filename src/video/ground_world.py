"""
ground_world.py
---------------
The road around the car for the 3D view, built from many frames at once instead of each frame
on its own.

Every analysed frame's segmentation (road, lane markings, crosswalks, curbs, sidewalk; 5 per
second) is projected onto the ground (bev.Geometry) and added to a raster in the fixed ground
frame (world.Pose). The road at time t is drawn from the frames between WINDOW_BEFORE seconds
before and WINDOW_AFTER after: a lane line seen in 90 frames out of 100 is a line, a speck seen
once is not; road hidden behind a car at t was seen a moment before or after; the road under our
own car was seen ahead of it. One frame's segmentation drew lines that broke up and wobbled.

Only a window of frames, never the whole drive: the pose drifts slowly (speed ±5 %), harmless
over 18 s, not over minutes. Implemented as a sliding sum — each frame is added when it enters
the window and subtracted when it leaves — on a raster that moves along with the car.

Per cell, from the counts: lane marking / crosswalk / curb when that class makes up at least
SHARE of the times the cell was seen, else sidewalk or road by majority.

Also read off per frame: lane lines and curbs across the road every 5 m ahead
(lanes_world.parquet: t, z, kind line / curb / edge_left / edge_right, x) — for putting vehicles in their lanes.
"""

from __future__ import annotations

import json
import subprocess
import zlib

import cv2
import numpy as np
import pandas as pd

from src.video.bev import Geometry
from src.video.compute import progress
from src.video.scene import CODES, DENSE_FPS

CELL = 0.1                       # m per raster cell
SPAN = 160.0                     # raster side, metres (moves with the car)
WINDOW_BEFORE, WINDOW_AFTER = 12.0, 6.0
Z_SEEN = (3.0, 20.0)             # ground used from this close … this far (beyond: one image row
                                 # covers > 0.4 m of road, rows leave gaps)
LANE_BANDS = np.arange(5.0, 41.0, 5.0)    # lane lines read every 5 m ahead …
LANE_HALF_M = 3.0                         # … over ±3 m of road (a dash of a dashed line is ~3 m, its gap ~6 m)
LANE_MIN_SHARE = 0.12                     # a line covers at least this share of the band's length
LANE_MIN_GAP_M = 1.8
FILL_M = 0.6                     # never-seen cells take the nearest seen cell's class within this
X_SEEN = 16.0
SAMPLE = 2                       # every 2nd pixel of the class map
SHARE = {"lane_marking": 0.28, "crosswalk": 0.30, "curb": 0.30}
LAYERS = ["road", "lane_marking", "crosswalk", "curb", "sidewalk"]


class Window:
    """Counts per layer (+ how often seen) on a raster in the ground frame that follows the car."""

    def __init__(self):
        n = int(SPAN / CELL)
        self.n = n
        self.count = np.zeros((len(LAYERS) + 1, n, n), np.float32)      # last layer: seen at all (float: OpenCV resamples it directly)
        self.origin = None                                               # world (x, z) of cell (0, 0)

    def recentre(self, x: float, z: float) -> bool:
        """Keep the car in the middle half of the raster (origin in 20-m steps); True if it moved."""
        if self.origin is not None and abs(x - (self.origin[0] + SPAN / 2)) < SPAN / 4 \
                and abs(z - (self.origin[1] + SPAN / 2)) < SPAN / 4:
            return False
        self.origin = np.floor(np.array([x - SPAN / 2, z - SPAN / 2]) / 20.0) * 20.0
        self.count[:] = 0
        return True

    def add(self, pts: tuple[np.ndarray, np.ndarray, np.ndarray], sign: int):
        x, z, layer = pts
        i = ((x - self.origin[0]) / CELL).astype(int)
        j = ((z - self.origin[1]) / CELL).astype(int)
        ok = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        i, j, layer = i[ok], j[ok], layer[ok].astype(np.int64)
        if not len(i):
            return
        i0, i1, j0, j1 = i.min(), i.max() + 1, j.min(), j.max() + 1          # count within this frame's box
        w, h = i1 - i0, j1 - j0
        flat = (j - j0) * w + (i - i0)
        L = len(LAYERS)
        per = np.bincount(layer * (h * w) + flat, minlength=L * h * w).reshape(L, h, w)
        self.count[:L, j0:j1, i0:i1] += sign * per
        self.count[L, j0:j1, i0:i1] += sign * per.sum(0)

    def sample(self, mx: np.ndarray, mz: np.ndarray) -> np.ndarray:
        """Counts at raster positions (cell coordinates) → (layers + 1, *mx.shape)."""
        return np.stack([cv2.remap(c, mx, mz, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0) for c in self.count])


def classes(cnt: np.ndarray, fill_px: float) -> np.ndarray:
    """Layer index + 1 per cell from its counts (0 = never seen); gaps filled from the nearest seen cell."""
    seen = cnt[-1]
    out = (seen > 0).astype(np.uint8) * np.uint8(LAYERS.index("road") + 1)
    out[cnt[LAYERS.index("sidewalk")] > 0.5 * seen] = LAYERS.index("sidewalk") + 1
    two = seen >= 2
    for name in ("curb", "crosswalk", "lane_marking"):                    # later = drawn on top
        k = LAYERS.index(name)
        out[two & (cnt[k] >= SHARE[name] * seen)] = k + 1
    # gaps between projected image rows (far away, or standing still): nearest seen cell
    dist, lab = cv2.distanceTransformWithLabels((out == 0).astype(np.uint8), cv2.DIST_L2, 3,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    src = out[out > 0]
    if len(src):
        fill = (out == 0) & (dist <= fill_px)
        out[fill] = src[lab[fill] - 1]
    return out


def lane_lines(layer: np.ndarray, X: np.ndarray, Z: np.ndarray, t: float) -> list[dict]:
    """Lane lines and curbs across the road every LANE_BANDS metres ahead (car frame): a column of
    the ground image whose band is mostly marking / curb. (t, z, kind, x) rows."""
    from scipy.signal import find_peaks
    px = 1 / abs(X[1] - X[0])
    out = []
    for z in LANE_BANDS:
        rows = np.abs(Z - z) <= LANE_HALF_M
        if not rows.any():
            continue
        band = layer[rows]
        for kind in ("lane_marking", "curb"):
            prof = (band == LAYERS.index(kind) + 1).mean(0)
            prof = np.convolve(prof, np.ones(3) / 3, mode="same")
            peaks, _ = find_peaks(prof, height=LANE_MIN_SHARE, distance=max(1, int(LANE_MIN_GAP_M * px)))
            out += [dict(t=t, z=float(z), kind="line" if kind == "lane_marking" else "curb", x=round(float(X[p]), 2))
                    for p in peaks]
        # the road surface we are on: the run of road columns around X = 0. Columns never seen
        # (a car stood there the whole window) do not end it — cars stand on the road
        is_road = np.isin(band, [LAYERS.index(k) + 1 for k in ("road", "lane_marking", "crosswalk")]).mean(0) > 0.5
        unseen = (band == 0).mean(0) > 0.5
        run = is_road | unseen
        c = int(np.argmin(np.abs(X)))
        if run[c]:
            off = np.flatnonzero(~run)
            lo = off[off < c].max() + 1 if (off < c).any() else 0
            hi = off[off > c].min() - 1 if (off > c).any() else len(run) - 1
            real = np.flatnonzero(is_road[lo:hi + 1]) + lo                # end at the last real road column
            if len(real):
                if lo > 0:
                    out.append(dict(t=t, z=float(z), kind="edge_left", x=round(float(X[real[0]]), 2)))
                if hi < len(run) - 1:
                    out.append(dict(t=t, z=float(z), kind="edge_right", x=round(float(X[real[-1]]), 2)))
    return out


def frame_points(geo: Geometry, row: dict, pose, t: float, lut: np.ndarray):
    """One frame's ground pixels in the ground frame: (x, z, layer)."""
    m = np.frombuffer(zlib.decompress(row["mask"]), np.uint8).reshape(json.loads(row["mask_shape"]))
    m = m[::SAMPLE, ::SAMPLE]
    rr, cc = np.nonzero(lut[m] >= 0)
    layer = lut[m[rr, cc]]
    y = row["top"] + (rr * SAMPLE + 0.5) * row["step"]
    x = (cc * SAMPLE + 0.5) * row["step"]
    xu, yu = geo.undistort(x, y)
    ok = (yu > geo.y_h + 3) & (y < geo.bonnet - 4)
    Z = geo.fz * geo.h / np.maximum(yu - geo.y_h, 1e-3)
    X = (xu - geo.cx) * geo.h / np.maximum(yu - geo.y_h, 1e-3)
    ok &= (Z > Z_SEEN[0]) & (Z < Z_SEEN[1]) & (np.abs(X) < X_SEEN)
    wx, wz = pose.to_world(t, X[ok], Z[ok])
    return wx, wz, layer[ok]


def render(drv: dict, geo: Geometry, pose, out_path, x_range, z_range, px_per_m: int, style: dict, dark,
           span: tuple[float, float] | None = None) -> int:
    """surround_ground.mp4 from the windowed ground raster (same frame grid and look as surround.py)."""
    folder = drv["folder"]
    meta = json.loads((folder / "meta.json").read_text())
    dense = pd.read_parquet(folder / "scene_dense.parquet").sort_values("t").reset_index(drop=True)
    rows = dense.to_dict("records")
    dt = dense.t.to_numpy()
    lut = np.full(len(CODES), -1, np.int8)
    for k, name in enumerate(LAYERS):
        lut[CODES.index(name)] = k
    lut[CODES.index("bike_lane")] = LAYERS.index("road")
    lut[CODES.index("curb_cut")] = LAYERS.index("curb")
    colours = np.array([dark] + [style[n] for n in LAYERS], np.uint8)      # 0 = never seen → dark
    W, H = int((x_range[1] - x_range[0]) * px_per_m), int((z_range[1] - z_range[0]) * px_per_m)
    X = x_range[0] + (np.arange(W) + 0.5) / px_per_m
    Z = z_range[1] - (np.arange(H) + 0.5) / px_per_m
    XX, ZZ = np.meshgrid(X, Z)
    times = np.round(np.arange(0, meta["end_s"], 1 / DENSE_FPS), 1)
    if span:                                          # a clip only (tests)
        times = times[(times >= span[0]) & (times <= span[1])]
    win = Window()
    cache: dict[int, tuple] = {}

    def points(j):
        if j not in cache:
            cache[j] = frame_points(geo, rows[j], pose, float(dt[j]), lut)
        return cache[j]

    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", str(DENSE_FPS), "-i", "-", "-c:v", "libx264", "-g", "1", "-crf", "20",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)], stdin=subprocess.PIPE)
    lo = hi = 0                                       # frames [lo, hi) are in the window
    lanes: list[dict] = []
    bar = progress(len(times), "surround ground (world)")
    for t in times:
        px, pz, _ = pose.at(t)
        new_lo = int(np.searchsorted(dt, t - WINDOW_BEFORE))
        new_hi = int(np.searchsorted(dt, t + WINDOW_AFTER, side="right"))
        if win.recentre(px, pz):                      # raster moved along (every ~40 m): rebuild it from
            for j in range(new_lo, new_hi):           # the frames in the window
                win.add(points(j), 1)
        else:
            for j in range(lo, min(new_lo, hi)):
                win.add(points(j), -1)
            for j in range(max(hi, new_lo), new_hi):
                win.add(points(j), 1)
        for j in [j for j in cache if j < new_lo]:
            del cache[j]
        lo, hi = new_lo, new_hi
        wx, wz = pose.to_world(t, XX, ZZ)
        mx = ((wx - win.origin[0]) / CELL - 0.5).astype(np.float32)
        mz = ((wz - win.origin[1]) / CELL - 0.5).astype(np.float32)
        layer = classes(win.sample(mx, mz), FILL_M * px_per_m)
        lanes += lane_lines(layer, X, Z, float(t))
        img = colours[layer]
        ff.stdin.write(np.ascontiguousarray(img).tobytes())
        bar.update(1)
    bar.close()
    ff.stdin.close()
    ff.wait()
    if not span:
        pd.DataFrame(lanes, columns=["t", "z", "kind", "x"]).to_parquet(folder / "lanes_world.parquet", index=False)
    return len(times)
