"""
speed.py
--------
Our speed from the video, in km/h: optical flow × metric depth.

Driving forward, a scene point at depth Z moves away from the vanishing point by T·r/Z per
frame (T metres driven, r its distance from the vanishing point in focal-length units), so
each tracked point gives T = radial motion · Z / r — metric, the same in a narrow street
(buildings 10 m away stream past fast) and in the open (fields 200 m away barely move),
where plain image motion confuses the two. Per analysed frame (5 per second):

    points     tracked through FLOW_PAIRS adjacent frame pairs (dense.py), vehicles excluded,
               undistorted (the wide lens squeezes motion near the edges)
    depth      Depth Anything fitted to the road plane (depth.metric), rescaled per frame by
               this frame's own road pixels (frame_factor: the model's scale wanders)
    rotation   fitted to distant points (turning and pitching move them, driving hardly does)
               and removed from all points
    speed      median T over near points; still points dropped when most move (raindrops,
               the dashboard's reflection in the windscreen)

The result is metric without calibration (2026_10_23_10_06: 0.99× GPS). If the video has
the dashcam's speed overlay, that one factor is still taken from it (GPS speed only, never
positions).

    2026_10_23_10_06 vs GPS          correlation   median error   90 % within
    scene flow (event_match)             0.47        13.7 km/h
    radial flow × depth, 1 pair          0.79         9.2
    + undistort, rotation removed        0.81         7.4
    + per-frame depth, 2-s median        0.84         6.6
    + windscreen points dropped          0.85         6.7         20.8 km/h

Usage:
    .venv-video/bin/python -m src.video.speed <drive>

Output: data/video/processed/<drive>/speed.parquet (t, kmh, kmh_raw, yaw_dps, depth_k, points);
        speed_* keys in calibration.json
"""

from __future__ import annotations

import argparse
import json
import zlib

import numpy as np
import pandas as pd

from src.video import depth as depth_mod
from src.video import drives
from src.video.bev import Geometry
from src.video.scene import CODES

Z_RANGE_M = (3.0, 80.0)
ROUNDS = 3               # rotation ↔ speed refinements
FAR_M = 30.0             # rotation from points this far (forward motion barely moves them)
NEAR_MIN_M = 8.0         # speed from points between these distances (closer: blurred at speed,
NEAR_MAX_M = 40.0        # tracking slips onto the windscreen's reflection)
STILL_PX = 0.3           # a point moving less than this (pixels per frame) …
STATIC_CELL = 16         # image cells (px) checked for windscreen reflections …
STATIC_SHARE = 0.5       # … static if most points there stay still while the car drives
MOVING_SHARE = 0.4       # … is windscreen dirt if at least this share of points move
# vanishing point of travel, x in focal-length units from the image centre: this dashcam looks
# ~4.5° right of the direction of travel — where points left and right of it agree on the speed
# (median-only fit, 2026_10_23_10_06: −0.08; 2026_10_24_15_48: −0.12; one side read 0.63× the
# other at 0). With the rotation refined (ROUNDS) the exact value matters little (±0.4 km/h).
FOE_X = -0.08
MIN_R = 0.1              # … and at least this far from the vanishing point (focal-length units)
MIN_POINTS = 30
# no speed from: the road surface (smooth, blurred at speed, under the dashboard's reflection —
# 0.03× GPS out of town), sky (wires: depth unknown), "other" (mostly the reflection itself)
NO_SPEED = [CODES.index(c) for c in ("road", "lane_marking", "crosswalk", "curb", "curb_cut", "sidewalk",
                                     "bike_lane", "sky", "other")]
SMOOTH_S = 2.0           # median over this window
FRAME_FIT_M = (6.0, 25.0)  # road pixels used to rescale each frame's depth
FRAME_FIT_MIN = 300
FRAME_FIT_CLIP = (0.6, 1.6)


def fit_motion(xn, yn, du, dv, Z, ex: float, ey: float, still_n: float, usable=None) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Speed per frame from tracked points with known depth (pinhole, normalised coordinates).

    The car moves along its heading, toward the vanishing point (ex, ey). A point at depth Z,
    at (x, y) in focal-length units, moves by
        du = (x − ex)·T/Z + x·y·wx − (1 + x²)·wy + y·wz
        dv = (y − ey)·T/Z + (1 + y²)·wx − x·y·wy − x·wz
    (T metres, w radians, per frame). Rotation moves points regardless of their distance,
    forward motion hardly moves distant ones: so the rotation is fitted to distant points
    (Z ≥ FAR_M, robust least squares), removed from all, and every remaining point gives
    T = radial motion · Z / distance from the vanishing point; the median is the speed.
    (Fitting T and w together let the rotation absorb part of the speed — a steady
    −10°/s "turn" on straight roads; correlation with GPS 0.68.)
    still_n: STILL_PX in focal-length units; usable: points allowed to give the speed.
    Returns (T, [wx, wy, wz], T of each point used, which points)."""
    # raindrops and the dashboard's reflection on the windscreen stay put in the image: drop
    # still points when most points move (else the car itself stands still)
    still = np.hypot(du, dv) < still_n
    use = ~still if (~still).sum() >= MOVING_SHARE * len(still) else np.ones_like(still)
    far = use & (Z >= FAR_M)
    if far.sum() < MIN_POINTS:
        far = use & (Z >= np.percentile(Z[use], 70))
    xf, yf = xn[far], yn[far]
    A = np.vstack([np.column_stack([xf * yf, -(1 + xf ** 2), yf]),
                   np.column_stack([1 + yf ** 2, -xf * yf, -xf])])
    ox, oy = xn - ex, yn - ey
    r = np.hypot(ox, oy)
    near = use & (r > MIN_R) & (Z >= NEAR_MIN_M) & (Z < NEAR_MAX_M)
    if usable is not None:
        near &= usable
    # distant points still move a little from driving (4 px per frame at 30 m, 80 km/h): remove
    # the motion the current speed implies before fitting the rotation again, and repeat
    t_est, T = 0.0, np.array([])
    for _ in range(ROUNDS):
        b = np.concatenate([du[far] - ox[far] * t_est / Z[far], dv[far] - oy[far] * t_est / Z[far]])
        w = np.ones_like(b)
        for _ in range(5):
            sw = np.sqrt(w)
            rot = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)[0]
            res = np.abs(A @ rot - b)
            c = 1.345 * max(np.median(res) / 0.6745, 1e-6)
            w = np.where(res <= c, 1.0, c / np.maximum(res, 1e-12))
        wx, wy, wz = rot
        ru = du - (xn * yn * wx - (1 + xn ** 2) * wy + yn * wz)
        rv = dv - ((1 + yn ** 2) * wx - xn * yn * wy - xn * wz)
        T = (ru * ox + rv * oy)[near] / r[near] * Z[near] / r[near]
        if len(T) < MIN_POINTS:
            break
        t_est = max(float(np.median(T)), 0.0)
    return (float(np.median(T)) if len(T) >= MIN_POINTS else np.nan), rot, T, near


def frame_factor(geo: Geometry, row: dict, D: np.ndarray) -> float:
    """This frame's depth error: Depth Anything's scale wanders from frame to frame (one drive-wide
    fit, depth.fit_to_ground, leaves e.g. 69 km/h read as 19). The road plane gives every road
    pixel's true distance, so the median ratio over this frame's road pixels rescales it."""
    from src.video.depth import GROUND
    from src.video.scene import CODES
    m = np.frombuffer(zlib.decompress(row["mask"]), np.uint8).reshape(json.loads(row["mask_shape"]))
    rr, cc = np.nonzero(np.isin(m, [CODES.index(c) for c in GROUND]))
    y, x = row["top"] + (rr + 0.5) * row["step"], (cc + 0.5) * row["step"]
    xu, yu = geo.undistort(x, y)
    ok = (yu > geo.y_h + 4) & (y < geo.bonnet - 4)
    Zg = geo.fz * geo.h / (yu[ok] - geo.y_h)
    Zd = D[np.clip((y[ok] / depth_mod.DOWN).astype(int), 0, D.shape[0] - 1),
           np.clip((x[ok] / depth_mod.DOWN).astype(int), 0, D.shape[1] - 1)]
    keep = (Zg > FRAME_FIT_M[0]) & (Zg < FRAME_FIT_M[1])
    if keep.sum() < FRAME_FIT_MIN:
        return 1.0
    return float(np.clip(np.median(Zg[keep] / Zd[keep]), *FRAME_FIT_CLIP))


def static_cells(motion: pd.DataFrame, shape: tuple[int, int]) -> np.ndarray:
    """Image cells (STATIC_CELL px) where tracked points stay put while the car drives: the
    dashboard's reflection in the windscreen, the windscreen's frame, dirt. Corners there are
    strong, so evenly spread points (track_points.py) put many of them on the reflection over
    the road — road points then read 0.03× the true speed (2026_10_24_15_48)."""
    H, W = shape[0] // STATIC_CELL + 1, shape[1] // STATIC_CELL + 1
    still, total = np.zeros((H, W)), np.zeros((H, W))
    for b in motion.pts.iloc[::7]:
        a = np.frombuffer(zlib.decompress(b), np.float16).reshape(-1, 4).astype(np.float32)
        m = np.hypot(a[:, 2], a[:, 3])
        if len(m) < MIN_POINTS or np.median(m) < 2.0:          # the car (nearly) stands: says nothing
            continue
        r, c = (a[:, 1] // STATIC_CELL).astype(int), (a[:, 0] // STATIC_CELL).astype(int)
        np.add.at(total, (r, c), 1)
        np.add.at(still, (r, c), m < STILL_PX)
    return (total >= 20) & (still > STATIC_SHARE * np.maximum(total, 1))


def raw_speed(drv: dict, points: list | None = None, foe_x: float | None = None, every: int = 1) -> pd.DataFrame:
    """Speed per analysed frame. points: if a list, every used point is added to it
    (t, class, Z, x, y, km/h) — to study which parts of the scene read too fast or slow.
    foe_x: vanishing point of travel (default FOE_X); every: use only every n-th analysed frame."""
    folder = drv["folder"]
    foe_x = FOE_X if foe_x is None else foe_x
    meta = json.loads((folder / "meta.json").read_text())
    fps = meta["fps"]
    geo = Geometry(drv, pd.read_parquet(folder / "scene_dense.parquet").iloc[0])
    motion = pd.read_parquet(folder / "motion.parquet")
    dense = pd.read_parquet(folder / "scene_dense.parquet").sort_values("t").reset_index(drop=True)
    dense_t = dense.t.to_numpy()
    td, maps = depth_mod.load(drv)
    static = static_cells(motion, (int(geo.bonnet) + 10, 960))
    rows = []
    keep_t = set(np.unique(motion.t)[::every])
    for t, g in motion.groupby("t"):
        if t not in keep_t:
            continue
        j = int(np.argmin(np.abs(td - t)))
        if abs(td[j] - t) > 0.05:
            continue
        D = depth_mod.metric(drv, maps[j])
        i = int(np.argmin(np.abs(dense_t - t)))
        k = frame_factor(geo, dense.iloc[i].to_dict(), D) if abs(dense_t[i] - t) < 0.05 else 1.0
        D = D * k
        row = dense.iloc[i]
        cmap = np.frombuffer(zlib.decompress(row["mask"]), np.uint8).reshape(json.loads(row["mask_shape"]))
        ms = []
        for r in g.itertuples():
            a = np.frombuffer(zlib.decompress(r.pts), np.float16).reshape(-1, 4).astype(np.float32)
            p, d = a[:, :2], a[:, 2:]
            Z = D[np.clip((p[:, 1] / depth_mod.DOWN).astype(int), 0, D.shape[0] - 1),
                  np.clip((p[:, 0] / depth_mod.DOWN).astype(int), 0, D.shape[1] - 1)]
            ok = (Z > Z_RANGE_M[0]) & (Z < Z_RANGE_M[1]) & (p[:, 1] < geo.bonnet - 5)
            ok &= ~static[np.clip((p[:, 1] // STATIC_CELL).astype(int), 0, static.shape[0] - 1),
                          np.clip((p[:, 0] // STATIC_CELL).astype(int), 0, static.shape[1] - 1)]
            if ok.sum() < MIN_POINTS:
                continue
            px, py = p[ok, 0], p[ok, 1]
            ri = ((py - row["top"]) / row["step"]).astype(int)
            cls = np.where(ri >= 0, cmap[np.clip(ri, 0, cmap.shape[0] - 1),
                                         np.clip((px / row["step"]).astype(int), 0, cmap.shape[1] - 1)], 255)
            # undistort both ends of each track: the wide lens squeezes motion near the edges
            x0, y0 = geo.undistort(p[ok, 0], p[ok, 1])
            x1, y1 = geo.undistort(p[ok, 0] + d[ok, 0], p[ok, 1] + d[ok, 1])
            xn, yn = (x0 - geo.cx) / geo.fl, (y0 - geo.cy) / geo.fl
            T, rot, Tp, near = fit_motion(xn, yn, (x1 - x0) / geo.fl, (y1 - y0) / geo.fl, Z[ok], foe_x, (geo.y_h - geo.cy) / geo.fl,
                                STILL_PX / geo.fl, ~np.isin(cls, NO_SPEED))
            ms.append((T, rot[1]))
            if points is not None and len(Tp):
                points.append(pd.DataFrame(dict(t=float(t), cls=cls[near], Z=Z[ok][near], x=px[near], y=py[near],
                                                kmh=Tp * fps * 3.6)))
        ms = [m for m in ms if np.isfinite(m[0])]
        if ms:
            m = np.median(ms, axis=0)
            rows.append(dict(t=float(t), kmh_raw=max(float(m[0]), 0.0) * fps * 3.6,
                             yaw_dps=float(np.degrees(m[1]) * fps), depth_k=k, points=len(ms)))
        else:
            rows.append(dict(t=float(t), kmh_raw=np.nan, yaw_dps=np.nan, depth_k=k, points=0))
    df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    win = max(1, int(round(SMOOTH_S * 5)))
    df["kmh_raw"] = df.kmh_raw.interpolate(limit=10).rolling(win, center=True, min_periods=1).median()
    return df


def scale(drv: dict, df: pd.DataFrame) -> tuple[float, str]:
    """One factor per drive from the overlay's GPS speed if there is one; else none (already metric)."""
    gps = drives.gps_track(drv)
    if gps is not None:
        v = np.interp(df.t, gps.video_s, gps.speed_kmh)
        ok = (v > 10) & (df.kmh_raw > 5)
        if ok.sum() > 50:
            return float(np.median(v[ok] / df.kmh_raw[ok])), "dashcam speed overlay"
    return 1.0, "none (metric from depth)"


def run(drv: dict) -> pd.DataFrame:
    df = raw_speed(drv)
    k, source = scale(drv, df)
    df["kmh"] = df.kmh_raw * k
    df.to_parquet(drv["folder"] / "speed.parquet", index=False)
    cal = drv["folder"] / "calibration.json"
    c = json.loads(cal.read_text()) if cal.exists() else {}
    c.update(speed_scale=round(k, 3), speed_scale_source=source)
    cal.write_text(json.dumps(c, indent=2))
    msg = f"→ {drv['folder'] / 'speed.parquet'}: {df.kmh.notna().mean():.0%} of samples measured; scale {k:.2f} ({source})"
    gps = drives.gps_track(drv)
    if gps is not None:
        t0, t1 = drv.get("street_window_s", [df.t.min(), df.t.max()])
        w = df[(df.t >= t0) & (df.t <= t1)].dropna()
        g = np.interp(w.t, gps.video_s, gps.speed_kmh)
        msg += (f"; vs GPS: correlation {np.corrcoef(w.kmh, g)[0, 1]:.2f}, median error "
                f"{np.median(np.abs(w.kmh - g)):.1f} km/h, 90 % within {np.percentile(np.abs(w.kmh - g), 90):.1f}")
    print(msg, flush=True)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    run(drives.load(ap.parse_args().drive))


if __name__ == "__main__":
    main()
