"""
speed.py
--------
Our speed from the video, in km/h: optical flow × metric depth.

Driving forward at v, a scene point at distance Z, r pixels from the vanishing point,
moves away from it at  v · r / Z  pixels per second. So each tracked point gives

    v = radial motion × fps × Z / r

and the median over the points (vehicles excluded) is the car's speed — metric, and the
same in a narrow street (buildings 10 m away stream past fast) and in the open (fields
200 m away barely move), where plain image motion confuses the two. Points come from
dense.py (FLOW_PAIRS adjacent frame pairs after each analysed frame, 5 per second), depth
from Depth Anything fitted to the road plane (depth.metric).

One scale factor remains (2026_10_23_10_06: the raw estimate read 0.79× GPS): calibrated
per drive from the dashcam's speed overlay when the video has one (GPS speed only, never
positions), else so that the distance driven equals the matched route's length.

    2026_10_23_10_06, 1 sample/s, 1 pair: correlation with GPS 0.79 (scene flow: 0.47),
    median error 9.2 km/h (13.7)

Usage:
    .venv-video/bin/python -m src.video.speed <drive>

Output: data/video/processed/<drive>/speed.parquet (t, kmh, kmh_raw, points); speed_* keys in calibration.json
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

MIN_R_PX = 60            # points close to the vanishing point move too little to measure
Z_RANGE_M = (2.0, 60.0)
MIN_POINTS = 30
SMOOTH_S = 1.0           # median over this window


def raw_speed(drv: dict) -> pd.DataFrame:
    folder = drv["folder"]
    meta = json.loads((folder / "meta.json").read_text())
    fps = meta["fps"]
    geo = Geometry(drv, pd.read_parquet(folder / "scene_dense.parquet").iloc[0])
    motion = pd.read_parquet(folder / "motion.parquet")
    td, maps = depth_mod.load(drv)
    centre = np.array([geo.cx, geo.y_h])
    rows = []
    for t, g in motion.groupby("t"):
        j = int(np.argmin(np.abs(td - t)))
        if abs(td[j] - t) > 0.05:
            continue
        D = depth_mod.metric(drv, maps[j])
        vs = []
        for r in g.itertuples():
            a = np.frombuffer(zlib.decompress(r.pts), np.float16).reshape(-1, 4).astype(np.float32)
            p, d = a[:, :2], a[:, 2:]
            rv = p - centre
            rr = np.linalg.norm(rv, axis=1)
            radial = (d * rv).sum(1) / np.maximum(rr, 1e-6)
            Z = D[np.clip((p[:, 1] / depth_mod.DOWN).astype(int), 0, D.shape[0] - 1),
                  np.clip((p[:, 0] / depth_mod.DOWN).astype(int), 0, D.shape[1] - 1)]
            ok = (rr > MIN_R_PX) & (Z > Z_RANGE_M[0]) & (Z < Z_RANGE_M[1]) & (p[:, 1] < geo.bonnet - 5)
            vs.append(radial[ok] * fps * Z[ok] / rr[ok])
        v = np.concatenate(vs) if vs else np.array([])
        rows.append(dict(t=float(t), kmh_raw=float(np.median(v)) * 3.6 if len(v) >= MIN_POINTS else np.nan, points=len(v)))
    df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    win = max(1, int(round(SMOOTH_S * 5)))
    df["kmh_raw"] = df.kmh_raw.clip(lower=0).interpolate(limit=10).rolling(win, center=True, min_periods=1).median()
    return df


def scale(drv: dict, df: pd.DataFrame) -> tuple[float, str]:
    """One factor per drive: from the overlay's GPS speed if there is one, else the matched route's length."""
    gps = drives.gps_track(drv)
    if gps is not None:
        v = np.interp(df.t, gps.video_s, gps.speed_kmh)
        ok = (v > 10) & (df.kmh_raw > 5)
        if ok.sum() > 50:
            return float(np.median(v[ok] / df.kmh_raw[ok])), "dashcam speed overlay"
    em = drv["folder"] / "event_match.json"
    if em.exists():
        L = json.loads(em.read_text())["route"]["d"][-1]
        t0, t1 = drv.get("street_window_s", [df.t.min(), df.t.max()])
        w = df[(df.t >= t0) & (df.t <= t1)]
        driven = float((w.kmh_raw.fillna(0) / 3.6 * w.t.diff().fillna(0)).sum())
        if driven > 0:
            return L / driven, "matched route length"
    return 1.0, "none"


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
