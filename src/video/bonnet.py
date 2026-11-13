"""
bonnet.py
---------
Where our own car is in the picture — measured per video, then kept out of everything.

The pipeline assumed the bottom 12 % of the frame (configs `camera.hood`). On this dashcam the
bonnet starts at ~0.79 of the height in the middle (lower towards the sides): the strip between
shows the bonnet mirroring road and sky, and was taken as road (ground projection, motion
points) and, now and then, as a car right in front of us (0.3–2 % of vehicle detections, one
box over the bonnet, in 73–408 frames per drive).

    measure   the bonnet's edge is a horizontal edge that stays put while the scene moves:
              averaged over the video's keyframes (hundreds, the whole drive) only persistent
              edges remain; per column the topmost one between EDGE_RANGE (below the dashboard's
              reflection in the windscreen) is the bonnet's edge, smoothed across columns.
              calibration.json: bonnet_curve (fraction of the height at BONNET_COLS columns),
              bonnet_frac (its highest point − margin) → drives.load sets camera.bonnet
    clean     vehicle boxes lying mostly on the bonnet are dropped from detections.parquet
              (the original kept as detections_raw.parquet), and wide boxes reaching deep into it
              (bonnet plus road, boxed as one "car")

Usage:
    .venv-video/bin/python -m src.video.bonnet <drive>
"""

from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np
import pandas as pd

from src.video import drives

SMALL_W, SMALL_H = 480, 270
EDGE_RANGE = (0.72, 0.95)
EDGE_MIN = 18.0                 # mean |vertical gradient| of a persistent edge (grey levels per pixel)
BONNET_COLS = 49
MARGIN = 0.01                   # of the height, above the edge
DISAGREE = 0.04                 # a curve this far from the other drives' (which agree) is not trusted
ON_BONNET = 0.5                 # a box with more than this share of its height below the edge is the bonnet
DEEP, WIDE = 0.08, 0.40         # … or a box this far below the edge and this wide (of height / width)
VEHICLES = ("car", "bus", "truck", "motorcycle")


def measure(drv: dict) -> dict:
    raw = subprocess.run(["nice", "-n", "10", "ffmpeg", "-v", "error", "-skip_frame", "nokey", "-i", str(drv["video"]),
                          "-vf", f"scale={SMALL_W}:{SMALL_H},format=gray", "-vsync", "vfr", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, SMALL_H, SMALL_W).astype(np.float32)
    edge = np.abs(np.diff(frames, axis=1).mean(0))                     # persistent horizontal edges
    lo, hi = int(EDGE_RANGE[0] * SMALL_H), int(EDGE_RANGE[1] * SMALL_H)
    cols = np.linspace(4, SMALL_W - 5, BONNET_COLS).astype(int)
    top = []
    for x in cols:
        col = edge[lo:hi, max(x - 4, 0):x + 5].mean(1)
        strong = np.flatnonzero(col >= max(EDGE_MIN, 0.35 * col.max()))
        top.append((lo + strong[0]) / SMALL_H if len(strong) else np.nan)
    top = pd.Series(top).interpolate(limit_direction="both").to_numpy()
    curve, kept = smooth_curve(top)
    source = "measured"
    others = [np.array(c) for c in other_curves(drv)]
    if len(others) >= 2:
        ref = np.median(others, axis=0)
        agree = all(np.abs(o - ref).max() <= DISAGREE for o in others)
        if agree and np.abs(curve - ref).max() > DISAGREE:          # e.g. at night the edge is not visible and
            curve, source = ref, "borrowed: same dashcam, other drives"   # the bonnet's lower edge was taken
    curve = np.round(curve, 4)
    res = dict(bonnet_curve=curve.tolist(), bonnet_frac=round(float(np.min(curve)) - MARGIN, 4),
               bonnet_keyframes=len(frames), bonnet_columns_fitted=round(kept, 2), bonnet_source=source)
    cal = drv["folder"] / "calibration.json"
    c = json.loads(cal.read_text()) if cal.exists() else {}
    c.update(res)
    cal.write_text(json.dumps(c, indent=2))
    print(f"→ {cal}: bonnet edge at {np.min(curve):.3f}–{np.max(curve):.3f} of the height (middle "
          f"{curve[len(curve) // 2]:.3f}; assumed so far {1 - drv['camera']['hood']:.2f}), {len(frames)} keyframes, {source}", flush=True)
    return res


def smooth_curve(top: np.ndarray) -> tuple[np.ndarray, float]:
    """A bonnet's edge is a smooth curve: robust quadratic over the columns (columns where a lower
    edge was taken drop out)."""
    x = np.linspace(0, 1, len(top))
    keep = np.ones(len(top), bool)
    for _ in range(6):
        c = np.polyfit(x[keep], top[keep], 2)
        r = np.abs(np.polyval(c, x) - top)
        keep = r <= max(0.012, np.percentile(r[keep], 70))
    return np.polyval(c, x), float(keep.mean())


def other_curves(drv: dict) -> list:
    out = []
    for cal in (drives.ROOT / "data" / "video" / "processed").glob("*/calibration.json"):
        if cal.parent.name == drv["key"]:
            continue
        c = json.loads(cal.read_text())
        if c.get("bonnet_source") == "measured":
            out.append(c["bonnet_curve"])
    return out


def curve_at(curve: list, x_frac) -> np.ndarray:
    return np.interp(x_frac, np.linspace(0, 1, len(curve)), curve)


def clean(drv: dict, curve: list):
    folder = drv["folder"]
    meta = json.loads((folder / "meta.json").read_text())
    W, H = meta["width"], meta["height"]
    raw = folder / "detections_raw.parquet"
    det = pd.read_parquet(raw if raw.exists() else folder / "detections.parquet")
    if not raw.exists():
        det.to_parquet(raw, index=False)
    edge = curve_at(curve, ((det.x1 + det.x2) / 2 / W).to_numpy()) * H
    below = (det.y2 - np.maximum(det.y1, edge)).clip(lower=0) / (det.y2 - det.y1).clip(lower=1)
    # a real car ahead is hidden by the bonnet: its box ends near the bonnet's edge. A wide box that
    # reaches far down into the bonnet is the bonnet (plus road), not a car
    deep = (det.y2 > edge + DEEP * H) & ((det.x2 - det.x1) > WIDE * W)
    drop = det.cls.isin(VEHICLES) & ((below > ON_BONNET) | deep)
    det[~drop].to_parquet(folder / "detections.parquet", index=False)
    v = det.cls.isin(VEHICLES)
    print(f"→ {folder / 'detections.parquet'}: {drop.sum()} bonnet boxes dropped ({drop.sum() / max(v.sum(), 1):.1%} of "
          f"vehicle detections, {det[drop].frame.nunique()} frames); original in detections_raw.parquet", flush=True)


def run(drv: dict):
    res = measure(drv)
    clean(drv, res["bonnet_curve"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    run(drives.load(ap.parse_args().drive))


if __name__ == "__main__":
    main()
