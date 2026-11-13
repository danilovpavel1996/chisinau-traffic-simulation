"""
road_geometry.py
----------------
Camera-independent road measurements from the per-frame class maps saved by
scene.py (no model run, no camera specs).

Perspective: on a straight, flat road the distance from the car's centre line to
the road edge, measured in an image row, grows in proportion to how far below the
horizon that row is. So

    half-width ratio  =  (road edge − image centre) / (row − horizon)

is about constant along a street — the road's half-width in units of the camera
height, the same for any dashcam. Per analysed frame:

    left_w, right_w   median half-width ratio over a band of rows ahead (BAND)
    width             left_w + right_w   → wide boulevard vs narrow street
    open_left/right   the ratio on that side jumps well above its recent level
                      → a side street opens there (junction without crosswalk)

Usage:
    .venv-video/bin/python -m src.video.road_geometry <drive>

Output: data/video/processed/<drive>/road.parquet
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.video import drives
from src.video.motion import ego_states
from src.video.scene import CODES, MASK_STEP, WIDTH

DRIVABLE = {"road", "lane_marking", "crosswalk", "car", "bus", "truck", "bike_lane"}
BAND = (0.06, 0.35)     # rows used: this fraction of (bottom − horizon) below the horizon
OPEN_JUMP = 1.6         # side ratio this × its running median = an opening
RUN_FRAMES = 11         # running-median window (analysed frames, ~20 s)


def horizon_row(drv: dict, mask_rows: int) -> float:
    """Horizon (focus-of-expansion row from motion) in class-map rows."""
    folder, cam = drv["folder"], drv["camera"]
    meta = json.loads((folder / "meta.json").read_text())
    e = ego_states(pd.read_parquet(folder / "ego.parquet"), meta["width"], cam["hfov_deg"], cam["hood"])
    ok = e.moving & ~e.turning & e.foe_y.between(0, meta["height"])
    y_full = float(np.median(e.foe_y[ok])) if ok.sum() > 50 else meta["height"] * 0.4
    scale = WIDTH / meta["width"] / MASK_STEP                     # full frame px → class-map rows
    return min(y_full * scale, mask_rows * 0.6)


def frame_widths(mask: np.ndarray, hz: float) -> tuple[float, float]:
    rows, cols = mask.shape
    drivable = np.isin(mask, [CODES.index(c) for c in DRIVABLE if c in CODES])
    cx = cols // 2
    r0, r1 = int(hz + BAND[0] * (rows - hz)) + 1, int(hz + BAND[1] * (rows - hz)) + 1
    left, right = [], []
    for r in range(max(r0, 0), min(r1, rows)):
        line = drivable[r]
        if not line[cx]:
            continue
        rr = cx
        while rr + 1 < cols and line[rr + 1]:
            rr += 1
        ll = cx
        while ll - 1 >= 0 and line[ll - 1]:
            ll -= 1
        dy = r - hz
        right.append((rr - cx) / dy); left.append((cx - ll) / dy)
    if len(right) < 3:
        return np.nan, np.nan
    return float(np.median(left)), float(np.median(right))


def run(drv: dict) -> pd.DataFrame:
    sc = pd.read_parquet(drv["folder"] / "scene.parquet").sort_values("t").reset_index(drop=True)
    if "mask" not in sc:
        raise RuntimeError("scene.parquet has no class maps — re-run scene.py (or --colab)")
    shape = json.loads(sc.mask_shape.iat[0])
    hz = horizon_row(drv, shape[0])
    rows = []
    for r in sc.itertuples():
        m = np.frombuffer(r.mask, np.uint8).reshape(json.loads(r.mask_shape))
        lw, rw = frame_widths(m, hz)
        rows.append(dict(t=r.t, left_w=lw, right_w=rw))
    df = pd.DataFrame(rows)
    df["width"] = df.left_w + df.right_w
    for side in ("left", "right"):
        base = df[f"{side}_w"].rolling(RUN_FRAMES, center=True, min_periods=3).median()
        df[f"open_{side}"] = df[f"{side}_w"] > OPEN_JUMP * base
    t0, t1 = drv.get("street_window_s", [df.t.min(), df.t.max()])
    df["in_window"] = df.t.between(t0, t1)
    df.to_parquet(drv["folder"] / "road.parquet", index=False)
    w = df[df.in_window]
    print(f"→ {drv['folder'] / 'road.parquet'}: horizon at class-map row {hz:.1f}/{shape[0]}; width measured in "
          f"{w.width.notna().mean():.0%} of frames; openings left {w.open_left.sum()}, right {w.open_right.sum()}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    run(drives.load(ap.parse_args().drive))


if __name__ == "__main__":
    main()
