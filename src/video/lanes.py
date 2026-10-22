"""
lanes.py
--------
Lanes from the road scanlines saved by scene.py (no extra model run).

Per analysed frame, on the scanlines in ROWS (the lowest row is mostly bonnet
on a phone), keeping the row that sees most lanes:
    drivable span around the car   road, lane markings, crosswalks and vehicles
                                   (vehicles occupy lanes), small gaps bridged
    lane boundaries                span edges + lane-marking centres inside it
    lanes                          intervals at least MIN_LANE × our lane wide
                                   (narrower = shoulder / bike lane / noise)
    lane_from_right                our interval (it contains the image centre =
                                   the car) counted from the right edge, 1 = rightmost
The camera is assumed to be centred in the car.

Dashed lines are invisible part of the time → counts are the max over WINDOW
analysed frames. Per street (with trace.parquet): median lanes seen, our lane,
vs OSM `lanes` (which is the total for BOTH directions on two-way streets).

Usage:
    .venv-video/bin/python -m src.video.lanes <drive>

Output: data/video/processed/<drive>/lanes.parquet (per frame) and
        lanes_by_street.csv
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.video import drives

ROWS = ("0.75", "0.85")    # 0.95 is mostly bonnet on a phone
WINDOW = 5                 # analysed frames (~10 s): dashed lines show up within a few dashes
DRIVABLE = {"road", "lane_marking", "crosswalk", "car", "bus", "truck", "bike_lane"}
GAP_PX = 8             # bridge tiny gaps (segmentation noise) inside the drivable span
MIN_LANE = 0.4         # an interval narrower than this × our lane is not a lane


def frame_lanes(runs: list, width: int) -> dict | None:
    cx = width / 2
    spans = []                                         # merged drivable spans
    for c, a, b in runs:
        if c in DRIVABLE:
            if spans and a - spans[-1][1] <= GAP_PX:
                spans[-1][1] = b
            else:
                spans.append([a, b])
    ours = next(((a, b) for a, b in spans if a <= cx <= b), None)
    if ours is None or ours[1] - ours[0] < 0.15 * width:
        return None
    L, R = ours
    marks = [(a + b) / 2 for c, a, b in runs if c == "lane_marking" and L < (a + b) / 2 < R]
    bounds = sorted({L, R, *marks})
    ivals = list(zip(bounds, bounds[1:]))
    mine = next(i for i, (a, b) in enumerate(ivals) if a <= cx <= b)
    my_w = ivals[mine][1] - ivals[mine][0]
    lanes = [(a, b) for a, b in ivals if b - a >= MIN_LANE * my_w]
    right_of_me = sum(1 for a, b in lanes if a >= ivals[mine][1])
    return dict(lanes_visible=len(lanes), lane_from_right=right_of_me + 1, markings=len(marks),
                span_px=R - L, my_lane_px=my_w, occluded=any(c in ("car", "bus", "truck") for c, a, b in runs
                                                             if a <= cx <= b))


def osm_lanes_per_direction(lanes: str, highway: str, oneway_share: float | None = None) -> float:
    """OSM 'lanes' is the total for both directions on two-way streets."""
    try:
        vals = [int(x) for x in str(lanes).replace("[", "").replace("]", "").replace("'", "").split(",") if x.strip()]
    except ValueError:
        return np.nan
    if not vals:
        return np.nan
    return float(np.median(vals))


def run(drv: dict):
    folder = drv["folder"]
    sc = pd.read_parquet(folder / "scene.parquet").sort_values("t")
    rows = []
    for r in sc.itertuples():
        lines = json.loads(r.scanlines)
        found = [x for x in (frame_lanes(lines.get(y, []), int(r.width)) for y in ROWS) if x]
        best = max(found, key=lambda x: x["lanes_visible"]) if found else {}
        rows.append(dict(t=r.t, **best))
    df = pd.DataFrame(rows)
    # dashed markings are visible only part of the time → max over a short window
    df["lanes"] = df.lanes_visible.rolling(WINDOW, center=True, min_periods=1).max()
    df["lane_from_right_w"] = df.lane_from_right.rolling(WINDOW, center=True, min_periods=1).max()
    tr = pd.read_parquet(folder / "trace.parquet").set_index("video_s")
    s = df.t.astype(int).clip(tr.index.min(), tr.index.max())
    df["street"] = tr.street.reindex(s).to_numpy()
    df["osm_lanes"] = tr.lanes.reindex(s).to_numpy()
    df["moving"] = tr.moving.reindex(s).to_numpy()
    df["in_window"] = tr.in_window.reindex(s).to_numpy()
    df.to_parquet(folder / "lanes.parquet", index=False)

    ok = df[df.lanes_visible.notna() & df.in_window.astype(bool)]
    by = (ok.groupby("street")
            .agg(frames=("t", "size"),
                 lanes_seen=("lanes", lambda x: int(np.median(x))),
                 lanes_seen_max=("lanes", "max"),
                 our_lane_from_right=("lane_from_right_w", lambda x: int(np.median(x))),
                 osm_lanes=("osm_lanes", lambda x: x.mode().iat[0] if len(x.mode()) else ""))
            .sort_values("frames", ascending=False))
    by = by[by.frames >= 5]
    by.to_csv(folder / "lanes_by_street.csv")
    print(f"lanes found in {df.lanes_visible.notna().mean():.0%} of {len(df)} frames")
    print(by.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    run(drives.load(ap.parse_args().drive))


if __name__ == "__main__":
    main()
