"""
ground.py
---------
The road-plane geometry measured from parked cars and the dashcam's speed: the
horizon row and the distance scale.

On a flat road a point Z metres ahead appears at image row  y = y_h + S / Z
(S = focal length × camera height, in inference pixels × metres). A parked car does
not move, so while we drive, the bottom of its box (where it touches the road) obeys

    1 / (y(t) − y_h)  =  (Z₀ − s(t)) / S          s(t) = distance driven

— a straight line in s with slope −1/S. With the distance driven from the overlay
speed, each parked car followed over a few seconds gives S; the horizon is the row
at which the most cars fall on straight lines, and S is their median.

The speed comes from the dashcam overlay (GPS). It is used only for this scale, never
for the route: the video-only route and position stay free of GPS positions. Videos
without an overlay keep the map-based calibration (calibrate.py).

Tried first and dropped (2026_10_23): tracking the road surface itself — dense
optical flow, flow after removing the static windscreen reflection, and matching
row-brightness profiles in inverse distance. The compressed asphalt has almost no
texture (1–3 grey levels change between frames at 60 km/h) and the reflection of the
dashboard dominates: the measured motion did not grow with the square of the row as
it must. Car boxes come from the detector and are not affected.

Usage:
    .venv-video/bin/python -m src.video.ground <drive>

Output: horizon_px, ground_scale (+ fit quality) added to calibration.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.video import drives
from src.video.motion import ego_states
from src.video.scene import WIDTH

MIN_KMH = 15             # driving, not creeping
MAX_YAW = 3.0            # deg/s: straight — a turn swings boxes sideways and up / down
MIN_POINTS = 5           # detections of one parked car …
MIN_DRIVEN_M = 6         # … over at least this much driving
EDGE = 0.01              # box touching the image border: its bottom is cut off
MIN_R2 = 0.95            # a car "on a straight line"
SEARCH_PX = 50           # horizon search range around the first guess
CAMERA_HEIGHT_M = (0.9, 2.0)    # scale / GeoCalib focal length must be a dashcam's height above the road
MAX_IQR_RATIO = 2.0      # cars must roughly agree (75th / 25th percentile of their scales)
NEAR_PX = 25             # use box bottoms at least this far below the first-guess horizon (≲ 30 m):
                         # far cars barely move in the image while their box edges jitter ± 2 px
SPLIT_PX = 4             # a parked car only comes closer: its bottom edge jumping up by more than
                         # this means the tracker gave its ID to the next car in the row
SPLIT_M = 6              # … and so does a gap of this much driving without a detection (5 per s ≈ 3 m)


def parked_cars(drv: dict, y_guess: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per parked car: (bottom row in inference px, metres driven) while driving straight."""
    folder, cam = drv["folder"], drv["camera"]
    meta = json.loads((folder / "meta.json").read_text())
    W, H = meta["width"], meta["height"]
    k = WIDTH / W
    gps = drives.gps_track(drv)
    if gps is None:
        return []
    g = gps.sort_values("video_s")
    driven = np.cumsum(np.r_[0, np.diff(g.video_s)] * g.speed_kmh.to_numpy() / 3.6)
    e = ego_states(pd.read_parquet(folder / "ego.parquet"), W, cam["hfov_deg"], cam["hood"])
    tracks = pd.read_parquet(folder / "tracks.parquet")
    ids = tracks[(tracks.label == "parked") & (tracks.cls == "car")].track_id
    det = pd.read_parquet(folder / "detections.parquet")
    det = det[det.track_id.isin(ids) & (det.x1 > EDGE * W) & (det.x2 < (1 - EDGE) * W)
              & (det.y2 < H * (1 - cam["hood"]) - 4)]
    det = det.assign(kmh=np.interp(det.t, g.video_s, g.speed_kmh),
                     yaw=np.interp(det.t, e.t, e.yaw_smooth.abs()),
                     s=np.interp(det.t, g.video_s, driven))
    det = det[(det.kmh >= MIN_KMH) & (det.yaw < MAX_YAW)]
    det = det[det.y2 * k > y_guess + NEAR_PX]
    out = []
    for _, c in det.sort_values("t").groupby("track_id"):
        y, s = c.y2.to_numpy() * k, c.s.to_numpy()
        cut = np.flatnonzero((np.diff(y) < -SPLIT_PX) | (np.diff(s) > SPLIT_M)) + 1
        for piece in np.split(np.arange(len(y)), cut):
            if len(piece) >= MIN_POINTS and np.ptp(s[piece]) >= MIN_DRIVEN_M:
                out.append((y[piece], s[piece]))
    return out


def fit(cars, y_guess: float) -> dict:
    """Horizon at which most parked cars move as a flat road predicts; the scale from their slopes."""
    if len(cars) < 10:
        return dict(ground_status=f"only {len(cars)} usable parked cars")
    best = None
    for yh in np.arange(y_guess - SEARCH_PX, y_guess + SEARCH_PX + 0.5, 1.0):
        scales, r2s = [], []
        for y, s in cars:
            if (y - yh).min() < 4:
                continue
            u = 1 / (y - yh)
            slope, icpt = np.polyfit(s, u, 1)
            r2 = 1 - np.var(u - (slope * s + icpt)) / max(np.var(u), 1e-12)
            if slope < 0:
                scales.append(-1 / slope); r2s.append(r2)
        r2s, scales = np.array(r2s), np.array(scales)
        good = r2s >= MIN_R2
        if good.sum() < 5:
            continue
        # all cars share one camera: at the true horizon they agree on the scale
        spread = float(np.std(np.log(scales[good])))
        score = (good.sum(), -spread)
        if best is None or score > best[0]:
            best = (score, float(yh), float(np.median(scales[good])),
                    float(np.percentile(scales[good], 25)), float(np.percentile(scales[good], 75)), float(np.median(r2s)))
    if best is None:
        return dict(ground_status="parked cars do not move like a flat road")
    (n, _), yh, s, q1, q3, r2 = best
    return dict(ground_status="calibrated", horizon_px=round(yh, 1), ground_scale=round(s, 1),
                ground_scale_iqr=[round(q1, 1), round(q3, 1)], ground_cars=int(n), ground_cars_total=len(cars),
                ground_median_r2=round(r2, 3))


def plausible(res: dict, lens_focal: float | None) -> dict:
    """
    Reject a result no real dashcam could give. At night (2026_10_23_19_26) parked-car boxes
    are loose (headlights, reflections): scale 4,839 with an interquartile range 3,561–7,178,
    i.e. a camera 8 m above the road given GeoCalib's focal length (599 px).
    """
    if res.get("ground_status") != "calibrated":
        return res
    q1, q3 = res["ground_scale_iqr"]
    why = []
    if q3 / max(q1, 1e-9) > MAX_IQR_RATIO:
        why.append(f"cars disagree (scales {q1:.0f}–{q3:.0f})")
    if lens_focal:
        h = res["ground_scale"] / lens_focal
        res["ground_camera_height_m"] = round(h, 2)
        if not CAMERA_HEIGHT_M[0] <= h <= CAMERA_HEIGHT_M[1]:
            why.append(f"implies a camera {h:.1f} m above the road")
    if why:
        res["ground_status"] = "rejected: " + "; ".join(why)
    return res


def run(drv: dict) -> dict:
    from src.video.bev import horizon_from_road
    f = (WIDTH / 2) / np.tan(np.radians(drv["camera"]["hfov_deg"] / 2))
    y_guess = horizon_from_road(drv, f)
    cal = drv["folder"] / "calibration.json"
    lens_focal = (json.loads(cal.read_text()) if cal.exists() else {}).get("lens_focal_px")
    res = plausible(fit(parked_cars(drv, y_guess), y_guess), lens_focal) if drives.gps_track(drv) is not None \
        else dict(ground_status="no speed overlay")
    c = {k: v for k, v in (json.loads(cal.read_text()) if cal.exists() else {}).items() if not k.startswith("ground")}
    c.update(res)
    cal.write_text(json.dumps(c, indent=2))
    print(f"→ {cal}: {res}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    run(drives.load(ap.parse_args().drive))


if __name__ == "__main__":
    main()
