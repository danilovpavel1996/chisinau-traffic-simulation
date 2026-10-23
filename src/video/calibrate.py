"""
calibrate.py
------------
Self-calibration of the camera's focal length against the map — no camera specs,
no GPS.

The video measures rotation in pixels; degrees = pixels / focal length. Once the
route is matched (event_match.py), the map gives the true heading change through
every bend and turn. Over 12-s windows with > MIN_MAP_TURN degrees of turning on the
map, the video's rotation (computed with the nominal field of view) is regressed on
the map's: the slope is how much the nominal focal length is off.

    2026_10_23_10_06, dashcam "120°": map slope 1.58, GPS slope 1.56 → ~95° effective

Depth of the spatial view (bev.py): with a speed overlay, horizon and scale come from
parked cars (ground.py, preferred); otherwise, second step here: approaching a crosswalk straight ahead,
the distance the view shows for it must shrink exactly by the distance driven — from
the video odometer, in metres through the map. The median ratio is the depth factor
(2026_10_23 after the focal step: view 0.69 × odometer, 0.70 × GPS → factor 1.45).

Applied only when the evidence agrees with itself — enough windows, ≥ MIN_AGREE of
them turning the same way as the map, a plausible slope. A wrong route (at night)
gives inconsistent windows and the nominal value stays.

The calibrated field of view is stored in processed/<drive>/calibration.json;
drives.load() uses it in place of camera.hfov_deg, so motion, events and the spatial
view all pick it up. After calibrating, events and the matching are re-run (turn
sizes enter the matching cost).

Usage:
    .venv-video/bin/python -m src.video.calibrate <drive>
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.video import drives
from src.video.motion import ego_states

WINDOW_S = 12            # s, rotation compared over windows this long …
STEP_S = 6               # … every 6 s
MIN_MAP_TURN = 25        # deg of heading change on the map within a window
MIN_WINDOWS = 6
MIN_AGREE = 0.75         # share of windows where video and map turn the same way
MIN_DEPTH_FRAMES = 6     # crosswalk-approach frames needed for the depth factor
DEPTH_RANGE = (0.5, 2.5)
DEPTH_PASSES = 3
SLOPE_RANGE = (0.5, 2.5)
SIGN = -1                # the video's yaw is positive to the left, map headings to the right


def windows(drv: dict) -> np.ndarray:
    """(map turn, video turn) in degrees per window, video with the nominal field of view."""
    f, cam = drv["folder"], drv["camera"]
    meta = json.loads((f / "meta.json").read_text())
    e = ego_states(pd.read_parquet(f / "ego.parquet"), meta["width"], cam["hfov_nominal"], cam["hood"])
    cum_yaw = np.cumsum(e.yaw_deg_s * e.t.diff().fillna(0)) * SIGN
    o = json.loads((f / "event_match.json").read_text())
    tr, rt = o["trace"], o["route"]
    kx = 111_320 * np.cos(np.radians(np.mean(rt["lat"])))
    r_lat, r_lon, r_d = map(np.asarray, (rt["lat"], rt["lon"], rt["d"]))

    def heading(d):                                  # map heading along the route, over ±10 m
        return np.degrees(np.arctan2((np.interp(d + 10, r_d, r_lon) - np.interp(d - 10, r_d, r_lon)) * kx,
                                     (np.interp(d + 10, r_d, r_lat) - np.interp(d - 10, r_d, r_lat)) * 111_320))

    s, d = np.asarray(tr["video_s"], float), np.asarray(tr["d"], float)
    hd = np.degrees(np.unwrap(np.radians(heading(d))))
    rows = []
    for a in np.arange(s[0], s[-1] - WINDOW_S, STEP_S):
        b = a + WINDOW_S
        m_turn = np.interp(b, s, hd) - np.interp(a, s, hd)
        v_turn = np.interp(b, e.t, cum_yaw) - np.interp(a, e.t, cum_yaw)
        if abs(m_turn) > MIN_MAP_TURN:
            rows.append((m_turn, v_turn))
    return np.array(rows).reshape(-1, 2)


def estimate(drv: dict) -> dict:
    return fit(windows(drv), drv["camera"]["hfov_nominal"])


def fit(w: np.ndarray, nominal: float) -> dict:
    """Field of view from (map turn, video turn) windows, or the nominal one if they disagree."""
    res = dict(hfov_nominal=nominal, windows=len(w), hfov_deg=nominal, status="kept nominal")
    if len(w) < MIN_WINDOWS:
        res["reason"] = f"only {len(w)} turning windows (need {MIN_WINDOWS})"
        return res
    agree = float(np.mean(np.sign(w[:, 0]) == np.sign(w[:, 1])))
    slope = float(np.sum(w[:, 0] * w[:, 1]) / np.sum(w[:, 0] ** 2))
    res.update(agree=round(agree, 2), slope=round(slope, 3))
    if agree < MIN_AGREE or not SLOPE_RANGE[0] <= slope <= SLOPE_RANGE[1]:
        res["reason"] = f"inconsistent: {agree:.0%} of windows agree, slope {slope:.2f}"
        return res
    # video angle ∝ 1 / focal → true focal = slope × nominal focal
    hfov = 2 * np.degrees(np.arctan(np.tan(np.radians(nominal / 2)) / slope))
    res.update(hfov_deg=round(float(hfov), 1), status="calibrated")
    return res


def run(drv: dict, rematch: bool = True) -> dict:
    """Always estimated from a match made with the nominal lens, so results don't depend on
    an earlier calibration; then events and the matching are re-run with the calibrated one."""
    from src.video import event_match, events
    (drv["folder"] / "calibration.json").unlink(missing_ok=True)
    drv = drives.load(drv["key"])
    if rematch:
        events.run(drv)
        event_match.run(drv)
    res = estimate(drv)
    (drv["folder"] / "calibration.json").write_text(json.dumps(res, indent=2))
    detail = res.get("reason") or f"{res['windows']} windows, slope {res['slope']}, {res['agree']:.0%} agree"
    print(f"→ {drv['folder'] / 'calibration.json'}: {res['status']}, field of view {res['hfov_nominal']}° → "
          f"{res['hfov_deg']}° ({detail})", flush=True)
    if rematch and res["status"] == "calibrated" and abs(res["hfov_deg"] - res["hfov_nominal"]) > 3:
        drv = drives.load(drv["key"])                # picks up the calibrated field of view
        events.run(drv)
        event_match.run(drv)
    if (drv["folder"] / "scene_dense.parquet").exists():
        res.update(depth(drives.load(drv["key"])))
        (drv["folder"] / "calibration.json").write_text(json.dumps(res, indent=2))
        if drives.gps_track(drv) is not None:          # speed overlay: horizon + scale from parked cars
            from src.video import ground
            res.update(ground.run(drives.load(drv["key"])))
        if (drv["folder"] / "depth.parquet").exists():   # depth in metres, fitted to the road geometry
            from src.video import depth
            res.update(depth.fit_to_ground(drives.load(drv["key"])))
        print(f"   depth: {res['depth_status']}" + (f", factor {res['depth_scale']}" if "depth_scale" in res else "")
              + f" ({res['depth_frames']} crosswalk-approach frames)", flush=True)
    return res


def depth(drv: dict) -> dict:
    """Depth factor of the spatial view from crosswalk approaches (see module doc)."""
    from src.video.bev import Geometry, crosswalk_depths
    (drv["folder"] / "calibration.json").write_text(json.dumps(
        {k: v for k, v in json.loads((drv["folder"] / "calibration.json").read_text()).items() if k != "depth_scale"}))
    dense = pd.read_parquet(drv["folder"] / "scene_dense.parquet").iloc[:1]
    geo = Geometry(drv, dense.iloc[0])
    out, factor = dict(depth_frames=0, depth_status="not calibrated"), 1.0
    for _ in range(DEPTH_PASSES):        # which approaches qualify depends on the scale → re-estimate
        geo.fz = geo.f * factor
        geo.z_min = geo.fz * geo.h / (geo.bonnet - geo.y_h)
        pairs = crosswalk_depths(drv, geo)
        if len(pairs) < MIN_DEPTH_FRAMES:
            break
        factor *= float(np.median(pairs[:, 1] / pairs[:, 0]))
        out = dict(depth_frames=len(pairs), depth_status="calibrated", depth_scale=round(factor, 3))
    if not DEPTH_RANGE[0] <= factor <= DEPTH_RANGE[1]:
        out = dict(depth_frames=out["depth_frames"], depth_status=f"implausible factor {factor:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    ap.add_argument("--no-rematch", action="store_true", help="estimate from the existing match, re-run nothing")
    a = ap.parse_args()
    run(drives.load(a.drive), rematch=not a.no_rematch)


if __name__ == "__main__":
    main()
