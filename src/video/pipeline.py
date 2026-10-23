"""
pipeline.py
-----------
Run every step for one drive, in order:

    perception   video → detections.parquet, ego.parquet          (slow: ~2× video length on CPU)
    overlay      → gps.parquet: dashcam's burnt-in time/speed/GPS (ground truth; skipped if none)
    traffic      → tracks.parquet, stops/turns/per_minute.csv, timeline.png
    scene        → scene.parquet + scene_dense.parquet: crosswalks, lane markings, curbs… (Google Colab GPU)
    localize     → match.json (video-only route on the map, accuracy vs truth / GPS)
    road         → road.parquet: road width + side-street openings from the class maps
    events       → events.parquet: junction events seen in the video (turns, roundabouts, crosswalks, signals)
    event_match  → event_match.json: video-only route from start + destination (event-based)
    calibrate    → calibration.json: field of view + depth scale measured against the map; re-match
    bev          → bev.mp4: spatial view — what the camera sees vs what the map expects
    strip        → strip.png: the whole drive stitched into one straightened top-down strip
    position     → trace.parquet: where the car is every second (GPS if available, else the match)
    lanes        → lanes.parquet, lanes_by_street.csv
    signs        → text.parquet: street names and named places read in the video (no number plates)
    streets      → stops_detail.csv, streets.csv + data/video/aggregate/ (all drives)
    dashboard    → data/video/dashboard.html (all drives) + this drive's dashboard_data.js

Usage:
    .venv-video/bin/python -m src.video.pipeline <drive>                    # all steps
    .venv-video/bin/python -m src.video.pipeline <drive> --from localize    # skip earlier steps

Add the drive to configs/drives.yaml first (video path, camera, street window).
"""

from __future__ import annotations

import argparse
import time

from src.video import drives

STEPS = ["overlay", "perception", "traffic", "scene", "road", "events", "event_match", "calibrate", "bev", "strip", "position", "lanes", "signs",
         "streets", "dashboard"]


def run_step(step: str, drv: dict):
    if step == "overlay":
        from src.video import overlay
        overlay.run(drv)
    elif step == "perception":
        from src.video import perception
        perception.run(drv)
    elif step == "traffic":
        from src.video import traffic
        traffic.run(drv)
    elif step == "scene":                     # > 5 min on the laptop → free Colab GPU
        from src.video import scene
        scene.run_colab(drv)
        scene.run_colab(drv, dense=True)
    elif step == "calibrate":
        from src.video import calibrate
        calibrate.run(drv)
    elif step == "bev":
        from src.video import bev
        bev.run(drv)
    elif step == "strip":
        from src.video import strip
        strip.run(drv)
    elif step == "road":
        from src.video import road_geometry
        road_geometry.run(drv)
    elif step == "events":
        from src.video import events
        events.run(drv)
    elif step == "event_match":
        from src.video import event_match
        event_match.run(drv)
    elif step == "localize":
        from src.video import localize
        localize.run(drv)
    elif step == "position":
        from src.video import position
        position.run(drv)
    elif step == "lanes":
        from src.video import lanes
        lanes.run(drv)
    elif step == "signs":
        from src.video import signs
        signs.run(drv)
    elif step == "streets":
        from src.video import street_stats
        street_stats.run(drv)
        street_stats.aggregate()
    elif step == "dashboard":
        from src.video.dashboard import build
        build.run(drv)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    ap.add_argument("--from", dest="start", choices=STEPS, default=STEPS[0], help="first step to run")
    a = ap.parse_args()
    drv = drives.load(a.drive)
    for step in STEPS[STEPS.index(a.start):]:
        t0 = time.time()
        print(f"\n━━ {step} ━━")
        run_step(step, drv)
        print(f"   {step} done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
