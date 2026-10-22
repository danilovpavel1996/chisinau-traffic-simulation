"""
pipeline.py
-----------
Run every step for one drive, in order:

    perception   video → detections.parquet, ego.parquet          (slow: ~2× video length on CPU)
    traffic      → tracks.parquet, stops/turns/per_minute.csv, timeline.png
    localize     → match.json (route on the map, accuracy if the drive has a truth)
    viewer       → viewer.html (+ 720p copy of the video)

Usage:
    .venv-video/bin/python -m src.video.pipeline <drive>                    # all steps
    .venv-video/bin/python -m src.video.pipeline <drive> --from localize    # skip earlier steps

Add the drive to configs/drives.yaml first (video path, camera, street window).
"""

from __future__ import annotations

import argparse
import time

from src.video import drives

STEPS = ["perception", "traffic", "localize", "viewer"]


def run_step(step: str, drv: dict):
    if step == "perception":
        from src.video import perception
        perception.run(drv)
    elif step == "traffic":
        from src.video import traffic
        traffic.run(drv)
    elif step == "localize":
        from src.video import localize
        localize.run(drv)
    elif step == "viewer":
        from src.video.viewer import build
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
