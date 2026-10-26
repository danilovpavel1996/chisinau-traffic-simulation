"""
track_points.py
---------------
Scene points tracked between every pair of adjacent frames (30 per second), for speed.py.
Runs on a Colab CPU runtime (whole drive, ~10 min on the laptop) or the laptop (tests).

    where    corners spread over a grid of image cells (GRID), so the road and both sides
             always get points — the strongest corners alone all sat in the trees in a tree
             tunnel and none on the road, where distance is known best (2026_10_24_15_48)
    check    tracked forward and back (Lucas–Kanade); a point that does not come back to
             within FB_MAX_PX is dropped — repeating guard-rail posts lock onto the next post
    skip     vehicles (detections of the nearest analysed frame)

Every pair is filed under the nearest analysed frame (dense.py, every FRAME_STEP-th frame),
whose depth map speed.py uses: pair = frame offset (−2 … 3).

Usage:
    .venv-video/bin/python -m src.video.track_points <drive> --colab
    .venv-video/bin/python -m src.video.track_points <drive> --start S --seconds N    # laptop test

Output: data/video/processed/<drive>/motion.parquet (t, pair, pts zlib float16 [x, y, dx, dy], n)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import time
import zlib

import cv2
import numpy as np
import pandas as pd

from src.video import drives
from src.video.dense import FRAME_STEP, input_height, input_video
from src.video.scene import WIDTH

GRID = (4, 6)                # rows × columns of image cells
PER_CELL = 30
FB_MAX_PX = 0.5
LK = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
VEHICLES = {"car", "bus", "truck", "motorcycle"}


def corners(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    H, W = gray.shape
    out = []
    for i in range(GRID[0]):
        for j in range(GRID[1]):
            y0, y1, x0, x1 = H * i // GRID[0], H * (i + 1) // GRID[0], W * j // GRID[1], W * (j + 1) // GRID[1]
            p = cv2.goodFeaturesToTrack(gray[y0:y1, x0:x1], PER_CELL, 0.003, 6, mask=mask[y0:y1, x0:x1])
            if p is not None:
                out.append(p[:, 0] + [x0, y0])
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 2), np.float32)


def track(video: str, boxes: dict, img_h: int, bonnet: int, first: int = 0, last: int | None = None) -> list[dict]:
    """Track points through every adjacent frame pair of the model-input video (self-contained:
    also runs on Colab). boxes: frame → vehicle boxes in this video's pixels."""
    cv2.setNumThreads(2)                                   # keep the laptop cool
    det_frames = np.array(sorted(boxes))
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), last or 10 ** 9)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    rows, prev, t0 = [], None, time.time()
    for f in range(first, n):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame[:img_h], cv2.COLOR_BGR2GRAY)
        if prev is not None:
            mask = np.zeros_like(prev)
            mask[:bonnet] = 255
            if len(det_frames):
                fd = det_frames[np.argmin(np.abs(det_frames - (f - 1)))]
                if abs(fd - (f - 1)) <= FRAME_STEP:
                    for x1, y1, x2, y2 in np.asarray(boxes[fd]).astype(int):
                        mask[max(y1, 0):y2, max(x1, 0):x2] = 0
            p0 = corners(prev, mask)
            if len(p0):
                p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, **LK)
                pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev, p1, None, **LK)
                good = (st1[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(pb - p0, axis=1) < FB_MAX_PX)
                a = np.column_stack([p0[good], p1[good] - p0[good]]).astype(np.float16)
                f0 = int(round(f / FRAME_STEP)) * FRAME_STEP          # nearest analysed frame
                rows.append(dict(t=float(f0 / fps), pair=int(f - f0), pts=zlib.compress(a.tobytes(), 6), n=len(a)))
        prev = gray
        if (f - first) % 1500 == 0 and f > first:
            el, done = time.time() - t0, f - first
            print(f"tracked {f}/{n} frames  {el / done * 1000:.0f} ms/frame  ~{el / done * (n - f) / 60:.1f} min left", flush=True)
    return rows


def inputs(drv: dict) -> tuple[dict, int, int]:
    folder = drv["folder"]
    meta = json.loads((folder / "meta.json").read_text())
    k = WIDTH / meta["width"]
    det = pd.read_parquet(folder / "detections.parquet")
    det = det[det.cls.isin(VEHICLES)]
    boxes = {int(f): (g[["x1", "y1", "x2", "y2"]].to_numpy() * k).round().astype(int).tolist() for f, g in det.groupby("frame")}
    bonnet = int(meta["height"] * drv["camera"]["bonnet"] * k) - 4
    return boxes, input_height(drv), bonnet


def run(drv: dict, seconds: float | None = None, start: float = 0.0):
    """On the laptop (a short test: --seconds)."""
    boxes, img_h, bonnet = inputs(drv)
    fps = json.loads((drv["folder"] / "meta.json").read_text())["fps"]
    first = int(start * fps)
    t0 = time.time()
    rows = track(str(input_video(drv)), boxes, img_h, bonnet, first, first + int(seconds * fps) if seconds else None)
    out = drv["folder"] / ("motion.parquet" if not seconds else "motion_test.parquet")
    save(rows, out, t0)


def save(rows: list[dict], out, t0: float):
    m = pd.DataFrame(rows)
    m.to_parquet(out, index=False)
    print(f"→ {out}: {len(m)} frame pairs, {m.n.median():.0f} points per pair (median); {(time.time() - t0) / 60:.1f} min", flush=True)


REMOTE = """
import json, time, zlib
import cv2, numpy as np, pandas as pd
GRID, PER_CELL, FB_MAX_PX, FRAME_STEP = {grid}, {per_cell}, {fb}, {step}
LK = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
{corners}
{track}
boxes = {{int(k): v for k, v in json.load(open("/content/boxes.json")).items()}}
rows = track("/content/input.mp4", boxes, {img_h}, {bonnet})
pd.DataFrame(rows).to_parquet("/content/motion.parquet", index=False)
print("done", len(rows), "frame pairs", flush=True)
"""


def run_kaggle(drv: dict):
    import inspect
    import tempfile
    from src.video import gpu
    boxes, img_h, bonnet = inputs(drv)
    script = REMOTE.format(grid=GRID, per_cell=PER_CELL, fb=FB_MAX_PX, step=FRAME_STEP, img_h=img_h, bonnet=bonnet,
                           corners=inspect.getsource(corners), track=inspect.getsource(track))
    work = drv["folder"] / "colab" / "kaggle_points"
    work.mkdir(parents=True, exist_ok=True)
    bj = work / "boxes.json"
    bj.write_text(json.dumps(boxes))
    t0 = time.time()
    gpu.run_job(f"points {drv['key']}", script, {"/content/input.mp4": input_video(drv), "/content/boxes.json": bj},
                       {"/content/motion.parquet": drv["folder"] / "motion.parquet"}, work, order=gpu.CLOUD)
    m = pd.read_parquet(drv["folder"] / "motion.parquet")
    print(f"→ {drv['folder'] / 'motion.parquet'}: {len(m)} frame pairs; Kaggle {(time.time() - t0) / 60:.1f} min", flush=True)


def run_colab(drv: dict, reuse: bool = False):
    """The whole drive on a Colab CPU runtime (~2 cores: slower than the laptop, but detached)."""
    import inspect
    import tempfile
    from src.video.colab import Session
    boxes, img_h, bonnet = inputs(drv)
    script = REMOTE.format(grid=GRID, per_cell=PER_CELL, fb=FB_MAX_PX, step=FRAME_STEP, img_h=img_h, bonnet=bonnet,
                           corners=inspect.getsource(corners), track=inspect.getsource(track))
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp, Session(f"points-{drv['key']}", gpu=None, reuse=reuse) as s:
        bj = Path(tmp) / "boxes.json"
        with open(bj, "w") as fh:
            json.dump(boxes, fh)
        s.upload(input_video(drv), "/content/input.mp4")
        s.upload(bj, "/content/boxes.json")
        s.start(script)
        if s.wait() != 0:
            print(s.log())
            raise RuntimeError("point tracking on Colab failed")
        s.download("/content/motion.parquet", drv["folder"] / "motion.parquet")
    m = pd.read_parquet(drv["folder"] / "motion.parquet")
    print(f"→ {drv['folder'] / 'motion.parquet'}: {len(m)} frame pairs, {m.n.median():.0f} points per pair; "
          f"Colab {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    ap.add_argument("--seconds", type=float, help="only N seconds (test → motion_test.parquet)")
    ap.add_argument("--start", type=float, default=0.0, help="with --seconds: from this second")
    ap.add_argument("--colab", action="store_true", help="the whole drive on a Colab CPU runtime")
    ap.add_argument("--kaggle", action="store_true", help="the whole drive on Kaggle")
    ap.add_argument("--reuse", action="store_true", help="with --colab: attach to the runtime if still alive")
    a = ap.parse_args()
    if a.kaggle:
        run_kaggle(drives.load(a.drive))
    elif a.colab:
        run_colab(drives.load(a.drive), a.reuse)
    else:
        run(drives.load(a.drive), a.seconds, a.start)


if __name__ == "__main__":
    main()
