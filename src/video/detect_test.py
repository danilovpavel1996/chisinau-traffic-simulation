"""
detect_test.py
--------------
Which detection and tracking settings see the scene best — measured on short clips before
spending GPU hours on whole drives (Colab T4).

    rates    the pipeline's detector + tracker (perception.py: RT-DETRv2, BoT-SORT with ReID)
             at 5 (today), 10, 15 and 30 frames per second; the tracker's memory kept at the
             same ~10 s whatever the rate
    tiles    at 10 fps also the upper image cut into two overlapping tiles, each through the
             detector on its own: a distant traffic light is ~1.6× larger in the detector's
             input (it shrinks every image to 640 px) — for lights missed today (15:19 on
             2026_10_24_17_08)

Scored on the laptop (score()), without GPS:
    vehicles   track IDs per minute; how long tracks last; new IDs born on a large box in the
               middle of the image — a car already in view getting a new ID (an ID switch)
    lights     share of 0.2-s moments with a light detected; landmarks found (world.lights)

Usage:
    .venv-video/bin/python -m src.video.detect_test <drive> 470:560 700:740 895:925   # run on Colab
    .venv-video/bin/python -m src.video.detect_test <drive> --score

Output: data/video/processed/<drive>/detect_test/<config>.parquet
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import time
import zipfile

import numpy as np
import pandas as pd

from src.video import drives
from src.video.drives import ROOT

RATES = {"5fps": 6, "10fps": 3, "15fps": 2, "30fps": 1}       # config → frame stride (30 fps video)
TILES_RATE = "10fps"
BUFFER_S = 10.0
TILE_TOP = 0.55                                                # tiles cover the upper 55 % of the image
SHIP = ["src/video/__init__.py", "src/video/perception.py", "src/video/compute.py", "src/video/drives.py",
        "src/video/botsort_dashcam.yaml", "configs/drives.yaml"]

REMOTE = '''
import json, sys, time
sys.path.insert(0, "/content/repo")
import cv2, numpy as np, pandas as pd
from src.video import perception as P
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import IterableSimpleNamespace, YAML
import inspect

CLIPS = {clips}
RATES = {rates}
TILES_RATE, BUFFER_S, TILE_TOP = "{tiles_rate}", {buffer_s}, {tile_top}
detect = P.RTDetrV2Detector(0.1, "cuda")
names = sorted(set(P.CLASSES.values()))

def tracker(stride):
    cfg = YAML.load(P.TRACKER)
    cfg["device"] = "cuda:0"
    cfg["track_buffer"] = int(BUFFER_S * 30 / stride)          # same ~10 s at every rate
    args = IterableSimpleNamespace(**cfg)
    return BOTSORT(args, frame_rate=30) if "frame_rate" in inspect.signature(BOTSORT.__init__).parameters else BOTSORT(args)

def tile_lights(frame):
    H, W = frame.shape[:2]
    top = frame[:int(H * TILE_TOP)]
    out = []
    for x0, x1 in ((0, int(W * 0.57)), (int(W * 0.43), W)):
        xyxy, cf, lab = detect(np.ascontiguousarray(top[:, x0:x1]))
        for b, c, n in zip(xyxy, cf, lab):
            if n == "traffic_light":
                out.append((b[0] + x0, b[1], b[2] + x0, b[3], float(c)))
    return out

def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)

todo = [(c, s) for c, s in RATES.items()] + [("tiles", RATES[TILES_RATE])]
t_all = time.time()
for config, stride in todo:
    rows = []
    for clip, (path, f_first) in CLIPS.items():
        cap = cv2.VideoCapture(path)
        trk = tracker(stride)
        f, t0 = 0, time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if f % stride == 0:
                F = f_first + f
                t = F / 30.0
                xyxy, cf, labels = detect(frame)
                tracks = trk.update(P.Dets(xyxy, cf, np.array([names.index(n) for n in labels], float)), frame)
                full = []
                if len(tracks):
                    idx = tracks[:, 7].astype(int)
                    for (x1, y1, x2, y2), p, name, tid in zip(xyxy[idx].astype(int), cf[idx], [labels[i] for i in idx],
                                                              tracks[:, 4].astype(int)):
                        rows.append(dict(clip=clip, frame=F, t=t, track_id=int(tid), cls=name, conf=float(p),
                                         x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2), source="full",
                                         light=P.light_state(frame, (x1, y1, x2, y2)) if name == "traffic_light" else None))
                        if name == "traffic_light":
                            full.append((x1, y1, x2, y2))
                if config == "tiles":
                    for b in tile_lights(frame):
                        if all(iou(b, a) < 0.3 for a in full):
                            x1, y1, x2, y2 = (int(v) for v in b[:4])
                            rows.append(dict(clip=clip, frame=F, t=t, track_id=-1, cls="traffic_light", conf=b[4],
                                             x1=x1, y1=y1, x2=x2, y2=y2, source="tile",
                                             light=P.light_state(frame, (x1, y1, x2, y2))))
            f += 1
        print(f"{{config}} {{clip}}: {{f}} frames in {{time.time() - t0:.0f}} s  (total {{(time.time() - t_all) / 60:.1f}} min)", flush=True)
    pd.DataFrame(rows).to_parquet(f"/content/out/{{config}}.parquet", index=False)
print("done", flush=True)
'''


def clip_videos(drv: dict, spans: list[tuple[float, float]]) -> dict:
    out = {}
    folder = drv["folder"] / "detect_test"
    folder.mkdir(exist_ok=True)
    for a, b in spans:
        name = f"{int(a)}_{int(b)}"
        p = folder / f"clip_{name}.mp4"
        if not p.exists():
            # exact frames from the full-resolution video (accurate seek), lightly compressed
            subprocess.run(["nice", "-n", "10", "ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-i", str(drv["video"]),
                            "-t", f"{b - a:.3f}", "-map", "0:v:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                            "-threads", "4", "-an", str(p)], check=True)
        out[name] = (p, int(round(a * 30)))
    return out


def run_colab(drv: dict, spans: list[tuple[float, float]]):
    from src.video.colab import Session
    clips = clip_videos(drv, spans)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for rel in SHIP:
            z.write(ROOT / rel, rel)
    code = drv["folder"] / "detect_test" / "code.zip"
    code.write_bytes(buf.getvalue())
    remote = {k: (f"/content/clips/{p.name}", f0) for k, (p, f0) in clips.items()}
    script = REMOTE.format(clips=json.dumps(remote), rates=json.dumps(RATES), tiles_rate=TILES_RATE,
                           buffer_s=BUFFER_S, tile_top=TILE_TOP)
    t0 = time.time()
    with Session(f"dettest-{drv['key']}") as s:
        s.upload(code, "/content/repo/code.zip")
        for p, _ in clips.values():
            s.upload(p, f"/content/clips/{p.name}")
        s.install("ultralytics", "onnxruntime-gpu", "transformers>=4.50")
        s.start('import zipfile, os; zipfile.ZipFile("/content/repo/code.zip").extractall("/content/repo"); '
                'os.makedirs("/content/out", exist_ok=True)\n' + script)
        if s.wait() != 0:
            print(s.log())
            raise RuntimeError("detection test on Colab failed")
        for config in list(RATES) + ["tiles"]:
            s.download(f"/content/out/{config}.parquet", drv["folder"] / "detect_test" / f"{config}.parquet")
    print(f"→ {drv['folder'] / 'detect_test'}: Colab {(time.time() - t0) / 60:.1f} min", flush=True)


def score(drv: dict):
    folder = drv["folder"] / "detect_test"
    meta = json.loads((drv["folder"] / "meta.json").read_text())
    W, H = meta["width"], meta["height"]
    rows = []
    for config in list(RATES) + ["tiles"]:
        f = folder / f"{config}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        v = d[d.cls.isin(["car", "bus", "truck", "motorcycle"])]
        minutes = sum((g.t.max() - g.t.min()) for _, g in d.groupby("clip")) / 60
        life = v.groupby(["clip", "track_id"]).t.agg(lambda s: s.max() - s.min())
        first = v.sort_values("t").groupby(["clip", "track_id"]).head(1)
        clip_start = d.groupby("clip").t.min()
        # a new ID on a large box well inside the image, not at a clip's start: a car already in view
        h = first.y2 - first.y1
        inner = (first.x1 > 0.08 * W) & (first.x2 < 0.92 * W)
        switch = first[(h > 0.09 * H) & inner & (first.t > first.clip.map(clip_start) + 1.0)]
        L = d[d.cls == "traffic_light"]
        grid = []
        for c, g in d.groupby("clip"):
            ts = np.round(np.arange(g.t.min(), g.t.max(), 0.2), 1)
            seen = set(np.round(np.floor(L[L.clip == c].t / 0.2) * 0.2, 1))
            grid += [t in seen for t in ts]
        rows.append(dict(config=config, frames=d.frame.nunique(), vehicle_ids_per_min=round(life.size / minutes),
                         track_life_median_s=round(float(life.median()), 1), id_switches_per_min=round(len(switch) / minutes, 1),
                         light_moments=f"{np.mean(grid):.0%}", light_detections=len(L),
                         tile_lights=int((L.source == "tile").sum()) if "source" in L else 0))
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    ap.add_argument("spans", nargs="*", help="clips as start:end seconds")
    ap.add_argument("--score", action="store_true", help="score the downloaded results")
    a = ap.parse_args()
    drv = drives.load(a.drive)
    if a.score:
        score(drv)
    else:
        run_colab(drv, [tuple(map(float, s.split(":"))) for s in a.spans])


if __name__ == "__main__":
    main()
