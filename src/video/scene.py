"""
scene.py
--------
Street-scene segmentation (Mask2Former, Mapillary Vistas classes) on selected
frames, for features that can be checked against the map:

    crosswalk markings      → OSM crossings / junctions
    lane markings, curbs    → lanes per direction, our lane, road edges (scanlines)
    guard rail, bridge      → OSM bridge=yes
    traffic signs / lights  → junctions, signals

Frames: every `EVERY_S` seconds of the street window plus candidate moments
from the cheap detectors (traffic lights, stops, turns). ~2 s/frame on a laptop CPU.

Usage:
    .venv-video/bin/python -m src.video.scene download-model   # once, 826 MB
    .venv-video/bin/python -m src.video.scene <drive>

Output: data/video/processed/<drive>/scene.parquet — one row per analysed frame:
class shares of the image above the bonnet, crosswalk position (lowest row),
and per scanline the left→right runs of (class, x0, x1) for lane/road analysis.

Model: facebook/mask2former-swin-large-mapillary-vistas-semantic in models/
(Mapillary Vistas: non-commercial licence — fine for research, revisit for a product).
"""

from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np
import pandas as pd
import torch

from src.video import drives
from src.video.compute import limit_cpu, progress
from src.video.drives import ROOT
from src.video.motion import ego_states, stops, turns

MODEL = ROOT / "models" / "mask2former-vistas"
EVERY_S = 2.0          # regular sampling inside the street window
WIDTH = 960            # inference width (px)
SCANLINES = (0.75, 0.85, 0.95)   # rows of the above-bonnet image, as fractions of its height
CLASSES = {            # Vistas label → short name stored in the output
    "Lane Marking - Crosswalk": "crosswalk",
    "Lane Marking - General": "lane_marking",
    "Road": "road",
    "Curb": "curb",
    "Sidewalk": "sidewalk",
    "Bike Lane": "bike_lane",
    "Guard Rail": "guard_rail",
    "Bridge": "bridge",
    "Traffic Sign (Front)": "sign_front",
    "Traffic Light": "traffic_light",
    "Car": "car", "Bus": "bus", "Truck": "truck",
}


def candidate_times(drv: dict) -> list[float]:
    """Regular grid over the street window + moments the cheap detectors flag."""
    folder, cam = drv["folder"], drv["camera"]
    meta = json.loads((folder / "meta.json").read_text())
    t0, t1 = drv.get("street_window_s", [0, meta["end_s"]])
    e = ego_states(pd.read_parquet(folder / "ego.parquet"), meta["width"], cam["hfov_deg"], cam["hood"])
    det = pd.read_parquet(folder / "detections.parquet")
    times = set(np.round(np.arange(t0, t1, EVERY_S), 1))
    lights = det[(det.cls == "traffic_light") & ((det.y2 - det.y1) > 0.03 * meta["height"])].t
    times |= set(np.round(lights.to_numpy()[::3], 1))                 # visible traffic lights
    for ev in (stops(e).t_start, turns(e).t_start, turns(e).t_end):   # stops and turns
        for t in ev:
            times |= {round(t - 2, 1), round(t, 1)}
    return sorted(t for t in times if t0 <= t <= t1)


def runs(row: np.ndarray, names: np.ndarray) -> list[list]:
    """Left→right runs of one segmentation row: [[class, x0, x1], …] (short runs dropped)."""
    out, start = [], 0
    for x in range(1, len(row) + 1):
        if x == len(row) or row[x] != row[start]:
            if x - start >= 3:
                out.append([str(names[row[start]]), start, x - 1])
            start = x
    return out


def run(drv: dict):
    limit_cpu()
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL).eval()
    id2label = model.config.id2label
    names = np.array([CLASSES.get(id2label[i], "other") for i in range(len(id2label))])

    times = candidate_times(drv)
    hood = drv["camera"]["hood"]
    out_file, partial = drv["folder"] / "scene.parquet", drv["folder"] / "scene.partial.parquet"
    rows = pd.read_parquet(partial).to_dict(orient="records") if partial.exists() else []
    done = {r["t"] for r in rows}                       # resume after an interruption
    todo = [t for t in times if t not in done]
    cap = cv2.VideoCapture(str(drv["video"]))
    t_start = time.time()
    bar = progress(len(times), "scene segmentation", initial=len(times) - len(todo))
    for n, t in enumerate(todo):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        img = cv2.cvtColor(fr[: int(fr.shape[0] * (1 - hood + 0.03))], cv2.COLOR_BGR2RGB)   # above the bonnet
        img = cv2.resize(img, (WIDTH, int(WIDTH * img.shape[0] / img.shape[1])))
        with torch.no_grad():
            out = model(**proc(images=img, return_tensors="pt"))
        seg = proc.post_process_semantic_segmentation(out, target_sizes=[img.shape[:2]])[0].numpy()
        h = seg.shape[0]
        r = dict(t=float(t), width=WIDTH, height=h)
        for short in set(CLASSES.values()):
            r[short] = float(np.isin(seg, np.flatnonzero(names == short)).mean())
        cw_rows = np.flatnonzero((names[seg] == "crosswalk").any(axis=1))
        r["crosswalk_bottom"] = float(cw_rows.max() / h) if len(cw_rows) else np.nan   # 1.0 = at the bonnet
        r["scanlines"] = json.dumps({f"{y:.2f}": runs(seg[int(y * h) - 1], names) for y in SCANLINES})
        rows.append(r)
        bar.update(1)
        if (n + 1) % 20 == 0:                           # checkpoint ≈ every minute
            pd.DataFrame(rows).to_parquet(partial, index=False)
    bar.close()
    df = pd.DataFrame(rows).sort_values("t")
    df.to_parquet(out_file, index=False)
    partial.unlink(missing_ok=True)
    print(f"→ {out_file}: {len(df)} frames, crosswalk in {(df.crosswalk > 0.002).mean():.0%} of them, "
          f"{time.time() - t_start:.0f} s", flush=True)


def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download("facebook/mask2former-swin-large-mapillary-vistas-semantic", local_dir=MODEL,
                      allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"])
    print(f"→ {MODEL}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml, or 'download-model'")
    a = ap.parse_args()
    download_model() if a.drive == "download-model" else run(drives.load(a.drive))


if __name__ == "__main__":
    main()
