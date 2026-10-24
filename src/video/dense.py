"""
dense.py
--------
Everything the spatial and 3D views need per frame, in one Colab GPU session:
street-scene segmentation (Mask2Former, Mapillary Vistas) and depth (Depth Anything V2
metric outdoor), on every 6th video frame — 5 per second, the very frames perception
detects vehicles on, so road, cars and depth line up.

Instead of thousands of frame images, one small video is uploaded: exactly the model
input (the image above the bonnet, 960 px wide; ~200 MB for 10 minutes). The VM decodes
it and runs both models; one upload and one runtime instead of two each.

    segmentation  class map of the lower image, every pixel (scene.describe_dense)
    depth         metres per pixel (model scale, fitted to the road later: depth.fit_to_ground),
                  at ½ of the 960-px image
    motion        feature points tracked through the next FLOW_PAIRS adjacent frame pairs
                  (vehicles excluded): position and motion per point, for speed.py — speed =
                  radial motion × fps × depth / distance from the vanishing point
                  (→ motion_dense.parquet; speed.py now uses track_points.py's points)

Usage:
    .venv-video/bin/python -m src.video.dense <drive>            # Colab T4
    .venv-video/bin/python -m src.video.dense <drive> --reuse    # attach to a runtime still alive

Output: data/video/processed/<drive>/scene_dense.parquet, depth.parquet
"""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

from src.video import drives
from src.video.depth import DOWN, MODEL as DEPTH_MODEL
from src.video.scene import CLASSES, CODES, DENSE_STEP, DENSE_TOP, WIDTH, describe_dense

FRAME_STEP = 6           # every 6th frame: 5 per second at 30 fps, the frames perception analyses
FLOW_PAIRS = 3           # adjacent frame pairs tracked after each analysed frame (speed.py)
SEG_MODEL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"

REMOTE = '''
import json, os, time, zlib
import cv2, numpy as np, pandas as pd, torch
from transformers import (AutoImageProcessor, Mask2FormerForUniversalSegmentation,
                          AutoModelForDepthEstimation)
CLASSES = {classes}
CODES = {codes}
DENSE_STEP, DENSE_TOP, DOWN, FRAME_STEP, IMG_H, FLOW_PAIRS = {step}, {top}, {down}, {frame_step}, {img_h}, {pairs}
det = pd.read_parquet("/content/detections.parquet") if os.path.exists("/content/detections.parquet") else None
K = {k}
VEH = {{"car", "bus", "truck", "motorcycle"}}
flow_rows, grays = [], {{}}

{describe}

seg_proc = AutoImageProcessor.from_pretrained("{seg}")
seg = Mask2FormerForUniversalSegmentation.from_pretrained("{seg}").eval().cuda().half()
names = np.array([CLASSES.get(seg.config.id2label[i], "other") for i in range(len(seg.config.id2label))])
dep_proc = AutoImageProcessor.from_pretrained("{depth}")
dep = AutoModelForDepthEstimation.from_pretrained("{depth}").eval().cuda().half()
cap = cv2.VideoCapture("/content/input.mp4")
fps = cap.get(cv2.CAP_PROP_FPS); n_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
n = (n_all + FRAME_STEP - 1) // FRAME_STEP
print("GPU:", torch.cuda.get_device_name(0), "|", n, "frames to analyse", flush=True)
seg_rows, dep_rows, f, t0 = [], [], 0, time.time()
while True:
    ok = cap.grab()
    if not ok:
        break
    if 0 < f % FRAME_STEP <= FLOW_PAIRS and (f - f % FRAME_STEP) in grays:  # the frames right after a sample
        g1 = cv2.cvtColor(cap.retrieve()[1], cv2.COLOR_BGR2GRAY)[:IMG_H]
        f0 = f - f % FRAME_STEP
        g0 = grays[f0][-1]
        pts = cv2.goodFeaturesToTrack(g0, 600, 0.01, 6)
        if pts is not None:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, None)
            ok = st[:, 0] == 1
            pp, qq = pts[ok, 0], nxt[ok, 0]
            if det is not None:
                for b in det[(det.frame == f0) & det.cls.isin(VEH)].itertuples():
                    keep = ~((pp[:, 0] >= b.x1 * K) & (pp[:, 0] <= b.x2 * K) & (pp[:, 1] >= b.y1 * K) & (pp[:, 1] <= b.y2 * K))
                    pp, qq = pp[keep], qq[keep]
            arr = np.column_stack([pp, qq - pp]).astype(np.float16)
            flow_rows.append(dict(t=float(f0 / fps), pair=int(f % FRAME_STEP), pts=zlib.compress(arr.tobytes(), 6), n=len(arr)))
        grays[f0].append(g1)
        if f % FRAME_STEP == FLOW_PAIRS:
            grays.pop(f0, None)
    if f % FRAME_STEP == 0:
        img = cv2.cvtColor(cap.retrieve()[1], cv2.COLOR_BGR2RGB)[:IMG_H]          # drop the padding row
        grays[f] = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)]
        t = f / fps
        inp = {{k: v.cuda().half() if v.dtype.is_floating_point else v.cuda()
               for k, v in seg_proc(images=img, return_tensors="pt").items()}}
        with torch.no_grad():
            out = seg(**inp)
        out.class_queries_logits = out.class_queries_logits.float()
        out.masks_queries_logits = out.masks_queries_logits.float()
        s = seg_proc.post_process_semantic_segmentation(out, target_sizes=[img.shape[:2]])[0].cpu().numpy()
        seg_rows.append(describe_dense(s, names, t))
        with torch.no_grad():
            d = dep(pixel_values=dep_proc(images=img, return_tensors="pt")["pixel_values"].cuda().half()).predicted_depth.float()
        h, w = img.shape[0] // DOWN, img.shape[1] // DOWN
        d = torch.nn.functional.interpolate(d[:, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        dep_rows.append(dict(t=float(t), depth=zlib.compress(d.cpu().numpy().astype(np.float16).tobytes(), 6),
                             shape=json.dumps([h, w])))
        k = len(seg_rows)
        if k % 200 == 0 or k == n:
            el = time.time() - t0
            print(f"dense {{k}}/{{n}} frames  {{el / k:.2f}} s/frame  ~{{el / k * (n - k) / 60:.1f}} min left", flush=True)
    f += 1
pd.DataFrame(seg_rows).to_parquet("/content/scene_dense.parquet", index=False)
pd.DataFrame(dep_rows).to_parquet("/content/depth.parquet", index=False)
pd.DataFrame(flow_rows).to_parquet("/content/motion.parquet", index=False)
print("done", len(seg_rows), "frames", flush=True)
'''


def input_video(drv: dict) -> Path:
    """The model input as a small video: the image above the bonnet, 960 px wide (= scene.crop)."""
    meta = json.loads((drv["folder"] / "meta.json").read_text())
    crop_h = int(meta["height"] * (1 - drv["camera"]["hood"] + 0.03))
    out_h = int(WIDTH * crop_h / meta["width"])
    out = drv["folder"] / "colab" / f"{drv['key']}_input{WIDTH}.mp4"
    if not out.exists():
        out.parent.mkdir(exist_ok=True)
        print(f"writing the model-input video → {out.name} ({WIDTH}×{out_h})…", flush=True)
        subprocess.run(["nice", "-n", "10", "ffmpeg", "-v", "error", "-stats", "-y", "-i", str(drv["video"]), "-map", "0:v:0",
                        # H.264 4:2:0 needs an even height: one padding row, cropped off again on the VM
                        "-vf", f"crop={meta['width']}:{crop_h}:0:0,scale={WIDTH}:{out_h}:flags=area,"
                               f"pad={WIDTH}:{out_h + out_h % 2}:0:0",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-threads", "4", "-pix_fmt", "yuv420p",
                        "-vsync", "passthrough", "-an", str(out)], check=True)
    return out


def input_height(drv: dict) -> int:
    meta = json.loads((drv["folder"] / "meta.json").read_text())
    return int(WIDTH * int(meta["height"] * (1 - drv["camera"]["hood"] + 0.03)) / meta["width"])


def run_colab(drv: dict, reuse: bool = False, collect: bool = False):
    from src.video.colab import Session
    folder, key = drv["folder"], drv["key"]
    video = input_video(drv)
    script = REMOTE.format(classes=json.dumps(CLASSES), codes=repr(CODES), step=DENSE_STEP, top=DENSE_TOP, down=DOWN,
                           frame_step=FRAME_STEP, img_h=input_height(drv), pairs=FLOW_PAIRS,
                           k=WIDTH / json.loads((folder / "meta.json").read_text())["width"], describe=inspect.getsource(describe_dense), seg=SEG_MODEL,
                           depth=DEPTH_MODEL)
    t0 = time.time()
    with Session(f"dense-{key}", reuse=reuse or collect) as s:
        if not collect:                              # --collect: the job already finished on the runtime
            s.upload(video, "/content/input.mp4")
            if (folder / "detections.parquet").exists():
                s.upload(folder / "detections.parquet", "/content/detections.parquet")
            s.install("transformers==4.46.3")
            s.start(script)
            if s.wait() != 0:
                print(s.log())
                raise RuntimeError("dense segmentation + depth on Colab failed")
        for name in ("scene_dense.parquet", "depth.parquet"):
            s.download(f"/content/{name}", folder / name)
        # the points speed.py uses come from track_points.py (every frame pair, spread evenly);
        # these (strongest corners, 3 pairs per analysed frame) are kept for comparison
        s.download("/content/motion.parquet", folder / "motion_dense.parquet")
    n = len(pd.read_parquet(folder / "scene_dense.parquet"))
    print(f"→ {folder}/scene_dense.parquet, depth.parquet: {n} frames ({n / json.loads((folder / 'meta.json').read_text())['end_s']:.1f} per s), "
          f"Colab {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    ap.add_argument("--reuse", action="store_true", help="attach to the drive's runtime if still alive")
    ap.add_argument("--collect", action="store_true", help="only download the results of a job that finished")
    a = ap.parse_args()
    run_colab(drives.load(a.drive), a.reuse, a.collect)


if __name__ == "__main__":
    main()
