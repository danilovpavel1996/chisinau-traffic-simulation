"""
depth.py
--------
Distance to every pixel: Depth Anything V2, metric outdoor version (Large, 335M
parameters, fine-tuned on Virtual KITTI driving scenes → metres), on the dense
frames (2 per second, the same cropped frames scene.py --dense segments), on a free
Colab T4.

The flat-road spatial view (bev.py) can only place things that touch the road; with
depth every pixel has a distance — cars, poles, traffic lights, kerbs, slopes — the
basis of the 3D surround view.

Stored per frame: the depth map at 1/DOWN of the inference image, float16, zlib.

Usage:
    .venv-video/bin/python -m src.video.depth <drive>            # on Colab (needs scene.py --dense frames)
    .venv-video/bin/python -m src.video.depth <drive> --reuse    # attach to a runtime still alive

Output: data/video/processed/<drive>/depth.parquet  (t, depth, shape)
"""

from __future__ import annotations

import argparse
import json
import time
import zlib

import numpy as np
import pandas as pd

from src.video import drives

MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
DOWN = 4                 # stored depth map: 1/4 of the 960-px inference image (240 × ~123)

REMOTE = '''
import io, os, time, zipfile, zlib
import numpy as np, pandas as pd, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
zipfile.ZipFile("/content/frames.zip").extractall("/content/frames")
files = sorted(os.listdir("/content/frames"))
proc = AutoImageProcessor.from_pretrained("{model}")
model = AutoModelForDepthEstimation.from_pretrained("{model}").to("cuda").eval().half()
print("GPU:", torch.cuda.get_device_name(0), "|", len(files), "frames", flush=True)
rows, t0 = [], time.time()
for n, f in enumerate(files):
    img = Image.open("/content/frames/" + f).convert("RGB")
    inp = proc(images=img, return_tensors="pt").to("cuda")
    with torch.no_grad():
        d = model(pixel_values=inp["pixel_values"].half()).predicted_depth.float()        # metres, model resolution
    h, w = img.height // {down}, img.width // {down}
    d = torch.nn.functional.interpolate(d[:, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    a = d.cpu().numpy().astype(np.float16)
    rows.append(dict(t=float(f[:-4]), depth=zlib.compress(a.tobytes(), 6), shape=f"[{{h}}, {{w}}]"))
    if (n + 1) % 100 == 0 or n + 1 == len(files):
        el = time.time() - t0
        print(f"depth {{n + 1}}/{{len(files)}} frames  {{el / (n + 1):.2f}} s/frame  ~{{el / (n + 1) * (len(files) - n - 1) / 60:.1f}} min left", flush=True)
pd.DataFrame(rows).to_parquet("/content/depth.parquet", index=False)
print("done", len(rows), "frames", flush=True)
'''


def run_colab(drv: dict, reuse: bool = False):
    from src.video.colab import Session
    from src.video.scene import export_colab
    folder, key = drv["folder"], drv["key"]
    frames = folder / "colab" / f"{key}_frames_dense.zip"
    if not frames.exists():
        export_colab(drv, dense=True)
    t0 = time.time()
    with Session(f"depth-{key}", reuse=reuse) as s:
        s.upload(frames, "/content/frames.zip")
        s.start(REMOTE.format(model=MODEL, down=DOWN))
        code = s.wait()
        if code != 0:
            print(s.log())
            raise RuntimeError(f"depth on Colab failed (exit {code})")
        s.download("/content/depth.parquet", folder / "depth.parquet")
    d = pd.read_parquet(folder / "depth.parquet")
    print(f"→ {folder / 'depth.parquet'}: {len(d)} frames, Colab {(time.time() - t0) / 60:.1f} min", flush=True)


def load(drv: dict) -> tuple[np.ndarray, list[np.ndarray]]:
    """Frame times and depth maps (metres, float32)."""
    d = pd.read_parquet(drv["folder"] / "depth.parquet").sort_values("t")
    maps = [np.frombuffer(zlib.decompress(b), np.float16).reshape(json.loads(s)).astype(np.float32)
            for b, s in zip(d.depth, d["shape"])]
    return d.t.to_numpy(), maps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    ap.add_argument("--reuse", action="store_true", help="attach to the drive's runtime if still alive")
    a = ap.parse_args()
    run_colab(drives.load(a.drive), a.reuse)


if __name__ == "__main__":
    main()
