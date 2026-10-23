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
DOWN = 2                 # stored depth map: 1/2 of the 960-px inference image (480 × ~245)

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


FIT_RANGE_M = (6.0, 35.0)    # road pixels whose flat-road distance is trusted (validated vs GPS)
GROUND = ("road", "lane_marking", "crosswalk")


def fit_to_ground(drv: dict) -> dict:
    """
    Metric depth that agrees with the measured road-plane geometry. The model reads this
    dashcam's distances too far, and not by one factor (2.5× close, 1.7× at 40 m: trained on
    narrower driving cameras), so per drive: on road pixels 6–35 m ahead, where the flat-road
    distance is calibrated (parked cars + speed), fit  depth_model = c · Z^b  and invert it.
    """
    from src.video.bev import Geometry
    from src.video.scene import CODES
    folder = drv["folder"]
    dense = pd.read_parquet(folder / "scene_dense.parquet").sort_values("t").reset_index(drop=True)
    geo = Geometry(drv, dense.iloc[0])
    td, maps = load(drv)
    road = [CODES.index(c) for c in GROUND]
    zs, ds = [], []
    for r, t in zip(dense.to_dict("records")[::3], dense.t[::3]):
        j = int(np.argmin(np.abs(td - t)))
        if abs(td[j] - t) > 0.3:
            continue
        D = maps[j]
        m = np.frombuffer(zlib.decompress(r["mask"]), np.uint8).reshape(json.loads(r["mask_shape"]))
        rr, cc = np.nonzero(np.isin(m, road))
        rr, cc = rr[::7], cc[::7]
        y, x = r["top"] + (rr + 0.5) * r["step"], (cc + 0.5) * r["step"]
        xu, yu = geo.undistort(x, y)
        ok = (yu > geo.y_h + 4) & (y < geo.bonnet - 4)
        Z = geo.fz * geo.h / (yu[ok] - geo.y_h)
        dz = D[np.clip((y[ok] / DOWN).astype(int), 0, D.shape[0] - 1), np.clip((x[ok] / DOWN).astype(int), 0, D.shape[1] - 1)]
        keep = (Z > FIT_RANGE_M[0]) & (Z < FIT_RANGE_M[1]) & (dz > 0)
        zs.append(Z[keep]); ds.append(dz[keep])
    Z, D = np.concatenate(zs), np.concatenate(ds)
    bins = np.geomspace(*FIT_RANGE_M, 12)
    idx = np.digitize(Z, bins)
    mz = np.array([np.median(Z[idx == i]) for i in range(1, len(bins)) if (idx == i).sum() > 200])
    md = np.array([np.median(D[idx == i]) for i in range(1, len(bins)) if (idx == i).sum() > 200])
    b, logc = np.polyfit(np.log(mz), np.log(md), 1)
    res = dict(depth_fit_c=round(float(np.exp(logc)), 4), depth_fit_b=round(float(b), 4), depth_fit_pixels=int(len(Z)))
    cal = folder / "calibration.json"
    c = json.loads(cal.read_text()) if cal.exists() else {}
    c.update(res)
    cal.write_text(json.dumps(c, indent=2))
    print(f"→ {cal}: model depth = {res['depth_fit_c']} · Z^{res['depth_fit_b']} ({len(Z)} road pixels)", flush=True)
    return res


def metric(drv: dict, depth_map: np.ndarray) -> np.ndarray:
    """Model depth → metres consistent with the road-plane geometry (fit_to_ground)."""
    c = json.loads((drv["folder"] / "calibration.json").read_text())
    return (np.maximum(depth_map, 1e-3) / c["depth_fit_c"]) ** (1 / c["depth_fit_b"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    ap.add_argument("--reuse", action="store_true", help="attach to the drive's runtime if still alive")
    ap.add_argument("--fit", action="store_true", help="only fit the stored depth to the road geometry")
    a = ap.parse_args()
    drv = drives.load(a.drive)
    if not a.fit:
        run_colab(drv, a.reuse)
    fit_to_ground(drv)


if __name__ == "__main__":
    main()
