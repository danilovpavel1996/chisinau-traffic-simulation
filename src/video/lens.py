"""
lens.py
-------
The camera's lens measured from the video itself with GeoCalib (ETH Zürich, ECCV
2024: a network + geometric optimisation that estimates focal length, radial
distortion and the camera's roll / pitch from one image), on a free Colab T4.

Run on N_FRAMES frames spread over the drive (the same cropped, 960-px frames the
segmentation sees); the median is the drive's lens. Compared with what the drive
already measured: the central focal length from parked cars + speed (ground.py) and
the average one from rotation vs the map (calibrate.py) — a pinhole lens would give
the same number twice; a lens with barrel distortion gives a larger central one.

Usage:
    .venv-video/bin/python -m src.video.lens <drive>

Output: lens_* keys added to calibration.json (+ lens_frames.json with every frame).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.video import drives

N_FRAMES = 80
CAMERA_MODEL = "simple_radial"      # focal length + one radial distortion coefficient k1

REMOTE = '''
import json, os, subprocess, sys, zipfile
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/cvg/GeoCalib"], check=True)
import torch
from geocalib import GeoCalib
zipfile.ZipFile("/content/frames.zip").extractall("/content/frames")
files = sorted(os.listdir("/content/frames"))
files = files[: {n}]                                                 # already spread over the street window
model = GeoCalib().to("cuda")
out = []
for n, f in enumerate(files):
    img = model.load_image("/content/frames/" + f).to("cuda")
    res = model.calibrate(img, camera_model="{cam}")
    cam = res["camera"]
    row = dict(t=float(f[:-4]))
    for name in ("f", "c", "size", "dist", "hfov", "vfov"):
        try:
            v = getattr(cam, name)
            row[name] = v.detach().cpu().flatten().tolist() if torch.is_tensor(v) else v
        except Exception as e:
            row[name] = None
    try:
        row["roll_pitch_deg"] = (res["gravity"].rp * 180 / 3.141592653589793).detach().cpu().flatten().tolist()
    except Exception:
        row["roll_pitch_deg"] = None
    out.append(row)
    if (n + 1) % 10 == 0:
        print(f"lens {{n + 1}}/{{len(files)}}", flush=True)
json.dump(out, open("/content/lens.json", "w"))
print("done", len(out), flush=True)
'''


def export_frames(drv: dict) -> Path:
    """N_FRAMES model-input images (scene.crop) spread over the street window, as a zip."""
    import zipfile
    import cv2
    from src.video.scene import crop
    out = drv["folder"] / "colab" / f"{drv['key']}_lens_frames.zip"
    out.parent.mkdir(exist_ok=True)
    meta = json.loads((drv["folder"] / "meta.json").read_text())
    t0, t1 = drv.get("street_window_s", [0, meta["end_s"]])
    cap = cv2.VideoCapture(str(drv["video"]))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for t in np.linspace(t0, t1, N_FRAMES + 2)[1:-1]:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, fr = cap.read()
            if ok:
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(crop(fr, drv["camera"]["hood"]), cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, 92])
                z.writestr(f"{t:08.1f}.jpg", buf.tobytes())
    return out


def run_colab(drv: dict, reuse: bool = False) -> dict:
    from src.video.colab import Session
    folder, key = drv["folder"], drv["key"]
    frames = export_frames(drv)
    t0 = time.time()
    with Session(f"lens-{key}", reuse=reuse) as s:
        s.upload(frames, "/content/frames.zip")
        s.start(REMOTE.format(n=N_FRAMES, cam=CAMERA_MODEL))
        if s.wait() != 0:
            print(s.log())
            raise RuntimeError("lens calibration on Colab failed")
        s.download("/content/lens.json", folder / "lens_frames.json")
    rows = json.loads((folder / "lens_frames.json").read_text())
    f = np.array([r["f"][0] for r in rows if r.get("f")])
    k1 = np.array([r["dist"][0] for r in rows if r.get("dist")])
    hfov = np.degrees([r["hfov"][0] if isinstance(r["hfov"], list) else r["hfov"] for r in rows if r.get("hfov") is not None])
    pitch = np.array([r["roll_pitch_deg"][1] for r in rows if r.get("roll_pitch_deg")])
    res = dict(lens_frames=len(rows), lens_focal_px=round(float(np.median(f)), 1),
               lens_focal_iqr=[round(float(np.percentile(f, 25)), 1), round(float(np.percentile(f, 75)), 1)],
               lens_k1=round(float(np.median(k1)), 4) if len(k1) else None,
               lens_hfov_deg=round(float(np.median(hfov)), 1) if len(hfov) else None,
               lens_pitch_deg=round(float(np.median(pitch)), 2) if len(pitch) else None)
    cal = folder / "calibration.json"
    c = json.loads(cal.read_text()) if cal.exists() else {}
    c.update(res)
    cal.write_text(json.dumps(c, indent=2))
    print(f"→ {cal}: {res} (Colab {(time.time() - t0) / 60:.1f} min)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    ap.add_argument("--reuse", action="store_true", help="attach to the drive's runtime if still alive")
    a = ap.parse_args()
    run_colab(drives.load(a.drive), a.reuse)


if __name__ == "__main__":
    main()
