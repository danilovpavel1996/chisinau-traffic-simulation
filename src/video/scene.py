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
    .venv-video/bin/python -m src.video.scene <drive>          # on this Mac (~2.5 s/frame CPU)
    .venv-video/bin/python -m src.video.scene <drive> --colab   # free Colab T4 via the Colab CLI (~minutes)
    .venv-video/bin/python -m src.video.scene <drive> --colab --dense   # 2 frames/s, fine class maps → bev.py
    .venv-video/bin/python -m src.video.scene <drive> --export-colab        # or by hand in the Colab web UI:
    .venv-video/bin/python -m src.video.scene <drive> --import ~/Downloads/scene.parquet

Output: data/video/processed/<drive>/scene.parquet — one row per analysed frame:
class shares of the image above the bonnet, crosswalk position (lowest row),
and per scanline the left→right runs of (class, x0, x1) for lane/road analysis.

--dense: scene_dense.parquet — DENSE_FPS frames per second over the whole video, each
a class map of the lower image (from DENSE_TOP down, every DENSE_STEP-th pixel,
zlib-compressed) — fine enough to project onto the ground for the spatial view
(bev.py); the sparse class map (every 8th pixel) has ~1 row per 15 m at 40 m.

Model: facebook/mask2former-swin-large-mapillary-vistas-semantic in models/
(Mapillary Vistas: non-commercial licence — fine for this personal project, see README → Licensing).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import subprocess
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
    "Street Light": "street_light", "Pole": "pole", "Utility Pole": "pole", "Traffic Sign Frame": "pole",
    "Curb Cut": "curb_cut", "Barrier": "barrier", "Building": "building", "Vegetation": "vegetation",
    "Sky": "sky", "Person": "person", "Bicyclist": "person", "Motorcyclist": "person",
}
MASK_STEP = 8          # stored class map: every 8th pixel (960 px → 120 columns)
DENSE_FPS = 5          # dense frames per second (dense.py: every 6th frame, as perception)
DENSE_STEP = 1         # dense class map: every pixel (was every 2nd)
DENSE_TOP = 0.35       # --dense: rows from this fraction of the image height down (the horizon is ~0.5)
CODES = sorted(set(CLASSES.values())) + ["other"]     # class map value → short name


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


def crop(frame: np.ndarray, hood: float) -> np.ndarray:
    """Above-the-bonnet RGB image at inference width (the exact model input)."""
    img = cv2.cvtColor(frame[: int(frame.shape[0] * (1 - hood + 0.03))], cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (WIDTH, int(WIDTH * img.shape[0] / img.shape[1])))


def describe(seg: np.ndarray, names: np.ndarray, t: float) -> dict:
    """One analysed frame: class shares, crosswalk position, road scanlines."""
    h = seg.shape[0]
    r = dict(t=float(t), width=WIDTH, height=h)
    for short in set(CLASSES.values()):
        r[short] = float(np.isin(seg, np.flatnonzero(names == short)).mean())
    cw_rows = np.flatnonzero((names[seg] == "crosswalk").any(axis=1))
    r["crosswalk_bottom"] = float(cw_rows.max() / h) if len(cw_rows) else np.nan   # 1.0 = at the bonnet
    r["scanlines"] = json.dumps({f"{y:.2f}": runs(seg[int(y * h) - 1], names) for y in SCANLINES})
    # small class map of the whole frame, for features computed later without the model
    code = {c: i for i, c in enumerate(CODES)}
    small = names[seg[::MASK_STEP, ::MASK_STEP]]
    r["mask"] = np.vectorize(code.get)(small).astype(np.uint8).tobytes()
    r["mask_shape"] = json.dumps(list(small.shape))
    return r


def describe_dense(seg: np.ndarray, names: np.ndarray, t: float) -> dict:
    """One frame for the spatial view: compressed class map of the lower image."""
    import zlib
    code = {c: i for i, c in enumerate(CODES)}
    lut = np.array([code.get(n, code["other"]) for n in names], dtype=np.uint8)
    top = int(seg.shape[0] * DENSE_TOP)
    small = lut[seg[top::DENSE_STEP, ::DENSE_STEP]]
    return dict(t=float(t), width=seg.shape[1], height=seg.shape[0], top=top, step=DENSE_STEP,
                mask=zlib.compress(small.tobytes(), 6), mask_shape=json.dumps(list(small.shape)))


def dense_times(drv: dict) -> list[float]:
    meta = json.loads((drv["folder"] / "meta.json").read_text())
    return [round(t, 1) for t in np.arange(0, meta["end_s"], 1 / DENSE_FPS)]


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
        img = crop(fr, hood)
        with torch.no_grad():
            out = model(**proc(images=img, return_tensors="pt"))
        seg = proc.post_process_semantic_segmentation(out, target_sizes=[img.shape[:2]])[0].numpy()
        rows.append(describe(seg, names, t))
        bar.update(1)
        if (n + 1) % 20 == 0:                           # checkpoint ≈ every minute
            pd.DataFrame(rows).to_parquet(partial, index=False)
    bar.close()
    save(drv, pd.DataFrame(rows), time.time() - t_start)
    partial.unlink(missing_ok=True)


def save(drv: dict, df: pd.DataFrame, seconds: float | None = None):
    out_file = drv["folder"] / "scene.parquet"
    df = df.sort_values("t")
    df.to_parquet(out_file, index=False)
    print(f"→ {out_file}: {len(df)} frames, crosswalk in {(df.crosswalk > 0.002).mean():.0%} of them"
          + (f", {seconds:.0f} s" if seconds else ""), flush=True)


# ── Google Colab (free GPU) ───────────────────────────────────────────────────
def export_colab(drv: dict, dense: bool = False):
    """
    Write the model-input frames + a self-contained notebook for Colab:
    processed/<drive>/colab/{<drive>_frames.zip, scene_colab.ipynb}. The frames are
    already cropped/resized, so the 3 GB video never leaves this machine.
    """
    import inspect
    import zipfile
    times = dense_times(drv) if dense else candidate_times(drv)
    hood = drv["camera"]["hood"]
    out = drv["folder"] / "colab"
    out.mkdir(exist_ok=True)
    zpath = out / f"{drv['key']}_frames{'_dense' if dense else ''}.zip"
    cap = cv2.VideoCapture(str(drv["video"]))
    bar = progress(len(times), "exporting frames for Colab")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, fr = cap.read()
            if ok:
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(crop(fr, hood), cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, 92])
                z.writestr(f"{t:08.1f}.jpg", buf.tobytes())
            bar.update(1)
    bar.close()
    if dense:                                    # the Colab CLI route only (run_colab)
        print(f"→ {zpath} ({zpath.stat().st_size / 1e6:.0f} MB, {len(times)} frames)")
        return
    code = remote_code()
    nb = notebook(drv["key"], zpath.name, code)
    (out / "scene_colab.ipynb").write_text(json.dumps(nb, indent=1))
    print(f"→ {zpath} ({zpath.stat().st_size / 1e6:.0f} MB, {len(times)} frames)\n→ {out / 'scene_colab.ipynb'}")


def notebook(key: str, zip_name: str, code: str) -> dict:
    md = lambda src: dict(cell_type="markdown", metadata={}, source=src)
    py = lambda src: dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=src)
    cells = [
        md(f"# Street-scene segmentation — drive `{key}`\n\n"
           "1. **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**.\n"
           f"2. When asked, upload `{zip_name}` (from `data/video/processed/{key}/colab/`).\n"
           "3. `scene.parquet` downloads at the end — then on the Mac run:\n"
           f"   `python -m src.video.scene {key} --import ~/Downloads/scene.parquet`"),
        py("!pip -q install 'transformers==4.46.3'\nimport torch\nprint('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE — switch the runtime to GPU!')"),
        py(f"from google.colab import files\nimport zipfile, os\nup = files.upload()\n"
           f"zipfile.ZipFile('{zip_name}').extractall('frames')\nprint(len(os.listdir('frames')), 'frames')"),
        py(code),
        py("import os, cv2, pandas as pd, torch\nfrom tqdm.auto import tqdm\n"
           "from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation\n"
           "M = 'facebook/mask2former-swin-large-mapillary-vistas-semantic'\n"
           "proc = AutoImageProcessor.from_pretrained(M)\n"
           "model = Mask2FormerForUniversalSegmentation.from_pretrained(M).eval().cuda().half()\n"
           "id2label = model.config.id2label\n"
           "names = np.array([CLASSES.get(id2label[i], 'other') for i in range(len(id2label))])\n"
           "rows = []\n"
           "for f in tqdm(sorted(os.listdir('frames'))):\n"
           "    img = cv2.cvtColor(cv2.imread('frames/' + f), cv2.COLOR_BGR2RGB)\n"
           "    inp = {k: v.cuda().half() if v.dtype.is_floating_point else v.cuda() for k, v in proc(images=img, return_tensors='pt').items()}\n"
           "    with torch.no_grad():\n"
           "        out = model(**inp)\n"
           "    out.class_queries_logits = out.class_queries_logits.float(); out.masks_queries_logits = out.masks_queries_logits.float()\n"
           "    seg = proc.post_process_semantic_segmentation(out, target_sizes=[img.shape[:2]])[0].cpu().numpy()\n"
           "    rows.append(describe(seg, names, float(f[:-4])))\n"
           "pd.DataFrame(rows).to_parquet('scene.parquet', index=False)\nprint(len(rows), 'frames done')"),
        py("files.download('scene.parquet')"),
    ]
    return dict(cells=cells, metadata=dict(accelerator="GPU", kernelspec=dict(name="python3", display_name="Python 3"),
                                           colab=dict(provenance=[])), nbformat=4, nbformat_minor=0)


COLAB = ROOT / ".venv-colab" / "bin" / "colab"     # official Google Colab CLI (own venv, Python ≥3.12)


def remote_code(dense: bool = False) -> str:
    """The constants and describe function shipped to Colab (`describe` = what is stored per frame)."""
    import inspect
    return "\n\n".join([
        "import json, zlib\nimport numpy as np",
        f"WIDTH = {WIDTH}\nSCANLINES = {SCANLINES!r}\nCLASSES = {json.dumps(CLASSES, indent=4)}\n"
        f"MASK_STEP = {MASK_STEP}\nCODES = {CODES!r}\nDENSE_STEP = {DENSE_STEP}\nDENSE_TOP = {DENSE_TOP}",
        inspect.getsource(runs), inspect.getsource(describe), inspect.getsource(describe_dense),
        "describe = describe_dense" if dense else ""])


def remote_script(code: str) -> str:
    """Self-contained segmentation job for a Colab GPU VM (frames already uploaded)."""
    return code + '''

import os, time, zipfile, cv2, pandas as pd, torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
with open("/content/frames.zip", "wb") as fz:                       # reassemble the uploaded parts
    for part in sorted(f for f in os.listdir("/content") if f.startswith("part_")):
        fz.write(open("/content/" + part, "rb").read())
zipfile.ZipFile("/content/frames.zip").extractall("/content/frames")
files = sorted(os.listdir("/content/frames"))
M = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
proc = AutoImageProcessor.from_pretrained(M)
model = Mask2FormerForUniversalSegmentation.from_pretrained(M).eval().cuda().half()
id2label = model.config.id2label
names = np.array([CLASSES.get(id2label[i], "other") for i in range(len(id2label))])
print("GPU:", torch.cuda.get_device_name(0), "|", len(files), "frames", flush=True)
rows, t0 = [], time.time()
for n, f in enumerate(files):
    img = cv2.cvtColor(cv2.imread("/content/frames/" + f), cv2.COLOR_BGR2RGB)
    inp = {k: v.cuda().half() if v.dtype.is_floating_point else v.cuda()
           for k, v in proc(images=img, return_tensors="pt").items()}
    with torch.no_grad():
        out = model(**inp)
    out.class_queries_logits = out.class_queries_logits.float()
    out.masks_queries_logits = out.masks_queries_logits.float()
    seg = proc.post_process_semantic_segmentation(out, target_sizes=[img.shape[:2]])[0].cpu().numpy()
    rows.append(describe(seg, names, float(f[:-4])))
    if (n + 1) % 50 == 0 or n + 1 == len(files):
        el = time.time() - t0
        print(f"  {n + 1}/{len(files)} frames  {el / (n + 1):.2f} s/frame  ~{el / (n + 1) * (len(files) - n - 1) / 60:.1f} min left", flush=True)
pd.DataFrame(rows).to_parquet("/content/scene.parquet", index=False)
print("done:", len(rows), "frames in", round(time.time() - t0), "s", flush=True)
'''


CHUNK_MB = 25          # Colab's file API: 30 MB uploads work, 99 MB → HTTP 500; no sub-folders


def split(path: Path, folder: Path, mb: int) -> list[Path]:
    folder.mkdir(exist_ok=True)
    for old in folder.glob("part_*"):
        old.unlink()
    parts, data = [], path.read_bytes()
    for i in range(0, len(data), mb * 1_000_000):
        part = folder / f"part_{i // (mb * 1_000_000):03d}"
        part.write_bytes(data[i:i + mb * 1_000_000])
        parts.append(part)
    return parts


def run_colab(drv: dict, dense: bool = False):
    """Segment on a free Colab T4 via the Colab CLI: upload frames → run → download → release the GPU."""
    import shlex
    out = drv["folder"] / "colab"
    zpath = out / f"{drv['key']}_frames{'_dense' if dense else ''}.zip"
    if not zpath.exists():
        export_colab(drv, dense)
    job = out / "remote_scene.py"
    job.write_text(remote_script(remote_code(dense)))
    result = out / ("scene_dense.colab.parquet" if dense else "scene.colab.parquet")
    session = ("dense-" if dense else "scene-") + drv["key"].replace("_", "-").lower()

    def colab(*args, check=True):
        cmd = [str(COLAB), "--auth", "oauth2", *args]
        print("$ colab " + " ".join(shlex.quote(a) for a in args), flush=True)
        return subprocess.run(cmd, check=check).returncode

    t0 = time.time()
    try:
        print("1/5 starting a T4 GPU runtime (first time: sign in to Google when asked)…", flush=True)
        colab("new", "-s", session, "--gpu", "T4")
        # Colab's file API rejects large single uploads and missing folders (HTTP 500) → parts in /content
        parts = split(zpath, out / "parts", CHUNK_MB)
        print(f"2/5 uploading {zpath.stat().st_size / 1e6:.0f} MB of frames in {len(parts)} parts…", flush=True)
        for n, part in enumerate(parts, 1):
            print(f"    part {n}/{len(parts)}", flush=True)
            colab("upload", "-s", session, str(part), f"/content/{part.name}")
        print("3/5 installing transformers on the VM…", flush=True)
        colab("install", "-s", session, "transformers==4.46.3")
        print("4/5 segmenting on the GPU…", flush=True)
        if colab("exec", "-s", session, "-f", str(job), check=False):
            # the CLI can time out waiting for the kernel's final reply although the job
            # finished (seen after 'done: 887 frames') → still try to fetch the result
            print("   (exec reported an error — checking whether the result exists anyway)", flush=True)
        print("5/5 downloading the result…", flush=True)
        colab("download", "-s", session, "/content/scene.parquet", str(result))
    finally:
        print("releasing the GPU runtime…", flush=True)
        subprocess.run([str(COLAB), "--auth", "oauth2", "stop", "-s", session])
    if dense:
        df = pd.read_parquet(result).sort_values("t")
        df.to_parquet(drv["folder"] / "scene_dense.parquet", index=False)
        print(f"→ {drv['folder'] / 'scene_dense.parquet'}: {len(df)} frames", flush=True)
    else:
        import_colab(drv, result)
    print(f"Colab segmentation done in {(time.time() - t0) / 60:.1f} min", flush=True)


def import_colab(drv: dict, path: Path):
    df = pd.read_parquet(path)
    need = {"t", "crosswalk", "crosswalk_bottom", "guard_rail", "scanlines"}
    if not need <= set(df.columns):
        raise ValueError(f"{path} is not a scene result (missing {need - set(df.columns)})")
    save(drv, df)


def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download("facebook/mask2former-swin-large-mapillary-vistas-semantic", local_dir=MODEL,
                      allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"])
    print(f"→ {MODEL}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml, or 'download-model'")
    ap.add_argument("--export-colab", action="store_true", help="write frames + notebook for Google Colab")
    ap.add_argument("--import", dest="imp", type=Path, help="scene.parquet downloaded from Colab")
    ap.add_argument("--colab", action="store_true", help="run on a free Colab T4 GPU via the Colab CLI")
    ap.add_argument("--dense", action="store_true", help="with --colab: 2 frames/s class maps for the spatial view")
    a = ap.parse_args()
    if a.drive == "download-model":
        return download_model()
    drv = drives.load(a.drive)
    if a.colab:
        run_colab(drv, a.dense)
    elif a.export_colab:
        export_colab(drv)
    elif a.imp:
        import_colab(drv, a.imp.expanduser())
    else:
        run(drv)


if __name__ == "__main__":
    main()
