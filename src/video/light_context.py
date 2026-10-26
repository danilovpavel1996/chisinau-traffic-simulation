"""
light_context.py
----------------
Traffic lights with context: which landmarks (world.lights) are real lights, which one is ours,
and what each showed over time.

    verify    more than half the landmarks were not traffic lights (2026_10_24_17_08, 24 checked by
              eye: 10 real; road signs, shop signs, lit displays, poles). Each landmark's three largest
              views are scored by CLIP (openai/clip-vit-large-patch14, zero-shot) against "a traffic
              light / one seen from behind / a pedestrian light" vs the look-alikes: kept 9 of 10 real,
              3 of 14 false (threshold 0.3). Runs on Kaggle (GPU), else Colab, else here
    map       OpenStreetMap's signals (junctions and signalised crossings) confirm a light nearby;
              their absence proves nothing — OSM marks 6 signalised junctions on that route, the
              video found real lights at many more
    role      ours: facing us (its colour seen), at vehicle-light height, over or beside our road
              when we pass it; pedestrian: low (< PED_MAX_M); others: the rest (other directions)
    memory    each light's colour over time (red / yellow / green), a state held up to HOLD_S

Usage:
    .venv-video/bin/python -m src.video.light_context <drive>

Output: data/video/processed/<drive>/lights.parquet (id, x, z, y, n, t_first, t_last, score, real,
        map_signal_m, facing, role), light_states.parquet (id, t, colour)
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import pandas as pd

from src.video import drives
from src.video.drives import ROOT

CLIP = "openai/clip-vit-large-patch14"
LIGHT = ["a traffic light", "a traffic light seen from behind", "a pedestrian traffic light"]
OTHER = ["a blue road sign", "a yellow diamond priority road sign", "a pedestrian crossing road sign",
         "a shop sign or logo", "an illuminated advertisement screen", "a building facade", "a utility pole",
         "the tail lights of a car", "trees and sky"]
REAL_MIN = 0.3
CROPS = 3
MAP_NEAR_M = 40.0
PED_MAX_M = 2.8
OURS_X_M = 10.0
FACING_MIN = 0.2             # share of its detections with a colour seen
HOLD_S = 3.0

SCORE_JOB = """
import glob, os, json, numpy as np, pandas as pd, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
inp = os.path.dirname(glob.glob("/kaggle/input/**/crops.parquet", recursive=True)[0])
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = CLIPModel.from_pretrained("{model}").eval().to(dev); p = CLIPProcessor.from_pretrained("{model}")
texts = {texts}
crops = pd.read_parquet(os.path.join(inp, "crops.parquet"))
out = []
for i in range(0, len(crops), 32):
    b = crops.iloc[i:i + 32]
    imgs = [Image.open(os.path.join(inp, f)).convert("RGB") for f in b.file]
    with torch.no_grad():
        pr = m(**p(text=texts, images=imgs, return_tensors="pt", padding=True).to(dev)).logits_per_image.softmax(1).cpu().numpy()
    out += pr[:, :{n_light}].sum(1).tolist()
crops["p_light"] = out
os.makedirs("/kaggle/working/out", exist_ok=True)
crops.to_parquet("/kaggle/working/out/scores.parquet", index=False)
print("scored", len(crops))
"""


def cut_crops(drv: dict, L: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """The CROPS largest views of every landmark, as image files in light_check/."""
    det = pd.read_parquet(drv["folder"] / "detections.parquet")
    det = det[det.cls == "traffic_light"]
    out = drv["folder"] / "light_check"
    out.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(drv["video"]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows = []
    for m in L.itertuples():
        o = obs[obs.mark == m.id].merge(det[["t", "conf", "x1", "y1", "x2", "y2"]], on=["t", "conf"])
        o = o.assign(area=(o.x2 - o.x1) * (o.y2 - o.y1)).sort_values("area", ascending=False).head(CROPS)
        for k, r in enumerate(o.itertuples()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(r.t * fps)))
            ok, f = cap.read()
            if not ok:
                continue
            pad = int(0.5 * max(r.x2 - r.x1, r.y2 - r.y1)) + 6
            c = f[max(r.y1 - pad, 0):r.y2 + pad, max(r.x1 - pad, 0):r.x2 + pad]
            name = f"lm{m.id:04d}_{k}.jpg"
            cv2.imwrite(str(out / name), c)
            rows.append(dict(mark=m.id, file=name))
    crops = pd.DataFrame(rows)
    crops.to_parquet(out / "crops.parquet", index=False)
    return crops


def score(drv: dict, crops: pd.DataFrame) -> pd.Series:
    """P(traffic light) per landmark (median over its crops), CLIP on Kaggle."""
    from src.video import kaggle_run
    import zipfile
    folder = drv["folder"] / "light_check"
    pack = folder / "crops.zip"                      # one upload instead of hundreds (Kaggle unpacks it)
    with zipfile.ZipFile(pack, "w", zipfile.ZIP_STORED) as z:
        for f in list(crops.file) + ["crops.parquet"]:
            z.write(folder / f, f)
    files = {"crops.zip": pack}
    ds = kaggle_run.push_dataset(f"chisinau light crops {drv['key']}", files, drv["folder"] / "colab" / "kaggle_crops")
    script = SCORE_JOB.format(model=CLIP, texts=json.dumps(LIGHT + OTHER), n_light=len(LIGHT))
    out = kaggle_run.run_kernel(f"chisinau light check {drv['key']}", script, [ds], folder / "kaggle_out",
                                drv["folder"] / "colab" / "kaggle_kernel_lights")
    s = pd.read_parquet(out / "out" / "scores.parquet")
    return s.groupby("mark").p_light.median()


def context(drv: dict, L: pd.DataFrame, obs: pd.DataFrame, pose, scores: pd.Series) -> pd.DataFrame:
    from scipy.spatial import cKDTree
    from src.video import map_layer
    L = L.copy()
    L["score"] = L.id.map(scores).fillna(0.0)
    L["real"] = L.score >= REAL_MIN
    col = obs[obs.mark >= 0].groupby("mark").light.agg(lambda s: np.mean(s.isin(["red", "yellow", "green"])))
    L["facing"] = L.id.map(col).fillna(0) >= FACING_MIN
    # sideways from our road when we pass it
    xs = []
    for m in L.itertuples():
        ts = np.arange(m.t_first, m.t_last + 20, 0.2)
        X, Z = pose.to_car(ts, m.x, m.z)
        k = np.flatnonzero(Z <= 0)
        xs.append(float(X[k[0]]) if len(k) else float(X[-1]))
    L["x_pass"] = xs
    L["role"] = np.where(L.y < PED_MAX_M, "pedestrian",
                         np.where(L.facing & (np.abs(L.x_pass) <= OURS_X_M), "ours", "other"))
    # map prior: OSM signals near, placed with our map position
    mpf = drv["folder"] / "map_pose.parquet"
    sig = ROOT / "data" / "osm" / "signals_lamps.parquet"
    L["map_signal_m"] = np.nan
    if mpf.exists() and sig.exists():
        mp = pd.read_parquet(mpf)
        S = pd.read_parquet(sig)
        S = S[S.kind != "street_lamp"]
        e, n = map_layer.utm().transform(S.lon.to_numpy(), S.lat.to_numpy())
        tree = cKDTree(np.column_stack([e, n]))
        P = []
        for m in L.itertuples():
            r = mp.iloc[int(np.argmin(np.abs(mp.t - m.t_first)))]
            X, Z = pose.to_car(r.t, m.x, m.z)
            th = np.radians(r.heading)
            P.append((r.E + X * np.cos(th) + Z * np.sin(th), r.N - X * np.sin(th) + Z * np.cos(th)))
        L["map_signal_m"] = tree.query(np.array(P))[0].round(1)
    return L


def states(obs: pd.DataFrame) -> pd.DataFrame:
    """Each light's colour over time: readings, each held for HOLD_S (memory between detections)."""
    o = obs[(obs.mark >= 0) & obs.light.isin(["red", "yellow", "green"])].sort_values("t")
    return o[["mark", "t", "light"]].rename(columns=dict(mark="id", light="colour")).reset_index(drop=True)


def run(drv: dict) -> pd.DataFrame:
    from src.video import world
    from src.video.bev import Geometry
    dense = pd.read_parquet(drv["folder"] / "scene_dense.parquet").sort_values("t")
    pose = world.Pose(drv)
    L, obs = world.lights(drv, Geometry(drv, dense.iloc[0]), pose)
    crops = cut_crops(drv, L, obs)
    sc = score(drv, crops)
    L = context(drv, L, obs, pose, sc)
    L.to_parquet(drv["folder"] / "lights.parquet", index=False)
    states(obs).to_parquet(drv["folder"] / "light_states.parquet", index=False)
    obs[["t", "conf", "mark"]].to_parquet(drv["folder"] / "light_obs.parquet", index=False)   # detection → landmark
    real = L[L.real]
    print(f"→ {drv['folder'] / 'lights.parquet'}: {len(L)} landmarks, {len(real)} real traffic lights "
          f"({(real.role == 'ours').sum()} ours, {(real.role == 'other').sum()} other directions, "
          f"{(real.role == 'pedestrian').sum()} pedestrian); {np.mean(real.map_signal_m < MAP_NEAR_M):.0%} "
          f"of the real ones near an OSM signal", flush=True)
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    run(drives.load(ap.parse_args().drive))


if __name__ == "__main__":
    main()
