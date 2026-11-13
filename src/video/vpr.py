"""
vpr.py
------
Where along the route are we — from what the camera sees, matched against Mapillary's street photos
(visual place recognition). Our position along the road is otherwise only known to ~25 m (heading
profile + odometer, map_layer.pose_on_map); a photo taken at the same spot looks like our frame.

    photos     Mapillary images along the matched route (video only; no GPS): per BIN_M of route, up
               to PER_BIN recent ones facing our direction of travel, no panoramas; 512 px wide
    frames     ours, one per second of the street window, cropped above the bonnet line, 512 px
    prints     a global descriptor per image (EigenPlaces, ResNet-50, 2048-d — trained to recognise a
               place across viewpoints, seasons, cameras), on Kaggle
    sequence   the frame sequence aligned to the route by dynamic programming: each frame votes for
               the route positions whose photos it resembles; steps between frames must agree with our
               odometer — one look-alike street cannot pull the position away (SeqSLAM-style)

Mapillary's photo positions are its contributors' GPS — external map data, not our dashcam's.

Usage:
    .venv-video/bin/python -m src.video.vpr <drive>

Output: data/video/processed/<drive>/vpr.parquet (t, d along the route, similarity, confident)
"""

from __future__ import annotations

import argparse
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.video import drives
from src.video.drives import ROOT

OUT = ROOT / "data" / "mapillary"
CORRIDOR_M = 25.0
BIN_M = 5.0
PER_BIN = 3
FACING_DEG = 60.0            # photo compass within this of our direction of travel
WIDTH = 512
FRAME_EVERY_S = 1.0
SEARCH_M = 200.0             # route positions searched around the current estimate
STEP_SLACK = 0.2             # metres between frames: odometer ± 35 % + JUMP_M
JUMP_M = 6.0
CONF_Z = 4.0               # a frame is confidently placed when its photo there is this unusually similar


def route_of(drv: dict):
    from src.video import map_layer
    rt = json.loads((drv["folder"] / "event_match.json").read_text())["route"]
    e, n = map_layer.utm().transform(rt["lon"], rt["lat"])
    route = np.column_stack([e, n])
    cum = np.r_[0, np.cumsum(np.hypot(*np.diff(route, axis=0).T))]
    return route, cum


def fetch_photos(drv: dict) -> pd.DataFrame:
    """Mapillary photo metadata along the matched route, thinned per bin, and the photos themselves."""
    from scipy.spatial import cKDTree
    from src.video import map_layer, mapillary
    folder = OUT / f"{drv['key']}_photos"
    meta_f = OUT / f"{drv['key']}_photos.parquet"
    if meta_f.exists():
        return pd.read_parquet(meta_f)
    folder.mkdir(parents=True, exist_ok=True)
    route, cum = route_of(drv)
    dd = np.arange(0, cum[-1], 1.0)
    line = np.column_stack([np.interp(dd, cum, route[:, 0]), np.interp(dd, cum, route[:, 1])])
    head = np.degrees(np.arctan2(*np.gradient(line, axis=0).T)) % 360       # route direction (east, north)
    rt = json.loads((drv["folder"] / "event_match.json").read_text())["route"]
    path = np.column_stack([rt["lat"], rt["lon"]])
    s = mapillary._session()
    rows = []
    for b in mapillary.tiles(path):
        for it in mapillary._get_box(s, "images", b, "id,captured_at,computed_geometry,computed_compass_angle,is_pano,thumb_1024_url"):
            g = it.get("computed_geometry")
            if not g or it.get("is_pano") or it.get("computed_compass_angle") is None or not it.get("thumb_1024_url"):
                continue
            rows.append(dict(id=it["id"], lon=g["coordinates"][0], lat=g["coordinates"][1], compass=it["computed_compass_angle"],
                             captured=it.get("captured_at"), url=it["thumb_1024_url"]))
    P = pd.DataFrame(rows).drop_duplicates("id")
    e, n = map_layer.utm().transform(P.lon.to_numpy(), P.lat.to_numpy())
    side, idx = cKDTree(line).query(np.column_stack([e, n]))
    P = P.assign(E=e, N=n, side=side, d=dd[idx], route_head=head[idx])
    P["facing"] = np.abs((P.compass - P.route_head + 180) % 360 - 180)
    P = P[(P.side <= CORRIDOR_M) & (P.facing <= FACING_DEG)]
    P["bin"] = (P.d // BIN_M).astype(int)
    P = P.sort_values("captured", ascending=False).groupby("bin").head(PER_BIN).reset_index(drop=True)
    print(f"{len(P)} Mapillary photos to download along {cum[-1] / 1000:.1f} km ({P['bin'].nunique()} of "
          f"{int(cum[-1] // BIN_M)} route bins have one)", flush=True)

    def get(r):
        f = folder / f"{r.id}.jpg"
        if f.exists():
            return True
        for _ in range(3):
            try:
                resp = s.get(r.url, timeout=60)        # a signed URL: the token header is not needed but harmless
                if resp.ok:
                    img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        h = int(img.shape[0] * WIDTH / img.shape[1])
                        cv2.imwrite(str(f), cv2.resize(img, (WIDTH, h), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
                        return True
            except Exception:
                time.sleep(2)
        return False

    t0, ok = time.time(), []
    with ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(get, P.itertuples())):
            ok.append(r)
            if (i + 1) % 500 == 0:
                print(f"   {i + 1}/{len(P)} photos ({(time.time() - t0) / (i + 1) * (len(P) - i - 1) / 60:.1f} min left)", flush=True)
    P = P[np.array(ok)].drop(columns=["url"]).reset_index(drop=True)
    P.to_parquet(meta_f, index=False)
    print(f"→ {meta_f}: {len(P)} photos, {sum(f.stat().st_size for f in folder.glob('*.jpg')) / 1e6:.0f} MB", flush=True)
    return P


def our_frames(drv: dict) -> pd.DataFrame:
    folder = drv["folder"] / "vpr"
    meta_f = folder / "frames.parquet"
    if meta_f.exists():
        return pd.read_parquet(meta_f)
    folder.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(drv["video"]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    t0, t1 = drv["street_window_s"]
    rows = []
    for t in np.arange(np.ceil(t0), t1, FRAME_EVERY_S):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, f = cap.read()
        if not ok:
            continue
        f = f[:int(f.shape[0] * drv["camera"]["bonnet"])]
        h = int(f.shape[0] * WIDTH / f.shape[1])
        name = f"f{int(t):05d}.jpg"
        cv2.imwrite(str(folder / name), cv2.resize(f, (WIDTH, h), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
        rows.append(dict(t=float(t), file=name))
    F = pd.DataFrame(rows)
    F.to_parquet(meta_f, index=False)
    return F


PRINT_JOB = """
import glob, os, zipfile, numpy as np, torch
from PIL import Image
import torchvision.transforms as T
zipfile.ZipFile("/content/photos.zip").extractall("/content/photos")
zipfile.ZipFile("/content/frames.zip").extractall("/content/frames")
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(torch.hub.get_dir(), exist_ok=True)          # EigenPlaces loads CosPlace's code too: trust both, no prompt
with open(os.path.join(torch.hub.get_dir(), "trusted_list"), "a") as fh:
    fh.write("gmberton_eigenplaces\\ngmberton_cosplace\\n")
model = torch.hub.load("gmberton/eigenplaces", "get_trained_model", backbone="ResNet50", fc_output_dim=2048,
                       trust_repo=True).eval().to(dev)
tf = T.Compose([T.Resize((320, 512)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
for name in ("photos", "frames"):
    files = sorted(glob.glob(f"/content/{{name}}/*.jpg"))
    out = []
    for i in range(0, len(files), 64):
        x = torch.stack([tf(Image.open(f).convert("RGB")) for f in files[i:i + 64]]).to(dev)
        with torch.no_grad():
            out.append(torch.nn.functional.normalize(model(x), dim=1).cpu().numpy().astype(np.float16))
    np.save(f"/content/{{name}}.npy", np.concatenate(out))
    with open(f"/content/{{name}}.txt", "w") as fh:
        fh.write("\\n".join(os.path.basename(f) for f in files))
    print(name, len(files), flush=True)
"""


def prints(drv: dict, photos: pd.DataFrame, frames: pd.DataFrame) -> tuple[dict, dict]:
    """EigenPlaces descriptors of the photos and our frames (Kaggle): {file: vector}."""
    import zipfile
    from src.video import gpu
    work = drv["folder"] / "vpr"
    done = work / "photos.npy"
    if not done.exists():
        pz, fz = work / "photos.zip", work / "frames.zip"
        with zipfile.ZipFile(pz, "w", zipfile.ZIP_STORED) as z:
            for i in photos.id:
                z.write(OUT / f"{drv['key']}_photos" / f"{i}.jpg", f"{i}.jpg")
        with zipfile.ZipFile(fz, "w", zipfile.ZIP_STORED) as z:
            for f in frames.file:
                z.write(work / f, f)
        gpu.run_job(f"vpr {drv['key']}", PRINT_JOB.format(), {"/content/photos.zip": pz, "/content/frames.zip": fz},
                           {f"/content/{n}.{x}": work / f"{n}.{x}" for n in ("photos", "frames") for x in ("npy", "txt")},
                           work / "kaggle")
    out = []
    for n in ("photos", "frames"):
        v = np.load(work / f"{n}.npy").astype(np.float32)
        names = (work / f"{n}.txt").read_text().split("\n")
        out.append(dict(zip(names, v)))
    return out[0], out[1]


def align(drv: dict, photos: pd.DataFrame, frames: pd.DataFrame, P: dict, Fv: dict, prior: pd.DataFrame) -> pd.DataFrame:
    """Dynamic programming over route bins: each frame's similarity to each bin (best photo in it), steps
    between consecutive frames bounded by the odometer, searched around the prior estimate."""
    from src.video.world import Pose
    route, cum = route_of(drv)
    nb = int(cum[-1] // BIN_M) + 1
    bins = photos.d.to_numpy() // BIN_M
    pv = np.stack([P[f"{i}.jpg"] for i in photos.id])
    frames = frames[frames.file.isin(Fv)].reset_index(drop=True)
    fv = np.stack([Fv[f] for f in frames.file])
    sim_img = fv @ pv.T                                               # frames × photos (cosine)
    S = np.full((len(frames), nb), np.nan, np.float32)
    for b in np.unique(bins):
        S[:, int(b)] = sim_img[:, bins == b].max(1)
    pose = Pose(drv)
    px, pz, _ = pose.at(frames.t.to_numpy())
    step = np.r_[0, np.hypot(np.diff(px), np.diff(pz))]
    d0 = np.interp(frames.t, prior.t, prior.d)
    n = len(frames)
    # per frame: how much more similar than its usual (z-score against the frame's median, MAD scale) —
    # a bin without a photo is neutral (0). (1 − similarity with 0.3 for empty bins made the path prefer
    # empty bins: a good match, 0.54, cost 0.46)
    med = np.median(sim_img, axis=1, keepdims=True)
    mad = np.median(np.abs(sim_img - med), axis=1, keepdims=True) * 1.4826 + 1e-3
    Z = (S - med) / mad
    cost = np.where(np.isnan(Z), 0.0, -np.clip(Z, -2, 8))
    D = np.full((n, nb), np.inf, np.float32)
    back = np.zeros((n, nb), np.int32)
    lo0, hi0 = int(max(0, (d0[0] - SEARCH_M) // BIN_M)), int(min(nb, (d0[0] + SEARCH_M) // BIN_M + 1))
    D[0, lo0:hi0] = cost[0, lo0:hi0]
    for i in range(1, n):
        lo, hi = int(max(0, (d0[i] - SEARCH_M) // BIN_M)), int(min(nb, (d0[i] + SEARCH_M) // BIN_M + 1))
        smin = max(0.0, step[i] * (1 - STEP_SLACK) - JUMP_M) / BIN_M
        smax = (step[i] * (1 + STEP_SLACK) + JUMP_M) / BIN_M
        a, b = int(np.floor(smin)), int(np.ceil(smax))
        prev = D[i - 1]
        for j in range(lo, hi):
            k0, k1 = max(0, j - b), max(0, j - a + 1)
            if k1 <= k0:
                continue
            k = k0 + int(np.argmin(prev[k0:k1]))
            if np.isfinite(prev[k]):
                D[i, j] = prev[k] + cost[i, j]
                back[i, j] = k
    j = int(np.argmin(D[-1]))
    path = np.empty(n, int)
    for i in range(n - 1, -1, -1):
        path[i] = j
        j = back[i, j]
    best = np.nanmax(np.where(np.isnan(S), -1, S), axis=1)
    on = np.array([S[i, path[i]] if not np.isnan(S[i, path[i]]) else np.nan for i in range(n)])
    z_on = np.array([Z[i, path[i]] if not np.isnan(Z[i, path[i]]) else np.nan for i in range(n)])
    return pd.DataFrame(dict(t=frames.t, d=path * BIN_M + BIN_M / 2, sim=on, best=best, z=z_on,
                             confident=z_on >= CONF_Z))


# ── geometric verification (SuperPoint + LightGlue, RANSAC) ────────────────
TOP_K = 5                    # candidate photos per frame (most similar fingerprints) …
VERIFY_M = 120.0             # … within this of the photo alignment's position
MIN_INLIERS = 40             # a frame-photo pair with at least this many consistent point matches is the same place

VERIFY_JOB = """
import json, os, subprocess, sys, zipfile, numpy as np, pandas as pd, torch, cv2
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/cvg/LightGlue.git"], check=True)
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
zipfile.ZipFile("/content/photos.zip").extractall("/content/photos")
zipfile.ZipFile("/content/frames.zip").extractall("/content/frames")
pairs = pd.read_parquet("/content/pairs.parquet")
dev = "cuda" if torch.cuda.is_available() else "cpu"
ext = SuperPoint(max_num_keypoints=1024).eval().to(dev)
mat = LightGlue(features="superpoint").eval().to(dev)
cache = {{}}
def feats(path):
    if path not in cache:
        with torch.no_grad():
            cache[path] = ext.extract(load_image(path).to(dev))
        if len(cache) > 4000:
            cache.pop(next(iter(cache)))
    return cache[path]
inl = []
for i, r in enumerate(pairs.itertuples()):
    a, b = feats("/content/frames/" + r.frame), feats("/content/photos/" + r.photo)
    with torch.no_grad():
        m = rbd(mat({{"image0": a, "image1": b}}))
    k = m["matches"].cpu().numpy()
    n = 0
    if len(k) >= 8:
        p0 = rbd(a)["keypoints"].cpu().numpy()[k[:, 0]]; p1 = rbd(b)["keypoints"].cpu().numpy()[k[:, 1]]
        try:                                          # degenerate point sets make OpenCV's RANSAC assert
            F, mask = cv2.findFundamentalMat(p0, p1, cv2.USAC_MAGSAC, 1.5, 0.999, 10000)
            n = int(mask.sum()) if mask is not None else 0
        except cv2.error:
            n = 0
    inl.append(n)
    if (i + 1) % 500 == 0:
        print(i + 1, "pairs", flush=True)
pairs["inliers"] = inl
pairs.to_parquet("/content/verified.parquet", index=False)
print("done", len(pairs), flush=True)
"""


def verify(drv: dict, photos: pd.DataFrame, frames: pd.DataFrame, P: dict, Fv: dict, out: pd.DataFrame) -> pd.DataFrame:
    """Per frame, the TOP_K most similar photos near the aligned position, checked by point matching
    on Kaggle; returns the pairs with their inlier counts."""
    from src.video import gpu
    work = drv["folder"] / "vpr"
    vf = work / "verified.parquet"
    if vf.exists():
        return pd.read_parquet(vf)
    pv = np.stack([P[f"{i}.jpg"] for i in photos.id])
    pd_ = photos.d.to_numpy()
    rows = []
    for r in out.itertuples():
        f = frames.file[frames.t == r.t]
        if not len(f) or f.iat[0] not in Fv:
            continue
        near = np.flatnonzero(np.abs(pd_ - r.d) <= VERIFY_M)
        if not len(near):
            continue
        sim = pv[near] @ Fv[f.iat[0]]
        for j in near[np.argsort(-sim)[:TOP_K]]:
            rows.append(dict(t=r.t, frame=f.iat[0], photo=f"{photos.id.iat[j]}.jpg", d=float(pd_[j]), sim=float(pv[j] @ Fv[f.iat[0]])))
    pairs = pd.DataFrame(rows)
    pf = work / "pairs.parquet"
    pairs.to_parquet(pf, index=False)
    gpu.run_job(f"verify {drv['key']}", VERIFY_JOB.format(),
                       {"/content/photos.zip": work / "photos.zip", "/content/frames.zip": work / "frames.zip",
                        "/content/pairs.parquet": pf},
                       {"/content/verified.parquet": vf}, work / "kaggle_verify")
    return pd.read_parquet(vf)


def verified_positions(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per frame, the photo with the most consistent point matches, if it has at least MIN_INLIERS."""
    best = pairs.sort_values("inliers", ascending=False).groupby("t").head(1).sort_values("t")
    return best[best.inliers >= MIN_INLIERS].rename(columns={"d": "d_photo"}).reset_index(drop=True)


def run(drv: dict) -> pd.DataFrame:
    from src.video import map_layer
    photos = fetch_photos(drv)
    frames = our_frames(drv)
    P, Fv = prints(drv, photos, frames)
    prior = map_layer.pose_on_map(drv)
    out = align(drv, photos, frames, P, Fv, prior)
    ver = verified_positions(verify(drv, photos, frames, P, Fv, out))
    out = out.merge(ver[["t", "d_photo", "inliers"]], on="t", how="left")
    out["verified"] = out.d_photo.notna()
    out.to_parquet(drv["folder"] / "vpr.parquet", index=False)
    msg = f"→ {drv['folder'] / 'vpr.parquet'}: {len(out)} frames, {out.confident.mean():.0%} confident"
    if drives.gps_track(drv) is not None:
        msg += "; " + evaluate(drv, out, prior)
    print(msg, flush=True)
    return out


def evaluate(drv: dict, out: pd.DataFrame, prior: pd.DataFrame) -> str:
    """Along-route error vs the GPS answer key, for the photo alignment and the prior."""
    from scipy.spatial import cKDTree
    from src.video import map_layer
    g = drives.gps_track(drv).sort_values("video_s")
    route, cum = route_of(drv)
    dd = np.arange(0, cum[-1], 1.0)
    line = np.column_stack([np.interp(dd, cum, route[:, 0]), np.interp(dd, cum, route[:, 1])])
    ge, gn = map_layer.utm().transform(np.interp(out.t, g.video_s, g.lon), np.interp(out.t, g.video_s, g.lat))
    true = dd[cKDTree(line).query(np.column_stack([ge, gn]))[1]]
    e_v = np.abs(out.d - true)
    e_p = np.abs(np.interp(out.t, prior.t, prior.d) - true)
    c = out.confident.to_numpy()
    msg = (f"along the route: photos median {np.median(e_v):.0f} m (≤10 m {np.mean(e_v <= 10):.0%}), "
           f"confident ones {np.median(e_v[c]) if c.any() else float('nan'):.0f} m; prior {np.median(e_p):.0f} m (≤10 m {np.mean(e_p <= 10):.0%})")
    if "verified" in out and out.verified.any():
        v = out.verified.to_numpy()
        e_g = np.abs(out.d_photo.to_numpy()[v] - true[v])
        msg += f"; verified photos ({v.mean():.0%} of frames) {np.median(e_g):.0f} m (≤10 m {np.mean(e_g <= 10):.0%})"
    return msg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    run(drives.load(ap.parse_args().drive))


if __name__ == "__main__":
    main()
