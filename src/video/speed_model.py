"""
speed_model.py
--------------
A learned correction for the video's own speed (speed.py), trained on drives whose dashcam
overlay shows the GPS speed — the answer key — and applied to any video, with or without one.

speed.py measures speed from tracked points and their depth; how right that is depends on the
road: trees beside a downhill street read too fast, a bridge with only distant buildings and
raindrops reads too slow (2026_10_23_10_06). So per analysed frame the model sees what the
measurement was made of — per kind of scene point (trees, buildings, the road's surface and
markings, poles and signs, barriers, the upper image) their median speed and share, how far
they are, how much they disagree — and learns the true speed from it. Gradient-boosted trees
(scikit-learn), fitted to GPS speed only; GPS positions are never used.

Scored leave-one-drive-out: each drive is predicted by a model that never saw it.

Usage:
    .venv-video/bin/python -m src.video.speed_model train <drive> <drive> …   # needs motion.parquet + overlay
    .venv-video/bin/python -m src.video.speed_model apply <drive>             # → speed.parquet kmh_model

Output: models/speed_model.joblib; speed.parquet column kmh_model
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.video import drives
from src.video.drives import ROOT
from src.video.scene import CODES

MODEL = ROOT / "models" / "speed_model.joblib"
GROUPS = {
    "ground": ("road", "lane_marking", "crosswalk", "curb", "curb_cut", "sidewalk", "bike_lane"),
    "trees": ("vegetation",),
    "buildings": ("building",),
    "vertical": ("pole", "street_light", "sign_front", "traffic_light"),
    "barriers": ("barrier", "guard_rail", "bridge"),
    "other": ("other", "sky", "car", "bus", "truck", "person"),
}
UPPER = 255                       # points above the segmented part of the image
SMOOTH_S = 2.0
MIN_GPS_KMH = 0.0


def features(drv: dict) -> pd.DataFrame:
    """Per analysed frame: the measured speed and what it was measured on."""
    from src.video import speed
    pts: list = []
    sp = speed.raw_speed(drv, pts)
    P = pd.concat(pts, ignore_index=True)
    group_of = np.full(256, "other", dtype=object)
    for g, names in GROUPS.items():
        for n in names:
            group_of[CODES.index(n)] = g
    group_of[UPPER] = "upper"
    P["group"] = group_of[P.cls.to_numpy()]
    by_t = P.groupby("t")
    f = pd.DataFrame(dict(
        med=by_t.kmh.median(),
        q25=by_t.kmh.quantile(0.25),
        q75=by_t.kmh.quantile(0.75),
        n=by_t.size(),
        z_med=by_t.Z.median(),
        x_abs=by_t.x.apply(lambda x: float(np.median(np.abs(x - 480)))),
    ))
    for g in list(GROUPS) + ["upper"]:
        q = P[P.group == g].groupby("t")
        f[f"{g}_med"] = q.kmh.median()
        f[f"{g}_share"] = (q.size() / f.n).fillna(0)
    f = f.reset_index()
    f = f.merge(sp[["t", "kmh_raw", "yaw_dps", "depth_k"]], on="t", how="left")
    f["yaw_abs"] = f.yaw_dps.abs()
    win = max(1, int(round(SMOOTH_S * 5)))
    cols = [c for c in f.columns if c != "t"]
    f[cols] = f[cols].rolling(win, center=True, min_periods=1).median()
    f["drive"] = drv["key"]
    return f


def with_truth(drv: dict, f: pd.DataFrame) -> pd.DataFrame:
    gps = drives.gps_track(drv)
    if gps is None:
        raise SystemExit(f"{drv['key']}: no speed overlay — cannot train on it")
    gps = gps.sort_values("video_s")
    f = f.copy()
    f["gps"] = np.interp(f.t, gps.video_s, gps.speed_kmh)
    # only where the overlay was read around this moment (no long interpolation gaps)
    near = np.abs(gps.video_s.to_numpy()[np.clip(np.searchsorted(gps.video_s, f.t), 0, len(gps) - 1)] - f.t) < 2
    return f[near & (f.gps >= MIN_GPS_KMH)]


def X(f: pd.DataFrame) -> pd.DataFrame:
    return f.drop(columns=[c for c in ("t", "drive", "gps", "yaw_dps") if c in f.columns])


def make_model():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=80, l2_regularization=1.0, loss="absolute_error")


def score(pred, f) -> str:
    e = np.abs(pred - f.gps)
    return (f"correlation {np.corrcoef(pred, f.gps)[0, 1]:.2f}, median error {np.median(e):.1f} km/h, "
            f"90 % within {np.percentile(e, 90):.1f}")


def train(keys: list[str]):
    import joblib
    data = []
    for k in keys:
        drv = drives.load(k)
        f = with_truth(drv, features(drv))
        print(f"{k}: {len(f)} frames with GPS speed", flush=True)
        data.append(f)
    D = pd.concat(data, ignore_index=True)
    if len(keys) > 1:
        print("leave-one-drive-out (each drive predicted by a model that never saw it):")
        for k in keys:
            tr, te = D[D.drive != k], D[D.drive == k]
            m = make_model().fit(X(tr), tr.gps)
            print(f"  {k}  raw: {score(te.kmh_raw.fillna(0), te)}")
            print(f"  {' ' * len(k)}  model: {score(np.clip(m.predict(X(te)), 0, None), te)}", flush=True)
    m = make_model().fit(X(D), D.gps)
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(dict(model=m, columns=list(X(D).columns), drives=keys), MODEL)
    print(f"→ {MODEL} (trained on {len(D)} frames from {len(keys)} drives)")


def apply(drv: dict) -> pd.DataFrame:
    import joblib
    pack = joblib.load(MODEL)
    f = features(drv)
    pred = np.clip(pack["model"].predict(X(f)[pack["columns"]]), 0, None)
    sp_path = drv["folder"] / "speed.parquet"
    sp = pd.read_parquet(sp_path)
    sp["kmh_model"] = np.interp(sp.t, f.t, pred)
    sp.to_parquet(sp_path, index=False)
    msg = f"→ {sp_path}: kmh_model (model trained on {', '.join(pack['drives'])})"
    if drives.gps_track(drv) is not None:
        ft = with_truth(drv, f.assign(pred=pred))
        t0, t1 = drv.get("street_window_s", [ft.t.min(), ft.t.max()])
        ft = ft[(ft.t >= t0) & (ft.t <= t1)]
        msg += f"; vs GPS: {score(ft.pred, ft)}" + (" (trained on this drive)" if drv["key"] in pack["drives"] else "")
    print(msg, flush=True)
    return sp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["train", "apply"])
    ap.add_argument("drives", nargs="+", help="drive keys in configs/drives.yaml")
    a = ap.parse_args()
    if a.action == "train":
        train(a.drives)
    else:
        for k in a.drives:
            apply(drives.load(k))


if __name__ == "__main__":
    main()
