"""
mapillary.py
------------
Coverage check: does Mapillary (crowd-sourced street photos, with traffic signs, lights and other
points detected in them) have what our videos need along our routes in Chișinău?

Per drive, in a corridor of CORRIDOR_M around the driven path, from Mapillary's public API (v4):
    images     photos taken there: how many, how much of the route has one within 20 m, how recent
    features   detected map features (traffic lights, signs, crosswalk markings, …)
    lights     Mapillary's traffic lights vs OpenStreetMap's signals vs our verified light
               landmarks (light_context.py): which source sees which

The path here comes from the GPS overlay — only to know where to look, for this check; nothing of it
goes into the pipeline. The token is read from ~/.mapillary/token (the owner's; never printed, sent
in a request header, not in the URL).

Usage:
    .venv-video/bin/python -m src.video.mapillary <drive> [<drive> …]

Output: data/mapillary/<drive>_images.parquet, <drive>_features.parquet
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.video import drives
from src.video.drives import ROOT

API = "https://graph.mapillary.com"
TOKEN = Path.home() / ".mapillary" / "token"
OUT = ROOT / "data" / "mapillary"
CORRIDOR_M = 30.0
TILE_DEG = 0.004             # query boxes (the API limits a box's area)
LIGHTS = "object--traffic-light"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["Authorization"] = "OAuth " + TOKEN.read_text().strip()
    return s


def _get(s: requests.Session, path: str, params: dict) -> list[dict]:
    out, url = [], f"{API}/{path}"
    while url:
        for attempt in range(4):
            r = s.get(url, params=params, timeout=60)
            if r.status_code == 429 or (r.status_code >= 500 and attempt < 1):
                time.sleep(5 * (attempt + 1))
                continue
            break
        r.raise_for_status()
        j = r.json()
        out += j.get("data", [])
        url = j.get("paging", {}).get("next")
        params = None                                       # the next link carries them
    return out


def _get_box(s: requests.Session, path: str, b: tuple, fields: str, depth: int = 0) -> list[dict]:
    """All items in a box; a box the server cannot answer (HTTP 500: too much in it) is split in four."""
    try:
        return _get(s, path, dict(bbox=",".join(f"{v:.6f}" for v in b), fields=fields, limit=2000))
    except requests.HTTPError as e:
        if e.response is None or e.response.status_code < 500 or depth >= 4:
            raise
        w, so, ea, n = b
        mx, my = (w + ea) / 2, (so + n) / 2
        return [it for q in ((w, so, mx, my), (mx, so, ea, my), (w, my, mx, n), (mx, my, ea, n))
                for it in _get_box(s, path, q, fields, depth + 1)]


def path_of(drv: dict) -> np.ndarray:
    g = drives.gps_track(drv).sort_values("video_s")
    t0, t1 = drv.get("street_window_s", [g.video_s.min(), g.video_s.max()])
    g = g[(g.video_s >= t0) & (g.video_s <= t1)]
    return g[["lat", "lon"]].to_numpy()


def tiles(path: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Query boxes covering the path (west, south, east, north)."""
    keys = {(int(np.floor(lon / TILE_DEG)), int(np.floor(lat / TILE_DEG))) for lat, lon in path}
    return [(i * TILE_DEG, j * TILE_DEG, (i + 1) * TILE_DEG, (j + 1) * TILE_DEG) for i, j in sorted(keys)]


def fetch(drv: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    OUT.mkdir(parents=True, exist_ok=True)
    s = _session()
    path = path_of(drv)
    imgs, feats = [], []
    for b in tiles(path):
        for it in _get_box(s, "images", b, "id,captured_at,computed_geometry,geometry"):
            geo = it.get("computed_geometry") or it.get("geometry")
            if geo:
                imgs.append(dict(id=it["id"], lon=geo["coordinates"][0], lat=geo["coordinates"][1],
                                 captured=pd.to_datetime(it.get("captured_at"), unit="ms")))
        for it in _get_box(s, "map_features", b, "id,object_value,geometry,last_seen_at"):
            feats.append(dict(id=it["id"], value=it.get("object_value"), lon=it["geometry"]["coordinates"][0],
                              lat=it["geometry"]["coordinates"][1], last_seen=it.get("last_seen_at")))
    I = pd.DataFrame(imgs).drop_duplicates("id") if imgs else pd.DataFrame(columns=["id", "lon", "lat", "captured"])
    F = pd.DataFrame(feats).drop_duplicates("id") if feats else pd.DataFrame(columns=["id", "value", "lon", "lat", "last_seen"])
    I.to_parquet(OUT / f"{drv['key']}_images.parquet", index=False)
    F.to_parquet(OUT / f"{drv['key']}_features.parquet", index=False)
    return I, F


def route_features(drv: dict, kinds: tuple[str, ...] = (LIGHTS,)) -> pd.DataFrame:
    """Mapillary map features along the matched route (video only — event_match), cached per drive:
    id, value, lon, lat. For the pipeline: no GPS involved."""
    cache = OUT / f"{drv['key']}_route_features.parquet"
    if not cache.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        rt = json.loads((drv["folder"] / "event_match.json").read_text())["route"]
        path = np.column_stack([rt["lat"], rt["lon"]])
        d = np.r_[0, np.cumsum(np.hypot(np.diff(path[:, 0]) * 111_320, np.diff(path[:, 1]) * 75_900))]
        dense = np.column_stack([np.interp(np.arange(0, d[-1], 50), d, path[:, 0]), np.interp(np.arange(0, d[-1], 50), d, path[:, 1])])
        s = _session()
        rows = []
        for b in tiles(dense):
            for it in _get_box(s, "map_features", b, "id,object_value,geometry"):
                rows.append(dict(id=it["id"], value=it.get("object_value"), lon=it["geometry"]["coordinates"][0],
                                 lat=it["geometry"]["coordinates"][1]))
        pd.DataFrame(rows, columns=["id", "value", "lon", "lat"]).drop_duplicates("id").to_parquet(cache, index=False)
    F = pd.read_parquet(cache)
    return F[F.value.fillna("").str.startswith(kinds)].reset_index(drop=True)


def report(drv: dict, I: pd.DataFrame, F: pd.DataFrame) -> dict:
    from scipy.spatial import cKDTree
    from src.video import map_layer
    to = map_layer.utm()
    path = path_of(drv)
    pe, pn = to.transform(path[:, 1], path[:, 0])
    P = np.column_stack([pe, pn])
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    samples = P[np.searchsorted(seg, np.arange(0, seg[-1], 10.0)).clip(0, len(P) - 1)]
    tp = cKDTree(P)
    res = dict(route_km=round(seg[-1] / 1000, 2))
    if len(I):
        ie, inn = to.transform(I.lon.to_numpy(), I.lat.to_numpy())
        I = I.assign(d=tp.query(np.column_stack([ie, inn]))[0])
        near = I[I.d <= CORRIDOR_M]
        covered = cKDTree(np.column_stack([ie, inn])[I.d <= CORRIDOR_M]).query(samples)[0] <= 20 if len(near) else np.zeros(len(samples), bool)
        res.update(images=len(near), route_with_photo_within_20m=f"{np.mean(covered):.0%}",
                   newest=str(near.captured.max().date()) if len(near) else None,
                   photos_since_2024=f"{np.mean(near.captured >= '2024-01-01'):.0%}" if len(near) else None)
    else:
        res.update(images=0)
    if len(F):
        fe, fn = to.transform(F.lon.to_numpy(), F.lat.to_numpy())
        F = F.assign(d=tp.query(np.column_stack([fe, fn]))[0], E=fe, N=fn)
        Fc = F[F.d <= CORRIDOR_M]
        kinds = Fc.value.str.split("--").str[:2].str.join("--").value_counts()
        res["features"] = len(Fc)
        res["feature_kinds"] = kinds.head(8).to_dict()
        L = Fc[Fc.value.str.startswith(LIGHTS)]
        res["mapillary_lights"] = len(L)
        # vs OSM signals and our verified lights (placed with GPS for this comparison only)
        sig = pd.read_parquet(ROOT / "data" / "osm" / "signals_lamps.parquet")
        sig = sig[sig.kind != "street_lamp"]
        se, sn = to.transform(sig.lon.to_numpy(), sig.lat.to_numpy())
        S = np.column_stack([se, sn])
        S = S[tp.query(S)[0] <= CORRIDOR_M]
        res["osm_signals"] = len(S)
        ours = our_lights(drv)
        res["our_verified_lights"] = len(ours)
        if len(L):
            ML = L[["E", "N"]].to_numpy()
            res["mapillary_lights_near_osm_signal"] = f"{np.mean(cKDTree(S).query(ML)[0] <= 25):.0%}" if len(S) else "–"
            if len(ours):
                res["our_lights_near_a_mapillary_light"] = f"{np.mean(cKDTree(ML).query(ours)[0] <= 25):.0%}"
                res["mapillary_lights_near_one_of_ours"] = f"{np.mean(cKDTree(ours).query(ML)[0] <= 25):.0%}"
    return res


def our_lights(drv: dict) -> np.ndarray:
    """Our verified light landmarks in UTM, placed with the GPS track (comparison only)."""
    from src.video import map_layer, world
    f = drv["folder"] / "lights.parquet"
    if not f.exists():
        return np.zeros((0, 2))
    L = pd.read_parquet(f)
    L = L[L.real]
    pose = world.Pose(drv)
    g = drives.gps_track(drv).sort_values("video_s")
    to = map_layer.utm()
    out = []
    for m in L.itertuples():
        X, Z = pose.to_car(m.t_first, m.x, m.z)
        e, n = to.transform(np.interp([m.t_first - 1.5, m.t_first, m.t_first + 1.5], g.video_s, g.lon),
                            np.interp([m.t_first - 1.5, m.t_first, m.t_first + 1.5], g.video_s, g.lat))
        th = np.arctan2(e[2] - e[0], n[2] - n[0])
        out.append((e[1] + X * np.cos(th) + Z * np.sin(th), n[1] - X * np.sin(th) + Z * np.cos(th)))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drives", nargs="+", help="drive keys in configs/drives.yaml")
    for k in ap.parse_args().drives:
        drv = drives.load(k)
        I, F = fetch(drv)
        print(k, json.dumps(report(drv, I, F), indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
