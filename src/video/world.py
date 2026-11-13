"""
world.py
--------
A fixed ground frame the car drives through, from the video alone — so that what stands still
(traffic lights, parked cars) stays still, and what moves, moves smoothly.

    pose        our position and heading every analysed frame: speed and rotation from
                speed.py (flow × metric depth; rotation from distant points with GeoCalib's
                focal length: turns within ~10 % of GPS), integrated. Locally exact enough; it
                drifts over minutes, which does not matter for things seen for seconds
    lights      traffic lights as landmarks: every detection is a line of sight from where we
                were (bearing from the undistorted box centre) plus a rough distance (metric
                depth, nearest part of the box). Detections are grouped by geometry — a new one
                joins the landmark that lies on its line of sight at a consistent distance (the
                tracker's ids are no use: half the light tracks have ≤ 4 detections, one id
                covered 649) — and each landmark's position is the least-squares point of all its
                lines of sight and distances: well fixed once we have driven past at an angle,
                where one frame's depth wobbles by metres. Height from the elevation angles.
                Colour: the state seen at each moment (red / yellow / green), kept for a while
    vehicles    each detection moved into the ground frame; per track a parked car gets one
                position (the median), a moving one a smooth path (local linear fit over
                ±SMOOTH_S); back in the car frame for drawing

Car frame (surround.py): X metres to the right, Z ahead. Ground frame: x, z with heading ψ
(radians, 0 = our heading at the start, + = turned right).

Usage:
    .venv-video/bin/python -m src.video.world <drive> [--from S --to S]    # report on a clip
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.video import depth as depth_mod
from src.video import drives
from src.video.bev import Geometry
from src.video.scene import WIDTH

# traffic lights
GATE_ANGLE = 0.06            # rad: a detection joins a landmark this close to its line of sight …
GATE_M = 1.5                 # … or within this many metres of it
GATE_DEPTH = 0.45            # … at a distance within this fraction of its depth reading
ANGLE_SIGMA = 0.012          # rad: bearing noise of one detection (box centre)
DEPTH_SIGMA = 0.30           # depth reading noise, fraction of the distance
MIN_OBS, MIN_SPAN_S = 4, 0.8 # a landmark needs this many detections over this long
MERGE_M = 2.0                # landmarks closer than this are one light
LIGHT_Z = (3.0, 80.0)
COLOUR_HOLD_S = 3.0          # a seen colour holds this long without a new reading
SHOW_BEFORE_S, SHOW_AFTER_S = 4.0, 30.0
# vehicles
SMOOTH_S = 0.6
VEHICLES = ("car", "bus", "truck", "motorcycle")


class Pose:
    """Our position (x, z metres) and heading ψ in the ground frame at any time."""

    def __init__(self, drv: dict):
        s = pd.read_parquet(drv["folder"] / "speed.parquet").sort_values("t")
        t = s.t.to_numpy()
        v = s.kmh_raw.fillna(0).clip(lower=0).to_numpy() / 3.6
        w = np.radians(s.yaw_dps.fillna(0).to_numpy())
        w = pd.Series(w).rolling(3, center=True, min_periods=1).median().to_numpy()   # single-frame spikes
        dt = np.r_[0, np.diff(t)]
        psi = np.cumsum(w * dt)
        mid = psi - w * dt / 2                                   # heading half-way through each step
        self.t, self.psi = t, psi
        self.yaw_abs = np.abs(pd.Series(np.degrees(w)).rolling(5, center=True, min_periods=1).mean().to_numpy())
        self.x = np.cumsum(v * dt * np.sin(mid))
        self.z = np.cumsum(v * dt * np.cos(mid))
        self.v = v

    def at(self, t):
        return np.interp(t, self.t, self.x), np.interp(t, self.t, self.z), np.interp(t, self.t, self.psi)

    def to_world(self, t, X, Z):
        px, pz, psi = self.at(t)
        c, s = np.cos(psi), np.sin(psi)
        return px + X * c + Z * s, pz - X * s + Z * c

    def to_car(self, t, x, z):
        px, pz, psi = self.at(t)
        dx, dz = np.asarray(x) - px, np.asarray(z) - pz
        c, s = np.cos(psi), np.sin(psi)
        return dx * c - dz * s, dx * s + dz * c

    def yaw_at(self, t: float) -> float:
        """|rotation| in deg/s (surround.EgoMotion's interface)."""
        return float(np.interp(t, self.t, self.yaw_abs))

    def between(self, t0: float, t1: float) -> tuple[float, float]:
        """(metres driven, radians turned right) from t0 to t1 — surround.EgoMotion's interface."""
        x0, z0, p0 = self.at(t0)
        x1, z1, p1 = self.at(t1)
        d = np.hypot(x1 - x0, z1 - z0)
        return float(d if (x1 - x0) * np.sin(p0) + (z1 - z0) * np.cos(p0) >= 0 else -d), float(p1 - p0)


# ── traffic lights ──────────────────────────────────────────────────────────
def light_observations(drv: dict, geo: Geometry, pose: Pose) -> pd.DataFrame:
    """One row per traffic-light detection: line of sight in the ground frame, depth, elevation."""
    folder = drv["folder"]
    meta = json.loads((folder / "meta.json").read_text())
    k = WIDTH / meta["width"]
    det = pd.read_parquet(folder / "detections.parquet")
    det = det[det.cls == "traffic_light"].sort_values("t")
    td, maps = depth_mod.load(drv)
    cache: dict = {}
    rows = []
    for b in det.itertuples():
        j = int(np.argmin(np.abs(td - b.t)))
        d = None
        if abs(td[j] - b.t) <= 0.26:
            if j not in cache:
                cache.clear()
                cache[j] = depth_mod.metric(drv, maps[j])
            D = cache[j]
            r0, r1 = int(b.y1 * k) // depth_mod.DOWN, int(b.y2 * k) // depth_mod.DOWN + 1
            c0, c1 = int(b.x1 * k) // depth_mod.DOWN, int(b.x2 * k) // depth_mod.DOWN + 1
            patch = D[max(r0, 0):min(r1, D.shape[0]), max(c0, 0):min(c1, D.shape[1])]
            if patch.size:
                d = float(np.percentile(patch, 20))             # the housing is the nearest thing in its box
        xu, yu = geo.undistort((b.x1 + b.x2) / 2 * k, (b.y1 + b.y2) / 2 * k)
        bearing = np.arctan((xu - geo.cx) / geo.fl)              # + = to the right
        elev = np.arctan((geo.y_h - yu) / geo.fl)
        px, pz, psi = pose.at(b.t)
        a = psi + bearing
        rows.append(dict(t=b.t, track_id=b.track_id, conf=b.conf, light=b.light, ox=px, oz=pz,
                         ux=np.sin(a), uz=np.cos(a), depth=d if d and LIGHT_Z[0] < d < LIGHT_Z[1] else np.nan,
                         elev=elev))
    return pd.DataFrame(rows)


def solve(o: pd.DataFrame) -> tuple[float, float, float]:
    """Least-squares point of lines of sight (perpendicular offset) and depth readings (along).
    Returns (x, z, rms residual in metres)."""
    ox, oz, ux, uz, dep = (o[c].to_numpy() for c in ("ox", "oz", "ux", "uz", "depth"))
    nx, nz = uz, -ux                                              # normal to each line of sight
    guess = np.nanmedian(dep) if np.isfinite(dep).any() else 20.0
    dd = np.where(np.isfinite(dep), dep, guess)
    sp = ANGLE_SIGMA * dd + 0.3
    A = [np.column_stack([nx, nz]) / sp[:, None]]
    b = [(nx * ox + nz * oz) / sp]
    ok = np.isfinite(dep)
    if ok.any():
        sd = DEPTH_SIGMA * dep[ok] + 0.5
        A.append(np.column_stack([ux[ok], uz[ok]]) / sd[:, None])
        b.append((ux[ok] * ox[ok] + uz[ok] * oz[ok] + dep[ok]) / sd)
    A, b = np.vstack(A), np.concatenate(b)
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    perp = (p[0] - ox) * nx + (p[1] - oz) * nz
    return float(p[0]), float(p[1]), float(np.sqrt(np.mean(perp ** 2)))


def lights(drv: dict, geo: Geometry, pose: Pose) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Landmarks (id, x, z, y height, n, t_first, t_last, rms) and the observations with their landmark."""
    obs = light_observations(drv, geo, pose)
    marks: list[dict] = []
    owner = np.full(len(obs), -1)
    for i, r in enumerate(obs.itertuples()):
        best, best_score = None, np.inf
        for m in marks:
            if r.t - m["t_last"] > 6.0:                          # out of sight for long: not this one
                continue
            dx, dz = m["x"] - r.ox, m["z"] - r.oz
            along = dx * r.ux + dz * r.uz
            if along <= 1.0:
                continue
            perp = abs(dx * r.uz - dz * r.ux)
            if perp > max(GATE_M, GATE_ANGLE * along):
                continue
            if np.isfinite(r.depth) and abs(along - r.depth) > GATE_DEPTH * max(along, r.depth) + 2:
                continue
            score = perp / max(GATE_M, GATE_ANGLE * along) + (0.0 if r.track_id in m["tracks"] else 0.5)
            if score < best_score:
                best, best_score = m, score
        if best is None:
            if not np.isfinite(r.depth):
                continue
            best = dict(id=len(marks), idx=[], tracks=set(), x=r.ox + r.ux * r.depth, z=r.oz + r.uz * r.depth, t_last=r.t)
            marks.append(best)
        best["idx"].append(i)
        best["tracks"].add(r.track_id)
        best["t_last"] = r.t
        owner[i] = best["id"]
        if len(best["idx"]) >= 2:
            best["x"], best["z"], _ = solve(obs.iloc[best["idx"]])
    obs["mark"] = owner
    out = []
    for m in marks:
        o = obs.iloc[m["idx"]]
        if len(o) < MIN_OBS or o.t.max() - o.t.min() < MIN_SPAN_S:
            continue
        x, z, rms = solve(o)
        dist = np.hypot(x - o.ox, z - o.oz)
        y = float(np.median(geo.h + dist * np.tan(o.elev)))
        if not 1.5 < y < 10:
            continue
        out.append(dict(id=m["id"], x=x, z=z, y=y, n=len(o), t_first=o.t.min(), t_last=o.t.max(), rms=rms))
    L = pd.DataFrame(out, columns=["id", "x", "z", "y", "n", "t_first", "t_last", "rms"])
    # one light seen as two landmarks (e.g. a gap in the detections): merge close ones
    if len(L) > 1:
        L = L.sort_values("n", ascending=False).reset_index(drop=True)
        keep, alias = [], {}
        for r in L.itertuples():
            twin = next((k for k in keep if np.hypot(k.x - r.x, k.z - r.z) < MERGE_M), None)
            if twin is None:
                keep.append(r)
            else:
                alias[r.id] = twin.id
        obs["mark"] = obs.mark.map(lambda m: alias.get(m, m))
        L = pd.DataFrame([k._asdict() for k in keep]).drop(columns="Index")
        for k in L.itertuples():                                 # re-solve with the merged detections
            o = obs[obs.mark == k.id]
            x, z, rms = solve(o)
            L.loc[k.Index, ["x", "z", "rms", "n", "t_first", "t_last"]] = [x, z, rms, len(o), o.t.min(), o.t.max()]
    obs.loc[~obs.mark.isin(set(L.id)), "mark"] = -1
    return L.sort_values("t_first").reset_index(drop=True), obs


def light_colour(obs: pd.DataFrame, mark: int, t: float) -> str:
    o = obs[(obs.mark == mark) & (obs.t <= t) & (obs.t >= t - COLOUR_HOLD_S) & obs.light.isin(["red", "yellow", "green"])]
    return o.light.iloc[-1] if len(o) else "unknown"


# ── vehicles ────────────────────────────────────────────────────────────────
def smooth_tracks(rows: pd.DataFrame, pose: Pose, labels: dict) -> pd.DataFrame:
    """rows: t, track_id, X, Z (car frame) → adds xs, zs: smoothed car-frame positions."""
    xw, zw = pose.to_world(rows.t.to_numpy(), rows.X.to_numpy(), rows.Z.to_numpy())
    rows = rows.assign(xw=xw, zw=zw, xs=rows.X, zs=rows.Z)
    for tid, g in rows.groupby("track_id"):
        if len(g) < 3:
            continue
        if labels.get(tid) == "parked":
            sx = np.full(len(g), np.median(g.xw)); sz = np.full(len(g), np.median(g.zw))
        else:
            t = g.t.to_numpy()
            sx, sz = np.empty(len(g)), np.empty(len(g))
            for i, ti in enumerate(t):
                m = np.abs(t - ti) <= SMOOTH_S
                if m.sum() >= 3:
                    w = 1 - (np.abs(t[m] - ti) / (SMOOTH_S * 1.01)) ** 2
                    A = np.column_stack([np.ones(m.sum()), t[m] - ti])
                    sx[i] = np.linalg.lstsq(A * w[:, None], g.xw.to_numpy()[m] * w, rcond=None)[0][0]
                    sz[i] = np.linalg.lstsq(A * w[:, None], g.zw.to_numpy()[m] * w, rcond=None)[0][0]
                else:
                    sx[i], sz[i] = g.xw.iat[i], g.zw.iat[i]
        X, Z = pose.to_car(g.t.to_numpy(), sx, sz)
        rows.loc[g.index, "xs"], rows.loc[g.index, "zs"] = X, Z
    return rows


# ── vehicle headings ────────────────────────────────────────────────────────
DIMS = {"car": (1.8, 4.5, 1.5), "bus": (2.5, 12.0, 3.0), "truck": (2.4, 8.0, 3.0), "motorcycle": (0.8, 2.0, 1.4)}
MOVING_MS = 1.5              # a track moving at least this fast (ground frame) shows its own direction
EDGE_PX = 6                  # boxes touching the image edge or the bonnet: shape cut off, not used


def view_angle(aspect: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """|angle| between a vehicle's long axis and our line of sight from its box shape: seen at α it
    is W·|cos α| + L·|sin α| wide and H high (W, L, H per class)."""
    a = np.radians(np.arange(0, 91))
    out = np.empty(len(aspect))
    for c in np.unique(cls):
        W, L, H = DIMS.get(c, DIMS["car"])
        f = (W * np.cos(a) + L * np.sin(a)) / H
        top = int(np.argmax(f))                                    # widest at atan(L / W): beyond, no new info
        m = cls == c
        out[m] = a[np.clip(np.searchsorted(f[:top + 1], aspect[m]), 0, top)]
    return out


def _axis_mean(ang: np.ndarray) -> tuple[float, float]:
    """Mean and spread (0 = all equal) of directions modulo 180°."""
    z = np.exp(2j * ang).mean()
    return float(np.angle(z) / 2), float(1 - abs(z))


def vehicle_headings(obs: pd.DataFrame, pose: Pose, W: int, H_bonnet: float, labels: dict) -> np.ndarray:
    """
    Heading of every vehicle observation, degrees relative to ours (0 = our direction, + = right).

    A vehicle's long axis from its box shape: a car side-on is ~2.5× wider than high, from behind
    ~1×. Seen at α from its axis, its direction is our line of sight to it ± α — two candidates per
    frame. In the ground frame the true direction stays put while we drive past and the line of
    sight turns; the other candidate drifts: per track the steadier one is the axis. (Labels from
    image motion gave cars waiting on a crossing street "same direction", drawn parallel to us —
    2026_10_24_17_08 at 7:53.) Front or back: from the track's own motion when it moves, else the
    label (oncoming → towards us; waiting on a crossing street → towards our road).
    obs: t, track_id, cls, X, Z (car frame, smoothed), x1, y1, x2, y2 (image pixels).
    """
    t = obs.t.to_numpy()
    ray = np.arctan2(obs.X.to_numpy(), obs.Z.to_numpy())
    psi = pose.at(t)[2]
    ok = ((obs.x1 > EDGE_PX) & (obs.x2 < W - EDGE_PX) & (obs.y2 < H_bonnet - EDGE_PX)).to_numpy()
    aspect = ((obs.x2 - obs.x1) / (obs.y2 - obs.y1).clip(lower=1)).to_numpy()
    alpha = view_angle(aspect, obs.cls.to_numpy())
    xw, zw = pose.to_world(t, obs.X.to_numpy(), obs.Z.to_numpy())
    out = np.zeros(len(obs))
    for tid, idx in obs.groupby("track_id").indices.items():
        label = labels.get(tid, "")
        prior = {"oncoming": np.pi, "crossing": None}.get(label, 0.0)       # relative to our heading
        world_prior = None if prior is None else float(np.median(psi[idx])) + prior
        v = idx[ok[idx]]
        axis = None
        if len(v):
            plus, minus = psi[v] + ray[v] + alpha[v], psi[v] + ray[v] - alpha[v]
            (mp, sp), (mm, sm) = _axis_mean(plus), _axis_mean(minus)
            if len(v) >= 3 and abs(sp - sm) > 0.02:
                axis = mp if sp < sm else mm
            else:                                               # too short to tell: the one nearer the prior
                ref = world_prior if world_prior is not None else float(np.median(psi[v] + ray[v])) + np.pi / 2
                axis = min((mp, mm), key=lambda a: abs(np.sin(a - ref)))
        tt = t[idx]
        moving = None
        if len(idx) >= 3 and tt[-1] - tt[0] >= 0.8:
            dx, dz = xw[idx[-1]] - xw[idx[0]], zw[idx[-1]] - zw[idx[0]]
            if np.hypot(dx, dz) / (tt[-1] - tt[0]) >= MOVING_MS:
                moving = float(np.arctan2(dx, dz))
        if axis is None:
            heading = moving if moving is not None else (world_prior if world_prior is not None else float(np.median(psi[idx])))
        else:
            ref = moving if moving is not None else world_prior
            if ref is None:                                     # crossing, standing: facing our road
                mx, mz = xw[idx].mean(), zw[idx].mean()
                px, pz, _ = pose.at(float(np.median(tt)))
                ref = float(np.arctan2(px - mx, pz - mz))
            heading = axis if np.cos(axis - ref) >= 0 else axis + np.pi
        out[idx] = np.degrees((heading - psi[idx] + np.pi) % (2 * np.pi) - np.pi)
    return out


# ── report on a clip ────────────────────────────────────────────────────────
def report(drv: dict, t0: float, t1: float):
    dense = pd.read_parquet(drv["folder"] / "scene_dense.parquet").sort_values("t")
    geo = Geometry(drv, dense.iloc[0])
    pose = Pose(drv)
    L, obs = lights(drv, geo, pose)
    o = obs[(obs.t >= t0) & (obs.t <= t1)]
    print(f"{drv['key']} {t0:.0f}–{t1:.0f} s: {len(o)} light detections → {o[o.mark >= 0].mark.nunique()} landmarks "
          f"({(o.mark >= 0).mean():.0%} of detections used); drive total {len(L)} landmarks")
    for m in L[(L.t_last >= t0) & (L.t_first <= t1)].itertuples():
        oo = obs[obs.mark == m.id]
        # how much the single-frame (depth) position wobbles vs the landmark
        ok = oo.depth.notna()
        px, pz = oo.ox[ok] + oo.ux[ok] * oo.depth[ok], oo.oz[ok] + oo.uz[ok] * oo.depth[ok]
        wob = np.median(np.hypot(px - m.x, pz - m.z)) if ok.any() else np.nan
        X, Z = pose.to_car(m.t_first, m.x, m.z)
        print(f"  light {m.id}: {m.n} detections {m.t_first:.1f}–{m.t_last:.1f} s, height {m.y:.1f} m, first seen at "
              f"X {X:+.1f} Z {Z:.1f} m; lines of sight within {m.rms:.2f} m; single-frame positions wobbled {wob:.1f} m")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    ap.add_argument("--from", dest="t0", type=float, default=0)
    ap.add_argument("--to", dest="t1", type=float, default=1e9)
    a = ap.parse_args()
    report(drives.load(a.drive), a.t0, a.t1)


if __name__ == "__main__":
    main()
