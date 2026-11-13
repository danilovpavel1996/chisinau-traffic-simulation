"""
motion.py
---------
The recording car's own motion, from the optical flow saved by perception.py:

    moving / stopped     road-surface flow above an adaptive threshold
    rotation (yaw)       deg/s, + = left; intercept of the horizontal-flow fit,
                         corrected for camera mis-alignment over the drive
    speed index          ∝ speed; road-plane estimate where the road is visible,
                         scene flow (rotation removed) × local ratio elsewhere
    turns, stops         events derived from the above

Known limits (2026-10-22 drive, measured against the true route):
  - turns are measured roughly right (right turn 64° vs ~70° true), but driving
    straight produces false rotation (−97° on straight Mihai Viteazul) → use
    short-term heading changes, not long-term absolute heading;
  - no speed signal is metric yet: metres per unit differ 2–2.6× between the
    first part of the drive and Mihai Viteazul / Calea Ieșilor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HFOV_DEG = 70    # default horizontal field of view (iPhone 1x video); per drive in configs/drives.yaml
YAW_TURN = 5.0   # deg/s of sustained rotation (1 s median) above which we are turning;
                 # junction turns run ~10–25 deg/s, phone-mount shake ~1–5 deg/s
MIN_TURN = 20    # deg; smaller heading changes are road curves or lane changes


def ego_states(ego: pd.DataFrame, W: int, hfov: float = HFOV_DEG, hood: float = 0.15) -> pd.DataFrame:
    """
    Per sampled frame: moving, yaw_deg_s, yaw_smooth, turning, flow_smooth
    (scene flow, rotation removed) and speed_idx. Windows are in seconds, so
    results don't depend on the sampling stride.
    """
    e = ego.copy()
    dt = float(e.t.diff().median())
    win = max(3, int(round(1.0 / dt)) | 1)            # odd number of samples ≈ 1 s
    smooth = lambda s: s.rolling(win, center=True, min_periods=1).median()

    if "yaw_slope" in e:
        # The road strip above the bonnet often has < 10 trackable points (smooth
        # asphalt, a vehicle close ahead): 35 % of frames at > 15 km/h on 2026_10_23,
        # and a missing value read as "stopped" froze the odometer. The forward
        # expansion of the whole scene (yaw_slope) is never missing and separates
        # stopped / moving ~2000× — fill the gaps with it, scaled by this drive's own
        # ratio between the two (no camera constants, no GPS).
        both = e.ego_mag.notna() & e.yaw_slope.notna() & (e.ego_mag > 0.6)
        if both.sum() > 50:
            k = float(np.median(e.ego_mag[both] / e.yaw_slope[both].abs().clip(lower=1e-3)))
            e["ego_mag_measured"] = e.ego_mag.notna()
            e["ego_mag"] = e.ego_mag.fillna(e.yaw_slope.abs() * k)

    e["moving"] = smooth(e.ego_mag) > max(0.6, 0.1 * np.nanpercentile(smooth(e.ego_mag), 95))

    focal = (W / 2) / np.tan(np.radians(hfov / 2))
    yaw = e.yaw_dx
    if "yaw_slope" in e and e.moving.sum() > 50:
        # Phone not aimed exactly along the direction of travel → part of the
        # forward-motion slope leaks into the intercept. Calibrate that leak over
        # the drive: rotation is zero-mean, the leak is ∝ slope.
        mv = e.moving & e.yaw_slope.notna() & e.yaw_dx.notna()
        c = np.polyfit(e.yaw_slope[mv], e.yaw_dx[mv], 1)[0]
        yaw = e.yaw_dx - c * e.yaw_slope
    e["yaw_deg_s"] = (np.degrees(yaw / focal) / dt).fillna(0)
    e["yaw_smooth"] = smooth(e.yaw_deg_s)
    e["turning"] = e.yaw_smooth.abs() > YAW_TURN

    # forward motion only: while turning the road also streams sideways by
    # ~yaw px/frame, which is rotation, not distance travelled
    e["flow_smooth"] = smooth(np.sqrt(np.clip(e.ego_mag ** 2 - yaw.fillna(0) ** 2, 0, None)))
    ground = ground_speed(e, W, hood=hood)
    e["speed_idx"] = e.flow_smooth if ground is None else smooth(ground).fillna(0)
    return e


def ground_speed(e: pd.DataFrame, W: int, n_bands: int = 8, band_top: float = 0.15,
                 hood: float = 0.45) -> pd.Series | None:
    """
    Speed index from the road plane. For a point on flat road at image row y,
    vertical flow ≈ v · (y − y_h)² / (f · camera height), y_h = horizon row
    (median focus-of-expansion row while driving straight). Dividing a band's
    flow by (y − y_h)² should remove the dependence on what the camera looks at.

    In practice (see module docstring) it is only marginally better than scene
    flow: plain asphalt has little texture, so the road is "visible" ~half the
    time, and gaps are filled with scene flow × the local ratio (≥30 % data per
    1-min window, clamped to 0.5–2.5× the drive-wide ratio — an unclamped ratio
    from a few outliers once produced 50× phantom speed).
    """
    cols = [f"band{i}_dy" for i in range(n_bands)]
    if not set(cols) <= set(e.columns) or "foe_y" not in e:
        return None
    H = W * 9 / 16
    straight = e.moving & ~e.turning & e.foe_y.between(0, H)
    if straight.sum() < 50:
        return None
    y_h = float(np.median(e.foe_y[straight]))
    edges = np.linspace(H * band_top, H * (1 - hood), n_bands + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    # usable bands: clearly below the horizon (above it: buildings, not road)
    # and clearly above the bonnet edge (reflections)
    lo, hi = y_h + 0.08 * H, H * (1 - hood) - 0.05 * H
    est = [e[c].clip(lower=0) / ((y - y_h) / H) ** 2 for c, y in zip(cols, centres) if lo < y < hi]
    if not est:
        return None
    g = pd.concat(est, axis=1).mean(axis=1)

    dt = float(e.t.diff().median())
    both = g.notna() & (e.flow_smooth > 0.5)
    glob = float(np.nanmedian(g[both] / e.flow_smooth[both]))
    win = int(60 / dt)
    roll = lambda s: s.where(both).rolling(win, center=True, min_periods=int(0.3 * win)).median()
    ratio = (roll(g) / roll(e.flow_smooth)).clip(0.5 * glob, 2.5 * glob).fillna(glob)
    g = g.where((g <= 4 * glob * e.flow_smooth + 1) | g.isna())   # drop impossible readings
    return g.fillna(e.flow_smooth * ratio)


def turns(e: pd.DataFrame, min_s: float = 1.5, merge_gap: float = 3.0) -> pd.DataFrame:
    """
    Ego manoeuvres. Scenery flowing left (yaw < 0) means we turn right.
    Turning periods closer than `merge_gap` s are one manoeuvre (a roundabout
    is right-left-right); its angle is the NET heading change ∫ yaw dt.
    Angles are approximate (they scale with the assumed field of view).
    """
    dt = e.t.diff().fillna(0)
    seg = (e.turning != e.turning.shift()).cumsum()
    spans = [(g.t.iat[0], g.t.iat[-1]) for _, g in e[e.turning].groupby(seg[e.turning])]
    merged = []
    for a, b in spans:
        if merged and a - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    out = []
    for a, b in merged:
        g = e[(e.t >= a) & (e.t <= b)]
        angle = float((g.yaw_deg_s * dt[g.index]).sum())
        if b - a < min_s or abs(angle) < MIN_TURN:
            continue
        out.append(dict(t_start=a, t_end=b, duration_s=round(b - a, 1),
                        direction="right" if angle < 0 else "left", angle_deg_approx=round(abs(angle))))
    return pd.DataFrame(out, columns=["t_start", "t_end", "duration_s", "direction", "angle_deg_approx"])


def stops(e: pd.DataFrame, min_s: float = 3.0) -> pd.DataFrame:
    """Standstill periods; `e` may carry a `light` column (colour of the dominant traffic light)."""
    seg = (e.moving != e.moving.shift()).cumsum()
    out = []
    for _, g in e[~e.moving].groupby(seg[~e.moving]):
        d = g.t.iat[-1] - g.t.iat[0]
        if d >= min_s:
            lights = g["light"].dropna() if "light" in g else pd.Series(dtype=object)
            out.append(dict(t_start=g.t.iat[0], t_end=g.t.iat[-1], duration_s=round(d, 1),
                            light_seen=lights.mode().iat[0] if len(lights) else "none",
                            cause="signal" if (lights == "red").any() else "congestion_or_other"))
    return pd.DataFrame(out, columns=["t_start", "t_end", "duration_s", "light_seen", "cause"])
