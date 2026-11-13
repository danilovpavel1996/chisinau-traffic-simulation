"""
events.py
---------
The video as a sequence of junction events — what event-based route matching
(event_match.py) aligns with the map.

Cues
    turn         net heading change of a manoeuvre (motion.turns), signed: + = right
    roundabout   a manoeuvre with much more rotation than its net angle (right-left-right)
    crosswalk    the car drives over a crosswalk (scene segmentation)
    signal       a traffic light detected close (large in the frame), last moment seen
    stop         the car comes to a standstill (possible junction ahead)
    opening      a side street opens on the right (road_geometry.py) — junctions
                 without crosswalks; right side only (the left is usually the
                 opposite carriageway: measured precision 20 % ≈ chance, right 50 %)
Cues within MERGE_S seconds are one junction event (a crosswalk and a turn at the
same junction). A stop only joins an event, it never makes one on its own.

Usage:
    .venv-video/bin/python -m src.video.events <drive>

Output: data/video/processed/<drive>/events.parquet
    t, turn_deg (signed), roundabout, crosswalk, signal, stopped, cues
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.video import drives
from src.video.localize import crossing_events
from src.video.motion import ego_states, stops, turns

MERGE_S = 8.0
SIGNAL_NEAR = 0.035     # traffic-light box height / frame height: close enough to be "this junction"
USE_SPEED_YAW = True
SIGNAL_CONF = 0.3       # median detector confidence of the cluster: red leaves through rain drops were
                        # "traffic lights" at 0.11–0.14; the real light passed at 509 s had 0.60 (2026_10_23)
ROUNDABOUT_EXTRA = 90   # deg of rotation beyond the net angle that marks a roundabout
USE_SIGNS = False      # junction signs as a matching cue: correct signs, but matching did not improve
                        # (day 39 → 28 %, night 33 → 33 %, 2026-10-24) — kept for display and later
SIGN_PROB = 0.85
JUNCTION_SIGNS = {"yield", "priority road", "stop", "no entry", "keep right", "keep left", "roundabout",
                  "right of way at next junction"}     # reliable GTSRB classes (blue arrows: confused with info signs)
OPENINGS = "road"   # "road" (road_geometry.py, right side) or "side_streets" (side_streets.py: more real
                    # side streets found, but ~4 in 5 detections false — matching got worse: 39 → 28 % day, 33 → 25 % night)


def cue_list(drv: dict) -> pd.DataFrame:
    folder, cam = drv["folder"], drv["camera"]
    meta = json.loads((folder / "meta.json").read_text())
    e = ego_states(pd.read_parquet(folder / "ego.parquet"), meta["width"], cam["hfov_deg"], cam["hood"])
    dt = e.t.diff().fillna(0)
    # turn angles from speed.py's rotation when it has run: fitted to distant scene points,
    # undistorted, with GeoCalib's focal length, vehicles masked. The old angle scales with the
    # assumed field of view (fitted to map turns, so to whichever route was matched: 132.9° on a
    # wrong route made a 90° left read as a 230° U-turn; 2026_10_24_17_08 at 17:16:36).
    # Turn size vs GPS heading change: 1.4–2.8× → 0.9–1.1× (night 1.4×)
    sp = folder / "speed.parquet"
    rot = pd.read_parquet(sp).dropna(subset=["yaw_dps"]) if USE_SPEED_YAW and sp.exists() else None
    cues = []
    for r in turns(e).itertuples():
        g = e[(e.t >= r.t_start) & (e.t <= r.t_end)]
        total = float((g.yaw_deg_s.abs() * dt[g.index]).sum())
        signed = r.angle_deg_approx * (1 if r.direction == "right" else -1)
        if rot is not None:
            q = rot[(rot.t >= r.t_start - 0.5) & (rot.t <= r.t_end + 0.5)]
            if len(q) > 1:
                signed = float(np.trapz(q.yaw_dps, q.t))
                total = float(np.trapz(q.yaw_dps.abs(), q.t))
        rb = total - abs(signed) > ROUNDABOUT_EXTRA
        cues.append(dict(t=(r.t_start + r.t_end) / 2, kind="roundabout" if rb else "turn", turn_deg=signed))
    for t in crossing_events(folder):
        cues.append(dict(t=float(t), kind="crosswalk"))
    det = pd.read_parquet(folder / "detections.parquet")
    # tracked lights only: since perception keeps every confident light (and adds tiled ones) for the
    # landmarks, untracked detections include street lamps at night (19_26: junctions 42 → 17 %)
    lights = det.cls == "traffic_light"
    keep = det.track_id >= 0
    lo, lc = folder / "light_obs.parquet", folder / "lights.parquet"
    if lo.exists() and lc.exists():                  # … plus every detection of a light verified as real (light_context)
        real = set(pd.read_parquet(lc).query("real").id)
        ob = pd.read_parquet(lo)
        ok = set(map(tuple, ob[ob.mark.isin(real)][["t", "conf"]].round(4).to_numpy()))
        keep = keep | pd.Series([(round(a, 4), round(b, 4)) in ok for a, b in zip(det.t, det.conf)], index=det.index)
    tl = det[lights & keep & ((det.y2 - det.y1) > SIGNAL_NEAR * meta["height"])].sort_values("t")
    if len(tl):
        grp = (tl.t.diff() > 4).cumsum()                     # one cluster per light we drive up to
        for _, g in tl.groupby(grp):
            if g.frame.nunique() >= 3 and g.conf.median() >= SIGNAL_CONF:
                cues.append(dict(t=float(g.t.iat[-1]), kind="signal"))
    for r in stops(e).itertuples():
        cues.append(dict(t=float(r.t_end), kind="stop"))
    sg = folder / "sign_types.parquet"
    if USE_SIGNS and sg.exists():                         # junction signs (sign_types.py): last sighting
        s = pd.read_parquet(sg)
        s = s[(s.prob >= SIGN_PROB) & s.name.isin(JUNCTION_SIGNS)].sort_values("t")
        for _, g in s.groupby("name"):
            for _, c in g.groupby((g.t.diff() > 1.5).cumsum()):
                if len(c) >= 2:
                    cues.append(dict(t=float(c.t.max()), kind="sign"))
    sf, rf = folder / "side_streets.parquet", folder / "road.parquet"
    if OPENINGS == "side_streets" and sf.exists():      # side streets from the calibrated ground plan, both sides
        for t in pd.read_parquet(sf).t:
            cues.append(dict(t=float(t), kind="opening"))
    elif rf.exists():                                     # older cue: right-side widening of the road
        rd = pd.read_parquet(rf)
        for t in rd[rd.open_right].t:
            cues.append(dict(t=float(t), kind="opening"))
    t0, t1 = drv.get("street_window_s", [0, 1e9])
    return pd.DataFrame(cues).query("@t0 <= t <= @t1").sort_values("t").reset_index(drop=True)


def merge(cues: pd.DataFrame) -> pd.DataFrame:
    events, cur = [], None
    for c in cues.itertuples():
        if cur is None or c.t - cur["t_last"] > MERGE_S:
            if cur and cur["cues"] != {"stop"}:
                events.append(cur)
            cur = dict(t_first=c.t, t_last=c.t, cues=set(), turn_deg=0.0, times=[])
        cur["cues"].add(c.kind)
        cur["t_last"] = c.t
        if c.kind != "sign":            # a sign is seen before / after its junction: it flags the event,
            cur["times"].append(c.t)    # it does not set its time
        if c.kind in ("turn", "roundabout") and abs(c.turn_deg) > abs(cur["turn_deg"]):
            cur["turn_deg"] = c.turn_deg
    if cur and cur["cues"] != {"stop"}:
        events.append(cur)
    when = lambda e: (min(e["times"]) + max(e["times"])) / 2 if e["times"] else (e["t_first"] + e["t_last"]) / 2
    out = pd.DataFrame([dict(t=when(e), turn_deg=round(e["turn_deg"]),
                             roundabout="roundabout" in e["cues"], crosswalk="crosswalk" in e["cues"],
                             signal="signal" in e["cues"], stopped="stop" in e["cues"],
                             opening="opening" in e["cues"], sign="sign" in e["cues"],
                             cues="+".join(sorted(e["cues"]))) for e in events])
    return out


def run(drv: dict) -> pd.DataFrame:
    ev = merge(cue_list(drv))
    ev["clock"] = drives.clock(drv, ev.t).strftime("%H:%M:%S")
    ev.to_parquet(drv["folder"] / "events.parquet", index=False)
    print(f"→ {drv['folder'] / 'events.parquet'}: {len(ev)} junction events")
    print(ev[["clock", "turn_deg", "cues"]].to_string(index=False))
    return ev


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    run(drives.load(ap.parse_args().drive))


if __name__ == "__main__":
    main()
