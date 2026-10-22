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
ROUNDABOUT_EXTRA = 90   # deg of rotation beyond the net angle that marks a roundabout


def cue_list(drv: dict) -> pd.DataFrame:
    folder, cam = drv["folder"], drv["camera"]
    meta = json.loads((folder / "meta.json").read_text())
    e = ego_states(pd.read_parquet(folder / "ego.parquet"), meta["width"], cam["hfov_deg"], cam["hood"])
    dt = e.t.diff().fillna(0)
    cues = []
    for r in turns(e).itertuples():
        g = e[(e.t >= r.t_start) & (e.t <= r.t_end)]
        total = float((g.yaw_deg_s.abs() * dt[g.index]).sum())
        signed = r.angle_deg_approx * (1 if r.direction == "right" else -1)
        rb = total - abs(signed) > ROUNDABOUT_EXTRA
        cues.append(dict(t=(r.t_start + r.t_end) / 2, kind="roundabout" if rb else "turn", turn_deg=signed))
    for t in crossing_events(folder):
        cues.append(dict(t=float(t), kind="crosswalk"))
    det = pd.read_parquet(folder / "detections.parquet")
    tl = det[(det.cls == "traffic_light") & ((det.y2 - det.y1) > SIGNAL_NEAR * meta["height"])].sort_values("t")
    if len(tl):
        grp = (tl.t.diff() > 4).cumsum()                     # one cluster per light we drive up to
        for _, g in tl.groupby(grp):
            if len(g) >= 3:
                cues.append(dict(t=float(g.t.iat[-1]), kind="signal"))
    for r in stops(e).itertuples():
        cues.append(dict(t=float(r.t_end), kind="stop"))
    rf = folder / "road.parquet"
    if rf.exists():
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
            cur = dict(t_first=c.t, t_last=c.t, cues=set(), turn_deg=0.0)
        cur["cues"].add(c.kind)
        cur["t_last"] = c.t
        if c.kind in ("turn", "roundabout") and abs(c.turn_deg) > abs(cur["turn_deg"]):
            cur["turn_deg"] = c.turn_deg
    if cur and cur["cues"] != {"stop"}:
        events.append(cur)
    out = pd.DataFrame([dict(t=(e["t_first"] + e["t_last"]) / 2, turn_deg=round(e["turn_deg"]),
                             roundabout="roundabout" in e["cues"], crosswalk="crosswalk" in e["cues"],
                             signal="signal" in e["cues"], stopped="stop" in e["cues"],
                             opening="opening" in e["cues"],
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
