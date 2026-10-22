"""
traffic.py
----------
Other road users, from the detections/tracks of perception.py.

Per track (vehicle seen in ≥3 sampled frames) we compute image-space features
and label the manoeuvre relative to the ego car (right-hand traffic):

    oncoming        grows fast, drifts left              → opposite-direction flow
    passed_by_ego   grows fast, drifts right             → slower / parked cars on our side
    overtook_ego    enters from frame edge, shrinks      → faster cars in our direction
    same_dir        roughly constant size, seen ≥2 s     → cars travelling with us (density)
    crossing        large lateral motion, ~constant size → cross traffic at junctions
    parked          moves exactly like the background while we drive → not traffic
    during_turn     seen mostly while we turn; image motion is rotation, not theirs

Before labelling, fragments of the same vehicle are stitched into one track and
bus/truck are merged ("heavy": trolleybuses flip between the two).

Thresholds are heuristic (v0); check them in the viewer's detection overlay.

Usage:
    .venv-video/bin/python -m src.video.traffic <drive>

Outputs (data/video/processed/<drive>/): tracks.parquet, per_minute.csv,
stops.csv, turns.csv, timeline.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.plots import PALETTE, set_style
from src.video import drives
from src.video.motion import ego_states, stops, turns

VEHICLES = ["car", "motorcycle", "bus", "truck"]
CLASS_GROUP = {"car": "car", "motorcycle": "motorcycle", "bus": "heavy", "truck": "heavy"}
GROW = 0.35      # 1/s, d(log size)/dt above which an object is approaching fast
SHRINK = -0.25   # 1/s, below which an object is pulling away
EDGE = 0.03      # fraction of width counted as "touching the frame edge"
CROSS_SPEED = 0.08  # frame widths per second of sideways motion to count as crossing


def stitch_tracks(v: pd.DataFrame, W: int, max_gap: float = 1.0) -> pd.DataFrame:
    """
    Merge track fragments: B continues A if B starts ≤max_gap s after A ends,
    close to where A was last seen and with a similar box size. Greedy, by gap.
    """
    v = v.copy()
    v["cxp"], v["cyp"] = (v.x1 + v.x2) / 2, (v.y1 + v.y2) / 2
    v["sz"] = np.sqrt((v.x2 - v.x1) * (v.y2 - v.y1))
    ends = v.sort_values("t").groupby("track_id").tail(1).set_index("track_id")
    starts = v.sort_values("t").groupby("track_id").head(1).set_index("track_id")

    cands = []
    for a, ea in ends.iterrows():
        for b, sb in starts.iterrows():
            gap = sb.t - ea.t
            if a == b or not (0 < gap <= max_gap):
                continue
            dist = np.hypot(sb.cxp - ea.cxp, sb.cyp - ea.cyp)
            ratio = sb.sz / ea.sz
            if dist <= max(0.06 * W, 0.6 * ea.sz) and 0.7 <= ratio <= 1.45:
                cands.append((gap, dist, a, b))

    parent, used_a, used_b = {}, set(), set()
    for _, _, a, b in sorted(cands):
        if a in used_a or b in used_b:
            continue
        used_a.add(a); used_b.add(b); parent[b] = a

    def root(x):
        while x in parent:
            x = parent[x]
        return x

    v["track_id"] = v.track_id.map(root)
    return v.drop(columns=["cxp", "cyp", "sz"])


def track_features(det: pd.DataFrame, W: int, H: int, hood: float,
                   ego_flags: pd.DataFrame) -> pd.DataFrame:
    v = det[det.cls.isin(VEHICLES) & (det.track_id >= 0)].copy()
    # the bonnet / dashboard is sometimes detected as a car
    v = v[(v.y1 + v.y2) / 2 < H * (1 - hood)]
    v["cls"] = v.cls.map(CLASS_GROUP)
    n_raw = v.track_id.nunique()
    v = stitch_tracks(v, W)
    print(f"stitching: {n_raw} raw tracks → {v.track_id.nunique()}")
    v["turning"] = v.frame.map(ego_flags.turning).fillna(False).astype(bool)
    ego_moving = v.frame.map(ego_flags.moving).fillna(False).astype(bool)

    # parked: while we drive straight, the box moves like the ground below it.
    # (During turns rotation moves everything alike, so those frames are skipped.)
    if "bg_dx" in v:
        bg = np.hypot(v.bg_dx, v.bg_dy)
        resid = np.hypot(v.obj_dx - v.bg_dx, v.obj_dy - v.bg_dy)
        v["bg_moving"] = (bg > 1.5) & ego_moving & ~v.turning
        v["static"] = v.bg_moving & (resid < np.maximum(1.0, 0.3 * bg))
        # clear evidence the vehicle moved on its own (any time we are not turning)
        v["world_moving"] = ~v.turning & (resid > np.maximum(2.5, 0.5 * bg))
    else:
        v["bg_moving"] = v["static"] = v["world_moving"] = False
    v["cx"] = (v.x1 + v.x2) / 2 / W
    v["size"] = np.sqrt((v.x2 - v.x1) * (v.y2 - v.y1)) / W

    rows = []
    for tid, g in v.groupby("track_id"):
        if len(g) < 3:
            continue
        g = g.sort_values("t")
        t = g.t.to_numpy()
        dur = t[-1] - t[0]
        if dur < 0.3:
            continue
        scale_rate = np.polyfit(t - t[0], np.log(g["size"].to_numpy()), 1)[0]
        first, last = g.iloc[0], g.iloc[-1]
        rows.append(dict(
            track_id=tid,
            cls=g.cls.mode().iat[0],
            t_first=t[0], t_last=t[-1], duration=dur, n=len(g),
            cx_start=first.cx, cx_end=last.cx, cx_mean=g.cx.mean(), dx=last.cx - first.cx,
            size_start=first["size"], size_end=last["size"], scale_rate=scale_rate,
            enters_edge=bool(first.x1 < EDGE * W or first.x2 > (1 - EDGE) * W or first.y2 > 0.97 * H),
            exits_left=bool(last.x1 < EDGE * W),
            exits_right=bool(last.x2 > (1 - EDGE) * W),
            turn_share=g.turning.mean(),
            n_bg_moving=int(g.bg_moving.sum()),
            static_share=g.static.sum() / max(g.bg_moving.sum(), 1),
            n_world_moving=int(g.world_moving.sum()),
        ))
    tr = pd.DataFrame(rows)
    if tr.empty:
        return tr
    tr["label"] = tr.apply(label_track, axis=1)
    return tr


def label_track(r: pd.Series) -> str:
    in_our_lane = 0.3 <= r.cx_mean <= 0.6   # queued ahead of us, not parked
    if (r.n_bg_moving >= 5 and r.static_share >= 0.75
            and r.n_world_moving <= 1 and not in_our_lane):
        return "parked"
    if r.turn_share > 0.3:
        return "during_turn"
    if r.scale_rate > GROW and (r.exits_left or (r.cx_mean < 0.5 and r.dx < 0)):
        return "oncoming"
    if r.scale_rate > GROW and (r.exits_right or (r.cx_mean >= 0.5 and r.dx > 0)):
        return "passed_by_ego"
    if r.scale_rate < SHRINK and r.enters_edge:
        return "overtook_ego"
    if abs(r.dx) > 0.25 and abs(r.dx) / r.duration > CROSS_SPEED and abs(r.scale_rate) <= GROW:
        return "crossing"
    if abs(r.scale_rate) <= GROW and r.duration >= 2.0:
        return "same_dir"
    return "other"


def dominant_light(det: pd.DataFrame) -> pd.DataFrame:
    """Per sampled frame: colour of the largest traffic light with a known colour."""
    tl = det[(det.cls == "traffic_light") & (det.light != "unknown")].copy()
    if tl.empty:
        return pd.DataFrame(columns=["frame", "light"])
    tl["area"] = (tl.x2 - tl.x1) * (tl.y2 - tl.y1)
    return tl.sort_values("area").groupby("frame").tail(1)[["frame", "light"]]


def per_minute(tr: pd.DataFrame, det: pd.DataFrame, e: pd.DataFrame) -> pd.DataFrame:
    t0 = e.t.min()
    minute = lambda t: ((t - t0) // 60).astype(int)

    counts = (tr.assign(minute=minute(tr.t_first))
                .pivot_table(index="minute", columns="label", values="track_id",
                             aggfunc="count", fill_value=0))
    veh = det[det.cls.isin(VEHICLES)]
    visible = veh.groupby("frame").size().reindex(e.frame, fill_value=0)
    e = e.assign(visible=visible.to_numpy(), minute=minute(e.t))
    base = e.groupby("minute").agg(t_start=("t", "min"),
                                   vehicles_visible_mean=("visible", "mean"),
                                   moving_share=("moving", "mean"),
                                   red_light_share=("light", lambda s: (s == "red").mean()))
    out = base.join(counts, how="left").fillna(0)
    sec = e.groupby("minute").t.agg(lambda s: s.max() - s.min() + 1e-9)
    if "oncoming" in out:
        out["oncoming_per_min"] = out.oncoming / sec * 60
    return out.round(3)


def plot_timeline(e: pd.DataFrame, tr: pd.DataFrame, out: Path):
    set_style()
    fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True,
                           gridspec_kw=dict(height_ratios=[1, 1, 1, 1.6]))
    ax[0].plot(e.t, e.flow_smooth, color=PALETTE["blue"], lw=1)
    ax[0].fill_between(e.t, 0, e.flow_smooth.max(), where=~e.moving,
                       color=PALETTE["red"], alpha=0.15, label="ego stopped")
    ax[0].set_ylabel("ego motion\n(flow px)")
    ax[0].legend(loc="upper right")

    ax[1].plot(e.t, e.yaw_smooth, color=PALETTE["orange"], lw=1)
    ax[1].fill_between(e.t, e.yaw_smooth.min(), e.yaw_smooth.max(), where=e.turning,
                       color=PALETTE["orange"], alpha=0.15, label="ego turning")
    ax[1].axhline(0, color="#888", lw=0.5)
    ax[1].set_ylabel("rotation, deg/s\n(+ left / − right)")
    ax[1].legend(loc="upper right")

    ax[2].plot(e.t, e.visible, color=PALETTE["purple"], lw=0.8)
    ax[2].set_ylabel("vehicles\nvisible")

    colors = {"oncoming": PALETTE["orange"], "same_dir": PALETTE["green"],
              "overtook_ego": PALETTE["blue"], "passed_by_ego": PALETTE["purple"],
              "crossing": PALETTE["red"], "parked": "#555555",
              "during_turn": PALETTE["yellow"], "other": "#BBBBBB"}
    for lab, g in tr.groupby("label"):
        ax[3].scatter(g.t_first, [lab] * len(g), s=12, color=colors.get(lab, "#999"))
    lc = {"red": PALETTE["red"], "yellow": PALETTE["yellow"], "green": PALETTE["green"]}
    lights = e.dropna(subset=["light"])
    ax[3].scatter(lights.t, ["light"] * len(lights), s=6, c=lights.light.map(lc))
    ax[3].set_xlabel("video time, s")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def run(drv: dict):
    folder, cam = drv["folder"], drv["camera"]
    meta = json.loads((folder / "meta.json").read_text())
    det = pd.read_parquet(folder / "detections.parquet")
    ego = pd.read_parquet(folder / "ego.parquet")
    H, hood = meta["height"], cam["hood"]

    e = ego_states(ego, meta["width"], cam["hfov_deg"], hood).merge(dominant_light(det), on="frame", how="left")
    tr = track_features(det, meta["width"], H, hood, e.set_index("frame")[["turning", "moving"]])
    det = det[~(det.cls.isin(VEHICLES) & ((det.y1 + det.y2) / 2 >= H * (1 - hood)))]
    e["visible"] = (det[det.cls.isin(VEHICLES)].groupby("frame").size()
                    .reindex(e.frame, fill_value=0).to_numpy())

    pm = per_minute(tr, det, e.drop(columns="visible"))
    st = stops(e)
    tn = turns(e)

    tr.to_parquet(folder / "tracks.parquet", index=False)
    pm.to_csv(folder / "per_minute.csv")
    st.to_csv(folder / "stops.csv", index=False)
    tn.to_csv(folder / "turns.csv", index=False)
    plot_timeline(e, tr, folder / "timeline.png")

    span = e.t.max() - e.t.min()
    print(f"\n── {folder.name}: {span:.0f} s analysed ──")
    print(f"vehicle tracks: {len(tr)}")
    print(tr.label.value_counts().to_string())
    print(f"median track duration: {tr.duration.median():.1f} s")
    print(f"\nego moving {e.moving.mean():.0%} of the time; {len(st)} stops ≥3 s")
    if len(st):
        print(st.to_string(index=False))
    print(f"\n{len(tn)} turns")
    if len(tn):
        print(tn.to_string(index=False))
    print("\nper minute:")
    print(pm.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    run(drives.load(ap.parse_args().drive))


if __name__ == "__main__":
    main()
