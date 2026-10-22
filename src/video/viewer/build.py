"""
viewer/build.py
---------------
Build an interactive viewer for one drive: the video next to the map, in sync.

    video     with optional live overlay of detected vehicles / traffic lights
    map       true route (green), matched route drawing itself as the video plays
              (red), video-only dead-reckoned shape (dashed), moving car marker,
              stops and turns; click the route to jump the video there
    timeline  moving / stopped / turning / red light per second; click to seek
    panel     clock time, street, speed, distance from the true route

Usage:
    .venv-video/bin/python -m src.video.viewer.build <drive>

Needs: match.json (src.video.localize), detections.parquet + ego.parquet
(src.video.perception). Writes data/video/processed/<drive>/viewer.html and,
if missing, a 720p H.264 copy of the video (browsers play it everywhere).
Open viewer.html in Chrome or Safari — no server needed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.video import drives
from src.video.motion import ego_states, stops, turns
from src.video.traffic import VEHICLES, dominant_light

PROXY = "viewer_720p.mp4"


def make_proxy(video: Path, out: Path):
    if out.exists():
        return
    print(f"transcoding {video.name} → {out.name} (one-off)…")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-map", "0:v:0",
                    "-vf", "scale=-2:720,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "26", "-movflags", "+faststart", "-an", str(out)], check=True)


def build_data(drv: dict) -> dict:
    folder = drv["folder"]
    meta = json.loads((folder / "meta.json").read_text())
    match = json.loads((folder / "match.json").read_text())
    det = pd.read_parquet(folder / "detections.parquet")
    e = ego_states(pd.read_parquet(folder / "ego.parquet"), meta["width"],
                   drv["camera"]["hfov_deg"], drv["camera"]["hood"])
    e = e.merge(dominant_light(det), on="frame", how="left")

    # per-second state strip
    sec = e.assign(s=e.t.astype(int)).groupby("s").agg(
        moving=("moving", "mean"), turning=("turning", "mean"),
        red=("light", lambda c: (c == "red").mean()), yaw=("yaw_smooth", "mean"))
    states = dict(s=sec.index.tolist(), moving=(sec.moving > 0.5).astype(int).tolist(),
                  turning=(sec.turning > 0.3).astype(int).tolist(), red=(sec.red > 0.3).astype(int).tolist())

    # speed & heading along the match
    tr = pd.DataFrame(match["trace"])
    kx = 111_320 * np.cos(np.radians(tr.lat.mean()))
    dx, dy = np.gradient(tr.lon) * kx, np.gradient(tr.lat) * 111_320
    tr["speed_kmh"] = (pd.Series(np.hypot(dx, dy) * 3.6).rolling(5, center=True, min_periods=1).mean()).round(0)
    tr["bearing"] = (np.degrees(np.arctan2(dx, dy)) % 360).round(0)
    match["trace"] = tr.to_dict(orient="list")

    # detection overlay (vehicles + traffic lights), boxes in source-video pixels
    hood_y = meta["height"] * (1 - drv["camera"]["hood"])
    d = det[det.cls.isin(VEHICLES + ["traffic_light"])]
    d = d[~(d.cls.isin(VEHICLES) & ((d.y1 + d.y2) / 2 > hood_y))]
    frames = []
    for t, g in d.groupby("t"):
        frames.append([round(t, 2), [[int(r.x1), int(r.y1), int(r.x2), int(r.y2),
                                      "light" if r.cls == "traffic_light" else "heavy" if r.cls in ("bus", "truck") else r.cls,
                                      r.light or ""] for r in g.itertuples()]])

    ev_stops = stops(e)
    ev_turns = turns(e)
    events = ([dict(t=float(r.t_start), kind="stop", text=f"stop {r.duration_s:.0f} s"
                    + (" · red light" if r.cause == "signal" else "")) for r in ev_stops.itertuples()]
              + [dict(t=float(r.t_start), kind="turn", text=f"turn {r.direction} ~{r.angle_deg_approx}°")
                 for r in ev_turns.itertuples()])
    from src.video.localize import crossing_events
    events += [dict(t=float(t), kind="crossing", text="crosswalk") for t in crossing_events(drv["folder"])]
    events.sort(key=lambda x: x["t"])

    return dict(drive=drv["key"], description=drv.get("description", ""), video=PROXY,
                src_size=[meta["width"], meta["height"]], duration=float(e.t.max()),
                truth_route=" → ".join(s.removeprefix("Strada ") for s in (drv.get("truth") or {}).get("streets", [])), match=match,
                states=states, detections=frames, events=events)


def run(drv: dict):
    make_proxy(drv["video"], drv["folder"] / PROXY)
    data = build_data(drv)
    html = (Path(__file__).with_name("template.html").read_text()
            .replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))
    out = drv["folder"] / "viewer.html"
    out.write_text(html)
    print(f"→ {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    run(drives.load(ap.parse_args().drive))


if __name__ == "__main__":
    main()
