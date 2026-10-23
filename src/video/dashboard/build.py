"""
dashboard/build.py
------------------
One dashboard for all recorded drives: pick a drive, watch its video next to
the map, in sync.

    video     + live overlay of detected vehicles / traffic lights
    spatial   bird's-eye view: what the camera sees vs what the map expects (bev.py)
    map       ground truth (dashcam GPS or the driver's directions), the matched
              route drawing itself as the video plays, video-only dead-reckoned
              shape, OSM traffic signals and crossings, stops / turns /
              crosswalks / re-seeds; click the route to jump the video there
    timeline  moving / stopped, turning, red light, crosswalk, guard rail, and
              speed (estimated vs GPS); click to seek
    panel     clock, matched street, speed, distance from the truth

A drive that is still being processed shows whatever exists so far.

Usage:
    .venv-video/bin/python -m src.video.dashboard.build            # all drives
    .venv-video/bin/python -m src.video.dashboard.build <drive>    # rebuild one drive's data

Writes data/video/dashboard.html, and per drive processed/<drive>/dashboard_data.js
(+ a 720p H.264 copy of the video that every browser plays). Open the dashboard
in Chrome or Safari — no server needed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.video import drives
from src.video.drives import PROCESSED, REGISTRY, ROOT

DASHBOARD = ROOT / "data" / "video" / "dashboard.html"
PROXY = "video_720p.mp4"
FEATURES = ROOT / "data" / "osm" / "junction_features.parquet"


def make_proxy(video: Path, out: Path):
    old = out.with_name("viewer_720p.mp4")                   # name used before the dashboard
    if old.exists() and not out.exists():
        old.rename(out)
    if out.exists():
        return
    print(f"transcoding {video.name} → {out.name} (one-off)…", flush=True)
    subprocess.run(["ffmpeg", "-v", "error", "-stats", "-y", "-i", str(video), "-map", "0:v:0",
                    "-vf", "scale=-2:720,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "26", "-movflags", "+faststart", "-an", str(out)], check=True)


def r6(a):
    return [round(float(x), 6) for x in a]


def build_data(drv: dict) -> dict:
    from src.video.localize import crossing_events, guard_rail_seconds
    from src.video.motion import ego_states, stops, turns
    from src.video.traffic import VEHICLES, dominant_light

    folder, cam = drv["folder"], drv["camera"]
    lat0, lon0, start = drives.anchor(drv)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                "default=nw=1:nk=1", str(drv["video"])], capture_output=True, text=True).stdout)
    secs = np.arange(0, int(dur) + 1)
    out = dict(key=drv["key"], description=drv.get("description", ""), video=f"processed/{drv['key']}/{PROXY}",
               duration=dur, anchor=[lat0, lon0],
               clock=list(drives.clock(drv, secs).strftime("%H:%M:%S")),   # local wall clock per video second
               street_window_s=drv.get("street_window_s", [0, dur]),
               start_pin=drv.get("start"), dest_pin=drv.get("destination"))
    if (folder / "bev.mp4").exists():
        out["bev"] = f"processed/{drv['key']}/bev.mp4"

    # ground truth
    gps = drives.gps_track(drv)
    if gps is not None:
        out["truth_kind"] = "GPS (dashcam overlay)"
        out["gps"] = dict(s=r6(gps.video_s), lat=r6(gps.lat), lon=r6(gps.lon), speed=r6(gps.speed_kmh))
        out["truth"] = [[a, b] for a, b in zip(out["gps"]["lat"], out["gps"]["lon"])]
    elif drv.get("truth"):
        out["truth_kind"] = "driver's directions"
        out["truth_route"] = " → ".join(s.removeprefix("Strada ") for s in drv["truth"]["streets"])

    # video-only route: event-based matching (event_match.py) if available, else the particle filter
    em, mf = folder / "event_match.json", folder / "match.json"
    if em.exists():
        e_m = json.loads(em.read_text())
        tr = pd.DataFrame(e_m["trace"])
        tr["street"] = ""
        m = dict(trace=tr.to_dict(orient="list"), method="event-based route matching", seed="-",
                 km=e_m["route_km"], accuracy=None, reseeds=[], dead_reckoning=None)
        out["event_eval"] = e_m.get("evaluation", {})
        out["event_route"] = " → ".join(n.removeprefix("Strada ") for n in e_m["streets"])
        mf = None
    if mf is not None and mf.exists():
        m = json.loads(mf.read_text())
    if em.exists() or (mf is not None and mf.exists()):
        tr = pd.DataFrame(m["trace"])
        kx = 111_320 * np.cos(np.radians(tr.lat.mean()))
        dx, dy = np.gradient(tr.lon) * kx, np.gradient(tr.lat) * 111_320
        tr["speed_kmh"] = pd.Series(np.hypot(dx, dy) * 3.6).rolling(5, center=True, min_periods=1).mean().round(0)
        if "kmh" in tr:                    # the video's own measured speed, not the speed of the marker
            tr["speed_kmh"] = tr["kmh"]
        tr["bearing"] = (np.degrees(np.arctan2(dx, dy)) % 360).round(0)
        if "street" not in tr:
            tr["street"] = ""
        keep = [c for c in ("video_s", "lat", "lon", "street", "speed_kmh", "bearing", "error_m", "d") if c in tr]
        out["match"] = dict(trace=tr[keep].to_dict(orient="list"), accuracy=m.get("accuracy"), km=m.get("km"),
                            method=m.get("method"), seed=m.get("seed"), dead_reckoning=m.get("dead_reckoning"),
                            reseeds=m.get("reseeds", []))
        if "truth" not in out and m.get("truth"):
            out["truth"] = m["truth"]

    # perception-based strips, events and detection boxes
    events = []
    if (folder / "ego.parquet").exists() and (folder / "meta.json").exists():
        meta = json.loads((folder / "meta.json").read_text())
        det = pd.read_parquet(folder / "detections.parquet")
        e = ego_states(pd.read_parquet(folder / "ego.parquet"), meta["width"], cam["hfov_deg"], cam["hood"])
        e = e.merge(dominant_light(det), on="frame", how="left")
        sec = e.assign(s=e.t.astype(int)).groupby("s").agg(
            moving=("moving", "mean"), turning=("turning", "mean"), red=("light", lambda c: (c == "red").mean()))
        out["states"] = dict(s=sec.index.tolist(), moving=(sec.moving > 0.5).astype(int).tolist(),
                             turning=(sec.turning > 0.3).astype(int).tolist(), red=(sec.red > 0.3).astype(int).tolist())
        out["src_size"] = [meta["width"], meta["height"]]
        hood_y = meta["height"] * (1 - cam["hood"])
        d = det[det.cls.isin(VEHICLES + ["traffic_light"])]
        d = d[~(d.cls.isin(VEHICLES) & ((d.y1 + d.y2) / 2 > hood_y))]
        out["detections"] = [[round(t, 2), [[int(r.x1), int(r.y1), int(r.x2), int(r.y2),
                                             "light" if r.cls == "traffic_light" else "heavy" if r.cls in ("bus", "truck") else r.cls,
                                             r.light or ""] for r in g.itertuples()]] for t, g in d.groupby("t")]
        events += [dict(t=float(r.t_start), kind="stop", text=f"stop {r.duration_s:.0f} s"
                        + (" · red light" if r.cause == "signal" else "")) for r in stops(e).itertuples()]
        events += [dict(t=float(r.t_start), kind="turn", text=f"turn {r.direction} ~{r.angle_deg_approx}°")
                   for r in turns(e).itertuples()]
    if (folder / "scene.parquet").exists():
        sc = pd.read_parquet(folder / "scene.parquet").sort_values("t")
        out["scene"] = dict(t=r6(sc.t), crosswalk=r6(sc.crosswalk), guard_rail=r6(sc.guard_rail))
        events += [dict(t=float(t), kind="crossing", text="crosswalk") for t in crossing_events(folder)]
    for t, n in (out.get("match") or {}).get("reseeds", []):
        events.append(dict(t=float(t), kind="reseed", text=f"re-seeded {n} hypotheses at a turn"))
    out["events"] = sorted(events, key=lambda x: x["t"])

    # traffic measures (position.py, street_stats.py, lanes.py, signs.py)
    if (folder / "trace.parquet").exists():
        tr = pd.read_parquet(folder / "trace.parquet")
        tr = tr[tr.in_window]
        out["speed_trace"] = dict(lat=r6(tr.lat), lon=r6(tr.lon), kmh=r6(tr.speed_kmh), s=tr.video_s.tolist(),
                                  street=tr.street.tolist(), source=tr.source.iat[0] if len(tr) else "")
    if (folder / "stops_detail.csv").exists():
        sd = pd.read_csv(folder / "stops_detail.csv")
        out["stops"] = sd.replace({np.nan: None}).to_dict(orient="records")
        for r in sd.itertuples():
            txt = f"stop {r.duration_s} s · {r.cause}"
            if pd.notna(r.next_signal_m):
                txt += f" ({int(r.next_signal_m)} m before signal)"
            events = [e for e in events if not (e["kind"] == "stop" and abs(e["t"] - r.t_start) < 3)]
            events.append(dict(t=float(r.t_start), kind="stop", text=txt))
    agg_sig = ROOT / "data" / "video" / "aggregate" / "signals.csv"
    if agg_sig.exists():
        sg = pd.read_csv(agg_sig)
        sg = sg[sg.drive == drv["key"]].dropna(subset=["lat"])
        out["signals"] = sg.replace({np.nan: None}).to_dict(orient="records")
    if (folder / "lanes.parquet").exists():
        ln = pd.read_parquet(folder / "lanes.parquet").dropna(subset=["lanes"])
        out["lanes"] = dict(t=r6(ln.t), n=ln.lanes.astype(int).tolist())
    if (folder / "text.parquet").exists():
        tx = pd.read_parquet(folder / "text.parquet")
        tx["what"] = tx.street.fillna(tx.get("place"))
        tx = tx.dropna(subset=["what"]).drop_duplicates(subset=["what"]).head(80)
        events += [dict(t=float(r.video_s), kind="text", text=f"read “{r.text}” → {r.what}") for r in tx.itertuples()]
    out["events"] = sorted(events, key=lambda x: x["t"])

    # OSM signals / crossings near the drive
    pts = np.array(out.get("truth") or [[lat0, lon0]])
    if FEATURES.exists():
        f = pd.read_parquet(FEATURES)
        f = f[f.lat.between(pts[:, 0].min() - 0.003, pts[:, 0].max() + 0.003)
              & f.lon.between(pts[:, 1].min() - 0.004, pts[:, 1].max() + 0.004)]
        out["osm"] = dict(lat=r6(f.lat), lon=r6(f.lon), kind=f.kind.tolist())
    return out


def build_drive(key: str) -> dict:
    drv = drives.load(key)
    make_proxy(drv["video"], drv["folder"] / PROXY)
    data = build_data(drv)
    (drv["folder"] / "dashboard_data.js").write_text(
        "window.DRIVE = " + json.dumps(data, separators=(",", ":"), default=str) + ";\n")
    for old in ("viewer.html",):                                   # replaced by the dashboard
        (drv["folder"] / old).unlink(missing_ok=True)
    print(f"  {key}: {'match ✓' if 'match' in data else 'no match yet'}, "
          f"{'GPS ✓' if 'gps' in data else 'no GPS'}, {len(data['events'])} events", flush=True)
    return data


def write_dashboard():
    index = []
    for key, d in {str(k): v for k, v in yaml.safe_load(REGISTRY.read_text()).items()}.items():
        if (PROCESSED / key / "dashboard_data.js").exists():
            index.append(dict(key=key, description=d.get("description", "")))
    html = (Path(__file__).with_name("template.html").read_text()
            .replace("/*__INDEX__*/[]", json.dumps(index, ensure_ascii=False)))
    DASHBOARD.write_text(html)
    print(f"→ {DASHBOARD}  ({len(index)} drives)")


def run(drv: dict):
    build_drive(drv["key"])
    write_dashboard()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", nargs="?", help="rebuild only this drive's data (default: all drives)")
    a = ap.parse_args()
    keys = [a.drive] if a.drive else [str(k) for k in yaml.safe_load(REGISTRY.read_text())]
    for k in keys:
        build_drive(k)
    write_dashboard()


if __name__ == "__main__":
    main()
