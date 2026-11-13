"""
overlay.py
----------
Read the text a dashcam burns into the picture — date/time, speed, GPS — once
per video second, with Apple's Vision OCR (macOS, no download).

    2026-10-22 18:16:51 | 2mph | 47°2.361N, 28°48.243E
    → clock, speed_kmh, lat, lon  (degrees + decimal minutes → decimal degrees)

This is ground truth for the drive (position and speed every second) and the
real clock across gaps between dashcam clips. The vision-only pipeline never
sees it: the strip lies below the bonnet line that perception ignores.

Usage:
    .venv-video/bin/python -m src.video.overlay <drive>

Output: data/video/processed/<drive>/gps.parquet
Needs in configs/drives.yaml: overlay: {height_px: 46}
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.video import drives
from src.video.compute import progress
from src.video.drives import ROOT

OCR_SRC = ROOT / "src" / "video" / "ocr" / "vision_ocr.swift"
OCR_BIN = ROOT / ".build" / "vision_ocr"
MEDIAN_S = 21           # rolling-median window (s) of the track used to catch repeated misreads
OFF_TRACK_M = 250       # m from that median track = a misread (a turn bends it far less)
MIN_SEG = 5             # only stretches this long are repaired (shorter ones: drop_glitches)
JUMP_SLACK, JUMP_M = 1.5, 60     # a step longer than 1.5 × overlay speed × time + 60 m is a misread
DIGIT_FIXES = sorted([(c, sgn * d * 10.0 ** -p) for c in ("lat", "lon") for p in range(1, 6)
                      for d in range(1, 10) for sgn in (1, -1)], key=lambda x: abs(x[1]))
PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2}).*?(\d+)\s*mph.*?"
    r"(\d{1,2})\s*°?\s*(\d{1,2}\.\d+)\s*'?\s*([NS]).*?(\d{1,3})\s*°?\s*(\d{1,2}\.\d+)\s*'?\s*([EW])")   # ° often dropped by OCR


def ocr_binary() -> Path:
    if not OCR_BIN.exists() or OCR_BIN.stat().st_mtime < OCR_SRC.stat().st_mtime:
        OCR_BIN.parent.mkdir(exist_ok=True)
        subprocess.run(["swiftc", "-O", "-o", str(OCR_BIN), str(OCR_SRC)], check=True)
    return OCR_BIN


TIME_ONLY = re.compile(r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})(?:.*?(\d+)\s*mph)?")


def parse(text: str) -> dict | None:
    t = text.replace("|", " ").replace("O", "0").replace("o", "0")     # 'Omph' → 0 mph
    t = t.replace("0mph", "0 mph").replace("N,", "N ").replace("′", "'")
    t = re.sub(r"[•º˚*]", "°", t)                                      # OCR variants of the degree sign
    t = re.sub(r"(°\s*\d{1,2}\.\d*) (\d)", r"\1\2", t)              # '°2.36 1N' → '°2.361N' (coordinates only)
    t = re.sub(r"(?<=\d) (?=[NSEW]\b)", "", t)                        # '2.367 N' → '2.367N'
    m = PATTERN.search(t)
    if not m:
        # no position yet (the dashcam needs ~1 min for a GPS fix) — keep the clock and speed
        tm = TIME_ONLY.search(t)
        if not tm:
            return None
        day, hms, mph = tm.groups()
        return dict(clock=datetime.fromisoformat(f"{day} {hms}"),
                    speed_kmh=round(int(mph) * 1.609344, 1) if mph else np.nan, lat=np.nan, lon=np.nan)
    day, hms, mph, la_d, la_m, ns, lo_d, lo_m, ew = m.groups()
    lat = (int(la_d) + float(la_m) / 60) * (1 if ns == "N" else -1)
    lon = (int(lo_d) + float(lo_m) / 60) * (1 if ew == "E" else -1)
    return dict(clock=datetime.fromisoformat(f"{day} {hms}"), speed_kmh=round(int(mph) * 1.609344, 1),
                lat=lat, lon=lon)


def repair_digits(df: pd.DataFrame, kx: float) -> pd.DataFrame:
    """
    A misread digit can persist far longer than the median window: 2026_10_24_17_08 read
    latitude 47.05… for 47.06… for 32 s on end (1.1 km off) — three such stretches made a
    13.2 km "GPS route" of an 8.7 km drive. The overlay's own speed says how far the car can
    move between two readings, so the track is cut where it jumps further; every piece but the
    longest is tried with one digit of latitude or longitude changed (± 1–9 in one decimal
    place), and the change is kept if it joins the piece smoothly to its neighbours.
    """
    df = df.copy()
    for _ in range(2):
        g = df[~df.bad]
        if len(g) < 3:
            return df
        lat, lon, t = g.lat.to_numpy(), g.lon.to_numpy(), g.video_s.to_numpy()
        speed = g.speed_kmh if "speed_kmh" in g else pd.Series(np.nan, index=g.index)
        v = speed.fillna(130).clip(lower=20).to_numpy() / 3.6      # unknown: up to 130 km/h
        allow = np.maximum(v[1:], v[:-1]) * np.diff(t) * JUMP_SLACK + JUMP_M
        step = np.hypot(np.diff(lat) * 111_320, np.diff(lon) * kx)
        cuts = np.flatnonzero(step > allow) + 1
        if not len(cuts):
            return df
        segs = np.split(np.arange(len(g)), cuts)
        main = max(range(len(segs)), key=lambda k: len(segs[k]))
        fixed = 0
        for k, seg in enumerate(segs):
            if k == main or len(seg) < MIN_SEG:
                continue
            a, b = seg[0], seg[-1]
            for col, delta in DIGIT_FIXES:
                la, lo = (lat[seg] + delta, lon[seg]) if col == "lat" else (lat[seg], lon[seg] + delta)
                ok_prev = a == 0 or np.hypot((la[0] - lat[a - 1]) * 111_320, (lo[0] - lon[a - 1]) * kx) <= allow[a - 1]
                ok_next = b == len(g) - 1 or np.hypot((lat[b + 1] - la[-1]) * 111_320, (lon[b + 1] - lo[-1]) * kx) <= allow[b]
                if ok_prev and ok_next:
                    df.loc[g.index[seg], col] = (la if col == "lat" else lo)
                    fixed += 1
                    break
        if not fixed:
            break
    return df


def drop_glitches(df: pd.DataFrame, kx: float) -> pd.DataFrame:
    """Short misreads: far from the rolling median track, or jumping > 60 m/s and back."""
    for _ in range(3):
        g = df[~df.bad]
        mid_lat = g.lat.rolling(MEDIAN_S, center=True, min_periods=3).median()
        mid_lon = g.lon.rolling(MEDIAN_S, center=True, min_periods=3).median()
        far = np.hypot((g.lat - mid_lat) * 111_320, (g.lon - mid_lon) * kx) > OFF_TRACK_M
        if not far.any():
            break
        df.loc[far[far].index, "bad"] = True
    for _ in range(3):
        g = df[~df.bad]
        step = np.hypot(g.lat.diff() * 111_320, g.lon.diff() * kx) / g.video_s.diff()
        jumpy = (step > 60) & (step.shift(-1) > 60)
        if not jumpy.any():
            break
        df.loc[jumpy[jumpy].index, "bad"] = True
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop OCR misreads: the clock must advance ~1 s per video s (bad_time), positions
    can't jump > 60 m/s and must exist (bad = not usable as a GPS position).
    A misread digit (0↔6, 3↔8 in the minutes: ~1 km off) often repeats for several
    seconds, so a position is also bad when it is far from the rolling median track.
    """
    df = df.sort_values("video_s").reset_index(drop=True)
    off = (df.clock - df.clock.min()).dt.total_seconds() - df.video_s
    df["bad_time"] = (off - off.rolling(15, center=True, min_periods=3).median()).abs() > 3
    df["bad"] = df.bad_time | df.lat.isna()
    kx = 111_320 * np.cos(np.radians(df.lat.median()))
    df = drop_glitches(df, kx)
    df = repair_digits(df, kx)             # long misread stretches, on what the glitch filters left
    df = drop_glitches(df, kx)
    return df


def run(drv: dict):
    cfg = drv.get("overlay")
    if not cfg:
        print("no overlay configured for this drive — skipped")
        return
    video, out = drv["video"], drv["folder"] / "gps.parquet"
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                "default=nw=1:nk=1", str(video)], capture_output=True, text=True).stdout)
    h = int(cfg["height_px"])
    with tempfile.TemporaryDirectory() as tmp:
        print(f"extracting {int(dur)} overlay strips…", flush=True)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps=1,crop=iw:{h}:0:ih-{h}",
                        str(Path(tmp) / "s%05d.png")], check=True)
        files = sorted(Path(tmp).glob("s*.png"))
        rows, bar = [], progress(len(files), "overlay OCR")
        binary = ocr_binary()
        for i in range(0, len(files), 100):
            batch = files[i:i + 100]
            res = subprocess.run([str(binary), *map(str, batch)], capture_output=True, text=True).stdout
            for line in res.splitlines():
                path, _, text = line.partition("\t")
                r = parse(text)
                video_s = int(Path(path).stem[1:]) - 1 + 0.5        # fps=1 takes the middle of each second
                rows.append(dict(video_s=video_s, text=text, **(r or {})))
            bar.update(len(batch))
        bar.close()
    df = pd.DataFrame(rows)
    ok = df.dropna(subset=["clock"])
    df = clean(ok).merge(df[df.clock.isna()][["video_s", "text"]], how="outer")
    df["bad"] = df.bad.fillna(True).astype(bool)
    df["bad_time"] = df.bad_time.fillna(True).astype(bool)
    df.to_parquet(out, index=False)
    good = df[~df.bad]
    print(f"→ {out}: {len(good)}/{len(df)} seconds read cleanly, clock {good.clock.min():%H:%M:%S}–"
          f"{good.clock.max():%H:%M:%S}, max speed {good.speed_kmh.max():.0f} km/h", flush=True)


def reparse(drv: dict):
    """Re-parse the saved OCR text (after improving `parse`) without running OCR again."""
    out = drv["folder"] / "gps.parquet"
    raw = pd.read_parquet(out)[["video_s", "text"]]
    parsed = pd.DataFrame([parse(t) or {} for t in raw.text])
    df = pd.concat([raw.reset_index(drop=True), parsed], axis=1)
    ok = df.dropna(subset=["clock"])
    df = clean(ok).merge(df[df.clock.isna()][["video_s", "text"]], how="outer")
    df["bad"] = df.bad.fillna(True).astype(bool)
    df["bad_time"] = df.bad_time.fillna(True).astype(bool)
    df.to_parquet(out, index=False)
    print(f"→ {out}: {(~df.bad).sum()}/{len(df)} seconds read cleanly")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    ap.add_argument("--reparse", action="store_true", help="re-parse saved OCR text, no OCR")
    a = ap.parse_args()
    (reparse if a.reparse else run)(drives.load(a.drive))


if __name__ == "__main__":
    main()
