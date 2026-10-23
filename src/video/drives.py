"""
drives.py
---------
Registry of recorded drives (configs/drives.yaml), their output folders, and
the one GPS anchor every phone video carries (start position + start time).
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]          # repository root
REGISTRY = ROOT / "configs" / "drives.yaml"
PROCESSED = ROOT / "data" / "video" / "processed"


def load(key: str) -> dict:
    """Settings of one drive, with absolute `video` and output `folder` paths."""
    # keys as text: YAML reads an unquoted 2026_10_22_18_16 as the integer 202610221816
    drives = {str(k): v for k, v in yaml.safe_load(REGISTRY.read_text()).items()}
    if key not in drives:
        raise KeyError(f"drive '{key}' not in {REGISTRY} (known: {', '.join(drives)})")
    d = dict(drives[key])
    d["key"] = key
    d["video"] = ROOT / d["video"]
    d["folder"] = PROCESSED / key
    d["folder"].mkdir(parents=True, exist_ok=True)
    # the configured field of view is only a starting guess; calibrate.py replaces it
    # with the value measured against the map, when that measurement is consistent
    d["camera"] = dict(d["camera"], hfov_nominal=d["camera"]["hfov_deg"])
    cal = d["folder"] / "calibration.json"
    if cal.exists():
        c = json.loads(cal.read_text())
        if c.get("status") == "calibrated":
            d["camera"]["hfov_deg"] = c["hfov_deg"]
    return d


def anchor(drv: dict) -> tuple[float, float, datetime]:
    """
    The one GPS fix a drive starts from, and its start time (video second 0):
    a phone video's QuickTime metadata, or the first reading of a dashcam's
    burnt-in overlay (gps.parquet from overlay.py).
    """
    gps = gps_track(drv)
    if gps is not None:
        first = gps.iloc[0]
        return float(first.lat), float(first.lon), first.clock - timedelta(seconds=float(first.video_s))
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags",
                          "-of", "json", str(drv["video"])], capture_output=True, text=True, check=True)
    tags = json.loads(out.stdout)["format"]["tags"]
    iso = tags["com.apple.quicktime.location.ISO6709"]          # +47.0586+028.8141+133.146/
    lat, lon = [float(x) for x in iso.replace("+", " +").replace("-", " -").split()[:2]]
    return lat, lon, datetime.fromisoformat(tags["com.apple.quicktime.creationdate"]).replace(tzinfo=None)


def gps_track(drv: dict):
    """Clean per-second GPS from a dashcam overlay, or None (ground truth, never fed to matching)."""
    import pandas as pd
    f = drv["folder"] / "gps.parquet"
    if not f.exists():
        return None
    g = pd.read_parquet(f)
    return g[~g.bad].sort_values("video_s").reset_index(drop=True)


def clock(drv: dict, video_s):
    """Wall-clock time of video seconds: from the dashcam overlay when present (handles gaps
    between clips), else start time + seconds."""
    import numpy as np
    import pandas as pd
    v = np.asarray(video_s, dtype=float)
    f = drv["folder"] / "gps.parquet"
    if f.exists():                     # every second with a readable overlay clock, GPS fix or not
        g = pd.read_parquet(f)
        g = g[~g.get("bad_time", g.bad).astype(bool)].dropna(subset=["clock"]).sort_values("video_s")
        epoch = g.clock.astype("int64") / 1e9
        return pd.to_datetime(np.interp(v, g.video_s, epoch), unit="s")
    _, _, start = anchor(drv)
    return pd.to_datetime(start) + pd.to_timedelta(v, unit="s")
