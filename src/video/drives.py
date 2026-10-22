"""
drives.py
---------
Registry of recorded drives (configs/drives.yaml), their output folders, and
the one GPS anchor every phone video carries (start position + start time).
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]          # repository root
REGISTRY = ROOT / "configs" / "drives.yaml"
PROCESSED = ROOT / "data" / "video" / "processed"


def load(key: str) -> dict:
    """Settings of one drive, with absolute `video` and output `folder` paths."""
    drives = yaml.safe_load(REGISTRY.read_text())
    if key not in drives:
        raise KeyError(f"drive '{key}' not in {REGISTRY} (known: {', '.join(drives)})")
    d = dict(drives[key])
    d["key"] = key
    d["video"] = ROOT / d["video"]
    d["folder"] = PROCESSED / key
    d["folder"].mkdir(parents=True, exist_ok=True)
    return d


def anchor(video: Path) -> tuple[float, float, datetime]:
    """Start GPS point and local start time from QuickTime (iPhone) metadata."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags",
                          "-of", "json", str(video)], capture_output=True, text=True, check=True)
    tags = json.loads(out.stdout)["format"]["tags"]
    iso = tags["com.apple.quicktime.location.ISO6709"]          # +47.0586+028.8141+133.146/
    lat, lon = [float(x) for x in iso.replace("+", " +").replace("-", " -").split()[:2]]
    return lat, lon, datetime.fromisoformat(tags["com.apple.quicktime.creationdate"])
