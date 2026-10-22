"""
drives.py
---------
Registry of recorded drives (src/config/drives.yaml) and their output folders.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REGISTRY = Path("src/config/drives.yaml")
PROCESSED = Path("data/video/processed")


def load(key: str) -> dict:
    drives = yaml.safe_load(REGISTRY.read_text())
    if key not in drives:
        raise KeyError(f"drive '{key}' not in {REGISTRY} (known: {', '.join(drives)})")
    d = drives[key]
    d["key"] = key
    d["video"] = Path(d["video"])
    d["folder"] = PROCESSED / key
    return d
