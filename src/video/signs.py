"""
signs.py
--------
Text in the street — street-name plates, direction boards, shop and building
names — read with Apple's Vision OCR (macOS, no download) from a full-resolution
frame every EVERY_S seconds, and matched against OSM street names.

A matched street name ("PETRICANI 86" on a house-number plate) or a named place
(a Rompetrol station, a Felicia pharmacy — OSM shops/amenities/brands) is a
strong, map-checkable observation — for localisation it is close to a GPS fix.

Privacy: number plates are personal data (Moldovan law 133/2011) — text that
looks like a plate is dropped before anything is stored; only a count is kept.

Usage:
    .venv-video/bin/python -m src.video.signs <drive>              # OCR + matching
    .venv-video/bin/python -m src.video.signs <drive> --rematch    # matching only (reuses text.parquet)

Output: data/video/processed/<drive>/text.parquet — every text line read
(video_s, text, street match + score), and a summary of matched street names.
"""

from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from src.video import drives
from src.video.compute import progress
from src.video.drives import ROOT
from src.video.overlay import ocr_binary
from src.video.roads import load_graph, street_name
from src.video.truth import norm

EVERY_S = 2
MIN_SCORE = 0.86        # fuzzy match ratio to accept a street name
GENERIC = {"strada", "str", "bulevardul", "bd", "piata", "calea", "stradela", "soseaua", "aleea"}
PLATE = re.compile(r"^(?=.*\d.*\d)(?=.*[A-ZА-Я])[A-ZА-Я0-9\s.•:·-]{4,11}$")    # letters + ≥2 digits, short
POIS = ROOT / "data" / "osm" / "pois.parquet"


def is_plate(text: str) -> bool:
    return bool(PLATE.match(text.strip().upper()))


def pois(osm_file=ROOT / "data" / "osm" / "chisinau_full.osm") -> pd.DataFrame:
    """Named shops / amenities / brands with positions (nodes) from the OSM extract, cached."""
    if POIS.exists():
        return pd.read_parquet(POIS)
    import xml.etree.ElementTree as ET
    rows = []
    for _, el in ET.iterparse(osm_file, events=("end",)):
        if el.tag == "node":
            tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
            if ("shop" in tags or "amenity" in tags or "brand" in tags) and ("name" in tags or "brand" in tags):
                for n in {tags.get("brand"), tags.get("name")} - {None}:
                    rows.append(dict(name=n, lat=float(el.get("lat")), lon=float(el.get("lon")),
                                     kind=tags.get("shop") or tags.get("amenity") or "brand"))
            el.clear()
        elif el.tag in ("way", "relation"):
            el.clear()
    df = pd.DataFrame(rows)
    df.to_parquet(POIS, index=False)
    return df


def street_names() -> dict[str, str]:
    """normalised core name → OSM name, e.g. 'mihai viteazul' → 'Strada Mihai Viteazul'."""
    out = {}
    for _, _, d in load_graph().edges(data=True):
        n = street_name(d)
        if n.startswith("("):
            continue
        for part in n.split(", "):
            core = norm(part)
            if len(core) >= 5:
                out[core] = part
    return out


def match(text: str, names: dict[str, str]) -> tuple[str | None, float]:
    t = norm(re.sub(r"[^\w\s.-]", " ", text))
    t = re.sub(r"\s+\d+[a-z]?$", "", t)                       # house number: 'petricani 86' → 'petricani'
    t = " ".join(w for w in t.split() if w.strip(".") not in GENERIC)
    if len(t) < 5:
        return None, 0.0
    best = difflib.get_close_matches(t, list(names), n=1, cutoff=MIN_SCORE)
    if not best:
        return None, 0.0
    return names[best[0]], difflib.SequenceMatcher(None, t, best[0]).ratio()


def run(drv: dict):
    video, folder = drv["video"], drv["folder"]
    hood = drv["camera"]["hood"]
    with tempfile.TemporaryDirectory() as tmp:
        print(f"extracting a frame every {EVERY_S} s…", flush=True)
        subprocess.run(["ffmpeg", "-v", "error", "-stats", "-i", str(video), "-vf",   # -stats: progress
                        f"fps=1/{EVERY_S},crop=iw:ih*{1 - hood:.2f}:0:0", "-q:v", "2",
                        str(Path(tmp) / "f%05d.jpg")], check=True)
        files = sorted(Path(tmp).glob("f*.jpg"))
        binary, rows = ocr_binary(), []
        bar = progress(len(files), "reading text")
        for i in range(0, len(files), 25):
            batch = files[i:i + 25]
            res = subprocess.run([str(binary), *map(str, batch)], capture_output=True, text=True).stdout
            for line in res.splitlines():
                path, _, text = line.partition("\t")
                video_s = (int(Path(path).stem[1:]) - 1) * EVERY_S + EVERY_S / 2
                for piece in text.split(" | "):
                    if piece.strip():
                        rows.append(dict(video_s=video_s, text=piece.strip()))
            bar.update(len(batch))
        bar.close()
    save(drv, pd.DataFrame(rows, columns=["video_s", "text"]))


def save(drv: dict, raw: pd.DataFrame):
    """Drop number plates, match street names and named places, store."""
    folder = drv["folder"]
    plates = raw.text.map(is_plate)
    df = raw[~plates][["video_s", "text"]].reset_index(drop=True)
    names = street_names()
    m = [match(t, names) for t in df.text]
    df["street"], df["score"] = [x[0] for x in m], [round(x[1], 2) for x in m]
    poi = pois()
    poi_names = {norm(n): n for n in poi.name if len(norm(n)) >= 5}
    pm = [match(t, poi_names) if s is None else (None, 0.0) for t, s in zip(df.text, df.street)]
    df["place"] = [x[0] for x in pm]
    df["clock"] = drives.clock(drv, df.video_s).strftime("%H:%M:%S") if len(df) else []
    df.attrs["plates_dropped"] = int(plates.sum())
    df.to_parquet(folder / "text.parquet", index=False)
    hits = df.dropna(subset=["street"])
    places = df.dropna(subset=["place"])
    print(f"→ {folder / 'text.parquet'}: {len(df)} text lines kept ({int(plates.sum())} number plates dropped), "
          f"{len(hits)} street-name and {len(places)} named-place matches")
    for col, sub in (("street", hits), ("place", places)):
        if len(sub):
            print(sub.groupby(col).agg(first=("clock", "first"), times=("text", "size"), example=("text", "first"))
                     .sort_values("first").to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    ap.add_argument("--rematch", action="store_true", help="re-run matching on the stored text (no OCR)")
    a = ap.parse_args()
    drv = drives.load(a.drive)
    if a.rematch:
        save(drv, pd.read_parquet(drv["folder"] / "text.parquet")[["video_s", "text"]])
    else:
        run(drv)


if __name__ == "__main__":
    main()
