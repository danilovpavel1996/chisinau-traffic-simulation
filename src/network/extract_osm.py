"""
extract_osm.py
--------------
Downloads the OSM bounding-box for the Airport↔Botanica corridor
using the Overpass API and saves it to data/osm/corridor.osm.

Usage
-----
    python src/network/extract_osm.py --config src/config/scenario.yaml
    python src/network/extract_osm.py --bbox 28.800 46.940 28.900 47.040
"""

import argparse
import os
import time
import urllib.request
import urllib.parse
import yaml


OVERPASS_URL = "https://overpass-api.de/api/interpreter"

OVERPASS_QUERY_TEMPLATE = """
[out:xml][timeout:90];
(
  way["highway"]({s},{w},{n},{e});
);
(._;>;);
out body;
"""


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_query(bbox: list[float]) -> str:
    """bbox = [minLon, minLat, maxLon, maxLat]  (YAML convention)"""
    w, s, e, n = bbox
    return OVERPASS_QUERY_TEMPLATE.format(s=s, w=w, n=n, e=e).strip()


def download_osm(bbox: list[float], out_file: str, retries: int = 3) -> None:
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    query = build_query(bbox)
    data = urllib.parse.urlencode({"data": query}).encode()

    for attempt in range(1, retries + 1):
        print(f"[extract_osm] Attempt {attempt}: querying Overpass…")
        try:
            req = urllib.request.Request(OVERPASS_URL, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                content = resp.read()
            with open(out_file, "wb") as f:
                f.write(content)
            size_kb = os.path.getsize(out_file) / 1024
            print(f"[extract_osm] Saved {out_file}  ({size_kb:.1f} KB)")
            return
        except Exception as exc:
            print(f"[extract_osm] Error: {exc}")
            if attempt < retries:
                time.sleep(5 * attempt)
    raise RuntimeError("All Overpass attempts failed.")


def main():
    parser = argparse.ArgumentParser(description="Download OSM corridor data")
    parser.add_argument("--config", default="src/config/scenario.yaml")
    parser.add_argument("--bbox", nargs=4, type=float,
                        metavar=("minLon", "minLat", "maxLon", "maxLat"),
                        help="Override bbox from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bbox = args.bbox if args.bbox else cfg["network"]["bbox"]
    out_file = cfg["network"]["osm_file"]

    print(f"[extract_osm] bbox = {bbox}")
    download_osm(bbox, out_file)


if __name__ == "__main__":
    main()
