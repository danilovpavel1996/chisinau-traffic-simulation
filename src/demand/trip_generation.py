"""
trip_generation.py
------------------
Converts the OD matrix CSV into a SUMO .rou.xml trip file.

For each (time_bin, origin, destination, trips) row:
  • sample `trips` departure times within the 15-min bin
  • pick a random origin edge and destination edge
  • assign vehicle type according to the mix in config

Routing (shortest path at free-flow) is delegated to SUMO's
`duarouter` — we only generate <trip> elements here, not full routes.
Duarouter is called at the end of this script.

Usage
-----
    python src/demand/trip_generation.py --config src/config/scenario.yaml
"""

from __future__ import annotations
import argparse
import os
import random
import subprocess
import yaml
import numpy as np
import pandas as pd
from lxml import etree
from zones import ZoneRegistry


VEHICLE_TYPES_XML = """\
    <vType id="passenger" accel="2.6" decel="4.5" sigma="0.5"
           length="4.5" maxSpeed="50" guiShape="passenger"/>
    <vType id="truck"     accel="1.3" decel="4.0" sigma="0.5"
           length="12.0" maxSpeed="30" guiShape="truck"/>
    <vType id="bus"       accel="1.2" decel="4.0" sigma="0.5"
           length="12.0" maxSpeed="25" guiShape="bus"/>
"""


def vehicle_type(mix: dict[str, float], rng: np.random.Generator) -> str:
    types = list(mix.keys())
    probs = np.array(list(mix.values()), dtype=float)
    probs /= probs.sum()
    return rng.choice(types, p=probs)


def generate_trips(
    od_df: pd.DataFrame,
    registry: ZoneRegistry,
    mix: dict[str, float],
    bin_sec: int,
    seed: int = 42,
) -> list[dict]:
    """
    Returns a list of trip dicts:
      {id, type, depart, from_edge, to_edge}
    """
    rng = np.random.default_rng(seed)
    trips = []
    trip_id = 0

    # Pre-compute edge lists once
    edge_map = {z.id: registry.get_edges(z.id) for z in registry.all_zones()}

    for _, row in od_df.iterrows():
        t_bin   = int(row["time_bin"])
        origin  = row["origin"]
        dest    = row["destination"]
        n       = int(row["trips"])

        orig_edges = edge_map.get(origin, [])
        dest_edges = edge_map.get(dest, [])

        if not orig_edges or not dest_edges:
            # Zone edges not assigned yet — generate placeholder TAZ-based trip
            for _ in range(n):
                depart = t_bin + float(rng.integers(0, bin_sec))
                trips.append({
                    "id":       f"t_{trip_id:07d}",
                    "type":     vehicle_type(mix, rng),
                    "depart":   round(depart, 1),
                    "fromTaz":  origin,
                    "toTaz":    dest,
                    "from_edge": None,
                    "to_edge":   None,
                })
                trip_id += 1
        else:
            for _ in range(n):
                depart     = t_bin + float(rng.integers(0, bin_sec))
                from_edge  = rng.choice(orig_edges)
                to_edge    = rng.choice(dest_edges)
                if from_edge == to_edge:
                    continue
                trips.append({
                    "id":        f"t_{trip_id:07d}",
                    "type":      vehicle_type(mix, rng),
                    "depart":    round(depart, 1),
                    "from_edge": from_edge,
                    "to_edge":   to_edge,
                    "fromTaz":   None,
                    "toTaz":     None,
                })
                trip_id += 1

    trips.sort(key=lambda x: x["depart"])
    return trips


def write_rou_xml(trips: list[dict], out_file: str) -> None:
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    root = etree.Element("routes")
    root.append(etree.Comment(" Vehicle type definitions "))

    for t in trips:
        trip_el = etree.SubElement(root, "trip")
        trip_el.set("id",     t["id"])
        trip_el.set("type",   t["type"])
        trip_el.set("depart", str(t["depart"]))

        if t["from_edge"]:
            trip_el.set("from", t["from_edge"])
            trip_el.set("to",   t["to_edge"])
        else:
            trip_el.set("fromTaz", t["fromTaz"])
            trip_el.set("toTaz",   t["toTaz"])

    tree = etree.ElementTree(root)
    tree.write(out_file, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    print(f"[trip_gen] Wrote {len(trips)} trips → {out_file}")


def run_duarouter(trips_file: str, net_file: str, routes_file: str) -> None:
    """Call SUMO's duarouter to convert trips into full routes."""
    cmd = [
        "duarouter",
        "--net-file",    net_file,
        "--trip-files",  trips_file,
        "--output-file", routes_file,
        "--ignore-errors",
        "--no-warnings",
        "--repair",
        "--repair.from",
        "--repair.to",
        "--begin",  "0",
        "--end",    "86400",
    ]
    print(f"[trip_gen] Running duarouter…")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[trip_gen] WARNING: duarouter returned non-zero exit code.")
    else:
        print(f"[trip_gen] Routes written → {routes_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config/scenario.yaml")
    parser.add_argument("--skip-router", action="store_true",
                        help="Skip duarouter (write trips only)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    od_file     = cfg["demand"]["od_file"]
    routes_file = cfg["demand"]["routes_file"]
    trips_file  = routes_file.replace(".rou.xml", ".trips.xml")
    net_file    = cfg["network"]["net_file"]
    mix         = cfg["demand"]["vehicle_mix"]
    bin_sec     = cfg["demand"]["time_bin_minutes"] * 60
    seed        = cfg["simulation"]["seed"]

    registry = ZoneRegistry.from_config(args.config)

    # Attempt automatic edge assignment from net.xml
    if os.path.exists(net_file):
        registry.assign_edges_from_net(net_file)
    else:
        print(f"[trip_gen] net_file not found ({net_file}) — using TAZ-based trips")

    od_df  = pd.read_csv(od_file)
    trips  = generate_trips(od_df, registry, mix, bin_sec, seed=seed)
    write_rou_xml(trips, trips_file)

    if not args.skip_router and os.path.exists(net_file):
        run_duarouter(trips_file, net_file, routes_file)
    else:
        print("[trip_gen] Skipping duarouter.")


if __name__ == "__main__":
    main()
