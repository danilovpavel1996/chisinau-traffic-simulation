"""
build_sumo_net.py
-----------------
Converts the OSM file into a SUMO .net.xml, then applies a
patch list of known lane-count / speed / direction corrections.

Usage
-----
    python src/network/build_sumo_net.py --config src/config/scenario.yaml
    python src/network/build_sumo_net.py --osm data/osm/corridor.osm \\
                                          --out data/sumo_net/network.net.xml

Dependencies
------------
    SUMO must be on PATH (netconvert command).
    pip install pyyaml lxml
"""

import argparse
import os
import subprocess
import json
import yaml
from lxml import etree


# ── Patch list ────────────────────────────────────────────────────────────────
# Edit this list as you verify the network in SUMO GUI.
# Confidence: increments each time you verify a patch in the real simulation.
NETWORK_PATCHES: list[dict] = [
    # {
    #   "type":   "lane_count",        # lane_count | speed | one_way
    #   "edge_id": "-123456789#0",     # SUMO edge id (from netconvert)
    #   "value":   3,                  # new value
    #   "note":    "OSM says 2 lanes, satellite shows 3",
    #   "confidence": 0.8,
    # },
]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_netconvert(osm_file: str, net_file: str, poly_file: str) -> None:
    """Call SUMO netconvert to build the network from OSM."""
    os.makedirs(os.path.dirname(net_file), exist_ok=True)
    cmd = [
        "netconvert",
        "--osm-files",          osm_file,
        "--output-file",        net_file,
        "--polygon-output",     poly_file,
        "--geometry.remove",    "true",
        "--roundabouts.guess",  "true",
        "--ramps.guess",        "true",
        "--junctions.join",     "true",
        "--tls.guess",          "true",
        "--tls.join",           "true",
        "--verbose",
    ]
    print(f"[build_sumo_net] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"netconvert failed (exit {result.returncode})")
    print(f"[build_sumo_net] Network written to {net_file}")


def apply_patches(net_file: str, patches: list[dict]) -> None:
    """Apply hand-crafted corrections to the generated .net.xml."""
    if not patches:
        print("[build_sumo_net] No patches to apply.")
        return

    tree = etree.parse(net_file)
    root = tree.getroot()
    applied = 0

    for patch in patches:
        edge_id = patch["edge_id"]
        ptype   = patch["type"]
        value   = patch["value"]

        edge_el = root.find(f'.//edge[@id="{edge_id}"]')
        if edge_el is None:
            print(f"  [WARN] Edge '{edge_id}' not found — skipping patch.")
            continue

        if ptype == "lane_count":
            lanes = edge_el.findall("lane")
            current = len(lanes)
            if current < value:
                # duplicate last lane to add lanes
                last = lanes[-1]
                for i in range(value - current):
                    new_lane = etree.SubElement(edge_el, "lane")
                    new_lane.attrib.update(last.attrib)
                    new_lane.set("index", str(current + i))
            elif current > value:
                for lane in lanes[value:]:
                    edge_el.remove(lane)
            print(f"  [PATCH] {edge_id}: lane_count {current} → {value}")
            applied += 1

        elif ptype == "speed":
            for lane in edge_el.findall("lane"):
                lane.set("speed", str(value))
            print(f"  [PATCH] {edge_id}: speed → {value} m/s")
            applied += 1

        elif ptype == "one_way":
            edge_el.set("function", "normal")
            # Reversing direction requires rebuilding from OSM; log for manual fix
            print(f"  [WARN] {edge_id}: one_way patch — manual verification needed.")

    tree.write(net_file, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    print(f"[build_sumo_net] Applied {applied}/{len(patches)} patches.")


def qa_summary(net_file: str) -> None:
    """Print a short network QA summary."""
    tree = etree.parse(net_file)
    root = tree.getroot()
    edges    = root.findall("edge[@function='normal']")
    junctions = root.findall("junction[@type='traffic_light']")
    print("\n── Network QA Summary ──────────────────────────")
    print(f"  Normal edges      : {len(edges)}")
    print(f"  Traffic lights    : {len(junctions)}")
    lane_counts = [len(e.findall('lane')) for e in edges]
    if lane_counts:
        print(f"  Avg lanes/edge    : {sum(lane_counts)/len(lane_counts):.2f}")
    print("────────────────────────────────────────────────\n")


def main():
    parser = argparse.ArgumentParser(description="Build SUMO network from OSM")
    parser.add_argument("--config", default="src/config/scenario.yaml")
    parser.add_argument("--osm",    help="Override OSM file path")
    parser.add_argument("--out",    help="Override net.xml output path")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    osm_file = args.osm or cfg["network"]["osm_file"]
    net_file = args.out or cfg["network"]["net_file"]
    poly_file = cfg["network"]["poly_file"]

    run_netconvert(osm_file, net_file, poly_file)
    apply_patches(net_file, NETWORK_PATCHES)
    qa_summary(net_file)


if __name__ == "__main__":
    main()
