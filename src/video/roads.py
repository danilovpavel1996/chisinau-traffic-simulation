"""
roads.py
--------
Chișinău drivable street graph (OpenStreetMap via osmnx) for map matching.

Usage:
    .venv-video/bin/python -m src.video.roads build      # download / refresh the graph

The graph is a cached download (data/osm/chisinau_drive.graphml, gitignored).
"""

from __future__ import annotations

import argparse

import numpy as np
import osmnx as ox

from src.video.drives import ROOT

GRAPH = ROOT / "data" / "osm" / "chisinau_drive.graphml"
BBOX = (28.74, 46.93, 28.96, 47.08)      # west, south, east, north: Chișinău municipality


def wrap(a):
    """Angle difference to (-180, 180]; positive = clockwise = right turn."""
    return (np.asarray(a) + 180) % 360 - 180


def load_graph():
    if not GRAPH.exists():
        raise FileNotFoundError(f"{GRAPH} missing — run: python -m src.video.roads build")
    return ox.load_graphml(GRAPH)


class Roads:
    """
    Directed edges with geometry, plus contra-flow copies of one-way streets.
    OSM one-way tags can be wrong or outdated (Mitropolit Dosoftei on the
    2026-10-22 drive), so driving against them is allowed — the matcher
    charges a penalty instead of forbidding it.
    """

    def __init__(self, G):
        self.G = G
        self.out = {}       # node -> [(v, key, length, coords)]
        self.ring = {(u, v, k) for u, v, k, d in G.edges(keys=True, data=True)
                     if d.get("junction") in ("roundabout", "circular")}
        self.contra = set()  # (u, v, k) = one-way edge v→u driven u→v
        for u, v, k, d in G.edges(keys=True, data=True):
            if "geometry" in d:
                xs, ys = d["geometry"].xy
                coords = list(zip(ys, xs))
            else:
                coords = [(G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])]
            self.out.setdefault(u, []).append((v, k, d["length"], coords))
            oneway = str(d.get("oneway", "False")) == "True"
            if oneway and (u, v, k) not in self.ring and not G.has_edge(v, u):
                self.out.setdefault(v, []).append((u, k, d["length"], coords[::-1]))
                self.contra.add((v, u, k))

    def data(self, edge) -> dict:
        """OSM attributes of an edge (of the original one for a contra-flow edge)."""
        u, v, k = edge
        return self.G.edges[(v, u, k) if edge in self.contra else edge]


def street_name(d: dict) -> str:
    n = d.get("name", "")
    return ", ".join(n) if isinstance(n, list) else (str(n) or "(unnamed)")


def build():
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(ROOT / "data" / "osm" / "cache")
    G = ox.graph_from_bbox(BBOX, network_type="drive", simplify=True)
    GRAPH.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, GRAPH)
    print(f"{len(G.nodes)} junctions, {len(G.edges)} road segments → {GRAPH}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build"])
    ap.parse_args()
    build()


if __name__ == "__main__":
    main()
