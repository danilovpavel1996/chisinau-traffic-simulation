"""
truth.py
--------
Ground-truth route of a drive, written the way a driver gives directions:
start point, the sequence of streets, end point (configs/drives.yaml → truth).

The path is the shortest one on the OSM graph where the listed streets are cheap,
unnamed links (roundabout rings, slip roads) are cheap too, and every other named
street costs 20× — then it is checked to really run through the listed streets in
that order (raises otherwise, so a wrong truth cannot silently score matches).
"""

from __future__ import annotations

import unicodedata

import networkx as nx
import osmnx as ox

OFF_ROUTE = 20.0     # cost multiplier for named streets not in the directions
LINK = 1.5           # cost multiplier for unnamed links


def norm(name: str) -> str:
    """'Strada Mirceşti' == 'Strada Mircești' == 'mircesti'."""
    s = name.replace("ş", "ș").replace("ţ", "ț").replace("Ş", "Ș").replace("Ţ", "Ț")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    for prefix in ("strada ", "bulevardul ", "str. ", "bd. "):
        s = s.removeprefix(prefix)
    return s.strip()


def edge_names(d: dict) -> set[str]:
    n = d.get("name", "")
    return {norm(x) for x in (n if isinstance(n, list) else [n]) if x}


def route_from_streets(G, start, streets: list[str], end) -> list[tuple[float, float]]:
    wanted = {norm(s) for s in streets}

    def cost(u, v, data):
        best = float("inf")
        for d in data.values():
            names = edge_names(d)
            f = 1.0 if names & wanted else LINK if not names else OFF_ROUTE
            best = min(best, d["length"] * f)
        return best

    near = lambda p: min(G.nodes, key=lambda n: ox.distance.great_circle(*p, G.nodes[n]["y"], G.nodes[n]["x"]))
    path = nx.shortest_path(G, near(start), near(end), weight=cost)

    pts, seq = [], []
    for u, v in zip(path, path[1:]):
        k = min(G[u][v], key=lambda k: G[u][v][k]["length"])
        d = G.edges[u, v, k]
        if "geometry" in d:
            xs, ys = d["geometry"].xy
            pts.extend(zip(ys, xs))
        else:
            pts.extend([(G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])])
        for n in edge_names(d) & wanted:
            if not seq or seq[-1] != n:
                seq.append(n)

    order = [norm(s) for s in streets]
    it = iter(seq)
    if not all(s in it for s in order):          # listed streets appear in this order
        raise ValueError(f"true route does not follow the directions: got {seq}, expected {order}")
    return pts
