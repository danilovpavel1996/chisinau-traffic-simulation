from __future__ import annotations
import math
import yaml
from dataclasses import dataclass, field
from lxml import etree


@dataclass
class Zone:
    id: str
    label: str
    centroid: tuple[float, float]   # (lon, lat)
    pop_weight: float
    job_weight: float
    edges: list[str] = field(default_factory=list)

    @property
    def attraction(self) -> float:
        return self.job_weight

    @property
    def production(self) -> float:
        return self.pop_weight


class ZoneRegistry:
    def __init__(self, zones: list[Zone]):
        self._zones: dict[str, Zone] = {z.id: z for z in zones}

    @classmethod
    def from_config(cls, config_path: str) -> "ZoneRegistry":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        zones = []
        for zid, zdata in cfg["zones"].items():
            zones.append(Zone(
                id=zid,
                label=zdata["label"],
                centroid=tuple(zdata["centroid"]),
                pop_weight=zdata["pop_weight"],
                job_weight=zdata["job_weight"],
            ))
        return cls(zones)

    @staticmethod
    def _lonlat_to_sumo(lon, lat, ox, oy):
        a=6378137.0; e2=0.00669438; k0=0.9996
        lon0=math.radians(27)
        lat_r=math.radians(lat); lon_r=math.radians(lon)
        N=a/math.sqrt(1-e2*math.sin(lat_r)**2)
        T=math.tan(lat_r)**2
        C=e2/(1-e2)*math.cos(lat_r)**2
        A=math.cos(lat_r)*(lon_r-lon0)
        M=a*((1-e2/4-3*e2**2/64)*lat_r-(3*e2/8+3*e2**2/32)*math.sin(2*lat_r)+(15*e2**2/256)*math.sin(4*lat_r))
        x=k0*N*(A+(1-T+C)*A**3/6)+500000
        y=k0*(M+N*math.tan(lat_r)*(A**2/2+(5-T+9*C)*A**4/24))
        return x+ox, y+oy

    def assign_edges_from_net(self, net_file: str, radius_m: float = 500.0) -> None:
        tree = etree.parse(net_file)
        root = tree.getroot()

        location = root.find("location")
        ox, oy = map(float, location.get("netOffset").split(","))

        # Convert all zone centroids to SUMO coords
        zone_sumo = {}
        for zone in self._zones.values():
            lon, lat = zone.centroid
            sx, sy = self._lonlat_to_sumo(lon, lat, ox, oy)
            zone_sumo[zone.id] = (sx, sy)
            print(f"[zones] {zone.id:20s} centroid SUMO=({sx:.1f},{sy:.1f})")

        for edge in root.findall("edge"):
            eid = edge.get("id", "")
            if eid.startswith(":"):
                continue
            lanes = edge.findall("lane")
            if not lanes:
                continue
            shape_str = lanes[0].get("shape", "")
            if not shape_str:
                continue
            coords = [tuple(map(float, p.split(","))) for p in shape_str.split()]
            if not coords:
                continue
            mid_x = sum(c[0] for c in coords) / len(coords)
            mid_y = sum(c[1] for c in coords) / len(coords)

            # Check if edge allows passenger vehicles
            allow = lanes[0].get("allow", "all")
            disallow = lanes[0].get("disallow", "")
            if allow != "all":
                if "passenger" not in allow:
                    continue
            if "passenger" in disallow:
                continue

            for zone in self._zones.values():
                cx, cy = zone_sumo[zone.id]
                if math.sqrt((mid_x - cx)**2 + (mid_y - cy)**2) < radius_m:
                    zone.edges.append(eid)

        for zone in self._zones.values():
            print(f"[zones] {zone.id:20s}: {len(zone.edges)} edges assigned")

    def assign_edges_manual(self, mapping: dict[str, list[str]]) -> None:
        for zid, edges in mapping.items():
            if zid in self._zones:
                self._zones[zid].edges = edges
        print("[zones] Manual edge assignment applied.")

    def get_zone(self, zone_id: str) -> Zone:
        return self._zones[zone_id]

    def get_edges(self, zone_id: str) -> list[str]:
        return self._zones[zone_id].edges

    def all_zones(self) -> list[Zone]:
        return list(self._zones.values())

    def zone_ids(self) -> list[str]:
        return list(self._zones.keys())

    def __repr__(self) -> str:
        return f"ZoneRegistry({list(self._zones.keys())})"
