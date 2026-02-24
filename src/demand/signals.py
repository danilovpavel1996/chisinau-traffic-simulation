"""
signals.py  (src/demand/signals.py)
-------------------------------------
Generates data/signals/baseline.add.xml — fixed-time signal plans
for the pilot intersections defined in scenario.yaml.

Phase structure (simplified):
  Each intersection has N phases.
  Green time is split proportionally between phases.
  Yellow = 4 s, All-red = 2 s after each phase.

Usage
-----
    python src/demand/signals.py --config src/config/scenario.yaml
"""

from __future__ import annotations
import argparse
import os
import yaml
from lxml import etree


YELLOW_SEC = 4
ALL_RED_SEC = 2

# Default green split fractions per phase type (must sum < 1; rest is minor)
PHASE_SPLITS = {
    2: [0.45, 0.45],
    3: [0.38, 0.35, 0.17],
    4: [0.30, 0.30, 0.20, 0.10],
}

# SUMO phase state strings per phase type
# G = protected green, g = permissive, y = yellow, r = red
# These are stubs — real strings depend on junction topology from netconvert
PHASE_STATE_STUBS = {
    "NS_straight": "GGrrGGrr",
    "EW_straight": "rrGGrrGG",
    "NS_left":     "ggrrggrr",
    "EW_left":     "rrggrrgg",
}


def build_phases(cycle: int, phase_names: list[str]) -> list[dict]:
    """Returns list of {duration, state} dicts."""
    n = len(phase_names)
    splits = PHASE_SPLITS.get(n, [1/n] * n)
    overhead = (YELLOW_SEC + ALL_RED_SEC) * n
    green_pool = max(cycle - overhead, n * 5)

    phases = []
    for i, name in enumerate(phase_names):
        g = max(5, int(green_pool * splits[i]))
        state = PHASE_STATE_STUBS.get(name, "G" * 8)
        phases.append({"duration": g,     "state": state,        "comment": name})
        phases.append({"duration": YELLOW_SEC,  "state": state.replace("G","y").replace("g","y"), "comment": "yellow"})
        phases.append({"duration": ALL_RED_SEC, "state": "r" * len(state), "comment": "all-red"})
    return phases


def generate_baseline_xml(cfg: dict, out_file: str) -> None:
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    root = etree.Element("additional")
    root.append(etree.Comment(
        " Baseline fixed-time signal plans — Airport↔Botanica corridor "
    ))

    signals_cfg = cfg["signals"]["pilot_intersections"]
    for sig in signals_cfg:
        jid    = sig["id"]
        cycle  = sig["cycle_length"]
        phases = build_phases(cycle, sig["phases"])

        tl = etree.SubElement(root, "tlLogic")
        tl.set("id",       jid)
        tl.set("type",     "static")
        tl.set("programID", "baseline")
        tl.set("offset",   "0")

        for ph in phases:
            p = etree.SubElement(tl, "phase")
            p.set("duration", str(ph["duration"]))
            p.set("state",    ph["state"])
            p.append(etree.Comment(ph["comment"]))

        root.append(etree.Comment(
            f" {sig['label']} — cycle {cycle}s, phases: {sig['phases']} "
        ))

    tree = etree.ElementTree(root)
    tree.write(out_file, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    print(f"[signals] Baseline written → {out_file}")

    # Print summary table
    print("\n── Signal Plan Summary ─────────────────────────────────")
    print(f"{'ID':25s} {'Cycle':>6s}  {'Phases'}")
    print("-" * 60)
    for sig in signals_cfg:
        print(f"{sig['id']:25s} {sig['cycle_length']:>5d}s  {', '.join(sig['phases'])}")
    print("────────────────────────────────────────────────────────\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config/scenario.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    generate_baseline_xml(cfg, cfg["signals"]["baseline_file"])


if __name__ == "__main__":
    main()
