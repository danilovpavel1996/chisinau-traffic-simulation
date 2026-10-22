"""
evaluate.py
-----------
Robustness check of the map matching: run every seed and score each against the
ground truth (not just the one the matcher keeps). Use this to judge changes —
a single run can look good or bad by luck.

Usage:
    .venv-video/bin/python -m src.video.evaluate <drive> [--seeds 5]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.video import drives
from src.video.compute import progress
from src.video.localize import EdgeTable, accuracy, crossing_events, guard_rail_seconds, odometry, run_filter
from src.video.roads import Roads, load_graph, street_name
from src.video.truth import route_from_streets


def run(drv: dict, seeds: int = 5):
    folder, cam = drv["folder"], drv["camera"]
    t_from, t_to = drv.get("street_window_s", [0, 1e9])
    lat0, lon0, _ = drives.anchor(drv["video"])
    od = odometry(folder, cam["hfov_deg"], t_from, t_to)
    G = load_graph(); T = EdgeTable(Roads(G))
    truth = route_from_streets(G, (lat0, lon0), drv["truth"]["streets"], tuple(drv["truth"]["end"]))
    rows = []
    bar = progress(seeds, "evaluating seeds")
    for seed in range(seeds):
        tr, scale, ev = run_filter(T, od, (lat0, lon0), None, seed, crossing_events(folder), guard_rail_seconds(folder))
        a = accuracy(tr, truth)
        names = [street_name(T.roads.data(T.key[e])) for e in tr.edge]
        streets = [n for i, n in enumerate(names) if n != "(unnamed)" and (i == 0 or n != names[i - 1])]
        rows.append(dict(seed=seed, evidence=round(ev), median_m=a["median_m"], within_30m=a["within_30m"],
                         coverage=a["coverage"], route=" → ".join(dict.fromkeys(s.removeprefix("Strada ") for s in streets))))
        bar.update(1)
    bar.close()
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
    print(df.to_string(index=False))
    cov = df.coverage.str.rstrip("%").astype(int)
    print(f"\ncoverage over {seeds} runs: median {int(np.median(cov))} %, worst {cov.min()} %, "
          f"runs ≥ 90 %: {(cov >= 90).sum()}/{seeds}; the kept run (max evidence) = seed {df.evidence.idxmax()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    run(drives.load(a.drive), a.seeds)


if __name__ == "__main__":
    main()
