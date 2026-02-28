#!/usr/bin/env python3
"""
Generate trips_deckgl.json from fcd_full.xml
Memory-efficient: only keeps vehicles active during morning peak.
Writes output in chunks to avoid RAM explosion.

Usage:
    python3 make_deckgl.py
"""

import os, json, re, time
from collections import defaultdict

BASE    = os.path.dirname(os.path.abspath(__file__))
FCD     = os.path.join(BASE, 'data/outputs/fcd_full.xml')
OUT     = os.path.join(BASE, 'data/outputs/trips_deckgl.json')

MORNING_START = 7 * 3600   # 25200
MORNING_END   = 9 * 3600   # 32400
BUFFER        = 60         # 1 min buffer either side
MAX_VEHICLES  = 8000       # cap for visualization performance
MIN_WAYPOINTS = 5

print(f"Reading FCD: {FCD}")
print(f"Window: {MORNING_START//3600}:00 – {MORNING_END//3600}:00")
fcd_size = os.path.getsize(FCD)

RE_TIME  = re.compile(r'time="([\d.]+)"')
RE_VID   = re.compile(r'\bid="([^"]+)"')
RE_X     = re.compile(r'\bx="([\d.]+)"')
RE_Y     = re.compile(r'\by="([\d.]+)"')
RE_SPEED = re.compile(r'speed="([\d.]+)"')

vehicles  = {}   # vid -> [[time, lon, lat, speed], ...]
cur_time  = -1
in_window = False
t0        = time.time()
bytes_read = 0
last_pct   = -1

with open(FCD, 'r', buffering=1 << 23) as f:
    for line in f:
        bytes_read += len(line)

        if '<timestep' in line:
            m = RE_TIME.search(line)
            if m:
                cur_time  = float(m.group(1))
                in_window = (MORNING_START - BUFFER <= cur_time
                             <= MORNING_END   + BUFFER)
                if cur_time > MORNING_END + BUFFER:
                    break  # stop early — nothing more to collect

                # Progress report
                pct = int(bytes_read / fcd_size * 100)
                if pct > last_pct:
                    last_pct = pct
                    h = cur_time / 3600
                    print(f"  {h:.2f}h | {pct}% | "
                          f"{len(vehicles)} vehicles tracked | "
                          f"{time.time()-t0:.0f}s", end='\r')

        elif in_window and '<vehicle' in line:
            vm = RE_VID.search(line)
            xm = RE_X.search(line)
            ym = RE_Y.search(line)
            sm = RE_SPEED.search(line)
            if vm and xm and ym and sm:
                vid = vm.group(1)
                if vid not in vehicles:
                    vehicles[vid] = []
                vehicles[vid].append([
                    int(cur_time),
                    round(float(xm.group(1)), 5),
                    round(float(ym.group(1)), 5),
                    round(float(sm.group(1)) * 3.6, 1)
                ])

print(f"\n  Scan complete in {time.time()-t0:.0f}s")
print(f"  Total vehicles seen: {len(vehicles)}")

# Filter: keep only vehicles with enough waypoints
valid = {vid: wps for vid, wps in vehicles.items()
         if len(wps) >= MIN_WAYPOINTS}
print(f"  Vehicles with {MIN_WAYPOINTS}+ waypoints: {len(valid)}")

# If too many, sample the most active ones (most waypoints = longest trip)
if len(valid) > MAX_VEHICLES:
    sorted_vids = sorted(valid.keys(), key=lambda v: len(valid[v]), reverse=True)
    valid = {vid: valid[vid] for vid in sorted_vids[:MAX_VEHICLES]}
    print(f"  Sampled top {MAX_VEHICLES} most active vehicles")

# Write output
deckgl = [{'id': vid, 'waypoints': wps} for vid, wps in valid.items()]
with open(OUT, 'w') as f:
    json.dump(deckgl, f, separators=(',', ':'))

size_mb = os.path.getsize(OUT) / 1e6
print(f"  Written {len(deckgl)} vehicles → {OUT} ({size_mb:.1f} MB)")
print("Done!")
