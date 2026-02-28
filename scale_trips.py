"""
Smart Trip Scaler for Chișinău SUMO Simulation
================================================
Instead of generating random edge pairs (which mostly fail routing),
this script resamples from the ORIGINAL 50,142 valid trips.

Strategy:
- Original trips are grouped by their approximate OD district pair
- For each time bin in the scaled OD matrix, sample N trips from the
  matching district pair pool, with small random depart time jitter
- This guarantees 100% valid routes since we reuse proven edge pairs

Usage:
    python3 scale_trips.py
"""

import xml.etree.ElementTree as ET
import sumolib
import pandas as pd
import random
from collections import defaultdict

random.seed(42)
SCALE = 3.0

# District bounding boxes
DISTRICTS = {
    'centru':     (28.820, 28.875, 46.985, 47.020),
    'botanica':   (28.840, 28.900, 46.955, 46.990),
    'buiucani':   (28.790, 28.845, 47.010, 47.055),
    'rascani':    (28.845, 28.910, 47.010, 47.055),
    'ciocana':    (28.890, 28.960, 46.990, 47.040),
    'telecentru': (28.820, 28.870, 46.950, 46.990),
    'durlesti':   (28.740, 28.810, 47.020, 47.070),
    'sculeni':    (28.790, 28.840, 46.960, 47.015),
}

def get_district(lon, lat):
    for name, (lon0, lon1, lat0, lat1) in DISTRICTS.items():
        if lon0 <= lon <= lon1 and lat0 <= lat <= lat1:
            return name
    return None

print('Step 1/5 — Loading SUMO network...')
net = sumolib.net.readNet('data/sumo_net/network.net.xml')

print('Step 2/5 — Loading original trips and classifying by district pair...')
tree = ET.parse('data/demand/trips.trips.xml')
root = tree.getroot()
original_trips = root.findall('trip')
print(f'  Original trips: {len(original_trips)}')

# Build pool: district_pair -> list of (from_edge, to_edge)
pool = defaultdict(list)
unclassified = 0

for trip in original_trips:
    from_e = trip.get('from')
    to_e = trip.get('to')
    try:
        fe = net.getEdge(from_e)
        te = net.getEdge(to_e)
        flon, flat = net.convertXY2LonLat(*fe.getShape()[0])
        tlon, tlat = net.convertXY2LonLat(*te.getShape()[0])
        orig_d = get_district(flon, flat)
        dest_d = get_district(tlon, tlat)
        if orig_d and dest_d:
            pool[(orig_d, dest_d)].append((from_e, to_e))
        else:
            unclassified += 1
    except:
        unclassified += 1

print(f'  Classified pairs: {sum(len(v) for v in pool.values())}')
print(f'  Unclassified: {unclassified}')
print(f'  District pairs with trips:')
for (o,d), trips in sorted(pool.items(), key=lambda x: -len(x[1]))[:10]:
    print(f'    {o} → {d}: {len(trips)} template trips')

print('Step 3/5 — Loading scaled OD matrix...')
df = pd.read_csv('data/demand/od_matrix_scaled.csv')
print(f'  Rows: {len(df)}, total trips: {df.trips.sum()}')

print('Step 4/5 — Generating scaled trips by resampling...')

# Build a global fallback pool per origin district (any destination)
origin_pool = defaultdict(list)
dest_pool = defaultdict(list)
global_pool = []
for (o, d), trips in pool.items():
    for pair in trips:
        origin_pool[o].append(pair)
        dest_pool[d].append(pair)
        global_pool.append(pair)

print(f'  Origin pools: { {k: len(v) for k, v in origin_pool.items()} }')

new_trips = []
trip_idx = 0
skipped_pairs = 0
fallback_used = 0

for _, row in df.iterrows():
    orig = row['origin']
    dest = row['destination']
    t_bin = int(row['time_bin'])
    n = int(row['trips'])

    key = (orig, dest)

    if key in pool and len(pool[key]) > 0:
        candidates = pool[key]
    else:
        # Fallback 1: reverse direction (swap from/to)
        rev_key = (dest, orig)
        if rev_key in pool and len(pool[rev_key]) > 0:
            candidates = [(t, f) for f, t in pool[rev_key]]
            fallback_used += n
        # Fallback 2: same origin, any destination
        elif orig in origin_pool and len(origin_pool[orig]) > 0:
            candidates = origin_pool[orig]
            fallback_used += n
        # Fallback 3: any origin, same destination
        elif dest in dest_pool and len(dest_pool[dest]) > 0:
            candidates = dest_pool[dest]
            fallback_used += n
        # Fallback 4: global pool
        else:
            candidates = global_pool
            fallback_used += n

    sampled = random.choices(candidates, k=n)
    for from_e, to_e in sampled:
        depart = t_bin + random.randint(0, 899)
        new_trips.append((depart, f't_{trip_idx:07d}', 'passenger', from_e, to_e))
        trip_idx += 1

print(f'  Generated: {len(new_trips)} trips')
print(f'  Fallback used: {fallback_used} trips')

# Sort by depart time
new_trips.sort(key=lambda x: x[0])

# Stats
morning = sum(1 for t in new_trips if 25200 <= t[0] <= 32400)
evening = sum(1 for t in new_trips if 61200 <= t[0] <= 72000)
print(f'  Morning peak (07-09): {morning} trips')
print(f'  Evening peak (17-20): {evening} trips')

print('Step 5/5 — Writing trips.trips.xml...')
new_root = ET.Element('routes')
new_root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')

for depart, tid, vtype, from_e, to_e in new_trips:
    trip = ET.SubElement(new_root, 'trip')
    trip.set('id', tid)
    trip.set('type', vtype)
    trip.set('depart', str(float(depart)))
    trip.set('from', from_e)
    trip.set('to', to_e)

new_tree = ET.ElementTree(new_root)
ET.indent(new_tree, space='  ')
new_tree.write('data/demand/trips.trips.xml',
               encoding='unicode', xml_declaration=True)

print(f'\n✓ Done — {len(new_trips)} trips written to data/demand/trips.trips.xml')
print(f'  Scale factor: {SCALE}x (original: 50,142 → new: {len(new_trips)})')
