#!/usr/bin/env python3
"""
Fast Post-processing pipeline for Chișinău SUMO simulation
Uses line-by-line parsing with 30s sampling for speed.
"""

import os, sys, json, re, math
from collections import defaultdict
import time

print("=" * 65)
print("  Chișinău Traffic Simulation — Post-Processing Pipeline")
print("=" * 65)

BASE    = os.path.dirname(os.path.abspath(__file__))
NET     = os.path.join(BASE, 'data/sumo_net/network.net.xml')
FCD     = os.path.join(BASE, 'data/outputs/fcd_full.xml')
OUT_DIR = os.path.join(BASE, 'data/outputs')

MORNING_START, MORNING_END = 7*3600,  9*3600
EVENING_START, EVENING_END = 17*3600, 20*3600
SAMPLE_EVERY = 30

print(f"\n[1/5] Parsing FCD (sampling every {SAMPLE_EVERY}s in peak windows)...")
t0 = time.time()

edge_speed_sum   = defaultdict(float)
edge_speed_count = defaultdict(int)

current_time = -1
in_peak      = False
sample_this  = False

RE_TIME  = re.compile(r'time="([\d.]+)"')
RE_LANE  = re.compile(r'lane="([^"]+)"')
RE_SPEED = re.compile(r'speed="([\d.]+)"')

fcd_size   = os.path.getsize(FCD)
bytes_read = 0
last_report = -1

with open(FCD, 'r', buffering=1 << 23) as f:
    for line in f:
        bytes_read += len(line)

        if '<timestep' in line:
            m = RE_TIME.search(line)
            if m:
                current_time = float(m.group(1))
                in_peak = (MORNING_START <= current_time <= MORNING_END or
                           EVENING_START <= current_time <= EVENING_END)
                sample_this = in_peak and (int(current_time) % SAMPLE_EVERY == 0)

                h = current_time / 3600
                bucket = int(h * 2)
                if bucket > last_report:
                    last_report = bucket
                    pct = bytes_read / fcd_size * 100
                    elapsed = time.time() - t0
                    print(f"      {h:.1f}h sim | {pct:.1f}% file | "
                          f"{len(edge_speed_count)} edges | {elapsed:.0f}s elapsed")

        elif sample_this and '<vehicle' in line:
            lm = RE_LANE.search(line)
            sm = RE_SPEED.search(line)
            if lm and sm:
                edge_id = lm.group(1).rsplit('_', 1)[0]
                speed   = float(sm.group(1)) * 3.6
                edge_speed_sum[edge_id]   += speed
                edge_speed_count[edge_id] += 1

print(f"      Done in {time.time()-t0:.0f}s — "
      f"{len(edge_speed_count)} edges, "
      f"{sum(edge_speed_count.values())} samples")

print("\n[2/5] Computing congestion metrics + loading network...")
import sumolib, pandas as pd
net = sumolib.net.readNet(NET)

rows = []
for edge_id, count in edge_speed_count.items():
    mean_speed = edge_speed_sum[edge_id] / count
    try:
        edge = net.getEdge(edge_id)
        ff     = max(edge.getSpeed() * 3.6, 10.0)
        length = edge.getLength()
    except:
        ff, length = 50.0, 50.0
    rows.append({
        'edge_id':        edge_id,
        'mean_speed_kmh': round(mean_speed, 2),
        'mean_speed_rel': round(mean_speed / ff, 4),
        'freeflow_kmh':   round(ff, 1),
        'length_m':       round(length, 1),
        'sample_count':   count,
        'peak_flow':      round(count / 4.0, 1),
    })

df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, 'edge_congestion.csv')
df.to_csv(csv_path, index=False)
print(f"      {len(df)} edges → {csv_path}")
print(f"      Mean speed ratio : {df.mean_speed_rel.mean():.3f}")
print(f"      Median speed     : {df.mean_speed_kmh.median():.1f} km/h")
print(f"      SR < 0.30        : {(df.mean_speed_rel < 0.30).sum()} edges")
print(f"      SR < 0.50        : {(df.mean_speed_rel < 0.50).sum()} edges")

print("\n[3/5] Generating roads_congestion.geojson...")

congestion = {r['edge_id']: r for _, r in df.iterrows()}

def sr_to_color(sr):
    if sr < 0.25: return [204, 0,   0  ]
    if sr < 0.45: return [255, 68,  0  ]
    if sr < 0.60: return [255, 136, 0  ]
    if sr < 0.75: return [255, 187, 0  ]
    if sr < 0.90: return [136, 204, 0  ]
    return               [0,   170, 68 ]

def offset_line(coords, offset_m):
    result = []
    n = len(coords)
    for i, pt in enumerate(coords):
        if i == 0:     dx,dy = coords[1][0]-coords[0][0],   coords[1][1]-coords[0][1]
        elif i == n-1: dx,dy = coords[-1][0]-coords[-2][0], coords[-1][1]-coords[-2][1]
        else:          dx,dy = coords[i+1][0]-coords[i-1][0], coords[i+1][1]-coords[i-1][1]
        L = math.sqrt(dx*dx + dy*dy)
        if L == 0: result.append(pt); continue
        result.append([pt[0]+(-dy/L)*offset_m/75000, pt[1]+(dx/L)*offset_m/111000])
    return result

def base_id(eid):
    return re.sub(r'^-?(\d+).*', r'\1', eid)

roundabout_edges = {eid for rb in net.getRoundabouts() for eid in rb.getEdges()}

groups = defaultdict(list)
for e in net.getEdges():
    eid = e.getID()
    if eid.startswith(':') or eid not in congestion: continue
    groups[(base_id(eid), len(e.getLanes()), eid.startswith('-'))].append(e)

features = []
for (bid, n_lanes, is_neg), edges in groups.items():
    def seg_num(e):
        m = re.search(r'#(\d+)$', e.getID()); return int(m.group(1)) if m else 0
    edges = sorted(edges, key=seg_num)
    merged = []
    for e in edges:
        ll = [list(net.convertXY2LonLat(x,y)) for x,y in e.getShape()]
        if merged and ll:
            if abs(ll[0][0]-merged[-1][0])<5e-5 and abs(ll[0][1]-merged[-1][1])<5e-5:
                ll = ll[1:]
        merged.extend(ll)
    if len(merged) < 2: continue
    srs,flows,ws = [],[],[]
    for e in edges:
        row = congestion.get(e.getID())
        if row is None: continue
        srs.append(row['mean_speed_rel']); flows.append(row['peak_flow']); ws.append(e.getLength())
    if not srs: continue
    W = sum(ws)
    avg_sr   = sum(s*w for s,w in zip(srs,ws))/W
    avg_flow = sum(f*w for f,w in zip(flows,ws))/W
    color    = sr_to_color(avg_sr)
    is_rb    = any(e.getID() in roundabout_edges for e in edges)
    if n_lanes == 1:
        features.append({'type':'Feature',
            'geometry':{'type':'LineString','coordinates':merged},
            'properties':{'speed_ratio':round(float(avg_sr),3),'peak_flow':int(avg_flow),
                'color':color,'n_lanes':1,'is_roundabout':is_rb}})
    else:
        for li in range(n_lanes):
            off = (li-(n_lanes-1)/2.0)*3.2
            features.append({'type':'Feature',
                'geometry':{'type':'LineString','coordinates':offset_line(merged,off)},
                'properties':{'speed_ratio':round(float(avg_sr),3),'peak_flow':int(avg_flow),
                    'color':color,'n_lanes':n_lanes,'is_roundabout':is_rb}})

geojson_path = os.path.join(OUT_DIR, 'roads_congestion.geojson')
with open(geojson_path,'w') as f:
    json.dump({'type':'FeatureCollection','features':features}, f, separators=(',',':'))
print(f"      {len(features)} features → {geojson_path}")

print("\n[4/5] Generating trips_deckgl.json (morning peak)...")
t1 = time.time()

RE_VID = re.compile(r'\bid="([^"]+)"')
RE_X   = re.compile(r'\bx="([\d.]+)"')
RE_Y   = re.compile(r'\by="([\d.]+)"')

vehicles   = {}
cur_time   = -1

with open(FCD, 'r', buffering=1 << 23) as f:
    for line in f:
        if '<timestep' in line:
            m = RE_TIME.search(line)
            if m:
                cur_time = float(m.group(1))
                if cur_time > MORNING_END + 120:
                    break
        elif MORNING_START-60 <= cur_time <= MORNING_END+60 and '<vehicle' in line:
            vm=RE_VID.search(line); xm=RE_X.search(line)
            ym=RE_Y.search(line);  sm=RE_SPEED.search(line)
            if vm and xm and ym and sm:
                vid = vm.group(1)
                if vid not in vehicles: vehicles[vid] = []
                vehicles[vid].append([int(cur_time),
                    round(float(xm.group(1)),6), round(float(ym.group(1)),6),
                    round(float(sm.group(1))*3.6,1)])

deckgl = [{'id':vid,'waypoints':wps} for vid,wps in vehicles.items() if len(wps)>=5]
deckgl_path = os.path.join(OUT_DIR, 'trips_deckgl.json')
with open(deckgl_path,'w') as f:
    json.dump(deckgl, f, separators=(',',':'))
print(f"      {len(deckgl)} vehicles → {deckgl_path} ({time.time()-t1:.0f}s)")

print("\n[5/5] Validation vs Google Maps ground truth...")

ground_truth = {
    'Calea Ieșilor → Centru': {'bbox':(28.830,28.855,46.960,46.975),'google':8.6},
    'Botanica → Primărie':    {'bbox':(28.840,28.870,46.955,46.990),'google':7.1},
    'Moscova → UTM':          {'bbox':(28.845,28.870,47.005,47.030),'google':8.8},
    'Alba Iulia → Bd. Dacia': {'bbox':(28.800,28.860,47.005,47.045),'google':10.7},
    'Ciocana → Centru':       {'bbox':(28.890,28.950,46.990,47.030),'google':9.7},
    'Muncești → Bd. Ștefan':  {'bbox':(28.820,28.865,46.950,46.985),'google':12.5},
}

edge_coords = {}
for _, row in df.iterrows():
    try:
        e = net.getEdge(row['edge_id'])
        lon,lat = net.convertXY2LonLat(*e.getShape()[0])
        edge_coords[row['edge_id']] = (lon, lat, row['mean_speed_kmh'])
    except: pass

print(f"\n  {'Corridor':<30} {'Google':>7} {'Sim':>7} {'Ratio':>6}  Status")
print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*6}  {'-'*12}")
for name, info in ground_truth.items():
    lon0,lon1,lat0,lat1 = info['bbox']
    speeds = [spd for _,(lon,lat,spd) in edge_coords.items()
              if lon0<=lon<=lon1 and lat0<=lat<=lat1]
    if speeds:
        sim   = sum(speeds)/len(speeds)
        ratio = sim/info['google']
        status = "✅ GOOD" if ratio<1.5 else ("⚠️  HIGH" if ratio<2.5 else "❌ TOO FAST")
        print(f"  {name:<30} {info['google']:>7.1f} {sim:>7.1f} {ratio:>6.1f}x  {status}")
    else:
        print(f"  {name:<30} {info['google']:>7.1f} {'N/A':>7} {'?':>6}   ⚠️  NO DATA")

print("\n" + "="*65)
print("  Pipeline complete!")
print("="*65)
