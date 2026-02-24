# Chisinau Commute Upgrade
## Airport ↔ Botanica Corridor Simulation

A data-science pipeline to simulate, measure, and optimise traffic flow on the
Airport–Botanica corridor in Chișinău using **SUMO** + **Python** + **TraCI**.

---

## Quick Start

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Make sure SUMO is on your PATH
#    Download: https://sumo.dlr.de/docs/Downloads.php
sumo --version

# 3. Download OSM corridor data
python src/network/extract_osm.py --config src/config/scenario.yaml

# 4. Build SUMO network
python src/network/build_sumo_net.py --config src/config/scenario.yaml

# 5. Generate OD matrix
python src/demand/od_matrix.py --config src/config/scenario.yaml

# 6. Generate trips (+ duarouter routing)
python src/demand/trip_generation.py --config src/config/scenario.yaml

# 7. Generate baseline signal plans
python src/demand/signals.py --config src/config/scenario.yaml

# 8. Run baseline simulation (fixed-time signals)
python src/simulation/run_sumo.py --config src/config/scenario.yaml

# 9. Extract KPIs and bottlenecks
python src/simulation/metrics.py --config src/config/scenario.yaml

# 10. Generate baseline report + charts
python src/analysis/baseline_report.py --config src/config/scenario.yaml

# 11. Run adaptive simulation (TraCI max-pressure controller)
python src/simulation/run_sumo.py --config src/config/scenario.yaml \
       --mode traci --controller adaptive_pressure

# 12. Compare baseline vs adaptive
python src/analysis/compare_runs.py \
       --baseline data/outputs \
       --compare  data/outputs/adaptive \
       --out       data/outputs/comparison
```

---

## Project Structure

```
chisinau-commute-upgrade/
├── data/
│   ├── osm/                  # Raw OSM export
│   ├── sumo_net/             # SUMO network files (.net.xml, .poly.xml)
│   ├── demand/               # OD matrix CSV + trip/route XML
│   ├── signals/              # Fixed-time signal plans (.add.xml)
│   └── outputs/              # Simulation outputs + figures
│
├── src/
│   ├── config/
│   │   └── scenario.yaml     # ← All parameters live here
│   │
│   ├── network/
│   │   ├── extract_osm.py    # Download corridor OSM via Overpass
│   │   ├── build_sumo_net.py # netconvert + patch list
│   │   └── clean_network_notes.md  # QA checklist
│   │
│   ├── demand/
│   │   ├── zones.py          # TAZ definitions + edge assignment
│   │   ├── od_matrix.py      # Gravity model → OD CSV
│   │   ├── trip_generation.py# OD CSV → SUMO trips + duarouter
│   │   └── signals.py        # Fixed-time signal XML generator
│   │
│   ├── simulation/
│   │   ├── run_sumo.py       # Batch or TraCI simulation runner
│   │   ├── metrics.py        # Parse outputs → KPIs + bottlenecks
│   │   └── controllers/
│   │       ├── adaptive_pressure.py  # Max-pressure adaptive controller
│   │       └── fixed_time.py         # No-op pass-through
│   │
│   └── analysis/
│       ├── baseline_report.py   # Reproducible report + figures
│       ├── compare_runs.py      # Before/after comparison
│       └── plots.py             # Shared plot utilities
│
├── notebooks/
│   └── corridor_analysis.ipynb  # Interactive exploration
│
├── requirements.txt
└── README.md
```

---

## Milestones

| Milestone | Goal | Status |
|-----------|------|--------|
| **A1** | Corridor network imported and QA'd | ☐ |
| **A2** | 5–10 signalised intersections with fixed-time plans | ☐ |
| **A3** | Synthetic OD demand generated | ☐ |
| **A4** | Full-day simulation running | ☐ |
| **B1** | KPIs and bottlenecks extracted | ☐ |
| **B2** | Baseline report reproducible | ☐ |
| **C1** | Max-pressure adaptive controller live | ☐ |
| **C2** | Baseline vs adaptive comparison | ☐ |

---

## Key Configuration

All experiment parameters live in **`src/config/scenario.yaml`**:

- `network.bbox` — bounding box for OSM download
- `simulation.peak_hours` — morning / midday / evening windows
- `demand.total_daily_vehicles` — scale OD demand up/down
- `demand.peak_multipliers` — shape the temporal distribution
- `signals.pilot_intersections` — cycle lengths and phase lists
- `signals.adaptive` — min/max green, pressure threshold

---

## KPIs Produced

**Corridor**
- Mean / median / p90 travel time
- Total delay (vehicle-hours)
- Completion rate

**Network**
- Per-edge speed heatmap (edge × 15-min bin)
- Top-10 bottleneck ranking (congested minutes)
- Mean queue length per approach (via edge density)

---

## Validation Sanity Checks

Before calibrating against real data, verify:
- [ ] Free-flow travel time looks reasonable (not < 2 min, not > 2 h)
- [ ] Morning/evening peaks create visible congestion in heatmap
- [ ] Bottlenecks occur at intuitively obvious intersections
- [ ] Increasing `total_daily_vehicles` increases delay **nonlinearly**

---

## Known Risks

| Risk | Mitigation |
|------|-----------|
| OSM lane counts wrong | Patch list in `build_sumo_net.py`; `clean_network_notes.md` |
| Signal phase strings don't match junction | Verify in SUMO-GUI after netconvert |
| OD matrix drives unrealistic routes | Tune `beta` in gravity model; check duarouter warnings |
| Spillback not captured | Use `edgedata` density; add detectors near key intersections |

---

## Dependencies

- **SUMO ≥ 1.18** — https://sumo.dlr.de
- Python 3.10+ with packages in `requirements.txt`
