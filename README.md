# Chișinău Traffic

Modelling Chișinău's traffic to find ways to relieve its congestion — from data,
not hand-tuned numbers.

Two parts:

| | What | Status |
|---|---|---|
| **Simulation** | City-scale SUMO model: 150k trips/day, 265 signalised intersections, deck.gl visualisation | v2, Feb 2026 — demand is a hand-scaled 8-district OD matrix → [docs/simulation.md](docs/simulation.md) |
| **Video data** | Dashcam / phone videos → route on the map, speeds, stops, traffic lights, surrounding vehicles | in progress, Oct 2026 → [docs/video_pipeline.md](docs/video_pipeline.md) |

The video pipeline exists to replace the simulation's guessed demand and signal
timings with measurements crowdsourced from drivers.

## Repository layout

```
configs/
  scenario.yaml          simulation scenario
  drives.yaml            recorded drives: video, camera, ground truth
src/
  network/ demand/ simulation/ analysis/     SUMO simulation
  video/
    perception.py        detection + tracking + optical flow
    motion.py            the car's own motion: speed, rotation, turns, stops
    traffic.py           other road users
    roads.py             OpenStreetMap street graph
    localize.py          particle-filter map matching
    viewer/              synced video + map viewer
    pipeline.py          all steps for one drive
docs/                    methods, findings, simulation guide
tests/                   fast unit tests
notebooks/
data/                    inputs and outputs (large files gitignored)
```

## Quick start — video

Two environments: the simulation runs on any Python ≥3.10; the video pipeline needs
Python 3.11 (PyTorch on Intel Macs stops at 2.2 / NumPy < 2).

```bash
python3.11 -m venv .venv-video && .venv-video/bin/pip install -r requirements-video.txt
.venv-video/bin/python -m src.video.roads build          # street graph (once)
```

1. Copy the video to `data/video/raw/` and add an entry to `configs/drives.yaml`.
2. Run the pipeline (perception takes ~2× the video length on a laptop CPU):
   ```bash
   .venv-video/bin/python -m src.video.pipeline 2026_10_22_10_21AM
   ```
3. Open `data/video/processed/<drive>/viewer.html` in Chrome or Safari.

Tests: `.venv-video/bin/python -m pytest tests -q`

## Quick start — simulation

See [docs/simulation.md](docs/simulation.md) (SUMO 1.26, `requirements.txt`).
