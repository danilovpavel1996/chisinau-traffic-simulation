# Video pipeline — methods and findings

Goal: turn dashcam / phone videos uploaded by Chișinău drivers into traffic data
(where the car was, how fast, where it stopped and why, what traffic was around it)
to calibrate the city simulation with measurements instead of hand-tuned numbers.

## Steps

| Step | Module | Output (`data/video/processed/<drive>/`) |
|---|---|---|
| Detection, tracking, optical flow | `src/video/perception.py` | `detections.parquet`, `ego.parquet`, `meta.json` |
| Own motion: speed, rotation, turns, stops | `src/video/motion.py` | (used by the steps below) |
| Other vehicles: oncoming / parked / same direction…, counts | `src/video/traffic.py` | `tracks.parquet`, `stops.csv`, `turns.csv`, `per_minute.csv`, `timeline.png` |
| Map matching (particle filter on OSM) | `src/video/localize.py` | `match.json` |
| Synced video + map viewer | `src/video/viewer/` | `viewer.html`, `viewer_720p.mp4` |

Run everything: `.venv-video/bin/python -m src.video.pipeline <drive>`.
Drives and their settings live in `configs/drives.yaml`.

## Localisation method

A phone video carries **one** GPS fix (start position) and the start time. The rest
of the route is reconstructed from the video:

- **Odometry** — per second: rotation (horizontal-flow intercept, corrected for camera
  mis-alignment) and a speed index (road-plane flow where the asphalt is visible,
  scene flow × local ratio elsewhere).
- **Particle filter** on the OSM drivable graph (4 000 hypotheses of edge, offset,
  distance scale, heading offset and its drift rate). Each second the hypotheses move
  by their own scale × speed index and are weighted by
  - heading **change over 8 s**, map vs video (turns, curves, roundabouts),
  - absolute heading with a per-hypothesis drift rate (pins the street),
  - stops: a new stop is most likely just before a junction (lights, give-way),
  - implausible speeds (> 90 km/h) — soft penalty,
  - driving against an OSM one-way — penalty, not forbidden (tags can be wrong).
- 5 independent runs; the one with the highest **evidence** (log marginal likelihood)
  is kept — chosen without looking at the ground truth.

## Results — drive 2026-10-22 (home → Zity Mall, 5.3 km, 15 min, iPhone)

Ground truth: route confirmed by the driver (Google Maps), see `configs/drives.yaml`.

| Version | Median error | Within 30 m | Covers true route |
|---|---|---|---|
| Turn-event beam search (removed) | — | — | ~40 %, stops early |
| Particle filter, absolute heading only | 147 m | 44 % | 41 % |
| + stops near junctions, speed sanity | 2 m | 55–92 % | 40–48 % |
| + windowed heading, drift-rate state, adaptive scale (current) | **2 m** | **89 %** | **62 %** |

Median error is at the metric's resolution (true route sampled every 5 m). The match
follows Mircești → roundabout (correct first exit) → Mihai Viteazul, but undercounts
distance there and never reaches the final right turn to Calea Ieșilor.

## Findings log

What was learned, including what did not work — kept so it is not re-tried blindly.

1. **Phone set-up and parking manoeuvres** (first ~95 s, last ~60 s) look like turns.
   → `street_window_s` per drive.
2. **Rotation leaked forward motion** (median of horizontal flow): a steady fake left
   turn whenever moving. → fit `dx = rotation + slope·x`, rotation = intercept.
   Still: straight driving produces drift (−97° on straight Mihai Viteazul), while turns
   are measured about right (64° vs ~70°). → compare **short-term heading changes**.
3. **Essential-matrix visual odometry** (OpenCV, frame-to-frame) was *worse* on
   straights (+426° on Mircești). A real SLAM system would be needed; not pursued.
4. **Scene flow undercounts distance on wide streets** (far buildings move slowly in
   the image) — 2.3–2.65× between the first part of the drive and Mihai Viteazul.
5. **Asphalt patch above the bonnet** as speed sensor: failed — with the phone mounted
   low the patch is bonnet / windscreen reflection (reads ~0).
6. **Road-plane bands** (flow ÷ (row − horizon)²): only marginally better than scene
   flow (2.25×). An earlier "consistent within 8 %" result was an artefact of wrong
   segment boundaries and was retracted.
7. **Gap-filling ratio from a few outliers** produced 50× phantom speed (400 m of
   distance that never happened) → minimum data per window + clamping.
8. **Hard speed-cap penalty** punished correct hypotheses for measurement noise and
   shrank the distance scale → soft penalty.
9. **Roundabouts** judged edge by edge, and **chains of gentle ramp bends**, produced
   wrong exits in the turn-event matcher; the particle filter handles both by
   comparing heading changes over time.
10. **OSM one-way tag** on Mitropolit Dosoftei contradicts the driven route → contra-flow
    allowed at a penalty.

## Next

- Drives with **phone gyroscope + GPS** logged alongside the video (e.g. Sensor
  Logger): exact rotation, and per-second ground truth to measure the vision-only
  pipeline honestly.
- **Road / lane segmentation** for metric speed (lane width as a ruler), lane count
  (match to OSM `lanes`) and lane position (turn prior).
- More drives before further tuning — every threshold here comes from one drive.
