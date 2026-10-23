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

## Video data

**Goal.** Drivers upload dashcam videos. From each video, without relying on GPS,
we work out where the car was every second on the real street map and what the
traffic was like: speed, stops and their cause, queues before traffic lights,
lanes and surrounding vehicles. Those measurements later replace the simulation's
guessed demand and signal timings.

**What the uploader gives:** the video (date and time are read from the video
itself), a **start point** and a **destination** — pins on a street (yards and car
parks are not in the street map). Nothing else: no GPS track, no camera settings, no
tuning per dashcam. Videos that do carry GPS (burnt into the
image by the dashcam) are used as answer keys to measure how well the video-only
method works.

### Where we are (2026-10-23)

| | Status |
|---|---|
| Detect vehicles, traffic lights, crosswalks, lanes, curbs from the video | ✅ works in daylight; at night lights and crosswalks are weak |
| Own motion: moving / stopped, speed, turns, roundabouts | ✅ moving / stopped 88 % vs GPS; turn angles right after calibration; ⚠️ speed rough (± ~11 km/h, correlation 0.5) |
| Which **route** was driven (video only, start + destination) | ✅ daylight: right route on both daylight drives (out of 300 candidates) |
| Which **junction at which moment** (target ≈ 80 %) | ⚠️ 44 % in daylight (8 of 18 junctions); position within 30 m at 50 % of junction moments — see plan, step 1 |
| Night drives | ❌ rotation drift; events are noise — postponed |
| Traffic measures (stops, queues before signals, speeds per street) | ✅ from GPS drives, combined over all drives in `data/video/aggregate/` |
| **Spatial view**: what the camera sees, projected onto the ground, vs what the map expects | ✅ depth within ~7 % of GPS after calibration against the map |
| **Camera self-calibration** (no specs, no GPS): field of view from map turns, depth from crosswalk approaches | ✅ dashcam "120°" measured as 99° (GPS check agrees); refuses when the evidence is inconsistent |
| Dashboard: video · spatial view · map (GPS point and video-only point) | ✅ `data/video/dashboard.html` |

### Models and data sources — what detects what

| Data | Model / method | Runs where | Notes |
|---|---|---|---|
| Cars, buses, trucks, motorcycles, traffic lights (boxes) | **YOLO11s** (Ultralytics, COCO-trained) | laptop CPU | every 6th frame (5 per s); bonnet detections dropped |
| Same vehicle across frames (track IDs) | **ByteTrack** (`bytetrack_dashcam.yaml`) | laptop | |
| Traffic light colour | colour heuristic (brightest saturated hue in the box) | laptop | daylight only in practice |
| Own motion: moving / stopped, speed, rotation, horizon | **optical flow** (OpenCV Lucas–Kanade, sparse corners) | laptop | speed: flow between adjacent frames; rotation: over 0.2 s |
| Other vehicles: parked / oncoming / same direction / crossing | **dense optical flow** (OpenCV Farneback): box motion vs road under it | laptop | |
| Crosswalks, lane markings, curbs, road, sidewalk, guard rails, poles, barriers | **Mask2Former Swin-L, Mapillary Vistas** (Meta, via Hugging Face) | free **Google Colab T4 GPU** (~2 min per drive), laptop fallback | weights licensed for non-commercial use (CC BY-NC) — fine here, see Licensing |
| Road width, side streets opening on the right | geometry on the segmentation maps (half-width ÷ distance below horizon) | laptop | camera-independent |
| Dashcam overlay: time, speed, GPS | **Apple Vision OCR** (macOS built-in) | laptop | GPS = answer key only |
| Street-name plates, named places | Apple Vision OCR + fuzzy match to OSM names | laptop | number plates detected and **deleted** (personal data, law 133/2011) |
| Streets, junctions, roundabouts, signals, crossings, lanes | **OpenStreetMap** (osmnx graph + `chisinau_full.osm`) | laptop | free, no paid traffic APIs |
| Route + position from video events | **event alignment** (dynamic programming, no ML) | laptop | ~30 s per drive |
| Camera field of view, depth scale | **self-calibration against the map**: video rotation vs map heading change over 12-s windows; crosswalk distances vs distance driven | laptop | ~1 min per drive (includes re-matching) |
| Spatial view (bird's-eye plan of the road ahead) | flat-road projection of the segmentation; horizon from where the road ends, camera height from car widths, calibrated depth | laptop | ~20 s per drive |

### The pipeline, step by step

`.venv-video/bin/python -m src.video.pipeline <drive>` runs every step in order (steps that
would take more than ~5 min on the laptop run on Google Colab);
`--from <step>` restarts from a step. Outputs go to `data/video/processed/<drive>/`.

1. **overlay** (`overlay.py`). OCR of the dashcam's burnt-in line, once per second,
   gives time, speed and GPS → `gps.parquet`. The video clock is kept even before the
   dashcam has a GPS fix. Misread digits (a 0 read as a 6 puts the car 1 km away)
   are dropped against the rolling median track. Used for evaluation and for traffic
   measures; skipped for videos without an overlay.
2. **perception** (`perception.py`). YOLO + ByteTrack + optical flow on every 6th
   frame → `detections.parquet` (boxes, tracks, light colour, box motion) and
   `ego.parquet` (own motion signals). Slowest step: ~1–2× the video length on the CPU.
3. **traffic** (`traffic.py`, `motion.py`). Own motion becomes moving / stopped, speed
   index, turns and stops. Vehicle tracks are labelled (oncoming, parked, same
   direction, crossing, overtaking) → `tracks.parquet`, `per_minute.csv`, `timeline.png`.
4. **scene** (`scene.py --colab`, then `--colab --dense`). Mask2Former segmentation on
   the Colab GPU, two passes: candidate frames → `scene.parquet` (class shares, crosswalk
   position, road scanlines, a small class map); and 2 frames per second with a fine class
   map of the road area → `scene_dense.parquet`, for the spatial view (~5 min for a
   12-min drive). Frames are cropped and resized before upload; the video never leaves
   the laptop.
5. **road** (`road_geometry.py`). Road width and side-street openings from the class
   maps → `road.parquet`.
6. **events** (`events.py`). The video as a list of **junction events**: turns (signed
   angle), roundabouts, crosswalks crossed, a traffic light passed close, stops,
   openings on the right. Cues within 8 s are one event → `events.parquet`.
7. **event_match** (`event_match.py`). The core of video-only localisation:
   - lists up to 300 candidate routes from start to destination (≤ 1.5× the shortest);
   - for each route, lists its **sites** in order: junctions with the turn made there,
     roundabouts as one manoeuvre, OSM signals and crossings;
   - aligns the video events with the sites. Matching costs are low when the turn
     direction and size, crosswalk, signal and roundabout agree, and when the distance
     driven since the **previous** match fits the map. Missed sites and false events
     cost according to how visible they are;
   - the cheapest route wins; matched sites anchor the per-second position.

   The route runs from the start pin to the destination pin themselves (pins snap to
   the nearest street segment).

   → `event_match.json`. With GPS it also scores itself (junction matching rate).
8. **calibrate** (`calibrate.py`). The camera calibrates itself against the map, with
   no camera specs and no GPS: the matched route's bends give the true heading changes,
   and the video's measured rotation over the same windows gives the lens's field of view.
   Then depth: approaching a crosswalk, its distance in the spatial view must shrink by
   the distance driven. Applied only when the evidence agrees; then events and matching
   are re-run → `calibration.json` (used by every later step).
9. **bev** (`bev.py`). The **spatial view**, every half second: a bird's-eye plan of the
   ~40 m ahead. Colour = what the camera sees (road, lane markings, crosswalks,
   sidewalk, curbs, grass verges; vehicles from their boxes). White = the OSM streets,
   crossings and signals as the map expects them from the **video-only** position and
   heading; thin green = the same from the GPS position, when there is GPS. Where colour
   and white disagree, the video-only position is wrong → `bev.mp4`.
10. **position** (`position.py`). Where the car is every second, snapped to an OSM
   street (from GPS if the video has it, else from the video match) → `trace.parquet`.
11. **lanes** (`lanes.py`). Lanes visible and our lane, from road scanlines →
   `lanes.parquet`, `lanes_by_street.csv`.
12. **signs** (`signs.py`). OCR every 2 s: street names and named places matched to
    OSM; number plates deleted → `text.parquet`.
13. **streets** (`street_stats.py`). Each stop with its cause (queue for the signal
    ahead, red light, other), vehicles ahead, and per-street distance, speed and
    standing time; also combined over all drives → `stops_detail.csv`, `streets.csv`,
    `data/video/aggregate/{stops,streets,signals}.csv`.
14. **dashboard** (`dashboard/build.py`). One page for all drives, three boxes in sync:
    the video, the spatial view, and the map with the GPS point and the video-only point; plus the timeline, events, stops and queues → `data/video/dashboard.html`.

Older method, kept for comparison: the particle filter (`localize.py`, `evaluate.py`)
works partly in daylight and fails at night; it is no longer in the default pipeline.

### Test drives

| Drive | Camera | Light | GPS | Video-only result |
|---|---|---|---|---|
| `2026_10_22_10_21AM` home → Zity Mall, ~5 km | iPhone | day | no (truth = Google directions) | full correct route (99 % of it, all within 30 m); clear margin over the runner-up |
| `2026_10_22_18_16` Zity Mall → home, 7 km | dashcam | dusk / night | yes | 81 % of junctions on the chosen route; 0 % at the right moment |
| `2026_10_23_10_06` home → Zity Mall, same route, ~5 km | dashcam | day | yes | right route (20 of 21 junctions on it); 44 % of junctions at the right moment; position within 30 m at 50 % of junction moments; median error 59 m; lens calibrated 120° → 99°. Pins: after the U-turn on Mircești → the turn into the mall |

Measured on the evening dashcam drive against GPS: 56 % of junctions get a video cue;
crosswalk precision 83 %; all 5 real turns caught, plus 6 false turns. On the morning
drive, moving / stopped agrees with GPS 88 % of the time and the video odometer is off
by a median of 103 m over 5.3 km. Details, methods and the findings log:
[docs/video_pipeline.md](docs/video_pipeline.md).

### Plan

In priority order. Review and reorder this list as priorities change.

1. **Junction detection in daylight, through the spatial view** — the current
   bottleneck. About half of the junctions show no cue or only a weak "side street
   opens" cue, so the alignment skips them.
   1. ~~*Calibrate the depth scale against the map*~~ — done (2026-10-23): field of view
      from map turns, depth from crosswalk approaches; depth within ~7 % of GPS.
   1b. *Tesla-style surround view*: one rendered scene of the road around the car — road
      edges, lane lines, crosswalks, other vehicles as 3D boxes where they are — replacing
      the flat "ahead" plan. (A straightened strip of the whole drive was tried and dropped
      from the dashboard: `strip.py` stays as a possible detector input.)
   2. *Review together*: you watch the spatial view and mark where a junction or
      feature is missed; that drives the detector.
   3. *Detector on the ground plan*: road opening left / right at a measured distance
      ahead, crosswalks, stop lines — compared with the junction the map expects next.
      Also vehicles crossing ahead.
   4. *Traffic-sign types*: a yield sign (red triangle), stop, priority road, roundabout
      signs announce a junction tens of metres ahead and say which kind. Needs a
      sign-type detector (e.g. a YOLO model trained on the Mapillary traffic-sign
      dataset, on Colab); today only "a sign" is segmented, not its type.

   Target: ≈ 80 % of junctions at the right moment (±8 s) on `2026_10_23_10_06`.
2. **Tighter position between junctions.** After the first alignment, re-fit the
   odometer scale stretch by stretch from the matched junctions, then align again.
   Target: position within 30 m at most junction moments (now 22 %).
3. **More daylight drives with GPS** as answer keys: different routes, and
   other dashcams to check that nothing is camera-specific.
4. **Traffic data for the simulation.** Other vehicles as measurements (flows,
   oncoming counts), signal waits and queue lengths per junction, speeds per street;
   then feed `data/video/aggregate/` into the SUMO calibration.
5. **Night drives.** Rotation drift, street lamps taken for traffic lights. Use
   traffic-light colour, headlights and cross traffic at night.
6. **3D reconstruction** of the road ahead (monocular depth on Colab), after the flat
   spatial view: exact lanes, distances, kerbs and slopes.
7. **Platform readiness:**
   - upload flow (video + start and destination pins on a street);
   - perception on Colab too (rule: anything > 5 min on the laptop runs on Colab);
   - privacy review (plates are deleted today; faces too?);
   - disk cleanup (joined videos duplicate the original clips).

### Licensing

This is a personal, non-commercial project. The Mask2Former Mapillary Vistas weights are
licensed for non-commercial use only, which fits. If the project ever becomes
commercial, the segmentation model must be replaced or licensed first.

## Repository layout

```
configs/
  scenario.yaml          simulation scenario
  drives.yaml            recorded drives: video, camera, ground truth
src/
  network/ demand/ simulation/ analysis/     SUMO simulation
  video/
    pipeline.py          all steps for one drive, in order (--from <step>)
    drives.py            drive registry (configs/drives.yaml), video clock
    compute.py           CPU limits, progress bars
    overlay.py           dashcam burnt-in time / speed / GPS (Apple Vision OCR) → answer key
    perception.py        YOLO11s + ByteTrack + optical flow
    motion.py            the car's own motion: moving, speed, rotation, turns, stops
    traffic.py           other road users: oncoming, parked, same direction…
    scene.py             Mask2Former segmentation (local or Google Colab GPU)
    road_geometry.py     road width, side streets opening
    events.py            the video as a list of junction events
    event_match.py       video-only route + position: align events with route sites
    roads.py, truth.py   OpenStreetMap street graph; truth route from driving directions
    position.py          where the car is every second, on OSM streets
    calibrate.py         camera self-calibration against the map (field of view, depth)
    bev.py               spatial view: what the camera sees vs what the map expects
    strip.py             (not in the pipeline) the drive stitched into one straightened strip
    lanes.py, signs.py   lanes; street names / named places read in the video
    street_stats.py      stops, queues before signals, speeds per street (+ all drives)
    localize.py, evaluate.py   older particle-filter localisation (comparison only)
    dashboard/           one page for all drives: synced video + map
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
.venv-video/bin/python -m src.video.roads features       # crossings + signals (once, needs data/osm/chisinau_full.osm)
.venv-video/bin/python -m src.video.scene download-model  # segmentation model (once, 826 MB)
```

1. Copy the video to `data/video/raw/` and add an entry to `configs/drives.yaml`.
2. Run the pipeline (perception takes ~2× the video length on a laptop CPU):
   ```bash
   .venv-video/bin/python -m src.video.pipeline 2026_10_22_10_21AM
   ```
3. Open `data/video/dashboard.html` in Chrome or Safari and pick the drive.

Tests: `.venv-video/bin/python -m pytest tests -q`

## Quick start — simulation

See [docs/simulation.md](docs/simulation.md) (SUMO 1.26, `requirements.txt`).
