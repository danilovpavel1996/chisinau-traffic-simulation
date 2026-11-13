# Chisinau Traffic

Modelling Chișinău's traffic to find ways to relieve its congestion — from data,
not hand-tuned numbers.

Two parts:

| | What | Status |
|---|---|---|
| **Simulation** | City-scale SUMO model: 150k trips/day, 265 signalised intersections, deck.gl visualisation | v2, Feb 2026 — demand is a hand-scaled 8-district OD matrix → [docs/simulation.md](docs/simulation.md) |
| **Video pipeline** | Dashcam / phone videos → route on the map, speeds, stops, traffic lights, surrounding vehicles — no GPS needed | paused Oct 2026 → [docs/video_pipeline.md](docs/video_pipeline.md) |

The video pipeline exists to replace the simulation's guessed demand and signal
timings with measurements crowdsourced from drivers.

---

## Video pipeline

### The idea

Drivers upload dashcam or phone videos of their commute. From each video, **without
relying on GPS**, we work out:

- **where** the car was every second on the real street map,
- **how fast** it was going,
- **where it stopped** and why (red light, queue, turning),
- **what traffic was like** around it (vehicles, lanes, queues before signals).

**What the uploader gives:** the video, a **start point** and a **destination** —
two pins on a map. Nothing else: no GPS track, no camera specs, no per-dashcam
tuning. Videos that do carry GPS (burnt into the image) are used only as answer
keys to measure how well the video-only method works.

### Results (Oct 2026, 5 test drives)

**Map position (video only, no GPS)**

| Drive | Conditions | Median error | Within 30 m |
|---|---|---|---|
| 10_06 home → mall | day, rain | 11.7 m | 90% |
| 10_02 home → mall | day, dry | 15.4 m | 78% |
| 17_08 downtown | day | 15.6 m | 72% |
| 19_26 bd. Moscova → Ciocana | **night**, wet | 16.4 m | 83% |
| 15_48 city centre | day | 15.2 m | 85% |

**Speed** (video odometer vs GPS): 0.94 correlation, 2.9 km/h median error (dry);
0.93, 4.8 km/h (rain).

**Junctions matched at the right moment:** 40 of 85 across all drives (47%).

**Lane placement** (lateral lock, map edge vs seen kerb): median error 0.09–0.71 m
after the lock, down from 1.2–3.6 m before.

### How it works

The pipeline runs 14 steps, from raw video to a browsable dashboard:

1. **Detection & tracking** — RT-DETRv2 + BoT-SORT on a free GPU (Modal / Kaggle / Colab): vehicles, traffic lights, crosswalks, lane markings, curbs
2. **Own motion** — speed from optical flow × metric depth; rotation, turns, stops
3. **Scene segmentation** — Mask2Former (Mapillary Vistas) on a free GPU
4. **Event-based route matching** — dynamic programming aligns video junction events (turns, crosswalks, signals, stops) with candidate routes from start to destination; the cheapest route wins; no ML
5. **Camera self-calibration** — field of view from map bends, depth scale from crosswalk approaches; no camera specs needed
6. **3D surround view** — Tesla-style bird's-eye: road, markings, vehicles on lanes, traffic lights in colour
7. **Lane counting from video** — where OpenStreetMap has no lane tag, infer lane counts from the video (lane lines, kerbs); apply only when multiple drives agree
8. **Per-track lane binding** — each vehicle bound to one lane via Viterbi; lane changes as explicit events
9. **Dashboard** — all drives on one page: synced video + 3D view + map with GPS and video-only position

Run everything:
```bash
.venv-video/bin/python -m src.video.pipeline <drive>
```

GPU-heavy steps (detection, segmentation) run on free cloud GPUs via a dispatcher
that tries Modal → Kaggle → Colab in order, falling through on failure.

### Models and data sources

| What | Model / method | Runs on |
|---|---|---|
| Vehicles, traffic lights | RT-DETRv2 (ResNet-101, COCO) | free cloud GPU |
| Tracking with re-ID | BoT-SORT + yolo26s-reid | free cloud GPU |
| Scene segmentation | Mask2Former Swin-L (Mapillary Vistas) | free cloud GPU |
| Own speed, rotation | optical flow (Lucas-Kanade) × metric depth | laptop |
| Vehicle classification | dense optical flow (Farneback) | laptop |
| Depth | Depth Anything V2 (metric outdoor) | free cloud GPU |
| Overlay OCR (GPS answer key) | Apple Vision OCR (macOS) | laptop |
| Route matching | dynamic programming (no ML) | laptop |
| Self-calibration | video rotation vs map heading | laptop |
| Map data | OpenStreetMap (free, no paid APIs) | laptop |

### Test drives

| Drive | Camera | Light | GPS | Route found? |
|---|---|---|---|---|
| `2026_10_22_10_21AM` | iPhone | day | no | correct (99% within 30 m) |
| `2026_10_22_18_16` | dashcam | dusk/night | yes | correct route, junction timing weak |
| `2026_10_23_10_06` | dashcam | day, rain | yes | correct; 11.7 m median; lens self-calibrated 120° → 99° |
| `2026_10_23_19_26` | dashcam | night, wet | yes | correct; 16.4 m median |
| `2026_10_24_17_08` | dashcam | day | yes | correct; 15.6 m median |
| `2026_10_26_10_02` | dashcam | day, dry | yes | correct; 15.4 m median; best speed (2.9 km/h error) |
| `2026_10_26_15_48` | dashcam | day | yes | correct; 15.2 m median |

---

## Future plans

The project is paused as of Oct 2026. Here is what comes next, in priority order:

### Near-term

1. **Junction detection toward 80%** (currently 47%). Better cues: Mapillary traffic
   sign types (yield, stop, priority road signs announce junctions), and night-robust
   features (headlights, cross-traffic).

2. **Per-track lane binding on video-only lanes.** Where the map has no lane data,
   bind vehicles to lanes visible in the video using the same Viterbi approach.

3. **Mapillary traffic signs as junction cues.** Download sign photos along each
   candidate route; use sign type and position as additional matching evidence.

4. **Detection on Modal.** Run the full perception step (RT-DETRv2 + BoT-SORT) on
   Modal for the first time with a new drive, instead of only Colab/Kaggle.

### Medium-term

5. **Building heights from video.** Estimate building heights from the dashcam to
   improve the 3D scene (shadow, occlusion).

6. **Signal state vs stop/go.** Correlate traffic light colour with our stops — is our
   light red when we stop? Use for signal timing extraction.

7. **Sign types for new drives.** Train or use a sign-type detector (GTSRB or
   Mapillary Traffic Sign) on Colab to classify traffic signs, not just detect them.

8. **Finer position from photo matches.** Visual place recognition (NetVLAD / cosine
   similarity) against Mapillary street-level photos to anchor position between
   junctions.

### Long-term vision

9. **Platform.** Upload flow for drivers (video + start/destination pins); automated
   pipeline on cloud GPUs; privacy review (plates are deleted, faces next).

10. **Feed the simulation.** Use the aggregated traffic measurements (flows, signal
    waits, queue lengths, speeds per street) to calibrate the SUMO model with real
    data instead of hand-tuned OD matrices.

11. **Night drives.** Solve rotation drift and street-lamp / traffic-light confusion
    at night.

12. **More cities.** The pipeline uses OpenStreetMap and free models — nothing is
    specific to Chișinău except the test data.

---

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
    gpu.py               GPU dispatcher: Modal → Kaggle → Lightning → Colab
    modal_run.py          Modal runner (free monthly credits, starts in seconds)
    kaggle_run.py         Kaggle runner (30 GPU h/week)
    colab.py             Colab runner (free tier, may reclaim mid-job)
    compute.py           CPU limits, progress bars
    overlay.py           dashcam overlay OCR (time, speed, GPS → answer key)
    perception.py        RT-DETRv2 + BoT-SORT + optical flow
    motion.py            own motion: moving, speed, rotation, turns, stops
    speed.py             speed from tracked points × metric depth
    traffic.py           other vehicles: oncoming, parked, same direction...
    scene.py             Mask2Former segmentation (cloud GPU or local)
    road_geometry.py     road width, side streets
    events.py            video as a list of junction events
    event_match.py       video-only route + position: events aligned with route sites
    roads.py, truth.py   OpenStreetMap street graph; truth route
    position.py          per-second position on OSM streets
    calibrate.py         camera self-calibration (field of view, depth)
    ground.py, lens.py   distance scale, lens model (GeoCalib)
    depth.py             per-pixel depth (Depth Anything V2)
    bev.py               spatial view: camera vs map
    surround.py          3D surround view data (ground memory, vehicles on lanes)
    world.py             fixed ground frame for the 3D view
    map_layer.py         SUMO lanes/junctions on the map; lateral lock; lane binding
    lane_count.py        lane counts from video where OSM has none
    vpr.py               visual place recognition (Mapillary photos)
    light_context.py     traffic light role scoring
    lanes.py, signs.py   lanes; street names (plates deleted)
    street_stats.py      stops, queues, speeds per street
    dashboard/           one page for all drives: video + 3D + map
docs/                    methods, findings, simulation guide
tests/                   28 unit tests
notebooks/
data/                    inputs and outputs (large files gitignored)
logs/                    job scripts and logs
```

## Quick start — video

Two environments: the simulation needs Python ≥ 3.10; the video pipeline needs
Python 3.11 (PyTorch on Intel Macs stops at 2.2 / NumPy < 2).

```bash
python3.11 -m venv .venv-video && .venv-video/bin/pip install -r requirements-video.txt
.venv-video/bin/python -m src.video.roads build          # street graph (once)
.venv-video/bin/python -m src.video.roads features       # crossings + signals (once)
.venv-video/bin/python -m src.video.scene download-model  # segmentation model (once, 826 MB)
```

1. Copy the video to `data/video/raw/` and add an entry to `configs/drives.yaml`.
2. Run the pipeline:
   ```bash
   .venv-video/bin/python -m src.video.pipeline 2026_10_22_10_21AM
   ```
3. Open `data/video/dashboard.html` in a browser and pick the drive.

GPU steps run on free cloud GPUs (Modal → Kaggle → Colab). Set up accounts:
- **Modal:** `pip install modal && modal token new` (saves to `~/.modal.toml`)
- **Kaggle:** place your token at `~/.kaggle/access_token`
- **Colab:** `pip install colab-connect` (free tier, no setup)

Tests: `.venv-video/bin/python -m pytest tests -q`

## Quick start — simulation

See [docs/simulation.md](docs/simulation.md) (SUMO 1.26, `requirements.txt`).

## Licensing

This is a personal, non-commercial project. The Mask2Former Mapillary Vistas weights
are licensed for non-commercial use only (CC BY-NC), which fits. If the project ever
becomes commercial, the segmentation model must be replaced or licensed first.
