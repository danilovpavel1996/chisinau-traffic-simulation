# Video pipeline — methods and findings

Goal: turn dashcam / phone videos uploaded by Chișinău drivers into traffic data
(where the car was, how fast, where it stopped and why, what traffic was around it)
to calibrate the city simulation with measurements instead of hand-tuned numbers.

## Steps

| Step | Module | Output (`data/video/processed/<drive>/`) |
|---|---|---|
| Detection, tracking, optical flow | `src/video/perception.py` | `detections.parquet`, `ego.parquet`, `meta.json` |
| Street-scene segmentation (Mask2Former, Mapillary Vistas) | `src/video/scene.py` | `scene.parquet` (crosswalks, lane markings, curbs, guard rails, scanlines) |
| Own motion: speed, rotation, turns, stops | `src/video/motion.py` | (used by the steps below) |
| Other vehicles: oncoming / parked / same direction…, counts | `src/video/traffic.py` | `tracks.parquet`, `stops.csv`, `turns.csv`, `per_minute.csv`, `timeline.png` |
| Map matching (particle filter on OSM) | `src/video/localize.py` | `match.json` |
| Dashcam overlay OCR (time, speed, GPS → ground truth) | `src/video/overlay.py` | `gps.parquet` |
| Dashboard of all drives (synced video + map) | `src/video/dashboard/` | `data/video/dashboard.html`, `dashboard_data.js`, `video_720p.mp4` |

| Position every second (GPS overlay, else video match), snapped to OSM | `src/video/position.py` | `trace.parquet` |
| Lanes seen / our lane (road scanlines) | `src/video/lanes.py` | `lanes.parquet`, `lanes_by_street.csv` |
| Street names + named places read in the video (plates dropped) | `src/video/signs.py` | `text.parquet` |
| Stops, queues before signals, speeds per street; all drives combined | `src/video/street_stats.py` | `stops_detail.csv`, `streets.csv`, `data/video/aggregate/` |

Run everything: `.venv-video/bin/python -m src.video.pipeline <drive>`.
Segmentation on a free Colab GPU: `python -m src.video.scene <drive> --colab` (~3 min vs ~40 on the laptop).
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
  - driving against an OSM one-way — penalty, not forbidden (tags can be wrong),
  - **driving over a crosswalk** (segmentation): the hypothesis should be within ~25 m of
    a mapped junction, crossing or signal (OSM features projected onto the graph).
- **Re-seeding at turns** (augmented MCL): when the video shows a clear turn (>45° in 8 s)
  that <20 % of hypotheses can follow, the worst 25 % are re-placed at junctions within
  1.5 km where that turn is possible (not in the first 60 s: parking-lot manoeuvres).
- 5 independent runs; the one with the highest **evidence** (log marginal likelihood)
  is kept — chosen without looking at the ground truth.

## Results — drive 2026-10-22 (home → Zity Mall, 5.3 km, 15 min, iPhone)

Ground truth: the driver's Google Maps directions (street sequence), see `configs/drives.yaml`.
Rows above the line were measured against an earlier, wrong truth (it continued down
Mihai Viteazul instead of turning onto Columna) and are not comparable.

| Version | Median error | Within 30 m | Covers true route |
|---|---|---|---|
| Turn-event beam search (removed) | — | — | ~40 %, stops early |
| Particle filter, absolute heading only | 147 m | 44 % | 41 % |
| + stops near junctions, speed sanity | 2 m | 55–92 % | 40–48 % |
| + windowed heading, drift-rate state, adaptive scale | 2 m | 89 % | 62 % |
| *corrected truth (Columna → Henri Coandă → Calea Ieșilor):* | | | |
| same filter | 2 m | 82 % | 70 % |
| + crosswalk anchors | 2 m | 82 % | 70 % |
| + re-seeding at turns (current) | **2 m** | **89 %** | **80 %** |

Median error is at the metric's resolution (true route sampled every 5 m). The match now
follows the whole route: Mircești → Petricani → roundabout (1st exit) → Mihai Viteazul →
right turn at 10:32 → Henri Coandă → Calea Ieșilor. Remaining: the campus exit (driveways
not in the street graph) and the 330 m of Columna (re-seeding jumped to the next junction).

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
10. **OSM one-way tag**: contra-flow is allowed at a penalty (map tags can be wrong).
11. **A self-made ground truth was wrong** (waypoints instead of the driver's directions)
    → truth is now the street sequence from the directions, validated against the map.
12. **Crosswalks** (segmentation): 16/16 detected events were real (stripe detector: ~50 %);
    one crosswalk is re-detected while standing at it → counted once per drive-over.
    Two crossings on Mircești are missing in OSM (2:21, 3:07 into the video).
13. **Anchors alone can't fix a 500 m along-track error** on a street with several
    junctions — the filter snaps to the nearest one. The decisive cue was the turn:
    re-seeding at turns moved coverage 70 % → 80 %.

## Next

- Drives with **phone gyroscope + GPS** logged alongside the video (e.g. Sensor
  Logger): exact rotation, and per-second ground truth to measure the vision-only
  pipeline honestly.
- **Road / lane segmentation** for metric speed (lane width as a ruler), lane count
  (match to OSM `lanes`) and lane position (turn prior).
- More drives before further tuning — every threshold here comes from one drive.

## Traffic measures — first results (2026-10-22)

Position from the dashcam's GPS overlay (evening) and from the video match (morning).

**Evening, Zity Mall → home (GPS):** 16 min of the 22-min drive were spent on 1.65 km of
Calea Ieșilor — median 8 km/h while moving, 62 % of the time standing, 30 stops. The stops
form one ~1 km queue that took ~13 min (793 s) to clear, headed by the traffic signal at
Mihai Viteazul × Mitropolit Dosoftei (if Piața D. Cantemir has signals missing in OSM, part
of it belongs there). Mircești, Petricani and Mihai Viteazul: free flow, 53–56 km/h.

**Morning (video match — speeds approximate):** 56 s waiting 42 m before the signal at the
end of Mircești; a 25 s red light on Petricani with 4 vehicles ahead.

**Lanes** (road scanlines, max over ~10 s for dashed markings): consistent between drives
on the same street (Mihai Viteazul 4/4, Calea Ieșilor 3/3 = 3 per direction as in OSM) but
over-counts narrow streets (parked cars) — a ±1 road-width class, not an exact count.

**Text** (Apple Vision OCR, a frame every 2 s): mostly number plates (now dropped for
privacy) and shop names. Street-name plates are rarely legible; named places matched
against OSM POIs are right in 10/16 cases on the GPS drive (within 150 m) — soft evidence.

## Findings log (continued)

14. **Night breaks optical-flow rotation** (dashcam, dusk): 83° false rotation per 30 s on
    straight roads; vision-only map matching failed completely. Daytime tuning does not
    transfer — GPS (overlay / phone) is the position source when available; GPS drives are
    the training data for making vision-only robust later.
15. **Red-light colour detection works in daylight, not at night** (glare) → queues are
    attributed to the next signal AHEAD on the driven route instead.
16. **Colab file API**: uploads > ~30 MB and into missing folders fail (HTTP 500) → 25 MB parts.
17. **YAML** reads an unquoted 2026_10_22_18_16 as an integer → drive keys are quoted.

## Event-based route matching (2026-10-23)

Input per video: start + destination (given by the uploader), date/time from the video.
The video becomes junction events (turns, roundabouts, crosswalks, close traffic lights,
right-side openings); 300 candidate routes start → destination are aligned with them
(Needleman–Wunsch, detectability-based costs, odometer timing), re-ranked by road-width
agreement (rank correlation of the video's road width with the map's road size).

| Drive | Route chosen | Result |
|---|---|---|
| Morning, daylight (phone) | Mircești → Petricani → Mihai Viteazul → Columna → Henri Coandă → Cantemir → Calea Ieșilor | **99 % of the true route, all within 30 m** (12 → 20 events once side-street openings were added; margin to the runner-up small) |
| Evening, dusk (dashcam) | 80 % of the junctions passed are on the chosen route | right junction at the right moment: 0 % — night events are mostly noise |

Measured cues (evening drive vs GPS): right-side openings precision 50 % (chance 19 %),
left side ≈ chance (opposite carriageway); cross traffic at night 33 % (chance 22 %) — not used.
Road width (half-width ÷ distance below the horizon, from the class maps) orders boulevards
vs side streets within a video; queued cars inflate it.

## First daylight dashcam drive — 2026_10_23_10_06 (2026-10-23)

Home → Zity Mall, the same route as the phone drive, dashcam with GPS overlay
(answer key). First run: the right route lost to a wrong one (via Ștefan cel Mare),
and only 4 of 21 junctions came out at the right moment. Five causes, all fixed:

| Problem | Evidence | Fix |
|---|---|---|
| GPS answer key had 1 km jumps | 0↔6 / 3↔8 misreads repeated for several seconds; the track measured 14.9 km for a ~5 km trip | `overlay.clean`: drop fixes > 250 m from the 21-s rolling median track (both dashcam drives re-parsed) |
| Speed saturated above ~50 km/h | over a 0.2 s stride the road near the car moves past the tracker's reach; 60–80 km/h read like 30 | speed flow between adjacent frames × stride; rotation stays on the full stride (one-frame rotation was noise → fake turns) |
| Missing road measurement read as "stopped" | the road strip had < 10 trackable points in 35 % of frames at > 15 km/h; odometer froze on Calea Ieșilor | fill gaps with the scene's forward expansion (`yaw_slope`, never missing, ~2000× stopped/moving contrast), scaled by the drive's own ratio. Moving/stopped vs GPS 73 → 88 %; odometer error median 303 → 103 m over 5.3 km |
| Unknown way out of the start pin | car left the yard north-west, U-turned on Mircești: +800 m before the route starts; timing measured from the start shifted every later junction | alignment scores timing **between consecutive matches** (20 % of the stretch + 60 m); start / destination ties count ¼ |
| Road-width re-ranking | flipped the right route to a wrong one; never changed the phone drive's choice | off (`WIDTH_WEIGHT = 0`) |

The evaluation now also counts **physical junctions** (OSM nodes within 40 m passed within
6 s are one crossing: a dual carriageway is 2–4 nodes) with a match ±8 s within 30 m of it,
and whether the video-only position is within 30 m at the moment GPS passes the junction.

| Drive | Route | Junctions on route | Right junction ±8 s (strict / physical) | Position ≤ 30 m at junction | Margin to runner-up |
|---|---|---|---|---|---|
| 2026_10_23_10_06, day, dashcam | **correct** | 20/21 (95 %) | 24 % / **33 %** (6/18) | 22 % | 9.7 vs 12.7 |
| 2026_10_22_10_21AM, day, phone | **correct** (99 % of it, all within 30 m) | — (no GPS) | — | — | 14.7 vs 17.7 (was 7.6 vs 7.75) |
| 2026_10_22_18_16, dusk, dashcam | wrong in the middle | 21/26 (81 %) | 0 % / 0 % | 0 % | — night, postponed |

What limits the junction rate now is detection, not alignment: of the 18 physical junctions
on the morning drive, half show no video cue or only a right-side opening (weak, many
false ones), so the alignment skips them. Next: more junction cues in daylight (vehicles
crossing ahead, side streets on both sides from curb / sidewalk gaps, stop lines), then
refit the odometer scale between matched junctions — see README → Plan.

## Spatial view, first version (2026-10-23)

`bev.py` projects the dense segmentation (2 frames/s, Colab, 4.8 min for 12 min of video)
onto the ground and draws the map's expectation on top, from the video-only pose
(white) and, for evaluation, the GPS pose (thin green). Dashboard: video · spatial view · map.

- **Horizon**: the motion-based focus of expansion sat ~25 px too high (row 252 vs 277 of
  540) and projected the tree line as road 10–40 m ahead. Now: where the road ends straight
  ahead on open road (10th percentile of the road's top row over straight driving).
- **Upright classes** (cars, poles, signs, guard rails, buildings) smear into long streaks
  when projected — in a bend the guard rail covered the road ahead. Only ground classes are
  drawn; vehicles come from their detection boxes.
- **Camera height** from car widths (1.8 m): 1.28 m (1,576 cars), lateral scale fine.
- **Depth scale is off**: the nominal 120° pinhole gives too short a focal length — turn
  angles from the video are ~1.5× (1.1–2.7×) the GPS ones, and a car ~15 m ahead is drawn
  at ~8 m. Next: calibrate it against the map (ground motion between matched junctions =
  their distance on OSM).
- First use: at Columna × Henri Coandă (533 s) the camera sees a side street opening on the
  right where the GPS-pose map has the junction, while the video-only-pose map shows a
  straight street — the video-only position is behind there.

## Red arrow 500 m ahead at the start (2026-10-23, reported from the dashboard)

At 10:08:33 the video-only position was 827 m along the route, the truth 320 m, and the
panel showed 115 km/h (GPS 63). Two causes:
- a **false traffic light** at 179 s — red autumn leaves through rain drops, YOLO confidence
  0.11–0.32 — matched to the signal at Mircești × Petricani (passed at 216 s). The signal cue
  now needs a median confidence ≥ 0.3 over ≥ 3 frames (the real light at 509 s: 0.60);
- the **road-plane speed** read ~2× too high going downhill on Mircești (it depends on the
  road's tilt). The odometer now uses scene flow: at 147 s the position is 302 m (true 320).

Result: physical junctions at the right moment 33 → **44 %**, position within 30 m at the
junction moment 22 → **44 %**, median position error 62 → 57 m; both daylight routes still
correct. The speed panel showed the marker's speed (driven by the anchors); it now shows the
video's own measured speed (scaled to the route length) — rough: correlation 0.5 with GPS,
median difference 11 km/h.

## Camera self-calibration against the map (2026-10-23)

No camera specs, no GPS: `calibrate.py`, after the first matching.

**Field of view.** Over 12-s windows with > 25° of heading change on the matched route, the
video's rotation (nominal lens) is regressed on the map's. Daylight dashcam: slope **1.48**
from the map, **1.56** from GPS → the "120°" lens is effectively **99°**. Guards: ≥ 6 windows,
≥ 75 % turning the same way, slope 0.5–2.5 — the night drive (wrong route, 60 % agree) and
the phone (4 windows) keep the nominal value. Estimated always from a nominal-lens match,
so the result does not depend on earlier runs; events and matching are then re-run: position
within 30 m at junction moments 44 → 50 %, runner-up margin 2.9 → 3.0.

**Depth.** Approaching a crosswalk straight ahead, its distance in the spatial view must
shrink by the distance driven (video odometer, metres through the map). Before: view / GPS
distance 0.51 (nominal lens), 0.70 (calibrated lens). Depth factor (3 passes — which approaches
qualify depends on the scale): **1.54** → view / GPS **0.93** (IQR 0.86–1.12, 17 frames). The
camera height from car widths is 1.37 m with the calibrated lens. With the depth right, the
nearest visible ground is 4.3 m (the bonnet hides the rest).

Seen in the spatial view at 10:08:32: Strada Domniței on the map (white, video-only position)
lines up with the road surface opening on the right in the camera layer.

## Stitched top-down strip (2026-10-23)

`strip.py` paints, every 0.5 s, the sharpest band of the calibrated ground plan (from the
nearest visible ground, 4.3 m, to +14 m) into one long strip at the video-only distance along
the route; each 0.25 m cell keeps its most-seen class. With an 8 m band the strip had gaps at
speed (0.5 s at 50 km/h = 7 m, and the anchored distance can run faster between anchors);
14 m → ground seen along 99 % of 5.15 km, built in ~7 s. The map's side streets are stubs at
the angle they leave the route. Where a curb gap and a stub are at different distances, the
video-only position is off by that much — the starting point for a junction detector (next).

## Detector upgrade: RT-DETRv2 + BoT-SORT with ReID, on Colab (2026-10-23)

Same drive (2026_10_23_10_06), street window, YOLO11s + ByteTrack (laptop) vs RT-DETRv2-R101 +
BoT-SORT with camera-motion compensation and yolo26s-reid (Colab T4, 32 min for 12 min of video):

| | YOLO11s + ByteTrack | RT-DETRv2 + BoT-SORT/ReID |
|---|---|---|
| vehicles per frame | 6.1 | **11.4** (near cars YOLO missed, distant traffic, parked rows) |
| vehicle tracks ≥ 10 frames | 370 | **793** |
| close traffic-light clusters | 2 (1 false: red leaves through rain) | 7: 3 at OSM signals, a real pedestrian light missing from OSM (259 s, 0.82), 1 false (a trolleybus's lit route number, 0.39) |
| moving / stopped vs GPS (flow re-run on Colab) | 92 % | 90 % |
| speed correlation with GPS | 0.51 | 0.46 |

Localisation did not measurably change: physical junctions at the right moment 44 → 39 %,
position within 30 m at junction moments 50 → 33 % (one or two junctions of 18). The lens
calibrated from the map moved 99° → 105.5° because it depends on the route timing; GPS gives the
same turn scale for both runs (1.56 / 1.57). One drive with 18 junctions swings ± 2 junctions
with such knock-on effects — judging changes like this needs more daylight drives with GPS.
The ReID model's GPU backend needs "cuda:0": with "cuda" it fell back to the CPU (3 s per frame).

## Depth Anything V2 (metric outdoor, Large) — first check (2026-10-23)

`depth.py` on Colab: 1,426 dense frames in 8.1 min. The *shape* is good (correlation 0.75 with
the flat-road distance of cars ahead), the *metric scale* is not, and not uniform:
depth / flat-road distance on road pixels straight ahead falls from 2.5 (0–10 m) to 1.7 (40 m);
on crosswalk approaches depth / GPS = 1.25. Causes: the model was trained on ~80° driving
cameras (Virtual KITTI), this dashcam is ~100°+ (things look smaller → farther), plus the bonnet
and windscreen reflections near the bottom of the image.

Found while checking: the ground geometry's own scale is unstable between runs (view / GPS 0.93
in one run, 0.73 after re-running perception). The depth factor rests on only 2–3 crosswalk
approaches per drive, where the video odometer's local error (± 20–30 % over a short stretch)
goes straight into the scale. Needed: a camera model calibrated from the video's geometry itself
(lens distortion from straight lines, focal length from vanishing points), then depth fitted onto
that ground geometry.

## Distance scale from parked cars and the dashcam's speed (2026-10-23)

The depth scale rested on 2–3 crosswalk approaches per drive and swung between runs (view / GPS
0.93 → 0.73). `ground.py` measures the horizon and the scale (focal length × camera height)
from **parked cars**: their box bottom must move as a flat road predicts while we drive at the
**overlay speed** (GPS speed only — never GPS positions, so the route stays video-only).

Tried first on the road surface and dropped: dense optical flow, flow after removing the static
windscreen reflection (median of 120 frames), and matching row-brightness profiles in inverse
distance. The compressed asphalt changes by 1–3 grey levels between frames at 60 km/h and the
dashboard's reflection dominates — measured motion did not grow with the square of the row.

Parked cars needed two fixes: far cars (30–80 m) barely move in the image while box edges jitter,
so only bottoms ≥ 25 px below the horizon (≲ 30 m); and the tracker passes one ID along a row of
parked cars, so tracks are split where the bottom edge jumps up or detections pause for > 6 m.
At the true horizon all cars agree on one scale.

2026_10_23_10_06: 47 usable car segments, 13 on clean straight lines → horizon row 272, scale 888
→ spatial view / GPS distance on crosswalk approaches **0.93** (IQR 0.86–1.07, 18 frames), now
independent of the odometer. Camera height (car widths) 1.38 m; central focal length 646 px
(~73° at the centre) vs ~415 px from rotation over the whole width (~99°): the lens is not a
pinhole — it magnifies the centre more (mild barrel distortion). Next: measure the lens (GeoCalib).

## Lens (GeoCalib) and metric depth (2026-10-23)

`lens.py` — GeoCalib on 80 frames (Colab, 5 min): focal length **600 px** (IQR 510–715) at the
centre of the 960-px image (~77° field of view), radial distortion **k1 = −0.24** (barrel: the
edges squeezed ~15 % toward the centre). Agrees with the central focal length from parked cars +
speed (646 px, within 8 %); the "99°" from rotation was an average over the distorted width.
GeoCalib's camera tilt (~3°) is rough; the measured horizon is kept. The spatial view now
undistorts pixels before projecting them (things beside the car were drawn too close to the
centre line).

`depth.py --fit` — Depth Anything reads this dashcam's distances too far, not by one factor;
per drive, fitted on 719,000 road pixels 6–35 m ahead (calibrated flat-road distances):
model depth = 2.21 · Z^0.887. Independent checks: crosswalk distances from depth alone vs GPS
**0.99** (IQR 0.95–1.11); traffic lights **4.3 m** above the road (IQR 3.9–6.4) — realistic.

## Second night drive — 2026_10_23_19_26 (Bd. Moscova → Ciocana, 2026-10-23)

Four dashcam clips, the first trimmed by the driver and re-saved as HEVC: re-encoded to H.264 and
joined through MPEG-TS (in-band parameter sets), 632 s, decodes cleanly. The first Colab upload
(1.3 GB, 54 parts) died when the free runtime was reclaimed half-way: uploads now retry per part,
and perception uploads a compressed copy (same 18,967 frames, ~⅓ the size).

- **Route (video only): identical to GPS** — Bd. Moscova → Studenților → Mihail Sadoveanu →
  Profesor Ion Dumeniuc; 11 of 12 junctions on it; 33 % of junctions at the right moment (the
  first night drive: 0 %); median position error 76 m. Pins: start on Bd. Moscova at 0 s, end where
  the car turns off the street network (439.4 s, from the video's own turn detection).
- **Lens (GeoCalib) is stable day and night**: focal 598.7 px / k1 −0.243 vs 599.5 / −0.241 in the
  morning — same dashcam, same answer.
- **Parked-car scale fails at night** (loose boxes: headlights, reflections): 4,839, i.e. a camera
  8 m above the road. Now rejected by a plausibility check (scale / GeoCalib focal = camera height
  0.9–2.0 m; cars agreeing within 2×); fallback: GeoCalib focal length + horizon where the road
  ends + camera height from car widths (1.23 m). Crosswalk distances vs GPS: **1.01**
  (15 frames). Turn-based lens calibration correctly refused (night windows inconsistent).

## Detailed surroundings: 5 frames/s, full-resolution segmentation (2026-10-24)

`dense.py` replaces the separate dense-segmentation and depth uploads: one small video of the
model input (above the bonnet, 960 px, ~350 MB for 10 min; one padding row because H.264 4:2:0
needs an even height, cropped off on the VM) is uploaded once; Mask2Former and Depth Anything run
in one Colab session on **every 6th frame (5 per second — the frames perception detects on)**;
class maps at every pixel (was every 2nd), depth at ½ (was ¼). ~0.37 s/frame on a T4 for both.
Plan view drawn at 20 px/m (was 10), 3D ground texture 32 px/m (was 16).

The 3D road surface is now a plain `<video>` placed in 3D with CSS (three.js CSS3DRenderer) under a
transparent WebGL layer for the cars and lights: from a file (file://) Chrome will not hand a video
to WebGL, but CSS shows it — the 3D view no longer needs the local viewer. The ground under and
behind the car (hidden by the bonnet) continues the nearest ground seen.

Colab robustness: a progress check that lost its connection hung for ever (no timeout on the CLI
call) — every call now has a time limit, 10 failed checks in a row raise a clear error, and
`dense.py --collect` downloads the results of a job that finished on a runtime still alive.

## 3D view: ground memory, vehicle headings; a side-street detector (2026-10-24)

**Ground memory** (`surround.py`). Our own motion from the video alone (`EgoMotion`: speed from
scene flow in metres through the map route, rotation from yaw) carries the ground seen in earlier
frames along: road hidden behind a car or under ours shows what was seen there a moment ago;
fades after 8 s unseen (faster while turning — the rotation scale is uncertain at night). Only the
sharp near field (≤ 20 m) is remembered: far ground is coarse and smeared into streaks.

**Vehicle headings.** Measuring every vehicle's heading from its track minus our motion failed:
our speed from the video is off by up to ~3 m/s over 0.2 s, so parked cars got headings and
oncoming cars a median of 30°. Now from the traffic step's labels (image motion vs the road beside
the vehicle): ahead / overtaking / parked 0°, oncoming 180°, crossing ±90° by its sideways motion;
a measured heading only for turning / unlabelled vehicles moving ≥ 4 m/s.

**Side-street detector** (`side_streets.py`, experimental, not used by the matching). Road surface
reaching ≥ 3 m further sideways than the street's usual edge over ≥ 5 m of length, in the
calibrated ground plan, merged by predicted passing time. Against GPS (correct side, ± 4 s):

| | right side: precision / found | both sides: precision / found |
|---|---|---|
| old right-side opening cue, day | 25 % / 3 of 9 | — |
| side_streets.py, day | 22 % / 8 of 9 | 21 % / 10 of 14 |
| old right-side opening cue, night | 12 % / 2 of 7 | — |
| side_streets.py, night | 23 % / 5 of 7 | 29 % / 8 of 10 |

Much higher recall, same low precision — and in the matching it made things worse (physical
junctions at the right moment 39 → 28 % day, 33 → 25 % night): the extra false openings mislead
the alignment. Causes seen: occluders driving away (a trolleybus uncovering the opposite lanes),
car-park entrances (real openings, not in the street map), timing. Tried and dropped: counting
vehicles as road (parked rows became road, precision 7 %), rejecting openings with parked cars
(real side streets have parked cars too, recall 71 → 14 %). Needs another cue to confirm a mouth
(curb return, crosswalk, sign types, a vision-language check) before it can help the matching.

## Traffic-sign types (2026-10-24)

`sign_types.py`: Mapillary "traffic sign (front)" regions every 6th frame → ResNet-18 trained on
GTSRB (43 German classes, Vienna Convention like Moldova's) on Colab, once (models/). Crops saved.
By eye: daylight yield 12/12, priority road 11/12, no entry 12/12, keep right 11/12 correct; at
night priority road and no entry 12/12, "yield" mostly the backs of yield signs (still a junction);
the blue arrow classes (ahead only) absorb Moldova's blue information / lane / pedestrian signs —
GTSRB has no "other sign" class. Reliable junction classes vs GPS junctions (10-s window): day
precision 68 % (chance 41 %), recall 62 %; night 37 % (chance 29 %). Priority-road signs come ~8 s
before their junction, yield / keep-right at it. As a matching cue (also as a flag only, not moving
the event time) it did not help: day 39 → 28 %, night 33 → 33 % — off (`events.USE_SIGNS`).

## Speed from flow × depth (2026-10-24)

Image motion depends on how far the scene is (narrow street vs open road). With metric depth each
tracked point gives v = radial motion × fps × Z / (pixels from the vanishing point); the median
over points is the car's speed. Test on 2026_10_23_10_06 (1 sample/s, one frame pair): correlation
with GPS **0.79** (scene flow: 0.47), median error **9.2 km/h** (13.7) after one scale factor (raw
0.79× GPS). `dense.py` now also tracks points through 3 adjacent frame pairs after each analysed
frame (motion.parquet); `speed.py` turns them into km/h, scaled by the overlay speed if present,
else the matched route length.

### Speed, second round (2026-10-24 evening)

Motion now captured for every analysed frame (5/s, 3 adjacent frame pairs each). Error sources
found and removed, on 2026_10_23_10_06 vs GPS (street window):

| step | correlation | median error | 90 % within | raw scale |
|---|---|---|---|---|
| radial flow × depth, median over points | 0.79 | 8.7 km/h | 26.5 | 0.86× |
| 6 unknowns (speed, side/vertical drift, 3 rotations) at once | 0.72 | 10.6 | 35.8 | 0.98× |
| direction fixed to the vanishing point, rotation fitted with speed | 0.68 | 9.5 | 33.5 | 1.02× — a steady −10°/s "turn" absorbed speed |
| **rotation from distant points only (Z ≥ 30 m), speed from near ones, undistorted** | 0.81 | 7.4 | 23.7 | 1.00× |
| + depth rescaled per frame by its own road pixels, 2-s median | 0.84 | 6.6 | 21.3 | 0.99× |
| + still points dropped (raindrops, the dashboard's reflection) | 0.84–0.85 | 6.6–6.7 | 20.8 | 0.98× |

The estimate is now metric without any calibration. Errors left depend on the road: on the
tree-lined downhill of Mircești it reads ~1.2× (vegetation points), on the Bd. Moscova bridge
~0.4× (only distant buildings, wires and raindrops to track); per scene class no single
class is unbiased everywhere. Hence `speed_model.py`: gradient-boosted trees fitted to GPS
speed on what each frame's speed was measured on (median speed and share of points per scene
class, distance, spread, depth correction, turning). Trained on one half of 10_06 and tested
on the other: correlation 0.83 → 0.89, median error 7.4 → 6.2 km/h (and 0.87 → 0.88, 6.1 → 5.2
the other way round). Next: train on the out-of-town drive 2026_10_24_15_48 (few cars, long
fast stretches), the city drive 2026_10_24_17_08 and the night drive, scored leave-one-drive-out.

As the matcher's odometer the better speed has not helped yet (physical junctions 7/18 → 6/18,
position within 30 m 6/18 → 7/18): the matching is limited by the junction events, not the
distance between them. Dead-reckoned from the start alone it drifts up to ~400 m (scene flow:
~350 m) because the per-road bias accumulates — the model should fix exactly that.

### New drives, GPS digit repair, turn angles from the rotation fit (2026-10-25)

- **The "13.2 km detour" was a broken answer key.** The overlay OCR read latitude 47.05… for
  47.06… for 32 s on end (three stretches, each ~1.1 km off) — longer than the median filter's
  window. `overlay.repair_digits` cuts the track where it jumps further than the overlay's own
  speed allows and shifts whole stretches (≥ 5 s) by one digit when that joins them to their
  neighbours. 2026_10_24_17_08: 13.2 → 8.94 km by GPS (video odometer: 8.68 km). Other drives
  unchanged.
- **Turn angles from speed.py's rotation.** The old angle scales with the assumed field of view,
  which is fitted to the map turns of the matched route — 132.9° on a wrong route made the left
  turn onto Bd. Moscova (17:16:36) read 230°, a "U-turn", so the matcher went straight on.
  The rotation fit uses GeoCalib's focal length, undistorted distant points, vehicles masked:
  turn size vs GPS heading change 1.4–2.8× → 0.9–1.1× (night 1.4×).
- **route_search.py**: candidate routes built junction by junction from the events and the
  metric odometer (Viterbi with a beam), added to the shortest-first candidates — for real
  detours (> 1.5× the shortest route), which shortest-first listing can never reach.
  On the four drives it has not changed the chosen route yet.
- Speed model with four drives, leave-one-drive-out: still no consistent gain (out of town
  8.5 → 14.5 km/h worse; city drives ±0) — not used.

| drive | route | physical junctions ±8 s | position ≤ 30 m at junctions |
|---|---|---|---|
| 2026_10_23_10_06 (day) | right | 7/18 (39 %) | 6/18 (33 %) |
| 2026_10_23_19_26 (night) | right | 5/12 (42 %) — was 33 % | 2/12 (17 %) |
| 2026_10_24_15_48 (out of town, map part) | right | 2/4 | 2/4 |
| 2026_10_24_17_08 (city, 8.9 km) | 66 % of junctions on it — was 17 % | 12/33 (36 %) — was 4/33 | 14/33 (42 %) — was 2/33 |

Speed vs GPS, whole drives, no calibration: correlation 0.93 / 0.90 / 0.82 / 0.93, median
error 4.8 / 3.6 / 8.8 / 3.5 km/h (day / night / out of town / city).

### 3D view: world frame, light landmarks, road from many frames, lanes (2026-10-25)

- `world.py` — our pose (x, z, heading) from speed.py, integrated. Traffic lights become
  landmarks: detections grouped by geometry (tracker ids for lights were useless: half the light
  tracks ≤ 4 detections, one id over 649), position = least squares over all lines of sight and
  depth readings (consistent within 0.3–0.65 m; a single frame's position wobbled 2–6 m), drawn
  with a pole, colour held 3 s. 76 landmarks on 2026_10_24_17_08.
- Vehicles smoothed per track in the ground frame, parked cars fixed: frame-to-frame jump
  1.49 → 0.17 m median (90 %: 10.3 → 2.6 m), parked cars 5.8 m → 0.
- `ground_world.py` — the road from all frames 12 s before … 6 s after, in the ground frame
  (sliding sums on a raster that moves with the car, 10-cm cells): lane lines continuous instead
  of breaking up per frame, gaps behind cars and under our car filled. ~6 min per 20-min drive.
- Lanes: lines, curbs and the road's extent read every 5 m ahead (lanes_world.parquet); lanes not
  painted or not seen are spaced by the road's own lane width out to its edge; moving vehicles
  are pulled to the middle of their lane — 48 % of moving-vehicle positions fall in a lane
  (only lines actually read: 24 %); the rest mostly at junctions and in turns.
- Waiting for a Colab GPU: `detect_test.py` (5/10/15/30 fps tracking, tiled detection of
  distant lights) on three clips of 2026_10_24_17_08.

### Detection / tracking test on clips (2026-10-25, `detect_test.py`)

2026_10_24_17_08, clips 470–560 s (left turn + parked cars), 700–740 s (junction with lights),
895–925 s (the light at 15:19); RT-DETRv2 + BoT-SORT/ReID as in perception.py, tracker memory
kept at ~10 s at every rate.

| rate | vehicle IDs / min | track life (median) | close cars (> 80 px): new ID while already close and inside the image / min | moments with a (tracked) light |
|---|---|---|---|---|
| 5 fps | 155 | 3.6 s | 12 | 24 % |
| 10 fps | 241 | 2.2 s | 24 | 49 % |
| 15 fps | 294 | 1.8 s | 33 | 56 % |
| 30 fps | 377 | 1.4 s | 40 | 66 % |

- More frames did **not** keep identities better with this tracker — the opposite. Not
  fragments of one car (joining pieces ≤ 0.6 s apart changes little: 1.03–1.07 pieces per
  car): more short new tracks, also on close cars. BoT-SORT is tuned for 5 fps (birth threshold,
  match threshold, camera-motion compensation); a higher rate needs a re-tuned or different
  association — planned in the ground frame, with lanes (README plan). Detection stays at 5 fps.
- Traffic lights: the tracker discarded most light detections (only confirmed tracks were
  kept): at 15:19 one detection at 5 fps vs 29 at 10 fps. Fixed in perception.py: every light
  ≥ 0.3 kept, tracked or not (world.py groups them by geometry).
- Tiles (upper image in two overlapping tiles): real distant / side-on lights the full image
  misses, but below 0.4 confidence mostly false (median tile confidence 0.14). Kept ≥ 0.4.
- Compression: the lightly compressed clips gave +40 % light detections at 15:19 and +36 % at the
  junction vs the CRF-30 upload copy (vehicles ±5 %). For the next full runs: better quality,
  uploaded and detected in time chunks (one Colab job each, survives a reclaimed GPU).
- Colab free GPUs were taken back twice mid-job: jobs now per run and clip, downloaded as they
  finish, resumed on rerun; a vanished session fails at once.

### Map layer: SUMO lanes around the car, and where on the map we are (2026-10-26, `map_layer.py`)

The SUMO network (netconvert from OSM) as tables: 121,231 vehicle lanes, 65,985 junctions (265
with traffic lights); no crossings in it (netconvert was run without them — OSM's 3,124 crossing
points cover that). To draw it around the car, our position on the map must be right to a few
metres. Video-only position vs GPS (answer key), median / within 10 m / within 30 m:

| method | 10_06 (day) | 17_08 (city) | 19_26 (night) |
|---|---|---|---|
| event_match trace (matched junctions + flow odometer) | 73 m / 7 % / 25 % | 53 m / 8 % / 29 % | 59 m / 8 % / 21 % |
| metric odometer only, scaled to the route length | 136 m / 1 % / 9 % | 86 m / 4 % / 14 % | 52 m / 14 % / 35 % |
| our path's shape fitted to the route (rigid, 60-s windows) | 80 m | 40 m | 51 m |
| turns matched to route corners + odometer between | 68 m | 33 m | 35 m |
| **heading-profile alignment (DTW of heading change vs metres driven)** | **25 m / 12 % / 61 %** | **28 m / 18 % / 51 %** | 46 m / 14 % / 35 % |
| + landmark anchors (crosswalk / light passings matched to OSM / SUMO) | 22 m | 47 m | 24 m |
| + road-surface registration (FFT, seen road vs map lanes) | dropped: from the true position it wandered 7–9 m off, 25 m off it stayed 25 m off | | |

Kept: heading-profile alignment (`pose_on_map` → map_pose.parquet). Not yet good enough to draw
junction geometry in place (needs ≲ 5 m along the road); sideways the video's own lane lines can
lock it. The out-of-town drive's map part (one straight road) has no profile to align: 650 m off.
Landmark anchors off: the video's crosswalk passings were a median 21–30 m from any OSM crossing at
the true position, light passings far outnumber signalised junctions (several heads, pedestrian /
false lights) — both need better detectors before they can pin the position.

### Traffic lights with context, OSM buildings, Kaggle for every GPU step (2026-10-26)

- Light landmarks checked by eye (24 random, 2026_10_24_17_08): only 10 real — the rest road signs
  (blue, priority diamonds, crosswalk signs), shop signs, lit displays, poles. OSM is incomplete
  here: 6 signalised junctions mapped on that route, all 6 found by the video, but well-seen real
  lights stand at many unmapped places — the map can confirm a light, not rule one out.
- `light_context.py`: each landmark's three largest views scored by CLIP (clip-vit-large-patch14,
  zero-shot, on Kaggle) against the look-alikes: on the 24, kept 9/10 real and 3/14 false (base
  model: 7/10 and 4/14). City drive: 147 of 286 landmarks real — 59 ours (facing us, at vehicle
  height, over / beside our road), 41 other directions, 47 pedestrian; 31 % near an OSM signal.
  3D view: only real lights, ours full size, others smaller and dim, pedestrian small.
- Junction events: the signal cue from tracked lights plus every detection of a verified light
  (all untracked lights made night matching drop 42 → 17 %: street lamps).
- OSM buildings (69,010; height measured 2 %, from floors 7 %, else by type) drawn translucent in
  the 3D view along the route.
- Kaggle runs the Colab jobs unchanged (`kaggle_run.run_job`: inputs to /content paths, outputs
  collected): perception, scene, dense, points, lights — Colab only as fallback.

### Mapillary coverage (2026-10-26, `mapillary.py`)

City route 2026_10_24_17_08 (8.9 km), 30-m corridor: 21,453 street photos, 57 % of the route with
one within 20 m, newest 2026-10-17, 31 % since 2024; 11,264 detected map features — supports 2,718,
signs 1,995, traffic signs (typed) 1,685, street lamps 1,446, markings 638, **traffic lights 480**
(OSM: 14 signals). 83 % of Mapillary's lights lie within 25 m of one of our verified lights; 54 % of
ours have a Mapillary light near. Much richer than OSM for lights and signs: the map prior for
lights, typed junction signs, and fixed points to pin our position along the road. Token: the
owner's, in ~/.mapillary/token (header, never in a URL); data is CC BY-SA — fine for this personal,
non-commercial project.

**Mapillary pins: tried, off.** Our verified lights vs Mapillary's, to fix how far along the route we
are. Mapillary's "traffic lights" along the city route are 866 heads spaced 1–2 m — the same lights
detected again photo after photo, spread along the streets rather than one point per light —
clustered into only 8–19 long "spots"; at the true (GPS) position only 16–17 % of our passings were
within 15 m of one. Light-to-light offset votes: 25 → 34 m (10_06), 28 → 34 m (17_08); in-order
spot matching: 25 → 32 m, 28 → 55 m. Mapillary lights stay useful to confirm that a light exists
(83 % of them within 25 m of one of ours), not to pin a position to a few metres.

### Visual place recognition against Mapillary photos (2026-10-26, `vpr.py`)

2026_10_23_10_06: 1,857 Mapillary photos along the matched route (≤ 3 per 5 m, facing our way, no
panoramas; 63 % of the 5-m bins have one), our frames once a second above the bonnet; EigenPlaces
(ResNet-50, 2048-d) descriptors on Kaggle. The descriptors discriminate: at the true spot similarity
0.54 (median), any pair 0.17; the single best photo within ±200 m lies within 15 m of the truth for 48 %
of frames. Sequence alignment (DP over 5-m route bins, steps bounded by the odometer ± 20 %, per-frame
z-scores; a fixed cost for photo-less bins first made the path prefer them — 32 m): along the route
20 m median, ≤ 10 m 25 % (prior 25 m, 17 %); confident frames (z ≥ 4, 57 %) 16 m.
Fused as pins (median per 10 s) into the map position: 24.5 → 21.5 m, ≤ 10 m 16 → 26 %.

The GPS overlay appears to lag the video by ~1 s: shifting it by +1 s lowers the confident-frame
error 16 → 12 m — every GPS score so far includes ~10 m of this at city speed.
Error left varies by stretch (+26 m at 3:00, +32 m at 9:00, ~0 elsewhere): look-alike straight
roads, or contributors' GPS on some Mapillary sequences. Next: geometric verification with local
features (relative position between our frame and the photo), more photos per bin.

**Geometric verification + GPS lag (2026-10-26).** Per frame the 5 most similar photos within ±120 m,
checked on Kaggle with SuperPoint + LightGlue and a RANSAC fundamental matrix. The GPS overlay trails
the video — measured per drive as the shift where its speed best matches the video's own speed: +1.2,
+2.0, +2.6, +1.0 s (10_06, 19_26, 15_48, 17_08) → `gps_lag_s`, applied in `drives.gps_track` (answer
key and speed scale only). With that, 10_06: verified photos with ≥ 100 inlier matches (55 % of
frames) 9 m median, 58 % within 10 m (best of the 5 candidates, an oracle, 8 m — the photos available
are the limit); map position with these as pins **11.5 m median, 44 % within 10 m, 89 % within 30 m**
(16.8 m, 23 %, 70 % without).

### New dry drive, the matcher's odometer, GPU dispatcher (2026-10-26)

**2026_10_26_10_02** (home → Zity Mall, dry after rain; same route as the rainy 10_06). Window
133–901 s from the video's own turns. Dry vs rain: speed vs GPS 0.94 / 2.9 km/h (10_06: 0.93 /
4.8); 48k vehicle detections (34k), 3,025 traffic-light detections (1,351); 30 real lights (37);
photo matching 12 m median along the route, 42 % within 10 m (the best of all drives); map position
15.4 m median, 78 % within 30 m (10_06: 11.5 m, 89 %). The clear view helps seeing, not placing.

**The matcher's odometer is now speed.py** (`USE_SPEED`). Junction events in the slow queue on Calea
Ieșilor came out 230–330 m ahead on both 10_06 and 10_02 although the speed there is right (735 m
driven vs 684 m GPS): the scene-flow odometer was the weak part. Tested once before on 10_06 alone with
the older speed (no gain); now, on five drives, physical junctions at the right moment:

| drive | scene flow | speed.py | + route length |
|---|---|---|---|
| 10_06 | 6/18 | 7/18 | 7/18 |
| 17_08 | 13/33 | 14/33 | 14/33 |
| 19_26 | 6/12 | 9/12 | 9/12 |
| 15_48 | 0/3 (wrong road) | 1/3 | **2/3, right road** |
| 10_02 | 4/19 | 8/19 | 8/19 |
| total | 29/85 | 39/85 | **40/85** |

**Route length.** The matcher stretched the odometer to fit each candidate route, so a longer route
cost nothing: 15_48 took a 3.54 km detour (Bd. Dacia, Str. Aeroport) over the 2.83 km of Șos. Muncești
it drove. The raw speed reads 0.81–0.96 × the true distance on five drives (median 0.86 — one constant
for this camera); a route needing the odometer more than ~12 % off that is penalised (`length_cost`).
The other four drives are unchanged.

Photo-pin settings swept (inliers 40–150, one pin per 5–20 s, 1–3 photos): the current ones
(100, 10 s, 3) are within 0.6 m of the best on average — not changed.

**GPU dispatcher** (`gpu.py`): Modal → Kaggle → Lightning (not set up) → Colab → the laptop, falling
through on refusal or failure. `modal_run.py`: inputs on a Modal volume, a T4, the image cached
(numpy 2 for current pyarrow; pandas < 3 and OpenCV < 5 as on the laptop). All remote steps go through it
(dense, scene, points, perception, light check, photo matching and verification); the steps with their own
Colab code keep it as the last resort. The light check on Modal reproduced Kaggle's scores (≤ 0.02).

15_48 on the right road: photo matching (Modal) 36 m along the route, verified photos 24 m (49 % of
frames); map position **52.8 m** median, 27 % within 30 m (was 626 m on the wrong road). Still the
weakest drive: only 158 s inside the street map, on a long, nearly straight road — the heading-profile
alignment has little to hold on to, and few photos.

**Straight roads (2026-10-26).** 15_48's odometer was 2 % right (2,892 m vs 2,840 m), but the
heading-profile alignment put it 200 m ahead in the first 10 s and it stayed there: a straight road
has no shape to align. A pull toward the odometer's guess (`ODO_PULL`, 2° per step per 100 m) fixes it,
but on city routes it hurt (night drive 15.8 → 21–26 m, also when applied only on their straight
parts). The routes separate clearly by the map alone — 15_48 91 % straight (0.7 bends/km), the city
routes 48–60 % (2.9–3.8 bends/km) — so the pull is used only on routes ≥ 80 % straight.
15_48: **52.8 → 15.2 m** median, 27 → 85 % within 30 m; the four city drives unchanged.

### Vehicles bound to one lane per track (2026-10-26)

Snapped frame by frame, a car between two map lanes flipped between them: 0.57 lane switches per track,
26 % of tracks (17_08), 20.8 sideways jumps > 1.5 m per 1,000 steps. Now each track's lanes are chosen
for the whole track (Viterbi, `map_layer.snap_vehicles`): a lane is the k-th from the kerb in its
direction (SUMO lane index — ids change at every edge even straight on); staying is free, a lane change
costs `LANE_CHANGE`, being off every lane costs as much as a 2.5 m miss each frame. The car sits on that
lane's centre line and points along it.

| 17_08 | per frame | change cost 12 | **25** | 50 |
|---|---|---|---|---|
| jumps > 1.5 m / 1,000 steps | 20.8 | 11.3 | **8.4** | 5.8 |
| tracks with a lane switch | 26 % | 18 % | **16 %** | 11 % |
| positions on a map lane | 14,059 | 13,653 | **13,144** | 12,291 |

(10_06: 20.4 → 11.5 jumps at 25.) 25: jumps −45–60 %, 5–7 % fewer positions on a lane; at 50 real
positions start falling off the lanes. Cars on the video's own lanes (no map lane) are still per frame.

### Our lane on the map, lane counts from the video, frozen cars (2026-10-27)

Reported on 10_02: the map lanes shifted at 2:20 (Mircești), the car in the middle lane at 10:11 on
Bd. Mihai Viteazul although it drove in the right one, a car "stuck" on the right at 2:13.

- **Sideways lock.** It matched the map's lane boundaries to the lane lines the video sees — ambiguous by
  a whole lane, since lines repeat every lane width — and it could shift ±6 m only, while the route runs
  on the street's centre line (on a 3 + 3 boulevard our lane is ~7 m from it). Now the map's right
  edge is also matched to the kerb / road edge the video sees, lines count by their mean (one reading
  per 5-m band), ±12 m. Map right edge vs seen kerb, median: 10_02 1.54 → 0.09 m, 10_06 3.57 → 0.15,
  17_08 3.42 → 0.20, 19_26 1.21 → 0.13, 15_48 3.51 → 0.71 m. At 10:11 now in the right lane. (GPS is too
  coarse sideways — 1–4 m — to score lanes.)
- **Lane counts** (`lane_count.py`). Only 44 % of Chișinău's car roads have an OSM `lanes` tag (residential
  15 %, secondary 74 %); without one SUMO gives 1 lane per direction — Mircești has 1 + 2. Per second:
  the lane lines between the kerbs count the lanes (the far kerb is often read too near: Mircești 6.8 m
  for 3 lanes); per street and drive, which lanes we drove in (right-hand traffic: those are our
  direction's) and the vehicles seen in each lane. Drives both ways settle the split: towards the mall
  only the right lane of 3 was driven, towards home the right two → **1 + 2**, as the driver counts.
  On the 45 OSM-tagged streets measured, the method gets a carriageway's lane count right 33 % of the
  time and the split 11 % (extra lines for parking / bus strips, far kerbs, and vehicle directions from
  box shape or image motion are unreliable), so the map uses a count only where ≥ 2 drives agree on the
  total: 3 ways so far (Mircești, two stretches of Petricani). `data/video/lane_counts.parquet` is
  city-wide: every drive adds to it.
- **Frozen cars.** The 3D view stores only moments with detections and kept the last one through gaps:
  a car last seen at 2:09 stood until the next detection at 2:14. Objects older than 0.4 s are hidden.

**Lane counts, second pass (2026-10-27).** Solid vs dashed from single frames (the multi-frame ground
raster closes a dashed line's gaps): along each line the painted share of 4–24 m ahead — Mircești's
solid centre line 0.85, the dashed line between the oncoming lanes 0.20. But lines between our own lanes
read solid near junctions (Mihai Viteazul at 10:11: 0.90), so "first solid line left of the lanes we
drove" is only a hint. Also dropped: lines < 1.5 m from the kerb or each other (edge lines, double
readings; 2.3 m dropped Mircești's far lanes, which look 1.9 m wide), outer lanes of mostly parked cars.
On the 52 OSM-tagged ways measured: lane total 31 → 33–37 %, split 12 → 13–15 % — no real gain. What
does work is the lanes we drove in, from drives both ways; the map now uses a video count only on ways
driven both ways (or one-way) with ≥ 2 drives agreeing: Mircești 2 + 1 (towards home + towards the mall)
and three Petricani ways.
