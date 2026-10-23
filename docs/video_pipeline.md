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
