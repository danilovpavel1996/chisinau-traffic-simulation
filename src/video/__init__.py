"""
Dashcam / phone video → traffic data.

    perception.py   detection + tracking + optical flow  (one pass over the video)
    motion.py       the car's own motion: speed, rotation, turns, stops
    traffic.py      other road users: track labels (oncoming, parked, …), per-minute counts
    scene.py        street-scene segmentation: crosswalks, lanes, curbs, guard rails, signs (+ class maps)
    road_geometry.py camera-independent road width and side-street openings from the class maps
    events.py       the video as junction events (turns, roundabouts, crosswalks, signals, openings)
    event_match.py  video-only route from start + destination: events aligned with map routes
    roads.py        OpenStreetMap street graph
    localize.py     particle-filter map matching of a GPS-less drive
    truth.py        ground-truth route from a driver's street directions
    position.py     where the car is every second (GPS overlay, else the match), snapped to OSM streets
    lanes.py        lanes seen / our lane from the road scanlines
    signs.py        street names and named places read in the video (Apple Vision OCR, plates dropped)
    street_stats.py stops, queues before signals, speeds per street; aggregate over all drives
    overlay.py      dashcam burnt-in time / speed / GPS → ground truth (Apple Vision OCR)
    dashboard/      one page for all drives: synced video + map
    evaluate.py     robustness: score every matcher run against the truth
    pipeline.py     all steps for one drive
    drives.py       drive registry (configs/drives.yaml)
"""
