"""
Dashcam / phone video → traffic data.

    perception.py   detection + tracking + optical flow  (one pass over the video)
    motion.py       the car's own motion: speed, rotation, turns, stops
    traffic.py      other road users: track labels (oncoming, parked, …), per-minute counts
    scene.py        street-scene segmentation: crosswalks, lanes, curbs, guard rails, signs
    roads.py        OpenStreetMap street graph
    localize.py     particle-filter map matching of a GPS-less drive
    truth.py        ground-truth route from a driver's street directions
    viewer/         synced video + map viewer
    pipeline.py     all steps for one drive
    drives.py       drive registry (configs/drives.yaml)
"""
