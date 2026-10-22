"""
Dashcam / phone video → traffic data.

    perception.py   detection + tracking + optical flow  (one pass over the video)
    motion.py       the car's own motion: speed, rotation, turns, stops
    traffic.py      other road users: track labels (oncoming, parked, …), per-minute counts
    roads.py        OpenStreetMap street graph
    localize.py     particle-filter map matching of a GPS-less drive
    viewer/         synced video + map viewer
    pipeline.py     all steps for one drive
    drives.py       drive registry (configs/drives.yaml)
"""
