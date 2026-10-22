"""
Dashcam video → traffic observations.

detect_track.py  one pass over a video: YOLO detection + ByteTrack tracking,
                 ego-motion (optical flow) and traffic-light colour
analyze.py       tracks → per-track manoeuvre labels + per-minute traffic stats
"""
