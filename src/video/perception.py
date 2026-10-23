"""
perception.py
-------------
Single pass over a drive's video:
  - YOLO detection + ByteTrack tracking of vehicles and traffic lights
  - colour (red / yellow / green) of every detected traffic light
  - optical flow: the car's own motion (speed, rotation, horizon row) and each
    vehicle's motion relative to the road below it (parked vs moving)

Usage:
    .venv-video/bin/python -m src.video.perception <drive> [--ego-only]

Settings (stride, bonnet line) come from configs/drives.yaml.
Outputs (data/video/processed/<drive>/):
    detections.parquet   one row per (sampled frame, detected object), incl. object vs
                         road flow (obj_dx/dy, bg_dx/dy) for parked-car detection
    ego.parquet          one row per sampled frame: ego_mag, yaw_dx/yaw_slope,
                         bandN_dy + foe_y (road-plane speed)
    meta.json            video + run parameters
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.video import drives
from src.video.compute import limit_cpu, progress
from src.video.drives import ROOT

# COCO class ids we care about
CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 9: "traffic_light"}
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}

MODEL = ROOT / "models" / "yolo11s.pt"   # yolo11s: ~2x the oncoming/light recall of yolo11n, 1.45x slower on CPU
TRACKER = str(Path(__file__).with_name("bytetrack_dashcam.yaml"))


# ── Traffic light colour ──────────────────────────────────────────────────────
def light_state(frame: np.ndarray, box: tuple[int, int, int, int]) -> str:
    """Classify a traffic-light crop by its brightest saturated hue."""
    x1, y1, x2, y2 = box
    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
    if crop.size == 0 or crop.shape[0] < 6:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (s > 90) & (v > 150)
    counts = {
        "red":    int(np.sum(lit & ((h < 10) | (h > 165)))),
        "yellow": int(np.sum(lit & (h >= 12) & (h <= 35))),
        "green":  int(np.sum(lit & (h >= 45) & (h <= 95))),
    }
    best = max(counts, key=counts.get)
    # need a lit blob of a few pixels relative to the crop size
    return best if counts[best] >= max(4, 0.02 * crop.shape[0] * crop.shape[1]) else "unknown"


N_BANDS = 8          # horizontal bands between BAND_TOP and the bonnet line
BAND_TOP = 0.15      # fraction of frame height where the bands start


def foe_row(p: np.ndarray, d: np.ndarray, sel: np.ndarray) -> float:
    """
    Row of the focus of expansion: the point every flow vector points away
    from when driving straight (= the horizon row for a forward-looking camera).
    Least-squares intersection of the flow lines.
    """
    m = sel & (np.linalg.norm(d, axis=1) > 0.5)
    if m.sum() < 20:
        return np.nan
    pp, dd = p[m], d[m] / np.linalg.norm(d[m], axis=1, keepdims=True)
    nrm = np.column_stack([-dd[:, 1], dd[:, 0]])          # line normals
    A, b = nrm, np.sum(nrm * pp, axis=1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    return float(sol[1])


def robust_line(x: np.ndarray, y: np.ndarray, rounds: int = 3) -> tuple[float, float]:
    """Least-squares y = a + b·x, refitted after trimming points beyond 2.5 MAD."""
    keep = np.ones(len(x), bool)
    a = b = 0.0
    for _ in range(rounds):
        A = np.column_stack([np.ones(keep.sum()), x[keep]])
        (a, b), *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        res = y - (a + b * x)
        mad = np.median(np.abs(res[keep])) + 1e-6
        keep = np.abs(res) < 2.5 * 1.4826 * mad
        if keep.sum() < 10:
            break
    return float(a), float(b)


# ── Scene motion ──────────────────────────────────────────────────────────────
class SceneFlow:
    """
    Optical flow per sampled frame (all values in full-resolution pixels per
    sampled frame). Sparse corners for the ego signals, dense flow for boxes:

    ego_mag   median flow magnitude on the road surface → moving / stopped
    bandN_dy  vertical flow of central points in N row bands, and foe_y the
              horizon row → road-plane speed in motion.py
    yaw_dx    rotation (turning), as horizontal flow at the image centre. Fit over
              all background points: dx = yaw_dx + yaw_slope · (x − centre)/width.
              Rotation shifts every point equally (intercept); driving forward
              streams points outward ∝ distance from centre (slope). A plain
              median mixes the two → a fake left turn whenever the car moves.
    per box   flow of the vehicle vs flow of the road surface just below it;
              a parked car moves exactly like the ground it stands on (residual ≈ 0)

    The speed signals (ego_mag, bandN_dy) are measured between the sampled frame
    and the frame just before it, × the stride (same units): across a whole
    stride (0.2 s) the road near the car moves further than the tracker can
    follow above ~50 km/h, and speed saturated (60–80 km/h looked like 30).
    Rotation and the horizon stay on the full stride — over one frame the
    rotation is 6× smaller than the tracking noise and fake turns appeared.

    The bottom `hood` fraction of the frame (dashboard / bonnet) is ignored.
    """

    def __init__(self, width: int, height: int, scale: float = 0.5, hood: float = 0.15):
        self.scale = scale
        self.w, self.h = int(width * scale), int(height * scale)
        self.hood_y = self.h * (1 - hood)
        self.prev = None

    def _median(self, d: np.ndarray, sel: np.ndarray, min_n: int = 4):
        return np.median(d[sel], axis=0) / self.scale if sel.sum() >= min_n else None

    def _box_motion(self, prev: np.ndarray, gray: np.ndarray, sb: list) -> list:
        """
        Dense (Farneback) flow at quarter resolution, so plain asphalt below a car
        still gets a motion estimate. Per box: median flow inside the box vs the
        road strip just below it (same distance as the car → no parallax).
        """
        q = 0.5  # relative to the half-res working frame
        flow = cv2.calcOpticalFlowFarneback(cv2.resize(prev, None, fx=q, fy=q),
                                            cv2.resize(gray, None, fx=q, fy=q), None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        hq, wq = flow.shape[:2]
        occupied = np.zeros((hq, wq), bool)
        qb = [tuple(int(round(c * q)) for c in b) for b in sb]
        for x1, y1, x2, y2 in qb:
            occupied[max(y1, 0):y2, max(x1, 0):x2] = True
        hood_q = int(self.hood_y * q)

        out = []
        for x1, y1, x2, y2 in qb:
            x1, x2 = max(x1, 0), min(x2, wq)
            bw, bh = x2 - x1, y2 - y1
            if bw < 4 or bh < 4:
                out.append(None)
                continue
            inner = flow[y1 + bh // 6:y2 - bh // 6, x1 + bw // 6:x2 - bw // 6].reshape(-1, 2)
            g0, g1 = y2, min(y2 + max(3, int(0.3 * bh)), hood_q)
            gpatch = flow[g0:g1, x1:x2]
            gfree = ~occupied[g0:g1, x1:x2]
            ground = gpatch[gfree].reshape(-1, 2)
            if len(inner) < 4 or len(ground) < 6:
                out.append(None)
                continue
            mo = np.median(inner, axis=0) / (self.scale * q)
            mb = np.median(ground, axis=0) / (self.scale * q)
            out.append(dict(obj_dx=float(mo[0]), obj_dy=float(mo[1]),
                            bg_dx=float(mb[0]), bg_dy=float(mb[1])))
        return out

    def update(self, frame: np.ndarray, boxes: list[tuple[int, int, int, int]],
               before: np.ndarray | None = None, gap: int = 1):
        """before: the video frame just before this one (speed flow over 1 frame, × gap)."""
        gray = cv2.cvtColor(cv2.resize(frame, (self.w, self.h)), cv2.COLOR_BGR2GRAY)
        ego = dict(ego_mag=np.nan, yaw_dx=np.nan, yaw_slope=np.nan, foe_y=np.nan,
                   **{f"band{i}_dy": np.nan for i in range(N_BANDS)})
        per_box = [None] * len(boxes)
        if self.prev is None:
            self.prev = gray
            return ego, per_box

        prev, self.prev = self.prev, gray
        s = self.scale
        sb = [(x1 * s, y1 * s, x2 * s, y2 * s) for x1, y1, x2, y2 in boxes]
        mask = np.zeros_like(gray)
        mask[:int(self.hood_y)] = 255

        def track(ref, k):
            """Background corner flow ref → this frame (× k): positions, displacements, mask."""
            pts = cv2.goodFeaturesToTrack(ref, maxCorners=1200, qualityLevel=0.01, minDistance=5, mask=mask)
            if pts is None or len(pts) < 20:
                return None
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(ref, gray, pts, None)
            good = st.ravel() == 1
            p, d = nxt[good].reshape(-1, 2), (nxt[good] - pts[good]).reshape(-1, 2) * k
            in_any = np.zeros(len(p), bool)
            for x1, y1, x2, y2 in sb:
                in_any |= (p[:, 0] >= x1) & (p[:, 0] <= x2) & (p[:, 1] >= y1) & (p[:, 1] <= y2)
            return p, d, ~in_any

        full = track(prev, 1.0)
        if full is None:
            return ego, per_box
        near = full
        if before is not None:
            near = track(cv2.cvtColor(cv2.resize(before, (self.w, self.h)), cv2.COLOR_BGR2GRAY), float(gap)) or full

        p, d, bg = near                                    # speed: one-frame flow
        x, y = p[:, 0], p[:, 1]

        # road band: 30% of frame height just above the bonnet line
        road = (bg & (y > self.hood_y - 0.3 * self.h) & (y < self.hood_y)
                & (x > self.w * 0.15) & (x < self.w * 0.85))
        if road.sum() >= 10:
            ego["ego_mag"] = float(np.median(np.linalg.norm(d[road], axis=1)) / s)
        # ground-plane speed: vertical flow of central (lane) points per image row band;
        # analyze.py divides by (row − horizon)² → speed independent of street width
        centre = bg & (x > self.w * 0.3) & (x < self.w * 0.7)
        edges = np.linspace(self.h * BAND_TOP, self.hood_y, N_BANDS + 1)
        for i in range(N_BANDS):
            sel = centre & (y >= edges[i]) & (y < edges[i + 1])
            if sel.sum() >= 5:
                ego[f"band{i}_dy"] = float(np.median(d[sel, 1]) / s)

        p, d, bg = full                                    # rotation + horizon: full-stride flow
        x, y = p[:, 0], p[:, 1]
        scene = bg & (y > self.h * 0.02) & (y < self.hood_y)
        if scene.sum() >= 20:
            r, slope = robust_line((x[scene] - self.w / 2) / self.w, d[scene, 0])
            ego["yaw_dx"], ego["yaw_slope"] = r / s, slope / s
        ego["foe_y"] = foe_row(p, d, bg) / s
        if sb:
            per_box = self._box_motion(prev, gray, sb)
        return ego, per_box


# ── Main pass ─────────────────────────────────────────────────────────────────
def run(drv: dict, ego_only: bool = False, start: float = 0.0, duration: float | None = None,
        imgsz: int = 640, conf: float = 0.1) -> Path:
    video, stride, hood = drv["video"], drv["stride"], drv["camera"]["hood"]
    limit_cpu()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    f_start = int(start * fps)
    f_end = n_total if duration is None else min(n_total, f_start + int(duration * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)

    out_dir = drv["folder"]
    saved = None
    if ego_only:  # recompute motion only; mask cars with the saved detections
        saved = {f: g for f, g in pd.read_parquet(out_dir / "detections.parquet").groupby("frame")}
    else:
        model = YOLO(str(MODEL))
    flow = SceneFlow(W, H, hood=hood)

    det_rows, ego_rows = [], []
    t0 = time.time()
    f = f_start
    last = None                                         # the decoded frame before the current one
    bar = progress((f_end - f_start + stride - 1) // stride, "perception" + (" (motion only)" if ego_only else ""))
    while f < f_end:
        ok, frame = cap.read()
        if not ok:
            break
        before, last = last, frame
        if (f - f_start) % stride:
            f += 1
            continue
        t = f / fps

        rows = []
        if saved is not None:
            g = saved.get(f)
            veh = [] if g is None else g[g.cls.isin(VEHICLE_CLASSES)][["x1", "y1", "x2", "y2"]].to_numpy()
            ego, _ = flow.update(frame, [tuple(b) for b in veh], before, stride)
            ego_rows.append(dict(frame=f, t=t, **ego))
            bar.update(1)
            f += 1
            continue

        res = model.track(frame, persist=True, tracker=TRACKER, imgsz=imgsz,
                          conf=conf, classes=list(CLASSES), verbose=False)[0]
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
            cls = res.boxes.cls.cpu().numpy().astype(int)
            cf = res.boxes.conf.cpu().numpy()
            ids = (res.boxes.id.cpu().numpy().astype(int)
                   if res.boxes.id is not None else np.full(len(cls), -1))
            for (x1, y1, x2, y2), c, p, tid in zip(xyxy, cls, cf, ids):
                name = CLASSES[c]
                if name in VEHICLE_CLASSES and (y1 + y2) / 2 > H * (1 - hood):
                    continue  # bonnet / dashboard detected as a car
                rows.append(dict(frame=f, t=t, track_id=int(tid), cls=name, conf=float(p),
                                 x1=x1, y1=y1, x2=x2, y2=y2,
                                 light=light_state(frame, (x1, y1, x2, y2))
                                 if name == "traffic_light" else None))

        # flow on the clean frame, before anything is drawn on it
        veh = [r for r in rows if r["cls"] in VEHICLE_CLASSES]
        ego, per_box = flow.update(frame, [(r["x1"], r["y1"], r["x2"], r["y2"]) for r in veh], before, stride)
        for r, m in zip(veh, per_box):
            r.update(m or {})
        det_rows.extend(rows)
        ego_rows.append(dict(frame=f, t=t, **ego))


        bar.update(1)
        f += 1

    bar.close()
    cap.release()

    ego_df = pd.DataFrame(ego_rows)
    ego_df.to_parquet(out_dir / "ego.parquet", index=False)
    if ego_only:
        print(f"ego.parquet rewritten ({len(ego_df)} frames, {time.time() - t0:.0f} s)")
        return out_dir
    det = pd.DataFrame(det_rows)
    det.to_parquet(out_dir / "detections.parquet", index=False)
    meta = dict(video=str(video.relative_to(ROOT)), fps=fps, width=W, height=H, stride=stride,
                start_s=start, end_s=f / fps, model=MODEL.name, imgsz=imgsz, conf=conf,
                hood=hood, tracker=Path(TRACKER).name,
                n_sampled_frames=len(ego_df), n_detections=len(det),
                runtime_s=round(time.time() - t0, 1))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drive", help="drive key in configs/drives.yaml")
    ap.add_argument("--ego-only", action="store_true",
                    help="recompute ego.parquet only, reusing saved detections (fast)")
    ap.add_argument("--start", type=float, default=0.0, help="start time, seconds (for quick tests)")
    ap.add_argument("--duration", type=float, default=None, help="seconds to process (default: all)")
    a = ap.parse_args()
    run(drives.load(a.drive), a.ego_only, a.start, a.duration)


if __name__ == "__main__":
    main()
