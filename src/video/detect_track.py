"""
detect_track.py
---------------
Single pass over a dashcam video:
  - YOLO detection + ByteTrack tracking of vehicles and traffic lights
  - optical flow: ego speed index, turning (yaw), and per-vehicle motion
    relative to its background (parked vs moving)
  - colour (red / yellow / green) of every detected traffic light

Usage:
    .venv-video/bin/python -m src.video.detect_track data/video/raw/clip.mp4 \
        --start 540 --duration 180 --preview [--model models/yolo11s.pt --tag s]

Outputs (data/video/processed/<video_name>/):
    detections.parquet   one row per (sampled frame, detected object), incl. object vs
                         background flow (obj_dx/dy, bg_dx/dy) for parked-car detection
    ego.parquet          one row per sampled frame: ego_mag (speed index), yaw_dx (turning)
    meta.json            video + run parameters
    preview.mp4          annotated video (with --preview)
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

# COCO class ids we care about
CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 9: "traffic_light"}
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}

OUT_ROOT = Path("data/video/processed")


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
    road_dy   downward flow of the asphalt just above the bonnet (fails if the
              phone sits low: the patch is bonnet / reflection)
    bandN_dy  vertical flow of central points in N row bands, and foe_y the
              horizon row → ground-plane speed in analyze.py
    yaw_dx    rotation (turning), as horizontal flow at the image centre. Fit over
              all background points: dx = yaw_dx + yaw_slope · (x − centre)/width.
              Rotation shifts every point equally (intercept); driving forward
              streams points outward ∝ distance from centre (slope). A plain
              median mixes the two → a fake left turn whenever the car moves.
    per box   flow of the vehicle vs flow of the road surface just below it;
              a parked car moves exactly like the ground it stands on (residual ≈ 0)

    The bottom `hood` fraction of the frame (dashboard / bonnet) is ignored.
    """

    def __init__(self, width: int, height: int, scale: float = 0.5, hood: float = 0.15):
        self.scale = scale
        self.w, self.h = int(width * scale), int(height * scale)
        self.hood_y = self.h * (1 - hood)
        self.prev = None

    def _median(self, d: np.ndarray, sel: np.ndarray, min_n: int = 4):
        return np.median(d[sel], axis=0) / self.scale if sel.sum() >= min_n else None

    def _road_speed(self, prev: np.ndarray, gray: np.ndarray, sb: list) -> float:
        """
        Metric speed index: downward flow of the asphalt patch just above the
        bonnet, centre of the lane. That patch is at a fixed distance from the
        camera (fixed height and tilt), so its flow ∝ speed regardless of the
        surroundings — unlike whole-scene flow, which depends on how close the
        buildings are. NaN when a vehicle covers the patch.
        """
        y1, y2 = int(self.hood_y - 0.12 * self.h), int(self.hood_y - 0.02 * self.h)
        x1, x2 = int(0.35 * self.w), int(0.65 * self.w)
        for bx1, by1, bx2, by2 in sb:
            if by2 > y1 and bx1 < x2 and bx2 > x1:
                return np.nan
        a, b = prev[y1 - 20:y2 + 20, x1:x2], gray[y1 - 20:y2 + 20, x1:x2]
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        return float(np.median(flow[20:-20, :, 1]) / self.scale)

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

    def update(self, frame: np.ndarray, boxes: list[tuple[int, int, int, int]]):
        gray = cv2.cvtColor(cv2.resize(frame, (self.w, self.h)), cv2.COLOR_BGR2GRAY)
        ego = dict(ego_mag=np.nan, yaw_dx=np.nan, yaw_slope=np.nan, road_dy=np.nan, foe_y=np.nan,
                   **{f"band{i}_dy": np.nan for i in range(N_BANDS)})
        per_box = [None] * len(boxes)
        if self.prev is None:
            self.prev = gray
            return ego, per_box

        prev, self.prev = self.prev, gray
        mask = np.zeros_like(gray)
        mask[:int(self.hood_y)] = 255
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=1200, qualityLevel=0.01,
                                      minDistance=5, mask=mask)
        if pts is None or len(pts) < 20:
            return ego, per_box
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None)
        good = st.ravel() == 1
        p, d = nxt[good].reshape(-1, 2), (nxt[good] - pts[good]).reshape(-1, 2)
        x, y = p[:, 0], p[:, 1]

        s = self.scale
        sb = [(x1 * s, y1 * s, x2 * s, y2 * s) for x1, y1, x2, y2 in boxes]
        in_any = np.zeros(len(p), bool)
        for x1, y1, x2, y2 in sb:
            in_any |= (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)
        bg = ~in_any

        # road band: 30% of frame height just above the bonnet line
        road = (bg & (y > self.hood_y - 0.3 * self.h) & (y < self.hood_y)
                & (x > self.w * 0.15) & (x < self.w * 0.85))
        if road.sum() >= 10:
            ego["ego_mag"] = float(np.median(np.linalg.norm(d[road], axis=1)) / s)
        scene = bg & (y > self.h * 0.02) & (y < self.hood_y)
        if scene.sum() >= 20:
            r, slope = robust_line((x[scene] - self.w / 2) / self.w, d[scene, 0])
            ego["yaw_dx"], ego["yaw_slope"] = r / s, slope / s

        ego["road_dy"] = self._road_speed(prev, gray, sb)
        # ground-plane speed: vertical flow of central (lane) points per image row band;
        # analyze.py divides by (row − horizon)² → speed independent of street width
        centre = bg & (x > self.w * 0.3) & (x < self.w * 0.7)
        edges = np.linspace(self.h * BAND_TOP, self.hood_y, N_BANDS + 1)
        for i in range(N_BANDS):
            sel = centre & (y >= edges[i]) & (y < edges[i + 1])
            if sel.sum() >= 5:
                ego[f"band{i}_dy"] = float(np.median(d[sel, 1]) / s)
        ego["foe_y"] = foe_row(p, d, bg) / s
        if sb:
            per_box = self._box_motion(prev, gray, sb)
        return ego, per_box


# ── Main pass ─────────────────────────────────────────────────────────────────
TRACKER = str(Path(__file__).with_name("bytetrack_dashcam.yaml"))
LIGHT_BGR = {"red": (0, 0, 255), "yellow": (0, 215, 255), "green": (0, 200, 0)}


def draw(frame: np.ndarray, rows: list[dict], t: float, ego: dict):
    for r in rows:
        color = LIGHT_BGR.get(r["light"], (255, 160, 40))
        cv2.rectangle(frame, (r["x1"], r["y1"]), (r["x2"], r["y2"]), color, 2)
        label = f'{r["cls"]} #{r["track_id"]}' + (f' {r["light"]}' if r["light"] else "")
        cv2.putText(frame, label, (r["x1"], max(r["y1"] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f't={t:7.1f}s  ego={ego["ego_mag"]:5.1f}  yaw={ego["yaw_dx"]:+5.1f}',
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)


def run(video: Path, stride: int, start: float, duration: float | None, model_name: str,
        imgsz: int, conf: float, hood: float, preview: bool, tag: str = "",
        ego_only: bool = False) -> Path:
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

    out_dir = OUT_ROOT / (video.stem + (f"_{tag}" if tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = None
    if ego_only:  # recompute motion only; mask cars with the saved detections
        saved = {f: g for f, g in pd.read_parquet(out_dir / "detections.parquet").groupby("frame")}
        preview = False
    else:
        model = YOLO(model_name)
    flow = SceneFlow(W, H, hood=hood)
    writer = None
    if preview:
        writer = cv2.VideoWriter(str(out_dir / "preview.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (W, H))

    det_rows, ego_rows = [], []
    t0 = time.time()
    f = f_start
    while f < f_end:
        ok, frame = cap.read()
        if not ok:
            break
        if (f - f_start) % stride:
            f += 1
            continue
        t = f / fps

        rows = []
        if saved is not None:
            g = saved.get(f)
            veh = [] if g is None else g[g.cls.isin(VEHICLE_CLASSES)][["x1", "y1", "x2", "y2"]].to_numpy()
            ego, _ = flow.update(frame, [tuple(b) for b in veh])
            ego_rows.append(dict(frame=f, t=t, **ego))
            if len(ego_rows) % 500 == 0:
                print(f"  t={t:7.1f}s  {len(ego_rows) / (time.time() - t0):4.1f} frames/s", flush=True)
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
        ego, per_box = flow.update(frame, [(r["x1"], r["y1"], r["x2"], r["y2"]) for r in veh])
        for r, m in zip(veh, per_box):
            r.update(m or {})
        det_rows.extend(rows)
        ego_rows.append(dict(frame=f, t=t, **ego))

        if writer is not None:
            draw(frame, rows, t, ego)
            writer.write(frame)

        if len(ego_rows) % 100 == 0:
            done = (f - f_start) / max(f_end - f_start, 1)
            print(f"  {done:5.1%}  t={t:7.1f}s  {len(ego_rows) / (time.time() - t0):4.1f} frames/s",
                  flush=True)
        f += 1

    cap.release()
    if writer is not None:
        writer.release()

    ego_df = pd.DataFrame(ego_rows)
    ego_df.to_parquet(out_dir / "ego.parquet", index=False)
    if ego_only:
        print(f"ego.parquet rewritten ({len(ego_df)} frames, {time.time() - t0:.0f} s)")
        return out_dir
    det = pd.DataFrame(det_rows)
    det.to_parquet(out_dir / "detections.parquet", index=False)
    meta = dict(video=str(video), fps=fps, width=W, height=H, stride=stride,
                start_s=start, end_s=f / fps, model=model_name, imgsz=imgsz, conf=conf,
                hood=hood, tracker=Path(TRACKER).name,
                n_sampled_frames=len(ego_df), n_detections=len(det),
                runtime_s=round(time.time() - t0, 1))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--stride", type=int, default=3, help="process every Nth frame (3 → ~8 fps at 25 fps)")
    ap.add_argument("--start", type=float, default=0.0, help="start time, seconds")
    ap.add_argument("--duration", type=float, default=None, help="seconds to process (default: all)")
    ap.add_argument("--model", default="models/yolo11s.pt",
                    help="yolo11s: ~2x the oncoming/light recall of yolo11n, 1.45x slower on CPU")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.1,
                    help="detector threshold; low on purpose, ByteTrack filters (see bytetrack_dashcam.yaml)")
    ap.add_argument("--hood", type=float, default=0.15, help="bottom fraction of frame that is dashboard/bonnet")
    ap.add_argument("--tag", default="", help="suffix for the output folder, to compare runs")
    ap.add_argument("--preview", action="store_true", help="write annotated preview.mp4")
    ap.add_argument("--ego-only", action="store_true",
                    help="recompute ego.parquet only, reusing saved detections (fast)")
    a = ap.parse_args()
    run(a.video, a.stride, a.start, a.duration, a.model, a.imgsz, a.conf, a.hood, a.preview, a.tag,
        a.ego_only)


if __name__ == "__main__":
    main()
