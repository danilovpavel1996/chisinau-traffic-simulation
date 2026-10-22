"""
compute.py
----------
Keep the laptop usable while heavy steps run: cap CPU threads, lower priority,
and show progress with time remaining.

    TRAFFIC_THREADS=4 (default) → ~60–65 % of a 6-core MacBook; raise for speed,
    lower for silence. Applies to PyTorch and OpenCV.
"""

from __future__ import annotations

import os

THREADS = int(os.environ.get("TRAFFIC_THREADS", 4))
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, str(THREADS))


def limit_cpu():
    """Cap PyTorch / OpenCV threads and lower process priority (call before heavy work)."""
    import cv2
    import torch
    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(1)   # allowed only once per process
    except RuntimeError:
        pass
    cv2.setNumThreads(THREADS)
    try:
        os.nice(10)              # lower priority: interactive apps stay responsive
    except OSError:
        pass
    print(f"CPU: {THREADS} threads, low priority (TRAFFIC_THREADS to change)", flush=True)


def progress(total: int, desc: str, initial: int = 0):
    """Progress bar with rate and time remaining; works in a terminal and in log files."""
    from tqdm import tqdm
    return tqdm(total=total, initial=initial, desc=desc, unit="frame", smoothing=0.05,
                mininterval=2.0, dynamic_ncols=True,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed} elapsed, {remaining} left, {rate_fmt}]")
