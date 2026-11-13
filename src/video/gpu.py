"""
gpu.py
------
One entry point for remote GPU jobs, trying the free services in order of reliability (the owner's
choice, 2026-10-26) and falling through when one refuses, has no credits or fails:

    1. Modal       modal_run.py    free monthly credits, starts in seconds, never taken back mid-job
    2. Kaggle      kaggle_run.py   30 GPU h a week; refuses when two sessions already run (409)
    3. Lightning   (not set up yet: needs the owner's `lightning login`)
    4. Colab       colab.py        unpublished limits, may reclaim the runtime mid-job
    (the MacBook last: the caller's own local code path)

A job is a script written for /content paths (as on Colab), its inputs {remote path: local file} and
outputs {remote path: local file}. GPU_ORDER (environment, e.g. "kaggle,colab") overrides the order.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

ORDER = ("modal", "kaggle", "lightning", "colab")
CLOUD = ORDER[:-1]          # for steps that keep their own Colab code as the fallback


def _modal(name, script, uploads, downloads, work, pip):
    from src.video import modal_run
    modal_run.run_job(name, script, uploads, downloads, work, pip)


def _kaggle(name, script, uploads, downloads, work, pip):
    from src.video import kaggle_run
    kaggle_run.run_job(name, script, uploads, downloads, work, pip)


def _lightning(name, script, uploads, downloads, work, pip):
    raise RuntimeError("Lightning AI is not set up (no `lightning login` yet)")


def _colab(name, script, uploads, downloads, work, pip):
    from src.video.colab import Session
    with Session(name.replace(" ", "-")[:40]) as s:
        for remote, local in uploads.items():
            s.upload(Path(local), remote)
        if pip:
            s.install(*pip)
        s.start(script)
        if s.wait() != 0:
            print(s.log())
            raise RuntimeError(f"{name} failed on Colab")
        for remote, local in downloads.items():
            s.download(remote, Path(local))


BACKENDS = dict(modal=_modal, kaggle=_kaggle, lightning=_lightning, colab=_colab)


def run_job(name: str, script: str, uploads: dict[str, Path], downloads: dict[str, Path], work: Path,
            pip: list[str] | None = None, order: tuple[str, ...] | None = None) -> str:
    """Run the job on the first service that accepts and finishes it; returns that service's name."""
    order = order or tuple(os.environ.get("GPU_ORDER", ",".join(ORDER)).split(","))
    errors = []
    for b in order:
        t0 = time.time()
        try:
            BACKENDS[b](name, script, uploads, downloads, work / b, pip)
        except Exception as e:
            errors.append(f"{b}: {str(e)[:300]}")
            print(f"{name}: {b} failed after {(time.time() - t0) / 60:.1f} min ({str(e)[:300]}); trying the next…", flush=True)
            continue
        print(f"{name}: done on {b} in {(time.time() - t0) / 60:.1f} min", flush=True)
        return b
    raise RuntimeError(f"{name}: no GPU service could run it —\n" + "\n".join(errors))
