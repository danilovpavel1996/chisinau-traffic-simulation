"""
modal_run.py
------------
GPU jobs on Modal (modal.com): free monthly credits, a container that starts in seconds and is not
taken back mid-job. The same job scripts as on Colab / Kaggle (inputs at /content paths).

    inputs    uploaded to a Modal volume (chisinau-jobs/<job>/in/…), copied to their /content paths
    run       the script on a T4 (an image with torch + the job's own pip packages, cached by Modal)
    outputs   copied to the volume (…/out/…) and downloaded; the job's folder is removed after

The account's token is in ~/.modal.toml (the owner's `modal token new`; never read here).
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

VOLUME = "chisinau-jobs"
GPU = "T4"
BASE_PIP = ["torch==2.5.1", "torchvision==0.20.1", "opencv-python-headless<5", "pandas<3", "pyarrow", "scipy", "pillow"]


class Unavailable(RuntimeError):
    """Modal could not run the job (not configured, no credits, outage) — try the next service."""


def configured() -> bool:
    return (Path.home() / ".modal.toml").exists()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")[:40]


def run_job(name: str, script: str, uploads: dict[str, Path], downloads: dict[str, Path], work: Path,
            pip: list[str] | None = None, timeout_h: float = 6.0) -> None:
    import modal
    if not configured():
        raise Unavailable("no Modal token (~/.modal.toml)")
    job = f"{_slug(name)}-{int(time.time())}"
    try:
        vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
        up, down = {}, {}
        with vol.batch_upload(force=True) as b:
            for i, (remote, local) in enumerate(uploads.items()):
                rel = f"/{job}/in/{i}_{Path(remote).name}"
                b.put_file(str(local), rel)
                up[remote] = "/vol" + rel
        for i, remote in enumerate(downloads):
            down[remote] = f"/vol/{job}/out/{i}_{Path(remote).name}"
        image = (modal.Image.debian_slim(python_version="3.11")
                 .apt_install("libgl1", "libglib2.0-0", "git", "ffmpeg")
                 .pip_install(*(BASE_PIP + list(pip or [])))
                 # a job's own `pip install` must not swap numpy under the already-imported pandas
                 .run_commands("python -c \"import numpy; print('numpy==' + numpy.__version__)\" > /constraints.txt")
                 .env({"PIP_CONSTRAINT": "/constraints.txt"}))
        app = modal.App(f"chisinau-{_slug(name)[:30]}")
    except Exception as e:                                     # setting up failed: Modal unreachable / quota
        raise Unavailable(f"Modal set-up failed: {e}") from e

    @app.function(image=image, gpu=GPU, timeout=int(timeout_h * 3600), volumes={"/vol": vol}, serialized=True)
    def run(script: str, up: dict, down: dict):
        import os as _os
        import shutil as _sh
        for r, src in up.items():
            _os.makedirs(_os.path.dirname(r) or "/", exist_ok=True)
            _sh.copy(src, r)
        _os.makedirs("/content", exist_ok=True)
        _os.chdir("/content")
        exec(compile(script, "job", "exec"), {"__name__": "__main__"})
        for r, dst in down.items():
            _os.makedirs(_os.path.dirname(dst), exist_ok=True)
            _sh.copy(r, dst)
        import modal as _m
        _m.Volume.from_name(VOLUME).commit()

    t0 = time.time()
    print(f"running {name} on Modal ({GPU})…", flush=True)
    try:
        with modal.enable_output(), app.run():
            run.remote(script, up, down)
    except modal.exception.Error as e:                         # infrastructure (no credits, outage …)
        msg = str(e).lower()
        if any(k in msg for k in ("billing", "credit", "quota", "limit", "unavailable", "capacity")):
            raise Unavailable(f"Modal refused: {e}") from e
        raise
    for remote, dst in down.items():
        local = downloads[remote]
        local.parent.mkdir(parents=True, exist_ok=True)
        with open(local, "wb") as fh:
            for chunk in vol.read_file(dst.removeprefix("/vol")):
                fh.write(chunk)
    try:
        vol.remove_file(f"/{job}", recursive=True)
    except Exception:
        pass
    print(f"Modal job {name} done in {(time.time() - t0) / 60:.1f} min", flush=True)
