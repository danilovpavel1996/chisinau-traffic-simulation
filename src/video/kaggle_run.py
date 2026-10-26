"""
kaggle_run.py
-------------
GPU jobs on Kaggle notebooks (free: ~30 GPU hours a week, T4) when Colab's free GPUs are used up.

    dataset   the job's input files (video chunks, code) as a private Kaggle dataset
    kernel    a private script notebook on a T4 that reads the dataset and writes its results to
              /kaggle/working/out — runs to the end without a connection from the laptop
    results   downloaded when the kernel has finished

The Kaggle command-line tool reads the account's API token from ~/.kaggle/kaggle.json, which the
owner creates (kaggle.com → Settings → API → Create New Token); this code never reads it.
Notebook GPUs and internet access need a phone-verified account.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from src.video.drives import ROOT

KAGGLE = ROOT / ".venv-video" / "bin" / "kaggle"
POLL_S = 60
ACCELERATOR = "NvidiaTeslaT4"


def kaggle(*args: str, capture: bool = True, check: bool = True, timeout: float = 3600) -> str:
    r = subprocess.run([str(KAGGLE), *args], capture_output=capture, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args[:3])} failed: {(r.stderr or r.stdout or '').strip()[-500:]}")
    return (r.stdout or "") + (r.stderr or "")


def username() -> str:
    out = kaggle("config", "view")
    m = re.search(r"username:\s*(\S+)", out)
    if not m:
        raise RuntimeError("no Kaggle account configured: create ~/.kaggle/kaggle.json (kaggle.com → Settings → API)")
    return m.group(1)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")[:50]


def push_dataset(name: str, files: dict[str, Path], stage: Path) -> str:
    """Upload files (name → local path) as a private dataset (a new version if it exists); returns its id."""
    ds = f"{username()}/{slug(name)}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for n, p in files.items():
        try:
            os.link(p, stage / n)                         # no copy of the videos
        except OSError:
            shutil.copy2(p, stage / n)
    (stage / "dataset-metadata.json").write_text(json.dumps(dict(title=name[:50], id=ds, licenses=[dict(name="CC0-1.0")])))
    size = sum(p.stat().st_size for p in files.values()) / 1e6
    exists = kaggle("datasets", "status", ds, check=False).strip().lower().endswith("ready")
    print(f"uploading {len(files)} files ({size:.0f} MB) to the Kaggle dataset {ds} ({'new version' if exists else 'new'})…", flush=True)
    if exists:
        kaggle("datasets", "version", "-p", str(stage), "-m", time.strftime("%Y-%m-%d %H:%M"), "-r", "skip", capture=False)
    else:
        kaggle("datasets", "create", "-p", str(stage), "-r", "skip", capture=False)
    for _ in range(120):                                  # Kaggle processes the upload before kernels can mount it
        if kaggle("datasets", "status", ds, check=False).strip().lower().endswith("ready"):
            break
        time.sleep(30)
    shutil.rmtree(stage)
    return ds


def run_kernel(name: str, script: str, datasets: list[str], out_dir: Path, stage: Path) -> Path:
    """Run `script` as a private GPU script kernel with the datasets mounted under /kaggle/input;
    wait for it; download /kaggle/working into out_dir."""
    kid = f"{username()}/{slug(name)}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    (stage / "job.py").write_text(script)
    (stage / "kernel-metadata.json").write_text(json.dumps(dict(
        id=kid, title=name[:50], code_file="job.py", language="python", kernel_type="script", is_private=True,
        enable_gpu=True, enable_internet=True, dataset_sources=datasets, competition_sources=[], kernel_sources=[])))
    print(f"starting the Kaggle kernel {kid} on {ACCELERATOR}…", flush=True)
    kaggle("kernels", "push", "-p", str(stage), "--accelerator", ACCELERATOR, capture=False)
    t0, last = time.time(), ""
    while True:
        time.sleep(POLL_S)
        st = kaggle("kernels", "status", kid, check=False).strip()
        state = st.split('"')[-2] if '"' in st else st.split()[-1]
        if st != last:
            print(f"   {time.strftime('%H:%M')} {state} ({(time.time() - t0) / 60:.0f} min)", flush=True)
            last = st
        if any(k in st.lower() for k in ("complete", "error", "cancel")):
            break
    out_dir.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", kid, "-p", str(out_dir), capture=False, check=False)
    if "complete" not in st.lower():
        log = next(out_dir.glob("*.log"), None)
        raise RuntimeError(f"Kaggle kernel {kid} ended: {st}" + (f"\n{log.read_text()[-3000:]}" if log else ""))
    shutil.rmtree(stage, ignore_errors=True)
    return out_dir
