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
PUSH_TRIES, PUSH_WAIT_S = 12, 300


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


def dataset_files(ds: str) -> dict[str, int]:
    out = kaggle("datasets", "files", ds, "--page-size", "200", check=False)
    return {p[0]: int(p[1]) for p in (ln.split() for ln in out.splitlines()) if len(p) >= 2 and p[1].isdigit()}


def push_dataset(name: str, files: dict[str, Path], stage: Path, reuse_if: list[str] | None = None) -> str:
    """Upload files (name → local path) as a private dataset (a new version if it exists); returns its id.
    reuse_if: if the dataset already holds these files at these sizes, no upload (a rerun)."""
    ds = f"{username()}/{slug(name)}"
    if reuse_if:
        have = dataset_files(ds)
        if all(have.get(n) == files[n].stat().st_size for n in reuse_if):
            print(f"the Kaggle dataset {ds} already holds these files — no upload", flush=True)
            return ds
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
    try:
        if exists:
            kaggle("datasets", "version", "-p", str(stage), "-m", time.strftime("%Y-%m-%d %H:%M"), "-r", "skip", capture=False)
        else:
            kaggle("datasets", "create", "-p", str(stage), "-r", "skip", capture=False)
    except RuntimeError:                                  # the CLI reported failure after 858 uploads although
        time.sleep(30)                                    # the dataset was created: trust Kaggle's own status
        if not kaggle("datasets", "status", ds, check=False).strip().lower().endswith("ready"):
            raise
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
    for attempt in range(PUSH_TRIES):                     # 409: too many sessions at once (free tier) — wait
        try:
            kaggle("kernels", "push", "-p", str(stage), "--accelerator", ACCELERATOR, capture=False)
            break
        except RuntimeError:
            if attempt == PUSH_TRIES - 1:
                raise
            print(f"   Kaggle would not start it (busy?) — retrying in {PUSH_WAIT_S // 60} min", flush=True)
            time.sleep(PUSH_WAIT_S)
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


# ── any remote script: the Colab jobs unchanged, on Kaggle ─────────────────
JOB_HEAD = """
import glob, json, os, shutil, subprocess, sys
_JOB_UP, _JOB_DOWN, _JOB_PIP = {up}, {down}, {pip}     # names no job script uses
_JOB_IN = os.path.dirname(glob.glob("/kaggle/input/**/" + next(iter(_JOB_UP.values())), recursive=True)[0])
for _r, _n in _JOB_UP.items():                        # each input where the script expects it
    os.makedirs(os.path.dirname(_r), exist_ok=True)
    shutil.copy(os.path.join(_JOB_IN, _n), _r)
if _JOB_PIP:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_JOB_PIP], check=True)
os.chdir("/content")
"""
JOB_TAIL = """
os.makedirs("/kaggle/working/out", exist_ok=True)
for _r, _n in _JOB_DOWN.items():
    shutil.copy(_r, os.path.join("/kaggle/working/out", _n))
print("job done", flush=True)
"""


def quota_left_h() -> float:
    out = kaggle("quota", check=False)
    m = re.search(r"GPU\s+\S+h\s+([\d.]+)h", out)
    return float(m.group(1)) if m else 0.0


def run_job(name: str, script: str, uploads: dict[str, Path], downloads: dict[str, Path], work: Path,
            pip: list[str] | None = None) -> None:
    """Run a remote script written for Colab (/content paths) on a Kaggle T4: uploads {remote path:
    local file} become a private dataset, the outputs {remote path: local file} are downloaded."""
    work.mkdir(parents=True, exist_ok=True)
    up_names = {r: f"in{i}_{Path(r).name}.bin" for i, r in enumerate(uploads)}   # .bin: Kaggle must not unpack zips
    down_names = {r: f"out{i}_{Path(r).name}" for i, r in enumerate(downloads)}
    ds = push_dataset(f"chisinau {name}", {up_names[r]: p for r, p in uploads.items()}, work / "dataset",
                      reuse_if=list(up_names.values()))
    job = JOB_HEAD.format(up=json.dumps(up_names), down=json.dumps(down_names), pip=json.dumps(pip or [])) + script + JOB_TAIL
    out = run_kernel(f"chisinau {name} job", job, [ds], work / "out", work / "kernel")
    for r, local in downloads.items():
        shutil.move(str(out / "out" / down_names[r]), local)
