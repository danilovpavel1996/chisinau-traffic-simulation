"""
colab.py
--------
Run a long job on a free Google Colab T4 GPU through the official Colab CLI
(.venv-colab), for every step that would take more than ~5 minutes on the laptop.

    with Session("perception-<drive>") as s:      # new T4 runtime; released at the end
        s.upload(path)                            # any size: 25 MB parts, joined on the VM
        s.install("ultralytics", ...)
        s.start(script)                           # runs in the background on the VM
        s.wait()                                  # prints the job's progress until it ends
        s.download(remote, local)

`colab exec` waits only --timeout seconds (default 30) for a reply — a job run
directly "timed out" although it kept running — so the job runs detached
(nohup) and is polled with short execs that print its log tail.

The user signs in to Google themselves the first time (browser OAuth); no
credentials are handled here.
"""

from __future__ import annotations

import shlex
import subprocess
import tempfile
import time
from pathlib import Path

from src.video.drives import ROOT

COLAB = ROOT / ".venv-colab" / "bin" / "colab"
CHUNK_MB = 25            # the file API rejects single uploads above ~30 MB (HTTP 500)
POLL_S = 30
UPLOAD_TRIES = 3
JOB, LOG, DONE = "/content/job.py", "/content/job.log", "/content/job.done"


class Session:
    def __init__(self, name: str, gpu: str = "T4", reuse: bool = False):
        self.name = name.replace("_", "-").lower()
        self.gpu = gpu
        self.reuse = reuse           # attach to a runtime that is still alive (e.g. after the laptop crashed)

    def colab(self, *args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
        cmd = [str(COLAB), "--auth", "oauth2", *args]
        if not capture:
            print("$ colab " + " ".join(shlex.quote(a) for a in args), flush=True)
        return subprocess.run(cmd, check=check, capture_output=capture, text=True)

    def __enter__(self) -> "Session":
        if self.reuse:
            print(f"reusing the running GPU runtime '{self.name}'…", flush=True)
            return self
        print(f"starting a {self.gpu} GPU runtime '{self.name}' (first time: sign in to Google when asked)…", flush=True)
        self.colab("new", "-s", self.name, "--gpu", self.gpu)
        return self

    def has(self, remote: str, size: int) -> bool:
        """Is this exact file already on the VM (same size)?"""
        out = self.run(f"import os; print('@@', os.path.getsize({remote!r}) if os.path.exists({remote!r}) else -1)")
        return f"@@ {size}" in out

    def __exit__(self, *exc):
        print("releasing the GPU runtime…", flush=True)
        subprocess.run([str(COLAB), "--auth", "oauth2", "stop", "-s", self.name])

    def run(self, code: str, timeout: float = 120) -> str:
        """Execute a short snippet on the VM and return what it printed."""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
        out = self.colab("exec", "-s", self.name, "-f", f.name, "--timeout", str(timeout), check=False, capture=True)
        Path(f.name).unlink()
        return (out.stdout or "") + (out.stderr or "")

    def upload(self, path: Path, remote: str):
        """Upload a file of any size to `remote` (full path on the VM), unless it is already there."""
        if self.reuse and self.has(remote, path.stat().st_size):
            print(f"{path.name} already on the VM", flush=True)
            return
        data, mb = path.read_bytes(), CHUNK_MB * 1_000_000
        parts = (len(data) + mb - 1) // mb
        print(f"uploading {path.name} ({len(data) / 1e6:.0f} MB, {parts} part{'s' * (parts > 1)})…", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            names = []
            for i in range(parts):
                p = Path(tmp) / f"part_{i:04d}"
                p.write_bytes(data[i * mb:(i + 1) * mb])
                for attempt in range(UPLOAD_TRIES):          # a part can fail transiently
                    if self.colab("upload", "-s", self.name, str(p), f"/content/{p.name}", check=False).returncode == 0:
                        break
                    if "@@alive" not in self.run("print('@@alive')", timeout=60):
                        raise RuntimeError("the Colab runtime disconnected during the upload (free runtimes can be "
                                           "reclaimed) — run again; consider a smaller file")
                    time.sleep(5 * (attempt + 1))
                else:
                    raise RuntimeError(f"upload of {p.name} failed {UPLOAD_TRIES} times")
                names.append(f"/content/{p.name}")
        self.run(f"""import os
os.makedirs(os.path.dirname({remote!r}), exist_ok=True)
with open({remote!r}, "wb") as out:
    for p in {names!r}:
        out.write(open(p, "rb").read()); os.remove(p)
print(os.path.getsize({remote!r}), "bytes")""", timeout=300)

    def install(self, *packages: str):
        self.colab("install", "-s", self.name, *packages)

    def start(self, script: str, cwd: str = "/content"):
        """Run `script` detached on the VM; its output goes to the job log."""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(script)
        self.colab("upload", "-s", self.name, f.name, JOB)
        Path(f.name).unlink()
        self.run(f"""import subprocess, os
for p in ({LOG!r}, {DONE!r}):
    if os.path.exists(p): os.remove(p)
subprocess.Popen(f"cd {cwd} && python -u {JOB} > {LOG} 2>&1; echo $? > {DONE}", shell=True,
                 start_new_session=True)
print("started")""")

    def wait(self, poll_s: float = POLL_S) -> int:
        """Print the job's progress until it ends; returns its exit code."""
        seen = ""
        while True:
            out = self.run(f"""import os
log = open({LOG!r}, errors="replace").read() if os.path.exists({LOG!r}) else ""
print(log[-3000:].replace("\\r", "\\n"))
print("@@EXIT", open({DONE!r}).read().strip() if os.path.exists({DONE!r}) else "running")""", timeout=60)
            lines = [ln for ln in out.splitlines() if ln.strip() and not ln.startswith("@@EXIT")]
            last = lines[-1] if lines else ""
            if last and last != seen:
                print("   " + last[-160:], flush=True)
                seen = last
            if "@@EXIT" in out and "@@EXIT running" not in out:
                code = out.split("@@EXIT", 1)[1].split()[0]
                return int(code) if code.lstrip("-").isdigit() else -1
            time.sleep(poll_s)

    def log(self) -> str:
        return self.run(f"print(open({LOG!r}, errors='replace').read()[-8000:])")

    def download(self, remote: str, local: Path):
        local.parent.mkdir(parents=True, exist_ok=True)
        self.colab("download", "-s", self.name, remote, str(local))
