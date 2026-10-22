"""
viewer/serve.py
---------------
Tiny local web server for the drive viewers, with HTTP Range support so the
video can be seeked (python -m http.server cannot do that).

Usage:
    .venv-video/bin/python -m src.video.viewer.serve [--port 8765]
    → http://localhost:8765/<drive>/viewer.html
"""

from __future__ import annotations

import argparse
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from src.video.drives import PROCESSED

ROOT = str(PROCESSED)


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if not m or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
        end = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
        end = min(end, size - 1)
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        n = getattr(self, "_remaining", None)
        if n is None:
            return super().copyfile(source, outputfile)
        while n > 0:
            chunk = source.read(min(1 << 16, n))
            if not chunk:
                break
            outputfile.write(chunk)
            n -= len(chunk)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    a = ap.parse_args()
    httpd = ThreadingHTTPServer(("127.0.0.1", a.port), partial(RangeHandler, directory=ROOT))
    print(f"serving {ROOT} on http://localhost:{a.port}/", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
