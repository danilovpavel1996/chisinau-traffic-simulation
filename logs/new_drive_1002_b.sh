#!/bin/zsh
# 2026_10_26_10_02, part B: laptop steps (+ light check and photo matching on the GPU services)
set -o pipefail
cd /Users/danilovpavel/Documents/traffic_chisinau/chisinau-commute-upgrade
D=2026_10_26_10_02
step() { echo "== $D: $1 ($(date +%H:%M))"; nice -n 10 .venv-video/bin/python -W ignore -c "from src.video import drives, pipeline; pipeline.run_step('$1', drives.load('$D'))" || exit 1; }
for s in road events event_match calibrate speed; do step $s; done
echo "== $D: light check ($(date +%H:%M))"; .venv-video/bin/python -W ignore -m src.video.light_context $D || exit 1
for s in events event_match; do step $s; done
echo "== $D: photo matching ($(date +%H:%M))"; nice -n 10 .venv-video/bin/python -W ignore -m src.video.vpr $D || exit 1
echo "== $D: map position ($(date +%H:%M))"; .venv-video/bin/python -W ignore -m src.video.map_layer $D || exit 1
for s in bev surround position streets dashboard; do step $s; done
echo "PART B DONE ($(date +%H:%M))"
