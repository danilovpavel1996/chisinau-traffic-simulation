#!/bin/zsh
# 2026_10_24_15_48 on its right route: photo matching, light map prior, map position, 3D view, dashboard
set -o pipefail
cd /Users/danilovpavel/Documents/traffic_chisinau/chisinau-commute-upgrade
D=2026_10_24_15_48
echo "== $D: photo matching ($(date +%H:%M))"; nice -n 10 .venv-video/bin/python -W ignore -m src.video.vpr $D || exit 1
echo "== $D: map position ($(date +%H:%M))"; .venv-video/bin/python -W ignore -m src.video.map_layer $D || exit 1
echo "== $D: light check ($(date +%H:%M))"; .venv-video/bin/python -W ignore -m src.video.light_context $D || exit 1
echo "== $D: 3D view ($(date +%H:%M))"; nice -n 10 .venv-video/bin/python -W ignore -m src.video.surround $D --objects || exit 1
nice -n 10 .venv-video/bin/python -W ignore -c "from src.video import drives, pipeline; pipeline.run_step('dashboard', drives.load('$D'))"
echo "FIX 15_48 DONE ($(date +%H:%M))"
